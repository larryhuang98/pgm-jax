# Holonomic bond constraints (SHAKE / RATTLE) for flexible molecules

`pgm_jax/md/constraints.py`, `FlexibleSimulation(..., constraints="h-bonds" | "all-bonds", hmr=...)`.
Tests: `tests/test_shake.py` (and `test_hmr.py`, `test_md_macro.py`). Validation:
`scripts/validate_shake.py` (fitted pGM methanol), `scripts/shake_vs_pmemd.py` (against
pmemd.pgm.cuda with SHAKE), `scripts/bench_shake.py` (cost per call).

**In short.** Distance constraints hold the X-H bonds (`"h-bonds"`) or every bond
(`"all-bonds"`) of the flexible templates at their reference lengths; rigid templates (water,
ions) are always held by their three distances. The integrator is g-BAOAB / velocity Verlet with
SHAKE after each drift and RATTLE (the mass-weighted projection onto the constraint tangent space)
after each kick and thermostat step, so flexible molecules run at 2 fs, and at 4 fs with hydrogen
mass repartitioning. Constraints are solved exactly per cluster (Newton, closed-form small
systems) for the usual clusters and iteratively (matrix-free) for large ones.

## Usage

```python
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box

tpl = FlexibleTemplate.load("runs/flex/methanol.flex")
pos, H = liquid_box(tpl, 216, density=0.75)
sim = FlexibleSimulation(
    System([tpl.pgm] * 216),
    [tpl] * 216,
    pos,
    H,
    MDSettings(),
    dt=0.002,
    ensemble="npt",
    thermostat="bussi",
    constraints="h-bonds",
)  # 2 fs
sim = FlexibleSimulation(..., dt=0.004, constraints="h-bonds", hmr=3.024)  # 4 fs
sim = FlexibleSimulation(..., dt=0.002, constraints="all-bonds")  # every bond
sim.observables()  # ... "shake_err" (largest relative length error), "rattle_err" (largest
# |r^ . (v_a - v_b)| over the RMS relative speed of the constrained pairs)
```

- `constraints="none"` (default for flexible templates: every bond flexible, dt 0.5 fs),
  `"h-bonds"` (bonds between a hydrogen and a heavy atom, Amber's `ntc=2`), `"all-bonds"`
  (`ntc=3`). The lengths are the templates' reference bond lengths (`FlexibleTemplate.bond_lengths()`,
  e.g. `req` of an Amber topology through `amber_template`). The bonded energy of a constrained bond
  is then constant (zero for harmonic bonds) and needs no special treatment (Amber's `ntf=2` only
  skips computing it).
- `hmr=` (hydrogen mass repartitioning, `constraints.hmr_masses`): one hydrogen mass for every
  molecule or one value per molecule.
- Works with every ensemble and thermostat (NVE, Langevin, Bussi, GLE; the GLE auxiliary momenta
  are projected too), the Monte Carlo barostat, multiple time stepping (`mts=`; RATTLE after each
  kick of each level), restraints, virtual sites (never constrained), alchemical regions and replica
  exchange (batched replicas: the constraint solves are vmapped like the rest of the step).
- The log prints the constraint layout, e.g. `# constraints (h-bonds): 864 constraints in 432
  clusters: 432 x (4 atoms, 3 constraints); 3021 degrees of freedom`.
- `Constraints(pairs, d0, masses)` can be used on its own: `positions(x_new, x_ref)` (SHAKE),
  `momenta(x, p, masses)` (RATTLE), `violation(x)`, `velocity_violation(x, p, masses)`,
  `describe()`.

## Method

**Equations.** Constraint c holds atoms a_c, b_c at d_c. With s_c = r_a - r_b before a drift and
r~ the unconstrained drifted positions, SHAKE finds the multipliers lambda of

    r' = r~ + M^-1 sum_c lambda_c s_c (+ at a_c, - at b_c),    sigma_c = |r'_a - r'_b|^2 - d_c^2 = 0,

and RATTLE removes the component of the momenta that would change a length:
p' = p + sum_c mu_c r_c (+ / -) with (B M^-1 B^T) mu = -B v, B the constraint gradients at the new
positions. This is the orthogonal projection onto the tangent space in the M^-1 metric; it is
linear, idempotent and leaves the total momentum unchanged (constraint forces are internal).

**Clusters and blocks.** The constraint graph splits into clusters (connected components: an O-H,
a CH3, a water triangle, a whole methanol with every bond). A block is a padded array (clusters x
atoms x constraints) whose clusters (all copies of a molecule type, all CH3 groups of a protein,
all waters) are solved together: this is the per-template batching. Clusters of up to three
constraints (X-H, XH2, XH3, water) share one block (the calls are launch-bound, so one padded block
is faster than one per size); larger clusters get one block per number of constraints, so a solute
with every bond constrained does not pad the waters to its size.

- **Dense solver** (clusters of at most `dense_max` = 12 constraints): Newton on the cluster's
  own C x C system, J_cd = 2 r_c . s_d K_cd with K_cd = sum_a inc_ac inc_ad / m_a, a fixed number
  of iterations (4, unrolled: no device reduction or host round trip per step). Newton converges
  quadratically (relative length errors about 1e-3, 1e-6, 1e-12 after 1, 2, 3 iterations for the
  displacement of a 2-4 fs step), so 4 iterations reach float64 round-off. Systems of 1-3
  unknowns are solved in closed form (adjugate), larger ones by unrolled Gaussian elimination
  without pivoting (the matrices are, up to the difference between r and s, Gram matrices of the
  mass-weighted constraint gradients: symmetric positive definite), which XLA fuses into
  element-wise kernels over the whole block (a batched LU would launch a factorisation per call).
  RATTLE solves the linear C x C system once.
- **Iterative solver** (larger clusters, e.g. every bond of a protein: 1,2xx constraints in one
  cluster): one flat list, matrix-free (the products B M^-1 B^T x are two gathers). SHAKE by
  quasi-Newton iterations lambda <- lambda - J0^-1 sigma with J0 = 2 S K S^T, the Jacobian at the
  reference vectors (symmetric positive definite and fixed during the call), each solved by
  Jacobi-preconditioned CG to 1e-3; the outer loop runs until max |sigma_c| / d_c^2 <= `tol`
  (1e-10) (linear convergence at the rate |r - s| / d: 5-8 iterations for a 2-4 fs step). RATTLE:
  CG on R K R^T to `rattle_tol` (1e-11 of the initial residual). Both are while loops that test
  convergence every 8 CG iterations (each test is a device-host synchronisation).
- **Water.** A rigid water is a 3-constraint cluster, solved by the dense Newton to round-off: this
  is the solution SETTLE computes analytically (the same equations: displacements along the old
  bond vectors), so no separate SETTLE is needed. It costs little (table below).

**Integrator.** g-BAOAB (Leimkuhler & Matthews 2016) on the atoms: B (kick, RATTLE),
A (drift h/2, SHAKE + RATTLE), O (thermostat on the mass-scaled momenta, projected), A (drift h/2,
SHAKE), forces, B (kick, RATTLE). The drift that is followed by the closing kick leaves its RATTLE
to the kick, which projects at the same positions (P(p + hF) = P(Pp + hF)): one projection fewer
per step, the same trajectory to round-off (test). The drift before the O step keeps its RATTLE,
because the O step books the heat of the projected momenta (`econs`). NVE: B A B with SHAKE in the
drift and RATTLE in the kicks (velocity Verlet with RATTLE, symplectic and time-reversible).

**Degrees of freedom.** N_f = 3 N_real - N_constraints, minus 3 when the total momentum is
conserved: NVE and Bussi rescaling (the constraint forces are internal, so neither the constraints
nor the global rescaling change it; drawn momenta carry no net momentum, given ones are kept). Langevin and GLE act
on every degree of freedom (N_f without the -3). `temp_com` uses 3 N_mol (- 3), `temp_internal`
3 N_real - 3 N_mol - N_constraints.

**Pressure and the barostat.** The Monte Carlo barostat scales the molecular centres of mass and
translates every molecule rigidly, which keeps every constraint (and the momenta); the acceptance
uses N_mol kT ln(V'/V). The virial pressure is the molecular one (centre-of-mass kinetic energy,
forces times centres of mass): constraint forces act within molecules and sum to zero on each, so
they do not enter it (the atomic virial would need the constraint virial, sum_c lambda_c r_c r_c /
dt^2; the molecular form avoids it). Section 3 checks it: the NPT runs average 1 bar.

## Validation

All runs: the fitted pGM methanol of the paper (`runs/flex/methanol.flex`: class II bonded terms,
pGM with every pair, GAFF Lennard-Jones), 216 molecules (1,296 atoms), 0.9 nm cutoff, PME, LJ
tail, mixed precision and dipole tolerance 1e-5 unless noted, 298 K; starting from 200 ps of NPT
at 1 fs with X-H constraints (`scripts/validate_shake.py equil`). Configurations: `none` (every
bond flexible), `hb` (X-H bonds), `ab` (every bond), `hmr` (3.024 amu hydrogens); the number is the
time step in fs.

### 1. Constraints and the RATTLE condition at every step

`shake_err` (largest |r|/d0 - 1) and `rattle_err` (largest |r^ . (v_a - v_b)| over the RMS relative
speed of the constrained pairs), checked after every step for the first 200 steps and at every
0.1 ps report of the NVE runs below, and at every frame of the NVT/NPT samples:

| solver | largest shake_err | largest rattle_err |
|---|---|---|
| dense (216 methanols, X-H or all bonds, 0.5-5 fs, with and without HMR) | 1.1e-14 | 7.1e-15 |
| dense, peptide in water (X-H bonds; all bonds with the 32-constraint cluster forced dense) | 9.5e-15 | 7.8e-15 |
| iterative (the peptide with every bond: one 32-constraint cluster; tol 1e-10) | 5e-11 | 9e-14 |
| iterative, ubiquitin (1,237 constraints, one cluster; one call from a 2 fs-sized displacement) | 3.3e-11 | 1.9e-13 |

The dense solver reaches float64 round-off in its 4 Newton iterations; the iterative one stops at
its tolerance. `tests/test_shake.py` checks the solvers against each other (1e-10), the projection
properties (idempotent, M^-1-orthogonal, momentum-conserving), the tangency of the RATTLE velocities
by finite differences (length changes O(e^2) along them, O(e) without), and the constraints at every
step of NVE, NVT (Langevin, Bussi, GLE), NPT and multiple time stepping runs.

### 2. NVE energy conservation

2 ps Bussi at the configuration's own time step and masses, then 20 ps NVE (10 ps for float64 at
tolerance 1e-9); drift = slope of E_tot per degree of freedom; fluctuation = RMS about the linear fit
(`scripts/validate_shake.py nve`; the NVE runs ran on the CPU while the GPUs were busy). <T> is the
full-step kinetic temperature (see section 3 on its bias at large steps).

| run | constraints | dt (fs) | H mass (amu) | N_f | <T> full step (K) | drift, mixed (kT/ns/dof) | drift, float64 tol 1e-9 | E fluctuation, mixed (kJ/mol) | CG it./step |
|---|---|---|---|---|---|---|---|---|---|
| none-0.5 | none | 0.5 | 1.008 | 3885 | 296 | -0.0037 | -0.0060 | 0.88 | 4.6 |
| none-1 | none | 1 | 1.008 | 3885 | 292 | -0.0121 |  | 3.25 | 6.0 |
| hb-0.5 | h-bonds | 0.5 | 1.008 | 3021 | 301 | -0.0004 | +0.0043 | 0.37 | 3.0 |
| hb-1 | h-bonds | 1 | 1.008 | 3021 | 298 | +0.0062 | +0.0092 | 0.83 | 5.0 |
| hb-2 | h-bonds | 2 | 1.008 | 3021 | 296 | +0.0029 | -0.0004 | 3.01 | 6.9 |
| hb-2.5 | h-bonds | 2.5 | 1.008 | 3021 | 286 | +0.0092 |  | 4.84 | 7.0 |
| hmr-2 | h-bonds | 2 | 3.024 | 3021 | 303 | +0.0083 |  | 1.94 | 5.7 |
| hmr-3 | h-bonds | 3 | 3.024 | 3021 | 287 | -0.0094 |  | 4.34 | 6.9 |
| hmr-4 | h-bonds | 4 | 3.024 | 3021 | 278 | +0.2718 | +0.1795 | 8.06 | 7.2 |
| hmr-5 | h-bonds | 5 | 3.024 | 3021 | energy not finite after 400 steps | | | | |
| ab-2 | all-bonds | 2 | 1.008 | 2805 | 296 | +0.0049 | +0.0136 | 2.90 | 6.9 |
| ab-3 | all-bonds | 3 | 1.008 | 2805 | 278 | +0.0177 |  | 8.45 | 8.0 |
| ab-hmr-4 | all-bonds | 4 | 3.024 | 2805 | 281 | +0.0090 |  | 5.01 | 7.0 |
| ab-hmr-5 | all-bonds | 5 | 3.024 | 2805 | 272 | +0.0290 |  | 9.09 | 8.0 |

- Every configuration conserves the energy up to 3 fs; the drifts (0.005-0.02 kT/ns/dof) are at
  the level of the induction tolerance (the float64 runs at 1e-9 show the same), and the
  fluctuation grows as dt^2 (0.37, 0.83, 3.0 kJ/mol at 0.5, 1, 2 fs with X-H constraints). X-H
  constraints at 1 fs fluctuate as much as the unconstrained model at 0.5 fs, at 2 fs 3.4x as much
  (still 0.02 kT sqrt(N_f)).
- **Hydrogen mass repartitioning with X-H constraints stops at 3 fs for methanol**: with 3.024 amu
  hydrogens the methyl carbon keeps 5.96 amu, which raises the C-O stretch to about 1,300 cm^-1
  (period 25 fs); at 4 fs the energy drifts by 0.2-0.3 kT/ns/dof and at 5 fs the run fails.
  Constraining every bond removes that mode: **all bonds + HMR runs at 4 fs** (drift 0.009),
  and even 5 fs stays bounded (0.03). (Ubiquitin with X-H constraints and 3.024 amu hydrogens runs
  at 4 fs: docs/protein_ff.md.)
- The CG iterations per step grow with the step (4.6 unconstrained at 0.5 fs, 3.0 with X-H
  constraints at 0.5 fs, 5.0 at 1 fs, 6.9 at 2 fs, 7-8 at 3-5 fs): the dipole predictor extrapolates over
  a longer interval. This is part of the cost of a larger step (Speed).

Peptide in water (ACE-ALA-SER-NME, 252 rigid TIP3P-geometry waters, Na+ and Cl-; placeholder pGM,
ff19SB-form bonded terms; float64, tolerance 1e-6; 5 ps Bussi, 10 ps NVE;
`scripts/validate_shake.py peptide`):

| run | constraint blocks | N_f | drift (kT/ns/dof) | E fluctuation (kJ/mol) | max shake_err | max rattle_err | CG it./step |
|---|---|---|---|---|---|---|---|
| h-bonds 2 fs | 6 x (2 atoms, 1 constraints); 1 x (3 atoms, 2 constraints); 255 x (4 atoms, 3 constraints) | 1597 | +0.004 | 0.72 | 9e-15 | 5e-15 | 9.8 |
| all-bonds 2 fs | 252 x (3 atoms, 3 constraints); iterative: 32 constraints on 33 atoms (tol 1e-10) | 1582 | -0.020 | 0.64 | 5e-11 | 4e-14 | 9.9 |
| all-bonds 4 fs H 3.024 | 252 x (3 atoms, 3 constraints); iterative: 32 constraints on 33 atoms (tol 1e-10) | 1582 | +0.026 | 0.98 | 5e-11 | 9e-14 | 10.7 |
| h-bonds 4 fs H 3.024 | 6 x (2 atoms, 1 constraints); 1 x (3 atoms, 2 constraints); 255 x (4 atoms, 3 constraints) | 1597 | +0.001 | 1.75 | 9e-15 | 9e-15 | 11.4 |
| all-bonds 2 fs dense | 252 x (3 atoms, 3 constraints); 1 x (33 atoms, 32 constraints) | 1582 | +0.024 | 0.72 | 9e-15 | 6e-15 | 10.0 |
| all-bonds 4 fs H 3.024 dense | 252 x (3 atoms, 3 constraints); 1 x (33 atoms, 32 constraints) | 1582 | -0.020 | 0.72 | 1e-14 | 8e-15 | 10.6 |

Every variant conserves the energy; the iterative solver (tolerance 1e-10) and the dense solve of the
same 32-constraint cluster give the same drift and fluctuation (the drifts of these 10 ps runs of a
freshly heated system are dominated by its relaxation, not by the solver).

### 3. Equilibrium properties against the time step (216 methanols, NPT)

NPT at 298 K and 1 bar, Langevin 1/ps (which, unlike Bussi, does not use the degree-of-freedom
count, so the measured temperature tests it), Monte Carlo barostat every 0.1 ps; six independent
runs per configuration (20 ps of equilibration with the configuration's own settings from the
common equilibrated state, then 0.1 ns, a frame every 0.5 ps); errors are standard errors over
blocks (`scripts/validate_shake.py sample ... --seed k`, `analyze`). The reference is X-H
constraints at 1 fs (BAOAB's configurational error is O(dt^2): a quarter of that at 2 fs).
Distributions: O-O and O-HO radial distribution functions, the H-C-O-H dihedral, the C-O-H angle
and the C-O length (the last two are flexible with X-H constraints); the last column gives the
largest deviation from the reference over all bins in units of the combined standard error.

| run | density (g/cm^3) | U (kJ/mol per molecule) | T full step (K) | T half step (K) | T_com (K) | P (bar) | C-O-H (deg) | C-O (nm) | g_OO peak (nm / height) | max deviation from hb-1 (sigma): g_OO / g_OH / H-C-O-H / C-O-H / C-O |
|---|---|---|---|---|---|---|---|---|---|---|
| hb-1 | 0.7950 +- 0.0010 | -800.652 +- 0.030 | 295.86 +- 0.35 | 298.24 +- 0.35 | 297.91 +- 0.59 | 15 +- 11 | 107.296 +- 0.009 | 0.14285 +- 0.00000 | 0.297 / 2.131 | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 |
| hb-2 | 0.7913 +- 0.0011 | -800.591 +- 0.039 | 288.72 +- 0.45 | 298.24 +- 0.46 | 298.63 +- 0.56 | -15 +- 14 | 107.291 +- 0.008 | 0.14285 +- 0.00000 | 0.297 / 2.181 | 3.2 / 2.6 / 2.3 / 2.5 / 3.2 |
| ab-hmr-4 | 0.7953 +- 0.0016 | -801.934 +- 0.040 | 281.88 +- 0.32 | 298.35 +- 0.33 | 297.34 +- 0.72 | -8 +- 13 | 107.360 +- 0.009 | 0.14237 +- 0.00000 | 0.302 / 2.146 | 3.0 / 2.2 / 2.1 / 4.3 / 3757.6 |
| none-0.5 | 0.7854 +- 0.0015 | -796.310 +- 0.060 | 295.96 +- 0.38 | 297.98 +- 0.38 | 297.50 +- 1.08 | 20 +- 15 | 107.311 +- 0.018 | 0.14279 +- 0.00000 | 0.297 / 2.149 | 3.6 / 2.7 / 3.9 / 2.7 / 3.2 |

- **X-H constraints at 2 fs reproduce 1 fs**: potential energy within 0.06 +- 0.05 kJ/mol per
  molecule, every distribution within 2.3-3.2 sigma at its worst bin (the expected maximum of
  |z| over 70-200 bins is about 2.5-3), pressure 1 bar within its error; the density is 0.0037 +-
  0.0015 g/cm^3 lower (2.5 sigma; 0.6 ns per configuration).
- **Every bond constrained with 3.024 amu hydrogens at 4 fs**: density 0.7953 +- 0.0016 as at
  1 fs. It is a different model (the C-O bond is rigid: 0.14237 nm instead of a mean of 0.14285),
  hence the shifted C-O length, a slightly wider C-O-H angle and a potential energy 1.3 kJ/mol
  lower (a C-O stretch no longer stores kT/2).
- **Constraining the X-H bonds changes the liquid itself** (against the unconstrained model at
  0.5 fs): the potential energy drops by 4.3 kJ/mol per molecule (four stretches of kT/2 = 5.0
  kJ/mol, less their coupling) and the density rises from 0.7854 +- 0.0015 to 0.7950 +- 0.0010
  (+1.2 %): the constrained O-H and C-H lengths are the bonded model's reference lengths, not the
  thermal averages, and the covalent and induced dipoles follow. A model parameterised for
  flexible bonds needs this checked (or refitted) before it is run with constraints.
- **Pressure.** The virial pressure (molecular virial; constraint forces cancel within each
  molecule) averages 1 bar within its error in every NPT run (-15 +- 14 to 20 +- 15 bar): the
  virial is consistent with the Monte Carlo barostat's ensemble.

**Temperature.** The full-step kinetic temperature of BAOAB (and of velocity Verlet) is low at
large steps: for a harmonic mode of frequency w, <p^2> = kT (1 - (w h)^2 / 4) at full steps, while
the configurations are sampled exactly. With X-H constraints at 2 fs the H-C-H and C-O-H bends
(w h ~ 0.5) make the whole liquid read 289 K instead of 298 K; at 1 fs 296 K, at 4 fs (all bonds, HMR) 282 K. The half-step mean
(K(p - hF/2) + K(p + hF/2)) / 2 = K(p) + h^2/8 (PF) M^-1 (PF), which leapfrog codes such as Amber
report, is exact in the harmonic limit; `observables()["temp_half"]` (`half_step_kinetic()`) gives
it. With Langevin, whose friction does not use the degree-of-freedom count, it reads 298.0-298.4 +- 0.4 K
in every run of the table, which checks N_f = 3N - N_c: had the 864 X-H constraints of the box not been removed from
N_f, the temperature would read 22 % low.

### 4. Against pmemd.pgm with SHAKE (same model, independent code)

125 methanols (Amber's MEOHBOX, 750 atoms, 2.009 nm box), placeholder pGM (Amber charges as
Gaussian charges, pGM-pol polarizabilities and radii), parm10 bonded terms and LJ from the prmtop,
written with `write_pgm_prmtop` (`scripts/shake_vs_pmemd.py`). Single point (float64, dipole
tolerance 1e-9, with the LJ tail): EELEC -8948.3885 (engine) / -8948.3876 kcal/mol (pmemd.pgm; the
1e-7 is the known PME influence-function factor), BOND 15.2645, ANGLE 38.2507, DIHED 33.3117,
VDWAALS -236.8170 in both: EPtot differs by 0.004 kJ/mol. MD: NVT 298 K, Langevin 1/ps, X-H bonds
constrained (pmemd: SHAKE, ntc = ntf = 2, tol 1e-7; engine: `constraints="h-bonds"`), 9 A,
PME ~0.63 A order 6, LJ tail, dipole tolerance 1e-5; independent runs from pmemd's minimised
structure (pmemd: 16 x (10 ps heating, 50 ps equilibration, 0.25 ns), CPU pmemd.pgm; engine:
8 x (50 ps, 0.25 ns), mixed precision on the CPU), one block per run.

| run | density (g/cm^3) | U (kJ/mol per molecule) | T full step (K) | T half step (K) | T_com (K) | P (bar) | C-O-H (deg) | C-O (nm) | g_OO peak (nm / height) | max deviation from pm1fs (sigma): g_OO / g_OH / H-C-O-H / C-O-H / C-O |
|---|---|---|---|---|---|---|---|---|---|---|
| pmemd.pgm, 1 fs | - | -296.106 +- 0.014 | 297.79 +- 0.25 | - | - | - | 109.005 +- 0.006 | 0.14050 +- 0.00000 | 0.312 / 1.492 | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 |
| pmemd.pgm, 2 fs | - | -295.700 +- 0.009 | 298.05 +- 0.16 | - | - | - | 109.017 +- 0.004 | 0.14049 +- 0.00000 | 0.318 / 1.471 | 3.5 / 2.7 / 3.3 / 5.3 / 4.8 |
| engine, 1 fs | - | -296.268 +- 0.011 | 295.68 +- 0.26 | 297.76 +- 0.27 | 298.43 +- 0.48 | - | 109.019 +- 0.008 | 0.14050 +- 0.00001 | 0.312 / 1.493 | 3.1 / 3.1 / 3.1 / 2.5 / 3.1 |
| engine, 2 fs | - | -296.268 +- 0.011 | 289.32 +- 0.24 | 297.64 +- 0.24 | 297.39 +- 0.39 | - | 109.010 +- 0.006 | 0.14049 +- 0.00000 | 0.312 / 1.485 | 2.4 / 3.0 / 3.1 / 3.3 / 3.2 |

- **Same Hamiltonian, same answer.** The engine's potential energy is the same at 1 and 2 fs
  (-296.268 +- 0.011 kJ/mol per molecule both), while pmemd's Langevin integrator moves with the
  step (-296.106 +- 0.014 at 1 fs, -295.700 +- 0.009 at 2 fs). Extrapolated as dt^2, pmemd's
  dt -> 0 value is -296.241 +- 0.02, within 0.027 +- 0.023 of the engine. The distributions of
  both engine runs match pmemd at 1 fs within 2.4-3.3 sigma at the worst bin (pmemd's own 2 fs run
  differs from its 1 fs run by up to 5.3 sigma). The unconstrained C-O length (0.14049-0.14050 nm)
  and the C-O-H angle (109.01 deg) agree to their errors.
- **Temperature.** pmemd reports the leapfrog average (297.8-298.1 K); the engine's half-step
  estimator gives the same (297.6-297.8 K), its full-step value 289.3 K at 2 fs and 295.7 K at 1 fs
  (the bias of section 3, 4x smaller at half the step). Degrees of freedom: 1,750 for 750 atoms and
  500 constraints (Langevin), as pmemd's.
- Constraints held to 8e-15 (lengths) and 8.5e-15 (RATTLE) at every frame of the engine runs.

### 5. Speed

216 methanols (1,296 atoms), one RTX PRO 6000 Blackwell, mixed precision, dipole tolerance 1e-5,
NVT with Bussi 0.5 ps, 0.9 nm cutoff, PME 36^3; 10 ps after 2 ps of warm-up, no output
(`scripts/validate_shake.py speed`):

| run | constraints | dt (fs) | H (amu) | ms/step | CG it./step | ns/day | vs unconstrained 0.5 fs |
|---|---|---|---|---|---|---|---|
| none-0.5 | none | 0.5 | 1.008 | 0.641 | 4.6 | 67.4 | 1.00 |
| hb-0.5 | X-H | 0.5 | 1.008 | 0.699 | 3.0 | 61.8 | 0.92 |
| hb-1 | X-H | 1 | 1.008 | 0.830 | 5.0 | 104.0 | 1.54 |
| hb-2 | X-H | 2 | 1.008 | 0.958 | 6.9 | 180.3 | **2.67** |
| hmr-3 | X-H | 3 | 3.024 | 0.942 | 7.0 | 275.1 | 4.08 |
| ab-2 | all bonds | 2 | 1.008 | 1.050 | 6.9 | 164.5 | 2.44 |
| ab-hmr-4 | all bonds | 4 | 3.024 | 1.183 | 7.0 | 292.1 | **4.33** |

The step costs more at larger dt mainly because the induced-dipole predictor extrapolates over a
longer interval (4.6 CG iterations per step at 0.5 fs, 7 at 2-4 fs; about 0.066 ms per iteration
here); the constraints themselves cost about 0.15 ms per step for this small system (two SHAKE and
four RATTLE calls per NVT step, launch-bound: see the per-call table), every bond 0.1 ms more
(5-constraint clusters by elimination). 2 fs with X-H constraints is 2.7x faster than the
unconstrained 0.5 fs step, all bonds + HMR at 4 fs 4.3x.

Cost per call (`scripts/bench_shake.py`, same GPU, float64 positions; a displacement of the size of
a 2 fs step):

| system | constraints | layout | SHAKE (ms) | RATTLE (ms) |
|---|---|---|---|---|
| 4,096 rigid waters | 12,288 | 4,096 x (3, 3) | 0.041 | 0.017 |
| 216 methanols, X-H | 864 | 432 x (4 atoms, 3 constraints) | 0.031 | 0.017 |
| 216 methanols, all bonds | 1,080 | 216 x (6, 5), Gaussian elimination | 0.099 | 0.029 |
| ubiquitin in water, X-H (15,955 atoms) | 15,353 | 5,294 x (4, 3) | 0.035 | 0.024 |
| ubiquitin in water, all bonds | 15,961 | 4,908 waters dense + 1,237 iterative | 0.88 | 0.39 |

The 4,096-water block costs what the previous single-block code did (0.042 / 0.028 ms,
`runs/bench_cons.py` of master on the same GPU). Two findings shaped the layout: products written as
einsums became batched tiny matrix products (4x slower; now broadcast products that XLA fuses), and
one padded block of all small clusters beats one block per size (ubiquitin X-H: 0.224 vs 0.251 ms
in an earlier version), because the calls are launch-bound: clusters of up to three constraints
therefore share one block. The iterative solver (ubiquitin, every bond) needs 5-8 quasi-Newton
iterations of 8-16 CG steps for SHAKE and 24-40 CG steps for RATTLE; it is 25x the X-H cost and would
add about 2.5 ms to a 5 ms ubiquitin step.

## Limits

- Distance constraints only: no angle constraints (e.g. GROMACS' h-angles), no rigid bodies of more
  than three atoms by constraints (a flexible template with `"all-bonds"` keeps its angles flexible).
- No separate SETTLE kernel: water is the dense 3-constraint Newton, which gives SETTLE's solution
  to round-off at a small cost (0.041 ms per SHAKE, 0.017 ms per RATTLE for 4,096 waters: section 5).
- The iterative solver (clusters above 12 constraints, e.g. `"all-bonds"` on a protein) iterates
  to a tolerance with device-host synchronisation every 8 CG iterations; it is several times the
  cost of the dense blocks (ubiquitin, section 5). For proteins, X-H constraints with HMR (4 fs) are
  the efficient choice; the iterative path is for completeness and correctness.
- Methanol-like molecules with CH3 groups do not take HMR 3.024 at 4 fs with X-H constraints only
  (the light methyl carbon); use `"all-bonds"` or 3 fs.
- `temp_half` needs the forces of the current step and is not defined with multiple time stepping
  (the fast and slow forces enter at different steps); `temp_K`, `ekin` and `etot` stay full-step
  values (the conserved quantity of velocity Verlet).
- The molecular virial is used for the pressure (constraint forces drop out); an atomic virial with
  the constraint virial is not implemented.
- NVE energy drifts, the samples of sections 3 and 4 ran on the CPU (the GPU was held by other jobs);
  the speeds are GPU numbers. Samples are 0.3-0.6 ns per configuration (216 methanols) and 2-4 ns
  (125 methanols): enough for 0.001-0.002 g/cm^3 and 0.01-0.06 kJ/mol, not for smaller effects.
