# Path-integral MD: PIMD, TRPMD, RPMD (nuclear quantum effects)

`pgm_jax/md/pimd.py`; `scripts/pimd_water.py` (quantum flexible pGM water), `scripts/pimd_validate.py`
(model systems), `scripts/pimd_openmm.py` (comparison with OpenMM's RPMDIntegrator). Tests:
`tests/test_pimd.py`.

**In short.** A ring polymer of P beads per atom samples the quantum Boltzmann distribution of the
nuclei (PIMD) and gives approximate quantum dynamics (RPMD, thermostatted RPMD). The beads are
batched with `jax.vmap` over the ordinary pGM force field of the flexible engine
(`FlexibleSimulation`): one compiled program evaluates every bead, each bead with its own induced
dipoles and dipole-predictor history. Normal-mode propagation (exact free ring polymer, or its
Cayley form), PILE-L / PILE-G thermostats, TRPMD and plain RPMD, primitive and centroid-virial
kinetic-energy estimators (per element), the molecular centroid-virial pressure, bead-averaged
molecular dipoles, and ring-polymer contraction of the intermolecular forces. Flexible molecules
only (rigid-body water is not a ring polymer of atoms; see Limits); NVT and Monte Carlo NPT.

| Check | Result |
|---|---|
| Harmonic oscillators, P = 1-64, beta hbar omega = 2.5 and 17.8 | <V>, primitive and centroid-virial <K> = the exact P-bead values within 1-2 standard errors; 1/P^2 convergence to the quantum value |
| Free particles (PILE-L, PILE-G, TRPMD) | every normal-mode temperature 299.6-300.4 K at 300 K; mode spreads exact to 0.2 % |
| RPMD (NVE ring polymer) | H_P fluctuation O(dt^2); model and pGM water |
| OpenMM 8.2 RPMDIntegrator, with and without contraction | 32 observables agree within 2.2 standard errors |
| pGM beads | vmapped / chunked beads = single evaluations (1e-7); contraction forces = -dU/dq (2e-6); barostat trial energy = U (1e-7) |
| Flexible pGM water, 512 molecules, 298 K, P = 32 | KE_H = 148.5 +- 0.1 meV (about 152 converged in P), KE_O 55.2 meV; O-H and H-H peaks broadened, O-O unchanged; dipole 2.04 -> 2.14 D; TRPMD D = 2.2e-5 cm^2/s (classical 4.1) |
| Speed (512 waters, dt 0.25 fs) | P = 32: 7.8 ms/step (2.8 ns/day); contracted to 8 beads 4.2 ms (5.1 ns/day), to 1 bead 2.5 ms (8.8 ns/day) |

## Theory and implementation

Ring-polymer Hamiltonian (physical masses on every bead, sampled at P T):

    H_P = sum_k sum_i [ |p_i^k|^2 / 2 m_i + m_i omega_P^2 |q_i^k - q_i^(k+1)|^2 / 2 ] + U(q),
    U = sum_k V(q^k),   omega_P = P kB T / hbar,   hbar = 0.0635078 kJ/mol ps.

Normal modes: q~_l = sum_j C_jl q_j with the real orthonormal C (centroid, cos/sin pairs, the
alternating mode for even P), frequencies omega_l = 2 omega_P sin(pi l / P).

**Integrator** (BAOAB order, Liu, Li & Liu JCP 145, 024103 (2016); PILE of Ceriotti et al. JCP 133,
124104 (2010)): `B(dt/2) A(dt/2) O(dt) A(dt/2) [forces] B(dt/2)`, one force evaluation per step,
the A and O steps in normal-mode coordinates (two P x P transforms per step). A is the exact free
ring-polymer step (`propagator="exact"`: rotation of each mode) or its Cayley transform (`"cayley"`,
default; Korol, Bou-Rabee & Miller, JCP 151, 124103 (2019)): both conserve every mode's energy, so
both sample the free ring polymer exactly, and Cayley does not resonate as P grows. RPMD is
`B A(dt) B`.

**Thermostats** (O step at kB T_P = P kB T; the heat is booked, `econs` = H_P - heat is conserved):

| mode | internal modes l > 0 | centroid |
|---|---|---|
| `"pimd"`, `thermostat=PILE("l", tau_centroid)` | Langevin gamma_l = 2 lam omega_l (lam = 1: critical damping) | Langevin 1/tau_centroid |
| `"pimd"`, `thermostat=PILE("g", tau_centroid)` | same | Bussi global rescaling, time constant tau_centroid (gentler on the centroid and the dipole predictor) |
| `"trpmd"` (Rossi, Ceriotti & Manolopoulos JCP 140, 234116 (2014)) | gamma_l = 2 lam omega_l, lam = 1/2 by default | none: the centroid dynamics estimates Kubo-transformed correlation functions |
| `"rpmd"` | none | none (NVE ring polymer) |

**Estimators** (per atom, reported as sums and per-element means in meV):

    primitive        K_i = 3 P kB T / 2 - (1/P) sum_k m_i omega_P^2 |q_i^k - q_i^(k+1)|^2 / 2
    centroid virial  K_i = 3 kB T / 2 - (1/2P) sum_k (q_i^k - qbar_i) . f_i^k
    potential        <V> = <U> / P
    pressure         P = N_mol kB T / V - tr( (1/P) sum_k dV(q^k)/d eps ) / 3V

The pressure uses a molecular centroid virial: every bead of a molecule is translated with the
molecular centre of mass of the centroid (the springs and the intramolecular terms do not change;
the ideal term is that of the classical molecular centroids). `observables()` also gives the
centroid temperature and the temperature of every normal mode (`estimators()["t_modes"]`), the
bead kinetic temperature (= T), and bead-averaged molecular dipoles (`molecular_dipoles()`:
charges, covalent and induced dipoles of every bead; `dipole_D` is |<mu>_beads|, `dipole_bead_D`
<|mu|>_beads).

**pGM beads** (`PGMBeads`). The force field of a `FlexibleSimulation` (`PGMForceField` + bonded
terms) is vmapped over the beads. Each bead keeps its own `InductionState` (converged dipoles and
the 4-step predictor history); the predictor's step counter is shared (kept unbatched), so its
fused / unfused switch stays a real branch under vmap, as for batched replicas (`remd.py`). One
neighbour list serves all beads: it is built on the centroid, and its radius is enlarged by
`bead_margin` (default 0.08 nm), the largest distance of a bead atom from its centroid (atom
lists: pairs of centroid atoms within cutoff + 2 margins; molecule-centre lists: the group radius
grows by the margin). The driver checks the margin after every block (error, never silent), and
row / list overflows resize and repeat the block as in the classical driver. Ring polymers are
wrapped as a whole by the centroid's molecular centres. The force beads are evaluated in chunks of 8
vmapped beads run one after the other (`bead_chunk="auto"`; identical results): for 32 beads of 512
waters one vmap over all beads takes 15.2 ms per step, chunks of 8 take 7.8 ms (smaller working
sets per kernel).

**Ring-polymer contraction** (`contract=P'`; Markland & Manolopoulos, JCP 129, 024105 (2008)). The
potential is split as V = V_mono + (V - V_mono): V_mono is the sum of the gas-phase monomer
energies of the fitted flexible templates (bonded terms + all-pair intramolecular pGM with its
induction + intramolecular van der Waals, i.e. exactly the model the bonded terms were fitted
with), evaluated on all P beads; the intermolecular remainder (PME, induction, van der Waals) is
evaluated on P' beads obtained by truncating the normal modes, q' = T q with
T = sqrt(P'/P) C'_{jl} C_{kl} over the P' lowest modes (T T^T = (P'/P) I, the centroid is kept), and
its forces return through T^T:

    U = sum_k V_mono(q^k) + (P/P') sum_k' [V - V_mono](q'^k').

For pGM this split (and not the Ewald real/reciprocal one of the original paper) is natural: pGM
has no intramolecular exclusions, so the 1-2 / 1-3 electrostatics are part of the stiff monomer
potential and must see every bead; what is left is smooth on the scale of the ring polymer.
P' = 1 puts the intermolecular forces on the centroid.

## Flexible pGM water

`models.water.flexible_water(molecule)` makes a flexible water template from a pGM water molecule: bonded
terms (new family `bond_quartic`: K2 db^2/2 + K3 db^3 + K4 db^4, plus `angle_harm`, `angle_cubic`,
`bond_bond`, `bond_angle` of `pgm_jax.bonded`) fitted so that the gas-phase monomer potential
(bonded + all-pair intramolecular pGM electrostatics and induction) reproduces the q-TIP4P/F
intramolecular potential (Habershon, Markland & Manolopoulos, JCP 131, 024501 (2009): quartic
Morse O-H bonds, D = 116.09 kcal/mol, a = 2.287 A^-1, r_eq = 0.9419 A; harmonic bend
k = 87.85 kcal/mol/rad^2, theta_eq = 107.4 deg). The force constants enter linearly and are solved
by least squares on energies and forces of 2000 geometries (bonds +-0.08 A, angle +-9 deg),
inside a Nelder-Mead search over the reference values. A Morse bond with a fixed dissociation
energy could not absorb the strong intramolecular pGM Coulomb of this water (q_O = -2.04 e with
covalent dipoles): 3.9 kJ/mol RMS and a minimum drifting away; the quartic bond fits.

The electrostatics and Lennard-Jones are those of the pGM water of the README
(`~/pgm-gvdw-data/topology/rayl_512_v2.prmtop`, the 512-water box). The result
(`validation/pimd/pgm_water_flex.flex`, `python scripts/pimd_water.py template`):

| | flexible pGM water (gas phase) | q-TIP4P/F intramolecular target |
|---|---|---|
| O-H at the minimum | 0.9424 A | 0.9419 A |
| H-O-H at the minimum | 107.59 deg | 107.4 deg |
| harmonic bend / symmetric / antisymmetric stretch | 1600 / 3847 / 3916 cm^-1 | 1580 / 3853 / 3920 cm^-1 |
| fit error over 2000 geometries | 0.30 kJ/mol RMS energy, 25 kJ/mol/nm RMS force (target RMS force 3318) | |
| the MD engine (PME, one molecule in a 3 nm box) | 1599.5 / 3847.3 / 3915.6 (double), 1600.1 / 3847.3 / 3915.6 cm^-1 (mixed) | |

The last row checks that the periodic engine gives the gas-phase monomer surface the contraction
uses as its stiff reference (to 0.6 cm^-1).

So the flexible pGM water has q-TIP4P/F's gas-phase intramolecular surface and pGM's
intermolecular interactions (with its intramolecular induction responding to the liquid). It is a
new model: comparisons with q-TIP4P/F below are with a different (fixed-charge) model that shares
only the monomer potential; comparisons with experiment are indicative.

## Usage

```python
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate
from pgm_jax.md.pimd import PILE, PIMDSimulation

tpl = FlexibleTemplate.load("validation/pimd/pgm_water_flex.flex")
sim = FlexibleSimulation(
    System([tpl.pgm] * n),
    [tpl] * n,
    pos,
    H,
    MDSettings(),
    dt=0.00025,
    temperature=298.0,
    thermostat="bussi",
)
sim.minimize(200)
sim.run(8000)  # classical flexible equilibration
pi = PIMDSimulation(sim, beads=32, thermostat=PILE("g", tau_centroid=0.1))  # contract=8: contraction
pi.run(40000, prefix="qwater", report_every=200, traj_every=200, checkpoint_every=4000, report_pressure=True)
pi.set_mode("trpmd")
pi.run(80000, prefix="qwater_trpmd", report_every=200, traj_every=40)  # dynamics
pi.observables()  # temp_K, temp_centroid, epot, ekin_prim, ekin_cv, ke_H_cv_meV, ke_O_cv_meV, dipole_D, cg...
pi.pressure()
pi.centroid()  # (N, 3) nm
pi.beads()  # (P, N, 3) nm
pi.molecular_dipoles()
pi.save_checkpoint("q.pimd.chk")
pi.load_checkpoint("q.pimd.chk")
```

Any potential (model systems, tests): `PIMDIntegrator(PotentialEngine(V), masses, P, T, dt, mode,
thermostat)` with `V(x (N, 3), box)`; `PotentialEngine(V, soft=W, contract=P')` contracts `W`.

```bash
python scripts/pimd_water.py template                                   # the flexible pGM water
python scripts/pimd_water.py run --beads 32 --ps 10 --rdf --pressure --save --prefix runs/pimd/w32
python scripts/pimd_water.py run --beads 32 --contract 8 ...           # contraction
python scripts/pimd_water.py run --beads 32 --mode trpmd --load runs/pimd/w32.pimd.chk --classical-ps 0 --equil-ps 0 --ps 20
python scripts/pimd_water.py bench --beads 8 32 --contract 0 1 4       # speed
python scripts/pimd_validate.py harmonic | free | nve                  # model systems (CPU)
~/miniconda3/envs/colabfold/bin/python scripts/pimd_openmm.py openmm; python scripts/pimd_openmm.py pgmjax
```

`run` writes `prefix.log` (estimators every `--report-ps`), `prefix.json` (means with block
errors, econs drift, speed, diffusion coefficient of the centroid molecular centres), `prefix.rdf`
(bead-averaged intermolecular g_OO, g_OH, g_HH), `prefix.msd`, and with `--save` the checkpoint.

## Validation

All numbers: `validation/pimd/*.json` (model systems, CPU) and `validation/pimd/water/` (water on one
RTX PRO 6000 Blackwell: the `run` summaries `*.json`, bead-averaged g(r) `*.rdf`, RPMD logs, speed
benchmarks). Errors are standard errors from 10-20 blocks.

### 1. Harmonic oscillators: exact P-bead and quantum values

256 independent 3D oscillators (1.008 amu, 300 K), PILE-L (centroid 1/tau_centroid = omega), dt = 0.1/omega.
For the discretised oscillator <K> = <V> = (kT/2) sum_l omega^2 / (omega_l^2 + omega^2) per degree of
freedom at every P (both estimators have this average); the quantum value is (hbar omega / 4)
coth(beta hbar omega / 2). kJ/mol per degree of freedom (`python scripts/pimd_validate.py harmonic`):

| omega (beta hbar omega) | P | exact P-bead | exact / quantum - 1 | <V> | <K> primitive | <K> centroid virial |
|---|---|---|---|---|---|---|
| 700 /ps (17.8, an O-H stretch) | 1 | 1.2472 | -88.8 % | 1.2463(16) | 1.2472 | 1.2472 |
| | 4 | 4.5513 | -59.0 % | 4.5535(23) | 4.5512(2) | 4.5509(18) |
| | 8 | 7.4245 | -33.2 % | 7.4193(33) | 7.4254(9) | 7.4199(23) |
| | 16 | 9.7095 | -12.6 % | 9.7142(23) | 9.7032(26) | 9.7155(24) |
| | 32 | 10.7065 | -3.67 % | 10.7031(30) | 10.6969(33) | 10.7065(26) |
| | 64 | 11.0077 | -0.96 % | 11.0071(26) | 10.9819(67) | 11.0108(26) |
| 100 /ps (2.55) | 8 | 1.8378 | -1.08 % | 1.8374(15) | 1.8332(28) | 1.8380(2) |
| | 32 | 1.8565 | -0.068 % | 1.8534(15) | 1.8256(61) | 1.8569(3) |
| | 64 | 1.8575 | -0.017 % | 1.8544(12) | 1.8102(63) | 1.8581(2) |
| same, dt = 0.025/omega | 32 | 1.8565 | | 1.8546(12) | 1.8510(71) | 1.8566(3) |
| | 64 | 1.8575 | | 1.8572(13) | 1.8492(64) | 1.8578(2) |

The quantum limit is approached as 1/P^2 (the exact P-bead error falls 3.4-4x per doubling of P).
The centroid-virial estimator matches the exact value at every P with a P-independent error bar;
the primitive one has a variance growing with P and, at large P, a small time-step bias (its spring
term is a difference of large numbers: -2.5 % at P = 64 with dt = 0.1/omega, gone at dt/4). An
O-H stretch at 300 K needs P = 32 for 3.7 % and P = 64 for 1 %.

### 2. Free particles: thermostat and normal-mode temperatures

512 free particles, P = 16, 300 K, 3000 samples (`pimd_validate.py free`): every normal-mode
temperature (centroid included) is 299.6-300.4 K with PILE-L and PILE-G (centroid 299.9 / 299.7
K); the internal-mode spreads <q_l^2> are kT_P / (m omega_l^2) to 0.999-1.002; the centroid-virial
kinetic energy is 3kT/2 = 3.7415 kJ/mol exactly, the primitive 3.733 (its statistical error). With
TRPMD the internal modes are at 299.7-300.4 K and the centroid momentum is untouched (a test checks it
is conserved to 1e-10).

### 3. Energy conservation of the ring-polymer Hamiltonian (RPMD, no thermostat)

- Model (`pimd_validate.py nve`): 64 particles in the q-TIP4P/F O-H Morse expansion (omega 700 /ps),
  P = 32, 300 K, 10 ps: the RMS fluctuation of H_P per degree of freedom is 4.3e-3 / 1.2e-3 / 3.0e-4 kT
  at dt = 0.4 / 0.2 / 0.1 fs (Cayley; exact free step 5.3e-3 / 1.4e-3 / 3.5e-4), i.e. O(dt^2);
  drift at 0.2 fs 0.001 kT/ns per degree of freedom (Cayley at 0.4 fs: -0.25 kT/ns).
- pGM water, 8 molecules, P = 4, float64 (`tests/test_pimd.py`): the H_P fluctuation drops by 4x
  from dt = 0.1 to 0.05 fs.
- 512 pGM waters, P = 32, 1 ps of RPMD from the same PIMD state: H_P fluctuates with an RMS of
  60 kJ/mol at dt = 0.25 fs and 15 kJ/mol at 0.125 fs (1.6e-4 and 4e-5 kT per degree of freedom;
  O(dt^2)), with 0.1 ps block means wandering by +-50 and +-30 kJ/mol and no drift beyond that in 1 ps.
  Mixed precision with dipole tolerance 1e-5, 1e-7 and float64 with 1e-8 give the same H_P(t) to
  2 kJ/mol: the fluctuation is the integrator's, not the induction's. (Classical flexible water,
  P = 1: 0.3 kJ/mol RMS.)

Under PILE the booked conserved quantity drifts (about 0.01 kJ/mol/ps per bead degree of freedom
for water at 0.25 fs): Langevin friction on the stiff internal modes turns the splitting error of each
step into heat that no longer cancels. It is not a conservation diagnostic there; RPMD is.

### 4. Independent code: OpenMM 8.2 `RPMDIntegrator` (PILE-L, force-group contraction)

`scripts/pimd_openmm.py`: 64 particles (1.008 amu, 300 K), a stiff anharmonic well (the O-H Morse
expansion, omega 700 /ps) in force group 0 on every bead and a soft anharmonic potential (omega 60
/ps, cubic and quartic terms) in group 1, contracted to P' beads in both codes (OpenMM's Fourier
contraction vs `contraction_matrix`); dt 0.05 fs, 1 ps of equilibration and 7.5 ps of samples (3000). Observables from bead
positions (same definitions): pgm_jax / OpenMM, difference in combined standard errors:

| case | <V_stiff> (kJ/mol/particle) | <V_soft> | <x_c^2> (1e-3 nm^2) | spread <\|q_k - q_c\|^2> (1e-4 nm^2) |
|---|---|---|---|---|
| P = 8 | 7.415 / 7.447 (-1.6) | 2.847 / 2.866 (-0.5) | 1.183 / 1.198 (-0.8) | 2.758 / 2.746 (+1.6) |
| P = 8, soft on the centroid | 7.452 / 7.482 (-1.8) | 2.980 / 2.998 (-0.5) | 1.215 / 1.227 (-0.6) | 2.906 / 2.890 (+1.8) |
| P = 8, soft on P' = 3 | 7.462 / 7.466 (-0.2) | 2.946 / 2.903 (+1.2) | 1.227 / 1.208 (+1.1) | 2.759 / 2.765 (-0.8) |
| P = 32 | 10.640 / 10.587 (+2.2) | 2.863 / 2.891 (-0.9) | 1.178 / 1.193 (-1.0) | 2.930 / 2.935 (-0.9) |
| P = 32, soft on the centroid | 10.670 / 10.621 (+2.2) | 2.965 / 3.025 (-2.0) | 1.194 / 1.225 (-2.1) | 3.059 / 3.061 (-0.3) |
| P = 32, soft on P' = 5 | 10.634 / 10.597 (+1.5) | 2.900 / 2.916 (-0.6) | 1.188 / 1.201 (-0.9) | 2.941 / 2.939 (+0.3) |
| P = 32, dt 0.025 fs | 10.626 / 10.606 (+1.0) | 2.853 / 2.878 (-0.8) | 1.171 / 1.183 (-0.8) | 2.935 / 2.932 (+0.3) |
| P = 32, dt 0.1 fs | 10.626 / 10.665 (-1.8) | 2.834 / 2.888 (-1.6) | 1.161 / 1.186 (-1.6) | 2.937 / 2.942 (-0.6) |

All 32 differences are within 2.2 standard errors (chi^2 per point 1.6), including the contraction,
which changes the spread and <V_soft> by 5 % in both codes alike.

### 5. The pGM bead engine (tests)

`tests/test_pimd.py` (14 tests, float64): vmapped beads reproduce one-by-one force evaluations of
the flexible engine (1e-7 relative); ring-polymer contraction forces are -dU/dq by finite
differences (2e-6) and P' = P is no contraction; beads in `lax.map` chunks equal all beads vmapped
(to 1e-9 after 20 steps); the Monte Carlo trial energy equals the U of the force evaluation (1e-7)
and molecular scaling keeps the bond lengths; RPMD of pGM water conserves H_P to O(dt^2);
normal-mode transform round trip, spring-matrix eigenvalues, the contraction matrix (T T^T =
(P'/P) I; a smooth path is reproduced exactly), analytic free ring-polymer motion, the harmonic
estimators, mode temperatures, the water fit.

### 6. Quantum flexible pGM water

512 waters (1,536 atoms), 298 K, NVT at 0.9887 g/cm^3 (the box of the rigid model's NPT run), mixed
precision, 0.9 nm cutoff, PME, dipole tolerance 1e-5, dt 0.25 fs, PILE-G (tau_centroid 0.1 ps, lam 1),
Cayley step; 2 ps classical and 2 ps PIMD equilibration, then 10 ps (`scripts/pimd_water.py run`).
Kinetic energies are per atom (centroid virial; primitive in brackets); the dipole is the
bead-averaged molecular dipole (<|mu|> over beads in brackets); g peaks are the bead-averaged
intermolecular g(r) (O-O first peak at 2.76 A, O-H hydrogen-bond peak at 1.81-1.84 A, H-H at 2.24-2.26 A).

| P | P' | KE_H (meV) | KE_O (meV) | <V> - <V>_cl (kJ/mol/molecule) | p (bar) | dipole (D) | g_OO / g_OH / g_HH peak | ms/step |
|---|---|---|---|---|---|---|---|---|
| 1 (classical) | | 38.52 | 38.52 | 0 | -890 +- 35 | 2.037 | 3.21 / 1.46 / 1.42 | 0.94 |
| 8 | | 118.89 +- 0.10 (118.9 +- 0.1) | 51.15 +- 0.06 | +16.6 | -965 +- 90 | 2.099 (2.133) | 3.22 / 1.38 / 1.31 | 2.44 |
| 16 | | 139.65 +- 0.07 (139.5 +- 0.2) | 53.93 +- 0.06 | +20.9 | -1319 +- 56 | 2.123 (2.163) | 3.21 / 1.36 / 1.31 | 4.39 |
| 32 | | **148.48 +- 0.07** (148.0 +- 0.3) | 55.22 +- 0.06 | +22.6 | -1143 +- 73 | 2.143 (2.185) | 3.26 / 1.39 / 1.32 | 15.5 |
| 32 | 16 | 149.37 +- 0.08 (148.8 +- 0.3) | 55.07 +- 0.05 | +22.6 | -1093 +- 115 | 2.141 (2.177) | 3.30 / 1.40 / 1.32 | 6.82 |
| 32 | 8 | 150.35 +- 0.09 (150.0 +- 0.3) | 54.91 +- 0.05 | +22.7 | -1144 +- 49 | 2.136 (2.166) | 3.31 / 1.40 / 1.32 | 4.63 |
| 32 | 4 | 151.49 +- 0.08 (151.0 +- 0.3) | 54.53 +- 0.08 | +23.1 | -1070 +- 54 | 2.116 (2.136) | 3.24 / 1.37 / 1.32 | 3.47 |
| 32 | 1 | 152.66 +- 0.06 (152.3 +- 0.3) | 53.10 +- 0.04 | +23.3 | -1116 +- 59 | 2.074 | 3.16 / 1.34 / 1.31 | 2.82 |
| 32, TRPMD (20 ps) | | 148.51 +- 0.04 | 55.25 +- 0.02 | +22.0 | | 2.155 (2.197) | | 15.0 |

(The classical g peaks from a second 10 ps CPU run: 3.32 / 1.49 / 1.44, which shows the size of their
statistical error, about +-0.05-0.1.)

- **Quantum kinetic energy of H.** 38.5 (classical), 118.9, 139.7 and 148.5 meV at P = 1, 8, 16
  and 32. The increments (20.8 and 8.8 meV) fall as for the O-H oscillator of section 1 (ratio 2.35
  here, 2.29 there), whose P = 32 value is 0.41 of the 16 -> 32 increment below the limit: the
  converged value is about 152 meV (a plain 1/P^2 extrapolation from 16 and 32 gives 151.4). For
  comparison, and with a different model: q-TIP4P/F, whose gas-phase intramolecular potential this
  water shares, gives about 143 meV at room temperature in the PIMD literature (e.g.
  Habershon, Markland & Manolopoulos 2009; Ceriotti & Markland 2013); deep inelastic neutron
  scattering values for liquid water at room temperature are about 143-156 meV depending on the
  analysis (e.g. Pantalei et al., PRL 100, 177801 (2008): 143 +- 3 meV). The higher value here suggests a
  stiffer O-H stretch in the liquid than q-TIP4P/F's: pGM's intermolecular interactions red-shift it
  less. KE_O = 55 meV (classical 38.5). The two estimators agree (primitive within 0.5 meV).
- **Structure.** Nuclear quantum effects broaden the hydrogen-bond peak of g_OH (1.46-1.49 ->
  1.36-1.39) and the H-H peak (1.42-1.44 -> 1.31-1.32); the O-O peak is unchanged within its error.
  The molecular dipole grows from 2.04 to 2.14 D (zero-point stretching of the O-H bonds with pGM's
  covalent and induced dipoles).
- **Dynamics.** Centroid diffusion (TRPMD, P = 32, 20 ps, no finite-size correction, 2.22 nm box):
  D = 2.18e-5 cm^2/s; classical flexible water with the same code (P = 1, Bussi 0.1 ps, 20 ps):
  4.08e-5 (the rigid pGM water in NVE: 4.27e-5, `docs/thermostat_ideas.md`). In this model nuclear
  quantum effects slow diffusion by about 45 %, where q-TIP4P/F finds a speed-up of 10-20 %
  (Habershon et al. 2009): the "competing quantum effects" (delocalisation weakens hydrogen bonds,
  zero-point stretching strengthens them through the larger dipole) are tipped the other way by the
  larger dipole response of pGM. The pressure at this density also drops (-890 -> -1143 bar).
  This is a property of the model (not refitted for quantum nuclei), not a validation against
  experiment (D_exp = 2.3e-5 cm^2/s).
- **Ring-polymer contraction.** The monomer reference on all beads reproduces the isolated
  molecule exactly (table above), but in the liquid the intermolecular part with its induction
  couples to the O-H stretch (the liquid red shift), and with P' = 1 or 4 the high normal modes miss
  that coupling. The error falls steadily with P': KE_H +4.2, +3.0, +1.9 and +0.9 meV for P' = 1, 4,
  8 and 16 (dipole -0.07, -0.03, -0.007, -0.002 D), while the O-O structure and the pressure
  agree within their errors already at P' = 1 (<V> within 0.7 kJ/mol per molecule). With the
  beads in chunks of 8 (7.8 ms per step for all 32), P' = 16 gains only 1.2x (KE_H within 1 meV),
  P' = 8 1.9x (2 meV), P' = 4 2.4x and P' = 1 3.2x (structure and thermodynamics). The ms/step
  column of the table was measured in these runs (one vmap over all beads, reporting included). A
  reference that also carried the stretch-dependent response to the environment's field on every
  bead would allow fewer beads (not implemented).
- **NPT** (Monte Carlo every 100 steps, 1 bar, from the NVT states, 5 ps equilibration + 10 ps):
  classical flexible water 1.0315 +- 0.0030 g/cm^3, quantum (P = 32) 1.0467 +- 0.0046 g/cm^3,
  acceptance 0.50-0.52. Quantum nuclei make this model denser, by 0.015 +- 0.01 g/cm^3 with the
  doubled errors below (its classical flexible version is already denser than the rigid model it
  comes from, 1.006, and than experiment, 0.997), in line with the lower quantum pressure at fixed
  volume. KE_H at the NPT density: 148.52 +- 0.08 meV (NVT 148.48).
  Short runs: the volume correlation time is a few ps, so these densities carry about twice the
  quoted block errors.


## Speed

One RTX PRO 6000 Blackwell, 512 flexible pGM waters (1,536 atoms), mixed precision, 0.9 nm cutoff,
PME 30^3, dipole tolerance 1e-5, dt 0.25 fs, PILE-G; ms per step (ns/day), and CG iterations per
step (the batched solve runs until its slowest bead converges). `scripts/pimd_water.py bench`:

| P | all force beads | P' = 16 | P' = 8 | P' = 4 | P' = 1 |
|---|---|---|---|---|---|
| 1 (classical flexible) | 0.94 (23), 3.0 CG | | | | |
| 8 | 2.19 (9.9), 8.0 CG | | | 1.97 (11.0) | 1.44 (15.0) |
| 16 | 4.13 (5.2), 8.0 CG | | 3.46 (6.2) | | |
| 32 | **7.81 (2.8)**, 9.0 CG (15.2 in one vmap) | 6.64 (3.3) | 4.21 (5.1), 8.0 CG | 3.19 (6.8), 7.0 CG | 2.46 (8.8), 3.2 CG |
| 64 | 16.1 (1.3) | 7.92 (2.7) | | | |

- Beads batch well up to 16 per program (0.26 ms per bead against 0.94 ms for one classical
  step); 32 beads in one vmap are twice as slow as in four chunks of 8 run one after the other, hence
  `bead_chunk="auto"`.
- Contraction saves the pGM part on the removed beads but adds the monomer reference on every bead
  (gas-phase pGM of every molecule with its 9 x 9 induction solve, about 2.5 ms per step for 32 x 512
  molecules): P' = 8 is 1.9x faster than all 32 beads, P' = 1 3.2x. The centroid evaluations are
  smooth, so their dipoles need 3 CG iterations instead of 8-9.
- For comparison: rigid pGM water runs at 130 ns/day (1 fs, README); the flexible O-H stretch and the
  0.25 fs step cost 4x in time step, the beads another 8x (P = 32).
- CPU (48 cores of one node), P = 32 contracted to 1: 90 ms per step.


## Limits

- **Flexible molecules only.** Rigid bodies are not ring polymers of atoms (a rigid-rotor path
  integral needs rotational propagators and is not implemented), and constraints (SHAKE / RATTLE on
  beads) and virtual sites are refused. Quantum water therefore needs a flexible model, such as
  `models.water.flexible_water`. Rigid-water results of the repository are classical.
- NVT and isotropic Monte Carlo NPT (molecular centroid scaling). No multiple time stepping,
  restraints, alchemical regions or replica exchange with beads; biases on collective variables
  (`pgm_jax/bias`), external fields (`efield=`) and extended-Lagrangian dipoles (`MDSettings.induction.iel`)
  are refused as well (docs/CHANGES_2026-09.md).
- The dipole predictor is per bead: with the thermostat on the internal modes (and their
  high-frequency motion) the bead evaluations need 8-9 CG iterations per step at tolerance 1e-5
  (a batched solve runs until its slowest bead converges), against 3 for evaluations on the centroid
  and 3-5 in classical flexible MD.
- Ring-polymer contraction needs P' >= 8-16 of 32 beads for the quantum kinetic energy of flexible
  pGM water (section 6); fewer beads suffice for structure and thermodynamics.
- The time step is set by the flexible O-H stretch: 0.25 fs here (0.5 fs is common with q-TIP4P/F;
  not tested for pGM water).
- The conserved quantity is only a diagnostic for RPMD (no thermostat), see section 3; the
  kinetic temperature of the beads is 0.6 K low at 0.25 fs (the BAOAB kinetic-energy bias of the
  stiff stretch), which does not enter the estimators (they use T).
- Correlation functions of the centroid (IR spectra from the bead-averaged dipoles, velocity
  autocorrelation) are left to post-processing: `run(traj_every=...)` writes the centroid trajectory,
  `beads_traj_every` every bead, `molecular_dipoles()` the bead-averaged molecular dipoles.
- The flexible pGM water is a new model built for this validation, not refitted for quantum nuclei:
  at the rigid model's density its classical and quantum pressures are -900 and -1150 bar, and its
  quantum effects slow diffusion (section 6).
