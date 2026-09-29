# Enhanced sampling: metadynamics and OPES with autodiff collective variables

`pgm_jax/bias/` adds biases on collective variables (CVs) to both MD engines (`Simulation`,
`FlexibleSimulation`): static biases (umbrella windows, walls, analytic model potentials),
well-tempered metadynamics and OPES_METAD, several walkers in one compiled program, and the
analysis that turns biased runs into free energy surfaces. A CV is any JAX function of the
positions and the box; its derivatives, and so every bias force, come from `jax.grad` of
V(s(x)). Nothing is hand-differentiated, and a new CV is a few lines.

## Usage

```python
from pgm_jax.bias import BiasSet, MetaD, OPES, Harmonic, UpperWall, cv

phi, psi = cv.Dihedral(4, 6, 8, 14), cv.Dihedral(6, 8, 14, 16)  # periodic, rad
metad = MetaD(
    [phi, psi], sigma=0.35, height=1.2, pace=250, biasfactor=6.0, grid=(-np.pi, np.pi, 128)
)  # well-tempered, hills on a grid
sim = FlexibleSimulation(
    sys,
    templates,
    pos,
    H,
    MDSettings(),
    dt=0.002,
    temperature=300.0,
    constraints="h-bonds",
    bias=BiasSet([metad], colvar=100),
)
sim.run(5_000_000, report=5000, restart=50000, prefix="ala2")  # ala2.colvar, ala2.hills, ala2.bias, ala2.chk
sim.observables()["ebias"], sim.observables()["hills"]  # also bias_work (energy pumped in)
sim.cv_values(), sim.bias_energies()  # current CVs and V of each bias
sim.load_bias("ala2.bias")  # continue with a saved bias state
```

Walkers, umbrella windows and the analysis:

```python
from pgm_jax.bias.walkers import Walkers
from pgm_jax.bias import analysis as A
from pgm_jax.bias.io import read_table

wk = Walkers(sim, 12)  # 12 independent metaD runs in one program
wk = Walkers(sim, 8, shared=True)  # or multiple walkers of one bias
h = Harmonic(phi, at=0.0, kappa=150.0)  # umbrella windows: one centre per walker
sim = FlexibleSimulation(..., bias=BiasSet([h], colvar=100))
wk = Walkers(sim, 24, bias_states=[sim.state.bias._replace(parts=(h.state(at=c),)) for c in centres])
wk.run(750000, report=5000, restart=250000, prefix="us")  # us_wNN.colvar, us.walkers.chk

_, c = read_table("md_w00.colvar")
_, hl = read_table("md_w00.hills")  # step, time_ps, phi, psi, bias0_metad
steps, ct = A.metad_ct(hills, biasfactor, kT, periods, grid_points)  # c(t) after every hill
logw = A.ct_weights(c["step"], c["bias0_metad"], steps, ct, kT)
F = A.histogram_fes(np.stack([c["phi"], c["psi"]], 1), logw, [axis, axis], kT, periods)
F_bias = A.fes_from_bias(metad, state, points)  # -gamma/(gamma-1) V
F_wham, f = A.wham(samples, centres, kappas, axis, kT, period=2 * np.pi)
```

- **CVs** (`bias/cv.py`): `Distance`, `Angle`, `Dihedral` (periodic, IUPAC sign as the
  restraints), `Component`, `COMDistance` (mass-weighted groups), `Coordination` (PLUMED's
  rational switching function, all pairs of two groups), `RMSD` (optimal superposition by the
  quaternion method; the largest eigenvalue is differentiated by Hellmann-Feynman, so it stays
  regular when lower eigenvalues are degenerate; or translation only), `Linear` (combinations),
  `Custom(fn, period=...)` (any JAX function). Minimum images in the box for every difference.
- **Biases** (`bias/core.py`):
  - `StaticBias(cvs, fn)`, `Harmonic(cvs, at, kappa)` (V = kappa/2 ds^2, PLUMED's convention;
    centre and force constant are state variables, so umbrella windows share one compiled step
    and a centre can be moved without recompiling), `UpperWall` / `LowerWall` (PLUMED walls).
  - `MetaD(cvs, sigma, height, pace, biasfactor, grid=None)`: well-tempered metadynamics
    (Barducci, Bussi, Parrinello 2008; `biasfactor=None`: standard). Hills live in a device
    buffer (every hill evaluated each step, O(M)); with `grid=(lo, hi, bins)` (1 or 2 CVs) they
    are also accumulated on a grid of values and derivatives and read by cubic Hermite
    interpolation, O(1) per step whatever the number of hills. The interpolant is C1 and the
    forces are its exact gradient, so a static grid bias conserves energy too.
  - `OPES(cvs, sigma, pace, barrier, ...)`: OPES_METAD (Invernizzi and Parrinello 2020) as in
    PLUMED: compressed kernel density estimate of the unbiased distribution, bandwidth
    rescaling with N_eff, V = (1 - 1/gamma) kT log(P/Z + eps), Z recomputed after each
    deposition. Checked against a line-by-line re-implementation of PLUMED's update (1e-9).
  - `BiasSet(biases, colvar=n)`: the biases of one simulation, with a COLVAR buffer on the device.
- **Walkers** (`bias/walkers.py`): `Walkers(sim, W, shared=...)` runs W copies of the system as
  one vmapped program (the replica engine of `md/remd.py`): independent runs with their own
  biases (error bars; umbrella windows with `bias_states=`), or multiple-walker metadynamics /
  OPES with one shared bias.
- **Analysis** (`bias/analysis.py`): FES from the bias (-gamma/(gamma-1) V for metaD, -V/(1-1/gamma)
  for OPES), c(t) of Tiwary and Parrinello (2015) and frame weights exp((V - c(t))/kT), OPES
  weights exp(V/kT), weighted histograms of any CVs, 1D WHAM, RMSD after the best shift.
- **Model engine** (`bias/toy.py`): BAOAB Langevin of particles in analytic potentials (double
  well, Mueller-Brown, a particle on a ring) with the same biases, W walkers per program.

### How the bias enters the engines

- `MDState.bias` holds the bias state (hills, grid, kernels, normalisation, COLVAR buffer,
  work). The force evaluation adds V(s(x)) and its autodiff gradient together with the
  restraints (one `value_and_grad` for both), before the atomic forces are mapped to rigid
  bodies or spread from virtual sites.
- After each step, inside the compiled `fori_loop`: the COLVAR row (every `colvar` steps), then
  the updates due at this step (`lax.cond`). When a bias changes, the forces and `epot` of the
  state are corrected at once to the new bias at the same positions (a bias-only gradient), so
  the next half-kick uses the new bias; the change of V at fixed x is booked as heat (so
  `econs` stays conserved) and in `bias_work`. There are no host round-trips: the host only
  enlarges the buffers between blocks (`reserve`, which may re-trace) and collects COLVAR rows.
- The Monte Carlo barostat's trial energy and the pressure (virial by the molecular strain
  derivative) include the bias. With multiple time stepping the bias is in the slow group.
- Checkpoints (`prefix.chk`) contain the bias state; `prefix.bias` (`BiasSet.save/load`,
  `Simulation.load_bias`) carries a bias to another run, e.g. a static run with a converged bias.
- Replica exchange accepts static biases (the same in every replica); a time-dependent bias
  raises (it would have to stay with its temperature slot and enter the exchange criterion).

## Validation

All numbers below come from the scripts in `scripts/bias/` (outputs in `runs/bias/`, `runs/ala2/`).

### 1. Model potentials with exact free energies (`scripts/bias/validate_toy.py`)

BAOAB Langevin (`bias/toy.py`) at 300 K, mass 10 amu, friction 2/ps, dt 5 fs, 20 ns per run, with
8 independent runs (8 walkers, each with its own bias, in one program) and one run of 8 walkers
sharing a bias. MetaD: well-tempered, biasfactor 10, height 1 kJ/mol every 1 ps, on a grid. OPES:
barrier 30 kJ/mol, pace 1 ps. The FES comes from the final bias and from the reweighted
histogram (c(t) weights for metaD, exp(V/kT) for OPES; first 20 % discarded). It is compared with
the exact FES over the region F_exact < F_max, after the best constant shift. Columns: mean RMSD
of the 8 runs +- its standard error; RMSD of the average of the 8 FES (with the mean pointwise
error bar of that average); one run of 8 shared walkers. kJ/mol throughout.

| System (CVs, region) | Method | RMSD per run: bias | RMSD per run: reweighted | Average of 8, bias / reweighted (error bar) | 8 shared walkers, bias / reweighted |
|---|---|---|---|---|---|
| double well, barrier 25 (x; F < 20) | metaD | 0.209 +- 0.024 | 0.162 +- 0.009 | 0.073 / 0.088 (0.08 / 0.05) | 0.075 / 0.085 |
| | OPES | 0.225 +- 0.008 | 0.169 +- 0.011 | 0.153 / 0.079 (0.06 / 0.06) | 0.125 / 0.082 |
| particle on a ring (periodic angle; F < 25) | metaD | 0.290 +- 0.026 | 0.201 +- 0.017 | 0.102 / 0.062 (0.10 / 0.07) | 0.079 / 0.060 |
| | OPES | 0.231 +- 0.019 | 0.206 +- 0.014 | 0.074 / 0.073 (0.08 / 0.07) | 0.207 / 0.052 |
| Mueller-Brown x 0.25 (x, y; F < 30, 2,700 bins) | metaD, 20 ns | 0.634 +- 0.011 | 0.726 +- 0.005 | 0.240 / 0.334 (0.21 / 0.23) | 0.303 / 0.302 |
| | metaD, 60 ns | 0.436 +- 0.009 | 0.442 +- 0.007 | 0.158 / 0.236 (0.15 / 0.14) | |

Every 1D run is within 0.3 kJ/mol of the exact FES, and the averages of 8 runs agree with it
within their error bars. The 2D surface covers a 30 kJ/mol window with 2,700 bins; there, 60 ns
per run reach 0.44 kJ/mol. The periodic CV shows no seam at +-pi (the hills and kernels use the
nearest image, and the grid is periodic). OPES on the 2D Mueller-Brown model did not finish: with
bandwidth rescaling its kernels keep shrinking and multiplying in 2D, and Z costs O(K^2) per
deposition. A 20 ns run of 8 walkers reached 10 ns in 35 min on 16 cores, then slowed as K grew
and hit the 4 h job limit (see Limits).

### 1b. The same through the MD engine (`scripts/bias/engine_dw.py`)

Two non-interacting atoms (no charge, polarizability or van der Waals) in a 3 nm box, run by the
rigid-body engine (`Simulation`, Langevin 1/ps, 2 fs, float64). An external double well
U(r) = 15 ((r - 0.8)^2 / 0.25^2 - 1)^2 kJ/mol acts on their distance (a `StaticBias`), with walls
at 0.3 and 1.4 nm. The exact FES is F(r) = U(r) + walls - 2 kT ln r. Well-tempered metaD on r
(sigma 0.03 nm, 1 kJ/mol every 1 ps, biasfactor 8), 16 independent walkers of 3 ns in one
program (`Walkers`, 4,160 ns/day on 8 CPU cores). Region F < 20 kJ/mol:

| Estimate | RMSD per walker | RMSD of the 16-walker average (mean error bar) |
|---|---|---|
| final bias, -gamma/(gamma-1) V | 0.333 +- 0.013 kJ/mol | 0.143 (0.078) |
| c(t)-reweighted histogram | 0.310 +- 0.010 kJ/mol | 0.181 (0.064) |

The averages have chi^2 per point of 3.3 and 8.3: a residual of about 0.1-0.15 kJ/mol
(0.05 kT) that 3 ns runs do not average away. Its largest values (0.4-0.7) are at the steep walls.

### 2. Alanine dipeptide with pGM electrostatics (`scripts/bias/ala2.py`, `ala2_analyze.py`)

ACE-ALA-NME from tleap (ff19SB-form bonded terms with CMAP, placeholder pGM electrostatics with
induction, as `scripts/protein/remd_peptide.py`), alone in a 3.2 nm box (cutoff 1.2 nm, Ewald
coefficient 2.5/nm, PME 20^3, so its images are beyond the cutoff), flexible engine, X-H
constraints, Langevin 2/ps, 2 fs, 300 K, mixed precision.

- **Metadynamics:** well-tempered on (phi, psi), sigma 0.35 rad, 1.2 kJ/mol every 1 ps, biasfactor 6,
  128^2 periodic grid; 24 independent walkers (12 x 4 ns on the GPU in one program, at 1,930 ns/day
  aggregate, plus 12 x 3 ns on CPUs).
- **OPES:** OPES_METAD on (phi, psi), sigma0 0.2 rad, barrier 50 kJ/mol, pace 1 ps; 24 independent
  walkers (12 x 3 ns on the GPU, 12 x 2-3 ns on CPUs).
- **Reference, REMD:** temperature replica exchange (`md/remd.py`, batched), 8 replicas 300-700 K
  (geometric), exchanges every 0.5 ps (neighbour acceptance 0.64-0.70), 4 ns per replica on the GPU plus two independent 1.5 ns runs on CPUs (31,500 frames at 300 K).
  Frames every 0.2 ps of the 300 K slot, first 10 % discarded. Errors from 5 blocks.
- **Umbrella sampling in phi** (psi free), 24 windows 15 deg apart (kappa 150 kJ/mol/rad^2), each
  steered from the minimum to its centre over 50 ps and run for 1.5 ns (all windows of a job are
  walkers of one program, `bias_states=`), 1D WHAM.

F(phi) comes from the reweighted runs (first 20 % discarded, 10 deg bins), with errors from the
spread of the independent runs. Delta G = -kT ln(P(phi > 0) / P(phi < 0)) is the free energy
of the C7ax / alpha_L side.

| Method | Delta G(phi > 0), kJ/mol | F(phi) vs REMD: RMSD (max), chi^2 per bin | F(phi, psi) vs REMD: RMSD (max) |
|---|---|---|---|
| REMD, 300 K slot (reference) | 7.75 +- 0.26 | | |
| metaD, 24 runs | 7.88 +- 0.03 | 0.17 (0.38), 0.33 | 0.41 (1.17), 167 bins |
| OPES, 24 runs | 7.90 +- 0.08 | 0.17 (0.37), 0.39 | 0.43 (1.11), 161 bins |
| metaD, final bias -gamma/(gamma-1) V | | 0.35 | |
| umbrella in phi, WHAM | 7.40 +- 0.04 | 1.07 (2.86) | |

REMD comparisons use the bins with F_REMD < 12 kJ/mol (20 bins; at 300 K the barrier regions
get too few samples); the 2D comparisons use bins with F < 15 in both surfaces. MetaD and OPES
agree with REMD within its errors (chi^2 per bin below 1) and with each other to 0.18 kJ/mol
over 166 2D bins (max 0.43). Per run (3-4 ns), F(phi) is within 0.29 (metaD) and 0.40 (OPES)
kJ/mol RMSD of REMD.

**Umbrella sampling along phi alone fails as a reference here**, and the other runs show why. In
the windows centred at phi = -52.5 ... -7.5 deg, psi stays in the basin it was steered into for
all 1.5 ns (the fraction of frames with psi in (-126, 17) deg is exactly 0 or 1 per window, with
0-6 transitions against 129 and 2,508 in neighbouring windows). WHAM then misses the alpha_R /
C7eq exchange there: its F(phi) is up to 2.5 kJ/mol too high between -65 and -25 deg, and Delta G
comes out 0.5 kJ/mol too low. Metadynamics and OPES bias psi as well, and REMD crosses the psi
barrier at high temperature, so the three agree. A 1D umbrella reference would need replica
exchange between windows or a second biased CV.

### 3. Forces, energy conservation, tests

- **Forces vs finite differences** (`tests/test_bias.py`): the gradients of every CV (distance,
  angle, dihedral, coordination, COM distance, RMSD with and without alignment, linear, custom)
  in a skewed triclinic box, and the bias forces of a metaD + wall set in both engines, agree
  with central differences (2e-7 relative). The state's forces minus the unbiased ones equal
  -dV/dx, mapped to rigid-body forces and torques. After deposition inside the loop, the stored
  forces equal a fresh evaluation to 1e-7.
- **NVE** (`scripts/bias/nve_check.py`): eight pGM waters in a 3 nm box (no pair crosses the
  cutoff), float64, 4 ps. The CVs are the O-O distance and an O...H coordination number, with a
  wall at 0.6 nm. Maximum deviation of the conserved quantity (kJ/mol):

| Engine, dt | No bias | Static bias (40 hills; E_bias range) | Growing bias (hill every 200 steps; work done) |
|---|---|---|---|
| rigid bodies, 1 fs | 0.022 | 0.021 (4.1) | 0.017 (19.5) |
| rigid bodies, 0.5 fs | 0.020 | 0.020 (4.1) | 0.015 (37.0) |
| atoms + SHAKE, 1 fs | 0.020 | 0.025 (11.5) | 0.012 (19.3) |
| atoms + SHAKE, 0.5 fs | 0.021 | 0.019 (11.2) | 0.008 (36.4) |

  A static bias leaves energy conservation exactly as it is without one: the bias energy moves
  over 4-11 kJ/mol while the total stays within the unbiased run's 0.02 kJ/mol. With deposition,
  E_tot rises by the work of the updates (up to 37 kJ/mol). Because that work is booked as heat,
  `econs` stays within 0.02. A dihedral through two molecules (H-O...O-H) is a poor NVE CV: its
  gradient diverges when H-O...O is collinear, and one collision gave errors of up to 2 kJ/mol.
  This comes from the CV, not from the hook.
- **pytest** (`tests/test_bias.py`, 23 tests, ~4 min on 8 cores): CV values (dihedral =
  restraints, RMSD = Kabsch), static biases, metaD heights / periodicity / hill sum, grid vs hill
  sum (1D, 2D, periodic, C1 continuity), OPES against a line-by-line re-implementation of PLUMED's
  OPES_METAD (bias every step, kernels, Z, sum of weights: 1e-9; with and without recursive
  merging; full buffer), BiasSet COLVAR / save / load, MD forces and static NVE in both engines,
  deposition in the loop with NVE bookkeeping, files, checkpoint continuation, OPES in NVT, the
  pressure's bias virial, multiple time stepping, a flexible peptide with a phi/psi grid, REMD
  refusing a dynamic bias, walkers (independent = single runs; shared = W hills per pace),
  model engine with c(t), WHAM.


## Speed

Cost of the bias per MD step, from `scripts/bias/ala2.py bench` (same system and settings with
and without the bias; hills every 250 steps, COLVAR every 100 steps):

| System | Device | No bias (ms/step) | metaD, grid | metaD, hill list | OPES |
|---|---|---|---|---|---|
| alanine dipeptide in pGM water, 1,546 atoms, 2 fs | RTX PRO 6000 Blackwell | 1.103 | 1.187 (+7.7 %) | 1.180 (+7.0 %) | 1.092 (-1 %) |
| same, dipeptide alone (22 atoms) | RTX PRO 6000 Blackwell | 0.755 | 0.784 (+3.9 %) | 0.780 (+3.3 %) | 0.701 (-7 %) |
| alanine dipeptide in pGM water | 16 CPU cores | 43.7 | 43.0 (-2 %) | 44.1 (+1 %) | 43.7 (0 %) |

Differences of a few per cent either way are noise: each bias drives the peptide to other
conformations, which changes the CG iteration count. An earlier version evaluated two `lax.cond`s
after every step (is a COLVAR row due? an update?). On a GPU a conditional reads its predicate on the
host, and those two cost 0.1 ms per step (+11-12 % on both GPU systems). The step loop is now
split into plain inner loops of `stride` = gcd(colvar, paces) steps, with the bias work between
them (`md/integrate.strided_loop`).

The bias itself is a handful of small kernels: the CVs and their gradient (autodiff), a grid
lookup or a sum over hills or kernels, and one `value_and_grad` shared with the restraints.
Deposition adds a bias-only gradient every `pace` steps. On a 22-atom system that fixed cost is
visible (10 % on the GPU). For solvated systems it is lost in the force field. Walkers make small
systems efficient: 16 walkers of the vacuum dipeptide run at 1,390 ns/day aggregate on one GPU
(87 ns/day each; a single simulation runs at 230 ns/day), and 3 walkers at 143 ns/day on 8 CPU
cores. Model engine (CPU, 8 walkers): 8-10 us per step with a 1D grid, 70-115 us with a 2D grid
of 93,000 nodes or with OPES (the kernels are summed every step).

## Limits

- Grids for 1 or 2 CVs. With more CVs the hills are summed every step, O(number of hills).
- OPES: every kernel is evaluated at every step, and Z is recomputed from all kernel pairs (O(K^2))
  after each deposition. That is cheap for the ~10^2 kernels of the dipeptide, but slow for long
  2D runs where compression cannot keep K small (the Mueller-Brown model above). An incremental Z
  and a kernel neighbour list, as in PLUMED, are not implemented.
- OPES: OPES_METAD with fixed sigma0 or PLUMED's bandwidth rescaling. There is no adaptive sigma
  (sigma from CV fluctuations), no OPES_EXPLORE / OPES_EXPANDED, and no neighbour list over kernels.
- A non-periodic grid gives a flat bias (zero force) beyond its range; keep the CV inside with a
  wall. `Component` CVs are not wrapped, so use them without periodic boundaries or with molecules
  kept near the origin.
- Multiple walkers run in one process (one device, `jax.vmap`). They are NVT only, with no
  multiple time stepping, and not across processes or nodes. `Coordination` evaluates all
  |A| x |B| pairs, without a neighbour list.
- Replica exchange accepts static biases only. Parallel-tempering metadynamics (a bias per
  temperature slot inside the exchange criterion) is not implemented.
- No py-plumed interface (PLUMED is not installed in the environment).
- COLVAR rows are buffered on the device and written at the end of each block. The hills file is
  written at block ends. The bias temperature is one value per bias (the simulation's by default).
- Validation 2 is in vacuum. The solvated dipeptide was used only for the speed measurement,
  because GPU time was not available for a solvated FES comparison.
