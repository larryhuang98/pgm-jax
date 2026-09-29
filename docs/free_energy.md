# Alchemical free energies with pGM

`pgm_jax/md/alchemy.py` (lambda-dependent Hamiltonians, batched lambda windows, Hamiltonian replica
exchange, the gas-phase leg) and `pgm_jax/analysis/free_energy.py` (TI, BAR, MBAR, statistical
inefficiency) compute solvation free energies of small molecules, rigid (`Simulation`) or flexible
(`FlexibleSimulation`, e.g. a fitted methanol among rigid waters).
`scripts/solvation_free_energy.py` runs the whole protocol (`run`), analyses it (`analyze`) and
measures its cost (`bench`). Units: nm, ps, kJ/mol, K, e (results also in kcal/mol).
Parameter gradients of these free energies (fitting targets): `docs/fe_gradients.md`.

```python
from pgm_jax.md import free_energy as fe
from pgm_jax.md.alchemy import Alchemy, FreeEnergyRun, GasPhaseLeg, LambdaWindows, alchemical_system, standard_schedule
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.simulation import Simulation

sys, P = alchemical_system(sys, solute=0)  # the solute gets its own parameter keys
alch = Alchemy(sys, solute=0)  # soft core alpha 0.5, polarizability floor 1e-8
sim = Simulation(sys, pos, H, MDSettings(), params=P, alchemy=alch, thermostat="bussi", dt=0.002)
gas = GasPhaseLeg(alch, sim.positions()[sys.atom_slice(0)], "qpi")
L = standard_schedule(8)  # 8 electrostatics + 11 van der Waals windows
run = FreeEnergyRun(
    LambdaWindows(sim, L),
    sample_every=500,
    exchange_every=500,
    meta={"gas_delta_g": gas.delta_g(P), "gas_dudl": [gas.dudl(l, P) for l in L[:, 0]]},
)
run.run(1000000, prefix="wat", report_every=10000, checkpoint_every=50000)  # 2 ns per window
r = fe.estimate(
    fe.load("wat_fe.npz"), discard_ps=200, gas={"delta_g": gas.delta_g(P), "dudl": [gas.dudl(l, P) for l in L[:, 0]]}
)
r["dG_hyd_mbar_kcal"], r["dG_hyd_mbar_err_kcal"]
```

```bash
python scripts/solvation_free_energy.py run --model pgm -o runs/fe/pgm --ns 2          # water in pGM water
python scripts/solvation_free_energy.py run --prmtop x.prmtop --coords x.rst7 --solute 0 -o runs/fe/x
python scripts/solvation_free_energy.py run --solute-template methanol.flex -o runs/fe/meoh   # flexible solute
python scripts/solvation_free_energy.py analyze runs/fe/pgm_fe.npz --discard-ps 200
python scripts/solvation_free_energy.py bench --model pgm --windows 4,8,19
```

## Thermodynamic cycle and schedule

    A(gas)      --- Delta G_gas(1 -> 0) --->  A(gas, no electrostatics)
      |                                             | 0
    A(solution) --- Delta G_solv(1 -> 0) -->  A(solution, decoupled)

    Delta G_hyd = Delta G_gas(1 -> 0) - Delta G_solv(1 -> 0)

The decoupled solute is an ideal-gas molecule in either phase, so the right-hand leg is zero at
equal number density: the result is in Ben-Naim's standard state (1 M gas -> 1 M solution), the
one of tabulated experimental hydration free energies. Windows run at constant volume (after NPT
equilibration at full coupling, the box set to the mean volume). The Helmholtz free energy of the
transfer at that volume differs from the Gibbs free energy at 1 bar by P Delta V (4e-4 kcal/mol
for a water) and by the compression of the solvent, Delta V^2 / (2 V kappa_T) with Delta V the
solute's partial molar volume: 0.009 kcal/mol for a water in 512 (V = 15.6 nm^3), smaller than
the statistical errors below and shrinking as 1/V.

The solution leg has two stages (`standard_schedule(n_elec, vdw)`): the solute's electrostatics is
switched off with its van der Waals on (lambda_elec 1 -> 0 at lambda_vdw = 1, n_elec evenly spaced
windows), then its van der Waals with the environment (lambda_vdw 1 -> 0 at lambda_elec = 0,
default 0.9, 0.8, ..., 0.1, 0.05, 0). Charges are never on atoms that others can overlap, which is
what makes the second stage safe with a soft core (the order of Tinker's and OpenMM's AMOEBA
protocols, and of most fixed-charge protocols, e.g. Shirts & Pande, JCP 122, 134508 (2005)).

## What lambda does to each pGM term

| Term | lambda_elec (annihilation) | lambda_vdw (decoupling) |
|---|---|---|
| Gaussian charges q_s of the solute | q_s lambda_e | - |
| covalent dipoles (strengths c_s, i.e. permanent dipoles p_s) | c_s lambda_e | - |
| polarizabilities alpha_s | alpha_s [eps + (1 - eps) lambda_e], eps = 1e-8 | - |
| Gaussian radii (damping widths) | unchanged | - |
| solute-environment Lennard-Jones | - | Beutler soft core, lambda_v eps [1/w^2 - 2/w], w = (r/rmin)^6 + (alpha_sc/2)(1 - lambda_v) |
| its long-range correction | - | lambda_v x (the solute's share of the r^-6 tail) |
| intramolecular van der Waals of the solute (flexible) | - | unchanged (evaluated with the soft-core rows, unscaled) |
| intramolecular electrostatics, `intramolecular="annihilate"` | scaled with the rest (removed by the gas-phase leg) | - |
| same, `intramolecular="keep"` | kept: + E_gas(x_s; 1) - E_gas(x_s; lambda_e) | - |

**Electrostatics are annihilated**, as the ele-lambda of AMOEBA free energies in Tinker (Ren &
Ponder, JCC 23, 1497 (2002); JPC B 107, 5933 (2003); Shi, Wu, Ponder & Ren, JCC 32, 967
(2011)): the permanent multipoles and polarizabilities of the solute are scaled, so
every electrostatic interaction of the solute goes, its intramolecular ones included; in pGM those
are every intramolecular pair (no exclusions) and the solute's own induction. The gas-phase leg
removes the same intramolecular energy in vacuum. Decoupling by excluding the solute's
intermolecular pairs only (as fixed-charge codes do) does not carry over: PME and the induction
solve couple the solute's intra- and intermolecular terms through one charge density and one
linear system; the "keep" mode below reaches the decoupled end state another way. The radii are
not scaled: they are damping widths, not interaction
strengths, and keeping them keeps the dipole-dipole tensor fixed.

**Keeping the intramolecular electrostatics (`intramolecular="keep"`).** The Hamiltonian at lambda
can instead carry the correction E_gas(x_s; lambda_e = 1) - E_gas(x_s; lambda_e): the solute's
electrostatic energy in vacuum (the dense gas-phase pGM model of the lone solute, float64, at its
current geometry, every step, forces by autodiff) at full minus reduced coupling. The solute then
keeps its whole gas-phase intramolecular electrostatics at every lambda and only its coupling to
the environment (and to its periodic images) is switched off: the decoupled end state is the
gas-phase molecule, and Delta G_hyd = -Delta G_solv(1 -> 0) with no separate gas leg. Flexible pGM
molecules need this: their bonded terms were fitted together with the electrostatics of every
intramolecular pair (pGM has no exclusions), and annihilated they leave a molecule held by its
bonded terms alone (the fitted methanol at lambda_e = 0 in vacuum blew apart within 14 steps). The
correction costs a dense solve of 3 n_solute unknowns per step (negligible for small molecules).
For a rigid solute it is a constant at each lambda, so both modes give the same free energy (tests:
U_keep - U_annihilate = E_gas(1) - E_gas(lambda) to 1e-7 kJ/mol); with it dU/dlambda_e is the
gas-subtracted integrand below. This is the pGM analogue of decoupling (as opposed to annihilating)
a fixed-charge solute's electrostatics, done without separating intra- and intermolecular pairs in
PME and the induction solve, which pGM couples.

**The polarizability floor.** The induced dipoles minimise |mu|^2 / (2 alpha) - mu.E + mu T mu / 2.
Scaling alpha_s by lambda keeps the solve well posed for every lambda > 0 (the Jacobi-preconditioned
CG operator is I - alpha^1/2 T alpha^1/2, whose coupling only shrinks with alpha_s), but alpha_s = 0
puts 1/0 into the operator and into the polarization energy. Tinker masks the solute's induced
dipoles at lambda = 0 (`douind = .false.`), OpenMM's AMOEBA plugin tolerates a zero polarizability
because it iterates mu = alpha E. Here alpha_s(0) = eps alpha_s with eps = 1e-8 (`alpha_floor`):
the solve stays regular, U(lambda) is smooth up to the endpoint, dU/dlambda_e at lambda_e = 0 is
the exact limit (-(1 - eps) alpha_s |E|^2 / 2 per site plus the permanent terms, finite), and the
endpoint differs from masked dipoles by eps times the solute's induction energy (about 1e-7
kJ/mol; the decoupled state equals the box without the solute to 1e-9 kJ/mol, below).

**Van der Waals: Beutler soft core** (Beutler, Mark, van Schaik, Gerber & van Gunsteren, CPL 222,
529 (1994)) with alpha_sc = 0.5 and power 1, the choice of Shirts & Pande (2005), in Amber's rmin
form: lambda_v eps [1/w^2 - 2/w], w = (r/rmin)^6 + (alpha_sc/2)(1 - lambda_v), which is 4 eps
lambda_v [1/y^2 - 1/y], y = alpha_sc (1 - lambda_v) + (r/sigma)^6. It is Lennard-Jones at lambda_v =
1 and finite (with zero force) at r = 0 for lambda_v < 1. Only solute-environment pairs are soft;
AMOEBA uses Halgren's buffered 14-7 with its own soft core (Jiao, Golubkov, Darden & Ren, PNAS 105,
6290 (2008)) for the same purpose; pGM's van der Waals is Lennard-Jones. Beyond the cutoff the soft
core differs from lambda_v x LJ by (alpha_sc/2)(1 - lambda_v)(rmin/rc)^6 < 1e-3 of the term, so the
long-range correction of those pairs is scaled by lambda_v. GVDW (finite at overlap, it would need
no soft core) is refused for now.

**Derivatives.** The energy is variational in the induced dipoles, so dU/dlambda is the partial
derivative at the converged dipoles (Hellmann-Feynman): `Alchemy.dudl`, jax.grad of the fixed-mu
energy with respect to (lambda_e, lambda_v), checked against central differences of the energy with
the dipoles re-solved at every lambda (tests: 1e-9 relative inside, second-order one-sided
differences at the ends).

## Implementation and engine impact

The Hamiltonian at lambda = the ordinary force field (`forcefield.py`, unchanged) with the solute's
parameters at lambda_e and its van der Waals parameters set to zero, plus the soft-core term
evaluated in a dedicated small row set: each solute atom's candidate row from the neighbour list
(n_solute x C pairs, float64, forces by autodiff), the solute's intramolecular van der Waals pairs
(flexible molecules; from the special-pair table of `md/topology.py`, unscaled) and, with
`intramolecular="keep"`, the gas-phase correction. No pair kernel of the engine changes, which is
why the solute's tied parameters must be its own (`alchemical_system` copies the molecule with
every tying key prefixed `alch:`; `Alchemy` refuses shared keys). Hooks in shared files:

- `integrate.py`: `MDState.lam` ((2,) or None); `Integrator(alchemy=)`; `_forces(..., lam=)`
  computes through `Alchemy.compute` when an alchemical region is set; the Monte Carlo barostat's
  trial energy through `Alchemy.energy`.
- `simulation.py`: `Simulation(alchemy=)`; the pressure through `Alchemy.strain_derivative`; the
  cell-dipole recorder is refused with an alchemical region (it would not scale the solute).
- `flexible.py`: the same three hooks in `FlexibleIntegrator` (`_forces(..., lam=)`, the step, the
  barostat's trial energy), `FlexibleSimulation(alchemy=)`, its pressure through
  `Alchemy.strain_derivative`.

Without an alchemical region every hook is a Python branch that is not taken, so the compiled step
is the same program: a plain NVT (mixed) and NPT (double) run of 60 steps gives bit-identical
positions, box and energy on this branch and on master.

**Windows on the GPU** (`LambdaWindows`, a subclass of `remd.MDReplicas`). lambda is a traced value
of the state, like kT in temperature replica exchange, so all windows share one compiled step and
are one stacked state advanced by `jax.vmap` (NVT; `batched=False` runs them one after the other
through the engine's driver, e.g. for NPT). Resizing of the rows and lists, checkpoints and the
exchange bookkeeping are MDReplicas'.

**Samples** (`LambdaWindows.sample`, every `sample_every` steps): u[k, n] = beta U_k(x_n) for every
window k and every configuration n, and dU/dlambda of each window at its own lambda. U_k = E_ff(x;
lambda_e(k)) + E_sc(x; lambda_v(k)): E_ff needs the induced dipoles re-solved at lambda_e(k) (one
CG from the configuration's own dipoles per distinct lambda_e: the van der Waals stage shares
lambda_e = 0, so 19 windows need 8 solves per sample), E_sc is a small row sum (with "keep",
the gas-phase correction at lambda_e(k) is added too); the
lambda-independent terms (restraints, bonded energy of a flexible solute) are added, P V under NPT
(a per-configuration constant, cancelling in every estimator) is left out. The diagonal u[k, k] agrees
with beta times the step's potential energy (a different code path: analytic rows vs autodiff) to
0.004 kT RMS in mixed precision for pGM water (0.0008 kT for TIP3P).

**Hamiltonian replica exchange** (`exchange_every`, a multiple of `sample_every`): neighbouring
windows, even and odd pairs alternately, swap configurations with the Metropolis test of
`remd.metropolis` on the sampled u (no extra energy evaluation). A swap moves positions, momenta,
box, neighbour list and dipoles; the forces and dipoles are then re-evaluated in the slot's
Hamiltonian, the energy change is booked as heat (so `econs` stays continuous) and the dipole
predictor restarts from the new dipoles.

## Estimators (`free_energy.py`)

- **Decorrelation**: per window, samples after `discard_ps` are subsampled every ceil(g) with g
  the statistical inefficiency (Chodera et al., JCTC 3, 26 (2007); FFT autocorrelation, summed to
  the first non-positive value) of the energy difference to the neighbouring window (BAR, MBAR;
  alchemlyb's 'dE') or of dU/dlambda along the path (TI). `detect_equilibration` picks the start
  that maximises the number of uncorrelated samples (Chodera, JCTC 12, 1799 (2016)).
- **MBAR** (Shirts & Chodera, JCP 129, 124105 (2008)): Newton steps or self-consistent updates,
  whichever leaves the smaller gradient (pymbar's adaptive scheme; comparing objective values fails
  near convergence because the objective is a sum of large numbers), started from neighbour
  estimates; asymptotic covariance W^T (I - W N W^T)^+ W by the SVD route (pymbar's default).
  Also the overlap matrix (smallest neighbour element reported; below ~0.03 windows are too far).
- **BAR** between neighbours (Bennett 1976; Shirts et al., PRL 91, 140601 (2003)), with Bennett's
  variance (Eq. 10a), summed along the path.
- **TI**: trapezoid rule along the polyline of (lambda_e, lambda_v) points, standard errors of the
  window means propagated. With the gas-phase leg, `estimate` also integrates the
  **gas-subtracted integrand** <dU/dlambda_e> - dE_gas/dlambda_e: for a rigid solute
  integral(dE_gas) is exactly Delta G_gas, and pGM's intramolecular electrostatics (for pGM water
  E_gas = -4097 kJ/mol, a non-linear function of lambda_e through the intramolecular induction)
  then drops out before the quadrature instead of being integrated by the trapezoid rule.
- Toy with an analytic answer (tests): harmonic oscillators u_k = K_k (x - x0_k)^2 / 2, f_k =
  ln(K_k / K_0) / 2: MBAR and BAR within their error bars over 40 repeats, the ratio of their mean
  error bar to the spread of the estimates within 0.7-1.4, TI converging to the same value.

## Gas-phase leg

With `intramolecular="annihilate"` (the default of `Alchemy`, and of the script for rigid solutes),
`GasPhaseLeg` evaluates the solute alone in vacuum with the same alchemical parameters (the
gas-phase `Model` with `ElecChannel`: every pair, dense induction solve, the kernels and Coulomb
constant of the MD engine). For a rigid solute its energy does not depend on the configuration, so
Delta G_gas(1 -> 0) = E_gas(0) - E_gas(1) exactly. Checked against the MD engine's energy of the
lone molecule in a 4.2 nm periodic box (1e-3 kJ/mol; the periodic self-image and PME error of the
intramolecular terms are that small there), for a rigid water and a flexible methanol
(`lone_solute`: the lone molecule in a large box, which also serves to sample the gas-phase leg of
a flexible solute with the same engine if one annihilates its electrostatics). With
`intramolecular="keep"` the gas-phase leg is part of the Hamiltonian (above).

## Protocol checklist

- Build the system with `alchemical_system` (the solute's own parameter keys) and equilibrate at
  full coupling under NPT (`Alchemy` makes the barostat's trial energies alchemical), then run the
  windows NVT at the mean volume, batched on one GPU.
- `standard_schedule(8)` (8 electrostatics + 11 van der Waals windows) gave neighbour acceptances
  of 0.35-0.98 and an MBAR overlap of at least 0.087 for water and methanol (the lowest around
  lambda_vdw = 0.4, where the soft-core integrand peaks); check `overlap_min` (> 0.03) and the
  acceptance in the log for other solutes, and add windows where they drop.
- Sample every 1 ps (500 steps at 2 fs; the samples cost 0.5-1.5 % of the run) with Hamiltonian
  exchange at every sample: the statistical inefficiencies are then 1.0-2.3 (mostly below 1.4).
- Discard at least the detected equilibration time (`analyze` prints it per window) and compare
  the two halves of the run.
- For pGM solutes use MBAR or BAR, or TI of the gas-subtracted integrand: the solute's own
  induction makes the raw TI integrand of the electrostatics stage non-linear, and the trapezoid
  rule over 8 windows is then off by about 0.4 kcal/mol for water (below).
- Rigid solutes: `intramolecular="annihilate"` with `GasPhaseLeg` (or "keep": the same result);
  flexible pGM solutes: "keep".

## Validation

### Checks of the machinery (`tests/test_alchemy.py`, float64 unless noted)

| Check | Result |
|---|---|
| lambda = (1, 1) against the original Hamiltonian: energy, forces, pressure, Monte Carlo trial energy (rigid water; flexible methanol among rigid waters) | 1e-12 relative; an NPT run follows the plain one step by step |
| dU/dlambda (Hellmann-Feynman, fixed dipoles) against central differences with the dipoles re-solved, lambda = (0.6, 1), (0.3, 0.7), (0, 0.4), (0, 0.05); flexible solute (0.5, 0.7) | 1e-9 relative (1e-6 tolerance) |
| same at the ends lambda_elec = 0 (polarizability floor), 1 and lambda_vdw = 0 (second-order one-sided differences) | 1e-5 relative |
| decoupled end state lambda = (0, 0) against the box without the solute | 9e-10 kJ/mol of 4.4e4; environment forces 1e-7; solute forces < 1e-5 |
| flexible solute at (0, 0): the box without it + its bonded and intramolecular LJ energy (+ its whole gas-phase electrostatics with "keep") | 1e-8 relative |
| soft core at r = 0 (an oxygen on the solute's oxygen) | finite, zero force, U = lambda eps (1/w^2 - 2/w) exactly; LJ at lambda_vdw = 1; tail x lambda_vdw |
| pressure of the lambda Hamiltonian (lambda = (0.4, 0.6)) against a volume finite difference | 1e-6 relative |
| batched windows (vmap) against sequential windows; u_n(x_n) = beta U of the step | 1e-10 |
| Hamiltonian exchange: energies re-evaluated after a swap = sampled u_k(x_n); untouched windows unchanged | 1e-6 kJ/mol; bit-identical |
| keep vs annihilate on a rigid solute | U_keep - U_annihilate = E_gas(1) - E_gas(lambda) to 1e-7 kJ/mol |
| gas-phase leg against the MD engine's lone molecule in a 4.2 nm box (rigid water; flexible methanol) | 1e-3 kJ/mol |
| MBAR, BAR on harmonic oscillators (f_k = ln(K_k/K_0)/2), 40 repeats of 5 x 400 samples | mean within 3 standard errors; mean error bar / spread of the estimates within 0.7-1.4; MBAR with an unsampled state; TI converges |
| statistical inefficiency of AR(1) series (phi = 0, 0.8, 0.95) | within 10 % of (1 + phi) / (1 - phi) |
| production settings (PME 48^3 order 6, beta 4 nm^-1, mixed): lone solute, periodic minus gas-phase annihilation energy, 40 random placements | TIP3P -0.045 kJ/mol, pGM -0.015 (the tin-foil self-image energy; placement spread 1e-4): the finite-size error of the solution leg, < 0.01 kcal/mol |
| mixed-precision noise of the sampled u: u_k(x_k) against beta U of the step (other code path) | 0.004 kT RMS (pGM), 0.0008 kT (TIP3P) |
| one sample of the 19 pGM-water windows (production checkpoint) in mixed vs double precision | u_k(x_n) - u_n(x_n): 0.0015 kT RMS, 0.005 kT max; dU/dlambda 0.003 kJ/mol |
| plain NVT (mixed) and NPT (double) runs against master, 60 steps | bit-identical |

### Hydration free energies

Settings of the water runs: the 512-water truncated octahedron of `~/pgm-gvdw-data/inputs/lj/inpcrd.restrt`
(the box of the README's validation), one water the solute, 298 K; 0.9 nm cutoff with the
Lennard-Jones long-range correction, PME 48^3 order 6, beta 4 nm^-1, dipole tol 1e-5, mixed
precision; rigid bodies, 2 fs, Bussi thermostat (tau 1 ps). 100 ps NPT (1 bar, Monte Carlo
barostat) at full coupling, then NVT at the mean volume of its second half; 19 windows
(`standard_schedule(8)`: lambda_elec = 1, 6/7, ..., 0 at lambda_vdw = 1, then lambda_vdw = 0.9, 0.8,
..., 0.1, 0.05, 0), batched on one GPU; samples and Hamiltonian exchanges every 1 ps; the first 200
ps of every window discarded (the detected equilibration times are 0-280 ps; discarding 400 ps
changes the results by less than 0.01 kcal/mol). The pGM3P-25 and methanol runs (below) differ
where stated. Errors: one standard error (MBAR covariance, BAR variance, TI propagated), from
samples subsampled with the statistical inefficiency (1.0-2.3, mostly below 1.4: with exchanges,
samples 1 ps apart are nearly independent).

| Model | Run | Delta G_hyd MBAR | BAR | TI | TI, gas-subtracted | halves (MBAR) | electrostatics / vdW (MBAR) |
|---|---|---|---|---|---|---|---|
| TIP3P | 19 x 1.5 ns | **-6.12 +- 0.05** | -6.10 +- 0.04 | -6.18 +- 0.06 | (= TI) | -6.21 +- 0.08 / -6.06 +- 0.08 | -8.23 / +2.11 |
| pGM water of the README box | 19 x 2 ns | **-4.45 +- 0.05** | -4.46 +- 0.04 | -4.78 +- 0.06 | -4.38 +- 0.06 | -4.38 +- 0.06 / -4.53 +- 0.07 | -6.95 / +2.50 |
| pGM3P-25 with the paper's geometry and Lennard-Jones | 19 x 1.2 ns | **-4.91 +- 0.07** | -4.92 +- 0.06 | -5.25 +- 0.08 | -4.83 +- 0.08 | -4.81 +- 0.10 / -4.96 +- 0.10 | -7.58 / +2.68 |
| methanol (flexible, `intramolecular="keep"`) in pGM water | 19 x 1.2 ns | **-2.65 +- 0.07** | -2.67 +- 0.06 | -2.68 +- 0.08 | (= TI) | -2.65 +- 0.10 / -2.65 +- 0.10 | -4.80 / +2.15 |
| methanol held rigid (reference geometry; annihilation + exact gas leg) in pGM water | 19 x 0.9 ns | **-2.36 +- 0.09** | -2.30 +- 0.07 | -2.41 +- 0.10 | -2.37 +- 0.10 | -2.36 +- 0.12 / -2.35 +- 0.13 | -4.63 / +2.27 |

Values in kcal/mol. The electrostatic and van der Waals parts are those of the hydration free
energy: Delta G_gas - Delta G_solv,elec and -Delta G_solv,vdW.

- **TIP3P** (point charges, `elec="q"`, TIP3P's geometry and Lennard-Jones; density at 1 bar
  0.980 g/cm^3): -6.12 +- 0.05 kcal/mol by MBAR, BAR and TI within their errors. Literature
  values for TIP3P's hydration free energy of itself are about -6.1 (the range -6.1 to -6.3 of
  alchemical calculations with different cutoff and long-range treatments; e.g. Shirts & Pande,
  JCP 122, 134508 (2005); the exact table value could not be re-checked here, the PubMed page was
  rate-limited); experiment -6.3 (Ben-Naim & Marcus, JCP 81, 2016 (1984)). This validates the
  machinery end to end: Hamiltonian, windows, sampling, estimators and the gas-phase leg.
- **pGM water** (the model of the README box, `rayl_512_v2.prmtop`: pGM3P-25's electrostatics, q_O =
  -2.04056 e, q_H = +1.02028 e, covalent dipoles -0.019120 e nm (O -> H) and +0.0085877 e nm (H ->
  O), Gaussian radii 0.060515 / 0.053623 nm, polarizabilities 1.1177 / 0.3296 A^3, as in Wu et al.,
  JCTC 21, 3563 (2025) and github.com/yxwu21/pGM3P-25, on TIP3P's geometry (0.9572 A, 104.5 deg) and
  TIP3P's Lennard-Jones (sigma 3.1507 A, epsilon 0.1520 kcal/mol on O, none on H); density at 1 bar
  1.016 g/cm^3): **-4.45 +- 0.05 kcal/mol** (MBAR; BAR -4.46 +- 0.04), against -6.3 in experiment. I
  know of no published pGM hydration free energy to compare with. MBAR, BAR and the gas-subtracted
  TI agree within 0.07 kcal/mol; the raw TI integrand misses by 0.33 kcal/mol: <dU/dlambda_e> is
  dominated by the solute's intramolecular electrostatics (-8374 kJ/mol at lambda_e = 1), which its
  own induction makes non-linear in lambda_e, and the trapezoid rule over 8 windows cannot integrate
  it; subtracting the exact gas-phase integrand, or MBAR / BAR, removes that error. Discarding 400
  ps instead of 200 changes MBAR by 0.008 kcal/mol. The model binds a water 1.9 kcal/mol too weakly:
  its electrostatic part (-6.95 kcal/mol) is 1.3 kcal/mol weaker than TIP3P's (-8.23) and its cavity
  / van der Waals part 0.4 kcal/mol more costly (+2.50 against +2.11, with the same Lennard-Jones:
  pGM water is denser, 1.016 g/cm^3 against 0.980). This is in line with the weak polarity of this
  liquid (dipole 1.99 D, static dielectric constant 31: `docs/dielectric.md`).
- **pGM3P-25 with the published geometry and Lennard-Jones** (`--model pgm3p25`: the same
  electrostatic parameters with O-H 0.9745 A, 103.64 deg and sigma 3.18156 A, epsilon 0.14473
  kcal/mol, as `scripts/water_dielectric.py`; the hydrogens rebuilt about each oxygen of the box,
  200 ps NPT (1.014 g/cm^3), 300 ps of each window discarded): **-4.91 +- 0.07 kcal/mol** (BAR -4.92
  +- 0.06, gas-subtracted TI -4.83 +- 0.08). The published geometry and Lennard-Jones strengthen the
  electrostatic part by 0.6 kcal/mol (-7.58; the longer O-H bond gives a larger molecular dipole)
  and weaken the van der Waals part by 0.2 (+2.68): 0.46 kcal/mol more favourable than the box of
  the README, still 1.4 kcal/mol short of experiment. So the box of the README is indeed not the
  full published pGM3P-25 (its geometry and Lennard-Jones are TIP3P's, as docs/dielectric.md found),
  and the difference matters here at the 0.5 kcal/mol level.
- **Methanol, a flexible solute** (the class II bonded fit that `examples/flex_methanol_check.py`
  writes to `runs/flex/methanol.flex`, its pGM electrostatics from an ESP fit, q_O = -1.007, q_H(O) = +0.776, q_C = -0.103, q_H(C) = +0.111 e
  with covalent dipoles, and GAFF's Lennard-Jones; put at the centre of the pGM-water box, 8 waters
  removed; the flexible engine, waters rigid by constraints, the solute's X-H bonds constrained, 2
  fs, 200 ps NPT, 300 ps of each window discarded; `intramolecular="keep"`): **-2.65 +- 0.07
  kcal/mol** (MBAR; BAR and TI within 0.03; discarding 500 ps: -2.61 +- 0.09), against -5.11 in
  experiment (Ben-Naim & Marcus 1984). Held rigid at its reference geometry instead (the rigid
  engine, annihilation with the exact gas-phase leg, 19 x 0.9 ns): -2.36 +- 0.09. The two runs share
  no code path for the solute's intramolecular terms (flexible vs rigid engine, kept vs annihilated
  intramolecular electrostatics, gas-phase leg in the Hamiltonian vs separate and exact) and differ by 0.29 +- 0.11
  kcal/mol, both parts more favourable for the flexible molecule (electrostatics by 0.17, van der
  Waals by 0.12 kcal/mol), as expected from its relaxation in water (bond angles and the C-O
  torsion). The model binds methanol 2.5 kcal/mol too weakly, as expected from its parts: this pGM
  methanol with GAFF's Lennard-Jones has a heat of vaporization 2 kcal/mol too low (README, Fitting
  to liquid properties), and the water is the weakly polar pGM water above.

### Cost (one RTX PRO 6000 Blackwell; 512 waters, 1,536 atoms, 2 fs; `solvation_free_energy.py bench`)

| | TIP3P (`elec="q"`) | pGM water |
|---|---|---|
| plain MD, same system | 0.339 ms/step (510 ns/day) | 0.771 ms/step (224 ns/day) |
| one alchemical window (alchemy hooks, soft-core rows) | 0.362 ms/step (+7 %) | 0.787 ms/step (+2 %) |
| 4 / 8 / 19 windows batched, ms per window-step | 0.211 / 0.234 / 0.221 | 0.383 / 0.380 / 0.394 |
| 19 batched windows: ns/day per window (aggregate) | 41 (782) | 23 (438) |
| one sample of all 19 windows (u_k(x_n) with 8 dipole re-solves, dU/dlambda) | 10 ms (0.5 % at 1 ps) | 53 ms (1.4 % at 1 ps) |
| production, samples + exchanges every 1 ps, ns/day per window (aggregate) | 39.4 (749) | 22.8 (432) |
| GPU time of the runs above (19 windows, NPT included) | 1.5 ns/window: 57 min | 2 ns/window: 2.2 h |

Flexible methanol in pGM water (the flexible engine, 1,518 atoms, constraints) ran its 19 windows
at 21.4 ns/day per window (407 aggregate), as fast as rigid pGM water.
Batching the windows doubles the throughput of pGM water (1.5x for TIP3P, whose step has no
induction solve): the GPU is full at about 4 windows of 1,500 atoms, so more windows cost
proportionally more. The statistical error scales as 1/sqrt(time): +-0.1 kcal/mol for water in
pGM water takes about 0.6 ns per window (0.2 ns of it discarded), 40 minutes of one GPU.

## Limits

- One solute molecule (rigid in `Simulation`, or flexible in `FlexibleSimulation`); flexible pGM
  solutes with `intramolecular="keep"` (the script refuses annihilation for them).
- Neutral solutes only: annihilating a net charge in a periodic box needs finite-size corrections
  (Rocklin, Mobley, Dill & Huenenberger, JCP 139, 184103 (2013)); refused.
- Van der Waals forms "lj" and "none" (GVDW refused).
- The solute interacts with its own periodic images while lambda_e > 0 (a neutral molecule's
  image energy is ~0.01 kJ/mol in these boxes; not corrected).
- Batched windows run NVT (under vmap the barostat's trial energy would be evaluated every step);
  NPT windows run sequentially (`batched=False`).
- The cell-dipole recorder (`run(dipoles_every=n)`) is refused with an alchemical region.
