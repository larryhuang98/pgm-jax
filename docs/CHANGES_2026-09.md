# Changes, September 2026: nine feature branches merged

Nine feature branches, each started from master `5c92e96` (208 tests), were merged into the
`integration` branch of a separate clone in the order below (least invasive first), tested after
every merge, and master was then fast-forwarded to it. Each feature has its own document in
`docs/`; this page summarises them, lists what the integration changed, and states which
combinations of the new features work and which are refused.

| Order | Branch | Merge commit | Main code |
|---|---|---|---|
| 1 | qmfit | `398150b` | `pgm_jax/fit/qm.py`, `scripts/qmfit/`, `data/qm/` |
| 2 | bias | `2064c72` | `pgm_jax/bias/`, `bias=` in `Simulation` / `FlexibleSimulation` |
| 3 | iface | `18e5a6b` | `pgm_jax/interfaces/` (ASE, i-PI, OpenMM) |
| 4 | fegrad | `144e639` | `pgm_jax/md/fe_grad.py`, `alchemy.py`, `scripts/solvation_free_energy.py` |
| 5 | fit2 | `b416290` | `pgm_jax/fit/`, `scripts/fit_multi.py` |
| 6 | pimd | `ab758ef` | `pgm_jax/md/pimd.py`, `bond_quartic` bonded family |
| 7 | shake | `0f6df74` | `pgm_jax/md/constraints.py`, `flexible.py` |
| 8 | efield | `dbe4689` | `pgm_jax/md/efield.py`, `finite_field.py`, `forcefield.py`, integrators |
| 9 | iel | `ed6eff8` | `pgm_jax/md/iel.py`, `forcefield.py` (`MDSettings(iel=...)`) |

Integration commits (on top of the merges): `a8b8b12`, `a76b676`, `cda49af`, `2209446`, `4770fca`,
`06ba7ec` and the commit that adds this page (see "Integration" below).

## The features

**QM cluster fitting** (`docs/qmfit.md`). pGM parameters (charges, covalent dipoles, radii,
polarizabilities, LJ or GVDW) are fitted by least squares with exact Jacobians to interaction
energies, SAPT components, 3-body energies, rigid-body forces and monomer properties of clusters.
A psi4 water set comes with it (844 records: 757 dimers including Smith-type stationary structures,
scans, liquid pairs and the pairs of larger clusters; liquid trimers to pentamers; the WATER27
clusters; CCSD(T)/CBS-quality totals, SAPT0, MP2 many-body terms and forces; 274 core-hours). The
refitted pGM3P-25 water lowers the held-out interaction-energy RMSE from 1.50 to 0.75 kcal/mol
(LJ, pmemd-compatible) and to 0.57 kcal/mol with per-atom GVDW (3-body energies 0.18, forces 0.49).

**Enhanced sampling** (`docs/enhanced_sampling.md`). Collective variables are JAX functions of
positions and box, and bias forces come from `jax.grad` of V(s(x)): static biases (umbrellas,
walls), well-tempered metadynamics (hill list or periodic grid) and OPES_METAD, deposited inside
the compiled loop with the energy change booked as heat; walkers (independent or one shared bias)
in one vmapped program; FES, c(t) reweighting and WHAM. On model potentials every run is within
0.3 kJ/mol of the exact free-energy surface; for alanine dipeptide with pGM electrostatics,
metadynamics and OPES give Delta G(phi > 0) = 7.88 +- 0.03 and 7.90 +- 0.08 kJ/mol against 7.75 +-
0.26 from replica exchange (F(phi) RMSD 0.17 kJ/mol). A grid bias costs +4-8 % per step on the GPU.

**Interfaces to other codes** (`docs/interfaces.md`). A device-resident `PGMEngine` (one jitted call
per configuration, dipole-history slots for beads) drives an ASE calculator, an i-PI socket client
and an OpenMM `PythonForce`. Single points equal the native engine to rounding (ASE energy 2e-16
relative, stress 3e-13 against finite differences); NVE, NVT, NPT (OpenMM Monte Carlo barostat:
density 1.0176 +- 0.0020 against native 1.0183 +- 0.0010 g/cm^3) and i-PI path integrals (P = 32:
KE_H 148.53 +- 0.06 against native 148.48 +- 0.07 meV) agree with the native runs within their
errors. An external force evaluation costs 1.4-1.5x a native MD step on the GPU.

**Free-energy parameter gradients** (`docs/fe_gradients.md`). Hydration (alchemical) free energies
come with d DeltaG/d theta for every table parameter of solute and solvent (end-state and
MBAR-weighted estimators at the converged dipoles, block-jackknife errors, +1 % run time), a
`FreeEnergyTarget` for fitting codes, and `FreeEnergyRun(param_grad=)`. The branch also fixed the
sampler cache key of `LambdaWindows` (the neighbour-list layout is part of it). Finite differences
over 11 independent free-energy calculations (water: solute charge and r_min scales; flexible
methanol) agree within 1.1 standard errors; a one-step fit of methanol's charge scale to experiment
is shown.

**Multi-target liquid fits with uncertainties** (`docs/liquid_fit.md`). Ensemble gradients of the
density, heat of vaporization, static dielectric constant (the induced dipoles' response by an
adjoint solve), liquid dipole, g(r), thermal expansion and compressibility; gas-phase targets;
trust-region Levenberg-Marquardt with priors and a parameter covariance propagated to predictions.
Per-frame derivatives match central differences to 2e-6; ensemble gradients agree with finite
differences between independent 4 ns runs of 512 waters (|z| <= 2). A demonstration fit of the base
pGM water gives, in a 10 ns check, density 0.9963 g/cm^3, Hvap 10.491 kcal/mol and eps 81.5 +- 2.5
(targets 0.997, 10.52, 78.4).

**Path-integral MD** (`docs/pimd.md`). PIMD with PILE-L / PILE-G, thermostatted RPMD and RPMD for
flexible molecules, beads batched with `jax.vmap` (chunks of 8), Monte Carlo NPT, primitive and
centroid-virial estimators, ring-polymer contraction; a flexible pGM water with the q-TIP4P/F
monomer surface (new `bond_quartic` bonded family). Exact harmonic-oscillator results for
P = 1-64 and OpenMM's RPMDIntegrator are reproduced; 512 waters at 298 K and P = 32 give KE_H =
148.5 +- 0.1 meV (q-TIP4P/F about 143, experiment 143-156), at 7.8 ms per step (4.2 ms contracted
to 8 beads).

**Bond constraints** (`docs/shake.md`). SHAKE / RATTLE in g-BAOAB order for X-H bonds or all bonds,
exact per-cluster Newton in batched blocks and a matrix-free solver for clusters above 12
constraints; degrees of freedom 3N - N_c (- 3 with NVE or Bussi, drawn momenta without net
momentum), `temp_half`. Liquid methanol with X-H constraints at 2 fs reproduces 1 fs (density
within 0.004 g/cm^3), constraints and RATTLE hold to 1e-14, and against pmemd.pgm with SHAKE the
potential energy agrees within 0.03 kJ/mol per molecule; 2.7x the unconstrained speed at 2 fs,
4.3x at 4 fs with all bonds and HMR. One expectation in `tests/test_vsites.py` changed with the
degrees of freedom.

**External electric fields** (`docs/efield.md`). Static, oscillating and constant-D uniform fields
act on the Gaussian charges, the covalent dipoles and the induced dipoles (right-hand side of the
induction equations); the amplitude is a state variable, so `FieldReplicas` runs +-E copies in one
vmapped program. Finite-field dielectric constants reproduce the fluctuation results: pGM3P-25 33.9
+- 0.5 at +-0.1 V/nm (fluctuations 34.2 +- 0.7), the base pGM water 73.1 +- 0.4 (73.2 +- 0.2), TIP3P
96.8 +- 2.0 (OpenMM gives the same saturated response); the gas-phase response equals the molecular
polarizability to 1e-14; NVE conserves econs with static, oscillating and constant-D fields. A
static field costs 0.6 % per step on the GPU, constant D 14 %.

**Extended-Lagrangian induced dipoles** (`docs/iel.md`). iEL/0-SCF: auxiliary dipoles follow
Niklasson's dissipative time-reversible Verlet, one field sweep and a block-Jacobi update per step,
forces the exact gradient of a shadow energy; iEL/SCF-k runs k CG iterations from the auxiliary
dipoles. 1.5x the speed of the SCF solver at tol 1e-5 (336 ns/day for 512 waters at 2 fs); NVE drift
+0.001 (1 fs) and +0.013 to +0.020 (2 fs) kT/ns/dof; dipoles within 1.7e-3 RMS of converged ones;
pGM3P-25 eps 34.2 +- 0.5 (SCF 33.9 +- 0.6) with density, <U>, D, rotational times and g_OO equal
to SCF within errors.

## Integration

Conflicts were in `README.md` (every merge: feature bullets, Layout rows, docs links, Limits; all
kept, the test count set at the end), `pgm_jax/md/flexible.py` (bias / shake / efield / iel:
signatures and docstrings carry `bias=`, `constraint_options=` and `efield=` together; the shake
degrees of freedom and projections kept), `integrate.py` (bias's strided loop now runs the step with
efield's E(t) work booking; `MDState` has both the bias and the field fields; the barostat passes
field and bias state), `mts.py`, `simulation.py` (both hooks), and `forcefield.py` (efield's field
terms and iel's shadow terms in the same energy and force routines; `compute` passes the field to
either solver). Fixes made on the integration branch:

- `MTSIntegrator.init` passes the bias state (`FlexibleSimulation.minimize` with `mts=` raised a
  TypeError after the bias merge).
- Independent and shared walkers (`bias/walkers.py`) step through the same step as a single
  simulation, including the heat of a time-dependent field.
- The external field enters the extended-Lagrangian dipole equations: the auxiliary-dipole residual
  r(x), the warm-up solve and the iEL/SCF operator include E (or F(M) at constant D with its kappa
  term), and at constant D the shadow energy subtracts kappa |sum delta|^2 / 2, which the
  preconditioner does not contain.
- The barostat's converged reference energy of iEL/0-SCF includes the field and the bias.
- `PGMForceField.strain_derivative` returns the full tensor (below).
- Pair rows without iEL are evaluated in the original order again, so field-free SCF runs are
  bitwise identical to master (the iel branch had reassociated one product: 1e-13 relative in
  double-precision forces).
- README Limits and the iel / efield / pimd documents describe the combinations.

**strain_derivative.** The interfaces branch found the lower off-diagonal components of
`PGMForceField.strain_derivative` off by ~3e-3 relative. Root cause: the row displacements
(`_displacements`) and `volume` assume a lower-triangular box, and a strain eps_ab with a > b takes
H (1 + eps)^T out of that form. The fix keeps the hot path unchanged: the six components that keep
the box lower triangular are differentiated as before, and the other three follow from rotation
invariance (`full_strain_derivative`: the atomic strain derivative W + G, with G the intramolecular
part of molecular scaling, satisfies W_at - W_at^T = T^T - T, where T collects dE/dv (x) v of the
vectors held fixed: the induced dipoles, the field and its dipole offset). `Alchemy.strain_derivative`
uses the same helper. The full tensor now matches central differences of strained configurations
rotated back to a lower-triangular box to 1e-6 of its largest component (molecular and atomic
scaling, with a constant field and at constant D; `tests/test_integration.py`). Two existing tests
(`tests/test_flux.py`, `tests/test_vsites.py`) checked lower components against finite differences
of the engine's energy at non-triangular boxes, which carry the same error (the old tensor and the
old differences agreed with each other); they now evaluate the strained boxes in a lower-triangular
frame (`box.lower_triangular_frame`, also checking a new upper component). The native pressure
(trace) is unchanged; `PGMEngine` still reports the symmetric tensor built from the upper
components.

## Combinations of the new features

| Combination | Status | Checked by |
|---|---|---|
| iEL + external field (constant E, constant D, E(t)) | works: the field is in the auxiliary-dipole step and the shadow energy | shadow forces vs finite differences (E and D, Jacobi and block, omega 1 and 0.9), field-polarized SCF solution as fixed point, iEL/SCF step, NVE in a field |
| iEL + bias | works | bias forces add, NVE with a static umbrella |
| external field + bias | works (biases are field independent) | forces, field energy unchanged, NVE |
| walkers + E(t) | works (work of E(t) booked) | walker 0 = single simulation (epot, heat) |
| shake + PIMD | refused (PIMD takes no constraints) | `tests/test_integration.py` |
| PIMD + bias / external field / iEL | refused | same |
| interfaces (`PGMEngine`) + iEL | refused (iEL needs the native sequence of steps) | same |
| `PGMEngine.from_simulation` + field / bias | refused (the engine would drop them) | same |
| iEL + MTS / alchemy / differentiable; field + alchemy; field + NPT with charged molecules | refused (as in the branches) | `tests/test_iel.py`, `tests/test_efield.py` |
| shake's degrees of freedom (3N - N_c - 3 with NVE / Bussi) with the thermostats used by the bias, efield, iface and pimd code and tests | consistent: biases, fields and walkers use the engine's `integ.dof` (Bussi target, temperatures); PIMD has its own ring-polymer thermostats; the interfaces leave thermostats to the external code (their validation scripts use `integ.dof` of the native run) | full suite |
| fit2 / fegrad differentiable paths with the field terms of the induction solve | unchanged without a field: FrameAnalyzer bitwise equal to the fit2 branch, the differentiable solve (values and derivatives) bitwise equal to master | identity runs (below) |
| iface `PGMEngine` with the merged force field | works (the engine calls `compute` with the new optional arguments at their defaults) | `tests/test_interfaces.py` |

Identity checks without the new features (CPU; `scripts/efield_identical.py`: 300 steps of NVT
Bussi, mixed precision, 2 fs; the differentiable solve in double and mixed precision): the rigid run
and the differentiable dipole solve with its position and box derivatives are bitwise identical to
master; FrameAnalyzer values and derivatives are bitwise identical to the fit2 branch; the flexible
run (water by constraints) differs from master by design (shake: degrees of freedom and drawn
momenta with Bussi, projection order) and is bitwise identical to the shake branch.
Speed (GPU, `scripts/bench_md.py`, 512 pGM3P-25 waters, mixed precision, alternating runs): rigid
engine at 1 fs 0.732 / 0.711 ms per step for master and 0.737 / 0.713 for the integration branch
(noise); atoms with constraints at 2 fs 0.851 (master) and 0.834 ms (shake's projection order).

## Tests

`pytest -q` has 356 tests (208 on master before the merges; 17 of them in `tests/test_integration.py`,
the cross-feature checks). Full suite on the integration branch (`06ba7ec`), CPU, 32 cores
(cpu-short): 354 passed, 2 skipped, in 67.7 min. The two skipped tests need a real i-PI server and
OpenMM >= 8.4 (`IPI_ROOT`, and OpenMM from its own environment on `PYTHONPATH`, as in
`docs/interfaces.md`); run with those on the same code they pass (2 passed). After every merge the
branch's own tests and the tests of the files it touched ran on a snapshot of the branch; after the
efield merge the full suite ran as well (323 passed, 2 skipped).

## Known limits

The per-feature limits are in each document and summarised in README, Limits. In short: QM fits are
gas-phase fits of one rigid molecule kind (water); free-energy gradients of solvent parameters are
unbiased but noisy; liquid fits cover rigid molecules; walkers are single-device NVT without MTS and
OPES has no adaptive sigma or kernel neighbour list; the interfaces expose neither virtual sites,
alchemy, MTS, fields, biases nor iEL, and OpenMM's CUDA platform cannot share a GPU with JAX in
exclusive-process mode; path integrals take flexible molecules only (no constraints, virtual sites, MTS, restraints,
alchemy, biases, fields or iEL); constraints are distances only; external fields are uniform (no
NPT with charged molecules, no alchemy); iEL at 2 fs drifts +0.013-0.020 kT/ns/dof and is refused
with MTS, alchemy, differentiable solves, PIMD and the interfaces; iEL in a field is checked by
tests but not by production runs.
