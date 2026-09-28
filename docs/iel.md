# Extended-Lagrangian induced dipoles (iEL/0-SCF, iEL/SCF)

`MDSettings(iel="0scf")` replaces the per-step induced-dipole solve (predictor + conjugate
gradients to `dipole_tol`) by auxiliary dipoles x that are propagated with the atoms, as in the
inertial extended Lagrangian of Albaugh, Niklasson and Head-Gordon (iEL/0-SCF: J. Phys. Chem. Lett.
8, 1714 (2017); iEL/SCF: Albaugh, Demerdash & Head-Gordon, J. Chem. Phys. 143, 174104 (2015)) and
Niklasson's extended-Lagrangian Born-Oppenheimer MD. iEL/0-SCF does **no SCF iteration**: one field
sweep per step, and the forces are the exact gradient of a shadow energy U~(R, x). Per step it costs
the permanent-field sweep and the force evaluation that the SCF path also does, and nothing else
(the SCF path adds 4-6 CG iterations).

## Usage

```python
from pgm_jax.md.forcefield import MDSettings
s = MDSettings(iel="0scf")                    # iEL/0-SCF (defaults: block preconditioner, K = 7)
s = MDSettings(iel="scf", iel_iter=2)         # iEL/SCF-2: two CG iterations from the auxiliary dipoles
sim = Simulation(sys, pos, H, settings=s, ...)          # rigid bodies, or FlexibleSimulation
```

Scripts: `--iel 0scf` (and `--iel-iter`, `--iel-order`, `--iel-precond`, `--iel-omega`,
`--iel-kappa`, `--iel-alpha`, `--iel-no-shadow`) in `run_md.py`, `bench_md.py` and
`water_dielectric.py`. `scripts/iel_validate.py` makes the validation runs below and
`scripts/iel_cost.py` the cost per force call.

| setting | default | meaning |
|---|---|---|
| `iel` | `"none"` | `"0scf"`, `"scf"`, or `"none"` (predictor + CG, unchanged) |
| `iel_precond` | `"block"` | 0-SCF step delta = omega W r: W = alpha (`"jacobi"`) or M^-1, M = 1/alpha + the intramolecular row tensors of every molecule of 2-8 atoms (`"block"`; larger molecules Jacobi) |
| `iel_order` | 7 | K of Niklasson's dissipation (0, 3-9); 0 is exactly time reversible |
| `iel_omega` | 1.0 | scale of the 0-SCF step (see Energy flow) |
| `iel_iter` | 1 | `"scf"`: CG iterations per step from x (0: to `dipole_tol`) |
| `iel_kappa`, `iel_alpha` | table | kappa = (omega_x dt)^2 and the dissipation strength a; default Niklasson's values for K |
| `iel_shadow` | True | 0-SCF forces: exact gradient of U~ (False: fixed-dipole forces at mu) |

`pgm_jax.md.iel.response_spectrum(ff, pos, H, idx)` gives the extreme eigenvalues of W A (power
iteration) and `spectral_radius(lam, K)` the damping of the recurrence.

## Equations

pGM energy with induced dipoles mu (quadratic in mu): U(R, mu) = U_es(q, p(R) + mu) +
sum |mu|^2/(2 alpha) + U_vdW; U_es is the electrostatic energy (rows + PME + self; Gaussian
screened, no exclusions), A = 1/alpha - T its Hessian in mu, r(R, mu) = b - A mu = -dU/dmu the
residual (field minus mu/alpha), zero at the converged dipoles mu*(R).

**Auxiliary dipoles.** Niklasson's dissipative Verlet (JCP 130, 214109 (2009), Table I):

    x_{n+1} = 2 x_n - x_{n-1} + kappa (mu_n - x_n) + a sum_{k=0..K} c_k x_{n-k}

with mu_n the dipoles of step n obtained from x_n. K = 7: kappa = 1.86, a = 0.0016,
c = (-36, 99, -88, 11, 32, -25, 8, -1). The last term damps the free oscillations of x (float32 noise,
kicks); it is of high order in dt (phase lag of x at the librational frequencies: 1e-5 rad for K = 5,
6e-9 for K = 7 at 2 fs). This is the omega_x -> infinity limit of an extended Lagrangian with a
harmonic well for x centred on the response: no fictitious dipole mass or dipole thermostat.

**iEL/0-SCF.** One field sweep at x gives r(x). With a symmetric positive W (below), the dipoles
are mu = x + delta, delta = omega W r(x), and the energy is the shadow potential

    U~(R, x) = stat_delta [ U(R, x + delta) - (1/2) delta^T B delta ],   B = A - (omega W)^-1
             = U(R, x) - (omega/2) r^T W r.

Because U~ is stationary in delta, its exact force at fixed x is the fixed-dipole force at mu plus
the explicit R-dependence of (1/2) delta^T B delta:

    F~ = -dU/dR |_{mu} + d/dR [ U_es(0, delta) - (1/omega) E_M(delta) ] |_{delta},

U_es(0, delta) the dipole-dipole energy of delta alone (rows, PME, self) and E_M the part of it
inside M (the intramolecular row pairs of the blocks; 0 for Jacobi); a constant
(1/omega - 1) |delta|^2/(2 alpha) completes U~ for omega != 1. Both terms are computed in the same
passes over the rows and the same PME call as the forces (a delta argument of `_row_terms` and
`_nonpair`): no response derivative, no iteration. U~ - U* = -(omega/2) r^T (W - A^-1/omega) r is
second order in the error of x. mu = x + delta are the dipoles of every observable (cell dipole,
`induced=` files, virial).

Block preconditioner: M is assembled every step from the special (intramolecular) row entries,
G1 I - G2 x x^T per pair plus 1/alpha on the diagonal, one dense 9 x 9 block per water, and solved in
float64 (a batched LU; 7 % of a step at 512 waters, 6 % at 4096). For the 512-water box the spectrum
of W A is 0.68-1.85 with W = alpha and 0.71-1.56 with W = M^-1 (10-90 %: 0.73-1.39 and 0.82-1.20).

**iEL/SCF.** `iel_iter` CG iterations (Jacobi preconditioned, peek step) start from x with the
residual of the same fused sweep; forces are the fixed-dipole forces at the result (not an exact
gradient; the time-reversible x keeps the error from accumulating, Albaugh et al. 2015).

**Start and restarts.** The first max(K, 2) + 1 steps are solved to `dipole_tol` and fill the
history with converged dipoles (the head x_n replaced by mu*); a state from an SCF run
(checkpoint) starts the same way, and so does an accepted Monte Carlo volume move. With
iEL/0-SCF the barostat compares converged energies U* at both volumes (the dynamics samples U~),
one extra solve every `barostat_interval` steps.

## Energy flow and stability

E = E_kin + U~(R, x) changes only through the auxiliary dipoles: dE/dt = dU~/dx . xdot with
dU~/dx = A (I - omega W A) (x - mu*). The free oscillations e of x about mu* therefore carry the
energy (1/2) e^T A (I - omega W A) e: in an eigenmode lambda of W A it has the sign of
1 - omega lambda. Damping a positive mode (omega lambda < 1) takes energy from the atoms, damping a
negative one gives energy to them, and positive and negative modes together make the undamped
recurrence (K = 0) unstable. At 2 fs the auxiliary frequencies (omega_x dt = 1.2-2.3 rad per step,
bounded by the Verlet limit 2) are only 3-6 times the librational ones (0.3-0.4 rad per step), so
this exchange is not negligible; at 1 fs it is. Measured (NVE, 512 waters, drift of E in
kT/ns per degree of freedom, from the README restart, 80-200 ps unless noted):

| 0-SCF variant | 2 fs | 1 fs |
|---|---|---|
| Jacobi, K = 5, omega = 1 (Niklasson's usual) | +0.20 (heats) | +0.003 to +0.007 |
| Jacobi, K = 3 | T 473 K after 15 ps (unstable) | |
| Jacobi, K = 0 (no dissipation) | NaN after 61 ps | +0.007 |
| Jacobi, K = 7 / K = 9 | +0.047 / +0.026 (1 ns) | -0.002 / +0.002 |
| Jacobi, K = 5, omega = 0.9 / 0.8 / 0.7 / 0.5 | +0.045 / -0.11 / -0.44 / -2.6 (cools) | |
| Jacobi, K = 0, omega = 0.5 (all modes positive) | -0.46, T falls to 265 K in 200 ps | **+0.0006** |
| block, K = 5 / K = 0 | +0.095 / NaN after 34 ps | |
| **block, K = 7 (default)** | **+0.013 (1 ns)** | **+0.0012** |
| block, K = 9 | +0.013 (1 ns) | |
| SCF, mu4 + CG, tol 1e-5 (reference) | +0.0035 (1 ns) | +0.0014 |
| SCF, tol 1e-4 | +0.030 | +0.011 |

So: at 1 fs every stable 0-SCF variant conserves the energy as well as tol 1e-5; at 2 fs the
default (block, K = 7) drifts +0.013 kT/ns/dof over 1 ns, less than the SCF solver at tol 1e-4, and
with a thermostat (Bussi 1 ps) this is 0.03 K per ns of heating, removed without effect (below).
For strict NVE at 2 fs use iEL/SCF-2 (-0.003 to -0.006) or omega = 0.5, K = 0 at 1 fs.

## Validation (512 pGM waters of the README box, mixed precision, one RTX PRO 6000 / CPU)

Single points and small systems (pytest, `tests/test_iel.py`, float64): the 0-SCF forces agree
with central finite differences of U~ at fixed x to 1e-7 relative (Jacobi, omega = 0.8, block,
block omega = 0.9; the fixed-dipole forces at mu alone miss by 400 times the finite-difference
error); U~ - U* falls by a factor 4 when the error of x is halved (second order); without dissipation 60 steps forward and 60 back return
the positions to 1e-9 nm (0-SCF and SCF-2); the dipoles of a 64-water run stay within 2e-3 of the
converged ones; energy conservation in a tiny box as with converged dipoles.

**Dynamics** (`iel_validate.py`: 4 segments of 5 ps Bussi NVT + 50 ps NVE at 2 fs from the
equilibrated box of the SCF reference run, density 1.0178, the same initial velocities for every
method; converged reference dipoles and energies in float64 at tol 1e-9 every 1 ps; mean +- standard
error over the segments):

| | SCF tol 1e-5 | **0-SCF block K7** | 0-SCF Jacobi K5 | SCF-1 (K5) | SCF-2 (K5) |
|---|---|---|---|---|---|
| CG iterations / step | 5.98 | **0** | 0 | 1 | 2 |
| NVE drift (kT/ns/dof) | +0.0035 | **+0.012 +- 0.007** | +0.205 | -0.076 | -0.0033 |
| rms of E_tot about the drift (kJ/mol) | 0.66 | 1.7 | 2.5 | 1.08 | 0.71 |
| dipole error RMS |mu - mu*| / RMS |mu*| | 7.8e-7 | **1.57e-3** | 1.89e-3 | 1.06e-3 | 2.3e-4 |
| largest atomic dipole error (e nm) | 1.1e-7 | 1.7e-4 | 1.7e-4 | 1.4e-4 | 3.0e-5 |
| U - U*(float64) (kJ/mol, of -2.1e6) | +0.18 | -0.11 | -0.32 | +0.23 | +0.05 |
| D (1e-9 m^2/s) | 3.73 +- 0.21 | 3.75 +- 0.11 | 3.81 +- 0.12 | 3.70 +- 0.10 | 3.73 +- 0.19 |
| tau_1 of the dipole axis (ps) | 2.41 +- 0.11 | 2.44 +- 0.05 | 2.42 +- 0.05 | 2.36 +- 0.07 | 2.52 +- 0.11 |
| tau_2 (ps) | 0.855 +- 0.039 | 0.862 +- 0.022 | 0.857 +- 0.028 | 0.845 +- 0.027 | 0.884 +- 0.031 |
| <T> of the NVE segments (K) | 294.2 | 294.9 | 296.3 (heats) | 293.7 | 294.7 |
| g_OO first peak (nm, height) | 0.281, 2.976 | 0.281, 2.979 | 0.279, 2.992 | 0.279, 2.981 | 0.281, 2.983 |
| max / rms deviation of g_OO from SCF | (halves of SCF: 0.049 / 0.009) | 0.037 / 0.007 | 0.023 / 0.007 | 0.026 / 0.006 | 0.026 / 0.006 |
| mean molecular dipole (D) | 1.988 | 1.988 | 1.988 | 1.988 | 1.988 |

D (centre-of-mass MSD, 2-20 ps) and the rotational correlation times agree with SCF within the
statistical errors of these runs (3-5 %; seed-to-seed differences of T dominate). The dipole error
of 0-SCF (1.6e-3 relative, 3e-3 D per molecule) is not visible in any property.

**Equilibrium** (NPT 298 K / 1 bar, Bussi 1 ps, Monte Carlo barostat every 100 steps, 2 fs, cell
dipole every 25 steps; the protocol of the reference run of `docs/dielectric.md`; tin-foil eps with
eps_inf from the cell polarizability; jackknife errors):

| | SCF tol 1e-5 (reference, 14.9 ns) | **0-SCF block K7 (7.8 ns)** | 0-SCF Jacobi K5 (EPS_J5_NS) | SCF-2 (EPS_S2_NS) |
|---|---|---|---|---|
| eps | 31.02 +- 0.35 | **30.35 +- 0.44** | EPS_J5 | EPS_S2 |
| eps_inf | 1.798 | 1.797 | EPS_J5_INF | EPS_S2_INF |
| density (g/cm^3) | 1.0179 +- 0.0003 | 1.0173 +- 0.0006 | RHO_J5 | RHO_S2 |
| <U> (kJ/mol) | -2115060 +- 5 | -2115070 +- 8 | U_J5 | U_S2 |
| mean molecular dipole (D) | 1.9865 | 1.9864 | MU_J5 | MU_S2 |
| <T>, T_trans, T_rot (K) | 296.1, 297.0, 295.3 | 297.0, 296.6, 297.3 | T_J5 | T_S2 |

eps, density, <U> and the molecular dipole agree within 1-1.5 standard errors. The one systematic
difference is the equipartition between rotations and translations at 2 fs. With the global
(Bussi) thermostat, T_rot - T_trans is -1.7 K with SCF (an integration effect of the 2 fs step), and
TROT_TEXT. The heat that the auxiliary dipoles give the librations is removed only through the global
kinetic energy, so it shows up as slightly warmer rotations. Jacobi K5 heats 15 times faster than
block K7 in NVE.

## Speed

One RTX PRO 6000 Blackwell, mixed precision, 9 A cutoff, PME 48^3 per 512 waters (order 6),
`bench_md.py` (same GPU and session for every row; ms/step, ns/day in brackets):

| | 512 waters, Langevin 1/ps, 1 fs | 512, Bussi, 2 fs | 4,096 waters, Langevin, 1 fs | 4,096, Bussi, 2 fs |
|---|---|---|---|---|
| SCF, mu4 + CG, tol 1e-5 (5.5-6 iterations) | 0.770 (112) | 0.789 (219) | 2.029 (42.6) | 2.076 (83.2) |
| SCF, tol 1e-4 (4 iterations) | 0.672 (129) | 0.672 (257) | 1.699 (50.9) | 1.793 (96.4) |
| **iEL/0-SCF, block (default)** | | **0.506 (341)** | | **1.394 (124.0)** |
| iEL/0-SCF, Jacobi | 0.469 (184) | 0.473 (366) | 1.192 (72.5) | 1.309 (132.0) |
| iEL/SCF-1 | 0.497 (174) | 0.508 (340) | 1.270 (68.0) | 1.366 (126.5) |
| iEL/SCF-2 | 0.553 (156) | 0.546 (316) | 1.440 (60.0) | 1.519 (113.7) |

At 2 fs, iEL/0-SCF (block) is 1.56x (512 waters) and 1.49x (4,096) faster than the SCF solver at
tol 1e-5, and 1.33x / 1.29x faster than tol 1e-4 while conserving the energy better; iEL/SCF-2 gives
1.44x / 1.37x. Force call alone (`iel_cost.py`, 512 / 4,096 waters): CG iterations 1, 2, 3, 4, 6 cost
0.331, 0.384, 0.453, 0.563, 0.665 / 1.009, 1.134, 1.284, 1.442, 1.724 ms and 0-SCF (Jacobi) 0.292 /
0.842 ms, less than one CG iteration: the shadow terms ride in the force passes. The rest of a step
(integrator, neighbour list, bookkeeping) is 0.18 ms at 512 waters. In NPT the restart after each
accepted volume move (max(K, 2) + 1 converged steps) adds 0.25 CG iterations per step on average
(barostat every 100 steps, about 50 % acceptance).

## Limits

- At 2 fs the auxiliary dipoles are not adiabatically separated from the water librations (their
  frequencies are bounded by the Verlet limit 2/dt). The default drifts +0.013 kT/ns/dof in NVE,
  which is less than SCF at tol 1e-4 but 4 times SCF at tol 1e-5, and with a global thermostat the
  rotations run about 2 K warmer than with SCF. For strict NVE at 2 fs use iEL/SCF-2, or 0-SCF at
  1 fs. No choice of omega and K removes the exchange at 2 fs: all modes positive (omega lambda_max
  < 1) cools the atoms, some modes negative heats them, and without dissipation the mixed case is
  unstable.
- Niklasson's usual K = 5 with the Jacobi step heats at 2 fs (+0.2 kT/ns/dof) and K <= 4 is unstable
  there. Check other systems with `response_spectrum` (omega lambda_max) and a short NVE run.
- The dipoles are 1.6e-3 (relative RMS) from converged ones, where SCF at tol 1e-5 is at 8e-7: fine
  for eps, structure and dynamics above, but not for dipole-matching fits or anything that needs
  converged dipoles at each frame (use SCF there).
- The block preconditioner covers molecules of 2-8 atoms (water, small solutes); the atoms of larger
  molecules (proteins) get the Jacobi step. The flexible engine is supported but validated only on
  rigid water by constraints (`tests/test_iel.py`), not on flexible or large molecules.
- Not combined with multiple time stepping, alchemical regions, or `differentiable=True` (refused),
  or with replica exchange (not tested). The CG of the SCF path is not block-preconditioned.
- The virial (reported pressure) is taken at fixed mu = x + delta, without the shadow term; the
  barostat does not use it (Monte Carlo on converged energies).
- The Car-Parrinello-like variant (fictitious dipole mass with a cold dipole thermostat) is not
  implemented. It has the same cost per step as 0-SCF and the same adiabaticity limit at 2 fs:
  0-SCF with omega = 0.5 and no dissipation is effectively this scheme without the thermostat, and it
  cooled the atoms by 30 K in 200 ps at 2 fs.
