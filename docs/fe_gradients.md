# Free energies as fitting targets: parameter gradients of alchemical free energies

`pgm_jax/md/fe_grad.py` gives hydration (solvation) free energies, and any alchemical free
energy computed with `pgm_jax/md/alchemy.py`, **with their gradients with respect to every
force-field parameter** of the solute and of the solvent, and statistical errors of both, in a
form a fitting code can use as one more target. Units: kJ/mol, nm, e, e nm (tables also in kcal/mol).

At a glance: dDeltaG/dtheta = <dU_{K-1}/dtheta>_{K-1} - <dU_0/dtheta>_0 (+ the gas-phase leg), at
the converged induced dipoles, for the whole parameter table from the samples the windows already
take (+1 % run time); MBAR-weighted and end-state estimators with block-jackknife errors; checked
against finite differences over 11 independent free-energy calculations (water: solute charge and
r_min scales; flexible methanol: charge and r_min scales), all within 1.1 standard errors, and
exactly on analytic and zero-variance cases. A one-step fit of methanol's charge scale to its
experimental hydration free energy is shown.

## Theory

For a Hamiltonian U_k(x; theta) sampled at fixed volume and temperature, f_k(theta) = -kT ln
Z_k(theta) has the exact derivative

    df_k/dtheta = < dU_k/dtheta >_k .

The solution leg of the cycle (`docs/free_energy.md`) goes from window 0 (full coupling) to window
K-1 (decoupled), so

    d DeltaG_solv/dtheta = < dU_{K-1}/dtheta >_{K-1} - < dU_0/dtheta >_0 ,
    d DeltaG_hyd/dtheta  = d DeltaG_gas/dtheta - d DeltaG_solv/dtheta .

- Only the end states enter. With annihilation the decoupled end state does not see the
  solute's charges, dipoles, polarizabilities (up to the floor eps = 1e-8) or its coupling to
  the solvent, but it does see every solvent parameter, and with `intramolecular="keep"` the
  solute's own gas-phase electrostatics and (flexible molecules) intramolecular van der Waals.
- **Gas-phase leg.** For a rigid solute DeltaG_gas = E_gas(0) - E_gas(1) is exact, and so is its
  gradient (`gas_leg_gradient`, autodiff through the dense gas-phase pGM solve). With
  `intramolecular="keep"` the leg is part of the Hamiltonian and the identity handles it.
- **dU/dtheta at the converged dipoles.** The pGM energy is stationary in the induced dipoles, so
  dU_k/dtheta is the partial derivative at the dipoles converged in Hamiltonian k
  (Hellmann-Feynman for parameters; no derivative of the CG solve). `ParamGradients` re-solves the
  dipoles of every window's configuration in the two end-state Hamiltonians (from the
  configuration's own dipoles) and takes one reverse-mode pass of the fixed-dipole energy with
  respect to the whole parameter table, batched over the windows with `jax.vmap` on the device.
- theta is the **parameter table** (`sys.table`, every quantity: q, cov, radius, alpha,
  lj_rmin_half, lj_sqrt_eps, ...; `ParamSpace` flattens it, names `quantity:key`). Any other
  parameterization theta' -> table follows by the chain rule (`FEGradient.chain`,
  `FreeEnergyTarget.value_and_grad(theta_fn, theta)`), e.g. scale factors, atom types shared by
  solute and solvent (`alchemical_map(sys0, sysA)` maps the original table onto the alchemical one,
  so a gradient with respect to the water model sums the solute copy and the solvent).
- **Fixed volume.** The windows run NVT at the mean NPT volume of the reference parameters, so
  the free energy and its gradient are Helmholtz quantities at that volume. The difference from
  the Gibbs free energy at 1 bar is small (`docs/free_energy.md`: P DeltaV and DeltaV^2/(2 V
  kappa_T), below 0.01 kcal/mol for a water in 512); its parameter derivative is smaller still.

## Estimators and errors (`gradient_estimate`)

| Estimator | What | Meaning |
|---|---|---|
| `mbar` | E_k[dU_k/dtheta] = sum_n W_nk dU_k/dtheta(x_n) with the MBAR weights of the end states over the samples of every window | the exact derivative of the MBAR free energy of the perturbed end states reweighted from the sampled mixture: consistent with the MBAR DeltaG |
| `end` | the plain averages over the end windows' own samples | the identity directly |

Errors: **block jackknife over time** (default 10 contiguous blocks; the same time block is left out
of every window, which keeps inside a block both the time correlation of each window and the
correlation between windows coupled by Hamiltonian exchange). The replicates are kept
(`FEGradient.jk_value`, `jk_grad`), so the error of any projection (a scale direction, a chain-rule
product) is exact, not propagated from per-entry errors. The free energy is MBAR on all samples
(its jackknife error is reported with the asymptotic MBAR error of `free_energy.estimate`). The two
estimators agree within their errors in every run below; the MBAR-weighted one has 1.3-2.4 times
smaller errors (it also uses the samples of the neighbouring windows) and is the default of
`FreeEnergyTarget`.

## Usage

```python
from pgm_jax.md import fe_grad as fg
from pgm_jax.md.alchemy import FreeEnergyRun, GasPhaseLeg, LambdaWindows, standard_schedule

win = LambdaWindows(sim, standard_schedule(8))
pg = fg.ParamGradients(win)                                  # end states 0 and K-1, whole table
gas = GasPhaseLeg(alch, xyz_solute, "qpi")                   # rigid solute, annihilation
dg_gas, g_gas = fg.gas_leg_gradient(gas, P, pg.space)
run = FreeEnergyRun(win, sample_every=500, exchange_every=500, param_grad=pg,
                    meta={"gas_delta_g": dg_gas, "gas_grad": g_gas.tolist(), ...})
run.run(500000, prefix="wat")
r = fg.gradient_estimate(fe.load("wat_fe.npz"), discard_ps=100, gas={"delta_g": dg_gas, "grad": g_gas})
h = r["hyd"]["mbar"]                                         # FEGradient
h.value, h.value_err, h.grad, h.grad_err                     # kJ/mol, per table entry
h.project(pg.space.scale_direction(p_flat, "charge"))        # dG/d ln s_charge of the solute, error

# as a fitting target (theta_fn: theta -> parameter table of the run's system)
t = fg.FreeEnergyTarget.from_npz("wat_fe.npz", discard_ps=100, experiment=-6.3 * 4.184, sigma=0.2 * 4.184)
t.value_and_grad(theta_fn, theta)      # value, grad (per theta), errors, chi2, dchi2
t.estimate(theta_fn, theta, unit="kcal/mol")   # y, J, cov_y, J_err, leave-one-block-out replicates
fg.combine([(1.0, a), (-1.0, b)])      # relative / transfer free energies, log P (independent runs)
```

```bash
python scripts/solvation_free_energy.py run --model pgm -o runs/fg/w --grad             # + gradients
python scripts/solvation_free_energy.py run ... --grad --solute-scale charge=1.1,eps=0.9  # scaled solute
python scripts/solvation_free_energy.py run ... --start-from runs/fe/pgm.fe.chk --elec-only   # no NPT; 8 windows
python scripts/solvation_free_energy.py analyze runs/fg/w_fe.npz --discard-ps 100       # prints the gradients
python scripts/fe_gradient_check.py --group charge runs/fg/a_fe.npz runs/fg/b_fe.npz runs/fg/c_fe.npz
```

`analyze` prints the hydration free energy, its derivatives along the scale directions of the
solute's and of the environment's parameters (charge = charges and covalent dipoles together, eps,
rmin, alpha, radius; d/d ln s) by both estimators, and dG/dp of every solute entry.

## Validation

### Exact and analytic checks (`tests/test_fe_grad.py`, float64, CPU)

| Check | Result |
|---|---|
| dU/dP at the converged dipoles (autodiff, fixed mu) against central differences of the energy with the dipoles re-solved, along random directions of the whole table (charges, covalent dipoles, radii, polarizabilities, LJ of solute and solvent); annihilation at lambda = (1, 1), (0.5, 1), (0, 0.4); keep at (0.4, 1), (0, 0) | relative 1e-12 to 5e-10 |
| the estimators as exact derivatives on stored frames (5 windows x 8 samples of 64 waters): 'end' = d/dtheta of the exponential-averaging free energies of the perturbed end states, 'mbar' = d/dtheta of the MBAR free energies of the perturbed end states reweighted from the sampled mixture (energies re-solved at theta +- h) | 5e-8 to 1e-5 relative (the finite-difference roundoff) |
| sampler: batched windows = sequential = direct autodiff per configuration; rigid (annihilate) and flexible methanol (keep; its gas-phase electrostatics at the decoupled end = dE_gas/dq) | 1e-7 |
| **exactly known case**: a lone rigid water in a 4.2 nm box, windows lambda_elec = 1, 0.5, 0; its energy does not depend on the configuration, so DeltaG and its gradient are E_gas(0) - E_gas(1) and its autodiff gradient, with zero variance | both estimators: DeltaG to 2e-4 kJ/mol of 699; gradient to 3.7e-5 relative (the box's image / PME error); error bars 1e-3 of 2e4 (zero) |
| gas-phase leg gradient against central differences of E_gas(0) - E_gas(1) | 1e-6 |
| **analytic toy**: harmonic states u_k = K_k exp(c_k theta) x^2 / 2, d(f_3 - f_0)/dtheta = (c_3 - c_0)/2 = -0.7, correlated samples (AR(1), phi = 0.9, 4 x 2000), 40 repeats | end -0.6983 +- 0.0072, MBAR -0.7040 +- 0.0062; jackknife error bar / spread of the estimates 0.98 and 0.93; free energy 0.8075 +- 0.0054 (exact 0.8047), error ratio 0.97 |
| driver: samples dudp (S, 2, K, M) with names and sampled parameters in the npz, restart with gradients, refusal to mix, `load_windows`; target API (chain rule = projection, chi^2 gradient, fit layout, `alchemical_map` sums the solute copy and the solvent key, `combine`) | pass |

### Finite differences over independent free-energy calculations

**A. Water in pGM water, the solute's charge scale** (charges and covalent dipoles of the solute
water times s; polarizabilities unchanged). Only the electrostatics stage depends on these
parameters (the van der Waals stage runs at lambda_elec = 0), so each run is the 8 electrostatics
windows (lambda_elec = 1 ... 0 at lambda_vdw = 1) plus the exact gas-phase leg: G(s) = DeltaG_gas(s) -
DeltaG_elec(s), which differs from the hydration free energy by the (s-independent) van der Waals
stage. Settings of `docs/free_energy.md` (512 waters, PME 48^3, mixed precision, 2 fs, Bussi, samples
and Hamiltonian exchange every 1 ps); the windows start from the configurations of the 2-ns
production run at s = 1 (`--start-from`), 0.6 ns per window, the first 100 ps discarded (500
samples per window); `scripts/fe_gradient_check.py`.

| s | G(s) (MBAR, kcal/mol) | dG/ds MBAR-weighted | dG/ds end states |
|---|---|---|---|
| 0.9 | -5.280 +- 0.058 | -15.43 +- 0.13 | -15.40 +- 0.21 |
| 1.0 | -6.912 +- 0.054 | -18.57 +- 0.08 | -18.67 +- 0.12 |
| 1.1 | -8.964 +- 0.040 | -21.75 +- 0.10 | -21.83 +- 0.21 |

| Comparison | finite difference | from the gradients | z |
|---|---|---|---|
| 0.9 -> 1.0: (G(1) - G(0.9)) / 0.1 vs trapezoid | -16.32 +- 0.79 | -17.00 +- 0.08 (end -17.04) | 0.85 (0.89) |
| 1.0 -> 1.1 | -20.52 +- 0.68 | -20.16 +- 0.06 (end -20.25) | -0.53 (-0.39) |
| 0.9 -> 1.1, central difference vs g(1) | -18.42 +- 0.35 | -18.57 +- 0.08 (end -18.67) | 0.40 (0.67) |
| 0.9 -> 1.1 vs Simpson (g(0.9) + 4 g(1) + g(1.1)) / 6 | -18.42 +- 0.35 | -18.57 +- 0.06 (end -18.65) | 0.43 (0.63) |

kcal/mol per unit s; errors: block jackknife (10 blocks; the asymptotic MBAR errors of G are
0.059, 0.073, 0.071). The gradients agree with the finite differences within 0.9 standard errors,
and the gradient is 4-9 times sharper than the finite difference from the same simulations. G(s)
grows faster than s^2 (d ln G / d ln s = 2.69 +- 0.02 at s = 1, where linear response of the solvent
would give 2): the solvent's response to the solute's charges is not linear (hydrogen bonds
strengthen), so the curvature is real; Simpson's rule, exact for a cubic G, and the plain central
difference agree here.

**B. Water in pGM water, the solute's Lennard-Jones r_min scale** (R* of the solute's oxygen
times s): both stages depend on it (the solute-water Lennard-Jones at lambda_vdw = 1 and the soft
core, whose w = (r/rmin)^6 + ... depends on r_min), so each run is all 19 windows plus the exact gas
leg (which does not depend on r_min). 0.4 ns per window from the production configurations at s =
1, the first 50 ps discarded (350 samples per window).

| s | G(s) = DeltaG_hyd (MBAR, kcal/mol) | dG/ds MBAR-weighted | dG/ds end states |
|---|---|---|---|
| 0.97 | -5.717 +- 0.058 [0.106] | 40.94 +- 0.54 | 41.82 +- 0.88 |
| 1.00 | -4.453 +- 0.084 [0.107] | 37.11 +- 0.51 | 37.58 +- 0.98 |
| 1.03 | -3.542 +- 0.099 [0.099] | 32.07 +- 0.36 | 32.38 +- 0.86 |

| Comparison | finite difference | from the gradients | z |
|---|---|---|---|
| 0.97 -> 1.00 vs trapezoid | 42.1 +- 3.4 | 39.03 +- 0.37 (end 39.70) | 0.91 (0.70) |
| 1.00 -> 1.03 | 30.4 +- 4.3 | 34.59 +- 0.31 (end 34.98) | -0.98 (-1.06) |
| 0.97 -> 1.03, central difference vs g(1) | 36.2 +- 1.9 | 37.11 +- 0.51 (end 37.58) | -0.44 (-0.62) |
| 0.97 -> 1.03 vs Simpson | 36.2 +- 1.9 | 36.91 +- 0.35 (end 37.42) | -0.34 (-0.58) |

[ ]: asymptotic MBAR error of G (subsampled); the z values use the jackknife errors (with the
asymptotic ones they are smaller). The hydration free energy at s = 1 (-4.45 +- 0.08 from 0.35 ns
per window, started from the production windows) is the production value of `docs/free_energy.md`
(-4.45 +- 0.05).

**C. Flexible methanol (intramolecular="keep") in pGM water, and a one-parameter fit to
experiment.** The methanol of `docs/free_energy.md` (class II bonded terms, ESP-fitted pGM
electrostatics, GAFF Lennard-Jones; X-H bonds constrained, rigid waters; 19 windows; the gas-phase
leg inside the Hamiltonian, so both end states depend on the solute's electrostatic parameters).
0.4 ns per window from the production windows, 50 ps discarded (350 samples per window).

| s (solute charge scale) | DeltaG_hyd (kcal/mol) | dG/ds MBAR-weighted | dG/ds end states |
|---|---|---|---|
| 1.0 (the model) | -2.741 +- 0.094 [0.113] | -10.15 +- 0.13 | -10.19 +- 0.20 |
| 1.2335 (one Newton step) | -5.321 +- 0.084 [0.132] | -13.02 +- 0.13 | -12.92 +- 0.20 |

At s = 1 the hydration free energy agrees with the 1.2-ns production value (-2.65 +- 0.07); the
experimental value is -5.11 (Ben-Naim & Marcus 1984). One Newton step on the charge scale with the
computed gradient, s_1 = 1 + (-5.11 + 2.741) / (-10.145) = 1.2335 (`examples/hydration_target.py`
prints it from the npz, with the chi^2 and the fit layout), was run: -5.32 +- 0.08, 0.21 kcal/mol
past the target, as expected from the curvature (dG/ds grows by 28 % over the step; a quadratic
model through G(1) and g(1) predicted -5.30). The next Newton step from s_1, with the gradient
computed there, is s_2 = 1.2335 + (-5.11 + 5.321) / (-13.02) = 1.217. The pair is also a
finite-difference check of the flexible "keep" path: (G(1.2335) - G(1)) / 0.2335 = -11.05 +- 0.54
against the trapezoid of the gradients -11.58 +- 0.09 (end states -11.56 +- 0.14), z = 0.98 (0.91).
(A charge scale of 1.23 is only a demonstration of the machinery, not a proposed methanol model:
the missing hydration comes as much from the weakly polar pGM water, `docs/free_energy.md`.)

**D. The same methanol, the solute's Lennard-Jones r_min scale** (R* of every methanol atom times
s: its coupling to water through the soft core and, with "keep", its intramolecular van der Waals
at both end states). 0.4 ns per window from the production windows, 50 ps discarded.

| s | DeltaG_hyd (kcal/mol) | dG/ds MBAR-weighted | dG/ds end states |
|---|---|---|---|
| 0.95 | -3.372 +- 0.085 [0.103] | 13.64 +- 0.61 | 13.85 +- 0.71 |
| 1.00 | -2.741 +- 0.094 [0.113] | 12.48 +- 0.34 | 12.37 +- 0.51 |
| 1.05 | -2.065 +- 0.077 [0.109] | 12.10 +- 0.23 | 11.70 +- 0.43 |

| Comparison | finite difference | from the gradients | z |
|---|---|---|---|
| 0.95 -> 1.00 vs trapezoid | 12.6 +- 2.5 | 13.06 +- 0.35 (end 13.11) | -0.17 (-0.19) |
| 1.00 -> 1.05 | 13.5 +- 2.4 | 12.29 +- 0.21 (end 12.04) | 0.51 (0.61) |
| 0.95 -> 1.05, central difference vs g(1) | 13.1 +- 1.1 | 12.48 +- 0.34 (end 12.37) | 0.50 (0.56) |
| 0.95 -> 1.05 vs Simpson | 13.1 +- 1.1 | 12.61 +- 0.25 (end 12.51) | 0.40 (0.47) |

**Summary of the finite-difference checks**: 4 parameter directions, 11 independent free-energy
calculations; all 16 comparisons (both estimators) within 1.06 standard errors (|z| mean 0.6); the
gradient is 3-10 times more precise than the finite difference built from the same simulations.

## Cost

One RTX PRO 6000 Blackwell, pGM water box (512 waters, 1,536 atoms), mixed precision, the whole
table (13 solute + 13 solvent entries); `solvation_free_energy.py bench --grad`:

| | 8 windows | 19 windows |
|---|---|---|
| MD, ms per window-step (batched) | 0.400 | 0.395 |
| one sample of u_k(x_n) and dU/dlambda (existing) | 15 ms | 50 ms |
| one parameter-gradient sample (2 end states x K configurations: dipole re-solve + reverse pass) | 20 ms | 43 ms |
| gradient overhead at samples every 1 ps (500 steps) | 1.2 % | 1.1 % |

The runs of the validation below (samples, gradients and exchanges every 1 ps) ran at 49-51
ns/day per window with 8 windows, 21-22 with 19 windows of water and 20.3 with flexible methanol
(segments of 0.2-0.6 ns, compilation included), as the production runs without gradients (22.8 and
21.4, `docs/free_energy.md`). The gradient costs nothing compared with the free energy
itself, and it is statistically much sharper than a finite difference of free energies: 0.6 ns of
8 windows gave the solute-charge derivative of the hydration free energy to +-0.08 kcal/mol per ln
s (0.4 %), where a central difference over +-10 % with two free energies of the same length has
+-0.5 kcal/mol.

## Limits

- **Solvent parameters: correct but noisy.** d DeltaG/dtheta for a solvent parameter is the
  difference of two averages of an extensive quantity (dU/dtheta of the whole box), whose
  fluctuations grow as the square root of the number of solvent molecules, while the answer is
  local. The estimators are unbiased (the identity holds for every parameter), but 0.6 ns of 8
  windows gave the pGM-water charge scale of a water's hydration free energy only to +-6-9 kcal/mol
  per ln s (the solute's to +-0.1). Fitting a water model to hydration free energies needs much
  longer runs or a local estimator (not implemented).
- End states only: the intermediate windows enter through MBAR weights; no TI-type path estimator
  of the gradient (it would need d^2U/dlambda dtheta and covariances, no better for these cases).
- Helmholtz free energy at the volume of the reference parameters (the windows run NVT); the
  change of the equilibrium volume with theta is not followed (second order, below 0.01 kcal/mol
  for the free energy itself).
- Parameters of the table only (charges, covalent dipoles, radii, polarizabilities, Lennard-Jones,
  and the other table quantities); the bonded parameters of a flexible solute's template are not
  in the table and are not differentiated.
- Gradients at the dipoles converged to the run's tolerance (1e-5): the Hellmann-Feynman
  derivative has an error of first order in the dipole residual (~1e-5 relative, far below the
  statistical error).
- One solute; neutral; Lennard-Jones (GVDW and charge flux are refused by Alchemy, as for the free
  energies themselves). A sampled gas-phase leg (flexible solute with annihilation) has no
  gradient support; flexible solutes use intramolecular="keep", where the identity covers the gas
  phase.
- Relative free energies, transfer free energies and partition coefficients: `combine` adds
  independent results (values, gradients, errors in quadrature); only water boxes were run here, so
  no log P was computed.
