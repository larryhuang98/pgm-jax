# Fitting pGM to liquid and gas-phase properties: dielectric-constant gradients, multi-target fits, uncertainties

`pgm_jax/fit/` fits pGM parameters (charges, covalent dipoles, polarizabilities, radii, Lennard-Jones)
of a rigid-molecule liquid to several targets at once: density, heat of vaporization, static
dielectric constant, liquid molecular dipole, O-O g(r), gas-phase dipole and polarizability. Every
liquid observable comes with its exact ensemble gradient (fluctuation formulas with JAX derivatives
of each frame, including the induced dipoles' response to the parameters), a statistical error from
block jackknife, and the fitted parameters come with a covariance that is propagated to any predicted
property. `scripts/fit_multi.py` runs the iterations; `scripts/liquid_fit_tools.py` analyses them.

## Usage

```bash
# the base pGM water (Amber test pgm_512wat) toward experiment, six global scale factors
python scripts/fit_multi.py -o runs/fit/demo --model base --params q,cov,alpha,radius,lj_r,lj_eps --prior 0.3 \
    --targets density=0.997:0.002,hvap=10.52:0.05,eps=78.4:1.5,gas_dipole=1.855:0.01,gas_polarizability=1.47:0.01,liquid_dipole \
    --equil 50 --prod 2000 --every 0.5 --iters 8 --max-minutes 35        # resumable: rerun the same line
# measure only (bare target names): values, Jacobians, errors at --start; segments at fixed theta
python scripts/fit_multi.py -o runs/fit/c0 --params q,alpha --targets density,hvap,eps,liquid_dipole --fixed --iters 4
python scripts/liquid_fit_tools.py combine runs/fit/c0 --params q,alpha --targets density,hvap,eps,liquid_dipole
# small boxes on CPUs: batched NVT replicas (jax.vmap), e.g. 16 x 64 waters
python scripts/fit_multi.py -o runs/fit/r --coords box64.rst7 --cutoff 0.45 --skin 0.08 --ewald-beta 6 --nfft 20 \
    --ensemble nvt --replicas 16 --nblocks 32 --targets hvap=...,eps=...,gas_dipole=...
```

Targets are `name=value:sigma[:weight]` (a bare name is evaluated and propagated, not fitted);
`sigma` is the tolerance the fit may leave (experimental or model error); `rdf=FILE:sigma:weight`
takes a reference g(r). Parameters are global scale factors theta = ln s on the prmtop's table
(`q`, `cov`, `alpha`, `radius`, `lj_r`, `lj_eps`: the Bayesian optimisation's scale_q, scale_p,
scale_pol, scale_rad and the LJ R*, epsilon scales); in Python, `Param(quantity, "scale" | "shift",
keys=[...])` acts on individual tying keys (per-type values).

```python
from pgm_jax.fit import FrameAnalyzer, GasPhase, LiquidSamples, Objective, ParameterSpace, RDFSpec, Target
from pgm_jax.fit.liquid import LiquidFit
space = ParameterSpace.scales(sys.table, ["q", "cov", "alpha", "radius", "lj_r", "lj_eps"], prior_sigma=0.3)
gas = GasPhase(sys.molecules[0], pos[sys.atom_slice(0)], sys.table, space)
obj = Objective([Target("density", 0.997, 0.002), Target("eps", 78.4, 1.5), Target("gas_dipole", 1.855, 0.01), ...],
                space, gas=gas)
fit = LiquidFit(sys, pos, H, space, obj, settings=MDSettings(pme_grid=(48,) * 3), dt=0.002, prod_ps=2000, prefix="fit")
fit.run(theta0, iters=6)                               # prefix.json: every iteration, prediction checks, UQ
an = FrameAnalyzer(sys, H, settings, space)            # any frames (positions, box, dipole guess), batched
out = an.analyze(theta, [(pos_k, H_k, mu_k), ...])     # U, dU, M, dM, alpha, dalpha, D, dD, V, rdf
```

## Method

**Per frame** (`frames.py`, vmapped over chunks of frames, pair rows from a dense cutoff search so
no neighbour-list state is needed): the induced dipoles are re-solved to a tight tolerance from the
MD's dipoles, then

- U and dU/dtheta at the converged dipoles (the energy is variational in mu: Hellmann-Feynman);
- the cell dipole M = M_q + M_perm + M_ind and dM/dtheta. The induced part depends on theta through
  A(theta) mu = b(theta); with the adjoint A lambda_c = e_c (unit vector on every atom, A symmetric)
  dM_c/dtheta = d(M_q + M_perm)_c/dtheta + lambda_c . d(b - A mu)/dtheta at fixed mu, lambda;
- lambda_c is the response to a uniform unit field, so the same three solves give the cell
  polarizability alpha_cell (eps_inf) and d alpha_cell/dtheta = -(1/3) sum_c lambda_c . dA/dtheta lambda_c;
- the mean molecular dipole D (one more adjoint solve) and the volume, g(r) histograms.

One `jax.jacrev` of (U, M, alpha_cell, D) per frame gives all derivatives.

**Ensemble** (`estimators.py`): every liquid observable is a function of averages of per-frame
quantities, differentiated through the linear-exponential reweighted average
<a>(delta) = sum_k w_k(delta) (a_k + da_k/dtheta . delta), w_k ~ exp(-beta dU_k/dtheta . delta); at
delta = 0 its derivative is the fluctuation formula d<a>/dtheta = <da/dtheta> - beta cov(a, dU/dtheta).
For the dielectric constant (tin-foil Ewald, adiabatic induced dipoles, docs/dielectric.md)

    eps = 1 + 4 pi <alpha_cell/V> + 4 pi KE (<M.M> - <M>.<M>) / (3 kB T <V>)
    d eps/dtheta = 4 pi d<alpha_cell/V> + 4 pi KE / (3 kB T) [ (d<M.M> - 2 <M>.d<M>) / <V> - (<M.M> - <M>.<M>) d<V> / <V>^2 ],

with d<M.M> = <2 M . dM/dtheta> - beta cov(M.M, dU/dtheta) etc. Heat of vaporization
Hvap = U_gas(theta) - <U>/N + RT with the monomer energy of the rigid molecule (pGM with every
pair; equal to the MD engine's energy of one molecule in a large box, pytest), gas-phase dipole and
polarizability from `GasPhase` (exact, autodiff). Statistical errors: jackknife over contiguous
blocks for all observables together (their covariance) and their Jacobians.

**Fit** (`optimize.py`): residuals sqrt(w)(y - t)/s with s^2 = sigma_tol^2 + sigma_stat^2 plus a
Gaussian prior on theta; Levenberg-Marquardt step in a trust region |d/sigma_prior| <= radius; the
radius adapts to the ratio of achieved to predicted chi2 decrease measured by the next simulation.
Each record holds the predictions for the next iteration: linear, linear-exponential reweighting
and (optional, `--exact k`) exact reweighting of every k-th frame re-evaluated at the new
parameters, with n_eff.

**Uncertainty**: at the fixed point theta_fit is linear in the measured observables,
C_theta = G J_w^T S Sigma_y S J_w G with G = (J_w^T J_w + Sigma_prior^-1)^-1 (sampling error of the
fitted parameters; = (J^T Sigma_y^-1 J)^-1 without tolerances and prior); `posterior_cov` G reads
the tolerances as Gaussian target errors. Propagation: J_p C_theta J_p^T for any property p. Block
bootstrap of y, J and the step as a check.

## Validation

**Per frame and estimators (pytest, `tests/test_liquid_fit.py`, 13 tests, ~75 s on CPU).**

| Check | Result |
|---|---|
| U, M, alpha_cell, mean molecular dipole of FrameAnalyzer vs PGMForceField / CellDipole (float64) | 1e-8 relative (U), 1e-10 (M, D), 1e-9 (alpha_cell) |
| dU, dM, d alpha_cell, dD vs central differences with the dipoles re-solved, six scale factors (q, cov, alpha, radius, R*, eps), 30 waters + 4 methanols, float64 | max deviation < 2e-6 of the largest component |
| mixed precision (tol 1e-6) vs float64 derivatives | < 2e-4 relative (dU < 1e-4) |
| vmapped batches vs single frames; g(r) vs a numpy histogram | 1e-9 |
| estimator Jacobians (density, Hvap, eps, liquid dipole, alpha_p, kappa_t) vs the explicit fluctuation formulas on synthetic frames | 1e-8 relative |
| GasPhase energy vs the MD engine's energy of one molecule in a 6 nm box; gas-phase gradients vs finite differences | 2e-3 kJ/mol (image terms); 1e-6 |
| LM step and C_theta on a linear-Gaussian model (4000 noise realisations, correlated errors) | spread of fitted theta = C_theta within 10 %; C_theta = (J^T S^-1 J)^-1 without tolerances |

**Reference values of the base water** (512 waters, NPT 298 K, 2 fs, 2 ns, pgm_jax; brief/Amber: eps 72-73,
density 0.975, liquid dipole 2.54 D): eps 73.3 +- 1.5 (eps_inf 2.006), density 0.9763 +- 0.0011 g/cm^3,
liquid dipole 2.537 D, Hvap 6.812 +- 0.003 kcal/mol, gas-phase dipole 1.858 D, polarizability 1.984 A^3.
pGM3P-25 (10 ps, CPU): density 1.016, liquid dipole 2.12 D (reference 2.12 D), gas dipole 1.462 D,
polarizability 1.488 A^3.

**(a) Ensemble gradients against finite differences between independent simulations.** Runs at
ln s_q = -delta, 0, +delta (charges scaled), each with its own gradient; the finite difference
(y+ - y-)/(2 delta) is the mean slope over the interval, compared with Simpson's rule over the three
gradients, (g- + 4 g0 + g+)/6. Errors: jackknife over blocks (replicas for the batched runs).

512 base waters, NPT (Monte Carlo barostat), 2 fs, Bussi, 4 ns per point (8000 frames), delta = 0.03
(first 2 ns dropped in the second column):

| Observable | y(-0.03) / y(0) / y(+0.03) | FD slope | gradient (Simpson) | z | z (last 2 ns) |
|---|---|---|---|---|---|
| eps | 57.0 / 74.9 / 92.9 (+-0.8) | 599 +- 21 | 727 +- 84 | -1.5 | -1.1 |
| density (g/cm^3) | 0.9116 / 0.9753 / 1.0306 | 1.983 +- 0.017 | 1.905 +- 0.057 | +1.3 | +0.6 |
| Hvap (kcal/mol) | 5.939 / 6.809 / 7.739 | 29.99 +- 0.03 | 29.64 +- 0.18 | +2.0 | +1.2 |
| liquid dipole (D) | 2.3791 / 2.5371 / 2.6972 | 5.303 +- 0.003 | 5.271 +- 0.016 | +2.0 | +1.0 |
| eps_inf | 1.929 / 2.004 / 2.071 | 2.366 +- 0.021 | 2.273 +- 0.069 | +1.3 | +0.6 |
| alpha_p (1/K) | 2.1e-3 / 1.5e-3 / 1.3e-3 | -0.014 +- 0.002 | -0.006 +- 0.010 | -0.7 | +0.4 |
| kappa_t (1/bar) | 2.2e-4 / 1.5e-4 / 1.3e-4 | -0.0017 +- 0.0002 | -0.0009 +- 0.0007 | -1.2 | -0.5 |

64 base waters (cutoff 0.45 nm, Ewald 6 /nm, PME 20^3; a toy box for statistics), NVT, 16 batched
replicas per point, delta = 0.08 (eps 51 -> 78 -> 123: strongly nonlinear):

| Observable | FD slope, 2 fs (CPU, 16 x 350-600 ps) | Simpson | z | FD slope, 1 fs (GPU, 16 x 400 ps) | Simpson | z |
|---|---|---|---|---|---|---|
| eps | 447.6 +- 10.5 | 425.3 +- 28.2 | +0.7 | 427.4 +- 8.4 | 393.1 +- 17.0 | +1.8 |
| U/N (kJ/mol) | -4547.07 +- 0.09 | -4546.94 +- 0.24 | -0.5 | -4546.88 +- 0.12 | -4546.92 +- 0.23 | +0.1 |
| Hvap (kcal/mol) | 27.32 +- 0.02 | 27.28 +- 0.06 | +0.6 | 27.27 +- 0.03 | 27.28 +- 0.05 | -0.1 |
| eps_inf | -0.0538 +- 0.0002 | -0.0535 +- 0.0007 | -0.4 | -0.0539 +- 0.0002 | -0.0535 +- 0.0005 | -0.9 |
| liquid dipole (D) | 4.940 +- 0.002 | 4.935 +- 0.007 | +0.7 | 4.936 +- 0.002 | 4.935 +- 0.005 | +0.2 |

The fluctuation part of these derivatives is large for eps (it is almost all of it) and small for the
energy (22 of 4527 kJ/mol per molecule); both agree. With 2-4 segments (first runs) three of the 2 fs
comparisons were at z ~ 3; they came down with more data (and with the first segment dropped),
an equilibration transient after changing theta rather than an integrator bias (1 fs and 2 fs agree).

Statistical precision: the gradient of eps is much noisier than eps itself (512 waters, 4 ns:
eps +- 1 %, d eps/d ln s_q +- 13-35 %), and at a fixed simulation time its relative error grows
roughly as sqrt(N) (covariance of two extensive fluctuating quantities): 16 x 400 ps of 64 waters give
the slope to 4-5 %. Independent simulations at theta +- delta measure a directional derivative more
precisely than the fluctuation formula; the formula gives all parameters' derivatives from one run.

## Speed

## Limits

- Rigid molecules (the `Simulation` engine). `FrameAnalyzer` evaluates the PGMForceField energy
  (pGM + intermolecular van der Waals): flexible molecules would need their bonded and intramolecular
  LJ terms in U and a gas-phase ensemble for Hvap (`scripts/fit_liquid.py` has the latter for LJ only);
  charge flux and virtual sites are refused.
- `GasPhase` is one rigid molecule (the monomer geometry of the liquid); `fit_multi.py` handles boxes
  of one molecule type (the library takes any system; gas-phase targets per molecule type).
- Reweighting predictions collapse for the steps a fit takes: with 512 waters n_eff ~ 1 of 4000 for
  |d theta| ~ 0.05 in ln s_q (linear-exponential and exact alike), so the next iteration is predicted
  by the linear model (liquid) and exact gas-phase values, and verified by the next simulation.
- Steps in the strongly nonlinear regime (base water: eps and density change by 2x for 8 % charge
  scaling) are only as good as the trust region; the ratio test uses noisy chi2 values and no step is
  rejected (the next simulation always runs at the new parameters).
- The parameter covariance is the linearised sampling covariance at the fixed point; the Jacobian's
  own noise enters only through the bootstrap.
- `alpha_p` (thermal expansion) and `kappa_t` (compressibility) have exact estimator gradients
  (third cumulants through the reweighted averages; pytest) but were not used or validated in fits;
  there are no other temperature-derivative targets (Cp, TMD).
- Batched replicas run NVT only (md/remd.MDReplicas); NPT runs one simulation.
