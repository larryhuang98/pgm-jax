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
scale_pol, scale_rad and the LJ R*, epsilon scales), or per tying key: `alpha@OW,alpha@HW` (one scale per
atom type); in Python, `Param(quantity, "scale" | "shift", keys=[...])` also gives additive per-key
values. `--prior` sets the Gaussian prior width on each ln s (regularisation toward the prmtop's values,
or `--prior-center`), `--radius` the initial trust radius in units of the prior widths, `--exact k`
the exact-reweighting prediction on every k-th frame, `--fixed` a measurement at fixed theta in
resumable segments, `--max-minutes` the time budget of one (GPU) job. Outputs: `prefix.json` (every
iteration: observables, errors and their covariance, Jacobian and its errors, the check of the previous
prediction, step, predictions, C_theta, posterior covariance, propagated errors, bootstrap),
`prefix_frames*.npz` (per-frame data, for re-analysis), `prefix_state.npz`, `prefix.log`.

```python
from pgm_jax.fit import FrameAnalyzer, GasPhase, LiquidSamples, Objective, ParameterSpace, RDFSpec, Target
from pgm_jax.fit.liquid import LiquidFit

space = ParameterSpace.scales(sys.table, ["q", "cov", "alpha", "radius", "lj_r", "lj_eps"], prior_sigma=0.3)
gas = GasPhase(sys.molecules[0], pos[sys.atom_slice(0)], sys.table, space)
obj = Objective(
    [Target("density", 0.997, 0.002), Target("eps", 78.4, 1.5), Target("gas_dipole", 1.855, 0.01), ...], space, gas=gas
)
fit = LiquidFit(sys, pos, H, space, obj, settings=MDSettings(pme_grid=(48,) * 3), dt=0.002, prod_ps=2000, prefix="fit")
fit.run(theta0, iters=6)  # prefix.json: every iteration, prediction checks, UQ
an = FrameAnalyzer(sys, H, settings, space)  # any frames (positions, box, dipole guess), batched
out = an.analyze(theta, [(pos_k, H_k, mu_k), ...])  # U, dU, M, dM, alpha, dalpha, D, dD, V, rdf
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

O-O g(r) (reweighting-only gradient, d<g>/dtheta = -beta cov(g, dU/dtheta)), same 512-water runs, 56 bins of
0.01 nm from 0.245 to 0.8 nm: rms z 1.01, 93 % of the bins within |z| < 2 (max 2.9), correlation of the
finite-difference and fluctuation slopes over the bins 0.989 (g(0.275 nm) 2.139, 2.190, 2.237 at ln s_q = -0.03, 0, +0.03).

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

**(c) Calibration of the uncertainties.** The 16 replicas of the 64-water run at theta = 0 (2 fs,
600 ps each after 40 ps) are independent trajectories: each gives its own observables, jackknife
errors (10 blocks of 60 ps, conservative rule), Jacobian, fit and predicted C_theta
(`liquid_fit_tools.py calib-rep`, targets at the pooled means, no tolerances, wide prior). Spread
over the fits (95 % interval of a standard deviation from 16 samples) against the predictions:

| Quantity | spread over independent runs | predicted (rms) | bootstrap |
|---|---|---|---|
| eps (single replica) | 2.78 (2.05-4.30) | jackknife 2.76 | |
| Hvap (kcal/mol) | 0.0096 (0.0071-0.0149) | jackknife 0.0101 | |
| liquid dipole (D) | 0.0008 (0.0006-0.0012) | jackknife 0.0008 | |
| ln s_q fitted to eps (1 parameter) | 0.0076 (0.0056-0.0118) | 0.0074 | 0.0082 |
| ln s_q, ln s_pol fitted to eps + Hvap | 0.19 (0.14-0.30), 0.89 (0.66-1.38) | 0.26, 1.18 | 0.37, 1.72 |
| ln s_q, ln s_eps(LJ) fitted to eps + Hvap | 0.012 (0.009-0.019), 0.72 (0.53-1.11) | 0.020, 1.15 | 0.023, 1.31 |
| same, fits of 2 replicas (8 fits) | 0.0033 (0.0022-0.0068), 0.17 (0.11-0.35) | 0.0080, 0.44 | 0.012, 0.64 |

The jackknife errors of the observables are accurate. The parameter errors are right for a
well-conditioned fit and conservative (up to ~2x) when a Jacobian element is small and noisy
(d eps/d ln s_eps(LJ) = 3 +- 12 per replica makes some replicas' J nearly singular). With blocks shorter
than the dipole's slow tail (300 ps replicas, 10 blocks of 30 ps) the eps jackknife error was 1.6x too
small; halving the number of blocks and keeping the larger variance (the default, `conservative`)
guards against that.

**(b) Recovery test** (512 waters, NPT, 2 ns per iteration): the base water perturbed to
(ln s_q, ln s_pol, ln s_R, ln s_eps) = (0.02, -0.05, 0.01, 0.10) and fitted back to the base model's
own values (density 0.97633, Hvap 6.81193, eps 73.26 from a 2 ns reference run, tolerances = the
reference's statistical errors; gas dipole 1.85807 D, polarizability 1.98364 A^3, tolerance 0.001),
prior width 1 (negligible), trust radius 0.1:

| iter | ln s_q | ln s_pol | ln s_R | ln s_eps | density | Hvap | eps | gas dipole | gas pol | liquid dipole | chi2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.0200 | -0.0500 | +0.0100 | +0.1000 | 0.9737(11) | 7.503(3) | 85.0(16) | 1.9875 | 1.8993 | 2.673 | 50717.6 |
| 1 | +0.0009 | +0.0005 | -0.0090 | +0.1820 | 0.9750(10) | 6.828(3) | 73.0(12) | 1.8603 | 1.9844 | 2.538 | 23.3 |
| 2 | -0.0001 | -0.0000 | -0.0084 | +0.1567 | 0.9774(9) | 6.820(3) | 75.0(17) | 1.8578 | 1.9836 | 2.535 | 5.1 |
| 3 | +0.0000 | +0.0000 | +0.0020 | -0.0408 | 0.9766(14) | 6.811(3) | 74.2(22) | 1.8582 | 1.9837 | 2.538 | 0.2 |
| fitted (next theta) | +0.0001 +- 0.0004 | +0.0000 +- 0.0006 | +0.0028 +- 0.0139 | -0.057 +- 0.258 | 0.9767 (pred.) | 6.8115 | 74.0 | 1.8582 | 1.9837 | 2.538 +- 0.004 | |

The trust radius (0.1, then 0.2) limited the first step; after it every prediction was confirmed by the
next run (z <= 1.9 for the liquid observables, trust ratios 1.00, 0.79, 1.00). The fitted parameters
recover the truth (0, 0, 0, 0) within the posterior errors (z = 0.15, 0.04, 0.20, -0.22; Mahalanobis
distance 0.9 for 4 parameters). The gas-phase targets fix q and alpha to 1e-4; R* and the LJ well depth
are nearly degenerate for density + Hvap + eps (correlation -0.99 of their errors): the fit moves
along that valley (iterations 1-2 at ln s_eps = +0.18, +0.16, within 1.7 sigma) and C_theta says so.
Propagated to the unfitted liquid dipole: +- 0.004 D (measured 2.535-2.538 at iterations 1-3, true 2.537).
Before the trust-region damping used the prior metric (Levenberg) instead of Marquardt's diag(J^T J), a
first attempt with q and cov both free (nearly degenerate: both set the molecular dipole) stepped into
the flat LJ direction; that run was stopped and the damping fixed.

**(d) Demonstration: the base pGM water toward experiment** (preliminary; 512 waters, NPT, 2 ns per
iteration; six global scale factors q, cov, alpha, radius, R*, eps; prior width 0.3 on each ln s,
i.e. regularised toward the base model; targets density 0.997 +- 0.002, Hvap 10.52 +- 0.05 kcal/mol,
eps 78.4 +- 1.5, gas dipole 1.855 +- 0.01 D, gas polarizability 1.47 +- 0.01 A^3):

| iter | ln s_q | ln s_cov | ln s_pol | ln s_rad | ln s_R | ln s_eps | density | Hvap | eps | gas dipole | gas pol | liquid dipole | chi2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.9763(11) | 6.812(3) | 73.3(15) | 1.8581 | 1.9836 | 2.537 | 8204.8 |
| 1 | +0.0113 | -0.0184 | -0.1973 | -0.0481 | +0.0170 | +0.2191 | 1.0009(5) | 8.324(3) | 89.4(28) | 2.0027 | 1.6496 | 2.657 | 2478.2 |
| 2 | +0.0723 | -0.0497 | -0.2805 | -0.1961 | +0.0958 | -0.0041 | 0.9218(5) | 9.298(5) | 59.7(40) | 1.7771 | 1.4823 | 2.489 | 1987.7 |
| 3 | +0.0688 | -0.0452 | -0.2900 | -0.1819 | +0.0841 | -0.1526 | 0.9905(7) | 10.486(6) | 74.5(35) | 1.8409 | 1.4749 | 2.609 | 13.3 |
| 4 | +0.0415 | -0.1270 | -0.2926 | -0.1847 | +0.0760 | -0.0623 | 0.9966(6) | 10.491(5) | 80.7(48) | 1.8525 | 1.4706 | 2.623 | 0.7 |
| 5 | +0.0711 | -0.0215 | -0.2955 | -0.1755 | +0.0737 | -0.0415 | 0.9968(4) | 10.457(3) | 72.0(35) | 1.8539 | 1.4699 | 2.611 | 4.4 |

Scale factors at iteration 4: q x1.042, cov x0.881, alpha x0.746, radius x0.831, R* x1.079, LJ eps x0.940
(the Bayesian-optimisation pGM3P-25 scaled alpha by 0.771 and radii by 0.750). All five targets are met
within their tolerances (chi2 0.65 for five targets; eps 80.7 +- 4.8 at 2 ns), with the liquid
dipole at 2.62 D (base 2.54, pGM3P-25 2.12). The path: iteration 1 (radius 1) cut the chi2 by 3x;
iteration 2 overshot (density 0.92) after the trust radius had grown to 2, and its ratio 0.20 shrank
the radius; iterations 3-4 then converged. Along the way the linear predictions were poor for the
large early steps (eps predicted 113.6, measured 89.4) and within the errors for the small late ones
(iteration 4: density 0.9971 predicted, 0.9966 measured; eps 76.2 predicted, 80.7 +- 4.8 measured);
reweighting (n_eff 1-3 of 4000 frames for the early steps) only helped for the last ones.
Sampling errors of the parameters at iteration 4: ln s_q 0.029, ln s_cov 0.105, ln s_pol 0.0024,
ln s_rad 0.008, ln s_R 0.003, ln s_eps 0.056: charges and covalent dipoles are nearly degenerate
(both make the molecular dipole), and so are R* and eps; the polarizability, radius and R* scales are
well determined. Iteration 5 moved along the q-cov valley (q x1.074, cov x0.979) with the same quality
(chi2 4.4; density 0.9968, Hvap 10.457, eps 72.0 +- 3.5; every prediction within z 1.4): at 2 ns per
iteration the fit has reached the noise of eps (+- 3.5-5), and the chi2 ratio (-4.6) shrank the radius.
Each iteration is one GPU job of about 17 minutes.

A 10 ns run at the iteration-4 parameters (5 x 2 ns, 20000 frames): density 0.9963 +- 0.0003 g/cm^3,
Hvap 10.491 +- 0.002 kcal/mol, eps 81.5 +- 2.5 (eps_inf 1.748), gas dipole 1.852 D, polarizability
1.471 A^3, liquid dipole 2.623 D; not targeted: compressibility 5.0e-5 /bar (experiment 4.5e-5) and
thermal expansion 1e-5 +- 3e-5 /K (experiment 2.6e-4: this parameter set has its density maximum near
298 K; alpha_p is available as a target with its gradient). eps is within 2 sigma of 78.4; tightening it
needs iterations of ~10 ns (the Jacobian of eps at 10 ns: d eps/d ln s_q = 420 +- 240).


## Speed

One RTX PRO 6000 Blackwell, 512 waters, mixed precision, frames every 0.5 ps (250 steps of 2 fs):
the analysis of a frame (tight dipole re-solve, four adjoint solves, one jacrev of six outputs for
3-6 parameters, g(r)), vmapped in chunks of 8, takes 4-7 ms, 2-3 % of the MD time between frames
(2 ns of NPT MD + 4000 frames: 15-19 min, i.e. 150-190 ns/day). 64 waters in 16 batched replicas:
50 ps per replica (1 fs) in 37 s including the analysis. On CPUs (48 cores) a frame of 512 waters takes
0.87 s. The per-iteration estimates (jackknife over blocks, 200 bootstrap resamples, LM with exact
gas-phase terms) take seconds.

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
