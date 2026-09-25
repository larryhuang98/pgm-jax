# Thermostats for polarizable MD: why Langevin hurts the dipole predictor, and what to use instead

Status: research notes with prototype measurements (2026-09-25). The prototypes live in
`runs/langevin/` on the cluster (`pred.py`, `gle.py`); nothing here is in the MD engine yet.
All numbers: pGM water, 4096 molecules (12k atoms), constraints engine, tol 1e-5, 1 fs unless noted.

## 1. The effect

The CG start for the induced dipoles is a polynomial extrapolation of past solutions (cubic
"mu4" by default). Its error is the (K+1)-th finite difference of mu along the trajectory.

| thermostat | cubic predictor error (rel. rms) | CG iterations, pgm_jax | CG, pmemd-pgm mu4 | CG, pmemd delta4 |
|---|---|---|---|---|
| none (NVE) | 8.3e-5 | 4.0 | 4.0 | 3.0 |
| Langevin 0.1/ps | 4.0e-4 | 5.0 | 5.6 | 5.0 |
| Langevin 1/ps | 1.3e-3 | 6.0 | 6.1 | 6.0 |
| Langevin 5/ps | 2.8e-3 | 6.0 | 7.0 | 7.0 |
| Bussi (global), tau 0.1-1 ps | 8.4e-5 | 4.0 | 4.0 (ntt=11) | 3.0 (ntt=11) |

## 2. Theory

**Predictor error under Langevin.** With the noise acting directly on the momenta, the
positions are only C^{1,1/2} in time. The noise part of the (K+1)-th difference of x is

    h * sigma_v * sqrt(C(2K-2, K-1)),   sigma_v = sqrt(kT/m (1 - exp(-2 gamma h))) ~ sqrt(2 gamma h kT/m)

so it scales as sqrt(gamma) h^{3/2}. It grows with the order (sqrt 6 for cubic). NVE gives
O(h^{K+1}) instead. Measured: the error scales exactly as sqrt(gamma), and quadratic beats cubic
at gamma = 5. Field-anchored predictors (mu = alpha E_perm(new x) + extrapolated Delta) cut the
noise part about 3x but cannot remove it. The fresh kick changes mu through mutual induction,
and no history-based guess can see that without another field sweep. The pmemd-pgm CPU
least-squares predictor (`ls`) is the best one in NVE (3.5 iterations) but gets 6.1 under
Langevin 1/ps.

**Smooth GLE family.** Mass-scaled momentum p, auxiliary momenta s_1..s_k, y = (p, s),
dy = -A y dt + B dW. The distribution exp(-beta [U + p^2/2 + |s|^2/2]) is invariant iff
B B^T = kT (A + A^T). If the noise must not act on p directly (B_p = 0), this forces A_pp = 0
and antisymmetric couplings. The admissible class is therefore conservative couplings, with
friction and noise only on the auxiliaries.

- Chain (noise on the last link): the random force reaching p has been integrated k times, so
  x is in C^{k+1,1/2}, and the noise part of the predictor error falls to h^{k+3/2}. The memory
  kernel is Mori's continued fraction, K(z) = a1^2 / (z + a2^2 / (z + ... + ak^2 / (z + g))).
  - k = 1: K(w) = a^2 g / (g^2 + w^2), a low-pass kernel with zero-frequency friction a^2/g.
  - Fresh noise per step in p, relative to Langevin with the same zero-frequency friction, is
    g h / sqrt(3). Measured: 5.75e-3 for g = 10/ps (theory 5.77e-3).
- Band-pass variant (derived here). Use (p, s, q) with p-s coupling a, s-q coupling w0, and
  friction plus noise on the middle variable s. This gives
  - K(z) = a^2 z / (z^2 + g z + w0^2);
  - K(0) = 0, so diffusion is unperturbed to leading order;
  - a peak a^2/g at w0, and an w^-2 tail;
  - the noise is still only on an auxiliary, so the trajectory stays C^{2,1/2}.
- Thermostat efficiency is roughly integral g(w) K(w) dw over the velocity density of states
  g(w). The design is a filter problem: strong K where the kinetic energy lives, weak K near 1/h
  (predictor), and K(0) small if transport properties matter.

## 3. Schemes tested (1 fs; D from 60 ps at 2 fs; cold start from 146 K)

| scheme | CG | cubic error | T after 0.5 / 1 ps | D (1e-5 cm^2/s) |
|---|---|---|---|---|
| NVE | 4.00 | 7.9e-5 | 250 / 248 (no thermostat) | 4.27 |
| Langevin 1/ps | 6.00 | 1.3e-3 | 260 / 276 | 3.56 (-17 %) |
| Langevin 5/ps | 6.01 | 2.8e-3 | | 2.36 (-45 %) |
| global Bussi, tau 0.1 ps | 4.00 | 8.4e-5 | | (known to leave D almost unchanged) |
| Bussi per group of 64 molecules, tau 0.1 | 4.54 | 1.6e-4 | 289 / 296 | |
| Bussi per molecule, tau 0.1 | 5.98 | 1.1e-3 | 292 / 292 | |
| sparse Langevin, 5/ps every 50 steps | 4.21 | 7.4e-4 | 296 / 295 | |
| low-pass GLE k=1 (a^2/g = 5/ps, g = 50/ps) | 4.00 | 1.2e-4 | 275 / 287 | 2.33 (-45 %) |
| band-pass GLE (a 22.4, g 100, w0 50 /ps) | 4.22 | 1.8e-4 | 282 / 288 | 4.00 (-6 %) |

- The band-pass GLE thermalizes faster than Langevin 1/ps and perturbs diffusion about 3x less.
  It needs two fewer CG iterations per step at 1 fs. Its K(w) at 0/10/30/100/300/1000 rad/ps is
  0/0.74/3.9/3.2/0.53/0.05 per ps.
- Kinetic temperatures in all runs, Langevin included, sit about 1.5-3 K below 298 K. Standard
  deviations are 2.5 K against a canonical 2.7 K. This is the same for Langevin, so it is the
  integrator's discretization, not the thermostat. Still to check with the configurational
  temperature and longer runs.

## 4. Prior art (literature search 2026-09-25; "not found" means not found, not proven new)

| idea | status |
|---|---|
| Thermostat noise degrading SCF extrapolation / raising SCF iterations | Not found. Martinez et al. 2015 (XL-BOMD) asked, and found Langevin at 1/ps "fully compatible" with fixed SCF cycles. They did not measure predictor error. |
| Scaling law of section 2 (discretized noise vs extrapolation order) | Not found. |
| Noise only on auxiliaries, antisymmetric couplings | Done as a class: Leimkuhler & Sachs 2022, Ottobre & Pavliotis 2011. GLE framework: Ceriotti, Bussi & Parrinello 2009/2010. |
| Smoothness as a design goal | Done in MCMC: third-order Langevin (Mou et al. 2021); K-th order Langevin with Lagrange interpolation (Mahajan et al. 2025). Ceriotti 2009 used low-pass noise so shells / CP electrons are not heated. Not found for SCF extrapolation in MD. |
| Band-pass kernel with K(0) = 0 and no direct noise on p | Not found. Morrone et al. 2011 is high-pass with a white-noise floor; Rossi et al. 2018 builds mode-selective thermostats. |
| Local (grouped) Bussi | Done: suggested by Bussi & Parrinello 2008; GROMACS tc-grps; CP2K CSVR regions. |
| Sparse Andersen / Langevin | Done: Andersen 1980; E & Li 2008; GROMACS nsttcouple. A kick-aware predictor was not found. |
| Field-anchored predictors | Done in AIMD: Arias, Payne & Joannopoulos 1992; Alfe 1999. LS dipole predictor: Wang & Skeel 2005. Robustness to thermostat noise not found. |
| Cold thermostats on auxiliary dipoles | Done: Lamoureux & Roux 2003 (Drude); Albaugh et al. 2015 (iEL); An et al. 2021 and Tan et al. 2020 (stochastic XLMD). |
| Noise projected away from dipole-sensitive directions; long-wavelength-only Langevin | Not found (quick search). A band-subspace Langevin in normal-mode coordinates exists (FIMD, 2026). |

## 5. Open questions and next steps

1. Optimal k=1/k=2 designs under a smoothness constraint (maximize integral g K subject to
   K(1/h) <= eps and K(0) <= K0), and whether a closed form exists.
2. Canonical checks: kinetic-energy histogram, configurational temperature, h -> 0 against
   Langevin. Dynamics: D, orientational relaxation, VACF against NVE.
3. Proteins: at 2 fs with HMR the time-step error dominates (ubiquitin 13 -> 12 iterations with
   Bussi). The payoff is larger at 1 fs, for MTS inner steps, and once the predictor improves.
4. Implementation: exact (k+1)x(k+1) O-step per degree of freedom (auxiliaries projected with
   RATTLE) in pgm_jax, then a per-atom kernel in pmemd-pgm.
