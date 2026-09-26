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

## 3b. Energy conservation under thermostats (effective energy drift)

With a thermostat, E_tot is not conserved. The right measure is the effective energy
H~ = U + K (+ K_aux) - Q, where Q is the heat exchanged in the thermostat steps (their kinetic
energy change, constraint projections included). Exact dynamics conserve H~; its drift is
integration error plus the non-conservative work of incompletely converged dipoles.
`runs/langevin/drift.py`, 4096 waters, 100 ps (tol 1e-5, mixed) or 50 ps (tol 1e-8, double);
slopes in kT/ns/dof, linear fit (mean of 5 segment slopes +- s.e.). "rms at lag" = rms change of
detrended H~ over 1 and 20 ps: it stays flat for a bounded fluctuation and grows for a random walk.

| thermostat | 1 fs, tol 1e-5 | 2 fs, tol 1e-5 | 2 fs, exact SCF (double, 1e-8) | 2 fs rms at lag 1 / 20 ps (kJ/mol) |
|---|---|---|---|---|
| NVE | +0.0010 | +0.0035 (+0.0034 +- 0.0002) | -0.0001 | 2.2 / 2.3 (bounded) |
| Bussi, tau 1 ps | +0.0009 | +0.0033 (+0.0028 +- 0.0004) | -0.0004 | 2.7 / 2.5 (bounded) |
| Langevin 1/ps | +0.0004 | +0.0068 (+0.0054 +- 0.0016) | -0.0001 (+0.003 +- 0.007) | 2.8 / 4.0 (grows) |
| Langevin 5/ps | +0.0006 | +0.0116 (+0.0089 +- 0.0041) | +0.0059 (+0.017 +- 0.003) | 3.8 / 10.0 (grows) |
| band-pass GLE | +0.0015 | +0.0072 (+0.0035 +- 0.0012) | +0.0072 | 3.3 / 5.2 |
| sparse Langevin 5/ps every 50 steps | +0.0005 | +0.0177 (+0.0187 +- 0.0016) | | 3.1 / 7.0 (grows) |

- NVE and Bussi drift only through the dipole solve: the drift vanishes with an exact solve.
- Per-atom stochastic thermostats add two things at 2 fs:
  - a systematic drift that grows with the friction and persists with an exact solve. This is
    shadow work of the discretized Langevin dynamics (Sivak, Chodera & Crooks 2013, PRX 3,
    011007), not specific to pGM.
  - a random walk of H~.
- At 1 fs all thermostats are within noise of each other.
- The band-pass GLE does not reduce the drift. It couples at 30-100 rad/ps (librations), where
  the O(h^2 w^2) shadow-energy error lives. A thermostat design objective should therefore also
  penalize coupling to fast modes, roughly integral K(w) w^2 g(w) dw, which competes with fast
  thermalization.
- pmemd-pgm (250 ps runs, `runs/langevin/drift/`) does not report the thermostat heat, so its
  NVT E_tot cannot show integration drift. E_tot fluctuates canonically, about 75-105 kcal/mol
  rms for ntt=3, ntt=11 and the middle scheme. Berendsen suppresses it to 12-16 kcal/mol, which
  makes Langevin traces look "driftier" than Berendsen.
  - <T> and <EPtot> agree across thermostats within about 2 s.e.
  - NVE drift at tol 1e-4 is -0.005 to +0.0015 kT/ns/dof.
  - Measuring H~ in pmemd needs a patch that accumulates the heat of the Langevin update.

## 3c. After equilibration: E_tot is stationary, H~ still drifts

A thermostatted run must have a stationary E_tot once equilibrated. The heat produced by
integration and SCF errors is removed by the thermostat. For gamma = 1/ps and an NVE heating rate
of 0.0035 kT/ns/dof, the steady-state temperature shift is about r/(2 gamma), i.e. 2e-6 relative,
invisible. A persistent E_tot trend after equilibration therefore means the run is not
equilibrated, or there is a bug. The H~ drift of section 3b is a different quantity: the
bookkeeping of the heat the thermostat silently absorbs.

- **pmemd-pgm**, 500 ps Bussi equilibration, then 1 ns per thermostat at 2 fs
  (`runs/langevin/drift/parse_prod.py`). E_tot trends in kcal/mol/ns, from 10 block means:
  - ntt=3, gamma 1: -7.6 +- 19.6 (shipped solver), +20.9 +- 14.6 (mu4 fused)
  - ntt=3, gamma 5: -2.9 +- 8.4 (shipped), +8.3 +- 7.8 (mu4)
  - ntt=11 (Bussi): -67 +- 43 (shipped, partial parse), -5 +- 35 (mu4, before the failure below)
  All are stationary. The trends in the earlier 250 ps runs were relaxation from the starting
  structure.
  - Failure to follow up in pmemd-pgm: the experimental mu4 + fused path with ntt=11 went to NaN
    abruptly at step 277,500 (T about 299 K just before; no gradual heating). The shipped solver
    with ntt=11 completed. Earlier 250 ps runs of the same combination were fine, so this is a
    rare event.
- **pgm_jax**, 100 ps Langevin equilibration, then 200 ps per thermostat at 2 fs. E_tot block
  means are flat for Langevin within statistics; Bussi (tau 1 ps) was still relaxing. H~ drift
  (kT/ns/dof):
  - NVE +0.0034; Bussi +0.0034
  - Langevin 1/ps +0.0051 (segments +0.0042 +- 0.0011); Langevin 5/ps +0.0125 (+0.0118 +- 0.0023)
  - band-pass GLE +0.0073
  These confirm section 3b.

## 3d. Theory: can the memory kernel carry polarization information?

Extended state z = (x, y), y = (p~, s) mass-scaled momenta and auxiliaries, O-part
dy = -A dt y + B dW.

1. **Pointwise FDT theorem.** If A = A(x) and B = B(x) depend on the configuration only, and
   B(x) B(x)^T = kT (A(x) + A(x)^T) at every x, then rho ~ exp(-beta [U(x) + |y|^2/2]) is
   invariant. The Hamiltonian Liouvillian annihilates any function of the energy. The O-part acts
   at fixed x, and N(0, kT I) solves its Lyapunov equation. Polarization information may
   therefore enter through anything that is a function of the current configuration:
   - alpha_i, the permanent field E(x), the dipole tensor T(x);
   - the converged BO dipoles mu*(x);
   - the response Jacobian J(x) = d mu*/dx, or any approximation to it.
   The approximation quality affects only efficiency, never the ensemble. The friction must be
   smooth in x, or the splitting error and the shadow work grow.
2. **Not allowed: dependence on velocities or on solver history.**
   - dmu/dt = J v, the predictor error (a 4th difference along the trajectory), iteration counts,
     residuals and past dipoles all carry momentum information.
   - An O-step whose coefficients depend on p no longer preserves the Maxwellian: it needs the
     Ito correction kT div_p (B B^T / 2kT) and more.
   - A history-dependent rule makes the process non-Markovian in z, so the invariance argument
     fails.
   - "Adapt the friction to how badly the SCF is doing" is therefore wrong as a feedback rule.
     The consistent way is to promote the feedback variable to a state variable with its own
     FDT-consistent equation (as adaptive Langevin does for the friction). The ensemble is then
     exact by construction.
3. **Not allowed: thermalizing the polarization.** If the induced dipoles become dynamical bath
   variables at temperature T (extended Lagrangian or Drude, or a Markovian kernel whose
   auxiliaries are the dipoles), the nuclei see the free energy
       F(x) = -1/2 E^T A^-1 E + (kT/2) ln det A(x) + const,   A = alpha^-1 - T(x).
   The first term is the BO polarization energy. The second is a temperature-dependent,
   many-body term from thermal dipole fluctuations: a classical "Drude dispersion" that the LJ
   term already counts. The ensemble is exact only for cold auxiliaries (T* -> 0, the Drude dual
   thermostat; stochastic XLMD with T ~ eps^1/2) or for SCF.
   - A Fixman-like counter-potential -(kT/2) ln det A(x) would restore it. Its gradient,
     tr(A^-1 dA), needs stochastic trace estimates, i.e. several extra solves per step.
   - With finite-mass auxiliaries, integrating them out also puts a polarization memory into the
     nuclear equation of motion, K(t) ~ J^T cos(Omega t) J. The dynamics is BO only when
     Omega >> the nuclear frequencies.
4. **Physics of the kernel.** In Mori-Zwanzig terms, projecting out the electrons of a
   ground-state insulator gives an instantaneous (adiabatic) response, not a friction. Electronic
   friction is nonadiabatic and exponentially small in the gap. A "polarization kernel" therefore
   has no physical content: the thermostat is a sampling device. Its only physical requirement is
   the ensemble (item 1). For dynamics, the requirement is to perturb the observables of interest
   as little as possible.
5. **Legitimate polarization-aware designs.**
   - A configuration-dependent friction tensor that keeps noise out of the directions that move
     the induced dipoles: Gamma(x) = gamma (I + c J~^T J~)^-1, where J~ is, for example, the local
     direct-response Jacobian alpha dE_perm/dx. This is canonical by item 1. It needs a sparse
     linear solve per step for the noise, so it is expensive for what it buys.
   - The same weighting inside the smooth GLE couplings a(x).
   - Long-wavelength-only noise is a cheap proxy: long-wavelength kicks barely change local fields.
   The SCF goal is already met by temporal smoothness (noise through auxiliaries) or by global
   rescaling, without any polarization information.
6. **Design as an optimization over K(w)** (Markovian embedding, a_pp = 0):
   - canonical: automatic, from pointwise FDT;
   - SCF: small noise power near pi/h, and smoothness order k;
   - energy bookkeeping (shadow work): integral K(w) w^2 h^2 g(w) dw small;
   - thermalization: integral K(w) g(w) dw >= target;
   - dynamics: K small at w -> 0 and in the band of the observables.
   Fast modes (librations) equilibrate through anharmonic coupling, so a thermostat confined to
   the slow band (about 5-50 rad/ps) serves all goals at once. The first band-pass (w0 50,
   width 100 rad/ps) couples too strongly at 100 rad/ps, which is why its H~ drift is as large
   as Langevin's.
   Test of the principle, slow-band GLE (a 9.5, g 30, w0 20 /ps). Its K(w) at
   0/10/30/100/300/1000 rad/ps is 0/1.50/2.30/0.27/0.03/0.003 per ps.
   - H~ drift +0.0052 +- 0.0004 kT/ns/dof at 2 fs: fast band-pass 0.0073, Langevin 1/ps 0.0051,
     NVE 0.0034.
   - CG 4.00 at 1 fs (Langevin 6.00); cubic predictor error 9.0e-5 (NVE 7.9e-5).
   - Cold start 146 K -> 277 K after 1 ps, the same as Langevin 1/ps (276 K).
   - D 3.95 (-7.5 %; Langevin 1/ps -17 %).
   Relative to Langevin 1/ps it gives the same thermalization and H~ drift, 2 fewer CG
   iterations, and half the diffusion perturbation.

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
| Shadow work / effective-energy bookkeeping for discrete Langevin | Done: Sivak, Chodera & Crooks 2013 (PRX); Bussi et al. 2007 use the effective energy to monitor integration error. |
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
