# NOTES: eps gradient + multi-target liquid/gas fitting + UQ (branch fit2)

## Plan
- pgm_jax/fit/: params.py (ParameterSpace), frames.py (FrameAnalyzer: per-frame U, M, alpha_cell, D, V, g(r)
  and explicit derivatives, adjoint of the induction solve), estimators.py (LiquidSamples: fluctuation
  formulas via reweighted averages, jackknife, bootstrap; GasPhase), optimize.py (Target, Objective,
  LM trust region, covariance), liquid.py (LiquidFit driver, JSON records, resume).
- scripts/fit_multi.py (fit / measure), scripts/liquid_fit_tools.py (combine, fd, calib).
- tests/test_liquid_fit.py (10 tests, ~1 min CPU).

## Status log
- 01:40 start. CPU MD 512 base waters: 4.7 ns/day (48 cores, mixed). GPU ~220 ns/day but all five GPUs
  run the owner's pmemd back to back; gpu_run.sh waits.
- 02:20 tests pass (10). Probe on base water CPU: analysis 0.87 s/frame (48 cores).
  base water: rho 0.988 (4 ps), Hvap 6.85 kcal/mol, liquid dipole 2.54 D, gas dipole 1.858 D, gas pol 1.984 A^3.
  p25: rho 1.016, Hvap 9.49, liquid dipole 2.12 D, gas dipole 1.462 D, gas pol 1.488 A^3 (10 ps, CPU).
- The 512 boxes are truncated octahedra (not cubic).  Small box for CPU validations: 216 waters cut in
  fractional coordinates (runs/fit/smallbox.py), cutoff 0.7 nm (elec_cutoff_settings).

## To do
- (a) FD of d eps / d ln s_q: runs at -d, 0, +d (independent), compare with gradients.
- (b) recovery test; (c) calibration over seeds; (d) demo from base water toward experiment.
- docs/liquid_fit.md, README bullets; full test suite.
- 03:30-04:00 all GPUs busy with the owner's pmemd (13 s / 130 s jobs back to back, gaps < 1 s).  CPU queue
  memory-bound (others allocate whole-node memory); my jobs use --mem.  CPU: 64-water box, 16 batched NVT
  replicas (vmap): 38 ns/day aggregate on 16 cores (cutoff 0.45, beta 6, PME 20^3); 125 waters: 16 ns/day.
- 04:03 GPU obtained (gpu-2-2).  demo512: GPU analysis 7 ms/frame, 2 ns + 4000 frames in ~15 min.
  iter 0 (base): rho 0.97633(114), Hvap 6.81193(303), eps 73.26(147), gas mu 1.85807, gas pol 1.98364,
  liquid dipole 2.53745(29), eps_inf 2.0056.  iter 1: rho 1.0009, Hvap 8.32, eps 89.4, gas mu 2.003, pol 1.650.
  Linear predictions off for large steps (eps predicted 113.6, got 89.4): nonlinearity.
- 04:45 step(): gas-phase observables kept exact (nonlinear GN iterations inside the trust region);
  radius_max 2.  Queue: runs/gpu_queue.sh (FD 512: q -0.08/0/+0.08, 4 ns each, then demo continues),
  runs/gpu_queue2.sh (recovery rec512: start (0.03,-0.10,0.06,0.15) in (q,cov,alpha,lj_eps), targets = base values).
- CPU: v64_{m,c,p}: 16 replicas x 64 waters NVT, q -0.08/0/+0.08 (params q,alpha,lj_eps), 12 x 50 ps segments;
  demo216 (NPT 216 waters, cutoff 0.7) queued.
- 06:00-07:10 results:
  * demo512 iter 3 (q 1.071, cov 0.956, pol 0.748, rad 0.834, R* 1.088, eps 0.859): rho 0.9905, Hvap 10.486,
    eps 74.5(35), gas mu 1.841, gas pol 1.475, liquid mu 2.609 D; chi2 13.3.
  * FD 512 NPT (4 ns each): s_q -0.08: rho 0.761 eps 33.9; -0.03: rho 0.912 eps 57.0(7); +0.03: rho 1.031 eps 92.9(10).
    d eps/d ln s_q gradient: 381(81) at -0.03, 603(217) at +0.03 (noisy: relative error grows ~sqrt(N) at fixed time).
  * 64 waters, 16 NVT replicas, q -/+0.08: at 2 fs (CPU) the fluctuation terms of energy, dipole, eps_inf are ~4 %
    smaller than FD (z ~ 3); at 1 fs (GPU) they agree (z 0.1-0.9): integrator bias at 2 fs.  eps: FD 427(8) vs
    Simpson 393(17), z 1.8 (8 segments).
  * tau_M ~ 1 ps for base water (fast), but a small slow tail: jackknife with 10 blocks of 30 ps underestimates the
    eps error of single 300 ps replicas (3.1 vs spread 5.0) -> conservative jackknife (max with half the blocks).
  * gpu_run.sh lock race: several of my waiters -> double starts (rc=1).  Now one master queue (runs/gpu_master.sh).
  * Local analysis in the Cowork VM (jax 0.6 CPU) of frames copied from the cluster (CPU queue saturated).
- 07:50 correction: with more segments (7/12/9) the 2 fs CPU FD (64 waters) agrees too (energy z -0.5, hvap +0.6,
  eps +0.7, eps_inf -0.4, dipole +0.7; --skip 1: all |z| < 0.9).  The early z ~ 3 came from 2-4 segments
  (equilibration transient after the 40 ps start / chance); no evidence of a 2 fs bias.
- gpu_master2.sh: fd512_c -> rec512 -> one demo job -> w64 to 24 segments -> demo; then gpu_queue6.sh (x64, 2 fs GPU).
- 08:10 FD 512 (m3/c/p3, 4 ns each): all |z| <= 2.0 (full), <= 1.2 dropping the first 2 ns.  Calibration (64-water
  replicas, 600 ps each): jackknife errors of eps/Hvap/dipole = replica spread; parameter errors right for 1-param and
  well-conditioned fits, conservative (<= 2x) when J has a small noisy element.
- 08:40 first recovery (q,cov,alpha,lj_eps from (0.03,-0.10,0.06,0.15)) went off: q/cov nearly degenerate and the
  Marquardt diag(A) damping pushed the trust-region step into lj_eps.  Fixed: damping in the prior metric (exact TR
  subproblem).  Killed that run; rec512b: (q, alpha, R*, eps) from (0.02,-0.05,0.01,0.10).
- rec512b iter 1: all targets within ~1-4 sigma; next theta (-0.0001, 0.0000, -0.0084, 0.157), posterior sd
  (0.0003, 0.0006, 0.0050, 0.097), corr(R*, eps) -0.996; Mahalanobis 8.5 (chi2_4 95 % = 9.5).
- full suite (cluster CPU, 07:05 code): 221 passed in 41 min.
- 10:05 rec512b done (4 iterations x 2 ns): chi2 50718 -> 23 -> 5.1 -> 0.2; fitted (0.0001, 0.0000, 0.0028, -0.057)
  +- (0.0004, 0.0006, 0.014, 0.26) posterior; Mahalanobis 0.9.  RDF FD (512): 56 bins, rms z 1.01.
- 10:30 demo512 iter 4: chi2 0.65 (rho 0.9966, Hvap 10.491, eps 80.7(48), gas mu 1.8525, pol 1.4706, liquid mu 2.623);
  iter 5 (along the q-cov valley): chi2 4.4, eps 72.0(35).  demo512_final: 10 ns at iteration-4 parameters
  (runs/gpu_master4.sh), for eps to +-1.5, alpha_p, kappa_t, g(r).
- Final tests: tests/test_liquid_fit.py 13 passed (cluster CPU, final code); full suite 221 passed (07:05 code).

## What is left / ideas
- Longer runs (or several replicas) per iteration once the fit is near the targets: eps +-1.5 needs ~10 ns of
  512 waters; the Jacobian of eps is the noisiest piece (small boxes or FD along the step direction help).
- Jacobians from small-box replicas (cheap, precise) with values from the large box; NPT batched replicas.
- Step rejection in the trust region (currently every step is taken); per-key fits (alpha@OW etc.) not run.
- Flexible molecules (bonded + intramolecular LJ in U, gas-phase ensemble), charge flux, virtual sites.
- 14:20 demo512_final (10 ns at iteration-4 theta): rho 0.9963(3), Hvap 10.491(2), eps 81.5(25), eps_inf 1.748,
  liquid mu 2.623 D, gas mu 1.852, pol 1.471; kappa_t 5.0e-5 /bar, alpha_p 1e-5(3e-5) /K (exp 2.6e-4).
- 14:50 full suite with the final code: 221 passed (31.8 min, cpu-short 32 cores).
