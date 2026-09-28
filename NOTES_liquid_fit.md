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
