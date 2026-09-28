# NOTES: free energies as fitting targets (feature 8, branch fegrad)

## Plan
- pgm_jax/md/fe_grad.py: ParamSpace (flat table), ParamGradients (dU_k/dP of the end-state
  Hamiltonians at every window's configuration, at re-solved dipoles, vmapped), gas_leg_gradient,
  gradient_estimate ("end" and "mbar" estimators, block jackknife errors, replicates kept),
  FEGradient (project / chain), FreeEnergyTarget (fitting API), combine (relative / transfer / logP),
  alchemical_map (P0 -> PA), scaled_params / scale_direction (charge, eps, rmin, alpha, radius).
- alchemy.py: FreeEnergyRun(param_grad=...) stores samples "dudp" (S, T, K, M) + meta
  (dudp_targets, dudp_names, params_flat); load_windows(chk) starts from another run's windows.
- scripts/solvation_free_energy.py: run --grad, --solute-scale, --start-from; analyze prints gradients.
- scripts/fe_gradient_check.py: FD (theta +- delta, independent runs) vs gradient (central + Simpson).
- tests/test_fe_grad.py.
- docs/fe_gradients.md, README bullets.

## Validation plan
1. FD with independent runs: (a) water in pGM water, solute charge scale, electrostatics stage only
   (the vdW stage does not depend on the solute's charges exactly), s = 0.9/1.0/1.1 or 0.95/1.05;
   (b) methanol (flexible, keep) LJ eps scale (all 19 windows), 3 points.
2. Exact: lone rigid solute (zero variance) = gradient of E_gas(0)-E_gas(1); harmonic toy (analytic,
   calibrated jackknife errors); estimator = derivative of reweighted FE on stored frames.
3. pytest.

## Log
- 04:10 smoke (water elec-only, 20 ps, from the alch clone checkpoint): works; partial hydration
  (gas - elec stage) -7.17 +- 0.30 kcal/mol; d/dln s_charge(solute) -17.9 +- 0.5 (MBAR) / -18.0 +- 0.6
  (end); environment gradients +-20 kcal/mol from 20 ps (extensive noise).
- GPU bench (bench --grad): grad sample 8 windows 20 ms, 19 windows 43 ms (u sample 15 / 50 ms);
  0.40 ms per window-step => 1.1 % overhead at 1 ps sampling.
- The GPUs are shared with the owner pmemd jobs that cycle every few s: runs/gpu_fast.sh = gpu_run.sh
  with a 2 s free-check and retries of jobs failing within 90 s. Lesson: do not kill a waiting
  gpu_run.sh with TERM (its trap removes the lock and continues); kill -9 then remove own lock.
- 04:20 chain started (runs/fg/chain.sh): A water charge elec-only s=1.0,0.9,1.1 (0.6 ns); B water
  rmin s=1.0,0.97,1.03 (0.4 ns, 19 windows); C1 methanol reference (0.4 ns). Then C2: methanol at
  the charge scale fitted to experiment (-5.11 kcal/mol) with the C1 gradient.
- 05:13 A done (runs/fg/wq090/100/110, checkA.log): G = -5.280/-6.912/-8.964 kcal/mol; dG/ds -15.43/-18.57/-21.75;
  FD outer -18.42 +- 0.35 vs centre -18.57 +- 0.08 (z 0.4), Simpson -18.57 (z 0.4); pairs z 0.85, -0.53.
- full suite (before the h change of one test): 221 passed in 36.6 min (runs/full1.log).
- tests_print.log: HF rel errors 1e-12..5e-10; reweighting identities 5e-8..1e-5; lone solute 3.7e-5
  relative (PME/image), error bars 1e-3 of 2e4; harmonic: end -0.6983 +- 0.0072, mbar -0.7040 +- 0.0062
  (exact -0.7), error bar / spread 0.98, 0.93; value 0.8075 +- 0.0054 (exact 0.8047), 0.97.
- B done (wr097/100/103, 0.4 ns x 19 windows): G = -5.717 +- 0.058 / -4.453 +- 0.084 / -3.542 +- 0.099;
  d/dln s_rmin = 39.7 / 37.1 / 33.0; outer FD 36.3 +- 1.9 vs g(1) 37.1 +- 0.5 (z -0.4). checkB pending (CPU queue).
- C1 me100 (flexible methanol keep, 0.4 ns): G = -2.741 +- 0.094 (production -2.65 +- 0.07); d/dln s_charge
  -10.145 +- 0.125. C2 (chain2.sh) at s = 1.2335 (Newton step to -5.11). D (chain3.sh): methanol rmin 0.95, 1.05.
- C2 meq1234 (charge 1.2335, one Newton step to -5.11): G = -5.321 +- 0.084, d/ds -13.02; pair FD -11.05 +- 0.54 vs
  trapezoid -11.58 +- 0.09 (z 0.98). examples/hydration_target.py prints the step from the npz.
- D (methanol rmin 0.95 / 1.05) queued (chain3.sh); GPU shared with iface / bias.
- full suite (final code): 221 passed in 44 min (runs/full2.log). D mer105 running from 09:54.
- D done (mer095/me100/mer105, checkD.log): outer FD 13.07 +- 1.14 vs g(1) 12.48 +- 0.34 (z 0.5), Simpson z 0.4.
- All GPU work finished 10:53 (coordinator: keep GPU use short; none left). Docs complete.
## Status: done. Not done: local (low-noise) estimator for solvent parameters; log P (no second solvent box);
  bonded (template) parameters of flexible solutes; volume response.
