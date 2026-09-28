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
