# NOTES: external electric fields (branch efield)

## Plan
- [x] md/efield.py: ExternalField (E0 V/nm, omega, phase), units (VNM_TO_INTERNAL, EPS_FACTOR), finite_field_eps
- [x] forcefield.py: efield=(E, offset) in compute / energy / energy_fixed_mu / strain_derivative;
      E on the induction RHS (plain, fused, ls, energy(), differentiable custom_vjp incl. d/dE);
      forces q E, torques via dE/dd; charge-flux path (phi -= E.r); Result.dipole = M
- [x] integrate.py / flexible.py / mts.py / simulation.py: MDState.efield (amplitude, state variable),
      fshift (re-wrapped charged molecules), fdip (M); field at t_{n+1}; work of E(t) booked in heat;
      NPT + charged molecules refused; alchemy refused; set_field(); observables efield, field_energy, Mx..Mz
- [x] channels.ElecChannel(efield=...) gas phase
- [x] md/finite_field.py: FieldReplicas (batched +-E copies, MDReplicas engine), read_series, analyse
- [x] scripts/finite_field.py run / analyse; scripts/validate_efield.py gas / box1 / nve
- [x] tests/test_efield.py
- [x] validation runs: p25 / tip3p / base finite field (GPU gpu-2-1), NVE (CPU), OpenMM TIP3P, constant D
- [x] docs/efield.md, README bullets
- [x] optional constant D
- [x] full test suite (232 passed)

## Log
- densities for NVT: p25 1.010 (pgm_jax NPT, docs/dielectric.md), base 0.983 (epsp/base/jax logs), tip3p 0.986 (docs)
- CPU, 3 replicas of 512 p25 waters, 32 cores: 0.91 ns/day per replica (GPU needed for production)
- flexible-engine NVE test in the tiny box: drift dominated by LJ truncation at 0.6 nm (same with/without field);
  tests use vdw="none" for the drift checks
- validate_efield gas: dmu vs alpha_mol E 1e-14 rel, energy exact, FD forces 3e-5 (h 1e-5); box1: one water in
  L = 2..6 nm boxes: (alpha_box - alpha_gas)/alpha_gas * V = 0.006 nm^3 (image field ~1/V)
- NVE 512 p25 (mixed, 1 fs, 20 ps after 10 ps NVT): drift no field 0.0002, 0.1 V/nm 0.0022 (noise; econs rms 0.25),
  0.5 V/nm 0.0005 kT/ns/dof (T rises 293 -> 318 K: orientation releases 466 kJ/mol)
- E(t): first version booked -(E1-E0).M1: 0.5 V/nm at 200 cm^-1 heated to 600 K, econs drift 1.28 kT/ns/dof (2 % of
  the absorbed 10^4 kJ/mol; systematic dE.alpha.dE/2 per step). Trapezoid of dH/dt (extended-phase-space VV):
  0.2 V/nm, 1062 kJ/mol absorbed in 20 ps, econs drift -0.0012 kT/ns/dof, rms 0.29 kJ/mol
- GPU gpu-2-1: p25 10 replicas (+-0.02, 0.05, 0.1, 0.2, 2 x 0) 1 ns: 42.0 ns/day per replica, 420 aggregate (2 fs)
  quick awk means (50 ps skipped): eps(+-0.02) 35.6, (0.05) 34.7, (0.1) 33.0, (0.2) 33.3
- CPU (32 cores): TIP3P 2 replicas 2.9 ns/day each
- 1 ns x 10 replicas (50 ps skipped), pairs:
  p25   0.02 35.57+-4.99 | 0.05 34.70+-1.71 | 0.1 32.98+-0.60 | 0.2 33.34+-0.42 ; zero-field fluct 33.74+-1.93, 33.39+-1.09
  tip3p 0.02 106.65+-4.21 | 0.05 92.68+-3.00 | 0.1 87.47+-0.83 | 0.2 72.57+-0.20 ; fit<=0.1 100.35+-2.96; zero 97.1, 96.3
  base  0.02 71.69+-2.91 | 0.05 74.01+-1.02 | 0.1 70.47+-0.33 | 0.2 63.35+-0.21 ; fit<=0.2 73.08+-0.40 (chi2 2.8/2); zero 74.7, 72.3
- zero-field references (validate_efield fluct, skip 200 ps): tip3p.dip (dipoles agent, NPT 9.8 ns) 103.77+-2.98, 1-ns
  segments std 9.13; pgm3p25.dip 34.20+-0.70, std 2.19; base r1..r5 (5 x 9.8 ns) 73.23+-0.17, std 1.87 (tau_M ~1 ps)
- GPU speeds (2 fs, 10 replicas): p25 42 ns/day/replica (420 agg), tip3p 73 (729), base 42 (421)
- cluster note: sbatch without --mem takes the whole node's memory (125G): runs/cpu_mem.sh adds --mem
- OpenMM TIP3P check (scripts/openmm_tip3p_field.py, CPU 16 threads ~31 ns/day): +-0.05, 0.1, 0.2 x 1.5 ns submitted
- p25 after 2 ns: pairs 0.02 34.56+-2.98, 0.05 35.27+-1.14, 0.1 33.89+-0.47, 0.2 33.23+-0.24; fit(all) 34.39+-0.55
  (c 29+-16); zero copies 33.07+-0.48, 34.39+-0.81
- OpenMM TIP3P (1.45 ns/replica): +0.1 88.92+-0.98, -0.1 87.92+-1.08, +0.2 71.40+-0.44, -0.2 71.13+-0.17
  -> pairs 88.4+-0.7 (pgm_jax 87.47+-0.83), 71.3+-0.2 (pgm_jax 72.57+-0.20)
- full suite: 230 passed (35 min, 32 cores); zero-field identity with master: bitwise (efield_identical.py)
- GPU lock on gpu-2-1 is contended by the iel agent (their jobs ~70 min); remaining work bundled in runs/gpu_all.sh
  (tip3p +1 ns, p25 constant D +-3.3/6.6 0.5 ns, speed, tip3p +-0.2 at 1 fs)
- tip3p after 2 ns: pairs 0.02 101.1+-4.3, 0.05 93.7+-1.7, 0.1 88.1+-0.5, 0.2 72.2+-0.2; fit<=0.1 96.8+-2.0
  (c 871+-210); zero copies 99.1+-3.2, 100.3+-2.8. 1 fs +-0.2: 71.8+-0.3 (OpenMM 71.3+-0.2)
- p25 constant D (+-3.3, +-6.6; 0.45 ns): 34.6+-0.8, 32.4+-0.5; <E> 0.095, 0.205 V/nm; tau_M 0.3 ps
- GPU speed (validate_efield speed): none 0.755 ms/step, static 0.913, E(t) 0.916, D 0.951; alternating repeat
  0.806 vs 0.909; forces() 0.528 vs 0.664 ms. HLO (CPU): +8 fusions, +4 reduces -> after merging the M sums and
  computing M once: +4, +2. New GPU measurement queued (runs/gpu_chain9.sh)
- after merging the M sums: GPU alternating none/static 0.803/0.808 ms/step (+0.6 %), forces() 0.557/0.578 ms;
  sequential validate_efield speed: 0.769 / 0.812 / 0.820 (E(t)) / 0.878 (D); FieldReplicas x4 457 ns/day aggregate
- OpenMM TIP3P +-0.05 (1.95 ns): 95.9+-1.7 (pgm_jax 93.7+-1.7)
- full suite after all changes: 232 passed (32 min, 32 cores)
- DONE. Not done: absorption spectra from E(t) vs the IR spectrum; NPT at constant D not run; D-mode CG sum in float64
  (+14 % per step) could be made cheaper
