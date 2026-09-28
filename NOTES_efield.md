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
- [ ] validation runs: p25 / tip3p / base finite field (GPU gpu-2-1), NVE (CPU)
- [ ] docs/efield.md, README bullets
- [ ] optional constant D
- [ ] full test suite

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
