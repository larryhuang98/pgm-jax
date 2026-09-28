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
