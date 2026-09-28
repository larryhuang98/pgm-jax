# NOTES: path-integral MD (branch pimd)

## Plan
- pgm_jax/md/pimd.py: RingPolymer (normal modes, exact/Cayley free RP step), PILE-L/PILE-G, TRPMD, RPMD,
  PIMDIntegrator (BAOAB: B A O A B, forces once per step), estimators (primitive, centroid virial, per
  element), PotentialEngine (analytic V for tests), PGMBeads (pGM force field of a FlexibleSimulation
  vmapped over beads; per-bead induction state, shared predictor counter; one neighbour list of the
  centroid with a bead margin), ring-polymer contraction (monomer gas-phase model on all beads,
  intermolecular remainder on P' beads), PIMDSimulation driver (blocks, overflow handling, log,
  centroid/bead trajectories, checkpoints, molecular centroid-virial pressure).
- Flexible pGM water: rayl_512_v2 pGM water + bonded terms fitted so that the gas-phase monomer PES
  = q-TIP4P/F intramolecular PES (pimd.flexible_water). New bonded family bond_quartic.
- Rigid bodies / constraints: not supported (documented).  NVT only.

## Decisions
- H_P convention: physical masses, beads at P T, omega_P = P kT / hbar.  kT_P used by the thermostat.
- Primitive estimator: 3PkT/2 - (1/P) sum spring (first version forgot 1/P; fixed).
- Contraction T = sqrt(P'/P) C'_{jl} C_{kl} over the P' lowest modes (MM2008).
- Water fit: Morse bond with the table De could not absorb the huge intramolecular pGM Coulomb
  (q_O = -2.04 e with covalent dipoles); quartic bond + angle harm/cubic + bond-bond + bond-angle,
  refs by Nelder-Mead, force constants by linear least squares: rms 0.30 kJ/mol, 25 kJ/mol/nm
  (target rms force 3318), min 0.09424 nm / 107.59 deg (target 0.09419 / 107.4), harmonic
  1600 / 3847 / 3916 cm-1 (target 1580 / 3853 / 3920).

## Status / log
- 21:00 GPUs gpu-2-0..4 all busy with larry's alchemical/pathenv opt_driver2.sh (pmemd, outside the
  lock); not interfering. CPU work meanwhile (cpu-long partition via runs/cpu_long.sh).
- tests/test_pimd.py: 12 tests pass on CPU (~6 min).
- Model validation (validation/pimd/*.json): HO omega=100, 700 rad/ps, P=1..64: V, Kprim, Kcv = exact
  P-bead values within error; Kprim at large P has a small time-step bias (dt = 0.1/omega; gone at
  0.025/omega), Kcv none. Free particle: all mode temperatures 299.6-300.4 K, spreads 1.000+-0.002.
  RPMD NVE: sd(H_P) ~ dt^2, both propagators.
- OpenMM RPMDIntegrator comparison (scripts/pimd_openmm.py): P=8 full / c1 / c3 agree within ~1-2 sd.
- CPU water: classical (P=1) 1.38 ns/day at 0.5 fs, 48 cores; P=32 contracted to 1: 90 ms/step.
  P=32->1 KE_H ~152 meV; CG 3.2/step with centroid contraction vs ~7 with bead evaluations.
