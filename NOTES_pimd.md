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
- 22:30 full suite (before barostat test): 219 passed (34 min, 32 cores).
- CPU P=32 contracted to centroid, 512 waters, 298 K, NVT at 0.9887 g/cm3, 10 ps: KE_H 152.6+-0.1 meV (cv),
  152.1+-0.3 (prim); KE_O 53.1; p -799+-53 bar (classical flexible -876+-50); dipole 2.083 D (classical
  2.044); g_OO peak 3.24 (classical 3.32) at 2.76 A.  CG 3.2/step.
- econs under PILE drifts (~0.01 kJ/mol/ps per bead dof): Langevin on the stiff internal modes
  randomizes the BAOAB splitting error; not a conservation diagnostic (use RPMD NVE).
- OpenMM comparison, 6 cases x 4 observables: all within 2.2 sd (chi2/dof 1.7); dt check submitted.
- GPU campaign script runs/pimd/campaign.sh waits for the pathenv driver on gpu-2-0 to finish.
- 23:15 GPU (gpu-2-0 free after pathenv driver). bench (512 waters, dt 0.25 fs): P=8 2.19 ms/step
  (9.9 ns/day), 8->1 1.44, 8->4 1.97; P=32 15.2 ms (1.42 ns/day), 32->1 2.46 (8.8), 32->4 3.19 (6.8).
- w32 (P=32 full, GPU, 10 ps): KE_H 148.48+-0.07 meV cv (148.0+-0.3 prim), KE_O 55.22, p -1143+-73 bar,
  dipole 2.143 D (<|mu|> 2.185), g_OO 3.257 at 2.76 A, CG 9.0/step (max over beads), 15.5 ms/step.
  => centroid contraction (P'=1) overestimates KE_H by 4 meV (2.8 %), dipole 0.06 D low.
- 23:45 full suite 222 passed (36.8 min, 32 cores).
- GPU results: P=16 KE_H 139.65+-0.07 meV; P=32 148.48+-0.07; contraction P'=1 152.66, P'=4 151.49 (the
  intermolecular part, with its induction, couples strongly to the O-H stretch -> high modes needed).
  Isolated-molecule check: MD engine monomer frequencies = gas model to 0.04 (double) / 0.6 cm-1 (mixed),
  so the contraction offset is liquid-phase physics, not a split artefact.
  TRPMD P=32 20 ps: D 2.18e-5 cm2/s; classical flexible (P=1) D 4.08 (rigid pGM NVE 4.27 in
  thermostat_ideas.md); PIMD P=32 (PILE-G) D 2.57.  NQE slow diffusion in this model (dipole 2.04 -> 2.14 D).
  RDF: OO peak unchanged (3.2-3.3 at 2.76 A); OH H-bond peak 1.46-1.49 -> 1.34-1.39; HH 1.42-1.44 -> 1.31.
- pathenv multi_worker on gpu-2-0 retries when the GPU is busy; my jobs now take the GPU in a gap
  (PIMD_WAIT_GPU=1 in scripts/pimd_water.py; runs/pimd/campaign3.sh).
- 00:10-00:45 GPU batches (single process holding the GPU: scripts/pimd_water.py batch, PIMD_WAIT_GPU=1):
  P=8 KE_H 118.9; P'=16 149.37, P'=8 150.35 (contraction error +0.9 / +1.9 meV); bead chunks: P=32 15.2 ->
  7.8 ms/step with chunks of 8 (now default "auto"); P=64 16.1 ms; RPMD NVE 512 waters P=32: H_P RMS
  60 / 15 kJ/mol at 0.25 / 0.125 fs, no drift beyond block wander; identical with tol 1e-7 / double.
  NPT: classical 1.0315+-0.003, quantum 1.0467+-0.0046 g/cm3. bead_margin default 0.08 (RPMD tripped 0.061).
- docs/pimd.md, README (feature bullet, layout row, limits, 222 tests) written.

## Left / ideas
- Contraction with a reference that includes the monomer's response to the environment field.
- Kubo-transformed correlation functions / IR spectra from TRPMD (post-processing only now).
- 0.5 fs time step for flexible pGM water not tested; rigid-rotor PIMD not implemented.
