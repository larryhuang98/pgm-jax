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
