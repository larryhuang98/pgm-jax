# NOTES: holonomic bond constraints (item 11), branch `shake`

Running log so the work can be resumed.

## State found at start (2026-09-28)
- md/constraints.py already had SHAKE/RATTLE per cluster (exact Newton, all clusters padded to the
  largest one), `hmr_masses`; flexible.py: `constraints="none" | "h-bonds"`, RigidTemplate (water by
  3 constraints), g-BAOAB with RATTLE after kicks/drifts/O steps, DOF = 3N - nc (- 3 NVE),
  MC barostat scales molecular centres (constraints kept), molecular virial (constraint forces are
  intramolecular, so they drop out). MTS (mts.py) already calls RATTLE after its kicks.
- README Limits still says "no bond constraints yet (dt 0.5 fs)"; no validation of constrained
  flexible liquids (only water via RigidTemplate and proteins).

## Plan
1. constraints.py: blocks by cluster size (no padding of every cluster to the largest), exact Newton
   for small clusters, matrix-free iterative SHAKE/RATTLE (quasi-Newton + CG) for large clusters
   (all-bonds of a protein), velocity (RATTLE) diagnostic, describe().
2. flexible.py: constraints="all-bonds"; Bussi DOF (-3, total momentum conserved); rattle_err
   observable; skip redundant RATTLE after drifts that are followed by a projected kick.
3. tests/test_shake.py.
4. scripts/validate_shake.py: NVE drift vs dt, equilibrium vs dt, speed; pmemd.pgm (SHAKE) comparison.
5. docs/shake.md, README.

## Log
- 02:10 constraints.py rewritten: _DenseBlock (clusters grouped by number of constraints; Newton,
  closed form C<=3, unrolled Gauss C<=12), _SparseBlock (matrix-free quasi-Newton + PCG, while
  loops, tol 1e-10), errors()/velocity_violation(); `atoms`/`invm` kept (flat) for test_hmr.
- flexible.py: "all-bonds"; momentum_conserved (NVE, Bussi): dof - 3 and net momentum removed at
  init; rattle_err observable; drift before a kick skips RATTLE (exact, test); log line.
- tests/test_shake.py (7 tests). Small methanol box has cutoff noise in NVE: use an 8-molecule
  cluster in a 3.6 nm box, cutoff 1.7 nm (no pair crosses). E fluctuation h-bonds 0.5/1/2 fs:
  0.084/0.337/1.37 kJ/mol (dt^2), none 0.5 fs 0.094.
- runs/meoh: MEOHBOX 2x2x2 by tleap (cpptraj on the cluster lacks libmvec) -> 1000 methanols;
  write_pgm_prmtop + single point vs pmemd.pgm CPU: EELEC -71587.1132 vs -71587.1014 kcal/mol
  (1.6e-7, PME lambda factor), BOND/ANGLE/DIHED/VDWAALS equal to 1e-4.
- GPUs: all five occupied by the owner's pmemd.cuda jobs since ~00:40; every agent's gpu_run waits.
  CPU nodes saturated too (queue). scripts: validate_shake.py, shake_vs_pmemd.py, bench_shake.py.
