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
- 04:13 eq.npz from the GPU (NVT 2 ps 0.5 fs + NPT 200 ps at 1 fs X-H; density ~0.79). Then GPUs
  blocked again (remd2 holds gpu-2-4); CPU jobs need `--mem` (default = whole node, one job per
  node): runs/cpul.sh adds --mem=16G.
- NVE mixed (CPU, 216 MeOH, 20 ps after 2 ps Bussi): shake/rattle errors <= 1e-14 every step;
  drift kT/ns/dof: none-0.5 -0.004, hb-0.5 -0.0004, hb-1 +0.006, hb-2 +0.003, hb-2.5 +0.009,
  hmr-2 +0.008, hmr-3 -0.009, hmr-4 +0.27 (!), ab-2 +0.005, ab-3 +0.018, ab-hmr-4 +0.009,
  ab-hmr-5 +0.029. HMR 3.024 leaves the methyl C at 5.96 amu: C-O stretch at ~1320 cm-1 limits
  X-H-only constraints to 3 fs; all-bonds + HMR runs at 4 fs. NVE T at full steps is low at
  large dt (278-287 K at 3-4 fs): full-step kinetic energy bias of velocity Verlet.
- bug fixed: frames per step used fs/ps mix (ZeroDivision) in sample/engine_md.
- meoh125 single point vs pmemd.pgm: EELEC -8948.3885 / -8948.3876, others equal.
- running: 16 pmemd.pgm CPU runs (0.25 ns each), 8 engine CPU runs (0.25 ns), samples hb-2,
  hmr-4, ab-2, ab-hmr-4 on CPU (1 ns); GPU queue (runs/gpu_queue_shake.sh): bench, hb-0.5, none-0.5.
