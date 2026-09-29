# NOTES: interfaces to other MD codes (item 20), branch iface

Running log so the work can be resumed.

## Environment
- ASE 3.29 is already in the pgmjax env.
- OpenMM: 8.3.1 (plm env) has no PythonForce.  PyPI wheels of 8.6.1 need glibc 2.34 (compute nodes
  have 2.28).  Own env `~/miniconda3/envs/pgmjax-iface-omm` (conda-forge openmm 8.6.1, python 3.12,
  created on the head node with CONDA_OVERRIDE_GLIBC=2.28).  Used from the pgmjax python through a
  directory holding only a symlink to its `openmm` package: `PYTHONPATH=runs/ommlib` (runs/ is
  gitignored; recreate with `mkdir -p runs/ommlib && ln -s ~/miniconda3/envs/pgmjax-iface-omm/lib/python3.12/site-packages/openmm runs/ommlib/openmm`).
- i-PI 3.3.0: pure-Python wheel unpacked into `runs/pylib` (PYTHONPATH=runs/pylib; server script
  `runs/pylib/ipi-3.3.0.data/scripts/i-pi`).
- Head node glibc 2.17 (no project python), compute nodes 2.28 and no network; the head node has
  network (curl to pypi / conda-forge).
- cpu-short is often congested by other agents; `runs/cpu_long.sh` = cpu_run.sh on cpu-long.

## Design
- `pgm_jax/interfaces/engine.py`: PGMEngine (periodic; rigid-molecule model or flexible templates),
  GasPhaseEngine.  One jitted call per configuration: unwrap molecules (bond tree, pointer doubling),
  wrap by COM, neighbour-list update, PGMForceField.compute, bonded terms, cell dipole; one packed
  array to the host.  Slots of induced-dipole histories (i-PI beads).  Overflow -> resize + repeat.
- Virial: ff.strain_derivative is only exact for strains that keep H lower triangular (eps upper +
  diagonal); the lower off-diagonal components came out wrong (3e-3 relative vs FD).  The engine
  uses triu(W) + triu(W,1).T -> exact (3e-13 vs FD of ASE energies).  Native code only uses the
  trace (pressure), so it was unaffected.
- ASE: PGMCalculator; FixRigidMolecules = vectorised SHAKE/RATTLE (ASE's FixBondLengths loops in
  Python: 0.6 s/step for 30 waters; ours agrees with it to 2e-13 and costs ~1 ms).

## Log
- 01:40 start; env checks; engine + ASE calculator written.
- engine vs native ff.compute: E identical, F 1e-12, W 1e-12; rotated general cell + atoms wrapped
  one by one: E 4e-12, F 1e-11.
- Slurm: jobs without --mem get the whole node's memory (MIN_MEMORY 0) and queue behind each
  other; with `--mem=8G` (runs/cpu_long.sh MEM=..., or sbatch -p cpu-short ... --mem=8G) they start
  at once on the partly used nodes.
- CPU (24 cores) ASE validation, 512 waters: double single point identical to native (E 2e-16
  relative, F 4e-10, mu 2e-15); NVE 4 ps drift native -0.004, ASE -0.0007 kT/ns/dof; E0 equal to
  1e-6 kJ/mol; native 20.4 ms/step, ASE 30.9 (engine 25.2).  Pressure factor bug in the script fixed.
  Native Langevin NVT came out at 290 K (6 ps; check on GPU with 20 ps).
- i-PI smoke test (32 flexible methanols, double): P=1, P=4 batched, P=4 serial with 4 slots run;
  step-0 potential = engine energy (1e-8, i-PI's output digits); batched and serial give identical
  series (same seed).
- stress default: "molecular" for the rigid-molecule model, "atomic" with templates.
- 05:46 GPU batch 1 (gpu-2-2): ASE validation done (validation/interfaces/ase.json): single point
  double identical (E 2e-16, F 4e-10, P 0.0 bar after the script fix), mixed within mixed noise;
  NVE 10 ps native +0.0011, ASE -0.0022 kT/ns/dof; NVT 20 ps T 296.8+-0.8 / 298.7+-0.8,
  U -4130.885+-0.073 / -4130.723+-0.074 kJ/mol/molecule. Speed: native 0.80 ms/step, ASE 4.26 (engine
  1.90 at that time). OpenMM step crashed (CUDA platform), i-PI script had a syntax error.
  (lock mess on gpu-2-2: gpu_run.sh's trap removes the lock dir unconditionally; killing a waiting
  gpu_run.sh with SIGTERM removed fit2's lock; SIGKILL my own waiting gpu_run.sh processes instead.)
- runs/gpu_any.sh: waits for any free gpu-2-x lock + idle GPU, then gpu_run.sh there.
- 06:32 GPU batch 2 (gpu-2-0): profile: native 0.61 ms/step, engine.compute 1.37 ms, _fn 0.93 ms
  (8 CG), ff.compute alone 0.87 (8 CG), native integ.forces 1.19 (8 CG).  Host overhead: one extra
  device->host transfer for nb.failed(nbr) -> use the error code packed in the output.
- OpenMM CUDA platform: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (conda nvrtc 13.4 vs driver 610), and in
  any case the GPUs are in exclusive-process mode = one context per device: OpenMM's own context and
  JAX's primary context cannot coexist ("CUDA_ERROR_DEVICE_UNAVAILABLE").  Use OpenMM CPU platform +
  JAX GPU (OpenMM integration + SETTLE ~0.5 ms/step for 1536 atoms).
- i-PI (GPU): NVE 2000 steps from the same state: U agrees with native to 2e-6 relative over the run
  (float32 chaos), E0 differs by 0.29 kJ/mol = KE 5e-5 (i-PI unit constants); drift both 0.08 kT/ns/dof
  (0.5 ps); NVT classical 2 ps T 299.2+-1.1 vs 296.2+-0.8, U -4088.85+-0.07 vs -4088.70+-0.09.
  PIMD P=8 (batch, one call per bead then): KE_H 119.18 vs native 118.89+-0.10 meV, KE_O 51.09 vs
  51.15+-0.06, <V> -2084795+-53 vs -2085188+-43 (2.5 ps vs 10 ps).  37 ms/step (engine 3.1 ms per
  bead, 39 % predictor resets: i-PI does not keep the bead order in batches -> slot = batch index is
  wrong).  Fixed: one-to-one assignment to previous structures (match_previous), and
  PGMEngine.compute_batch: all beads in one vmapped call (shared list of the centroid + bead margin,
  molecule or atom list; stacked histories with a shared predictor counter; chunks).
- 07:15 GPU batch 3: bench x1 (1536 atoms): native 0.70 ms, engine 1.01 (CG 4.0), ASE 4.2, OpenMM-CPU
  1.53; x2 (12288): native 1.90, engine 3.08, ASE 20.4 (deepcopy of the constraint list in
  atoms.copy(): fixed with a light copy + constraint arrays), OpenMM 3.49.  Jump test now uses the
  minimum image (the native driver's wrapping had reset the predictor: CG 7.2 -> 4.0).
  i-PI PIMD P=8, 10 ps, vmapped beads (first batch logic): KE_H 119.28+-0.03 (native 118.89+-0.10),
  KE_O 51.15+-0.03 (51.15+-0.06), <V> -2084885+-35 (native -2085188+-43); 22 ms/step (engine 1.5 ms
  per bead); 17 % of i-PI's batches are partial (padded with copies), 38 % resets.  i-PI's default
  splitting is OBABO (native BAOAB + Cayley): rerun with splitting="baoab", propagator="cayley".
- compute_batch rewritten: dedupe padding copies, rectangular assignment of the distinct structures
  to P slots, vmapped call over all P slots with absent slots kept (masked state update).
- CPU i-PI (16 cores): engine 105 ms per call vs native 54 ms per step (i-PI's polling thread, latency
  1e-4 s, competes with XLA's threads; cancelled the CPU PIMD run: 2 s/step).
- Native rigid-body Langevin (gamma 1/ps) came out 290-297 K in the 6-20 ps runs of these scripts
  (ASE / OpenMM Langevin 297-299 K); not investigated (outside this feature).
- 08:16 GPU batch 4: PIMD P=8 with i-PI splitting="baoab" + Cayley (as native): KE_H 118.80+-0.04
  (native 118.89+-0.10), KE_O 51.09+-0.01 (51.15+-0.06), <V>/molecule -4072.17+-0.09 (-4072.63+-0.08;
  first block high: equilibration).  P=32 (2.5 ps): KE_H 148.53+-0.06 (148.48+-0.07), KE_O 55.21+-0.03
  (55.22+-0.06), <V>/molecule -4066.45+-0.11 (-4066.64+-0.07).  Bench after fixes: x1 engine 1.01 ms
  vs native 0.73; x2 2.90 vs 1.89; ASE x2 12.3 ms (constraints 3.6); OpenMM-CPU 1.50 / 3.19.
- i-PI splits a step's beads over 1-2 batches (P=32: always 2) -> compute_batch evaluates only the
  batch's slots (padded to P/4, P/2, P).  Timing (no virial): P=1 3.21 ms/step (native 2.24), P=8
  11.8 (engine 6.4; native 2.44), P=32 33.1 (engine 17.1; native 15.5); serial P=8 20.8.
- Full suite (08:17 code): 221 passed in 35 min (cpu-short, 24 cores).
- 10:10 full suite on the final code: 221 passed in 34 min (cpu-short, 24 cores). Done.
