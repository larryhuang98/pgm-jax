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
