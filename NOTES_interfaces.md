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
