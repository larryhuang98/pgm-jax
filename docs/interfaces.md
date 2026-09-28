# Interfaces to other simulation codes: ASE, i-PI, OpenMM

`pgm_jax/interfaces/`: the pGM force field of pgm_jax driven by other MD codes. The external code
integrates (thermostats, barostats, constraints, path integrals, enhanced sampling, reporters);
pgm_jax computes energies, forces, virials and dipoles. All three interfaces sit on one
device-resident engine, so the jitted pGM call stays on the JAX device (CPU or GPU) and each MD
step crosses the process / language boundary once, with one array in each direction.

| Interface | Module | Route | What it gives |
|---|---|---|---|
| ASE | `pgm_jax.interfaces.ase` | `PGMCalculator` (ASE `Calculator`) | energy, forces, stress, cell dipole, induced dipoles; `FixRigidMolecules` (vectorised SHAKE / RATTLE for rigid water) |
| i-PI | `pgm_jax.interfaces.ipi` | socket client (pure Python), `python -m pgm_jax.interfaces.ipi` | energy, forces, virial, dipole extras; i-PI's batched requests (all beads of a ring polymer in one message) |
| OpenMM | `pgm_jax.interfaces.openmm` | `openmm.PythonForce` (OpenMM >= 8.4) | energy and forces inside any OpenMM `System`: integrators, SETTLE / SHAKE, Monte Carlo barostat, reporters |
| (all) | `pgm_jax.interfaces.engine` | `PGMEngine`, `GasPhaseEngine` | one jitted call per configuration, units nm / kJ/mol |

In short: energies, forces, dipoles and virials are those of the native engine to rounding; NVE,
NVT, NPT and path-integral runs driven by ASE, OpenMM and i-PI reproduce the native energy
conservation, temperatures, energies, density and quantum kinetic energies within their statistical
errors; a force evaluation from outside costs 1.4-1.5x a native MD step on the GPU (the round trip
of one synchronous call), plus what the driver itself costs per step.

## Installation

Nothing beyond pgm_jax for the engine and the i-PI client. The drivers:

- **ASE** (>= 3.23; 3.29 in the `pgmjax` environment): `pip install ase`.
- **i-PI** (server side only; the client is pure Python): `pip install ipi` (3.x). The tests and
  validation scripts find it through `IPI_ROOT` (a directory holding the `ipi` package and
  `ipi-*.data/scripts/i-pi`, e.g. an unpacked wheel) or an `i-pi` on the PATH.
- **OpenMM >= 8.4** (for `openmm.PythonForce`): `conda install -c conda-forge openmm` in the same
  Python as pgm_jax. On rayl8 the pip wheels of OpenMM 8.6 need glibc 2.34 (the compute nodes have
  2.28); conda-forge's build works. It is kept in its own environment,
  `~/miniconda3/envs/pgmjax-iface-omm` (OpenMM 8.6.1, python 3.12, made on the head node with
  `CONDA_OVERRIDE_GLIBC=2.28 conda create -p ... -c conda-forge python=3.12 openmm=8.6`). The pgmjax
  python imports it through a directory holding only a symlink to its `openmm` package:
  `PYTHONPATH=runs/ommlib` (with `runs/ommlib/openmm -> .../pgmjax-iface-omm/lib/python3.12/site-packages/openmm`),
  so nothing is installed into `pgmjax`. OpenMM's CUDA platform does not work next to JAX on rayl8's
  GPUs (Limits); its CPU platform does.

## Usage

### The engine

```python
from pgm_jax.interfaces import PGMEngine
eng = PGMEngine.from_amber("water.prmtop", "water.rst7", settings=MDSettings())     # rigid-molecule model
eng = PGMEngine(system, pos_nm, H_nm, settings, templates=[tpl] * n)               # flexible molecules
eng = PGMEngine.from_simulation(sim)                                               # a native Simulation's model
res = eng.compute(pos_nm, cell_nm, virial=True)
res.energy, res.forces, res.virial, res.terms       # kJ/mol, kJ/mol/nm, dE/deps (kJ/mol), elec / vdw / bonded
res.induced_dipoles, res.dipole                     # e nm (fetched from the device on access)
```

- **Models.** Without templates: the model of the rigid-molecule engine (`Simulation`): pGM with
  every pair, no intramolecular van der Waals, no bonded terms; the external code must keep the
  molecules rigid (ASE `FixRigidMolecules`, OpenMM constraints). With `templates`
  (`FlexibleTemplate` / `RigidTemplate`, one per molecule): `FlexibleSimulation`'s model (bonded
  terms, intramolecular van der Waals from `lj_min_sep`, charge flux). Restraints can be passed
  (`restraints=`). `GasPhaseEngine(model, system)` wraps the gas-phase `Model` (dense induction).
- **One jitted call per configuration.** Inside it: molecules made whole along their bond trees
  (pointer doubling; atoms may come wrapped one by one or never wrapped), whole molecules shifted
  into the cell, neighbour-list update (JAX-MD rebuilds when something moved more than skin/2),
  `PGMForceField.compute` (PME, pair rows, induced dipoles with the mu4 predictor), bonded terms,
  optionally the virial and the cell dipole (`eng.with_dipole`, set by the i-PI client; otherwise
  computed on first access). The host receives one packed float64 array (energies, flags, forces);
  induced dipoles stay on the device until asked for.
- **Induced-dipole history.** The engine keeps the predictor history between calls, so successive
  MD steps start the CG from the extrapolated dipoles as the native integrator does (same CG
  iteration counts). `slots=P` keeps P histories for interleaved configurations (ring-polymer
  beads sent one after the other): the first P calls fill the slots, later calls use the slot whose
  last configuration is closest; a caller can name the slot (`slot=k`). A jump larger than `jump`
  (0.05 nm, minimum image) restarts that slot's predictor.
- **Batches of beads.** `eng.compute_batch(X, cell)` (X: B x N x 3) evaluates close structures that
  share the cell in one vmapped call (chunks of 8 above 8): P slots with stacked dipole histories
  (the predictor's step counter shared, so its branch stays a real branch under vmap), one
  neighbour list of the slots' mean with the radius enlarged by `bead_margin`, exact duplicates
  evaluated once and every distinct structure assigned to the slot with the closest last
  configuration (one-to-one; i-PI reorders beads and splits a step over one or two batches).
- **Safety.** Row / neighbour-list overflows, an atom beyond the list radius of its group, a
  volume change above 10 % (barostats) or a box too small for the molecule list are detected after
  the call; the engine resizes or rebuilds and repeats the call (counted in `eng.stats`). Nothing
  is silent.
- **Cells.** Any right-handed cell. General (ASE) cells are rotated to the engine's reduced
  lower-triangular form; forces, virial and dipoles are rotated back.
- **Virial.** `stress="molecular"` (default for the rigid-molecule model: molecular centres scaled,
  as the native pressure) or `"atomic"` (default with templates: every atom scaled, what ASE's
  stress and i-PI's virial mean for flexible molecules). Both include the long-range-correction
  impulse term of `MDSettings.lj_lrc` as `Simulation.pressure()` does.
  `PGMForceField.strain_derivative` is exact only for strains that keep the box lower triangular
  (the diagonal and eps_ab with a < b); its lower off-diagonal components were wrong (a 3e-3
  relative error found by finite differences here; the native code uses only the trace). The
  engine returns the symmetric tensor built from the exact components, which matches finite
  differences of the energy to 3e-13.

### ASE

```python
from pgm_jax.interfaces.ase import PGMCalculator, atoms_from_system, rigid_constraints
atoms = atoms_from_system(eng.sys, pos_nm, H_nm)        # symbols, the system's masses, cell, pbc
atoms.set_constraint(rigid_constraints(eng.sys))        # rigid-molecule model only
atoms.calc = PGMCalculator(eng)
VelocityVerlet(atoms, 1.0 * units.fs).run(1000)         # or Langevin, NPT, BFGS, NEB, ...
atoms.get_stress(); atoms.calc.get_property("dipole"); atoms.calc.get_induced_dipoles()
```

Properties in ASE units (eV, Angstrom, e): `energy`, `free_energy`, `forces`, `stress` (Voigt),
`dipole` (cell dipole; its parts M_q, M_perm, M_ind in `results["dipole_components"]`),
`induced_dipoles` (N x 3). A stress asked for after the forces of the same configuration costs one
strain derivative at the converged dipoles, not a second dipole solve.
`FixRigidMolecules` (from `rigid_constraints`) holds molecules of up to three atoms rigid with
SHAKE (Newton iterations on the three multipliers of each molecule) and RATTLE (one batched 3 x 3
solve), vectorised over molecules. ASE's `FixBondLengths` gives the same positions and momenta
(2e-13) but loops over pairs in Python: 0.6 s per step for 30 waters (ours: under 1 ms).

### i-PI

```bash
i-pi input.xml &                                      # <ffsocket mode="unix"><address>pgm</address> ...
python -m pgm_jax.interfaces.ipi --template water.flex --nmol 512 --address pgm --unix   # PIMD: <batch_size>P</batch_size>
python -m pgm_jax.interfaces.ipi --prmtop water.prmtop --address localhost --port 31415   # inet
```

or `IPIClient(engine_or_factory, address, unix=True).run()` from Python. The client speaks the
i-PI protocol (STATUS / INIT / POSDATA / GETFORCE / EXIT, atomic units), including i-PI 3's batched
requests (`<batch_size>P</batch_size>` in `<ffsocket>`): the beads of a step arrive in one or two
messages, each evaluated in one vmapped call (`compute_batch`; `--no-vmap`: one call per bead), each
bead with its own dipole history. Without batching use `--slots P` (beads sent one at a time are
matched to the closest of P histories). Returned: energy,
forces, virial (-W, symmetric), extras `{"dipole": [...] (e Bohr), "cg_iterations": n}` (i-PI's
`dipole` property works). The engine is built on the first structure i-PI sends (`--settings` takes
MDSettings as JSON). `scripts/interfaces/ipi_tools.py` writes i-PI inputs (masses from the system,
xyz with the cell), starts the server and reads its output.

### OpenMM

```python
from pgm_jax.interfaces.openmm import PGMOpenMM
om = PGMOpenMM(eng)
system = om.system(rigid=True)                   # masses, box, constraints (SETTLE for water), pGM PythonForce, CMMotionRemover
system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 298 * unit.kelvin, 25))
sim = app.Simulation(om.topology(), system, openmm.LangevinMiddleIntegrator(298 * unit.kelvin, 1 / unit.picosecond,
                     0.002 * unit.picoseconds), openmm.Platform.getPlatformByName("CPU"))   # JAX on the GPU
sim.context.setPositions(om.positions()); sim.context.setPeriodicBoxVectors(*om.box())
sim.reporters.append(app.DCDReporter("traj.dcd", 1000)); sim.step(100000)
```

`om.force()` alone adds pGM to any System (other forces, e.g. restraints or a CustomCVForce, can be
combined; the particles must be the engine's atoms in order). Flexible templates: `system(rigid=False,
constraints="h-bonds")`.

**Routes considered.** (1) `openmm.PythonForce` (OpenMM >= 8.4): chosen. It needs nothing but
OpenMM, works on every platform, with every integrator, constraints and the Monte Carlo barostat;
the jitted pGM call stays on the JAX device, and per step OpenMM hands over positions and takes
forces (on the CUDA platform through host memory: 2 x 37 kB for 1,536 atoms, ~0.05 ms). (2)
openmm-torch's `TorchForce` with a DLPack bridge (jax2torch) could keep the arrays on the GPU, but a
TorchForce runs a TorchScript module and TorchScript cannot call into JAX; wrapping the call in a
Python-side module brings back a host round trip, and it needs openmm-torch (not installed in any
environment here). At best it would save the two small copies above. (3) A `CustomExternalForce`
with per-atom force parameters updated every step (`updateParametersInContext`) evaluates the
force only at the start of the step, gives no energy consistent with the forces and costs a
parameter upload per step: unusable with OpenMM's integrators and the barostat. OpenMM 8.3 (the
`plm` environment) has no PythonForce.

## Validation

`scripts/interfaces/validate_ase.py`, `validate_openmm.py`, `validate_ipi.py`; results in
`validation/interfaces/*.json`. One RTX PRO 6000 Blackwell (JAX on the GPU; OpenMM's integrator on
its CPU platform, see Limits). The 512-water pGM box of the README (rigid model: pGM3P-25 water,
9 A cutoff, PME, dipole tol 1e-5, mixed precision, dt 1 fs, 298 K) for ASE and OpenMM; the flexible
pGM water of the PIMD work (`validation/interfaces/pgm_water_flex.flex`, q-TIP4P/F monomer surface,
dt 0.25 fs) for i-PI, which has no rigid-body integrator. Native = `Simulation` /
`FlexibleSimulation` / native PIMD (`pgm_jax/md/pimd.py`, PIMD branch) from the same state.

| Check | Native pgm_jax | External code + pgm_jax |
|---|---|---|
| **ASE** single point, float64, tol 1e-10: energy / forces / induced dipoles / pressure | -2115058.99134313 kJ/mol | same to 2e-16 relative; forces 4.5e-10 kJ/mol/nm max (RMS force 1495); dipoles 2e-15 e nm; pressure (molecular virial) 6e-11 bar |
| ASE stress vs finite differences of ASE energies (all components, atomic strain) | | 3e-13 relative (pytest) |
| ASE NVE 10 ps, VelocityVerlet + FixRigidMolecules | drift +0.0011 kT/ns/dof, sigma(E) 0.23 kJ/mol | drift -0.0022 kT/ns/dof, sigma(E) 0.25 kJ/mol; E(0) equal to 1e-6 kJ/mol |
| ASE Langevin 1/ps, 20 ps: T, U per molecule | 296.8 +- 0.8 K, -4130.89 +- 0.07 kJ/mol | 298.7 +- 0.8 K, -4130.72 +- 0.07 kJ/mol |
| **OpenMM** single point, float64 (PythonForce, State) | | energy identical, forces 2.9e-10 kJ/mol/nm max |
| OpenMM NVE 10 ps, VerletIntegrator + SETTLE | drift +0.0004 kT/ns/dof, sigma(E) 0.25 | drift +0.0028 kT/ns/dof, sigma(E) 0.27 |
| OpenMM LangevinMiddle 1/ps, 20 ps: T, U | 296.6 +- 1.5 K, -4130.84 +- 0.12 | 297.3 +- 1.6 K, -4130.94 +- 0.07 |
| OpenMM MonteCarloBarostat 1 bar (every 25 steps), 50 ps: density | 1.0183 +- 0.0010 g/cm^3 (native MC barostat, every 25 steps) | 1.0176 +- 0.0020 g/cm^3 |
| **i-PI** NVE, same state and masses, 2000 steps (velocity Verlet in both) | | potential energy along the trajectory equal to 2e-6 relative (float32 chaos); E_tot(0) differs by 0.29 kJ/mol = 5e-5 of the kinetic energy (i-PI's unit constants); same drift |
| i-PI classical NVT (SVR vs Bussi, tau 0.1 ps), 2 ps: T, U per molecule | 296.2 +- 0.8 K, -4088.70 +- 0.09 | 299.2 +- 1.1 K, -4088.85 +- 0.07 |
| i-PI PIMD P = 8, PILE-G, BAOAB + Cayley (as native), 10 ps: centroid-virial KE per H / per O | 118.89 +- 0.10 / 51.15 +- 0.06 meV | 118.80 +- 0.04 / 51.09 +- 0.01 meV (i-PI's default OBABO + exact: 119.28 +- 0.03 / 51.15 +- 0.03) |
| i-PI PIMD P = 8, bead-averaged potential per molecule | -4072.63 +- 0.08 kJ/mol | -4072.17 +- 0.09 kJ/mol |
| i-PI PIMD P = 32, BAOAB + Cayley, 2.5 ps (native 10 ps): KE per H / per O; potential per molecule | 148.48 +- 0.07 / 55.22 +- 0.06 meV; -4066.64 +- 0.07 kJ/mol | 148.53 +- 0.06 / 55.21 +- 0.03 meV; -4066.45 +- 0.11 kJ/mol |

The single points are the same code (the engine calls `PGMForceField.compute`), so they agree to
rounding; the MD rows check the plumbing (units, cells, wrapping, constraints, virial signs, dipole
histories) through the external integrators. Differences are within 1-2 standard errors (block
averages, 5 blocks; the errors of 2-20 ps runs are underestimates), with one exception: the P = 8
potential energy is 0.46 kJ/mol per molecule (0.011 %) above the native value, 3.8 of these block
errors (its first 2 ps block is the highest: equilibration of the ring polymer from a classical
start). The kinetic energies agree once i-PI integrates as the native PIMD does (BAOAB splitting,
Cayley free ring-polymer step); with i-PI's default OBABO splitting the H kinetic energy is 0.4 meV
(0.3 %) higher, a finite-time-step difference between the two integrators, not an interface error.
i-PI's batches: it splits the P beads of a step over one or two batches, in any order, and pads a
partial batch with copies of its last structure; the engine matches every structure to its own
dipole history (0 predictor restarts, 7.3 CG iterations per bead vs 7.95 native).

## Cost per step (overhead)

`scripts/interfaces/bench_interfaces.py`: rigid pGM3P-25 water, mixed precision, NVE, dt 1 fs,
the same GPU; `engine` = `PGMEngine.compute` alone on consecutive MD frames (what every interface pays
per force evaluation), the others are complete MD steps.

| ms per MD step | 1,536 atoms (512 waters) | 12,288 atoms (4,096 waters) |
|---|---|---|
| native `Simulation` (rigid bodies) | 0.73 | 1.89 |
| `PGMEngine.compute` (one force evaluation; 4.0 CG iterations, as native) | 1.01 (1.4x) | 2.90 (1.5x) |
| same with the virial (strain derivative) | 2.43 | 8.32 |
| OpenMM VerletIntegrator + SETTLE (CPU platform) + PythonForce | 1.50 (engine 0.92, State -> numpy 0.02) | 3.19 (engine 2.17, 0.03) |
| ASE VelocityVerlet + FixRigidMolecules + PGMCalculator | 3.33 (engine 1.57, constraints 0.55) | 12.25 (engine 2.72, constraints 3.60) |

- **Engine vs native.** The native engine runs blocks of steps inside one XLA program; an external
  code needs one call per force evaluation: host -> device positions, one dispatch, the dipole solve
  with the same predictor and CG iterations, device -> host forces (one packed array). The
  difference is the latency of one synchronous round trip and a few small kernels (molecules made
  whole, list centres): +0.3 ms per step at 1,536 atoms and +1.0 ms at 12,288. On the CPU the
  difference is small: OpenMM + engine 46.7 ms per step vs native 40.3 (16 cores, 1,536 atoms;
  NPT density 1.0197 +- 0.0027 vs 1.0179 +- 0.0033, `validation/interfaces/openmm_cpu.json`).
- **OpenMM** adds its integrator and SETTLE on the CPU platform plus the State -> numpy conversion
  (0.02-0.03 ms).
- **ASE** adds Python per step: `FixRigidMolecules` (SHAKE / RATTLE, vectorised numpy),
  VelocityVerlet's array arithmetic and the calculator's change checks (a copy of the Atoms without
  its constraints: ASE's own `atoms.copy()` deep-copies the constraint objects and cost 13 ms per step
  for 12,288 atoms). ASE is the most flexible and the slowest driver.
- **i-PI** (flexible water, 1,536 atoms, dt 0.25 fs, no virial; ms per MD step):

  | | native pgm_jax | i-PI + pgm_jax | of which the engine |
  |---|---|---|---|
  | classical NVE (P = 1) | 2.24 | 3.21 | 1.68 |
  | PIMD P = 8, beads batched (one vmapped call) | 2.44 | 11.8 | 6.4 |
  | PIMD P = 8, one engine call per bead | | 20.8 | 11.9 |
  | PIMD P = 32, beads batched (chunks of 8) | 15.5 | 33.1 | 17.1 |

  The rest is i-PI's own Python (normal modes, thermostats, estimators, sockets): 1.5 ms per step
  at P = 1, 5.4 ms at P = 8, 16 ms at P = 32. With the virial every step (pressure estimators,
  barostats) P = 8 takes 19 ms and P = 32 99 ms per step. i-PI is the driver for what it offers
  beyond the native engine (its integrators and estimators, committees, enhanced sampling), not
  for speed; batching the beads (`<batch_size>` = P) halves its cost.
- **Virial.** A stress / virial costs one strain derivative at the converged dipoles (autodiff of
  the energy at fixed dipoles): 1.4 ms at 1,536 atoms, 5.4 ms at 12,288 (the native engine computes it only for reports and
  the barostat does not need it). It is computed only when asked (ASE `get_stress`, i-PI
  with `virial=True`, the default of the client since i-PI's barostats and pressure estimators need
  it; `--no-virial` for NVT runs that do not print the pressure).

## Limits

- **Rigid molecules** are held by the external code: ASE (`FixRigidMolecules`, molecules up to three
  atoms) and OpenMM (distance constraints: SETTLE for water). i-PI has no rigid-molecule integrator
  that works with path integrals, so the i-PI client needs flexible templates. Rigid molecules of more
  than three atoms (methanol) have no constraint set here; use flexible templates.
- **OpenMM CUDA platform**: on rayl8's GPUs (exclusive-process compute mode: one context per device)
  OpenMM's own CUDA context and JAX's cannot coexist in one process, and the conda-forge OpenMM 8.6.1
  build also fails to load its kernels with driver 610 (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, nvrtc
  13.4). Use OpenMM's CPU platform with JAX on the GPU (its integrator costs ~0.5 ms per step for
  1,536 atoms); on GPUs in the default compute mode OpenMM's CUDA platform and JAX can share a GPU
  (not tested here). A TorchForce/DLPack route would avoid the host copies but not this.
- **OpenMM gives no virial** to a PythonForce: pressure reporters and anisotropic barostats that
  need the virial are not available; the Monte Carlo barostat works (it uses energies).
- **Virtual sites, alchemical regions, multiple time stepping, charge-flux free energies** of the
  native engine are not exposed (the engine refuses virtual sites); restraints are.
- **Batches** (`compute_batch`, i-PI with `batch_size`) assume structures that stay close to each
  other and share the cell (ring-polymer beads): one neighbour list of the mean of the P slots'
  configurations with the list radius enlarged by `bead_margin` (0.08 nm; an atom further than that
  makes the engine enlarge the margin and repeat; if the box cannot hold the enlarged list the
  structures are evaluated one by one). A batch is padded to P/4, P/2 or P structures (one compiled
  program per size). Unrelated structures (replicas at different temperatures) should use
  `vmap_beads=False`: one call per structure, still with separate dipole histories.
- **PIMD integrators differ**: i-PI's default splitting (OBABO, exact free ring polymer) and the
  native BAOAB + Cayley sample slightly different finite-dt ensembles; compare at equal splitting.
- **GPU sharing**: the PythonForce and ASE calculators call the engine from Python; one process per
  GPU. The i-PI client serves one i-PI instance (several clients: one engine each).
- i-PI's shared-memory transport (`mode="shm"`) and the MPI driver are not implemented in the
  client (unix and inet sockets are; with batches the socket traffic is one message per step).
