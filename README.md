# pgm-jax

The polarizable Gaussian multipole (pGM) force field in JAX: energies, forces, induced dipoles and
virials are differentiable functions of the coordinates, the parameters and the box. It includes a
GPU molecular dynamics engine that reads Amber `prmtop` / `rst7` files, and tools to fit parameters.

## Install

```bash
pip install -e .            # from the repository root; provides the `pgm-jax` command
pip install -e ".[md,test]" # + jax-md (needed for molecular dynamics) and pytest
```

Requires Python >= 3.10 and JAX (install the CUDA build of `jax` for GPUs). Without installing, put
the repository on the path: `PYTHONPATH=/path/to/pgm-jax python scripts/md/run_md.py ...`.

**Check the installation.** `examples/water512` is a box of 512 pGM3P-25 waters. Ten NVE steps
take a few seconds; the potential energy of step 10 must be -2114884.26 kJ/mol (CPU or GPU, double
precision; -2114884.1 to -2114884.2 in mixed precision):

```bash
pgm-jax md -p examples/water512/pgm3p25_512.prmtop -c examples/water512/pgm3p25_512.rst7 -o check \
    --nsteps 10 --report-every 5 --thermostat none --barostat none --dt-fs 1.0 --precision double
pytest -q tests/test_water512.py     # the same run on CPU (double) and GPU (double, mixed); GPU tests skip without a GPU
```

## Use

**Molecular dynamics** (Amber inputs and outputs, Amber-style options and Angstrom):

```bash
python scripts/md/run_md.py -p water.prmtop -c water.rst7 -o md --barostat mc --temp 298 --press 1 \
    --nsteps 100000 --dt-fs 1.0 --cut 9.0 --nfft 48 48 48 --order 6 --ew-coeff 0.4 --vdwmeth 1 \
    --gamma 2.0 --report-every 1000 --traj-every 1000 --checkpoint-every 10000
```

This writes `md.log`, `md.nc` (Amber NetCDF trajectory), `md.rst7` and `md.chk`; continue with
`--continue-from md.chk`. The same run from Python:

```python
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import Bussi

sim = Simulation.from_amber("water.prmtop", "water.rst7", settings=MDSettings(cutoff=0.9), thermostat=Bussi(1.0))
sim.run(100_000, prefix="md", report_every=1000)
```

**Energies and forces** of a system in a periodic box:

```python
from pgm_jax import PeriodicModel

# system: pgm_jax.System, box: (3, 3) nm, positions: (N, 3) nm
model = PeriodicModel(system, box, positions, cutoff=0.9, vdw="lj", lj_lrc=True)  # vdw: "lj" | "de" | "gvdw" | "none"
energies = model.energy(positions)  # jax.grad gives forces, parameter and box derivatives
```

**Command line.** `pgm-jax --help` lists the subcommands (`md`, `pimd`, `remd`, `dielectric`,
`solvation`, `fit-liquid`, `fit-qm`, ...); each wraps a script in `scripts/`. Script options carry
their unit in the name (`--dt-fs`, `--time-ns`, `--temperature-K`); `run_md.py` keeps Amber's.

Other data and programs (Amber builds, water boxes, QM data) are found through environment
variables with defaults, listed in `pgm_jax/paths.py`.

## Test

```bash
pytest -q                    # full suite (about 70 min on 32 CPU cores)
pytest -q -m "not slow"      # fast suite (about 30 min)
```

## Documentation

- `docs/reference.md`: the complete reference (model, MD, PIMD, replica exchange, fitting, validation against Amber, speed).
- `docs/model_options.md`: electrostatics levels, van der Waals forms (LJ, double exponential, GVDW), bonded sets.
- `docs/howto_vdw.md`, `docs/howto_bonded.md`, `docs/liquid_fit.md`, `docs/qmfit.md`: parameterization.
- `docs/dielectric.md`, `docs/free_energy.md`, `docs/pimd.md`, `docs/interfaces.md` (ASE, i-PI, OpenMM).
- `docs/dev/style_guide.md`, `tests/README.md`: contributing.
