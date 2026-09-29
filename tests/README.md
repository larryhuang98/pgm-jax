# pgm_jax tests

The test suite checks the physics and numerics of pgm_jax against references that do not depend
on the code under test: analytic results (harmonic oscillators, closed-form kernels, exact
Ewald sums), finite differences (forces, virials, parameter gradients with the induced dipoles
re-solved), Amber / sander / pmemd-pgm outputs, published formulas (OpenMM / sander virtual
sites, PLUMED's OPES), and the golden outputs of the regression harness.  Everything runs in
float64 on the CPU unless a test says otherwise; no test needs a GPU.

## Layout

| Path | Contents |
|---|---|
| `test_<feature>.py` | one module per library module or feature (`test_md.py`: MD engine core, `test_mts.py`: multiple time stepping, ...); the module docstring says what is checked against what |
| `_systems.py` | shared test systems, settings and helpers (water, methanol, boxes, templates, alchemical and flux systems, peptides, `fd_check`, `md_settings`, marker helpers); the toy molecules and boxes are those of `pgm_jax.models.toy` |
| `conftest.py` | enables `jax_enable_x64` before anything is imported |
| `test_docstrings.py` | docstring completeness of `pgm_jax/` (expected failure until P8 is merged) and of `tests/` |
| `data/` | small tleap inputs (peptide in TIP3P / TIP4P-Ew, TIP4P-Ew / TIP5P boxes) and the legacy checkpoints of `test_checkpoints.py` (with the script that wrote them) |
| `regression/` | the golden-output harness (`regress.py`, `regression_cases.py`, `regression_systems.py`, `golden/`) |

Conventions:

- Test modules never import each other; a builder used by more than one module goes into
  `_systems.py`.  White-box tests of private internals stay next to the module's other tests.
- Test names read `test_<unit>_<property>` where the module does not already name the unit
  (`test_periodic_box_gradient`, `test_nve_energy_conservation`).
- Every module, test, fixture and helper has a docstring: tests state the property checked, the
  reference and, where it is not obvious, where the tolerance comes from (finite-difference
  truncation, solver tolerance, float32, statistical error); helpers what they build, with sizes
  and units (docs/dev/style_guide.md, section 14).
- Random numbers come from seeded numpy generators or JAX keys: every run compares the same
  numbers.

## Markers

Registered in `pyproject.toml` (`--strict-markers`):

| Marker | Meaning |
|---|---|
| `slow` | takes more than about 20 s on 32 CPU cores (63 tests, about 60 % of the run time) |
| `needs_data` | needs data outside the repository: the pGM3P-25 box (`PGM_GVDW_DATA`), the Amber `pgm_4wat` test (`AMBERHOME`), the QM set `data/qm/` |
| `needs_external` | needs an external program: `pmemd.pgm` (`PGM_PMEMD_BIN`), i-PI (`IPI_ROOT`), OpenMM >= 8.4 |
| `optional_deps` | needs an optional Python package: RDKit (peptides), ASE, networkx |
| `regression` | the golden-output harness (opt-in, see below) |

Tests with `needs_data`, `needs_external` or `optional_deps` skip (never fail) when their data,
program or package is missing; `pytest -rs` lists the reasons.

## Running

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests                      # everything (CPU, ~70 min on 32 cores)
JAX_PLATFORMS=cpu python -m pytest -q tests -m "not slow"        # the fast suite (~30 min)
python -m pytest -q tests/test_md.py -k strain                   # one module / one test
python -m pytest -q tests -m "not (needs_data or needs_external)" -rs   # a machine without the extras
python -m pytest -q -s tests/test_fe_grad.py                     # with the diagnostic prints
```

The test modules compile many JAX programs; set `OMP_NUM_THREADS` to the cores you have.

## Regression harness

`tests/regression/` runs 35 cases (single points and short trajectories of every engine and
feature) and compares their outputs with the golden files recorded on master, bitwise by
default:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 python tests/regression/regress.py list
JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 python tests/regression/regress.py check [--group a] [--out report.json]
PGM_REGRESSION=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 python -m pytest tests/regression -q   # the same as tests
```

The golden files are valid for one JAX / XLA build and CPU family (docs/api_design.md, section 7);
they are never edited by hand, only re-recorded on master after an environment change.  A
change of the engine API updates the "API adapter" section of `regression_cases.py` in the same
commit; the numbers must not change.

## Checks for new tests

```bash
python scripts/dev/check_docstrings.py tests --list                 # missing docstrings
ruff check tests && ruff format --check tests
ruff check --config ruff-docstrings.toml --exit-zero tests          # numpy docstring style
```
