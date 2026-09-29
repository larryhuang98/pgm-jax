# How to: bonded terms for a flexible pGM molecule

Goal: bond, angle, torsion (and coupling) parameters for a molecule whose electrostatics is pGM
with **all pairs** (no 1-2/1-3/1-4 exclusions), so that the molecule can be simulated flexibly in
the MD engine. The bonded terms only carry what the all-pair pGM electrostatics and the
Lennard-Jones from 1-5 pairs on do not, so they are fitted *on top of* that nonbonded model, to
DFT energies and forces.

Everything below runs on a GPU node (`ssh gpu-2-x; conda activate pgmjax; cd ~/project/pGM-JAX`),
except the DFT and ESP jobs, which are Slurm arrays on the CPU partition.

## 1. Reference data for a new molecule

| Step | Command | Output |
|---|---|---|
| topology + conformers (RDKit) | add the SMILES to `pgm_jax/bonded/study/molecules.py` (`MOLECULES`), then `python scripts/bonded/build_molecules.py` | `data/bonded/molecules/<name>.json` |
| MACE-OFF minimum | `python scripts/bonded/mace_min.py <name>` | `data/bonded/frames/<name>_min.npz` |
| sampling | `python scripts/bonded/mace_sample.py <name>` (Langevin MD at 500 K for training, 298 K for testing, relaxed torsion scans) | `data/bonded/frames/<name>_md.npz`, `_scan*.npz` |
| DFT labels | `python scripts/bonded/make_dft_tasks.py`, then `sbatch runs/bonded/dft.sh` (wB97M-D3(BJ)/def2-TZVPPD, psi4) | `data/bonded/dft/<name>__<key>__<start>.npz` |
| pGM parameters | `python scripts/bonded/pgm_params.py prep`, `sbatch runs/bonded/esp.sh`, `python scripts/bonded/pgm_params.py fit` (B3LYP/aug-cc-pVTZ ESP, py_resp, pGM-pol table, LJ from GAFF) | `data/bonded/params/<name>.json` |

`make_dft_tasks.py` only writes tasks for frames without labels, so it can be rerun after
failures (`--retry`, `--exclude`, `--only`; `--help`). The slurm scripts of `make_dft_tasks.py` and
`pgm_params.py prep` run psi4 with `--psi4-python` (default `$PGM_PSI4_PYTHON`, else `python`).

## 2. Fit and export

```bash
python examples/fit_bonded_template.py <name>                       # class II set of Abdullah et al.
python examples/fit_bonded_template.py <name> --families diag+ub    # or any set / "fam1+fam2+..."
```

prints the test errors on the 298 K frames (energy MAE, force error, kcal/mol and kcal/mol/A) and
writes `runs/flex/<name>.flex`. Families are listed in `pgm_jax/bonded/terms.py` (`REGISTRY`);
named sets are in `scripts/bonded/experiments.py` (`FAMILY_SETS`). For systematic comparisons
(several molecules, typed parameters shared across molecules, leave-one-out, torsion scans) use
`scripts/bonded/experiments.py run ...`; `data/reports/bonded/README.md` has the findings so far.

In Python:

```python
from pgm_jax.bonded.data import frames, mol_spec
from pgm_jax.bonded.fit import Fitter
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.bonded import terms as T
from pgm_jax.md.flexible import FlexibleTemplate

spec = mol_spec("methanol")
model = BondedModel([spec], BondedSettings(families=T.PAPER))  # pGM all pairs + LJ 1-5+
fit = Fitter(model, {0: {"train": frames("methanol", "train500"), "test": frames("methanol", "test298")}})
P = fit.fit(model.init_params())
print(fit.metrics(P, "test"))
FlexibleTemplate.from_fit(model, P).save("runs/flex/methanol.flex")
```

Only fits that use the MD engine's model can be exported: pGM with all pairs, no refitted
charges and no learned pair scales (`FlexibleTemplate` checks this). Charge flux
(`BondedSettings(flux=1 | 2)`) runs in MD as fitted: `docs/charge_flux.md`.

## 3. Check it in MD

```bash
python examples/flex_methanol_check.py                      # methanol: forces vs gas phase, NVE, NPT
python examples/run_flexible_liquid.py runs/flex/<name>.flex --molecules 216 --time-ps 100
python scripts/bonded/md_check.py <run> --mols <name> --families paper   # gas-phase MD: stays bounded? fluctuations vs MACE-OFF
```

Things to look at: the NVE energy drift (use `dt` 0.5 fs with X-H bonds), whether the class II
cross terms stay bounded at 298 K (they are unbounded below far from equilibrium), the internal
temperature (`temp_internal` in the log) and the liquid density.

## 4. Adding a term family

A family is a JAX function of internal coordinates with its own index set, parameters and keys;
see any entry of `REGISTRY` in `pgm_jax/bonded/terms.py` (about 20 lines each) and the finite
difference test in `tests/test_bonded.py`, which checks every registered family automatically.
