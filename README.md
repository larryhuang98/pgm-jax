# pGM-JAX

The polarizable Gaussian multipole (pGM) model in JAX: Gaussian charges, covalent permanent
dipoles and induced Gaussian dipoles, every atom pair interacting (no 1-2/1-3 masking), in the gas
phase and under periodic boundaries. Energies, forces (autodiff), induced dipoles, molecular
polarizabilities and n-body decompositions, batched over geometries. Validated against Amber
(sander and pmemd-pgm) and PyRESP.

Extracted on 2026-09-23 from `~/project/evoff` at commit `73d961c` (the pGM core of that
project, unchanged except for imports; evoff keeps its own copy). Not included: the searched pair
terms and expression language, the LLM search, the datasets and the Psi4/S66 pipeline.

## Layout

| Path | What is in it |
|---|---|
| `pgm_jax/units.py` | units (nm, e, kJ/mol) and constants, incl. Amber's pGM Coulomb constant |
| `pgm_jax/system.py` | `Molecule` (template + pGM parameters), `System` (molecules flattened into static index arrays) |
| `pgm_jax/kernels.py` | Gaussian pair kernels: Coulomb erf(b r)/r, overlap, C6 dampings (gd6, tt6) |
| `pgm_jax/channels.py` | `ElecChannel` (pGM electrostatics + induction), `elec_decomposition` (SAPT-like elst/ind), `molecular_polarizability` |
| `pgm_jax/ewald.py` | `PeriodicPGM`: Ewald sum, triclinic boxes, CG induction solve with exact Hessian-vector products |
| `pgm_jax/solver.py` | dense linear induction solve; Newton solver with implicit differentiation (`lax.custom_root`) for non-quadratic response |
| `pgm_jax/model.py` | `Model` = list of channels: energies, forces, batching, interaction and n-body energies |
| `pgm_jax/param.py` | Amber pGM prmtop reader, JSON save/load, py_resp `.chg` + pGM-pol table reader, bond-graph atom mapping |
| `tests/` | `pytest -q` (Amber-dependent tests skip if the Amber files are missing) |
| `scripts/validate_amber.py` | full comparison with sander / pmemd-pgm on 512 waters, and with PyRESP |
| `scripts/bench.py` | timings on the current device (gpu-2-3, float64: 512 periodic waters 27 ms per force call; 4000 dimers 4 ms; 45,316 trimers 0.11 s) |
| `validation/amber_ref/` | the Amber reference runs (inputs + outputs) used by `validate_amber.py compare` |
| `validation/validate_amber.json` | the numbers below |

## Use

```python
import jax; jax.config.update("jax_enable_x64", True)
from pgm_jax import ElecChannel, Model, System, read_prmtop_pgm

w = read_prmtop_pgm("rayl_512_v2.prmtop")[0]         # pGM3P-25 water
sys = System([w] * 3)
model = Model([lambda s: ElecChannel()])
e = model.energy_fn(sys)(pos, None)                    # pos (9, 3) nm -> {"perm", "ind", "total"} kJ/mol
f = model.forces_fn(sys)(pos, None)                    # kJ/mol/nm
nb = model.nbody(sys, coords)                          # coords (B, 9, 3): interaction, 2- and 3-body
```

## Running on rayl8

The login node's glibc is too old for jaxlib: run on a GPU node (all nodes share /home8).

```bash
ssh gpu-2-3
source ~/miniconda3/etc/profile.d/conda.sh && conda activate evoff    # same environment as evoff
cd ~/project/pGM-JAX
pytest -q
python scripts/validate_amber.py compare     # gas phase + periodic vs Amber, ~1 min
python scripts/validate_amber.py pyresp      # monomer vs PyRESP
python scripts/bench.py
```

## Validation (512 pGM3P-25 waters; same Coulomb constant as Amber)

| Comparison | Energy diff (kcal/mol) | Force RMS diff (kcal/mol/Å) | Induced dipoles |
|---|---|---|---|
| gas-phase cluster vs sander (no cutoff) | 2 × 10⁻⁵ of −505,230 | 4.2 × 10⁻⁵ (RMS force 38.8) | |
| periodic vs pmemd-pgm (PME, tight) | 4 × 10⁻⁵ of −506,165 | 8.8 × 10⁻⁷ (RMS force 35.7) | every atom to 1.2 × 10⁻¹⁰ e·Å RMS |
| water monomer vs PyRESP | | | 8 × 10⁻¹⁰ a.u. |

Conventions worth knowing:

- **Coulomb constant.** The model uses CODATA (332.06371 kcal Å/mol). Amber's pGM code uses
  Tinker's 332.05382, 2.98 × 10⁻⁵ lower; comparisons rescale by `units.KE_AMBER_PGM / units.KE`.
- **Covalent dipoles** follow py_resp/Amber: p_i = Σ_k c_ik · unit(r_j(k) − r_i).
- **Energy split.** `ElecChannel` returns `perm` and `ind` of the whole system. For intermolecular
  electrostatics vs induction comparable with SAPT use `elec_decomposition`.

## Limits

- Gas-phase induction builds the dense 3n × 3n matrix: fine for clusters up to a few thousand atoms.
- Periodic code is plain Ewald (no PME), with a neighbour list built once from a reference
  geometry: correct for single points and small displacements, not yet an MD engine (no list
  updates, integrator or virial).
- float64 throughout; single precision is untested.
