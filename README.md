# pGM-JAX

The polarizable Gaussian multipole (pGM) model with Lennard-Jones, in JAX. **Fixed functional
form, everything differentiable:** energies, forces, induced dipoles, polarizabilities and virials
are JAX functions of the coordinates, of the parameters and, for periodic systems, of the box, to
any order. Validated against Amber (sander, pmemd-pgm) and PyRESP.

- **Electrostatics:** Gaussian charges, covalent permanent dipoles and induced Gaussian dipoles,
  with every atom pair interacting (no 1-2/1-3 masking).
- **Van der Waals:** Lennard-Jones between molecules (Amber form, Lorentz-Berthelot).
- **Systems:** gas phase (`Model`) and periodic (`PeriodicModel`, Ewald, triclinic boxes).

Started on 2026-09-23 from the pGM core of `~/project/evoff` (commit `73d961c`); this repository
is where the two projects diverge (evoff searches over functional forms, pGM-JAX keeps pGM's).

## Parameters are inputs

`Molecule` holds topology and initial values. `ParamTable` turns them into the free parameters:
for each quantity, a list of keys with one value per key. Atoms with the same key share one value
and one gradient. `System` is topology plus index arrays into the table. The parameter pytree
(`table.initial()`, a dict of arrays) is an argument of every energy function.

| Quantity | Meaning (units) | Default tying |
|---|---|---|
| `q` | Gaussian charge (e) | per molecule, symmetry-equivalent atoms tied (as py_resp) |
| `cov` | covalent dipole strength (e nm), p_i += c·unit(r_j − r_i) | per molecule, by the pair of symmetry classes |
| `radius` | pGM Gaussian radius R (nm) | atom type (as the pGM-pol table) |
| `alpha` | isotropic polarizability (nm³) | atom type |
| `lj_rmin_half` | LJ R* = r_min/2 (nm) | atom type |
| `lj_sqrt_eps` | √ε (√(kJ/mol)); the square root keeps gradients finite at ε = 0 | atom type |

Symmetry classes come from colour refinement on bonds + covalent-dipole pairs, so the keys don't
depend on atom order (for example `WAT:OW`, `WAT:HW`, `WAT:OW>HW`, `MeOH:h1`). To tie differently,
pass `Molecule(keys={quantity: [one key per atom or per covalent dipole]})`. Build one
`ParamTable` from all molecules and pass it to every `System` that should share parameters.

```python
import jax, jax.numpy as jnp; jax.config.update("jax_enable_x64", True)
from pgm_jax import ElecChannel, LJChannel, Model, ParamTable, PeriodicModel, System, read_prmtop_pgm

w = read_prmtop_pgm("rayl_512_v2.prmtop")[0]           # pGM3P-25 water: pGM + LJ + bonds + masses
sys = System([w] * 3)
P = sys.table.initial()                                  # {"q": (2,), "cov": (2,), "alpha": (2,), ...}
model = Model([ElecChannel(), LJChannel()])
E = model.energy_fn(sys)                                 # E(pos, P) -> {"perm", "ind", "vdw", "total"} kJ/mol
dE_dP = jax.grad(lambda p: E(pos, p)["total"])(P)        # same pytree as P
F = model.forces_fn(sys)(pos, P)                         # kJ/mol/nm
dL_dP = jax.grad(lambda p: jnp.sum((model.forces_fn(sys)(pos, p) - F_ref) ** 2))(P)   # force matching

box = PeriodicModel(sys512, H, pos512, rc=1.0, b0=3.8, lj_lrc=True)
box.energy(pos512, P, H)                                 # H: lattice vectors as rows (nm)
jax.grad(lambda h: box.energy(pos512, P, h)["total"])(H) # box derivative
box.pressure(pos512, P)                                  # static pressure (bar), molecular virial
box.elec.induced_dipoles(pos512, P)                      # differentiable too
```

How induction is differentiated: the induced dipoles minimise a quadratic functional. For the gas
phase the solve is dense; for periodic systems it is conjugate gradients (`lax.custom_linear_solve`,
implicit differentiation). The periodic energy is wrapped by `solver.variational`: first
derivatives use stationarity (one solve, like Hellmann-Feynman), and higher derivatives
differentiate through the solve. A deliberately broken version (no derivative through the solve)
gives second-order parameter gradients that are off by 12–100 %, so the second-order tests are
sensitive to this.

## Layout

| Path | What is in it |
|---|---|
| `pgm_jax/system.py` | `Molecule`, `ParamTable` (tying), `System` (topology + index arrays, `expand(params)`) |
| `pgm_jax/units.py` | units (nm, e, kJ/mol) and constants, incl. Amber's pGM Coulomb constant |
| `pgm_jax/kernels.py` | Gaussian pair kernels: Coulomb erf(b r)/r, overlap, C6 dampings (gd6, tt6) |
| `pgm_jax/channels.py` | `ElecChannel`, `elec_decomposition` (SAPT-like elst/ind), `molecular_polarizability` |
| `pgm_jax/lj.py` | `LJChannel` (gas phase), `PeriodicLJ` (cutoff, optional long-range correction) |
| `pgm_jax/ewald.py` | `PeriodicPGM`: Ewald with integer image and k-vector lists, so the box is a JAX input; neutralising background |
| `pgm_jax/periodic.py` | `PeriodicModel` (elec + LJ, shared neighbour list): energy, forces, `strain_derivative`, `pressure` |
| `pgm_jax/solver.py` | dense induction solve; `variational`; Newton solver with implicit differentiation |
| `pgm_jax/model.py` | gas-phase `Model`: energies, forces, batching, n-body energies (compiled once per topology) |
| `pgm_jax/param.py` | Amber pGM prmtop reader (incl. LJ, bonds, masses), JSON save/load, py_resp `.chg` + pGM-pol table, atom mapping |
| `tests/` | `pytest -q`: 26 tests, incl. finite-difference checks of every derivative |
| `scripts/validate_amber.py` | comparison with sander / pmemd-pgm / PyRESP (`compare`, `pyresp`, `virial`) |
| `scripts/bench.py` | timings on the current device |
| `validation/` | Amber reference runs (inputs + outputs) and `validate_amber.json` |

## Running on rayl8

The login node's glibc is too old for jaxlib: run on a GPU node (all nodes share /home8).

```bash
ssh gpu-2-3
source ~/miniconda3/etc/profile.d/conda.sh && conda activate evoff    # same environment as evoff
cd ~/project/pGM-JAX
pytest -q                                    # ~1 min
python scripts/validate_amber.py compare     # gas phase + periodic vs Amber, ~1 min
python scripts/validate_amber.py pyresp      # monomer vs PyRESP
python scripts/validate_amber.py virial      # molecular virial vs sander (ntp=1)
python scripts/bench.py
```

## Validation (512 pGM3P-25 waters; same Coulomb constant as Amber)

| Comparison | Result |
|---|---|
| gas-phase cluster vs sander (no cutoff) | EELEC diff 2 × 10⁻⁵ of −505,230 kcal/mol; force RMS diff 4.2 × 10⁻⁵ kcal/mol/Å (RMS force 38.8); VDWAALS 623.8568 exact |
| periodic vs pmemd-pgm (PME, tight) | EELEC diff 4 × 10⁻⁵ of −506,165; force RMS diff 8.8 × 10⁻⁷; induced dipoles, every atom, 1.2 × 10⁻¹⁰ e·Å RMS; VDWAALS 690.1238 exact |
| molecular virial vs sander (ntp=1), vdwmeth=0 | 547.49803 vs 547.498 kcal/mol |
| same, vdwmeth=1 (LJ long-range correction) | 610.73566 vs 610.7357; VDWAALS 669.0446 exact |
| water monomer vs PyRESP | induced dipoles to 8 × 10⁻¹⁰ a.u. |
| 4-water Amber test (`pgm_4wat`) | EELEC −2164.48, VDWAALS 6.7727 (pytest) |

Speed on gpu-2-3 (float64), 512 periodic waters with LJ: energy 25 ms, forces 27 ms, dE/dparams 26 ms,
virial 28 ms, gradient of a force-matching loss 50 ms.

Conventions worth knowing:

- **Coulomb constant.** The model uses CODATA (332.06371 kcal Å/mol). Amber's pGM code uses
  Tinker's 332.05382, 2.98 × 10⁻⁵ lower; comparisons rescale by `units.KE_AMBER_PGM / units.KE`.
- **Energy split.** `perm` and `ind` are whole-system terms; for intermolecular electrostatics vs
  induction comparable with SAPT use `elec_decomposition`.
- **Pressure with the LJ tail.** `strain_derivative` is the exact derivative of the energy.
  `pressure` / `virial_derivative` add the cutoff-impulse part of the continuum tail
  (`PeriodicLJ.tail_virial`), giving the standard P_tail = 2 E_tail / V that Amber uses.
- **Cutoffs.** Periodic pairs beyond rc are masked on the current distance (hard cutoff, as
  Amber); a neighbour list built with `skin` stays valid for small displacements.

## Limits

- LJ is intermolecular only (every intramolecular pair excluded, as for rigid molecules in
  Amber); intramolecular LJ and bonded terms are not implemented.
- The gas-phase induction solve is dense (3n × 3n): fine up to a few thousand atoms.
- Periodic: plain Ewald (no PME); the neighbour list and k-vectors are built once, so no MD
  engine yet (no list updates, integrator or thermostat).
- float64 throughout; single precision is untested.
- `kernels.gd6_jax` jumps by ~4 % at x = 0.15 where it switches to its small-x series (inherited;
  not used by the model yet).
