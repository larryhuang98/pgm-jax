# pGM-JAX

The polarizable Gaussian multipole (pGM) model with Lennard-Jones, in JAX. **Fixed functional
form, everything differentiable:** energies, forces, induced dipoles, polarizabilities and virials
are JAX functions of the coordinates, of the parameters and, for periodic systems, of the box, to
any order. Validated against Amber (sander, pmemd-pgm) and PyRESP.

- **Electrostatics:** Gaussian charges, covalent permanent dipoles and induced Gaussian dipoles,
  with every atom pair interacting (no 1-2/1-3 masking).
- **Van der Waals:** Lennard-Jones between molecules (Amber form, Lorentz-Berthelot).
- **Systems:** gas phase (`Model`), periodic (`PeriodicModel`, Ewald, triclinic boxes), and
  **molecular dynamics with JAX-MD** (`pgm_jax.md`: smooth PME, neighbour lists, pmemd-pgm's induction
  solver, rigid molecules, NVE / Langevin NVT / Monte Carlo NPT, Amber inputs and outputs).

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

## Molecular dynamics (JAX-MD)

```bash
python scripts/run_md.py -p water.prmtop -c water.rst7 -o md --ensemble npt --temp 298 --press 1 \
    --nsteps 100000 --dt 1.0 --cut 9.0 --nfft 48 48 48 --order 8 --ew-coeff 0.4 --vdwmeth 1 \
    --dipole-tol 1e-5 --gamma 2.0 --barostat-interval 100 --report 1000 --traj 1000 --restart 10000
```

Writes `md.log` (energies, translational/rotational temperatures, density, solver iterations,
ns/day), `md.nc` (Amber NetCDF trajectory; cpptraj/VMD/MDTraj read it), `md.rst7` (Amber NetCDF
restart) and `md.chk` (complete state; continue with `--checkpoint md.chk`). From Python:
`Simulation.from_amber(prmtop, coords, settings=MDSettings(...), ensemble=..., ...).run(...)`.

How it works:

- **Electrostatics** exactly as the rest of pGM-JAX, with smooth PME for the reciprocal part
  (Gaussian charges + total dipoles spread with B-spline derivatives; OpenMM's spline conventions).
  Direct-space pairs run over the rows of a dense, full JAX-MD neighbour list, compacted every step to
  the pairs inside the cutoff; per-atom sums, analytic pair forces (grad G_n = -G_{n+1} x), no
  scatter-adds.
- **Induced dipoles** as pmemd-pgm: permanent-field right-hand side, multi-order least-squares
  extrapolation (`dipole_scf_init=3`, order 3, 2 steps), preconditioned CG with pmemd-pgm's
  convergence test (max|alpha r| / mean|alpha E_perm| <= `dipole_scf_tol`), peek step (0.65).
  The short-range inner-CG preconditioner (`scf_local_niter`) is implemented but off by default:
  on the GPU a Jacobi-preconditioned iteration is cheaper than the saved iterations.
  Forces are Hellmann-Feynman at the converged dipoles (the energy is variational in mu).
- **Rigid molecules** (every molecule; the model has no bonded terms) as JAX-MD rigid bodies:
  NO_SQUISH quaternion integration from JAX-MD `simulate`. Equivalent to SHAKE-rigid water.
- **Thermostat**: BAOAB Langevin with an exact Ornstein-Uhlenbeck step on centre-of-mass and
  body-frame angular momenta. **Barostat**: isotropic Monte Carlo, molecular scaling (Amber
  `barostat=2`), adaptive step.
- **Precision**: `mixed` (default) evaluates pair kernels, PME and CG vectors in float32 and keeps
  positions, energies and dot products in float64; `double` is float64 throughout.
- **Boxes**: any reduced triclinic box, including Amber's truncated octahedron (exact minimum image;
  cutoff + skin must be below half the smallest box height).

Two JAX-MD 0.2.29 issues are worked around (and would affect other users of those functions):
its rigid-body Langevin step draws the quaternion-momentum noise with a diagonal covariance, which
leaves rotations 15-25 K too cold for water (replaced by the exact O step above); and its
neighbour list sets the MALFORMED_BOX error bit for every *valid* box matrix (inverted predicate),
so that bit is ignored. `pgm_jax.md` imports only JAX-MD's simulation core (space, partition,
simulate, rigid_body), because its top-level import pulls in flax, which does not import with
JAX 0.11.

### Validation (512 pGM3P-25 waters)

| Check | Result |
|---|---|
| Single point vs pmemd-pgm (PME 72^3, order 8, float64) | EELEC 4e-5 kcal/mol; forces 9e-7 kcal/mol/A RMS; induced dipoles 2e-11 e A RMS; VDW exact |
| Same, mixed precision | EELEC 0.05-0.07 kcal/mol of 5e5; forces 3e-3 kcal/mol/A RMS (RMS force 36) |
| Analytic row forces vs autodiff; forces and virial vs finite differences | 2e-10 (float64); pytest |
| NVE, dt 1 fs, 10-20 ps | drift 0.001-0.007 kT/ns per degree of freedom (mixed and double, tol 1e-4 to 1e-6); 2 fs also stable |
| NVT, gamma 2/ps | T 297-299 K; translational = rotational temperature |
| NPT 298 K / 1 bar, settings of the GVDW paper (8 A, PME 48^3 order 8, no LJ tail, tol 1e-4), 2 x 100 ps | density 1.0058 +- 0.011 and 1.0107 +- 0.012 g/cm^3 (Amber: 1.0076 +- 0.012); g_OO first peak 2.79 A, 3.00 (Amber paper: 2.795 A, 2.99) |

### Speed (one RTX PRO 6000 Blackwell; NVT, 9 A cutoff, PME order 6, dipole_scf_tol 1e-5, dt 1 fs)

| System | pgm_jax mixed | pgm_jax double | pmemd.pgm.cuda_SPFP | pmemd-pgm CPU (1 core) |
|---|---|---|---|---|
| 512 waters (1,536 atoms) | 97.8 ns/day | 33.9 | 95.6 | 1.7 |
| 4,096 waters (12,288 atoms) | 19.4 (21.9 with skin 2 A) | 6.8 | 68.7 | |

At 1.5k atoms JAX-MD runs as fast as pmemd's GPU code; at 12k atoms pmemd is ~3x faster (its
kernels are hand-written CUDA; ours are XLA-compiled). Most of our step is the dipole CG (6-7
iterations of a row product and a PME convolution) and JAX-MD's cell-list rebuild (float32; a
larger skin helps on large systems).

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
| `pgm_jax/md/` | MD engine: `forcefield.py` (PME + direct rows + induction solver), `pme.py`, `kernels.py`, `neighbors.py` (JAX-MD lists), `rigid.py` (JAX-MD rigid bodies), `integrate.py`, `simulation.py`, `io.py` (Amber NetCDF), `box.py` |
| `scripts/run_md.py` | MD from an Amber prmtop + inpcrd/rst7 (Amber-style options) |
| `scripts/bench_md.py`, `scripts/pgm_supercell.py` | MD speed benchmark; replicate a pGM prmtop for larger systems |
| `tests/` | `pytest -q`: 34 tests, incl. finite-difference checks of every derivative and the MD engine |
| `scripts/validate_amber.py` | comparison with sander / pmemd-pgm / PyRESP (`compare`, `pyresp`, `virial`) |
| `scripts/bench.py` | timings on the current device |
| `validation/` | Amber reference runs (inputs + outputs) and `validate_amber.json` |

## Running on rayl8

The login node's glibc is too old for jaxlib: run on a GPU node (all nodes share /home8).

```bash
ssh gpu-2-3
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pgmjax   # clone of evoff + jax-md 0.2.29
cd ~/project/pGM-JAX
pytest -q                                    # ~2 min
python scripts/validate_amber.py compare     # gas phase + periodic vs Amber, ~1 min
python scripts/validate_amber.py pyresp      # monomer vs PyRESP
python scripts/validate_amber.py virial      # molecular virial vs sander (ntp=1)
python scripts/bench.py
python scripts/bench_md.py --replicate 2     # MD speed, 4096 waters
```

The `pgmjax` environment was made with `conda create -n pgmjax --clone evoff` plus
`pip install --no-index --find-links ~/project/pGM-JAX-wheels jax-md==0.2.29` (the GPU nodes have no
internet; the wheels were downloaded elsewhere). evoff's environment is untouched.

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
- Outside `pgm_jax.md` everything is float64; single precision is used (and validated) only in MD's mixed mode.
- `kernels.gd6_jax` jumps by ~4 % at x = 0.15 where it switches to its small-x series (inherited;
  not used by the model yet).
