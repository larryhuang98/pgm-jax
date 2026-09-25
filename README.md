# pGM-JAX

The polarizable Gaussian multipole (pGM) model with Lennard-Jones, in JAX. **Fixed functional
form, everything differentiable:** energies, forces, induced dipoles, polarizabilities and virials
are JAX functions of the coordinates, of the parameters and, for periodic systems, of the box, to
any order. Validated against Amber (sander, pmemd-pgm) and PyRESP.

- **Electrostatics:** Gaussian charges, covalent permanent dipoles and induced Gaussian dipoles,
  with every atom pair interacting (no 1-2/1-3 masking). Levels `elec = "q" | "qp" | "qi" | "qpi"`
  (charges only, + permanent dipoles, charges + induction, full pGM); covalent Gaussian
  quadrupoles (derived and tested; gas phase and fitting only for now).
- **Van der Waals:** Lennard-Jones (Amber form, Lorentz-Berthelot) or GVDW, the Gaussian-density
  van der Waals of pmemd-pgm (`vdw = "lj" | "gvdw"`, Gaussian or Slater repulsion).
- **Bonded term sets** for flexible molecules: Amber/GAFF forms (GAFF import), the explored
  class II and new families, and fast neural bonded terms (a graph network writes the
  parameters of analytic terms once; MD cost = classical terms).
- **Systems:** gas phase (`Model`), periodic (`PeriodicModel`, Ewald, triclinic boxes), and
  **molecular dynamics with JAX-MD** (`pgm_jax.md`: smooth PME, neighbour lists, pmemd-pgm's induction
  solver, rigid or flexible molecules, NVE / Langevin NVT / Monte Carlo NPT, Amber inputs and outputs).
- **Parameterization:** gradients of QM losses (energies, forces, dipoles, ESP) by autodiff;
  gradients of liquid properties (density, heat of vaporization) by fluctuation formulas over MD
  frames; bonded terms for flexible pGM molecules (`pgm_jax.bonded`).

**Getting started with parameterization:** `docs/howto_bonded.md` (bond, angle, torsion terms
for flexible molecules) and `docs/howto_vdw.md` (Lennard-Jones from liquid properties and gas-phase
data). The model options (electrostatics levels, quadrupoles, GVDW, the three bonded
sets) are described, with their checks, in `docs/model_options.md`. The software paper (LaTeX + PDF) is in `paper/`.

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
    --nsteps 100000 --dt 1.0 --cut 9.0 --nfft 48 48 48 --order 6 --ew-coeff 0.4 --vdwmeth 1 \
    --dipole-tol 1e-5 --gamma 2.0 --barostat-interval 100 --report 1000 --traj 1000 --restart 10000
```

Writes `md.log` (energies, translational/rotational temperatures, density, solver iterations,
ns/day), `md.nc` (Amber NetCDF trajectory; cpptraj/VMD/MDTraj read it), `md.rst7` (Amber NetCDF
restart) and `md.chk` (complete state; continue with `--checkpoint md.chk`). From Python:
`Simulation.from_amber(prmtop, coords, settings=MDSettings(...), ensemble=..., ...).run(...)`.

Force or dipole matching on fixed frames (any number of frames; each gets its own rows):

```python
from pgm_jax.md.forcefield import MDSettings, PGMForceField
ff = PGMForceField(system, H, MDSettings(precision="double", dipole_tol=1e-8, differentiable=True))
rows = ff.rows_for(pos, H)                                   # host side, once per frame

def loss(theta):
    res = ff.compute(pos, H, rows, ff.init_induction(), theta)   # nm, kJ/mol/nm, e nm
    return jnp.sum((res.forces - F_ref) ** 2) + w * jnp.sum((res.induction.mu - mu_ref) ** 2)

g = jax.jit(jax.grad(loss))(system.params0)                  # same pytree as the parameters
```

How it works:

- **Electrostatics** exactly as the rest of pGM-JAX, with smooth PME for the reciprocal part
  (Gaussian charges + total dipoles spread with B-spline derivatives; OpenMM's spline conventions).
  Direct-space pairs run over full rows (every pair in both rows): each atom's intramolecular
  partners, then its intermolecular neighbours, compacted every step to the pairs inside the
  cutoff and stored as a structure of arrays; per-atom sums, analytic pair forces
  (grad G_n = -G_{n+1} x), no scatter-adds.
- **Neighbour list** of molecular centres (JAX-MD cell list, float32): rotations never invalidate
  it, so for water it is rebuilt every ~30 steps instead of ~10 for an atom list, at a tenth of
  the cost. Each step every atom keeps the molecules whose centre can bring an atom inside its
  cutoff. Small boxes fall back to an atom list. Overflows are detected, resized and the block
  repeated, never silent.
- **Induced dipoles** as pmemd-pgm's GPU code: cubic extrapolation of the last four converged
  dipoles (`mu4`) with the fused initial residual (the permanent-field sweep is done for
  d = p + guess), Jacobi-preconditioned CG with pmemd-pgm's convergence test
  (max|alpha r| / mean|alpha E_perm| <= `dipole_scf_tol`), peek step (0.65). pmemd-pgm CPU's
  least-squares extrapolation (`--predictor ls`, `dipole_scf_init=3`) and the short-range
  inner-CG preconditioner (`scf_local_niter`) are available; on the GPU a Jacobi iteration is
  cheaper than the iterations they save. Forces are Hellmann-Feynman at the converged dipoles
  (the energy is variational in mu).
- **Rigid molecules** (every molecule; the model has no bonded terms) as JAX-MD rigid bodies:
  NO_SQUISH quaternion integration from JAX-MD `simulate`. Equivalent to SHAKE-rigid water.
- **Thermostat**: BAOAB Langevin with an exact Ornstein-Uhlenbeck step on centre-of-mass and
  body-frame angular momenta. **Barostat**: isotropic Monte Carlo, molecular scaling (Amber
  `barostat=2`), adaptive step.
- **Differentiable forces and dipoles** (`MDSettings(differentiable=True)`): `compute()` returns
  energy, forces and induced dipoles that `jax.grad` / `jax.vjp` can differentiate with respect to
  the parameters, positions and box, e.g. for force or dipole matching. The dipole solve is
  differentiated implicitly: A mu = b(theta) gives mu_bar . dmu = lam . d(b - A mu) with
  A lam = mu_bar, one extra CG with the same (symmetric) operator (`adjoint_tol`, default 1e-6).
  The forward pass is unchanged, so MD runs at the same speed with the option on. Checked
  against finite differences with the dipoles re-solved (1e-7 to 1e-10 relative, float64); mixed
  precision gradients agree with float64 to 1e-6 to 5e-5. A gradient of forces + dipoles costs
  ~15 forward evaluations (30 ms at 12k atoms). Trajectories themselves are not differentiated
  (neighbour-list rebuilds, Monte Carlo moves and Langevin noise); for ensemble averages use
  reweighting, which needs only dU/dtheta per frame.
- **Precision**: `mixed` (default) evaluates pair kernels, PME and CG vectors in float32 and keeps
  positions, energies and dot products in float64; `double` is float64 throughout. float32 matrix
  products are requested at full precision: by default NVIDIA GPUs use TF32 for them, which made
  the dipole spread of the PME 1e-3 inaccurate (7x larger mixed-precision force error).
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
| Same, mixed precision | EELEC 0.05 kcal/mol of 5e5; forces 4e-4 kcal/mol/A RMS (RMS force 36); dipoles 8e-7 e A RMS |
| Analytic row forces vs autodiff; forces and virial vs finite differences | 2e-10 (float64); pytest |
| NVE, mixed, dt 1 fs, 20 ps | drift 0.0001 kT/ns per degree of freedom at dipole_scf_tol 1e-5 (default), 0.02 at 1e-4; dt 2 fs, 1e-4: 0.06 |
| NVT, gamma 2/ps | T 297-299 K; translational = rotational temperature |
| NPT 298 K / 1 bar, settings of the GVDW paper (8 A, PME 48^3 order 8, no LJ tail, tol 1e-4), 2 x 100 ps | density 1.0058 +- 0.011 and 1.0107 +- 0.012 g/cm^3 (Amber: 1.0076 +- 0.012); g_OO first peak 2.79 A, 3.00 (Amber paper: 2.795 A, 2.99) |

### Speed (one RTX PRO 6000 Blackwell; NVT, gamma 1/ps, 9 A cutoff, PME 48^3 per 512 waters, order 6, dt 1 fs)

| System | pgm_jax mixed | pgm_jax double | pmemd.pgm.cuda_SPFP | pmemd-pgm CPU (1 core) |
|---|---|---|---|---|
| 512 waters (1,536 atoms), dipole_scf_tol 1e-5 | 130 ns/day | 47 | 95.6 | 1.7 |
| 4,096 waters (12,288 atoms), dipole_scf_tol 1e-5 | 44 | 9.8 | 68.7 | |
| 512 waters, tol 1e-4 | 148 | | | |
| 4,096 waters, tol 1e-4 | 53 (1.63 ms/step) | | | |

`python scripts/bench_md.py --replicate 2` reproduces a row (pmemd inputs: `scf_local_niter=3`,
the same grid, cutoff and tolerance). What made the step fast (12k atoms, tol 1e-5: 4.5 -> 2.0
ms): the molecular-centre neighbour list (list cost ~1 ms -> 0.03 ms per step); pair rows as a structure of
arrays (the row kernels are memory bound: 5x faster than (N, C, 3) rows); a direct PME dipole
gradient in the CG instead of autodiff; closed-form 3x3 box inverses (jnp.linalg.inv launches LU
factorisations every step); int32 PME indices; mu4 + fused residual (one field sweep fewer). The
remaining gap to pmemd at 12k atoms is hand-written CUDA vs XLA: per CG iteration pmemd runs one
fused kernel per term, XLA several kernels with the loop condition checked on the host.

## Flexible molecules in MD

`pgm_jax.md.flexible` runs molecules with bonded terms fitted by `pgm_jax.bonded`. pGM
electrostatics already includes every intramolecular pair, so a flexible molecule adds

    E_intra = E_bonded(R) + LJ over pairs >= lj_min_sep bonds apart (+ lj14_scale x 1-4 LJ),

exactly the gas-phase model the bonded terms were fitted with. Atoms are integrated individually
(velocity Verlet / BAOAB Langevin; the MC barostat scales molecular centres), molecules are kept
whole across the boundaries, and the neighbour list is still built between molecular centres
(with the molecule's radius plus a margin, checked every block).

```python
from pgm_jax.md.flexible import FlexibleTemplate, FlexibleSimulation, liquid_box
tpl = FlexibleTemplate.from_fit(model, P)          # after fitting pgm_jax.bonded; .save() / .load()
pos, H = liquid_box(tpl, 216, density=0.55)        # dilute start; NPT compresses it
sim = FlexibleSimulation(System([tpl.pgm] * 216), [tpl] * 216, pos, H, MDSettings(),
                         dt=0.0005, ensemble="npt", temperature=298.0)
sim.run(200000, report=2000, prefix="meoh")        # log columns include temp_com and temp_internal
```

`examples/fit_bonded_template.py` (fit + export), `examples/run_flexible_liquid.py` (box, NVT,
NPT) and `examples/flex_methanol_check.py` (forces vs the gas-phase model, NVE, NPT). Only fits
with the engine's model can be exported (pGM with all pairs, no flux, no refitted charges).
216 methanols (1,296 atoms), mixed precision, dt 0.5 fs: 52 ns/day on one GPU (NPT), density
0.789 +- 0.002 g/cm^3 with GAFF LJ and pGM electrostatics (experiment 0.7866); NVE drift below
0.005 kT/ns per degree of freedom (`paper/scripts/flex_methanol.py`).

NPT from a loose start changes the box a lot: the driver rebuilds the neighbour lists when the
volume has drifted by more than 10 % or when a block keeps overflowing (then the block is split).

## Fitting to liquid properties

`scripts/fit_liquid.py` fits Lennard-Jones parameters to the liquid density and heat of
vaporization: each iteration is one NPT simulation; per frame, dU/dtheta is taken by JAX at the
converged induced dipoles (the energy is variational in them, so no derivative of the solve is
needed); the fluctuation formula d<A>/dtheta = <dA/dtheta> - beta cov(A, dU/dtheta) gives the
Jacobian of rho and dHvap; a damped Gauss-Newton step gives the next parameters, and the next
simulation checks the step's prediction.

```bash
python scripts/fit_liquid.py methanol --iters 6                              # flexible, to experiment
python scripts/fit_liquid.py water --start 0.0296,-0.357 --targets 1.0177,8.638   # recovery test
```

Results (paper, section 6.2): water recovers s_R = 1.0005 +- 0.0005, s_eps = 0.985 +- 0.009 from a
start at (1.03, 0.70) in two iterations of 200 s; flexible methanol from GAFF LJ (dHvap 6.97
kcal/mol, 2 kcal/mol too low) to experiment (0.7866 g/cm^3, 8.946 kcal/mol) in four iterations,
at s_R = 1.047, s_eps = 1.510. `--params type` fits one R* and one eps scale per atom type. See
`docs/howto_vdw.md` for per-type parameters and other targets.

## Bonded terms for flexible molecules (`pgm_jax.bonded`)

pGM has no 1-2/1-3/1-4 exclusions, so the valence (bonded) terms of a flexible pGM molecule only
carry what the all-pair electrostatics and LJ beyond 1-4 do not. `pgm_jax.bonded` fits and compares
bonded functional forms for that setting, after Abdullah et al. (arXiv 2504.14398), whose
bonded-only 1-4 treatment with class II couplings it reproduces as one option.

- `topology.py`: internal coordinates and coupling index sets from the bond graph; tying keys
  (per-molecule symmetry classes, or atom environments to a depth for transferable types).
- `terms.py`: a registry of term families, each a JAX function of internal coordinates with its
  own index set, parameters and keys: the paper's class II set (Morse bonds, cosine angles,
  Fourier torsions, bond-bond, bond-angle, angle-angle, torsion-bond, torsion-angle,
  angle-angle-torsion, impropers), topological pair potentials (Urey-Bradley, 1-3 and 1-4
  exponential), a factorised torsion coupling (`torsion_mod`), extended quadratic couplings, a
  pyramidalisation out-of-plane term, a torsion x out-of-plane coupling (`torsion_oop`) and the
  twist of 3-coordinated centres (`twist`, the Winkler-Dunitz angle; fixes amide rotation
  barriers), and electronic-structure-inspired terms: pi-axis conjugation (`conj`), signed-volume
  double wells (`volume`), sigma->sigma* and n->sigma* hyperconjugation (`hc_sigma`, `hc_lone`),
  Coulson hybrid-orbital angles, fixed or self-consistent (`angle_hyb`, `angle_hybsc`),
  distance-only and Gaussian-overlap topological pair terms. A new family is about 20 lines.
- `model.py`: `BondedModel` = bonded families + gas-phase pGM (all pairs, induced dipoles) + LJ
  from `lj_min_sep` bonds; options for a classical control (`elec_exclude`), Amber-like 1-4
  scaling (`elec14_scale`, `lj14_scale`), separate exclusion of the induction (`ind_exclude`),
  charge and covalent-dipole flux (`flux`), learned
  per-type scales of the 1-2/1-3/1-4 permanent pair energies (`escale`), pGM charges and covalent
  dipoles fitted with the bonded terms (`qfit`: typed values; `qbci`: typed bond-charge increments
  on the ESP charges), shared parameters across molecules (`typing="type"`); `esp()` gives the
  molecule's electrostatic potential on a grid.
- Term sets, `terms.SETS`: `"amber"` (harmonic bonds and angles, Amber torsions and
  impropers; `typing="amber"` with GAFF types, `amber.init_from_prmtop` starts from GAFF and matches
  cpptraj), `"protein"` (the Amber forms + `cmap`, a Fourier phi/psi correction per residue, found
  from the bond graph), `"explore"` (the class II set; add any registry family) and `"nn"` (`nn/`:
  a graph network predicts per-instance parameters of a basis set from the bond graph and pGM
  parameters, with residue context for the backbone maps; frozen for MD, so an MD step costs the
  same as the classical terms; `NNBonded.save` / `load` / `prepare` apply a trained network to new
  molecules). `amber.export_bonded` writes fitted protein-set parameters into a prmtop (checked
  against sander: `scripts/bonded/check_export.py`).
- `fit.py`: energy + force (+ dipole, + QM ESP restraint) loss with per-molecule offsets, L-BFGS
  (after Adam for the neural set) on everything (reference values, force constants, exponents, flux, charges) with JAX gradients,
  optional L1.
- `bench.py`: the paper's metrics, force-field-relaxed torsion scans.
- `scripts/bonded/`: molecule set (RDKit), MACE-OFF sampling (MD at 500/298 K, relaxed scans),
  DFT labels (psi4, wB97M-D3(BJ)/def2-TZVPPD, Slurm arrays), pGM parameters (ESP + py_resp), the
  experiments, the alanine dipeptide phi/psi surface (`x6_dipeptide.py`), gas-phase MD with the
  fitted force fields (`md_check.py`) and the report (`reports/bonded/`).

```python
from pgm_jax.bonded.data import frames, mol_spec
from pgm_jax.bonded.fit import Fitter
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.bonded import terms as T

spec = mol_spec("formic_acid")                                    # topology + pGM parameters + minimum
model = BondedModel([spec], BondedSettings(families=T.PAPER))     # pGM all pairs, LJ 1-5+
fit = Fitter(model, {0: {"train": frames("formic_acid", "train500"), "test": frames("formic_acid", "test298")}})
P = fit.fit(model.init_params())
print(fit.metrics(P, "test"))                                     # energy / force MAE (kcal/mol, /A), dipole RMSE (D)
```

Findings of the first study are in `reports/bonded/README.md`.

## Layout

| Path | What is in it |
|---|---|
| `pgm_jax/system.py` | `Molecule`, `ParamTable` (tying), `System` (topology + index arrays, `expand(params)`) |
| `pgm_jax/units.py` | units (nm, e, kJ/mol) and constants, incl. Amber's pGM Coulomb constant |
| `pgm_jax/kernels.py` | Gaussian pair kernels: Coulomb erf(b r)/r, overlap, C6 dampings (gd6, tt6) |
| `pgm_jax/channels.py` | `ElecChannel`, `elec_decomposition` (SAPT-like elst/ind), `molecular_polarizability` |
| `pgm_jax/lj.py` | `LJChannel` (gas phase), `PeriodicLJ` (cutoff, optional long-range correction) |
| `pgm_jax/vdw.py` | GVDW: `gvdw_pair`, `GVDWChannel`, `PeriodicGVDW`, `set_gvdw`, pmemd conversions, pGM3P-GVDW water |
| `pgm_jax/multipole.py` | Gaussian quadrupoles: derivation, covalent quadrupole basis (`with_quadrupoles`), pair and field kernels |
| `pgm_jax/options.py` | electrostatics levels and vdW forms shared by all models |
| `pgm_jax/ewald.py` | `PeriodicPGM`: Ewald with integer image and k-vector lists, so the box is a JAX input; neutralising background |
| `pgm_jax/periodic.py` | `PeriodicModel` (elec + LJ, shared neighbour list): energy, forces, `strain_derivative`, `pressure` |
| `pgm_jax/solver.py` | dense induction solve; `variational`; Newton solver with implicit differentiation |
| `pgm_jax/model.py` | gas-phase `Model`: energies, forces, batching, n-body energies (compiled once per topology) |
| `pgm_jax/param.py` | Amber pGM prmtop reader (incl. LJ, bonds, masses), JSON save/load, py_resp `.chg` + pGM-pol table, atom mapping |
| `pgm_jax/md/` | MD engine: `forcefield.py` (PME + direct rows + induction solver), `pme.py`, `kernels.py`, `neighbors.py` (JAX-MD lists), `rigid.py` (JAX-MD rigid bodies), `integrate.py`, `simulation.py`, `io.py` (Amber NetCDF), `box.py` |
| `pgm_jax/md/flexible.py` | flexible molecules in MD: `FlexibleTemplate` (bonded fit -> MD), `FlexibleSimulation`, `liquid_box` |
| `scripts/fit_liquid.py` | LJ from liquid density + heat of vaporization (ensemble gradients, Gauss-Newton) |
| `examples/`, `docs/` | fit-and-run examples; how-tos for bonded and van der Waals parameterization |
| `paper/` | the pGM-JAX paper (LaTeX, PDF, figure data and scripts) |
| `pgm_jax/bonded/` | bonded terms for flexible pGM molecules: `topology.py` (incl. peptide backbone and residues from the graph), `terms/` (registry and `SETS`: `core`, `classical`, `class2`, `explore`, `cmap`), `model.py` (`BondedTerms`, `BondedModel`), `fit.py`, `bench.py`, `data.py`, `molecules.py`, `amber.py` (GAFF / ff19SB import, prmtop export), `nn/` (neural bonded terms: `features`, `layers`, `instances`, `model`) |
| `pgm_jax/prmtop.py` | Amber prmtop as raw sections: read, edit, write (unknown sections such as pGM's kept verbatim) |
| `scripts/bonded/` | the bonded study: sampling, DFT labels, pGM parameters, experiments, report |
| `scripts/run_md.py` | MD from an Amber prmtop + inpcrd/rst7 (Amber-style options) |
| `scripts/bench_md.py`, `scripts/pgm_supercell.py` | MD speed benchmark; replicate a pGM prmtop for larger systems |
| `tests/` | `pytest -q`: 76 tests, incl. finite-difference checks of every derivative, the MD engine and the model options |
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

- LJ is intermolecular only in `Model`, `PeriodicModel` and the rigid-molecule MD engine (every
  intramolecular pair excluded, as for rigid molecules in Amber). Flexible molecules
  (`pgm_jax.md.flexible`) add bonded terms and intramolecular LJ from 1-5 on; they have no bond
  constraints yet (dt 0.5 fs).
- `fit_liquid.py` does not yet differentiate <U_gas> for molecules with intramolecular LJ pairs.
- Quadrupoles are in the gas phase and in bonded fitting, not yet in Ewald/PME or MD (templates
  with quadrupoles are refused there); no fitted quadrupole values yet.
- GVDW per atom type (geometric A and C6, arithmetic b) generalises pmemd-pgm's single global
  set for LJ-bearing pairs; identical for pGM3P water.
- The gas-phase induction solve is dense (3n × 3n): fine up to a few thousand atoms.
- `PeriodicModel` uses plain Ewald with a neighbour list and k-vectors built once (for
  single points and gradients); MD uses `pgm_jax.md` (smooth PME, list updates, integrators).
- Outside `pgm_jax.md` everything is float64; single precision is used (and validated) only in MD's mixed mode.
- `kernels.gd6_jax` jumps by ~4 % at x = 0.15 where it switches to its small-x series (inherited;
  not used by the model yet).
