# pGM + ML bonded protein force field: code structure and usage

This note is the map of the code for the protein force-field project: pGM electrostatics (and
LJ or GVDW) for the nonbonded part, bonded terms from Amber forms + CMAP whose parameters come
from typed fits or from the neural bonded model, MD with the pgm_jax engine or exported to
Amber. Everything below is in the repository, and each item has at least one test.

## Layers

| Layer | Module | What it does |
|---|---|---|
| Residue chemistry | `protein/residues.py` | bond orders Amber topologies do not carry; terminal residue keys (NMET, CGLY) |
| pGM parameters | `protein/library.py` | `ResidueLibrary`: q, alpha, radius by residue and atom name; covalent dipoles incl. "-C" / "+N" partners (JSON) |
| System | `protein/amber.py` | `load_amber`: a tleap system -> molecules (protein, water, ions), bonded-model inputs; `amber_template` |
| Bonded forms | `bonded/terms/` | `SETS["protein"]` = Amber bonds, angles, torsions, impropers + `cmap` (Fourier phi/psi map; `cmap6`: order 6) |
| Topology | `bonded/topology.py` | backbone phi/psi quintuples and residues from the bond graph; linear-time for proteins |
| Neural bonded | `bonded/nn/` | `NNBConfig`, `NNBonded` (`for_molecules`, `prepare`, `save` / `load`); residue context for the backbone map |
| Fitting | `bonded/model.py`, `bonded/fit.py` | `BondedModel` (bonded + gas-phase pGM), `Fitter` (Adam + L-BFGS) |
| Export | `bonded/amber.py`, `prmtop.py` | `export_bonded`: per-instance parameters + CMAP into a prmtop (pmemd-pgm / sander) |
| pmemd-pgm | `protein/pmemd.py` | `write_pgm_prmtop`: the engine's whole model (pGM, LJ, exclusions, 1-4, bonded, masses) as a pmemd-pgm prmtop; `pmemd_mdin`, `pmemd_grid` |
| MD topology | `md/topology.py` | neighbour-list groups (heavy-atom groups), special pairs with vdW weights, constraints |
| Constraints | `md/constraints.py` | SHAKE / RATTLE per cluster, vectorised; hydrogen mass repartitioning |
| MD | `md/flexible.py` | `FlexibleTemplate` (`from_fit`, `from_network`), `RigidTemplate`, `FlexibleSimulation` (g-BAOAB, `minimize`) |
| Top-down | `ensemble.py` | `Reweighting` (averages, n_eff, chi2 and gradients), Karplus J couplings, phi/psi regions |

Design rules that keep it reusable:

- **Parameter shapes live with the network.** `NNBonded` keeps its configuration and
  vocabulary (key skeletons, slots, typed-table keys) together with its weights. A trained
  network applies to any new molecule through `prepare`, and a molecule with unseen term kinds
  is refused rather than silently mispredicted.
- **Only frozen tables reach MD.** `FlexibleTemplate.from_fit` / `from_network` evaluate the
  network once, so the MD step contains only the analytic terms.
- **The bonded half is separate.** `BondedTerms` holds just the bonded part (any size, used by
  MD templates). `BondedModel` adds the gas-phase pGM for fitting on fragments.
- **The pair structure is one object.** `MDTopology` is the only place that knows which pairs
  are excluded or scaled. The force field and the neighbour lists read it, and
  `MDTopology.rigid` reproduces the rigid-water engine exactly.
- **Amber files are edited, not regenerated.** `Prmtop` keeps every section verbatim, so pGM
  sections survive and `export_bonded` only replaces the bonded ones.

## From a PDB to MD

```bash
python scripts/protein/build_amber.py 1ubq.pdb runs/protein/ubq --buffer 10   # tleap: ff19SB topology, TIP3P box
```

```python
from pgm_jax.protein import ResidueLibrary, amber_template, load_amber
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate
from pgm_jax.md.forcefield import MDSettings

asys = load_amber("runs/protein/ubq.prmtop", "runs/protein/ubq.inpcrd",
                  electrostatics=ResidueLibrary.load("pgm_residues.json"))   # or "placeholder"
k = [i for i, m in enumerate(asys.molecules) if m.kind == "protein"][0]
tpl = amber_template(asys.molecules[k], "runs/protein/ubq.prmtop")          # ff19SB-form bonded + CMAP
# or, with a trained network:  net, P = NNBonded.load("nnb.pkl")
#                              tpl = FlexibleTemplate.from_network(net, P, asys.molecules[k].spec, lj14_scale=0.5)
sim = FlexibleSimulation(asys.system(), asys.templates({k: tpl}), asys.system_positions(), asys.box,
                         MDSettings(dipole_tol=1e-4), dt=0.002, ensemble="npt",
                         constraints="h-bonds", hmr=3.024,
                         thermostat="bussi", tau_t=1.0)   # fastest with pGM (docs/thermostat_ideas.md)
sim.minimize(300)            # after minimization, equilibrate with thermostat="langevin" (faster warm-up)
sim.run(500000, report=5000, traj=5000, prefix="ubq")
```

## Production MD with pmemd-pgm

pmemd.pgm.cuda runs the engine's model 3 to 7 times faster from about 16k atoms on (table below).
`write_pgm_prmtop` writes that model into the system's tleap prmtop and `pmemd_mdin` the matching
nonbonded settings, so both codes simulate the same system:

```python
from pgm_jax.protein import pmemd_grid, pmemd_mdin, write_pgm_prmtop
templates = asys.templates({k: tpl})                       # exactly what FlexibleSimulation gets
write_pgm_prmtop(asys, "ubq_pgm.prmtop", templates, hmr=3.024)
st = MDSettings(pme_grid=pmemd_grid(asys.box))             # a PME grid both codes accept
open("min.in", "w").write(pmemd_mdin(st, asys.box, maxcyc=500))                      # tleap clashes
open("heat.in", "w").write(pmemd_mdin(st, asys.box, nstlim=4000, dt=0.0005, tempi=0.0))
open("md.in", "w").write(pmemd_mdin(st, asys.box, nstlim=500000, dt=0.002, irest=1))
```

```bash
python scripts/protein/write_pgm_prmtop.py ubq.prmtop ubq.inpcrd ubq_pgm.prmtop --library lib.json --hmr 3.024 --mdin ubq
pmemd.pgm.cuda_SPFP -O -i ubq.min.in -p ubq_pgm.prmtop -c ubq.inpcrd -o min.out -r min.rst7   # then ubq.heat.in, ubq.md.in
```

What is written (details in `protein/pmemd.py`):

- **pGM**: the `POL_GAUSS_*` sections with the engine's values (`system.expand(params)`, so fitted
  parameter tables too). pgm_jax's covalent-dipole convention is pmemd-pgm's.
- **Van der Waals**: LJ types and tables rebuilt from the engine's per-atom R* and eps. The
  exclusion list holds the pairs without regular LJ in the engine: every pair of a rigid molecule,
  pairs fewer than `lj_min_sep` bonds apart in a flexible one. pmemd-pgm keeps excluded pairs in
  its pGM electrostatics, so neither code has electrostatic exclusions.
- **1-4 pairs**: one dihedral with the 1-4 flag per pair 3 bonds apart. pmemd-pgm drops the
  separate 1-4 electrostatics and hard-codes SCNB = 1. `lj14_scale = 1/2` (Amber's, `amber_template`)
  is therefore written as the CHARMM-type 1-4 LJ tables (`LENNARD_JONES_14_*` = lj14_scale x LJ).
  pmemd reads them, on the CPU and the GPU, when `FORCE_FIELD_TYPE` names CHARMM. The other
  CHARMM sections are present and empty.
- **Bonded terms**: `export_bonded` (the Fourier CMAP tabulated on Amber's 24 x 24 grid), or the
  input's own terms with `templates=None` (e.g. ff19SB with its CMAP grids).
- **Rigid water**: the SHAKE / SETTLE lengths are the RigidTemplate distances. A template takes
  the geometry of the first water, which in tleap's boxes differs from TIP3P's 0.9572 / 1.5136 A
  by up to 3e-4 A. **Masses**: `hmr` as in `FlexibleSimulation`.

Single points: pmemd-pgm on the written file against the engine at the same coordinates
(float64, dipole tolerance 1e-9; cut 9 A, Ewald coefficient 0.4 A^-1, PME spacing <= 0.8 A, order
6; placeholder electrostatics, `amber_template`; `scripts/protein/check_pgm_prmtop.py`,
`validation/check_pgm_prmtop.json`). Energies are in kcal/mol, forces in kcal/mol/A. The first
EELEC and force numbers are with each code's own PME. The second ones are with pmemd's
influence-function factor in the engine (see below).

| System (atoms) | pmemd | EELEC | EELEC diff | BOND, ANGLE, DIHED, 1-4 NB, VDWAALS: max diff | CMAP diff | max force diff (backbone-map atoms excluded) | max force diff, backbone-map atoms |
|---|---|---|---|---|---|---|---|
| 512 pGM3P-25 waters, round trip (1,536) | CPU | -506,165 | 0.084 / 4e-5 | 7e-6 | - | 1e-3 / 4e-6 | - |
| 4,096 pGM waters, round trip (12,288) | CPU, DPFP | -2,223,199 | 1.17 / 3e-5 | 3e-5 | - | 2e-3 / 1e-6 | - |
| same | SPFP | | 1.25 / 0.073 | 2e-3 | - | 2e-3 / 8e-4 | - |
| ACE-ALA-SER-NME in TIP3P, NaCl (791) | CPU | -31,607 | 7e-3 / 2e-5 | 4e-5 | 1e-5 | 1e-4 / 2e-6 | 2e-3 |
| Trp-cage in TIP3P (6,215) | CPU, DPFP | -248,643 | 0.045 / 9e-5 | 5e-5 | 3e-3 | 1e-4 / 2e-5 | 0.047 |
| same | SPFP | | 0.051 / 6e-3 | 2e-3 | 3e-3 | 5e-4 / 5e-4 | 0.047 |

pmemd prints energies to 1e-4 kcal/mol. The round trips also run pmemd on the original pGM
prmtops, with identical energies. The remaining differences:

- **Coulomb constant.** pmemd-pgm uses Tinker's 332.05382 kcal A/mol, 2.98e-5 below the engine's
  CODATA value, so its electrostatic energies and forces are the engine's times 0.9999702. A
  prmtop cannot change this. The comparison gives the engine pmemd-pgm's constant (charges and
  covalent dipoles times its square root).
- **PME influence function.** pmemd multiplies the Euler-spline influence function by
  lambda(m)^2 (`factor_lambda` in pme_recip_dat.F90), and the engine does not. At 0.8 A and order
  6 the electrostatic energies differ by 2-5e-7 relative and the forces by up to 2e-3. With the
  factor in the engine (`check_pgm_prmtop.py --amber-lambda`) or with a fine grid (0.4 A,
  order 8: `--tight`), EELEC agrees to 1e-4 kcal/mol (at most 6e-10 relative) and the forces
  to 2e-5.
- **CMAP.** pmemd interpolates the 24 x 24 tabulation of the Fourier map bicubically. For
  Trp-cage the CMAP energy differs by 3e-3 kcal/mol and the forces on backbone atoms by up to 0.05
  kcal/mol/A (CMAP forces reach 8.6). Amber's format fixes the grid (pmemd allocates 24 x 24).
- **SPFP** adds about 3e-8 relative to EELEC and 5-8e-4 kcal/mol/A to the forces.
- pmemd's CPU code removes the net PME force (`netfrc=1`); its GPU code and the engine do not
  (the check sets `netfrc=0`). pmemd.pgm.cuda also needs at least three neighbour-list cells per box
  dimension (the 512-water box and the small peptide box run on the CPU only at 9 A), PME orders
  4 to 6, and grids that are multiples of 4 with factors 2, 3, 5 (`pmemd_grid`).

MD with pmemd.pgm.cuda_SPFP (placeholder electrostatics, `amber_template`; 500 minimisation steps,
2 ps at 0.5 fs heating from 0 K, then NVT at 298 K, Langevin 1/ps, dt 2 fs, SHAKE on X-H bonds,
rigid water, no HMR, 9 A, PME <= 0.8 A order 6, dipole_scf_tol 1e-5; `check_pgm_prmtop.py md`).
The engine column is the table under "Speed and size": the same settings, with HMR.

| System | Atoms | pmemd.pgm.cuda ms/step | ns/day | engine ms/step | pmemd speed-up |
|---|---|---|---|---|---|
| Trp-cage 1L2Y | 6,215 | 1.41-1.46 | 119-122 | 1.72 | 1.2 |
| Ubiquitin 1UBQ | 15,955 | 1.77 | 97 | 5.06 | 2.9 |
| DHFR 1RX2 | 25,780 | 2.51 | 69 | 7.68 | 3.1 |
| MBP 1OMP | 46,329 | 3.27 | 53 | 22.0 | 6.7 |

Trp-cage, 200 ps from three seeds in each code: pmemd holds 297.5-297.8 K (sd 3.5-3.9 K) with no
SHAKE failures. The CA RMSD from the NMR model, averaged over the second 100 ps, is 1.9, 2.3 and 2.9
A (maximum 3.4 A). The engine gives 1.6, 2.0 and 3.0 A (maximum 3.3 A) with the same model and
protocol (`check_pgm_prmtop.py md-engine`, 87-95 ns/day). Both codes show the same 2-3 A drift, a
property of this test model (placeholder electrostatics, the order-3 Fourier version of ff19SB's
CMAP), not of either code. pmemd's `tempi` draws velocities for every degree of freedom before
SHAKE, so a constrained system starts about 1.5x too hot (440 K for Trp-cage). Three pmemd runs
started that way drifted to 3.3-4.3 A, so `md` heats from 0 K instead.

## Training the neural bonded model for proteins

```python
from pgm_jax.bonded import terms as T
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.bonded.fit import Fitter

st = BondedSettings(families=("nnb",), nn_basis=T.PROTEIN, lj14_scale=0.5)     # Amber forms + CMAP, residue context on
model = BondedModel(fragment_specs, st)                     # capped dipeptides / tripeptides with pGM parameters
P = Fitter(model, data).fit(model.init_params(), adam_steps=3000, maxiter=4000)
model.nnb.save("nnb.pkl", P["nnb"])
```

## Top-down refinement by reweighting

Sample with the current parameters, then fit to solution data (J couplings, helicities) by
reweighting the saved frames. The gradient reaches the CMAP coefficients or the network weights.
Resample when `n_eff` drops.

```python
from pgm_jax.ensemble import KARPLUS, Reweighting, backbone_torsions, karplus
from pgm_jax.md.io import read_trajectory

frames, _, _ = read_trajectory("prod.nc", atoms=protein_atoms)          # Amber NetCDF, A
X = frames * 0.1                                                         # nm
# theta: the parameters being refined (here the CMAP coefficients); terms: the BondedTerms of the protein
rw = Reweighting(lambda th, R: terms.bonded_energy(0, R, with_cmap(P, th)), th0, X, temperature=298.0)
phi, psi = backbone_torsions(X, terms.mols[0].top)                       # (frames, residues), IUPAC sign
J = karplus(phi, *KARPLUS["3J_HNHA_Vogeli2007"])
loss, grad = rw.chi2_and_grad(th, [(J, J_exp, 0.5)])                     # (values, target, sigma)
print(rw.n_eff(th))                                                      # resample when this drops
```

## Checks and numbers

| Check | Result |
|---|---|
| Export to prmtop, then sander (ACE-ALA-ALA-NME, per-instance parameters) | BOND, ANGLE, DIHED, CMAP agree to < 3e-5 kcal/mol; 1-4 VDW / EEL unchanged (`validation/check_export.json`, `scripts/bonded/check_export.py`) |
| ff19SB CMAP imported as a Fourier map (same peptide, 2 CMAP terms) | approximate: off by 1.5 kcal/mol (order 3) / 0.46 kcal/mol (order 6, `cmap6`) at that conformation |
| Alanine dipeptide phi/psi surface, grid-only fit (MAE) | Amber forms 0.67 kcal/mol; + CMAP 0.55 |
| 29-atom peptide in heavy-atom groups: MD forces vs gradient of the gas-phase model | max error < 0.2 % of the rms force (`tests/test_md_macro.py`) |
| Rigid water by constraints vs rigid bodies | same energy (1e-8 relative); the rigid-body engine is unchanged (2.05 ms/step, 12k atoms) |
| SHAKE / RATTLE | exact to 1e-12; 4096 waters: 0.042 ms per SHAKE |
| Peptide + water, X-H constraints + HMR, 2 fs; solvated ACE-ALA-SER-NME | stable (tests) |
| pmemd-pgm prmtop of the engine's model, pmemd.pgm single points (water, peptide, Trp-cage; CPU, DPFP, SPFP) | all terms agree to pmemd's print precision; forces to 2e-5 kcal/mol/A (with pmemd's PME factor in the engine) except the CMAP interpolation (section "Production MD with pmemd-pgm"; `tests/test_pgm_prmtop.py`) |

Benchmark: `python scripts/protein/bench_protein.py sys.prmtop sys.inpcrd [--library lib.json] [--tol 1e-4]`.

## Speed and size

One RTX PRO 6000 Blackwell (96 GB), mixed precision, NVT, dt 2 fs, X-H constraints + HMR, rigid
water, 9 A cutoff, PME spacing 0.8 A order 6, dipole tol 1e-5, placeholder electrostatics,
rectangular TIP3P box (10 A buffer unless noted). `scripts/protein/bench_protein.py`.

| System | Atoms (protein) | ms/step | ns/day | CG iterations | GPU memory peak | Setup + minimise/compile |
|---|---|---|---|---|---|---|
| Trp-cage 1L2Y | 6,215 (304) | 1.72 | 100 | 12 | 0.3 GiB | 12 s + 12 s |
| Ubiquitin 1UBQ | 15,955 (1,231) | 5.06 | 34 (41 at tol 1e-4) | 12 (9) | 1.0 GiB | 16 s + 16 s |
| DHFR 1RX2 | 25,780 (2,489) | 7.68 | 22.5 | 14 | 2.5 GiB | 25 s + 20 s |
| MBP 1OMP | 46,329 (5,737) | 22.0 | 7.9 | 14 | 4.3 GiB | 49 s + 30 s |
| MBP, 20 A buffer | 93,180 | 51.8 | 3.3 | 14 | 9.6 GiB | 83 s + 57 s |
| MBP, 32 A buffer | 180,213 | 105.5 | 1.6 | 15 | 11.3 GiB | 217 s + 100 s |

Pure pGM water with the same engine (constraints, dt 2 fs, `scripts/bench_md.py --engine
constraints --grid 36`): 12k atoms 2.14 ms/step, 41k 11.9, 98k 35.0, with 6-7 CG iterations.

- Memory is not the limit: 180k atoms use about 11 GiB. The limit is time per step.
- The cost per atom grows from about 0.3 ms per 1000 atoms (up to 26k atoms) to 0.5-0.6
  (46k and above). Pure water shows the same growth (0.17 -> 0.36), so it is the engine, not the
  protein terms (probably the pair rows no longer staying in cache; not yet profiled).
- Protein boxes need twice the CG iterations of pure water at the same time step (12-15 against
  6-7). That accounts for most of the extra cost per atom. Whether the cause is the placeholder
  parameters or the predictor is still open.


### What limits the speed (measured, ubiquitin unless noted)

- **The direct-space dipole matvec is memory bound.** It stores 24 bytes per pair (index,
  displacement, two kernels) for about 390 pairs per atom. This fits the 128 MB L2 cache of the
  card up to about 14k atoms. In pure water, 12k atoms take 0.04 ms per matvec and 98k atoms
  2.2 ms. The ubiquitin box needs 150 MB, and one matvec takes 0.45 ms against 0.065 ms for the
  PME part. For comparison, pmemd-pgm's direct field sweep at 98k atoms takes 0.23 ms.
- **Why the CG needs 12-15 iterations.** Each iteration only halves the residual (about 4x in
  pure water). The slowest components sit on aromatic ring carbons and on the arginine CZ. The
  operator depends only on alpha, R and the geometry, not on the charges, so this is a property
  of the pGM-pol parameters, not of the placeholder charges. At 2 fs the starting residual is
  largest on charged side chains (Arg NH, Lys NZ): 0.13 against 0.036 on water.
  `scripts/protein/cg_diag.py` prints this analysis.
- **Shorter real-space cutoff with a larger Ewald coefficient.** Cutoff 0.7 nm, beta
  5.14 nm^-1, grid 0.062 nm. This halves the pair rows, so the matvec fits in cache again.
  Ubiquitin: 34 -> 55 ns/day. DHFR (26k atoms): 22.5 -> 28. Trp-cage: unchanged (already in
  cache). Force error against a tight reference, electrostatics only: 6.5e-5 relative rms
  (2.8e-5 at the current 0.9 nm / 4.0 / 0.08 nm). In this test LJ was cut at 0.7 nm as well;
  production needs a separate electrostatics cutoff.
- **dipole tol 1e-4 on top:** 64 ns/day, 1.9x the baseline.
- **Local preconditioner** (`local_niter` 2, 0.3 nm): 13 -> 7 iterations, but only 7 % faster.
  The inner sweeps cost about what they save, as in pmemd-pgm.
- **Thermostat** (`thermostat="bussi"`): per-atom Langevin noise spoils the dipole predictor.
  Ubiquitin at 2 fs: 34.2 -> 36.1 ns/day; at 1 fs: 19.7 -> 21.7 ns/day (CG 10.4 -> 8.4).
  Together with the cutoff lever and tol 1e-4: 63.4 ns/day at 2 fs. See docs/thermostat_ideas.md.
- **Truncated octahedron** (`build_amber.py --box oct`): ubiquitin 15,955 -> 15,238 atoms,
  34 -> 38 ns/day. DHFR 25,780 -> 22,492 atoms, 22.5 -> 25.7 ns/day.
- **Langevin thermostat and the dipole predictor.** Langevin (BAOAB here, ntt=3 in pmemd) gives
  every atom an independent random velocity kick each step, so the trajectory is no longer
  smooth. The cubic extrapolation of the dipoles (error = 4th difference along the trajectory)
  then has a noise part ~ sqrt(gamma) dt^1.5 instead of ~ dt^4, and higher orders amplify it.
  4096 waters, 1 fs, tol 1e-5, relative predictor error / mean CG iterations:

  | thermostat | pgm_jax cubic predictor | pgm_jax CG | pmemd-pgm mu4 CG | pmemd delta4 CG |
  |---|---|---|---|---|
  | none (NVE) | 8.3e-5 | 4.0 | 4.0 | 3.0 |
  | Langevin 0.1/ps | 4.0e-4 | 5.0 | 5.6 | 5.0 |
  | Langevin 1/ps | 1.3e-3 | 6.0 | 6.1 | 6.0 |
  | Langevin 5/ps | 2.8e-3 | 6.0 | 7.0 | 7.0 |
  | Bussi, tau 0.1-1 ps | 8.4e-5 | 4.0 | 4.0 (ntt=11) | 3.0 (ntt=11) |

  In this setting, pmemd-pgm with ntt=11 (Bussi) instead of ntt=3 runs at 0.73 ms/step instead
  of 0.96 (mu4) and 0.71 instead of 1.04 (delta4), with correct temperatures. At 2 fs the
  time-step error already dominates, so the gain is smaller (pmemd mu4: 7.0 -> 5.4 iterations).
  Solvated proteins with HMR at 2 fs gain little (ubiquitin 13 -> 12 iterations, 1 %); at 1 fs
  ubiquitin goes 10 -> 8 (8 % faster) and Trp-cage 9 -> 7 (6 %).
- **Rejected:** fp16 / bf16 storage of the rows (1.7x faster matvec at 98k, but 5e-4 / 3e-3
  relative matvec error), recomputing the kernels on the fly in XLA (no gain), spatial sorting
  of the molecules (no gain).

## Open items

- **pGM residue library.** The pGM parameters of the amino acids are still needed:
  multi-conformation ESP of capped fragments, fitted with covalent dipoles (and
  `ResidueLibrary.from_fits`). Until then, `placeholder` uses Amber charges and pGM
  polarizabilities and is only for testing the pipeline.
- **CG iterations in protein boxes.** 12-15 per step against 6-7 for pure pGM water at the same
  2 fs, so dt is not the cause. Candidates: the placeholder parameters (Amber charges + pGM
  polarizabilities, water included), the mu4 predictor, the preconditioner. Bringing it to the
  water level would make protein MD about 1.5x faster.
- **Water model.** Protein systems currently use tleap's TIP3P geometry. `load_amber(water=...)`
  swaps in the pGM water model.
- **Fragment training data.** SPICE dipeptides are at the same DFT level as our labels; φ/ψ
  and χ scans are still to be computed.
