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
| MD topology | `md/topology.py` | neighbour-list groups (heavy-atom groups), special pairs with vdW weights, constraints |
| Constraints | `md/constraints.py` | SHAKE / RATTLE per cluster, vectorised; hydrogen mass repartitioning, one mass or per molecule (`AmberSystem.hmr`) |
| Restraints | `md/restraints.py` | positional, distance, angle, dihedral, centre-of-mass distance (Amber NMR form); `AmberSystem.select` / `position_restraints` |
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
                         MDSettings(dipole_tol=1e-4), dt=0.002, ensemble="npt",   # 4 fs: see below
                         constraints="h-bonds", hmr=3.024,     # or asys.hmr({"water": 4.0, "protein": 3.024})
                         thermostat="bussi", tau_t=1.0)   # fastest with pGM (docs/thermostat_ideas.md)
sim.minimize(300)            # after minimization, equilibrate with thermostat="langevin" (faster warm-up)
sim.run(500000, report=5000, traj=5000, prefix="ubq")
```

Restraints (`pgm_jax/md/restraints.py`; Amber conventions, E = k x^2, k in kJ/mol/nm^2 or
kJ/mol/rad^2, `KCAL_A2` = 418.4 per kcal/mol/A^2) go in with `restraints=` or, on a running
simulation, `set_restraints` (which recompiles the step). Positional restraints on the protein
built from the current positions, released in stages:

```python
from pgm_jax.md.restraints import KCAL_A2, DihedralRestraint, harmonic
x0, H0 = sim.positions_nm(), sim.state.box                   # e.g. after minimize()
for k in (10.0, 5.0, 1.0, 0.1):                              # kcal/mol/A^2 on the heavy atoms
    sim.set_restraints(asys.position_restraints(k * KCAL_A2, "heavy", x0, H0))   # scaling "com"
    sim.run(25000, report=5000, prefix=f"eq_k{k:g}")         # log column erestraint
sim.set_restraints(None)
# a phi restraint (IUPAC sign, rad): atoms in system order, e.g. from prot.atom_names
sim.set_restraints(DihedralRestraint([[c0, n1, ca1, c1]], harmonic(np.radians(-63.0)), k=50.0))
```

Under NPT the reference of a "com" restraint moves with the molecule's scaled centre, which is
what the molecular Monte Carlo barostat does to the protein; "fractional" scales every reference
point with the box, "none" keeps it fixed. Restraints are not stored in checkpoints: pass them
again when continuing. Cost on ubiquitin (2 fs, NVT, 4.9 ms/step): 603 heavy-atom positional
restraints are within the timing noise (0.1 ms); adding 144 distance and dihedral restraints and a
centre-of-mass restraint costs 5 %.

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
| Ubiquitin, HMR 3.024 (protein) / 3.024 or 4.0 (water), Bussi, 4 fs, 420 ps | stable (SHAKE 2e-14, econs drift < 1e-4 kT/ns/dof); <U> +83 +- 25 kJ/mol above 1 fs (about 1 K) |
| Restraints (positional, distance, angle, dihedral, centre-of-mass distance) | forces and strain derivative = finite differences (float64); NVE with restraints conserves the energy in both engines (`tests/test_restraints.py`) |

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

### Time step and hydrogen masses per molecule kind (ubiquitin)

`FlexibleSimulation(hmr=asys.hmr({"water": 4.0, "protein": 3.024, "ion": None}))` gives water
hydrogens 4.0 amu (oxygen 10.0) and protein hydrogens 3.024 (a CH3 carbon keeps 5.96 amu; at
4.0 it would keep 3.0). Ubiquitin, 15,955 atoms (4,908 waters), NVT 298 K, Bussi tau 1 ps, X-H
constraints, rigid water, dipole tol 1e-5, mixed precision, placeholder electrostatics. Every run
starts from one structure (minimised, 100 ps at 2 fs, 820 ps at 4 fs), re-equilibrates 20 ps and
samples 400 ps every 0.5 ps:
`bench_protein.py ubq.prmtop ubq.inpcrd --coords equil.rst7 --minimize 0 --thermostat bussi
--equil-ps 20 --prod-ps 400 --dt ... [--hmr-water 4.0]`. dU = <U> - <U>(1 fs), errors from 10
block averages of 40 ps; econs drift in kT/ns per degree of freedom; ns/day from 2000 unsampled
steps. The largest constraint error was 2.5e-14 in every run.

| dt | H mass protein / water (amu) | ns/day | CG iterations | <T> (K) | econs drift | dU (kJ/mol) |
|---|---|---|---|---|---|---|
| 1 fs | 3.024 / 3.024 | 21.9 | 8.52 | 297.3 | -0.0000 | 0 +- 16 |
| 2 fs | 3.024 / 3.024 | 35.3 | 12.83 | 296.8 | -0.0000 | -9 +- 26 |
| 4 fs | 3.024 / 3.024 | 56.3 | 16.70 | 293.8 | -0.0000 | +83 +- 25 |
| 4 fs | 3.024 / 4.0 | 56.3 | 16.71 | 293.9 | +0.0001 | +93 +- 24 |
| 5 fs | 3.024 / 3.024 | 68.7 | 18.17 | 291.1 | +0.057 | +170 +- 24 |
| 5 fs | 3.024 / 4.0 | 68.2 | 18.19 | 291.1 | +0.086 | +185 +- 18 |

- **4 fs is stable and 1.6x faster than 2 fs.** Over 420 ps per run (and the 820 ps of
  equilibration), the constraints hold to 1e-14, the temperature fluctuates as at 1 fs, and
  econs drifts no more than at 1-2 fs.
- **Its configurational error is small.** <U> lies 80-95 kJ/mol above the 1 fs value, 0.02 kJ/mol
  per water. With d<U>/dT = var(U) / (k_B T^2) = 85 kJ/mol/K, that is about 1 K of configurational
  temperature. 2 fs cannot be told apart from 1 fs. The kinetic temperature reads 0.7 K low at
  1 fs, 4 K low at 4 fs and 7 K low at 5 fs: the discretisation of the kinetic estimator, as in
  pure water.
- **The water hydrogen mass changes nothing measurable here.** 4.0 instead of 3.024 gives the same
  CG iterations, speed, drift and <U> at 2, 4 and 5 fs. In pure pGM water, heavier hydrogens do
  cut the CG count (docs/thermostat_ideas.md 3f). In the protein box the protein atoms
  presumably set it: the convergence test takes the largest residual, and aromatic and charged
  side chains converge slowest (above).
- **5 fs drifts.** econs gains 0.04-0.09 kT/ns/dof (replicates with either water mass) and <U>
  rises by 170-230 kJ/mol.
- **Replicate.** A first set of 200 ps runs from a start equilibrated for only 120 ps gave the
  same picture: 4 fs +86 / +114 kJ/mol (water 3.024 / 4.0), 5 fs +227 / +210 (drift 0.073 /
  0.041), 2 fs with water 4.0 +26 +- 32.
- **In short.** For equilibrium sampling of this protein, 4 fs with 3.024 amu hydrogens is the
  fastest stable setting. Per-kind masses are for systems where one mass does not fit all: a
  uniform 4.0 would strip CH3 carbons to 3 amu, while water-dominated systems gain from 4.0
  (pure water: 4-5 fs).

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
