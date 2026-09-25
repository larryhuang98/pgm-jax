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
                         constraints="h-bonds", hmr=3.024)
sim.minimize(300)
sim.run(500000, report=5000, traj=5000, prefix="ubq")
```

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
| Solvated Trp-cage, placeholder electrostatics, X-H constraints + HMR, 2 fs | 96 ns/day |
| Solvated ubiquitin (16k atoms), same settings | 35 ns/day (dipole tol 1e-5), 41 ns/day (1e-4) |

Benchmark: `python scripts/protein/bench_protein.py sys.prmtop sys.inpcrd [--library lib.json] [--tol 1e-4]`.

## Open items

- **pGM residue library.** The pGM parameters of the amino acids are still needed:
  multi-conformation ESP of capped fragments, fitted with covalent dipoles (and
  `ResidueLibrary.from_fits`). Until then, `placeholder` uses Amber charges and pGM
  polarizabilities and is only for testing the pipeline.
- **CG iterations at 2 fs.** Dipole extrapolation (mu4) is less accurate at 2 fs; ubiquitin
  needs 9-13 CG iterations against 4-6 for water at 1 fs. Tuning the predictor and
  preconditioner is the next speed item.
- **Water model.** Protein systems currently use tleap's TIP3P geometry. `load_amber(water=...)`
  swaps in the pGM water model.
- **Fragment training data.** SPICE dipeptides are at the same DFT level as our labels; φ/ψ
  and χ scans are still to be computed.
