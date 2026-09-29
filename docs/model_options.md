# Model options: electrostatics, van der Waals, bonded term sets

pGM-JAX started as pGM with Lennard-Jones. It now has a menu of nonbonded and bonded options.
Every option is a JAX function of coordinates, parameters and box, so everything stays
differentiable. This note says what each option is, where it is available and how it was
checked.

| Part | Options | Default |
|---|---|---|
| Electrostatics `elec` | `"q"` Gaussian charges · `"qp"` + permanent covalent dipoles · `"qi"` charges + induced dipoles · `"qpi"` pGM | `"qpi"` |
| Quadrupoles `quadrupoles` | covalent Gaussian quadrupoles (in the code, off by default; not in MD yet) | off |
| Van der Waals `vdw` | `"lj"` Lennard-Jones · `"gvdw"` Gaussian-density vdW (`gvdw_rep` `"gauss"` / `"slater"`) · `"none"` | `"lj"` |
| Bonded sets `families` | `T.SETS["amber"]` Amber/GAFF forms · `T.SETS["explore"]` class II and our families · `T.SETS["nn"]` neural bonded terms | explore |
| Charge flux `flux` | `1` bond charge flux + covalent-dipole flux c0 + jc db · `2` + jc2 db² | 0 (off) |

Where the options apply:

| | gas phase (`ElecChannel`, `Model`) | periodic Ewald (`PeriodicModel`) | bonded fitting (`BondedSettings`) | MD (`MDSettings`, flexible templates) |
|---|---|---|---|---|
| `elec` levels | yes (`ElecChannel.level`) | yes | yes | yes |
| quadrupoles | yes | no | yes (+ ESP) | no (rejected with an error) |
| `vdw` = LJ / GVDW | yes (`LJChannel`, `GVDWChannel`) | yes | yes | yes (+ tail correction, virial) |
| bonded sets | – | – | yes | yes (fitted templates; the NN set frozen) |
| charge flux | – | – | yes (`BondedSettings.flux`) | yes (fitted templates, `md/flux.py`; `docs/charge_flux.md`) |

A flexible template carries the settings it was fitted with; `FlexibleSimulation` refuses MD
settings that differ (`FlexibleTemplate.check_settings`).

## 1. Electrostatics levels

All four levels use the pGM Gaussian distributions (exponent from the atomic radii) and let
every atom pair interact, with no 1-2/1-3 masking.

- `"q"`: Gaussian charges only, a fixed-charge Gaussian model, the cheapest level.
- `"qp"`: charges plus pGM's permanent covalent dipoles (the covalent basis vectors, CBV).
  This is the simplified, non-polarizable pGM. It keeps the anisotropy of the ESP and costs
  about the same as `"q"` in MD.
- `"qi"`: charges plus induced Gaussian dipoles.
- `"qpi"`: full pGM.

```python
from pgm_jax.channels import ElecChannel

e = ElecChannel.level("qp").energy(pos, sys)[0]  # {"perm": ...}; "ind" only with induction
from pgm_jax.periodic import PeriodicModel

pm = PeriodicModel(sys, H, pos, elec="q", vdw="gvdw", gvdw_rep="slater")
from pgm_jax.md.forcefield import MDSettings

st = MDSettings().replace(elec="qp", vdw="lj")  # no induction solve in MD
from pgm_jax.bonded.model import BondedSettings

bs = BondedSettings(elec="qi", vdw="gvdw")
```

Without induction the MD engine skips the solver: no CG iterations and a zero polarization
energy. Checks: in MD, `"q"`, `"qp"` and `"qi"` match exact Ewald to 2e-6 relative. Forces
match autodiff and finite differences (`tests/test_md_options.py`, `tests/test_options.py`).

## 2. Gaussian quadrupoles (derivation and set-up)

**Interaction.** Each atom carries a Gaussian charge q, dipole p and traceless quadrupole Θ
(Buckingham convention), all with the atom's Gaussian width. A pair interacts through
multipole operators applied to the Gaussian Coulomb kernel φ = erf(β r)/r:

    E_ij = O_i O_j φ(|x|),   x = r_i − r_j,
    O_i = q_i + p_i·∇ + (1/3) Θ_i:∇∇,   O_j = q_j − p_j·∇ + (1/3) Θ_j:∇∇.

The radial functions are B_0 = φ and B_(n+1) = −(1/r) dB_n/dr, the same ones the dipoles use
(closed form, with a series at small βr). Contracting the derivatives of φ with traceless Θ gives

    E_ij = q_i q_j B0 + [q_i (p_j·x) − q_j (p_i·x)] B1 + (p_i·p_j) B1 − (p_i·x)(p_j·x) B2
         + (1/3)[q_i (xΘ_j x) + q_j (xΘ_i x)] B2
         + (1/3)[(p_j·x)(xΘ_i x) − (p_i·x)(xΘ_j x)] B3 + (2/3)[p_i Θ_j x − p_j Θ_i x] B2
         + (1/9)[(xΘ_i x)(xΘ_j x) B4 − 4 (xΘ_i Θ_j x) B3 + 2 (Θ_i:Θ_j) B2].

The field of j's permanent multipoles at i is the right-hand side of the induction equations:

    E_i = q_j x B1 − p_j B1 + (p_j·x) x B2 − (2/3) Θ_j x B2 + (1/3)(xΘ_j x) x B3.

The quadrupoles therefore polarize the induced dipoles as well. `tests/test_options.py` checks
the energy and the field against nested automatic derivatives of the operator form, to 1e-9.
One point matters for Gaussians. The Gaussian kernel is not harmonic inside the clouds
(∇²φ ≠ 0), so the trace of a second moment would change the energy. The quadrupole is
therefore defined traceless: 5 components per atom.

**Can the quadrupole use the CBV idea? Yes.** The dipole is p_i = Σ_j c_ij u_ij over unit
vectors to bonded partners. The quadrupole counterpart is a sum of traceless products of
the same unit vectors:

    Θ_i = Σ_(j,k) t_ijk S(u_ij, u_ik),   S(u, v) = (3/4)(u vᵀ + v uᵀ) − (1/2)(u·v) 1.

- **j = k** gives a uniaxial quadrupole along the bond, S(u,u) = (3uuᵀ − 1)/2, with Θ_zz = t
  along u.
- **j ≠ k** gives the anisotropy in the plane of two partners, for example lone-pair plane
  versus π axis.
- **Terminal atoms** (carbonyl O, halogens, H) have one partner, so they would only get the
  axial term. `terminal13=True` adds their 1-3 neighbours as virtual partners.

For atoms with two or more partners, these tensors span every quadrupole allowed by the local
symmetry. As with the covalent dipoles, there are no local frames: Θ follows the geometry
smoothly, and forces come from autodiff with no frame torques to code. Strengths are typed
like the CBV dipoles. Keys: `mol:Q:C>O` (axial) and `mol:Q:C>H|O` (pair), stored in
`Molecule.quad` and `ParamTable` quantity `"quad"`.

```python
from pgm_jax.multipole import with_quadrupoles

mq = with_quadrupoles(mol)  # adds the (i, j, k, t) terms, t = 0
ch = ElecChannel(quadrupoles=True)  # or ElecChannel.level("qpi", quadrupoles=True)
bs = BondedSettings(quadrupoles=True)  # fitting; BondedModel.esp() includes (1/3)(xΘx)B2
```

**Status.** The quadrupoles are in the gas phase and bonded fitting, and they are tested. With
t = 0 the results equal pGM exactly. They are not yet in Ewald/PME or MD, and templates with
quadrupoles are rejected there. They also have no fitted values yet. Next steps:

- fit t to the ESP together with the charges and dipoles (py_resp-style, now by autodiff);
- add the quadrupole terms to the PME reciprocal sum, which needs the second-order spline
  derivatives the dipoles already use for forces.

## 3. Van der Waals: LJ and GVDW

**LJ** is the Amber form with Lorentz-Berthelot mixing (`lj_rmin_half`, `lj_sqrt_eps`), as before.

**GVDW** is the Gaussian-density vdW of Huang, Luo and Duan (the `pgm_gvdw_letter` manuscript;
pmemd-pgm `igvdw = 1`). It uses the same Gaussian exponent as the pGM electrostatics,
y = β_ij r with β_ij = 1/√(2(R_i² + R_j²)):

    U_ij = A_ij exp(−B_ij y²)   (gvdw_rep="gauss",  pmemd gvdw_rep_form = 0)
         | A_ij exp(−B_ij y)    (gvdw_rep="slater", gvdw_rep_form = 1)
           − C6_ij F(y)/r⁶,
    F(y) = [erf y − (2y/√π)(1 + 2y²/3) e^(−y²)]² + (1/2)[(4y³/(3√π)) e^(−y²)]².

F is the damping of the dipole-dipole dispersion between two Gaussian clouds. F/y⁶ tends to
8/(9π) at short range, so the energy is finite everywhere. It is evaluated from a 15-term
exact-rational series below y = 0.6 (truncation < 1e-13) and in closed form above, which keeps
it safe in float32.

- **Parameters.** Per atom type: `gvdw_sqrt_a`, `gvdw_sqrt_c6`, `gvdw_b`. Combining rules:
  A_ij = a_i a_j, C6_ij = c_i c_j, B_ij = (b_i + b_j)/2.
- **Setting them.** `set_gvdw(mol, {"OW": (sqrt_a, sqrt_c6, b)})`. `from_pmemd(A_kcal,
  C6_kcal_A6, b)` converts pmemd's input.
- **Water.** `vdw.PGM3P_GVDW` holds the manuscript's pGM3P water parameters.
- **Long range.** The tail correction is −2π(Σ c_i)²/(3 V r_c³), with its virial.

**Validation** against pmemd-pgm with 512 pGM3P waters (`scripts/validate_gvdw.py`,
`validation/validate_gvdw.json`):

| | pmemd VDWAALS (kcal/mol) | pGM-JAX | force RMSD (kcal/mol/Å) |
|---|---|---|---|
| Slater | 784.7282 | 784.72818 | 6.9e-7 |
| Gauss | 868.3699 | 868.36986 | |

**Liquid water with our MD engine** at the manuscript's settings (NPT, 298 K, 1 bar, 200 ps × 2
seeds, `scripts/md_gvdw_water.py`):

| vdW | density (g/cm³) | manuscript | speed (ns/day) |
|---|---|---|---|
| GVDW Slater | 0.9995 ± 0.0013, 0.9984 ± 0.0012 | 0.999 | 125–130 |
| GVDW Gauss | 0.9967 ± 0.0006, 0.9979 ± 0.0010 | 0.997 | 124–135 |
| LJ (pGM3P) | 1.0068 ± 0.0014, 1.0095 ± 0.0010 | | 124–128 |

**Open choices:**

- pmemd uses one global (A, C6, b) for the pairs of LJ-bearing atoms. Per-type values with
  geometric/geometric/arithmetic mixing are our generalization for mixtures and molecules,
  to be agreed on.
- C8/C10 terms would follow from the same B_n machinery (quadrupole-dipole dispersion) if they
  are wanted later.

## 4. Bonded term sets

`BondedSettings(families=T.SETS[name])`:

### `"amber"`: the Amber/GAFF forms

`bond_harm`, `angle_harm`, `torsion_amber` (n = 1–4, phases 0/180 as signed K_n) and
`improper_amber` (K(1 − cos 2ω)). Use it with `typing="amber"` (GAFF atom types from the
molecule's `types`) and `lj14_scale=0.5`. The electrostatics stay pGM, all pairs.

- `scripts/bonded/gaff_prmtop.py` makes GAFF prmtops (antechamber types, parmchk2, tleap). It
  has been run for the 18 molecules.
- `bonded.amber.init_from_prmtop(model, P, {0: "gaff.prmtop"})` starts from the GAFF values.
  Unit conversions: Kb = 2 K_b · 418.4, Ka = 2 K_θ · 4.184, K_n = ±PK · 4.184.
- `with_amber_impropers(spec, prmtop)` takes Amber's improper atom order.

**Checked** against cpptraj on a NetCDF trajectory (`scripts/bonded/check_amber_import.py`): the
bond, angle, dihedral and improper energies agree to about 1e-4 kcal/mol for methanol,
formamide, acetaldehyde, alanine dipeptide, benzene, pyridine and acetate. This set is the easy
entry point for students. Parameters are GAFF-like, so they can be read, compared and refitted
by autodiff.

### `"explore"`: our families

The class II set of the bonded study (`T.PAPER`) by default. Any of the registry's families can
be added: Urey-Bradley, 1-3/1-4 pair terms, conjugation, hyperconjugation, hybrid-orbital
angles, twist, and others. See `docs/howto_bonded.md` and `reports/bonded/README.md`.

### `"nn"`: neural bonded terms (NNB), fast by construction

The idea: **the network replaces atom typing, not the energy function.** A small graph network
reads the molecule once and writes the parameters of ordinary analytic bonded terms, per term
instance. MD then evaluates only those analytic terms, so an MD step costs the same as a
classical force field.

- **Stage 1: the parameter network, run once per molecule.**
  - Input: the bond graph, with atom features element, degree, bond orders, ring and aromatic
    flags, and the atom's pGM charge, polarizability, radius and |covalent dipole|.
  - Message passing: 3 layers, width 32, with bond-order edge features.
  - Readouts: permutation-symmetric per component kind. A bond is (h_i + h_j, h_i ⊙ h_j).
    An angle is (h_j, h_i + h_k, h_i ⊙ h_k). A torsion is symmetric under i-j-k-l → l-k-j-i.
    An improper is symmetric in its outer atoms.
  - Output: the parameters of each instance of the class II families. Instances are
    decomposed by the same key function that types the classical terms, so couplings such as
    bond-angle or torsion-bond know which bond or angle they refer to.
  - Reference values: b₀ and θ₀ are the minimum geometry plus bounded learned shifts (`ref=
    "geometry"`), or come from covalent radii and hybridization with no QM geometry
    (`ref="predicted"`). The shift networks also see the reference value itself, because
    graph-equivalent angles are not alike in the minimum geometry.
  - Symmetry: the parameters are exactly invariant under permutations of equivalent atoms, and
    the energy is invariant under rotation (tested).
- **Typed table + residual** (`nn_table_depth=0`, `nn_resid_l2`): every head outputs
  "element-typed table value + network residual". A penalty on the residual (training
  molecules) shrinks the model to a classical element-typed force field where the data do not
  ask for more. The table is readable like a parameter file.
- **Stage 2: the energy.** Exactly the class II expressions with these per-instance parameters.
  `FlexibleTemplate.from_fit` freezes stage 1 (`NNBonded.freeze`), so the MD step has no
  network in it.
- **Related work.** Graph networks that assign MM parameters already exist: espaloma (Wang et
  al., Chem. Sci. 2022; espaloma-0.3, 2024) and Grappa (Seute et al., Chem. Sci. 2025). Both
  predict class I terms next to a fixed-charge, 1-4-scaled nonbonded model.
- **What is specific here:**
  - The nonbonded part is pGM: polarizable, all pairs, no exclusions. The bonded network only
    fills the short-range remainder, and it is trained with that nonbonded model in the loss.
  - The inputs include the pGM parameters, so the bonded and electrostatic terms are
    consistent by construction.
  - The basis can be any registry family set: class II couplings, our new terms, or the Amber
    forms.
  - The typed table plus shrunk residual gives a readable classical force field as the
    fallback.
  - Everything is one JAX program with the nonbonded model, so charges, polarizabilities and
    vdW can be fitted jointly.
  - The frozen-table idea extends to proteins through residue templates: stage 1 runs once
    per residue type, or once per molecule for ligands.

**Speed** (500 alanine dipeptides, 11 000 atoms, energy + forces, one GPU,
`scripts/bonded/bench_sets.py`):

| set | ms per evaluation |
|---|---|
| amber | 0.43 |
| explore (class II) | 0.38 |
| nn (frozen) | 0.30 |

Stage 1 takes 0.5 ms per molecule, once. For scale, a pGM MD step for 12 000 atoms takes about
2 ms.

**Accuracy** (`scripts/bonded/nnb_experiments.py`, summary by `scripts/bonded/nnb_summary.py`;
pGM all pairs; energy MAE kcal/mol / force MAE kcal/mol/Å on 298 K frames, training on 500 K
frames).

*Per molecule.* The network is fitted to one molecule, as the class II terms are.

| molecule | NNB | class II (symmetry-typed) |
|---|---|---|
| ethane | 0.10 / 1.26 | 0.11 / 1.34 |
| methanol | 0.11 / 1.87 | 0.13 / 2.33 |
| methylamine | 0.16 / 2.03 | 0.18 / 2.35 |
| acetaldehyde | 0.14 / 1.84 | 0.21 / 2.98 |
| formic acid | 0.11 / 1.92 | 0.23 / 4.67 |
| formamide | 0.13 / 1.88 | 0.21 / 3.68 |
| fluorochloroethane | 1.44 / 2.54 | 0.30 / 3.10 |
| chloroformic acid | 0.15 / 2.30 | 0.28 / 5.34 |
| chloromethanol | 0.22 / 2.59 | 0.31 / 3.82 |
| acetate | 0.27 / 2.95 | 0.31 / 4.39 |
| methylammonium | 1.27 / 8.15 | 0.85 / 7.43 |
| hydrogen phosphate | 0.27 / 3.64 | 0.27 / 4.22 |

Two findings:

- **Forces:** NNB beats class II for 11 of 12 molecules, with the same energy expressions. Its
  parameters are per instance, not per symmetry class.
- **Energies:** fluorochloroethane is the exception. Its training loss is 3.5× lower than class
  II's, but the 298 K energies are worse, so the network overfits this molecule's many unique
  instances.

Two fixes made this possible. The reference-angle shifts must see the reference angle itself,
and θ₀ needs up to ±0.35 rad: class II θ₀ ends up 0.15–0.28 rad from the minimum
geometry, with large bond-angle couplings compensating.

*Transfer, leaving one molecule out.* Train on 11 molecules, test on the held-out one.
Baselines are the element-typed class II and class I terms of the bonded study.

| held out | NNB, table + residual (shrinkage 10) | NNB, table + residual (shrinkage 1) | NNB, network only (predicted ref) | class II, element-typed | class I, element-typed |
|---|---|---|---|---|---|
| ethane | 1.62 / 17.4 | 3.72 / 23.2 | 1.01 / 11.2 | 1.07 / 13.2 | 0.64 / 6.5 |
| methanol | 1.22 / 14.4 | 1.45 / 16.0 | 2.45 / 22.0 | 1.60 / 17.7 | 1.20 / 15.7 |
| chloroformic acid | 2.32 / 19.5 | 2.69 / 34.1 | 6.68 / 53.5 | 3.26 / 41.5 | 1.66 / 26.4 |
| methylamine | 5.99 / 34.8 | 4.82 / 44.3 | 3.85 / 37.2 | 2.94 / 23.3 | 3.49 / 29.2 |
| acetaldehyde | 1.99 / 25.3 | 2.17 / 32.7 | 2.76 / 22.9 | 3.12 / 32.1 | 1.18 / 16.5 |
| formic acid | 1.67 / 30.5 | 1.51 / 32.1 | 3.23 / 45.5 | 2.37 / 32.0 | 1.68 / 25.6 |
| chloromethanol | 2.11 / 18.5 | 3.12 / 25.7 | 3.71 / 28.2 | 1.83 / 19.0 | 1.98 / 17.2 |
| acetate | 2.72 / 46.2 | 3.24 / 31.4 | 5.73 / 100.7 | 2.74 / 46.5 | 1.65 / 21.0 |
| methylammonium | 5.44 / 74.9 | 2.93 / 35.5 | 3.00 / 35.3 | 2.57 / 39.5 | 2.04 / 30.8 |
| hydrogen phosphate | 13.30 / 110.6 | 14.61 / 136.6 | 37.60 / 255.7 | 7.27 / 74.0 | 7.27 / 74.1 |
| formamide | 4.92 / 40.9 | 3.40 / 30.8 | 2.40 / 21.9 | 4.76 / 37.6 | 1.69 / 22.7 |
| fluorochloroethane | 5.23 / 42.5 | 3.53 / 33.1 | 3.90 / 33.6 | 4.08 / 29.0 | 1.90 / 12.6 |
| mean (12) | 4.04 / 39.6 | 3.93 / 39.6 | 6.36 / 55.6 | 3.13 / 33.8 | 2.20 / 24.9 |

**With 11 training molecules, NNB does not transfer better than element typing.**

- **In training** it fits far better. Training-set means are 0.27 / 2.8 (network only) and
  0.45 / 5.9 (table, shrinkage 10), against about 1.7 / 11 for element-typed class II.
- **On a new molecule** it is worse on average. Wins such as chloroformic acid and methanol
  are offset by failures on chemistry the other 11 molecules do not cover: the only P
  compound, and amines versus ammonium.
- **The typed table with a shrunk residual** limits the damage but does not beat class I.
- **Other variants tried** did not help on the first folds either: the class I basis, and
  narrower reference shifts.

This matches the rest of the bonded study: element-typed class I transfers best at this data
size. A network needs many molecules. The next step for the ML-bonded force field is a
training set of hundreds of molecules, sampled and labelled with the existing pipeline
(`scripts/bonded/`: MACE-OFF sampling, ωB97M-D3(BJ) labels, pGM parameters). The
architecture, speed and MD path are ready for it.

Settings: `BondedSettings(families=T.SETS["nn"], nn_ref="geometry", nn_table_depth=0,
nn_resid_l2=10.0)` for transfer experiments. Fitting uses `Fitter.fit(..., adam_steps=3000)`:
Adam, then L-BFGS.
