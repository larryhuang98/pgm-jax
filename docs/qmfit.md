# Fitting pGM parameters to QM cluster data

`pgm_jax/fit/qm.py` fits pGM parameters (charges, covalent dipoles, Gaussian radii,
polarizabilities, Lennard-Jones or GVDW) directly to quantum-chemical data of molecular clusters:
interaction energies, SAPT components, many-body (2- and 3-body) energies, rigid-body forces and
monomer properties. `data/qm/water_qm.json` is a water reference set built for it (psi4, 757
dimers, 77 trimers to pentamers, the WATER27 hexamers and octamers), and
`scripts/qmfit/fit_water_qm.py` fits and validates pGM water against it.

## What maps to what

For a cluster of rigid molecules the gas-phase model (all atom pairs interact, no masking) gives the
interaction energy E(cluster) - sum E(monomer), split as in `channels.elec_decomposition`:

| pGM term | meaning | QM counterpart |
|---|---|---|
| `elst` | interaction of the self-polarized monomers (Gaussian charges, covalent and intramolecular induced dipoles); the Gaussian overlap gives charge penetration | SAPT electrostatics (E_elst^(10)) |
| `ind` | relaxation of the induced dipoles in the cluster (<= 0) | SAPT induction incl. exchange-induction and dHF |
| `vdw` | Lennard-Jones or GVDW | SAPT exchange + dispersion |
| `total` | elst + ind + vdw | CCSD(T)/CBS estimate of the interaction energy |
| 3-body | only induction contributes (every other term is pairwise) | CP MP2/aug-cc-pVTZ 3-body energy |
| net force and torque on each rigid molecule | from the gradient of the interaction energy | gradient of the CP MP2/aTZ interaction energy (monomer terms exert neither) |
| monomer dipole, polarizability | charges + permanent + intramolecular induced dipoles; `molecular_polarizability` | CCSD/aug-cc-pVTZ |

## The water reference set (`data/qm/water_qm.json`)

All monomers are the rigid pGM3P-25 water (r_OH 0.9745 A, HOH 103.64 deg), superposed on the
source geometry (mass-weighted Kabsch, centre of mass kept), so the model and the QM see the same
geometry and the interaction energies contain no monomer deformation. A model with another rigid
geometry (the base parameters use 0.9572 A / 104.49 deg) is evaluated on the same clusters with its
own monomers superposed (`qmfit.superpose_monomers`).

| set | records | where the geometries come from |
|---|---|---|
| `smith` | 6 | Smith-type dimer stationary structures (Cs open = the minimum, Cs planar, Ci and C2h cyclic, C2v bifurcated, planar C2v bifurcated), optimized at MP2/aug-cc-pVDZ within their point group (`scripts/qmfit/smith_opt.py`; the C2 cyclic start converged onto C2h) |
| `radial` | 37 | O-O scans of four of them, 2.4-8 A |
| `angular` | 20 | around the minimum: acceptor flap (-60..100 deg), donor bend (+-40 deg), acceptor twist (30..180 deg) |
| `liquid2` | 235 | pairs from pGM liquid snapshots (p25_4096, p25_512, base_4096 restarts), stratified in R_OO from 2.4 to 6.5 A |
| `liquid3`, `liquid4`, `liquid5` | 57, 12, 8 | a molecule and 2-4 neighbours from the snapshots (compact first-shell trimers, 12 extended trimers) |
| `water27` | 10 | the (H2O)n clusters of WATER27 (GMTKN55): dimer, cyclic trimer, tetramer, pentamer, the prism, cage, book and cyclic hexamers, D2d and S4 octamers |
| `pairs` | 458 | every pair of every cluster with n >= 3 (2-body corrections, and more liquid-like dimers) |

Levels (psi4 1.11, frozen core, density fitting; `scripts/qmfit/psi4_clusters.py`):

- dimers: SAPT0/jun-cc-pVDZ (elst, exch, ind incl. dHF, disp; sSAPT0 too), counterpoise-corrected
  MP2/aug-cc-pVTZ and aug-cc-pVQZ (HF/aQZ + X^-3 extrapolated correlation = MP2/CBS) and
  DF-CCSD(T)/aug-cc-pVTZ; the reference is `E.ref` = MP2/CBS + [CCSD(T) - MP2]/aTZ;
  for the 235 liquid pairs also the gradient of the CP MP2/aTZ interaction energy;
- clusters: CP MP2/aug-cc-pVTZ many-body expansion in the cluster basis (every subset of up to 3
  molecules and the whole cluster; the octamers only the whole cluster and the monomers), giving
  the 2-body, 3-body and >= 4-body energies; the reference `E.ref` is the MP2/aTZ interaction energy
  plus the CCSD(T)/CBS correction of every pair (computed in the dimer basis);
- monomer: CCSD/aug-cc-pVTZ dipole and static polarizability.

### Checks of the reference data

| check | value |
|---|---|
| water dimer minimum (rigid monomers at the MP2/aDZ Smith Cs geometry), `E.ref` | -5.085 kcal/mol; MP2/CBS -5.061, dCCSD(T)/aTZ -0.023. Literature De (relaxed monomers): -5.02 +- 0.05 (Tschumper et al. 2002), -4.97 (WATER27); the difference is the monomer deformation energy (+0.03-0.1) |
| WATER27 clusters: `E.ref` (rigid monomers) vs the literature De (relaxed) | dimer -5.09 / -4.97, trimer -16.09 / -15.71, tetramer -27.90 / -27.35, pentamer -36.51 / -35.88, prism -47.19 / -45.99, cage -46.95 / -45.73, book -46.27 / -45.29, cyclic -45.04 / -44.30, octamers -74.31, -74.33 / -72.49, -72.45: E.ref is 0.1-1.9 kcal/mol lower, growing with n as the deformation energy does |
| hexamer order | E.ref: prism < cage < book < cyclic, as CCSD(T)/CBS in the literature (Bates & Tschumper 2009; WATER27) |
| composite vs its parts (757 dimers, vs `E.ref`) | MP2/aTZ RMSE 0.25, MP2/aQZ 0.12, MP2/CBS 0.07 kcal/mol; dCCSD(T)/aTZ rms 0.07, max 0.37 |
| SAPT0/jun-cc-pVDZ total vs `E.ref` | RMSE 0.46, mean +0.17, max 2.3 kcal/mol (sSAPT0: 0.50): the components are a guide, not a better reference than the total |
| many-body (MP2/aTZ, CP) | hexamers: 3-body -8.0 (prism), -8.3 (cage), -9.5 (book), -10.8 (cyclic); >= 4-body -0.4 to -1.7 kcal/mol |
| monomer, CCSD/aTZ at the rigid geometry | dipole 1.868 D (experiment 1.855), polarizability 1.439 A^3 (experiment 1.47) |

Cost: 3350 psi4 tasks, 274 core-hours (CCSD(T)/aTZ 202, MP2 aTZ + aQZ 43, many-body 18,
gradients 7, SAPT0 4), 1.5 h of wall time on 12 workers x 16 cores. Per dimer on 16 cores: SAPT0
1.2 s, CP MP2/aTZ 2.4 s, aQZ 10 s, CCSD(T)/aTZ 55 s, MP2/aTZ gradient 6.4 s; MBE(3) trimer 12 s,
pentamer 2 min, hexamer 5.5 min. A fit of 11 parameters to the 495 training clusters takes 10-20 s
on 8 CPU cores (the residuals and their Jacobian for all clusters in one compiled call).

## Usage

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pgm_jax import read_prmtop_pgm
from pgm_jax.fit.qm import (
    ClusterModel,
    FitWeights,
    ParamMap,
    QMFit,
    QMSet,
    error_table,
    evaluate,
    format_table,
    rigid_minimize,
    rigid_water,
)

data = QMSet.load("data/qm/water_qm.json")  # records + monomer properties
train, test = data.split(lambda r: "base_4096" in r["id"] or r["id"].startswith("water27") or r["set"] == "smith")
w = read_prmtop_pgm("p25_512.prmtop")[0]  # starting parameters
cm = ClusterModel(w, vdw="lj", monomer_xyz_nm=rigid_water(0.9745, 103.64) * 0.1)
pm = ParamMap(
    cm.table,
    [w],
    {"q": "all", "cov": "all", "radius": "all", "alpha": "all", "lj_rmin_half": ["OW"], "lj_sqrt_eps": ["OW"]},
)
fit = QMFit(
    cm, pm, train, FitWeights(total=1, elst=0.1, ind=0.1, exch_disp=0.1, nb3=1, dipole=1, polarizability=1, prior=0.01)
)
res = fit.fit()  # scipy least_squares, exact Jacobian
P = pm.params(jnp.asarray(res.x))  # parameter pytree for every pgm_jax model
print(format_table(error_table(evaluate(cm, test, P))))  # RMSE / MAE per set, kcal/mol
E_min, X_min = rigid_minimize(cm, test.records[0]["xyz_A"], P)  # the model's own rigid-body minimum
```

- `ClusterModel(mol, vdw="lj" | "gvdw")`: per-cluster components (`components`, `batch`,
  `batch_grad`), monomer dipole and polarizability. Clusters of one kind of rigid molecule
  (`System([mol] * n)`), compiled once per cluster size and batched with `jax.vmap`.
- `ParamMap(table, molecules, free)`: `free` = {quantity: "all" | [keys]}; bounds and scales per
  quantity (`BOUNDS`, `SCALES`); free charges move in the null space of the neutrality
  constraints, so every molecule keeps its charge.
- `FitWeights`: weights of the residual groups: `total` (the reference interaction energy
  `E.ref`), `elst`, `ind`, `exch_disp` (SAPT components, dimers), `nb3` (3-body energies of
  clusters), `force` (net force and torque on each rigid molecule), `dipole`, `polarizability`
  (monomer), `prior` (ridge towards the starting values, in units of `SCALES`). Energy residuals
  are divided by sigma_E sqrt(n_pairs) (1 + max(E_ref, 0) / e_soft), so larger clusters and
  repulsive geometries count less.
- `QMFit.fit()` minimizes |r(theta)|^2 by trust-region reflective least squares (bounds) with the
  Jacobian from `jax.jacfwd`; `QMFit.loss` is differentiable for other optimizers.
- `evaluate`, `error_table`, `format_table`: predictions and error statistics per set.
- Scripts: `scripts/qmfit/smith_opt.py`, `build_water_clusters.py`, `psi4_clusters.py` (QM worker;
  any number of slurm workers share one queue through atomic claims and can be restarted),
  `collect_water_qm.py` (dataset), `fit_water_qm.py` (`baseline`, `fit NAME [--free ...] [--w ...]
  [--vdw gvdw] [--init ...]`, `summary`).

## Validation: pGM water against the set

Training set: the scans and everything cut from the pGM3P-25 snapshots (495 clusters: 20
angular, 37 radial, 175 liquid pairs, 38 trimers, 8 tetramers, 5 pentamers, 212 pairs of those
clusters). Test set (348): everything from the base_4096 snapshot (60 liquid pairs, 19 trimers, 4
tetramers, 3 pentamers, 246 pairs of those and of the WATER27 clusters), the 6 Smith-type
structures and the 10 WATER27 clusters. Starting point for every fit: pGM3P-25
(`~/project/epsp/p25_512.prmtop`); baselines pGM3P-25 and the base parameters
(`~/project/epsp/base/base_512.prmtop`, evaluated with their own rigid geometry on the same
clusters). Errors in kcal/mol (forces kcal/mol/A). `validation/qmfit/{final,combo,probe}.sh`
reproduce every row (run from the clone root; each fit writes `runs/qmfit/<name>.json` and
`data/qm/fits/<name>.json`, `fit_water_qm.py summary` prints the table); the reports are kept in
`validation/qmfit/`.

Columns: RMSE of the interaction energy on the training and test sets and MAE on the test set;
test RMSE of the SAPT components (dimers) and of the 3-body energies (clusters); the model's
interaction energy at the QM dimer minimum (QM -5.085) and the model's own rigid-body minimum
(literature -5.0); order of the four hexamers at the rigidified WATER27 geometries (QM and
literature: prism < cage < book < cyclic; otherwise the model's order); monomer dipole and
polarizability (CCSD/aTZ 1.868 D, 1.439 A^3); test RMSE of the net forces on the molecules
(rms QM force 3.7).

| model | E_int train | E_int test | MAE test | elst | ind | exch+disp | 3-body | 3-body W27 / MP2 | dimer at QM min | model dimer min | hexamer order | dipole D | alpha A^3 | force test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pGM3P-25 | 0.97 | 1.50 | 0.63 | 3.59 | 1.48 | 5.04 | 1.53 | 0.59 | -4.38 | -4.91 | cage < prism < book < cyclic | 1.46 | 1.49 | 1.31 |
| base | 1.07 | 2.61 | 0.92 | 4.62 | 1.77 | 6.07 | 2.25 | 0.40 | -3.40 | -3.73 | ok | 1.86 | 1.98 | - |
| lj_total | 0.96 | 1.53 | 0.63 | 3.59 | 1.48 | 5.00 | 1.53 | 0.59 | -4.40 | -4.90 | cage < prism | 1.46 | 1.49 | 1.27 |
| **all_total** | 0.45 | 0.75 | 0.26 | 3.21 | 1.44 | 4.72 | 1.14 | 0.70 | -4.93 | -4.95 | ok | 1.89 | 1.45 | 1.11 |
| all_total_F | 0.46 | 0.98 | 0.33 | 3.19 | 1.54 | 4.74 | 1.42 | 0.62 | -4.84 | -4.85 | ok | 1.89 | 1.45 | 0.92 |
| all_sapt_0.01 | 0.48 | 0.55 | 0.24 | 2.94 | 1.27 | 4.28 | 0.61 | 0.84 | -5.04 | -5.04 | book < cage < prism < cyclic | 1.91 | 1.45 | 1.34 |
| all_sapt_0.03 | 0.58 | 0.49 | 0.26 | 2.63 | 1.06 | 3.76 | 0.19 | 0.99 | -5.16 | -5.26 | book < cage < cyclic < prism | 1.94 | 1.45 | 1.69 |
| all_sapt_0.1 | 0.92 | 0.76 | 0.39 | 1.25 | 0.97 | 2.22 | 0.27 | 1.02 | -5.22 | -5.42 | book < cyclic < cage < prism | 1.97 | 1.46 | 3.16 |
| all_sapt_0.3 | 1.34 | 1.09 | 0.62 | 1.57 | 0.49 | 1.99 | 1.42 | 1.37 | -5.62 | -6.32 | cage < prism < book < cyclic | 2.11 | 1.45 | 4.41 |
| all_sapt_1 | 1.20 | 1.30 | 0.57 | 0.84 | 0.56 | 1.37 | 1.77 | 1.47 | -5.65 | -5.91 | cyclic < book < cage < prism | 2.11 | 1.55 | 4.65 |
| all_sapt_only | 1.67 | 1.40 | 0.72 | 0.79 | 0.48 | 1.22 | 1.95 | 1.52 | -5.76 | -6.18 | book < cage < cyclic < prism | 2.06 | 1.50 | 6.29 |
| gvdw_total | 0.29 | 0.50 | 0.21 | 2.88 | 1.21 | 4.07 | 0.34 | 0.92 | -4.83 | -4.87 | cage < prism < book < cyclic | 1.89 | 1.44 | 0.64 |
| gvdw_sapt_0.3 | 0.75 | 0.96 | 0.63 | 1.02 | 0.65 | 1.10 | 1.07 | 1.27 | -4.23 | -5.69 | ok | 2.04 | 1.46 | 1.12 |
| gvdw_sapt_only | 1.39 | 2.21 | 1.23 | 0.79 | 0.48 | 0.57 | 1.95 | 1.52 | -4.03 | -6.09 | ok | 2.06 | 1.50 | 2.38 |
| rec_lj | 0.54 | 0.52 | 0.26 | 2.78 | 1.22 | 4.10 | 0.39 | 0.90 | -5.05 | -5.12 | book < cage < cyclic < prism | 1.92 | 1.46 | 1.31 |
| **rec_gvdw** | 0.33 | 0.57 | 0.33 | 2.08 | 1.13 | 3.07 | 0.18 | 1.00 | -4.79 | -5.02 | cage < prism < book < cyclic | 1.94 | 1.42 | 0.49 |
| rec_gvdw_nosapt | 0.31 | 0.45 | 0.24 | 2.63 | 1.22 | 3.79 | 0.28 | 0.94 | -4.82 | -4.91 | cage < prism < book < cyclic | 1.91 | 1.45 | 0.56 |
| probe_nb3x10 | 0.53 | 0.44 | 0.23 | 3.13 | 1.17 | 4.41 | 0.36 | 0.91 | -4.94 | -5.02 | book < cage < prism < cyclic | 1.89 | 1.46 | 1.40 |
| probe_nomono | 0.44 | 0.44 | 0.20 | 3.24 | 1.38 | 4.68 | 0.46 | 0.89 | -4.95 | -4.99 | prism < book < cage < cyclic | 1.92 | 1.86 | 1.03 |


Fits: `lj_total` Lennard-Jones of O only, totals only; `all_*` q, the two covalent dipoles, both
radii, both polarizabilities and the O Lennard-Jones (9 parameters) with totals, 3-body energies
(weight 1), monomer dipole and polarizability, a weak prior, and SAPT components with the weight
given in the name (`_F`: forces 0.1; `_sapt_only`: no totals); `gvdw_*` the same with GVDW on O
and H instead of Lennard-Jones (13 parameters); `rec_*`: totals, SAPT 0.03 (`_nosapt`: 0), 3-body 10, forces 0.1; `probe_nb3x10`: 3-body weight 10; `probe_nomono`: no monomer targets.

What the numbers say:

- **pGM3P-25** is a good 2-body model near liquid geometries (test liquid pairs 0.65, pairs of
  clusters 0.49 kcal/mol RMSE) but underbinds the dimer minimum (-4.38 at the QM geometry, -4.91 at
  its own minimum), gets the Smith relative energies wrong (Cs planar +1.5 vs 0.65), has 59-76 % of
  the MP2 3-body energy, underbinds the hexamers by 7.5-10.5 kcal/mol (prism -39.2 vs -47.2) and puts
  the cage below the prism. Its gas-phase dipole is 1.46 D. **The base parameters** are worse
  (test RMSE 2.6; 3-body 26-45 % of MP2, their larger radii damp the induction; polarizability
  1.98 A^3), although their hexamer order is right.
- Refitting only the Lennard-Jones changes nothing (1.50 -> 1.53): the error is in the
  electrostatics and induction. Refitting all nine pGM parameters to totals halves the test error
  (0.75, MAE 0.26), fixes the dimer (-4.93 / -4.95), the Smith relative energies (within 0.3) and the
  hexamer order, with dipole 1.89 D and polarizability 1.45 A^3.
- **Components vs totals.** With Lennard-Jones on O only the vdW term cannot follow SAPT
  exchange + dispersion (test RMSE 4.7 when not fitted, 1.2-1.4 at best). Small component weights
  (0.01-0.03) still help the held-out totals and the 3-body energies (test E_int 0.49, 3-body 0.19)
  because they pin the radii (charge penetration) and the damping of the induction; larger weights
  trade totals for components (weight 1: components 0.8 / 0.6 / 1.4, totals 1.3, forces 4.6).
  Fitting components only gives the elst and ind terms of SAPT to 0.8 and 0.5 kcal/mol, but
  totals to 1.4 (LJ) and 2.2 (GVDW): SAPT0/jun-cc-pVDZ is itself 0.46 kcal/mol RMSE from the
  reference, and the pGM terms are not complete (no charge transfer, no exchange-induction).
  GVDW on O and H follows exchange + dispersion much better than LJ (0.57 vs 1.2 with components
  only).
- **3-body energies** are the test of the induction model (the model's 3-body energy is pure
  induction). pGM3P-25 reproduces the MP2/aTZ 3-body energies of the liquid trimers to pentamers to
  0.2-0.24 kcal/mol RMSE but gives only 59 % of the WATER27 (tetramer to hexamer) 3-body energy
  (RMSE 3.3), the base parameters 40 % (their larger radii damp the induction). With the QM
  monomer polarizability (1.44 A^3) and fitted radii, rec_gvdw gets 100 % (RMSE 0.11 / 0.16 / 0.41
  / 0.20 for the test trimers / tetramers / pentamers / WATER27) and all_sapt_0.03 99 %; fits to
  totals alone keep ~70 % (all_total). Without the monomer targets the polarizability goes to
  1.86 A^3 (`probe_nomono`): the fit then buys 3-body energy with polarizability instead of
  with less damping.
- **Hexamers** (kcal/mol; at the rigidified WATER27 geometries, in brackets the model's own
  rigid-body minimum from `rigid_minimize` started there):

  | | prism | cage | book | cyclic |
  |---|---|---|---|---|
  | CCSD(T)/CBS* (this set, rigid monomers) | -47.19 | -46.95 | -46.27 | -45.04 |
  | WATER27 literature De (relaxed monomers) | -45.99 | -45.73 | -45.29 | -44.30 |
  | pGM3P-25 | -39.22 (-41.03) | -39.42 (-41.93) | -37.91 (-40.64) | -34.59 (-37.96) |
  | base | -33.94 (-35.51) | -32.61 (-34.42) | -31.64 (-34.12) | -31.57 (-33.88) |
  | all_total | -42.94 (-43.44) | -42.82 (-43.42) | -42.46 (-43.07) | -41.55 (-42.28) |
  | rec_gvdw | -44.96 (-45.34) | -45.23 (-45.61) | -43.58 (-44.25) | -41.21 (-41.97) |

  The four isomers lie within 2.2 kcal/mol; the fits bring the absolute energies from 8-13 kcal/mol
  (pGM3P-25, base) to 2-4 kcal/mol too weak, but the order is fragile: all_total (and base,
  gvdw_sapt_0.3) have the literature order, the GVDW fits put the cage 0.27 below the prism, and
  several Lennard-Jones fits with better totals put the book first. The isomer order is not a
  training target here.
- **Forces** (net force and torque on each molecule, test liquid pairs vs CP MP2/aTZ; rms QM
  force 3.7 kcal/mol/A): 1.31 (pGM3P-25), 1.11 (all_total), 0.92 when fitted (`all_total_F`),
  0.49 (rec_gvdw); large component weights degrade them (4-6).
- **Per set** (test RMSE / MAE of E_int, kcal/mol):

  | | liquid pairs | pairs of clusters | trimers | tetramers | pentamers | WATER27 | Smith |
  |---|---|---|---|---|---|---|---|
  | pGM3P-25 | 0.65 / 0.46 | 0.49 / 0.36 | 0.89 / 0.70 | 1.03 / 0.82 | 1.54 / 1.21 | 8.15 / 7.29 | 1.28 / 1.20 |
  | base | 0.63 / 0.44 | 0.67 / 0.49 | 1.30 / 1.11 | 2.26 / 1.95 | 3.52 / 3.19 | 14.65 / 12.95 | 0.81 / 0.50 |
  | all_total | 0.32 / 0.19 | 0.19 / 0.12 | 0.61 / 0.50 | 0.37 / 0.31 | 0.59 / 0.56 | 4.17 / 3.57 | 0.21 / 0.15 |
  | rec_gvdw | 0.29 / 0.22 | 0.32 / 0.26 | 0.54 / 0.43 | 0.53 / 0.46 | 1.20 / 1.14 | 2.62 / 2.22 | 0.32 / 0.30 |

  Smith-type relative energies (Cs planar, Ci cyclic, C2h cyclic, C2v bifurcated, planar C2v
  bifurcated; QM 0.65, 0.75, 1.11, 1.94, 2.86): pGM3P-25 1.52, 0.85, 2.34, 2.38, 3.15; base -0.02,
  -0.89, -0.70, 0.32, 1.20; all_total 0.57, 0.69, 1.40, 1.87, 2.67; rec_gvdw 0.85, 0.23, 1.01,
  2.02, 2.75.

## Fitted parameters

`data/qm/fits/<name>.json` (`pgm_jax.param.load_molecule`; rigid geometry 0.9745 A / 103.64 deg).
Two are recommended, depending on the engine:

| | pGM3P-25 | all_total (LJ; pmemd-pgm compatible) | rec_gvdw (GVDW on O and H; pgm_jax) |
|---|---|---|---|
| q_O (e) | -2.0406 | -1.9855 | -1.2557 |
| covalent dipole O>H, H>O (e nm) | -0.01912, 0.00859 | -0.02196, 0.00730 | -0.01041, -0.01343 |
| radius O, H (nm) | 0.0605, 0.0536 | 0.0648, 0.0556 | 0.0665, 0.0371 |
| alpha O, H (nm^3) | 1.118e-3, 3.30e-4 | 6.87e-4, 5.24e-4 | 7.21e-4, 4.84e-4 |
| LJ O: R*, eps | 0.1786 nm, 0.606 kJ/mol | 0.1943 nm, 0.217 kJ/mol | - |
| GVDW O: sqrt A, sqrt C6, b | - | - | 62.83, 0.06997, 1.4315 |
| GVDW H: sqrt A, sqrt C6, b | - | - | 2.992, 0.00222, 0.158 |

all_total is the most conservative refit (totals, 3-body, monomer targets; physical LJ; the
Smith energies, the dimer and the hexamer order right; 70 % of the cluster 3-body energy).
rec_gvdw is the best-balanced model (3-body energies, forces, components) but needs per-atom GVDW.
The Lennard-Jones fits with SAPT weights (`all_sapt_*`, `rec_lj`) drive the O Lennard-Jones to a
large R* with a tiny well (e.g. rec_lj: R* 0.322 nm, eps 5e-4 kJ/mol): the O-only r^-12/r^-6 cannot
represent SAPT exchange + dispersion, so they are not recommended despite their test errors.

## Tests

`tests/test_qmfit.py` (8 tests, 2.7 min on 8 CPU cores):

- dataset IO: `QMSet` save / load round trip, `label`, `split`; the committed set loads, its dimer
  minimum is within [-5.5, -4.5] kcal/mol and its monomers are rigid;
- the per-cluster components equal the `Model` interaction energy E(ABC) - sum E(X) to 1e-10, and
  the 3-body energy of `Prepared` equals `Model.nbody`; the model's 3-body energy is pure induction
  (perm and vdw 3-body < 1e-9);
- `ParamMap`: theta0 gives the starting parameters, any theta keeps every molecule neutral;
- rigid-body forces: monomer energies exert no net force or torque; the net force from the
  interaction gradient equals the finite difference along a rigid translation (1e-6);
  `superpose_monomers` keeps the centres of mass;
- the loss gradient (totals, SAPT components, 3-body, forces, dipole, polarizability, prior; 9
  parameters) equals central finite differences (2e-5 relative);
- a synthetic target made by the model at perturbed parameters (7 parameters) is recovered
  exactly: loss < 1e-12 and theta to 1e-5;
- the committed fit `data/qm/fits/all_total.json` gives the dimer and prism energies of its report.

## Limits

- One kind of rigid molecule per data set (`ClusterModel` builds `System([mol] * n)`); mixtures
  and flexible monomers need one template per molecule kind and monomer-deformation terms.
- The QM set is water only, at the rigid pGM3P-25 geometry; the base model is evaluated on the same
  clusters with its own monomers superposed (the QM energies are not recomputed for its geometry).
- The reference for clusters with n >= 3 is MP2/aTZ plus pairwise CCSD(T)/CBS corrections: the
  3-body and higher terms are MP2/aTZ (no CCSD(T) 3-body correction). Forces are MP2/aTZ. The octamers
  have no many-body decomposition. SAPT is SAPT0/jun-cc-pVDZ (dimers only; no SAPT for clusters).
- The pGM terms map onto SAPT only approximately: there is no charge-transfer or
  exchange-induction term (they sit in SAPT induction), and the Lennard-Jones on O alone cannot
  follow SAPT exchange + dispersion; per-atom GVDW (O and H) is available in pgm_jax
  (`GVDWChannel`, `PeriodicGVDW`) but not in pmemd-pgm, which takes one global GVDW parameter set.
- Gas-phase fits only: the fitted parameters were not tested in the liquid (density, heat of
  vaporization, dielectric constant). `scripts/fit_liquid.py` gives liquid-property gradients;
  a joint objective is the sum of the two (the QM residuals of `QMFit.residuals` and the liquid
  Gauss-Newton rows).
- Hexamer isomer order is a small difference (0.2-2.2 kcal/mol) between large numbers; it is not a
  training target and is reproduced by some fits only.
