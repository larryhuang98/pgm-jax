# Charge flux in MD

Charges and covalent-dipole strengths that change with bond lengths (`BondedSettings.flux` in
the bonded fits, `pgm_jax/md/flux.py` in MD). A flexible template fitted with flux runs in MD
unchanged: same functional form, same parameters, same sign conventions.

## Model

For a bond b = (i, j) of length r_b with reference length b0_b (the fit's `P["ref"]["b0"]`, the
b0 of the bond-stretch terms) and parameter key k (the bond's b0 key):

    db_b = r_b - b0_b
    q_i  = q_i^0 - sum_{b = (i, .)} s_b jb_k db_b + sum_{b = (., i)} s_b jb_k db_b     bond charge flux
    c_m  = c_m^0 + jc_k db_b [+ jc2_k db_b^2]      every covalent dipole m along bond b (dipole flux)

- As a bond stretches, the charge s_b jb_k db_b moves from its first atom to its second. The sign
  s_b is +1 when the first atom has the lower tying key of the fit (the canonical order of the
  bonded model's atom classes), -1 when it has the higher one, and 0 for a bond between two
  equivalent atoms, which carries no charge flux (its covalent dipoles still flux). So a
  molecule keeps its total charge, and symmetry-equivalent bonds move charge the same way.
- Both covalent dipoles of a bond (i -> j and j -> i) take the bond's jc (and jc2 with
  `flux=2`). Covalent dipoles along virtual bonds have no flux.
- Units: jb e/nm, jc e (e nm of dipole per nm of stretch), jc2 e/nm.

The fitted methanol (below) has jb = 2.7 e/nm on C-O (O more negative as the bond stretches),
-0.23 on C-H and -0.43 on O-H, and jc = 1.26 e on O-H: a 0.01 nm stretch changes that covalent
dipole by 0.013 e nm, about its own size. In the liquid the flux moves 0.003 e per atom on
average.

## Using it

```python
tpl = FlexibleTemplate.from_fit(model, P)           # a BondedModel with BondedSettings(flux=1 or 2)
sim = FlexibleSimulation(System([tpl.pgm] * 216), [tpl] * 216, pos, H, MDSettings(), dt=0.0005)
sim.ff.flux                                         # ChargeFlux: bonds, b0, keys, signs, cov_bond, params
```

`examples/fit_bonded_template.py methanol --flux 1 --wmu 1` fits one (`--wmu` puts the gas-phase
dipoles in the loss; the flux parameters mostly act on them). `FlexibleSimulation` builds the
`ChargeFlux` of the whole system from its templates (`ChargeFlux.from_templates`; molecules
without flux, rigid ones included, contribute nothing) and prints it in the log header. A
`ChargeFlux` can also be built directly from per-bond arrays (global atom pairs, b0, parameter
index, sign, covalent dipole -> bond map, parameter vectors) and passed to
`PGMForceField(..., flux=)`.

Parameters: `ChargeFlux.params = {"jb", "jc"[, "jc2"]}`, one value per bond key, one block per
template. The force field takes them from `params["flux"]` when the parameter pytree has that
entry and from `ChargeFlux.params` otherwise, so they can be differentiated like any other
parameter:

```python
theta = {**sys.params0, "flux": {k: jnp.asarray(v) for k, v in ff.flux.params.items()}}
g = jax.grad(lambda th: ff.compute(pos, H, idx, ff.init_induction(), th).energy["total"])(theta)
```

(with `MDSettings(differentiable=True)` for gradients of forces and dipoles). A parameter pytree
with a `"flux"` entry is refused by a force field without flux.

Rigid molecules have a fixed geometry, so their flux is a constant shift: `molecule_at(tpl, xyz)`
returns the template's pGM molecule with the charges and covalent dipoles of the flux model at
that geometry, for `RigidTemplate` or the rigid-body engine. X-H bonds constrained at their
reference lengths (`constraints="h-bonds"`) have db = 0 and carry no flux; the flux forces of a
constrained bond lie along it and are removed with the constraint forces.

`write_pgm_prmtop` refuses templates with flux (pmemd-pgm has no charge flux).

## How the engine does it

The energy is E(R, q(R), c(R), mu) with the induced dipoles variational, so

    F = -dE/dR|_{q,c,mu} - sum_i phi_i dq_i/dR - sum_m (dE/dc_m) dc_m/dR,

with phi_i = dE/dq_i the potential at atom i (times KE) and dE/dc_m = (dE/dd_i) . u_m for the
covalent dipole m on atom i along u_m. In `PGMForceField.compute`:

1. q(R), c(R) are formed first (`ChargeFlux.charges`, bond-local); the induction right-hand side
   uses them.
2. `_energy_forces_flux` is `_energy_forces` plus the potential: one more row sum
   (`_row_potential`, sum_k q_k G0 + (d_k . x) G1, which XLA fuses into the force pass over the
   rows), and the reciprocal, self and background parts from the same autodiff of the PME energy
   with q among the differentiated arguments. dE/dc comes from the covalent-frame pull-back the
   engine already does, taken with respect to c as well.
3. (phi, dE/dc) go back through the flux map with one vector-Jacobian product (`jax.vjp` of
   `ChargeFlux.charges`: per bond dE/d(db) = s jb (phi_j - phi_i) + sum over its covalent dipoles
   of dE/dc (jc + 2 jc2 db), then -+dE/d(db) u on its two atoms). Nothing differentiates through
   the solve or the rows. The map is written with gathers (per-atom bond tables); on the GPU this
   was as fast as or faster than scatter-adds in the map and than a hand-written gather-only
   pull-back or a per-atom formulation (2-4 % per step at a fixed CG count for all of them).

Every other use of the charges takes them at its own positions (`PGMForceField.charges_at`):
the Monte Carlo barostat's trial energies, `energy_fixed_mu` and hence the strain derivative
(virial), the differentiable path (the solve's implicit derivative sees q(R) and p(c(R)) as
inputs, so gradients reach jb, jc, jc2) and the cell dipole (`md/dipoles.py`). The engine's
molecular scaling translates molecules rigidly, so bond lengths and the flux do not change under
it; `strain_derivative(molecular=False)` stretches bonds and the flux follows by autodiff.
Without flux none of this code runs.

## Validation

Single points and derivatives, float64, dipoles solved to 1e-12 (`scripts/validate_flux.py` on the
fitted methanol with quadratic flux, `--flux 2`; `tests/test_flux.py` on made-up parameters):

| Check | Result |
|---|---|
| q(R), c(R) vs `BondedModel._flux` (perturbed geometry; flux shifts up to 0.0055 e, 0.0037 e nm) | 1e-17 |
| Gas-phase pGM energy of the bonded model vs `Model(ElecChannel)` of the molecule at those charges | 2.8e-16 relative; induced dipoles 1e-18 e nm |
| MD engine, one molecule in a 5 nm box: flux vs the flux-free engine at the same charges | energy 4.2e-16 relative, induced dipoles 1.5e-15 |
| MD forces vs gradient of the gas-phase model (flux forces up to 2000 kJ/mol/nm) | max 0.17 kJ/mol/nm of RMS 1250 (zero flux: 0.15; periodic images) |
| 32 flexible methanols (1.46 nm box): forces vs autodiff of E(R, q(R), c(R)) at fixed mu | 3e-15 relative |
| Same, forces vs central differences with the dipoles re-solved (24 components, h = 1e-5 nm) | 2.5e-7 relative |
| Molecular strain derivative (tr W) vs energy under box scaling | 1.3e-11 relative |
| Atomic strain derivative (bonds stretch, flux active), 6 components | 2.4e-8 relative |
| Differentiable path: d(w.F + w'.mu)/d(all parameters), /d(jb, jc, jc2), /d positions, /d box | 9e-12 to 1.5e-9 relative |
| dE/d(jb, jc, jc2), 9 parameters | 1.9e-8 relative |

MD with fitted templates (`scripts/flux_md.py`; one RTX PRO 6000 Blackwell, mixed precision,
tol 1e-5, dt 0.5 fs). Liquid methanol, 216 molecules, NPT 298 K / 1 bar (Langevin 1/ps, barostat
every 25 steps), 100
ps of equilibration from a dilute lattice and 100 ps sampled (5 blocks); gas phase: 256
independent molecules with the fitted gas-phase model (BondedModel), 100 ps each
(`scripts/flux_md.py liquid <template>`, `gas <template>`). The three
templates are fits of the same class II bonded set with the same weights (energies, forces,
dipoles; `examples/fit_bonded_template.py methanol --wmu 1 --maxiter 4000 [--flux 1 | 2]`):

| | no flux | flux 1 | flux 2 (+ jc2) | experiment |
|---|---|---|---|---|
| gas-phase fit, 298 K test frames: energy / force MAE (kcal/mol, /A), dipole RMSE (D) | 0.129 / 2.33 / 0.161 | 0.129 / 2.23 / 0.131 | 0.128 / 2.20 / 0.131 | |
| density (g/cm^3) | 0.7815 +- 0.0008 | 0.7860 +- 0.0019 | 0.7877 +- 0.0014 | 0.7866 |
| <U>/molecule, liquid (kJ/mol) | -796.35 +- 0.07 | -813.25 +- 0.05 | -812.10 +- 0.08 | |
| <U>, gas (kJ/mol) | -769.73 +- 0.02 | -785.71 +- 0.02 | -784.18 +- 0.02 | |
| heat of vaporization <U_gas> - <U_liq>/N + RT (kJ/mol) | 29.10 +- 0.07 | 30.02 +- 0.05 | 30.40 +- 0.08 | 37.4 |
| mean molecular dipole, liquid (D) | 2.274 | 2.262 | 2.270 | |
| mean molecular dipole, gas (D) | 1.680 | 1.677 | 1.678 | 1.70 |
| mean charge shift by the flux in the liquid (e per atom) / mean bond stretch db (nm) | | 0.0031 / 0.0009 | 0.0034 / 0.0008 | |
| kinetic temperature (K) | 296.6 +- 0.8 | 295.2 +- 0.6 | 295.0 +- 1.1 | 298 |
| CG iterations per step (Langevin, NPT) | 5.31 | 5.54 | 5.56 | |

The absolute energies differ between the fits (the intramolecular electrostatics changes with the
flux); the heat of vaporization compares them. With flux the liquid is 0.6-0.8 % denser and
0.9-1.3 kJ/mol more cohesive, the molecular dipole in the liquid changes by less than 0.5 %.
These shifts are of the size of the differences between bonded fits without flux (the paper's
methanol template, fitted to energies and forces only, gives 0.789 g/cm^3 with the same MD settings),
so methanol does not show a large condensed-phase effect of the gas-phase-fitted flux; its
point, polarization of the bonds by the liquid, needs fits to condensed-phase data (the
differentiable path gives the gradients).

NVE from the end of each liquid run (216 molecules, dt 0.5 fs, 3885 degrees of freedom; drift
from a linear fit of E_tot, per ns and degree of freedom; RMS deviation of E_tot from the fit;
`scripts/flux_md.py nve <template> --nve_ps 200` and `--nve_ps 0 --double_ps 100`):

| | no flux | flux 1 | flux 2 |
|---|---|---|---|
| mixed, tol 1e-5, 200 ps: drift (kT/ns/dof) | 1.1e-5 | 1.0e-4 | -1.6e-5 |
| RMS deviation (kT) / CG iterations per step | 0.387 / 4.99 | 0.404 / 5.04 | 0.413 / 5.04 |
| double, tol 1e-8, 100 ps: drift (kT/ns/dof) | -1.7e-4 | 1.8e-4 | 1.6e-4 |
| RMS deviation (kT) / CG iterations per step | 0.371 / 9.12 | 0.404 / 10.03 | 0.420 / 10.03 |

The drifts are at the noise of these run lengths (10 ps runs scatter by +-4e-3, 40 ps ones by
+-2e-3) and far below the 0.004 kT/ns/dof of the paper's 20 ps check without flux. The E_tot
fluctuation, the dt error of velocity Verlet on the fastest bond vibrations, grows by 5-10 %
with flux, presumably because the flux forces, which act along the bonds, change those vibrations.

Speed (one RTX PRO 6000 Blackwell; the flux-1 methanol liquid, NVT, mixed precision, tol 1e-5,
dt 0.5 fs, 0.9 nm, PME spacing 0.08 nm; the same template with and without its flux, alternating
blocks of 4000 steps in one process, 3 runs each; `scripts/flux_md.py speed [--replicate 2]
[--thermostat bussi] [--fixed_iter 5]`):

| | 1,296 atoms: no flux | flux | 10,368 atoms: no flux | flux |
|---|---|---|---|---|
| exactly 5 CG iterations per step (the cost of the flux terms) | 0.800 ms | 0.824 ms (+3.1 %) | 1.422 ms | 1.467 ms (+3.2 %) |
| Bussi 1 ps: ms per step (CG iterations) | 0.744 (5.00) | 0.761 (5.04), +2.3 % | 1.428 (5.00) | 1.494 (5.32), +4.7 % |
| Langevin 1/ps: ms per step (CG iterations) | 0.802 (5.00) | 0.844 (5.31), +5.3 % | 1.410 (5.00) | 1.553 (5.98), +10.2 % |
| NVE (from the liquid runs above): CG iterations | 4.99 | 5.04 | | |

The flux terms themselves cost 25-45 us per step (3 %), nearly all of it the flux map and its
vector-Jacobian product: the potential rows and the PME charge gradient are free (XLA fuses them
into passes the engine makes anyway; measured by switching each off). The rest is the dipole
solver: the charges now follow every bond vibration, so the right-hand side of the induction
changes faster and the predicted dipoles are a little worse. Per-atom white noise (Langevin) jitters the
bond lengths and hence the charges from step to step, which the extrapolation cannot follow:
+0.3 iterations at 1,296 atoms and +1 at 10,368 (the convergence test takes the largest residual
of any atom). With Bussi's global rescaling the trajectory stays smooth (+0.04 and +0.3), and in
NVE at tol 1e-5 the counts are nearly the same (5.04 vs 4.99; at tol 1e-8, 10.0 vs 9.1); Bussi is
also the fastest thermostat without flux (README, Thermostats). Without flux the lowered program of the MD step is identical to the one
before this feature (same StableHLO for flexible NVT / NPT in both precisions, the rigid engine
and the differentiable path), so its speed is unchanged.

## Limits

- Bond charge flux only (charge moving along bonds as they stretch) and covalent-dipole flux; no
  angle-dependent flux (the bonded model has none either).
- The rigid-body engine has no flux path: freeze the charges with `molecule_at`.
- No pmemd-pgm export.
