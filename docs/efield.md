# External electric fields and the finite-field dielectric constant

`pgm_jax/md/efield.py` (field, units, formulas), `pgm_jax/md/finite_field.py` (batched field
replicas, analysis); `Simulation(..., efield=...)`, `FlexibleSimulation(..., efield=...)`,
`ElecChannel(efield=...)` (gas phase); `scripts/run_md.py --efield / --efield-freq /
--displacement`; `scripts/finite_field.py` (finite-field eps runs and analysis);
`scripts/validate_efield.py` (validation). Tests: `tests/test_efield.py`.

**In short.** A uniform external field acts on the whole pGM charge density: Gaussian charges,
covalent (permanent) dipoles and induced dipoles, which respond to it through the induction
equations. Static fields E0, oscillating fields E0 cos(w t + phi) and constant electric
displacement D are available in both MD engines (and E in the gas-phase model), jit-compatible,
with no cost without a field (the field-free code path is unchanged) and negligible cost with one.
The finite-field dielectric constant from +-E replicas (batched on one device) reproduces the
fluctuation results of TIP3P, pGM3P-25 and the Amber test pGM water at a fraction of the cost
(numbers below).

## Usage

```python
from pgm_jax.md.efield import ExternalField, displacement

sim = Simulation(system, positions, box, settings, thermostat="bussi", efield=(0.0, 0.0, 0.1))  # V/nm
sim.run(50000, prefix="md", report_every=1000, dipoles_every=25)  # log: efield, field_energy, Mx My Mz (e nm)
sim.set_field((0.0, 0.0, 0.2))  # new amplitude, no recompilation
ExternalField.from_wavenumber((0, 0, 0.2), 200.0)  # E(t) = E0 cos(2 pi c nu t), nu in cm^-1
ExternalField((0, 0, 0.2), omega=37.7, phase=0.0)  # omega in rad/ps
displacement((0.0, 0.0, 3.0))  # constant D: D/eps0 = 3 V/nm
FlexibleSimulation(sys, templates, pos, H, settings, efield=...)  # same for the flexible engine
ElecChannel(efield=(0.0, 0.0, 0.1))  # gas phase: energy "field", induced dipoles respond
ff.compute(pos, H, idx, ind, efield=(E, None))  # force field: (E V/nm, dipole offset) or (D, offset, "D")
```

```bash
python scripts/run_md.py -p water.prmtop -c water.rst7 --ensemble nvt --efield 0 0 0.1 ...
python scripts/finite_field.py run --model p25 --density 1.010 --fields 0.05 0.1 0.2 --zero 2 --ns 1 -o ff/p25
python scripts/finite_field.py analyse ff/p25.ffd --skip-ps 50      # eps per replica, per +-E pair, fluctuations
```

`FieldReplicas(sim, fields)` (finite_field.py) runs copies of a simulation in different fields as
one `jax.vmap`-ed program (the batched engine of the replica exchange, without exchanges), writing
the cell dipole of every replica to `prefix.ffd`; `read_series` / `analyse` evaluate it.

## Physics and conventions

Energy, forces and induction (SI in the formulas, model units in the code):

    U(R, mu; E) = U_pGM(R, mu) - E . M,      M = sum_i q_i r_i + sum_i p_i + sum_i mu_i

- A Gaussian charge or dipole density has the moments of the point multipole at its centre, so a
  uniform field sees point multipoles (and no quadrupole energy).
- Forces F_i += q_i E; the field's torque on each covalent dipole reaches the atoms through the
  covalent frame (the same vector-Jacobian product as the other dipole forces: dE/dd -= E);
  with charge flux the external potential -E . r_i enters the charge pull-back.
- The induced dipoles minimise U, so E is added to the permanent field on the right-hand side,
  (alpha^-1 - T) mu = E_perm + E, everywhere the engine forms it (plain and fused initial
  residual, least-squares predictor, barostat trial energies, the adjoint of the differentiable
  solve, which also gives dM/dE).
- Reported: energy["field"] = -E . M (kJ/mol), part of the total; Result.dipole = M;
  MDState.fdip = M of the last force evaluation; log columns efield, field_energy, Mx My Mz.
- Units: E in V/nm (1 V/nm on 1 e: 96.485 kJ/mol/nm); internally e/nm^2 (energy KE M . F),
  F = 0.694462 E.

Periodic boundary conditions. Smooth PME omits k = 0: tin-foil (conducting) boundary conditions,
so the applied field is the macroscopic (Maxwell) field in the sample (no depolarising field).
M uses the engines' whole molecules. For neutral molecules sum q r does not change when a
molecule is shifted by a lattice vector, so the field energy is continuous across re-wrapping.
Charged molecules: the driver books Q_k L of each re-wrapped molecule in MDState.fshift (M is
then the itinerant, unwrapped dipole), so E_tot stays conserved in NVE while the field drives the
ions across the cell (tested); forces never depend on it.

Virial and NPT. The molecular virial and the barostat translate molecules rigidly, so for neutral
molecules the field term is invariant at fixed mu: it adds exactly nothing to the molecular virial
(tested), and the Monte Carlo trial energies see the field through the induced dipoles re-solved in
the trial box. NPT is therefore the constant-pressure ensemble of U - E . M at constant Maxwell
field, electrostriction included. With charged molecules the term -E . sum Q_k R_k is not
invariant under scaling (it depends on the image of each ion), and NPT with a field is refused. NVT
at the zero-field density is the recommended protocol for dielectric constants.

Time-dependent fields, E(t) = E0 cos(w t + phi). E0 is a state variable (MDState.efield: replicas
can carry different amplitudes; `set_field` changes it without recompiling), w and phi are static.
The forces of step n+1 are evaluated at t_{n+1}. The Hamiltonian depends on time, dH/dt|_x =
-dE/dt . M (Hellmann-Feynman). The step is velocity Verlet in phase space extended by (t, p_t),
whose shadow energy H + p_t is conserved when p_t takes -dH/dt|_x (dt/2) in each half kick; the
driver books that energy, (dt/2)(dH/dt|_n + dH/dt|_{n+1}), in MDState.heat, so econs = E_tot +
|aux|^2/2 - heat stays conserved as with a thermostat. (A first version booked the switch
-(E_{n+1} - E_n) . M_{n+1}; that is off by dE . alpha_cell . dE / 2 per step and drifted by 2 % of
the absorbed energy at resonance.)

Constant electric displacement (`displacement(D)`, D given as D/eps0 in V/nm; Stengel, Spaldin &
Vanderbilt, Nat. Phys. 5, 304 (2009); Zhang & Sprik, PRB 93, 144201 (2016)):

    U_D = U_pGM(R, mu) + (eps0 V / 2) |E(M)|^2,     E(M) = D/eps0 - M / (eps0 V)

The field acting on the charges and dipoles is the Maxwell field E(M), which follows the
polarization; the induction operator gains the symmetric all-to-all term (4 pi / V) sum_j mu_j
(model units), so the solver is still conjugate gradients with the same preconditioner. U_D
depends on V: the molecular virial and the barostat include it (autodiff of the strain, tested
against finite differences). eps = D / (eps0 <E>) = (D/eps0) / (D/eps0 - <M.e>/(eps0 V)).
D = 0 is open circuit in every direction: the full depolarising field -P/eps0 acts on any macroscopic polarization.

## The finite-field dielectric constant

In the linear regime, with tin-foil boundary conditions (E the Maxwell field),

    eps = 1 + <M . e>_E / (eps0 V |E|)                                    (single replica)
    eps = 1 + (<M . e>_{+E} - <M . e>_{-E}) / (2 eps0 V |E|)              (+-E pair)

with M the total dipole, induced dipoles included, so the electronic part eps_inf needs no
separate calculation (the fluctuation formula needs eps_inf from the cell polarizability, since
adiabatic dipoles carry no thermal fluctuation of their own). In pgm_jax units (M e nm, V nm^3,
E V/nm) eps - 1 = 18.0951 <M.e> / (V |E|). The pair removes the zero-field bias of a finite run
(<M>_0 vanishes only on average) and every even-order term; several |E| show the onset of
dielectric saturation, eps(E) = eps(0) - c E^2.

Statistical error. <M.e> over a run of length T with integrated correlation time tau has the
variance 2 tau <dM_e^2> / T, and <dM_e^2> = (eps - eps_inf) eps0 V kB T, so

    sigma_pair(eps) = sqrt((eps - eps_inf) kB T / (eps0 V)) sqrt(tau / T) / |E|,
    sigma_fluct(eps) = (eps - eps_inf) sqrt(2 tau / (3 T'))                (zero-field run of length T').

At equal cost (T' = 2T) the ratio of variances, i.e. the cost ratio at equal error, is
(<M.e>_E / sd(M_e))^2 / 3: the induced dipole must exceed the thermal fluctuation of M. It grows
as E^2 and as V (the fluctuation estimate is size independent in cost per error, the finite-field
one gains with V at the same E), up to where the response saturates.

## Validation

### Single molecules, forces, energy conservation (`tests/test_efield.py`, `scripts/validate_efield.py`)

| Check | Result |
|---|---|
| Gas phase (`ElecChannel(efield=E)`, E = (0.02, -0.05, 0.1) V/nm): sum mu(E) - sum mu(0) vs `molecular_polarizability` . E | 1.6e-14 (pGM3P-25 water), 8.9e-15 (methanol), 2.2e-14 (water-methanol-water cluster), relative |
| Gas phase energy vs E(0) - E.M0 - E.alpha.E/2 (exact for a linear response) | 0 to print precision (tests: 1e-9 relative) |
| Gas phase forces vs central finite differences (h = 1e-5 nm, every coordinate) | max 3e-5 (water), 1.3e-6 (methanol), 1.5e-5 (cluster) kJ/mol/nm; net force 0, torque = M x E (tests) |
| One pGM3P-25 water in a periodic box (engine, float64): alpha_zz from the field response vs gas phase | 7.8e-4 (L = 2 nm), 2.3e-4 (3), 9.7e-5 (4), 2.9e-5 (6 nm): the difference x V is constant (0.006 nm^3), the field of the periodic images |
| MD forces vs autodiff of the energy at fixed mu (elec q, qp, qpi; constant E and constant D) | < 1e-9 relative (float64) |
| MD forces vs finite differences with the dipoles re-solved | < 2e-6 relative |
| Charges only: the field adds exactly q E (constant D: q E(M)) to the forces | 1e-9 kJ/mol/nm |
| Periodic linear response at fixed nuclei: dM_ind = alpha_cell . E (`CellDipole.polarizability`, Ewald couplings) | 1e-8 relative; energy quadratic in E to 1e-9 |
| Constant D at fixed nuclei: dM = (1 + 4 pi alpha_cell / V)^-1 alpha_cell dD | 1e-8 relative |
| dU/dE = -M and dM/dE = alpha_cell through the differentiable solve (custom_vjp); dU/dD = V eps0 E(M) | 1e-8, 1e-7 relative |
| Strain derivative: static field adds nothing for neutral molecules; constant D matches finite differences of the molecular scaling | 1e-8; 1e-5 relative |
| Zero field (`efield=(0, 0, 0)`) vs no field | energies and forces 1e-12, trajectories 1e-12 |
| No field vs master (`scripts/efield_identical.py`: rigid and flexible engines, 300 steps, mixed, CPU) | bitwise identical |
| MTS(inner=1) with a time-dependent field vs the ordinary step (both engines) | positions 1e-11, econs 1e-8, booked work 1e-9 |
| Charge flux (flexible methanols with fitted-size flux), constant E and D: forces vs autodiff and vs differences with re-solved dipoles | 1e-9; 1e-6 relative |
| Charged molecules (Na+, Cl- in water, 2 V/nm, NVE): the field drives the ions across the cell | re-wrapping booked as Q L in MDState.fshift; econs continuous |

NVE, 512 pGM3P-25 waters (`validate_efield.py nve`: density 1.010, mixed precision, dipole tol 1e-5,
1 fs, 20 ps from the same 10 ps NVT state; drift from a linear fit of econs, 40 points):

| Field | econs drift (kT/ns/dof) | econs rms (kJ/mol) | notes |
|---|---|---|---|
| none | +0.0002 | 0.26 | |
| E = 0.1 V/nm | +0.0022 | 0.25 | <field energy> -21 kJ/mol |
| E = 0.5 V/nm | +0.0005 | 0.26 | the molecules orient: -466 kJ/mol of field energy heats the box from 293 to 318 K |
| E = 0.2 cos(w t) V/nm, 200 cm^-1 | -0.0012 | 0.29 | resonant absorption: 1062 kJ/mol of work in 20 ps (293 -> 320 K), booked; E_tot changes by 1047 |
| constant D, D/eps0 = 3 V/nm | +0.0015 | 0.25 | |

The drifts are all within the noise of a 20 ps fit (0.002 kT/ns/dof corresponds to 0.34 kJ/mol over
the run, against 0.25 kJ/mol rms); none grows with the field. With the first (right-end) booking of the
work of E(t), 0.5 V/nm at 200 cm^-1 (10^4 kJ/mol absorbed, the box heated to 600 K) drifted by
1.28 kT/ns/dof, 2 % of the absorbed energy; with the trapezoid the resonant run above conserves econs
to 0.02 % of the absorbed energy.

### Finite-field dielectric constant (512 waters, NVT, 298 K)

Protocol (`scripts/finite_field.py run`): the 512-water boxes of the project (`~/project/epsp`),
scaled to the model's NPT density in pgm_jax (pGM3P-25 1.010, base pGM 0.983, TIP3P 0.986 g/cm^3), 10
replicas in one vmapped program on one RTX PRO 6000: +-E along z for |E| = 0.02, 0.05, 0.1, 0.2
V/nm and two zero-field copies; rigid bodies, 2 fs, Bussi 1 ps, 0.9 nm cutoff with the LJ tail, PME
48^3 order 6, dipole tol 1e-5, mixed precision; M every 50 fs, the first 50 ps discarded; errors
from 10 contiguous blocks per replica (tau_M 1-8 ps, blocks >= 95 ps). References: the fluctuation
formula on 10 ns zero-field NPT runs of the same models in pgm_jax (`validate_efield.py fluct`, 200 ps
discarded, jackknife; for base pGM five independent runs).

eps per +-E pair (the antisymmetric combination; 1 sigma):

| Model | +-0.02 V/nm | +-0.05 | +-0.1 | +-0.2 | fit eps0 - c E^2 | zero-field copies (fluctuations) | reference: fluctuations, 10 ns zero field |
|---|---|---|---|---|---|---|---|
| pGM3P-25 (2 x 1.95 ns per field) | 34.6 +- 3.0 | 35.3 +- 1.1 | **33.9 +- 0.5** | 33.2 +- 0.2 | **34.4 +- 0.6** (all, c = 29 +- 16, chi2 0.9 / 2) | 33.1 +- 0.5, 34.4 +- 0.8 | 34.2 +- 0.7 (docs/dielectric.md: 33.9 +- 0.7; pmemd.pgm.cuda 34.3 +- 0.6) |
| base pGM water, q_O -1.73 (2 x 0.95 ns) | 71.7 +- 2.9 | 74.0 +- 1.0 | 70.5 +- 0.3 | 63.4 +- 0.2 | **73.1 +- 0.4** (all, c = 244 +- 12, chi2 2.8 / 2) | 74.7 +- 2.6, 72.3 +- 1.6 | 73.2 +- 0.2 (5 runs x 9.8 ns); 72-73 |
| TIP3P, point charges (2 x 1.95 ns) | 101.1 +- 4.3 | 93.7 +- 1.7 | 88.1 +- 0.5 | 72.2 +- 0.2 | **96.8 +- 2.0** (|E| <= 0.1, c = 871 +- 210, chi2 1.5 / 1) | 99.1 +- 3.2, 100.3 +- 2.8 | 103.8 +- 3.0 (pmemd pipeline 97.3 +- 0.8; literature 94-104, 94 in Vega & Abascal 2011) |
| TIP3P, OpenMM 8.2 (1.45 ns per replica; 1.95 at 0.05) | | 95.9 +- 1.7 | 88.4 +- 0.7 | 71.3 +- 0.2 | | | |

- **The finite-field and fluctuation estimates agree** for all three models, within the error bars,
  and the fluctuation estimate of the zero-field copies of each batch agrees with the long references.
- **Linear regime.** pGM3P-25 is nearly linear up to 0.2 V/nm (c = 29 +- 16 (V/nm)^-2: -3 +- 2 % at
  0.2 V/nm, -1 % at 0.1), base pGM saturates by 4 % at 0.1 V/nm and 13 % at 0.2, TIP3P by ~2 % at 0.05 V/nm (fit), 9 % at 0.1 and 25 % at 0.2 (beyond the quadratic term: the fit up to 0.2 gives
  93.8 +- 0.6 with chi2 4.0 / 2, so the fit is restricted to |E| <= 0.1). The saturation follows
  the orientational polarization per molecule, <cos theta> ~ <M.e> / (N mu): 0.24 for pGM3P-25 at
  0.2 V/nm (whose induced dipoles respond linearly), 0.2-0.3 for base pGM and TIP3P at 0.1 V/nm. For
  eps ~ 70-100 fields of 0.02-0.05 V/nm are linear within the errors; the fit eps0 - c E^2 over several
  |E| uses the larger, more precise fields.
- **Independent code.** OpenMM 8.2 (CPU; the same prmtop, box and density, SETTLE, PME, LangevinMiddle
  1/ps, CustomExternalForce -q E z) gives the same response of TIP3P: 95.9 +- 1.7 at +-0.05 V/nm (pgm_jax 93.7 +- 1.7), 88.4 +- 0.7 at +-0.1 V/nm
  (pgm_jax 88.1 +- 0.5) and 71.3 +- 0.2 at +-0.2 V/nm (pgm_jax 72.2 +- 0.2 at 2 fs, 71.8 +- 0.3 at
  1 fs, 1.4 sigma from OpenMM: part of the 3-sigma difference at 2 fs is the time step of the
  rigid-body integrator, which matters in the deeply saturated regime; OpenMM: SETTLE +
  LangevinMiddle at 2 fs, 1.45 ns per replica).

Cost at equal error. Measured errors of the finite-field estimates against the measured error of
the fluctuation formula for the same total MD time (scatter of eps over 1-ns segments of the 10 ns
references: 2.19 for pGM3P-25, 1.87 for base pGM, 9.13 for TIP3P, scaled by 1/sqrt(time)):

| Model, estimator | MD time | error | fluctuation error at that MD time | finite field cheaper by |
|---|---|---|---|---|
| pGM3P-25, pair +-0.2 V/nm (saturation -1.2 +- 0.6 not removed) | 3.9 ns | 0.24 | 1.11 | 21x |
| pGM3P-25, pair +-0.1 V/nm (saturation -0.3 +- 0.15) | 3.9 ns | 0.47 | 1.11 | **5.6x** |
| pGM3P-25, fit over +-0.1 and +-0.2 (4 copies, saturation removed) | 7.8 ns | 0.63 | 0.78 | 1.5x |
| base pGM, pair +-0.05 V/nm | 1.9 ns | 1.02 | 1.36 | 1.8x |
| base pGM, fit over +-0.02..0.2 (8 copies) | 7.6 ns | 0.40 | 0.68 | 2.9x |
| TIP3P, pair +-0.05 V/nm (saturation -2.2 +- 0.5 not removed) | 3.9 ns | 1.7 | 4.6 | 7.7x |
| TIP3P, fit over +-0.02..0.1 (6 copies) | 11.7 ns | 2.0 | 2.7 | 1.8x |

The gain is (<M.e> / sd(M_e))^2 / 3 (module docstring of finite_field.py): large for a moderate eps
that stays linear to large fields (pGM3P-25), small when saturation limits the field (TIP3P) or when M
decorrelates fast (the base pGM water: tau_M ~ 1 ps against 6-8 ps for the other two models, which
makes its fluctuation estimate unusually cheap). At fixed field the gain grows with the box volume.

Constant D (pGM3P-25, D/eps0 = +-3.3 and +-6.6 V/nm, 4 copies, 2 x 0.45 ns per |D| after 50 ps;
eps = D / (D - <M.e>/(eps0 V))): the Maxwell field settles at <E> = 0.095 and 0.205 V/nm, and

| D/eps0 (V/nm) | <E> (V/nm) | eps (pair) | tau of M (ps) |
|---|---|---|---|
| +-3.3 | 0.095 | 34.6 +- 0.8 | 0.3 |
| +-6.6 | 0.205 | 32.4 +- 0.5 | 0.3 |

in agreement with the constant-E results at the same Maxwell fields (33.9 +- 0.5, 33.2 +- 0.2). At
constant D the polarization relaxes with the longitudinal time (0.3 ps against 5-7 ps) and its
fluctuations are suppressed by ~eps, but eps = D/(eps0 <E>) amplifies the error of <M> by eps^2 / D:
per ns of MD the error is the same as at constant E with the same Maxwell field (0.50 in 0.9 ns here,
0.50 for +-0.2 V/nm scaled to 0.9 ns).

## Speed

The field adds O(N) work per force evaluation (q E on the forces, E on the right-hand side and in
the dipole-frame pull-back, the sum M); constant D adds one O(N) sum per CG iteration. On one RTX PRO
6000 Blackwell (512 pGM3P-25 waters, rigid bodies, 2 fs, NVT Bussi, mixed precision, dipole tol 1e-5,
`validate_efield.py speed`):

| Run | ms/step | ns/day | CG iterations |
|---|---|---|---|
| no field | 0.769 | 225 | 6.00 |
| static E, 0.1 V/nm | 0.812 | 213 | 6.00 |
| E(t), 0.1 V/nm, 200 cm^-1 | 0.820 | 211 | 6.00 |
| constant D, D/eps0 = 3 V/nm | 0.878 | 197 | 6.00 |
| no field / static E, alternating, 3 x 5,000 steps each | 0.802, 0.803, 0.806 / 0.808, 0.808, 0.809 | +0.6 % | 6.00 |
| one force evaluation, no field / static E | 0.557 / 0.578 ms | | |
| `FieldReplicas` x1 / x2 / x4 / x10 (M every 25 steps) | 1.01 / 1.10 / 1.51 / 3.89 | 172 / 158 / 114 / 45 per copy (172 / 315 / 457 / 445 aggregate) | 6.0 |

The single-shot rows are sequential runs (the first is favoured by the GPU's state); the alternating
repeats give the cost of a static field: 0.005 ms per step (+0.6 %), no extra CG iteration. The first
version cost 0.10-0.16 ms per step (+13-20 %): it summed M in two float64 column reductions and did so
twice per force evaluation (the compiled step had 8 more fusions and 4 more reductions than the
field-free one); M is now one reduction, computed once (4 and 2 more). Constant D adds the all-to-all
term, one float64 sum of the dipoles, to every CG iteration (+0.1 ms, +14 %). Batching pays up to
about 4 copies of 512 waters (457 ns/day aggregate, 2.7x one copy).

On the CPU (32 cores, 1 fs NVE, the runs of the NVE table): 3.14 ns/day without a field, 3.10, 3.08,
3.11 and 3.13 ns/day with 0.1 V/nm, 0.5 V/nm, E(t) and constant D. Without a field the compiled step
is the field-free one (bitwise identical trajectories to master).

Production runs of the validation (10 copies of 512 waters, M every 25 steps, 2 fs, first version of
the field code): 42 ns/day per copy (420 aggregate) for the pGM waters, 73 (729 aggregate) for TIP3P.

## Limits

- **Uniform fields only**, acting on the point multipole moments of the Gaussian densities (exact for
  a uniform field); no field gradients. The covalent quadrupoles are not in the MD engine (gas phase
  only), and a uniform field exerts neither force nor torque on a quadrupole.
- **Refused combinations** (explicit errors): an alchemical region with a field (the field would act
  on the unscaled solute charges); NPT with a field and charged molecules; `FieldReplicas` with NPT,
  MTS or charged molecules; path integrals (`md/pimd.py`) and `PGMEngine.from_simulation` (the
  external-code interfaces) with a field.
- **Other features** (docs/CHANGES_2026-09.md, `tests/test_integration.py`): extended-Lagrangian
  dipoles (docs/iel.md) take the field in the auxiliary-dipole step and the shadow energy (constant E
  and D); biases on collective variables add to the field forces; walkers book the work of E(t) as a
  single simulation does. `PGMForceField.strain_derivative` with a field returns the full tensor.
- **Charged molecules**: the itinerant dipole of re-wrapped ions is booked by the `Simulation` /
  `FlexibleSimulation` drivers only; the batched replica engines (replica exchange, field replicas)
  re-wrap without booking it (for an electrolyte the "dielectric constant" also contains the
  conduction current, which the finite-field formula does not separate).
- **Linear response** has to be checked: TIP3P and the base pGM water saturate measurably at
  0.1 V/nm; run several |E| and fit (or stay at |E| <= 0.05 V/nm for eps ~ 100).
- **Constant D** is validated at the level of forces, the virial, linear response at fixed nuclei,
  energy conservation and the dielectric constant of pGM3P-25; NPT at constant D is implemented
  (the volume dependence of the D term is in the virial and the trial energies) but was not run.
- **Time-dependent fields**: energy bookkeeping validated (econs conserved while the field pumps
  10^3 kJ/mol into the box); absorption spectra from E(t) runs were not compared with the IR spectrum
  of the zero-field M(t) correlation. The gas-phase `ElecChannel` takes a static field only.
- The field amplitude is a traced state variable, the frequency and phase are static (a new omega
  recompiles the step).
