# pGM-JAX clean-up: survey and API design

Status: accepted with the owner's decisions (section 9, which overrides the recommendations of
the earlier sections where they differ); being executed phase by phase on branch `cleanup`
(from master `e72c57c`). P0 added this document, the regression harness (`tests/regression/`),
the GPU speed check (`scripts/dev/gpu_bench.sh`) and the ruff configuration.

Scope chosen by the owner: clean-up plus API redesign. Breaking API changes are allowed, and
scripts, tests and docs are updated in the same commits. Lint and format with ruff (isort,
pyflakes, pyupgrade; line length 120). Physics results must not change: MD stays bitwise
identical wherever the change does not touch numerics, and jitted hot paths must not get slower.

Contents: 1 Inventory, 2 Problems, 3 Target design, 4 Migration table, 5 Phased plan,
6 Risks, what is not changed, open questions, 7 Regression harness, 8 Ruff report,
9 Decisions.

---------------------------------------------------------------------------------------------------

## 1. Inventory

### 1.1 Package layout (`pgm_jax/`, 88 files, 22,611 lines)

| Module | Lines | Responsibility |
|---|---:|---|
| `__init__.py` | 27 | Re-exports the gas-phase and Ewald core (`Model`, `PeriodicModel`, `ElecChannel`, `System`, ...) |
| `system.py` | 318 | `Molecule`, `ParamTable` (tied parameters), `System` (topology + index arrays) |
| `param.py` | 310 | Parameter sources: pGM prmtop reader (`read_prmtop_pgm`), PyRESP files, pol table, molecule JSON |
| `prmtop.py` | 177 | Generic prmtop section reader/writer (`Prmtop`) |
| `channels.py` | 210 | Gas-phase pGM electrostatics channel (`ElecChannel`), polarizability, SAPT-like decomposition |
| `lj.py`, `vdw.py` | 85, 172 | LJ and GVDW channels, gas and periodic |
| `ewald.py`, `periodic.py` | 193, 98 | Reference Ewald pGM (`PeriodicPGM`), `PeriodicModel` (energy, virial, pressure) |
| `model.py` | 81 | `Model`: list of channels, n-body decomposition |
| `kernels.py`, `multipole.py`, `solver.py` | 78, 152, 71 | Density kernels, Gaussian quadrupoles, variational induction solve |
| `options.py`, `units.py` | 30, 16 | Model option names (`elec`, `vdw`), a few constants |
| `ensemble.py` | 103 | Reweighting of ensemble averages (protein force-field refinement: J couplings) |
| `qmfit.py` | 500 | Fitting pGM parameters to QM cluster data (`QMSet`, `ClusterModel`, `ParamMap`, `QMFit`) |
| `md/forcefield.py` | 1394 | `MDSettings` and `PGMForceField`: PME, pair rows, CG / predictor / iEL induction, virial (hot path) |
| `md/pme.py`, `md/kernels.py`, `md/neighbors.py`, `md/box.py` | 177, 62, 162, 96 | PME, erf kernels, JAX-MD neighbour lists, box helpers |
| `md/integrate.py`, `md/rigid.py`, `md/thermostats.py` | 453, 143, 196 | Rigid-body integrator (`Integrator`, `MDState`), rigid bodies, Langevin / Bussi / GLE |
| `md/simulation.py` | 430 | `Simulation` driver (rigid molecules; Amber I/O, blocks, logs, checkpoints) |
| `md/flexible.py` | 811 | `FlexibleTemplate`, `RigidTemplate`, `FlexibleIntegrator`, `FlexibleSimulation`, `liquid_box` |
| `md/constraints.py`, `md/topology.py`, `md/vsites.py`, `md/flux.py` | 468, 187, 481, 262 | SHAKE/RATTLE, pair topology, virtual sites, charge flux |
| `md/mts.py`, `md/iel.py` | 678, 94 | r-RESPA integrators; iEL CLI helpers (engine in forcefield.py) |
| `md/restraints.py`, `md/efield.py` | 412, 201 | Restraints; external fields |
| `md/pimd.py` | 1088 | Ring polymer, PILE, `PIMDSimulation`, plus the q-TIP4P/F flexible-water model builder |
| `md/remd.py` | 667 | `MDReplicas` (batched / sequential replica engine), `ReplicaExchange` |
| `md/finite_field.py` | 314 | `FieldReplicas` (+-E copies) and the finite-field analysis |
| `md/alchemy.py`, `md/free_energy.py`, `md/fe_grad.py` | 844, 360, 473 | Alchemical Hamiltonian, `LambdaWindows`, `FreeEnergyRun`; TI/BAR/MBAR; dDeltaG/dtheta |
| `md/dipoles.py`, `md/dielectric.py`, `md/io.py` | 371, 180, 179 | Cell dipole and its files; dielectric analysis; Amber NetCDF / restart I/O |
| `md/_jaxmd.py` | 24 | Imports JAX-MD's simulation core without flax |
| `bias/` (7 files) | 1,855 | CVs, MetaD / OPES / static biases, `Walkers`, COLVAR/HILLS I/O, FES analysis, toy Langevin |
| `bonded/` (19 files) | 3,226 | Bonded topology, term families (classical, class II, CMAP, explored), `BondedModel`, fitting, Amber import/export, neural bonded terms, study data (`bench.py`, `data.py`, `molecules.py`, `terms/explore.py`) |
| `fit/` (6 files) | 1,189 | Liquid fitting: `ParameterSpace`, `FrameAnalyzer`, `LiquidSamples`, `Objective`, `LiquidFit` |
| `interfaces/` (5 files) | 1,510 | `PGMEngine` (device-resident force calls), ASE calculator, i-PI client, OpenMM `PythonForce` |
| `protein/` (5 files) | 1,001 | `load_amber` / `AmberSystem`, residue library, `write_pgm_prmtop`, `pmemd_mdin` |

### 1.2 Public entry points

| Entry point | Where | Role |
|---|---|---|
| `Molecule`, `ParamTable`, `System` | `system.py` | Topology, tied parameters (`sys.table.initial()`, `sys.params0`) |
| `Model`, `ElecChannel`, `LJChannel`, `GVDWChannel` | `model.py`, `channels.py`, `lj.py`, `vdw.py` | Gas phase: energies, forces, n-body, parameter gradients |
| `PeriodicModel`, `PeriodicPGM`, `pressure_bar`, `strain_derivative` | `periodic.py`, `ewald.py` | Reference Ewald model, differentiable in box and parameters |
| `read_prmtop_pgm`, `load_molecule`, `save_molecule` | `param.py` | Parameter input |
| `MDSettings`, `PGMForceField`, `elec_cutoff_settings` | `md/forcefield.py` | MD force field (32 settings fields), `compute()` / `rows_for()` / `strain_derivative()` |
| `Simulation(sys, pos_nm, H_nm, settings, dt, ensemble, temperature, gamma, pressure, barostat_interval, seed, vel_nm_ps, params, log, neighbor_list, thermostat, tau_t, restraints, alchemy, mts, bias, efield)` | `md/simulation.py:54` | Rigid-body MD, `run()`, `save()`, `load()`, `observables()`, `pressure()` |
| `FlexibleSimulation(sys, templates, pos_nm, H_nm, ..., r_margin, constraints, hmr, max_single, constraint_options)` | `md/flexible.py:565` | Atomistic MD (bonded terms, constraints, vsites, flux), `minimize()` |
| `FlexibleTemplate`, `RigidTemplate`, `liquid_box` | `md/flexible.py` | Molecule templates for the flexible engine |
| `MTS`, `Restraints` + kinds, `ExternalField` / `displacement`, `VirtualSite` | `md/mts.py`, `md/restraints.py`, `md/efield.py`, `md/vsites.py` | Options passed to the engines |
| `PIMDSimulation(sim, beads, mode, thermostat, tau0, lam, propagator, contract, bead_margin, seed, dt, spread, ensemble, pressure, barostat_interval, bead_chunk, log)` | `md/pimd.py:703` | Path-integral MD on a `FlexibleSimulation` |
| `MDReplicas`, `ReplicaExchange`, `geometric_ladder` | `md/remd.py` | Temperature REMD (batched with vmap or sequential) |
| `FieldReplicas`, `read_series`, `analyse` | `md/finite_field.py` | Finite-field dielectric constant |
| `Alchemy`, `alchemical_system`, `LambdaWindows`, `FreeEnergyRun`, `GasPhaseLeg`, `standard_schedule` | `md/alchemy.py` | Alchemical free energies |
| `estimate`, `mbar`, `bar`, `ti`, ... | `md/free_energy.py` | Estimators |
| `ParamSpace`, `ParamGradients`, `gradient_estimate`, `FreeEnergyTarget` | `md/fe_grad.py` | Free-energy parameter gradients |
| `MetaD`, `OPES`, `Harmonic`, walls, `BiasSet`, `cv.*`, `Walkers` | `bias/` | Enhanced sampling |
| `ParameterSpace`, `FrameAnalyzer`, `LiquidSamples`, `GasPhase`, `Objective`, `Target`, `LiquidFit` | `fit/` | Liquid-property fitting |
| `QMSet`, `ClusterModel`, `ParamMap`, `FitWeights`, `QMFit` | `qmfit.py` | QM cluster fitting |
| `BondedModel`, `BondedSettings`, `MolSpec`, `terms.*`, `NNB` | `bonded/` | Bonded terms and their fit |
| `PGMEngine`, `GasPhaseEngine`, `PGMCalculator` (ASE), i-PI client, OpenMM force | `interfaces/` | External MD codes |
| `load_amber`, `AmberSystem`, `amber_template`, `ResidueLibrary`, `write_pgm_prmtop`, `pmemd_mdin`, `pmemd_grid` | `protein/` | Protein pipeline and pmemd-pgm export |
| `Reweighting`, `karplus` | `ensemble.py` | Reweighting of experimental observables |

### 1.3 Scripts (77 files, 12,474 lines) and examples (4)

| Group | Scripts |
|---|---|
| Production drivers | `run_md.py` (Amber-style CLI), `solvation_free_energy.py`, `finite_field.py`, `pimd_water.py`, `fit_liquid.py`, `fit_multi.py`, `protein/write_pgm_prmtop.py`, `protein/build_amber.py`, `pgm3p25_prmtop.py`, `pgm_supercell.py` |
| Analysis | `dielectric.py`, `trajectory_dipoles.py`, `liquid_fit_tools.py`, `bias/ala2_analyze.py` |
| Validation vs Amber / OpenMM / i-PI / ASE / theory | `validate_*.py` (8), `shake_vs_pmemd.py`, `pimd_validate.py`, `pimd_openmm.py`, `openmm_tip3p_field.py`, `iel_validate.py`, `water_dielectric.py`, `validate_eps_gradient.py`, `fe_gradient_check.py`, `efield_identical.py`, `md_gvdw_water.py`, `flux_md.py`, `bias/{validate_toy,engine_dw,nve_check,ala2}.py`, `interfaces/validate_{ase,ipi,openmm}.py`, `protein/{check_pgm_prmtop,elec_accuracy,remd_peptide}.py` |
| Benchmarks | `bench.py`, `bench_md.py`, `bench_shake.py`, `iel_cost.py`, `interfaces/bench_interfaces.py`, `protein/bench_protein.py`, `protein/cg_diag.py`, `bonded/bench_sets.py` |
| QM data | `qmfit/{build_water_clusters,psi4_clusters,collect_water_qm,fit_water_qm,smith_opt}.py` |
| Bonded study (research record) | 24 files in `scripts/bonded/` (experiments, LOO tables, MACE sampling, DFT labels, report figures) |
| Examples | `examples/{fit_bonded_template,run_flexible_liquid,flex_methanol_check,hydration_target}.py` |

43 of 77 scripts have a `__main__` guard, 60 insert the repo into `sys.path`, 34 use argparse.

### 1.4 Tests

35 files, 291 test functions, 356 collected tests (8,644 lines), flat in `tests/`, plus
`tests/data/` (tleap prmtops: peptide in TIP3P / TIP4P-Ew, small TIP4P-Ew / TIP5P boxes; 424 kB).
Four tests depend on files outside the repository (`~/pgm-gvdw-data`, pmemd, i-PI, OpenMM) and skip
without them. Test modules import helpers from each other (`from test_grad import water` in 12
files, `test_md_macro._water_box` in 6, `test_md.small_box/settings` in 8, `test_hmr._cluster` in 3,
...). New in this branch: `tests/regression/` (section 7).

### 1.5 Documentation and repository root

`README.md` (400 lines: overview, MD feature list with numbers, validation and speed tables);
`docs/` 20 files (feature docs, how-tos, `CHANGES_2026-09.md`); 9 `NOTES_*.md` in the root
(branch work notes of the September merge); `paper/` (LaTeX); tracked result directories
`validation/` (1.7 MB: Amber reference outputs and validation JSON), `reports/bonded/` (figures),
`data/` (QM sets, bonded study data); `examples/`; untracked `runs/` (ignored).

---------------------------------------------------------------------------------------------------

## 2. Problems found

### 2.1 Parameter names and units differ between engines and classes

| Concept | Spellings found (file:line) |
|---|---|
| Temperature | `temperature` (`Simulation`, `md/simulation.py:55`; `ReplicaExchange` uses `temperatures`), `T` (`LiquidFit`, `fit/liquid.py:52`; `LiquidSamples`, `fit/estimators.py:54`; `md/dielectric.py:49-162`; `md/finite_field.py:306`), `kT` (`bias/analysis.py:54-129`), `temperature=None` meaning "bind later" (`bias/core.py:57`), `self.T0` / `self.T` attributes |
| Thermostat coupling | `gamma` (1/ps) and `tau_t` (ps) as separate engine arguments (`md/simulation.py:55-57`), `tau` in `Bussi(tau)` and `make_thermostat(spec, gamma, tau)` (`md/thermostats.py:91,182`), `tau0` + `lam` + `thermostat="pile-l"` in PIMD (`md/pimd.py:703`); the unused one of `gamma` / `tau_t` is silently ignored |
| Ensemble | `ensemble="nve"|"nvt"|"npt"` plus `thermostat=` plus `pressure=` and `barostat_interval=`: four arguments for two choices; `ensemble="nve"` with the default `thermostat="langevin"` is legal and ignores the thermostat |
| Pressure | `pressure` (bar, constructors), `pressure_bar` (`fit/estimators.py:54`), `pressure: bool` = "report the pressure" in `PIMDSimulation.run` (`md/pimd.py:880`), `pressure_every_report` in `Simulation.run` (`md/simulation.py:344`) |
| Time step and time | `dt` in ps in the library (default 0.001 in `Simulation`, 0.0005 in `FlexibleSimulation`, `md/flexible.py:566`; `None` = from `sim` in PIMD); `*_ps` durations in `LiquidFit` / fe estimators; CLIs `--dt` in fs in 10 scripts (`run_md.py:55`, `iel_validate.py:68`, `solvation_free_energy.py:459`, `pimd_water.py:280`, `finite_field.py:160`, ...) and in ps in 12 (`bench_md.py:48`, `bias/ala2.py:49`, `protein/bench_protein.py:39`, `interfaces/validate_ase.py:94`, ...) |
| Positions / box / velocities | `pos_nm`, `H_nm`, `vel_nm_ps` (engines), `pos`, `H` (force field, `PeriodicModel`), `positions`, `cell` (interfaces), `xyz` (I/O, Angstrom or nm), `box` (state), `X`, `R` |
| Cutoffs / Ewald / solver | `MDSettings(cutoff, ewald_beta, dipole_tol)` vs `PeriodicModel(rc, b0, lj_rc, k_tol, cg_tol)` (`periodic.py:51`); `tol` in `FrameAnalyzer` / `LiquidFit`; CLIs `--cut` in Angstrom (`run_md.py:63`, `trajectory_dipoles.py:69`) and in nm (`bench_md.py:42`, `solvation_free_energy.py:462`), `--cutoff` (`fit_multi.py:99`), `--es-cut` / `--elec-cut`, `--tol` / `--dipole-tol` / `--md-tol` |
| Beads | `beads` (`PIMDSimulation`), `nbeads` (`RingPolymer`, `PIMDIntegrator`), attribute `P` |
| Output prefix / intervals | `prefix` (engines), `out` (17 CLIs), `-o` (9), `--prefix` (2); `report` (steps) in engines vs `--report` in ps (`bias/ala2.py:67`, `protein/remd_peptide.py:53`) and `--report-ps`; `restart` (steps between checkpoints) vs `restart=` meaning "continue" elsewhere; `every` (FieldReplicas sampling) vs `sample_every`, `exchange_every` |
| Logging | `log=sys.stdout` default in `Simulation`, `PIMDSimulation`, `ReplicaExchange`, `LiquidFit`, i-PI client; `log=None` default in `FlexibleSimulation` (`md/flexible.py:567`); `log=None` is passed 141 times in the tests |
| Seeds | `seed` everywhere (good), but PRNG streams are derived differently per driver (`PRNGKey(seed)`, `random.split(PRNGKey(seed), n)`, numpy `SeedSequence([seed, 0x5EED])` in REMD) - document, do not change (would change trajectories) |

### 2.2 `run()`, `save()` and `load()` are similar but not the same

| Driver | Signature |
|---|---|
| `Simulation.run` (`md/simulation.py:343`) | `(nsteps, report=1000, traj=0, restart=0, prefix="md", pressure_every_report=False, append=False, dipoles=0, induced=0)` |
| `PIMDSimulation.run` (`md/pimd.py:879`) | `(nsteps, report=100, traj=0, beads_traj=0, restart=0, prefix="pimd", append=False, pressure=False)` |
| `ReplicaExchange.run` (`md/remd.py:561`) | `(nsteps, report=1000, traj=0, restart=0, prefix="remd" or None, append=False)` |
| `FieldReplicas.run` (`md/finite_field.py:104`) | `(nsteps, every=25, prefix="ff", report=5000, append=False, restart=0, extra=None, log=None)` |
| `Walkers.run` (`bias/walkers.py:224`) | `(nsteps, report=1000, restart=0, prefix="walkers", append=False, ...)` |
| `FreeEnergyRun.run` (`md/alchemy.py:718`) | `(nsteps, prefix="fe", report=0, restart=0)` - `prefix` is the 2nd positional argument here, the 5th in `Simulation.run` |

Checkpoints: six pickle formats (`prefix.chk` without a format tag, `md/simulation.py:407`;
`prefix.pimd.chk` with `{"format", "beads"}`, `md/pimd.py:934`; `.remd.chk`, `.fe.chk`, `.ffchk`,
`.walkers.chk`); `save(prefix)` in most drivers, `save(path)` in `FieldReplicas.save`
(`md/finite_field.py:144`) and `Walkers.save` (`bias/walkers.py:270`).

Observables: the key sets differ per engine and are documented nowhere in one place
(`temp_trans`/`temp_rot` in `Simulation`, renamed to `temp_com`/`temp_internal` by
`FlexibleSimulation.observables`, `md/flexible.py:727`; `temp_K`, `temp_centroid`, `ekin_cv`,
`ke_<el>_cv_meV` in PIMD).

### 2.3 `MDSettings` option sprawl

`MDSettings` (`md/forcefield.py:114-157`) is one frozen dataclass with 32 fields covering six
unrelated concerns: the model (`elec`, `vdw`, `gvdw_rep`, `lj_lrc`), cutoffs and the list
(`cutoff`, `elec_cutoff`, `skin`), PME (`ewald_beta`, `pme_grid`, `pme_spacing`, `pme_order`), the
SCF solver and predictor (`dipole_tol`, `max_iter`, `predictor`, `fused`, `norm_refresh`,
`local_cut`, `local_niter`, `peek`, `extrap_order`, `extrap_steps`), the extended Lagrangian (8
`iel_*` fields) and numerics (`precision`, `differentiable`, `adjoint_tol`). Neighbour-list options
live elsewhere (`neighbor_list=` in the engine constructors, `r_margin` in `FlexibleSimulation`,
`bead_margin` in PIMD and `PGMEngine`). The fields are only documented as trailing comments with
pmemd names; combinations that are refused are checked in several places.

### 2.4 Duplicated MD-loop, state, checkpoint and logging logic

- Run loop (block size = gcd of the intervals, advance, report, write frames, checkpoint, ns/day)
  written six times: `md/simulation.py:343-396`, `md/pimd.py:879-922`, `md/remd.py:561-625`,
  `md/finite_field.py:104-142`, `bias/walkers.py:224-268`, `md/alchemy.py:718-777`
  (`np.gcd.reduce` in five of them); the log-line format (`"# " + " ".join(f"{c:>14s}")`, `%14.6f`)
  copied in `md/simulation.py:382`, `md/pimd.py:905`, `md/remd.py:602`.
- Block advance with overflow detection, resize, recompile and repeat ("for attempt in range(6)"):
  `md/simulation.py:301-340`, `md/pimd.py:784-816`, `md/remd.py:400-440`, and twice in
  `interfaces/engine.py:492,728` (8 attempts); the "rebuild lists if the volume drifted by 10 %"
  logic in `md/simulation.py:287` and `md/pimd.py:770`.
- `FlexibleSimulation` subclasses `Simulation` but does not call `Simulation.__init__`
  (`md/flexible.py:565-627` repeats the construction and the log header), and overrides
  `_make_neighbors`, `_size_lists`, `_pressure`, `load` with near copies
  (`md/simulation.py:219-284,420-430` vs `md/flexible.py:673-766`); the pressure formula with the
  literal `16.605390671738466` appears in both.
- Checkpoint pickling/unpickling six times (2.2). `_print(self, s)` defined in 6 classes.
- Test-only access to internals because the public API has no "advance n steps without files":
  `sim._advance(n)` is called 67 times in 21 scripts and 89 times in 19 test files; scripts import
  the private `simulation._dedupe` (17 files) to build a `System` from a prmtop.

### 2.5 Duplicated helpers

- Constants: `KB` defined in `ensemble.py:32`, `fit/estimators.py:42`, `md/integrate.py:49`,
  `bias/core.py:39`; kJ/mol/nm^3 <-> bar in `periodic.py:16`, `fit/estimators.py:45`,
  `md/pimd.py:89`, `md/integrate.py:50` and as a literal in `md/simulation.py:229`,
  `md/flexible.py:750`; amu/nm^3 -> g/cm^3 in `md/simulation.py:39`, `fit/estimators.py:44`,
  `md/pimd.py:832`, `md/flexible.py:779` and 7 scripts; `units.py` exists but is imported by 9
  modules only.
- Boxes: `ewald.box_matrix` (`ewald.py:45`) and `md/io.box_from_cell` (`md/io.py:68`) build the
  same matrix; `md/box.py` (reduce, volume, inv3, min_image), `interfaces/engine.standard_cell`;
  `min_image` re-implemented in `scripts/iel_validate.py:47`, `scripts/qmfit/build_water_clusters.py:105`.
- Molecular centres of mass (`segment_sum(w * pos) / segment_sum(w)`) in `periodic.py:28`,
  `md/forcefield.py:1383`, `md/alchemy.py:416`, `md/restraints.py:397`, `md/flexible.py:325`.
- Statistics: `jackknife_cov` (`fit/estimators.py:159`), `jackknife_error` (`md/fe_grad.py:265`),
  `jackknife` (`md/dielectric.py:67`), `block_mean` (`md/finite_field.py:196`), two
  `correlation_time` (`md/dielectric.py:147`, `md/finite_field.py:204`),
  `statistical_inefficiency` (`md/free_energy.py:38`), `block_err` in two scripts.
- Parameter subsets as a flat vector: three abstractions for one idea: `fit.ParameterSpace`
  (`fit/params.py:46`, scale/shift of tied keys), `fe_grad.ParamSpace` (`md/fe_grad.py:67`),
  `qmfit.ParamMap` (`qmfit.py:264`, with neutrality constraints).
- prmtop parsing twice: `param._prmtop_sections` (`param.py:28`) and `prmtop.Prmtop`.
- Dipoles: cell / molecular dipole code in `md/dipoles.CellDipole`, `fit/frames._mol_dipoles`
  (`fit/frames.py:144`), `PIMDSimulation.molecular_dipoles` (`md/pimd.py:855`),
  `interfaces/engine._dipole_only` (`interfaces/engine.py:392`), `qmfit.ClusterModel.monomer_dipole`.
- CLI helpers inside the library: `md/iel.py:17 add_iel_arguments`, `md/mts.py:655 add_mts_arguments`.

### 2.6 Dead code, legacy paths, surprises

- Compatibility shims: `md/integrate.upgrade_state` (old checkpoints, `md/integrate.py:93`, used
  by both engines' `load`), `Neighbors = AtomNeighbors` (`md/neighbors.py:161`), molecules pickled
  before GVDW/quadrupoles (`system.py:111`), `PeriodicModel(lj=...)` duplicating `vdw="none"`
  (`periodic.py:51`), `bonded/nn/model.py:266` compatibility block.
- Unused variables in library code (pyflakes F841): `fit/liquid.py:108` (`dens`), `fit/liquid.py:179`
  (`s`), `interfaces/openmm.py:82` (`eng`).
- The default thermostat is Langevin (`md/simulation.py:57`) although `md/thermostats.py:34` calls it
  "kept for compatibility" and recommends Bussi.
- A user path in library code: `param.py:205` `PGM_POL_TABLE = ~/amber25/...`; hard-coded
  `/home8/larry/...` paths in `scripts/qmfit/{build_water_clusters,fit_water_qm}.py`.
- Research-study code shipped in the library: `bonded/bench.py`, `bonded/data.py`,
  `bonded/molecules.py`, `bonded/terms/explore.py` (488 lines of explored families).

### 2.7 Long functions and constructors

Functions over 100 lines: `protein/pmemd.write_pgm_prmtop` (154), `bonded/amber.export_bonded`
(144), `interfaces/engine.PGMEngine.compute_batch` (130), `protein/amber.load_amber` (113); 24
over 60 lines. Most arguments: `LiquidFit.__init__` 33, `FlexibleSimulation.__init__` 29,
`Simulation.__init__` 23, `pmemd_mdin` 20, `PIMDSimulation.__init__` 18, `Integrator.__init__` 17,
`PeriodicModel.__init__` 15, `PGMEngine.__init__` 14. Biggest modules: `md/forcefield.py` (1394),
`md/pimd.py` (1088), `interfaces/engine.py` (845), `md/alchemy.py` (844), `md/flexible.py` (811).

### 2.8 Docstrings and type hints

AST count over `pgm_jax/` (functions and methods whose name has no leading underscore, including
nested closures): 1,115, of which 573 have no docstring; 66 of 187 classes have none. Of 2,707
parameters 927 are annotated (34 %), 335 functions annotate the return. Docstrings are prose
(often excellent physics, units and references) but have no common structure: parameters, units
and return values are described in running text, so the units of an argument are hard to find.

### 2.9 Module placement

- `qmfit.py` is top-level, while the other fitting code is in `fit/` and `bonded/fit.py`.
- `ensemble.py` (reweighting for protein refinement) is top-level and its name clashes with the MD
  meaning of "ensemble".
- `md/` mixes the engine with analysis (`dielectric.py`, `free_energy.py` estimators, the analysis
  half of `finite_field.py`), fitting targets (`fe_grad.FreeEnergyTarget`) and model building
  (`pimd.flexible_water`, q-TIP4P/F, `pimd.py:951-1088`).
- `kernels.py` (density kernels, gas phase) and `md/kernels.py` (erf kernels, MD) have the same name.
- Reference implementations (`ewald.py`, `periodic.py`) and the production MD force field are not
  labelled as such.

### 2.10 Scripts, repository root

- Flat `scripts/` with production drivers, validations, benchmarks and one-off checks side by side;
  outputs default to `runs/...` or `validation/...` paths relative to the working directory.
- `sys.path.insert` in 60 scripts instead of an installed package; 34 scripts run at import (no
  `__main__` guard).
- Nine `NOTES_*.md` in the root, `reports/`, `validation/`, `examples/`, `paper/`, `data/` side by side;
  `docs/` has no index.

---------------------------------------------------------------------------------------------------

## 3. Target design

### 3.1 Principles

1. Evolutionary: move and rename, extract duplicated host-side code into shared helpers, and keep
   every jitted function (`PGMForceField` kernels, CG, PME, integrator steps, constraint solvers,
   bias deposition) and its order of floating-point operations as it is.
2. One concept, one name, one unit (3.3), everywhere: library, CLI, logs, docs.
3. Each engine is configured by the same small set of objects: `MDSettings` (force field),
   thermostat, barostat, and the run's output options (3.4, 3.5).
4. Public means documented and tested: whatever scripts or tests need (for example advancing
   without writing files) becomes public; nothing outside a module uses its `_private` names.
5. Every phase is small and verified by the full test suite, the regression harness (bitwise) and,
   when `md/` changes, the GPU speed check (section 5).

### 3.2 Package layout (target)

```
pgm_jax/
  __init__.py          core re-exports (unchanged set + System.from_prmtop)
  units.py             ALL physical constants and conversions (KB, BAR, AMU_NM3_G_CM3, KE, ...)
  system.py  param.py  prmtop.py (single prmtop parser)  options.py
  channels.py lj.py vdw.py model.py multipole.py solver.py      (gas-phase model: stay where they are)
  densities.py         (was kernels.py; avoids the clash with md/kernels.py)
  ewald.py periodic.py (reference Ewald model; docstrings say "reference, not the MD engine")
  md/
    forcefield.py      PGMForceField (hot path, unchanged numerics)
    settings.py        MDSettings and its groups (3.4)
    pme.py kernels.py neighbors.py box.py topology.py constraints.py vsites.py flux.py
    integrate.py rigid.py thermostats.py barostat.py(new: MC barostat config) mts.py restraints.py efield.py iel.py
    engine.py          (new) MDEngine base: construction, advance/resize, observables, pressure, checkpoints
    driver.py          (new) run loop, LogWriter, trajectory/restart writers, checkpoint format
    simulation.py      Simulation (rigid bodies)        flexible.py  FlexibleSimulation + templates
    pimd.py            PIMD engine only                 remd.py  replicas + exchange
    finite_field.py    FieldReplicas only               alchemy.py  Alchemy, LambdaWindows, FreeEnergyRun
    dipoles.py io.py
  analysis/            (new) stats.py (jackknife, blocks, inefficiency, correlation time),
                       dielectric.py, free_energy.py (TI/BAR/MBAR), finite_field.py (analyse, fits)
  fit/                 params.py (one ParameterSpace), frames.py estimators.py optimize.py liquid.py,
                       qm.py (was qmfit.py), free_energy.py (was md/fe_grad.py targets/estimators),
                       reweighting.py (was ensemble.py)
  models/              (new) water.py (q-TIP4P/F flexible water builder from md/pimd.py), GVDW tables
  bias/  bonded/  interfaces/  protein/   (as now; bonded study code -> bonded/study/)
  cli/                 (new) args.py (shared argparse groups: settings, thermostat, output; units),
                       main.py (single `pgm-jax` entry point for the production drivers)
```

Moves are plain `git mv` + import updates in one commit each; no numerical code changes in them.
Whether the old import paths keep thin re-export shims for one release is open question Q3.

### 3.3 Naming rules and units policy

Units (documented once in `units.py` and in the README):

- Library arguments, attributes, state and results use the internal units: nm, ps, amu, K, bar,
  kJ/mol, e, e nm, nm^3, V/nm, rad. No unit suffix for these (`positions`, `velocities`, `box`,
  `dt`, `temperature`, `pressure`, `cutoff`).
- A value in any other unit carries the unit as a suffix: `xyz_A`, `dt_fs`, `energy_kcal`,
  `dipole_D`, `box_A`; file readers/writers in Amber units say so in the name
  (`read_coordinates` -> returns `xyz_A`, as now documented in text only).
- Intervals counted in steps end in `_every` (`report_every`, `traj_every`, `checkpoint_every`,
  `sample_every`, `exchange_every`, `barostat.every`); durations in ps end in `_ps`
  (`equil_ps`, `discard_ps`).
- CLI options use the library names in kebab case (`--temperature`, `--pressure`, `--cutoff`,
  `--dt`, `--report-every`) and the library units, with one exception to be decided (Q1: `--dt`
  in fs or ps).

Names:

| Concept | Name |
|---|---|
| positions / velocities / box (lattice vectors as rows) | `positions`, `velocities`, `box` (short `pos`, `vel`, `H` only inside functions) |
| temperature(s), pressure | `temperature`, `temperatures`, `pressure` (bar); `kT` only for energies in kJ/mol |
| thermostat time scales | `Langevin(friction)` 1/ps, `Bussi(tau)` ps, `GLE.band()`, `PILE(tau_centroid, lam)` |
| barostat | `MonteCarloBarostat(pressure, every)` |
| solver tolerances | `dipole_tol` (MD induction), `adjoint_tol`; other iterative solvers `tol` in their own object |
| number of copies | `beads`, `replicas`, `walkers`, `windows` (ints); ladders `temperatures`, `lambdas`, `fields` |
| output | `prefix` (path prefix; `None` = no files), `log` (text stream or `None`) |
| checkpoint files | `save_checkpoint(path)`, `load_checkpoint(path)`; Amber restart `write_restart(path)` |
| random numbers | `seed` (int); streams derived as today (unchanged so trajectories stay identical) |

### 3.4 `MDSettings` in groups

```python
MDSettings(
    terms=Terms(elec="qpi", vdw="lj", gvdw_rep="gauss", lj_lrc=True),  # not "model": clashes with pgm_jax.Model
    cutoffs=Cutoffs(cutoff=0.9, elec_cutoff=None, skin=0.1),
    pme=PME(ewald_beta=4.0, grid=None, spacing=0.08, order=6),
    induction=Induction(
        tol=1e-5,
        max_iter=50,
        predictor="mu4",
        fused=True,
        norm_refresh=1000,
        peek=0.65,
        local_cut=0.3,
        local_niter=0,
        extrap_order=3,
        extrap_steps=2,
    ),
    iel=ExtendedLagrangian(
        scheme="none", iterations=1, order=7, kappa=None, alpha=None, precond="block", omega=1.0, shadow=True
    ),
    precision="mixed",
    differentiable=False,
    adjoint_tol=1e-6,
)
```

- All groups are frozen dataclasses with the current defaults, so `MDSettings()` is unchanged and
  hashing for jit caches still works. Derived properties (`pair_cutoff`, `elec_rc`, `dtype`,
  `describe_*`) stay on `MDSettings`.
- Convenience: `MDSettings.replace(**flat)` accepts the flat names of today
  (`s.replace(dipole_tol=1e-8, cutoff=0.6)`) so that tests and scripts that tweak one or two
  values stay readable; `elec_cutoff_settings()` returns a `Cutoffs`/`PME` pair.
- Validation of refused combinations (iEL + interfaces, quadrupoles in MD, ...) moves into
  `MDSettings.__post_init__` / one `check()` instead of being spread over the engines.
- Neighbour-list options that are not force-field settings (`neighbor_list` mode, `r_margin`,
  `bead_margin`) stay engine arguments, named alike in all engines.
- Whether to nest at all (vs. a flat class with grouped documentation) is Q6.

### 3.5 Unified engine API

```python
from pgm_jax.md import (Simulation, FlexibleSimulation, MDSettings, Bussi, Langevin, GLE,
                        MonteCarloBarostat)

sim = Simulation(system, positions, box, settings=MDSettings(),
                 dt=0.001, temperature=298.0,
                 thermostat=Bussi(tau=1.0),            # None -> NVE; strings "langevin"/"bussi"/"gle" = defaults
                 barostat=MonteCarloBarostat(pressure=1.0, every=100),   # None -> constant volume
                 velocities=None, seed=0, params=None,
                 restraints=None, bias=None, alchemy=None, efield=None, mts=None,
                 neighbor_list="auto", log=None)
sim = Simulation.from_amber(prmtop, coords, charges="pgm", **same_keywords)
flex = FlexibleSimulation(system, templates, positions, box, ...same keywords...,
                          constraints="h-bonds", hmr=None, constraint_options=None, r_margin=0.05)

sim.advance(n)                     # public, no files (was _advance)
sim.run(nsteps, prefix="md", report_every=1000, traj_every=0, checkpoint_every=0,
        dipoles_every=0, induced_every=0, report_pressure=False, append=False)
sim.observables(); sim.pressure(); sim.positions(); sim.velocities()
sim.save_checkpoint(path); sim.load_checkpoint(path); sim.write_restart(path)
sim.set_field(E); sim.set_restraints(r); sim.set_bias_state(b)          # unchanged

PIMDSimulation(sim, beads=32, mode="pimd", thermostat=PILE(kind="l", tau_centroid=0.2, lam=None),
               propagator="cayley", contract=None, barostat=None, bead_margin=0.08,
               bead_chunk="auto", spread=True, seed=0, log=None)       # dt, temperature, settings from sim
ReplicaExchange(sim, temperatures, exchange_every=500, batched=True, seed=0, log=None)
FieldReplicas(sim, fields, seed=0, log=None); Walkers(sim, walkers, shared=False, seed=0, log=None)
LambdaWindows(sim, lambdas, batched=True, seed=0); FreeEnergyRun(windows, sample_every=500, ...)
# every driver: .advance(n), .run(nsteps, *, prefix, report_every, traj_every, checkpoint_every,
#               append, **driver-specific keywords), .save_checkpoint(path), .load_checkpoint(path)
```

The shared core (host side only; the compiled step is not touched):

- `md/engine.py: MDEngine` - the common parts of `Simulation` and `FlexibleSimulation`: settings
  and box checks, neighbour-list construction and sizing, the block advance with overflow
  detection / resize / recompile / repeat and the 10 %-volume rebuild (one implementation, also
  used by PIMD, `MDReplicas` and `PGMEngine` through a small `Resizable` protocol), `pressure()`,
  observables, the log header, checkpoints. The two engines provide hooks (`positions(state)`,
  `list_centers(pos)`, `atom_velocities(state)`, `describe()`).
- `md/driver.py` - `run_blocks(driver, nsteps, events)` (gcd blocking, event callbacks),
  `LogWriter` (header once, append mode, fixed column format, `ns_per_day`), NetCDF / restart /
  dipole writers, and `Checkpoint` (one pickle container `{"format": "pgm_jax <kind>", "version":
  n, ...}` for all drivers, with a clear error for a checkpoint of another kind).
- Observable keys stay as they are (they are log-file columns that analysis scripts read):
  `temp_trans`/`temp_rot` (rigid bodies) and `temp_com`/`temp_internal` (atoms) are different
  quantities; all keys of all drivers are documented in one table (`docs/user/md.md`), and new
  keys follow the same pattern (`<quantity>[_<qualifier>]`, units in the table).
- Thermostats become the configuration objects that already exist (`Langevin`, `Bussi`, `GLE` in
  `md/thermostats.py`); `gamma` / `tau_t` / `ensemble` / `pressure` / `barostat_interval`
  disappear from the engine signatures. Q2 asks whether an `ensemble=` string stays as a shortcut.
- `log=None` by default (library code is quiet; scripts pass `log=sys.stdout`); Q5 asks whether
  to switch to the `logging` module instead.

### 3.6 Error messages

Current style is already good in most places and becomes the rule:
`raise ValueError(f"{argument}: {what is wrong} ({value!r}); {what to do}")`, lower case, no
trailing period, name the argument and the accepted values (as `options.elec_flags`,
`thermostats.make_thermostat`). `ValueError` / `TypeError` for arguments,
`NotImplementedError` for supported-later combinations (with the doc to read), `RuntimeError` for
failures at run time (persistent overflow), `FloatingPointError` for non-finite energies. No
`assert` for user input. Refused combinations are checked at construction, not in the first
`run()`.

### 3.7 Docstrings and type hints

- numpy style (Parameters / Returns / Raises / Notes / References): the physics prose of the module
  and class docstrings stays; each public function gets a Parameters section with the unit of
  every argument in brackets (`dt : float` / `Time step [ps].`). Private helpers and jitted
  closures: one-line docstring where the name is not enough.
- Type hints: every public signature fully annotated (`from __future__ import annotations`,
  `jax.Array`, `np.ndarray`, `ArrayLike`, `Sequence[...]`, `Literal["q","qp","qi","qpi"]` for
  option strings); private code optional. No runtime type checking. A type checker (mypy /
  pyright) is Q12.
- Enforced by review, and for docstring presence by a small check in `scripts/dev/check.sh`
  (ruff `D1xx` on `pgm_jax/` only, if the owner agrees: Q11).

### 3.8 Scripts

```
scripts/
  run/        run_md.py solvation_free_energy.py finite_field.py pimd_water.py remd_peptide.py write_pgm_prmtop.py build_amber.py ...
  analysis/   dielectric.py trajectory_dipoles.py liquid_fit_tools.py ala2_analyze.py
  fit/        fit_liquid.py fit_multi.py qmfit/*
  validate/   validate_*.py shake_vs_pmemd.py pimd_validate.py ... (each writes validation/<name>.json)
  bench/      bench_md.py bench.py bench_shake.py iel_cost.py bench_interfaces.py bench_protein.py
  dev/        gpu_bench.sh regression entry, check.sh, migrate_api.py (codemods for the phases)
  studies/bonded/   the bonded-study scripts (research record, not maintained against API changes: Q10)
```

- Every script: module docstring with a usage line, `main()` + `__main__` guard, argparse built
  from `pgm_jax/cli/args.py` groups (`add_settings_args`, `add_thermostat_args`,
  `add_output_args`, `add_iel_args`, `add_mts_args`), no `sys.path` edits (the package is
  installed with `pip install -e .`), no hard-coded user paths (data paths are arguments or
  `PGM_DATA` environment variable).
- One entry point `pgm-jax` (`[project.scripts]`) with subcommands for the production drivers only:
  `pgm-jax md | pimd | remd | fe | finite-field | dielectric | fit-liquid | write-prmtop`
  (thin wrappers of the `scripts/run/*.py` mains). Validation and benchmark scripts stay scripts.

### 3.9 Tests

```
tests/
  conftest.py            shared builders (water, methanol, small_box, water_box, cluster, templates)
                         and markers: slow, gpu, needs_data (pGM3P-25 files), needs_amber/openmm/ipi
  unit/core/ unit/md/ unit/bias/ unit/fit/ unit/bonded/ unit/interfaces/ unit/protein/
  integration/           test_md_macro, test_combinations, test_integration, examples smoke tests
  regression/            the golden-output harness (section 7)
  data/
```

No test imports another test module; helpers come from `conftest.py` (or `tests/helpers.py`).
Tests use the public API only (`advance`, not `_advance`), except white-box tests of a module's
internals, which live next to that module's other tests and say so.

### 3.10 Documentation

```
README.md                 short: what, install, 20-line example, links (feature numbers move to docs)
docs/index.md             map of the docs
docs/user/                model_options, howto_bonded, howto_vdw, md (engines, thermostats, outputs), units
docs/features/            charge_flux dielectric efield enhanced_sampling fe_gradients free_energy iel
                          interfaces liquid_fit mts pimd protein_ff qmfit shake virtual_sites thermostat_ideas
docs/dev/                 api_design.md (this file), CHANGES_2026-09.md, dev_notes/NOTES_*.md (moved from the root)
validation/ reports/      tracked results of record (unchanged location; Q15)
```

API examples in the docs are updated in the phase that changes the API (4). Examples in
`examples/` get a smoke test (`tests/integration/test_examples.py`, tiny sizes).

---------------------------------------------------------------------------------------------------

## 4. Migration table

Counts are call sites found on master (`grep`), as lines / files, in scripts (S), tests (T) and
docs + README + examples (D). Every phase updates all of them in the same commit; a codemod
(`scripts/dev/migrate_api.py`, regex + manual review) does the mechanical keyword renames.

### 4.1 Engines and drivers

| Old | New | S | T | D |
|---|---|---|---|---|
| `Simulation(sys, pos_nm, H_nm, settings, dt=, ensemble=, temperature=, gamma=, pressure=, barostat_interval=, seed=, vel_nm_ps=, params=, log=sys.stdout, neighbor_list=, thermostat="langevin", tau_t=, restraints=, alchemy=, mts=, bias=, efield=)` | `Simulation(system, positions, box, settings, *, dt, temperature, thermostat="langevin", barostat=None, velocities, seed, params, restraints, alchemy, mts, bias, efield, neighbor_list, log=None)` | 33/18 | 53/15 | 9/7 |
| `FlexibleSimulation(sys, templates, pos_nm, H_nm, ..., log=None, r_margin, constraints, hmr, max_single, constraint_options)` | same keywords as `Simulation` + `constraints`, `hmr`, `constraint_options` (with `max_single` inside), `r_margin` | 30/19 | 65/25 | 22/11 |
| `ensemble="nve"` | `thermostat=None` (no barostat) | 72 (all `ensemble=`) | 127 | 12 |
| `ensemble="nvt", thermostat="langevin", gamma=g` | `thermostat=Langevin(friction=g)` (or `"langevin"` for the default 1/ps) | 23 (`gamma=`) | 19 | 4 |
| `thermostat="bussi", tau_t=t` | `thermostat=Bussi(tau=t)` | 20 (`tau_t=`) | 18 | 1 |
| `ensemble="npt", pressure=p, barostat_interval=n` | `barostat=MonteCarloBarostat(pressure=p, every=n)` | 14 | 11 | 2 |
| `vel_nm_ps=` / `pos_nm`, `H_nm` | `velocities=` / `positions`, `box` | 23/12 | 11/9 | 3 |
| `Simulation.from_amber(prmtop, coords, use_velocities, charges, **kw)` | unchanged name, new keywords as above | 5/3 | 4/3 | 3/2 |
| `sim._advance(n)` | `sim.advance(n)` | 67/21 | 89/19 | 0 |
| `sim.run(nsteps, report, traj, restart, prefix, pressure_every_report, append, dipoles, induced)` | `sim.run(nsteps, *, prefix, report_every, traj_every, checkpoint_every, report_pressure, append, dipoles_every, induced_every)` | 62 (all `.run(`) | 37 | 19 |
| `sim.save(prefix)`, `sim.load(path)` | `sim.save_checkpoint(path)`, `sim.write_restart(path)`, `sim.load_checkpoint(path)` (run() still writes `prefix.chk` + `prefix.rst7`, and `prefix.bias` with biases) | 16/9 | 19/15 | 3/3 |
| `sim.positions_nm()`, `sim.velocities_nm_ps()` | `sim.positions()`, `sim.velocities()`, `sim.box()` | 41/16 | 47/15 | 10/5 |
| `simulation._dedupe(read_prmtop_pgm(p, first_residue_only=False, charges=c))` | `System.from_prmtop(p, charges=c)` (identical molecules share one template) | 19/15 | 2/1 | 1/1 |
| `PIMDSimulation(sim, beads, mode, thermostat="pile-l", tau0, lam, propagator, contract, bead_margin, seed, dt, spread, ensemble, pressure, barostat_interval, bead_chunk, log=sys.stdout)` | `PIMDSimulation(sim, beads, mode, thermostat=PILE(kind="l", tau_centroid=0.2, lam=None), propagator, contract, barostat=None, bead_margin, bead_chunk, spread, seed, dt=None, log=None)` | 2/1 | 10/3 | 2 |
| `PIMDSimulation.run(..., beads_traj, pressure: bool)`, `set_mode(mode, thermostat, tau0, lam)` | `run(..., beads_traj_every, report_pressure)`, `set_mode(mode, thermostat=PILE(...))`; `centroid_nm()` / `beads_nm()` -> `centroid()` / `beads()` | 3 | 4 | 2 |
| `RingPolymer(nbeads, ...)`, `PIMDIntegrator(engine, masses, nbeads, ...)` | `beads` | 0 | 6 | 1 |
| `ReplicaExchange(sim, temperatures, exchange_every, batched, seed, log=sys.stdout)`; `.run(nsteps, report, traj, restart, prefix, append)`; `save(prefix)`/`load(path)` | `log=None`; `.run(nsteps, *, prefix, report_every, traj_every, checkpoint_every, append)`; `save_checkpoint`/`load_checkpoint` | 4/2 | 15/4 | 1 |
| `FieldReplicas.run(nsteps, every, prefix, report, append, restart, extra, log)`; `save(path)`/`load(path)` | `FieldReplicas(sim, fields, seed, log=None).run(nsteps, *, sample_every, prefix, report_every, checkpoint_every, append, extra)`; `save_checkpoint`/`load_checkpoint` | 2/2 | 2/2 | 1 |
| `Walkers(sim, n, shared, bias_states, seed)`; `.run(nsteps, report, restart, prefix, append, log)`; `save/load(path)` | `Walkers(sim, walkers, shared, bias_states, seed, log=None)`; `.run(nsteps, *, prefix, report_every, checkpoint_every, append)`; `save_checkpoint`/`load_checkpoint` | 4/2 | 5/3 | 4/1 |
| `FreeEnergyRun(windows, sample_every, exchange_every, seed, log, meta, param_grad)`; `.run(nsteps, prefix, report, restart)`; `save(prefix)`/`load(path)`/`load_windows(path)` | `FreeEnergyRun(windows, sample_every, exchange_every, seed, log=None, meta, param_grad)`; `.run(nsteps, *, prefix, report_every, checkpoint_every)`; `save_checkpoint`/`load_checkpoint`/`load_windows`, `save_samples(path)` (prefix_fe.npz) | 1/1 | 8/3 | 3/3 |
| `LiquidFit(sys_, pos, H, space, objective, T, pressure, settings, dt, thermostat="bussi", tau_t, gamma, barostat_interval, ..., tol, ..., log=sys.stdout, ..., ensemble, replicas, ...)` | `LiquidFit(system, positions, box, space, objective, temperature, settings, dt, thermostat=Bussi(), barostat=MonteCarloBarostat() (None: NVT), ..., dipole_tol, ..., log=None, ..., replicas, ...)` (P5: coupling objects, `log`; P7: `sys_`, `pos`, `H`, `T`, `tol` renamed; prefix.json key `"T"` -> `"temperature"`) | 1/1 | 3/1 | 1/1 |
| `Integrator(ff, rigid, neighbors, dt, ensemble, temperature, gamma, pressure, barostat_interval, params, thermostat, tau_t, ...)`, `FlexibleIntegrator(...)`, MTS variants | same keywords as the engines (thermostat / barostat objects) | 0 | ~6 | 0 |
| `make_thermostat(spec, gamma, tau)`, `Langevin(gamma)` | `make_thermostat(spec)` (string = defaults, or an object), `Langevin(friction)` | 0 | 3 | 1 |

Unchanged: `observables()` keys, log-file columns and every output file format except the
checkpoints (P4: versioned `.npz` with a JSON header; the old pickles are still read) and the
progress logs of `Walkers` (`prefix_walkers.log`), `FreeEnergyRun` (`prefix_fe.log`) and
`FieldReplicas` (`prefix.log`), which became tables in the format of `Simulation.run`'s log
(P4, `md/driver.LogTable`); `set_field`, `set_restraints`, `set_bias_state`, `restraint_energies`,
`bias_energies`, `cv_values`, `minimize`, `MDReplicas.advance/permute/potentials/observables`,
`LambdaWindows` (except `log`), `Alchemy`, `MTS`, `Restraints`, `ExternalField`, `VirtualSite`,
bias classes, `PGMForceField.compute/rows_for/strain_derivative/init_induction`.

### 4.2 Settings and models

| Old (`MDSettings` flat field) | New |
|---|---|
| `elec`, `vdw`, `gvdw_rep`, `lj_lrc` | `terms.elec`, `terms.vdw`, `terms.gvdw_rep`, `terms.lj_lrc` |
| `cutoff`, `elec_cutoff` | `cutoffs.cutoff`, `cutoffs.elec_cutoff` |
| `skin`; engine keyword `neighbor_list=` of `Simulation`, `FlexibleSimulation`, `PGMEngine` | `neighbors.skin`, `neighbors.mode` (`NeighborList(skin=0.1, mode="auto")`; flat names `skin`, `neighbor_list`) |
| `ewald_beta`, `pme_grid`, `pme_spacing`, `pme_order` | `pme.ewald_beta`, `pme.grid`, `pme.spacing`, `pme.order` (group class `PMESettings`: `PME` is the solver class of `md/pme.py`) |
| `dipole_tol`, `max_iter`, `predictor`, `fused`, `norm_refresh`, `local_cut`, `local_niter`, `peek`, `extrap_order`, `extrap_steps` | `induction.tol`, `.max_iter`, `.predictor`, `.fused`, `.norm_refresh`, `.local_cut`, `.local_niter`, `.peek`, `.extrap_order`, `.extrap_steps` |
| `iel`, `iel_iter`, `iel_order`, `iel_kappa`, `iel_alpha`, `iel_precond`, `iel_omega`, `iel_shadow` | `induction.iel.scheme`, `.iterations`, `.order`, `.kappa`, `.alpha`, `.precond`, `.omega`, `.shadow` (`ExtendedLagrangian`, nested in `Induction`) |
| property `induction` (bool: induced dipoles on) | `has_induction` (the name `induction` is the group) |
| `precision`, `differentiable`, `adjoint_tol` | unchanged (top level) |
| `MDSettings(**flat)` in 55 script / 63 test / 31 doc lines | `MDSettings().replace(**flat)` keeps the flat names working for construction; attribute reads (`s.cutoff`) change to the group path (~120 sites in `pgm_jax/`, mostly `md/forcefield.py`, `md/mts.py`, `interfaces/engine.py`) |
| (P7) `PeriodicModel(sys, H, pos_ref, rc, b0, skin, lj, lj_rc, lj_lrc, k_tol, cg_tol, elec, vdw, gvdw_rep)` | `PeriodicModel(system, box, positions_ref, cutoff, ewald_beta, skin, vdw_cutoff, lj_lrc, k_tol, dipole_tol, elec, vdw, gvdw_rep)` (`lj=False` removed: `vdw="none"`; the methods keep `energy(pos, params, H)` etc.) - S 1, T 5, D 2 |
| (P7) `PeriodicPGM(sys, H, pos_ref, b0, rc, skin, k_tol, cg_tol, nlist, elec)` | `PeriodicPGM(system, box, positions_ref, ewald_beta, cutoff, skin, k_tol, dipole_tol, nlist, elec)` - S 3, T 7, D 0 |

### 4.3 Fitting, analysis and module moves

| Old | New | S | T | D |
|---|---|---|---|---|
| `pgm_jax.qmfit` | `pgm_jax.fit.qm` | 3 | 1 | 2 |
| `pgm_jax.ensemble` (`Reweighting`, `karplus`, ...) | `pgm_jax.fit.reweighting` | 0 | 1 | 1 |
| `pgm_jax.md.dielectric`, `pgm_jax.md.free_energy` | `pgm_jax.analysis.dielectric`, `pgm_jax.analysis.free_energy` | 4 | 3 | 3 |
| `md.finite_field.{analyse, fluctuation_eps, saturation_fit, predicted_errors, block_mean, correlation_time}` | `analysis.finite_field.*` (`FieldReplicas`, `read_series` stay in `md.finite_field`) | 2 | 1 | 1 |
| jackknife / block / correlation helpers (2.5) | `analysis.stats.{jackknife, jackknife_cov, block_means, statistical_inefficiency, correlation_time}` (existing call sites re-pointed; numerics of each kept) | 3 | 2 | 0 |
| `md.fe_grad.{gradient_estimate, FEGradient, FreeEnergyTarget, combine}` | `fit.free_energy.*` (P7); `ParameterGradients`, `gas_leg_gradient` (evaluates the alchemical gas-phase Hamiltonian), `alchemical_map`, `scaled_params` stay in `md.fe_grad`; `jackknife_error` is `analysis.stats.jackknife_error` (P2) | 2 | 6 | 4/2 |
| `md.fe_grad.ParamGradients` | `md.fe_grad.ParameterGradients` (P7) | 2 | 6 | 3/2 |
| `fit.ParameterSpace(table, params, p0, prior_sigma)`, `fit.Param(quantity, kind="scale"\|"shift", keys, name, prior_sigma, extra)` | `fit.params.ParameterSpace(table, params, p0, prior_sigma, neutral)` with `Param(quantity, kind="scale"\|"shift"\|"values", keys, name, prior_sigma, bounds, step, extra)`; classmethods `scales(table, quantities, p0, prior_sigma)`, `values(table, free, p0, neutral, bounds, steps, prior_sigma)`, `from_names(names)`; attributes `names`, `n`, `theta0`, `lower`, `upper`, `step`, `prior_sigma`, `params`, `quantities`, `keys`, `slices`; the per-parameter index list `space.index` became `space.blocks[j].idx` and `index(name)` a method (P7, D9) | 3 | 6 | 2 |
| `md.fe_grad.ParamSpace(table, quantities)`, `ParamSpace.from_names(names)`; `.flatten`, `.unflatten`, `.index`, `.select`, `.scale_direction`; `md.fe_grad.SCALE_GROUPS` | `ParameterSpace.values(table, quantities)`, `ParameterSpace.from_names(names)` (same methods; `flatten` / `unflatten` need a values-only space); `fit.params.SCALE_GROUPS` (P7) | 5 | 4 | 3/3 |
| `qmfit.ParamMap(table, molecules, free, P0, bounds, scales)`; `.params(theta)`, `.scale`, `len()`; `QMFit(cm, pm, data, weights)`; `qm.SCALES` | `fit.qm.parameter_space(table, molecules, free, p0, bounds, steps)` -> a `ParameterSpace`; `space(theta)`, `.step`, `len()` (same `names`, `theta0`, `lower`, `upper`; charges keep the `q:null{j}` null-space coordinates); `QMFit(cm, space, data, weights)`; `qm.STEPS` (P7) | 8 | 5 | 9/5 |
| `LiquidSamples(frames, T, n_mol, mass, nblocks, pressure_bar)` (attribute `.T`), `FrameAnalyzer(sys, H, settings, space, rdf, tol, max_iter, chunk, margin, row_block)`, `GasPhase(molecule, pos, table, space, elec)` | `LiquidSamples(frames, temperature, n_mol, mass, nblocks, pressure)` (attribute `.temperature`), `FrameAnalyzer(system, box, settings, space, rdf, dipole_tol, max_iter, chunk, margin, row_block)`, `GasPhase(molecule, positions, table, space, elec)` (P7) | 4 | 5 | 2/2 |
| `analysis.dielectric.{fluctuation, jackknife, static_dielectric, block_errors, running, decomposition}(..., T, ...)`, `correlation_time(M, dt_ps, window)`, `ir_spectrum(M, dt_ps, V, T, segment_ps)`; `analysis.finite_field.fluctuation_eps(M, V, T, ...)`, `predicted_errors(eps, eps_inf, V, T, E, tau_ps, run_ps)` | `temperature` for `T` everywhere; `correlation_time(M, dt, window)`, `ir_spectrum(M, dt, V, temperature, segment_ps)` (`dt` in ps, the library unit); `predicted_errors(eps, eps_inf, V, temperature, field, tau_ps, run_ps)` (P7) | 10 | 9 | 0 |
| `md.pimd.{flexible_water, qtip4pf_intra, QTIP4PF, water_geometry, harmonic_frequencies, WATER_FAMILIES}` | `pgm_jax.models.water.*` | 2 | 2 | 1 |
| `md.iel.{add_iel_arguments, iel_settings}`, `md.mts.{add_mts_arguments, mts_from_args, mts_stats}` | `pgm_jax.cli.args.*` | 8 | 1 | 1 |
| `pgm_jax.kernels` | `pgm_jax.densities` | 0 | 1 | 0 |
| `bonded.{bench, data, molecules}`, `bonded.terms.explore` | `bonded.study.*` (registry keeps the explored families) | 12 | 3 | 1 |
| `md.neighbors.Neighbors` (alias) | `AtomNeighbors` | 0 | 4 | 0 |
| `KB` in `md.integrate` / `bias.core` / `fit.estimators` / `ensemble`, `BAR` in `md.integrate` / `md.pimd`, `periodic.KJMOL_NM3_BAR`, `fit.estimators.{BAR_KJ, G_CM3}`, `simulation.AMU_NM3_TO_G_CM3` | `pgm_jax.units.{KB, BAR_PER_KJMOL_NM3, KJMOL_NM3_PER_BAR, AMU_NM3_TO_G_CM3, ...}` (same literal values and expressions) | 7 | 10 | 0 |
| `param.PGM_POL_TABLE` (`~/amber25/...`) | argument of `read_pol_table(path)`, default from `PGM_POL_TABLE` env var | 2 | 1 | 1 |

### 4.4 Command-line options (all scripts; `pgm_jax/cli/args.py`)

| Old | New |
|---|---|
| `--T`, `--temp` | `--temperature` |
| `--press`, `--pressure` (bar) / `--pressure` (flag: report it) | `--pressure` (bar) / `--report-pressure` |
| `--gamma`; `--tau`, `--tautp`; `--tau0` | `--friction`; `--tau`; `--tau-centroid` |
| `--out`, `-o`, `--prefix` | `--prefix` (`-o` short form) |
| `--cut` (Angstrom in `run_md.py`, `trajectory_dipoles.py`; nm elsewhere), `--cutoff` | `--cutoff` (nm) |
| `--es-cut`, `--elec-cut` | `--elec-cutoff` (nm) |
| `--tol`, `--dipole-tol`, `--md-tol` | `--dipole-tol` (MD), `--analysis-tol` (frame analysis) |
| `--prec`, `--precision` | `--precision` |
| `--report` (steps), `--report` (ps), `--report-ps` | `--report-every` (steps) or `--report-ps` |
| `--restart` (steps between checkpoints), `--checkpoint` (file to continue from) | `--checkpoint-every`, `--continue-from` |
| `--dt` (fs in 10 scripts, ps in 12) | one unit everywhere (Q1) |

---------------------------------------------------------------------------------------------------

## 5. Phased plan

Every phase is a short series of small commits on `cleanup`, each of which passes:

- **pytest**: the full suite (`python -m pytest -q tests`, CPU, 32 cores; plus the GPU subset the
  owner runs today). Baseline at P0 (CPU, 32 cores): 354 passed, 3 skipped (two tests needing
  external codes, and the opt-in harness module) in 68 min,
- **harness**: `python tests/regression/regress.py check` on the CPU (16 threads): bitwise
  (`--rtol 0 --atol 0`) unless the phase says otherwise; the harness code itself is updated in the
  same commit when an API it calls changes (its calls are concentrated in the "API adapter" section
  of `regression_cases.py`),
- **speed** (phases that touch `pgm_jax/md/`, `interfaces/engine.py`, `bias/`): `scripts/dev/gpu_bench.sh`
  on one GPU, alternating with master; no case slower than master by more than the run-to-run
  noise (about 3 %, section 7.3) in both rounds.

| Phase | Content | Verification / tolerance |
|---|---|---|
| P0 (this) | Design document, regression harness + golden files, GPU bench script, ruff config (not applied), `.tools/` ignored | harness reproduces itself bitwise (section 7) |
| P1 lint | `ruff check --fix` (I001, F401, UP037), manual F841 / UP031 fixes, `noqa` only where intended; then `ruff format` as a separate commit listed in `.git-blame-ignore-revs` | pytest, harness bitwise; `ruff format` is AST-preserving (checked with an AST dump diff over all files) |
| P2 constants and helpers | `units.py` holds every constant (same literals / expressions); `analysis/stats.py`; one prmtop parser; `System.from_prmtop`; COM helper outside jitted kernels only; box helpers merged; delete `Neighbors` alias, unused variables | pytest, harness bitwise, speed |
| P3 module moves | `qmfit` -> `fit/qm`, `ensemble` -> `fit/reweighting`, `analysis/` package, `models/water.py`, `cli/args.py`, `densities.py`, `bonded/study/`; imports in scripts/tests/docs; pickle-compatible class paths (6.1) | pytest, harness bitwise (import-only change) |
| P4 engine core | `md/driver.py` (run loop, LogWriter, Checkpoint), `md/engine.py` (`MDEngine`, shared resize/advance); `Simulation`, `FlexibleSimulation`, PIMD, REMD, FieldReplicas, Walkers, FreeEnergyRun use them; public `advance()`; old names still accepted inside this phase | pytest, harness bitwise (including `md_rigid_run_files`: log columns, NetCDF, restart, dipole files, checkpoint continuation), speed |
| P5 engine API | thermostat / barostat objects, `positions`/`box`/`velocities`, `log=None`, `*_every` run keywords, `save_checkpoint`/`load_checkpoint`, PIMD / REMD / replicas / LiquidFit signatures; codemod of scripts, tests, docs | pytest, harness bitwise, speed |
| P6 settings | `MDSettings` groups + `replace(**flat)`; `PeriodicModel` names; `elec_cutoff_settings` | pytest, harness bitwise, speed (jit caches keyed by the settings hash) |
| P7 fitting API | `temperature`/`pressure`/`dipole_tol` names in `fit/`, analysis; one `ParameterSpace` (if Q8 = yes) | pytest, harness bitwise (`fit_frames`, `fe_windows`, `qmfit_synthetic`, `analysis_estimators`) |
| P8 docstrings and hints | numpy docstrings + annotations for every public name, module by module (no code changes) | pytest, `ruff check`, harness bitwise |
| P9 scripts and CLI | `scripts/` subfolders, shared CLI groups, `pgm-jax` entry point, no `sys.path` hacks or user paths; each production script run once on a tiny input (`scripts/dev/smoke_scripts.sh`) | pytest, harness, smoke runs of every production script |
| P10 tests and docs | `conftest.py` fixtures, `unit/` + `integration/` layout, markers; `docs/` index and folders, NOTES -> `docs/dev/dev_notes/`, README trimmed, examples smoke test | pytest (same 356 test ids + new), harness bitwise |

Each phase is merged into master only after the owner's review of the phase (or all at once at the
end: Q3 decides whether intermediate releases need compatibility shims).

---------------------------------------------------------------------------------------------------

## 6. Risks, deliberate non-changes, open questions

### 6.1 Risks and mitigations

- **Bitwise identity.** XLA can reorder operations when the traced program changes, even for
  equivalent Python; a refactor of host code that changes the jitted closure (captured constants,
  argument order, a helper that now uses `jnp` instead of `np`) can change the last bits.
  Mitigation: do not touch traced code in P2-P7; move code only between modules; the harness
  covers every engine and is run bitwise after each commit; the few intended deviations (none
  planned) must be justified in the commit message with the harness's max rel. deviation.
- **Constants.** `BAR = 1.0 / 16.605390671738466` and the literal `16.605390671738466` are
  different doubles in different places; `units.py` keeps both forms where they are used today
  (a product by `1/x` is not bitwise equal to a division by `x`).
- **Pickled files.** Checkpoints and `.flex` templates are pickles; they store class paths
  (`pgm_jax.md.integrate.MDState`, `pgm_jax.bonded.model.MolSpec`, ...). Moving or renaming
  those classes breaks existing files (e.g. the committed `validation/interfaces/pgm_water_flex.flex`).
  Mitigation: do not move pickled classes, or load through an `Unpickler.find_class` map of old
  paths; the new checkpoint header carries a version.
- **jit caches and static arguments.** `MDSettings` is hashed as a static argument; the grouped
  version must stay frozen and hashable, with the same equality semantics.
- **Hidden users of private names** (`sim.integ._forces`, `sim.ff._atoms`, `_size_lists`, `_rebuild*`:
  42 uses in scripts and tests). They are either made public (with a name) or rewritten.
- **CPU determinism depends on the XLA build.** The golden files are valid for jax/jaxlib 0.11.2,
  numpy 2.5.3 and this CPU family (Intel Xeon Gold 6418H, AVX-512). After an environment upgrade
  the harness must be re-recorded on master first (record the environment in `golden/*.json`).
- **Scope creep.** API redesign and moves are mechanical but large (about 700 call sites); phases
  are kept small and each updates scripts, tests and docs together.

### 6.2 Deliberately not changed

- Numerics and jitted hot paths: `PGMForceField` (rows, kernels, CG, predictor, iEL), PME, the
  integrator steps (rigid, flexible, MTS, PIMD), SHAKE/RATTLE solvers, thermostat O steps, bias
  deposition, alchemical Hamiltonian, the order of operations in all of them.
- Random-number streams (how keys are derived from `seed`), so trajectories stay identical.
- File formats of trajectories, restarts, logs (columns), `.dip`, `.mu.nc`, `.ffd`, COLVAR / HILLS,
  `_fe.npz`, REMD / FE JSON summaries; only the checkpoint header is new.
- The physics content of the docs (numbers, validation tables).
- Reference implementations `ewald.PeriodicPGM` / `PeriodicModel` stay separate from the MD force
  field (they are the differentiable reference used by the tests).
- JAX-MD workarounds in `md/_jaxmd.py`, `md/neighbors.py`.

### 6.3 Open questions for the owner

1. **Q1 `--dt` unit in CLIs**: ps everywhere (as the library and Amber's mdin) or fs everywhere
   (as `run_md.py` and 9 other scripts today)? Recommendation: ps, named `--dt`; `run_md.py` may
   additionally read an Amber mdin.
2. **Q2 `ensemble=` string**: remove it (thermostat / barostat objects only) or keep
   `ensemble="nve"|"nvt"|"npt"` as a shortcut that builds the default objects? Recommendation:
   remove (it is redundant and allows contradictions), 214 call sites change mechanically.
3. **Q3 compatibility shims**: hard break at each phase, or keep old import paths and keyword
   names with a `DeprecationWarning` until the end of the clean-up? Recommendation: shims only
   inside a phase, none after merge (single user code base).
4. **Q4 default thermostat**: keep Langevin (1/ps) or switch to Bussi (1 ps), which the docs
   recommend? Changing it changes the default trajectories (not the golden files, which name the
   thermostat explicitly).
5. **Q5 logging**: `log=None` default with explicit streams, or the `logging` module
   (`logging.getLogger("pgm_jax")`)?
6. **Q6 `MDSettings`**: nested groups (3.4) or keep the flat class and only document the groups?
   Nested is clearer; flat avoids ~120 attribute-path changes in the engine.
7. **Q7 checkpoints**: keep pickle with a versioned header, or move to `.npz` + JSON (portable,
   no class paths)? Should old checkpoints (no header) stay loadable?
8. **Q8 one `ParameterSpace`** for liquid fitting, free-energy gradients and QM fitting?
9. **Q9 single CLI entry point** `pgm-jax <subcommand>` for the production drivers: wanted?
10. **Q10 research code**: `scripts/bonded/` (24 scripts), `bonded/study`, `reports/`: keep
    maintained, keep frozen under `studies/`, or move to a separate repository?
11. **Q11 ruff scope**: `ruff format` removes the column alignment of 652 trailing comments in
    `pgm_jax/` and reflows 551 long lines; accept, or lint only (no formatter)? Also enable
    pydocstyle (`D`) for `pgm_jax/` and pycodestyle `E`/`W` beyond the requested `F`, `I`, `UP`?
12. **Q12 static typing**: add mypy / pyright to `scripts/dev/check.sh`?
13. **Q13 Python floor**: `requires-python >= 3.10` (target py310 for pyupgrade) is kept; raise to 3.12?
14. **Q14 `run_md.py` Amber units**: its `--cut` in Angstrom and Amber-like names (`--tautp`,
    `--ew-coeff`, `--vdwmeth`) mirror Amber on purpose: convert to the common names/units, or keep
    this one script Amber-flavoured?
15. **Q15 `validation/` and `reports/`**: keep tracked at the root, or move under `docs/`?
16. **Q16 README**: move the long feature list with numbers into `docs/features/` and keep a short
    README?

---------------------------------------------------------------------------------------------------

## 7. Regression harness (`tests/regression/`)

### 7.1 What it is

- `regression_systems.py`: self-contained builders of the test systems (toy pGM water, methanol,
  the skewed water/methanol box, flexible methanol with class II terms and optional charge flux,
  flexible water for PIMD, pGM3P-25 water from `~/pgm-gvdw-data`). Copied from the unit tests on
  purpose, so that reorganizing the tests cannot change the harness inputs.
- `regression_cases.py`: 35 cases, each returning a flat dict of arrays. The only code that knows
  the engine API is its "API adapter" section and the case bodies; a phase that changes an API
  updates them in the same commit, and the recorded numbers must still match.
- `regress.py`: `list`, `record` (writes `golden/<case>.npz` + `golden/<case>.json` with the
  environment: jax/jaxlib/numpy versions, host, CPU model, threads, commit), `check`
  (bitwise by default, `--rtol/--atol` for a stated tolerance, `--out report.json`; reports
  missing keys as failures and new keys as information; exit status 1 on any failure).
- `test_regression.py`: the same check as a parametrized pytest, opt-in with `PGM_REGRESSION=1`.

Run (CPU, float64 except one mixed-precision case, 16 threads):

```
~/project/cpu_run.sh <clone> <clone>/runs/rg/check.log 16 60 python tests/regression/regress.py check
# or per group (a..g) in parallel: ... regress.py check --group a
```

### 7.2 Coverage (golden files recorded on master `e72c57c`, 2.1 MB in total)

| Case | What is compared |
|---|---|
| `gas_model` | `Model` energies for elec q / qp / qi / qpi, forces, induced and permanent dipoles, dE/dparameters, molecular polarizability, n-body terms, external field, GVDW |
| `periodic_model` | `PeriodicModel` (Ewald) energies, forces, molecular and atomic strain derivatives, pressure, induced dipoles, `elec="qp", vdw="none"` |
| `ff_small_box` | `PGMForceField` single point (energy terms, forces, dipoles, CG iterations / residual, strain derivative) in double and mixed precision; parameter gradient of a force + dipole loss (`differentiable=True`) |
| `ff_pgm3p25_512` | the 512-water pGM3P-25 validation box at the default `MDSettings` (mixed) and in double |
| `efield_point`, `iel_point` | force field with an external field; iEL/0-SCF shadow energy and forces (block preconditioner) |
| `md_rigid_nve/langevin/bussi/gle/npt/mixed` | rigid pGM3P-25 water (64), 200 steps, snapshots every 50 steps: all observables, positions, velocities, box, induced dipoles; NPT pressure |
| `md_rigid_run_files` | `Simulation.run` with every output: log columns (except ns/day), NetCDF frames, rst7, `.dip` series, `.mu.nc`; continuation from the checkpoint in a new simulation |
| `md_rigid_mts`, `md_iel`, `md_efield`, `md_vsites`, `md_restraints_npt` | r-RESPA (short split); iEL/0-SCF MD; static field and constant D; TIP4P-Ew from a tleap prmtop (rigid engine and constrained flexible engine with a placed site); restraints of every kind under NPT |
| `flex_methanol_hbonds`, `flex_methanol_npt`, `flex_water_constraints`, `flex_flux`, `flex_mts` | flexible engine: minimize + X-H constraints (Bussi), NPT without constraints (Langevin), rigid water by SHAKE/RATTLE (GLE), charge flux (NVE), MTS special-pair split |
| `pimd`, `remd_batched`, `bias_metad`, `bias_walkers`, `field_replicas` | PIMD 4 beads (PILE-L, then TRPMD) with bead positions / momenta and estimators; batched REMD with exchanges (replica map, acceptance, energies, states); metadynamics + restraint (hills, COLVAR rows, bias energies); shared-bias walkers; FieldReplicas `.ffd` series |
| `alchemy_point`, `fe_windows` | alchemical Hamiltonian at lambda = (0.6, 0.8) (energy, forces, dipoles, pressure); batched lambda windows: reduced energies, dU/dlambda, dU/dtheta samples, `gradient_estimate`, `free_energy.estimate`, `FreeEnergyRun` with exchanges |
| `fit_frames`, `qmfit_synthetic`, `analysis_estimators` | `FrameAnalyzer` values and theta-derivatives; QM-fit components, predictions, residual loss and exact gradient; dielectric, BAR, MBAR, TI, statistical inefficiency, WHAM, jackknife, finite-field fits on synthetic data |
| `interfaces_engine` | `PGMEngine` energy, forces, dipoles, atomic virial for two successive configurations (predictor path) |
| `protein_peptide` | solvated ACE-ALA-SER-NME: `load_amber` + `amber_template` + flexible engine with X-H constraints and HMR (60 steps), the bytes of the written pmemd-pgm prmtop and mdin |

Not covered (outside the CPU harness): GPU-specific paths (bitwise GPU runs are not reproducible
because of atomics in the PME spreading), the external codes (ASE/i-PI/OpenMM drivers: covered by
their unit tests when installed), `LiquidFit` iterations and the bonded fitting (covered by pytest).

### 7.3 Status

- Recorded on the cpu-short nodes (Intel Xeon Gold 6418H), jax/jaxlib 0.11.2, numpy 2.5.3,
  16 threads: 35 cases in 436 s of serial CPU time (about 2 min wall in 7 parallel group jobs).
- Re-run with 16 threads: 35/35 bitwise. Re-run with 8 threads (one job, all cases): 35/35 bitwise.
- Sensitivity check: changing the last digit of `KB` in `md/integrate.py` (1e-16 relative) left
  `gas_model` bitwise and made `md_rigid_bussi` and `flex_flux` fail (51 and 35 arrays differ, relative
  deviations 1e-16 to 3e-11), i.e. the harness sees any change of the trajectories.
- GPU baseline (`scripts/dev/gpu_bench.sh`, master `e72c57c`, gpu-2-1, RTX PRO 6000 Blackwell),
  ms/step in two rounds:

| Case | Round 1 | Round 2 |
|---|---:|---:|
| rigid 512 waters, mixed, Langevin, 1 fs | 0.754 | 0.747 |
| rigid 4096 | 2.046 | 1.991 |
| constraints 512, 2 fs | 0.855 | 0.856 |
| constraints 4096, 2 fs, Bussi | 2.112 | 2.167 |
| iEL/0-SCF 512, 2 fs, Bussi | 0.491 | 0.498 |
| constraints + HMR + MTS 2, 6 fs, Bussi | 1.101 | 1.062 |
| rigid 512, double | 1.940 | 1.973 |

Run-to-run noise is up to 3.5 %; a phase passes when no case is slower than master by more
than that in both rounds of an alternating A/B run.

---------------------------------------------------------------------------------------------------

## 8. Ruff report (configuration added, not applied)

`pyproject.toml` now has `[tool.ruff]` (line length 120, target py310, `paper/`, `runs/`, `.tools/`
excluded), `[tool.ruff.lint] select = ["F", "I", "UP"]`, isort with `known-first-party = ["pgm_jax"]`,
format with double quotes. ruff 0.16.9 (static binary in `.tools/bin/ruff`, ignored by git; runs
on the head node). Numbers for the 204 tracked Python files of master (the new harness files are
already clean):

`ruff check`: 252 findings in 119 files, 217 fixable automatically (25 more with `--unsafe-fixes`).

| Rule | Count | pgm_jax | scripts | tests | examples | Auto-fix |
|---|---:|---:|---:|---:|---:|---|
| I001 unsorted imports | 159 | 16 | 80 | 54 | 9 | yes |
| F401 unused import | 30 | 5 | 17 | 8 | 0 | yes |
| UP037 quoted annotation | 28 | 28 | 0 | 0 | 0 | yes |
| UP031 printf-style formatting | 21 | 0 | 19 | 1 | 1 | unsafe / manual |
| F841 unused variable | 11 | 3 | 2 | 6 | 0 | manual |
| F821 undefined name | 3 | 0 | 3 | 0 | 0 | manual |
| Files with findings | 119 | 30 | 53 | 33 | 3 | |

Notable: F841 in library code at `fit/liquid.py:108,179` and `interfaces/openmm.py:82`; the three
F821 are `win` in `scripts/solvation_free_energy.py:384-387` (a closure over a loop variable that is
`del`eted at the end of the loop body: not a runtime bug, but the `del` should go).

`ruff format --diff`: 201 of 204 files would change (pgm_jax 85/88, scripts 77/77, tests 35/35,
examples 4/4), 16,726 changed lines (pgm_jax +4,798 / -2,032, scripts +4,850 / -1,668,
tests +2,422 / -823, examples +103 / -30). Most of it is (a) 551 lines longer than 120 characters
wrapped, (b) column-aligned trailing comments collapsed to two spaces (652 aligned comments in
`pgm_jax/`; about a third of the removed lines differ only in whitespace), (c) one statement per
line, magic trailing commas and quote normalization. ruff 0.16 also
formats Python code blocks in Markdown: README.md and 15 files in `docs/` would change (exclude
`*.md` or accept; part of Q11).

---------------------------------------------------------------------------------------------------

## 9. Decisions (owner, 2026-09-29)

The owner answered the open questions of 6.3; where the owner delegated, the coordinator chose.
These decisions override the recommendations above where they differ.

| # | Decision | Where it lands |
|---|---|---|
| D1 | Hard breaks: no compatibility shims for renamed APIs or moved modules; every call site in the repository is updated in the same commit (library, scripts, tests, examples, docs, `paper/` scripts that import `pgm_jax`). (Q3) | every phase |
| D2 | Script CLI units: common MD units with the unit in the option name: `--dt-fs`, `--cutoff-nm`, `--time-ns`, `--temperature-K`, `--pressure-bar`, ... `scripts/md/run_md.py` keeps its Amber style (Angstrom, Amber option names) for comparisons with pmemd and says so in its help. (Q1, Q14) | P9 (`pgm_jax/cli/args.py` groups from P3 on) |
| D3 | Checkpoints: a new versioned format, `.npz` arrays + JSON header with a format version, for every driver; old `.chk` pickles stay loadable through a reader / converter, tested on a real old checkpoint. (Q7) | P4 |
| D4 | Research code (`scripts/bonded/`, the `reports/` and `validation/` producing scripts, one-off validation scripts) is cleaned to the same standard as the library (docstrings, structure, no `sys.path` edits, no hard-coded user paths) and organized by purpose. (Q10) | P3 (imports, paths), P9 (layout, docstrings) |
| D5 | Every function (public and private, including non-trivial nested functions) has a docstring or a clear comment: what it does, arguments with units, returns, non-obvious physics / algorithm notes; every module has a module docstring; clear module responsibilities, no duplication, small functions. | P8 (library), P9 (scripts); new code from P1 on |
| D6 | `ensemble=`, `gamma=`, `tau_t=`, `barostat_interval=` are removed in favour of thermostat / barostat objects (Q2). Defaults that exist today are kept so that results do not change (the default thermostat stays Langevin 1/ps; the docs recommend Bussi) (Q4). | P5 |
| D7 | Python `logging` (`logging.getLogger("pgm_jax...")`) for diagnostics; MD tables (log lines, observables) still go to their files / streams. (Q5) | P4, P5 |
| D8 | Nested `MDSettings` groups with a `replace(**flat)` convenience. (Q6) | P6 |
| D9 | One `ParameterSpace` replaces `fit.ParameterSpace`, `fe_grad.ParamSpace` and `qmfit.ParamMap`. (Q8) | P7 |
| D10 | A single `pgm-jax` CLI entry point (`[project.scripts]`) dispatching to subcommands for the main scripts. (Q9) | P9 |
| D11 | numpy docstring style; ruff rules E, W, F, I, UP, B (subset) and D (numpy convention), D enforced on the library and the scripts; `ruff format` accepted (column alignment of comments goes); type hints on all function signatures, no mypy / pyright gate for now; Python floor 3.10. (Q11-Q13) | P1 (E, W, F, I, UP, B, format), P8/P9 (D, hints) |
| D12 | README shortened to overview + quick start + links; the long feature and validation material moves into `docs/` with an index; `NOTES_*.md` -> `docs/dev/notes/`. (Q16) | P10 |
| D13 | Data layout (decided here): `data/` is the single top-level home of tracked data: `data/inputs/` (QM sets, bonded-study molecules and frames; today's `data/qm`, `data/bonded`), `data/validation/` (today's `validation/`: Amber reference outputs and validation JSON results), `data/reports/` (today's `reports/`: figures and tables of studies). Scripts write there through one `pgm_jax.cli.paths` helper (repository-relative, overridable with `PGM_DATA`). (Q15) | P9 (moves together with the scripts that write them) |

B subset (D11): B905 (`zip` without `strict`), B008 (calls in default arguments: the defaults are
frozen dataclasses such as `MDSettings()`) and B023 (loop variables in closures: the flagged
closures are jitted and called inside the same iteration) are not selected; the rest of B is.
E741 (ambiguous names `l`, `I`, `O`) is not selected: they are the physics notation (angular
index, identity matrix). E402 is resolved by removing the `sys.path` edits (P3) and by setting
`jax_enable_x64` in `tests/conftest.py` instead of in every test module (P1).

### 9.1 Execution notes (P1-P3)

- P1: ruff rules E, W, F, I, UP, B (subset as above) enforced on the whole repository; D (numpy
  convention) is configured and switched on in P8/P9 together with the docstrings.
- P2: `pgm_jax/units.py` holds every constant.  Values that differed between modules (Bohr radius of
  three CODATA releases) are kept under their own names so that no result changes (owner, after P3:
  the three Bohr values stay separate for now; the owner decides later which one to keep);
  `KJMOL_NM3_PER_BAR = 1.0 / BAR_PER_KJMOL_NM3` is computed as before (bitwise the same double).
  `md.box` gained `box_from_cell` / `cell_parameters` / `centers_of_mass`, `md.io`
  `read_coordinates_nm`, `analysis/stats.py` the time-series statistics; one prmtop parser
  (`prmtop.Prmtop`); `System.from_prmtop` / `param.read_prmtop_molecules` / `param.share_identical`
  replace `simulation._dedupe`.  The full test run after P2 found four imports of removed names in
  tests and scripts (`md.pimd.HBAR`, `interfaces.ipi.BOHR_NM`, `dielectric.E_NM` / `C_LIGHT`,
  `efield.E_CHARGE`); they are fixed in the P3 commits, and `runs/v/check_imports.py`-style static
  checks (every imported name and every attribute of an imported module exists) now run with ruff.
- P3: the location helper of D13 is `pgm_jax/paths.py` (not `pgm_jax/cli/paths.py`): the library
  itself needs it (the pGM-pol table of `param`).  External data and programs are found through
  environment variables with the development defaults (`PGM_GVDW_DATA`, `AMBERHOME`,
  `PGM_PMEMD_BIN`, ...), so no script names a user's home directory any more.  Shared example
  systems live in `pgm_jax/models/toy.py` (used by the harness, scripts and, from P10, the tests),
  the bonded-study catalogue and loaders in `pgm_jax/bonded/study/` (`families.py`, `data.py`,
  `gas_md.py`), the i-PI helpers in `pgm_jax/interfaces/ipi_tools.py`.  No `sys.path` edits remain
  except in `scripts/validation/efield_identical.py` and the subprocess script of `validate_vsites.py identical`,
  which load another code tree on purpose.  `bonded/terms/core._N` is a numpy array, so importing
  `pgm_jax` creates no JAX array and `jax_enable_x64` may be set after the imports.

### 9.2 Execution notes (P4-P6)

- P4 (engine core): `md/driver.py` holds the host-side pieces every driver shares (`block_length`,
  `retry_block`, `advance_with_rebuilds`, `LogTable`, `Stopwatch`, the checkpoint format) and
  `md/engine.py` the `MDEngine` base class of `Simulation` and `FlexibleSimulation` (neighbour lists
  and sizes, blocks with overflow handling, observables, pressure, restraints / biases / fields, the
  run loop and checkpoints; the two engines keep only their construction and four hooks).
  `PIMDSimulation`, `MDReplicas` / `ReplicaExchange`, `Walkers` (which lost its copy of the replica
  resize), `FieldReplicas` and `FreeEnergyRun` use the same retry, tables and checkpoints.  The
  compiled steps are untouched (harness 35/35 bitwise after every commit; GPU speed as before).
  Checkpoints: an `.npz` archive with a `__header__` JSON entry `{"format": "pgm_jax checkpoint",
  "version": 1, "kind": ..., "content": ...}`; state pytrees are stored leaf by leaf and rebuilt on
  the driver's current state as template (`encode_content` / `decode_content`).  Old pickle
  checkpoints are recognised and read by every `load_checkpoint`; saving converts them.  Tests:
  checkpoints of every driver written with the code of `e72c57c` (`tests/data/legacy_checkpoints`,
  generator included) and one of a real run of the old code (`real_npt.chk`, older than `e72c57c`).
  Not moved onto the core: `interfaces/engine.PGMEngine` (its per-slot / per-batch retry loops
  re-point slots, change the bead margin and rebuild batch programs; forcing them into
  `retry_block` would not remove code).  The progress logs of `Walkers`, `FreeEnergyRun` and
  `FieldReplicas` are now `LogTable` tables (a format change; nothing in the repository parses them).
- P5 (engine API): as in table 4.1.  Diagnostics go to Python loggers under `pgm_jax`
  (`pgm_jax.cli.args.setup_logging()` in the scripts prints them as `# ...` lines); `log=` is the
  stream that receives the log-table rows (default `None`).  `pgm_jax.cli.args.coupling_from_options`
  maps the scripts' unchanged `--ensemble/--thermostat/--gamma/--tau/--press` options to the objects
  (the CLI renames are P9).  `protein/pmemd.pmemd_mdin` writes Amber input and keeps Amber's
  `ensemble`, `gamma`, `tau_t` names; `interfaces/ipi_tools` (i-PI XML) keeps i-PI's.  `bias.toy.ToyLangevin`
  (a toy integrator, not an engine thermostat) keeps `gamma`.  `LiquidFit` got the thermostat /
  barostat objects; its other names (`T`, `tol`, `sys_`, ...) belong to P7.
- P6 (settings): `MDSettings(terms, cutoffs, neighbors, pme, induction, precision, differentiable,
  adjoint_tol)` with `Terms`, `Cutoffs`, `NeighborList` (skin and the list kind, which was the
  engines' `neighbor_list=` keyword), `PMESettings` and `Induction` (with the extended-Lagrangian
  dipoles nested as `induction.iel`, an `ExtendedLagrangian`); `MDSettings().replace(**flat)` takes
  the old flat names (`FLAT_SETTINGS`), whole groups and the top-level fields.  The old boolean
  property `induction` is `has_induction`.  `precision`, `differentiable` and `adjoint_tol` stay top
  level (one field each would make one-field groups).  `PeriodicModel` / `PeriodicPGM` names are
  left to P7.

### 9.3 Execution notes (P7)

- One `pgm_jax.fit.params.ParameterSpace` (D9) replaces the three parameter maps.  Its blocks are
  `Param`s of kind `"scale"` (theta = ln s, as the old `fit.ParameterSpace`), `"shift"` (added in
  native units) or `"values"` (theta is the parameter itself, as `fe_grad.ParamSpace` and
  `qmfit.ParamMap`); charges in a `"values"` block with `neutral=molecules` are parameterized in the
  null space of the neutrality constraints (the old `ParamMap` behaviour, names `q:null{j}`).  Each
  kind keeps the exact array operations of the class it replaces, so every fit, QM-fit and
  free-energy-gradient result is bitwise the same (harness cases `fit_frames`, `qmfit_synthetic`,
  `fe_windows`, `periodic_model`).  `fit.qm.parameter_space(...)` builds the QM-fit space with the
  default bounds (`BOUNDS`) and step sizes (`STEPS`, was `SCALES`).
- The free-energy estimators and fitting targets moved to `pgm_jax/fit/free_energy.py`;
  `md/fe_grad.py` keeps the sampling side (`ParameterGradients`, `gas_leg_gradient`,
  `alchemical_map`, `scaled_params`).  `gas_leg_gradient` stays with the sampling code (it
  differentiates the alchemical gas-phase Hamiltonian), unlike the plan in table 4.3.
- Names: `temperature` [K], `pressure` [bar], `dipole_tol`, `cutoff`, `ewald_beta`, `box`,
  `positions`, `system` in `fit/`, `analysis/`, `PeriodicModel` and `PeriodicPGM` constructors;
  time steps in ps (the library unit) are `dt`, durations keep `_ps` (`tau_ps`, `run_ps`,
  `segment_ps`, `equil_ps`, ...).  The per-configuration methods of the reference models
  (`energy(pos, params, H)`, `forces`, `strain_derivative`, ...) keep their short argument names;
  they are positional everywhere.  `PeriodicLJ` / `PeriodicGVDW` (internal to `PeriodicModel`) keep
  `rc`.  `LiquidFit` writes `"temperature"` instead of `"T"` in prefix.json (nothing in the
  repository reads it).
- `paper/main.tex` names the new class; the PDF was not rebuilt.
- Docstring tooling for P8: `scripts/dev/check_docstrings.py` (AST: every module, class, function and
  method, nested functions from 6 lines, per-file or per-directory counts, `--fail`,
  `find_missing()` for a test); ruff D (numpy convention) runs report-only through
  `ruff check --config ruff-docstrings.toml --exit-zero` (the enforced configuration in
  `pyproject.toml` is unchanged); the standard is `docs/dev/style_guide.md`.
