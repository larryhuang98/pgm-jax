# Cell dipole, induced dipoles and the static dielectric constant

`pgm_jax/md/dipoles.py` computes the dipole moment of the periodic cell and records it during MD;
`pgm_jax/md/dielectric.py` and `scripts/dielectric.py` turn the recorded series into the static
dielectric constant (with error bars and convergence) and, optionally, an infrared spectrum.

## Usage

```bash
python scripts/run_md.py -p water.prmtop -c water.rst7 -o md --dt 2.0 --thermostat bussi \
    --nsteps 5000000 --report 5000 --restart 250000 --dipoles 25          # M every 50 fs -> md.dip
python scripts/dielectric.py md.dip --skip 200                             # eps, errors, convergence
python scripts/dielectric.py ir.dip --ir ir_spectrum.dat                   # M sampled every step
```

```python
sim.run(nsteps, report=5000, traj=5000, dipoles=25, induced=5000, prefix="md")   # also md.mu.nc
from pgm_jax.md.dipoles import cell_dipole, CellDipole, read_dipoles
cell_dipole(sim)            # {"charge", "perm", "ind", "total"} e nm, "debye": the same in D, "molecular" (nmol, 3)
CellDipole(ff).components(pos, H, mu)       # fixed frames: rows M_q, M_perm, M_ind (e nm)
CellDipole(ff).polarizability(pos, H, idx)  # cell polarizability tensor (nm^3), tin-foil Ewald
meta, d = read_dipoles(["md.dip", "md2.dip"])   # continuation segments; superseded records dropped
```

Both engines are supported (`Simulation`, rigid bodies; `FlexibleSimulation`, atoms with
constraints). `dipoles=n` does not shorten the driver's blocks: M is evaluated on the device
inside the block (a `lax.scan` over sub-blocks of the integrator), so sampling every step costs
no host round trip; on 512 pGM waters (2 fs, sampling every 25 steps) the speed was unchanged
(199 vs 196 ns/day). Samples of a block that the driver repeats after a list overflow are
discarded. The Amber trajectory, restart and log files are unchanged.

`prefix.dip` is a text table with a commented header (`# key = value`: temperature, ensemble,
thermostat, time step, sampling interval, atom and molecule counts, net charge, charged molecules,
electrostatics level) and the columns
`step time_ps temp_K volume_nm3 Mq_x Mq_y Mq_z Mp_x Mp_y Mp_z Mi_x Mi_y Mi_z mol_dipole alpha_nm3`
(dipoles in e nm; `mol_dipole`: mean |dipole| of the molecules; `alpha_nm3`: the cell's
electronic polarizability, 1/3 of its trace, every 100th sample (`DipoleRecorder.alpha_every`), `nan`
otherwise or when a solve did not converge; `temp_K`: the
kinetic temperature). `prefix.mu.nc` holds per-atom induced dipoles (NetCDF-3: `time`, `step`,
`induced_dipoles (frame, atom, spatial)` in e nm, float32).

## Definitions and conventions

    M = M_q + M_perm + M_ind,   M_q = sum_k sum_{i in k} q_i (r_i - R_k),   M_perm = sum_i p_i,   M_ind = sum_i mu_i

- A Gaussian charge or dipole has the dipole moment of the point multipole at its centre, so M is
  the dipole moment of the model's charge density. Levels without permanent or induced dipoles
  (`elec = "q" | "qi" | "qp"`) drop those parts.
- Molecules are whole: both engines keep them whole and shift only whole molecules by lattice
  vectors, and `CellDipole` uses the engine's positions as they are (no minimum image, which would
  break molecules longer than half the box).
- For neutral molecules M_q = sum_i q_i r_i, independent of the origin and of the periodic image
  of each molecule, hence continuous across the driver's re-wrapping.
- A charged molecule (ion, charged residue) contributes its dipole about its centre of mass
  (physical masses), as GROMACS `gmx dipoles`. The translational part sum_k Q_k R_k is left out:
  it jumps by Q_k L whenever molecule k is wrapped, it is the time integral of the ionic current
  (conductivity), and for a net-charged cell it depends on the origin. M is then the "molecular"
  dipole M_D of the electrolyte literature, defined for any cell; the header records the net
  charge and the number of charged molecules, and `scripts/dielectric.py` refuses such series
  unless `--molecular` is given.

## The static dielectric constant with adiabatic induced dipoles

Smooth PME omits the k = 0 term: the box is surrounded by a conductor (tin-foil boundary
conditions), so the uniform field acting on the charges is the Maxwell field and
eps - 1 = (4 pi / 3V) tr d<M>/dF (F in e/nm^2, energy -KE M.F). The pGM induced dipoles are
adiabatic: they minimise the energy at each configuration and have no thermal fluctuations of
their own. With U(R, F) = min_mu E(R, mu) - KE M.F,

    d<M>/dF = beta KE (<M M> - <M><M>) + <dM/dF at fixed nuclei>,

and the second term is the cell's electronic polarizability alpha_cell. Therefore

    eps = eps_inf + (<M.M> - <M>.<M>) / (3 eps0 <V> kB T),     eps_inf = 1 + 4 pi <alpha_cell / V>,

with M the total dipole (induced dipoles included). The fluctuation of the total dipole alone
misses eps_inf - 1: a frozen polarizable crystal has M = 0 at every step and eps = eps_inf > 1.
`CellDipole.polarizability` computes alpha_cell exactly for the model (three CG solves with the
force field's induction operator for unit fields along x, y, z; the Ewald dipole couplings are
included, so no Clausius-Mossotti approximation). For pGM water eps_inf = 1.80: it adds 0.8 to eps.
Drew & Gilson (JCTC 21, 6964 (2025), Eqs. 17-18) use the same formula for their
induced-dipole water, with eps_inf approximated by 1 + sum_j alpha_j / (eps0 V) (no local field);
Drude models with thermalized Drude particles would carry the electronic part in the fluctuations
instead.

Units are SI (M in C m, V in m^3); `dielectric.fluctuation` equals 4 pi KE <dM^2> / (3 V kB T) in
the model's units (tested). T is the thermostat target: the kinetic temperature reads 2-4 K low at
2-4 fs (discretization of the kinetic estimator; the configurations are canonical at T).

**Error bars.** M relaxes slowly (tau_M, from an exponential fit of its autocorrelation, is 3.7 ps
for the pGM box, 6.7 ps with the pGM3P-25 geometry and 6.9 ps for TIP3P; the Debye time of water is
8.3 ps), so samples are correlated. The error is the jackknife over
contiguous blocks (leave one block out, full estimator including <M>^2); `scripts/dielectric.py`
prints it against the number of blocks (it must plateau) and the running estimate against the run
length. For a Gaussian M with exponential correlation the relative error of eps - eps_inf is
sqrt(2 tau_M / (3 T_run)), independent of the system size, so small boxes are the cheapest way to
converge eps (512 waters: 1.3 % after 15 ns for tau_M = 3.7 ps).

**Infrared spectrum** (`--ir`): alpha(w) n(w) = beta w^2 C(w) / (6 c eps0 V), C(w) the Fourier
transform of <M(0).M(t)> (classical line shape = quantum line shape with the harmonic correction),
from Welch periodograms. Sample M every step or two; use physical masses (no HMR).

## Validation (512 waters, 298 K, 1 bar)

Settings of every run: 512 waters in the truncated octahedron of
`~/pgm-gvdw-data/inputs/lj/inpcrd.restrt`, NPT 298 K / 1 bar (Bussi, tau 1 ps; Monte Carlo
barostat every 100 steps), PME 48^3 order 6, beta 4 nm^-1, 0.9 nm cutoff with the LJ tail, dipole
tol 1e-5, mixed precision, one RTX PRO 6000; M every 25 steps; the first 200 ps (500 ps after the
geometry change of the pGM3P-25 run) discarded; errors are jackknife over 10 blocks (the block
tables plateau at these values, and they agree with sqrt(2 tau_M / 3 T_run)).
`scripts/water_dielectric.py` makes every run (the production run of the pGM box used
`run_md.py --dt 2.0 --thermostat bussi --nfft 48 48 48 --order 6 --dipoles 25`, stopped at 11.5 ns
and continued from its 11.0 ns checkpoint with `--checkpoint`; `read_dipoles` keeps the continued
records).

| Model | Engine, dt | Run | Density (g/cm^3) | <mu_mol> (D) | tau_M (ps) | eps_inf | eps |
|---|---|---|---|---|---|---|---|
| pGM box of the README (`rayl_512_v2.prmtop`) | rigid bodies, 2 fs | 14.9 ns | 1.018 | 1.99 | 3.7 | 1.798 | **31.0 +- 0.4** |
| same | constraints, H mass 4.0, 4 fs | 9.8 ns | 1.018 | 1.99 | (4.2, heavier H) | 1.798 | **30.6 +- 0.5** |
| pGM3P-25 with the paper's geometry and LJ | rigid bodies, 2 fs | 9.5 ns | 1.010 | 2.13 | 6.7 | 1.795 | **33.9 +- 0.7** |
| TIP3P control (point charges, `elec="q"`) | rigid bodies, 2 fs | 9.8 ns | 0.986 | 2.35 | 6.9 | 1 | **103.8 +- 3.0** |
| pGM3P-25, published (Wu et al., JCTC 21, 3563 (2025)) | pmemd-pgm | | 1.003 | 2.413 | | | 84.3 |
| TIP3P, literature | | | 0.98 | 2.35 | | | 89-104 (94, Vega & Abascal, PCCP 13, 19663 (2011)) |
| Experiment | | | 0.997 | ~2.9 | 8.3 (Debye) | 1.78 (n^2) | 78.4 |

Running estimate of the pGM box (eps against the length of the run used):

| run used (ns) | 1.9 | 3.7 | 5.6 | 7.5 | 9.3 | 11.2 | 13.0 | 14.9 |
|---|---|---|---|---|---|---|---|---|
| eps | 30.5 +- 0.8 | 30.9 +- 0.5 | 30.7 +- 0.7 | 30.8 +- 0.6 | 31.3 +- 0.5 | 31.1 +- 0.4 | 31.1 +- 0.4 | 31.0 +- 0.35 |

The jackknife error of the full run is 0.25-0.42 for 4 to 50 blocks (0.35 for 10) and
sqrt(2 tau_M / 3 T_run) predicts 0.38. With 1 + fluctuation alone (no eps_inf) the result would
be 30.2.

What the numbers say:

- **The recording and the formula are right.** The TIP3P control lands inside the literature
  range (89-104; reported values differ by the long-range and cutoff treatment and by run length),
  the single-molecule test reproduces the gas-phase model's total dipole and polarizability, and
  eps_inf = 1.798 from the cell polarizability agrees with Clausius-Mossotti on the model's
  gas-phase molecular polarizability (1.481 A^3 -> 1.803; without the local field, as Drew &
  Gilson, 1.63).
- **The two engines agree** (rigid bodies at 2 fs vs atoms with constraints, heavier hydrogens and
  4 fs: 31.0 +- 0.4 and 30.6 +- 0.5), as they should for a configurational property.
- **pGM water's eps is about 31-34, not 84.** The charges of pGM3P-25 are large
  (q_O = -2.04 e) and every intramolecular pair interacts, so the covalent dipoles and the
  intramolecular induction oppose the charge dipole: per molecule 5.74 D from the charges, 4.11 D
  with the covalent dipoles, 1.38 D with induction in the gas phase, 1.99 D in the liquid. The
  fluctuation term of the box splits accordingly (14.9 ns): charges alone 229, charges + covalent
  dipoles 117, charges + induced dipoles 94, all three 29.2. The induced dipoles are strongly
  anti-correlated with the permanent ones (cross term -118). The published 84.3 and 2.413 D do
  not follow from the same parameters with the complete dipole; leaving out the covalent dipoles
  comes closest (eps = 96 here, 100 with the paper's geometry and LJ). pGM-JAX's induced dipoles
  match pmemd-pgm's to 1e-10 and its gas-phase dipoles PyRESP's, so the difference is not in the
  model's electrostatics; the paper does not describe how its dipoles and eps were computed. Worth
  settling before eps is used as a fitting target.
- The README's box (`rayl_512_v2.prmtop`) has pGM3P-25's electrostatic parameters (identical to
  those in the paper's repository github.com/yxwu21/pGM3P-25, whose README lists the charges
  divided by 18.2223, Amber's charge scaling) on TIP3P's
  geometry (0.9572 A, 104.5 deg) and TIP3P's Lennard-Jones (A = 581936, B = 594.8); the paper's
  model has 0.9745 A, 103.64 deg and A = 622716, B = 600.41. The difference moves eps from 31 to
  34 and the liquid dipole from 1.99 to 2.13 D.

**Infrared spectrum** (pGM box, rigid bodies, NVT 2 fs, M every step, 180 ps after 20 ps;
`scripts/water_dielectric.py --ensemble nvt --ns 0.2 --dipoles 1`, then `scripts/dielectric.py
--ir`): one broad librational band with its maximum at about 470 cm^-1 (alpha n = 2.6e3 cm^-1)
and a shoulder near 250 cm^-1 (hydrogen-bond stretch), falling to 5 % of the maximum by
1000 cm^-1; rigid molecules have no intramolecular bands. Liquid water's librational band peaks
near 680 cm^-1; the band of the pGM box lies about 200 cm^-1 lower.

## Independent check: pmemd.pgm.cuda with the published parameters

To separate the model from the engine, pGM3P-25 with its published geometry and Lennard-Jones was
written as a pmemd-pgm topology (`scripts/pgm3p25_prmtop.py`) and sampled with pmemd.pgm.cuda
itself; the cell dipoles of its trajectories were then evaluated with the model's induced dipoles
re-solved at every frame (`scripts/trajectory_dipoles.py`, tol 1e-6, one frame per ps).

Runs: pmemd.pgm.cuda_SPFP (`~/ambers/pgm-larry-install`), 4,096 waters (the 512-water box 2 x 2 x 2;
pmemd.pgm.cuda needs three neighbour-list cells across the box), NPT 298 K / 1 bar, Langevin 1/ps,
Monte Carlo barostat, SETTLE, 2 fs, 9 A cutoff, PME 96^3 order 6, ew_coeff 0.4, vdwmeth 1,
dipole_scf_tol 1e-5 (`pgm3p25_prmtop.py --mdin`); four independent runs of 0.1 ns + 3 ns on four GPUs
(114 ns/day each), the first 300 ps of each discarded: 10.8 ns. The heat of vaporization uses the
gas-phase energy of one molecule at the published geometry (-976.42 kcal/mol with pmemd-pgm's
Coulomb constant; `Model`, dense pGM) and <EPtot>/N from pmemd.

| pGM3P-25, published parameters | pmemd.pgm.cuda (this check) | pgm_jax (table above) | Wu et al. 2025 |
|---|---|---|---|
| density (g/cm^3) | 1.0097 +- 0.0002 | 1.010 | 1.003 |
| heat of vaporization (kcal/mol) | 9.54 | | 9.847 |
| mean molecular dipole, liquid (D) | 2.124 | 2.13 | 2.413 |
| molecular dipole, gas phase (D) | 1.462 (experiment 1.855) | | |
| eps (M_q + M_perm + M_ind, + eps_inf 1.794) | **34.3 +- 0.6** | **33.9 +- 0.7** | 84.3 |

Cross-check of the dipoles against the pGM CPU codes (`~/ambers/pgm-larry`, `dipole_print=1`), same
coordinates (512 waters, published parameters) and PME settings:

- sander (`pGM_compute_dipole`: system moment = sum of q r + permanent dipoles + induced dipoles
  over whole molecules, the quantity whose variance it prints for eps): (18.896334, -15.957502,
  2.330521) e A; pgm_jax's M_q + M_perm + M_ind: (18.8963, -15.9575, 2.3306) e A.
- pmemd.pgm (CPU; `fort.100`, per-water total moment, charges + permanent + induced dipoles):
  pgm_jax's molecular dipoles agree to 1.4e-7 e A for every molecule that is whole in pmemd's image
  coordinates (451 of 512; mean 2.10 D). pmemd sums image (wrapped) coordinates, so the 61 molecules
  split across the box print meaningless moments (e.g. 27 e A instead of 0.44): that output must
  not be used for molecules that cross the box boundary.
- One molecule in vacuum (sander, ntb = 0): EEL -976.4185 kcal/mol and total dipole
  (-0.0736, -0.0359, -0.2932) e A = 1.462 D, both as pgm_jax.

So the dipole of the model, induced dipoles included, is the one the pGM CPU codes compute.

Other definitions of M on the same pmemd frames (1 + fluctuation; errors 2 %): pGM charges only 236,
charges + covalent dipoles 123, charges + induced dipoles 100, TIP3P's point charges (-0.834 / +0.417 e)
40.3.

- **The two engines agree** (34.3 +- 0.6 and 33.9 +- 0.7; liquid dipole 2.12 and 2.13 D; density
  1.0097 and 1.010): eps of about 34 is a property of pGM3P-25 with these parameters, not of
  pgm_jax's sampling or analysis.
- **The published dipole is not the model's.** 2.413 D is exactly the dipole of TIP3P's charges at
  the published geometry, 2 x 0.417 e x 0.9745 A x cos(51.82 deg) = 2.4132 D, the same for every
  molecule and frame. The repository's `dipole_calc.py` (github.com/yxwu21/pGM3P-25) takes the
  charges from the topology with MDAnalysis, which reads the prmtop's CHARGE section; tleap fills it
  with TIP3P's charges, and pmemd-pgm never uses it (it reads POL_GAUSS_MONOPOLES_LIST). No
  covalent or induced dipoles enter. `pgm3p25_prmtop.py` therefore writes the pGM monopoles into
  CHARGE (in Amber units), so charge-only tools at least see the model's charges.
- **The published eps = 84.3 is not reproduced** by any of these definitions of M on these
  trajectories (TIP3P's point charges give 40, not 84). The public repository does not contain the
  dielectric analysis (its `analysis.py` calls scripts in a directory on this cluster that is not
  public), so how 84.3 was obtained cannot be checked from here.
- Density and heat of vaporization differ from the published ones by 0.7 % and 3 %; the paper does
  not give all its run settings (512 waters, CPU pmemd; cutoff and long-range correction not stated).
- The model is under-polar in the gas phase (1.46 D against 1.855 D): consistent with eps of about 34
  and with its hydration free energy (-4.91 +- 0.07 kcal/mol against -6.3; `docs/free_energy.md`).

## Limits

- Charged molecules: M is the molecular dipole M_D (no ionic current), so for electrolytes eps
  from its fluctuations leaves out the M_D-current cross correlation and the conductivity
  (`--molecular` to accept it). A net-charged cell needs no special treatment.
- The fluctuation formula is the one for conducting (tin-foil) boundary conditions, which is what
  the PME of `pgm_jax.md` implements; it does not apply to other boundary conditions.
- Under NPT the formula uses <V>; the V-M^2 correlation is neglected (far below the statistical
  error for liquids).
- `prefix.dip` is text, about 230 bytes per sample (a sample every 25 steps for 20 ns at 2 fs:
  90 MB). Per-atom induced dipoles are written at the driver's block ends, so their interval
  enters the block length (as `traj` does).
- The cell polarizability costs three CG solves (from zero, at `dipole_tol`) every
  `DipoleRecorder.alpha_every` = 100 samples; a solve that does not converge within `max_iter`
  gives `nan`, and `scripts/dielectric.py` reports how many did.
- Kinetic observables (IR spectrum, relaxation times) need physical masses; with hydrogen mass
  repartitioning only eps and other configurational averages are meaningful.
