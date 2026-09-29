# Virtual sites

Virtual sites are massless interaction sites whose positions are functions of other atoms of the
same molecule: the M site of TIP4P-type water, the lone pairs of TIP5P or of an ether oxygen, a
sigma-hole site beyond a halogen, a charge off a bond.  In pGM-JAX a site is an atom of its
`Molecule` with zero mass, listed with its construction in `Molecule.vsites`.  Like any atom it
carries a charge and may carry a pGM Gaussian radius, a polarizability, covalent dipoles and van
der Waals parameters.  Both MD engines support sites (`Simulation`: rigid molecules;
`FlexibleSimulation`: atoms with bonded terms and constraints). Amber extra points are read from
prmtops. The code is in `pgm_jax/md/vsites.py`.

## Defining sites

```python
from pgm_jax.md.vsites import VirtualSite, VirtualSites
from pgm_jax.system import Molecule, System

# TIP4P-Ew as point charges: O, H1, H2 and the M site (element "EP": massless)
tip4pew = Molecule(
    "T4E",
    ["O", "H", "H", "EP"],
    ["OW", "HW", "HW", "EP"],
    q=[0.0, 0.52422, 0.52422, -1.04844],
    radius=[1e-4] * 4,
    alpha=[0.0] * 4,
    lj_rmin_half=[0.1776, 0, 0, 0],
    lj_sqrt_eps=[0.8252, 0, 0, 0],  # R* (nm), sqrt(kJ/mol)
    vsites=[VirtualSite.tip4p(3, 0, 1, 2, d_om=0.0125)],
)  # (1 - 2a, a, a) average
VirtualSites.of(System([tip4pew] * 512)).place(pos, H)  # positions with every site rebuilt
sim = Simulation(System([tip4pew] * 512), pos, H, MDSettings(elec="q"), barostat=MonteCarloBarostat(), dt=0.002)

# or from tleap (source leaprc.water.tip4pew): extra points become "amber" sites
sim = Simulation.from_amber("tip4pew.prmtop", "tip4pew.rst7", charges="amber", settings=MDSettings(elec="q"))
asys = load_amber("protein_opc.prmtop", "protein_opc.inpcrd")  # water with extra points: RigidTemplate + site
```

| Kind | Constructor | Position (d_k = mi(r_k - r_host), host = first parent) |
|---|---|---|
| `average2` | `VirtualSite.average2(s, a, b, w_a, w_b)` | w_a r_a + w_b r_b, w_a + w_b = 1 (OpenMM TwoParticleAverageSite; w outside [0, 1] extrapolates) |
| `average3` | `VirtualSite.average3(s, a, b, c, w_a, w_b, w_c)`, `VirtualSite.tip4p(...)` | w_a r_a + w_b r_b + w_c r_c, weights summing to 1 (ThreeParticleAverageSite) |
| `outofplane` | `VirtualSite.out_of_plane(s, a, b, c, w_ab, w_ac, w_x)` | r_a + w_ab d_b + w_ac d_c + w_x (d_b x d_c), w_x in nm^-1 (OutOfPlaneSite, GROMACS 3out) |
| `local` | `VirtualSite.local(s, atoms, wo, wx, wy, p)` | origin sum wo_k r_k (sum 1); x = sum wx_k r_k, y = sum wy_k r_k (sums 0); e_x = x/\|x\|, e_z = x cross y normalised, e_y = e_z cross e_x; origin + p_x e_x + p_y e_y + p_z e_z (LocalCoordinatesSite) |
| `amber` | `VirtualSite.amber(s, center, first, third, p, middle=None)` | Amber's extra-point frame (Stone & Alderton): u, v unit vectors from the centre to first and third (or to the midpoints of the carbon's bonds for a carbonyl oxygen), e_z = -unit(u + v), e_x = unit(v - u), e_y = e_z cross e_x; centre + p . (e_x, e_y, e_z) |

Rules, checked when a system is set up: the weights of the averages sum to 1 (and those of the
local frame to 1, 0, 0), so that every construction is translation invariant; a site has mass 0;
its parents are real atoms of the same molecule (no sites built from sites); no frame is
degenerate. Every construction uses minimum-image displacements from the host, so a flexible
molecule may straddle the box. Sites take part in the colour refinement of the tying keys through
their parents (two sites of the same type on different hosts get different charge keys; radii,
polarizabilities and LJ tie by atom type, so give sites with different values different types).

## Forces, energy and virial

The energy is U(R, s(R)) with s the site positions as functions of the real atoms. The force on
the real atoms is F_R + (ds/dR)^T F_s: `VirtualSites.spread` evaluates it as the vector-Jacobian
product of `place` (automatic differentiation, exact for every kind), which conserves energy and,
because every construction is equivariant under rigid motions, the total force and torque.

Amber's `orient_frc` spreads an extra point's force differently: as a force and a torque on the
frame, distributed by rotations about the frame vectors. On TIP4P-Ew the two agree on every
rigid-body component (per-molecule force and torque to 1.8e-6 kcal/mol/A, the float32 precision of
sander's force file) and differ, by 9e-5 kcal/mol/A RMS per atom, only in the internal components
that the constraints of a rigid water cancel. For a flexible frame the transposed Jacobian is the
energy-conserving choice.

The pressure uses the molecular virial: under the molecular scaling of the virial and of the Monte
Carlo barostat a molecule is translated rigidly, and so are its sites. The atomic (affine) strain
derivative (`PGMForceField.strain_derivative(molecular=False)`) would need the sites rebuilt from
the deformed parents and is refused with sites.

## The MD engines

**Rigid molecules** (`Simulation`): a site is one more point of the rigid template, at zero mass.
Its position is placed from the parents of the input coordinates before the templates are built;
its force enters the body force and torque like any other atom's. Centre of mass, inertia,
wrapping, `momenta_from_velocities` and the degrees of freedom (6 per molecule) are unchanged by
massless points.

**Flexible molecules** (`FlexibleSimulation`, g-BAOAB with SHAKE / RATTLE): sites are not
integrated. They have no momentum and are excluded from the constraints, the thermostat (their
noise is masked), the kinetic energy, the degrees of freedom (3 per real atom minus the
constraints) and hydrogen mass repartitioning (massless atoms neither give nor take mass). The
integrator keeps unit placeholder masses on the sites, whose momenta stay zero, so the drifts of
the g-BAOAB step do not move them; they are rebuilt once per step, after the last drift (after
SHAKE) and before the forces, and the forces are spread to the parents before every momentum
update. Without sites the integrator is the one before sites existed, bit for bit.
`RigidTemplate` holds up to three real atoms plus sites (constrained TIP4P-Ew, TIP5P, OPC);
`FlexibleTemplate`s may carry sites if their bonded terms do not involve them.

**Pair topology** (`md/topology.py`): a site is part of its host. It shares the host's
neighbour-list group, takes the host's graph distances for the van der Waals weights (so it is
excluded from its host and from whatever its host is excluded from, and a site on an atom 1-4 to
another gets the 1-4 weight), and bonds to sites (Amber's topologies list one per extra point)
are not bonds of the graph. pGM electrostatics has no exclusions: sites interact with every atom,
within the molecule too (a constant for rigid molecules).

Trajectories and restarts contain the sites' coordinates, as Amber's do. Restart velocities of
sites are the rigid-body velocities (rigid engine) or zero (flexible engine).

## pGM on sites

- Charges, Gaussian radii and polarizabilities on sites work as on atoms; an induced dipole on a
  site responds to the field at the site and its force is spread like any other site force.
- **alpha = 0** marks a non-polarizable atom or site (also useful for point-charge atoms in
  general): its induced dipole is fixed at 0. The Jacobi-preconditioned CG never moves it (the
  preconditioner is alpha r), every predictor starts it at 0, and mu^2 / (2 alpha) and mu / alpha
  are evaluated as 0 there, so there is no 0/0 in values or gradients. The derivative with respect
  to an alpha that is exactly 0 is returned as 0 (the one-sided derivative -|E|^2 / 2 of a
  vanishing polarizability is not). The result equals the limit alpha -> 0 (tested). pmemd-pgm
  treats alpha <= 1e-6 A^3 as non-polarizable; the engine uses exactly 0. The mask is decided
  when the force field is built (`PGMForceField.alpha_mask`: some alpha of the system's parameter
  table is 0), so that systems without such atoms run the plain division, bit for bit as before;
  parameters passed later with new zeros need a force field built from a table that has them.
- **Covalent dipoles** (p_i += c unit(r_j - r_i)) may have a site as i or j. Their gradient
  reaches the site's position and is spread with the site forces. The two points must not
  coincide (checked at setup).
- Gas-phase and Ewald models (`Model`, `PeriodicModel`) take positions as given: place the sites
  with `VirtualSites.place(pos, None)` and spread forces with `VirtualSites.spread`. The same
  holds for `PGMForceField.compute` in force matching: its forces are per particle.

## Amber extra points

`read_prmtop_pgm` (used by `Simulation.from_amber`) and `protein.load_amber` turn atoms of type
`EP` (mass 0) into `amber` sites. The frames come from the bond graph by the rules of sander and
pmemd (`extra_pts.F90` `define_frames`, `frameon = 1`): an extra point is bonded to its centre at
the equilibrium length `req` of that bond;

| Centre's other neighbours | Extra points | Local position p |
|---|---|---|
| two hydrogens (BONDS_INC_HYDROGEN), one EP: TIP4P, OPC | on the H-O-H bisector toward the hydrogens | (0, 0, -req) |
| two hydrogens, two EPs (TIP5P); two heavy atoms; one heavy + one hydrogen | tetrahedral lone pairs in the local zy plane | (0, +-sin 54.735 req, cos 54.735 req); types S / SH: (0, +-req, 0); one EP: (0, 0, req) |
| one heavy atom, no hydrogen (carbonyl oxygen) | frame of the carbon's other two bond midpoints | (+-sin 60 req, 0, cos 60 req); one EP: (0, 0, req) |

Neighbour order follows the bond lists, as in Amber (it decides which lone pair is which).
Topologies that Amber rejects (more than two real neighbours, EP-EP bonds) raise; pmemd's custom
`VIRTUAL_SITE_FRAMES` section is not read (raises). sander replaces the extra points of the input
coordinates at the start, and so does the engine (tleap's library monomers and boxes place them
only approximately, up to 0.3 A off for TIP5P's monomer).

- `read_prmtop_pgm(prmtop, charges="amber")` / `Simulation.from_amber(..., charges="amber")` reads a
  classical prmtop: point charges `CHARGE / 18.2223` (Gaussian radius 1e-4 nm), no polarizability,
  no covalent dipoles; run it with `MDSettings(elec="q")`. The default `charges="pgm"` reads the
  POL_GAUSS sections of a pGM prmtop (and raises if there are none).
- `load_amber` recognises water with extra points (TIP4P-Ew, OPC, TIP5P: a water residue with three
  real atoms) and gives it a `RigidTemplate`; the placeholder library gives extra points their Amber
  charge as a point charge and no polarizability. Extra points outside water raise (the bonded
  models have no sites).
- `write_pgm_prmtop` refuses systems with sites: pmemd.pgm (CPU) spreads extra-point forces in its
  pGM branch (`pme_force.F90`, `orient_frc`), but pmemd.pgm.cuda's pGM force path
  (`cuda/pgm_gpu.cpp`) never calls `kOrientForces`, so on the GPU the extra points' forces would not
  reach their frames.

## Validation (`scripts/validate_vsites.py`)

TIP4P-Ew (Horn et al., J. Chem. Phys. 120, 9665 (2004)): 512 waters from tleap
(`leaprc.water.tip4pew`, 24.88 A lattice box), Amber's SHAKE geometry (0.9572 / 1.5136 A), EP at
0.125 A, point charges (elec "q"), 9 A cutoff with the LJ tail correction.

**Single point against sander** (AmberTools 25; float64; PME 64^3 order 8, ew_coeff 0.4 A^-1, exact
erfc (`eedmeth=3`), `netfrc=0`, 9 A, `vdwmeth=1`; an equilibrated frame; `validate_vsites.py sander`,
`compare`). The engine's energy includes each molecule's intramolecular Coulomb energy (-351.6
kcal/mol per water with point charges; pGM has no exclusions), which is subtracted with its forces;
its electrostatics are rescaled to sander's Coulomb constant (18.2223^2).

| Quantity | sander | pgm_jax | Difference |
|---|---|---|---|
| EELEC (kcal/mol) | -6748.3452 | -6748.34525 | 5e-5 (sander prints 1e-4) |
| VDWAALS (kcal/mol) | 1060.8288 | 1060.82884 | 4e-5 |
| Extra-point positions (sander's own placement) | | | 1.0e-6 A (float32 trajectory precision) |
| Force and torque on every molecule (RMS) | | | 1.8e-6 kcal/mol/A, 1.2e-6 kcal/mol (float32 force file; RMS atomic force 18.7) |
| Atomic forces (RMS, max) | | | 8.7e-5, 3.1e-4 kcal/mol/A: internal components only (orient_frc vs transposed Jacobian, above) |

**Energy conservation** (NVE, mixed precision, 50 ps from the equilibrated box, drift of E_tot per
degree of freedom; `validate_vsites.py nve`):

| Engine | dt 1 fs | dt 2 fs |
|---|---|---|
| Rigid bodies (`Simulation`) | -8.1e-5 kT/ns | 5.6e-4 kT/ns |
| Atoms + SHAKE / RATTLE + placed site (`FlexibleSimulation`, `RigidTemplate`) | -5.2e-5 kT/ns | -7.3e-4 kT/ns |

**Liquid at 298 K, 1 atm** (NPT, Bussi 1 ps, Monte Carlo barostat every 100 steps, mixed
precision, 0.2 ns equilibration first; errors: standard error of 10 block means; `validate_vsites.py
npt`, `analyse`). <U> is the intermolecular potential energy per molecule (with the LJ tail). Horn
et al. (Table V, 512 waters, Ewald, LJ switched between 9.0 and 9.5 A plus the tail, velocity Verlet
1 fs, T_internal 297.7 K for a 298 K bath): density 0.9954 +- 0.0003 g/cm^3, <U> = -5687.4 kcal/mol
for 512 waters = -11.108 kcal/mol (-46.477 kJ/mol) per molecule, uncertainty of the total below 1.9
kcal/mol (0.004 per molecule).

| Run | T_kin (K) | Density (g/cm^3) | <U> (kcal/mol) | <U> (kJ/mol) | ns/day |
|---|---|---|---|---|---|
| Horn et al. 2004 | 297.7 | 0.9954 +- 0.0003 | -11.108 +- 0.004 | -46.48 | |
| Rigid bodies, 1 fs, 3 ns | 297.2 | 0.9941 +- 0.0004 | -11.115 +- 0.003 | -46.51 +- 0.01 | 206 |
| Constraints + site, 1 fs, 3 ns | 297.2 | 0.9942 +- 0.0005 | -11.116 +- 0.003 | -46.51 +- 0.01 | 188 |
| Rigid bodies, 2 fs, 4 ns | 295.5 | 0.9950 +- 0.0004 | -11.119 +- 0.004 | -46.52 +- 0.02 | 406 |
| Constraints + site, 2 fs, 4 ns | 295.6 | 0.9949 +- 0.0005 | -11.119 +- 0.004 | -46.52 +- 0.02 | 362 |

The two engines agree with each other within their errors. At 2 fs the kinetic temperature of
BAOAB is 2.5 K below the bath (an O(dt^2) property of the momenta; configurations are sampled more
accurately: <U> agrees with the 1 fs runs). Against Horn et al. the density is 0.04-0.13 % lower and
<U> 0.007-0.011 kcal/mol (0.1 %) lower, 1-2.6 combined standard errors: the size expected from the
different treatment of the cutoff (a hard 9 A cutoff with Amber's tail correction here, a 9.0-9.5 A
switch of the LJ and a molecule-based taper of the real-space Ewald sum there) and of temperature
and pressure control.

**pGM with sites** (`validate_vsites.py pgm`; `tests/test_vsites.py`): 64 pGM waters in a periodic box,
each with a charged, polarizable Gaussian M site (a covalent dipole from the oxygen to it), two
charged, non-polarizable out-of-plane lone pairs and non-polarizable hydrogens; float64, dipoles
re-solved to 1e-12 at every displaced point. Spread forces against central differences of the
energy of the real atoms: largest relative error 1.7e-7 (36 components); molecular strain derivative
(all nine components): 6.0e-7; induced dipoles of the alpha = 0 atoms and sites exactly 0. The tests
cover every kind, the Amber frames, the topology, both engines (rigid and constrained TIP4P-Ew give
the same energies and body forces to 1e-9 and the same NVE trajectories), flexible methanol with a
site of every kind (Langevin, X-H constraints, HMR, NVE), replica exchange (batched and sequential),
and a peptide in TIP4P-Ew water from tleap through `load_amber`.

**Speed** (one RTX PRO 6000 Blackwell; mixed precision, NVT Bussi, dt 2 fs, 9 A, PME 0.08 nm order 6;
`validate_vsites.py bench`). The control is the same water without its site (the M charge on the
oxygen): the difference is the cost of one more interaction site per molecule plus, in the flexible
engine, the placement and spreading.

| Waters (atoms with / without the site) | Engine | TIP4P-Ew | Control (no site) |
|---|---|---|---|
| 512 (2,048 / 1,536) | rigid bodies | 0.358 ms/step (483 ns/day) | 0.330 ms/step (524 ns/day) |
| 512 | constraints + placed site | 0.418 (413) | 0.327 (529) |
| 4,096 (16,384 / 12,288) | rigid bodies | 1.040 (166) | 0.757 (228) |
| 4,096 | constraints + placed site | 1.138 (152) | 0.773 (224) |

(Means of two runs.) In the rigid engine the site costs what one more interaction site costs (8 %
at 512 waters, 37 % at 4,096, where the atom pairs, 16 per pair of waters instead of 9, dominate).
The flexible engine adds the placement and the spreading: 0.06-0.08 ms per step (15 % of the step
at 512 waters, 7 % at 4,096). On their own, in a compiled loop, they take 0.015 and 0.025-0.030 ms per
call at both sizes: a few small kernels (placement: five fusions; spreading: two, with one
scatter-add for all forces), bound by launch latency rather than by the number of sites.

Without sites nothing changes: the code path is the previous one, and the results are bitwise
those of the commit before this feature (`validate_vsites.py identical`, CPU: rigid NPT in mixed and
double precision, constrained NVT); `scripts/bench_md.py --replicate 2` (4,096 pGM waters, four runs
each, alternating with the previous code): rigid 1.975 ms/step (before: 1.987), constraints at 2 fs
2.229 ms/step (before: 2.223); the run-to-run spread is 3 %.

## Limitations

- Sites are built from real atoms only (no sites of sites) and must belong to the parents'
  molecule.
- Flexible templates with sites: the bonded terms must not involve the sites (the bonded fitting
  models have no site constructions); Amber topologies with extra points outside water are refused
  by `load_amber`.
- pmemd-pgm export of systems with sites is refused (see above); pmemd's custom
  `VIRTUAL_SITE_FRAMES` are not read.
- The atomic (affine) strain derivative is refused with sites; the engines use the molecular one.
