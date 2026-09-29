# Multiple time stepping (r-RESPA) for pGM MD

`pgm_jax/md/mts.py`; `FlexibleSimulation(..., dt=outer, mts=MTS(...))` and
`Simulation(..., mts=MTS(...))`; `--mts N` (and `--mts-*`) in `scripts/bench_md.py`,
`scripts/protein/bench_protein.py` and `scripts/run_md.py`. Tests: `tests/test_mts.py`.

**In short.** The framework integrates force groups at their own time steps (slow / optional
short-range / bonded) for both engines, exactly: one fast step per outer step is the ordinary
integrator to 1e-14, the groups sum to the full force, the NVE step is time-reversible. For pGM,
what limits the outer step is not the cost split but the physics of the fast motions:

- **Solvated protein (ubiquitin, HMR, X-H constraints, Bussi):** the bonded terms and the pGM
  interactions of the topologically close pairs every 2.33 fs, everything else (induced-dipole
  solve, PME, all other pairs) every 7 fs (`MTS(inner=3, split="special")`, dt 7 fs) keeps the
  accuracy of the best single step (4 fs): <U> within 7 +- 28 kJ/mol of it over 200-300 ps,
  econs drift 0.006 kT/ns/dof, the internal temperature 3 K closer to the target. It runs at
  **1.42x** the speed of the 4 fs step (108.6 against 76.4 ns/day with the quadratic dipole
  predictor; 1.79x over the 4 fs step with the 0.9 nm electrostatics cutoff). At 6 fs (bonded
  or special split, 1.23-1.28x) the configurational energy is that of 2 fs. Backbone RMSD and
  radius of gyration over 300 ps (6 / 2 fs) match the single step.
- Why 7-8 fs is the limit: pGM has no electrostatic exclusions, so the 1-2 and 1-3
  electrostatics of flexible molecules vibrate with the bonds and must be fast (the plain bonded
  split drifts from 7 fs on). The short-range intermolecular forces (hydrogen bonds, with the
  polarization that couples them to the environment) then make every split drift at 8 fs
  (0.01-0.05 kT/ns/dof) and fail at 9-12 fs. Heavier water hydrogens do not help; a short-range
  pGM model (mutual induction, Gaussian-screened pairs) in the fast group cuts the drift at 8 fs
  4x but costs what it gains.
- **Pure pGM water:** the single-step integrator with 4 amu hydrogens at 5 fs stays the fastest
  at equal accuracy; MTS at 8 / 4 fs is 1.22x faster but shifts <U> by +0.26 kJ/mol per molecule
  and the density by -0.2 %. With rigid bodies and physical masses, 4 / 2 fs is 1.32x faster
  than 2 fs, with a 10x larger drift.

## Method

**Factorisation.** The propagator is split as in r-RESPA (Tuckerman, Berne & Martyna, JCP 97,
1990 (1992)), with BAOAB-type placement of the thermostat:

    B_slow(dt/2) [B_fast(h/2) A(h) B_fast(h/2)]^n B_slow(dt/2),     h = dt / n,

and recursively for a third level (bonded terms every h / m inside each fast step). B_g(t) is the
kick p += t F_g (RATTLE after each kick with constraints; for rigid bodies the atomic forces are
mapped to the centre-of-mass force and quaternion torque), A the drift (SHAKE + RATTLE, or the
NO_SQUISH free rotor). All groups are functions of the positions only, so the map is
symplectic and time-reversible. The simulation's `dt` is the outer step: steps, times, ns/day,
barostat intervals and outputs all count outer steps.

**Thermostat placement** (`o_step`). "outer" (default): one O step of length dt in the middle of
the outer step, as in BAOAB-RESPA (Lagardere et al. 2019). For n = 1 this is exactly BAOAB; for
n even it sits between two fast steps, for n odd in the middle of the middle one's drift
(recursively with three levels). "inner": an O step of length h in every innermost drift. Both
book the heat of every O step, so `econs` = E_tot + |aux|^2/2 - heat remains the conserved
effective energy at outer steps.

**Force groups** (`split`):

- `"bonded"` (flexible engine): fast = bonded terms + restraints, slow = everything nonbonded
  (the ordinary force-field evaluation with the induced-dipole solve and PME).
- `"special"` (flexible engine): fast = bonded terms + restraints + the pGM and van der Waals
  interactions of the special pairs (md/topology.py: the rest of a small molecule, the atoms of
  the nearby heavy-atom groups of a large one), unswitched, with the Gaussian-screened Coulomb
  erf(a r)/r and the fast induction model over these pairs; slow = F_full - F_fast. pGM has no
  electrostatic exclusions, so the 1-2 and 1-3 electrostatics of a flexible molecule vary with
  its bond and angle vibrations as the bonded terms do; this split moves them to the fast group
  at the cost of a sweep over the rows of the flexible molecules (at most 29 partners each; the
  internal pairs of rigid molecules only exert forces that their constraints remove).
- `"short"`: fast = a cheap short-range nonbonded model (below) + bonded + restraints (or, with
  `bonded` > 1, a middle level with the short-range model and an innermost bonded level, as
  RESPA1); slow = F_full - F_fast evaluated at the same positions. Every group sums exactly to
  the full force, so n = 1 is the ordinary integrator.

**The fast nonbonded model.** A conservative potential of the positions alone, over the pairs
closer than r_short (0.5 nm), switched off from r_short - 0.1 nm by
S(r) = 1 - t^3 (10 - 15 t + 6 t^2) (C2):

    U_fast = 1/2 sum_i sum_k S(r_ik) [KE e_ik + w_ik e_vdW(r_ik)] + KE U_ind

- e_ik is the pGM pair energy of the Gaussian charges and dipoles (permanent + induced) with the
  kernel erf(a_ik r)/r - erf(beta_s r)/r: the Gaussian-screened Coulomb minus its smooth
  long-range part, beta_s = 2.327 / r_short so that erfc(beta_s r_short) = 1e-3. The screening
  is essential. The bare Coulomb switched atom by atom cuts through neutral molecules: in
  ubiquitin's water the fast net force per water molecule was then 4,063 kJ/mol/nm rms against
  115 for the real force, the slow group cancelled it, and the integrator blew up within 1 ps.
  With the screened kernel, a pair at the switch carries 1e-3 of its Coulomb interaction.
- U_ind, `polarization`: `"mutual"` (default) -1/2 E.mu1 with nu0 = alpha E, mu1 = nu0 +
  alpha T nu0, E the switched permanent field and T the switched dipole tensor (one mutual
  iteration, the second order of the perturbation series behind ExPT / OPT: Simmonett et al.,
  JCP 143, 074115 (2015); 145, 164101 (2016)); `"direct"` -1/2 E.alpha E; `"none"`. The
  forces are exact gradients, -mu1.dE/dx - 1/2 nu0 (dT/dx) nu0 plus the covalent-dipole frames,
  from three sweeps over the pair list: no solve, and the fast energy is conserved by the inner
  integrator (checked to 1e-12 against autodiff).
- The pair list holds the pairs closer than r_short + buffer (0.1 nm). It is compacted at every
  outer step from the electrostatic rows of the full evaluation (`compute(keep_geometry=True)`),
  checked at every fast evaluation (sum of the two largest atomic displacements since the build <
  buffer), and rebuilt from the neighbour list inside a `lax.cond` if needed (1.6 % of the fast
  steps for ubiquitin at 8 fs), so no pair inside r_short is ever missed. Displacements are
  taken in float64 (exact for bonded pairs in mixed precision).

**Predictor.** One solve per outer step, so the dipole history is spaced by dt. With fast
induced dipoles the history is anchored (`anchor`, default on): it stores mu - mu_fast, and the
guess is mu_fast(x_new) + the extrapolation of mu - mu_fast. mu_fast at the new positions is
known before the solve (the last fast evaluation), so the fused initial residual still applies.
CG iterations per solve, ubiquitin (tol 1e-5):

| outer step, split | mu4 | mu3 (quadratic) |
|---|---|---|
| 4 fs single step | 16.9 | 16.7 |
| 6 fs, bonded (no fast dipoles) | 18.3 | 17.7 |
| 8 fs, short 8 / 4 / 2, not anchored | 19.3 | 18.6 |
| 8 fs, short 8 / 4 / 2, anchored | 18.1 | 17.4 |

The anchor saves 1.2 iterations at 8 fs, and with the history spaced by 6-8 fs the quadratic
extrapolation beats the cubic one by 0.6-0.7 iterations (the higher order amplifies what is not
smooth at that spacing): with MTS, `MDSettings().replace(predictor="mu3")` is 2-4 % faster. The solve at 8
fs then costs what it costs at 4 fs without MTS.

**Other pieces.** The Monte Carlo barostat runs at outer steps (its trial energy is the full
energy); an accepted move re-evaluates every group at the scaled positions. Restraints are in
the fastest group. Checkpoints hold the MTS state; a checkpoint of an ordinary run continues
under MTS (the groups are evaluated at its first step). Replica exchange refuses MTS for now
(the swap would have to move the MTS state too, and its driver does not grow the pair list).

## Prior art

- **r-RESPA** (Tuckerman, Berne & Martyna 1992) and the reversible impulse schemes
  (Grubmueller et al. 1991): the factorisation used here. Splitting the Ewald real-space sum
  across levels with the reciprocal part outermost: Procacci, Darden & Marchi, J. Phys. Chem.
  100, 10464 (1996); Zhou, Harder, Xu & Berne, JCP 115, 2348 (2001).
- **Resonances.** Impulse MTS is linearly unstable when the outer step approaches half the
  period of a fast mode (Biesiadecki & Skeel, J. Comput. Phys. 109, 318 (1993)) and has
  nonlinear instabilities from about a third of it (Ma, Izaguirre & Skeel, SIAM J. Sci. Comput.
  24, 1951 (2003)). This is the classic 5 fs barrier of biomolecular MTS.
- **Thermostats against resonances.** Colored-noise (GLE) thermostats damp the resonant modes
  and extend the stable outer step (Morrone, Markland, Ceriotti & Berne, JCP 134, 014103 (2011)).
  Stochastic isokinetic Nose-Hoover (SIN(R)) removes resonances entirely by holding the kinetic
  energy of every degree of freedom, and allows outer steps of 100 fs for configurational
  sampling at the price of the dynamics (Leimkuhler, Margul & Tuckerman, Mol. Phys. 111, 3579
  (2013)); for polarizable models: Margul & Tuckerman, JCTC 12, 2170 (2016).
- **AMOEBA in Tinker-HP** (Lagardere, Aviat, Piquemal et al., JPCL 10, 2593 (2019)):
  BAOAB-RESPA (O step at the outer level; bonded inner, nonbonded outer) and BAOAB-RESPA1 (bonded
  inner, short-range nonbonded and short-range polarization at an intermediate step, the rest at
  a 10 fs outer step with HMR), up to 7x over 1 fs Verlet. The polarization at the fast level
  there is a short-range solve; the non-iterative, analytically differentiable truncated CG
  (Aviat et al., JCTC 13, 180 (2017)) is the related tool for conservative approximate
  polarization.
- **OpenMM** `MTSIntegrator` / `MTSLangevinIntegrator`: force groups at their own intervals,
  BAOAB-type O step at the outer level.

What is new here: the special-pair split for a force field without electrostatic exclusions,
the fast model for pGM (Gaussian-screened short-range kernel, one-iteration mutual induction with
exact forces), the anchored dipole predictor, and the measurements below.

## Measurements

One RTX PRO 6000 Blackwell, mixed precision, dipole tol 1e-5, Bussi tau 1 ps, NVT at 298 K
unless noted; ns/day counts simulated time with the outer step.

### What goes into the slow force

Ubiquitin in water (15,955 atoms, 4,908 waters), one configuration and 1 fs of dynamics. For the
waters the forces are projected on the rigid bodies (net force, torque): only those move a
rigid water. rms change per fs of the full, fast and slow forces:

| fast model (r_short / beta_s) | water torque: full / fast / slow (kJ/mol) | water net force: slow (full 4.81) | protein heavy atoms: slow (full 278) |
|---|---|---|---|
| bare Coulomb, direct (0.5 / -) | 0.434 / 7.27 / 7.24 | 142 | 50 |
| screened, none (0.5 / 5.19) | 0.434 / 0.316 / 0.139 | 0.80 | 5.2 |
| screened, direct (0.5 / 4.66) | 0.434 / 0.323 / 0.131 | 0.79 | 2.5 |
| screened, mutual (0.5 / 4.66) | 0.434 / 0.336 / 0.118 | 0.76 | 2.0 |
| screened, direct (0.6 / 4.0) | 0.434 / 0.357 / 0.099 | 0.56 | 2.7 |
| screened, mutual (0.6 / 4.0) | 0.434 / 0.387 / 0.071 | 0.49 | 2.4 |
| screened, none (0.8 / 3.0) | 0.434 / 0.584 / 0.170 | 1.00 | 6.6 |
| screened, mutual (0.8 / 3.0) | 0.434 / 0.438 / 0.044 | 0.35 | 3.5 |

- The screened kernel makes the slow force on the protein 100x slower than the full force.
- On water it leaves 10-30 % of the librational torque variation in the slow force. Most of it
  is induction beyond the fast model: without fast induction the fast force overshoots the
  real one, and the mutual iteration halves the rest.
- Taking more of it into the fast group needs a longer pair list (0.8 nm: the whole
  electrostatic row) and so a fast step that costs a third of a full one.

### Ubiquitin: stability and accuracy

100 ps (unless noted) sampled every 0.5 ps after 10 ps, each run from the same structure (100 ps
at 2 fs, 300 ps at 4 fs). The ns/day of these runs include the sampling and vary by +-5 % with
the load of the node; the clean speeds are in the next section. Most special-split runs of 100 ps
used a first version whose fast group also held the internal pairs of the rigid waters (their
forces are removed by the constraints: the same dynamics, a slower fast step); the 200 ps runs
and "special 6 / 2, no fast induction" use only the rows of the flexible molecules.

HMR 3.024 amu on every hydrogen, X-H constraints, rigid water, electrostatics cut at 0.7 nm (LJ
0.9 nm). dU = <U> - <U>(2 fs), with the errors of 10 block averages; drift of econs in kT/ns per
degree of freedom. "bonded a / b": nonbonded every a fs, bonded every b fs; "special a / b":
special pairs and bonded every b fs, the rest every a fs; "short a / b / c": slow every a,
short-range pGM model every b, bonded every c fs. Fast induction "mutual" unless noted.

| setting | ns/day | CG / solve | dU (kJ/mol) | econs drift | T centre of mass / internal (K) |
|---|---|---|---|---|---|
| 2 fs (reference) | 45.7 | 12.9 | 0 +- 26 | -0.0001 | 297.4 / 296.6 |
| 4 fs (best single step) | 76.1 | 16.7 | +57 +- 32 | 0.0000 | 296.5 / 291.9 |
| same, 300 ps | 74.8 | 16.7 | +38 +- 34 | +0.0001 | 296.5 / 291.8 |
| bonded 5 / 2.5 | 84.2 | 17.6 | +138 +- 40 | 0.0000 | 297.5 / 296.2 |
| bonded 6 / 2 | 91.9 | 18.3 | +56 +- 40 | +0.0025 | 296.4 / 295.1 |
| bonded 6 / 2, O step inner | 92.2 | 18.3 | +30 +- 40 | -0.0003 | 296.5 / 295.0 |
| bonded 6 / 1.5 | 93.9 | 18.3 | +68 +- 37 | +0.0028 | 297.1 / 295.9 |
| bonded 6 / 3 | 101.7 | 18.5 | +260 +- 38 | +0.0027 | 297.8 / 295.4 |
| bonded 7 / 1.75 | 100.5 | 18.8 | +102 +- 49 | +0.051 | 297.1 / 295.0 |
| bonded 7 / 3.5 | 114.6 | 18.9 | +388 +- 35 | +0.050 | 297.2 / 295.0 |
| bonded 8 / 2 | 109.7 | 19.3 | +218 +- 38 | +0.23 | 295.7 / 295.1 |
| short 8 / 2.67 / 1.33 | 90.1 | 18.0 | +83 +- 40 | +0.0046 | 297.8 / 296.5 |
| short 8 / 4 / 2 | 100.7 | 18.1 | +188 +- 39 | +0.010 | 297.9 / 296.9 |
| short 8 / 4 / 2, direct induction | 101.9 | 18.6 | +199 +- 46 | +0.0075 | 297.8 / 296.7 |
| short 9 / 3 / 1.5 | 99.0 | 18.8 | +964 +- 55 | +2.5 | 286.1 / 306.3 |
| special 6 / 2 | 85.5 | 17.5 | -2 +- 42 | -0.0004 | 296.6 / 295.4 |
| special 6 / 2, 200 ps, another seed | 90.4 | 17.3 | +11 +- 33 | +0.0002 | 296.5 / 295.0 |
| special 6 / 2, no fast induction | 96.7 | 18.3 | -2 +- 45 | +0.0013 | 296.5 / 295.6 |
| **special 7 / 2.33**, 200 ps | 105.0 | 17.9 | +45 +- 32 | +0.0055 | 296.4 / 294.6 |
| special 7 / 2.33, no fast induction | 109.0 | 18.8 | +62 +- 30 | +0.0085 | 296.5 / 294.0 |
| same, 200 ps, another seed | 105.4 | 18.8 | +46 +- 33 | +0.0081 | 295.5 / 294.6 |
| special 7 / 1.75 | 95.7 | 17.8 | +103 +- 41 | +0.0076 | 297.4 / 295.2 |
| special 7 / 1.75, no fast induction | 99.6 | 18.8 | +65 +- 47 | +0.0046 | 296.7 / 296.2 |
| special 7 / 2.33, no fast induction, water H 4 amu | 105.9 | 18.8 | +74 +- 40 | +0.011 | 295.7 / 295.0 |
| special 8 / 2.67, no fast induction | 108.6 | 19.3 | +148 +- 39 | +0.048 | 295.6 / 293.7 |
| special 8 / 2 | 102.1 | 18.3 | +166 +- 49 | +0.042 | 296.6 / 294.9 |
| special 8 / 2, no fast induction, water H 4 amu | 102.1 | 19.2 | +138 +- 41 | +0.056 | 297.2 / 295.7 |
| special 8 / 2, no fast induction | 110.6 | 19.2 | +128 +- 43 | +0.049 | 296.9 / 295.3 |
| special 8 / 4 / 2 | 105.7 | 18.4 | +193 +- 41 | +0.047 | 296.0 / 295.5 |
| special 9 / 2.25 | 118.3 | 18.7 | +352 +- 54 | +0.65 | 294.5 / 297.0 |

Screens of 20 ps (same start; "short" with the fast model at 0.5 / 4.66 or 0.6 / 4.0):

- **Short-range forces at 4 fs with the bonded terms in the same group are unstable:** 8 / 4
  blew up after 7-17 ps with direct induction and with mutual induction at 0.6 nm, survived 20
  ps with a +0.04 drift at 0.5 nm, and 12 / 4 drifted by +15 kT/ns/dof (centre-of-mass 267 K,
  internal 323 K). The same short-range step with the bonded terms at 2 fs (8 / 4 / 2) is
  stable. The outer kicks resonate with heavy-atom stretches and angles, which HMR speeds up (a
  CH3 carbon keeps 6 amu); an 8 fs outer step is half the period of a 16 fs mode.
- **Thermostats do not rescue larger steps.** 8 / 4 with Langevin 1/ps at the outer O step
  stays together but drifts by +0.51 kT/ns/dof with dU +490 kJ/mol; with the O step inner, +0.80;
  the slow-band GLE blows up; 12 / 4 with Langevin blows up.
- Outer steps of 10-12 fs drift by 0.4-15 kT/ns/dof whatever the inner steps.

So for this protein:

- The error of the 4 fs single step comes from the bonded modes: bonded terms at 3 fs cost
  +200 kJ/mol more than at 2 fs, whatever the outer step. Once they are integrated finely, the
  nonbonded forces tolerate 6 fs at the accuracy of the single step.
- At 7-8 fs the bonded split drifts (+0.05 and +0.23 kT/ns/dof). Most of that is the 1-2 / 1-3
  pGM electrostatics, which the bonded split leaves in the slow group: the special split cuts
  the drift at 8 fs from 0.23 to 0.04-0.05 and at 7 fs from 0.05 to 0.008.
- What remains at 8 fs comes from the intermolecular forces left in the slow group: 0.04-0.05
  kT/ns/dof with the special split (the same with 4 amu water hydrogens, so it is not the
  librational frequency as such), 0.01 when a short-range pGM model is fast too (short 8 / 4 /
  2), with dU +130-190 kJ/mol in both. 9 fs is beyond the limit in every split.

### Pure pGM water (4,096 pGM3P-25 molecules, 12,288 atoms)

`scripts/bench_md.py --replicate 2 --elec-cut 0.7 --thermostat bussi`, 20 ps sampled after 2
ps; <U> relative to the best single step; fast model 0.5 nm, mutual.

| engine, setting | ns/day | CG / solve | econs drift | dU (kJ/mol) | T (K), groups |
|---|---|---|---|---|---|
| constraints, H 4 amu, 5 fs | 168.6 | 7.0 | +0.0050 | 0 +- 112 | 294.6 / 293.1 |
| constraints, H 4 amu, 4 fs | 132.3 | 6.7 | +0.0003 | +128 +- 145 | 296.0 / 294.8 |
| constraints, H 4 amu, 8 / 4 | 200.7 | 8.0 | +0.027 | +897 +- 185 | 299.4 / 298.2 |
| constraints, H 4 amu, 10 / 3.33 | 211.7 | 8.0 | +0.18 | +19 +- 162 | 297.8 / 298.8 |
| constraints, H 4 amu, 10 / 5 | 237.0 | 8.0 | +0.12 | +1220 +- 137 | 300.8 / 299.1 |
| constraints, H 4 amu, 12 / 4 | 251.0 | 8.2 | +0.45 | +147 +- 144 | 298.2 / 297.6 |
| constraints, H 4 amu, 15 / 5 | 292.9 | 9.0 | +1.75 | +313 +- 146 | 298.1 / 298.1 |
| rigid bodies, 2 fs | 76.2 | 6.0 | +0.0028 | 0 +- 54 | 299.0 / 295.7 |
| rigid bodies, 4 / 2 | 104.3 | 7.0 | +0.040 | +378 +- 112 | 299.6 / 298.3 |
| rigid bodies, 6 / 3 | 130.5 | 8.0 | +0.32 | -284 +- 176 | 297.6 / 298.2 |
| rigid bodies, 6 / 2 | 146.4 | 8.0 | +0.31 | +622 +- 127 | 303.3 / 298.8 |

(T groups: centre of mass / internal for constraints, translational / rotational for rigid
bodies. dU of 4,096 waters: 300 kJ/mol is 1 K of configurational temperature.)

- Water has no bonded modes, but its librations still limit the outer step: the slow force
  keeps 27 % of the torque variation with the 0.5 nm model, and outer steps of 8 fs and more
  either drift or shift <U>.
- The 4 amu single step at 5 fs is already a large step (the librations are slower): MTS does
  not beat it at equal accuracy.

### Structure and ensemble checks

- **Ubiquitin, 300 ps** (after 10 ps, frames every 2 ps; `bench_protein.py --traj-ps 2`,
  which prints the CA RMSD to the first frame after superposition, radius of
  gyration of the heavy atoms):

  | setting | <U> (kJ/mol) | CA RMSD mean / last half / max (A) | Rg (A) | T COM / internal (K) | econs drift |
  |---|---|---|---|---|---|
  | 4 fs single step | -2630797.8 +- 21.6 | 1.36 / 1.43 / 2.02 | 11.79 +- 0.09 | 296.5 / 291.8 | +0.0001 |
  | bonded 6 / 2 | -2630791.8 +- 18.2 | 1.72 / 1.66 / 2.47 | 11.77 +- 0.08 | 296.7 / 294.9 | +0.0006 |

  The same <U> within 6 kJ/mol, the same Rg, a stable backbone in both (the RMSD difference is
  that of two diverging trajectories), and internal temperatures closer to the target with MTS
  (the fast modes are integrated at 2 fs).
- **O-O radial distribution function** of pGM water (40 ps after 1 ps): the first peak at
  0.281 nm in every run, height 2.961 (5 fs single step, 4 amu H), 2.938 (MTS 8 / 4), 2.957
  (rigid bodies, 2 fs), 2.945 (rigid bodies, MTS 4 / 2).

- **NPT** (Monte Carlo barostat every 25 outer steps at 1 bar, trial energies with the full
  force field): pGM water with 4 amu hydrogens, 60 ps after 25 ps (5 fs) or 40 ps (8 / 4):

  | setting | density (g/cm^3) | <U> - <U>(5 fs) (kJ/mol) | T centre of mass / internal (K) | g_OO first peak (nm), height |
  |---|---|---|---|---|
  | 5 fs single step | 1.0184 +- 0.0010 | 0 +- 85 | 295.5 / 292.8 | 0.281, 2.953 |
  | MTS 8 / 4 | 1.0160 +- 0.0010 | +1080 +- 110 | 299.5 / 298.3 | 0.279, 2.932 |

  The barostat works at outer steps; the 8 / 4 split shifts the density by -0.2 % and <U> by
  +0.26 kJ/mol per molecule, as in NVT. (The ubiquitin test system cannot be checked in NPT:
  its placeholder water, TIP3P charges as Gaussians, has a virial pressure of +1,200 bar at the
  tleap density of 0.78 g/cm^3 and expands without bound, with or without MTS.)

### Speed

Clean runs, one after the other on an idle node (`bench_protein.py --steps 3000` from the
equilibrated structure; `bench_md.py --replicate 2 --steps 3000`). The single step of this
branch runs at the speed of master (75.9 and 75.1 ns/day).

| system | electrostatics cutoff | setting | ms / outer step | ns/day | speed-up |
|---|---|---|---|---|---|
| ubiquitin (15,955 atoms) | 0.7 nm | 4 fs single step | 4.52 | 76.4 | 1 |
| | 0.7 nm | 4 fs single step, predictor mu3 | 4.53 | 76.2 | 1.00 |
| | 0.7 nm | bonded 6 / 2 | 5.30 | 97.8 | 1.28 |
| | 0.7 nm | bonded 6 / 3 | 5.12 | 101.3 | 1.33 |
| | 0.7 nm | special 6 / 2 | 5.54 | 93.6 | 1.23 |
| | 0.7 nm | special 6 / 2, mu3 | 5.43 | 95.6 | 1.25 |
| | 0.7 nm | special 6 / 2, no fast induction | 5.34 | 97.1 | 1.27 |
| | 0.7 nm | special 7 / 2.33 | 5.69 | 106.3 | 1.39 |
| | 0.7 nm | **special 7 / 2.33, mu3 (recommended)** | 5.57 | 108.6 | **1.42** |
| | 0.7 nm | special 7 / 2.33, no fast induction | 5.75 | 105.1 | 1.38 |
| | 0.7 nm | special 7 / 2.33, no fast induction, mu3 | 5.37 | 112.7 | 1.48 |
| | 0.7 nm | special 7 / 1.75, no fast induction | 6.04 | 100.2 | 1.31 |
| | 0.7 nm | short 8 / 4 / 2 | 6.50 | 106.4 | 1.39 |
| | 0.7 nm | short 8 / 4 / 2, mu3 | 6.65 | 103.9 | 1.36 |
| | 0.9 nm | 4 fs single step | 5.69 | 60.8 | 0.80 |
| | 0.9 nm | bonded 6 / 2 | 6.90 | 75.2 | 0.98 |
| | 0.9 nm | special 7 / 2.33, mu3 | 6.53 | 92.7 | 1.21 |
| | 0.9 nm | short 8 / 4 / 2 | 8.22 | 84.1 | 1.10 |
| pGM water, constraints, H 4 amu (12,288 atoms) | 0.7 nm | 5 fs single step | 2.73 | 158.5 | 1 |
| | 0.7 nm | 8 / 4 | 3.58 | 193.1 | 1.22 |
| pGM water, rigid bodies | 0.7 nm | 2 fs single step | 2.29 | 75.4 | 1 |
| | 0.7 nm | 4 / 2 | 3.48 | 99.2 | 1.32 |

Repeated runs of one setting differ by up to 3-4 %. The fast levels are cheap: the bonded terms
with SHAKE / RATTLE cost 0.3 ms per fast step, the special-pair model on the protein's 1,231
rows 0.01-0.08 ms more; what the outer step buys is fewer induced-dipole solves (4.4 ms each).

- At the accuracy of the 4 fs single step: special 7 / 2.33, **1.42x** (1.39-1.48x with the
  predictor and fast-induction variants).
- At the accuracy of 2 fs: special or bonded 6 / 2, 1.23-1.28x.
- The separate electrostatics cutoff and MTS multiply: from the 0.9 nm single step (60.8
  ns/day) to 0.7 nm with special 7 / 2.33 (108.6) is 1.79x. pmemd.pgm.cuda runs this system at
  97 ns/day at 2 fs (docs/protein_ff.md).

### Where the time goes

Ubiquitin, ms per outer step (each part timed on its own, so with its launch overhead):

| | ms |
|---|---|
| single step 4 fs | 4.31 |
| MTS bonded 6 / 3: step | 4.91 |
| - full evaluation (18.5 CG iterations, rows, PME) | 4.40 |
| - bonded forces + kick + drift, per fast step | 0.23 + 0.07 + 0.07 |
| MTS short 8 / 4 (mutual): step | 6.07 |
| - full evaluation + pair list | 4.39 |
| - fast evaluation (short-range model 0.31, bonded 0.12, list check 0.06) | 0.60 |
| MTS short 8 / 4 / 2: step | 6.78 |
| - short-range level / bonded level evaluation | 0.44 / 0.23 |
| list rebuild from the neighbour list (1.6 % of fast steps) | 0.6 |

The fast short-range evaluation (136 pairs per atom) costs 14 % of a full evaluation, the
bonded level (bonded forces, SHAKE and RATTLE) 5 %.

## Recommended settings

- **Proteins in water with HMR and X-H constraints:** `MTS(inner=3, split="special")` with dt =
  7 fs and `MDSettings().replace(predictor="mu3", **elec_cutoff_settings(0.7))` (`--dt 0.007 --mts 3
  --mts-split special --predictor mu3 --elec-cut 0.7`): the accuracy of the 4 fs single step at
  1.42x its speed. For the accuracy of 2 fs, the same at dt = 6 fs (1.25x). Bussi, with the O
  step at the outer level (the defaults).
- **Water-dominated systems:** the single step with 4 amu water hydrogens at 4-5 fs
  (docs/thermostat_ideas.md): MTS is faster (8 / 4 fs, 1.22x) but not at equal accuracy. With
  rigid bodies and physical masses, 4 / 2 fs MTS is 1.32x faster than 2 fs, less accurate.
- Keep Bussi: Langevin or GLE at the outer O step hide the resonant heating in the heat
  bookkeeping instead of removing it (drift 0.5-0.8 kT/ns/dof at 8 / 4); the O step at the inner
  level makes no difference with Bussi (6 / 2: dU +30 against +56).
- MTS and replica exchange are not combined yet (`ReplicaExchange` refuses MTS).

## Next steps

- **The 8 fs barrier** comes from the short-range intermolecular forces left in the slow group
  (hydrogen bonds and the polarization that couples them to the environment). A fast model that
  captures them at the cost of a sweep over ~30 partners per atom (a short solve on a 0.35 nm
  list, or TCG as in Tinker-HP) would move it; the short-range model here captures them at the
  cost of 136 partners in three sweeps, which is what it gains.
- **SIN(R)** (Leimkuhler, Margul & Tuckerman 2013; Margul & Tuckerman 2016) is the known way
  past the resonance barrier for configurational sampling: outer steps of tens of fs, canonical
  configurations, dynamics given up. It needs an isokinetic constraint per degree of freedom
  compatible with holonomic constraints and rigid bodies.
- Replica exchange with MTS: swap the MTS state with the configuration and grow the short-range
  list in the replica driver.
- Merging kicks at equal positions (one RATTLE per drift) would save a few per cent in the
  three-level scheme.
