# NOTES: extended-Lagrangian induced dipoles (item 14, branch iel)

Running log so the work can be resumed.

## Design (done)
- MDSettings(iel="none"|"0scf"|"scf", iel_iter, iel_order=K, iel_kappa, iel_alpha); engine in
  pgm_jax/md/forcefield.py (_solve_iel, _dipole_energy_forces), CLI helpers + stability in
  pgm_jax/md/iel.py; --iel options in run_md.py, bench_md.py, water_dielectric.py.
- Auxiliary dipoles x: InductionState.xl (K1, N, 3) = [x_{n+1}, x_n, ...]; Niklasson dissipative
  Verlet, table of JCP 130, 214109 (2009).
- iEL/0-SCF shadow energy U~(R,x) = U(R,x) - sum alpha r^2/2 = U(R, x+delta) - U_es(0,delta),
  delta = alpha r(x), stationary in delta -> forces = fixed-dipole forces at mu = x + delta
  + grad_R U_es(0, delta) (dipole-only rows + PME + self).
- Warm-up: first max(K,2)+1 steps converged (dipole_tol), head replaced by mu*.  Accepted MC volume
  move: ff.energy resets count -> warm-up again.  Barostat with shadow: converged U* at both volumes.
- Refused: differentiable, MTS, alchemy.
- Spectrum of alpha A, 512 pGM waters (runs/eig.py): 0.679 .. 1.853 (median 0.92).  K=5 recurrence:
  spectral radius <= 0.968 on that range (stable up to lam ~ 1.99).

## Log
- 2026-09-28 02:30: implementation + tests/test_iel.py (10 tests, pass on CPU, ~90 s).  Shadow forces
  vs central FD: error -> 0 as h^2 (h 3e-6: 1e-4 of 2500), fixed-dipole forces at mu off by 0.046.
- scripts/iel_validate.py: from an equilibrated .chk (copied: runs/iel/scf_prod15ns.chk = the 15 ns SCF
  NPT run of ~/project/pGM-JAX-dipoles/runs/eps512, eps 31.0 +- 0.4), box scaled to density 1.0178,
  seeds x (Bussi NVT equil + NVE prod): drift, dipole error vs float64 tol 1e-9, D, tau1/2, RDF.
  CPU smoke test (2 x 1 ps, 2 fs, 0scf): mu rel RMS error 1.9e-3, U~ - U* = -0.35 kJ/mol (of -2.1e6).
- GPU gpu-2-1 blocked by the owner's pmemd stream since ~02:30; NVE drift runs moved to CPU
  (runs/iel/nvecpu.sh -> runs/iel/nve_dt*.log, bench_md.py --ensemble nve --ps).
- 03:40 fused the shadow correction into _row_terms / _nonpair (delta argument); iel_shadow option
  (False: fixed-dipole forces at mu).  tests: test_iel + test_md + test_flux 26 pass.
- CPU jobs: sbatch default mem = whole node (125G) -> one job per node; use --mem=6000 (scontrol
  update MinMemoryNode=6000 on pending jobs).  Only touch jobs named iel-* / pgmjax-pGM-JAX-iel.
- 04:08-04:22 GPU (runs/iel/bench1.log, bench2.log).  Cost per force call (512 w): scf-k 0.331,
  0.384, 0.453, 0.563 (k=1..4), 0.665 (6); 0scf 0.292 ms.  4096 w: 1.009 .. 1.724; 0scf 0.842.
  NVT speed 512 w Langevin 1 fs: SCF 1e-5 0.770 ms (112 ns/d), 1e-4 0.672, 0scf 0.469 (184), scf1 0.497;
  Bussi 2 fs: 1e-5 0.789 (219), 1e-4 257, 0scf 0.473 (366), scf1 340.
  4096 w: Langevin 1 fs 1e-5 2.029 (42.6), 1e-4 50.9, 0scf 1.192 (72.5), scf1 68.0; Bussi 2 fs 83.2 / 96.4 / 132.0 / 126.5.
  NVE (README restart, 50/100 ps): 1 fs drift kT/ns/dof: 1e-5 +0.0014, 1e-4 +0.0106, 0scf +0.0060,
  scf1 -0.0013, scf2 +0.0012.  2 fs: 1e-5 +0.0036, 1e-4 +0.030, 0scf +0.158 (!), scf1 -0.079, scf2 -0.006.
  -> 0scf heats at 2 fs from this start; CPU dyn run (equilibrated start, 2 fs) had 2e-4.  bench3.sh:
  dissipation order / kappa / no-shadow variants at 2 fs + dyn protocol on GPU.
- CPU dyn (2 fs, 50 ps NVE, equilibrated start): SCF 1e-5 mu rel err 7.8e-7, dU +0.18 kJ/mol (mixed vs
  double), econs rms 0.65 kJ/mol; 0scf mu rel err 1.9e-3, dU -0.33, econs rms 2.0 kJ/mol.
- eps replicas (12 x 1.5 ns NPT, 0scf, CPU): runs/iel/epscpu.sh -> runs/iel/eps_0scf_s*.{dip,log}
- 04:30-05:30 NVE 2 fs experiments (GPU bench3/4, CPU nve*): drift kT/ns/dof, 0scf Jacobi:
  K=5 +0.20/+0.21 (GPU/CPU), K=7 +0.047, K=3 explodes (T 473 K at 15 ps), K=0 (no dissipation) NaN at
  61 ps, kappa 1.4 +0.25, no-shadow +0.19/+0.18; omega 0.8 -0.106, omega 0.8 K7 -0.038, omega 0.7 -0.44,
  omega 0.8 K0 NaN.  scf-1 K3: -9.8 (T -> 92 K).  1 fs: 0scf K5 +0.003..+0.006, K7 -0.002, K0 +0.007.
  -> the drift at 2 fs is set by the dissipation (not by the phase lag at nuclear frequencies, which is
  1e-5 rad for K=5: /tmp/lag.py) acting on the auxiliary modes near Nyquist, and its sign by omega.
- block preconditioner: spectrum of M^-1 A 0.706-1.559 (10-90 %: 0.82-1.20) vs alpha A 0.679-1.853
  (runs/eig2.py); FD test of the block shadow forces passes.
- reference eps (scripts/iel_validate.py --eps prod.dip --skip 200): 31.02 +- 0.35, eps_inf 1.798,
  density 1.0179 +- 0.0003, mu_mol 1.9865 D, U -2115060 +- 5 kJ/mol, T 296.13.
- dyn (2 fs NVE 4 x 50 ps from the equilibrated box, runs/iel/dyn_*.json): D (1e-9 m2/s) SCF 3.73 +- 0.21,
  0scf 3.81 +- 0.12, scf1 3.70 +- 0.10; tau2 0.855 / 0.857 / 0.845 ps; mu rel err 7.8e-7 / 1.9e-3 / 1.1e-3;
  U~ - U* +0.18 / -0.32 / +0.23 kJ/mol; drift 0.0035 / 0.205 / -0.076.  (early dyn JSONs have drift in
  kT/ps/dof; --combine converts.)
- 05:45 released the 9 held eps replicas (0scf K5 Jacobi 2 fs, the literature scheme).
- 06:00-07:30 energy-flow picture (docs/iel.md): auxiliary modes carry energy of the sign of
  1 - omega lambda (lambda: eigenvalues of W A).  omega = 1: modes lambda > 1 negative -> damping them
  heats (K5 +0.2), K0 unstable (NaN 60 ps); omega 0.9 +0.045, 0.8 -0.106, 0.7 -0.44, 0.5 -2.6 (K5),
  0.5 K0 -0.46 with T falling to 265 K in 200 ps (energy flows into the undamped auxiliary modes: at
  2 fs they are not adiabatically separated from the librations).  1 fs, omega 0.5 K0: +0.0006,
  block omega 0.6 K0: +0.0034.  Jacobi K9 (omega 1): +0.008 (80 ps CPU); block K7 +0.019, block K5 +0.095,
  block K0 NaN at 34 ps.
- block preconditioner cost: 512 w NVE 0.41 vs 0.36 ms (Jacobi); Bussi 2 fs 0.506 vs 0.473 ms; 4096 w 1.394 vs 1.309.
- iEL/SCF-2 Bussi 2 fs: 512 w 0.546 ms (316 ns/d), 4096 w 1.519 ms (113.7).
- GPU queue: bench6 (1 ns NVE of block K7 / Jacobi K9 / block K9 / SCF; 1 fs 200 ps; dyn block K7;
  eps NPT 8 ns block K7 = runs/iel/geps_blk7_s100) then bench7 (dyn scf-2, eps scf-2 8 ns).
- 07:40 defaults -> block + K7 (1 ns NVE 2 fs: block K7 +0.013, block K9 +0.013, Jacobi K9 +0.026, SCF
  1e-5 +0.0035; 1 fs 200 ps: block K7 +0.0012, Jacobi K9 +0.0020, Jacobi K5 +0.0069).  Full suite 222 pass.
- dyn (GPU) block K7: D 3.75 +- 0.11, tau2 0.862, mu err 1.57e-3, dU -0.11, drift +0.012; SCF-2: D 3.73,
  tau2 0.884, mu err 2.3e-4, drift -0.0033, econs rms 0.71 kJ/mol.
- eps (NPT 2 fs Bussi, skip 200 ps): block K7 2 x 7.8 ns GPU (geps_blk7_s100/101): 30.51 +- 0.25
  (30.35, 30.62), density 1.0177, U -2115067 +- 4, mu 1.9865, T 296.9 (T_tr 296.7, T_rot 297.2);
  SCF-2 7.8 ns: 31.03 +- 0.71, 1.0175, -2115038 +- 8, T_rot - T_tr -0.8; reference SCF 31.02 +- 0.35,
  T_rot - T_tr -1.7.  Jacobi K5 CPU replicas (interim 10.2 ns): 30.46 +- 0.21, density 1.0188 +- 0.0005,
  U -2115094 +- 5, T_rot - T_tr +2.9.
- 1 fs Langevin block: 512 w 0.507 ms (170 ns/d), 4096 w 1.299 ms (66.5).
- TODO at ~11:30: final pooled eps of eps_0scf_s* (scripts/iel_validate.py --eps ... --skip 50), fill
  docs/iel.md placeholders (EPS_J5, RHO_J5, U_J5, MU_J5, T_J5, TROT_TEXT), commit.
