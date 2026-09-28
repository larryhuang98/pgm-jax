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
