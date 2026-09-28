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
