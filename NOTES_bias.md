# NOTES: enhanced sampling (item 13) — branch `bias`

Running log so the work can be resumed.

## Design (2026-09-28)
- Package `pgm_jax/bias/`: `cv.py` (CVs = JAX functions of (pos, H)), `core.py` (StaticBias,
  Harmonic, walls, MetaD, OPES, BiasSet/BiasState), `analysis.py` (FES, c(t), weights, histogram,
  WHAM), `io.py` (COLVAR/HILLS files), `toy.py` (Langevin engine for model potentials, walkers).
- MD hook: `MDState.bias` (BiasState pytree); `Integrator._forces(..., bias=st.bias)` adds bias
  energy + autodiff forces together with restraints (`_extra_energy`); `_run` calls
  `_bias_post` after each step: COLVAR row (every `colvar` steps, device buffer), then updates due
  (lax.cond), with the forces/epot corrected to the new bias at the same positions (bias-only
  grad), energy change booked as heat and BiasState.work. MTS: correction also added to the slow
  level. Barostat trial energies and pressure include the bias.
- Driver: reserve() buffers before each block (hills / kernels / COLVAR rows; shape change =>
  retrace), drain COLVAR rows after, files prefix.colvar / .hills / .bias; checkpoint includes bias.

## Progress log
- Commit fbbb43f: core package + MD hook + walkers + tests (19 tests pass on CPU, test_walkers added after).
- Toy validation (scripts/bias/validate_toy.py): 10 us/step for 8 walkers on CPU. dw metaD 20 ns: RMSD 0.21+-0.02
  (bias), 0.16+-0.01 (reweighted) per run; shared 8 walkers 0.075/0.085.
- OPES reserve was O(n_updates) -> buffer padding made OPES slow; now doubles at 80 % fill, forced merge
  on overflow (OPESState.forced counter).
- GPU vacuum ala2 (22 atoms): 0.75 ms/step single; metaD +10 %; 16 independent walkers 1386 ns/day aggregate.
- CPU vacuum ala2: 3.1 ms/step (16 cores) with PME 40^3; vacuum settings now beta 2.5, grid 20^3.
- NVE (8 waters, rigid, float64): static bias E max dev 0.010 kJ/mol with 17 kJ/mol bias range.
- Running: runs/ala2/chain.sh (GPU: umbrella 24 x 2 ns, metaD 12 x 4 ns, OPES 12 x 4 ns, 40-min segments,
  --resume); CPU: tests, nve_check, toy_all, engine_dw smoke.
- OPES recursive merging (PLUMED default) added, checked against the reference implementation.
- read_table: equal steps kept (walkers), only rows superseded by a continuation dropped.
- CPU queue: jobs without --mem take the whole node's memory; runs/cpul.sh now passes --mem (MEM=6G).
- GPU gpu-2-3 starved by fegrad's back-to-back chain (message sent to main). Fallback: ala2 on CPU:
  umbrella runs/ala2/usc_{0..21} (8 jobs x 3 windows x 1.5 ns), metaD mdc1-4 and OPES opc1-4 (3 walkers x
  3 ns each); GPU chain2.sh (md, op: 12 walkers x 4 ns) waits for the lock.
- CPU vacuum ala2: 1.28 ms/step single (grid 20, beta 2.5), 3 walkers 143 ns/day aggregate, 12 walkers 147.
- Full suite (before strided change): 230 passed (36 min, 32 cores).
- Strided loop: per-step lax.cond removed (GPU overhead 12 % -> 8 % solvated, 11 % -> 4 % vacuum).
- Umbrella/WHAM reference in phi FAILS for windows -52.5..-7.5 deg: psi trapped (0 or 1 fraction, <10
  transitions in 1.5 ns). metaD (GPU md, 12x4 ns) and OPES (CPU) agree with each other (2D RMSD 0.38 over
  159 bins F<15; dG(phi>0) 7.83+-0.04 / 7.89+-0.18) but not with WHAM (7.40, F max dev 2.4 near phi -45).
  -> REMD reference (8 replicas 300-700 K): GPU chain5.sh (remd 4 ns then OPES GPU), CPU remdc41/42 (1.5 ns).
- GPU gpu-2-3: fegrad / iface chains take the lock back-to-back; my jobs get in only occasionally.
- FINAL (10:30): full suite 231 passed (35 min, 32 cores) after the strided-loop change.
  ala2 vacuum: REMD (GPU 8x4 ns + CPU 2x8x1.5 ns) dG(phi>0) 7.75+-0.26; metaD 24 runs 7.88+-0.03, F(phi) vs REMD
  RMSD 0.17 (chi2 0.33); OPES 24 runs 7.90+-0.08, 0.17 (chi2 0.39); metaD vs OPES 2D 0.18. WHAM(phi only) 7.40
  (psi trapped in windows -52.5..-7.5). Docs: docs/enhanced_sampling.md. Analysis: runs/ala2/an6.sh -> an6.json.
- Not done: OPES MB 2D (too slow: O(K^2) Z, growing K), py-plumed, PT-metaD, solvated FES comparison.
