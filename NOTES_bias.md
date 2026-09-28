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
