# How to: van der Waals (Lennard-Jones) parameters

Lennard-Jones parameters in pGM-JAX are `lj_rmin_half` (R* = r_min / 2, nm) and `lj_sqrt_eps`
(the square root of epsilon, sqrt(kJ/mol)), tied by atom type by default. They can be fitted to
gas-phase data (dimer energies) with automatic differentiation, and to liquid properties with
ensemble gradients from MD.

## 1. Liquid density and heat of vaporization

```bash
python scripts/fit_liquid.py methanol --iters 6                         # to experiment
python scripts/fit_liquid.py water --start 0.0296,-0.357 --targets 1.01,8.6 --iters 8   # recovery test
```

Each iteration runs one NPT simulation, stores frames every 0.2 ps, and for each frame computes
the potential energy U, the density and dU/dtheta (JAX, at the converged induced dipoles). The
fluctuation formula

    d<A>/dtheta = <dA/dtheta> - beta (<A dU/dtheta> - <A><dU/dtheta>)

gives the Jacobian of rho = M/V and dHvap = <U_gas> - <U_liq>/N + RT, and a damped Gauss-Newton
step (clipped) gives the next parameters. The log prints the values predicted for the next
iteration from the Jacobian, next to what the next simulation gives: this is the check that the
gradients are right and the step is within the linear range. Results go to
`runs/liquid/<system>.json`.

What to change for your own system:

- **Parameters.** `ParamMap` maps theta to the parameter table. `--params global` (default)
  uses two global scales (all R* times s_R, all epsilon times s_eps); `--params type` uses one
  pair of scales per Lennard-Jones atom type with epsilon > 0 (the log lists their names, e.g.
  `ln s_R[MeOH:c3]`). With two targets and more parameters the Gauss-Newton step is the
  minimum-norm step, so for per-type fits add targets (other liquids, temperatures, gas-phase
  dimers) or a prior. Any other map works the same way: dU/dtheta is taken by
  `jax.value_and_grad` through it, so no derivative code changes.
- **System.** `build()` returns the System, coordinates, box and time step; rigid molecules come
  from an Amber pGM prmtop (`Simulation`), flexible ones from a `FlexibleTemplate` (see
  `howto_bonded.md`).
- **Targets and weights.** `EXP` and `--sig_rho`, `--sig_dh` (the scale of each residual in the
  Gauss-Newton objective). More observables are one more row of the Jacobian: any function of a
  frame (energy, volume, box) works with the same formula; for enthalpy-derived properties
  (heat capacity, thermal expansion, compressibility) take the fluctuation expressions and
  differentiate them the same way.
- **Gas phase.** For rigid molecules, U_gas is the monomer energy. For flexible molecules it is
  sampled by gas-phase Langevin MD with the bonded model. If a molecule has intramolecular LJ
  pairs (1-5 or more bonds apart, or scaled 1-4), <U_gas> depends on the LJ parameters too and
  its gradient needs the same fluctuation formula over the gas-phase frames (not implemented
  yet; the script stops with a message).
- **Statistics.** Errors are block standard errors (5 blocks). The Jacobian is noisier than the
  averages; 200 ps per iteration was enough for water and methanol with two parameters. More
  parameters need longer runs or more molecules.

## 2. Gas-phase data

Dimer and cluster energies (e.g. S66-type data or SAPT components) are fitted with the gas-phase
`Model`:

```python
from pgm_jax import ElecChannel, LJChannel, Model, System

model = Model([ElecChannel(), LJChannel()])
E = model.energy_fn(System([a, b]))  # E(pos, P) -> {"perm", "ind", "vdw", "total"}
Ea, Eb = model.energy_fn(System([a])), model.energy_fn(System([b]))


def loss(P):
    e_int = jax.vmap(lambda x: E(x, P)["total"] - Ea(x[:na], P)["total"] - Eb(x[na:], P)["total"])(X)
    return jnp.mean((e_int - E_qm) ** 2)


g = jax.grad(loss)(P)  # same pytree as P: only LJ keys if you mask the rest
```

`elec_decomposition` splits the intermolecular pGM energy into electrostatic and induction parts
comparable with SAPT. Combining a gas-phase loss with the liquid Gauss-Newton residuals is a sum
of the two objectives.

## 3. Checks

- `pytest -q tests/test_md.py tests/test_flexible.py` (engine, derivatives, flexible molecules).
- After a fit, run the final parameters longer (e.g. 1 ns) and compare density and dHvap with
  the fit's last iteration.
