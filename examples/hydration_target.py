"""A hydration free energy as a fitting target: value, gradient, chi^2 and one Newton step.

The gradient is taken with respect to a scale s of one group of the solute's parameters
(theta = ln s).  The theta -> parameter table map is any JAX function (here: the solute's group,
e.g. charges and covalent dipoles, times exp(theta)); FreeEnergyTarget checks that it reproduces
the sampled table at theta = 0 and applies the chain rule, with jackknife errors of the projected
gradient.  docs/fe_gradients.md describes the method.

Usage:

    python examples/hydration_target.py runs/fg/me100_fe.npz --experiment-kcal -5.11 --group charge --discard-ps 50
    python examples/hydration_target.py --help

Inputs: the npz written by `scripts/free_energy/solvation_free_energy.py run --grad` (or
FreeEnergyRun with param_grad).
Outputs: printed value, gradient, chi^2, Newton step and the fit layout (y, J, covariances).
Units: kcal/mol (--experiment-kcal, --sigma-kcal and the printed values), --discard-ps ps.
Runtime: seconds.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.fit.free_energy import FreeEnergyTarget
from pgm_jax.fit.params import SCALE_GROUPS
from pgm_jax.md import fe_grad as fg
from pgm_jax.units import KCAL

jax.config.update("jax_enable_x64", True)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and print the target's value, gradient and Newton step (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="free-energy run with parameter gradients (.npz)")
    ap.add_argument(
        "--experiment-kcal", type=float, required=True, help="experimental hydration free energy [kcal/mol]"
    )
    ap.add_argument("--sigma-kcal", type=float, default=0.2, help="uncertainty of the target [kcal/mol]")
    ap.add_argument("--group", default="charge", choices=sorted(SCALE_GROUPS), help="parameter group that is scaled")
    ap.add_argument("--discard-ps", type=float, default=100.0, help="equilibration discarded per window [ps]")
    a = ap.parse_args(argv)
    t = FreeEnergyTarget.from_npz(
        a.npz, discard_ps=a.discard_ps, experiment=a.experiment_kcal * KCAL, sigma=a.sigma_kcal * KCAL, name="dG_hyd"
    )
    space = t.space
    P = space.unflatten(jnp.asarray(t.p))

    def theta_fn(theta):
        """Return the parameter table with the solute's group scaled by s = exp(theta[0])."""
        return fg.scaled_params(space, P, {a.group: jnp.exp(theta[0])})

    r = t.value_and_grad(theta_fn, jnp.zeros(1))
    g, ge = r["grad"][0] / KCAL, r["grad_err"][0] / KCAL
    print(
        f"DeltaG_hyd = {r['value'] / KCAL:.3f} +- {r['value_err'] / KCAL:.3f} kcal/mol (experiment {a.experiment_kcal})"
    )
    print(f"dDeltaG_hyd / d ln s_{a.group} = {g:.3f} +- {ge:.3f} kcal/mol;  chi2 = {r['chi2']:.2f}")
    step = (a.experiment_kcal - r["value"] / KCAL) / g
    print(f"Newton step: ln s = {step:+.4f} (s = {np.exp(step):.4f}); first-order s: {1 + step:.4f}")
    est = t.estimate(theta_fn, jnp.zeros(1), unit="kcal/mol")
    print(
        "fit layout:",
        {k: np.round(np.asarray(v, float), 4).tolist() for k, v in est.items() if k in ("y", "J", "cov_y", "J_err")},
    )


if __name__ == "__main__":
    main()
