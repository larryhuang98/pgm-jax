"""A hydration free energy as a fitting target: value, gradient with respect to a scale of the
solute's parameters (theta = ln s), chi^2 against experiment and one Newton step.

    python examples/hydration_target.py runs/fg/me100_fe.npz --experiment -5.11 --group charge --discard-ps 50

The npz is the output of `scripts/solvation_free_energy.py run --grad` (or FreeEnergyRun with
param_grad).  theta -> parameter table is any JAX function (here: the solute's charges and covalent
dipoles times exp(theta)); FreeEnergyTarget checks that it reproduces the sampled table at theta = 0
and applies the chain rule, with jackknife errors of the projected gradient.  docs/fe_gradients.md."""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgm_jax.md import fe_grad as fg  # noqa: E402

KCAL = 4.184


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--experiment", type=float, required=True, help="kcal/mol")
    ap.add_argument("--sigma", type=float, default=0.2, help="kcal/mol")
    ap.add_argument("--group", default="charge", choices=sorted(fg.SCALE_GROUPS))
    ap.add_argument("--discard-ps", type=float, default=100.0)
    a = ap.parse_args()
    t = fg.FreeEnergyTarget.from_npz(a.npz, discard_ps=a.discard_ps, experiment=a.experiment * KCAL,
                                     sigma=a.sigma * KCAL, name="dG_hyd")
    space = t.space
    P = space.unflatten(jnp.asarray(t.p))

    def theta_fn(theta):                          # ln s -> table (the solute's group scaled by s)
        return fg.scaled_params(space, P, {a.group: jnp.exp(theta[0])})

    r = t.value_and_grad(theta_fn, jnp.zeros(1))
    g, ge = r["grad"][0] / KCAL, r["grad_err"][0] / KCAL
    print(f"DeltaG_hyd = {r['value'] / KCAL:.3f} +- {r['value_err'] / KCAL:.3f} kcal/mol (experiment {a.experiment})")
    print(f"dDeltaG_hyd / d ln s_{a.group} = {g:.3f} +- {ge:.3f} kcal/mol;  chi2 = {r['chi2']:.2f}")
    step = (a.experiment - r["value"] / KCAL) / g
    print(f"Newton step: ln s = {step:+.4f} (s = {np.exp(step):.4f}); first-order s: {1 + step:.4f}")
    est = t.estimate(theta_fn, jnp.zeros(1), unit="kcal/mol")
    print("fit layout:", {k: np.round(np.asarray(v, float), 4).tolist() for k, v in est.items() if k in ("y", "J", "cov_y", "J_err")})


if __name__ == "__main__":
    main()
