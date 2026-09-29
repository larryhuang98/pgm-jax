"""Speed of the three bonded term sets at MD time (energy + forces of many copies of a molecule).

The bonded terms of --copies copies of a molecule (test frames), vmapped over copies as
FlexibleMolecules does, on one device: the amber set, the explore set (class II) and the neural set
(timed frozen: stage-1 coefficients evaluated once, as in MD; stage 1 itself is timed separately).
Couplings are set nonzero so that nothing is skipped.

Usage:

    python scripts/bonded/bench_sets.py [--mol alanine_dipeptide] [--copies 500]
    python scripts/bonded/bench_sets.py --help

Inputs: the study data of the molecule (pgm_jax.bonded.study.data).
Outputs: runs/bonded/nnb/bench_sets.json and printed timings.
Units: ms per evaluation.
Runtime: GPU, a minute.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.bonded import terms as T
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.bonded.study.data import load
from pgm_jax.paths import repo_path

jax.config.update("jax_enable_x64", True)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, time the three sets and write the JSON (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mol", default="alanine_dipeptide", help="molecule of the study")
    ap.add_argument("--copies", type=int, default=500, help="copies evaluated together")
    ap.add_argument("--reps", type=int, default=200, help="timed evaluations")
    a = ap.parse_args(argv)
    specs, data = load([a.mol], with_scans=False)
    X0 = np.asarray(data[0]["test"].X)
    X = jnp.asarray(np.concatenate([X0] * (a.copies // len(X0) + 1))[: a.copies])
    n_atoms = int(X.shape[0] * X.shape[1])
    out = {"molecule": a.mol, "copies": a.copies, "atoms": n_atoms, "device": str(jax.devices()[0]), "sets": {}}
    for name, fams in (
        ("amber", T.SETS["amber"]),
        ("explore (class II)", T.SETS["explore"]),
        ("nn (frozen)", T.SETS["nn"]),
    ):
        model = BondedModel(specs, BondedSettings(families=fams))
        P = model.init_params()
        for f in model.fams:  # nonzero couplings so that nothing is skipped
            P[f] = {k: v + 0.1 for k, v in P[f].items()}
        if model.nnb is not None:
            rng = np.random.default_rng(0)
            P["nnb"] = jax.tree_util.tree_map(lambda v: v + 0.05 * jnp.asarray(rng.normal(size=v.shape)), P["nnb"])
            coef = jax.jit(lambda p: model.nnb.coefficients(p, 0))
            jax.block_until_ready(coef(P["nnb"]))
            t0 = time.perf_counter()
            for _ in range(20):
                jax.block_until_ready(coef(P["nnb"]))
            stage1 = (time.perf_counter() - t0) / 20
            P = dict(P)
            P["nnb"] = model.nnb.freeze(P["nnb"])
        f = jax.jit(jax.vmap(jax.value_and_grad(lambda R: model.bonded_energy(0, R, P))))
        jax.block_until_ready(f(X))
        t0 = time.perf_counter()
        for _ in range(a.reps):
            jax.block_until_ready(f(X))
        dt = (time.perf_counter() - t0) / a.reps
        out["sets"][name] = {"ms_per_eval": 1e3 * dt}
        if model.nnb is not None:
            out["sets"][name]["stage1_ms_per_molecule"] = 1e3 * stage1
        print(name, out["sets"][name], flush=True)
    os.makedirs(repo_path("runs", "bonded", "nnb"), exist_ok=True)
    with open(repo_path("runs", "bonded", "nnb", "bench_sets.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
