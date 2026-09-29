"""Speed of the three bonded term sets at MD time: energy + forces of the bonded terms of many
copies of a molecule (vmapped over copies, as FlexibleMolecules does), one GPU.
    python scripts/bonded/bench_sets.py [--mol alanine_dipeptide] [--copies 500]
The neural set is timed frozen (stage-1 coefficients evaluated once, as in MD); stage 1 itself
is timed separately."""

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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser()
ap.add_argument("--mol", default="alanine_dipeptide")
ap.add_argument("--copies", type=int, default=500)
ap.add_argument("--reps", type=int, default=200)
a = ap.parse_args()
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
os.makedirs(os.path.join(ROOT, "runs/bonded/nnb"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "runs/bonded/nnb/bench_sets.json"), "w"), indent=1)
