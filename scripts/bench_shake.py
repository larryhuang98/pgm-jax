"""Cost of SHAKE (positions) and RATTLE (momenta) per call on the current device (docs/shake.md).

    python scripts/bench_shake.py [--ubq runs/protein/ubq]      # -> runs/shake/bench_shake.json
Systems: 4,096 rigid waters (3 constraints each), 216 methanols (fitted template, X-H bonds or every
bond), ubiquitin in water (tleap system, X-H bonds or every bond: the protein's 1,2xx bonds are one
cluster, solved iteratively); blocks by cluster size (default) against one padded block."""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.md.constraints import Constraints, hmr_masses
from pgm_jax.md.flexible import FlexibleTemplate, liquid_box
from pgm_jax.md.topology import MDTopology
from pgm_jax.paths import resource
from pgm_jax.system import System

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jax.config.update("jax_enable_x64", True)


def timeit(f, *a, reps=200):
    out = f(*a)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = f(*a)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps * 1e3


def bench(label, pairs, d0, m, x, res, **kw):
    C = Constraints(pairs, d0, m, **kw)
    rng = np.random.default_rng(0)
    X = jnp.asarray(x)
    for _ in range(3):
        X = C.positions(X, X)
    Y = X + 0.002 * jnp.asarray(rng.normal(size=x.shape))  # ~ one 2 fs drift
    P = jnp.asarray(rng.normal(size=x.shape) * np.sqrt(np.maximum(m, 1e-9))[:, None])
    pos = jax.jit(C.positions)
    mom = jax.jit(lambda q, p: C.momenta(q, p, m))
    z = pos(Y, X)
    res[label] = {
        "constraints": C.nc,
        "atoms": int(len(x)),
        "blocks": C.describe(),
        "shake_ms": timeit(pos, Y, X),
        "rattle_ms": timeit(mom, z, P),
        "violation": float(C.violation(z)),
        "rattle_err": float(C.velocity_violation(z, mom(z, P), m)),
    }
    print(label, json.dumps(res[label]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubq", default=resource("ubq_runs"))
    a = ap.parse_args()
    res = {"device": str(jax.devices()[0])}
    # 4,096 waters
    nw = 4096
    t = np.radians(104.52 / 2)
    w = np.array(
        [[0, 0, 0], [0.09572 * np.sin(t), 0.09572 * np.cos(t), 0], [-0.09572 * np.sin(t), 0.09572 * np.cos(t), 0]]
    )
    rng = np.random.default_rng(0)
    x = np.concatenate([w + rng.uniform(0, 5, 3) for _ in range(nw)])
    pairs = np.array([(3 * k + i, 3 * k + j) for k in range(nw) for i, j in ((0, 1), (0, 2), (1, 2))])
    d0 = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
    bench("water4096", pairs, d0, np.tile([15.999, 1.008, 1.008], nw), x, res)
    # 216 methanols
    tpl = FlexibleTemplate.load(os.path.join(ROOT, "runs/flex/methanol.flex"))
    x, H = liquid_box(tpl, 216, 0.75, seed=1, min_dist=0.18)
    sys_ = System([tpl.pgm] * 216)
    for cons in ("h-bonds", "all-bonds"):
        top = MDTopology.build(sys_, [tpl.md_rule(cons)] * 216)
        m = np.asarray(sys_.masses, float)
        bench(f"meoh216 {cons}", top.constraints, top.constraint_d0, m, x, res)
        bench(f"meoh216 {cons}, one block", top.constraints, top.constraint_d0, m, x, res, bucket=False)
    # ubiquitin in water
    if os.path.exists(a.ubq + ".prmtop"):
        from pgm_jax.protein import amber_template, load_amber

        asys = load_amber(a.ubq + ".prmtop", a.ubq + ".inpcrd")
        prot = {k: amber_template(m, a.ubq + ".prmtop") for k, m in enumerate(asys.molecules) if m.kind == "protein"}
        templates = asys.templates(prot)
        sys_ = asys.system()
        x = asys.system_positions()
        m = hmr_masses(sys_, 3.024)
        for cons in ("h-bonds", "all-bonds"):
            top = MDTopology.build(sys_, [t.md_rule(cons) for t in templates])
            bench(f"ubq {cons}", top.constraints, top.constraint_d0, m, x, res)
            if cons == "h-bonds":
                bench(f"ubq {cons}, one block", top.constraints, top.constraint_d0, m, x, res, bucket=False)
    os.makedirs(os.path.join(ROOT, "runs/shake"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "runs/shake/bench_shake.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
