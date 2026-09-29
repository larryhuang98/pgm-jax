"""Validation 1 of pgm_jax.bias: model potentials with exact free energy surfaces (bias/toy.py).

    python scripts/bias/validate_toy.py SYSTEM METHOD [--ns 20] [--walkers 8] [--shared] --out runs/bias/toy

SYSTEM: dw (double well in x, barrier 25 kJ/mol), ring (periodic angle on a ring), mb (Mueller-Brown x 0.25,
2 CVs); METHOD: metad (well-tempered, grid) | opes.  W walkers run independently (each its own bias:
W independent runs in one vmapped program) or, with --shared, as multiple walkers of one bias.
At the end, for every run: the FES from the bias (-f V) and the reweighted histogram (c(t) weights
for metadynamics, exp(V/kT) for OPES; frames after --skip of the run), each compared with the exact
FES over the region F_exact < --fmax after the best constant shift (RMSD, max).  Writes OUT_SYSTEM_METHOD.json
and .npz (FES arrays)."""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pgm_jax.bias import OPES, MetaD, cv  # noqa: E402
from pgm_jax.bias import analysis as A  # noqa: E402
from pgm_jax.bias.core import KB  # noqa: E402
from pgm_jax.bias.toy import ToyLangevin, double_well, mueller_brown, ring  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("system", choices=("dw", "ring", "mb"))
ap.add_argument("method", choices=("metad", "opes"))
ap.add_argument("--ns", type=float, default=20.0)
ap.add_argument("--walkers", type=int, default=8)
ap.add_argument("--shared", action="store_true")
ap.add_argument("--T", type=float, default=300.0)
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--mass", type=float, default=10.0)
ap.add_argument("--pace", type=int, default=200)
ap.add_argument("--height", type=float, default=1.0)
ap.add_argument("--biasfactor", type=float, default=10.0)
ap.add_argument("--barrier", type=float, default=30.0)
ap.add_argument("--sample", type=int, default=100)
ap.add_argument("--skip", type=float, default=0.2, help="fraction of the run discarded before reweighting")
ap.add_argument("--fmax", type=float, default=None)
ap.add_argument("--segments", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/bias/toy")
a = ap.parse_args()

kT = KB * a.T
if a.system == "dw":
    U = double_well(barrier=25.0)
    cvs = [cv.Component(0, 0)]
    sigma, grid = [0.1], (-2.5, 2.5, 500)
    axes = [np.linspace(-1.6, 1.6, 81)]
    x0 = [[-1.0, 0.0, 0.0]]
    periods = [0.0]
    fmax = 20.0 if a.fmax is None else a.fmax
    Fex = U.fes(axes[0])
elif a.system == "ring":
    U = ring()
    cvs = [cv.Custom(lambda x, H: jnp.arctan2(x[0, 1], x[0, 0]), period=2 * np.pi, name="theta")]
    sigma, grid = [0.2], (-np.pi, np.pi, 360)
    axes = [A.periodic_axis(90)]
    x0 = [[1.0, 0.0, 0.0]]
    periods = [2 * np.pi]
    fmax = 25.0 if a.fmax is None else a.fmax
    Fex = U.fes(axes[0])
else:
    U = mueller_brown(scale=0.25)
    cvs = [cv.Component(0, 0), cv.Component(0, 1)]
    sigma, grid = [0.05, 0.05], ((-1.8, -0.6), (1.3, 2.4), (310, 300))
    axes = [np.linspace(-1.5, 1.1, 53), np.linspace(-0.4, 2.1, 51)]
    x0 = [[0.62, 0.03, 0.0]]
    periods = [0.0, 0.0]
    fmax = 30.0 if a.fmax is None else a.fmax
    gx, gy = np.meshgrid(*axes, indexing="ij")
    Fex = U.fes(gx, gy)
Fex = Fex - Fex.min()
pts, shape = A.mesh(*axes)
mask = Fex < fmax

if a.method == "metad":
    bias = MetaD(cvs, sigma=sigma, height=a.height, pace=a.pace, biasfactor=a.biasfactor, grid=grid)
else:
    bias = OPES(cvs, sigma=sigma, pace=a.pace, barrier=a.barrier)
sim = ToyLangevin(
    U,
    x0,
    mass=a.mass,
    temperature=a.T,
    dt=a.dt,
    gamma=a.gamma,
    bias=bias,
    walkers=a.walkers,
    shared=a.shared,
    seed=a.seed,
)
nsteps = int(round(a.ns * 1000 / a.dt))
seg = nsteps // a.segments
seg -= seg % a.sample
outs = []
t0 = time.time()
for k in range(a.segments):
    outs.append(sim.run(seg, sample=a.sample))
    print(f"segment {k + 1}/{a.segments}: {time.time() - t0:.0f} s", flush=True)
wall = time.time() - t0
out = {key: np.concatenate([o[key] for o in outs]) for key in ("step", "cv", "bias")}
W = a.walkers
bstates = (
    [sim.state.bias.parts[0]]
    if a.shared
    else [jax.tree_util.tree_map(lambda x: x[w], sim.state.bias.parts[0]) for w in range(W)]
)
kT = KB * a.T
res = {
    "args": vars(a),
    "fmax": fmax,
    "wall_s": wall,
    "steps": nsteps,
    "us_per_step": wall / nsteps * 1e6,
    "bias": bias.describe(),
    "runs": [],
}
F_bias, F_rw = [], []
keep = out["step"] > a.skip * nsteps
fine = (
    A.mesh(
        *[
            np.linspace(ax[0] - 1.0, ax[-1] + 1.0, 400) if p == 0 else A.periodic_axis(360)
            for ax, p in zip(axes, periods)
        ]
    )[0]
    if len(axes) == 1
    else A.mesh(np.linspace(-1.8, 1.3, 156), np.linspace(-0.6, 2.4, 151))[0]
)
for r, st in enumerate(bstates):
    Fb = A.fes_from_bias(bias, st, pts).reshape(shape)
    if a.shared:
        s, v, steps = out["cv"].reshape(-1, len(cvs)), out["bias"][..., 0].reshape(-1), np.repeat(out["step"], W)
    else:
        s, v, steps = out["cv"][:, r], out["bias"][:, r, 0], out["step"]
    if a.method == "metad":
        hs, ct = A.metad_ct(bias.hills(st), a.biasfactor, kT, periods, fine)
        if a.shared:  # the hills of one deposition step come from all walkers: c(t) after the last of them
            last = np.append(hs[1:] != hs[:-1], True)
            hs, ct = hs[last], ct[last]
        lw = A.ct_weights(steps, v, hs, ct, kT)
    else:
        lw = A.opes_weights(v, kT)
    k = np.repeat(keep, W) if a.shared else keep
    Fh = A.histogram_fes(s[k], lw[k], axes, kT, periods)
    rb, mb, _ = A.align_rmsd(Fb, Fex, mask)
    rh, mh, _ = A.align_rmsd(Fh, Fex, mask)
    x = s[:, 0]
    info = bias.info(st)
    res["runs"].append(
        {
            "rmsd_bias": rb,
            "max_bias": mb,
            "rmsd_reweight": rh,
            "max_reweight": mh,
            **{kk: float(vv) for kk, vv in info.items()},
        }
    )
    F_bias.append(Fb)
    F_rw.append(Fh)
    print(
        f"run {r}: FES from the bias RMSD {rb:.3f} (max {mb:.2f}) kJ/mol, reweighted RMSD {rh:.3f} (max {mh:.2f}); "
        f"{info}"
    )
for name, Fs in (("bias", F_bias), ("reweight", F_rw)):
    R = np.array([rr[f"rmsd_{name}"] for rr in res["runs"]])
    Fs = np.array([A.align_rmsd(F, Fex, mask)[2] for F in Fs])
    Fm = Fs.mean(0)
    err = Fs.std(0, ddof=1) / np.sqrt(len(Fs)) if len(Fs) > 1 else np.zeros_like(Fm)
    rm, mm, _ = A.align_rmsd(Fm, Fex, mask)
    res[name] = {
        "rmsd_mean": float(R.mean()),
        "rmsd_sem": float(R.std(ddof=1) / np.sqrt(len(R))) if len(R) > 1 else 0.0,
        "rmsd_of_average": rm,
        "max_of_average": mm,
        "mean_pointwise_sem": float(err[mask].mean()),
        "chi2_per_point": float(np.mean(((Fm - Fex)[mask] / np.maximum(err[mask], 1e-9)) ** 2))
        if len(Fs) > 1
        else None,
    }
    print(
        f"{name}: RMSD per run {R.mean():.3f} +- {res[name]['rmsd_sem']:.3f} kJ/mol; average of {len(Fs)} runs: RMSD "
        f"{rm:.3f}, max {mm:.2f}, mean error bar {res[name]['mean_pointwise_sem']:.3f}"
    )
print(f"{nsteps} steps x {W} walkers in {wall:.0f} s ({res['us_per_step']:.1f} us per step)")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
tag = f"{a.out}_{a.system}_{a.method}{'_shared' if a.shared else ''}"
json.dump(res, open(tag + ".json", "w"), indent=1)
np.savez(tag + ".npz", axes=np.array(axes, dtype=object), Fex=Fex, F_bias=np.array(F_bias), F_rw=np.array(F_rw))
