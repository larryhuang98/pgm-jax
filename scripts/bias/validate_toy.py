"""Validation 1 of pgm_jax.bias: model potentials with exact free energy surfaces (bias/toy.py).

SYSTEM: dw (double well in x, barrier 25 kJ/mol), ring (periodic angle on a ring), mb (Mueller-Brown
x 0.25, 2 CVs); METHOD: metad (well-tempered, grid) | opes.  W walkers run independently (each its
own bias: W independent runs in one vmapped program) or, with --shared, as multiple walkers of one
bias, with the toy Langevin integrator (bias.toy.ToyLangevin).  At the end, for every run: the FES
from the bias (-f V) and the reweighted histogram (c(t) weights for metadynamics, exp(V/kT) for
OPES; frames after --skip-fraction of the run), each compared with the exact FES over the region
F_exact < --fmax-kJ after the best constant shift (RMSD, max).  See docs/enhanced_sampling.md.

Usage:

    python scripts/bias/validate_toy.py SYSTEM METHOD [--time-ns 20] [--walkers 8] [--shared] --out runs/bias/toy
    python scripts/bias/validate_toy.py --help

Inputs: none.
Outputs: <out>_<system>_<method>[_shared].json (per-run and average RMSDs) and .npz (FES arrays);
printed summary.
Units: --time-ns ns, --dt-fs fs, --temperature-K K, --friction-per-ps 1/ps, --mass-amu amu,
--height-kJ, --barrier-kJ and --fmax-kJ kJ/mol, --sample-every and --pace steps; the toy
coordinates in nm.
Runtime: CPU, minutes.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.bias import OPES, MetaD, cv
from pgm_jax.bias import analysis as A
from pgm_jax.bias.toy import ToyLangevin, double_well, mueller_brown, ring
from pgm_jax.cli.args import add_dt_arg, add_seed_arg, add_temperature_arg
from pgm_jax.units import KB

jax.config.update("jax_enable_x64", True)


def toy_system(a: argparse.Namespace) -> SimpleNamespace:
    """Return the model system: potential U, CVs, bias widths and grid, FES axes, start, periods, fmax, exact FES."""
    if a.system == "dw":
        U = double_well(barrier=25.0)
        cvs = [cv.Component(0, 0)]
        sigma, grid = [0.1], (-2.5, 2.5, 500)
        axes = [np.linspace(-1.6, 1.6, 81)]
        x0 = [[-1.0, 0.0, 0.0]]
        periods = [0.0]
        fmax = 20.0 if a.fmax_kJ is None else a.fmax_kJ
        Fex = U.fes(axes[0])
    elif a.system == "ring":
        U = ring()
        cvs = [cv.Custom(lambda x, H: jnp.arctan2(x[0, 1], x[0, 0]), period=2 * np.pi, name="theta")]
        sigma, grid = [0.2], (-np.pi, np.pi, 360)
        axes = [A.periodic_axis(90)]
        x0 = [[1.0, 0.0, 0.0]]
        periods = [2 * np.pi]
        fmax = 25.0 if a.fmax_kJ is None else a.fmax_kJ
        Fex = U.fes(axes[0])
    else:
        U = mueller_brown(scale=0.25)
        cvs = [cv.Component(0, 0), cv.Component(0, 1)]
        sigma, grid = [0.05, 0.05], ((-1.8, -0.6), (1.3, 2.4), (310, 300))
        axes = [np.linspace(-1.5, 1.1, 53), np.linspace(-0.4, 2.1, 51)]
        x0 = [[0.62, 0.03, 0.0]]
        periods = [0.0, 0.0]
        fmax = 30.0 if a.fmax_kJ is None else a.fmax_kJ
        gx, gy = np.meshgrid(*axes, indexing="ij")
        Fex = U.fes(gx, gy)
    Fex = Fex - Fex.min()
    return SimpleNamespace(U=U, cvs=cvs, sigma=sigma, grid=grid, axes=axes, x0=x0, periods=periods, fmax=fmax, Fex=Fex)


def run(a: argparse.Namespace, t: SimpleNamespace) -> tuple:
    """Run the biased toy dynamics in --segments segments; return (sim, bias, samples, steps, wall time [s])."""
    U, cvs, sigma, grid, x0 = t.U, t.cvs, t.sigma, t.grid, t.x0
    dt = a.dt_fs / 1000
    if a.method == "metad":
        bias = MetaD(cvs, sigma=sigma, height=a.height_kJ, pace=a.pace, biasfactor=a.biasfactor, grid=grid)
    else:
        bias = OPES(cvs, sigma=sigma, pace=a.pace, barrier=a.barrier_kJ)
    sim = ToyLangevin(
        U,
        x0,
        mass=a.mass_amu,
        temperature=a.temperature_K,
        dt=dt,
        gamma=a.friction_per_ps,
        bias=bias,
        walkers=a.walkers,
        shared=a.shared,
        seed=a.seed,
    )
    nsteps = int(round(a.time_ns * 1000 / dt))
    seg = nsteps // a.segments
    seg -= seg % a.sample_every
    outs = []
    t0 = time.time()
    for k in range(a.segments):
        outs.append(sim.run(seg, sample=a.sample_every))
        print(f"segment {k + 1}/{a.segments}: {time.time() - t0:.0f} s", flush=True)
    wall = time.time() - t0
    out = {key: np.concatenate([o[key] for o in outs]) for key in ("step", "cv", "bias")}
    return sim, bias, out, nsteps, wall


def analyse(
    a: argparse.Namespace, t: SimpleNamespace, sim: ToyLangevin, bias: object, out: dict, nsteps: int, wall: float
) -> dict:
    """Compare the FES of every run (from the bias and reweighted) with the exact FES; return the results."""
    axes, periods, fmax, Fex = t.axes, t.periods, t.fmax, t.Fex
    cvs = t.cvs
    pts, shape = A.mesh(*axes)
    mask = Fex < fmax
    W = a.walkers
    bstates = (
        [sim.state.bias.parts[0]]
        if a.shared
        else [jax.tree_util.tree_map(lambda x: x[w], sim.state.bias.parts[0]) for w in range(W)]
    )
    kT = KB * a.temperature_K
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
    keep = out["step"] > a.skip_fraction * nsteps
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
            f"{name}: RMSD per run {R.mean():.3f} +- {res[name]['rmsd_sem']:.3f} kJ/mol; "
            f"average of {len(Fs)} runs: RMSD {rm:.3f}, max {mm:.2f}, "
            f"mean error bar {res[name]['mean_pointwise_sem']:.3f}"
        )
    res["_F"] = (F_bias, F_rw)
    return res


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run, analyse and write the results (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system", choices=("dw", "ring", "mb"), help="model system")
    ap.add_argument("method", choices=("metad", "opes"), help="adaptive bias")
    ap.add_argument("--time-ns", type=float, default=20.0, help="run length [ns]")
    ap.add_argument("--walkers", type=int, default=8, help="walkers")
    ap.add_argument("--shared", action="store_true", help="the walkers share one bias")
    add_temperature_arg(ap, 300.0)
    add_dt_arg(ap, 5.0)
    ap.add_argument("--friction-per-ps", type=float, default=2.0, help="Langevin friction [1/ps]")
    ap.add_argument("--mass-amu", type=float, default=10.0, help="particle mass [amu]")
    ap.add_argument("--pace", type=int, default=200, help="steps between depositions")
    ap.add_argument("--height-kJ", type=float, default=1.0, help="metaD hill height [kJ/mol]")
    ap.add_argument("--biasfactor", type=float, default=10.0, help="metaD bias factor")
    ap.add_argument("--barrier-kJ", type=float, default=30.0, help="OPES barrier [kJ/mol]")
    ap.add_argument("--sample-every", type=int, default=100, help="steps between samples")
    ap.add_argument("--skip-fraction", type=float, default=0.2, help="fraction of the run discarded before reweighting")
    ap.add_argument("--fmax-kJ", type=float, default=None, help="region of the comparison: F < fmax [kJ/mol]")
    ap.add_argument("--segments", type=int, default=10, help="run segments (progress lines)")
    add_seed_arg(ap)
    ap.add_argument("-o", "--out", default="runs/bias/toy", help="output prefix")
    a = ap.parse_args(argv)
    t = toy_system(a)
    sim, bias, out, nsteps, wall = run(a, t)
    res = analyse(a, t, sim, bias, out, nsteps, wall)
    F_bias, F_rw = res.pop("_F")
    print(f"{nsteps} steps x {a.walkers} walkers in {wall:.0f} s ({res['us_per_step']:.1f} us per step)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tag = f"{a.out}_{a.system}_{a.method}{'_shared' if a.shared else ''}"
    with open(tag + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez(tag + ".npz", axes=np.array(t.axes, dtype=object), Fex=t.Fex, F_bias=np.array(F_bias), F_rw=np.array(F_rw))


if __name__ == "__main__":
    main()
