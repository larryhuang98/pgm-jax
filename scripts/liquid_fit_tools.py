"""Analysis of pgm_jax/fit runs (scripts/fit_multi.py outputs): combine segments, finite-difference
check of the ensemble gradients, calibration of the parameter uncertainties.

    # observables, Jacobians and jackknife errors from saved frames (segments of one run at fixed theta)
    python scripts/liquid_fit_tools.py combine runs/fit/c0 --params q --targets density,hvap,eps,liquid_dipole
    # finite differences between independent runs at theta -/+ delta e_j vs the fluctuation-formula gradient
    python scripts/liquid_fit_tools.py fd --minus runs/fit/m --center runs/fit/c0 --plus runs/fit/p --delta 0.1 --param 0 \
        --params q --targets density,hvap,eps,liquid_dipole
    # spread of fitted parameters over independent fits vs the predicted sampling errors
    python scripts/liquid_fit_tools.py calib runs/fit/cal_s*.json

The same --params / --targets / --model options as fit_multi.py define the objective."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
from fit_multi import add_arguments, setup  # noqa: E402

from pgm_jax.fit import LiquidSamples  # noqa: E402

GRAD_KEYS = ("dU", "dM", "dalpha", "dD")


def load_frames(prefix, skip: int = 0, select=None):
    """Concatenate prefix_frames*.npz (segments in order); returns (frames dict, theta).  select:
    indices of the parameters to keep (the frames of a run carry derivatives for all of its own)."""
    files = sorted(glob.glob(prefix + "_frames*.npz"))[skip:]
    if not files:
        raise SystemExit(f"no frames for {prefix}")
    parts = [dict(np.load(f)) for f in files]
    th = parts[0]["theta"]
    for p in parts[1:]:
        if not np.allclose(p["theta"], th):
            raise SystemExit(f"{prefix}: segments at different theta")
    keys = [k for k in parts[0] if k not in ("theta", "rep")]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if "rep" in parts[0]:                  # batched replicas: order by replica, then time (contiguous blocks)
        rep = np.concatenate([p["rep"] for p in parts])
        order = np.argsort(rep, kind="stable")
        out = {k: v[order] for k, v in out.items()}
        out["rep"] = rep[order]
    if select is not None:
        th = th[select]
        for k in GRAD_KEYS:
            out[k] = out[k][..., select]
    return out, th


def _select(a):
    return None if not getattr(a, "select", "") else [int(x) for x in a.select.split(",")]


def estimate(S, a, prefix):
    fr, th = load_frames(prefix, a.skip, _select(a))
    fr.pop("rep", None)
    s = LiquidSamples(fr, a.T, S["sys"].nmol, float(np.sum(S["sys"].masses)), a.nblocks)
    return S["obj"].estimate(s, th), s


def cmd_combine(S, a):
    for pre in a.prefixes:
        est, s = estimate(S, a, pre)
        print(f"# {pre}: {s.F} frames, theta {np.round(est.theta, 5).tolist()}")
        for i, n in enumerate(est.names):
            print(f"   {n:20s} {est.y[i]:12.5f} +- {est.err[i]:9.5f}   d/dtheta " +
                  " ".join(f"{j:11.4f} +- {e:8.4f}" for j, e in zip(est.J[i], est.J_err[i])))
        if a.json:
            json.dump(est.as_dict(), open(pre + "_combined.json", "w"), indent=1)


def cmd_fd(S, a):
    em, sm = estimate(S, a, a.minus)
    ec, sc = estimate(S, a, a.center)
    ep, sp = estimate(S, a, a.plus)
    j = a.param
    dth = ep.theta[j] - em.theta[j]
    print(f"# finite differences along {S['space'].names[j]}: theta {em.theta[j]:+.4f} / {ec.theta[j]:+.4f} / "
          f"{ep.theta[j]:+.4f}; frames {sm.F} / {sc.F} / {sp.F}")
    rows = []
    for i, n in enumerate(ec.names):
        fd = (ep.y[i] - em.y[i]) / dth
        fd_err = np.hypot(ep.err[i], em.err[i]) / abs(dth)
        g = [em.J[i, j], ec.J[i, j], ep.J[i, j]]
        ge = [em.J_err[i, j], ec.J_err[i, j], ep.J_err[i, j]]
        simpson = (g[0] + 4 * g[1] + g[2]) / 6.0                          # mean slope over [-d, +d] (exact for cubics)
        simpson_err = np.sqrt(ge[0] ** 2 + 16 * ge[1] ** 2 + ge[2] ** 2) / 6.0
        z = (fd - simpson) / np.hypot(fd_err, simpson_err)
        rows.append({"name": n, "y": [em.y[i], ec.y[i], ep.y[i]], "y_err": [em.err[i], ec.err[i], ep.err[i]],
                     "fd": fd, "fd_err": fd_err, "grad": g, "grad_err": ge, "simpson": simpson, "simpson_err": simpson_err, "z": z})
        print(f"   {n:16s} y {em.y[i]:10.4f} {ec.y[i]:10.4f} {ep.y[i]:10.4f} (+- {ec.err[i]:.4f})  FD {fd:11.4f} +- {fd_err:9.4f}  "
              f"gradient {g[0]:11.4f} {g[1]:11.4f} {g[2]:11.4f} (+- {ge[1]:.4f})  Simpson {simpson:11.4f} +- {simpson_err:9.4f}  z {z:+.2f}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=float)


def cmd_calib(a):
    th, err, errb = [], [], []
    for f in a.files:
        d = json.load(open(f))
        r = d["records"][-1]
        th.append(r["next_theta"])
        err.append(r["uq"]["theta_err"])
        errb.append(r["uq"].get("bootstrap_theta_sd", [np.nan] * len(r["next_theta"])))
    th, err, errb = np.array(th), np.array(err), np.array(errb)
    n = len(th)
    sd = th.std(0, ddof=1)
    names = json.load(open(a.files[0]))["names"]
    print(f"# {n} independent fits")
    for k, nm in enumerate(names):
        # chi-square interval of the sample standard deviation (95 %)
        from scipy.stats import chi2
        lo, hi = sd[k] * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1)), sd[k] * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1))
        print(f"   {nm:12s} mean {th[:, k].mean():+.5f}  spread {sd[k]:.5f} (95 % {lo:.5f}-{hi:.5f})  "
              f"predicted (jackknife) {np.sqrt(np.mean(err[:, k] ** 2)):.5f}  bootstrap {np.sqrt(np.nanmean(errb[:, k] ** 2)):.5f}")


def cmd_calib_rep(S, a):
    """Independent fits from groups of replicas of one run: spread of the fitted parameters and of the
    observables over the groups vs the predicted (jackknife + propagation, bootstrap) errors."""
    fr, th = load_frames(a.prefixes[0], a.skip, _select(a))
    rep = fr.pop("rep")
    R = int(rep.max()) + 1
    g = a.group
    obj = S["obj"]
    fits, errs, errb, ys, yerr = [], [], [], [], []
    for k in range(R // g):
        sel = (rep >= k * g) & (rep < (k + 1) * g)
        s = LiquidSamples({key: v[sel] for key, v in fr.items()}, a.T, S["sys"].nmol, float(np.sum(S["sys"].masses)), a.nblocks)
        est = obj.estimate(s, th)
        st = obj.step(est, radius=np.inf)
        cov = obj.covariance(est)
        b = obj.bootstrap(s, est, np.inf, a.nboot, seed=k)
        fits.append(th + st["delta"])
        errs.append(cov["theta_err"])
        errb.append(b["theta_sd"])
        ys.append(est.y)
        yerr.append(est.err)
    fits, errs, errb, ys, yerr = map(np.array, (fits, errs, errb, ys, yerr))
    n = len(fits)
    from scipy.stats import chi2
    print(f"# {n} independent fits of {g} replicas each ({R} replicas, {len(rep)} frames); targets "
          + ", ".join(f"{t.name}={t.value}" for t in obj.targets if t.fit))
    rows = []
    for j, nm in enumerate(S["space"].names):
        sd = fits[:, j].std(ddof=1)
        lo, hi = sd * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1)), sd * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1))
        pj, pb = np.sqrt(np.mean(errs[:, j] ** 2)), np.sqrt(np.mean(errb[:, j] ** 2))
        rows.append({"param": nm, "spread": sd, "spread_95": [lo, hi], "predicted": pj, "bootstrap": pb, "mean": fits[:, j].mean()})
        print(f"   {nm:14s} fitted {fits[:, j].mean():+.5f}  spread {sd:.5f} (95 % {lo:.5f}-{hi:.5f})  predicted {pj:.5f}  bootstrap {pb:.5f}")
    names = obj.layout()[0]
    for i, nm in enumerate(names):
        sd = ys[:, i].std(ddof=1)
        lo, hi = sd * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1)), sd * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1))
        pe = np.sqrt(np.mean(yerr[:, i] ** 2))
        rows.append({"observable": nm, "spread": sd, "spread_95": [lo, hi], "jackknife": pe, "mean": ys[:, i].mean()})
        print(f"   {nm:14s} mean {ys[:, i].mean():10.4f}  spread {sd:.4f} (95 % {lo:.4f}-{hi:.4f})  jackknife {pe:.4f}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combine")
    c.add_argument("prefixes", nargs="+")
    f = sub.add_parser("fd")
    f.add_argument("--minus", required=True)
    f.add_argument("--center", required=True)
    f.add_argument("--plus", required=True)
    f.add_argument("--param", type=int, default=0)
    k = sub.add_parser("calib")
    k.add_argument("files", nargs="+")
    r = sub.add_parser("calib-rep")
    r.add_argument("prefixes", nargs=1)
    r.add_argument("--group", type=int, default=1, help="replicas per independent fit")
    r.add_argument("--nboot", type=int, default=100)
    for p in (c, f, r):
        add_arguments(p)
        p.add_argument("--skip", type=int, default=0, help="segments to drop at the start")
        p.add_argument("--json", default="")
        p.add_argument("--select", default="", help="indices of the run's parameters to use (e.g. 0,2)")
    a = ap.parse_args()
    if a.cmd == "calib":
        return cmd_calib(a)
    S = setup(a)
    {"combine": cmd_combine, "fd": cmd_fd, "calib-rep": cmd_calib_rep}[a.cmd](S, a)


if __name__ == "__main__":
    main()
