"""Analysis of liquid-fit runs (scripts/fitting/fit_multi.py outputs; the `pgm-jax analyze-fit` command).

Subcommands: combine (observables, Jacobians and jackknife errors from the saved frames of the
segments of one run at fixed theta), fd (finite differences between independent runs at
theta -/+ delta e_j against the fluctuation-formula gradient, averaged with Simpson's rule over
the three runs), calib (spread of the fitted parameters over independent fits against the
predicted sampling errors), calib-rep (the same with independent fits from groups of replicas of
one batched run).  The same --params / --targets / --model options as fit_multi.py define the
objective (docs/liquid_fit.md).

Usage:

    python scripts/fitting/liquid_fit_tools.py combine runs/fit/c0 --params q --targets density,hvap,eps,liquid_dipole
    python scripts/fitting/liquid_fit_tools.py fd --minus runs/fit/m --center runs/fit/c0 --plus runs/fit/p
        --param 0 --params q --targets density,hvap,eps,liquid_dipole
    python scripts/fitting/liquid_fit_tools.py calib runs/fit/cal_s*.json
    python scripts/fitting/liquid_fit_tools.py calib-rep runs/fit/rep --group 4 --params q --targets ...
    python scripts/fitting/liquid_fit_tools.py combine --help

Inputs: <prefix>_frames*.npz of the runs (and their JSON files for calib).
Outputs: printed tables; with --json the rows (combine: <prefix>_combined.json).
Units: those of pgm_jax.fit (theta: ln scales; observables in the target units).
Runtime: seconds to minutes (CPU).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import glob
import json

import jax
import numpy as np
from fit_multi import add_arguments, setup
from scipy.stats import chi2

from pgm_jax.fit import LiquidSamples

jax.config.update("jax_enable_x64", True)
GRAD_KEYS = ("dU", "dM", "dalpha", "dD")


def load_frames(prefix: str, skip: int = 0, select: list[int] | None = None) -> tuple[dict, np.ndarray]:
    """Return the concatenated frames of prefix_frames*.npz (segments in order) and their theta.

    Parameters
    ----------
    prefix : str
        Output prefix of the run.
    skip : int
        Segments dropped at the start.
    select : list of int, optional
        Indices of the parameters to keep (the frames of a run carry derivatives for all of its own).

    Returns
    -------
    frames : dict
        Arrays by key, frames concatenated (batched replicas ordered by replica, then time).
    theta : np.ndarray
        Parameters of the run.

    Raises
    ------
    SystemExit
        No frames, or segments at different theta.
    """
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
    if "rep" in parts[0]:  # batched replicas: order by replica, then time (contiguous blocks)
        rep = np.concatenate([p["rep"] for p in parts])
        order = np.argsort(rep, kind="stable")
        out = {k: v[order] for k, v in out.items()}
        out["rep"] = rep[order]
    if select is not None:
        th = th[select]
        for k in GRAD_KEYS:
            out[k] = out[k][..., select]
    return out, th


def _select(a: argparse.Namespace) -> list[int] | None:
    """Return the parameter indices of --select (None: all)."""
    return None if not getattr(a, "select", "") else [int(x) for x in a.select.split(",")]


def estimate(S: dict, a: argparse.Namespace, prefix: str) -> tuple[object, LiquidSamples]:
    """Return the objective's estimate (observables, Jacobian, errors) of a run and its samples."""
    fr, th = load_frames(prefix, a.skip, _select(a))
    fr.pop("rep", None)
    s = LiquidSamples(fr, a.temperature_K, S["sys"].nmol, float(np.sum(S["sys"].masses)), a.nblocks)
    return S["obj"].estimate(s, th), s


def cmd_combine(S: dict, a: argparse.Namespace) -> None:
    """Print (and with --json write) the estimates of each run (`combine`)."""
    for pre in a.prefixes:
        est, s = estimate(S, a, pre)
        print(f"# {pre}: {s.F} frames, theta {np.round(est.theta, 5).tolist()}")
        for i, n in enumerate(est.names):
            print(
                f"   {n:20s} {est.y[i]:12.5f} +- {est.err[i]:9.5f}   d/dtheta "
                + " ".join(f"{j:11.4f} +- {e:8.4f}" for j, e in zip(est.J[i], est.J_err[i]))
            )
        if a.json:
            with open(pre + "_combined.json", "w") as fh:
                json.dump(est.as_dict(), fh, indent=1)


def cmd_fd(S: dict, a: argparse.Namespace) -> None:
    """Compare finite differences between the runs at theta -/+ delta with the gradients (`fd`).

    The finite difference (y+ - y-) / (theta+ - theta-) is the mean slope over the interval; it is
    compared with Simpson's average (g- + 4 g0 + g+) / 6 of the fluctuation-formula gradients of
    the three runs (exact for cubic y), as z = difference / combined error.
    """
    em, sm = estimate(S, a, a.minus)
    ec, sc = estimate(S, a, a.center)
    ep, sp = estimate(S, a, a.plus)
    j = a.param
    dth = ep.theta[j] - em.theta[j]
    print(
        f"# finite differences along {S['space'].names[j]}: theta {em.theta[j]:+.4f} / {ec.theta[j]:+.4f} / "
        f"{ep.theta[j]:+.4f}; frames {sm.F} / {sc.F} / {sp.F}"
    )
    rows = []
    for i, n in enumerate(ec.names):
        fd = (ep.y[i] - em.y[i]) / dth
        fd_err = np.hypot(ep.err[i], em.err[i]) / abs(dth)
        g = [em.J[i, j], ec.J[i, j], ep.J[i, j]]
        ge = [em.J_err[i, j], ec.J_err[i, j], ep.J_err[i, j]]
        simpson = (g[0] + 4 * g[1] + g[2]) / 6.0  # mean slope over [-d, +d] (exact for cubics)
        simpson_err = np.sqrt(ge[0] ** 2 + 16 * ge[1] ** 2 + ge[2] ** 2) / 6.0
        z = (fd - simpson) / np.hypot(fd_err, simpson_err)
        rows.append(
            {
                "name": n,
                "y": [em.y[i], ec.y[i], ep.y[i]],
                "y_err": [em.err[i], ec.err[i], ep.err[i]],
                "fd": fd,
                "fd_err": fd_err,
                "grad": g,
                "grad_err": ge,
                "simpson": simpson,
                "simpson_err": simpson_err,
                "z": z,
            }
        )
        print(
            f"   {n:16s} y {em.y[i]:10.4f} {ec.y[i]:10.4f} {ep.y[i]:10.4f} (+- {ec.err[i]:.4f})  FD {fd:11.4f} +- "
            f"{fd_err:9.4f}  "
            f"gradient {g[0]:11.4f} {g[1]:11.4f} {g[2]:11.4f} (+- {ge[1]:.4f})  Simpson {simpson:11.4f} +- "
            f"{simpson_err:9.4f}  z {z:+.2f}"
        )
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1, default=float)


def cmd_calib(a: argparse.Namespace) -> None:
    """Print the spread of the fitted parameters over independent fits against their predicted errors (`calib`)."""
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
        lo, hi = sd[k] * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1)), sd[k] * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1))
        print(
            f"   {nm:12s} mean {th[:, k].mean():+.5f}  spread {sd[k]:.5f} (95 % {lo:.5f}-{hi:.5f})  "
            f"predicted (jackknife) {np.sqrt(np.mean(err[:, k] ** 2)):.5f}  bootstrap "
            f"{np.sqrt(np.nanmean(errb[:, k] ** 2)):.5f}"
        )


def cmd_calib_rep(S: dict, a: argparse.Namespace) -> None:
    """Fit groups of replicas of one run independently and compare the spreads with the predicted errors.

    Spread of the fitted parameters and of the observables over the groups against the predicted
    (jackknife + propagation, bootstrap) errors (`calib-rep`).
    """
    fr, th = load_frames(a.prefixes[0], a.skip, _select(a))
    rep = fr.pop("rep")
    R = int(rep.max()) + 1
    g = a.group
    obj = S["obj"]
    fits, errs, errb, ys, yerr = [], [], [], [], []
    for k in range(R // g):
        sel = (rep >= k * g) & (rep < (k + 1) * g)
        s = LiquidSamples(
            {key: v[sel] for key, v in fr.items()},
            a.temperature_K,
            S["sys"].nmol,
            float(np.sum(S["sys"].masses)),
            a.nblocks,
        )
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
    print(
        f"# {n} independent fits of {g} replicas each ({R} replicas, {len(rep)} frames); targets "
        + ", ".join(f"{t.name}={t.value}" for t in obj.targets if t.fit)
    )
    rows = []
    for j, nm in enumerate(S["space"].names):
        sd = fits[:, j].std(ddof=1)
        lo, hi = sd * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1)), sd * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1))
        pj, pb = np.sqrt(np.mean(errs[:, j] ** 2)), np.sqrt(np.mean(errb[:, j] ** 2))
        rows.append(
            {
                "param": nm,
                "spread": sd,
                "spread_95": [lo, hi],
                "predicted": pj,
                "bootstrap": pb,
                "mean": fits[:, j].mean(),
            }
        )
        print(
            f"   {nm:14s} fitted {fits[:, j].mean():+.5f}  spread {sd:.5f} (95 % {lo:.5f}-{hi:.5f})  predicted "
            f"{pj:.5f}  bootstrap {pb:.5f}"
        )
    names = obj.layout()[0]
    for i, nm in enumerate(names):
        sd = ys[:, i].std(ddof=1)
        lo, hi = sd * np.sqrt((n - 1) / chi2.ppf(0.975, n - 1)), sd * np.sqrt((n - 1) / chi2.ppf(0.025, n - 1))
        pe = np.sqrt(np.mean(yerr[:, i] ** 2))
        rows.append({"observable": nm, "spread": sd, "spread_95": [lo, hi], "jackknife": pe, "mean": ys[:, i].mean()})
        print(f"   {nm:14s} mean {ys[:, i].mean():10.4f}  spread {sd:.4f} (95 % {lo:.4f}-{hi:.4f})  jackknife {pe:.4f}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1, default=float)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser with the subcommands combine, fd, calib and calib-rep."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combine", help="estimates from the frames of runs")
    c.add_argument("prefixes", nargs="+", help="output prefixes of the runs")
    f = sub.add_parser("fd", help="finite differences between runs vs the gradient")
    f.add_argument("--minus", required=True, help="prefix of the run at theta - delta e_j")
    f.add_argument("--center", required=True, help="prefix of the run at theta")
    f.add_argument("--plus", required=True, help="prefix of the run at theta + delta e_j")
    f.add_argument("--param", type=int, default=0, help="index j of the varied parameter")
    k = sub.add_parser("calib", help="spread of independent fits vs predicted errors")
    k.add_argument("files", nargs="+", help="fit JSON files")
    r = sub.add_parser("calib-rep", help="independent fits from groups of replicas")
    r.add_argument("prefixes", nargs=1, help="output prefix of the batched run")
    r.add_argument("--group", type=int, default=1, help="replicas per independent fit")
    r.add_argument("--nboot", type=int, default=100, help="bootstrap samples")
    for p in (c, f, r):
        add_arguments(p)
        p.add_argument("--skip", type=int, default=0, help="segments to drop at the start")
        p.add_argument("--json", default="", help="write the results to this JSON file")
        p.add_argument("--select", default="", help="indices of the run's parameters to use (e.g. 0,2)")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the subcommand (see the module docstring)."""
    a = build_parser().parse_args(argv)
    if a.cmd == "calib":
        return cmd_calib(a)
    S = setup(a)
    {"combine": cmd_combine, "fd": cmd_fd, "calib-rep": cmd_calib_rep}[a.cmd](S, a)


if __name__ == "__main__":
    main()
