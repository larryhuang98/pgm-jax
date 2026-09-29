"""Finite-difference check of free-energy parameter gradients with independent runs.

    python scripts/fe_gradient_check.py --group charge runs/fg/wq090_fe.npz runs/fg/wq100_fe.npz runs/fg/wq110_fe.npz

Each run (solvation_free_energy.py run --grad --solute-scale GROUP=s) gives the hydration free energy
G(s) (MBAR) and its gradient dG/ds (fe_grad.gradient_estimate, MBAR-weighted and end-state
estimators, block jackknife errors).  The runs are independent, so G(s_j) - G(s_i) is compared with
the integral of the gradient between them: the trapezoid rule over neighbouring runs and, for three
equally spaced runs, Simpson's rule (exact for a cubic G(s)) and the central value
g(s_mid) (the central difference, truncation error delta^2 G'''/6).  Values in kcal/mol."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgm_jax.analysis import free_energy as fe  # noqa: E402
from pgm_jax.md import fe_grad as fg  # noqa: E402
from pgm_jax.units import KCAL


def point(path, group, discard_ps, n_blocks, solute=True):
    d = fe.load(path)
    meta = d["meta"]
    s = float(meta.get("solute_scale", {}).get(group, 1.0))
    gas = {"delta_g": meta["gas_delta_g"], "grad": meta.get("gas_grad")} if "gas_delta_g" in meta else None
    r = fg.gradient_estimate(d, discard_ps=discard_ps, gas=gas, n_blocks=n_blocks)
    leg = "hyd" if "hyd" in r else "solv"
    space = fg.ParamSpace.from_names(r["names"])
    v = space.scale_direction(np.asarray(meta["params_flat"], float), group, solute)
    out = {"path": path, "s": s, "samples_per_window": r["samples_per_window"], "leg": leg}
    m = r[leg]["mbar"]
    out["G"], out["G_err"] = m.value / KCAL, m.value_err / KCAL
    g = fe.estimate(
        d, discard_ps=discard_ps, gas=None if gas is None else {"delta_g": gas["delta_g"], "dudl": meta["gas_dudl"]}
    )
    out["G_err_mbar_asymptotic"] = (g["dG_hyd_mbar_err"] if gas is not None else g["mbar_err"]) / KCAL
    for est in ("mbar", "end"):
        a, b = r[leg][est].project(v)  # dG/d ln s
        out[f"g_{est}"], out[f"g_{est}_err"] = a / s / KCAL, b / s / KCAL
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", help="runs at increasing scales of the group")
    ap.add_argument("--group", default="charge", choices=sorted(fg.SCALE_GROUPS))
    ap.add_argument("--environment", action="store_true", help="the group's environment parameters, not the solute's")
    ap.add_argument("--discard-ps", type=float, default=200.0)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--json")
    a = ap.parse_args()
    pts = sorted((point(p, a.group, a.discard_ps, a.blocks, not a.environment) for p in a.npz), key=lambda x: x["s"])
    print(
        f"# group {a.group} ({'environment' if a.environment else 'solute'}), discard {a.discard_ps} ps, "
        f"{a.blocks} jackknife blocks; kcal/mol"
    )
    print("#     s    samples     G (MBAR)          [asympt.]   dG/ds MBAR-weighted    dG/ds end states")
    for p in pts:
        print(
            f"  {p['s']:6.3f} {p['samples_per_window']:7d}  {p['G']:9.4f} +- {p['G_err']:.4f} "
            f"[{p['G_err_mbar_asymptotic']:.4f}]"
            f"  {p['g_mbar']:9.3f} +- {p['g_mbar_err']:.3f}   {p['g_end']:9.3f} +- {p['g_end_err']:.3f}"
        )
    res = {"points": pts, "pairs": []}
    print("# neighbouring runs: finite difference vs trapezoid of the gradients (z = difference / combined error)")
    for p, q in zip(pts[:-1], pts[1:]):
        ds = q["s"] - p["s"]
        fd, fde = (q["G"] - p["G"]) / ds, math.hypot(q["G_err"], p["G_err"]) / ds
        row = {"s": [p["s"], q["s"]], "fd": fd, "fd_err": fde}
        line = f"  {p['s']:.3f} -> {q['s']:.3f}: FD {fd:9.3f} +- {fde:.3f}"
        for est in ("mbar", "end"):
            tr = 0.5 * (p[f"g_{est}"] + q[f"g_{est}"])
            te = 0.5 * math.hypot(p[f"g_{est}_err"], q[f"g_{est}_err"])
            z = (fd - tr) / math.hypot(fde, te)
            row[est] = [tr, te, z]
            line += f" | {est} {tr:9.3f} +- {te:.3f} (z {z:+.2f})"
        res["pairs"].append(row)
        print(line)
    if len(pts) == 3 and abs((pts[1]["s"] - pts[0]["s"]) - (pts[2]["s"] - pts[1]["s"])) < 1e-9:
        p0, p1, p2 = pts
        ds = p2["s"] - p0["s"]
        fd, fde = (p2["G"] - p0["G"]) / ds, math.hypot(p2["G_err"], p0["G_err"]) / ds
        print(f"# outer runs {p0['s']:.3f} -> {p2['s']:.3f}: FD {fd:.3f} +- {fde:.3f}")
        res["outer"] = {"fd": fd, "fd_err": fde}
        for est in ("mbar", "end"):
            g = [p[f"g_{est}"] for p in pts]
            e = [p[f"g_{est}_err"] for p in pts]
            simp = (g[0] + 4 * g[1] + g[2]) / 6
            simpe = math.sqrt(e[0] ** 2 + 16 * e[1] ** 2 + e[2] ** 2) / 6
            zc = (fd - g[1]) / math.hypot(fde, e[1])
            zs = (fd - simp) / math.hypot(fde, simpe)
            res["outer"][est] = {"central": [g[1], e[1], zc], "simpson": [simp, simpe, zs]}
            print(
                f"  {est:5s}: gradient at the centre {g[1]:.3f} +- {e[1]:.3f} (z {zc:+.2f}); Simpson {simp:.3f} +- "
                f"{simpe:.3f} (z {zs:+.2f})"
            )
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
