#!/usr/bin/env python
"""Regression harness: record golden outputs of the cases in regression_cases.py, or check the
current code against them.

    python tests/regression/regress.py list
    python tests/regression/regress.py record [--only a,b] [--group a]     # writes golden/<case>.npz
    python tests/regression/regress.py check  [--only ...] [--rtol 0 --atol 0] [--out report.json]

pgm_jax must be importable (pip install -e ., or PYTHONPATH=<repository>).  Run on the CPU
(JAX_PLATFORMS=cpu) with a fixed thread count; the golden files were recorded with
16 threads (OMP_NUM_THREADS=16) on the cpu-short nodes of rayl8, see golden/meta_<group>.json.
check compares every array bitwise (NaNs equal) by default; with --rtol / --atol it accepts
|a - b| <= atol + rtol |b| and reports the largest deviations.  Exit status 1 on any mismatch,
missing key or failed case.  Keys that exist only in the new output are reported, not failed."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback

import jax
import numpy as np
import regression_cases as C

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # the repository (git commit of the recording)
GOLDEN = os.path.join(HERE, "golden")


def _select(only: str | None, group: str | None) -> list[str]:
    names = list(C.CASES)
    if only:
        want = only.split(",")
        bad = [w for w in want if w not in C.CASES]
        if bad:
            raise SystemExit(f"unknown cases: {bad}; `list` shows them")
        names = want
    if group:
        names = [n for n in names if C.CASES[n]["group"] in group.split(",")]
    return names


def _environment() -> dict:
    import jaxlib

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    cpu = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "cpu": cpu,
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "devices": [str(d) for d in jax.devices()],
        "commit": commit,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _run(name: str) -> tuple[dict, float]:
    t0 = time.time()
    out = C.CASES[name]["fn"]()
    clean = {}
    for k, v in out.items():
        a = np.asarray(v)
        if a.dtype == object:
            raise TypeError(f"{name}: {k} is not a numeric / string array")
        clean[k] = a
    return clean, time.time() - t0


def _compare(new: dict, old: dict, rtol: float, atol: float) -> dict:
    rep = {"missing": sorted(set(old) - set(new)), "extra": sorted(set(new) - set(old)), "diff": {}}
    for k in sorted(set(old) & set(new)):
        a, b = new[k], old[k]
        if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
            rep["diff"][k] = {"reason": f"shape/dtype {a.shape} {a.dtype} vs golden {b.shape} {b.dtype}"}
            continue
        if a.dtype.kind in "fc":
            if np.array_equal(a, b, equal_nan=True):
                continue
            d = np.abs(a - b)
            scale = np.abs(b)
            ok = bool(np.all((d <= atol + rtol * scale) | (np.isnan(a) & np.isnan(b))))
            rel = float(np.nanmax(d / np.where(scale > 0, scale, 1.0))) if d.size else 0.0
            rep["diff"][k] = {"max_abs": float(np.nanmax(d)) if d.size else 0.0, "max_rel": rel, "within_tol": ok}
        elif not np.array_equal(a, b):
            rep["diff"][k] = {"reason": "values differ", "within_tol": False}
    rep["ok"] = not rep["missing"] and all(v.get("within_tol", False) for v in rep["diff"].values())
    rep["bitwise"] = not rep["missing"] and not rep["diff"]
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["list", "record", "check"])
    ap.add_argument("--only", help="comma-separated case names")
    ap.add_argument("--group", help="comma-separated case groups (a..f)")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--golden", default=GOLDEN, help="directory of the golden files")
    ap.add_argument("--out", help="write the check report (JSON) here")
    a = ap.parse_args()
    names = _select(a.only, a.group)
    if a.action == "list":
        for n in names:
            c = C.CASES[n]
            print(f"{n:24s} group {c['group']}  {'(needs ' + c['needs'] + ') ' if c['needs'] else ''}{c['doc']}")
        return
    os.makedirs(a.golden, exist_ok=True)
    env = _environment()
    print(f"# {a.action}: {len(names)} cases; {json.dumps(env)}", flush=True)
    report, failed = {"environment": env, "cases": {}}, False
    for n in names:
        if not C.available(n):
            print(f"{n:24s} SKIPPED (needs {C.CASES[n]['needs']})", flush=True)
            report["cases"][n] = {"skipped": True}
            continue
        try:
            out, dt = _run(n)
        except Exception:  # noqa: BLE001
            failed = True
            tb = traceback.format_exc()
            print(f"{n:24s} ERROR\n{tb}", flush=True)
            report["cases"][n] = {"error": tb}
            continue
        path = os.path.join(a.golden, f"{n}.npz")
        if a.action == "record":
            np.savez_compressed(path, **out)
            with open(os.path.join(a.golden, f"{n}.json"), "w") as fh:
                json.dump({"case": n, "seconds": round(dt, 1), "keys": len(out), "environment": env}, fh, indent=1)
            print(f"{n:24s} recorded {len(out)} arrays in {dt:.1f} s", flush=True)
            continue
        if not os.path.exists(path):
            failed = True
            print(f"{n:24s} NO GOLDEN FILE", flush=True)
            report["cases"][n] = {"error": "no golden file"}
            continue
        with np.load(path) as g:
            old = {k: g[k] for k in g.files}
        rep = _compare(out, old, a.rtol, a.atol)
        rep["seconds"] = round(dt, 1)
        report["cases"][n] = rep
        failed |= not rep["ok"]
        status = "BITWISE" if rep["bitwise"] else ("WITHIN TOL" if rep["ok"] else "FAILED")
        print(
            f"{n:24s} {status} ({len(old)} arrays, {len(rep['diff'])} differ, {len(rep['missing'])} missing, "
            f"{len(rep['extra'])} new) {dt:.1f} s",
            flush=True,
        )
        for k, v in list(rep["diff"].items())[:12]:
            print(f"    {k}: {v}", flush=True)
        for k in rep["missing"][:12]:
            print(f"    missing: {k}", flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
