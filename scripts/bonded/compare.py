"""Print per-molecule and mean test errors of several result files of the bonded study.

For the molecules common to all files (methanethiol excluded): test energy and force MAEs, the
largest error of the relaxed torsion scans, the parameter counts and the dipole RMSE.

Usage:

    python scripts/bonded/compare.py NAME1 NAME2 ...          # runs/bonded/results/<NAME>.json
    python scripts/bonded/compare.py --help

Inputs: runs/bonded/results/<name>.json (scripts/bonded/experiments.py).
Outputs: the printed tables.
Units: kcal/mol, kcal/mol/A, D.
Runtime: seconds.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pgm_jax.paths import repo_path

RES = repo_path("runs", "bonded", "results")


def smax(v: dict) -> float:
    """Return the largest relaxed-scan error [kcal/mol] of a molecule's result (inf: the relaxation left the basin)."""
    xs = []
    for x in v["scans"].values():
        if "error" in x:
            continue
        xs.append(
            x["relaxed_max"]
            if "relaxed_max" in x
            else (np.inf if x.get("relax_ok") is False else x.get("sp_max", np.nan))
        )
    return max(xs) if xs else np.nan


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and print the tables (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="result names (runs/bonded/results/<name>.json)")
    names = ap.parse_args(argv).names
    excl = {"methanethiol"}
    D = {}
    for n in names:
        with open(os.path.join(RES, f"{n}.json")) as fh:
            D[n] = json.load(fh)["molecules"]
    common = [m for m in D[names[0]] if all(m in D[n] for n in names) and m not in excl]
    w = max(len(n) for n in names)
    print(f"{'':{w}s} " + " ".join(f"{m[:10]:>10s}" for m in common) + "      mean")
    for key in ("E_MAE", "F_MAE"):
        print(f"-- test {key} ({'kcal/mol' if key == 'E_MAE' else 'kcal/mol/A'})")
        for n in names:
            v = [D[n][m]["test"][key] for m in common]
            print(f"{n:{w}s} " + " ".join(f"{x:10.3f}" for x in v) + f"  {np.mean(v):8.3f}")
    if any(D[n][m]["scans"] for n in names for m in common):
        print("-- relaxed torsion scan, max |error| (kcal/mol; inf = the relaxation left the basin)")
        for n in names:
            v = [smax(D[n][m]) for m in common]
            fin = [x for x in v if np.isfinite(x)]
            print(
                f"{n:{w}s} "
                + " ".join(f"{x:10.2f}" for x in v)
                + f"  {np.mean(fin) if fin else np.nan:8.2f}  fails {sum(1 for x in v if np.isinf(x))}"
            )
    print("-- parameters (per group)")
    for n in names:
        print(f"{n:{w}s} " + " ".join(f"{D[n][m]['n_params_group']:10d}" for m in common))
    if any("mu_RMSE_D" in D[n][common[0]]["test"] for n in names):
        print("-- dipole RMSE (D)")
        for n in names:
            v = [D[n][m]["test"].get("mu_RMSE_D", np.nan) for m in common]
            print(f"{n:{w}s} " + " ".join(f"{x:10.3f}" for x in v) + f"  {np.nanmean(v):8.3f}")


if __name__ == "__main__":
    main()
