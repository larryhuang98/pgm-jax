"""Summarize the neural bonded terms against the element-typed class II / class I terms.

Leave-one-molecule-out transfer and per-molecule fits: reads runs/bonded/nnb/{loo,permol}_*.json
(scripts/bonded/nnb_experiments.py) and the bonded study's
runs/bonded/results/loo0_<mol>_{paper,diag}_pgm.json (scripts/bonded/experiments.py), prints
markdown tables and writes runs/bonded/nnb/summary.json.

Usage:

    python scripts/bonded/nnb_summary.py
    python scripts/bonded/nnb_summary.py --help

Inputs: the result files above.
Outputs: runs/bonded/nnb/summary.json, markdown tables on stdout.
Units: energy MAE [kcal/mol] / force MAE [kcal/mol/A].
Runtime: seconds.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from pgm_jax.paths import repo_path

#: result label -> file pattern below runs/bonded/nnb
RUNS = {
    "nnb_table10": "loo_geometry_vT3*.json",  # typed table + residual, resid_l2 10, geometry ref
    "nnb_table1": "loo_geometry_vT1*.json",  # typed table + residual, resid_l2 1
    "nnb_predicted": "loo_predicted_[ab].json",  # network only, predicted ref
}
LABELS = [
    "NNB, table + residual (shrinkage 10)",
    "NNB, table + residual (shrinkage 1)",
    "NNB, network only (predicted ref)",
    "class II, element-typed",
    "class I, element-typed",
]


def merged(nnb: str, pattern: str) -> dict:
    """Return the "molecules" records of all files matching pattern in nnb, merged (later files win)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(nnb, pattern))):
        with open(f) as fh:
            out.update(json.load(fh)["molecules"])
    return out


def baseline(res: str, mol: str, form: str) -> tuple[float, float] | None:
    """Return (energy MAE, force MAE) of the held-out molecule of loo0_<mol>_<form>_pgm.json, None if absent."""
    f = os.path.join(res, f"loo0_{mol}_{form}_pgm.json")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        t = json.load(fh)["molecules"][mol]["test"]
    return t["E_MAE"], t["F_MAE"]


def fmt(v: tuple[float, float] | None) -> str:
    """Return "E / F" of an (energy, force) MAE pair, "" for None."""
    return "" if v is None else f"{v[0]:.2f} / {v[1]:.1f}"


def main(argv: list[str] | None = None) -> None:
    """Print the tables and write the summary (see the module docstring)."""
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    nnb, res = repo_path("runs", "bonded", "nnb"), repo_path("runs", "bonded", "results")
    summary = {"loo": {}, "permol": {}}
    loo = {k: merged(nnb, p) for k, p in RUNS.items()}
    mols = list(dict.fromkeys(sum((list(v) for v in loo.values()), [])))
    cols = {k: [] for k in list(RUNS) + ["class_ii", "class_i"]}
    if mols:
        print("Leave one molecule out (held-out molecule, 298 K frames): energy MAE kcal/mol / force MAE kcal/mol/A\n")
        print("| held out | " + " | ".join(LABELS) + " |")
        print("|---" * (len(LABELS) + 1) + "|")
        for m in mols:
            row = {k: ((loo[k][m]["test"]["E_MAE"], loo[k][m]["test"]["F_MAE"]) if m in loo[k] else None) for k in RUNS}
            row["class_ii"] = baseline(res, m, "paper")
            row["class_i"] = baseline(res, m, "diag")
            summary["loo"][m] = row
            print(f"| {m} | " + " | ".join(fmt(row[k]) for k in cols) + " |")
        complete = [m for m in mols if all(summary["loo"][m][k] is not None for k in cols)]
        mean = {k: tuple(np.mean([summary["loo"][m][k] for m in complete], 0)) for k in cols} if complete else {}
        if mean:
            print(
                f"| mean ({len(complete)}) | " + " | ".join(f"{mean[k][0]:.2f} / {mean[k][1]:.1f}" for k in cols) + " |"
            )
        summary["loo_mean"] = {k: list(v) for k, v in mean.items()}
        summary["loo_n"] = len(complete)

    per = merged(nnb, "permol_geometry_final.json")
    if per:
        print("\nPer molecule (trained on 500 K frames of the molecule, tested on 298 K frames)\n")
        print("| molecule | NNB | class II (symmetry-typed) |")
        print("|---|---|---|")
        for m, r in per.items():
            summary["permol"][m] = {k: (v["E_MAE"], v["F_MAE"]) for k, v in r.items()}
            print(
                f"| {m} | {r['nnb']['E_MAE']:.2f} / {r['nnb']['F_MAE']:.2f} | {r['paper']['E_MAE']:.2f} / "
                f"{r['paper']['F_MAE']:.2f} |"
            )
    with open(os.path.join(nnb, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
