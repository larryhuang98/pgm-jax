"""Neural bonded terms vs the element-typed class II / class I terms: leave-one-molecule-out
transfer and per-molecule fits.  Reads runs/bonded/nnb/{loo,permol}_*.json and the bonded
study's runs/bonded/results/loo0_<mol>_{paper,diag}_pgm.json; prints markdown tables and writes
runs/bonded/nnb/summary.json.

    python scripts/bonded/nnb_summary.py
"""

import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NNB = os.path.join(ROOT, "runs/bonded/nnb")
RES = os.path.join(ROOT, "runs/bonded/results")


def merged(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(NNB, pattern))):
        out.update(json.load(open(f))["molecules"])
    return out


def baseline(mol, form):
    f = os.path.join(RES, f"loo0_{mol}_{form}_pgm.json")
    if not os.path.exists(f):
        return None
    t = json.load(open(f))["molecules"][mol]["test"]
    return t["E_MAE"], t["F_MAE"]


summary = {"loo": {}, "permol": {}}
RUNS = {
    "nnb_table10": "loo_geometry_vT3*.json",  # typed table + residual, resid_l2 10, geometry ref
    "nnb_table1": "loo_geometry_vT1*.json",  # typed table + residual, resid_l2 1
    "nnb_predicted": "loo_predicted_[ab].json",
}  # network only, predicted ref
LABELS = [
    "NNB, table + residual (shrinkage 10)",
    "NNB, table + residual (shrinkage 1)",
    "NNB, network only (predicted ref)",
    "class II, element-typed",
    "class I, element-typed",
]
loo = {k: merged(p) for k, p in RUNS.items()}
mols = list(dict.fromkeys(sum((list(v) for v in loo.values()), [])))
cols = {k: [] for k in list(RUNS) + ["class_ii", "class_i"]}
if mols:
    print("Leave one molecule out (held-out molecule, 298 K frames): energy MAE kcal/mol / force MAE kcal/mol/A\n")
    print("| held out | " + " | ".join(LABELS) + " |")
    print("|---" * (len(LABELS) + 1) + "|")
    for m in mols:
        row = {k: ((loo[k][m]["test"]["E_MAE"], loo[k][m]["test"]["F_MAE"]) if m in loo[k] else None) for k in RUNS}
        row["class_ii"] = baseline(m, "paper")
        row["class_i"] = baseline(m, "diag")
        summary["loo"][m] = row
        f = lambda v: "" if v is None else f"{v[0]:.2f} / {v[1]:.1f}"
        print(f"| {m} | " + " | ".join(f(row[k]) for k in cols) + " |")
    complete = [m for m in mols if all(summary["loo"][m][k] is not None for k in cols)]
    mean = {k: tuple(np.mean([summary["loo"][m][k] for m in complete], 0)) for k in cols} if complete else {}
    if mean:
        print(f"| mean ({len(complete)}) | " + " | ".join(f"{mean[k][0]:.2f} / {mean[k][1]:.1f}" for k in cols) + " |")
    summary["loo_mean"] = {k: list(v) for k, v in mean.items()}
    summary["loo_n"] = len(complete)

per = merged("permol_geometry_final.json")
if per:
    print("\nPer molecule (trained on 500 K frames of the molecule, tested on 298 K frames)\n")
    print("| molecule | NNB | class II (symmetry-typed) |")
    print("|---|---|---|")
    for m, r in per.items():
        summary["permol"][m] = {k: (v["E_MAE"], v["F_MAE"]) for k, v in r.items()}
        print(
            f"| {m} | {r['nnb']['E_MAE']:.2f} / {r['nnb']['F_MAE']:.2f} | {r['paper']['E_MAE']:.2f} / {r['paper']['F_MAE']:.2f} |"
        )
json.dump(summary, open(os.path.join(NNB, "summary.json"), "w"), indent=1)
