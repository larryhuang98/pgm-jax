"""Print the leave-one-molecule-out table with element-level typing (depth 0): held-out test errors.

Per bonded family and held-out molecule: parameter coverage and test energy / force MAEs of the pGM,
classical and x13 electrostatics (runs/bonded/results/loo0_<molecule>_<family>_<elec>.json of
scripts/bonded/experiments.py), and their means.

Usage:

    python scripts/bonded/loo0_table.py
    python scripts/bonded/loo0_table.py --help

Inputs: runs/bonded/results/loo0_*.json.
Outputs: runs/bonded/results/loo0_summary.json and the printed tables.
Units: kcal/mol (E), kcal/mol/A (F).
Runtime: seconds.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from pgm_jax.paths import repo_path

RES = repo_path("runs", "bonded", "results")


def main(argv: list[str] | None = None) -> None:
    """Parse the (empty) command line, print the tables and write the summary (see the module docstring)."""
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    rows = {}
    for f in sorted(glob.glob(os.path.join(RES, "loo0_*.json"))):
        tag = os.path.basename(f)[5:-5]
        if tag == "summary":
            continue
        head, fam, el = tag.rsplit("_", 2)
        with open(f) as fh:
            d = json.load(fh)
        r = d["molecules"][head]
        rows.setdefault(fam, {}).setdefault(head, {})[el] = (
            r["test"]["E_MAE"],
            r["test"]["F_MAE"],
            r.get("coverage", np.nan),
        )
    out = {}
    for fam, mols in rows.items():
        print(f"== family {fam}: held-out molecule, element-typed parameters fitted on the other 11")
        els = [e for e in ("pgm", "cls", "x13") if any(e in v for v in mols.values())]
        print(
            f"{'held out':20s} {'coverage':>8s} "
            + " ".join(f"{'E ' + e:>8s}" for e in els)
            + " "
            + " ".join(f"{'F ' + e:>8s}" for e in els)
        )
        E = {e: [] for e in els}
        F = {e: [] for e in els}
        for mol, v in mols.items():
            if not all(e in v for e in els):
                continue
            cov = next(v[e][2] for e in els)
            print(
                f"{mol:20s} {cov:8.2f} "
                + " ".join(f"{v[e][0]:8.3f}" for e in els)
                + " "
                + " ".join(f"{v[e][1]:8.2f}" for e in els)
            )
            for e in els:
                E[e].append(v[e][0])
                F[e].append(v[e][1])
        if E[els[0]]:
            print(
                f"{'mean':20s} {'':8s} "
                + " ".join(f"{np.mean(E[e]):8.3f}" for e in els)
                + " "
                + " ".join(f"{np.mean(F[e]):8.2f}" for e in els)
                + f"  (n={len(E[els[0]])})"
            )
        out[fam] = {"molecules": mols}
    with open(os.path.join(RES, "loo0_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
