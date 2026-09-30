"""Print the leave-one-molecule-out transfer table (typed parameters): held-out test errors, pGM vs classical.

Reads runs/bonded/results/loo_<molecule>_<pgm|cls>.json of scripts/bonded/experiments.py and prints
per held-out molecule the parameter coverage and the test energy and force MAEs of the pGM and the
classical (point-charge) electrostatics, and their means.

Usage:

    python scripts/bonded/loo_table.py
    python scripts/bonded/loo_table.py --help

Inputs: runs/bonded/results/loo_*.json.
Outputs: the printed table.
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
    """Parse the (empty) command line and print the table (see the module docstring)."""
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    rows = {}
    for f in sorted(glob.glob(os.path.join(RES, "loo_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        tag = os.path.basename(f)[4:-5]
        mol, el = tag.rsplit("_", 1)
        r = d["molecules"][mol]
        rows.setdefault(mol, {})[el] = (r["test"]["E_MAE"], r["test"]["F_MAE"], r.get("coverage", np.nan))
    print(f"{'held out':20s} {'coverage':>8s} {'E pGM':>7s} {'E cls':>7s} {'F pGM':>7s} {'F cls':>7s}")
    E = {"pgm": [], "cls": []}
    F = {"pgm": [], "cls": []}
    for mol, v in rows.items():
        if "pgm" not in v or "cls" not in v:
            continue
        print(
            f"{mol:20s} {v['pgm'][2]:8.2f} {v['pgm'][0]:7.3f} {v['cls'][0]:7.3f} {v['pgm'][1]:7.2f} {v['cls'][1]:7.2f}"
        )
        for k in ("pgm", "cls"):
            E[k].append(v[k][0])
            F[k].append(v[k][1])
    print(
        f"{'mean':20s} {'':8s} {np.mean(E['pgm']):7.3f} {np.mean(E['cls']):7.3f} {np.mean(F['pgm']):7.2f} "
        f"{np.mean(F['cls']):7.2f}"
    )


if __name__ == "__main__":
    main()
