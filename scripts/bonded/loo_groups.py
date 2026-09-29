"""Print the leave-one-out transfer by chemical group: energy and force MAE per bonded family.

Groups of the 12 molecules (carbonyl/carboxyl, amine/ammonium/phosphate, other) and all 12, for the
pGM, classical and x13 electrostatics (runs/bonded/results/loo0_*.json of scripts/bonded/experiments.py).

Usage:

    python scripts/bonded/loo_groups.py
    python scripts/bonded/loo_groups.py --help

Inputs: runs/bonded/results/loo0_*.json.
Outputs: runs/bonded/results/loo_groups.json and the printed table.
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
G = {
    "carbonyl/carboxyl": ["acetaldehyde", "acetate", "formic_acid", "chloroformic_acid", "formamide"],
    "amine/ammonium/phosphate": ["methylamine", "methylammonium", "hydrogen_phosphate"],
    "other (alkane, alcohol, halides)": ["ethane", "methanol", "chloromethanol", "fluorochloroethane"],
}  # chemical groups of the held-out molecules


def main(argv: list[str] | None = None) -> None:
    """Parse the (empty) command line, print the table and write loo_groups.json (see the module docstring)."""
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    rows = {}
    for f in glob.glob(os.path.join(RES, "loo0_*.json")):
        tag = os.path.basename(f)[5:-5]
        if tag == "summary":
            continue
        head, fam, el = tag.rsplit("_", 2)
        with open(f) as fh:
            r = json.load(fh)["molecules"][head]
        rows.setdefault(fam, {}).setdefault(el, {})[head] = (r["test"]["E_MAE"], r["test"]["F_MAE"])
    out = {}
    for fam in (
        "diag",
        "diag+b1",
        "diag+b1e10",
        "diag+q1",
        "diag+q1e",
        "diag+q1e10",
        "diag+es",
        "diag+es14",
        "diag+ub",
        "diag+p14",
        "paper",
        "diag+conj",
        "diag+hc",
        "diag+new",
        "hyb",
        "hybsc",
        "chem",
        "chem+hyb",
        "dist",
        "dist+chem",
        "diag+ovl",
    ):
        if fam not in rows:
            continue
        print(f"== {fam}")
        for g, mols in list(G.items()) + [("all 12", sum(G.values(), []))]:
            line = f"  {g:34s}"
            for el in ("pgm", "cls", "x13"):
                v = [rows[fam].get(el, {}).get(m) for m in mols]
                if all(x is not None for x in v):
                    line += f"  {el} E {np.mean([x[0] for x in v]):5.2f} F {np.mean([x[1] for x in v]):5.1f}"
                    out.setdefault(fam, {}).setdefault(g, {})[el] = [
                        float(np.mean([x[0] for x in v])),
                        float(np.mean([x[1] for x in v])),
                    ]
            print(line)
    with open(os.path.join(RES, "loo_groups.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
