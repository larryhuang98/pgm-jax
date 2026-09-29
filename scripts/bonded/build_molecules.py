"""Build the molecule set of the bonded study: data/bonded/molecules/<name>.json.

Each file holds the elements, bonds and RDKit conformers of one molecule of
pgm_jax.bonded.study.molecules.MOLECULES (docs/howto_bonded.md, data/reports/bonded/README.md).

Usage:

    python scripts/bonded/build_molecules.py
    python scripts/bonded/build_molecules.py --help

Inputs: none (RDKit builds the conformers).
Outputs: data/bonded/molecules/<name>.json; one printed line per molecule.
Units: Angstrom (conformers).
Runtime: seconds.
"""

from __future__ import annotations

import argparse
import json
import os

from pgm_jax.bonded.study.molecules import MOLECULES, build
from pgm_jax.paths import repo_path


def main(argv: list[str] | None = None) -> None:
    """Parse the (empty) command line and write every molecule (see the module docstring)."""
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    out = repo_path("data", "bonded", "molecules")
    os.makedirs(out, exist_ok=True)
    for name in MOLECULES:
        d = build(name)
        with open(os.path.join(out, f"{name}.json"), "w") as fh:
            json.dump(d, fh, indent=1)
        print(
            f"{name:20s} {d['subset']} q={d['charge']:+d} atoms {len(d['elements']):2d} bonds {len(d['bonds']):2d} "
            f"conformers {len(d['conformers'])}"
        )


if __name__ == "__main__":
    main()
