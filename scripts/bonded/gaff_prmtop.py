"""Write GAFF topologies of the bonded-study molecules (the initial values of the "amber" term set).

antechamber types (runs/bonded/pgm/<name>/mol.mol2, from scripts/bonded/pgm_params.py prep) ->
parmchk2 -> tleap -> runs/bonded/pgm/<name>/gaff.prmtop, read by pgm_jax.bonded.amber.init_from_prmtop.

Usage:

    python scripts/bonded/gaff_prmtop.py methanol ethanol ...      (AmberTools on the PATH)
    python scripts/bonded/gaff_prmtop.py --help

Inputs: runs/bonded/pgm/<name>/mol.mol2.
Outputs: runs/bonded/pgm/<name>/{mol.frcmod, leap.in, gaff.prmtop, gaff.inpcrd}; one printed line
per molecule.
Units: Amber's.
Runtime: seconds per molecule.
"""

from __future__ import annotations

import argparse
import os
import subprocess

from pgm_jax.paths import repo_path

LEAP = """source leaprc.gaff
loadamberparams mol.frcmod
m = loadmol2 mol.mol2
saveamberparm m gaff.prmtop gaff.inpcrd
quit
"""


def gaff_prmtop(name: str) -> str:
    """Run parmchk2 and tleap for one molecule and return the path of its gaff.prmtop."""
    wd = repo_path("runs", "bonded", "pgm", name)
    subprocess.run(["parmchk2", "-i", "mol.mol2", "-f", "mol2", "-o", "mol.frcmod", "-s", "gaff"], cwd=wd, check=True)
    with open(os.path.join(wd, "leap.in"), "w") as fh:
        fh.write(LEAP)
    subprocess.run(["tleap", "-f", "leap.in"], cwd=wd, check=True, stdout=subprocess.DEVNULL)
    return os.path.join(wd, "gaff.prmtop")


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and write the topologies (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="molecule names")
    for name in ap.parse_args(argv).names:
        path = gaff_prmtop(name)
        print(name, "->", path, os.path.exists(path))


if __name__ == "__main__":
    main()
