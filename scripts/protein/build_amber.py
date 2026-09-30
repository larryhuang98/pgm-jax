"""Build a solvated Amber topology from a PDB or a residue sequence with tleap (`pgm-jax build-amber`).

The result (topology, hydrogens, termini, water box and ions) is what pgm_jax.protein.load_amber
reads (docs/protein_ff.md).  The structure is cleaned here (first model, ATOM records only: no
waters or ligands, first alternate location, hydrogens removed so that tleap adds them with Amber's
names); tleap adds hydrogens and termini (HIS as HIE unless renamed), neutralises with Na+ / Cl-
(before solvating, so no ion lands on a water) and solvates in a rectangular TIP3P box (--box oct:
truncated octahedron; the water is replaced by the pGM water model when the system is loaded).
With --sequence, tleap's `sequence` builds the chain from its residue library (extended backbone,
Amber residue names including caps and terminal variants, e.g. "ACE ALA NME" or "NALA ALA CALA").

Usage:

    python scripts/protein/build_amber.py 1ubq.pdb runs/protein/ubq --buffer-A 10 [--model 1] [--ff ff19SB]
    python scripts/protein/build_amber.py --sequence "ACE ALA ALA ALA NME" runs/remd/ala3 --buffer-A 8
    python scripts/protein/build_amber.py --help

Inputs: the PDB file (or --sequence); tleap on PATH (AmberTools, AMBERHOME).
Outputs: <out>.prmtop, <out>.inpcrd, <out>.pdb, <out>_clean.pdb, <out>_leap.in, <out>_leap.log;
the last line of tleap's log is printed.
Units: --buffer-A in Angstrom (tleap's unit).
Runtime: seconds.
"""

from __future__ import annotations

import argparse
import os
import subprocess

BOXES = {"tip3p": "TIP3PBOX", "opc": "OPCBOX", "opc3": "OPC3BOX"}  # tleap solvent box per water model


def clean_pdb(src: str, dst: str, model: int = 1) -> None:
    """Write the ATOM records of one model of a PDB file without hydrogens and alternate locations.

    Parameters
    ----------
    src : str
        Input PDB.
    dst : str
        Output PDB (ATOM and TER records, END).
    model : int
        MODEL number to keep (files without MODEL records count as model 1).
    """
    out, cur, seen_model = [], 1, False
    with open(src) as fh:
        for ln in fh:
            rec = ln[:6].strip()
            if rec == "MODEL":
                cur = int(ln.split()[1])
                seen_model = True
                continue
            if rec == "ENDMDL" and seen_model and cur == model:
                break
            if cur != model or rec != "ATOM":
                if rec == "TER" and cur == model:
                    out.append("TER\n")
                continue
            alt, el = ln[16], (ln[76:78].strip() or ln[12:16].strip()[0])
            if alt not in (" ", "A") or el == "H" or ln[12:16].strip().startswith("H"):
                continue
            out.append(ln[:16] + " " + ln[17:])  # blank alternate-location column
    with open(dst, "w") as fh:
        fh.write("".join(out) + "END\n")


def leap_input(load: str, a: argparse.Namespace) -> str:
    """Return the tleap input: force field, water, the molecule (`load` command), ions, solvent, outputs."""
    return f"""source leaprc.protein.{a.ff}
source leaprc.water.{a.water}
{load}
addions m Na+ 0
addions m Cl- 0
{"solvateoct" if a.box == "oct" else "solvatebox"} m {BOXES[a.water]} {a.buffer_A}
saveamberparm m {a.out}.prmtop {a.out}.inpcrd
savepdb m {a.out}.pdb
quit
"""


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, clean the structure and run tleap (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", nargs="?", help="input PDB (omit with --sequence)")
    ap.add_argument("out", help="output prefix (directory created)")
    ap.add_argument("--sequence", default=None, help='residue names for tleap\'s sequence, e.g. "ACE ALA ALA ALA NME"')
    ap.add_argument("--buffer-A", type=float, default=10.0, help="solvent buffer [A]")
    ap.add_argument("--ff", default="ff19SB", help="protein force field (leaprc.protein.<ff>)")
    ap.add_argument("--water", default="tip3p", choices=sorted(BOXES), help="water model of the box")
    ap.add_argument("--model", type=int, default=1, help="MODEL of the PDB file")
    ap.add_argument("--box", default="rect", choices=("rect", "oct"), help="rectangular box or truncated octahedron")
    a = ap.parse_args(argv)
    if (a.pdb is None) == (a.sequence is None):
        ap.error("give either a PDB file or --sequence")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.sequence is None:
        clean = a.out + "_clean.pdb"
        clean_pdb(a.pdb, clean, a.model)
        load = f"m = loadpdb {clean}"
    else:
        load = "m = sequence { " + " ".join(a.sequence.replace(",", " ").split()) + " }"
    with open(a.out + "_leap.in", "w") as fh:
        fh.write(leap_input(load, a))
    with open(a.out + "_leap.log", "w") as log:
        subprocess.run(["tleap", "-f", a.out + "_leap.in"], check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(a.out + "_leap.log") as fh:
        print(fh.read().splitlines()[-1])


if __name__ == "__main__":
    main()
