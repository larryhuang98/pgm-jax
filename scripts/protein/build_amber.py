"""PDB (or a residue sequence) -> solvated Amber topology with tleap (the topology, hydrogens,
termini, water box and ions that pgm_jax.protein.load_amber reads).

    python scripts/protein/build_amber.py 1ubq.pdb runs/protein/ubq --buffer 10 [--model 1] [--ff ff19SB]
    python scripts/protein/build_amber.py --sequence "ACE ALA ALA ALA NME" runs/remd/ala3 --buffer 8

The structure is cleaned here (first model, ATOM records only: no waters or ligands, first
alternate location, hydrogens removed so that tleap adds them with Amber's names); tleap adds
hydrogens and termini (HIS as HIE unless renamed), neutralises with Na+ / Cl- (before solvating,
so no ion lands on a water) and solvates in a rectangular TIP3P box (--box oct: truncated
octahedron; the water is replaced by the pGM water model when the system is loaded).
With --sequence, tleap's `sequence` builds the chain from its residue library (extended backbone,
Amber residue names including caps and terminal variants, e.g. "ACE ALA NME" or "NALA ALA CALA").
Needs AmberTools (AMBERHOME)."""

import argparse
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("pdb", nargs="?", help="input PDB (omit with --sequence)")
ap.add_argument("out", help="output prefix (directory created)")
ap.add_argument("--sequence", default=None, help='residue names for tleap\'s sequence, e.g. "ACE ALA ALA ALA NME"')
ap.add_argument("--buffer", type=float, default=10.0, help="solvent buffer (A)")
ap.add_argument("--ff", default="ff19SB")
ap.add_argument("--water", default="tip3p")
ap.add_argument("--model", type=int, default=1)
ap.add_argument("--box", default="rect", choices=("rect", "oct"), help="rectangular box or truncated octahedron")
a = ap.parse_args()
if (a.pdb is None) == (a.sequence is None):
    ap.error("give either a PDB file or --sequence")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
clean = a.out + "_clean.pdb"


def clean_pdb(src, dst, model=1):
    out, cur, seen_model = [], 1, False
    for ln in open(src):
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
        out.append(ln[:16] + " " + ln[17:])
    open(dst, "w").write("".join(out) + "END\n")


if a.sequence is None:
    clean_pdb(a.pdb, clean, a.model)
    load = f"m = loadpdb {clean}"
else:
    load = "m = sequence { " + " ".join(a.sequence.replace(",", " ").split()) + " }"
box = {"tip3p": "TIP3PBOX", "opc": "OPCBOX", "opc3": "OPC3BOX"}[a.water]
leap = f"""source leaprc.protein.{a.ff}
source leaprc.water.{a.water}
{load}
addions m Na+ 0
addions m Cl- 0
{"solvateoct" if a.box == "oct" else "solvatebox"} m {box} {a.buffer}
saveamberparm m {a.out}.prmtop {a.out}.inpcrd
savepdb m {a.out}.pdb
quit
"""
open(a.out + "_leap.in", "w").write(leap)
subprocess.run(
    ["tleap", "-f", a.out + "_leap.in"], check=True, stdout=open(a.out + "_leap.log", "w"), stderr=subprocess.STDOUT
)
print(open(a.out + "_leap.log").read().splitlines()[-1])
