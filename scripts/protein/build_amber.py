"""PDB -> solvated Amber topology with tleap (the topology, hydrogens, termini, water box and ions
that pgm_jax.protein.load_amber reads).

    python scripts/protein/build_amber.py 1ubq.pdb runs/protein/ubq --buffer 10 [--model 1] [--ff ff19SB]

The structure is cleaned here (first model, ATOM records only: no waters or ligands, first
alternate location, hydrogens removed so that tleap adds them with Amber's names); tleap adds
hydrogens and termini (HIS as HIE unless renamed), neutralises with Na+ / Cl- (before solvating, so no ion lands on a
water) and solvates in a rectangular TIP3P box (the water is replaced by the pGM water model when
the system is loaded).  Needs AmberTools (AMBERHOME)."""
import argparse
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("pdb")
ap.add_argument("out", help="output prefix (directory created)")
ap.add_argument("--buffer", type=float, default=10.0, help="solvent buffer (A)")
ap.add_argument("--ff", default="ff19SB")
ap.add_argument("--water", default="tip3p")
ap.add_argument("--model", type=int, default=1)
a = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
clean = a.out + "_clean.pdb"


def clean_pdb(src, dst, model=1):
    out, cur, seen_model = [], 1, False
    for ln in open(src):
        rec = ln[:6].strip()
        if rec == "MODEL":
            cur = int(ln.split()[1]); seen_model = True
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


clean_pdb(a.pdb, clean, a.model)
box = {"tip3p": "TIP3PBOX", "opc": "OPCBOX", "opc3": "OPC3BOX"}[a.water]
leap = f"""source leaprc.protein.{a.ff}
source leaprc.water.{a.water}
m = loadpdb {clean}
addions m Na+ 0
addions m Cl- 0
solvatebox m {box} {a.buffer}
saveamberparm m {a.out}.prmtop {a.out}.inpcrd
savepdb m {a.out}.pdb
quit
"""
open(a.out + "_leap.in", "w").write(leap)
subprocess.run(["tleap", "-f", a.out + "_leap.in"], check=True, stdout=open(a.out + "_leap.log", "w"), stderr=subprocess.STDOUT)
print(open(a.out + "_leap.log").read().splitlines()[-1])
