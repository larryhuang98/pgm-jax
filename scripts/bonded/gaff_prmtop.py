"""GAFF bonded parameters for the bonded-study molecules: antechamber types (runs/bonded/pgm/<name>/
mol.mol2, from pgm_params.py prep) -> parmchk2 -> tleap -> runs/bonded/pgm/<name>/gaff.prmtop, the
initial values of the "amber" term set (pgm_jax.bonded.amber.init_from_prmtop).
    python scripts/bonded/gaff_prmtop.py methanol ethanol ...      (AmberTools on the PATH)"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEAP = """source leaprc.gaff
loadamberparams mol.frcmod
m = loadmol2 mol.mol2
saveamberparm m gaff.prmtop gaff.inpcrd
quit
"""
for name in sys.argv[1:]:
    wd = os.path.join(ROOT, "runs/bonded/pgm", name)
    subprocess.run(["parmchk2", "-i", "mol.mol2", "-f", "mol2", "-o", "mol.frcmod", "-s", "gaff"], cwd=wd, check=True)
    open(os.path.join(wd, "leap.in"), "w").write(LEAP)
    subprocess.run(["tleap", "-f", "leap.in"], cwd=wd, check=True, stdout=subprocess.DEVNULL)
    print(name, "->", os.path.join(wd, "gaff.prmtop"), os.path.exists(os.path.join(wd, "gaff.prmtop")))
