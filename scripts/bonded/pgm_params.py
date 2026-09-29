"""pGM electrostatic parameters for the bonded-study molecules, the standard pGM route
(as evoff/scripts/param_s66.py): GAFF types (antechamber) -> B3LYP/aug-cc-pVTZ ESP (psi4,
scripts/bonded/qm_esp.py) -> two-stage py_resp (ipol=5 pGM-perm, all 1-2/1-3 pairs included)
with the pGM-pol table; LJ from GAFF.  Geometry: the MACE-OFF minimum.

    python scripts/bonded/pgm_params.py prep     # runs/bonded/pgm/<name>/input.json
    sbatch runs/bonded/esp.sh                     # ESP (written by prep)
    python scripts/bonded/pgm_params.py fit      # data/bonded/params/<name>.json
"""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pgm_jax.bonded.molecules import MOLECULES  # noqa: E402
from pgm_jax.param import PGM_POL_TABLE, molecule_from_pyresp, read_pol_table, save_molecule  # noqa: E402

AMBER = os.path.expanduser("~/amber25")
ENV = dict(os.environ, AMBERHOME=AMBER, PATH=f"{AMBER}/bin:" + os.environ["PATH"])
VER = os.environ.get("PGMVER", "")  # "" (single minimum) or "2" (multi-conformer ESP)
NCONF = 5
WD = os.path.join(ROOT, f"runs/bonded/pgm{VER}")
OUT = os.path.join(ROOT, f"data/bonded/params{VER}")


def gaff_types(elements, xyz, charge, wd):
    with open(os.path.join(wd, "mol.pdb"), "w") as fh:
        for k, (e, x) in enumerate(zip(elements, xyz)):
            fh.write(
                f"HETATM{k + 1:5d} {e + str(k + 1):<4s} MOL A   1    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00          {e:>2s}\n"
            )
        fh.write("END\n")
    subprocess.run(
        [
            "antechamber",
            "-i",
            "mol.pdb",
            "-fi",
            "pdb",
            "-o",
            "mol.mol2",
            "-fo",
            "mol2",
            "-at",
            "gaff",
            "-nc",
            str(charge),
            "-dr",
            "no",
            "-pf",
            "y",
        ],
        cwd=wd,
        env=ENV,
        check=True,
        capture_output=True,
    )
    types, on = [], False
    for ln in open(os.path.join(wd, "mol.mol2")):
        if ln.startswith("@<TRIPOS>ATOM"):
            on = True
            continue
        if ln.startswith("@<TRIPOS>") and on:
            break
        if on and ln.strip():
            types.append(ln.split()[5])
    assert len(types) == len(elements)
    return types


def gaff_lj(path=os.path.join(AMBER, "dat/leap/parm/gaff.dat")):
    """{type: (R* A, eps kcal/mol)} from the MOD4 RE section."""
    lines = open(path).read().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("MOD4"))
    out = {}
    for ln in lines[k + 1 :]:
        t = ln.split()
        if not t or ln.startswith("END"):
            break
        out[t[0].lower()] = (float(t[1]), float(t[2]))
    return out


def prep():
    table = read_pol_table()
    names = []
    for name, (smi, charge, subset) in MOLECULES.items():
        fr = os.path.join(ROOT, "data/bonded/frames", f"{name}_md.npz")
        if not os.path.exists(fr):
            fr = os.path.join(ROOT, "data/bonded/frames", f"{name}_min.npz")
        if not os.path.exists(fr):
            print(f"{name:20s} no frames yet")
            continue
        d = json.load(open(os.path.join(ROOT, "data/bonded/molecules", f"{name}.json")))
        xyz = np.load(fr)["minima"][0]
        wd = os.path.join(WD, name)
        os.makedirs(wd, exist_ok=True)
        types = gaff_types(d["elements"], xyz, charge, wd)
        missing = sorted({t for t in types if t.lower() not in table})
        rec = {"name": name, "elements": d["elements"], "types": types, "xyz_A": xyz.tolist(), "charge": charge}
        if VER == "2" and fr.endswith("_md.npz"):  # minimum + evenly spaced 500 K training frames
            tr = np.load(os.path.join(ROOT, "data/bonded/frames", f"{name}_md.npz"))["train500"]
            pick = np.linspace(0, len(tr) - 1, NCONF - 1).astype(int)
            rec["xyz_A_list"] = [xyz.tolist()] + [tr[k].tolist() for k in pick]
        json.dump(rec, open(os.path.join(wd, "input.json"), "w"), indent=1)
        names.append(name)
        print(f"{name:20s} {' '.join(types)}" + (f"   MISSING pGM-pol: {missing}" if missing else ""))
    open(os.path.join(ROOT, f"runs/bonded/pgm{VER}_list.txt"), "w").write("\n".join(names) + "\n")
    open(os.path.join(ROOT, f"runs/bonded/esp{VER}.sh"), "w").write(f"""#!/bin/bash
#SBATCH --job-name=pgmjax-esp
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-{len(names) - 1}
#SBATCH --output={ROOT}/runs/bonded/slurm/esp{VER}_%a.out
cd {ROOT}
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" runs/bonded/pgm{VER}_list.txt)
export PSI_SCRATCH=/tmp/larry/pgmjax_esp_$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID; mkdir -p $PSI_SCRATCH
~/miniconda3/envs/psi4/bin/python scripts/bonded/qm_esp.py runs/bonded/pgm{VER}/$NAME --threads 16 --memory 32
rm -rf $PSI_SCRATCH
""")


def fit():
    table, lj = read_pol_table(), gaff_lj()
    os.makedirs(OUT, exist_ok=True)
    pr = os.path.join(AMBER, "bin")
    for name in open(os.path.join(ROOT, f"runs/bonded/pgm{VER}_list.txt")).read().split():
        wd = os.path.join(WD, name)
        if not os.path.exists(os.path.join(wd, "esp.dat")):
            print(f"{name:20s} no ESP yet")
            continue
        inp = json.load(open(os.path.join(wd, "input.json")))
        sh = lambda cmd: subprocess.run(cmd, cwd=wd, env=ENV, check=True, capture_output=True, text=True)
        nconf = len(inp.get("xyz_A_list", [0]))
        sh(
            [
                f"{pr}/pyresp_gen.py",
                "-i",
                "esp.dat",
                "-p",
                "perm",
                "-q",
                str(inp["charge"]),
                "-n",
                str(nconf),
                "-f1",
                "1st.in",
                "-f2",
                "2nd.in",
            ]
        )
        sh(
            [
                f"{pr}/py_resp.py",
                "-O",
                "-i",
                "1st.in",
                "-o",
                "1st.out",
                "-s",
                "1st.esp",
                "-t",
                "1st.chg",
                "-ip",
                PGM_POL_TABLE,
                "-e",
                "esp.dat",
            ]
        )
        sh(
            [
                f"{pr}/py_resp.py",
                "-O",
                "-i",
                "2nd.in",
                "-o",
                "2nd.out",
                "-q",
                "1st.chg",
                "-s",
                "2nd.esp",
                "-t",
                "2nd.chg",
                "-ip",
                PGM_POL_TABLE,
                "-e",
                "esp.dat",
            ]
        )
        m = molecule_from_pyresp(
            name, inp["elements"], inp["types"], os.path.join(wd, "2nd.chg"), table, n_atoms=len(inp["elements"])
        )
        d = json.load(open(os.path.join(ROOT, "data/bonded/molecules", f"{name}.json")))
        m.bonds = [tuple(b) for b in d["bonds"]]
        m.lj_rmin_half = np.array([lj[t.lower()][0] * 0.1 for t in inp["types"]])
        m.lj_sqrt_eps = np.sqrt(np.array([lj[t.lower()][1] * 4.184 for t in inp["types"]]))
        m.extra["xyz_ref_A"] = np.array(inp["xyz_A"])
        save_molecule(m, os.path.join(OUT, f"{name}.json"))
        rr = [ln for ln in open(os.path.join(wd, "2nd.out")) if "RRMS" in ln.upper()]
        print(
            f"{name:20s} q {' '.join(f'{x:+.3f}' for x in m.q)}  ncov {len(m.cov)}  sum {m.q.sum():+.3f}  {rr[-1].strip() if rr else ''}"
        )


if __name__ == "__main__":
    {"prep": prep, "fit": fit}[sys.argv[1]]()
