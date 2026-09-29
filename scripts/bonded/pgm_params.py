"""pGM electrostatic parameters for the bonded-study molecules by the standard pGM route.

As evoff/scripts/param_s66.py: GAFF types (antechamber) -> B3LYP/aug-cc-pVTZ ESP (psi4,
scripts/bonded/qm_esp.py) -> two-stage py_resp (ipol=5 pGM-perm, all 1-2/1-3 pairs included) with
the pGM-pol table; LJ from GAFF.  Geometry: the MACE-OFF minimum; with --multi-conformer also
NCONF - 1 evenly spaced 500 K training frames (the "2" set: runs/bonded/pgm2, data/bonded/params2).

Usage:

    python scripts/bonded/pgm_params.py prep     # runs/bonded/pgm/<name>/input.json, runs/bonded/esp.sh
    sbatch runs/bonded/esp.sh                    # ESP (written by prep)
    python scripts/bonded/pgm_params.py fit      # data/bonded/params/<name>.json
    python scripts/bonded/pgm_params.py prep --multi-conformer
    python scripts/bonded/pgm_params.py --help

Inputs: data/bonded/frames/<name>_{md,min}.npz, data/bonded/molecules/<name>.json, AmberTools
(antechamber, py_resp; pgm_jax.paths resource "amberhome").
Outputs: prep: runs/bonded/pgm<set>/<name>/input.json, runs/bonded/pgm<set>_list.txt,
runs/bonded/esp<set>.sh (slurm array); fit: data/bonded/params<set>/<name>.json.
Units: Angstrom (geometries), GAFF R* [A] and eps [kcal/mol] converted to nm and kJ/mol.
Runtime: prep and fit minutes (CPU); the ESP array hours.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

from pgm_jax.bonded.study.molecules import MOLECULES
from pgm_jax.param import PGM_POL_TABLE, molecule_from_pyresp, read_pol_table, save_molecule
from pgm_jax.paths import repo_path, resource
from pgm_jax.units import KCAL

NCONF = 5  # conformers of the multi-conformer set (minimum + NCONF - 1 training frames)


def amber_env() -> tuple[str, dict]:
    """Return AMBERHOME and the environment for AmberTools subprocesses (its bin/ first on PATH)."""
    amber = resource("amberhome")
    return amber, dict(os.environ, AMBERHOME=amber, PATH=f"{amber}/bin:" + os.environ["PATH"])


def gaff_types(elements: list[str], xyz: np.ndarray, charge: int, wd: str) -> list[str]:
    """Return the GAFF atom types of a molecule (antechamber on <wd>/mol.pdb, written from xyz [A])."""
    with open(os.path.join(wd, "mol.pdb"), "w") as fh:
        for k, (e, x) in enumerate(zip(elements, xyz)):
            fh.write(
                f"HETATM{k + 1:5d} {e + str(k + 1):<4s} MOL A   1    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00    "
                f"      {e:>2s}\n"
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
        env=amber_env()[1],
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


def gaff_lj(path: str | None = None) -> dict[str, tuple[float, float]]:
    """Return {type: (R* [A], eps [kcal/mol])} from the MOD4 RE section of gaff.dat (default: AMBERHOME's)."""
    if path is None:
        path = os.path.join(amber_env()[0], "dat/leap/parm/gaff.dat")
    with open(path) as fh:
        lines = fh.read().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("MOD4"))
    out = {}
    for ln in lines[k + 1 :]:
        t = ln.split()
        if not t or ln.startswith("END"):
            break
        out[t[0].lower()] = (float(t[1]), float(t[2]))
    return out


def prep(ver: str = "", psi4_python: str = "python") -> None:
    """Write the ESP inputs of every molecule with frames and the slurm array script.

    Parameters
    ----------
    ver : str
        "" (single minimum) or "2" (multi-conformer ESP).
    psi4_python : str
        Python with psi4, written into the slurm script.
    """
    wdir = repo_path("runs", "bonded", f"pgm{ver}")
    table = read_pol_table()
    names = []
    for name, (_smi, charge, _subset) in MOLECULES.items():
        fr = repo_path("data", "bonded", "frames", f"{name}_md.npz")
        if not os.path.exists(fr):
            fr = repo_path("data", "bonded", "frames", f"{name}_min.npz")
        if not os.path.exists(fr):
            print(f"{name:20s} no frames yet")
            continue
        with open(repo_path("data", "bonded", "molecules", f"{name}.json")) as fh:
            d = json.load(fh)
        xyz = np.load(fr)["minima"][0]
        wd = os.path.join(wdir, name)
        os.makedirs(wd, exist_ok=True)
        types = gaff_types(d["elements"], xyz, charge, wd)
        missing = sorted({t for t in types if t.lower() not in table})
        rec = {"name": name, "elements": d["elements"], "types": types, "xyz_A": xyz.tolist(), "charge": charge}
        if ver == "2" and fr.endswith("_md.npz"):  # minimum + evenly spaced 500 K training frames
            tr = np.load(repo_path("data", "bonded", "frames", f"{name}_md.npz"))["train500"]
            pick = np.linspace(0, len(tr) - 1, NCONF - 1).astype(int)
            rec["xyz_A_list"] = [xyz.tolist()] + [tr[k].tolist() for k in pick]
        with open(os.path.join(wd, "input.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
        names.append(name)
        print(f"{name:20s} {' '.join(types)}" + (f"   MISSING pGM-pol: {missing}" if missing else ""))
    root = repo_path()
    with open(repo_path("runs", "bonded", f"pgm{ver}_list.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    with open(repo_path("runs", "bonded", f"esp{ver}.sh"), "w") as fh:
        fh.write(f"""#!/bin/bash
#SBATCH --job-name=pgmjax-esp
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-{len(names) - 1}
#SBATCH --output={root}/runs/bonded/slurm/esp{ver}_%a.out
cd {root}
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" runs/bonded/pgm{ver}_list.txt)
export PSI_SCRATCH=/tmp/$USER/pgmjax_esp_$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID; mkdir -p $PSI_SCRATCH
{psi4_python} scripts/bonded/qm_esp.py runs/bonded/pgm{ver}/$NAME --threads 16 --memory-GB 32
rm -rf $PSI_SCRATCH
""")


def fit(ver: str = "") -> None:
    """Run the two-stage py_resp fit of every molecule with an ESP; write data/bonded/params<ver>/<name>.json."""
    amber, env = amber_env()
    wdir, out = repo_path("runs", "bonded", f"pgm{ver}"), repo_path("data", "bonded", f"params{ver}")
    table, lj = read_pol_table(), gaff_lj()
    os.makedirs(out, exist_ok=True)
    pr = os.path.join(amber, "bin")
    with open(repo_path("runs", "bonded", f"pgm{ver}_list.txt")) as fh:
        names = fh.read().split()
    for name in names:
        wd = os.path.join(wdir, name)
        if not os.path.exists(os.path.join(wd, "esp.dat")):
            print(f"{name:20s} no ESP yet")
            continue
        with open(os.path.join(wd, "input.json")) as fh:
            inp = json.load(fh)

        def sh(cmd, wd=wd):
            """Run an AmberTools command in the molecule's directory (output captured)."""
            return subprocess.run(cmd, cwd=wd, env=env, check=True, capture_output=True, text=True)

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
        with open(repo_path("data", "bonded", "molecules", f"{name}.json")) as fh:
            d = json.load(fh)
        m.bonds = [tuple(b) for b in d["bonds"]]
        m.lj_rmin_half = np.array([lj[t.lower()][0] * 0.1 for t in inp["types"]])
        m.lj_sqrt_eps = np.sqrt(np.array([lj[t.lower()][1] * KCAL for t in inp["types"]]))
        m.extra["xyz_ref_A"] = np.array(inp["xyz_A"])
        save_molecule(m, os.path.join(out, f"{name}.json"))
        with open(os.path.join(wd, "2nd.out")) as fh:
            rr = [ln for ln in fh if "RRMS" in ln.upper()]
        print(
            f"{name:20s} q {' '.join(f'{x:+.3f}' for x in m.q)}  ncov {len(m.cov)}  sum {m.q.sum():+.3f}  "
            f"{rr[-1].strip() if rr else ''}"
        )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run prep or fit (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("prep", "fit"), help="prep: ESP inputs and slurm script; fit: py_resp fits")
    ap.add_argument(
        "--multi-conformer", action="store_true", help="the multi-conformer set (pgm2/params2; was PGMVER=2)"
    )
    ap.add_argument(
        "--psi4-python",
        default=os.environ.get("PGM_PSI4_PYTHON", "python"),
        help="python with psi4 for the slurm script (default: $PGM_PSI4_PYTHON or python)",
    )
    a = ap.parse_args(argv)
    ver = "2" if a.multi_conformer else ""
    if a.mode == "prep":
        prep(ver, a.psi4_python)
    else:
        fit(ver)


if __name__ == "__main__":
    main()
