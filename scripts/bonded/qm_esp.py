"""Compute the pGM electrostatic reference of one molecule, the pGM/PyRESP protocol (Wang et al. JCTC 2019).

B3LYP/aug-cc-pVTZ electrostatic potential on Merz-Kollman shells (1.4, 1.6, 1.8, 2.0 x the MK
radii, 6 points per A^2) and the dipole, at the sampling minimum (or at every conformer of
xyz_A_list).  The Merz-Kollman shell code is that of evoff/qm/psi4_monomer.py.

Usage:

    <python with psi4> scripts/bonded/qm_esp.py runs/bonded/pgm/<name> --threads 16 --memory-GB 32
    python scripts/bonded/qm_esp.py --help

Inputs: <dir>/input.json {name, elements, types, xyz_A (or xyz_A_list), charge}.
Outputs: <dir>/esp_<k>.dat per conformer and esp.dat (py_resp format, the blocks in sequence),
<dir>/qm.json (dipoles [a.u.], number of points, time), <dir>/psi4.out.
Units: Angstrom (input), bohr and atomic units (esp.dat), a.u. (dipoles).
Runtime: CPU, minutes to an hour per conformer.  This script does not import pgm_jax (it runs in a
psi4 environment).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

BOHR = 0.52917721067  # A per bohr
MK_RADII = {"H": 1.20, "C": 1.50, "N": 1.50, "O": 1.40, "S": 1.75, "F": 1.35, "Cl": 1.70, "P": 1.80}  # A
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17}


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n points (n, 3) spread evenly on the unit sphere (Fibonacci lattice)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5**0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def mk_points(
    xyz: np.ndarray, elements: list[str], scales: tuple = (1.4, 1.6, 1.8, 2.0), density: float = 6.0
) -> np.ndarray:
    """Return the Merz-Kollman points [A]: shells at scales x MK radius, density [1/A^2], outside all other shells."""
    rad = np.array([MK_RADII[e] for e in elements])
    pts = []
    for s in scales:
        R = s * rad
        for i in range(len(xyz)):
            n = max(int(density * 4 * np.pi * R[i] ** 2), 10)
            p = xyz[i] + R[i] * fibonacci_sphere(n)
            dd = np.linalg.norm(p[:, None, :] - xyz[None, :, :], axis=-1)
            pts.append(p[np.all(dd >= R[None, :] - 1e-8, axis=1)])
    return np.concatenate(pts)


def write_espdat(
    path: str,
    xyz_A: np.ndarray,
    elements: list[str],
    types: list[str],
    pts_A: np.ndarray,
    esp_au: np.ndarray,
    name: str = "MOL",
) -> None:
    """Write a py_resp esp.dat: atoms and points in bohr, the ESP in atomic units."""
    n, m = len(xyz_A), len(pts_A)
    with open(path, "w") as fh:
        fh.write(f"{n:5d}{m:5d}    0 {name[:3].upper():3s}{n:9d}{m:7d}\n")
        for k, (x, e, t) in enumerate(zip(xyz_A / BOHR, elements, types)):
            fh.write(" " * 18 + "".join(f"{v:16.7E}" for v in x) + f"{Z[e]:4d} {t:3s} {e}{k + 1:<4d}\n")
        for v, p in zip(esp_au, pts_A / BOHR):
            fh.write(f"{v:16.7E}" + "".join(f"{c:16.7E}" for c in p) + "\n")


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run psi4 on every conformer and write the ESP files (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="molecule directory with input.json")
    ap.add_argument("--threads", type=int, default=16, help="psi4 threads")
    ap.add_argument("--memory-GB", type=float, default=30, help="psi4 memory [GB]")
    a = ap.parse_args(argv)
    import psi4

    with open(os.path.join(a.dir, "input.json")) as fh:
        d = json.load(fh)
    psi4.set_memory(f"{a.memory_GB} GB")
    psi4.set_num_threads(a.threads)
    psi4.core.set_output_file(os.path.join(a.dir, "psi4.out"), False)
    geoms = [np.array(g) for g in d.get("xyz_A_list", [d["xyz_A"]])]
    psi4.set_options({"basis": "aug-cc-pvtz", "scf_type": "df", "d_convergence": 1e-8})
    t0 = time.time()
    dips, npts, parts = [], 0, []
    for c, xyz in enumerate(geoms):
        psi4.core.clean()
        lines = [f"{d['charge']} 1"] + [f"{e} {x:.8f} {y:.8f} {z:.8f}" for e, (x, y, z) in zip(d["elements"], xyz)]
        mol = psi4.geometry("\n".join(lines + ["symmetry c1", "no_reorient", "no_com"]))
        e, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)
        psi4.oeprop(wfn, "DIPOLE")
        dips.append(
            np.array(wfn.variable("SCF DIPOLE") if wfn.has_variable("SCF DIPOLE") else wfn.variable("CURRENT DIPOLE"))
            .ravel()
            .tolist()
        )
        pts = mk_points(xyz, d["elements"])
        esp = np.array(psi4.core.ESPPropCalc(wfn).compute_esp_over_grid_in_memory(psi4.core.Matrix.from_array(pts)))
        part = os.path.join(a.dir, f"esp_{c}.dat")
        write_espdat(part, xyz, d["elements"], d["types"], pts, esp, d["name"])
        parts.append(part)
        npts += len(pts)
    with open(os.path.join(a.dir, "esp.dat"), "w") as fh:  # py_resp multi-conformer input: blocks in sequence
        for p in parts:
            with open(p) as src:
                fh.write(src.read())
    qm = {
        "name": d["name"],
        "level": "B3LYP/aug-cc-pVTZ",
        "n_conf": len(geoms),
        "dipole_au": dips[0],
        "dipoles_au": dips,
        "n_esp_points": int(npts),
        "time_s": time.time() - t0,
    }
    with open(os.path.join(a.dir, "qm.json"), "w") as fh:
        json.dump(qm, fh, indent=1)
    print(d["name"], "done", time.time() - t0)


if __name__ == "__main__":
    main()
