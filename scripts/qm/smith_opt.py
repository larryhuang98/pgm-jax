"""Optimize Smith-type water dimer stationary structures with psi4, keeping their point group.

A start geometry in the point group is built from rigid waters (O-H 0.96 A, 104.5 deg; O-O
2.91 A, bifurcated 3.0 A), optimized at DF-MP2/aug-cc-pVDZ (frozen core) with psi4; optking keeps
the point group (symmetry set from the start geometry, symmetrized to 1e-3).  The structures are
part of the water-cluster QM set (scripts/qm/build_water_clusters.py; docs/qmfit.md).

Names: Cs_open Cs_planar Ci_cyclic C2_cyclic C2h_cyclic C2v_bifurcated C2v_planar_bifurcated

Usage:

    <python with psi4> scripts/qm/smith_opt.py NAME OUTDIR [--threads 8] [--opt-coordinates redundant]
    python scripts/qm/smith_opt.py --help

Inputs: none.
Outputs: <out>/<name>.json (point groups, MP2 energy [Eh], elements, optimized xyz [A], time),
<out>/<name>.out (psi4 output); the JSON is printed.
Units: A (geometries), Eh (energy).
Runtime: CPU, minutes.  Needs psi4 (imported in main; this script does not import pgm_jax).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

R, TH = 0.96, np.radians(104.5)  # O-H [A], H-O-H [rad] of the start geometries
NAMES = ["Cs_open", "Cs_planar", "Ci_cyclic", "C2_cyclic", "C2h_cyclic", "C2v_bifurcated", "C2v_planar_bifurcated"]


def water(O: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return a water (O, H, H) [A] with its bisector along u and its plane spanned by u and v.

    Parameters
    ----------
    O : array (3,)
        Oxygen position [A].
    u : array (3,)
        Direction of the H-O-H bisector (normalized here).
    v : array (3,)
        In-plane direction (its component along u is removed).

    Returns
    -------
    np.ndarray (3, 3)
        Coordinates [A].
    """
    u = np.asarray(u, float)
    u /= np.linalg.norm(u)
    v = np.asarray(v, float)
    v = v - u * (u @ v)
    v /= np.linalg.norm(v)
    O = np.asarray(O, float)
    c, s = np.cos(TH / 2), np.sin(TH / 2)
    return np.array([O, O + R * (c * u + s * v), O + R * (c * u - s * v)])


def donor_along_x(O: np.ndarray) -> np.ndarray:
    """Return a water (3, 3) [A] with its oxygen at O and its first O-H along +x, in the xy plane."""
    u = np.array([np.cos(TH / 2), -np.sin(TH / 2), 0.0])
    v = np.array([np.sin(TH / 2), np.cos(TH / 2), 0.0])
    return water(O, u, v)


def build(name: str) -> np.ndarray:
    """Return the start geometry (6, 3) [A] of the named dimer (donor first).

    Raises
    ------
    KeyError
        An unknown name.
    """
    d = 2.91  # O-O distance [A]
    if name == "Cs_open":
        D = donor_along_x([0, 0, 0])
        b = np.radians(57)
        A = water([d, 0, 0], [np.cos(b), np.sin(b), 0], [0, 0, 1])
    elif name == "Cs_planar":
        D = donor_along_x([0, 0, 0])
        b = np.radians(57)
        A = water([d, 0, 0], [np.cos(b), np.sin(b), 0], [-np.sin(b), np.cos(b), 0])
    elif name in ("Ci_cyclic", "C2_cyclic", "C2h_cyclic"):
        # molecule 1 at -x; its H-bonded O-H at 50 deg from the O-O axis (ring), the free O-H
        # rotated out of the plane by `tilt` about the bonded O-H; molecule 2 is the image of 1
        O1 = np.array([-1.40, 0.0, 0.0])
        a1 = np.radians(50.0)
        h1 = np.array([np.cos(a1), np.sin(a1), 0.0])
        a2 = a1 + TH
        h2 = np.array([np.cos(a2), np.sin(a2), 0.0])
        tilt = 0.0 if name == "C2h_cyclic" else np.radians(35)
        perp = h2 - h1 * (h1 @ h2)
        ph = np.linalg.norm(perp)
        perp /= ph
        h2 = h1 * (h1 @ h2) + ph * (np.cos(tilt) * perp + np.sin(tilt) * np.cross(h1, perp))
        M1 = np.array([O1, O1 + R * h1, O1 + R * h2])
        M2 = M1 * np.array([-1, -1, 1]) if name == "C2_cyclic" else -M1
        D, A = M1, M2
    elif name == "C2v_bifurcated":
        D = water([0, 0, 0], [1, 0, 0], [0, 1, 0])
        A = water([3.0, 0, 0], [1, 0, 0], [0, 0, 1])
    elif name == "C2v_planar_bifurcated":
        D = water([0, 0, 0], [1, 0, 0], [0, 1, 0])
        A = water([3.0, 0, 0], [1, 0, 0], [0, 1, 0])
    else:
        raise KeyError(name)
    return np.vstack([D, A])


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, optimize the dimer and write its JSON (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=NAMES, help="stationary structure")
    ap.add_argument("out", help="output directory")
    ap.add_argument("--threads", type=int, default=8, help="psi4 threads")
    ap.add_argument(
        "--opt-coordinates", default="redundant", help="optking opt_coordinates (redundant, cartesian, both, ...)"
    )
    a = ap.parse_args(argv)
    import psi4

    name, out, nt = a.name, a.out, a.threads
    os.makedirs(out, exist_ok=True)
    X = build(name)
    el = ["O", "H", "H", "O", "H", "H"]
    psi4.set_memory("12 GB")
    psi4.set_num_threads(nt)
    psi4.core.set_output_file(os.path.join(out, f"{name}.out"), False)
    geo = (
        "\n".join(f"{e} {x:.10f} {y:.10f} {z:.10f}" for e, (x, y, z) in zip(el, X))
        + "\nunits angstrom\nno_com\nno_reorient\n"
    )
    mol = psi4.geometry("0 1\n" + geo)
    pg0 = mol.schoenflies_symbol()
    mol = psi4.geometry("0 1\n" + geo + f"symmetry {pg0}\n")  # keep the point group during the steps
    mol.symmetrize(1e-3)
    psi4.set_options(
        {
            "basis": "aug-cc-pvdz",
            "scf_type": "df",
            "mp2_type": "df",
            "freeze_core": True,
            "g_convergence": "gau_tight",
            "geom_maxiter": 200,
            "opt_coordinates": a.opt_coordinates,
        }
    )
    t0 = time.time()
    e = psi4.optimize("mp2", molecule=mol)
    Xo = np.asarray(mol.geometry()) * psi4.constants.bohr2angstroms
    rec = {
        "name": name,
        "point_group_start": pg0,
        "point_group": mol.schoenflies_symbol(),
        "E_mp2_adz": e,
        "elements": el,
        "xyz_A": Xo.tolist(),
        "sec": time.time() - t0,
        "level": "MP2/aug-cc-pVDZ (DF, fc) optimized",
    }
    with open(os.path.join(out, f"{name}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
