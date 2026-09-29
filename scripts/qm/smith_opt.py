"""Water dimer stationary structures (Smith-type, symmetry-constrained): build a start geometry in
the point group, optimize at MP2/aug-cc-pVDZ with psi4 (optking keeps the point group), write xyz.

    python scripts/qmfit/smith_opt.py NAME OUTDIR [threads]
Names: Cs_open Cs_planar Ci_cyclic C2_cyclic C2h_cyclic C2v_bifurcated C2v_planar_bifurcated
"""

import json
import os
import sys
import time

import numpy as np

R, TH = 0.96, np.radians(104.5)


def water(O, u, v):
    u = np.asarray(u, float)
    u /= np.linalg.norm(u)
    v = np.asarray(v, float)
    v = v - u * (u @ v)
    v /= np.linalg.norm(v)
    O = np.asarray(O, float)
    c, s = np.cos(TH / 2), np.sin(TH / 2)
    return np.array([O, O + R * (c * u + s * v), O + R * (c * u - s * v)])


def donor_along_x(O):
    """Water with O at O and its first O-H along +x, in the xy plane."""
    u = np.array([np.cos(TH / 2), -np.sin(TH / 2), 0.0])
    v = np.array([np.sin(TH / 2), np.cos(TH / 2), 0.0])
    return water(O, u, v)


def build(name):
    d = 2.91
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


def main():
    import psi4

    name, out = sys.argv[1], sys.argv[2]
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 8
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
            "opt_coordinates": os.environ.get("OPT_COORDS", "redundant"),
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
    json.dump(rec, open(os.path.join(out, f"{name}.json"), "w"), indent=1)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
