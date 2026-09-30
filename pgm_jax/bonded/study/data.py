"""Load the reference data of the bonded study in pGM-JAX units.

Molecules (topology + pGM parameters, `mol_spec`), DFT-labelled frames (MACE-OFF sampling,
wB97M-D3(BJ)/def2-TZVPPD labels, `frames`), torsion scans, the QM ESP of the pGM fit
(`esp_data`), and the per-molecule train / test / scan sets of the study (`load`).  Files live
under data/bonded (`DATA`): molecules/<name>.json, frames/<name>_*.npz, dft/<name>__<key>__*.npz,
params/<name>.json.  See docs/howto_bonded.md for how they are made.

Units: positions nm, energies kJ/mol, forces kJ/mol/nm, dipoles e nm (converted from the files'
A, hartree, hartree/bohr and atomic units).
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Sequence

import numpy as np

from ...param import load_molecule
from ...paths import repo_path
from ...units import BOHR_NM, HARTREE_KJMOL
from ..fit import FrameSet
from ..model import MolSpec
from .molecules import MOLECULES

DATA = repo_path("data/bonded")  # molecules/, frames/, dft/, params/


def mol_spec(name: str, with_pgm: bool = True) -> MolSpec:
    """Return the MolSpec of a study molecule (reference geometry: the first MACE-OFF minimum).

    Parameters
    ----------
    name : str
        Molecule name (study/molecules.py).
    with_pgm : bool
        Load its pGM parameters (params/<name>.json).
    """
    d = json.load(open(os.path.join(DATA, "molecules", f"{name}.json")))
    fr = np.load(os.path.join(DATA, "frames", f"{name}_md.npz"))
    pgm = None
    if with_pgm:
        pgm = load_molecule(os.path.join(DATA, "params", f"{name}.json"))
        assert pgm.elements == d["elements"], name
    return MolSpec(
        name, d["elements"], [tuple(b) for b in d["bonds"]], d["bond_orders"], d["charge"], fr["minima"][0] * 0.1, pgm
    )


def frames(name: str, key: str) -> FrameSet | None:
    """Return the DFT-labelled frames of one set, all chunks, or None if there are none.

    Parameters
    ----------
    name : str
        Molecule name.
    key : str
        Set: "train500", "test298", "scan0", "scan1", ..., or "scan2d".

    Returns
    -------
    FrameSet or None
        Frames sorted by their index in the sampled set (scan2d: file order), with extras "index",
        "src" (chunk file) and, for scans, "angle" [deg] and, for 1D scans, "torsion" (atoms) and
        "mace_E" (MACE-OFF energies [eV]).
    """
    pat = (
        os.path.join(DATA, "dft", f"{name}__{key}__*.npz")
        if key != "scan2d"
        else os.path.join(DATA, "dft", f"{name}__scan2d_*__*.npz")
    )
    files = sorted(glob.glob(pat))
    if not files:
        return None
    X, E, F, D, idx, src = [], [], [], [], [], []
    for f in files:
        z = np.load(f)
        if len(z["index"]) == 0:
            continue
        X.append(z["X"] * 0.1)
        E.append(z["energy_Eh"] * HARTREE_KJMOL)
        F.append(-z["gradient_Eh_bohr"] * HARTREE_KJMOL / BOHR_NM)
        D.append(z["dipole_au"] * BOHR_NM)
        idx.append(z["index"])
        src += [os.path.basename(f)] * len(z["index"])
    extra = {"index": np.concatenate(idx), "src": np.array(src)}
    if key.startswith("scan") and key != "scan2d":
        s = np.load(os.path.join(DATA, "frames", f"{name}_scan.npz"))
        extra["angle"] = s[key + "_angle"][extra["index"]]
        extra["torsion"] = np.repeat(s["torsions"][int(key[4:])][None], len(extra["index"]), 0)
        extra["mace_E"] = s[key + "_E"][extra["index"]]
    if key == "scan2d":
        ang = []
        for f in files:
            lo = os.path.basename(f).split("__")[1]
            a = np.load(os.path.join(DATA, "frames", f"{name}_{lo}.npz"))["angles"]
            ang.append(a[np.load(f)["index"]])
        extra["angle"] = np.concatenate(ang)
    order = np.argsort(extra["index"], kind="stable") if key != "scan2d" else np.arange(len(extra["index"]))
    fs = FrameSet(np.concatenate(X), np.concatenate(E), np.concatenate(F), np.concatenate(D), extra)
    return fs.subset(order)


def scan_keys(name: str) -> list[str]:
    """Return the torsion-scan set names ("scan0", ...) of a molecule (empty without scans)."""
    p = os.path.join(DATA, "frames", f"{name}_scan.npz")
    if not os.path.exists(p):
        return []
    return [f"scan{t}" for t in range(len(np.load(p)["torsions"]))]


def esp_data(name: str, n_points: int = 800, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the QM ESP of the pGM parameter fit at `n_points` random points, or None without the file.

    runs/bonded/pgm/<name>/esp.dat (B3LYP/aug-cc-pVTZ at the MACE-OFF minimum).

    Returns
    -------
    R : np.ndarray (n, 3)
        Atom positions [nm].
    grid : np.ndarray (k, 3)
        ESP points [nm].
    V : np.ndarray (k,)
        Potential [hartree/e].
    """
    p = repo_path("runs/bonded/pgm", name, "esp.dat")
    if not os.path.exists(p):
        return None
    lines = open(p).read().split("\n")
    na, npt = int(lines[0].split()[0]), int(lines[0].split()[1])
    R = np.array([[float(v) for v in ln.split()[:3]] for ln in lines[1 : 1 + na]]) * BOHR_NM
    E = np.array([[float(v) for v in ln.split()[:4]] for ln in lines[1 + na : 1 + na + npt]])
    k = np.random.default_rng(seed).choice(npt, size=min(n_points, npt), replace=False)
    return R, E[k, 1:] * BOHR_NM, E[k, 0]


def concat(sets: Sequence[FrameSet | None]) -> FrameSet:
    """Return the frames of several FrameSets as one (None and empty sets skipped; extras dropped)."""
    sets = [s for s in sets if s is not None and len(s)]
    return FrameSet(
        np.concatenate([s.X for s in sets]),
        np.concatenate([s.E for s in sets]),
        np.concatenate([s.F for s in sets]),
        np.concatenate([s.mu for s in sets]),
    )


def mol_list(spec: str) -> list[str]:
    """Return the molecule names of a comma-separated list of names and subsets ("A1", "A2", "A3", "A4", "B")."""
    out = []
    for tok in spec.split(","):
        out += [n for n, v in MOLECULES.items() if v[2] == tok] if tok in ("A1", "A2", "A3", "A4", "B") else [tok]
    return out


def load(names: Sequence[str], with_scans: bool = True) -> tuple[list[MolSpec], dict]:
    """Return the MolSpecs and data of the named molecules for a Fitter.

    Molecules without train500 or test298 frames are skipped (with a printed note).  The training
    set includes the torsion scans with at least 20 frames.

    Returns
    -------
    specs : list of MolSpec
        The molecules loaded.
    data : dict
        {index: {"train": FrameSet, "test": FrameSet, "scans": {key: FrameSet}}}.
    """
    data, specs = {}, []
    for n in names:
        tr = frames(n, "train500")
        te = frames(n, "test298")
        if tr is None or te is None:
            print(f"  {n}: no DFT frames yet, skipped", flush=True)
            continue
        scans = {k: frames(n, k) for k in scan_keys(n)} if with_scans else {}
        scans = {k: v for k, v in scans.items() if v is not None and len(v) >= 20}
        specs.append(mol_spec(n))
        data[len(specs) - 1] = {"train": concat([tr] + list(scans.values())), "test": te, "scans": scans}
    return specs, data
