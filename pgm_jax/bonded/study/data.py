"""Reference data of the bonded study: molecules (topology + pGM parameters) and DFT-labelled
frames (MACE-OFF sampling, wB97M-D3(BJ)/def2-TZVPPD labels), in pGM-JAX units."""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from ...param import load_molecule
from ...paths import repo_path
from ...units import BOHR_NM, HARTREE_KJMOL
from ..fit import FrameSet
from ..model import MolSpec
from .molecules import MOLECULES

DATA = repo_path("data/bonded")


def mol_spec(name: str, with_pgm: bool = True) -> MolSpec:
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
    """DFT-labelled frames of one set ("train500", "test298", "scan0", ..., "scan2d"), all chunks."""
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
    p = os.path.join(DATA, "frames", f"{name}_scan.npz")
    if not os.path.exists(p):
        return []
    return [f"scan{t}" for t in range(len(np.load(p)["torsions"]))]


def esp_data(name, n_points=800, seed=0):
    """QM ESP of the pGM parameter fit (runs/bonded/pgm/<name>/esp.dat, B3LYP/aug-cc-pVTZ at the
    MACE-OFF minimum): (atom positions nm, grid nm, potential hartree/e), `n_points` random points."""
    p = repo_path("runs/bonded/pgm", name, "esp.dat")
    if not os.path.exists(p):
        return None
    lines = open(p).read().split("\n")
    na, npt = int(lines[0].split()[0]), int(lines[0].split()[1])
    R = np.array([[float(v) for v in ln.split()[:3]] for ln in lines[1 : 1 + na]]) * BOHR_NM
    E = np.array([[float(v) for v in ln.split()[:4]] for ln in lines[1 + na : 1 + na + npt]])
    k = np.random.default_rng(seed).choice(npt, size=min(n_points, npt), replace=False)
    return R, E[k, 1:] * BOHR_NM, E[k, 0]


def concat(sets):
    sets = [s for s in sets if s is not None and len(s)]
    return FrameSet(
        np.concatenate([s.X for s in sets]),
        np.concatenate([s.E for s in sets]),
        np.concatenate([s.F for s in sets]),
        np.concatenate([s.mu for s in sets]),
    )


def mol_list(spec):
    out = []
    for tok in spec.split(","):
        out += [n for n, v in MOLECULES.items() if v[2] == tok] if tok in ("A1", "A2", "A3", "A4", "B") else [tok]
    return out


def load(names, with_scans=True):
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
