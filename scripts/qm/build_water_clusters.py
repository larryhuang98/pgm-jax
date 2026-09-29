"""Build the geometries of the water-cluster QM set (rigid monomers at the pGM3P-25 geometry).

Every monomer is replaced by the rigid water of the model (r_OH, HOH angle of the pGM3P-25
prmtop / restart: 0.9745 A, 103.64 deg), superposed on the source monomer (mass-weighted Kabsch,
centre of mass kept), so model and QM see the same geometry and the interaction energies contain
no monomer deformation (docs/qmfit.md).  The QM jobs of each record are run by
scripts/qm/psi4_clusters.py and collected by scripts/qm/collect_water_qm.py.

Sets (record "set"):
  smith        symmetry-constrained MP2/aug-cc-pVDZ stationary structures of the dimer
               (scripts/qm/smith_opt.py; PGM_QMDATA/smith/*.json)
  radial       O-O scans of four of them (the partner translated along O...O)
  angular      scans around the minimum: acceptor flap, donor bend, acceptor twist
  liquid2      pairs from pGM liquid snapshots, stratified in R_OO
  water27      the (H2O)n, n = 2..6 and 8 clusters of WATER27 (GMTKN55; Bryantsev et al.
               JCTC 2009, 5, 1016), monomers rigidified
  liquid3/4/5  trimers, tetramers, pentamers cut from the snapshots (a molecule and neighbours)
  pairs        every pair of every cluster with n >= 3 (2-body corrections to the cluster MP2)

Usage:

    python scripts/qm/build_water_clusters.py [--smith DIR] [--out data/qm/water_geoms.json]
    python scripts/qm/build_water_clusters.py --help

Inputs: the Smith structures (--smith, default PGM_QMDATA/smith), the liquid snapshots (PGM_EPSP:
p25_4096.rst7, p25_512.rst7, base/base_4096.rst7), data/qm/water27_raw.json.
Outputs: the geometry file (--out; records with id, set, n, xyz_A, jobs, meta); printed counts.
Units: Angstrom, degrees.
Runtime: seconds.  The random choices use a fixed seed (the file is reproducible).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter

import numpy as np

from pgm_jax.paths import repo_path, resource

R_OH, HOH = 0.9745, 103.64  # pGM3P-25 rigid water (p25_512.rst7), Angstrom / degrees
MASS = np.array([15.999, 1.008, 1.008])  # O, H, H [amu] (Kabsch weights)
SNAPSHOTS = {
    "p25_4096": resource("epsp", "p25_4096.rst7"),
    "p25_512": resource("epsp", "p25_512.rst7"),
    "base_4096": resource("epsp", "base/base_4096.rst7"),
}
DIMER_JOBS = ["sapt0", "mp2:aug-cc-pvtz", "mp2:aug-cc-pvqz", "ccsdt:aug-cc-pvtz"]
CLUSTER_JOBS = ["mbe:mp2:aug-cc-pvtz:3"]


def model_water() -> np.ndarray:
    """Return the model's rigid water (O, H, H) [A], O at the origin, bisector along +y."""
    t = np.radians(HOH) / 2
    return np.array([[0, 0, 0], [R_OH * np.sin(t), R_OH * np.cos(t), 0], [-R_OH * np.sin(t), R_OH * np.cos(t), 0]])


def kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotation Rm (3, 3) and translation t (3,) minimizing sum w |Rm P + t - Q|^2."""
    pc, qc = (w[:, None] * P).sum(0) / w.sum(), (w[:, None] * Q).sum(0) / w.sum()
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rm, qc - Rm @ pc


def rigidify(X: np.ndarray) -> np.ndarray:
    """Return the coordinates (3n, 3) [A] (O, H, H per molecule) with the model's rigid water superposed on each."""
    X = np.asarray(X, float).reshape(-1, 3, 3)
    M = model_water()
    out = []
    for w in X:
        Rm, t = kabsch(M, w, MASS)
        out.append(M @ Rm.T + t)
    return np.concatenate(out)


def group_waters(atoms: list) -> np.ndarray:
    """Return (3n, 3) coordinates O, H, H per molecule from [[el, x, y, z], ...] in any order (H to the nearest O)."""
    el = [a[0] for a in atoms]
    X = np.array([a[1:] for a in atoms], float)
    O = [k for k, e in enumerate(el) if e == "O"]
    H = [k for k, e in enumerate(el) if e == "H"]
    own = {o: [] for o in O}
    for h in H:
        own[min(O, key=lambda o: np.linalg.norm(X[h] - X[o]))].append(h)
    assert all(len(v) == 2 for v in own.values())
    return np.concatenate([X[[o] + own[o]] for o in O])


def rot(axis: np.ndarray, ang: float) -> np.ndarray:
    """Return the rotation matrix (3, 3) about `axis` by `ang` [rad] (Rodrigues)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def read_rst7(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the coordinates (N, 3) [A] and the box (3, 3) [A] (vectors as rows) of an ASCII rst7."""
    L = open(path).read().splitlines()
    n = int(L[1].split()[0])
    v = []
    for ln in L[2:]:
        v += [float(ln[i : i + 12]) for i in range(0, len(ln), 12) if ln[i : i + 12].strip()]
    X = np.array(v[: 3 * n]).reshape(-1, 3)
    a, b, c, al, be, ga = v[-6:]
    al, be, ga = np.radians([al, be, ga])
    H = np.zeros((3, 3))
    H[0] = [a, 0, 0]
    H[1] = [b * np.cos(ga), b * np.sin(ga), 0]
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    H[2] = [cx, cy, np.sqrt(c * c - cx * cx - cy * cy)]
    return X, H


IMAGES = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)  # neighbouring cells


def min_image(d: np.ndarray, H: np.ndarray, Hinv: np.ndarray) -> np.ndarray:
    """Return the shortest periodic image of displacement(s) d (..., 3) in box H (checks the 27 nearest cells)."""
    f = d @ Hinv
    d0 = (f - np.round(f)) @ H
    cand = d0[..., None, :] + IMAGES @ H
    k = np.argmin(np.linalg.norm(cand, axis=-1), axis=-1)
    return np.take_along_axis(cand, k[..., None, None], axis=-2)[..., 0, :]


class Snapshot:
    """A liquid water snapshot with whole molecules (O, H, H), for cutting clusters.

    Attributes
    ----------
    name : str
    H, Hinv : np.ndarray (3, 3)
        Box [A] and its inverse.
    W : np.ndarray (M, 3, 3)
        Molecules, hydrogens imaged next to their oxygen [A].
    O : np.ndarray (M, 3)
        Oxygen positions [A].
    """

    def __init__(self, name: str, path: str) -> None:
        """Read the snapshot from an ASCII rst7 and make the molecules whole."""
        X, H = read_rst7(path)
        self.name, self.H, self.Hinv = name, H, np.linalg.inv(H)
        W = X.reshape(-1, 3, 3)
        W[:, 1:] = W[:, :1] + min_image(W[:, 1:] - W[:, :1], H, self.Hinv)  # whole molecules
        self.W = W
        self.O = W[:, 0]

    def neighbours(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the O-O distances (M,) [A] (inf for i itself) and minimum-image vectors (M, 3) from molecule i."""
        d = min_image(self.O - self.O[i], self.H, self.Hinv)
        r = np.linalg.norm(d, axis=-1)
        r[i] = np.inf
        return r, d

    def cluster(self, idx: list[int]) -> np.ndarray:
        """Return the molecules idx imaged around molecule idx[0], (3n, 3) [A], centred."""
        c = self.O[idx[0]]
        out = []
        for k in idx:
            sh = min_image(self.O[k] - c, self.H, self.Hinv) - (self.O[k] - c)
            out.append(self.W[k] + sh)
        X = np.concatenate(out)
        return X - X.mean(0)


def rec(rid: str, set_: str, X: np.ndarray, jobs: list[str], **meta) -> dict:
    """Return a geometry record (id, set, n, xyz_A rounded to 1e-8 A, jobs, meta = the keywords)."""
    X = np.asarray(X)
    return {
        "id": rid,
        "set": set_,
        "n": len(X) // 3,
        "xyz_A": np.round(X, 8).tolist(),
        "jobs": list(jobs),
        "meta": meta,
    }


def roo(X: np.ndarray, i: int = 0, j: int = 1) -> float:
    """Return the O-O distance [A] of molecules i and j of X (O, H, H per molecule)."""
    return float(np.linalg.norm(X[3 * i] - X[3 * j]))


def dimer_records(smith_dir: str) -> list[dict]:
    """Return the monomer, the Smith stationary structures and the radial and angular dimer scans.

    Structures that converged onto another one (same MP2 energy to 1e-6 Eh) are kept once.
    """
    records = [rec("monomer/p25", "monomer", model_water(), ["props:ccsd:aug-cc-pvtz"])]
    smith = {}
    for fn in sorted(os.listdir(smith_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(smith_dir, fn)) as fh:
                d = json.load(fh)
            smith[d["name"]] = d
    seen = []
    for name, d in sorted(smith.items(), key=lambda kv: kv[1]["E_mp2_adz"]):
        if any(abs(d["E_mp2_adz"] - e) < 1e-6 for e in seen):
            continue  # converged onto another structure (C2 -> C2h)
        seen.append(d["E_mp2_adz"])
        X = rigidify(d["xyz_A"])
        records.append(
            rec(
                f"smith/{name}",
                "smith",
                X,
                DIMER_JOBS,
                point_group=d["point_group"],
                E_mp2_adz_opt=d["E_mp2_adz"],
                R_OO=roo(X),
            )
        )
    for name, dists in (
        ("Cs_open", [2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]),
        ("Cs_planar", [2.5, 2.7, 3.2, 3.6, 4.0, 5.0]),
        ("Ci_cyclic", [2.5, 2.65, 3.0, 3.3, 3.6, 4.0, 5.0]),
        ("C2v_bifurcated", [2.6, 2.8, 3.2, 3.5, 4.0, 5.0]),
    ):
        X = rigidify(smith[name]["xyz_A"])
        u = (X[3] - X[0]) / roo(X)
        for R in dists:
            Y = X.copy()
            Y[3:] += (R - roo(X)) * u
            records.append(rec(f"radial/{name}/{R:.2f}", "radial", Y, DIMER_JOBS, parent=name, R_OO=R))
    X = rigidify(smith["Cs_open"]["xyz_A"])
    z = np.cross(X[1] - X[0], X[3] - X[0])  # normal of the mirror plane (donor plane)
    z /= np.linalg.norm(z)
    for deg in (-60, -40, -20, 20, 40, 60, 80, 100):  # acceptor flap (about the normal, at O_B)
        Y = X.copy()
        Y[3:] = (X[3:] - X[3]) @ rot(z, np.radians(deg)).T + X[3]
        records.append(
            rec(f"angular/flap/{deg:+d}", "angular", Y, DIMER_JOBS, parent="Cs_open", angle=deg, R_OO=roo(Y))
        )
    for deg in (-40, -25, -12, 12, 25, 40):  # donor bend (about the normal, at O_A)
        Y = X.copy()
        Y[:3] = (X[:3] - X[0]) @ rot(z, np.radians(deg)).T + X[0]
        records.append(
            rec(f"angular/bend/{deg:+d}", "angular", Y, DIMER_JOBS, parent="Cs_open", angle=deg, R_OO=roo(Y))
        )
    ax = X[3] - X[0]
    for deg in (30, 60, 90, 120, 150, 180):  # acceptor twist about O...O
        Y = X.copy()
        Y[3:] = (X[3:] - X[3]) @ rot(ax, np.radians(deg)).T + X[3]
        records.append(
            rec(f"angular/twist/{deg:03d}", "angular", Y, DIMER_JOBS, parent="Cs_open", angle=deg, R_OO=roo(Y))
        )
    return records


def liquid_pairs(snaps: dict, rng: np.random.Generator) -> list[dict]:
    """Return pairs from the snapshots, stratified in R_OO (bins of [A] with counts split over the snapshots)."""
    records = []
    bins = [
        (2.40, 2.70, 30),
        (2.70, 2.90, 40),
        (2.90, 3.20, 40),
        (3.20, 3.60, 35),
        (3.60, 4.20, 35),
        (4.20, 5.00, 30),
        (5.00, 6.50, 25),
    ]
    for sname, frac in (("p25_4096", 0.6), ("p25_512", 0.15), ("base_4096", 0.25)):
        s = snaps[sname]
        for lo, hi, cnt in bins:
            m = int(round(cnt * frac))
            got = 0
            while got < m:
                i = int(rng.integers(len(s.O)))
                r, _ = s.neighbours(i)
                cand = np.nonzero((r >= lo) & (r < hi))[0]
                if len(cand) == 0:
                    continue
                j = int(rng.choice(cand))
                Y = rigidify(s.cluster([i, j]))
                jobs = DIMER_JOBS + ["mp2grad:aug-cc-pvtz"]
                records.append(rec(f"liquid2/{sname}/{i}-{j}", "liquid2", Y, jobs, snapshot=sname, R_OO=roo(Y)))
                got += 1
    return records


def liquid_clusters(snaps: dict, rng: np.random.Generator) -> list[dict]:
    """Return trimers (first shell, and first + second shell), tetramers and pentamers cut from the snapshots.

    First shell: O-O < 3.3 A; second shell 3.3-5.0 A; pentamers are a molecule and its four nearest
    neighbours.
    """
    clusters = []
    for sname, n3, n3x, n4, n5 in (("p25_4096", 30, 8, 8, 5), ("base_4096", 15, 4, 4, 3)):
        s = snaps[sname]
        made = 0
        while made < n3 + n3x + n4 + n5:
            i = int(rng.integers(len(s.O)))
            r, _ = s.neighbours(i)
            first = np.nonzero(r < 3.3)[0]
            if made < n3:
                if len(first) < 2:
                    continue
                idx, kind = [i] + list(rng.choice(first, 2, replace=False)), "liquid3"
            elif made < n3 + n3x:
                second = np.nonzero((r >= 3.3) & (r < 5.0))[0]
                if len(first) < 1 or len(second) < 1:
                    continue
                idx, kind = [i, int(rng.choice(first)), int(rng.choice(second))], "liquid3"
            elif made < n3 + n3x + n4:
                if len(first) < 3:
                    continue
                idx, kind = [i] + list(rng.choice(first, 3, replace=False)), "liquid4"
            else:
                idx, kind = [i] + list(np.argsort(r)[:4]), "liquid5"
            idx = [int(k) for k in idx]
            Y = rigidify(s.cluster(idx))
            clusters.append(rec(f"{kind}/{sname}/" + "-".join(map(str, idx)), kind, Y, CLUSTER_JOBS, snapshot=sname))
            made += 1
    return clusters


def water27_records() -> tuple[list[dict], list[dict]]:
    """Return the WATER27 dimer record(s) and cluster records (monomers rigidified).

    Clusters up to 6 molecules get the 3-body many-body expansion, the octamers only the 1-body
    terms and the whole cluster.
    """
    with open(repo_path("data", "qm", "water27_raw.json")) as fh:
        W27 = json.load(fh)
    labels = {
        "H2O2": "dimer",
        "H2O3": "trimer",
        "H2O4": "tetramer",
        "H2O5": "pentamer",
        "H2O6": "prism",
        "H2O6c": "cage",
        "H2O6b": "book",
        "H2O6c2": "cyclic",
        "H2O8d2d": "octamer_D2d",
        "H2O8s4": "octamer_S4",
    }
    dimers, clusters = [], []
    for k, lab in labels.items():
        Y = rigidify(group_waters(W27[k]))
        n = len(Y) // 3
        if n == 2:
            dimers.append(rec(f"water27/{k}", "water27", Y, DIMER_JOBS, label=lab))
        else:
            jobs = ["mbe:mp2:aug-cc-pvtz:3"] if n <= 6 else ["mbe:mp2:aug-cc-pvtz:1"]
            clusters.append(rec(f"water27/{k}", "water27", Y, jobs, label=lab))
    return dimers, clusters


def pair_records(clusters: list[dict]) -> list[dict]:
    """Return every pair of every cluster (the 2-body corrections)."""
    records = []
    for c in clusters:
        X = np.asarray(c["xyz_A"])
        for i, j in itertools.combinations(range(c["n"]), 2):
            Y = np.concatenate([X[3 * i : 3 * i + 3], X[3 * j : 3 * j + 3]])
            records.append(rec(f"{c['id']}/p{i}-{j}", "pairs", Y, DIMER_JOBS, parent=c["id"], pair=[i, j], R_OO=roo(Y)))
    return records


def check_records(records: list[dict]) -> None:
    """Check unique ids and rigid monomers; store the shortest intermolecular distance as meta d_min [A].

    Raises
    ------
    AssertionError
        Duplicate ids or a monomer that is not rigid.
    """
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in records:  # sanity: rigid monomers, no clashes
        X = np.asarray(r["xyz_A"])
        W = X.reshape(-1, 3, 3)
        assert np.allclose(np.linalg.norm(W[:, 1] - W[:, 0], axis=1), R_OH, atol=1e-6)
        dmin = min(
            (
                np.linalg.norm(W[i][:, None] - W[j][None], axis=-1).min()
                for i, j in itertools.combinations(range(len(W)), 2)
            ),
            default=9,
        )
        r["meta"]["d_min"] = round(float(dmin), 4)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, build all records and write the geometry file (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smith", default=resource("qmdata", "smith"), help="directory of the smith_opt.py JSON files")
    ap.add_argument("-o", "--out", default=repo_path("data", "qm", "water_geoms.json"), help="geometry file")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(20260928)
    records = dimer_records(a.smith)
    snaps = {k: Snapshot(k, p) for k, p in SNAPSHOTS.items()}
    records += liquid_pairs(snaps, rng)
    clusters = liquid_clusters(snaps, rng)
    dimers, w27 = water27_records()
    records += dimers
    clusters += w27
    records += clusters
    records += pair_records(clusters)
    check_records(records)
    out = {
        "about": "water clusters, rigid pGM3P-25 monomers (r_OH 0.9745 A, HOH 103.64 deg); coordinates in Angstrom, "
        "atoms O,H,H per molecule; built by scripts/qm/build_water_clusters.py",
        "records": records,
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(Counter(r["set"] for r in records), len(records))
    print("tasks", Counter(j for r in records for j in r["jobs"]))
    print("min contact", min(r["meta"]["d_min"] for r in records))


if __name__ == "__main__":
    main()
