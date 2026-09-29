"""Geometries of the water-cluster QM set (rigid monomers at the pGM3P-25 geometry).

Every monomer is replaced by the rigid water of the model (r_OH, HOH angle of the pGM3P-25
prmtop / restart: 0.9745 A, 103.64 deg), superposed on the source monomer (mass-weighted Kabsch,
centre of mass kept), so model and QM see the same geometry and the interaction energies contain
no monomer deformation.

Sets (record "set"):
  smith        symmetry-constrained MP2/aug-cc-pVDZ stationary structures of the dimer
               (scripts/qmfit/smith_opt.py; ~/project/qmdata/smith/*.json)
  radial       O-O scans of four of them (the partner translated along O...O)
  angular      scans around the minimum: acceptor flap, donor bend, acceptor twist
  liquid2      pairs from pGM liquid snapshots, stratified in R_OO
  water27      the (H2O)n, n = 2..6 and 8 clusters of WATER27 (GMTKN55; Bryantsev et al.
               JCTC 2009, 5, 1016), monomers rigidified
  liquid3/4/5  trimers, tetramers, pentamers cut from the snapshots (a molecule and neighbours)
  pairs        every pair of every cluster with n >= 3 (2-body corrections to the cluster MP2)

    python scripts/qmfit/build_water_clusters.py [--smith DIR] [--out data/qm/water_geoms.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R_OH, HOH = 0.9745, 103.64  # pGM3P-25 rigid water (p25_512.rst7), Angstrom / degrees
MASS = np.array([15.999, 1.008, 1.008])
SNAPSHOTS = {
    "p25_4096": "/home8/larry/project/epsp/p25_4096.rst7",
    "p25_512": "/home8/larry/project/epsp/p25_512.rst7",
    "base_4096": "/home8/larry/project/epsp/base/base_4096.rst7",
}
DIMER_JOBS = ["sapt0", "mp2:aug-cc-pvtz", "mp2:aug-cc-pvqz", "ccsdt:aug-cc-pvtz"]


def model_water():
    t = np.radians(HOH) / 2
    return np.array([[0, 0, 0], [R_OH * np.sin(t), R_OH * np.cos(t), 0], [-R_OH * np.sin(t), R_OH * np.cos(t), 0]])


def kabsch(P, Q, w):
    """Rotation Rm and translation minimizing sum w |Rm P + t - Q|^2."""
    pc, qc = (w[:, None] * P).sum(0) / w.sum(), (w[:, None] * Q).sum(0) / w.sum()
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rm, qc - Rm @ pc


def rigidify(X):
    """(3n, 3) O,H,H per molecule -> the same with the model's rigid water superposed on each."""
    X = np.asarray(X, float).reshape(-1, 3, 3)
    M = model_water()
    out = []
    for w in X:
        Rm, t = kabsch(M, w, MASS)
        out.append(M @ Rm.T + t)
    return np.concatenate(out)


def group_waters(atoms):
    """[[el, x, y, z], ...] with any atom order -> (3n, 3) O,H,H per molecule (H to the nearest O)."""
    el = [a[0] for a in atoms]
    X = np.array([a[1:] for a in atoms], float)
    O = [k for k, e in enumerate(el) if e == "O"]
    H = [k for k, e in enumerate(el) if e == "H"]
    own = {o: [] for o in O}
    for h in H:
        own[min(O, key=lambda o: np.linalg.norm(X[h] - X[o]))].append(h)
    assert all(len(v) == 2 for v in own.values())
    return np.concatenate([X[[o] + own[o]] for o in O])


def rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def read_rst7(path):
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


IMAGES = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)


def min_image(d, H, Hinv):
    """Shortest periodic image of displacement(s) d (..., 3)."""
    f = d @ Hinv
    d0 = (f - np.round(f)) @ H
    cand = d0[..., None, :] + IMAGES @ H
    k = np.argmin(np.linalg.norm(cand, axis=-1), axis=-1)
    return np.take_along_axis(cand, k[..., None, None], axis=-2)[..., 0, :]


class Snapshot:
    def __init__(self, name, path):
        X, H = read_rst7(path)
        self.name, self.H, self.Hinv = name, H, np.linalg.inv(H)
        W = X.reshape(-1, 3, 3)
        W[:, 1:] = W[:, :1] + min_image(W[:, 1:] - W[:, :1], H, self.Hinv)  # whole molecules
        self.W = W
        self.O = W[:, 0]

    def neighbours(self, i):
        d = min_image(self.O - self.O[i], self.H, self.Hinv)
        r = np.linalg.norm(d, axis=-1)
        r[i] = np.inf
        return r, d

    def cluster(self, idx):
        """Molecules idx imaged around molecule idx[0], (3n, 3), centred."""
        c = self.O[idx[0]]
        out = []
        for k in idx:
            sh = min_image(self.O[k] - c, self.H, self.Hinv) - (self.O[k] - c)
            out.append(self.W[k] + sh)
        X = np.concatenate(out)
        return X - X.mean(0)


def rec(rid, set_, X, jobs, **meta):
    X = np.asarray(X)
    return {
        "id": rid,
        "set": set_,
        "n": len(X) // 3,
        "xyz_A": np.round(X, 8).tolist(),
        "jobs": list(jobs),
        "meta": meta,
    }


def roo(X, i=0, j=1):
    return float(np.linalg.norm(X[3 * i] - X[3 * j]))


def main(a):
    rng = np.random.default_rng(20260928)
    records = [rec("monomer/p25", "monomer", model_water(), ["props:ccsd:aug-cc-pvtz"])]
    # ---- Smith-type stationary structures and scans
    smith = {}
    for fn in sorted(os.listdir(a.smith)):
        if fn.endswith(".json"):
            d = json.load(open(os.path.join(a.smith, fn)))
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
    # ---- liquid snapshots
    snaps = {k: Snapshot(k, p) for k, p in SNAPSHOTS.items()}
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
    cl_jobs = ["mbe:mp2:aug-cc-pvtz:3"]
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
            clusters.append(rec(f"{kind}/{sname}/" + "-".join(map(str, idx)), kind, Y, cl_jobs, snapshot=sname))
            made += 1
    W27 = json.load(open(os.path.join(ROOT, "data/qm/water27_raw.json")))
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
    for k, lab in labels.items():
        Y = rigidify(group_waters(W27[k]))
        n = len(Y) // 3
        if n == 2:
            records.append(rec(f"water27/{k}", "water27", Y, DIMER_JOBS, label=lab))
        else:
            jobs = ["mbe:mp2:aug-cc-pvtz:3"] if n <= 6 else ["mbe:mp2:aug-cc-pvtz:1"]
            clusters.append(rec(f"water27/{k}", "water27", Y, jobs, label=lab))
    records += clusters
    for c in clusters:  # 2-body corrections
        X = np.asarray(c["xyz_A"])
        for i, j in itertools.combinations(range(c["n"]), 2):
            Y = np.concatenate([X[3 * i : 3 * i + 3], X[3 * j : 3 * j + 3]])
            records.append(rec(f"{c['id']}/p{i}-{j}", "pairs", Y, DIMER_JOBS, parent=c["id"], pair=[i, j], R_OO=roo(Y)))
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
    out = {
        "about": "water clusters, rigid pGM3P-25 monomers (r_OH 0.9745 A, HOH 103.64 deg); coordinates in Angstrom, "
        "atoms O,H,H per molecule; built by scripts/qmfit/build_water_clusters.py",
        "records": records,
    }
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    from collections import Counter

    print(Counter(r["set"] for r in records), len(records))
    print("tasks", Counter(j for r in records for j in r["jobs"]))
    print("min contact", min(r["meta"]["d_min"] for r in records))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smith", default=os.path.expanduser("~/project/qmdata/smith"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/qm/water_geoms.json"))
    main(ap.parse_args())
