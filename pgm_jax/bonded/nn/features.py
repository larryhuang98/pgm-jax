"""Graph inputs of the neural bonded model: atom and bond features, reference values of bonds and
angles, residues.  Everything here depends on the molecule only (not on the network)."""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from .. import terms as T
from ..topology import _ring_bonds

ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
N_FEAT = len(ELEMENTS) + 4 + 3 + 4          # element, degree 1-4, bond-order sum / ring / aromatic, pGM
N_EDGE = 4                                   # bond order 1, 1.5, 2, 3
# Pyykko covalent radii (A): single, double, triple
RCOV = {"H": (0.32, 0.32, 0.32), "C": (0.75, 0.67, 0.60), "N": (0.71, 0.60, 0.54), "O": (0.63, 0.57, 0.53),
        "F": (0.64, 0.59, 0.53), "P": (1.11, 1.02, 0.94), "S": (1.03, 0.94, 0.95), "Cl": (0.99, 0.95, 0.93),
        "Br": (1.14, 1.09, 1.10), "I": (1.33, 1.29, 1.25)}


def rcov(e: str, order: float) -> float:
    r1, r2, r3 = RCOV.get(e, (0.9, 0.85, 0.8))
    if order <= 1:
        return r1
    if order < 2:
        return r1 + (order - 1) * (r2 - r1)
    return r2 if order < 3 else r3


def bond_orders(spec) -> dict:
    return {tuple(sorted(b)): float(o) for b, o in zip(spec.bonds, spec.bond_orders)}


def graph_inputs(spec, top, ref: str = "geometry", pgm_features: bool = True) -> dict:
    """Atom features X (n, N_FEAT), directed edges (src, dst) with bond-order features ef, bond and
    angle reference values (nm, rad), residues (Topology.residue) of one molecule."""
    n = len(spec.elements)
    order = bond_orders(spec)
    nbr = [[] for _ in range(n)]
    for i, j in top.bonds:
        nbr[i].append(j); nbr[j].append(i)
    bo_sum = np.zeros(n)
    for (i, j), o in order.items():
        bo_sum[i] += o; bo_sum[j] += o
    ring = np.zeros(n)
    for rb in _ring_bonds([tuple(x) for x in top.bonds], nbr):
        for x in rb:
            ring[x] = 1.0
    pg = spec.pgm
    cov_abs = np.zeros(n)
    if pg is not None:
        for i, j, c in pg.cov:
            cov_abs[i] += abs(c)
    X = []
    for i, e in enumerate(spec.elements):
        f = [1.0 if e == x else 0.0 for x in ELEMENTS] + [1.0 if len(nbr[i]) == d else 0.0 for d in (1, 2, 3, 4)]
        f += [bo_sum[i] / 4.0, ring[i],
              1.0 if any(abs(order.get(tuple(sorted((i, j))), 1) - 1.5) < 1e-6 for j in nbr[i]) else 0.0]
        f += [float(pg.q[i]), float(pg.alpha[i]) * 1e3, float(pg.radius[i]) * 10.0, cov_abs[i] * 50.0] \
            if (pg is not None and pgm_features) else [0.0] * 4
        X.append(f)
    src = np.concatenate([top.bonds[:, 0], top.bonds[:, 1]])
    dst = np.concatenate([top.bonds[:, 1], top.bonds[:, 0]])
    bo = np.array([order.get(tuple(sorted(b)), 1.0) for b in top.bonds])
    ef = np.stack([bo == 1, np.abs(bo - 1.5) < 1e-6, bo == 2, bo == 3], -1).astype(float)
    ef = np.concatenate([ef, ef])
    if ref == "geometry":
        G = T.geometry(jnp.asarray(spec.ref_xyz), top)
        b_ref, th_ref = np.asarray(G["b"]), np.asarray(G["th"])
    elif ref == "predicted":
        b_ref = np.array([(rcov(spec.elements[i], bo[k]) + rcov(spec.elements[j], bo[k])) * 0.1
                          for k, (i, j) in enumerate(top.bonds)])
        tet = math.acos(-1 / 3)
        th_ref = np.array([math.pi if (len(nbr[j]) == 2 and bo_sum[j] >= 3.5) else
                           (2 * math.pi / 3 if (len(nbr[j]) == 3 and bo_sum[j] >= 3.5) else tet)
                           for i, j, k in top.angles])
    else:
        raise ValueError("ref: geometry | predicted")
    residue = np.asarray(top.residue if getattr(top, "residue", None) is not None else np.zeros(n, int))
    De = np.asarray([T.morse_depth(spec.elements[i], spec.elements[j], order.get(tuple(sorted((i, j))), 1))
                     for i, j in top.bonds])
    return {"X": np.asarray(X, float), "src": src, "dst": dst, "ef": ef, "n": n, "top": top,
            "bonds": np.asarray(top.bonds), "angles": np.asarray(top.angles).reshape(-1, 3),
            "b_ref": b_ref, "th_ref": th_ref, "residue": residue, "n_res": int(residue.max()) + 1 if n else 0,
            "De": De}
