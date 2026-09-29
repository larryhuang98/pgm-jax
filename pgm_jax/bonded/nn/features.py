"""Build the graph inputs of the neural bonded model: atom and bond features, reference values, residues.

Everything here depends on the molecule only (not on the network).  Atom features (N_FEAT):
one-hot element (ELEMENTS), one-hot degree 1-4, bond-order sum / 4, ring flag, aromatic flag
(a bond of order 1.5), and four pGM parameters (charge, polarizability, Gaussian radius, sum of
|covalent dipoles|, scaled to O(1); zeros without pGM parameters or with pgm_features=False).
Edge features (N_EDGE): one-hot bond order 1, 1.5, 2, 3.

Units: reference values nm and rad; RCOV in A (Pyykko covalent radii).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from .. import terms as T
from ..topology import _ring_bonds

if TYPE_CHECKING:
    from ..model import MolSpec
    from ..topology import Topology

ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
N_FEAT = len(ELEMENTS) + 4 + 3 + 4  # element, degree 1-4, bond-order sum / ring / aromatic, pGM
N_EDGE = 4  # bond order 1, 1.5, 2, 3
# Pyykko covalent radii (A): single, double, triple
RCOV = {
    "H": (0.32, 0.32, 0.32),
    "C": (0.75, 0.67, 0.60),
    "N": (0.71, 0.60, 0.54),
    "O": (0.63, 0.57, 0.53),
    "F": (0.64, 0.59, 0.53),
    "P": (1.11, 1.02, 0.94),
    "S": (1.03, 0.94, 0.95),
    "Cl": (0.99, 0.95, 0.93),
    "Br": (1.14, 1.09, 1.10),
    "I": (1.33, 1.29, 1.25),
}


def rcov(e: str, order: float) -> float:
    """Return the covalent radius [A] of element `e` in a bond of order `order`.

    Single, double, triple radii from RCOV (0.9, 0.85, 0.8 A for elements not in the table);
    orders between 1 and 2 interpolate linearly (aromatic 1.5).
    """
    r1, r2, r3 = RCOV.get(e, (0.9, 0.85, 0.8))
    if order <= 1:
        return r1
    if order < 2:
        return r1 + (order - 1) * (r2 - r1)
    return r2 if order < 3 else r3


def bond_orders(spec: MolSpec) -> dict:
    """Return {sorted atom pair: bond order} of a MolSpec."""
    return {tuple(sorted(b)): float(o) for b, o in zip(spec.bonds, spec.bond_orders)}


def graph_inputs(spec: MolSpec, top: Topology, ref: str = "geometry", pgm_features: bool = True) -> dict:
    """Return the graph inputs of one molecule.

    Parameters
    ----------
    spec : MolSpec
        The molecule (pGM parameters optional).
    top : Topology
        Its topology.
    ref : {"geometry", "predicted"}
        Reference values from the minimum geometry `spec.ref_xyz`, or predicted from covalent
        radii (bonds) and hybridisation (angles: 180 deg at 2-coordinated centres and 120 deg at
        3-coordinated centres with a bond-order sum >= 3.5, else tetrahedral).
    pgm_features : bool
        Include the pGM atom features.

    Returns
    -------
    dict
        "X" (n, N_FEAT) atom features, "src", "dst" (2 nb,) directed edges, "ef" (2 nb, N_EDGE)
        edge features, "n", "top", "bonds" (nb, 2), "angles" (na, 3), "b_ref" (nb,) [nm],
        "th_ref" (na,) [rad], "residue" (n,), "n_res", "De" (nb,) Morse depths [kJ/mol].

    Raises
    ------
    ValueError
        For an unknown `ref`.
    """
    n = len(spec.elements)
    order = bond_orders(spec)
    nbr = [[] for _ in range(n)]
    for i, j in top.bonds:
        nbr[i].append(j)
        nbr[j].append(i)
    bo_sum = np.zeros(n)
    for (i, j), o in order.items():
        bo_sum[i] += o
        bo_sum[j] += o
    ring = np.zeros(n)
    for rb in _ring_bonds([tuple(x) for x in top.bonds], nbr):
        for x in rb:
            ring[x] = 1.0
    pg = spec.pgm
    cov_abs = np.zeros(n)
    if pg is not None:
        for i, _j, c in pg.cov:
            cov_abs[i] += abs(c)
    X = []
    for i, e in enumerate(spec.elements):
        f = [1.0 if e == x else 0.0 for x in ELEMENTS] + [1.0 if len(nbr[i]) == d else 0.0 for d in (1, 2, 3, 4)]
        f += [
            bo_sum[i] / 4.0,
            ring[i],
            1.0 if any(abs(order.get(tuple(sorted((i, j))), 1) - 1.5) < 1e-6 for j in nbr[i]) else 0.0,
        ]
        f += (
            # pGM features scaled to O(1): q [e], alpha [nm^3] x 1e3, radius [nm] x 10, |covalent dipoles| [e nm] x 50
            [float(pg.q[i]), float(pg.alpha[i]) * 1e3, float(pg.radius[i]) * 10.0, cov_abs[i] * 50.0]
            if (pg is not None and pgm_features)
            else [0.0] * 4
        )
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
        b_ref = np.array(
            [
                (rcov(spec.elements[i], bo[k]) + rcov(spec.elements[j], bo[k])) * 0.1
                for k, (i, j) in enumerate(top.bonds)
            ]
        )
        tet = math.acos(-1 / 3)  # tetrahedral angle; 120 deg (sp2) and 180 deg (sp) below
        th_ref = np.array(
            [
                math.pi
                if (len(nbr[j]) == 2 and bo_sum[j] >= 3.5)
                else (2 * math.pi / 3 if (len(nbr[j]) == 3 and bo_sum[j] >= 3.5) else tet)
                for i, j, k in top.angles
            ]
        )
    else:
        raise ValueError("ref: geometry | predicted")
    residue = np.asarray(top.residue if getattr(top, "residue", None) is not None else np.zeros(n, int))
    De = np.asarray(
        [T.morse_depth(spec.elements[i], spec.elements[j], order.get(tuple(sorted((i, j))), 1)) for i, j in top.bonds]
    )
    return {
        "X": np.asarray(X, float),
        "src": src,
        "dst": dst,
        "ef": ef,
        "n": n,
        "top": top,
        "bonds": np.asarray(top.bonds),
        "angles": np.asarray(top.angles).reshape(-1, 3),
        "b_ref": b_ref,
        "th_ref": th_ref,
        "residue": residue,
        "n_res": int(residue.max()) + 1 if n else 0,
        "De": De,
    }
