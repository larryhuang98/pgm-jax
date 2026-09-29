"""Small example molecules and boxes: a pGM-like water, a methanol, a water / methanol / water gas
cluster, water lattices and a skewed triclinic water / methanol box.  Used by the tests, the
regression harness, examples and validation scripts (made-up but physically sensible parameters;
nm, e, e nm, nm^3, sqrt(kJ/mol))."""

from __future__ import annotations

import numpy as np

from ..md.box import reduce_box
from ..system import Molecule, System

METHANOL_BONDS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)]


def water():

    return Molecule(
        "WAT",
        ["O", "H", "H"],
        ["OW", "HW", "HW"],
        np.array([-0.8, 0.4, 0.4]),
        np.array([0.06, 0.05, 0.05]),
        np.array([1.0e-3, 0.3e-3, 0.3e-3]),
        cov=[(0, 1, -0.02), (0, 2, -0.02), (1, 0, 0.008), (2, 0, 0.008)],
        lj_rmin_half=[0.178, 0.0, 0.0],
        lj_sqrt_eps=[0.80, 0.0, 0.0],
        bonds=[(0, 1), (0, 2)],
    )


def methanol():

    x = (
        np.array(
            [
                [-0.0467, 0.6590, 0.0],
                [-0.0467, -0.7598, 0.0],
                [-1.0830, 0.9930, 0.0],
                [0.4406, 1.0735, 0.8902],
                [0.4406, 1.0735, -0.8902],
                [0.8785, -1.0591, 0.0],
            ]
        )
        * 0.1
    )
    m = Molecule(
        "MeOH",
        ["C", "O", "H", "H", "H", "H"],
        ["c3", "oh", "h1", "h1", "h1", "ho"],
        np.array([0.12, -0.62, 0.02, 0.02, 0.02, 0.44]),
        np.array([0.07, 0.06, 0.05, 0.05, 0.05, 0.05]),
        np.array([1.2e-3, 0.8e-3, 0.4e-3, 0.4e-3, 0.4e-3, 0.3e-3]),
        cov=[(0, 1, 0.01), (1, 0, -0.01), (1, 5, -0.02), (5, 1, 0.005)]
        + [c for h in (2, 3, 4) for c in ((0, h, 0.002), (h, 0, -0.002))],
        lj_rmin_half=[0.19, 0.172, 0.139, 0.139, 0.139, 0.02],
        lj_sqrt_eps=[0.33, 0.85, 0.20, 0.20, 0.20, 0.1],
        bonds=METHANOL_BONDS,
    )
    return m, x


def water_geometry(r=0.09572, theta_deg=104.52):
    t = np.radians(theta_deg / 2)
    return np.array([[0, 0, 0], [r * np.sin(t), r * np.cos(t), 0], [-r * np.sin(t), r * np.cos(t), 0]])


def _rot(rng):
    return np.linalg.qr(rng.normal(size=(3, 3)))[0]


def cluster(seed=0):
    """Gas phase: water + methanol + water, ~0.3 nm apart (nm)."""

    rng = np.random.default_rng(seed)
    w = water_geometry()
    m, xm = methanol()
    pos = np.concatenate([w, (xm - xm.mean(0)) @ _rot(rng).T + [0.33, 0.05, 0.0], w @ _rot(rng).T + [0.12, 0.30, 0.08]])
    return System([water(), m, water()]), pos


def water_lattice(n_side=4, spacing=0.31, seed=0, geometry=None):
    """n_side^3 randomly rotated waters on a cubic lattice: positions (nm), cubic box, geometry."""
    rng = np.random.default_rng(seed)
    w = water_geometry() if geometry is None else np.asarray(geometry, float)
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                pos.append(w @ Q.T + (np.array([i, j, k]) + 0.5) * spacing)
    return np.concatenate(pos), np.eye(3) * n_side * spacing, w


def small_box(seed=0, nw=30, nm=4):
    """Waters and methanols on a jittered lattice in a skewed (reduced) triclinic box, nm."""

    rng = np.random.default_rng(seed)
    H = reduce_box(np.array([[1.75, 0.0, 0.0], [0.45, 1.70, 0.0], [-0.40, 0.50, 1.65]]))
    w = water_geometry()
    m, xm = methanol()
    xm = xm - xm.mean(0)
    W = water()
    mols, pos = [], []
    grid = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(3)], float)
    grid = (grid + 0.5) / np.array([4, 4, 3])
    for n, f in enumerate(grid[rng.permutation(len(grid))][: nw + nm]):
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        c = f @ H + rng.normal(scale=0.01, size=3)
        if n < nm:
            mols.append(m)
            pos.append(xm @ R.T + c)
        else:
            mols.append(W)
            pos.append((w - w.mean(0)) @ R.T + c)
    return System(mols), np.concatenate(pos), H


def water_cluster_box():
    """Eight waters in a 3 nm box (cutoff 1.2 nm: no pair crosses the cutoff)."""
    pos, _, w = water_lattice(n_side=2, spacing=0.31)
    return pos + 1.2, np.eye(3) * 3.0, w
