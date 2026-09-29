"""Build small example molecules and boxes for tests, the regression harness, examples and validation.

Contents: water and methanol (Molecule templates with made-up but physically sensible
parameters), water_geometry, cluster (a water / methanol / water gas cluster), water_lattice
(waters on a cubic lattice), small_box (a skewed triclinic water / methanol box),
water_cluster_box (eight waters in a large box), METHANOL_BONDS.

The parameters are not fitted to anything: they only need to exercise every term (charges,
covalent dipoles in both directions, polarizabilities, LJ on some atoms).  Changing them changes
the golden files of the regression harness.

Units: nm, e, e nm, nm^3, sqrt(kJ/mol).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..md.box import reduce_box
from ..system import Molecule, System

METHANOL_BONDS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)]


def water() -> Molecule:
    """Return a pGM-like water template "WAT" (O, H, H; types OW, HW, HW).

    Charges -0.8/0.4 e, radii 0.06/0.05 nm, polarizabilities 1.0e-3/0.3e-3 nm^3, covalent dipoles
    O->H (-0.02 e nm) and H->O (0.008 e nm), LJ on O only (R* 0.178 nm, sqrt(eps) 0.80).
    """
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


def methanol() -> tuple[Molecule, np.ndarray]:
    """Return a methanol template "MeOH" (C, O, H, H, H, H; GAFF-like types) and a geometry (6, 3) [nm].

    Covalent dipoles along C-O, O-H and C-H in both directions, LJ on every atom; the geometry is a
    standard staggered methanol.
    """
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


def water_geometry(r: float = 0.09572, theta_deg: float = 104.52) -> np.ndarray:
    """Return O, H, H coordinates (3, 3) [nm] of a water, O at the origin, bisector along +y, in the xy plane.

    `r` is the O-H length [nm] and `theta_deg` the H-O-H angle [degrees] (defaults: TIP3P geometry).
    """
    t = np.radians(theta_deg / 2)
    return np.array([[0, 0, 0], [r * np.sin(t), r * np.cos(t), 0], [-r * np.sin(t), r * np.cos(t), 0]])


def _rot(rng: np.random.Generator) -> np.ndarray:
    """Return a random orthogonal matrix (3, 3) (QR of a Gaussian matrix; may be improper)."""
    return np.linalg.qr(rng.normal(size=(3, 3)))[0]


def cluster(seed: int = 0) -> tuple[System, np.ndarray]:
    """Return a gas-phase water + methanol + water cluster: its System and positions (12, 3) [nm].

    The molecules are about 0.3 nm apart, with random orientations from `seed`.
    """
    rng = np.random.default_rng(seed)
    w = water_geometry()
    m, xm = methanol()
    pos = np.concatenate([w, (xm - xm.mean(0)) @ _rot(rng).T + [0.33, 0.05, 0.0], w @ _rot(rng).T + [0.12, 0.30, 0.08]])
    return System([water(), m, water()]), pos


def water_lattice(
    n_side: int = 4, spacing: float = 0.31, seed: int = 0, geometry: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return n_side^3 randomly rotated waters on a cubic lattice.

    Parameters
    ----------
    n_side : int
        Waters per box edge.
    spacing : float
        Lattice spacing [nm].
    seed : int
        Seed of the orientations.
    geometry : ArrayLike (3, 3), optional
        Water geometry [nm]; None: water_geometry().

    Returns
    -------
    positions : np.ndarray (3 n_side^3, 3)
        [nm], the O of each water at a lattice point (cell centres).
    box : np.ndarray (3, 3)
        Cubic box [nm] of edge n_side spacing.
    geometry : np.ndarray (3, 3)
        The water geometry used [nm].
    """
    rng = np.random.default_rng(seed)
    w = water_geometry() if geometry is None else np.asarray(geometry, float)
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                pos.append(w @ Q.T + (np.array([i, j, k]) + 0.5) * spacing)
    return np.concatenate(pos), np.eye(3) * n_side * spacing, w


def small_box(seed: int = 0, nw: int = 30, nm: int = 4) -> tuple[System, np.ndarray, np.ndarray]:
    """Return waters and methanols on a jittered lattice in a skewed (reduced) triclinic box.

    Parameters
    ----------
    seed : int
        Seed of the placement and orientations.
    nw, nm : int
        Numbers of waters and methanols (nw + nm <= 48 lattice sites).

    Returns
    -------
    system : System
        Methanols first, then waters.
    positions : np.ndarray (N, 3)
        [nm], molecules centred on the lattice sites (0.01 nm jitter).
    box : np.ndarray (3, 3)
        Triclinic box [nm], lattice vectors as rows.
    """
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


def water_cluster_box() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eight waters in a 3 nm cubic box: positions (24, 3) [nm], box (3, 3) [nm], water geometry.

    With a 1.2 nm cutoff no pair crosses the cutoff (a periodic model that equals the gas phase
    up to the Ewald images).
    """
    pos, _, w = water_lattice(n_side=2, spacing=0.31)
    return pos + 1.2, np.eye(3) * 3.0, w
