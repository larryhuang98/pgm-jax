"""Small, self-contained test systems for the regression harness (float64, CPU).

Copied from the unit tests on purpose: the harness must keep producing the same inputs while the
tests are reorganized.  Only numpy and the model-building API are used here (Molecule, System,
bonded templates); everything engine-specific is in regression_cases.py."""

from __future__ import annotations

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
DATA = os.path.join(TESTS, "data")
PGM3P25_TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
PGM3P25_RST = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")

METHANOL_BONDS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)]


def water():
    from pgm_jax.system import Molecule

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
    from pgm_jax.system import Molecule

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
    from pgm_jax.system import System

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
    from pgm_jax.md.box import reduce_box
    from pgm_jax.system import System

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


def pgm3p25_available() -> bool:
    return os.path.exists(PGM3P25_TOP) and os.path.exists(PGM3P25_RST)


def pgm3p25_water():
    """(Molecule, geometry nm) of pGM3P-25 water: parameters from rayl_512_v2.prmtop, geometry of
    the first water of the equilibrated 512-water restart."""
    from pgm_jax.md.io import read_coordinates
    from pgm_jax.param import read_prmtop_pgm

    m = read_prmtop_pgm(PGM3P25_TOP)[0]
    xyz, _, _ = read_coordinates(PGM3P25_RST)
    return m, np.asarray(xyz[:3], float) * 0.1


# ----------------------------------------------------------------------------- bonded templates
def methanol_template(flux: int = 0, seed: int = 3):
    """Flexible methanol: class II bonded terms at their initial values, 1-4 LJ scaled by 0.5;
    flux > 0 adds made-up charge-flux parameters of the size of a fit (order 1 or 2)."""
    import jax.numpy as jnp

    from pgm_jax.bonded import terms as T
    from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec
    from pgm_jax.md.flexible import FlexibleTemplate

    m, x = methanol()
    spec = MolSpec("methanol", list(m.elements), METHANOL_BONDS, [1] * len(METHANOL_BONDS), 0, x, m)
    kw = {"flux": flux} if flux else {}
    model = BondedModel([spec], BondedSettings(families=T.PAPER, lj14_scale=0.5, **kw))
    P = model.init_params()
    if flux:
        rng = np.random.default_rng(seed)
        nk = len(P["flux"]["jb"])
        P["flux"] = {"jb": jnp.asarray(rng.uniform(-3, 3, nk)), "jc": jnp.asarray(rng.uniform(-1, 1, nk))}
        if flux >= 2:
            P["flux"]["jc2"] = jnp.asarray(rng.uniform(-8, 8, nk))
    return FlexibleTemplate.from_fit(model, P), x


def flexible_water_template():
    """Flexible water (quartic bonds, harmonic + cubic angle, cross terms) for PIMD."""
    import jax.numpy as jnp

    from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec
    from pgm_jax.md.flexible import FlexibleTemplate
    from pgm_jax.md.pimd import WATER_FAMILIES

    m = water()
    t = math.radians(104.5)
    x = np.array([[0, 0, 0], [0.0957, 0, 0], [0.0957 * math.cos(t), 0.0957 * math.sin(t), 0]])
    spec = MolSpec("WAT", ["O", "H", "H"], [(0, 1), (0, 2)], [1, 1], 0, x, m)
    model = BondedModel([spec], BondedSettings(families=WATER_FAMILIES))
    P = model.init_params()
    P["bond_quartic"]["K2"] = jnp.full_like(P["bond_quartic"]["K2"], 4.5e5)
    P["angle_harm"]["Ka"] = jnp.full_like(P["angle_harm"]["Ka"], 350.0)
    return FlexibleTemplate.from_fit(model, P), x


def flexible_water_box(n_side=2, L=1.5, seed=0):
    tpl, x = flexible_water_template()
    rng = np.random.default_rng(seed)
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                c = (np.array([i, j, k]) + 0.5) * L / n_side + rng.normal(scale=0.02, size=3)
                pos.append((x - x.mean(0)) @ R.T + c)
    return tpl, np.concatenate(pos), np.eye(3) * L
