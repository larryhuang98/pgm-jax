"""Small, self-contained test systems for the regression harness (float64, CPU).

The example molecules and boxes come from pgm_jax.models.toy (a change there shows up as a
failure of the harness); the pGM3P-25 water and the bonded templates are built here.  Everything
engine-specific is in regression_cases.py.
"""

from __future__ import annotations

import math
import os

import numpy as np

from pgm_jax.models.toy import (  # noqa: F401  (the harness's builders)
    METHANOL_BONDS,
    cluster,
    methanol,
    small_box,
    water,
    water_cluster_box,
    water_geometry,
    water_lattice,
)
from pgm_jax.paths import resource

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
DATA = os.path.join(TESTS, "data")
PGM3P25_TOP = resource("gvdw_data", "topology/rayl_512_v2.prmtop")
PGM3P25_RST = resource("gvdw_data", "inputs/lj/inpcrd.restrt")


def pgm3p25_available() -> bool:
    """Return whether the pGM3P-25 topology and restart are present (PGM_GVDW_DATA)."""
    return os.path.exists(PGM3P25_TOP) and os.path.exists(PGM3P25_RST)


def pgm3p25_water():
    """Return the pGM3P-25 water molecule and its geometry [nm].

    (Molecule, geometry nm) of pGM3P-25 water: parameters from rayl_512_v2.prmtop, geometry of
    the first water of the equilibrated 512-water restart.
    """
    from pgm_jax.md.io import read_coordinates
    from pgm_jax.param import read_prmtop_pgm

    m = read_prmtop_pgm(PGM3P25_TOP)[0]
    xyz, _, _ = read_coordinates(PGM3P25_RST)
    return m, np.asarray(xyz[:3], float) * 0.1


# ----------------------------------------------------------------------------- bonded templates
def methanol_template(flux: int = 0, seed: int = 3):
    """Return a flexible methanol template (class II terms, optional charge flux) and its geometry.

    Flexible methanol: class II bonded terms at their initial values, 1-4 LJ scaled by 0.5;
    flux > 0 adds made-up charge-flux parameters of the size of a fit (order 1 or 2).
    """
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
    """Return a flexible water template for PIMD and its geometry [nm].

    Flexible water (quartic bonds, harmonic + cubic angle, cross terms) for PIMD.
    """
    import jax.numpy as jnp

    from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec
    from pgm_jax.md.flexible import FlexibleTemplate
    from pgm_jax.models.water import WATER_FAMILIES

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
    """Return (template, positions [nm], box [nm]) of randomly rotated flexible waters on a lattice.

    Parameters
    ----------
    n_side : int
        Waters per box edge.
    L : float
        Box edge [nm].
    seed : int
        Seed of the orientations and the 0.02 nm jitter.
    """
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
