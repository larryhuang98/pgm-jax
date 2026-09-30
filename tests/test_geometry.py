"""Rigid geometry of the prmtop (md/geometry.py, Simulation.from_amber conform_geometry)."""

import numpy as np
import pytest

from pgm_jax.md.geometry import conform_rigid_geometry
from pgm_jax.md.io import read_coordinates_nm
from pgm_jax.paths import pgm3p25_files

TOP, CRD = pgm3p25_files()[:2]


def _oh(pos):
    """Return the three bond lengths (O-H1, O-H2, H1-H2) of every rigid water, shape (N/3,) each."""
    p = np.asarray(pos).reshape(-1, 3, 3)
    return np.linalg.norm(p[:, 1] - p[:, 0], axis=1), np.linalg.norm(p[:, 2] - p[:, 0], axis=1), np.linalg.norm(p[:, 2] - p[:, 1], axis=1)


def test_conform_restores_bond_lengths():
    """Stretched O-H bonds are rebuilt at the prmtop's lengths, the O atoms and the plane are kept."""
    pos, _, _ = read_coordinates_nm(CRD)
    ref, n0 = conform_rigid_geometry(TOP, pos)
    p = np.array(ref).reshape(-1, 3, 3)
    p[:, 1] = p[:, 0] + 1.03 * (p[:, 1] - p[:, 0])
    p[:, 2] = p[:, 0] + 0.98 * (p[:, 2] - p[:, 0])
    new, n = conform_rigid_geometry(TOP, p.reshape(-1, 3))
    assert n == len(p)
    a, b, c = _oh(new)
    a0, b0, c0 = _oh(ref)
    np.testing.assert_allclose(a, a0, atol=1e-9)
    np.testing.assert_allclose(b, b0, atol=1e-9)
    np.testing.assert_allclose(c, c0, atol=1e-9)
    np.testing.assert_allclose(new.reshape(-1, 3, 3)[:, 0], p[:, 0])
    again, m = conform_rigid_geometry(TOP, new)
    assert m == 0
