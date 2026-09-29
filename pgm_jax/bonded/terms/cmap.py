"""Backbone correction map (CMAP): a smooth 2D function of the phi/psi torsions of every residue.

    E = sum_(m, n) [a_mn cos(m phi) cos(n psi) + b_mn cos(m phi) sin(n psi)
                   + c_mn sin(m phi) cos(n psi) + d_mn sin(m phi) sin(n psi)],   0 <= m, n <= M,

without the constant, (2M + 1)^2 - 1 coefficients per instance (48 for M = 3).  A Fourier series
instead of Amber's bicubic grid: smooth to every order (forces, second derivatives), few
coefficients for a network or a fit to determine, and it tabulates onto Amber's 24 x 24 grid for
export (`cmap_grid`; bonded/amber.py).  Order 3 ("cmap") suits maps fitted to QM data; existing
Amber maps are rougher (ff19SB's alanine map: rms 0.43 kcal/mol from order 3, 0.19 from order 6
("cmap6"), 0.06 from order 11), so imports of those are approximate.  The instances are the phi/psi quintuples
C(i-1), N, CA, C, N(i+1) of Topology.cmaps (found from the bond graph); the tying key is the
oriented quintuple ("cmap" kind), so typed fits get one map per residue environment.
Units: kJ/mol, rad."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .core import Family, _dihedral, register

ORDER = 3


def basis_orders(order: int = ORDER):
    """(m, n, kind) of every basis function; kind 0..3 = cos cos, cos sin, sin cos, sin sin."""
    out = []
    for m in range(order + 1):
        for n in range(order + 1):
            for kind in range(4):
                if (kind in (1, 3) and n == 0) or (kind in (2, 3) and m == 0) or (m == n == 0):
                    continue
                out.append((m, n, kind))
    return out


def cmap_basis(phi, psi, order: int = ORDER):
    """(..., nbasis) basis functions at the torsions phi, psi (rad)."""
    cols = []
    for m, n, kind in basis_orders(order):
        a = jnp.cos(m * phi) if kind in (0, 1) else jnp.sin(m * phi)
        b = jnp.cos(n * psi) if kind in (0, 2) else jnp.sin(n * psi)
        cols.append(a * b)
    return jnp.stack(cols, -1)


def cmap_grid(coef, order: int = ORDER, resolution: int = 24):
    """Energies (resolution, resolution) of one map on Amber's grid: phi (rows) and psi (columns)
    from -180 deg in steps of 360/resolution, same units as coef."""
    g = -np.pi + 2.0 * np.pi * np.arange(resolution) / resolution
    P, S = np.meshgrid(g, g, indexing="ij")
    return np.asarray(cmap_basis(jnp.asarray(P), jnp.asarray(S), order) @ jnp.asarray(coef))


def phi_psi(R, q):
    """phi, psi (rad) of the quintuples q (k, 5) in frame R (n, 3)."""
    return (
        _dihedral(R[q[:, 0]], R[q[:, 1]], R[q[:, 2]], R[q[:, 3]]),
        _dihedral(R[q[:, 1]], R[q[:, 2]], R[q[:, 3]], R[q[:, 4]]),
    )


class CMAPFourier(Family):
    """Fourier backbone map of order 3 (48 coefficients); CMAPFourier6: order 6 (168)."""

    name = "cmap"
    order = ORDER
    params = {"cm": ((len(basis_orders(ORDER)),), 0.0)}
    linear = ("cm",)

    def index(self, top, keyf):
        q = getattr(top, "cmaps", None)
        q = np.zeros((0, 5), int) if q is None else np.asarray(q, int).reshape(-1, 5)
        return {"q": q}, [keyf(t, "cmap") for t in q]

    def energy(self, G, dev, I, p):
        q = I["q"]
        if len(q) == 0:
            return 0.0
        phi, psi = phi_psi(G["R"], q)
        return jnp.sum(p["cm"] * cmap_basis(phi, psi, self.order))


class CMAPFourier6(CMAPFourier):
    name = "cmap6"
    order = 6
    params = {"cm": ((len(basis_orders(6)),), 0.0)}


register(CMAPFourier)
register(CMAPFourier6)
