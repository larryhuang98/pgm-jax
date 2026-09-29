"""Define the backbone correction map (CMAP): a smooth 2D Fourier function of each residue's phi/psi.

    E = sum_(m, n) [a_mn cos(m phi) cos(n psi) + b_mn cos(m phi) sin(n psi)
                   + c_mn sin(m phi) cos(n psi) + d_mn sin(m phi) sin(n psi)],   0 <= m, n <= M,

without the constant, (2M + 1)^2 - 1 coefficients per instance (48 for M = 3).  A Fourier series
instead of Amber's bicubic grid: smooth to every order (forces, second derivatives), few
coefficients for a network or a fit to determine, and it tabulates onto Amber's 24 x 24 grid for
export (`cmap_grid`; bonded/amber.py).  Order 3 ("cmap") suits maps fitted to QM data; existing
Amber maps are rougher (ff19SB's alanine map: rms 0.43 kcal/mol from order 3, 0.19 from order 6
("cmap6"), 0.06 from order 11), so imports of those are approximate.  The instances are the
phi/psi quintuples C(i-1), N, CA, C, N(i+1) of Topology.cmaps (found from the bond graph); the
tying key is the oriented quintuple ("cmap" kind), so typed fits get one map per residue
environment.

Contents: `basis_orders`, `cmap_basis`, `cmap_grid`, `phi_psi` and the families `CMAPFourier`
("cmap") and `CMAPFourier6` ("cmap6").

Units: kJ/mol, rad.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .core import Family, _dihedral, register

if TYPE_CHECKING:
    import jax

    from ..topology import Topology

ORDER = 3  # default Fourier order M of the "cmap" family (48 coefficients)


def basis_orders(order: int = ORDER) -> list[tuple[int, int, int]]:
    """Return (m, n, kind) of every basis function; kind 0..3 = cos cos, cos sin, sin cos, sin sin.

    Terms that vanish identically (sin with order 0) and the constant are left out, so there are
    (2 order + 1)^2 - 1 functions.

    Examples
    --------
    >>> len(basis_orders(3))
    48
    """
    out = []
    for m in range(order + 1):
        for n in range(order + 1):
            for kind in range(4):
                if (kind in (1, 3) and n == 0) or (kind in (2, 3) and m == 0) or (m == n == 0):
                    continue
                out.append((m, n, kind))
    return out


def cmap_basis(phi: ArrayLike, psi: ArrayLike, order: int = ORDER) -> jax.Array:
    """Return the basis functions at the torsions, jax.Array (..., nbasis), columns in `basis_orders` order.

    Parameters
    ----------
    phi, psi : ArrayLike
        Backbone torsions [rad] (broadcast together).
    order : int
        Fourier order M.
    """
    cols = []
    for m, n, kind in basis_orders(order):
        a = jnp.cos(m * phi) if kind in (0, 1) else jnp.sin(m * phi)
        b = jnp.cos(n * psi) if kind in (0, 2) else jnp.sin(n * psi)
        cols.append(a * b)
    return jnp.stack(cols, -1)


def cmap_grid(coef: ArrayLike, order: int = ORDER, resolution: int = 24) -> np.ndarray:
    """Return the energies of one map on Amber's grid, np.ndarray (resolution, resolution).

    phi along the rows and psi along the columns, both from -180 deg in steps of 360/resolution;
    same units as `coef` (kJ/mol for the family's coefficients).

    Parameters
    ----------
    coef : ArrayLike (nbasis,)
        Coefficients of the map.
    order : int
        Fourier order M of the coefficients.
    resolution : int
        Grid points per torsion (Amber: 24).
    """
    g = -np.pi + 2.0 * np.pi * np.arange(resolution) / resolution
    P, S = np.meshgrid(g, g, indexing="ij")
    return np.asarray(cmap_basis(jnp.asarray(P), jnp.asarray(S), order) @ jnp.asarray(coef))


def phi_psi(R: jax.Array, q: np.ndarray) -> tuple[jax.Array, jax.Array]:
    """Return phi = C(i-1)-N-CA-C and psi = N-CA-C-N(i+1) [rad] of the quintuples q (k, 5) in frame R (n, 3) [nm]."""
    return (
        _dihedral(R[q[:, 0]], R[q[:, 1]], R[q[:, 2]], R[q[:, 3]]),
        _dihedral(R[q[:, 1]], R[q[:, 2]], R[q[:, 3]], R[q[:, 4]]),
    )


class CMAPFourier(Family):
    """Fourier backbone map of order 3 ("cmap", 48 coefficients cm [kJ/mol] per key).

    Subclass `CMAPFourier6` ("cmap6") has order 6 (168 coefficients).

    Attributes
    ----------
    order : int
        Fourier order M (class attribute).
    """

    name = "cmap"
    order = ORDER
    params = {"cm": ((len(basis_orders(ORDER)),), 0.0)}
    linear = ("cm",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the quintuples "q" (k, 5) of Topology.cmaps, keyed by the oriented quintuple (kind "cmap")."""
        q = getattr(top, "cmaps", None)
        q = np.zeros((0, 5), int) if q is None else np.asarray(q, int).reshape(-1, 5)
        return {"q": q}, [keyf(t, "cmap") for t in q]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum over the residues of cm . basis(phi, psi) [kJ/mol] (0.0 without quintuples)."""
        q = I["q"]
        if len(q) == 0:
            return 0.0
        phi, psi = phi_psi(G["R"], q)
        return jnp.sum(p["cm"] * cmap_basis(phi, psi, self.order))


class CMAPFourier6(CMAPFourier):
    """Fourier backbone map of order 6 ("cmap6", 168 coefficients per key); otherwise as `CMAPFourier`."""

    name = "cmap6"
    order = 6
    params = {"cm": ((len(basis_orders(6)),), 0.0)}


register(CMAPFourier)
register(CMAPFourier6)
