"""Compute Lennard-Jones van der Waals in Amber's form, differentiable in positions, parameters and box.

Contents: lj_pair (the pair energy), LJChannel (gas phase), lj_long_range (continuum tail
correction), PeriodicLJ (periodic, hard cutoff).

    E_ij = eps_ij [ (rmin_ij / r)^12 - 2 (rmin_ij / r)^6 ],   rmin_ij = R*_i + R*_j,   eps_ij = s_i s_j

with R* = lj_rmin_half [nm] and s = lj_sqrt_eps [sqrt(kJ/mol)] per atom (Lorentz-Berthelot).

Only intermolecular pairs (including periodic images of the same molecule) are summed, as for
rigid molecules in Amber, where every intramolecular pair is excluded.  Intramolecular LJ
(1-4 and beyond) belongs with bonded terms, which are not implemented.

Units: nm, kJ/mol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .system import System

if TYPE_CHECKING:
    import jax


def lj_pair(r: ArrayLike, rmin: ArrayLike, eps: ArrayLike) -> jax.Array:
    """Return the Lennard-Jones pair energy eps [(rmin/r)^12 - 2 (rmin/r)^6] [kJ/mol].

    Parameters
    ----------
    r : ArrayLike
        Distance [nm], nonzero.
    rmin : ArrayLike
        Distance of the minimum [nm].
    eps : ArrayLike
        Well depth [kJ/mol].

    Returns
    -------
    jax.Array
        Elementwise pair energies [kJ/mol].
    """
    s6 = (rmin / r) ** 6
    return eps * (s6 * s6 - 2.0 * s6)


def _pair_params(P: Mapping[str, jax.Array], i: np.ndarray, j: np.ndarray) -> tuple[jax.Array, jax.Array]:
    """Return (rmin_ij [nm], eps_ij [kJ/mol]) of pairs (i, j) from per-atom parameters P (System.expand)."""
    return P["lj_rmin_half"][i] + P["lj_rmin_half"][j], P["lj_sqrt_eps"][i] * P["lj_sqrt_eps"][j]


@dataclass
class LJChannel:
    """Gas-phase Lennard-Jones channel: all intermolecular pairs, no cutoff.

    Parameters
    ----------
    name : str
        Key of the energy in the returned dict.
    """

    name: str = "vdw"

    def energy(
        self, pos: jax.Array, sys: System, params: Mapping[str, ArrayLike] | None = None
    ) -> tuple[dict[str, jax.Array], dict]:
        """Return ({name: LJ energy [kJ/mol]}, {}) of one configuration.

        `pos` (N, 3) positions [nm]; `params` the parameter pytree (None: initial values).
        """
        P = sys.expand(params)
        inter = sys.pair_inter
        i, j = sys.pair_i[inter], sys.pair_j[inter]
        r = jnp.linalg.norm(pos[i] - pos[j], axis=-1)
        rmin, eps = _pair_params(P, i, j)
        return {self.name: jnp.sum(lj_pair(r, rmin, eps))}, {}


def lj_long_range(P: Mapping[str, jax.Array], volume: ArrayLike, rc: float) -> jax.Array:
    """Return Amber's vdwmeth=1 continuum correction beyond rc for the r^-6 term [kJ/mol].

    Parameters
    ----------
    P : Mapping of str to jax.Array (N,)
        Per-atom parameters (System.expand): "lj_rmin_half" [nm], "lj_sqrt_eps" [sqrt(kJ/mol)].
    volume : ArrayLike
        Box volume [nm^3].
    rc : float
        Cutoff [nm].

    Returns
    -------
    jax.Array ()
        E_lrc = -2 pi / (3 V rc^3) sum_{i,j} B_ij over all ordered atom pairs (i = j included),
        B_ij = 2 eps_ij rmin_ij^6 [kJ/mol].

    Notes
    -----
    Uses (R*_i + R*_j)^6 = sum_k C(6,k) R*_i^k R*_j^(6-k), so the double sum is
    sum_k C(6,k) M_k M_(6-k) with M_k = sum_i s_i R*_i^k: O(N).  The pair distribution beyond rc is
    taken as uniform, and intramolecular pairs are not removed from the sum.
    """
    R, s = P["lj_rmin_half"], P["lj_sqrt_eps"]
    M = [jnp.sum(s * R**k) for k in range(7)]
    SB = 2.0 * sum(comb(6, k) * M[k] * M[6 - k] for k in range(7))
    return -2.0 * jnp.pi * SB / (3.0 * volume * rc**3)


class PeriodicLJ:
    """Periodic Lennard-Jones with a hard cutoff rc (Amber cut; vdwmeth=0), optional long-range correction (vdwmeth=1).

    Uses the integer-image neighbour list of ewald.neighbor_list, restricted to intermolecular pairs
    and images of the same molecule.  The energy is not shifted at the cutoff.

    Attributes
    ----------
    sys : System
    H : np.ndarray (3, 3)
        Reference box [nm], lattice vectors as rows.
    rc : float
        Cutoff [nm].
    lrc : bool
        Add the long-range correction.
    pi, pj, img : np.ndarray
        Kept pairs (P,), (P,) and their image vectors (P, 3).
    """

    def __init__(
        self,
        sys: System,
        H: ArrayLike,
        pos_ref: ArrayLike,
        rc: float = 1.0,
        skin: float = 0.0,
        lrc: bool = False,
        nlist: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Build the pair list.

        Parameters
        ----------
        sys : System
            System.
        H : ArrayLike (3, 3)
            Reference box [nm], lattice vectors as rows.
        pos_ref : ArrayLike (N, 3)
            Reference positions [nm] of the neighbour list.
        rc : float
            Cutoff [nm].
        skin : float
            Neighbour-list skin [nm] (ignored when `nlist` is given).
        lrc : bool
            Add the long-range correction (lj_long_range) to the energy.
        nlist : tuple of np.ndarray, optional
            Shared neighbour list (i, j, image) with a cutoff of at least rc; None builds one.
        """
        from .ewald import neighbor_list

        self.sys, self.H, self.rc, self.lrc = sys, np.asarray(H, float), rc, lrc
        i, j, img = nlist if nlist is not None else neighbor_list(pos_ref, self.H, rc + skin)
        keep = ~((sys.mol[i] == sys.mol[j]) & np.all(img == 0, axis=1))  # intermolecular or image
        self.pi, self.pj, self.img = i[keep], j[keep], img[keep]

    def energy(
        self, pos: jax.Array, params: Mapping[str, ArrayLike] | None = None, H: ArrayLike | None = None
    ) -> tuple[dict[str, jax.Array], dict]:
        """Return ({"vdw": energy [kJ/mol]}, {}) of one configuration.

        `pos` (N, 3) positions [nm]; `params` the parameter pytree (None: initial values); `H` (3, 3)
        box [nm] (None: reference box).  Pairs of the list beyond rc are masked with jnp.where.
        """
        P = self.sys.expand(params)
        H = jnp.asarray(self.H if H is None else H)
        x = pos[self.pi] - pos[self.pj] + jnp.asarray(self.img, float) @ H
        r = jnp.linalg.norm(x, axis=-1)
        rmin, eps = _pair_params(P, self.pi, self.pj)
        e = jnp.sum(jnp.where(r < self.rc, lj_pair(r, rmin, eps), 0.0))
        if self.lrc:
            e = e + lj_long_range(P, jnp.abs(jnp.linalg.det(H)), self.rc)
        return {"vdw": e}, {}

    def tail_virial(self, params: Mapping[str, ArrayLike] | None = None, H: ArrayLike | None = None) -> jax.Array:
        """Return the cutoff-impulse part of the long-range correction as a dE/d eps term (3, 3) [kJ/mol].

        The term is -E_lrc I (zero without lrc).  Added to the strain derivative it gives the standard
        continuum tail pressure P_tail = 2 E_lrc / V, which Amber (vdwmeth=1) uses; the strain
        derivative of the energy alone gives E_lrc / V, because pairs crossing the fixed cutoff are not
        seen by a derivative at fixed configuration.  `params`, `H` as in `energy`.
        """
        if not self.lrc:
            return jnp.zeros((3, 3))
        H = jnp.asarray(self.H if H is None else H)
        return -lj_long_range(self.sys.expand(params), jnp.abs(jnp.linalg.det(H)), self.rc) * jnp.eye(3)
