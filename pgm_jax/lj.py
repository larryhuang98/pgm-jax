"""Lennard-Jones van der Waals in Amber's form, differentiable in positions, parameters and box.

  E_ij = eps_ij [ (rmin_ij / r)^12 - 2 (rmin_ij / r)^6 ],   rmin_ij = R*_i + R*_j,   eps_ij = s_i s_j
with R* = lj_rmin_half and s = lj_sqrt_eps per atom (Lorentz-Berthelot).

Only intermolecular pairs (including periodic images of the same molecule) are summed, as for
rigid molecules in Amber, where every intramolecular pair is excluded.  Intramolecular LJ
(1-4 and beyond) belongs with bonded terms, which are not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import jax.numpy as jnp
import numpy as np

from .system import System


def lj_pair(r, rmin, eps):
    s6 = (rmin / r) ** 6
    return eps * (s6 * s6 - 2.0 * s6)


def _pair_params(P, i, j):
    return P["lj_rmin_half"][i] + P["lj_rmin_half"][j], P["lj_sqrt_eps"][i] * P["lj_sqrt_eps"][j]


@dataclass
class LJChannel:
    """Gas phase: all intermolecular pairs, no cutoff."""
    name: str = "vdw"

    def energy(self, pos, sys: System, params=None):
        P = sys.expand(params)
        inter = sys.pair_inter
        i, j = sys.pair_i[inter], sys.pair_j[inter]
        r = jnp.linalg.norm(pos[i] - pos[j], axis=-1)
        rmin, eps = _pair_params(P, i, j)
        return {self.name: jnp.sum(lj_pair(r, rmin, eps))}, {}


def lj_long_range(P, volume, rc):
    """Amber's vdwmeth=1 continuum correction beyond rc for the r^-6 term (energy, kJ/mol):
    -2 pi / (3 V rc^3) sum_{i,j} B_ij over all ordered atom pairs (i = j included),
    B_ij = 2 eps_ij rmin_ij^6.  Uses (R*_i + R*_j)^6 = sum_k C(6,k) R*_i^k R*_j^(6-k): O(n)."""
    R, s = P["lj_rmin_half"], P["lj_sqrt_eps"]
    M = [jnp.sum(s * R ** k) for k in range(7)]
    SB = 2.0 * sum(comb(6, k) * M[k] * M[6 - k] for k in range(7))
    return -2.0 * jnp.pi * SB / (3.0 * volume * rc ** 3)


class PeriodicLJ:
    """Periodic LJ with a hard cutoff rc (Amber cut; vdwmeth=0), optional long-range correction
    (vdwmeth=1).  Uses the integer-image neighbour list of ewald.neighbor_list."""

    def __init__(self, sys: System, H, pos_ref, rc: float = 1.0, skin: float = 0.0, lrc: bool = False, nlist=None):
        from .ewald import neighbor_list
        self.sys, self.H, self.rc, self.lrc = sys, np.asarray(H, float), rc, lrc
        i, j, img = nlist if nlist is not None else neighbor_list(pos_ref, self.H, rc + skin)
        keep = ~((sys.mol[i] == sys.mol[j]) & np.all(img == 0, axis=1))       # intermolecular or image
        self.pi, self.pj, self.img = i[keep], j[keep], img[keep]

    def energy(self, pos, params=None, H=None):
        P = self.sys.expand(params)
        H = jnp.asarray(self.H if H is None else H)
        x = pos[self.pi] - pos[self.pj] + jnp.asarray(self.img, float) @ H
        r = jnp.linalg.norm(x, axis=-1)
        rmin, eps = _pair_params(P, self.pi, self.pj)
        e = jnp.sum(jnp.where(r < self.rc, lj_pair(r, rmin, eps), 0.0))
        if self.lrc:
            e = e + lj_long_range(P, jnp.abs(jnp.linalg.det(H)), self.rc)
        return {"vdw": e}, {}

    def tail_virial(self, params=None, H=None):
        """Cutoff-impulse part of the long-range correction, as a dE/d eps term (3, 3): -E_lrc I
        (zero without lrc).  Added to the strain derivative it gives the standard continuum tail
        pressure P_tail = 2 E_lrc / V, which Amber (vdwmeth=1) uses; the strain derivative of the
        energy alone gives E_lrc / V, because pairs crossing the fixed cutoff are not seen by a
        derivative at fixed configuration."""
        if not self.lrc:
            return jnp.zeros((3, 3))
        H = jnp.asarray(self.H if H is None else H)
        return -lj_long_range(self.sys.expand(params), jnp.abs(jnp.linalg.det(H)), self.rc) * jnp.eye(3)
