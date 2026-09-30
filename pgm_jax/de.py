"""Compute the double-exponential (DE) van der Waals interaction of DEGAUSS, differentiable in all inputs.

Contents: de_pair (pair energy), de_pair_grad (energy and (1/r) dU/dr), de_groups, de_long_range and
de_tail_impulse (continuum tail correction and its cutoff impulse), DEChannel (gas phase), PeriodicDE
(periodic, hard cutoff).

    u_ij(r) = eps_ij / (alpha - beta) [ beta exp(alpha (1 - r/rm_ij)) - alpha exp(beta (1 - r/rm_ij)) ]

with the well depth eps_ij and the minimum-energy separation rm_ij of the Lennard-Jones combination
rules of the topology (rm_ij = R*_i + R*_j, eps_ij = s_i s_j from lj_rmin_half and lj_sqrt_eps), and the
dimensionless exponents alpha > beta > 0 (Huang, Duan, Wu, Luo, "DEGAUSS in AMBER", Eq. 3).  The
pair energy is finite at r = 0,

    u_ij(0) = eps_ij / (alpha - beta) (beta e^alpha - alpha e^beta),

has its minimum -eps_ij at r = rm_ij, and tends to zero at large r.  Paper I of DEGAUSS obtained
alpha = 18.17 and beta = 3.65 by parameterizing the DE form against the Lennard-Jones interaction of
classical force fields; DE_ALPHA and DE_BETA are these defaults.

As in lj.py only intermolecular pairs (including periodic images of the same molecule) are summed.

Units: nm, kJ/mol.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .lj import PeriodicLJ, _pair_params
from .system import System

DE_ALPHA = 18.17  # repulsive exponent of Paper I
DE_BETA = 3.65  # attractive exponent of Paper I


def de_pair(r: ArrayLike, rm: ArrayLike, eps: ArrayLike, alpha: float = DE_ALPHA, beta: float = DE_BETA) -> jnp.ndarray:
    """Return the double-exponential pair energy [kJ/mol].

    Parameters
    ----------
    r : ArrayLike
        Distance [nm]; zero is allowed (the energy is finite).
    rm : ArrayLike
        Distance of the minimum [nm], positive (a nonpositive value is taken as 1; the well depth of
        such atoms is zero).
    eps : ArrayLike
        Well depth [kJ/mol].
    alpha, beta : float
        Repulsive and attractive exponents, alpha > beta > 0.

    Returns
    -------
    jax.Array
        eps / (alpha - beta) [beta exp(alpha (1 - r/rm)) - alpha exp(beta (1 - r/rm))], elementwise.
    """
    rm = jnp.where(rm > 0.0, rm, 1.0)  # atoms without a van der Waals radius have eps = 0
    x = 1.0 - r / rm
    return eps / (alpha - beta) * (beta * jnp.exp(alpha * x) - alpha * jnp.exp(beta * x))


def de_pair_grad(
    r: ArrayLike, rm: ArrayLike, eps: ArrayLike, alpha: float = DE_ALPHA, beta: float = DE_BETA
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the double-exponential pair energy and its (1/r) dU/dr.

    Parameters
    ----------
    r, rm, eps, alpha, beta
        As de_pair; `r` must be nonzero for the second output.

    Returns
    -------
    e : jax.Array
        Pair energy [kJ/mol].
    d : jax.Array
        (1/r) dU/dr = eps alpha beta / ((alpha - beta) rm r) [exp(beta (1 - r/rm)) - exp(alpha (1 - r/rm))]
        [kJ/mol/nm^2] (the force on atom i is -d x_ij).
    """
    rm = jnp.where(rm > 0.0, rm, 1.0)  # atoms without a van der Waals radius have eps = 0
    x = 1.0 - r / rm
    ea, eb = jnp.exp(alpha * x), jnp.exp(beta * x)
    c = eps / (alpha - beta)
    return c * (beta * ea - alpha * eb), c * alpha * beta * (eb - ea) / (rm * r)


def de_groups(rmin_half: ArrayLike, sqrt_eps: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return one representative atom and the population of each distinct (R*, sqrt(eps)) pair of parameters.

    The DE tail is not separable in the atoms (rm_ij = R*_i + R*_j sits in an exponent), so the
    double sum runs over the distinct parameter sets with their populations.

    Parameters
    ----------
    rmin_half : ArrayLike (N,)
        Per-atom R* [nm] (concrete values, not traced).
    sqrt_eps : ArrayLike (N,)
        Per-atom sqrt(eps) [sqrt(kJ/mol)].

    Returns
    -------
    rep : np.ndarray (G,) int
        Index of the first atom of each group.
    count : np.ndarray (G,)
        Number of atoms of each group.
    """
    key = np.stack([np.asarray(rmin_half, float), np.asarray(sqrt_eps, float)], axis=1)
    _, rep, count = np.unique(key, axis=0, return_index=True, return_counts=True)
    return rep, count.astype(float)


def _tail_integral(k: jnp.ndarray, rc: float) -> jnp.ndarray:
    """Return int_rc^inf r^2 exp(-k r) dr = exp(-k rc) (rc^2/k + 2 rc/k^2 + 2/k^3) for k > 0."""
    return jnp.exp(-k * rc) * (rc**2 / k + 2.0 * rc / k**2 + 2.0 / k**3)


def de_long_range(
    P: Mapping[str, jnp.ndarray],
    volume: ArrayLike,
    rc: float,
    groups: tuple[np.ndarray, np.ndarray],
    alpha: float = DE_ALPHA,
    beta: float = DE_BETA,
) -> jnp.ndarray:
    """Return the continuum correction beyond rc for the DE interaction [kJ/mol].

    Parameters
    ----------
    P : Mapping of str to jax.Array (N,)
        Per-atom parameters (System.expand): "lj_rmin_half" [nm], "lj_sqrt_eps" [sqrt(kJ/mol)].
    volume : ArrayLike
        Box volume [nm^3].
    rc : float
        Cutoff [nm].
    groups : tuple of np.ndarray
        (representative atom, population) of the distinct parameter sets (de_groups).
    alpha, beta : float
        DE exponents.

    Returns
    -------
    jax.Array ()
        E_lrc = 2 pi / V sum_{i,j} int_rc^inf r^2 u_ij(r) dr over all ordered atom pairs (i = j
        included), with the two exponential integrals in closed form:
        int_rc^inf r^2 exp(-k r) dr = exp(-k rc) (rc^2/k + 2 rc/k^2 + 2/k^3), k = alpha/rm or beta/rm.
        The pair distribution beyond rc is taken as uniform, as for the Lennard-Jones tail of lj.py.
    """
    rep, n = groups
    R, s = P["lj_rmin_half"][rep], P["lj_sqrt_eps"][rep]
    rm = R[:, None] + R[None, :]
    eps = s[:, None] * s[None, :]
    rm = jnp.where(rm > 0.0, rm, 1.0)  # atoms without van der Waals radius have eps = 0
    ia = jnp.exp(alpha) * _tail_integral(alpha / rm, rc)
    ib = jnp.exp(beta) * _tail_integral(beta / rm, rc)
    integral = eps / (alpha - beta) * (beta * ia - alpha * ib)
    w = jnp.asarray(n[:, None] * n[None, :])
    return 2.0 * jnp.pi * jnp.sum(w * integral) / volume


def de_tail_impulse(
    P: Mapping[str, jnp.ndarray],
    volume: ArrayLike,
    rc: float,
    groups: tuple[np.ndarray, np.ndarray],
    alpha: float = DE_ALPHA,
    beta: float = DE_BETA,
) -> jnp.ndarray:
    """Return the scalar X of the cutoff impulse -X I of the tail in the strain derivative [kJ/mol].

    Parameters
    ----------
    P, volume, rc, groups, alpha, beta
        As de_long_range.

    Returns
    -------
    jax.Array ()
        X = 2 pi rc^3 / (3 V) sum_{i,j} u_ij(rc) over all ordered atom pairs.

    Notes
    -----
    The strain derivative of the tail energy at fixed configuration is -E_lrc I (E_lrc ~ 1/V).  The
    true tail virial, (2 pi / 3 V) sum_ij int_rc^inf r^3 u_ij'(r) dr = (2 pi / 3 V) sum_ij (-rc^3 u_ij(rc)
    - 3 int r^2 u_ij dr), differs from it by -X I: for a pure r^-6 tail X = E_lrc (the LJ rule of
    lj.PeriodicLJ.tail_virial), and in general X collects only the pair energy at the cutoff.
    """
    rep, n = groups
    R, s = P["lj_rmin_half"][rep], P["lj_sqrt_eps"][rep]
    rm = R[:, None] + R[None, :]
    eps = s[:, None] * s[None, :]
    u = de_pair(rc, rm, eps, alpha, beta)
    w = jnp.asarray(n[:, None] * n[None, :])
    return 2.0 * jnp.pi * rc**3 * jnp.sum(w * u) / (3.0 * volume)


class DEChannel:
    """Gas-phase double-exponential channel: all intermolecular pairs, no cutoff.

    Parameters
    ----------
    name : str
        Key of the energy in the returned dict.
    alpha, beta : float
        DE exponents.
    """

    def __init__(self, name: str = "vdw", alpha: float = DE_ALPHA, beta: float = DE_BETA) -> None:
        """Store the key and the exponents."""
        self.name, self.alpha, self.beta = name, alpha, beta

    def energy(
        self, pos: jnp.ndarray, sys: System, params: Mapping[str, ArrayLike] | None = None
    ) -> tuple[dict[str, jnp.ndarray], dict]:
        """Return ({name: DE energy [kJ/mol]}, {}) of one configuration.

        `pos` (N, 3) positions [nm]; `params` the parameter pytree (None: initial values).
        """
        P = sys.expand(params)
        inter = sys.pair_inter
        i, j = sys.pair_i[inter], sys.pair_j[inter]
        r = jnp.linalg.norm(pos[i] - pos[j], axis=-1)
        rm, eps = _pair_params(P, i, j)
        return {self.name: jnp.sum(de_pair(r, rm, eps, self.alpha, self.beta))}, {}


class PeriodicDE(PeriodicLJ):
    """Periodic double-exponential van der Waals with a hard cutoff rc and an optional continuum tail.

    The pair list, the cutoff handling and the attributes are those of lj.PeriodicLJ (intermolecular
    pairs and images of the same molecule, energy not shifted at the cutoff); `lrc` adds de_long_range.

    Attributes
    ----------
    alpha, beta : float
        DE exponents.
    groups : tuple of np.ndarray
        Distinct parameter sets (de_groups) of the initial parameters, used by the tail.
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
        alpha: float = DE_ALPHA,
        beta: float = DE_BETA,
    ) -> None:
        """Build the pair list (lj.PeriodicLJ) and the parameter groups of the tail.

        Parameters as PeriodicLJ, plus `alpha`, `beta` the DE exponents.
        """
        super().__init__(sys, H, pos_ref, rc=rc, skin=skin, lrc=lrc, nlist=nlist)
        self.alpha, self.beta = alpha, beta
        P0 = sys.expand(None)
        self.groups = de_groups(P0["lj_rmin_half"], P0["lj_sqrt_eps"])

    def _tail(self, P: Mapping[str, jnp.ndarray], H: ArrayLike) -> jnp.ndarray:
        """Return the continuum correction [kJ/mol] at box H."""
        return de_long_range(P, jnp.abs(jnp.linalg.det(H)), self.rc, self.groups, self.alpha, self.beta)

    def energy(
        self, pos: jnp.ndarray, params: Mapping[str, ArrayLike] | None = None, H: ArrayLike | None = None
    ) -> tuple[dict[str, jnp.ndarray], dict]:
        """Return ({"vdw": energy [kJ/mol]}, {}) of one configuration (see PeriodicLJ.energy)."""
        P = self.sys.expand(params)
        H = jnp.asarray(self.H if H is None else H)
        x = pos[self.pi] - pos[self.pj] + jnp.asarray(self.img, float) @ H
        r = jnp.linalg.norm(x, axis=-1)
        rm, eps = _pair_params(P, self.pi, self.pj)
        e = jnp.sum(jnp.where(r < self.rc, de_pair(r, rm, eps, self.alpha, self.beta), 0.0))
        if self.lrc:
            e = e + self._tail(P, H)
        return {"vdw": e}, {}

    def tail_virial(self, params: Mapping[str, ArrayLike] | None = None, H: ArrayLike | None = None) -> jnp.ndarray:
        """Return the cutoff-impulse part of the tail as a dE/d eps term (3, 3) [kJ/mol] (see de_tail_impulse).

        The term is -X I with X = 2 pi rc^3 / (3 V) sum_ij u_ij(rc) (zero without lrc); it completes the
        strain derivative of the tail energy to the continuum tail pressure.  `params`, `H` as in `energy`.
        """
        if not self.lrc:
            return jnp.zeros((3, 3))
        H = jnp.asarray(self.H if H is None else H)
        X = de_tail_impulse(
            self.sys.expand(params), jnp.abs(jnp.linalg.det(H)), self.rc, self.groups, self.alpha, self.beta
        )
        return -X * jnp.eye(3)
