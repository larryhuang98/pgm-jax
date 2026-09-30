"""Compute GVDW, the Gaussian-density van der Waals of Huang, Luo and Duan (pmemd-pgm `igvdw = 1`).

The other van der Waals option, Lennard-Jones, is in lj.py; the GVDW data set is located by
paths.resource("gvdw_data") (PGM_GVDW_DATA).  Contents: gvdw_G (the damped dispersion kernel),
gvdw_pair, gvdw_pair_params, pair_beta, gvdw_long_range, GVDWChannel (gas phase), PeriodicGVDW
(periodic, hard cutoff), and the parameter helpers from_pmemd, set_gvdw, PGM3P_GVDW.

GVDW pair energy, with the Gaussian pair exponent of the pGM electrostatics
b_ij = 1/sqrt(2 (R_i^2 + R_j^2)) (same densities as the charges and dipoles) and y = b_ij r:

    U_ij(r) = A_ij exp(-B_ij y^2)            (rep = "gauss", pmemd gvdw_rep_form = 0)
            | A_ij exp(-B_ij y)              (rep = "slater", Born-Mayer, gvdw_rep_form = 1)
              - C6_ij F(y) / r^6,

    F(y) = A(y)^2 + g(y)^2 / 2,   A(y) = erf(y) - (2 y/sqrt(pi)) (1 + 2 y^2/3) exp(-y^2),
                                  g(y) = (4 y^3 / (3 sqrt(pi))) exp(-y^2),

the damping of the dipole-dipole dispersion between two Gaussian clouds: F -> 1 at large y and
F(y)/y^6 -> 8/(9 pi) at y -> 0, so the energy is finite at every distance.  Per-atom parameters
(system.py) a = sqrt(A), c = sqrt(C6), b, combined as A_ij = a_i a_j, C6_ij = c_i c_j,
B_ij = (b_i + b_j)/2.  pmemd-pgm uses one global A_rep, C6 and b_rep_scale for the pairs of atoms
that carry LJ (in pGM3P water: O-O only); `set_gvdw` sets those per atom type.

Numerics: G(y) = F(y)/y^6 is evaluated from its Taylor series (15 terms, exact rational
coefficients, truncation < 1e-13 relative) for y < 0.6 and in closed form above; both branches
are safe in float32.  densities.gd6_jax evaluates the same F(y) for the gas-phase bonded code.

Units: nm, kJ/mol; A [kJ/mol], C6 [kJ/mol nm^6], B dimensionless.

See also docs/howto_vdw.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf
from jax.typing import ArrayLike

from .system import System
from .units import KCAL

if TYPE_CHECKING:
    import jax

    from .system import Molecule

_SQRT_PI = math.sqrt(math.pi)
C0 = 8.0 / (9.0 * math.pi)  # lim_{y->0} F(y)/y^6
# F(y)/y^6 = C0 sum_k _GC[k] y^(2k)
_GC = (
    1.0,
    -2.0,
    58 / 25,
    -188 / 105,
    2222 / 2205,
    -1532 / 3465,
    64244 / 405405,
    -162056 / 3378375,
    1587734 / 126351225,
    -211108 / 72747675,
    11902564 / 19860115275,
    -611992 / 5465775315,
    755028172 / 39529267903125,
    -213335384 / 71152682225625,
    15291334888 / 35078272337233125,
)
Y_SERIES = 0.6  # series below, closed form above


def _poly(y2: jax.Array, coefs: Sequence[float]) -> jax.Array:
    """Return sum_k coefs[k] y2^k by Horner's rule (elementwise)."""
    acc = jnp.zeros_like(y2) + coefs[-1]
    for c in coefs[-2::-1]:
        acc = acc * y2 + c
    return acc


def gvdw_G(y: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return (G, G'/y) with G(y) = F(y)/y^6, the damped dispersion kernel (dimensionless), elementwise.

    Parameters
    ----------
    y : jax.Array
        Reduced distance b_ij r (dimensionless), y >= 0.

    Returns
    -------
    G : jax.Array
        F(y)/y^6.
    dG : jax.Array
        G'(y)/y (for forces, (1/r) dU/dr).

    Notes
    -----
    Below Y_SERIES the Taylor series in y^2 (_GC) and its term-wise derivative are used; above, the
    closed form with F' = 2 A A' + g g', A' = (8/(3 sqrt(pi))) y^4 exp(-y^2),
    g' = (4/sqrt(pi)) y^2 (1 - 2 y^2/3) exp(-y^2).  Each branch is evaluated with its argument
    clipped to its own region (jnp.where) so that neither produces NaNs or infinite gradients.
    """
    small = y < Y_SERIES
    ys = jnp.where(small, y, 0.0)
    y2s = ys * ys
    Gs = C0 * _poly(y2s, _GC)
    Hs = C0 * _poly(y2s, tuple(2.0 * k * c for k, c in enumerate(_GC))[1:])
    yl = jnp.where(small, 1.0, y)
    y2 = yl * yl
    e = jnp.exp(-y2)
    A = erf(yl) - (2.0 / _SQRT_PI) * yl * (1.0 + 2.0 * y2 / 3.0) * e
    g = (4.0 / (3.0 * _SQRT_PI)) * y2 * yl * e
    F = A * A + 0.5 * g * g
    Ap = (8.0 / (3.0 * _SQRT_PI)) * y2 * y2 * e
    gp = (4.0 / _SQRT_PI) * y2 * (1.0 - 2.0 * y2 / 3.0) * e
    Fp = 2.0 * A * Ap + g * gp
    y6 = y2 * y2 * y2
    Gl = F / y6
    Hl = (Fp / y6 - 6.0 * F / (y6 * yl)) / yl
    return jnp.where(small, Gs, Gl), jnp.where(small, Hs, Hl)


def gvdw_pair(
    r: jax.Array,
    beta: jax.Array,
    A: jax.Array,
    C6: jax.Array,
    B: jax.Array,
    rep: str = "gauss",
    grad: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Return the GVDW pair energy (and optionally (1/r) dU/dr), elementwise.

    Parameters
    ----------
    r : jax.Array
        Distance [nm], > 0.
    beta : jax.Array
        Gaussian pair exponent b_ij [1/nm] (pair_beta).
    A : jax.Array
        Repulsion amplitude A_ij [kJ/mol].
    C6 : jax.Array
        Dispersion coefficient C6_ij [kJ/mol nm^6].
    B : jax.Array
        Repulsion exponent scale B_ij (dimensionless).
    rep : {"gauss", "slater"}
        Repulsion form: A exp(-B y^2) or A exp(-B y), y = beta r.
    grad : bool
        Also return (1/r) dU/dr.

    Returns
    -------
    jax.Array or tuple of jax.Array
        U [kJ/mol]; with grad, (U, (1/r) dU/dr [kJ/mol/nm^2]).

    Raises
    ------
    ValueError
        If `rep` is unknown.

    Notes
    -----
    U = U_rep - C6 F(y)/r^6 = U_rep - C6 beta^6 G(y) (module docstring).
    """
    y = beta * r
    if rep == "gauss":
        er = A * jnp.exp(-B * y * y)
        dr = -2.0 * B * beta * beta * er  # (1/r) dU_rep/dr
    elif rep == "slater":
        er = A * jnp.exp(-B * y)
        dr = -B * beta * er / r
    else:
        raise ValueError(f"unknown GVDW repulsion {rep!r} (gauss | slater)")
    G, H = gvdw_G(y)
    b2 = beta * beta
    b6 = b2 * b2 * b2
    e = er - C6 * b6 * G
    if not grad:
        return e
    return e, dr - C6 * b6 * b2 * H


def gvdw_pair_params(
    P: Mapping[str, jax.Array], i: np.ndarray, j: np.ndarray
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (A_ij [kJ/mol], C6_ij [kJ/mol nm^6], B_ij) of pairs (i, j) from per-atom parameters.

    A_ij = a_i a_j, C6_ij = c_i c_j (a, c the square roots stored per atom), B_ij = (b_i + b_j)/2;
    `P` is System.expand(params).
    """
    return (
        P["gvdw_sqrt_a"][i] * P["gvdw_sqrt_a"][j],
        P["gvdw_sqrt_c6"][i] * P["gvdw_sqrt_c6"][j],
        0.5 * (P["gvdw_b"][i] + P["gvdw_b"][j]),
    )


def gvdw_long_range(P: Mapping[str, jax.Array], volume: ArrayLike, rc: float) -> jax.Array:
    """Return the continuum correction beyond rc for the GVDW dispersion [kJ/mol].

    As Amber's vdwmeth = 1 for the r^-6 term, with F = 1 beyond the cutoff:
    E_lrc = -2 pi / (3 V rc^3) sum_{i,j} C6_ij over all ordered pairs (i = j included), and
    sum_{i,j} c_i c_j = (sum_i c_i)^2.  `P` per-atom parameters (System.expand), `volume` [nm^3],
    `rc` [nm].  The repulsion tail is neglected.
    """
    c = jnp.sum(P["gvdw_sqrt_c6"])
    return -2.0 * jnp.pi * c * c / (3.0 * volume * rc**3)


def pair_beta(P: Mapping[str, jax.Array], i: np.ndarray, j: np.ndarray) -> jax.Array:
    """Return the Gaussian pair exponent 1/sqrt(2 (R_i^2 + R_j^2)) [1/nm] of pairs (i, j) (as densities.gauss_bij)."""
    return 1.0 / jnp.sqrt(2.0 * (P["radius"][i] ** 2 + P["radius"][j] ** 2))


@dataclass
class GVDWChannel:
    """Gas-phase GVDW channel: all intermolecular pairs, no cutoff.

    Parameters
    ----------
    rep : {"gauss", "slater"}
        Repulsion form.
    name : str
        Key of the energy in the returned dict.
    """

    rep: str = "gauss"
    name: str = "vdw"

    def energy(
        self, pos: jax.Array, sys: System, params: Mapping[str, ArrayLike] | None = None
    ) -> tuple[dict[str, jax.Array], dict]:
        """Return ({name: GVDW energy [kJ/mol]}, {}) of one configuration.

        `pos` (N, 3) positions [nm]; `params` the parameter pytree (None: initial values).
        """
        P = sys.expand(params)
        inter = sys.pair_inter
        i, j = sys.pair_i[inter], sys.pair_j[inter]
        r = jnp.linalg.norm(pos[i] - pos[j], axis=-1)
        A, C6, B = gvdw_pair_params(P, i, j)
        return {self.name: jnp.sum(gvdw_pair(r, pair_beta(P, i, j), A, C6, B, self.rep))}, {}


class PeriodicGVDW:
    """Periodic GVDW with a hard cutoff rc and optional dispersion tail (the lj.PeriodicLJ analogue).

    The pair list is the one of lj.PeriodicLJ (intermolecular pairs and images); the energy is not
    shifted at the cutoff.  Attributes as PeriodicLJ, plus `rep` (repulsion form).
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
        rep: str = "gauss",
    ) -> None:
        """Build the pair list (through lj.PeriodicLJ).

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
            Add the dispersion long-range correction (gvdw_long_range).
        nlist : tuple of np.ndarray, optional
            Shared neighbour list (i, j, image); None builds one.
        rep : {"gauss", "slater"}
            Repulsion form (checked only when the energy is evaluated).
        """
        from .lj import PeriodicLJ

        base = PeriodicLJ(sys, H, pos_ref, rc=rc, skin=skin, lrc=lrc, nlist=nlist)
        self.sys, self.H, self.rc, self.lrc, self.rep = sys, base.H, rc, lrc, rep
        self.pi, self.pj, self.img = base.pi, base.pj, base.img

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
        A, C6, B = gvdw_pair_params(P, self.pi, self.pj)
        e = jnp.sum(jnp.where(r < self.rc, gvdw_pair(r, pair_beta(P, self.pi, self.pj), A, C6, B, self.rep), 0.0))
        if self.lrc:
            e = e + gvdw_long_range(P, jnp.abs(jnp.linalg.det(H)), self.rc)
        return {"vdw": e}, {}

    def tail_virial(self, params: Mapping[str, ArrayLike] | None = None, H: ArrayLike | None = None) -> jax.Array:
        """Return the cutoff-impulse term -E_lrc I (3, 3) [kJ/mol] of the dispersion tail.

        Zero without lrc; see lj.PeriodicLJ.tail_virial.  `params`, `H` as in `energy`.
        """
        if not self.lrc:
            return jnp.zeros((3, 3))
        H = jnp.asarray(self.H if H is None else H)
        return -gvdw_long_range(self.sys.expand(params), jnp.abs(jnp.linalg.det(H)), self.rc) * jnp.eye(3)


# ------------------------------------------------------------------ parameters
def from_pmemd(A_kcal: float, C6_kcal_A6: float, b: float = 1.0) -> tuple[float, float, float]:
    """Convert pmemd-pgm's GVDW pair parameters into per-atom (sqrt A, sqrt C6, b) in pGM-JAX units.

    Parameters
    ----------
    A_kcal : float
        gvdw_arep [kcal/mol].
    C6_kcal_A6 : float
        gvdw_c6 [kcal/mol A^6].
    b : float
        b_rep_scale (dimensionless).

    Returns
    -------
    tuple of float
        (sqrt(A) [sqrt(kJ/mol)], sqrt(C6) [sqrt(kJ/mol nm^6)], b), for atoms whose like pairs should
        reproduce the pmemd parameters.
    """
    return math.sqrt(A_kcal * KCAL), math.sqrt(C6_kcal_A6 * KCAL * 1e-6), float(b)


def set_gvdw(mol: Molecule, by_type: dict[str, tuple[float, float, float]]) -> Molecule:
    """Return a copy of a Molecule with GVDW parameters per atom type.

    Parameters
    ----------
    mol : Molecule
        Template molecule.
    by_type : dict of str to (float, float, float)
        {type: (sqrt_a [sqrt(kJ/mol)], sqrt_c6 [sqrt(kJ/mol nm^6)], b)} in pGM-JAX units (see
        from_pmemd); other types get no GVDW (a = c = 0, b = 1).  Extra entries (such as "rep" of
        PGM3P_GVDW) are ignored unless they are atom types of the molecule.

    Returns
    -------
    Molecule
    """
    sa = np.array([by_type[t][0] if t in by_type else 0.0 for t in mol.types])
    sc = np.array([by_type[t][1] if t in by_type else 0.0 for t in mol.types])
    b = np.array([by_type[t][2] if t in by_type else 1.0 for t in mol.types])
    return replace(mol, gvdw_sqrt_a=sa, gvdw_sqrt_c6=sc, gvdw_b=b)


# pmemd-pgm GVDW water models of the GVDW manuscript (O-O only; the GVDW manuscript data)
PGM3P_GVDW = {
    "slater": {"rep": "slater", "OW": from_pmemd(87500.0, 594.825035, 4.52)},
    "gauss": {"rep": "gauss", "OW": from_pmemd(422.0, 594.825035, 0.9453)},
}
