"""Define collective variables (CVs) as JAX functions of the atom positions and the box.

Contents: the base class `CV` and the CVs `Distance`, `Component`, `Angle`, `Dihedral`,
`COMDistance`, `Coordination` (with PLUMED's rational `switching` function), `RMSD`, `Linear`
and `Custom`; `CVSet`, the ordered CVs of one bias (values, periods, nearest-image differences,
canonical values); `wrap`, the nearest-image difference of periodic values.

A CV is a scalar function s(pos, H) of the atomic positions pos (N, 3) [nm] and the box H
(lattice vectors as rows, reduced lower-triangular form [nm]; None for a non-periodic system).
Every difference vector between atoms is taken as a minimum image in H (md/box.min_image).  The
derivatives, and so every bias force, come from autodiff (jax.grad of V(s(pos, H))): a new CV
is a function, with no hand-coded derivatives.  CVs are evaluated in float64.

    from pgm_jax.bias import cv
    phi = cv.Dihedral(4, 6, 8, 14)                       # periodic, (-pi, pi]
    d = cv.Distance(0, 7)                                # nm, minimum image
    n = cv.Coordination(oxygens, hydrogens, r0=0.25)     # sum of rational switching functions
    r = cv.RMSD(backbone, ref_xyz)                       # after optimal superposition (quaternion fit)
    c = cv.COMDistance(ligand, pocket, masses=sys.masses)
    s = cv.Linear([d, n], [1.0, -0.1])                   # linear combination
    f = cv.Custom(lambda pos, H: pos[3, 2], name="z3")   # anything else
    s_val = phi(pos, H)                                  # a float64 scalar; jax.grad(phi)(pos, H) its gradient

Periodic CVs (Dihedral, or any CV with `period`) have their differences wrapped to the nearest
image by the biases (hills and kernels on a circle).

Units: positions and boxes nm; CV values nm (distances, RMSD), rad (angles), dimensionless
(coordination numbers), or those of a `Custom` function.

References
----------
.. [1] S. K. Kearsley, Acta Crystallogr. A 45, 208 (1989).
.. [2] E. A. Coutsias, C. Seok, K. A. Dill, J. Comput. Chem. 25, 1849 (2004). doi:10.1002/jcc.20110

See also docs/enhanced_sampling.md; bias/core.py (the biases on these CVs).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

_TINY = 1e-60  # floor under squared norms: keeps sqrt and its gradient finite at zero length
_HI = jax.lax.Precision.HIGHEST  # full float32/float64 matmul precision on GPUs (no TF32)


def _mi(d: jax.Array, H: ArrayLike | None) -> jax.Array:
    """Return the minimum image of displacement(s) `d` (..., 3) [nm] in a reduced box `H` (identity for H None)."""
    if H is None:
        return d
    from ..md.box import min_image

    return min_image(d, jnp.asarray(H, jnp.float64))


def _norm(v: jax.Array) -> jax.Array:
    """Return the Euclidean norm over the last axis, floored at sqrt(_TINY) (finite gradient at zero)."""
    return jnp.sqrt(jnp.maximum(jnp.sum(v * v, -1), _TINY))


def wrap(ds: ArrayLike, period: ArrayLike) -> jax.Array:
    """Return differences `ds` wrapped to the nearest image where `period` > 0.

    Parameters
    ----------
    ds : ArrayLike
        Differences of CV values [CV units].
    period : ArrayLike
        Periods [CV units], broadcast against `ds` (0: not periodic, left unchanged).

    Returns
    -------
    jax.Array
        ds - period round(ds / period) in [-period/2, period/2] (a difference of exactly half a
        period keeps its sign, since jnp.round rounds half to even).
    """
    period = jnp.asarray(period, jnp.float64)
    p = jnp.where(period > 0, period, 1.0)
    return jnp.where(period > 0, ds - p * jnp.round(ds / p), ds)


class CV:
    """Base class of the collective variables.

    Subclasses define __call__(pos, H) -> scalar and `atoms()`.  CVs are plain Python objects
    holding static indices and parameters (captured as constants when traced).

    Attributes
    ----------
    name : str
        Name of the CV (COLVAR column name).
    period : float or None
        Period [CV units] (class or instance attribute); None: not periodic.
    lo : float or None
        Lower end of the canonical interval [lo, lo + period) [CV units]; None: -period/2.
    """

    name = "cv"
    period = None
    lo = None

    def __call__(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the CV value s(pos, H), a float64 scalar [CV units] (to be defined by subclasses).

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Atom positions [nm].
        H : ArrayLike (3, 3), optional
            Box, lattice vectors as rows (reduced, lower triangular) [nm]; None: no periodic boundaries.

        Returns
        -------
        jax.Array ()
            The CV value.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError

    def atoms(self) -> np.ndarray:
        """Return the indices of the atoms the CV depends on (np.ndarray of int; empty in the base class)."""
        return np.zeros(0, int)

    def grad(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return ds/dpos by autodiff, jax.Array (N, 3) [CV units / nm].

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Atom positions [nm].
        H : ArrayLike (3, 3), optional
            Box, lattice vectors as rows (reduced, lower triangular) [nm]; None: no periodic boundaries.
        """
        return jax.grad(lambda x: self(x, H))(jnp.asarray(pos, jnp.float64))

    def __repr__(self) -> str:
        """Return the CV's name."""
        return self.name


def _x(pos: ArrayLike) -> jax.Array:
    """Return the positions as a float64 JAX array."""
    return jnp.asarray(pos, jnp.float64)


class Distance(CV):
    """Distance |r_j - r_i| between two atoms [nm], minimum image."""

    def __init__(self, i: int, j: int, name: str | None = None) -> None:
        """Set up the CV.

        Parameters
        ----------
        i, j : int
            Atom indices.
        name : str, optional
            Name; None: "d{i}_{j}".
        """
        self.i, self.j = int(i), int(j)
        self.name = name or f"d{self.i}_{self.j}"

    def __call__(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return |r_j - r_i| [nm] (minimum image in H)."""
        x = _x(pos)
        return _norm(_mi(x[self.j] - x[self.i], H))

    def atoms(self) -> np.ndarray:
        """Return [i, j]."""
        return np.array([self.i, self.j])


class Component(CV):
    """One Cartesian component of an atom's position, or of its displacement from `origin` [nm].

    Not periodic and not wrapped: for model systems and walls, used with H None or unwrapped
    coordinates.
    """

    def __init__(self, i: int, axis: int, origin: float = 0.0, name: str | None = None) -> None:
        """Set up the CV.

        Parameters
        ----------
        i : int
            Atom index.
        axis : int
            Cartesian axis (0, 1, 2 for x, y, z).
        origin : float
            Offset subtracted from the coordinate [nm].
        name : str, optional
            Name; None: "x{i}", "y{i}" or "z{i}".
        """
        self.i, self.axis, self.origin = int(i), int(axis), float(origin)
        self.name = name or f"{'xyz'[self.axis]}{self.i}"

    def __call__(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return pos[i, axis] - origin [nm] (`H` is ignored)."""
        return _x(pos)[self.i, self.axis] - self.origin

    def atoms(self) -> np.ndarray:
        """Return [i]."""
        return np.array([self.i])


class Angle(CV):
    """Bond angle i-j-k [rad] in [0, pi], vertex j, minimum images."""

    def __init__(self, i: int, j: int, k: int, name: str | None = None) -> None:
        """Set up the CV.

        Parameters
        ----------
        i, j, k : int
            Atom indices (j is the vertex).
        name : str, optional
            Name; None: "angle{i}_{j}_{k}".
        """
        self.idx = (int(i), int(j), int(k))
        self.name = name or "angle" + "_".join(map(str, self.idx))

    def __call__(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the angle between r_i - r_j and r_k - r_j [rad] (atan2 of |u x v| and u.v: stable near 0 and pi)."""
        x = _x(pos)
        i, j, k = self.idx
        u, v = _mi(x[i] - x[j], H), _mi(x[k] - x[j], H)
        return jnp.arctan2(_norm(jnp.cross(u, v)), jnp.sum(u * v, -1))

    def atoms(self) -> np.ndarray:
        """Return [i, j, k]."""
        return np.array(self.idx)


class Dihedral(CV):
    """Dihedral angle i-j-k-l [rad] in (-pi, pi], periodic with period 2 pi.

    IUPAC sign convention, as md/restraints.py and pgm_jax.bonded (computed by
    md/restraints.dihedral_from_bonds from the minimum-image bond vectors).  The canonical interval
    is [-pi, pi) (`lo` = -pi).
    """

    period = 2.0 * np.pi
    lo = -np.pi

    def __init__(self, i: int, j: int, k: int, l: int, name: str | None = None) -> None:  # noqa: E741
        """Set up the CV.

        Parameters
        ----------
        i, j, k, l : int
            Atom indices along the chain.
        name : str, optional
            Name; None: "dih{i}_{j}_{k}_{l}".
        """
        self.idx = (int(i), int(j), int(k), int(l))
        self.name = name or "dih" + "_".join(map(str, self.idx))

    def __call__(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the dihedral angle [rad] in (-pi, pi]."""
        from ..md.restraints import dihedral_from_bonds

        x = _x(pos)
        a, b, c, d = self.idx
        return dihedral_from_bonds(_mi(x[b] - x[a], H), _mi(x[c] - x[b], H), _mi(x[d] - x[c], H))

    def atoms(self) -> np.ndarray:
        """Return [i, j, k, l]."""
        return np.array(self.idx)


def _centre(x: jax.Array, g: np.ndarray, w: np.ndarray, H: ArrayLike | None) -> jax.Array:
    """Return the weighted centre of group `g` [nm], from minimum-image displacements to its first atom.

    Parameters
    ----------
    x : jax.Array (N, 3)
        Positions [nm].
    g : np.ndarray (n,) int
        Atom indices of the group.
    w : np.ndarray (n,)
        Weights (any positive scale; normalised here).
    H : ArrayLike (3, 3) or None
        Box [nm]; None: no periodic boundaries.
    """
    d = _mi(x[g] - x[g[0]], H)
    return x[g[0]] + jnp.sum(jnp.asarray(w)[:, None] * d, 0) / float(np.sum(w))


class COMDistance(CV):
    """Distance [nm] between the weighted centres of two groups of atoms.

    Weights are the atom masses (from `masses`, the per-atom masses of the whole system) or equal.
    Each group is made whole by minimum images from its first atom, so a group must be smaller than
    half the box; the distance between the centres is a minimum image too.
    """

    def __init__(
        self,
        group_a: ArrayLike,
        group_b: ArrayLike,
        masses: ArrayLike | None = None,
        name: str | None = None,
    ) -> None:
        """Set up the CV.

        Parameters
        ----------
        group_a, group_b : ArrayLike
            Atom indices of the two groups.
        masses : ArrayLike (N,), optional
            Masses of all atoms of the system [amu]; None: equal weights.
        name : str, optional
            Name; None: "com_distance".

        Raises
        ------
        ValueError
            If a group is empty.
        """
        self.ga, self.gb = (np.asarray(g, int).reshape(-1) for g in (group_a, group_b))
        if not len(self.ga) or not len(self.gb):
            raise ValueError("empty group")
        w = (lambda g: np.ones(len(g))) if masses is None else (lambda g: np.asarray(masses, float)[g])
        self.wa, self.wb = w(self.ga), w(self.gb)
        self.name = name or "com_distance"

    def __call__(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the distance between the two weighted centres [nm]."""
        x = _x(pos)
        return _norm(_mi(_centre(x, self.gb, self.wb, H) - _centre(x, self.ga, self.wa, H), H))

    def atoms(self) -> np.ndarray:
        """Return the atoms of both groups (group A, then group B)."""
        return np.concatenate([self.ga, self.gb])


def switching(r: ArrayLike, r0: float, n: int = 6, m: int = 12, d0: float = 0.0) -> jax.Array:
    """Return PLUMED's rational switching function s(r) = (1 - x^n) / (1 - x^m), x = (r - d0) / r0.

    Parameters
    ----------
    r : ArrayLike
        Distances [nm].
    r0 : float
        Switching length [nm].
    n, m : int
        Exponents (n < m).
    d0 : float
        Offset [nm]: s = 1 for r <= d0.

    Returns
    -------
    jax.Array
        s(r), dimensionless, same shape as `r`.

    Notes
    -----
    At x = 1 the ratio is 0/0; for |x - 1| < 1e-4 the first-order expansion
    s = n/m + n (n - m) / (2 m) (x - 1) is used instead.  The unused branch is evaluated at x = 0.5
    so that its gradient stays finite.
    """
    x = jnp.maximum((r - d0) / r0, 0.0)
    e = x - 1.0
    near = jnp.abs(e) < 1e-4  # 0/0 at x = 1: first-order expansion instead
    xs = jnp.where(near, 0.5, x)  # any x != 1 in the unused branch keeps its gradient finite
    f = (1.0 - xs**n) / (1.0 - xs**m)
    return jnp.where(near, n / m + n * (n - m) / (2.0 * m) * e, f)


class Coordination(CV):
    """Coordination number sum_{i in A, j in B, i != j} s(|r_ij|) with the rational `switching` function.

    Every pair of the two groups is evaluated (|A| x |B| minimum-image distances, no neighbour
    list: groups of up to a few thousand pairs).  Pairs listed twice when the groups overlap count
    twice, as in PLUMED's COORDINATION with GROUPA/GROUPB.

    Attributes
    ----------
    r0 : float
        Switching length [nm].
    n, m : int
        Exponents of the switching function.
    d0 : float
        Offset [nm].
    pi, pj : np.ndarray (P,) int
        The atom pairs (i in A, j in B, i != j).
    """

    def __init__(
        self,
        group_a: ArrayLike,
        group_b: ArrayLike,
        r0: float,
        n: int = 6,
        m: int = 12,
        d0: float = 0.0,
        name: str | None = None,
    ) -> None:
        """Set up the CV.

        Parameters
        ----------
        group_a, group_b : ArrayLike
            Atom indices of the two groups.
        r0 : float
            Switching length [nm].
        n, m : int
            Exponents of the switching function (n < m).
        d0 : float
            Offset [nm].
        name : str, optional
            Name; None: "coordination".

        Raises
        ------
        ValueError
            If n >= m.
        """
        self.ga, self.gb = (np.asarray(g, int).reshape(-1) for g in (group_a, group_b))
        if int(n) >= int(m):
            raise ValueError("switching exponents need n < m")
        self.r0, self.n, self.m, self.d0 = float(r0), int(n), int(m), float(d0)
        ia, ib = np.meshgrid(self.ga, self.gb, indexing="ij")
        keep = ia != ib
        self.pi, self.pj = ia[keep], ib[keep]
        self.name = name or "coordination"

    def __call__(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the coordination number (dimensionless)."""
        x = _x(pos)
        r = _norm(_mi(x[self.pj] - x[self.pi], H))
        return jnp.sum(switching(r, self.r0, self.n, self.m, self.d0))

    def atoms(self) -> np.ndarray:
        """Return the atoms of both groups (unique, sorted)."""
        return np.unique(np.concatenate([self.ga, self.gb]))


class RMSD(CV):
    """Root-mean-square deviation [nm] of a group of atoms from reference positions.

    With align=True the deviation is taken after optimal superposition (translation and rotation)
    by the quaternion method [1]_ [2]_: with x and y the centred positions and reference and
    normalised weights w,

        RMSD^2 = sum_a w_a (|x_a|^2 + |y_a|^2) - 2 lambda_max(K(R)),  R_ij = sum_a w_a x_ai y_aj

    where K is the symmetric 4 x 4 key matrix of R (`_key_matrix`).  lambda_max is differentiated by
    Hellmann-Feynman, lambda' = v^T K' v with the eigenvector v held constant, which stays regular
    when lower eigenvalues are degenerate.  With align=False only the translation is removed.  The
    group is made whole by minimum images from its first atom.

    Attributes
    ----------
    idx : np.ndarray (n,) int
        Atom indices.
    w : np.ndarray (n,)
        Normalised weights (sum 1).
    ref : np.ndarray (n, 3)
        Reference positions, centred with the weights [nm].
    align : bool
        Rotational superposition on or off.

    References
    ----------
    .. [1] S. K. Kearsley, Acta Crystallogr. A 45, 208 (1989).
    .. [2] E. A. Coutsias, C. Seok, K. A. Dill, J. Comput. Chem. 25, 1849 (2004). doi:10.1002/jcc.20110
    """

    def __init__(
        self,
        atoms: ArrayLike,
        ref: ArrayLike,
        weights: ArrayLike | None = None,
        align: bool = True,
        name: str | None = None,
    ) -> None:
        """Set up the CV.

        Parameters
        ----------
        atoms : ArrayLike (n,)
            Atom indices.
        ref : ArrayLike (n, 3)
            Reference positions [nm].
        weights : ArrayLike (n,), optional
            Non-negative weights per atom (normalised to sum 1); None: equal.
        align : bool
            Optimal rotation (True) or translation only (False).
        name : str, optional
            Name; None: "rmsd".

        Raises
        ------
        ValueError
            If `ref` is not (n, 3), or the weights are not one non-negative value per atom with a
            positive sum.
        """
        self.idx = np.asarray(atoms, int).reshape(-1)
        ref = np.asarray(ref, float)
        if ref.shape != (len(self.idx), 3):
            raise ValueError(f"ref must be ({len(self.idx)}, 3) nm")
        w = np.ones(len(self.idx)) if weights is None else np.asarray(weights, float).reshape(-1)
        if w.shape != self.idx.shape or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights: one non-negative value per atom")
        self.w = w / w.sum()
        self.ref = ref - (self.w[:, None] * ref).sum(0)
        self.align = bool(align)
        self.name = name or "rmsd"

    def __call__(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the RMSD [nm] (floored at sqrt(_TINY) under the square root)."""
        x = _x(pos)[self.idx]
        x = x[0] + _mi(x - x[0], H)
        w = jnp.asarray(self.w)
        x = x - jnp.sum(w[:, None] * x, 0)
        y = jnp.asarray(self.ref)
        sxx = jnp.sum(w * jnp.sum(x * x, 1)) + jnp.sum(w * jnp.sum(y * y, 1))
        if not self.align:
            return jnp.sqrt(jnp.maximum(jnp.sum(w * jnp.sum((x - y) ** 2, 1)), _TINY))
        R = jnp.einsum("a,ai,aj->ij", w, x, y, precision=_HI)  # (3, 3) weighted correlation matrix
        K = _key_matrix(R)
        # Hellmann-Feynman: the eigenvector is a constant for autodiff, so d(lam) = v^T dK v
        _, V = jnp.linalg.eigh(jax.lax.stop_gradient(K))
        v = jax.lax.stop_gradient(V[:, -1])  # eigenvector of the largest eigenvalue
        lam = v @ K @ v
        return jnp.sqrt(jnp.maximum(sxx - 2.0 * lam, _TINY))

    def atoms(self) -> np.ndarray:
        """Return the atom indices."""
        return self.idx


def _key_matrix(R: jax.Array) -> jax.Array:
    """Return the symmetric 4 x 4 key matrix whose largest eigenvalue is max_Q sum_a w_a x_a . (Q y_a).

    `R` (3, 3) is the weighted correlation matrix R_ij = sum_a w_a x_ai y_aj; Q runs over rotations
    (quaternion parametrisation).
    """
    Sxx, Sxy, Sxz = R[0, 0], R[0, 1], R[0, 2]
    Syx, Syy, Syz = R[1, 0], R[1, 1], R[1, 2]
    Szx, Szy, Szz = R[2, 0], R[2, 1], R[2, 2]
    return jnp.array(
        [
            [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
            [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
            [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
            [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
        ]
    )


class Linear(CV):
    """Linear combination sum_k c_k s_k + offset of other CVs.

    Not periodic unless `period` is given.
    """

    def __init__(
        self,
        cvs: Sequence[CV],
        coeffs: ArrayLike,
        offset: float = 0.0,
        period: float | None = None,
        name: str | None = None,
    ) -> None:
        """Set up the CV.

        Parameters
        ----------
        cvs : sequence of CV
            The CVs.
        coeffs : ArrayLike
            One coefficient per CV.
        offset : float
            Constant added [CV units].
        period : float, optional
            Period [CV units]; None: not periodic.
        name : str, optional
            Name; None: "linear".

        Raises
        ------
        ValueError
            If the number of coefficients differs from the number of CVs.
        """
        self.cvs = list(cvs)
        self.c = np.asarray(coeffs, float).reshape(-1)
        if len(self.c) != len(self.cvs):
            raise ValueError("one coefficient per CV")
        self.offset = float(offset)
        self.period = None if period is None else float(period)
        self.name = name or "linear"

    def __call__(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return sum_k c_k s_k(pos, H) + offset."""
        return sum(float(c) * cv(pos, H) for c, cv in zip(self.c, self.cvs)) + self.offset

    def atoms(self) -> np.ndarray:
        """Return the atoms of all component CVs (unique, sorted)."""
        return np.unique(np.concatenate([cv.atoms() for cv in self.cvs] + [np.zeros(0, int)]))


class Custom(CV):
    """Any differentiable JAX function fn(pos, H) -> scalar as a CV.

    `period` makes it periodic (values then taken in [lo, lo + period)).  `atoms` only serves the
    index check (`CVSet.check`); the function may read any atom.
    """

    def __init__(
        self,
        fn: Callable[[jax.Array, ArrayLike | None], ArrayLike],
        period: float | None = None,
        lo: float | None = None,
        atoms: ArrayLike | None = None,
        name: str = "custom",
    ) -> None:
        """Set up the CV.

        Parameters
        ----------
        fn : callable
            fn(pos, H) -> scalar, with pos a float64 jax.Array (N, 3) [nm] and H the box [nm] or None.
        period : float, optional
            Period [CV units]; None: not periodic.
        lo : float, optional
            Lower end of the canonical interval [CV units]; None: -period/2.
        atoms : ArrayLike, optional
            Atom indices for the range check; None: none.
        name : str
            Name of the CV.
        """
        self.fn = fn
        self.period = None if period is None else float(period)
        self.lo = None if lo is None else float(lo)
        self._atoms = np.zeros(0, int) if atoms is None else np.asarray(atoms, int).reshape(-1)
        self.name = name

    def __call__(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return fn(pos, H) as a float64 scalar."""
        return jnp.asarray(self.fn(_x(pos), H), jnp.float64)

    def atoms(self) -> np.ndarray:
        """Return the atom indices given to the constructor."""
        return self._atoms


class CVSet:
    """An ordered set of d CVs: the values s (d,), periods and nearest-image arithmetic of one bias.

    Attributes
    ----------
    cvs : list of CV
        The CVs.
    periods : np.ndarray (d,)
        Periods [CV units] (0: not periodic).
    lows : np.ndarray (d,)
        Lower ends of the canonical intervals [CV units] (NaN: -period/2 or not periodic).
    """

    def __init__(self, cvs: CV | Sequence[CV] | CVSet) -> None:
        """Set up the set.

        Parameters
        ----------
        cvs : CV, sequence of CV, or CVSet
            The CVs (a CVSet is copied).

        Raises
        ------
        ValueError
            If there are no CVs.
        TypeError
            If an element is not a `CV`.
        """
        if isinstance(cvs, CVSet):
            cvs = cvs.cvs
        if isinstance(cvs, CV):
            cvs = [cvs]
        self.cvs = list(cvs)
        if not self.cvs:
            raise ValueError("no collective variables")
        for c in self.cvs:
            if not isinstance(c, CV):
                raise TypeError(f"not a CV: {c!r}")
        self.periods = np.array([0.0 if c.period is None else float(c.period) for c in self.cvs])
        self.lows = np.array([np.nan if c.lo is None else float(c.lo) for c in self.cvs])

    def __len__(self) -> int:
        """Return the number of CVs d."""
        return len(self.cvs)

    @property
    def names(self) -> list[str]:
        """Names of the CVs, list of str."""
        return [c.name for c in self.cvs]

    def values(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the CV vector s, jax.Array (d,) float64 [CV units].

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Atom positions [nm].
        H : ArrayLike (3, 3), optional
            Box, lattice vectors as rows (reduced, lower triangular) [nm]; None: no periodic boundaries.
        """
        return jnp.stack([jnp.asarray(c(pos, H), jnp.float64) for c in self.cvs])

    def diff(self, a: ArrayLike, b: ArrayLike) -> jax.Array:
        """Return a - b [CV units] with periodic components wrapped to the nearest image."""
        return wrap(a - b, self.periods)

    def canonical(self, s: ArrayLike) -> jax.Array:
        """Return `s` with periodic components mapped into [lo, lo + period) (lo = -period/2 unless given)."""
        P = jnp.asarray(self.periods)
        lo = jnp.asarray(np.where(np.isnan(self.lows), -0.5 * self.periods, self.lows))
        p = jnp.where(P > 0, P, 1.0)
        return jnp.where(P > 0, lo + jnp.mod(s - lo, p), s)

    def atoms(self) -> np.ndarray:
        """Return the atoms of all CVs (unique, sorted)."""
        return np.unique(np.concatenate([c.atoms() for c in self.cvs] + [np.zeros(0, int)]))

    def check(self, n_atoms: int) -> None:
        """Check that every CV atom index lies in [0, n_atoms).

        Raises
        ------
        ValueError
            If an index is out of range.
        """
        a = self.atoms()
        if a.size and (a.min() < 0 or a.max() >= n_atoms):
            raise ValueError(f"CV atom index out of range for {n_atoms} atoms")
