"""Collective variables (CVs): JAX functions s(pos, H) of the atomic positions (N, 3) nm and the
box H (lattice vectors as rows, nm; None for a non-periodic system).  Their derivatives, and so
every bias force, come from autodiff (jax.grad of V(s(pos, H))): a new CV is a function, with no
hand-coded derivatives.

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
image by the biases (hills and kernels on a circle).  Units: nm, rad."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

_TINY = 1e-60
_HI = jax.lax.Precision.HIGHEST


def _mi(d, H):
    """Minimum image of displacement(s) d (..., 3) in a reduced box H (identity for H None)."""
    if H is None:
        return d
    from ..md.box import min_image
    return min_image(d, jnp.asarray(H, jnp.float64))


def _norm(v):
    return jnp.sqrt(jnp.maximum(jnp.sum(v * v, -1), _TINY))


def wrap(ds, period):
    """Differences ds wrapped to [-period/2, period/2) where period > 0 (arrays broadcast; period
    0 = not periodic)."""
    period = jnp.asarray(period, jnp.float64)
    p = jnp.where(period > 0, period, 1.0)
    return jnp.where(period > 0, ds - p * jnp.round(ds / p), ds)


class CV:
    """A collective variable.  Subclasses define __call__(pos, H) -> scalar and atoms().
    period: None (not periodic) or the period (the value is taken in [lo, lo + period))."""
    name = "cv"
    period = None
    lo = None

    def __call__(self, pos, H=None):
        raise NotImplementedError

    def atoms(self) -> np.ndarray:
        return np.zeros(0, int)

    def grad(self, pos, H=None):
        """ds/dpos (N, 3), by autodiff."""
        return jax.grad(lambda x: self(x, H))(jnp.asarray(pos, jnp.float64))

    def __repr__(self):
        return self.name


def _x(pos):
    return jnp.asarray(pos, jnp.float64)


class Distance(CV):
    """|r_j - r_i| (nm), minimum image."""

    def __init__(self, i: int, j: int, name: str | None = None):
        self.i, self.j = int(i), int(j)
        self.name = name or f"d{self.i}_{self.j}"

    def __call__(self, pos, H=None):
        x = _x(pos)
        return _norm(_mi(x[self.j] - x[self.i], H))

    def atoms(self):
        return np.array([self.i, self.j])


class Component(CV):
    """One Cartesian component (axis 0, 1, 2) of an atom's position (nm), or of its displacement
    from `origin`.  Not periodic (for model systems and walls; use it with H None or unwrapped
    coordinates)."""

    def __init__(self, i: int, axis: int, origin=0.0, name: str | None = None):
        self.i, self.axis, self.origin = int(i), int(axis), float(origin)
        self.name = name or f"{'xyz'[self.axis]}{self.i}"

    def __call__(self, pos, H=None):
        return _x(pos)[self.i, self.axis] - self.origin

    def atoms(self):
        return np.array([self.i])


class Angle(CV):
    """Angle i-j-k (rad, [0, pi])."""

    def __init__(self, i: int, j: int, k: int, name: str | None = None):
        self.idx = (int(i), int(j), int(k))
        self.name = name or "angle" + "_".join(map(str, self.idx))

    def __call__(self, pos, H=None):
        x = _x(pos)
        i, j, k = self.idx
        u, v = _mi(x[i] - x[j], H), _mi(x[k] - x[j], H)
        return jnp.arctan2(_norm(jnp.cross(u, v)), jnp.sum(u * v, -1))

    def atoms(self):
        return np.array(self.idx)


class Dihedral(CV):
    """Dihedral i-j-k-l (rad, (-pi, pi]; IUPAC sign as md/restraints.py and pgm_jax.bonded).
    Periodic with period 2 pi."""
    period = 2.0 * np.pi
    lo = -np.pi

    def __init__(self, i: int, j: int, k: int, l: int, name: str | None = None):   # noqa: E741
        self.idx = (int(i), int(j), int(k), int(l))
        self.name = name or "dih" + "_".join(map(str, self.idx))

    def __call__(self, pos, H=None):
        from ..md.restraints import dihedral_from_bonds
        x = _x(pos)
        a, b, c, d = self.idx
        return dihedral_from_bonds(_mi(x[b] - x[a], H), _mi(x[c] - x[b], H), _mi(x[d] - x[c], H))

    def atoms(self):
        return np.array(self.idx)


def _centre(x, g, w, H):
    """Weighted centre of group g from minimum-image displacements to its first atom."""
    d = _mi(x[g] - x[g[0]], H)
    return x[g[0]] + jnp.sum(jnp.asarray(w)[:, None] * d, 0) / float(np.sum(w))


class COMDistance(CV):
    """Distance (nm) between the weighted centres of two groups (weights from `masses`, the
    per-atom masses of the whole system, or equal).  A group must be smaller than half the box."""

    def __init__(self, group_a, group_b, masses=None, name: str | None = None):
        self.ga, self.gb = (np.asarray(g, int).reshape(-1) for g in (group_a, group_b))
        if not len(self.ga) or not len(self.gb):
            raise ValueError("empty group")
        w = (lambda g: np.ones(len(g))) if masses is None else (lambda g: np.asarray(masses, float)[g])
        self.wa, self.wb = w(self.ga), w(self.gb)
        self.name = name or "com_distance"

    def __call__(self, pos, H=None):
        x = _x(pos)
        return _norm(_mi(_centre(x, self.gb, self.wb, H) - _centre(x, self.ga, self.wa, H), H))

    def atoms(self):
        return np.concatenate([self.ga, self.gb])


def switching(r, r0: float, n: int = 6, m: int = 12, d0: float = 0.0):
    """PLUMED's rational switching function s = (1 - x^n) / (1 - x^m), x = (r - d0) / r0 (1 for
    r <= d0), with its limit n/m + n (n - m) / (2 m) (x - 1) near x = 1 (no 0/0)."""
    x = jnp.maximum((r - d0) / r0, 0.0)
    e = x - 1.0
    near = jnp.abs(e) < 1e-4
    xs = jnp.where(near, 0.5, x)
    f = (1.0 - xs ** n) / (1.0 - xs ** m)
    return jnp.where(near, n / m + n * (n - m) / (2.0 * m) * e, f)


class Coordination(CV):
    """Coordination number sum_{i in A, j in B, i != j} s(|r_ij|) with the rational switching
    function `switching` (r0 nm, exponents n < m, offset d0).  Every pair of the two groups is
    evaluated (|A| x |B| distances: groups of up to a few thousand pairs); pairs listed twice when
    the groups overlap count twice, as in PLUMED's COORDINATION with GROUPA/GROUPB."""

    def __init__(self, group_a, group_b, r0: float, n: int = 6, m: int = 12, d0: float = 0.0,
                 name: str | None = None):
        self.ga, self.gb = (np.asarray(g, int).reshape(-1) for g in (group_a, group_b))
        if int(n) >= int(m):
            raise ValueError("switching exponents need n < m")
        self.r0, self.n, self.m, self.d0 = float(r0), int(n), int(m), float(d0)
        ia, ib = np.meshgrid(self.ga, self.gb, indexing="ij")
        keep = ia != ib
        self.pi, self.pj = ia[keep], ib[keep]
        self.name = name or "coordination"

    def __call__(self, pos, H=None):
        x = _x(pos)
        r = _norm(_mi(x[self.pj] - x[self.pi], H))
        return jnp.sum(switching(r, self.r0, self.n, self.m, self.d0))

    def atoms(self):
        return np.unique(np.concatenate([self.ga, self.gb]))


class RMSD(CV):
    """Root-mean-square deviation (nm) of `atoms` from the reference positions `ref` (n, 3), after
    optimal superposition (align=True: translation and rotation, the quaternion method of Kearsley /
    Coutsouris; the largest eigenvalue of the 4 x 4 key matrix is differentiated by
    Hellmann-Feynman, lambda' = v^T K' v, which stays regular at degenerate lower eigenvalues) or
    after translation only (align=False).  weights: per atom (default equal).  The group is made
    whole by minimum images from its first atom."""

    def __init__(self, atoms, ref, weights=None, align: bool = True, name: str | None = None):
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

    def __call__(self, pos, H=None):
        x = _x(pos)[self.idx]
        x = x[0] + _mi(x - x[0], H)
        w = jnp.asarray(self.w)
        x = x - jnp.sum(w[:, None] * x, 0)
        y = jnp.asarray(self.ref)
        sxx = jnp.sum(w * jnp.sum(x * x, 1)) + jnp.sum(w * jnp.sum(y * y, 1))
        if not self.align:
            return jnp.sqrt(jnp.maximum(jnp.sum(w * jnp.sum((x - y) ** 2, 1)), _TINY))
        R = jnp.einsum("a,ai,aj->ij", w, x, y, precision=_HI)
        K = _key_matrix(R)
        _, V = jnp.linalg.eigh(jax.lax.stop_gradient(K))
        v = jax.lax.stop_gradient(V[:, -1])
        lam = v @ K @ v
        return jnp.sqrt(jnp.maximum(sxx - 2.0 * lam, _TINY))

    def atoms(self):
        return self.idx


def _key_matrix(R):
    """Symmetric 4 x 4 matrix whose largest eigenvalue is max_rotation sum_a w_a x_a . (Q y_a)."""
    Sxx, Sxy, Sxz = R[0, 0], R[0, 1], R[0, 2]
    Syx, Syy, Syz = R[1, 0], R[1, 1], R[1, 2]
    Szx, Szy, Szz = R[2, 0], R[2, 1], R[2, 2]
    return jnp.array([
        [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
        [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
        [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
        [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]])


class Linear(CV):
    """sum_k c_k s_k + offset.  Not periodic unless `period` is given."""

    def __init__(self, cvs, coeffs, offset: float = 0.0, period=None, name: str | None = None):
        self.cvs = list(cvs)
        self.c = np.asarray(coeffs, float).reshape(-1)
        if len(self.c) != len(self.cvs):
            raise ValueError("one coefficient per CV")
        self.offset = float(offset)
        self.period = None if period is None else float(period)
        self.name = name or "linear"

    def __call__(self, pos, H=None):
        return sum(float(c) * cv(pos, H) for c, cv in zip(self.c, self.cvs)) + self.offset

    def atoms(self):
        return np.unique(np.concatenate([cv.atoms() for cv in self.cvs] + [np.zeros(0, int)]))


class Custom(CV):
    """Any JAX function fn(pos, H) -> scalar (differentiable); `period` if periodic (then values
    are taken in [lo, lo + period))."""

    def __init__(self, fn, period=None, lo=None, atoms=None, name: str = "custom"):
        self.fn = fn
        self.period = None if period is None else float(period)
        self.lo = None if lo is None else float(lo)
        self._atoms = np.zeros(0, int) if atoms is None else np.asarray(atoms, int).reshape(-1)
        self.name = name

    def __call__(self, pos, H=None):
        return jnp.asarray(self.fn(_x(pos), H), jnp.float64)

    def atoms(self):
        return self._atoms


class CVSet:
    """An ordered set of d CVs: values(pos, H) -> (d,), periods (d,) (0 = not periodic)."""

    def __init__(self, cvs):
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

    def __len__(self):
        return len(self.cvs)

    @property
    def names(self):
        return [c.name for c in self.cvs]

    def values(self, pos, H=None):
        return jnp.stack([jnp.asarray(c(pos, H), jnp.float64) for c in self.cvs])

    def diff(self, a, b):
        """a - b with periodic components wrapped to the nearest image."""
        return wrap(a - b, self.periods)

    def canonical(self, s):
        """Periodic components mapped into [lo, lo + period) (lo = -period/2 unless given)."""
        P = jnp.asarray(self.periods)
        lo = jnp.asarray(np.where(np.isnan(self.lows), -0.5 * self.periods, self.lows))
        p = jnp.where(P > 0, P, 1.0)
        return jnp.where(P > 0, lo + jnp.mod(s - lo, p), s)

    def atoms(self):
        return np.unique(np.concatenate([c.atoms() for c in self.cvs] + [np.zeros(0, int)]))

    def check(self, n_atoms: int) -> None:
        a = self.atoms()
        if a.size and (a.min() < 0 or a.max() >= n_atoms):
            raise ValueError(f"CV atom index out of range for {n_atoms} atoms")
