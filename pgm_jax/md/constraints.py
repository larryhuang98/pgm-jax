"""Holonomic distance constraints for MD: SHAKE positions and RATTLE momenta.

Contents: `Constraints` (the constraints of a system, split into clusters and solved by blocks:
`_DenseBlock` for small clusters, `_SparseBlock` for large ones), the helpers `_clusters`,
`_solve_small`, `_solve_gauss`, and hydrogen mass repartitioning (`repartition_masses`,
`hmr_masses`).

The constraints (i, j, d0) split into clusters, the connected components of the constraint graph
(a water triangle, a CH3 group, an O-H bond, a whole molecule with every bond constrained):

  positions (SHAKE)  r' = r~ + M^-1 sum_c lambda_c s_c (+ on atom a_c, - on b_c), s_c the constraint
                     vectors before the move, lambda solving
                     sigma_c(lambda) = |r'_a - r'_b|^2 - d0^2 = 0;
  momenta (RATTLE)   p' = p + sum_c mu_c r_c (+ / -), mu from the linear system
                     r_c . (v'_a - v'_b) = 0, i.e. (B M^-1 B^T) mu = -B v with B the constraint gradients.

Two solvers, chosen per cluster (float64 throughout; positions are float64 in every precision mode):

  dense    clusters of at most `dense_max` constraints (X-H groups, water, small molecules with every
           bond constrained): exact Newton on the cluster's small dense system, a fixed number of
           iterations (`n_iter`, unrolled; quadratic convergence, relative errors ~1e-3, 1e-6, 1e-12
           after 1, 2, 3 iterations for the displacements of an MD step).  Clusters of up to 3
           constraints (X-H, XH2, XH3, water) form one block, larger ones one block per size; all
           clusters of a block are solved at once (one per molecule copy: the per-template
           batching); systems of <= 3 are solved in closed form, larger ones
           by unrolled Gaussian elimination (the matrices are Gram matrices of the mass-weighted
           constraint gradients: symmetric positive definite, no pivoting needed).
  sparse   larger clusters (a protein with every bond constrained): one flat list, matrix-free.
           SHAKE by quasi-Newton iterations lambda -= J0^-1 sigma with J0 = 2 S K S^T, the Jacobian
           at the reference vectors (symmetric positive definite, fixed during a call), each solve by
           Jacobi-preconditioned CG; RATTLE by CG on R K R^T.  Both iterate to a tolerance
           (`tol`, relative; while loops).

The integrator (flexible.py) applies them in g-BAOAB order: SHAKE after every drift, RATTLE after
every kick, drift and thermostat step.  With X-H bonds constrained (`constraints="h-bonds"`) flexible
molecules run at 2 fs, with hydrogen mass repartitioning (`repartition_masses`; `hmr_masses`: one
hydrogen mass, or one per molecule) at 4 fs (docs/shake.md, docs/protein_ff.md).

Every solver works on arrays padded with one dummy atom (row N, inverse mass 0).  The solvers are
traceable JAX functions (called inside the compiled step); the clusters and blocks are built once
on the host.  The position and momentum steps are SHAKE [1]_ and RATTLE [2]_.

Units: nm, ps, amu.

References
----------
.. [1] J.-P. Ryckaert, G. Ciccotti, H. J. C. Berendsen, J. Comput. Phys. 23, 327 (1977).
.. [2] H. C. Andersen, J. Comput. Phys. 52, 24 (1983).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from ..system import System

DENSE_MAX = 12  # clusters with more constraints go to the matrix-free (sparse) solver


def _clusters(pairs: np.ndarray, n_atoms: int) -> list[tuple[list[int], list[int]]]:
    """Return the connected components of the constraint graph (union-find; host).

    Parameters
    ----------
    pairs : np.ndarray (nc, 2) int
        Constrained atom pairs.
    n_atoms : int
        Number of atoms.

    Returns
    -------
    list of (atoms, constraints)
        Per cluster: its sorted atoms and its constraint indices.
    """
    parent = list(range(n_atoms))

    def find(a: int) -> int:
        """Return the root of a (with path halving)."""
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    comp = {}
    for c, (i, _j) in enumerate(pairs):
        comp.setdefault(find(int(i)), []).append(c)
    out = []
    for cs in comp.values():
        atoms = sorted({int(a) for c in cs for a in pairs[c]})
        out.append((atoms, cs))
    return out


def _solve_gauss(A: jax.Array, b: jax.Array) -> jax.Array:
    """Return x solving the batched systems A x = b by unrolled Gaussian elimination.

    For symmetric positive definite (or nearly so) (..., C, C) systems, without pivoting:
    elementwise operations on the batch, which XLA fuses.  b and x are (..., C).
    """
    C = A.shape[-1]
    a = [[A[..., i, j] for j in range(C)] for i in range(C)]
    r = [b[..., i] for i in range(C)]
    for k in range(C):
        inv = 1.0 / a[k][k]
        for i in range(k + 1, C):
            f = a[i][k] * inv
            for j in range(k + 1, C):
                a[i][j] = a[i][j] - f * a[k][j]
            r[i] = r[i] - f * r[k]
    x = [None] * C
    for i in reversed(range(C)):
        s = r[i]
        for j in range(i + 1, C):
            s = s - a[i][j] * x[j]
        x[i] = s / a[i][i]
    return jnp.stack(x, -1)


def _solve_small(A: jax.Array, b: jax.Array) -> jax.Array:
    """Return x solving the batched (..., C, C) systems A x = b, closed form for C <= 3.

    jnp.linalg.solve launches an LU factorisation per call, slow for many tiny systems on GPUs;
    above C = 3 unrolled elimination (`_solve_gauss`).  b and x are (..., C).
    """
    C = A.shape[-1]
    if C == 1:
        return b / A[..., 0, 0:1]
    if C == 2:
        a, bb, c, d = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
        det = a * d - bb * c
        return jnp.stack([d * b[..., 0] - bb * b[..., 1], -c * b[..., 0] + a * b[..., 1]], -1) / det[..., None]
    if C == 3:
        r0, r1, r2 = A[..., 0, :], A[..., 1, :], A[..., 2, :]
        c0, c1, c2 = jnp.cross(r1, r2), jnp.cross(r2, r0), jnp.cross(r0, r1)  # adjugate columns
        det = jnp.sum(r0 * c0, -1)
        return (c0 * b[..., 0:1] + c1 * b[..., 1:2] + c2 * b[..., 2:3]) / det[..., None]
    return _solve_gauss(A, b)


class _DenseBlock:
    """Clusters of at most C constraints padded to (K clusters, A atoms, C constraints): exact Newton.

    Works on padded arrays (row n: a dummy atom of zero inverse mass).  Padding constraints are
    masked (`cmask`) and get identity rows in the Newton and RATTLE matrices.

    Attributes
    ----------
    atoms : jax.Array (K, A) int32
        Atoms of every cluster (padding n).
    inc : jax.Array (K, A, C)
        Incidence: +1 at a_c, -1 at b_c.
    d2 : jax.Array (K, C)
        Squared constraint lengths [nm^2] (1 for padding).
    cmask : jax.Array (K, C)
        1 for real constraints.
    ends : jax.Array (K, C, 2) int32
        Local atom indices (a_c, b_c).
    kmat : jax.Array (K, C, C)
        K_cd = sum_a inc_ac inc_ad / m_a [1/amu] (after `set_masses`).
    """

    kind = "dense"

    def __init__(
        self, clusters: list[tuple[list[int], list[int]]], pairs: np.ndarray, d0: np.ndarray, n: int, n_iter: int
    ) -> None:
        """Build the padded cluster arrays.

        Parameters
        ----------
        clusters : list of (atoms, constraints)
            Clusters of this block (`_clusters`).
        pairs : np.ndarray (nc, 2) int
            All constrained pairs.
        d0 : np.ndarray (nc,)
            All constraint lengths [nm].
        n : int
            Number of atoms (index of the padding atom).
        n_iter : int
            Newton iterations (unrolled).
        """
        A = max(len(a) for a, _ in clusters)
        C = max(len(c) for _, c in clusters)
        K = len(clusters)
        atoms = np.full((K, A), n, np.int32)
        inc = np.zeros((K, A, C))  # incidence: +1 at a_c, -1 at b_c
        d = np.ones((K, C))
        cmask = np.zeros((K, C))
        ends = np.zeros((K, C, 2), np.int32)
        for k, (at, cs) in enumerate(clusters):
            atoms[k, : len(at)] = at
            loc = {a: s for s, a in enumerate(at)}
            for s, c in enumerate(cs):
                i, j = loc[int(pairs[c, 0])], loc[int(pairs[c, 1])]
                inc[k, i, s], inc[k, j, s] = 1.0, -1.0
                ends[k, s] = (i, j)
                d[k, s], cmask[k, s] = d0[c], 1.0
        self.n, self.n_iter = n, int(n_iter)
        self.nc = int(cmask.sum())
        self.atoms = jnp.asarray(atoms)
        self.inc = jnp.asarray(inc)
        self.d2 = jnp.asarray(d * d)
        self.cmask = jnp.asarray(cmask)
        self.ends = jnp.asarray(ends)
        self.shape = (K, A, C)
        self._eye = jnp.eye(C)
        self._off = jnp.asarray(cmask[:, :, None] * cmask[:, None, :])
        self._pad = jnp.asarray(1.0 - cmask)[..., None] * self._eye

    def describe(self) -> str:
        """Return e.g. "512 x (3 atoms, 3 constraints)"."""
        K, A, C = self.shape
        return f"{K} x ({A} atoms, {C} constraints)"

    def set_masses(self, invm_padded: ArrayLike) -> None:
        """Set the inverse masses (N + 1,) [1/amu] (padding 0) and the coupling matrices K."""
        self.invm = jnp.asarray(invm_padded)[self.atoms]  # (K, A)
        # K_cd = sum_a inc_ac inc_ad / m_a: how constraint forces c move the length of d
        self.kmat = jnp.einsum("kac,ka,kad->kcd", self.inc, self.invm, self.inc)

    def _vectors(self, X: jax.Array) -> jax.Array:
        """Return the constraint vectors r_a - r_b (K, C, 3) of cluster positions X (K, A, 3)."""
        i, j = self.ends[..., 0], self.ends[..., 1]

        def take(idx: jax.Array) -> jax.Array:  # X[k, idx[k, c]] (K, C, 3)
            return jnp.take_along_axis(X, idx[..., None], axis=1)

        return take(i) - take(j)

    def _move(self, w: jax.Array, V: jax.Array) -> jax.Array:
        """Return the displacements sum_c inc_ac w_c V_c / m_a (K, A, 3) of weights w (K, C).

        Written as broadcast products, which XLA fuses; an einsum becomes batched tiny matrix
        products, 4x slower on the GPU.
        """
        return self.invm[..., None] * jnp.sum(self.inc[..., None] * (w[:, None, :, None] * V[:, None]), axis=2)

    @staticmethod
    def _gram(R: jax.Array, S: jax.Array) -> jax.Array:
        """Return the Gram matrices R_c . S_d (K, C, C) of vectors R, S (K, C, 3)."""
        return jnp.sum(R[:, :, None, :] * S[:, None, :, :], -1)

    def positions(self, xp: jax.Array, rp: jax.Array) -> jax.Array:
        """Return the padded positions with this block's clusters moved onto the constraints (SHAKE).

        Newton on sigma_c(lambda) = |R_c(lambda)|^2 - d_c^2 with R = r_a - r_b of
        X + move(lambda, S), S the constraint vectors of the reference positions; the Jacobian is
        J_cd = 2 (R_c . S_d) K_cd.  `n_iter` iterations, unrolled.

        Parameters
        ----------
        xp : jax.Array (N + 1, 3)
            Unconstrained positions, padded [nm].
        rp : jax.Array (N + 1, 3)
            Reference positions (before the move), padded [nm].

        Returns
        -------
        jax.Array (N + 1, 3)
            Positions [nm].
        """
        X, S = xp[self.atoms], self._vectors(rp[self.atoms])

        def resid(lam: jax.Array) -> tuple[jax.Array, jax.Array]:  # constraint vectors, sigma (masked)
            R = self._vectors(X + self._move(lam, S))
            return R, (jnp.sum(R * R, -1) - self.d2) * self.cmask

        lam = jnp.zeros(self.shape[::2], X.dtype)
        R, sig = resid(lam)
        for _ in range(self.n_iter):  # fixed count, unrolled: no reduction per step
            J = 2.0 * self._gram(R, S) * self.kmat * self._off + self._pad
            lam = lam - _solve_small(J, sig)
            R, sig = resid(lam)
        return xp.at[self.atoms].set(X + self._move(lam, S))

    def momenta(self, xp: jax.Array, pp: jax.Array, mp: jax.Array) -> jax.Array:
        """Return the padded momenta with the constraint-violating components removed (RATTLE).

        Solves (R_c . R_d) K_cd mu_d = -R_c . (v_a - v_b) per cluster and adds the constraint
        impulses.  xp (N + 1, 3) [nm], pp (N + 1, 3) [amu nm/ps], mp (N + 1,) [amu] (padding 1).
        """
        m = mp[self.atoms][..., None]
        R = self._vectors(xp[self.atoms])
        V = pp[self.atoms] / m
        rv = jnp.sum(R * self._vectors(V), -1) * self.cmask
        M = self._gram(R, R) * self.kmat * self._off + self._pad
        mu = -_solve_small(M, rv)
        return pp.at[self.atoms].set((V + self._move(mu, R)) * m)

    def errors(self, xp: jax.Array, pp: jax.Array | None, mp: jax.Array | None) -> tuple:
        """Return the block's errors: max relative length error, max |r^ . (v_a - v_b)|, sum |v_a - v_b|^2.

        Velocities [nm/ps]; without momenta (pp None) the last two are 0.
        """
        X = xp[self.atoms]
        R = self._vectors(X)
        r = jnp.sqrt(jnp.sum(R * R, -1))
        dl = jnp.max(jnp.abs(r / jnp.sqrt(self.d2) - 1.0) * self.cmask)
        if pp is None:
            return dl, 0.0, 0.0
        dV = self._vectors(pp[self.atoms] / mp[self.atoms][..., None])
        rv = jnp.abs(jnp.sum(R * dV, -1)) / jnp.where(r > 0, r, 1.0) * self.cmask
        return dl, jnp.max(rv), jnp.sum(jnp.sum(dV * dV, -1) * self.cmask)


class _SparseBlock:
    """Large clusters as one flat list of constraints, matrix-free.

    SHAKE by quasi-Newton iterations with J0 = 2 S K S^T solved by preconditioned CG, RATTLE by CG
    on R K R^T (while loops to `tol`).  K is the constraint coupling through the inverse masses
    (applied as `_vec(_move(.))`, never formed).

    Attributes
    ----------
    atoms : jax.Array (na,) int32
        Atoms of the block (global).
    al, bl : jax.Array (nc,) int32
        Local indices of the constraint ends.
    inc_c, inc_s : jax.Array (na, D)
        Constraints of every atom (padding nc) and their signs.
    d2 : jax.Array (nc,)
        Squared lengths [nm^2].
    tol, rattle_tol : float
        Relative tolerances of SHAKE (max |sigma_c| / d_c^2) and RATTLE (CG residual).
    max_iter, cg_max : int
        Largest numbers of quasi-Newton and CG iterations.
    inner_tol : float
        Relative tolerance of the CG inside each quasi-Newton iteration.
    unroll : int
        CG iterations per convergence test (class attribute).
    """

    kind = "sparse"

    unroll = 8  # CG iterations per convergence test (each test is a device-to-host sync)

    def __init__(
        self,
        pairs: ArrayLike,
        d0: ArrayLike,
        n: int,
        tol: float = 1e-10,
        max_iter: int = 100,
        inner_tol: float = 1e-3,
        cg_max: int = 400,
        rattle_tol: float = 1e-11,
    ) -> None:
        """Build the flat constraint list; arguments as in the class Attributes (pairs global)."""
        pairs = np.asarray(pairs, int).reshape(-1, 2)
        atoms = np.unique(pairs)
        loc = {int(a): k for k, a in enumerate(atoms)}
        al = np.array([loc[int(i)] for i in pairs[:, 0]], np.int32)
        bl = np.array([loc[int(j)] for j in pairs[:, 1]], np.int32)
        nc, na = len(pairs), len(atoms)
        rows = [[] for _ in range(na)]
        for c in range(nc):
            rows[al[c]].append((c, 1.0))
            rows[bl[c]].append((c, -1.0))
        D = max(len(r) for r in rows)
        inc_c = np.full((na, D), nc, np.int32)  # padding: a zero constraint vector
        inc_s = np.zeros((na, D))
        for a, r in enumerate(rows):
            for s, (c, sg) in enumerate(r):
                inc_c[a, s], inc_s[a, s] = c, sg
        self.n, self.nc = n, nc
        self.atoms = jnp.asarray(atoms.astype(np.int32))
        self.al, self.bl = jnp.asarray(al), jnp.asarray(bl)
        self.inc_c, self.inc_s = jnp.asarray(inc_c), jnp.asarray(inc_s)
        self.d2 = jnp.asarray(np.asarray(d0, float) ** 2)
        self.tol, self.max_iter, self.inner_tol, self.cg_max = float(tol), int(max_iter), float(inner_tol), int(cg_max)
        self.rattle_tol = float(rattle_tol)
        self.shape = (na, D, nc)

    def describe(self) -> str:
        """Return e.g. "iterative: 1200 constraints on 1150 atoms (tol 1e-10)"."""
        na, D, nc = self.shape
        return f"iterative: {nc} constraints on {na} atoms (tol {self.tol:g})"

    def set_masses(self, invm_padded: ArrayLike) -> None:
        """Set the inverse masses (N + 1,) [1/amu] and the Jacobi weights 1/m_a + 1/m_b."""
        self.invm = jnp.asarray(invm_padded)[self.atoms]  # (na,)
        self._w = self.invm[self.al] + self.invm[self.bl]

    def _acc(self, W: jax.Array) -> jax.Array:
        """Return sum over each atom's constraints of +-W_c (na, 3) (W (nc, 3))."""
        Wp = jnp.concatenate([W, jnp.zeros((1, 3), W.dtype)], 0)
        return jnp.sum(self.inc_s[..., None] * Wp[self.inc_c], axis=1)

    def _move(self, w: jax.Array, V: jax.Array) -> jax.Array:
        """Return the displacements sum_c +-w_c V_c / m_a (na, 3) of weights w (nc,)."""
        return self.invm[:, None] * self._acc(w[:, None] * V)

    def _vec(self, Y: jax.Array) -> jax.Array:
        """Return the constraint vectors Y_a - Y_b (nc, 3)."""
        return Y[self.al] - Y[self.bl]

    def _cg(self, V: jax.Array, b: jax.Array, tol: float, max_iter: int) -> tuple[jax.Array, jax.Array]:
        """Solve (V K V^T) x = b by Jacobi-preconditioned CG (a lax.while_loop).

        V K V^T is the Gram matrix of the mass-weighted constraint gradients; iterations stop when
        |r|_max <= tol |b|_max or after max_iter, with `unroll` iterations between tests.

        Returns
        -------
        x : jax.Array (nc,)
            Solution.
        iterations : jax.Array () int
            Iterations done (a multiple of `unroll`).
        """

        def op(x: jax.Array) -> jax.Array:  # (V K V^T) x
            return jnp.sum(V * self._vec(self._move(x, V)), -1)

        dinv = 1.0 / (jnp.sum(V * V, -1) * self._w)
        stop = tol * jnp.max(jnp.abs(b))
        z = dinv * b
        c0 = (jnp.zeros_like(b), b, z, jnp.sum(b * z), 0)

        def cond(c: tuple) -> jax.Array:  # carry (x, r, d, r.z, it): residual above tol and iterations left
            return (jnp.max(jnp.abs(c[1])) > stop) & (c[4] < max_iter)

        def body(c: tuple) -> tuple:
            """Run `unroll` preconditioned CG iterations (guarded against r.z = 0 after convergence)."""
            x, r, d, rz, it = c
            for _ in range(self.unroll):
                Ad = op(d)
                a = rz / jnp.where(rz != 0, jnp.sum(d * Ad), 1.0)
                x, r = x + a * d, r - a * Ad
                z = dinv * r
                rz1 = jnp.sum(r * z)
                d = z + (rz1 / jnp.where(rz != 0, rz, 1.0)) * d
                rz = rz1
            return x, r, d, rz, it + self.unroll

        out = jax.lax.while_loop(cond, body, c0)
        return out[0], out[4]

    def positions(self, xp: jax.Array, rp: jax.Array) -> jax.Array:
        """Return the padded positions with the block moved onto the constraints (quasi-Newton SHAKE).

        lambda -= J0^-1 sigma with J0 = 2 S K S^T at the reference vectors S, each solve by CG to
        `inner_tol`, until max |sigma_c| / d_c^2 <= tol or `max_iter` iterations.  xp, rp
        (N + 1, 3) [nm] as in _DenseBlock.positions.
        """
        X = xp[self.atoms]
        S = self._vec(rp[self.atoms])

        def resid(lam: jax.Array) -> tuple[jax.Array, jax.Array]:  # moved positions, sigma
            Y = X + self._move(lam, S)
            R = self._vec(Y)
            return Y, jnp.sum(R * R, -1) - self.d2

        def cond(c: tuple) -> jax.Array:  # carry (lambda, Y, sigma, it)
            return (jnp.max(jnp.abs(c[2]) / self.d2) > self.tol) & (c[3] < self.max_iter)

        def body(c: tuple) -> tuple:  # one quasi-Newton iteration
            lam, _, sig, it = c
            lam = lam - self._cg(S, 0.5 * sig, self.inner_tol, self.cg_max)[0]  # J0 = 2 S K S^T
            Y, sig = resid(lam)
            return lam, Y, sig, it + 1

        lam = jnp.zeros(self.nc, X.dtype)
        Y, sig = resid(lam)
        Y = jax.lax.while_loop(cond, body, (lam, Y, sig, 0))[1]
        return xp.at[self.atoms].set(Y)

    def momenta(self, xp: jax.Array, pp: jax.Array, mp: jax.Array) -> jax.Array:
        """Return the padded momenta projected onto the constraint tangent space (RATTLE by CG).

        Solves (R K R^T) mu = -R . (v_a - v_b) to `rattle_tol`; arguments as _DenseBlock.momenta.
        """
        m = mp[self.atoms][:, None]
        R = self._vec(xp[self.atoms])
        V = pp[self.atoms] / m
        rv = jnp.sum(R * self._vec(V), -1)
        mu = self._cg(R, -rv, self.rattle_tol, self.cg_max)[0]
        return pp.at[self.atoms].set((V + self._move(mu, R)) * m)

    def errors(self, xp: jax.Array, pp: jax.Array | None, mp: jax.Array | None) -> tuple:
        """Return the block's errors as _DenseBlock.errors."""
        R = self._vec(xp[self.atoms])
        r = jnp.sqrt(jnp.sum(R * R, -1))
        dl = jnp.max(jnp.abs(r / jnp.sqrt(self.d2) - 1.0))
        if pp is None:
            return dl, 0.0, 0.0
        dV = self._vec(pp[self.atoms] / mp[self.atoms][:, None])
        rv = jnp.abs(jnp.sum(R * dV, -1)) / r
        return dl, jnp.max(rv), jnp.sum(dV * dV)


class Constraints:
    """Distance constraints of a system, solved by SHAKE (positions) and RATTLE (momenta).

    The constraints are split into clusters (module docstring) and the clusters into blocks: dense
    blocks of small clusters and one sparse block of the large ones.  Built on the host; the
    methods are traceable (called inside the compiled step).  Not a pytree.

        cons = Constraints(pairs, d0, masses)
        x = cons.positions(x_new, x_old)          # SHAKE
        p = cons.momenta(x, p, masses)            # RATTLE

    Attributes
    ----------
    n : int
        Number of atoms N.
    nc : int
        Number of constraints.
    n_iter : int
        Newton iterations of the dense solver.
    n_clusters : int
        Number of clusters.
    blocks : list of _DenseBlock and _SparseBlock
        The solver blocks.
    """

    def __init__(
        self,
        pairs: ArrayLike,
        d0: ArrayLike,
        masses: ArrayLike,
        n_iter: int = 4,
        dense_max: int = DENSE_MAX,
        tol: float = 1e-10,
        rattle_tol: float = 1e-11,
        bucket: bool = True,
    ) -> None:
        """Split the constraints into clusters and blocks.

        Parameters
        ----------
        pairs : ArrayLike (nc, 2) int
            Constrained atom pairs.
        d0 : ArrayLike (nc,)
            Constraint lengths [nm].
        masses : ArrayLike (N,)
            Masses [amu] (massless virtual sites are never constrained).
        n_iter : int
            Newton iterations of the dense solver.
        dense_max : int
            Largest cluster (number of constraints) for the dense solver.
        tol : float
            Relative tolerance of the iterative SHAKE (large clusters: max |sigma_c| / d_c^2).
        rattle_tol : float
            Tolerance of its RATTLE (max residual of B v relative to its initial value).
        bucket : bool
            Clusters of up to 3 constraints form one block and larger ones one block per size
            (False: every small cluster padded to the largest, one block).  Each block costs a few
            kernel launches per call, so on a GPU one padded block of X-H groups and waters is
            faster than one block per size (docs/shake.md).
        """
        pairs = np.asarray(pairs, int).reshape(-1, 2)
        d0 = np.asarray(d0, float).reshape(-1)
        self.n = len(masses)
        self.nc = len(pairs)
        self.n_iter = int(n_iter)
        cl = _clusters(pairs, self.n)
        self.n_clusters = len(cl)
        small = [c for c in cl if len(c[1]) <= dense_max]
        big = [c for c in cl if len(c[1]) > dense_max]
        groups = {}
        for c in small:  # <= 3 constraints: one block (closed-form solves)
            groups.setdefault(max(len(c[1]), 3) if bucket else 0, []).append(c)
        self.blocks = [_DenseBlock(g, pairs, d0, self.n, n_iter) for _, g in sorted(groups.items())]
        if big:
            idx = np.array([c for _, cs in big for c in cs], int)
            self.blocks.append(_SparseBlock(pairs[idx], d0[idx], self.n, tol, rattle_tol=rattle_tol))
        m = np.asarray(masses, float)
        invm = np.concatenate(
            [np.divide(1.0, m, out=np.zeros_like(m), where=m != 0), [0.0]]
        )  # massless sites: never in a cluster
        self.set_masses(invm)

    def set_masses(self, invm_padded: ArrayLike) -> None:
        """Set the inverse masses (N + 1,) [1/amu] (last entry: the padding atom, 0) of every block."""
        for b in self.blocks:
            b.set_masses(invm_padded)

    # compatibility: the constrained atoms and their inverse masses as the solvers see them (flat)
    @property
    def atoms(self) -> jax.Array:
        """Constrained atoms of every block, flattened (padding N included)."""
        return jnp.concatenate([b.atoms.reshape(-1) for b in self.blocks]) if self.blocks else jnp.zeros(0, jnp.int32)

    @property
    def invm(self) -> jax.Array:
        """Inverse masses [1/amu] of `atoms`, as the solvers see them."""
        return jnp.concatenate([b.invm.reshape(-1) for b in self.blocks]) if self.blocks else jnp.zeros(0)

    def describe(self) -> str:
        """Return one line for the log header (constraints, clusters, blocks)."""
        if not self.nc:
            return "no constraints"
        return f"{self.nc} constraints in {self.n_clusters} clusters: " + "; ".join(b.describe() for b in self.blocks)

    @staticmethod
    def _pad(x: jax.Array) -> jax.Array:
        """Return x with one zero row appended (the padding atom)."""
        return jnp.concatenate([x, jnp.zeros((1,) + x.shape[1:], x.dtype)], 0)

    def positions(self, x_new: jax.Array, x_ref: jax.Array) -> jax.Array:
        """Return x_new moved along the constraint vectors of x_ref onto the constraint surface (SHAKE).

        x_new, x_ref and the result are (N, 3) [nm].
        """
        if self.nc == 0:
            return x_new
        xp, rp = self._pad(x_new), self._pad(x_ref)
        for b in self.blocks:
            xp = b.positions(xp, rp)
        return xp[:-1]

    def momenta(self, x: jax.Array, p: jax.Array, masses: ArrayLike) -> jax.Array:
        """Return the momenta with the constraint-violating components removed (RATTLE).

        The mass-weighted orthogonal projection onto the tangent space of the constraint surface
        at x.  x (N, 3) [nm], p (N, 3) [amu nm/ps], masses (N,) [amu]; result (N, 3).
        """
        if self.nc == 0:
            return p
        xp, pp = self._pad(x), self._pad(p)
        mp = jnp.concatenate([jnp.asarray(masses, p.dtype).reshape(-1), jnp.ones(1, p.dtype)])
        for b in self.blocks:
            pp = b.momenta(xp, pp, mp)
        return pp[:-1]

    def violation(self, x: jax.Array) -> jax.Array | float:
        """Return the largest relative deviation ||r| / d0 - 1| over the constraints (x (N, 3) [nm])."""
        if self.nc == 0:
            return 0.0
        xp = self._pad(x)
        return jnp.max(jnp.stack([jnp.asarray(b.errors(xp, None, None)[0]) for b in self.blocks]))

    def velocity_violation(self, x: jax.Array, p: jax.Array, masses: ArrayLike) -> jax.Array | float:
        """Return the RATTLE violation: max |r^_c . (v_a - v_b)| relative to the RMS |v_a - v_b|.

        The RMS relative speed of the constrained pairs; 0 when every constraint is rigid in time.
        x (N, 3) [nm], p (N, 3) [amu nm/ps], masses (N,) [amu].
        """
        if self.nc == 0:
            return 0.0
        xp, pp = self._pad(x), self._pad(p)
        mp = jnp.concatenate([jnp.asarray(masses, p.dtype).reshape(-1), jnp.ones(1, p.dtype)])
        e = [b.errors(xp, pp, mp) for b in self.blocks]
        worst = jnp.max(jnp.stack([jnp.asarray(t[1]) for t in e]))
        rms = jnp.sqrt(sum(t[2] for t in e) / self.nc)
        return worst / jnp.maximum(rms, 1e-300)


def repartition_masses(
    masses: ArrayLike, elements: Sequence[str], bonds: ArrayLike, h_mass: ArrayLike = 3.024
) -> np.ndarray:
    """Return masses after hydrogen mass repartitioning (host).

    Every hydrogen gets h_mass, taken from the heavy atom it is bonded to (total mass unchanged).
    Massless atoms (virtual sites) neither give nor take mass.

    Parameters
    ----------
    masses : ArrayLike (N,)
        Masses [amu].
    elements : Sequence[str] (N,)
        Element symbols ("H" for hydrogen).
    bonds : ArrayLike (nb, 2) int
        Bonds (global indices).
    h_mass : ArrayLike
        Hydrogen mass [amu]: one mass for every hydrogen, or one value per atom (read at the
        hydrogens; NaN keeps that hydrogen's mass).

    Returns
    -------
    np.ndarray (N,)
        New masses [amu].

    Raises
    ------
    ValueError
        A per-atom h_mass of the wrong shape, or a non-positive mass left on a real atom.
    """
    m = np.asarray(masses, float).copy()
    real = m > 0
    target = np.asarray(h_mass, float)
    if target.ndim and target.shape != m.shape:
        raise ValueError(f"h_mass: a scalar or one value per atom ({len(m)})")
    target = np.broadcast_to(target, m.shape)
    for i, j in np.asarray(bonds, int).reshape(-1, 2):
        for h, x in ((i, j), (j, i)):
            if elements[h] == "H" and elements[x] != "H" and np.isfinite(target[h]) and real[h] and real[x]:
                dm = target[h] - m[h]
                m[h] += dm
                m[x] -= dm
    if np.any(m[real] <= 0):
        raise ValueError("hydrogen mass repartitioning left a non-positive mass")
    return m


def hmr_masses(sys: System, hmr: float | Sequence[float | None] | None) -> np.ndarray:
    """Return the per-atom masses of a System after hydrogen mass repartitioning along its bonds.

    Heavier hydrogens slow the fastest motions (librations of water, X-H bends) and so allow
    larger time steps; a CH3 carbon (12.01 amu) cannot give three hydrogens 4 amu each and keep a
    sensible mass, so the protein keeps 3.024 while water takes 4.0.

    Parameters
    ----------
    sys : System
        The system.
    hmr : float, Sequence of (float or None), or None
        None (the system's masses); one hydrogen mass [amu] for every molecule; or one value (or
        None: unchanged) per molecule, e.g. water 4.0 and protein 3.024 from
        AmberSystem.hmr({"water": 4.0, "protein": 3.024, "ion": None}).

    Returns
    -------
    np.ndarray (N,)
        Masses [amu].

    Raises
    ------
    TypeError
        A dict (use AmberSystem.hmr).
    ValueError
        A sequence of the wrong length, or from `repartition_masses`.
    """
    masses = np.asarray(sys.masses, float)
    if hmr is None:
        return masses
    if isinstance(hmr, dict):
        raise TypeError(
            "hmr: a mass or one value per molecule; for masses by molecule kind use AmberSystem.hmr({kind: mass})"
        )
    if np.ndim(hmr) == 0:
        per_mol = [float(hmr)] * sys.nmol
    else:
        per_mol = list(hmr)
        if len(per_mol) != sys.nmol:
            raise ValueError(f"hmr: one value per molecule ({sys.nmol}), got {len(per_mol)}")
    target = np.full(sys.n, np.nan)
    for k, h in enumerate(per_mol):
        if h is not None:
            target[sys.atom_slice(k)] = float(h)
    bonds = np.concatenate(
        [np.asarray(m.bonds, int).reshape(-1, 2) + sys.offsets[k] for k, m in enumerate(sys.molecules)]
    )
    return repartition_masses(masses, sys.elements, bonds, target)
