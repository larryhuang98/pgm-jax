"""Holonomic distance constraints for MD: SHAKE positions and RATTLE momenta.

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
Units nm, ps, amu."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

DENSE_MAX = 12                 # clusters with more constraints go to the matrix-free (sparse) solver


def _clusters(pairs, n_atoms):
    """Connected components of the constraint graph: list of (atoms, constraint indices)."""
    parent = list(range(n_atoms))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    comp = {}
    for c, (i, j) in enumerate(pairs):
        comp.setdefault(find(int(i)), []).append(c)
    out = []
    for cs in comp.values():
        atoms = sorted({int(a) for c in cs for a in pairs[c]})
        out.append((atoms, cs))
    return out


def _solve_gauss(A, b):
    """Batched solve of (..., C, C) symmetric positive definite (or nearly so) systems by unrolled
    Gaussian elimination without pivoting: elementwise operations on the batch, which XLA fuses."""
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


def _solve_small(A, b):
    """Batched solve of (..., C, C) systems: closed form for C <= 3 (jnp.linalg.solve launches an LU
    factorisation per call, slow for many tiny systems on GPUs), unrolled elimination above."""
    C = A.shape[-1]
    if C == 1:
        return b / A[..., 0, 0:1]
    if C == 2:
        a, bb, c, d = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
        det = a * d - bb * c
        return jnp.stack([d * b[..., 0] - bb * b[..., 1], -c * b[..., 0] + a * b[..., 1]], -1) / det[..., None]
    if C == 3:
        r0, r1, r2 = A[..., 0, :], A[..., 1, :], A[..., 2, :]
        c0, c1, c2 = jnp.cross(r1, r2), jnp.cross(r2, r0), jnp.cross(r0, r1)       # adjugate columns
        det = jnp.sum(r0 * c0, -1)
        return (c0 * b[..., 0:1] + c1 * b[..., 1:2] + c2 * b[..., 2:3]) / det[..., None]
    return _solve_gauss(A, b)


class _DenseBlock:
    """Clusters of at most C constraints padded to (K clusters, A atoms, C constraints): exact
    Newton.  Works on padded arrays (row n: a dummy atom of zero inverse mass)."""
    kind = "dense"

    def __init__(self, clusters, pairs, d0, n, n_iter):
        A = max(len(a) for a, _ in clusters)
        C = max(len(c) for _, c in clusters)
        K = len(clusters)
        atoms = np.full((K, A), n, np.int32)
        inc = np.zeros((K, A, C))                           # incidence: +1 at a_c, -1 at b_c
        d = np.ones((K, C))
        cmask = np.zeros((K, C))
        ends = np.zeros((K, C, 2), np.int32)
        for k, (at, cs) in enumerate(clusters):
            atoms[k, :len(at)] = at
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

    def describe(self):
        K, A, C = self.shape
        return f"{K} x ({A} atoms, {C} constraints)"

    def set_masses(self, invm_padded):
        self.invm = jnp.asarray(invm_padded)[self.atoms]                   # (K, A)
        # K_cd = sum_a inc_ac inc_ad / m_a: how constraint forces c move the length of d
        self.kmat = jnp.einsum("kac,ka,kad->kcd", self.inc, self.invm, self.inc)

    def _vectors(self, X):
        """Constraint vectors r_a - r_b (K, C, 3)."""
        i, j = self.ends[..., 0], self.ends[..., 1]
        take = lambda idx: jnp.take_along_axis(X, idx[..., None], axis=1)   # noqa: E731
        return take(i) - take(j)

    def _move(self, w, V):
        """(K, A, 3): sum_c inc_ac w_c V_c / m_a (broadcast products: XLA fuses them; an einsum
        becomes batched tiny matrix products, 4x slower on the GPU)."""
        return self.invm[..., None] * jnp.sum(self.inc[..., None] * (w[:, None, :, None] * V[:, None]), axis=2)

    @staticmethod
    def _gram(R, S):
        """(K, C, C): R_c . S_d."""
        return jnp.sum(R[:, :, None, :] * S[:, None, :, :], -1)

    def positions(self, xp, rp):
        X, S = xp[self.atoms], self._vectors(rp[self.atoms])

        def resid(lam):
            R = self._vectors(X + self._move(lam, S))
            return R, (jnp.sum(R * R, -1) - self.d2) * self.cmask

        lam = jnp.zeros(self.shape[::2], X.dtype)
        R, sig = resid(lam)
        for _ in range(self.n_iter):                 # fixed count, unrolled: no reduction per step
            J = 2.0 * self._gram(R, S) * self.kmat * self._off + self._pad
            lam = lam - _solve_small(J, sig)
            R, sig = resid(lam)
        return xp.at[self.atoms].set(X + self._move(lam, S))

    def momenta(self, xp, pp, mp):
        m = mp[self.atoms][..., None]
        R = self._vectors(xp[self.atoms])
        V = pp[self.atoms] / m
        rv = jnp.sum(R * self._vectors(V), -1) * self.cmask
        M = self._gram(R, R) * self.kmat * self._off + self._pad
        mu = -_solve_small(M, rv)
        return pp.at[self.atoms].set((V + self._move(mu, R)) * m)

    def errors(self, xp, pp, mp):
        """(largest relative length error, largest |r^ . (v_a - v_b)|, sum |v_a - v_b|^2) over the block."""
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
    """Large clusters as one flat list of constraints, matrix-free: SHAKE by quasi-Newton iterations
    with J0 = 2 S K S^T solved by preconditioned CG, RATTLE by CG on R K R^T (while loops to `tol`)."""
    kind = "sparse"

    unroll = 8                    # CG iterations per convergence test (each test is a device-to-host sync)

    def __init__(self, pairs, d0, n, tol=1e-10, max_iter=100, inner_tol=1e-3, cg_max=400, rattle_tol=1e-11):
        pairs = np.asarray(pairs, int).reshape(-1, 2)
        atoms = np.unique(pairs)
        loc = {int(a): k for k, a in enumerate(atoms)}
        al = np.array([loc[int(i)] for i in pairs[:, 0]], np.int32)
        bl = np.array([loc[int(j)] for j in pairs[:, 1]], np.int32)
        nc, na = len(pairs), len(atoms)
        rows = [[] for _ in range(na)]
        for c in range(nc):
            rows[al[c]].append((c, 1.0)); rows[bl[c]].append((c, -1.0))
        D = max(len(r) for r in rows)
        inc_c = np.full((na, D), nc, np.int32)              # padding: a zero constraint vector
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

    def describe(self):
        na, D, nc = self.shape
        return f"iterative: {nc} constraints on {na} atoms (tol {self.tol:g})"

    def set_masses(self, invm_padded):
        self.invm = jnp.asarray(invm_padded)[self.atoms]                   # (na,)
        self._w = self.invm[self.al] + self.invm[self.bl]

    def _acc(self, W):
        """(na, 3): sum over the atom's constraints of +-W_c."""
        Wp = jnp.concatenate([W, jnp.zeros((1, 3), W.dtype)], 0)
        return jnp.sum(self.inc_s[..., None] * Wp[self.inc_c], axis=1)

    def _move(self, w, V):
        return self.invm[:, None] * self._acc(w[:, None] * V)

    def _vec(self, Y):
        return Y[self.al] - Y[self.bl]

    def _cg(self, V, b, tol, max_iter):
        """Solve (V K V^T) x = b (the Gram matrix of the mass-weighted constraint gradients), Jacobi
        preconditioned, until |r|_max <= tol |b|_max; `unroll` iterations between tests.
        Returns (x, iterations)."""
        op = lambda x: jnp.sum(V * self._vec(self._move(x, V)), -1)      # noqa: E731
        dinv = 1.0 / (jnp.sum(V * V, -1) * self._w)
        stop = tol * jnp.max(jnp.abs(b))
        z = dinv * b
        c0 = (jnp.zeros_like(b), b, z, jnp.sum(b * z), 0)

        def cond(c):
            return (jnp.max(jnp.abs(c[1])) > stop) & (c[4] < max_iter)

        def body(c):
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

    def positions(self, xp, rp):
        X = xp[self.atoms]
        S = self._vec(rp[self.atoms])

        def resid(lam):
            Y = X + self._move(lam, S)
            R = self._vec(Y)
            return Y, jnp.sum(R * R, -1) - self.d2

        def cond(c):
            return (jnp.max(jnp.abs(c[2]) / self.d2) > self.tol) & (c[3] < self.max_iter)

        def body(c):
            lam, _, sig, it = c
            lam = lam - self._cg(S, 0.5 * sig, self.inner_tol, self.cg_max)[0]   # J0 = 2 S K S^T
            Y, sig = resid(lam)
            return lam, Y, sig, it + 1

        lam = jnp.zeros(self.nc, X.dtype)
        Y, sig = resid(lam)
        Y = jax.lax.while_loop(cond, body, (lam, Y, sig, 0))[1]
        return xp.at[self.atoms].set(Y)

    def momenta(self, xp, pp, mp):
        m = mp[self.atoms][:, None]
        R = self._vec(xp[self.atoms])
        V = pp[self.atoms] / m
        rv = jnp.sum(R * self._vec(V), -1)
        mu = self._cg(R, -rv, self.rattle_tol, self.cg_max)[0]
        return pp.at[self.atoms].set((V + self._move(mu, R)) * m)

    def errors(self, xp, pp, mp):
        R = self._vec(xp[self.atoms])
        r = jnp.sqrt(jnp.sum(R * R, -1))
        dl = jnp.max(jnp.abs(r / jnp.sqrt(self.d2) - 1.0))
        if pp is None:
            return dl, 0.0, 0.0
        dV = self._vec(pp[self.atoms] / mp[self.atoms][:, None])
        rv = jnp.abs(jnp.sum(R * dV, -1)) / r
        return dl, jnp.max(rv), jnp.sum(dV * dV)


class Constraints:
    """Distance constraints (pairs (nc, 2), lengths d0 (nc,) nm) of a system with `masses` (amu;
    massless virtual sites never constrained).  n_iter: Newton iterations of the dense solver;
    dense_max: largest cluster (number of constraints) for the dense solver; tol: relative
    tolerance of the iterative SHAKE (large clusters: max |sigma_c| / d_c^2), rattle_tol: of its
    RATTLE (max residual of B v relative to its initial value); bucket: clusters of up to 3 constraints form
    one block and larger ones one block per size (False: every small cluster padded to the largest,
    one block).  Each block costs a few kernel launches per call, so on a GPU one padded block of
    X-H groups and waters is faster than one block per size (docs/shake.md)."""

    def __init__(self, pairs, d0, masses, n_iter: int = 4, dense_max: int = DENSE_MAX, tol: float = 1e-10,
                 rattle_tol: float = 1e-11, bucket: bool = True):
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
        for c in small:                                  # <= 3 constraints: one block (closed-form solves)
            groups.setdefault(max(len(c[1]), 3) if bucket else 0, []).append(c)
        self.blocks = [_DenseBlock(g, pairs, d0, self.n, n_iter) for _, g in sorted(groups.items())]
        if big:
            idx = np.array([c for _, cs in big for c in cs], int)
            self.blocks.append(_SparseBlock(pairs[idx], d0[idx], self.n, tol, rattle_tol=rattle_tol))
        m = np.asarray(masses, float)
        invm = np.concatenate([np.divide(1.0, m, out=np.zeros_like(m), where=m != 0), [0.0]])   # massless sites: never in a cluster
        self.set_masses(invm)

    def set_masses(self, invm_padded):
        for b in self.blocks:
            b.set_masses(invm_padded)

    # compatibility: the constrained atoms and their inverse masses as the solvers see them (flat)
    @property
    def atoms(self):
        return jnp.concatenate([b.atoms.reshape(-1) for b in self.blocks]) if self.blocks else jnp.zeros(0, jnp.int32)

    @property
    def invm(self):
        return jnp.concatenate([b.invm.reshape(-1) for b in self.blocks]) if self.blocks else jnp.zeros(0)

    def describe(self) -> str:
        if not self.nc:
            return "no constraints"
        return (f"{self.nc} constraints in {self.n_clusters} clusters: "
                + "; ".join(b.describe() for b in self.blocks))

    @staticmethod
    def _pad(x):
        return jnp.concatenate([x, jnp.zeros((1,) + x.shape[1:], x.dtype)], 0)

    def positions(self, x_new, x_ref):
        """SHAKE: x_new moved along the constraint vectors of x_ref onto the constraint surface."""
        if self.nc == 0:
            return x_new
        xp, rp = self._pad(x_new), self._pad(x_ref)
        for b in self.blocks:
            xp = b.positions(xp, rp)
        return xp[:-1]

    def momenta(self, x, p, masses):
        """RATTLE: momenta with the constraint-violating components removed (the mass-weighted
        orthogonal projection onto the tangent space of the constraint surface at x)."""
        if self.nc == 0:
            return p
        xp, pp = self._pad(x), self._pad(p)
        mp = jnp.concatenate([jnp.asarray(masses, p.dtype).reshape(-1), jnp.ones(1, p.dtype)])
        for b in self.blocks:
            pp = b.momenta(xp, pp, mp)
        return pp[:-1]

    def violation(self, x):
        """Largest relative deviation |r| / d0 - 1 over the constraints."""
        if self.nc == 0:
            return 0.0
        xp = self._pad(x)
        return jnp.max(jnp.stack([jnp.asarray(b.errors(xp, None, None)[0]) for b in self.blocks]))

    def velocity_violation(self, x, p, masses):
        """RATTLE condition: largest |r^_c . (v_a - v_b)| over the constraints, relative to the RMS
        relative speed |v_a - v_b| of the constrained pairs (0 when every constraint is rigid in
        time)."""
        if self.nc == 0:
            return 0.0
        xp, pp = self._pad(x), self._pad(p)
        mp = jnp.concatenate([jnp.asarray(masses, p.dtype).reshape(-1), jnp.ones(1, p.dtype)])
        e = [b.errors(xp, pp, mp) for b in self.blocks]
        worst = jnp.max(jnp.stack([jnp.asarray(t[1]) for t in e]))
        rms = jnp.sqrt(sum(t[2] for t in e) / self.nc)
        return worst / jnp.maximum(rms, 1e-300)


def repartition_masses(masses, elements, bonds, h_mass=3.024) -> np.ndarray:
    """Hydrogen mass repartitioning: every hydrogen gets h_mass (amu), taken from the heavy atom it
    is bonded to (total mass unchanged).  h_mass: one mass for every hydrogen, or one value per atom
    (read at the hydrogens; NaN keeps that hydrogen's mass).  Massless atoms (virtual sites) neither
    give nor take mass."""
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


def hmr_masses(sys, hmr) -> np.ndarray:
    """Per-atom masses (amu) of a System after hydrogen mass repartitioning along its molecules'
    bonds.  hmr: None (the system's masses); one hydrogen mass for every molecule; or a sequence
    with one value (or None: unchanged) per molecule, e.g. water 4.0 and protein 3.024 from
    AmberSystem.hmr({"water": 4.0, "protein": 3.024, "ion": None}).  Heavier hydrogens slow the
    fastest motions (librations of water, X-H bends) and so allow larger time steps; a CH3 carbon
    (12.01 amu) cannot give three hydrogens 4 amu each and keep a sensible mass, so the protein
    keeps 3.024 while water takes 4.0."""
    masses = np.asarray(sys.masses, float)
    if hmr is None:
        return masses
    if isinstance(hmr, dict):
        raise TypeError("hmr: a mass or one value per molecule; for masses by molecule kind use "
                        "AmberSystem.hmr({kind: mass})")
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
    bonds = np.concatenate([np.asarray(m.bonds, int).reshape(-1, 2) + sys.offsets[k]
                            for k, m in enumerate(sys.molecules)])
    return repartition_masses(masses, sys.elements, bonds, target)
