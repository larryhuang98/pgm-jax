"""Holonomic distance constraints for MD: SHAKE positions and RATTLE momenta, cluster by cluster.

The constraints (i, j, d0) split into clusters, the connected components of the constraint graph
(a water triangle, a CH3 group, an O-H bond); every cluster is solved exactly and all clusters of
the system at once (padded to the largest cluster, float64):

  positions (SHAKE)  r' = r~ + M^-1 sum_c lambda_c s_c (+ on atom a_c, - on b_c), s_c the constraint
                     vectors before the move, lambda from Newton iterations on
                     sigma_c(lambda) = |r'_a - r'_b|^2 - d0^2 = 0 (a small dense system per cluster);
  momenta (RATTLE)   p' = p + sum_c mu_c r_c (+ / -), mu from the linear system
                     r_c . (v'_a - v'_b) = 0.

The integrator (flexible.py) applies them in g-BAOAB order: after every kick, drift and friction
step.  With hydrogen mass repartitioning (`repartition_masses`) and X-H bonds constrained,
flexible molecules run at 2 fs.  Units nm, ps, amu."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


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


def _solve_small(A, b):
    """Batched solve of (..., C, C) systems for C <= 3 in closed form (jnp.linalg.solve launches an
    LU factorisation per call, slow for many tiny systems on GPUs); general solve above."""
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
    return jnp.linalg.solve(A, b[..., None])[..., 0]


class Constraints:
    def __init__(self, pairs, d0, masses, n_iter: int = 4):
        pairs = np.asarray(pairs, int).reshape(-1, 2)
        d0 = np.asarray(d0, float).reshape(-1)
        self.n = len(masses)
        self.nc = len(pairs)
        self.n_iter = int(n_iter)
        cl = _clusters(pairs, self.n)
        A = max([len(a) for a, _ in cl] + [1])
        C = max([len(c) for _, c in cl] + [1])
        K = len(cl)
        atoms = np.full((K, A), self.n, np.int32)          # padding: a dummy atom of zero inverse mass
        inc = np.zeros((K, A, C))                           # incidence: +1 at a_c, -1 at b_c
        d = np.ones((K, C))
        cmask = np.zeros((K, C))
        ends = np.zeros((K, C, 2), np.int32)
        for k, (at, cs) in enumerate(cl):
            atoms[k, :len(at)] = at
            loc = {a: s for s, a in enumerate(at)}
            for s, c in enumerate(cs):
                i, j = loc[int(pairs[c, 0])], loc[int(pairs[c, 1])]
                inc[k, i, s], inc[k, j, s] = 1.0, -1.0
                ends[k, s] = (i, j)
                d[k, s], cmask[k, s] = d0[c], 1.0
        invm = np.concatenate([1.0 / np.asarray(masses, float), [0.0]])
        self.atoms = jnp.asarray(atoms)
        self.inc = jnp.asarray(inc)
        self.d2 = jnp.asarray(d * d)
        self.cmask = jnp.asarray(cmask)
        self.ends = jnp.asarray(ends)
        self.set_masses(invm)
        self.shape = (K, A, C)

    def set_masses(self, invm_padded):
        self.invm = jnp.asarray(invm_padded)[self.atoms]                   # (K, A)
        # K_cd = sum_a inc_ac inc_ad / m_a: how constraint forces c move the length of d
        self.kmat = jnp.einsum("kac,ka,kad->kcd", self.inc, self.invm, self.inc)

    def _gather(self, x):
        xp = jnp.concatenate([x, jnp.zeros((1, 3), x.dtype)], 0)
        return xp[self.atoms]                                              # (K, A, 3)

    def _scatter(self, x, X):
        xp = jnp.concatenate([x, jnp.zeros((1, 3), x.dtype)], 0)
        return xp.at[self.atoms].set(X)[:-1]

    def _vectors(self, X):
        """Constraint vectors r_a - r_b (K, C, 3)."""
        i, j = self.ends[..., 0], self.ends[..., 1]
        take = lambda idx: jnp.take_along_axis(X, idx[..., None], axis=1)
        return take(i) - take(j)

    def positions(self, x_new, x_ref):
        """SHAKE: x_new moved along the constraint vectors of x_ref onto the constraint surface."""
        if self.nc == 0:
            return x_new
        X, S = self._gather(x_new), self._vectors(self._gather(x_ref))
        eye = jnp.eye(self.shape[2])
        off = jnp.where(self.cmask[..., :, None] * self.cmask[..., None, :] > 0, 1.0, 0.0)

        def move(lam):                                    # (K, A, 3): sum_c inc_ac lam_c s_c / m_a
            return self.invm[..., None] * jnp.sum(self.inc[..., None] * (lam[:, None, :, None] * S[:, None]), axis=2)

        def resid(lam):
            R = self._vectors(X + move(lam))
            return R, (jnp.sum(R * R, -1) - self.d2) * self.cmask

        def body(c):
            lam, R, sig, it = c
            J = 2.0 * jnp.sum(R[:, :, None, :] * S[:, None, :, :], -1) * self.kmat * off \
                + eye * (1.0 - self.cmask[..., :, None])
            lam = lam - _solve_small(J, sig)
            R, sig = resid(lam)
            return lam, R, sig, it + 1

        # Newton converges quadratically: relative errors ~1e-3, 1e-6, 1e-12 after 1, 2, 3 steps for
        # the displacements of one MD step; a fixed count (unrolled) avoids a reduction per step
        lam0 = jnp.zeros(self.shape[::2])
        c = (lam0, *resid(lam0), 0)
        for _ in range(self.n_iter):
            c = body(c)
        lam = c[0]
        return self._scatter(x_new, X + move(lam))

    def momenta(self, x, p, masses):
        """RATTLE: momenta with the constraint-violating components removed."""
        if self.nc == 0:
            return p
        m = jnp.concatenate([jnp.asarray(masses, p.dtype).reshape(-1), jnp.ones(1, p.dtype)])[self.atoms]
        R = self._vectors(self._gather(x))
        V = self._gather(p) / m[..., None]
        rv = jnp.sum(R * self._vectors(V), -1) * self.cmask
        eye = jnp.eye(self.shape[2])
        M = jnp.sum(R[:, :, None, :] * R[:, None, :, :], -1) * self.kmat
        M = M * (self.cmask[..., :, None] * self.cmask[..., None, :]) + eye * (1.0 - self.cmask[..., :, None])
        mu = -_solve_small(M, rv)
        dV = self.invm[..., None] * jnp.sum(self.inc[..., None] * (mu[:, None, :, None] * R[:, None]), axis=2)
        return self._scatter(p, (V + dV) * m[..., None])

    def violation(self, x):
        """Largest relative deviation |r| / d0 - 1 over the constraints."""
        if self.nc == 0:
            return 0.0
        R = self._vectors(self._gather(x))
        return jnp.max(jnp.abs(jnp.sqrt(jnp.sum(R * R, -1) / self.d2) - 1.0) * self.cmask)


def repartition_masses(masses, elements, bonds, h_mass: float = 3.024) -> np.ndarray:
    """Hydrogen mass repartitioning: every hydrogen gets h_mass (amu), taken from the heavy atom it
    is bonded to (total mass unchanged)."""
    m = np.asarray(masses, float).copy()
    for i, j in bonds:
        for h, x in ((i, j), (j, i)):
            if elements[h] == "H" and elements[x] != "H":
                dm = h_mass - m[h]
                m[h] += dm
                m[x] -= dm
    if np.any(m <= 0):
        raise ValueError("hydrogen mass repartitioning left a non-positive mass")
    return m
