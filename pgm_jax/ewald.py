"""Periodic pGM electrostatics by Ewald summation (triclinic boxes).

Same model as `channels.ElecChannel` (Gaussian charges + covalent dipoles + induced Gaussian
dipoles, all pairs, no masking), under periodic boundary conditions, following Wei et al.
JCP 153, 114116 (2020), Sec. II D, with a plain Ewald reciprocal sum instead of PME:

  pair kernel  erf(b_ij r)/r = [erf(b_ij r) - erf(b0 r)]/r   (direct, short range, r < rc)
                             +  erf(b0 r)/r                   (reciprocal, all pairs and images)
  self term    -(b0/sqrt(pi)) sum q^2 - (2 b0^3/(3 sqrt(pi))) sum |d|^2
  tin-foil boundary (no surface term), as in Amber.

With d = p + mu (all dipoles), U(q, d) is quadratic in d.  The induced dipoles minimise
G(mu) = U(q, p + mu) + sum |mu|^2/(2 alpha); G(mu*) is the total electrostatic energy
(EELEC in Amber), and forces are -dG/dR at fixed mu* (Hellmann-Feynman).  The linear system
is solved by conjugate gradients with exact Hessian-vector products (JAX autodiff).

Units: nm, e, kJ/mol.  Box: H with lattice vectors as ROWS (nm).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

from .channels import perm_dipoles
from .system import System
from .units import KE

SQRT_PI = 1.7724538509055159


def box_matrix(a, b, c, alpha, beta, gamma):
    """Amber/PDB convention: a along x, b in the xy plane.  Angles in degrees; returns rows."""
    al, be, ga = np.radians([alpha, beta, gamma])
    ax = np.array([a, 0.0, 0.0])
    bx = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(c ** 2 - cx ** 2 - cy ** 2)
    return np.array([ax, bx, [cx, cy, cz]])


def neighbor_list(pos: np.ndarray, H: np.ndarray, rc: float, chunk: int = 200_000):
    """All pairs i<j and lattice images with |r_i - r_j - n.H| < rc.  Returns (i, j, shift (P,3))."""
    n = len(pos)
    ii, jj = np.triu_indices(n, k=1)
    shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float) @ H
    # bring each displacement to the reduced cell first, then test the 27 neighbours
    Hinv = np.linalg.inv(H)
    out_i, out_j, out_s = [], [], []
    for s in range(0, len(ii), chunk):
        a, b = ii[s:s + chunk], jj[s:s + chunk]
        d = pos[a] - pos[b]
        frac = d @ Hinv
        base = -np.round(frac) @ H                            # d + base is near the origin
        dd = (d + base)[:, None, :] + shifts[None, :, :]       # candidate displacements
        r = np.linalg.norm(dd, axis=-1)
        k, m = np.nonzero(r < rc)
        out_i.append(a[k]); out_j.append(b[k]); out_s.append(base[k] + shifts[m])
    i = np.concatenate(out_i); j = np.concatenate(out_j); sh = np.concatenate(out_s)
    return i, j, sh                                           # displacement = pos[i] - pos[j] + sh


def kvectors(H: np.ndarray, kcut: float):
    """Half-space reciprocal vectors with 0 < |k| < kcut (rows)."""
    B = 2 * np.pi * np.linalg.inv(H).T                        # H[i] . B[j] = 2 pi delta_ij
    # integer range: |m_i| <= kcut * |H_i| / (2 pi), bounded generously for skewed cells
    Mx = int(np.ceil(kcut * np.max(np.linalg.norm(H, axis=1)) / (2 * np.pi))) + 1
    rng = range(-Mx, Mx + 1)
    m = np.array([[a, b, c] for a in rng for b in rng for c in rng], float)
    half = (m[:, 0] > 0) | ((m[:, 0] == 0) & (m[:, 1] > 0)) | ((m[:, 0] == 0) & (m[:, 1] == 0) & (m[:, 2] > 0))
    k = m[half] @ B
    kk = np.linalg.norm(k, axis=1)
    return k[(kk > 0) & (kk < kcut)]


def _pair_tensors(x, bij, b0):
    """Direct-space kernel g(|x|) = (erf(bij r) - erf(b0 r))/r, its gradient and Hessian in x."""
    g = lambda v: (erf(bij * jnp.linalg.norm(v)) - erf(b0 * jnp.linalg.norm(v))) / jnp.linalg.norm(v)
    return g(x), jax.grad(g)(x), jax.hessian(g)(x)


class PeriodicPGM:
    """pGM energy/forces for one System in a periodic box."""

    def __init__(self, sys: System, H: np.ndarray, pos_ref: np.ndarray, b0: float = 3.8, rc: float = 1.0,
                 k_tol: float = 1e-12, cg_tol: float = 1e-12):
        self.sys, self.H = sys, np.asarray(H, float)
        self.V = float(abs(np.linalg.det(self.H)))
        self.b0, self.rc, self.cg_tol = b0, rc, cg_tol
        self.pi, self.pj, self.shift = neighbor_list(np.asarray(pos_ref), self.H, rc)
        R = sys.radius
        self.bij = 1.0 / np.sqrt(2.0 * (R[self.pi] ** 2 + R[self.pj] ** 2))
        kcut = 2 * b0 * np.sqrt(-np.log(k_tol))
        self.k = kvectors(self.H, kcut)
        k2 = np.sum(self.k ** 2, 1)
        self.kfac = (4 * np.pi / self.V) * np.exp(-k2 / (4 * b0 ** 2)) / k2   # half-space x 2
        self.q = np.asarray(sys.q)
        self.alpha = np.asarray(sys.alpha)

    # -------------------------------------------------------------- energy U(q, d)
    def _direct_tensors(self, pos):
        x = pos[self.pi] - pos[self.pj] + self.shift
        return jax.vmap(lambda v, b: _pair_tensors(v, b, self.b0))(x, jnp.asarray(self.bij))

    def _U_dir(self, tens, d):
        f, gx, Hx = tens
        q, i, j = self.q, self.pi, self.pj
        e = (q[i] * q[j] * f - q[i] * jnp.sum(d[j] * gx, -1) + q[j] * jnp.sum(d[i] * gx, -1)
             - jnp.einsum("pa,pab,pb->p", d[i], Hx, d[j]))
        return jnp.sum(e)

    def _U_rec(self, pos, d):
        ph = pos @ self.k.T                                   # (n, K)
        kd = d @ self.k.T
        c, s = jnp.cos(ph), jnp.sin(ph)
        q = self.q[:, None]
        A = jnp.sum(q * c - kd * s, 0)
        B = jnp.sum(q * s + kd * c, 0)
        return jnp.sum(self.kfac * (A * A + B * B))

    def _U_self(self, d):
        b0 = self.b0
        return -(b0 / SQRT_PI) * np.sum(self.q ** 2) - (2 * b0 ** 3 / (3 * SQRT_PI)) * jnp.sum(d * d)

    def U(self, pos, d, tens=None):
        tens = self._direct_tensors(pos) if tens is None else tens
        return KE * (self._U_dir(tens, d) + self._U_rec(pos, d) + self._U_self(d))

    # ----------------------------------------------------------------- induction --
    def solve(self, pos):
        pos = jnp.asarray(pos)
        p = perm_dipoles(pos, self.sys)
        tens = jax.lax.stop_gradient(self._direct_tensors(pos))
        a = jnp.asarray(self.alpha)[:, None]
        G = lambda mu: self.U(pos, p + mu, tens) + KE * jnp.sum(mu * mu / (2 * a))
        g0 = jax.grad(G)(jnp.zeros_like(p))
        _, hvp = jax.linearize(jax.grad(G), jnp.zeros_like(p))
        A = lambda v: hvp(v)                                  # G is quadratic: Hessian is constant
        mu, _ = jax.scipy.sparse.linalg.cg(A, -g0, tol=self.cg_tol, maxiter=2000)
        return mu, p, tens

    def energy(self, pos):
        mu, p, tens = self.solve(pos)
        a = jnp.asarray(self.alpha)[:, None]
        e_perm = self.U(pos, p, tens)
        e_tot = self.U(pos, p + mu, tens) + KE * jnp.sum(mu * mu / (2 * a))
        return {"perm": e_perm, "ind": e_tot - e_perm, "total": e_tot}, {"mu": mu, "p": p}

    def forces(self, pos):
        mu, _, _ = self.solve(pos)
        mu = jax.lax.stop_gradient(mu)
        a = jnp.asarray(self.alpha)[:, None]

        def G(x):
            return self.U(x, perm_dipoles(x, self.sys) + mu) + KE * jnp.sum(mu * mu / (2 * a))

        return -jax.grad(G)(jnp.asarray(pos))
