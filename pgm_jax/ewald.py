"""Periodic pGM electrostatics by Ewald summation (triclinic boxes), differentiable in the
coordinates, the parameters and the box.

Same model as `channels.ElecChannel` (Gaussian charges + covalent dipoles + induced Gaussian
dipoles, all pairs, no masking), under periodic boundary conditions, following Wei et al.
JCP 153, 114116 (2020), Sec. II D, with a plain Ewald reciprocal sum instead of PME:

  pair kernel  erf(b_ij r)/r = [erf(b_ij r) - erf(b0 r)]/r   (direct, short range, r < rc)
                             +  erf(b0 r)/r                   (reciprocal, all pairs and images)
  self term    -(b0/sqrt(pi)) sum q^2 - (2 b0^3/(3 sqrt(pi))) sum |d|^2
  background   -pi Q^2 / (2 V b0^2) for a net charge Q (zero for neutral systems; keeps the
               energy independent of b0 while fitted charges drift)
  tin-foil boundary (no surface term), as in Amber.

With d = p + mu (all dipoles), U(q, d) is quadratic in d.  The induced dipoles minimise
G(mu) = U(q, p + mu) + sum |mu|^2/(2 alpha); G(mu*) is the total electrostatic energy (EELEC in
Amber).  mu* comes from conjugate gradients with exact Hessian-vector products; the energy is
wrapped with `solver.variational`, so first derivatives (forces, virial, parameter gradients)
cost one solve and higher derivatives differentiate through the solve.

Box: H with lattice vectors as ROWS (nm).  The neighbour list holds integer image vectors n
(displacement = r_i - r_j + n.H) and the reciprocal sum holds integer indices m
(k = 2 pi m.H^-T), both chosen once at a reference geometry and box; displacements and
k-vectors are computed from H inside JAX, so H can be differentiated.  Molecules must be whole
(covalent dipoles use plain coordinate differences).

Units: nm, e, kJ/mol.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

from .channels import perm_dipoles
from .densities import gauss_bij
from .solver import variational
from .system import System
from .units import KE

SQRT_PI = 1.7724538509055159


def neighbor_list(pos: np.ndarray, H: np.ndarray, rc: float, chunk: int = 200_000):
    """All pairs i<j and lattice images with |r_i - r_j + n.H| < rc.
    Returns (i, j, n) with n (P, 3) integer image vectors."""
    pos, H = np.asarray(pos, float), np.asarray(H, float)
    n = len(pos)
    ii, jj = np.triu_indices(n, k=1)
    cells = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float)
    shifts = cells @ H
    # bring each displacement to the reduced cell first, then test the 27 neighbours
    Hinv = np.linalg.inv(H)
    out_i, out_j, out_n = [], [], []
    for s in range(0, len(ii), chunk):
        a, b = ii[s : s + chunk], jj[s : s + chunk]
        d = pos[a] - pos[b]
        base = -np.round(d @ Hinv)  # integer: d + base.H is near the origin
        dd = (d + base @ H)[:, None, :] + shifts[None, :, :]  # candidate displacements
        r = np.linalg.norm(dd, axis=-1)
        k, m = np.nonzero(r < rc)
        out_i.append(a[k])
        out_j.append(b[k])
        out_n.append(base[k] + cells[m])
    i = np.concatenate(out_i)
    j = np.concatenate(out_j)
    img = np.rint(np.concatenate(out_n)).astype(np.int32)
    return i, j, img


def kvector_indices(H: np.ndarray, kcut: float) -> np.ndarray:
    """Integer indices m (half space) of reciprocal vectors k = 2 pi m.H^-T with 0 < |k| < kcut."""
    H = np.asarray(H, float)
    B = 2 * np.pi * np.linalg.inv(H).T  # H[i] . B[j] = 2 pi delta_ij
    # integer range: |m_i| <= kcut * |H_i| / (2 pi), bounded generously for skewed cells
    Mx = int(np.ceil(kcut * np.max(np.linalg.norm(H, axis=1)) / (2 * np.pi))) + 1
    rng = range(-Mx, Mx + 1)
    m = np.array([[a, b, c] for a in rng for b in rng for c in rng], float)
    half = (m[:, 0] > 0) | ((m[:, 0] == 0) & (m[:, 1] > 0)) | ((m[:, 0] == 0) & (m[:, 1] == 0) & (m[:, 2] > 0))
    m = m[half]
    kk = np.linalg.norm(m @ B, axis=1)
    return m[(kk > 0) & (kk < kcut)].astype(np.int32)


def _pair_tensors(x, bij, b0):
    """Direct-space kernel g(|x|) = (erf(bij r) - erf(b0 r))/r, its gradient and Hessian in x."""

    def g(v):
        return (erf(bij * jnp.linalg.norm(v)) - erf(b0 * jnp.linalg.norm(v))) / jnp.linalg.norm(v)

    return g(x), jax.grad(g)(x), jax.hessian(g)(x)


class PeriodicPGM:
    """pGM energy/forces for one System in a periodic box.

    The reference box and positions fix the neighbour list and the set of k-vectors; energies take
    any positions, parameters and box (default: the reference box).  Pairs of the list farther apart
    than the cutoff are masked, so a list built with a skin (rc + skin) stays valid for small displacements."""

    def __init__(
        self,
        system: System,
        box: np.ndarray,
        positions_ref: np.ndarray,
        ewald_beta: float = 3.8,
        cutoff: float = 1.0,
        skin: float = 0.0,
        k_tol: float = 1e-12,
        dipole_tol: float = 1e-12,
        nlist=None,
        elec: str = "qpi",
    ):
        """Build the Ewald model.

        Parameters
        ----------
        system : System
            The molecules.
        box : array (3, 3)
            Reference box [nm], lattice vectors as rows (fixes the k-vector set).
        positions_ref : array (N, 3)
            Reference positions [nm] of the neighbour list.
        ewald_beta : float
            Ewald coefficient [1/nm].
        cutoff : float
            Real-space cutoff [nm] (pairs of the list beyond it are masked).
        skin : float
            Neighbour-list skin [nm].
        k_tol : float
            Reciprocal-space truncation: exp(-k^2 / (4 beta^2)) below k_tol.
        dipole_tol : float
            Relative residual of the induced-dipole CG (jax.scipy.sparse.linalg.cg).
        nlist : tuple, optional
            A neighbour list (i, j, image) to share (PeriodicModel).
        elec : str
            "q" | "qp" | "qi" | "qpi" (options.py).
        """
        from .options import elec_flags

        self.pd, self.ind = elec_flags(elec)
        self.sys, self.H = system, np.asarray(box, float)
        b0, rc = ewald_beta, cutoff
        self.b0, self.rc, self.cg_tol = b0, rc, dipole_tol
        self.pi, self.pj, self.img = nlist if nlist is not None else neighbor_list(positions_ref, self.H, rc + skin)
        kcut = 2 * b0 * np.sqrt(-np.log(k_tol))
        self.m = kvector_indices(self.H, kcut)
        self._E = variational(self._G, self._solve)

    # -------------------------------------------------------------- energy U(q, d)
    def _box(self, H):
        return jnp.asarray(self.H if H is None else H)

    def _direct_tensors(self, pos, R, H):
        x = pos[self.pi] - pos[self.pj] + jnp.asarray(self.img, float) @ H
        bij = gauss_bij(R[self.pi], R[self.pj])
        f, gx, Hx = jax.vmap(lambda v, b: _pair_tensors(v, b, self.b0))(x, bij)
        w = jax.lax.stop_gradient(jnp.where(jnp.sum(x * x, -1) < self.rc**2, 1.0, 0.0))
        return f * w, gx * w[:, None], Hx * w[:, None, None]

    def _U_dir(self, tens, q, d):
        f, gx, Hx = tens
        i, j = self.pi, self.pj
        e = (
            q[i] * q[j] * f
            - q[i] * jnp.sum(d[j] * gx, -1)
            + q[j] * jnp.sum(d[i] * gx, -1)
            - jnp.einsum("pa,pab,pb->p", d[i], Hx, d[j])
        )
        return jnp.sum(e)

    def _U_rec(self, pos, q, d, H):
        B = 2 * jnp.pi * jnp.linalg.inv(H).T
        k = jnp.asarray(self.m, float) @ B  # (K, 3)
        V = jnp.abs(jnp.linalg.det(H))
        k2 = jnp.sum(k**2, 1)
        kfac = (4 * jnp.pi / V) * jnp.exp(-k2 / (4 * self.b0**2)) / k2  # half-space x 2
        ph = pos @ k.T  # (n, K)
        kd = d @ k.T
        c, s = jnp.cos(ph), jnp.sin(ph)
        A = jnp.sum(q[:, None] * c - kd * s, 0)
        Bs = jnp.sum(q[:, None] * s + kd * c, 0)
        return jnp.sum(kfac * (A * A + Bs * Bs))

    def _U_self(self, q, d):
        b0 = self.b0
        return -(b0 / SQRT_PI) * jnp.sum(q**2) - (2 * b0**3 / (3 * SQRT_PI)) * jnp.sum(d * d)

    def _U_bg(self, q, H):
        return -jnp.pi * jnp.sum(q) ** 2 / (2 * jnp.abs(jnp.linalg.det(H)) * self.b0**2)

    def U(self, pos, q, d, R, H, tens=None):
        tens = self._direct_tensors(pos, R, H) if tens is None else tens
        return KE * (self._U_dir(tens, q, d) + self._U_rec(pos, q, d, H) + self._U_self(q, d) + self._U_bg(q, H))

    # ----------------------------------------------------------------- induction --
    def _p(self, pos, P):
        return perm_dipoles(pos, self.sys, P["cov"]) if self.pd else jnp.zeros((self.sys.n, 3))

    def _G(self, mu, theta):
        pos, P, H = theta
        p = self._p(pos, P)
        return self.U(pos, P["q"], p + mu, P["radius"], H) + KE * jnp.sum(mu * mu / (2 * P["alpha"][:, None]))

    def _solve(self, theta):
        z = jnp.zeros((self.sys.n, 3))

        def gradG(mu):
            return jax.grad(self._G)(mu, theta)

        g0, hvp = jax.linearize(gradG, z)  # G is quadratic: the Hessian is constant
        mu, _ = jax.scipy.sparse.linalg.cg(hvp, -g0, tol=self.cg_tol, maxiter=2000)
        return mu

    def _theta(self, pos, params, H):
        return (jnp.asarray(pos), self.sys.expand(params), self._box(H))

    def induced_dipoles(self, pos, params=None, H=None):
        """mu* (n, 3), e nm; differentiable (implicit differentiation of the CG solve)."""
        return self._solve(self._theta(pos, params, H))

    def energy(self, pos, params=None, H=None):
        """-> ({perm, ind, total} kJ/mol, {p}).  'perm' is U of the permanent multipoles alone."""
        theta = self._theta(pos, params, H)
        pos, P, H = theta
        p = self._p(pos, P)
        e_perm = self.U(pos, P["q"], p, P["radius"], H)
        e_tot = self._E(theta) if self.ind else e_perm
        return {"perm": e_perm, "ind": e_tot - e_perm, "total": e_tot}, {"p": p}

    def forces(self, pos, params=None, H=None):
        return -jax.grad(lambda x: self.energy(x, params, H)[0]["total"])(jnp.asarray(pos))
