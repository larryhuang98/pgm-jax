"""Gaussian multipoles up to quadrupoles: the covalent quadrupole basis and analytic pair kernels.

Atom i carries a Gaussian charge q_i, dipole p_i and traceless quadrupole Th_i (Buckingham
convention: Th = (1/2) int rho (3 r r^T - r^2 1) for a point distribution), all with the
atom's Gaussian width.  Two atoms interact through multipole operators applied to the Gaussian
Coulomb kernel phi(r) = erf(b_ij r)/r (b_ij the pGM pair exponent):

    E_ij = O_i O_j phi(|x|),   x = r_i - r_j,
    O_i = q_i + p_i . grad + (1/3) Th_i : grad grad,     O_j = q_j - p_j . grad + (1/3) Th_j : grad grad

(gradients with respect to x; grad_j = -grad).  With the radial functions B_0 = phi,
B_{n+1} = -(1/r) dB_n/dr (md/kernels.erf_kernels: closed form, series for small b r):

    d_a phi    = -x_a B1
    d_ab phi   =  x_a x_b B2 - d_ab B1
    d_abc phi  = -x_a x_b x_c B3 + (d_ab x_c + d_ac x_b + d_bc x_a) B2
    d_abcd phi =  x_a x_b x_c x_d B4 - (d_ab x_c x_d + 5 more) B3 + (d_ab d_cd + d_ac d_bd + d_ad d_bc) B2

and, contracting with traceless Th (d_ab Th_ab = 0),

    E_ij = q_i q_j B0 + [q_i (p_j.x) - q_j (p_i.x)] B1 + (p_i.p_j) B1 - (p_i.x)(p_j.x) B2
         + (1/3) [q_i (x Th_j x) + q_j (x Th_i x)] B2
         + (1/3) [(p_j.x)(x Th_i x) - (p_i.x)(x Th_j x)] B3 + (2/3) [p_i Th_j x - p_j Th_i x] B2
         + (1/9) [(x Th_i x)(x Th_j x) B4 - 4 (x Th_i Th_j x) B3 + 2 (Th_i : Th_j) B2].

The field at i of the permanent multipoles of j (right-hand side of the induced dipoles) is
E_i = -grad V_j with V_j = q_j B0 + (p_j.x) B1 + (1/3)(x Th_j x) B2:

    E_i = q_j x B1 - p_j B1 + (p_j.x) x B2 - (2/3) Th_j x B2 + (1/3)(x Th_j x) x B3.

The charge and dipole terms are those of pGM (channels.py, md/forcefield.py); tests compare the
whole expression with nested automatic derivatives of the operator form.  Gaussian kernels are
not harmonic inside the clouds (lap phi != 0), so the trace of a second moment would change the
energy; the quadrupole here is traceless by definition (5 components per atom).

Covalent quadrupole basis (the frame-free analogue of pGM's covalent dipoles).  With unit
vectors u_ij = (r_j - r_i)/|r_j - r_i| to covalent (or virtual) partners,

    Th_i = sum_{terms (i, j, k)} t_ijk S(u_ij, u_ik),
    S(u, v) = (3/4)(u v^T + v u^T) - (1/2)(u.v) 1        (traceless; S(u, u) = (3 u u^T - 1)/2),

j == k: a uniaxial quadrupole along the bond (Th_zz = t along u); j != k: the anisotropy in the
plane of two partners.  The span of these tensors is every quadrupole compatible with the local
symmetry for atoms with two or more partners; a terminal atom (one partner) only gets the axial
part, and `cbv_quadrupole_terms(..., terminal13=True)` adds its 1-3 neighbours as virtual
partners (carbonyl O: lone-pair plane vs pi axis).  Like the covalent dipoles, the quadrupoles
follow the geometry without local frames and are smooth in the coordinates.  Units: e, nm, e nm^2.
"""
from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from .md.kernels import erf_kernels


# ------------------------------------------------------------------ covalent quadrupole basis
def S_tensor(u, v):
    """(..., 3, 3) traceless symmetric S(u, v) = (3/4)(u v^T + v u^T) - (1/2)(u.v) 1."""
    uv = u[..., :, None] * v[..., None, :]
    dot = jnp.sum(u * v, -1)[..., None, None]
    return 0.75 * (uv + jnp.swapaxes(uv, -1, -2)) - 0.5 * dot * jnp.eye(3)


def quadrupoles(pos, sys, t):
    """Traceless quadrupoles (n, 3, 3), e nm^2, from the covalent quadrupole terms of `sys`
    (sys.quad_ijk) with strengths t (per term, e.g. sys.expand(params)["quad"]).  `pos` may be a
    function-free array of positions; minimum images are the caller's business (molecules whole)."""
    n = pos.shape[0]
    if len(sys.quad_ijk) == 0:
        return jnp.zeros((n, 3, 3))
    i, j, k = sys.quad_ijk.T
    uj = pos[j] - pos[i]
    uk = pos[k] - pos[i]
    uj = uj / jnp.linalg.norm(uj, axis=-1, keepdims=True)
    uk = uk / jnp.linalg.norm(uk, axis=-1, keepdims=True)
    return jnp.zeros((n, 3, 3)).at[i].add(t[:, None, None] * S_tensor(uj, uk))


def cbv_quadrupole_terms(mol, pairs: bool = True, terminal13: bool = True, t: float = 0.0):
    """Quadrupole terms (i, j, k, t) of a Molecule from its covalent-dipole partners: an axial term
    per partner, a pair term per two partners (pairs=True), and for atoms with one partner its
    1-3 neighbours as virtual partners (terminal13=True).  Strengths start at t (0: no change
    to the energy until fitted)."""
    partners = [[] for _ in range(mol.n)]
    for i, j, _ in mol.cov:
        if j not in partners[i]:
            partners[i].append(j)
    nbr = [set() for _ in range(mol.n)]
    for a, b in mol.bonds:
        nbr[a].add(b); nbr[b].add(a)
    terms = []
    for i in range(mol.n):
        P = list(partners[i])
        if terminal13 and len(P) == 1:
            P += sorted(k for k in nbr[P[0]] if k != i and k not in P)
        for a, j in enumerate(P):
            terms.append((i, j, j, t))
            if pairs:
                terms += [(i, j, k, t) for k in P[a + 1:]]
    return terms


def with_quadrupoles(mol, **kw):
    """Copy of a Molecule with the covalent quadrupole basis (cbv_quadrupole_terms)."""
    return replace(mol, quad=cbv_quadrupole_terms(mol, **kw))


# ------------------------------------------------------------------ pair kernels
def _dot(a, b):
    return jnp.sum(a * b, -1)


def _xTx(x, T):
    return jnp.einsum("pa,pab,pb->p", x, T, x)


def multipole_pair_energy(x, a, qi, pi, Ti, qj, pj, Tj, quadrupoles: bool = True):
    """E_ij (e^2/nm) of Gaussian charges, dipoles and traceless quadrupoles for pairs (P,):
    x (P, 3) = r_i - r_j, a (P,) pair exponent; multiply by the Coulomb constant."""
    r = jnp.linalg.norm(x, axis=-1)
    B = erf_kernels(a, r, 5 if quadrupoles else 3)
    pix, pjx = _dot(pi, x), _dot(pj, x)
    e = qi * qj * B[0] + (qi * pjx - qj * pix) * B[1] + _dot(pi, pj) * B[1] - pix * pjx * B[2]
    if quadrupoles:
        e = e + quadrupole_pair_terms(x, B, qi, pi, Ti, qj, pj, Tj)
    return e


def quadrupole_pair_terms(x, B, qi, pi, Ti, qj, pj, Tj):
    """The terms of E_ij that contain a quadrupole (B = (B0, ..., B4) at |x|)."""
    xTi, xTj = _xTx(x, Ti), _xTx(x, Tj)
    pix, pjx = _dot(pi, x), _dot(pj, x)
    Tix, Tjx = jnp.einsum("pab,pb->pa", Ti, x), jnp.einsum("pab,pb->pa", Tj, x)
    e = (qi * xTj + qj * xTi) * B[2] / 3.0
    e = e + (pjx * xTi - pix * xTj) * B[3] / 3.0 + 2.0 * (_dot(pi, Tjx) - _dot(pj, Tix)) * B[2] / 3.0
    e = e + (xTi * xTj * B[4] - 4.0 * _dot(Tix, Tjx) * B[3] + 2.0 * jnp.sum(Ti * Tj, (-1, -2)) * B[2]) / 9.0
    return e


def multipole_field(x, a, qj, pj, Tj, quadrupoles: bool = True):
    """Field (P, 3), e/nm^2, at atom i of the Gaussian multipoles of atom j (x = r_i - r_j)."""
    r = jnp.linalg.norm(x, axis=-1)
    B = erf_kernels(a, r, 4 if quadrupoles else 3)
    pjx = _dot(pj, x)
    E = (qj * B[1] + pjx * B[2])[:, None] * x - pj * B[1][:, None]
    if quadrupoles:
        Tjx = jnp.einsum("pab,pb->pa", Tj, x)
        E = E - (2.0 / 3.0) * Tjx * B[2][:, None] + (_xTx(x, Tj) * B[3] / 3.0)[:, None] * x
    return E
