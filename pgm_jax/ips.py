"""Isotropic periodic sum (IPS) kernels: an alternative to Ewald sums for electrostatics, LJ and DE.

IPS replaces the periodic sum by a sum over the pairs inside the cutoff rc plus a smooth function of r
that stands for the isotropic periodic images (Wu and Brooks).  A pair of Gaussian charges interacts
through phi(r) = erf(a r) / r [a = 1 / sqrt(2 (R_i^2 + R_j^2)), Gaussian radii R]; the IPS pair function is

    Phi(r) = phi(r) + (1 / rc) sum_{k=0}^{N-1} c_k (r / rc)^{2k}         for r < rc, 0 beyond.

The N coefficients c_k make Phi and its first N - 1 derivatives vanish at r = rc.  For N = 4 these are the
coefficients of sander and pmemd (DEGAUSS: aipseg for a Gaussian of width g = a rc, AIPSE for the point
charge 1 / r); N is the IPS order (`ips_order`), larger N gives a smoother truncation.  The coefficients
depend on the pair only through g = a rc, in closed form (`elec_coefficients`), so they are
differentiable in the Gaussian radii.  The self interaction of an atom with its own images is the value
(and curvature) of the polynomial at r = 0.

Lennard-Jones IPS (`lj_ips_pair`) uses the fixed 3D-IPS coefficients of sander (ips.F90) for r^-12 and
r^-6, and the double exponential (`de_ips_pair`) the analytic form of DEGAUSS (ips_de.h).

Radial kernels: G_n(r) = (-(1/r) d/dr)^n Phi(r), the convention of md/kernels.py (grad_x G_n = -G_{n+1} x).

Units: nm, kJ/mol, e; the electrostatic kernels are in units of the Coulomb constant.

References
----------
.. [1] X. Wu, B. R. Brooks, J. Chem. Phys. 129, 154115 (2008).
"""

from __future__ import annotations

import math
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

_SQRT_PI = math.sqrt(math.pi)

# sander / pmemd 3D-IPS coefficients of r^-12 (aipsva) and r^-6 (aipsvc), ips.F90
AIPSVA = (5.0 / 787.0, 9.0 / 26.0, -3.0 / 13.0, 27.0 / 26.0)
AIPSVC = (7.0 / 16.0, 9.0 / 14.0, -3.0 / 28.0, 6.0 / 7.0)


def _falling(p: int, m: int) -> float:
    """Return p (p - 1) ... (p - m + 1), the m-th derivative factor of u^p (1 for m = 0)."""
    out = 1.0
    for i in range(m):
        out *= p - i
    return out


@lru_cache(maxsize=None)
def _inverse_matrix(order: int) -> np.ndarray:
    """Return the inverse of M[m, k] = d^m u^(2k) / du^m at u = 1 (order x order)."""
    M = np.array([[_falling(2 * k, m) for k in range(order)] for m in range(order)], float)
    return np.linalg.inv(M)


def _hermite(n: int, x: jnp.ndarray) -> jnp.ndarray:
    """Return the physicists' Hermite polynomial H_n(x) (recurrence)."""
    h0, h1 = jnp.ones_like(x), 2.0 * x
    if n == 0:
        return h0
    for j in range(1, n):
        h0, h1 = h1, 2.0 * x * h1 - 2.0 * j * h0
    return h1


def gaussian_derivatives(g: jnp.ndarray, order: int) -> jnp.ndarray:
    """Return d^m [erf(g u) / u] / du^m at u = 1 for m = 0 .. order - 1, shape g.shape + (order,).

    By the Leibniz rule with d^j erf(g u) / du^j = (2 / sqrt(pi)) g^j (-1)^(j-1) H_(j-1)(g u) exp(-g^2 u^2).
    g is finite (a very large g approaches the derivatives of 1 / u; `point_coefficients` is exact).
    """
    g = jnp.asarray(g)
    ex = jnp.exp(-g * g)
    erfg = erf(g)
    # E_j = d^j erf(g u)/du^j at u = 1
    E = [erfg]
    for j in range(1, order):
        E.append((2.0 / _SQRT_PI) * g**j * ((-1.0) ** (j - 1)) * _hermite(j - 1, g) * ex)
    out = []
    for m in range(order):
        s = 0.0
        for j in range(m + 1):
            p = m - j  # d^p (1/u) / du^p = (-1)^p p! u^(-1-p) -> (-1)^p p! at 1
            s = s + math.comb(m, j) * E[j] * ((-1.0) ** p) * math.factorial(p)
        out.append(s)
    return jnp.stack(out, axis=-1)


def elec_coefficients(g: jnp.ndarray, order: int = 4) -> jnp.ndarray:
    """Return the IPS polynomial coefficients c_k (shape g.shape + (order,)) of the Gaussian pair kernel.

    g = a rc [dimensionless] (finite; the point charge: `point_coefficients`).  Solves sum_k c_k D^m u^(2k) = -D^m [erf(g u)/u] at
    u = 1, m = 0 .. order - 1.
    """
    b = gaussian_derivatives(g, order)
    return -(b @ jnp.asarray(_inverse_matrix(order)).T)


def point_coefficients(order: int = 4) -> np.ndarray:
    """Return the coefficients of the point charge 1 / r (order 4: sander's AIPSE = -35/16, 35/16, -21/16, 5/16)."""
    b = np.array([(-1.0) ** m * math.factorial(m) for m in range(order)])
    return -_inverse_matrix(order) @ b


def poly_kernels(c: jnp.ndarray, r: jnp.ndarray, rc: float, nmax: int) -> list:
    """Return G_0 .. G_(nmax-1) of the IPS polynomial (1 / rc) sum_k c_k (r / rc)^(2k).

    c has shape r.shape + (order,) (per pair) or (order,); r [nm].  G_n = (-2)^n d^n/dt^n of the polynomial in
    t = r^2.
    """
    order = c.shape[-1]
    t = r * r
    out = []
    for n in range(nmax):
        s = jnp.zeros_like(r)
        for k in range(n, order):
            s = s + c[..., k] * _falling(k, n) * (t / rc**2) ** (k - n) / rc ** (2 * n)
        out.append(((-2.0) ** n) * s / rc)
    return out


def self_constants(c: jnp.ndarray, rc: float) -> tuple:
    """Return (G0(0), G1(0)) of the polynomial: the charge and dipole self-image coefficients.

    The self energy of atom i is 1/2 q_i^2 G0(0) + 1/2 G1(0) |d_i|^2 (KE units).
    """
    return c[..., 0] / rc, -2.0 * c[..., 1] / rc**3


def lj_ips_pair(r: jnp.ndarray, rmin: jnp.ndarray, eps: jnp.ndarray, rc: float) -> tuple:
    """Return the LJ IPS pair energy and (1/r) dU/dr (r < rc).

    The pair function of sander's 12-6 IPS with the LJ minimum form: A = eps rmin^12, B = 2 eps rmin^6,
    E = A/rc^12 (PVA - PIPSVAC) - B/rc^6 (PVC - PIPSVCC) with PVA = u^-12 + a0 + a1 u^4 + a2 u^8 + a3 u^12,
    PVC = u^-6 + a0 + a1 u^2 + a2 u^4 + a3 u^6, u = r / rc.
    """
    A = eps * rmin**12
    B = 2.0 * eps * rmin**6
    u2 = (r / rc) ** 2
    u4 = u2 * u2
    ua, uc = AIPSVA, AIPSVC
    pipsvac = 1.0 + sum(ua)
    pipsvcc = 1.0 + sum(uc)
    u6r = 1.0 / (u2 * u2 * u2)
    pva = u6r * u6r + ua[0] + u4 * (ua[1] + u4 * (ua[2] + u4 * ua[3]))
    pvc = u6r + uc[0] + u2 * (uc[1] + u2 * (uc[2] + u2 * uc[3]))
    # (1/r) d/dr of PVA and PVC: (1/r) d/dr u^n = n u^(n-2) / rc^2
    dva = (-12.0 * u6r * u6r + u4 * (4.0 * ua[1] + u4 * (8.0 * ua[2] + u4 * 12.0 * ua[3]))) / (u2 * rc**2)
    dvc = (-6.0 * u6r + u2 * (2.0 * uc[1] + u2 * (4.0 * uc[2] + u2 * 6.0 * uc[3]))) / (u2 * rc**2)
    e = A / rc**12 * (pva - pipsvac) - B / rc**6 * (pvc - pipsvcc)
    d = A / rc**12 * dva - B / rc**6 * dvc
    return e, d


def lj_ips_self(rmin: jnp.ndarray, eps: jnp.ndarray, rc: float) -> jnp.ndarray:
    """Return the LJ IPS self-image energy of one atom (half the pair value at r = 0 without the singular part).

    1/2 [A/rc^12 (a0 - PIPSVAC) - B/rc^6 (a0 - PIPSVCC)] (sander: pipsva0, pipsvc0).
    """
    A = eps * rmin**12
    B = 2.0 * eps * rmin**6
    pa = AIPSVA[0] - (1.0 + sum(AIPSVA))
    pc = AIPSVC[0] - (1.0 + sum(AIPSVC))
    return 0.5 * (A / rc**12 * pa - B / rc**6 * pc)


def de_ips_pair(r: jnp.ndarray, rmin: jnp.ndarray, eps: jnp.ndarray, alpha: float, beta: float, rc: float) -> tuple:
    """Return the DE IPS pair energy and (1/r) dU/dr (r < rc), the analytic form of DEGAUSS (ips_de.h).

    With x = r / rm, s = rc / rm, cn1 = eps beta e^alpha / (alpha - beta), cn2 = eps alpha e^beta / (alpha - beta):

        E = cn1 [e^(-alpha x) + e^(-alpha s) (e^(alpha (x - s)) + K(alpha s))]
          - cn2 [e^(-beta x)  + e^(-beta s)  (e^(beta (x - s))  + K(beta s))],   K(z) = (12 + 6 e^(-z) / z) / z^2.
    """
    rm = jnp.where(rmin > 0.0, rmin, 1.0)
    x, s = r / rm, rc / rm
    cn1 = eps * beta * jnp.exp(alpha) / (alpha - beta)
    cn2 = eps * alpha * jnp.exp(beta) / (alpha - beta)

    def part(a: float) -> tuple:
        """Return the value and x-derivative of one exponential branch with rate a."""
        z = a * s
        ez = jnp.exp(-z)
        k = (12.0 + 6.0 * ez / z) / (z * z)
        v = jnp.exp(-a * x) + ez * (jnp.exp(a * (x - s)) + k)
        dv = -a * jnp.exp(-a * x) + ez * a * jnp.exp(a * (x - s))  # d/dx
        return v, dv

    va, dva = part(alpha)
    vc, dvc = part(beta)
    e = cn1 * va - cn2 * vc
    d = (cn1 * dva - cn2 * dvc) / (rm * r)
    return e, d


def de_ips_self(rmin: jnp.ndarray, eps: jnp.ndarray, alpha: float, beta: float, rc: float) -> jnp.ndarray:
    """Return the DE IPS self-image energy of one atom: half the correction of the pair at r = 0 (no direct term)."""
    rm = jnp.where(rmin > 0.0, rmin, 1.0)
    s = rc / rm
    cn1 = eps * beta * jnp.exp(alpha) / (alpha - beta)
    cn2 = eps * alpha * jnp.exp(beta) / (alpha - beta)

    def part(a: float) -> jnp.ndarray:
        z = a * s
        ez = jnp.exp(-z)
        return ez * (ez + (12.0 + 6.0 * ez / z) / (z * z))

    return 0.5 * (cn1 * part(alpha) - cn2 * part(beta))
