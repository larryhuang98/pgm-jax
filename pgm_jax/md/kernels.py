"""Radial functions of Gaussian-screened Coulomb kernels, safe in single precision.

For phi_a(r) = erf(a r)/r the derivatives needed for charges and dipoles are
    B0 = phi_a,   B1 = -(1/r) dphi/dr,   B2 = -(1/r) dB1/dr,
so that grad_x phi = -B1 x and grad grad phi = B2 x x^T - B1 I (x = r_i - r_j).  Closed form:
    B_n = [(2n - 1) B_{n-1} - (2 a^2)^n exp(-a^2 r^2) / (a sqrt(pi))] / r^2,
(B3, for forces, likewise; grad_x B_n = -B_{n+1} x),
which cancels badly for a r < 1; there the series
    B_n = (2 a^(2n+1) / sqrt(pi)) 2^n sum_m (-s)^m / (m! (2m + 2n + 1)),   s = a^2 r^2
is used (15 terms: < 1e-12 relative for s < 1).
"""
from __future__ import annotations

import math

import jax.numpy as jnp
from jax.scipy.special import erf

_SQRT_PI = math.sqrt(math.pi)
_NSERIES = 15


def erf_kernels_closed(a, r, nmax: int = 3):
    """Closed-form recursion only: for pairs with a r >~ 0.7 (intermolecular pairs; a single
    evaluation of erf and exp).  Loses about one digit per order at a r ~ 0.7 in float32."""
    c = (2.0 * a / _SQRT_PI) * jnp.exp(-(a * r) ** 2)
    ir2 = 1.0 / (r * r)
    B = [erf(a * r) / r]
    a2 = a * a
    for n in range(1, nmax):
        B.append(((2 * n - 1) * B[-1] - (2.0 * a2) ** (n - 1) * c) * ir2)
    return tuple(B)


def erf_kernels(a, r, nmax: int = 3):
    """(B0, ..., B_{nmax-1}) of erf(a r)/r, elementwise; r > 0.  nmax = 3 or 4."""
    x = a * r
    s = x * x
    small = s < 1.0
    # series branch (evaluated with s clipped to the small region)
    ss = jnp.where(small, s, 0.0)
    terms = [jnp.ones_like(ss)]
    for m in range(1, _NSERIES):
        terms.append(terms[-1] * (-ss) / m)                # (-s)^m / m!
    c0 = 2.0 * a / _SQRT_PI
    a2 = a * a
    Bs = [c0 * (2.0 * a2) ** n * sum(t / (2 * m + 2 * n + 1) for m, t in enumerate(terms)) for n in range(nmax)]
    # closed form (with r clipped to the large region)
    rl = jnp.where(small, 1.0 / a, r)
    r2 = rl * rl
    c = c0 * jnp.exp(-(a * rl) ** 2)
    Bl = [erf(a * rl) / rl]
    for n in range(1, nmax):                              # B_n = [(2n-1) B_{n-1} - (2a^2)^(n-1) c] / r^2
        Bl.append(((2 * n - 1) * Bl[-1] - (2.0 * a2) ** (n - 1) * c) / r2)
    return tuple(jnp.where(small, b1, b2) for b1, b2 in zip(Bs, Bl))


def coulomb_kernels(r):
    """(B0, B1, B2) of 1/r."""
    ir = 1.0 / r
    ir2 = ir * ir
    return ir, ir * ir2, 3.0 * ir * ir2 * ir2
