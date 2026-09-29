"""Radial functions of Gaussian-screened Coulomb kernels, safe in single precision.

Functions: `erf_kernels` (series for small a r, closed form elsewhere; exact for all r > 0),
`erf_kernels_closed` (closed form only, for pairs known to have a r >~ 0.7) and
`coulomb_kernels` (the unscreened 1/r).  Used by md/forcefield.py, md/mts.py, bonded/model.py and
channels.py for the charge-charge, charge-dipole and dipole-dipole interactions of Gaussian
multipoles.

For phi_a(r) = erf(a r)/r the radial derivatives needed for charges and dipoles are

    B0 = phi_a,   B1 = -(1/r) dphi/dr,   B2 = -(1/r) dB1/dr,   B3 = -(1/r) dB2/dr,

so that grad_x phi = -B1 x and grad grad phi = B2 x x^T - B1 I (x = r_i - r_j), and
grad_x B_n = -B_{n+1} x.  In closed form

    B_n = [(2n - 1) B_{n-1} - (2 a^2)^n exp(-a^2 r^2) / (a sqrt(pi))] / r^2,

which cancels badly for a r < 1 (about one digit per order at a r ~ 0.7 in float32).  There the
Taylor series

    B_n = (2 a^(2n+1) / sqrt(pi)) 2^n sum_m (-s)^m / (m! (2m + 2n + 1)),   s = a^2 r^2

is used, truncated after 15 terms (m = 0..14; < 1e-12 relative for s < 1).  For a Gaussian pair
with widths (radii) R_i, R_j the screening parameter is a = 1 / sqrt(2 (R_i^2 + R_j^2)); a Ewald
real-space term uses a = beta.

Units: a [1/nm], r [nm], B_n [1/nm^(2n+1)].
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy.special import erf
from jax.typing import ArrayLike

if TYPE_CHECKING:
    import jax

_SQRT_PI = math.sqrt(math.pi)
_NSERIES = 15  # Taylor terms m = 0..14: truncation error < 1/15! ~ 1e-12 relative for s < 1


def erf_kernels_closed(a: ArrayLike, r: ArrayLike, nmax: int = 3) -> tuple[jax.Array, ...]:
    """Return (B0, ..., B_{nmax-1}) of erf(a r)/r from the closed-form recursion only.

    For pairs with a r >~ 0.7 (intermolecular pairs, Ewald real space): a single evaluation of erf
    and exp, but it loses about one digit per order at a r ~ 0.7 in float32 and cancels
    catastrophically as a r -> 0 (see the module docstring).  Elementwise and differentiable.

    Parameters
    ----------
    a : ArrayLike
        Screening parameter [1/nm], broadcastable with `r`.
    r : ArrayLike
        Distances [nm], r > 0.
    nmax : int
        Number of kernels returned (static).

    Returns
    -------
    tuple of jax.Array
        B0 [1/nm], B1 [1/nm^3], ... with the broadcast shape of `a` and `r`.
    """
    c = (2.0 * a / _SQRT_PI) * jnp.exp(-((a * r) ** 2))  # c = 2 a exp(-a^2 r^2) / sqrt(pi)
    ir2 = 1.0 / (r * r)
    B = [erf(a * r) / r]
    a2 = a * a
    for n in range(1, nmax):  # B_n = [(2n-1) B_{n-1} - (2a^2)^(n-1) c] / r^2
        B.append(((2 * n - 1) * B[-1] - (2.0 * a2) ** (n - 1) * c) * ir2)
    return tuple(B)


def erf_kernels(a: ArrayLike, r: ArrayLike, nmax: int = 3) -> tuple[jax.Array, ...]:
    """Return (B0, ..., B_{nmax-1}) of erf(a r)/r, accurate for every r > 0.

    Elementwise: the Taylor series where s = (a r)^2 < 1 and the closed-form recursion elsewhere
    (module docstring).  Both branches are evaluated on clipped arguments (s set to 0 in the large
    region, r set to 1/a in the small one) and combined with `jnp.where`, so that neither branch
    produces inf/nan and the result is differentiable in `a` and `r`.

    Parameters
    ----------
    a : ArrayLike
        Screening parameter [1/nm], broadcastable with `r`.
    r : ArrayLike
        Distances [nm], r > 0.
    nmax : int
        Number of kernels returned (static; 3 for charges and dipoles, 4 or 5 when their
        derivatives are needed).

    Returns
    -------
    tuple of jax.Array
        B0 [1/nm], B1 [1/nm^3], ..., B_{nmax-1} [1/nm^(2 nmax - 1)], broadcast shape of `a`, `r`.
    """
    x = a * r
    s = x * x
    small = s < 1.0
    # series branch (evaluated with s clipped to the small region)
    ss = jnp.where(small, s, 0.0)
    terms = [jnp.ones_like(ss)]
    for m in range(1, _NSERIES):
        terms.append(terms[-1] * (-ss) / m)  # (-s)^m / m!
    c0 = 2.0 * a / _SQRT_PI
    a2 = a * a
    # B_n = c0 (2 a^2)^n sum_m (-s)^m / (m! (2m + 2n + 1))
    Bs = [c0 * (2.0 * a2) ** n * sum(t / (2 * m + 2 * n + 1) for m, t in enumerate(terms)) for n in range(nmax)]
    # closed form (with r clipped to the large region)
    rl = jnp.where(small, 1.0 / a, r)
    r2 = rl * rl
    c = c0 * jnp.exp(-((a * rl) ** 2))
    Bl = [erf(a * rl) / rl]
    for n in range(1, nmax):  # B_n = [(2n-1) B_{n-1} - (2a^2)^(n-1) c] / r^2
        Bl.append(((2 * n - 1) * Bl[-1] - (2.0 * a2) ** (n - 1) * c) / r2)
    return tuple(jnp.where(small, b1, b2) for b1, b2 in zip(Bs, Bl))


def coulomb_kernels(r: ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (B0, B1, B2) = (1/r, 1/r^3, 3/r^5) of the unscreened Coulomb kernel 1/r.

    Parameters
    ----------
    r : ArrayLike
        Distances [nm], r > 0.

    Returns
    -------
    tuple of jax.Array
        B0 [1/nm], B1 [1/nm^3], B2 [1/nm^5], the shape of `r`.
    """
    ir = 1.0 / r
    ir2 = ir * ir
    return ir, ir * ir2, 3.0 * ir * ir2 * ir2
