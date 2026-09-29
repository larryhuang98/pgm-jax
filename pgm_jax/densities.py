"""Pair kernels derived from atomic densities.

Everything a channel needs about "two atomic clouds at distance r" lives here, so that a
new density shape is one new set of functions, not changes all over the code.

Contents: the dispersion damping functions tt6_jax (Tang-Toennies) and gd6_jax (Gaussian
density), the Gaussian kernels (gauss_b, gauss_bij, gauss_coulomb, gauss_overlap,
gauss_overlap_dimless) and the registry DENSITIES that channels look them up in.

Gaussian density (pGM):

    rho_i(r) = q_i (b_i/sqrt(pi))^3 exp(-b_i^2 r^2),    b_i = 1/(sqrt(2) R_i)
    Coulomb:   phi_ij(r) = erf(b_ij r)/r,                b_ij = b_i b_j / sqrt(b_i^2 + b_j^2)
    Overlap:   S_ij(r)   = (b_ij^2/pi)^{3/2} exp(-b_ij^2 r^2)   (integral of two unit clouds, nm^-3)
    C6 damping from the same density: gd6(b_ij r)  (Huang, Luo, Duan GVDW)

with R_i the pGM radius [nm] and b the Gaussian exponent [1/nm].  All functions are elementwise
jnp functions (broadcasting, differentiable, jit-compatible).

Units: nm, 1/nm; kernels without the Coulomb constant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy.special import erf
from jax.typing import ArrayLike

if TYPE_CHECKING:
    import jax

SQRT_PI = 1.7724538509055159  # sqrt(pi)


# ------------------------------------------------------ dispersion damping --


def tt6_jax(x: ArrayLike) -> jax.Array:
    """Return the Tang-Toennies f6 damping function f6(x) = 1 - exp(-x) sum_{k<=6} x^k/k!.

    Parameters
    ----------
    x : ArrayLike
        Reduced distance b r (dimensionless), x >= 0.

    Returns
    -------
    jax.Array
        f6(x) in [0, 1), same shape as `x`.

    Notes
    -----
    For x < 0.5 the equivalent series exp(-x) sum_{k>=7} x^k/k! (first five terms) is used, which
    avoids the cancellation of 1 - exp(-x) poly(x) for small x.

    References
    ----------
    .. [1] K. T. Tang, J. P. Toennies, J. Chem. Phys. 80, 3726 (1984).
    """
    poly = 1 + x * (1 + x * (1 / 2 + x * (1 / 6 + x * (1 / 24 + x * (1 / 120 + x / 720)))))
    f = 1 - jnp.exp(-x) * poly
    # series for small x avoids cancellation: f6 = exp(-x) * sum_{k>=7} x^k/k!
    small = jnp.exp(-x) * x**7 / 5040.0 * (1 + x / 8 + x**2 / 72 + x**3 / 720 + x**4 / 7920)
    return jnp.where(x < 0.5, small, f)


def gd6_jax(x: ArrayLike) -> jax.Array:
    """Return the Gaussian-density C6 damping F(x) of two Gaussian clouds (Huang, Luo, Duan GVDW).

    Parameters
    ----------
    x : ArrayLike
        Reduced distance b_ij r (dimensionless), x >= 0.

    Returns
    -------
    jax.Array
        F(x), same shape as `x`; F -> 1 for large x.

    Notes
    -----
    F = A^2 + g^2 / 2, from the damped dipole-dipole tensor of two Gaussian clouds:

        A = erf(x) - (2 x / sqrt(pi)) (1 + 2 x^2 / 3) exp(-x^2)
        g = (4 x^3 / (3 sqrt(pi))) exp(-x^2)

    For x < 0.15 the leading term 8 x^6 / (9 pi) is used; it cancels the r^-6 pole of C6/r^6
    exactly and avoids the cancellation in A.
    """
    x2 = x * x
    e1 = jnp.exp(-x2)
    A = erf(x) - 2 * x / jnp.sqrt(jnp.pi) * (1 + 2 * x2 / 3) * e1
    g = 4 * x2 * x / (3 * jnp.sqrt(jnp.pi)) * e1
    f = A * A + 0.5 * g * g
    small = 8 * x2**3 / (9 * jnp.pi)  # leading term, cancels the r^-6 pole exactly
    return jnp.where(x < 0.15, small, f)


# ---------------------------------------------------------------- Gaussian --


def gauss_b(radius: ArrayLike) -> jax.Array:
    """Return the Gaussian exponent b = 1/(sqrt(2) R) [1/nm] of a pGM radius R [nm] (density ~ exp(-b^2 r^2))."""
    return 1.0 / (jnp.sqrt(2.0) * radius)


def gauss_bij(Ri: ArrayLike, Rj: ArrayLike) -> jax.Array:
    """Return the pair exponent b_ij = 1/sqrt(2 (R_i^2 + R_j^2)) [1/nm] of two pGM radii [nm].

    Equal to b_i b_j / sqrt(b_i^2 + b_j^2) with b = gauss_b(R); pmemd-pgm writes it as
    sqrt(1/(r_gauss_i + r_gauss_j)) with r_gauss = 2 R^2.
    """
    return 1.0 / jnp.sqrt(2.0 * (Ri**2 + Rj**2))


def gauss_coulomb(r: ArrayLike, bij: ArrayLike) -> jax.Array:
    """Return the Gaussian Coulomb kernel erf(b_ij r)/r [1/nm], finite at r -> 0.

    Parameters
    ----------
    r : ArrayLike
        Distance [nm].
    bij : ArrayLike
        Pair exponent [1/nm] (gauss_bij).

    Returns
    -------
    jax.Array
        erf(b_ij r)/r [1/nm] (multiply by q_i q_j KE for an energy in kJ/mol).

    Notes
    -----
    For b_ij r < 1e-4 the Taylor expansion (2 b_ij/sqrt(pi)) (1 - (b_ij r)^2/3) is used; the
    division uses a safe r in that region so that gradients stay finite at r = 0.
    """
    x = bij * r
    safe = jnp.where(x < 1e-4, 1.0, r)
    small = 2.0 * bij / SQRT_PI * (1.0 - x**2 / 3.0)
    return jnp.where(x < 1e-4, small, erf(bij * safe) / safe)


def gauss_overlap(r: ArrayLike, bij: ArrayLike) -> jax.Array:
    """Return the overlap integral of two unit-normalised Gaussian clouds [1/nm^3].

    S_ij(r) = (b_ij^2/pi)^{3/2} exp(-b_ij^2 r^2), with r [nm] and b_ij [1/nm].
    """
    return (bij**2 / jnp.pi) ** 1.5 * jnp.exp(-((bij * r) ** 2))


def gauss_overlap_dimless(r: ArrayLike, bij: ArrayLike) -> jax.Array:
    """Return the overlap exp(-b_ij^2 r^2), normalised to 1 at r = 0 (dimensionless, in (0, 1])."""
    return jnp.exp(-((bij * r) ** 2))


# ---------------------------------------------------------------- registry --
# Channels look kernels up by density name so that switching density shape is a config change.

DENSITIES = {
    "gaussian": {
        "pair_exponent": gauss_bij,
        "coulomb": gauss_coulomb,
        "overlap": gauss_overlap_dimless,
        "c6_damping": lambda r, bij: gd6_jax(bij * r),
    },
}
