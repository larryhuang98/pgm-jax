"""Pair kernels derived from atomic densities.

Everything a channel needs about "two atomic clouds at distance r" lives here, so that a
new density shape is one new set of functions, not changes all over the code.

Gaussian density (pGM):  rho_i(r) = q_i (b_i/sqrt(pi))^3 exp(-b_i^2 r^2),  b_i = 1/(sqrt(2) R_i)
  Coulomb:   phi_ij(r) = erf(b_ij r)/r,           b_ij = b_i b_j / sqrt(b_i^2 + b_j^2)
  Overlap:   S_ij(r)   = (b_ij^2/pi)^{3/2} exp(-b_ij^2 r^2)     (integral of two unit clouds, nm^-3)
  C6 damping from the same density: gd6(b_ij r)  (Huang, Luo, Duan GVDW)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import erf

SQRT_PI = 1.7724538509055159


# ------------------------------------------------------ dispersion damping --


def tt6_jax(x):
    """Tang-Toennies f6 damping, 1 - exp(-x) sum_{k<=6} x^k/k!."""
    poly = 1 + x * (1 + x * (1 / 2 + x * (1 / 6 + x * (1 / 24 + x * (1 / 120 + x / 720)))))
    f = 1 - jnp.exp(-x) * poly
    # series for small x avoids cancellation: f6 = exp(-x) * sum_{k>=7} x^k/k!
    small = jnp.exp(-x) * x**7 / 5040.0 * (1 + x / 8 + x**2 / 72 + x**3 / 720 + x**4 / 7920)
    return jnp.where(x < 0.5, small, f)


def gd6_jax(x):
    """Gaussian-density C6 damping F(x) (Huang, Luo, Duan GVDW): F = A^2 + g^2/2 from the damped
    dipole-dipole tensor of two Gaussian clouds, x = b_ij r."""
    x2 = x * x
    e1 = jnp.exp(-x2)
    A = erf(x) - 2 * x / jnp.sqrt(jnp.pi) * (1 + 2 * x2 / 3) * e1
    g = 4 * x2 * x / (3 * jnp.sqrt(jnp.pi)) * e1
    f = A * A + 0.5 * g * g
    small = 8 * x2**3 / (9 * jnp.pi)  # leading term, cancels the r^-6 pole exactly
    return jnp.where(x < 0.15, small, f)


# ---------------------------------------------------------------- Gaussian --


def gauss_b(radius):
    """pGM radius R -> Gaussian exponent b (density ~ exp(-b^2 r^2))."""
    return 1.0 / (jnp.sqrt(2.0) * radius)


def gauss_bij(Ri, Rj):
    """Pair exponent: 1/sqrt(2 (R_i^2 + R_j^2)) (pmemd-pgm: sqrt(1/(r_gauss_i + r_gauss_j)), r_gauss = 2 R^2)."""
    return 1.0 / jnp.sqrt(2.0 * (Ri**2 + Rj**2))


def gauss_coulomb(r, bij):
    """erf(b r)/r, finite at r -> 0 (2 b/sqrt(pi))."""
    x = bij * r
    safe = jnp.where(x < 1e-4, 1.0, r)
    small = 2.0 * bij / SQRT_PI * (1.0 - x**2 / 3.0)
    return jnp.where(x < 1e-4, small, erf(bij * safe) / safe)


def gauss_overlap(r, bij):
    """Overlap integral of two unit-normalised Gaussian clouds (nm^-3)."""
    return (bij**2 / jnp.pi) ** 1.5 * jnp.exp(-((bij * r) ** 2))


def gauss_overlap_dimless(r, bij):
    """Overlap normalised to 1 at r = 0 (dimensionless, in (0, 1])."""
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
