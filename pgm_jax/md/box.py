"""Periodic boxes in Amber/OpenMM reduced triclinic form.

H holds the lattice vectors as rows (nm): a = (ax, 0, 0), b = (bx, by, 0), c = (cx, cy, cz), with
|bx| <= ax/2, |cx| <= ax/2, |cy| <= by/2.  In that form the sequential minimum-image reduction
(c, then b, then a) is exact for any pair closer than half the smallest of ax, by, cz, which
therefore bounds the cutoff (OpenMM uses the same rule).  Amber's truncated octahedron is in this
form.  JAX-MD boxes are the transpose (columns are lattice vectors, upper triangular)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def reduce_box(H) -> np.ndarray:
    """Bring a lower-triangular box to reduced form (same lattice)."""
    H = np.array(H, float)
    if np.any(np.abs(np.triu(H, 1)) > 1e-12 * np.abs(H).max()):
        raise ValueError("box must be lower triangular (a along x, b in the xy plane)")
    a, b, c = H
    c = c - b * np.round(c[1] / b[1])
    c = c - a * np.round(c[0] / a[0])
    b = b - a * np.round(b[0] / a[0])
    return np.array([a, b, c])


def max_cutoff(H) -> float:
    """Largest cutoff for which the sequential minimum image is exact."""
    H = np.asarray(H, float)
    return 0.5 * float(min(H[0, 0], H[1, 1], H[2, 2]))


def check_box(H, cutoff: float) -> None:
    Hr = reduce_box(H)
    if not np.allclose(Hr, H, atol=1e-9):
        raise ValueError("box is not in reduced form; use box.reduce_box(H) (same lattice)")
    if cutoff > max_cutoff(H):
        raise ValueError(f"cutoff + skin {cutoff:.4f} nm exceeds half the box height {max_cutoff(H):.4f} nm")


def min_image(dx, H):
    """Minimum-image displacement(s) (..., 3) for a reduced lower-triangular box (JAX)."""
    dx = dx - jnp.round(dx[..., 2:3] / H[2, 2]) * H[2]
    dx = dx - jnp.round(dx[..., 1:2] / H[1, 1]) * H[1]
    dx = dx - jnp.round(dx[..., 0:1] / H[0, 0]) * H[0]
    return dx


def volume(H):
    return jnp.abs(H[0, 0] * H[1, 1] * H[2, 2])


def to_fractional(x, H):
    """x = u @ H  ->  u (not wrapped)."""
    return x @ jnp.linalg.inv(H)


def wrap_fractional(x, H):
    u = to_fractional(x, H)
    return u - jnp.floor(u)


def jaxmd_box(H):
    """JAX-MD affine box T (upper triangular, columns are lattice vectors): x = T u."""
    return jnp.asarray(H).T
