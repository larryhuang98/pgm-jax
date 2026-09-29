"""Periodic boxes in Amber/OpenMM reduced triclinic form.

Contents: the reduced form (`reduce_box`, `check_box`, `max_cutoff`, `lower_triangular_frame`),
minimum image and fractional coordinates (`min_image`, `to_fractional`, `wrap_fractional`),
closed-form 3x3 algebra that fuses into jitted kernels (`volume`, `det3`, `inv3`), conversions to
JAX-MD boxes and Amber cell parameters (`jaxmd_box`, `box_from_cell`, `cell_parameters`), and
molecular centres of mass (`centers_of_mass`, for the molecular virial and the barostat).

H holds the lattice vectors as rows [nm]: a = (ax, 0, 0), b = (bx, by, 0), c = (cx, cy, cz),
with |bx| <= ax/2, |cx| <= ax/2, |cy| <= by/2.  In that form the sequential minimum-image
reduction (c, then b, then a) is exact for any pair closer than half the smallest of ax, by, cz,
which therefore bounds the cutoff (OpenMM uses the same rule).  Amber's truncated octahedron is in
this form.  JAX-MD boxes are the transpose (columns are lattice vectors, upper triangular).
Positions and fractional coordinates are related by x = u @ H (row vectors).

Functions marked "host" work on numpy arrays and are not traceable; the others are JAX and
differentiable.

Units: nm (cell lengths in whatever unit is given), degrees for cell angles.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def reduce_box(H: ArrayLike) -> np.ndarray:
    """Bring a lower-triangular box to reduced form (same lattice; host).

    Parameters
    ----------
    H : ArrayLike (3, 3)
        Lower-triangular box, lattice vectors as rows [nm].

    Returns
    -------
    np.ndarray (3, 3)
        Reduced box: c reduced by b (on the y component), then by a, then b by a (on the x
        component) with nearest-integer multiples, so that |bx| <= ax/2, |cx| <= ax/2,
        |cy| <= by/2.

    Raises
    ------
    ValueError
        If H has an upper-triangle element above 1e-12 of its largest element.
    """
    H = np.array(H, float)
    if np.any(np.abs(np.triu(H, 1)) > 1e-12 * np.abs(H).max()):
        raise ValueError("box must be lower triangular (a along x, b in the xy plane)")
    a, b, c = H
    c = c - b * np.round(c[1] / b[1])
    c = c - a * np.round(c[0] / a[0])
    b = b - a * np.round(b[0] / a[0])
    return np.array([a, b, c])


def lower_triangular_frame(x: ArrayLike, H: ArrayLike, *vectors: ArrayLike) -> tuple[np.ndarray, ...]:
    """Rotate positions, a box and further vectors into the frame where the box is lower triangular.

    Host-side.  The rotation Q comes from the QR decomposition H^T = Q R, with the signs chosen so
    that the diagonal of H Q = R^T is positive (the form the engine assumes).  A strained box
    H (1 + eps)^T with eps_ab != 0 for a > b is not lower triangular: evaluate its energy in this
    frame (the energy is rotation invariant).

    Parameters
    ----------
    x : ArrayLike (n, 3)
        Positions [nm].
    H : ArrayLike (3, 3)
        Box, lattice vectors as rows [nm].
    *vectors : ArrayLike (..., 3)
        Further row vectors to rotate (e.g. velocities, a field).

    Returns
    -------
    tuple of np.ndarray
        (x Q, H Q, *(v Q for v in vectors)).
    """
    Q, R = np.linalg.qr(np.asarray(H, float).T)
    Q = Q @ np.diag(np.sign(np.diag(R)))  # flip columns so that diag(H Q) = diag(R) > 0
    return (np.asarray(x, float) @ Q, np.asarray(H, float) @ Q) + tuple(np.asarray(v, float) @ Q for v in vectors)


def max_cutoff(H: ArrayLike) -> float:
    """Return the largest cutoff for which the sequential minimum image is exact [nm].

    Parameters
    ----------
    H : ArrayLike (3, 3)
        Reduced lower-triangular box, lattice vectors as rows [nm].

    Returns
    -------
    float
        min(ax, by, cz) / 2 [nm].
    """
    H = np.asarray(H, float)
    return 0.5 * float(min(H[0, 0], H[1, 1], H[2, 2]))


def check_box(H: ArrayLike, cutoff: float) -> None:
    """Check that a box is in reduced form and large enough for a cutoff (host).

    Parameters
    ----------
    H : ArrayLike (3, 3)
        Box, lattice vectors as rows [nm].
    cutoff : float
        Largest pair distance the caller needs under the minimum image [nm] (the callers pass the
        pair cutoff plus the neighbour-list skin).

    Raises
    ------
    ValueError
        If H differs from `reduce_box(H)` by more than 1e-9, or if `cutoff` exceeds
        `max_cutoff(H)`.
    """
    Hr = reduce_box(H)
    if not np.allclose(Hr, H, atol=1e-9):
        raise ValueError("box is not in reduced form; use box.reduce_box(H) (same lattice)")
    if cutoff > max_cutoff(H):
        raise ValueError(f"cutoff + skin {cutoff:.4f} nm exceeds half the box height {max_cutoff(H):.4f} nm")


def min_image(dx: jax.Array, H: jax.Array) -> jax.Array:
    """Return the minimum-image displacement(s) for a reduced lower-triangular box.

    Parameters
    ----------
    dx : jax.Array (..., 3)
        Displacements [nm].
    H : jax.Array (3, 3)
        Reduced box, lattice vectors as rows [nm].

    Returns
    -------
    jax.Array (..., 3)
        Displacements shifted by lattice vectors [nm]; exact for |dx| < max_cutoff(H).

    Notes
    -----
    Sequential reduction along c (by the z component), then b (y), then a (x), with nearest-integer
    multiples.  Round is piecewise constant, so the derivative with respect to `dx` is the identity.
    """
    dx = dx - jnp.round(dx[..., 2:3] / H[2, 2]) * H[2]
    dx = dx - jnp.round(dx[..., 1:2] / H[1, 1]) * H[1]
    dx = dx - jnp.round(dx[..., 0:1] / H[0, 0]) * H[0]
    return dx


def volume(H: jax.Array) -> jax.Array:
    """Return the volume of a lower- or upper-triangular box [nm^3] (product of the diagonal).

    Parameters
    ----------
    H : jax.Array (3, 3)
        Triangular box [nm].  For a general matrix use `abs(det3(H))`.

    Returns
    -------
    jax.Array ()
        |ax by cz| [nm^3].
    """
    return jnp.abs(H[0, 0] * H[1, 1] * H[2, 2])


def det3(H: jax.Array) -> jax.Array:
    """Return the determinant of a general 3x3 matrix in closed form.

    The closed form fuses into the surrounding kernel, unlike `jnp.linalg.det`, which launches LU
    factorisations on the GPU at every call.  Differentiable.

    Parameters
    ----------
    H : jax.Array (3, 3)
        Matrix (e.g. a box [nm]).

    Returns
    -------
    jax.Array ()
        det H (e.g. the signed box volume [nm^3]).
    """
    return (
        H[0, 0] * (H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1])
        - H[0, 1] * (H[1, 0] * H[2, 2] - H[1, 2] * H[2, 0])
        + H[0, 2] * (H[1, 0] * H[2, 1] - H[1, 1] * H[2, 0])
    )


def inv3(H: ArrayLike) -> jax.Array:
    """Return the inverse of a general 3x3 matrix by cofactors (differentiable; see det3).

    Parameters
    ----------
    H : ArrayLike (3, 3)
        Non-singular matrix (e.g. a box [nm]).

    Returns
    -------
    jax.Array (3, 3)
        adj(H) / det(H) (e.g. [1/nm]).
    """
    H = jnp.asarray(H)
    adj = jnp.stack(
        [
            jnp.stack(
                [
                    H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1],
                    H[0, 2] * H[2, 1] - H[0, 1] * H[2, 2],
                    H[0, 1] * H[1, 2] - H[0, 2] * H[1, 1],
                ]
            ),
            jnp.stack(
                [
                    H[1, 2] * H[2, 0] - H[1, 0] * H[2, 2],
                    H[0, 0] * H[2, 2] - H[0, 2] * H[2, 0],
                    H[0, 2] * H[1, 0] - H[0, 0] * H[1, 2],
                ]
            ),
            jnp.stack(
                [
                    H[1, 0] * H[2, 1] - H[1, 1] * H[2, 0],
                    H[0, 1] * H[2, 0] - H[0, 0] * H[2, 1],
                    H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0],
                ]
            ),
        ]
    )
    return adj / det3(H)


def to_fractional(x: jax.Array, H: jax.Array) -> jax.Array:
    """Return the fractional coordinates u of positions x = u @ H (not wrapped).

    Parameters
    ----------
    x : jax.Array (..., 3)
        Positions [nm].
    H : jax.Array (3, 3)
        Box, lattice vectors as rows [nm].

    Returns
    -------
    jax.Array (..., 3)
        u = x H^-1 (dimensionless), computed with Precision.HIGHEST (TF32 matmuls on the GPU would
        lose digits).
    """
    return jnp.matmul(x, inv3(H), precision=jax.lax.Precision.HIGHEST)


def wrap_fractional(x: jax.Array, H: jax.Array) -> jax.Array:
    """Return the fractional coordinates of positions wrapped into [0, 1).

    Parameters
    ----------
    x : jax.Array (..., 3)
        Positions [nm].
    H : jax.Array (3, 3)
        Box, lattice vectors as rows [nm].

    Returns
    -------
    jax.Array (..., 3)
        u - floor(u) with u = x H^-1 (dimensionless).
    """
    u = to_fractional(x, H)
    return u - jnp.floor(u)


def jaxmd_box(H: ArrayLike) -> jax.Array:
    """Return the JAX-MD affine box T = H^T (upper triangular, columns are lattice vectors): x = T u.

    Parameters
    ----------
    H : ArrayLike (3, 3)
        Box, lattice vectors as rows [nm].

    Returns
    -------
    jax.Array (3, 3)
        H^T [nm].
    """
    return jnp.asarray(H).T


def box_from_cell(lengths: Sequence[float], angles: Sequence[float]) -> np.ndarray:
    """Return the box matrix of cell parameters (Amber / PDB convention: a along x, b in the xy plane).

    Parameters
    ----------
    lengths : Sequence[float] (3,)
        a, b, c (any length unit; the result has the same unit).
    angles : Sequence[float] (3,)
        alpha, beta, gamma [deg].

    Returns
    -------
    np.ndarray (3, 3)
        Lattice vectors as rows, lower triangular (not reduced; see reduce_box).
    """
    a, b, c = lengths
    alpha, beta, gamma = angles
    al, be, ga = np.radians([alpha, beta, gamma])
    ax = np.array([a, 0.0, 0.0])
    bx = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    return np.array([ax, bx, [cx, cy, cz]])


def cell_parameters(H: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return the cell parameters (lengths, angles) of a box matrix (host).

    Parameters
    ----------
    H : ArrayLike (3, 3)
        Box, lattice vectors as rows (any length unit, e.g. Angstrom or nm).

    Returns
    -------
    lengths : np.ndarray (3,)
        |a|, |b|, |c| in the unit of `H`.
    angles : np.ndarray (3,)
        alpha (b, c), beta (a, c), gamma (a, b) [deg].
    """
    H = np.asarray(H, float)
    a, b, c = np.linalg.norm(H, axis=1)
    alpha = np.degrees(np.arccos(np.dot(H[1], H[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(H[0], H[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(H[0], H[1]) / (a * b)))
    return np.array([a, b, c]), np.array([alpha, beta, gamma])


def centers_of_mass(pos: jax.Array, masses: jax.Array, mol: jax.Array, nmol: int) -> jax.Array:
    """Return the mass-weighted centres of the molecules (JAX, traceable).

    Parameters
    ----------
    pos : jax.Array (N, 3)
        Atomic positions [nm].
    masses : jax.Array (N,)
        Atomic masses [amu].
    mol : jax.Array (N,) int
        Molecule index of every atom (0 .. nmol-1).
    nmol : int
        Number of molecules (static).

    Returns
    -------
    jax.Array (nmol, 3)
        Centres of mass [nm] (no minimum image: molecules must be whole).
    """
    return jax.ops.segment_sum(masses[:, None] * pos, mol, nmol) / jax.ops.segment_sum(masses, mol, nmol)[:, None]
