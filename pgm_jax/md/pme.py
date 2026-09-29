"""Smooth particle-mesh Ewald for charges and point dipoles: the reciprocal-space part of pGM.

Contents: the B-spline weights (`bspline`) and their Euler exponential-spline moduli
(`bspline_moduli`), the grid-size rule (`grid_size`), and `PME`, the reciprocal energy on a fixed
grid with a per-geometry `setup`, the influence function (`influence`), the charge/dipole spread
(`spread`), the energy (`energy`) and a hand-written dipole gradient (`grad_dipoles`) for the
conjugate-gradient induction solver.

The reciprocal energy of the smooth PME [1]_, with dipoles as in [2]_, is

    E_rec = sum_{m != 0} exp(-pi^2 |m*|^2 / beta^2) / (2 pi V |m*|^2) * B(m) * |F[Q](m)|^2

with m* = m H^-T the reciprocal lattice vector of the integer triple m [1/nm], beta the Ewald
screening parameter [1/nm], V the box volume [nm^3], B(m) the Euler exponential-spline moduli,
F the unnormalised DFT, and Q the charges and dipoles spread on a K1 x K2 x K3 grid with cardinal
B-splines M of order p:

    Q(k) = sum_i [ q_i M(w_i1 - k1) M(w_i2 - k2) M(w_i3 - k3) + e_i . grad_w (M M M) ],
    w_i = K * frac(r_i),   e_i = K * (d_i H^-1)   (the dipole in scaled fractional units),

i.e. a dipole is spread as d . grad_r of its charge spline.  The B-spline recursion, the index
convention (grid points floor(w) .. floor(w) + p - 1) and the moduli follow OpenMM's reference
PME.  Everything is a JAX function of positions, charges, dipoles and box, so energies, fields,
forces and box derivatives come from autodiff; the sums over m use the real-to-complex FFT (half
spectrum along K3 with weights 2 for the pairs +-m).

Precision: fractional coordinates and the influence function are computed in float64; the
spline weights, the grid and the FFTs in `PME.dtype` (float32 in mixed precision), with every
matmul at Precision.HIGHEST; the energy is accumulated in float64.

Units: nm, e, e nm; energies in e^2/nm (multiply by the Coulomb constant, units.py, for
kJ/mol).  H has the lattice vectors as rows.

References
----------
.. [1] U. Essmann, L. Perera, M. L. Berkowitz, T. Darden, H. Lee, L. G. Pedersen, J. Chem. Phys.
   103, 8577 (1995).
.. [2] C. Sagui, L. G. Pedersen, T. A. Darden, J. Chem. Phys. 120, 73 (2004).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike

from .box import det3, inv3

# float32 matrix products on NVIDIA GPUs default to TF32 (10-bit mantissa): about 1e-3 relative
# error in the dipole spread, which dominated the mixed-precision force error.  Always full precision.
_HI = jax.lax.Precision.HIGHEST


def bspline(f: jax.Array, order: int) -> tuple[jax.Array, jax.Array]:
    """Return the cardinal B-spline weights and their derivatives at fractional offsets f in [0, 1).

    Parameters
    ----------
    f : jax.Array (...,)
        Offset of the scaled fractional coordinate from the grid point below it (dimensionless).
    order : int
        Spline order p (static), >= 4.

    Returns
    -------
    theta : jax.Array (..., p)
        theta[..., k] = M_p(f + p - 1 - k), the weight of grid point floor(w) + k.
    dtheta : jax.Array (..., p)
        d theta / d f.

    Raises
    ------
    ValueError
        If order < 4.

    Notes
    -----
    OpenMM's reference recursion: M_2 = (1 - f, f), raised one order at a time to p - 1, the
    derivatives taken from the order p - 1 values (dM_p(x)/dx = M_{p-1}(x) - M_{p-1}(x - 1)), then
    the last order.
    """
    if order < 4:
        raise ValueError("PME order must be >= 4")
    one = jnp.ones_like(f)
    data = [jnp.zeros_like(f)] * order
    data[0] = one - f
    data[1] = f
    for j in range(3, order):
        div = 1.0 / (j - 1.0)
        data[j - 1] = div * f * data[j - 2]
        for k in range(1, j - 1):
            data[j - k - 1] = div * ((f + k) * data[j - k - 2] + (j - k - f) * data[j - k - 1])
        data[0] = div * (one - f) * data[0]
    ddata = [-data[0]] + [data[k - 1] - data[k] for k in range(1, order)]
    div = 1.0 / (order - 1.0)
    data[order - 1] = div * f * data[order - 2]
    for k in range(1, order - 1):
        data[order - k - 1] = div * ((f + k) * data[order - k - 2] + (order - k - f) * data[order - k - 1])
    data[0] = div * (one - f) * data[0]
    return jnp.stack(data, -1), jnp.stack(ddata, -1)


def bspline_moduli(K: int, order: int) -> np.ndarray:
    """Return the squared spline moduli |b(m)|^-2 along one grid axis (numpy, float64).

    Parameters
    ----------
    K : int
        Number of grid points along the axis.
    order : int
        Spline order p.

    Returns
    -------
    np.ndarray (K,)
        |sum_j M_p(j + 1) exp(2 pi i m j / K)|^2 for m = 0 .. K-1 (the denominators of B(m)).
        Values below 1e-7 (the zeros of odd orders at m = K/2) are replaced by the mean of the two
        neighbours, as in OpenMM.
    """
    data = np.asarray(bspline(jnp.zeros((), jnp.float64), order)[0], float)
    j = np.arange(order)
    m = np.arange(K)[:, None]
    arg = 2 * np.pi * m * j[None, :] / K
    mod = (data * np.cos(arg)).sum(1) ** 2 + (data * np.sin(arg)).sum(1) ** 2
    for i in range(K):  # odd orders have zeros at m = K/2
        if mod[i] < 1e-7:
            mod[i] = 0.5 * (mod[(i - 1) % K] + mod[(i + 1) % K])
    return mod


def grid_size(H: ArrayLike, spacing: float = 0.05, factors: Sequence[int] = (2, 3, 5)) -> tuple[int, int, int]:
    """Return the smallest FFT-friendly grid with at most `spacing` between lattice planes (host).

    Parameters
    ----------
    H : ArrayLike (3, 3)
        Box, lattice vectors as rows [nm].
    spacing : float
        Largest spacing between grid planes [nm].
    factors : Sequence[int]
        Allowed prime factors of the grid sizes.

    Returns
    -------
    tuple of int
        (K1, K2, K3): for each axis the smallest even n >= max(8, ceil(h / spacing)) with only
        `factors` as prime factors, h = V / |b x c| (and cyclic) the distance between lattice planes.
    """
    H = np.asarray(H, float)
    V = abs(np.linalg.det(H))
    heights = [
        V / np.linalg.norm(np.cross(H[1], H[2])),
        V / np.linalg.norm(np.cross(H[2], H[0])),
        V / np.linalg.norm(np.cross(H[0], H[1])),
    ]

    def ok(n: int) -> bool:  # n has only `factors` as prime factors
        for f in factors:
            while n % f == 0:
                n //= f
        return n == 1

    out = []
    for h in heights:
        n = max(8, int(np.ceil(h / spacing)))
        while not ok(n) or n % 2:  # even sizes only
            n += 1
        out.append(n)
    return tuple(out)


class PME:
    """Reciprocal-space pGM energy of charges and point dipoles on a fixed grid.

    `setup` does the per-geometry spline work once, so that the many spreads of one step (one per
    conjugate-gradient iteration of the induction solver) reuse it; `influence` depends only on the
    box.  The instance holds no traced state (grid, order, beta and dtype are fixed at construction),
    so its methods can be called inside jitted functions; it is not a pytree.

        pme = PME(grid_size(H), order=6, beta=3.5)
        S, G = pme.setup(pos, H), pme.influence(H)
        E = pme.energy(S, G, q, d)                       # [e^2/nm]

    Attributes
    ----------
    K : tuple of int
        Grid (K1, K2, K3).
    order : int
        B-spline order p.
    beta : float
        Ewald screening parameter [1/nm].
    dtype : jnp.dtype
        Dtype of the spline weights, grid and FFTs.

    Notes
    -----
    The per-geometry dict S returned by `setup` holds

        flat     int32 (N, p^3)    flattened grid indices ((k1 K2 + k2) K3 + k3) of every atom's
                                   p x p x p stencil
        th, dth  dtype (N, 3, p)   spline weights and derivatives along the three axes
        e_scale  dtype (3, 3)      H^-1 with column a scaled by K_a (d @ e_scale = scaled
                                   fractional dipole) [1/nm]
    """

    def __init__(self, grid: Sequence[int], order: int, beta: float, dtype: DTypeLike = jnp.float32) -> None:
        """Set up the grid, the spline moduli and the wave-vector indices.

        Parameters
        ----------
        grid : Sequence[int] (3,)
            Grid sizes (K1, K2, K3), e.g. from `grid_size`.
        order : int
            B-spline order p (>= 4).
        beta : float
            Ewald screening parameter [1/nm].
        dtype : jnp.dtype
            Dtype of the grid work (float32 for mixed precision, float64 for double).
        """
        self.K = tuple(int(k) for k in grid)
        self.order, self.beta, self.dtype = int(order), float(beta), dtype
        K1, K2, K3 = self.K
        mods = [bspline_moduli(k, self.order) for k in self.K]
        Binv = 1.0 / (mods[0][:, None, None] * mods[1][None, :, None] * mods[2][None, None, : K3 // 2 + 1])
        # half-space weights of the rfft axis: the planes m3 = 0 (and K3/2 for even K3) count once
        w3 = np.full(K3 // 2 + 1, 2.0)
        w3[0] = 1.0
        if K3 % 2 == 0:
            w3[-1] = 1.0
        self._Bw = jnp.asarray(Binv * w3[None, None, :])  # float64
        self._w3inv = jnp.asarray(1.0 / w3, dtype)  # undo the half-space weights
        self._m = [  # integer wave-vector indices along each axis (signed; half spectrum along K3)
            jnp.asarray(np.fft.fftfreq(K1, 1.0 / K1)),
            jnp.asarray(np.fft.fftfreq(K2, 1.0 / K2)),
            jnp.asarray(np.arange(K3 // 2 + 1, dtype=float)),
        ]
        self._Kf = jnp.asarray(np.array(self.K, float))
        self._ar = jnp.arange(self.order, dtype=jnp.int32)

    def influence(self, H: ArrayLike) -> jax.Array:
        """Return the influence function G(m) on the rfft grid (half spectrum along K3).

        Parameters
        ----------
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].

        Returns
        -------
        jax.Array (K1, K2, K3//2 + 1)
            exp(-pi^2 |m*|^2 / beta^2) / (2 pi V |m*|^2) times the spline moduli B(m) and the
            half-space weights (2, or 1 for the planes m3 = 0 and m3 = K3/2), 0 at m = 0 [1/nm]; in
            `dtype` (computed in float64).  Multiplied by |F[Q]|^2 it gives the energy in e^2/nm.
            Differentiable in H (box derivatives of the energy).
        """
        H = jnp.asarray(H, jnp.float64)
        R = inv3(H).T  # rows: reciprocal vectors
        m1, m2, m3 = self._m
        mv = m1[:, None, None, None] * R[0] + m2[None, :, None, None] * R[1] + m3[None, None, :, None] * R[2]
        msq = jnp.sum(mv * mv, -1)  # |m*|^2 [1/nm^2]
        V = jnp.abs(det3(H))
        safe = jnp.where(msq > 0, msq, 1.0)  # double where: no 0/0 in the value or gradient at m = 0
        G = jnp.where(msq > 0, jnp.exp(-(jnp.pi**2) * safe / self.beta**2) / (2 * jnp.pi * V * safe), 0.0)
        return (G * self._Bw).astype(self.dtype)

    def setup(self, pos: ArrayLike, H: ArrayLike) -> dict[str, jax.Array]:
        """Return the spline indices and weights of one geometry (the dict S of the class docstring).

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Positions [nm] (need not be wrapped).
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].

        Returns
        -------
        dict of str to jax.Array
            "flat", "th", "dth", "e_scale" (see the class docstring).  The fractional coordinates are
            computed in float64 and the offsets f cast to `dtype`; `flat` is piecewise constant, so
            derivatives with respect to positions flow through th and dth only.
        """
        H = jnp.asarray(H, jnp.float64)
        Hinv = inv3(H)
        u = jnp.matmul(jnp.asarray(pos, jnp.float64), Hinv, precision=_HI)
        w = (u - jnp.floor(u)) * self._Kf  # scaled fractional coordinates in [0, K)
        base = jnp.floor(w)
        f = (w - base).astype(self.dtype)
        th, dth = bspline(f, self.order)  # (N, 3, p)
        idx = (base.astype(jnp.int32)[:, :, None] + self._ar[None, None, :]) % jnp.asarray(self.K, jnp.int32)[
            None, :, None
        ]
        K1, K2, K3 = self.K
        # (N, p, p, p) flattened grid index of every stencil point
        flat = (idx[:, 0, :, None, None] * K2 + idx[:, 1, None, :, None]) * K3 + idx[:, 2, None, None, :]
        return {
            "flat": flat.reshape(pos.shape[0], -1),
            "th": th,
            "dth": dth,
            "e_scale": (Hinv * self._Kf[None, :]).astype(self.dtype),
        }

    def spread(self, S: dict[str, jax.Array], q: jax.Array, d: jax.Array) -> jax.Array:
        """Return the grid Q of charges and dipoles spread with the B-splines of `setup`.

        Parameters
        ----------
        S : dict of str to jax.Array
            Output of `setup`.
        q : jax.Array (N,)
            Charges [e].
        d : jax.Array (N, 3)
            Dipoles [e nm].

        Returns
        -------
        jax.Array (K1, K2, K3)
            Q(k) = sum_i [q_i t1 t2 t3 + e_i1 d1 t2 t3 + e_i2 t1 d2 t3 + e_i3 t1 t2 d3] (t = weights,
            d = derivatives along each axis, e = scaled fractional dipole), in `dtype` [e].
        """
        th, dth = S["th"], S["dth"]
        e = jnp.matmul(d.astype(self.dtype), S["e_scale"], precision=_HI)  # (N, 3) scaled fractional dipoles
        t1, t2, t3 = th[:, 0], th[:, 1], th[:, 2]
        d1, d2, d3 = dth[:, 0], dth[:, 1], dth[:, 2]
        a1 = q.astype(self.dtype)[:, None] * t1 + e[:, 0:1] * d1
        # q t1 t2 t3 + e1 d1 t2 t3 + e2 t1 d2 t3 + e3 t1 t2 d3
        val = (
            a1[:, :, None, None] * t2[:, None, :, None] * t3[:, None, None, :]
            + (e[:, 1:2] * t1)[:, :, None, None] * d2[:, None, :, None] * t3[:, None, None, :]
            + (e[:, 2:3] * t1)[:, :, None, None] * t2[:, None, :, None] * d3[:, None, None, :]
        )
        K1, K2, K3 = self.K
        Q = jnp.zeros(K1 * K2 * K3, self.dtype).at[S["flat"].reshape(-1)].add(val.reshape(-1))
        return Q.reshape(K1, K2, K3)

    def grad_dipoles(self, S: dict[str, jax.Array], G: jax.Array, q: jax.Array, d: jax.Array) -> jax.Array:
        """Return dU_rec/dd, the gradient of the reciprocal energy with respect to the dipoles.

        The potential derivative dU/dQ on the grid costs one r2c and one c2r FFT and is interpolated
        back to the atoms with the spline derivatives.  Same as jax.grad(energy) with respect to d,
        without the float64 accumulation and the transposes of the autodiff path (it runs once per
        conjugate-gradient iteration).

        Parameters
        ----------
        S : dict of str to jax.Array
            Output of `setup`.
        G : jax.Array (K1, K2, K3//2 + 1)
            Output of `influence`.
        q : jax.Array (N,)
            Charges [e].
        d : jax.Array (N, 3)
            Dipoles [e nm].

        Returns
        -------
        jax.Array (N, 3)
            dU_rec/dd [e/nm^2] (minus the reciprocal-space field, before the Coulomb constant), in
            `dtype`.

        Notes
        -----
        With U = sum_m G_m |F[Q]_m|^2 over the full spectrum, dU/dQ = 2 K1 K2 K3 irfft(G_m F[Q]_m); G
        carries the half-space weights, which are divided out first.  The chain rule through
        Q(e) with e = d @ e_scale gives g_a = sum_k phi(k) dQ(k)/de_a and dU/dd = g @ e_scale^T.
        """
        K1, K2, K3 = self.K
        p = self.order
        Q = self.spread(S, q, d)
        # U = sum_m G_m |FQ_m|^2 over the full spectrum; dU/dQ = 2 K1 K2 K3 irfft(G_m FQ_m)
        phi = jnp.fft.irfftn(jnp.fft.rfftn(Q) * (G * self._w3inv), Q.shape) * jnp.asarray(
            2.0 * K1 * K2 * K3, self.dtype
        )
        n = S["flat"].shape[0]
        ph = phi.reshape(-1)[S["flat"]].reshape(n, p, p, p)  # dU/dQ on every atom's stencil
        th, dth = S["th"], S["dth"]
        t1, t2, t3 = th[:, 0], th[:, 1], th[:, 2]
        d1, d2, d3 = dth[:, 0], dth[:, 1], dth[:, 2]
        A = jnp.einsum("nabc,nc->nab", ph, t3, precision=_HI)
        B = jnp.einsum("nabc,nc->nab", ph, d3, precision=_HI)
        g1 = jnp.einsum("nab,na,nb->n", A, d1, t2, precision=_HI)
        g2 = jnp.einsum("nab,na,nb->n", A, t1, d2, precision=_HI)
        g3 = jnp.einsum("nab,na,nb->n", B, t1, t2, precision=_HI)
        # g_a = dU/de_a (derivative spline along axis a); dU/dd = g e_scale^T
        return jnp.matmul(jnp.stack([g1, g2, g3], -1), S["e_scale"].T, precision=_HI)

    def energy(self, S: dict[str, jax.Array], G: jax.Array, q: jax.Array, d: jax.Array) -> jax.Array:
        """Return the reciprocal-space energy E_rec (module docstring), accumulated in float64.

        Parameters
        ----------
        S : dict of str to jax.Array
            Output of `setup`.
        G : jax.Array (K1, K2, K3//2 + 1)
            Output of `influence`.
        q : jax.Array (N,)
            Charges [e].
        d : jax.Array (N, 3)
            Dipoles [e nm].

        Returns
        -------
        jax.Array () float64
            E_rec [e^2/nm] (multiply by the Coulomb constant for kJ/mol).  Differentiable in positions
            (through S), box (through S and G), charges and dipoles.
        """
        FQ = jnp.fft.rfftn(self.spread(S, q, d))
        return jnp.sum((G * (FQ.real**2 + FQ.imag**2)).astype(jnp.float64))
