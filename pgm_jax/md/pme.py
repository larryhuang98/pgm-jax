"""Smooth particle-mesh Ewald for charges and point dipoles: the reciprocal-space part of pGM.

    E_rec = sum_{m != 0} exp(-pi^2 |m*|^2 / beta^2) / (2 pi V |m*|^2) * B(m) * |F[Q](m)|^2

(Essmann et al., JCP 103, 8577 (1995); dipoles as in Sagui, Pedersen & Darden, JCP 120, 73
(2004)), m* = m H^-T, B(m) the Euler exponential-spline moduli, F the unnormalised DFT, and Q the
charges and dipoles spread on a K1 x K2 x K3 grid with cardinal B-splines of order p:

    Q(k) = sum_i [ q_i M(w_i1 - k1) M(w_i2 - k2) M(w_i3 - k3) + e_i . grad_w (M M M) ],
    w_i = K * frac(r_i),   e_i = K * (d_i H^-1)   (the dipole in scaled fractional units),

i.e. a dipole is spread as d . grad_r of its charge spline.  B-spline recursion, index convention
and moduli follow OpenMM's reference PME.  Everything is a JAX function of positions, charges,
dipoles and box, so energies, fields, forces and box derivatives come from autodiff.
Units: nm, e, e nm; energies in e^2/nm (multiply by the Coulomb constant).  H has the lattice
vectors as rows.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def bspline(f, order: int):
    """Cardinal B-spline weights and their derivatives at fractional offset f in [0, 1).
    f (...,) -> theta, dtheta (..., order)."""
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
    """|b(m)|^-2 denominators: |sum_j M(j+1) exp(2 pi i m j / K)|^2 for m = 0..K-1 (numpy, float64)."""
    data = np.asarray(bspline(jnp.zeros((), jnp.float64), order)[0], float)
    j = np.arange(order)
    m = np.arange(K)[:, None]
    arg = 2 * np.pi * m * j[None, :] / K
    mod = (data * np.cos(arg)).sum(1) ** 2 + (data * np.sin(arg)).sum(1) ** 2
    for i in range(K):                                       # odd orders have zeros at m = K/2
        if mod[i] < 1e-7:
            mod[i] = 0.5 * (mod[(i - 1) % K] + mod[(i + 1) % K])
    return mod


def grid_size(H, spacing: float = 0.05, factors=(2, 3, 5)) -> tuple[int, int, int]:
    """Smallest FFT-friendly grid with at most `spacing` nm between planes."""
    H = np.asarray(H, float)
    V = abs(np.linalg.det(H))
    heights = [V / np.linalg.norm(np.cross(H[1], H[2])), V / np.linalg.norm(np.cross(H[2], H[0])),
               V / np.linalg.norm(np.cross(H[0], H[1]))]

    def ok(n):
        for f in factors:
            while n % f == 0:
                n //= f
        return n == 1

    out = []
    for h in heights:
        n = max(8, int(np.ceil(h / spacing)))
        while not ok(n) or n % 2:
            n += 1
        out.append(n)
    return tuple(out)


class PME:
    """Reciprocal-space pGM energy on a fixed grid.  `setup` does the per-geometry spline work
    once so that many spreads (one per CG iteration) reuse it."""

    def __init__(self, grid, order: int, beta: float, dtype=jnp.float32):
        self.K = tuple(int(k) for k in grid)
        self.order, self.beta, self.dtype = int(order), float(beta), dtype
        K1, K2, K3 = self.K
        mods = [bspline_moduli(k, self.order) for k in self.K]
        Binv = 1.0 / (mods[0][:, None, None] * mods[1][None, :, None] * mods[2][None, None, :K3 // 2 + 1])
        w3 = np.full(K3 // 2 + 1, 2.0)
        w3[0] = 1.0
        if K3 % 2 == 0:
            w3[-1] = 1.0
        self._Bw = jnp.asarray(Binv * w3[None, None, :])                      # float64
        self._m = [jnp.asarray(np.fft.fftfreq(K1, 1.0 / K1)), jnp.asarray(np.fft.fftfreq(K2, 1.0 / K2)),
                   jnp.asarray(np.arange(K3 // 2 + 1, dtype=float))]
        self._Kf = jnp.asarray(np.array(self.K, float))
        self._ar = jnp.arange(self.order)

    def influence(self, H):
        """G(m) on the rfft grid, including the spline moduli and half-space weights (e^2/nm)."""
        H = jnp.asarray(H, jnp.float64)
        R = jnp.linalg.inv(H).T                                                # rows: reciprocal vectors
        m1, m2, m3 = self._m
        mv = (m1[:, None, None, None] * R[0] + m2[None, :, None, None] * R[1] + m3[None, None, :, None] * R[2])
        msq = jnp.sum(mv * mv, -1)
        V = jnp.abs(jnp.linalg.det(H))
        safe = jnp.where(msq > 0, msq, 1.0)
        G = jnp.where(msq > 0, jnp.exp(-(jnp.pi ** 2) * safe / self.beta ** 2) / (2 * jnp.pi * V * safe), 0.0)
        return (G * self._Bw).astype(self.dtype)

    def setup(self, pos, H):
        """Spline indices and weights for one geometry."""
        H = jnp.asarray(H, jnp.float64)
        Hinv = jnp.linalg.inv(H)
        u = jnp.asarray(pos, jnp.float64) @ Hinv
        w = (u - jnp.floor(u)) * self._Kf
        base = jnp.floor(w)
        f = (w - base).astype(self.dtype)
        th, dth = bspline(f, self.order)                                       # (N, 3, p)
        idx = (base.astype(jnp.int32)[:, :, None] + self._ar[None, None, :]) % jnp.asarray(self.K, jnp.int32)[None, :, None]
        K1, K2, K3 = self.K
        flat = (idx[:, 0, :, None, None] * K2 + idx[:, 1, None, :, None]) * K3 + idx[:, 2, None, None, :]
        return {"flat": flat.reshape(pos.shape[0], -1), "th": th, "dth": dth,
                "e_scale": (Hinv * self._Kf[None, :]).astype(self.dtype)}

    def spread(self, S, q, d):
        """Grid Q (K1, K2, K3) of charges q (N,) and dipoles d (N, 3) e nm."""
        th, dth = S["th"], S["dth"]
        e = d.astype(self.dtype) @ S["e_scale"]                               # (N, 3) scaled fractional dipoles
        t1, t2, t3 = th[:, 0], th[:, 1], th[:, 2]
        d1, d2, d3 = dth[:, 0], dth[:, 1], dth[:, 2]
        a1 = q.astype(self.dtype)[:, None] * t1 + e[:, 0:1] * d1
        # q t1 t2 t3 + e1 d1 t2 t3 + e2 t1 d2 t3 + e3 t1 t2 d3
        val = (a1[:, :, None, None] * t2[:, None, :, None] * t3[:, None, None, :]
               + (e[:, 1:2] * t1)[:, :, None, None] * d2[:, None, :, None] * t3[:, None, None, :]
               + (e[:, 2:3] * t1)[:, :, None, None] * t2[:, None, :, None] * d3[:, None, None, :])
        K1, K2, K3 = self.K
        Q = jnp.zeros(K1 * K2 * K3, self.dtype).at[S["flat"].reshape(-1)].add(val.reshape(-1))
        return Q.reshape(K1, K2, K3)

    def energy(self, S, G, q, d):
        """Reciprocal energy (e^2/nm), float64 accumulation."""
        FQ = jnp.fft.rfftn(self.spread(S, q, d))
        return jnp.sum((G * (FQ.real ** 2 + FQ.imag ** 2)).astype(jnp.float64))
