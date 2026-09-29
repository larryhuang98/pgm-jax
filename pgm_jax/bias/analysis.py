"""Free energy surfaces from biased simulations (host side, numpy / JAX).

  fes_from_bias      F(s) = -f V(s) on a grid: f = gamma / (gamma - 1) (well-tempered metaD),
                     1 / (1 - 1/gamma) (OPES), 1 (standard metaD)
  metad_ct           c(t) of well-tempered metadynamics (Tiwary & Parrinello, JPCB 119, 736 (2015)),
                         c(t) = kT log[ int ds exp(gamma/(gamma-1) V(s,t)/kT) / int ds exp(V(s,t)/((gamma-1) kT)) ],
                     after every hill
  ct_weights         frame weights exp((V(s(t), t) - c(t)) / kT) (metaD) ...
  opes_weights       ... and exp(V(s(t), t) / kT) (OPES; any quasi-static bias)
  histogram_fes      F = -kT log(weighted histogram) on a grid (any CVs, e.g. ones that were not biased)
  wham               1D weighted histogram analysis of umbrella windows (reference free energies)
  align_rmsd         RMSD between two FES over a region after the best constant shift
Units: kJ/mol and the CVs' units."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def mesh(*axes):
    """Grid points (G, d) of the product of 1D axes, and the grid shape."""
    g = np.meshgrid(*[np.asarray(a, float) for a in axes], indexing="ij")
    return np.stack([x.reshape(-1) for x in g], 1), g[0].shape


def periodic_axis(n: int, lo: float = -np.pi, period: float = 2 * np.pi):
    """n bin centres covering one period [lo, lo + period)."""
    return lo + (np.arange(n) + 0.5) * period / n


def bias_on_grid(bias, state, points, chunk: int = 4096):
    """V(s) (G,) of one bias (core.Bias) at the grid points (G, d)."""
    f = jax.jit(jax.vmap(lambda s: bias.potential(state, s)))
    P = np.asarray(points, float).reshape(-1, bias.d)
    return np.concatenate([np.asarray(f(jnp.asarray(P[i : i + chunk]))) for i in range(0, len(P), chunk)])


def fes_from_bias(bias, state, points):
    """F (G,) = -factor V (min 0) with the bias's own factor (MetaD / OPES fes_factor())."""
    F = -bias.fes_factor() * bias_on_grid(bias, state, points)
    return F - F.min()


def _hill_values(centers, heights, sigma, periods, points):
    """(n_hills, G) Gaussian values of every hill at every grid point (numpy)."""
    d = points[None, :, :] - centers[:, None, :]
    P = np.asarray(periods, float)
    p = np.where(P > 0, P, 1.0)
    d = np.where(P > 0, d - p * np.round(d / p), d) / np.asarray(sigma)
    return heights[:, None] * np.exp(-0.5 * np.sum(d * d, -1))


def metad_ct(hills: dict, biasfactor: float, kT: float, periods, points, chunk: int = 256):
    """c(t) (kJ/mol) of well-tempered metadynamics after each hill (in deposition order).
    hills: MetaD.hills() (or read from prefix.hills: step, center, height, sigma); points: a grid
    (G, d) covering the explored CV space.  Returns (steps of the hills, c)."""
    g = float(biasfactor)
    C, h, sig = np.asarray(hills["center"], float), np.asarray(hills["height"], float), hills["sigma"]
    if C.ndim == 1:
        C = C[:, None]
    V = np.zeros(len(points))
    out = np.zeros(len(h))
    for i in range(0, len(h), chunk):
        H = _hill_values(C[i : i + chunk], h[i : i + chunk], sig, periods, points)
        Vc = V[None, :] + np.cumsum(H, 0)
        a, b = g / (g - 1.0) * Vc / kT, Vc / ((g - 1.0) * kT)
        am, bm = a.max(1, keepdims=True), b.max(1, keepdims=True)
        out[i : i + chunk] = kT * (
            (am[:, 0] + np.log(np.exp(a - am).sum(1))) - (bm[:, 0] + np.log(np.exp(b - bm).sum(1)))
        )
        V = Vc[-1]
    return np.asarray(hills["step"]), out


def ct_weights(frame_steps, frame_bias, hill_steps, ct, kT: float, strict: bool = True):
    """log weights (V(s(t), t) - c(t)) / kT of COLVAR frames: c(t) of the last hill deposited before
    the frame (strict: hill step < frame step, as the drivers record the bias before a same-step
    update), 0 before the first hill."""
    idx = np.searchsorted(np.asarray(hill_steps), np.asarray(frame_steps), side="left" if strict else "right") - 1
    c = np.where(idx >= 0, np.asarray(ct)[np.maximum(idx, 0)], 0.0)
    return (np.asarray(frame_bias) - c) / kT


def opes_weights(frame_bias, kT: float):
    """log weights V / kT of frames sampled with a (quasi-)static bias V (OPES)."""
    return np.asarray(frame_bias) / kT


def histogram_fes(s, logw, axes, kT: float, periods=None):
    """F = -kT log p on the grid of bin centres `axes` (list of 1D arrays, uniform spacing) from
    samples s (n, d) with log weights logw (n,) (None: unweighted).  Periodic dimensions (periods
    > 0) wrap into their period.  Empty bins get inf; min F = 0."""
    s = np.asarray(s, float)
    if s.ndim == 1:
        s = s[:, None]
    n, d = s.shape
    periods = np.zeros(d) if periods is None else np.asarray(periods, float)
    idx = []
    for k, a in enumerate(axes):
        a = np.asarray(a, float)
        da = a[1] - a[0]
        lo = a[0] - 0.5 * da
        x = s[:, k]
        if periods[k] > 0:
            x = lo + np.mod(x - lo, periods[k])
        i = np.floor((x - lo) / da).astype(int)
        idx.append(i)
    idx = np.stack(idx, 1)
    shape = tuple(len(a) for a in axes)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), 1)
    lw = np.zeros(n) if logw is None else np.asarray(logw, float)
    lw = lw - lw[ok].max()
    hist = np.zeros(shape)
    np.add.at(hist, tuple(idx[ok].T), np.exp(lw[ok]))
    with np.errstate(divide="ignore"):
        F = -kT * np.log(hist)
    return F - F[np.isfinite(F)].min()


def align_rmsd(F, F_ref, mask=None):
    """RMSD of F - F_ref over mask after subtracting the mean difference (the best constant shift);
    returns (rmsd, max |diff|, shifted F)."""
    F, F_ref = np.asarray(F, float), np.asarray(F_ref, float)
    m = (
        np.isfinite(F) & np.isfinite(F_ref)
        if mask is None
        else (np.asarray(mask) & np.isfinite(F) & np.isfinite(F_ref))
    )
    shift = np.mean((F - F_ref)[m])
    D = (F - shift - F_ref)[m]
    return float(np.sqrt(np.mean(D * D))), float(np.max(np.abs(D))), F - shift


def wham(samples, centers, kappas, axis, kT: float, period: float = 0.0, tol: float = 1e-10, max_iter: int = 100000):
    """1D WHAM of umbrella windows with biases kappa_k / 2 (s - c_k)^2 (Harmonic; differences
    wrapped for a periodic CV).  samples: list of 1D arrays (one per window); axis: bin centres
    (uniform).  Returns (F on the axis, min 0; inf where no samples, f_k window free energies)."""
    axis = np.asarray(axis, float)
    da = axis[1] - axis[0]
    lo = axis[0] - 0.5 * da
    K = len(samples)
    counts = np.zeros((K, len(axis)))
    N = np.zeros(K)
    for k, x in enumerate(samples):
        x = np.asarray(x, float)
        if period > 0:
            x = lo + np.mod(x - lo, period)
        i = np.floor((x - lo) / da).astype(int)
        i = i[(i >= 0) & (i < len(axis))]
        np.add.at(counts[k], i, 1.0)
        N[k] = len(i)
    d = axis[None, :] - np.asarray(centers, float)[:, None]
    if period > 0:
        d = d - period * np.round(d / period)
    bk = 0.5 * np.asarray(kappas, float)[:, None] * d * d / kT  # (K, bins) reduced biases
    num = counts.sum(0)
    f = np.zeros(K)
    for _ in range(int(max_iter)):
        den = np.sum(N[:, None] * np.exp(f[:, None] - bk), 0)
        with np.errstate(divide="ignore"):
            p = np.where(num > 0, num / den, 0.0)
        fn = -np.log(np.sum(p[None, :] * np.exp(-bk), 1))
        fn -= fn[0]
        if np.max(np.abs(fn - f)) < tol:
            f = fn
            break
        f = fn
    with np.errstate(divide="ignore"):
        F = -kT * np.log(p)
    return F - F[np.isfinite(F)].min(), f * kT
