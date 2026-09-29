"""Compute free energy surfaces from biased simulations (host side, numpy / JAX).

Contents:

    fes_from_bias      F(s) = -f V(s) on a grid: f = gamma / (gamma - 1) (well-tempered metaD),
                       1 / (1 - 1/gamma) (OPES), 1 (standard metaD)
    metad_ct           c(t) of well-tempered metadynamics [1]_ after every hill,
                           c(t) = kT log[ int ds exp(gamma/(gamma-1) V(s,t)/kT) / int ds exp(V(s,t)/((gamma-1) kT)) ]
    ct_weights         frame log weights (V(s(t), t) - c(t)) / kT (metaD) ...
    opes_weights       ... and V(s(t), t) / kT (OPES; any quasi-static bias)
    histogram_fes      F = -kT log(weighted histogram) on a grid (any CVs, e.g. ones that were not biased)
    wham               1D weighted histogram analysis of umbrella windows (reference free energies)
    align_rmsd         RMSD between two FES over a region after the best constant shift
    mesh, periodic_axis, bias_on_grid   grids of CV values and the bias on them

Weights are returned as logarithms (log w, dimensionless) so that large biases do not
overflow; histogram_fes shifts them before exponentiating.  Free energies are defined up to a
constant and returned with minimum 0.

    steps, ct = metad_ct(metad.hills(state), biasfactor, kT, periods, points)
    logw = ct_weights(colvar["step"], colvar["bias0_metad"], steps, ct, kT)
    F = histogram_fes(np.stack([colvar["phi"], colvar["psi"]], 1), logw, [axis, axis], kT, periods)

Units: energies and kT kJ/mol; CV values in the CVs' units (nm, rad); steps are MD steps.

References
----------
.. [1] P. Tiwary, M. Parrinello, J. Phys. Chem. B 119, 736 (2015). doi:10.1021/jp504920s

See also docs/enhanced_sampling.md; bias/core.py (the biases), bias/io.py (reading COLVAR and
HILLS files).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .core import Bias


def mesh(*axes: ArrayLike) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return the grid points of the product of 1D axes and the grid shape.

    Parameters
    ----------
    *axes : ArrayLike
        One 1D array of values per CV [CV units].

    Returns
    -------
    points : np.ndarray (G, d)
        Grid points in C order (last axis fastest).
    shape : tuple of int
        Grid shape (len of each axis).
    """
    g = np.meshgrid(*[np.asarray(a, float) for a in axes], indexing="ij")
    return np.stack([x.reshape(-1) for x in g], 1), g[0].shape


def periodic_axis(n: int, lo: float = -np.pi, period: float = 2 * np.pi) -> np.ndarray:
    """Return `n` bin centres covering one period [lo, lo + period) [CV units].

    Examples
    --------
    >>> periodic_axis(4, lo=0.0, period=4.0)
    array([0.5, 1.5, 2.5, 3.5])
    """
    return lo + (np.arange(n) + 0.5) * period / n


def bias_on_grid(bias: Bias, state: Any, points: ArrayLike, chunk: int = 4096) -> np.ndarray:
    """Return V(s) of one bias at grid points, np.ndarray (G,) [kJ/mol].

    Parameters
    ----------
    bias : Bias
        A bias of core.py (its temperature must be known for OPES).
    state : pytree
        The bias state.
    points : ArrayLike (G, d)
        CV values [CV units].
    chunk : int
        Points per jitted, vmapped call (bounds the memory).

    Notes
    -----
    Compiles `bias.potential` once per chunk shape (the last, shorter chunk recompiles).
    """
    f = jax.jit(jax.vmap(lambda s: bias.potential(state, s)))
    P = np.asarray(points, float).reshape(-1, bias.d)
    return np.concatenate([np.asarray(f(jnp.asarray(P[i : i + chunk]))) for i in range(0, len(P), chunk)])


def fes_from_bias(bias: Bias, state: Any, points: ArrayLike) -> np.ndarray:
    """Return the FES F = -factor V at grid points, with the bias's own `fes_factor` (min 0) [kJ/mol].

    Parameters
    ----------
    bias : Bias
        A MetaD or OPES bias (anything with `fes_factor`).
    state : pytree
        The bias state.
    points : ArrayLike (G, d)
        CV values [CV units].

    Returns
    -------
    np.ndarray (G,)
        Free energy [kJ/mol], minimum 0.
    """
    F = -bias.fes_factor() * bias_on_grid(bias, state, points)
    return F - F.min()


def _hill_values(
    centers: np.ndarray, heights: np.ndarray, sigma: ArrayLike, periods: ArrayLike, points: np.ndarray
) -> np.ndarray:
    """Return the value of every hill at every grid point, np.ndarray (n_hills, G) [kJ/mol].

    Parameters
    ----------
    centers : np.ndarray (n_hills, d)
        Hill centres [CV units].
    heights : np.ndarray (n_hills,)
        Hill heights [kJ/mol].
    sigma : ArrayLike (d,)
        Hill widths [CV units].
    periods : ArrayLike (d,)
        CV periods (0: not periodic); periodic differences are wrapped to the nearest image.
    points : np.ndarray (G, d)
        Grid points [CV units].
    """
    d = points[None, :, :] - centers[:, None, :]
    P = np.asarray(periods, float)
    p = np.where(P > 0, P, 1.0)
    d = np.where(P > 0, d - p * np.round(d / p), d) / np.asarray(sigma)
    return heights[:, None] * np.exp(-0.5 * np.sum(d * d, -1))


def metad_ct(
    hills: dict, biasfactor: float, kT: float, periods: ArrayLike, points: ArrayLike, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Return c(t) of well-tempered metadynamics after each hill, in deposition order [1]_.

    Parameters
    ----------
    hills : dict
        The hills: `MetaD.hills()` or the columns of prefix.hills, with keys "step" (n,),
        "center" (n, d) or (n,), "height" (n,) [kJ/mol], "sigma" (d,) [CV units].
    biasfactor : float
        gamma of the run (dimensionless).
    kT : float
        Thermal energy of the bias [kJ/mol].
    periods : ArrayLike (d,)
        CV periods (0: not periodic).
    points : ArrayLike (G, d)
        A uniform grid covering the explored CV space [CV units].
    chunk : int
        Hills processed at once (memory: chunk x G floats).

    Returns
    -------
    steps : np.ndarray (n,)
        Deposition steps of the hills.
    ct : np.ndarray (n,)
        c(t) after each hill [kJ/mol].

    Notes
    -----
        c(t) = kT log[ sum_s exp(gamma/(gamma-1) V(s,t)/kT) / sum_s exp(V(s,t)/((gamma-1) kT)) ]

    with the integrals as sums over the grid points (the volume element cancels in the ratio);
    both are evaluated by log-sum-exp.  V is rebuilt from the hills (exact hill sum, not the
    simulation's grid interpolant).

    References
    ----------
    .. [1] P. Tiwary, M. Parrinello, J. Phys. Chem. B 119, 736 (2015). doi:10.1021/jp504920s
    """
    g = float(biasfactor)
    C, h, sig = np.asarray(hills["center"], float), np.asarray(hills["height"], float), hills["sigma"]
    if C.ndim == 1:
        C = C[:, None]
    V = np.zeros(len(points))
    out = np.zeros(len(h))
    for i in range(0, len(h), chunk):
        H = _hill_values(C[i : i + chunk], h[i : i + chunk], sig, periods, points)
        Vc = V[None, :] + np.cumsum(H, 0)  # (chunk, G) bias after each hill of the chunk
        a, b = g / (g - 1.0) * Vc / kT, Vc / ((g - 1.0) * kT)  # exponents of the two integrands
        am, bm = a.max(1, keepdims=True), b.max(1, keepdims=True)  # log-sum-exp shifts
        out[i : i + chunk] = kT * (
            (am[:, 0] + np.log(np.exp(a - am).sum(1))) - (bm[:, 0] + np.log(np.exp(b - bm).sum(1)))
        )
        V = Vc[-1]
    return np.asarray(hills["step"]), out


def ct_weights(
    frame_steps: ArrayLike,
    frame_bias: ArrayLike,
    hill_steps: ArrayLike,
    ct: ArrayLike,
    kT: float,
    strict: bool = True,
) -> np.ndarray:
    """Return log weights (V(s(t), t) - c(t)) / kT of COLVAR frames of a metadynamics run.

    Parameters
    ----------
    frame_steps : ArrayLike (F,)
        Steps of the frames (COLVAR "step" column).
    frame_bias : ArrayLike (F,)
        Bias energy at each frame [kJ/mol] (COLVAR "bias{k}_metad" column).
    hill_steps : ArrayLike (n,)
        Deposition steps of the hills, increasing (`metad_ct`).
    ct : ArrayLike (n,)
        c(t) after each hill [kJ/mol].
    kT : float
        Thermal energy [kJ/mol].
    strict : bool
        Use the last hill deposited strictly before the frame (hill step < frame step), as the
        drivers record the bias before a same-step update; False: at or before.

    Returns
    -------
    np.ndarray (F,)
        Log weights (dimensionless); c = 0 before the first hill.
    """
    idx = np.searchsorted(np.asarray(hill_steps), np.asarray(frame_steps), side="left" if strict else "right") - 1
    c = np.where(idx >= 0, np.asarray(ct)[np.maximum(idx, 0)], 0.0)
    return (np.asarray(frame_bias) - c) / kT


def opes_weights(frame_bias: ArrayLike, kT: float) -> np.ndarray:
    """Return log weights V / kT of frames sampled with a (quasi-)static bias V such as OPES's.

    Parameters
    ----------
    frame_bias : ArrayLike (F,)
        Bias energy at each frame [kJ/mol].
    kT : float
        Thermal energy [kJ/mol].

    Returns
    -------
    np.ndarray (F,)
        Log weights (dimensionless).
    """
    return np.asarray(frame_bias) / kT


def histogram_fes(
    s: ArrayLike, logw: ArrayLike | None, axes: Sequence[ArrayLike], kT: float, periods: ArrayLike | None = None
) -> np.ndarray:
    """Return F = -kT log p on a grid of bin centres from weighted samples.

    Parameters
    ----------
    s : ArrayLike (n, d) or (n,)
        Samples [CV units].
    logw : ArrayLike (n,), optional
        Log weights (dimensionless); None: unweighted.
    axes : sequence of ArrayLike
        Bin centres per dimension (uniform spacing) [CV units].
    kT : float
        Thermal energy [kJ/mol].
    periods : ArrayLike (d,), optional
        Periods (> 0: samples wrapped into [first bin edge, + period)); None: none periodic.

    Returns
    -------
    np.ndarray (shape of the axes)
        Free energy [kJ/mol], minimum 0; inf in empty bins.  Samples outside the grid are
        dropped.
    """
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
    lw = lw - lw[ok].max()  # largest weight 1 (no overflow)
    hist = np.zeros(shape)
    np.add.at(hist, tuple(idx[ok].T), np.exp(lw[ok]))
    with np.errstate(divide="ignore"):
        F = -kT * np.log(hist)
    return F - F[np.isfinite(F)].min()


def align_rmsd(F: ArrayLike, F_ref: ArrayLike, mask: ArrayLike | None = None) -> tuple[float, float, np.ndarray]:
    """Return the RMSD of F - F_ref over a region after the best constant shift (the mean difference).

    Parameters
    ----------
    F, F_ref : ArrayLike
        Free energies on the same grid [kJ/mol].
    mask : ArrayLike (bool), optional
        Region to compare; None: everywhere.  Points where either is not finite are excluded.

    Returns
    -------
    rmsd : float
        Root-mean-square difference after the shift [kJ/mol].
    max_diff : float
        Largest absolute difference after the shift [kJ/mol].
    F_shifted : np.ndarray
        F minus the shift.
    """
    F, F_ref = np.asarray(F, float), np.asarray(F_ref, float)
    m = (
        np.isfinite(F) & np.isfinite(F_ref)
        if mask is None
        else (np.asarray(mask) & np.isfinite(F) & np.isfinite(F_ref))
    )
    shift = np.mean((F - F_ref)[m])
    D = (F - shift - F_ref)[m]
    return float(np.sqrt(np.mean(D * D))), float(np.max(np.abs(D))), F - shift


def wham(
    samples: Sequence[ArrayLike],
    centers: ArrayLike,
    kappas: ArrayLike,
    axis: ArrayLike,
    kT: float,
    period: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 100000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 1D WHAM free energy of umbrella windows and the window free energies.

    Parameters
    ----------
    samples : sequence of ArrayLike
        CV samples of each window (one 1D array per window) [CV units].
    centers : ArrayLike (K,)
        Window centres [CV units].
    kappas : ArrayLike (K,)
        Force constants [kJ/mol per CV unit^2]; the bias is kappa_k / 2 (s - c_k)^2 (`Harmonic`),
        differences wrapped for a periodic CV.
    axis : ArrayLike (B,)
        Bin centres (uniform spacing) [CV units].
    kT : float
        Thermal energy [kJ/mol].
    period : float
        Period of the CV [CV units] (0: not periodic).
    tol : float
        Convergence threshold on max |f_new - f_old| (in kT).
    max_iter : int
        Maximum number of iterations (no warning if it is reached).

    Returns
    -------
    F : np.ndarray (B,)
        Free energy on the axis [kJ/mol], minimum 0; inf where no window has samples.
    f : np.ndarray (K,)
        Window free energies [kJ/mol], f_0 = 0.

    Notes
    -----
    Self-consistent iteration of the WHAM equations on the histogram (b_kb = bias / kT of window k
    at bin b, n_kb counts, N_k samples inside the axis):

        p_b = sum_k n_kb / sum_k N_k exp(f_k - b_kb),   f_k = -log sum_b p_b exp(-b_kb)

    The bias is evaluated at the bin centres.
    """
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
        # WHAM equations: p_b = sum_k n_kb / sum_k N_k exp(f_k - b_kb),  exp(-f_k) = sum_b p_b exp(-b_kb)
        den = np.sum(N[:, None] * np.exp(f[:, None] - bk), 0)
        with np.errstate(divide="ignore"):
            p = np.where(num > 0, num / den, 0.0)
        fn = -np.log(np.sum(p[None, :] * np.exp(-bk), 1))
        fn -= fn[0]  # f_0 = 0 fixes the free constant
        if np.max(np.abs(fn - f)) < tol:
            f = fn
            break
        f = fn
    with np.errstate(divide="ignore"):
        F = -kT * np.log(p)
    return F - F[np.isfinite(F)].min(), f * kT
