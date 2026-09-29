"""Compute the static dielectric constant, its error and the infrared absorption from a cell dipole.

The cell-dipole time series is written by md/dipoles.py DipoleRecorder and read by read_dipoles;
scripts/dielectric.py drives this module.  Contents: fluctuation, eps_inf, jackknife,
static_dielectric, block_errors, running, decomposition (the static dielectric constant),
autocorrelation, correlation_time (the dipole's dynamics), ir_spectrum (infrared).

Tin-foil (conducting) Ewald boundary conditions and adiabatic induced dipoles (dipoles.py derives
the formula):

    eps = eps_inf + (<M.M> - <M>.<M>) / (3 eps0 <V> kB T),      eps_inf = 1 + 4 pi <alpha_cell / V>,

M the total cell dipole (charges, permanent and induced dipoles), alpha_cell the cell's electronic
polarizability (1/3 trace, nm^3).  SI throughout: M in C m (1 e nm = 1.602176634e-28 C m), V in m^3,
eps0 = 8.8541878128e-12 F/m, kB = 1.380649e-23 J/K.  In the model's units the fluctuation term is
4 pi KE (<M.M> - <M>.<M>) / (3 V kB T), KE = 1/(4 pi eps0) = 138.935458 kJ mol^-1 nm e^-2 (the same
up to the rounding of KE).  Under NPT <V> is the mean volume (the V-M^2 correlation is far below the
statistical error for a liquid).  T is the thermostat's target: the kinetic temperature estimator
reads a few K low at 2-5 fs (integrator discretization) while the configurations are canonical at T.

Statistical error.  M is a slow collective variable (its correlation time tau_M is ~10 ps for water),
so the samples are correlated.  The series is cut into B contiguous blocks, and the jackknife over
blocks (leave one block out, recompute eps with the full estimator including its <M>^2 term) gives
the standard error; it is valid once the blocks are much longer than tau_M, i.e. when it no longer
grows as the blocks get longer (block_errors lists it for several B).  For a Gaussian M with an
exponential correlation the relative error of eps - eps_inf is sqrt(2 tau_M / (3 T_run)), e.g.
1.3 % for tau_M = 10 ps and a 40 ns run, independent of the system size.

Infrared.  With the harmonic quantum correction, the absorption coefficient times the refractive
index is the classical expression

    alpha(w) n(w) = beta w^2 / (6 c eps0 V) C(w),     C(w) = int dt e^{-iwt} <dM(0).dM(t)>,
C(w) estimated by Welch-averaged periodograms of M(t), dt |sum_n h_n dM_n e^{-iw n dt}|^2 / sum_n h_n^2
(Hann windows h, half overlap).  The sampling interval must resolve the motion (<= 4 fs up to
4000 cm^-1); rigid water has only the translational and librational bands (below ~1200 cm^-1).
Kinetic properties need physical masses (no hydrogen mass repartitioning).

Units: inputs in library units (M e nm, V nm^3, T K, dt ps); SI inside; spectra in cm^-1.

See also docs/dielectric.md.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from ..units import C_LIGHT_M_S, E_NM_C_M, EPS0_SI, KB_SI
from .stats import jackknife_error


def _series(M: ArrayLike, V: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return M as (F, 3) and V broadcast to (F,) float arrays."""
    M = np.asarray(M, float).reshape(-1, 3)
    V = np.broadcast_to(np.asarray(V, float), (len(M),))
    return M, V


def fluctuation(M: ArrayLike, V: ArrayLike, temperature: float) -> float:
    """Return the fluctuation term of the static dielectric constant (tin-foil Ewald).

    Parameters
    ----------
    M : ArrayLike (F, 3)
        Cell dipole series [e nm].
    V : float or ArrayLike (F,)
        Volume [nm^3].
    temperature : float
        Temperature [K] (the thermostat's target).

    Returns
    -------
    float
        (<M.M> - <M>.<M>) / (3 eps0 <V> kB T) (dimensionless, SI constants).
    """
    M, V = _series(M, V)
    dM2 = np.mean(np.sum(M * M, axis=1)) - np.sum(np.mean(M, axis=0) ** 2)
    return float(dM2 * E_NM_C_M**2 / (3.0 * EPS0_SI * np.mean(V) * 1e-27 * KB_SI * temperature))


def eps_inf(alpha: ArrayLike, V: ArrayLike) -> tuple[float, float]:
    """Return eps_inf = 1 + 4 pi <alpha_cell / V> and its standard error.

    Parameters
    ----------
    alpha : ArrayLike (F,)
        Cell polarizability samples [nm^3], nan where not evaluated.
    V : float or ArrayLike (F,)
        Volume [nm^3].

    Returns
    -------
    eps_inf, err : float
        The mean over the finite samples and its standard error assuming uncorrelated samples (nan
        for one sample).

    Raises
    ------
    ValueError
        If no sample is finite.
    """
    a, V = np.asarray(alpha, float), np.broadcast_to(np.asarray(V, float), np.shape(alpha))
    ok = np.isfinite(a)
    if not ok.any():
        raise ValueError("no polarizability samples: pass eps_inf explicitly")
    x = 4.0 * np.pi * a[ok] / V[ok]
    return float(1.0 + x.mean()), float(x.std(ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan")


def jackknife(M: ArrayLike, V: ArrayLike, temperature: float, nblocks: int = 10) -> tuple[float, float]:
    """Return the fluctuation term (see `fluctuation`) and its delete-one-block jackknife error.

    Parameters
    ----------
    M : ArrayLike (F, 3)
        Cell dipole series [e nm].
    V : float or ArrayLike (F,)
        Volume [nm^3].
    temperature : float
        Temperature [K] (the thermostat's target).
    nblocks : int
        Contiguous blocks of the jackknife (np.array_split: all samples used).

    Returns
    -------
    value, err : float
        The fluctuation term (dimensionless) and its standard error.

    Raises
    ------
    ValueError
        Fewer than two blocks or fewer samples than blocks.

    Notes
    -----
    Each replicate recomputes the full estimator, <M>^2 term and mean volume included, from block
    sums (O(F) total).
    """
    M, V = _series(M, V)
    if nblocks < 2 or len(M) < nblocks:
        raise ValueError("need at least two blocks with one sample each")
    parts = np.array_split(np.arange(len(M)), nblocks)
    n = np.array([len(p) for p in parts], float)
    s1 = np.array([M[p].sum(0) for p in parts])
    s2 = np.array([np.sum(M[p] * M[p]) for p in parts])
    sv = np.array([V[p].sum() for p in parts])
    conv = E_NM_C_M**2 / (3.0 * EPS0_SI * 1e-27 * KB_SI * temperature)

    def f(nn: ArrayLike, a1: np.ndarray, a2: ArrayLike, av: ArrayLike) -> np.ndarray:
        """Return the fluctuation term from sample counts and sums of M, |M|^2 and V (vectorised over replicates)."""
        nn = np.asarray(nn, float)
        return (a2 / nn - np.sum((a1 / nn[..., None]) ** 2, axis=-1)) * conv / (av / nn)

    full = f(n.sum(), s1.sum(0), s2.sum(), sv.sum())
    loo = f(n.sum() - n, s1.sum(0) - s1, s2.sum() - s2, sv.sum() - sv)
    err = jackknife_error(loo)
    return float(full), float(err)


def static_dielectric(
    M: ArrayLike,
    V: ArrayLike,
    temperature: float,
    alpha: ArrayLike | None = None,
    eps_inf_value: float | None = None,
    nblocks: int = 10,
) -> dict:
    """Return the static dielectric constant eps = eps_inf + fluctuation term, with its jackknife error.

    Parameters
    ----------
    M : ArrayLike (F, 3)
        Cell dipole series [e nm].
    V : float or ArrayLike (F,)
        Volume [nm^3].
    temperature : float
        Temperature [K] (the thermostat's target).
    alpha : ArrayLike (F,), optional
        Cell polarizability samples [nm^3] (nan where not evaluated), for eps_inf.
    eps_inf_value : float, optional
        eps_inf given directly (1.0 for a model without induced dipoles); takes precedence over
        `alpha`.
    nblocks : int
        Contiguous blocks of the jackknife.

    Returns
    -------
    dict
        eps, err (fluctuation and eps_inf errors in quadrature), fluct, fluct_err, eps_inf,
        eps_inf_err, n, mean_M (3,) [e nm], rms_M [e nm], V (mean) [nm^3].

    Raises
    ------
    ValueError
        Neither alpha nor eps_inf_value.
    """
    M, V = _series(M, V)
    if eps_inf_value is None:
        if alpha is None:
            raise ValueError("pass the alpha_cell samples or eps_inf_value")
        ei, ei_err = eps_inf(alpha, V)
    else:
        ei, ei_err = float(eps_inf_value), 0.0
    fl, err = jackknife(M, V, temperature, nblocks)
    return {
        "eps": ei + fl,
        "err": float(np.hypot(err, ei_err)),
        "fluct": fl,
        "fluct_err": err,
        "eps_inf": ei,
        "eps_inf_err": ei_err,
        "n": len(M),
        "mean_M": M.mean(0),
        "rms_M": float(np.sqrt(np.mean(np.sum(M * M, 1)))),
        "V": float(np.mean(V)),
    }


def block_errors(
    M: ArrayLike, V: ArrayLike, temperature: float, blocks: Iterable[int] = (4, 5, 8, 10, 16, 20, 32, 50)
) -> list[tuple[int, float]]:
    """Return jackknife errors of the fluctuation term for several block counts (a plateau check).

    Parameters
    ----------
    M : ArrayLike (F, 3)
        Cell dipole series [e nm].
    V : float or ArrayLike (F,)
        Volume [nm^3].
    temperature : float
        Temperature [K] (the thermostat's target).
    blocks : iterable of int
        Block counts (those with at least two samples per block are used).

    Returns
    -------
    list of (int, float)
        (number of blocks, standard error).
    """
    return [(b, jackknife(M, V, temperature, b)[1]) for b in blocks if len(M) >= 2 * b]


def running(
    M: ArrayLike,
    V: ArrayLike,
    temperature: float,
    fractions: Iterable[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
    nblocks: int = 10,
) -> list[tuple[float, float, float]]:
    """Return the fluctuation term and its error from growing leading parts of the series (convergence).

    Parameters
    ----------
    M : ArrayLike (F, 3)
        Cell dipole series [e nm].
    V : float or ArrayLike (F,)
        Volume [nm^3].
    temperature : float
        Temperature [K] (the thermostat's target).
    fractions : iterable of float
        Fractions of the series (those with at least 2 nblocks samples are used).
    nblocks : int
        Contiguous blocks of the jackknife.

    Returns
    -------
    list of (float, float, float)
        (fraction, value, standard error).
    """
    M, V = _series(M, V)
    out = []
    for f in fractions:
        k = int(round(f * len(M)))
        if k >= 2 * nblocks:
            out.append((f,) + jackknife(M[:k], V[:k], temperature, nblocks))
    return out


def decomposition(parts: dict, V: ArrayLike, temperature: float) -> dict[str, float]:
    """Return the fluctuation term of M = sum of parts split into variance and cross terms.

    Parameters
    ----------
    parts : dict
        name -> dipole series (F, 3) [e nm], e.g. charges, permanent and induced dipoles.
    V : float or ArrayLike (F,)
        Volume [nm^3].
    temperature : float
        Temperature [K].

    Returns
    -------
    dict of str to float
        The variance terms <dA.dA> (key "A") and cross terms 2 <dA.dB> (key "AxB"), in units of
        eps; they sum to the fluctuation term of the total.
    """
    names = list(parts)
    d = {k: np.asarray(v, float) - np.mean(v, axis=0) for k, v in parts.items()}
    conv = E_NM_C_M**2 / (3.0 * EPS0_SI * float(np.mean(V)) * 1e-27 * KB_SI * temperature)
    out = {}
    for i, a in enumerate(names):
        out[a] = float(np.mean(np.sum(d[a] * d[a], 1)) * conv)
        for b in names[i + 1 :]:
            out[f"{a}x{b}"] = float(2.0 * np.mean(np.sum(d[a] * d[b], 1)) * conv)
    return out


def autocorrelation(M: ArrayLike, max_lag: int | None = None) -> np.ndarray:
    """Return the normalised autocorrelation <dM(0).dM(t)> / <dM.dM> for lags 0 ... max_lag.

    `M` (F, 3) [e nm]; `max_lag` None: F - 1.  FFT with zero padding, unbiased normalisation
    1/(F - t).  Returns (max_lag + 1,), dimensionless.
    """
    M = np.asarray(M, float).reshape(-1, 3)
    d = M - M.mean(0)
    n = len(d)
    max_lag = n - 1 if max_lag is None else min(int(max_lag), n - 1)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(d, nfft, axis=0)
    c = np.fft.irfft(np.sum(np.abs(F) ** 2, axis=1), nfft)[: max_lag + 1] / (n - np.arange(max_lag + 1))
    return c / c[0]


def correlation_time(M: ArrayLike, dt: float, window: tuple[float, float] = (0.8, 0.2)) -> float:
    """Return the correlation time of the cell dipole from a fit of ln C(t) (Debye-like decay).

    Parameters
    ----------
    M : ArrayLike (F, 3)
        Cell dipole series [e nm].
    dt : float
        Time between samples [ps].
    window : (float, float)
        The fit uses the lags where C(t) is between window[0] and window[1] (before C first drops
        below window[1]); lags 1 ... max(k1, 3) - 1 if fewer than three qualify.

    Returns
    -------
    float
        tau_M [ps] = -1 / slope of ln C(t); nan if C never falls below window[1].
    """
    c = autocorrelation(M)
    below = np.nonzero(c < window[1])[0]
    if len(below) == 0:
        return float("nan")
    k1 = below[0]
    sel = np.nonzero((c[:k1] <= window[0]) & (c[:k1] > 0))[0]
    if len(sel) < 3:
        sel = np.arange(1, max(k1, 3))
    t = sel * dt
    return float(-1.0 / np.polyfit(t, np.log(c[sel]), 1)[0])


def ir_spectrum(
    M: ArrayLike, dt: float, V: ArrayLike, temperature: float, segment_ps: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the infrared absorption spectrum from the cell dipole (linear response, classical).

    Parameters
    ----------
    M : ArrayLike (F, 3)
        Cell dipole series [e nm].
    dt : float
        Time between samples [ps].
    V : float or ArrayLike (F,)
        Volume [nm^3].
    temperature : float
        Temperature [K].
    segment_ps : float
        Length of the Welch segments [ps] (resolution about 1 / (c segment) = 3.3 cm^-1 for
        10 ps); Hann windows, half overlap.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Wavenumbers [cm^-1] and alpha(w) n(w) [cm^-1].

    Notes
    -----
    alpha(w) n(w) = w^2 C(w) / (6 c eps0 V kB T) (module docstring), with C(w) the Welch estimate;
    each segment's mean is removed.
    """
    M = np.asarray(M, float).reshape(-1, 3)
    n = min(len(M), max(8, int(round(segment_ps / dt))))
    h = np.hanning(n)
    dt_s = dt * 1e-12
    starts = range(0, len(M) - n + 1, max(1, n // 2))
    d = M - M.mean(0)
    S = np.zeros(n // 2 + 1)
    for s in starts:
        seg = d[s : s + n] - d[s : s + n].mean(0)
        S += np.sum(np.abs(np.fft.rfft(seg * h[:, None], axis=0)) ** 2, axis=1)
    S *= dt_s / np.sum(h * h) / len(starts) * E_NM_C_M**2  # C(w), C^2 m^2 s
    w = 2.0 * np.pi * np.fft.rfftfreq(n, dt_s)  # rad/s
    V_m3 = float(np.mean(V)) * 1e-27
    alpha_n = w**2 * S / (6.0 * C_LIGHT_M_S * EPS0_SI * V_m3 * KB_SI * temperature)  # 1/m
    return w / (2.0 * np.pi * C_LIGHT_M_S) / 100.0, alpha_n / 100.0
