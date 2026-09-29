"""Compute statistics of time series shared by the analysis code (host side, numpy).

Contents:

    block_mean(x, nblocks)             mean and its standard error from contiguous block means
    jackknife_error(replicates)        standard error from delete-one(-block) jackknife replicates
    jackknife_cov(replicates)          jackknife covariance of flattened replicates
    integrated_correlation_time(x, dt) sum of the normalised autocorrelation up to its first zero
    statistical_inefficiency(a, b)     g = 1 + 2 sum_t (1 - t/N) C(t), summed until C(t) <= 0 (after
                                       mintime lags); Chodera, Swope, Pitera, Seok & Dill [1]_,
                                       the algorithm of pymbar.timeseries
    detect_equilibration(a)            the start t0 that maximises the number of uncorrelated samples
                                       (N - t0) / g(t0) (Chodera [2]_)
    subsample(n, g)                    indices of roughly uncorrelated samples (every ceil(g)-th)

Jackknife convention: for B delete-one replicates x_b, var = (B - 1)/B sum_b (x_b - mean)^2.

Units: those of the series; times in ps where a function takes a time step.

References
----------
.. [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, K. A. Dill, J. Chem. Theory Comput. 3,
       26 (2007).
.. [2] J. D. Chodera, J. Chem. Theory Comput. 12, 1799 (2016).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def block_mean(x: ArrayLike, nblocks: int = 10) -> tuple[float, float]:
    """Return the mean of a series and its standard error from `nblocks` contiguous block means.

    Parameters
    ----------
    x : ArrayLike (F,)
        Series.
    nblocks : int
        Number of blocks; the first len(x) mod nblocks samples are left out of the blocks (not of
        the mean).

    Returns
    -------
    mean, sem : float
        Mean of all samples and std(block means, ddof=1) / sqrt(nblocks).
    """
    x = np.asarray(x, float)
    n = len(x) // nblocks * nblocks
    b = x[len(x) - n :].reshape(nblocks, -1).mean(1)
    return float(x.mean()), float(b.std(ddof=1) / np.sqrt(nblocks))


def jackknife_error(rep: ArrayLike) -> np.ndarray:
    """Return the standard error from delete-one-block jackknife replicates rep (B, ...), elementwise (...)."""
    r = np.asarray(rep, float)
    B = r.shape[0]
    return np.sqrt((B - 1) / B * np.sum((r - r.mean(0)) ** 2, axis=0))


def jackknife_cov(values: ArrayLike) -> np.ndarray:
    """Return the jackknife covariance (m, m) of leave-one-out estimates values (B, ...) flattened to (B, m)."""
    x = np.asarray(values, float).reshape(len(values), -1)
    B = len(x)
    d = x - x.mean(0)
    return (B - 1) / B * d.T @ d


def integrated_correlation_time(x: ArrayLike, dt: float) -> float:
    """Return the integrated autocorrelation time of a series sampled every dt [ps].

    tau = dt (1/2 + sum_{t >= 1} C(t)), with the normalised autocorrelation C (FFT, unbiased
    normalisation 1/(n - t)) summed up to its first non-positive value.

    Returns
    -------
    float
        tau [ps] (units of dt).
    """
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    c = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
    c = c / c[0]
    stop = np.argmax(c <= 0) if np.any(c <= 0) else n
    return float(dt * (0.5 + np.sum(c[1:stop])))


def statistical_inefficiency(a: ArrayLike, b: ArrayLike | None = None, mintime: int = 3) -> float:
    """Return the statistical inefficiency g >= 1 of a time series: the number of steps per uncorrelated sample.

    Parameters
    ----------
    a : ArrayLike (N,)
        Series.
    b : ArrayLike (N,), optional
        Second series for the cross-correlation (None: autocorrelation of `a`).
    mintime : int
        The sum of C(t) is not stopped at a non-positive value before lag `mintime`.

    Returns
    -------
    float
        g = 1 + 2 sum_t (1 - t/N) C(t), at least 1; 1 for fewer than two samples, a constant series
        or series of different lengths.

    Notes
    -----
    The correlation function is computed by FFT (zero-padded to a power of two), symmetrised in a
    and b; the algorithm of pymbar.timeseries (Chodera et al. 2007, module docstring).
    """
    A = np.asarray(a, float).ravel()
    B = A if b is None else np.asarray(b, float).ravel()
    N = len(A)
    if N < 2 or len(B) != N:
        return 1.0
    dA, dB = A - A.mean(), B - B.mean()
    s2 = float(np.mean(dA * dB))
    if s2 == 0.0:
        return 1.0
    n = 1 << int(math.ceil(math.log2(2 * N)))
    fa, fb = np.fft.rfft(dA, n), np.fft.rfft(dB, n)
    c = np.fft.irfft(fa.conj() * fb, n)[:N] + np.fft.irfft(fb.conj() * fa, n)[:N]  # sum_i dA_i dB_{i+t} + dB_i dA_{i+t}
    t = np.arange(1, N - 1)
    C = c[1 : N - 1] / (2.0 * (N - t) * s2)
    g = 1.0
    for tt, Ct in zip(t, C):
        if Ct <= 0.0 and tt > mintime:
            break
        g += 2.0 * Ct * (1.0 - tt / N)
    return max(g, 1.0)


def detect_equilibration(a: ArrayLike, nskip: int = 1) -> tuple[int, float, float]:
    """Return (t0, g, N_eff): the start of the production part that maximises N_eff = (N - t0) / g(a[t0:]).

        Every nskip-th start is tried (O(N^2 log N / nskip)); t0 is a sample index (Chodera 2016, module
    docstring).
    """
    A = np.asarray(a, float).ravel()
    N = len(A)
    best = (0, 1.0, 0.0)
    for t0 in range(0, max(N - 2, 1), max(int(nskip), 1)):
        g = statistical_inefficiency(A[t0:])
        neff = (N - t0) / g
        if neff > best[2]:
            best = (t0, g, neff)
    return best


def subsample(n: int, g: float) -> np.ndarray:
    """Return the indices 0, ceil(g), 2 ceil(g), ... below n: roughly uncorrelated samples."""
    return np.arange(0, int(n), max(1, int(math.ceil(g - 1e-12))))
