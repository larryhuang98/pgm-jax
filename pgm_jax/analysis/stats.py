"""Statistics of time series shared by the analysis code (host side, numpy).

block_mean(x, nblocks)             mean and its standard error from contiguous block means
jackknife_error(replicates)        standard error from delete-one(-block) jackknife replicates
jackknife_cov(replicates)          jackknife covariance of flattened replicates
integrated_correlation_time(x, dt) sum of the normalised autocorrelation up to its first zero
statistical_inefficiency(a, b)     g = 1 + 2 sum_t (1 - t/N) C(t), summed until C(t) <= 0 (after
                                   mintime lags); Chodera, Swope, Pitera, Seok & Dill, JCTC 3, 26
                                   (2007), the algorithm of pymbar.timeseries
detect_equilibration(a)            the start t0 that maximises the number of uncorrelated samples
                                   (N - t0) / g(t0) (Chodera, JCTC 12, 1799 (2016))
subsample(n, g)                    indices of roughly uncorrelated samples (every ceil(g)-th)
"""

from __future__ import annotations

import math

import numpy as np


def block_mean(x, nblocks: int = 10):
    """Mean and its standard error from `nblocks` contiguous block means."""
    x = np.asarray(x, float)
    n = len(x) // nblocks * nblocks
    b = x[len(x) - n :].reshape(nblocks, -1).mean(1)
    return float(x.mean()), float(b.std(ddof=1) / np.sqrt(nblocks))


def jackknife_error(rep) -> np.ndarray:
    """Standard error from delete-one-block jackknife replicates rep (B, ...)."""
    r = np.asarray(rep, float)
    B = r.shape[0]
    return np.sqrt((B - 1) / B * np.sum((r - r.mean(0)) ** 2, axis=0))


def jackknife_cov(values):
    """Jackknife covariance of the leave-one-out estimates values (B, ...) flattened: (m, m)."""
    x = np.asarray(values, float).reshape(len(values), -1)
    B = len(x)
    d = x - x.mean(0)
    return (B - 1) / B * d.T @ d


def integrated_correlation_time(x, dt: float) -> float:
    """Integrated autocorrelation time (ps) of a series sampled every dt ps (sum of the normalised
    autocorrelation up to its first zero crossing)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    c = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
    c = c / c[0]
    stop = np.argmax(c <= 0) if np.any(c <= 0) else n
    return float(dt * (0.5 + np.sum(c[1:stop])))


def statistical_inefficiency(a, b=None, mintime: int = 3) -> float:
    """g >= 1 of the time series a (and the cross-series b): the number of steps per uncorrelated
    sample.  Autocorrelation by FFT; the sum stops at the first non-positive C(t) after mintime lags."""
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


def detect_equilibration(a, nskip: int = 1) -> tuple:
    """(t0, g, N_eff): the first sample t0 of the production part (every nskip-th start tried),
    chosen to maximise N_eff = (N - t0) / g(a[t0:])."""
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
    """Indices 0, ceil(g), 2 ceil(g), ... below n: roughly uncorrelated samples."""
    return np.arange(0, int(n), max(1, int(math.ceil(g - 1e-12))))
