"""Static dielectric constant, its statistical error and the infrared absorption from a cell-dipole
time series (md/dipoles.py: DipoleRecorder writes it, read_dipoles reads it; scripts/dielectric.py).

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
Kinetic properties need physical masses (no hydrogen mass repartitioning)."""
from __future__ import annotations

import numpy as np

E_CHARGE = 1.602176634e-19        # C
EPS0 = 8.8541878128e-12           # F/m
KB_SI = 1.380649e-23              # J/K
C_LIGHT = 2.99792458e8            # m/s
E_NM = E_CHARGE * 1e-9            # 1 e nm in C m


def _series(M, V):
    M = np.asarray(M, float).reshape(-1, 3)
    V = np.broadcast_to(np.asarray(V, float), (len(M),))
    return M, V


def fluctuation(M, V, T: float) -> float:
    """(<M.M> - <M>.<M>) / (3 eps0 <V> kB T) for M (F, 3) in e nm, V in nm^3 (scalar or (F,)), T in K."""
    M, V = _series(M, V)
    dM2 = np.mean(np.sum(M * M, axis=1)) - np.sum(np.mean(M, axis=0) ** 2)
    return float(dM2 * E_NM ** 2 / (3.0 * EPS0 * np.mean(V) * 1e-27 * KB_SI * T))


def eps_inf(alpha, V) -> tuple[float, float]:
    """eps_inf = 1 + 4 pi <alpha_cell / V> from the samples where alpha_cell (nm^3) was evaluated
    (nan elsewhere); returns (eps_inf, standard error of the mean assuming uncorrelated samples)."""
    a, V = np.asarray(alpha, float), np.broadcast_to(np.asarray(V, float), np.shape(alpha))
    ok = np.isfinite(a)
    if not ok.any():
        raise ValueError("no polarizability samples: pass eps_inf explicitly")
    x = 4.0 * np.pi * a[ok] / V[ok]
    return float(1.0 + x.mean()), float(x.std(ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan")


def jackknife(M, V, T: float, nblocks: int = 10) -> tuple[float, float]:
    """Fluctuation term and its jackknife standard error over `nblocks` contiguous blocks."""
    M, V = _series(M, V)
    if nblocks < 2 or len(M) < nblocks:
        raise ValueError("need at least two blocks with one sample each")
    parts = np.array_split(np.arange(len(M)), nblocks)
    n = np.array([len(p) for p in parts], float)
    s1 = np.array([M[p].sum(0) for p in parts])
    s2 = np.array([np.sum(M[p] * M[p]) for p in parts])
    sv = np.array([V[p].sum() for p in parts])
    conv = E_NM ** 2 / (3.0 * EPS0 * 1e-27 * KB_SI * T)

    def f(nn, a1, a2, av):
        nn = np.asarray(nn, float)
        return (a2 / nn - np.sum((a1 / nn[..., None]) ** 2, axis=-1)) * conv / (av / nn)

    full = f(n.sum(), s1.sum(0), s2.sum(), sv.sum())
    loo = f(n.sum() - n, s1.sum(0) - s1, s2.sum() - s2, sv.sum() - sv)
    err = np.sqrt((nblocks - 1) / nblocks * np.sum((loo - loo.mean()) ** 2))
    return float(full), float(err)


def static_dielectric(M, V, T: float, alpha=None, eps_inf_value: float | None = None, nblocks: int = 10) -> dict:
    """eps with its jackknife error.  eps_inf from the alpha_cell samples, or given (eps_inf_value;
    1.0 for a model without induced dipoles)."""
    M, V = _series(M, V)
    if eps_inf_value is None:
        if alpha is None:
            raise ValueError("pass the alpha_cell samples or eps_inf_value")
        ei, ei_err = eps_inf(alpha, V)
    else:
        ei, ei_err = float(eps_inf_value), 0.0
    fl, err = jackknife(M, V, T, nblocks)
    return {"eps": ei + fl, "err": float(np.hypot(err, ei_err)), "fluct": fl, "fluct_err": err, "eps_inf": ei,
            "eps_inf_err": ei_err, "n": len(M), "mean_M": M.mean(0), "rms_M": float(np.sqrt(np.mean(np.sum(M * M, 1)))),
            "V": float(np.mean(V))}


def block_errors(M, V, T: float, blocks=(4, 5, 8, 10, 16, 20, 32, 50)) -> list[tuple[int, float]]:
    """(number of blocks, jackknife error of the fluctuation term) for several block counts."""
    return [(b, jackknife(M, V, T, b)[1]) for b in blocks if len(M) >= 2 * b]


def running(M, V, T: float, fractions=(0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0), nblocks: int = 10):
    """Fluctuation term and error from the first fraction f of the series: [(f, value, error)]."""
    M, V = _series(M, V)
    out = []
    for f in fractions:
        k = int(round(f * len(M)))
        if k >= 2 * nblocks:
            out.append((f,) + jackknife(M[:k], V[:k], T, nblocks))
    return out


def decomposition(parts: dict, V, T: float) -> dict:
    """Split the fluctuation term of M = sum of `parts` (name -> (F, 3)) into the variance terms
    <dA.dA> and the cross terms 2 <dA.dB>, in units of eps."""
    names = list(parts)
    d = {k: np.asarray(v, float) - np.mean(v, axis=0) for k, v in parts.items()}
    conv = E_NM ** 2 / (3.0 * EPS0 * float(np.mean(V)) * 1e-27 * KB_SI * T)
    out = {}
    for i, a in enumerate(names):
        out[a] = float(np.mean(np.sum(d[a] * d[a], 1)) * conv)
        for b in names[i + 1:]:
            out[f"{a}x{b}"] = float(2.0 * np.mean(np.sum(d[a] * d[b], 1)) * conv)
    return out


def autocorrelation(M, max_lag: int | None = None) -> np.ndarray:
    """Normalised autocorrelation <dM(0).dM(t)> / <dM.dM> for lags 0..max_lag (FFT, unbiased)."""
    M = np.asarray(M, float).reshape(-1, 3)
    d = M - M.mean(0)
    n = len(d)
    max_lag = n - 1 if max_lag is None else min(int(max_lag), n - 1)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(d, nfft, axis=0)
    c = np.fft.irfft(np.sum(np.abs(F) ** 2, axis=1), nfft)[:max_lag + 1] / (n - np.arange(max_lag + 1))
    return c / c[0]


def correlation_time(M, dt_ps: float, window=(0.8, 0.2)) -> float:
    """tau_M (ps) from a fit of ln C(t) over the window where C falls from window[0] to window[1]
    (Debye-like decay of the collective dipole); nan if C never falls below window[1]."""
    c = autocorrelation(M)
    below = np.nonzero(c < window[1])[0]
    if len(below) == 0:
        return float("nan")
    k1 = below[0]
    sel = np.nonzero((c[:k1] <= window[0]) & (c[:k1] > 0))[0]
    if len(sel) < 3:
        sel = np.arange(1, max(k1, 3))
    t = sel * dt_ps
    return float(-1.0 / np.polyfit(t, np.log(c[sel]), 1)[0])


def ir_spectrum(M, dt_ps: float, V, T: float, segment_ps: float = 10.0):
    """Infrared absorption alpha(w) n(w) (cm^-1) against wavenumber (cm^-1) from M(t) (e nm)
    sampled every dt_ps; Welch periodograms over segments of segment_ps (resolution
    ~ 1 / (c segment) = 3.3 cm^-1 for 10 ps), Hann windows, half overlap."""
    M = np.asarray(M, float).reshape(-1, 3)
    n = min(len(M), max(8, int(round(segment_ps / dt_ps))))
    h = np.hanning(n)
    dt = dt_ps * 1e-12
    starts = range(0, len(M) - n + 1, max(1, n // 2))
    d = M - M.mean(0)
    S = np.zeros(n // 2 + 1)
    for s in starts:
        seg = d[s:s + n] - d[s:s + n].mean(0)
        S += np.sum(np.abs(np.fft.rfft(seg * h[:, None], axis=0)) ** 2, axis=1)
    S *= dt / np.sum(h * h) / len(starts) * E_NM ** 2               # C(w), C^2 m^2 s
    w = 2.0 * np.pi * np.fft.rfftfreq(n, dt)                          # rad/s
    V_m3 = float(np.mean(V)) * 1e-27
    alpha_n = w ** 2 * S / (6.0 * C_LIGHT * EPS0 * V_m3 * KB_SI * T)   # 1/m
    return w / (2.0 * np.pi * C_LIGHT) / 100.0, alpha_n / 100.0
