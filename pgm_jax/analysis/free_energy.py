"""Estimate free energies of lambda windows: TI, BAR and MBAR with uncertainties (numpy / scipy; no pymbar).

Statistical inefficiency and equilibration detection come from analysis/stats.py.

Conventions: reduced energies u = beta U (dimensionless); free energies f in units of kT unless a
function says kJ/mol.  u_kn[k, n] is the reduced energy of sample n in state k (samples of all
states pooled, N_k of them from state k, in order).  The samples of alchemy.FreeEnergyRun
(prefix_fe.npz) hold u (S, K, K) with u[s, k, n] the reduced energy in state k of sample s of
window n, dudl (S, K, d) the lambda-gradients of U [kJ/mol], lambdas (K, d), kT [kJ/mol] and
time_ps (S,).

Contents:

    bar(w_F, w_R)                    Bennett's acceptance ratio ([1]_, [2]_), variance from
                                     Bennett's Eq. 10a (pymbar's default "BAR")
    mbar(u_kn, N_k)                  the MBAR equations [3]_ solved by Newton's method on their
                                     convex objective; asymptotic covariance
                                     Theta = W^T (I - W N W^T)^+ W by the SVD route
    mbar_differences(f, Theta)       all pairwise differences and their errors
    ti(lambdas, means, sems)         trapezoid rule along the path of (lambda_elec, lambda_vdw) points
    estimate(samples, ...)           everything above for the samples of alchemy.FreeEnergyRun
                                     (prefix_fe.npz), with the gas-phase leg for a hydration free energy
    equilibration_times, load        a check of the discarded time; reading prefix_fe.npz

Uncertainties are one standard error of the uncorrelated (subsampled) estimates.

Units: kT (reduced) or kJ/mol as stated; "_kcal" keys in kcal/mol; times in ps.

References
----------
.. [1] C. H. Bennett, J. Comput. Phys. 22, 245 (1976).
.. [2] M. R. Shirts, E. Bair, G. Hooker, V. S. Pande, Phys. Rev. Lett. 91, 140601 (2003).
.. [3] M. R. Shirts, J. D. Chodera, J. Chem. Phys. 129, 124105 (2008).
.. [4] P. V. Klimovich, M. R. Shirts, D. L. Mobley, J. Comput.-Aided Mol. Des. 29, 397 (2015).

See also docs/free_energy.md.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

from ..units import KCAL
from .stats import detect_equilibration, statistical_inefficiency, subsample


# ----------------------------------------------------------------------------- BAR
def bar(w_F: ArrayLike, w_R: ArrayLike, tol: float = 1e-12) -> tuple[float, float]:
    """Return the BAR free-energy difference 0 -> 1 and its standard error [kT].

    Parameters
    ----------
    w_F : ArrayLike (nF,)
        Forward works u_1(x) - u_0(x) for samples x of state 0 (reduced, dimensionless).
    w_R : ArrayLike (nR,)
        Reverse works u_0(x) - u_1(x) for samples x of state 1.
    tol : float
        Absolute tolerance of the root [kT].

    Returns
    -------
    df, err : float
        Delta f = f_1 - f_0 [kT] and its standard error (Bennett's Eq. 10a).

    Raises
    ------
    ValueError
        If a state has no samples.
    RuntimeError
        If no bracket of the root is found.

    Notes
    -----
    Solves sum_F f(M + w_F - df) = sum_R f(-M + w_R + df), f the Fermi function, M = ln(nF/nR),
    by Brent's method in log space, bracketed around the two exponential-averaging estimates.
    """
    wF, wR = np.asarray(w_F, float).ravel(), np.asarray(w_R, float).ravel()
    nF, nR = len(wF), len(wR)
    if nF == 0 or nR == 0:
        raise ValueError("BAR needs samples from both states")
    M = math.log(nF / nR)

    def zero(df: float) -> float:
        """Return log sum_F f(M + w_F - df) - log sum_R f(-M + w_R + df), with f(x) = 1 / (1 + e^x).

        Increasing in df and zero at the BAR estimate (log space: no overflow for large works).
        """
        return logsumexp(-np.logaddexp(0.0, M + wF - df)) - logsumexp(-np.logaddexp(0.0, -M + wR + df))

    # bracket around the exponential-averaging estimates, widened until the sign changes
    lo = -(logsumexp(-wF) - math.log(nF))
    hi = logsumexp(-wR) - math.log(nR)
    lo, hi = min(lo, hi) - 1.0, max(lo, hi) + 1.0
    for _ in range(200):
        if zero(lo) < 0.0 < zero(hi):
            break
        lo, hi = lo - (hi - lo), hi + (hi - lo)
    else:
        raise RuntimeError("BAR: no bracket for the free-energy difference")
    df = brentq(zero, lo, hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=500)
    fF = expit(-(M + wF - df))
    fR = expit(-(-M + wR + df))
    var = np.mean(fF**2) / (nF * np.mean(fF) ** 2) + np.mean(fR**2) / (nR * np.mean(fR) ** 2) - (1.0 / nF + 1.0 / nR)
    return float(df), float(math.sqrt(max(var, 0.0)))


# ----------------------------------------------------------------------------- MBAR
def _mbar_weights(u_kn: np.ndarray, N_k: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the MBAR log weights (K, N) and the log denominators (N,).

    log W_kn = f_k - u_kn - log sum_l N_l exp(f_l - u_ln); infinite or NaN f_k - u_kn become -inf
    (zero weight).  Each row of W sums to 1 at the MBAR solution.
    """
    with np.errstate(invalid="ignore"):
        a = f[:, None] - u_kn
    a = np.where(np.isfinite(a), a, -np.inf)
    with np.errstate(divide="ignore"):
        logden = logsumexp(a + np.log(N_k)[:, None], axis=0)
    return a - logden[None, :], logden


def mbar(
    u_kn: ArrayLike, N_k: ArrayLike, tol: float = 1e-10, max_iter: int = 500, f0: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the MBAR free energies and their asymptotic covariance.

    Parameters
    ----------
    u_kn : ArrayLike (K, N)
        Reduced energies of every pooled sample in every state (+inf allowed: zero weight).
    N_k : ArrayLike (K,)
        Samples drawn from each state, in the order of the pooled samples (state 0 must have
        samples; states without samples get f from the converged weights).
    tol : float
        Convergence: max_k |dF/df_k| / N_k < tol over the sampled states.
    max_iter : int
        Largest number of iterations.
    f0 : ArrayLike (K,), optional
        Starting free energies [kT] (shifted to f_0 = 0); None: the start below.

    Returns
    -------
    f : np.ndarray (K,)
        Free energies [kT], f_0 = 0.
    Theta : np.ndarray (K, K)
        Asymptotic covariance of f (_mbar_covariance).

    Raises
    ------
    ValueError
        Inconsistent N_k, NaN energies, or no samples in state 0.
    RuntimeError
        If the iteration does not converge.

    Notes
    -----
    The MBAR equations are the stationarity conditions of the convex function F(f) = sum_n log
    sum_k N_k exp(f_k - u_kn) - sum_k N_k f_k.  Start: neighbouring sampled states joined by the
    mean of the forward and reverse first-order estimates; then each iteration takes the Newton
    step or the self-consistent update, whichever leaves the smaller gradient (pymbar's
    "adaptive" scheme, which needs no objective comparisons: F is a large sum and its changes near
    convergence are below its rounding).
    """
    u = np.asarray(u_kn, float)
    Nk = np.asarray(N_k, float)
    K, N = u.shape
    if Nk.shape != (K,) or int(round(Nk.sum())) != N:
        raise ValueError("N_k must have one entry per state summing to the number of samples")
    if np.any(np.isnan(u)):
        raise ValueError("reduced energies contain NaN")
    if Nk[0] <= 0:
        raise ValueError("state 0 (the reference) needs samples")
    samp = np.nonzero(Nk > 0)[0]
    act = samp[1:]  # optimised (f_0 = 0 fixed)
    start = np.concatenate([[0], np.cumsum(Nk).astype(int)])
    if f0 is None:
        f = np.zeros(K)
        for a, b in zip(samp[:-1], samp[1:]):
            xa, xb = u[:, start[a] : start[a + 1]], u[:, start[b] : start[b + 1]]
            fwd = np.mean(np.where(np.isfinite(xa[b] - xa[a]), xa[b] - xa[a], 0.0))
            rev = np.mean(np.where(np.isfinite(xb[a] - xb[b]), xb[a] - xb[b], 0.0))
            f[b] = f[a] + 0.5 * (fwd - rev)
    else:
        f = np.asarray(f0, float) - float(np.asarray(f0, float)[0])

    def grad(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lw, _ = _mbar_weights(u, Nk, f)
        W = np.exp(lw)
        return W, Nk * (W.sum(axis=1) - 1.0)

    def gnorm(g: np.ndarray) -> float:
        return float(np.max(np.abs(g[act] / Nk[act]))) if len(act) else 0.0

    def update(f: np.ndarray, W: np.ndarray) -> np.ndarray:  # self-consistent MBAR equation
        f = f - np.log(np.maximum(W.sum(axis=1), 1e-300))
        return f - f[0]

    W, g = grad(f)
    for _ in range(max_iter):
        if gnorm(g) < tol:
            break
        cand = [update(f, W)]
        WN = W[act] * Nk[act, None]
        Hs = -WN @ WN.T
        Hs[np.diag_indices(len(act))] += Nk[act] * W[act].sum(axis=1)
        try:
            d = np.linalg.solve(Hs, -g[act])
            fn = f.copy()
            fn[act] += d
            cand.append(fn)
        except np.linalg.LinAlgError:
            pass
        res = [(gnorm(gg), ff, WW, gg) for ff in cand for WW, gg in [grad(ff)]]
        _, f, W, g = min(res, key=lambda t: t[0])
    else:
        raise RuntimeError(f"MBAR did not converge (gradient {gnorm(g):.2e})")
    f = update(f, W)  # also the states without samples
    lw, _ = _mbar_weights(u, Nk, f)
    return f, _mbar_covariance(np.exp(lw).T, Nk)


def _mbar_covariance(W: np.ndarray, N_k: np.ndarray) -> np.ndarray:
    """Return Theta = W^T (I - W N W^T)^+ W (K, K) ([3]_ of the module, Eq. 8) by the SVD of W (N, K).

    Theta = V S (I - S V^T N V S)^+ S V^T with W = U S V^T (pymbar's "svd-ew"); W are the MBAR
    weights (N, K), N = diag(N_k).
    """
    K = W.shape[1]
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    V = Vt.T
    S = np.diag(s)
    inner = np.eye(K) - S @ V.T @ np.diag(N_k) @ V @ S
    return V @ S @ np.linalg.pinv(inner) @ S @ V.T


def mbar_differences(f: ArrayLike, Theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Delta f_ij = f_j - f_i, its standard error) as (K, K) arrays [kT]."""
    f = np.asarray(f, float)
    d = np.diag(Theta)
    var = d[:, None] + d[None, :] - 2.0 * Theta
    return f[None, :] - f[:, None], np.sqrt(np.maximum(var, 0.0))


# ----------------------------------------------------------------------------- TI
def ti(lambdas: ArrayLike, means: ArrayLike, sems: ArrayLike | None = None) -> tuple[float, float]:
    """Return (Delta G, standard error) by the trapezoid rule along a path of lambda points.

    Parameters
    ----------
    lambdas : ArrayLike (K, d)
        Path points (e.g. (lambda_elec, lambda_vdw)).
    means : ArrayLike (K, d)
        Mean gradients <dU/dlambda> at the points (any energy unit).
    sems : ArrayLike (K, d), optional
        Their standard errors, taken as independent between windows (None: zero).

    Returns
    -------
    dG, err : float
        sum over segments of (G_k + G_{k+1})/2 . (lambda_{k+1} - lambda_k), in the unit of `means`.
    """
    L = np.asarray(lambdas, float).reshape(len(lambdas), -1)
    G = np.asarray(means, float).reshape(L.shape)
    E = np.zeros_like(G) if sems is None else np.asarray(sems, float).reshape(L.shape)
    dL = np.diff(L, axis=0)  # (K-1, d)
    val = float(np.sum(0.5 * (G[:-1] + G[1:]) * dL))
    c = np.zeros_like(L)  # weight of each window's mean
    c[:-1] += 0.5 * dL
    c[1:] += 0.5 * dL
    return val, float(np.sqrt(np.sum((c * E) ** 2)))


# ----------------------------------------------------------------------------- everything
def estimate(
    samples: Mapping[str, Any],
    discard_ps: float = 0.0,
    gas: dict | None = None,
    stride: int = 1,
    end_ps: float | None = None,
) -> dict[str, Any]:
    """Return the free energy of switching the solute off (first window -> last) by TI, BAR and MBAR.

    Parameters
    ----------
    samples : mapping
        Samples of alchemy.FreeEnergyRun (a dict or np.load(prefix_fe.npz)): u, dudl, lambdas, kT,
        time_ps (module docstring).
    discard_ps : float
        Samples up to this time are discarded [ps].
    gas : dict, optional
        Gas-phase leg: {"delta_g": Delta G_gas(1 -> 0), "dudl": <dE_gas/dlambda_elec> at each
        window's lambda_elec} in kJ/mol, optionally with "delta_g_err" and "dudl_err"; delta_g and
        delta_g_err may be dicts by method ("ti", "bar", "mbar") for a sampled gas phase (flexible
        solute: estimate() of its own run).  A rigid solute's gas leg is exact
        (alchemy.GasPhaseLeg: E_gas(0) - E_gas(1), no error).
    stride : int
        Keep every stride-th sample.
    end_ps : float, optional
        Last time used [ps]; None: all.

    Returns
    -------
    dict
        windows, samples_per_window, kT, discard_ps; g_dE, g_dudl (statistical inefficiencies per
        window); ti, ti_err, dudl_mean, dudl_sem; bar, bar_err, bar_steps, bar_steps_err; mbar,
        mbar_err, mbar_f, mbar_steps, overlap_min; with an electrostatics/vdW stage point also
        mbar_elec, mbar_vdw, ti_elec, ti_vdw (and errors); with `gas` also gas, dG_hyd_<method>
        (and _err) and dG_hyd_ti_sub(_err).  Energies in kJ/mol, and each scalar energy again as
        "<key>_kcal" in kcal/mol.

    Raises
    ------
    ValueError
        Fewer than 10 samples after discarding.

    Notes
    -----
    Per window, samples after discard_ps are subsampled with the statistical inefficiency of the
    energy difference to the neighbouring window ("dE", for BAR / MBAR; alchemlyb's choice) and of
    dU/dlambda along the path (TI).  With a gas leg the result also has the hydration (solvation)
    free energy Delta G_hyd = Delta G_gas - Delta G_solv (errors added in quadrature) and TI of the
    gas-subtracted integrand (the solute's intramolecular electrostatics removed before the
    quadrature).  The stage point is the window with lambda = (0, 1).
    """
    S = dict(samples)
    u = np.asarray(S["u"], float)
    dudl = np.asarray(S["dudl"], float)
    L = np.asarray(S["lambdas"], float)
    kT = float(S["kT"])
    t = np.asarray(S["time_ps"], float)
    last = np.inf if end_ps is None else float(end_ps) + 1e-9
    keep = np.nonzero((t > float(discard_ps) + 1e-9) & (t <= last))[0][:: max(int(stride), 1)]
    if len(keep) < 10:
        raise ValueError(f"only {len(keep)} samples after {discard_ps} ps")
    u, dudl = u[keep], dudl[keep]
    K = len(L)
    # direction of the path at each window (for dU/dlambda along it)
    dirs = np.zeros_like(L)
    seg = np.diff(L, axis=0)
    seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-300)
    dirs[:-1] += seg
    dirs[1:] += seg
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-300)
    out = {"windows": K, "samples_per_window": len(keep), "kT": kT, "discard_ps": float(discard_ps)}
    # ---- decorrelation per window
    g_de, g_ti, idx_de, idx_ti = [], [], [], []
    for k in range(K):
        nb = k + 1 if k + 1 < K else k - 1
        de = u[:, nb, k] - u[:, k, k]
        g1 = statistical_inefficiency(de)
        g2 = statistical_inefficiency(dudl[:, k] @ dirs[k])
        g_de.append(g1)
        g_ti.append(g2)
        idx_de.append(subsample(len(keep), g1))
        idx_ti.append(subsample(len(keep), g2))
    out["g_dE"], out["g_dudl"] = [float(x) for x in g_de], [float(x) for x in g_ti]
    # ---- TI
    means = np.array([dudl[idx_ti[k], k].mean(0) for k in range(K)])
    sems = np.array([dudl[idx_ti[k], k].std(0, ddof=1) / math.sqrt(max(len(idx_ti[k]), 2)) for k in range(K)])
    dg, e = ti(L, means, sems)
    out["dudl_mean"], out["dudl_sem"] = means.tolist(), sems.tolist()
    out["ti"], out["ti_err"] = dg, e
    # ---- BAR between neighbours
    dfs, errs = [], []
    for k in range(K - 1):
        wF = u[idx_de[k], k + 1, k] - u[idx_de[k], k, k]
        wR = u[idx_de[k + 1], k, k + 1] - u[idx_de[k + 1], k + 1, k + 1]
        df, de = bar(wF, wR)
        dfs.append(df)
        errs.append(de)
    out["bar_steps"], out["bar_steps_err"] = [kT * x for x in dfs], [kT * x for x in errs]
    out["bar"], out["bar_err"] = kT * float(np.sum(dfs)), kT * float(np.sqrt(np.sum(np.square(errs))))
    # ---- MBAR on every window's uncorrelated samples
    cols = [u[idx_de[k], :, k] for k in range(K)]  # (n_k, K) each: sample of window k in every state
    N_k = np.array([len(c) for c in cols])
    u_kn = np.concatenate(cols, axis=0).T
    f, Theta = mbar(u_kn, N_k)
    D, dD = mbar_differences(f, Theta)
    out["mbar"], out["mbar_err"] = kT * float(D[0, -1]), kT * float(dD[0, -1])
    out["mbar_f"] = (kT * f).tolist()
    out["mbar_steps"] = [kT * float(D[k, k + 1]) for k in range(K - 1)]
    out["overlap_min"] = _overlap_min(u_kn, N_k, f)
    # ---- stages (electrostatics, van der Waals) by MBAR
    stage = np.nonzero((L[:, 0] == 0.0) & (L[:, 1] == 1.0))[0]
    if len(stage) == 1 and 0 < stage[0] < K - 1:
        s = int(stage[0])
        out["mbar_elec"], out["mbar_elec_err"] = kT * float(D[0, s]), kT * float(dD[0, s])
        out["mbar_vdw"], out["mbar_vdw_err"] = kT * float(D[s, -1]), kT * float(dD[s, -1])
        out["ti_elec"] = ti(L[: s + 1], means[: s + 1], sems[: s + 1])[0]
        out["ti_vdw"] = ti(L[s:], means[s:], sems[s:])[0]
    # ---- hydration free energy with the gas-phase leg
    if gas is not None:

        def by(v: float | dict, m: str) -> float:
            return float(v[m]) if isinstance(v, dict) else float(v)

        out["gas"] = by(gas["delta_g"], "mbar")
        for m in ("ti", "bar", "mbar"):
            ge = by(gas.get("delta_g_err", 0.0), m)
            out[f"dG_hyd_{m}"] = by(gas["delta_g"], m) - out[m]
            out[f"dG_hyd_{m}_err"] = math.hypot(out[f"{m}_err"], ge)
        gd = np.asarray(gas["dudl"], float).reshape(K)
        gde = np.asarray(gas.get("dudl_err", np.zeros(K)), float).reshape(K)
        sub, sube = means.copy(), sems.copy()
        sub[:, 0] -= gd
        sube[:, 0] = np.hypot(sube[:, 0], gde)
        dgs, es = ti(L, sub, sube)
        out["dG_hyd_ti_sub"], out["dG_hyd_ti_sub_err"] = -dgs, es
    for key in list(out):
        if key.startswith(("ti", "bar", "mbar", "dG_", "gas")) and isinstance(out[key], float):
            out[key + "_kcal"] = out[key] / KCAL
    return out


def equilibration_times(samples: Mapping[str, Any]) -> np.ndarray:
    """Return, per window, the equilibration time [ps] that maximises the number of uncorrelated samples.

    The series is each window's reduced energy difference to its neighbouring window
    (stats.detect_equilibration, every len/100-th start tried): a check of discard_ps.
    """
    u = np.asarray(samples["u"], float)
    t = np.asarray(samples["time_ps"], float)
    K = u.shape[1]
    out = []
    for k in range(K):
        nb = k + 1 if k + 1 < K else k - 1
        t0, _, _ = detect_equilibration(u[:, nb, k] - u[:, k, k], nskip=max(1, len(t) // 100))
        out.append(t[t0] - t[0])
    return np.array(out)


def _overlap_min(u_kn: ArrayLike, N_k: ArrayLike, f: ArrayLike) -> float:
    """Return the smallest neighbour element O[k, k+1] of the MBAR overlap matrix O = W^T W N ([4]_ of the module).

    Below ~0.03 neighbouring windows overlap poorly.
    """
    lw, _ = _mbar_weights(np.asarray(u_kn, float), np.asarray(N_k, float), np.asarray(f, float))
    W = np.exp(lw).T
    O = W.T @ W * np.asarray(N_k, float)[None, :]
    return float(min(O[k, k + 1] for k in range(len(N_k) - 1)))


def load(path: str) -> dict:
    """Return prefix_fe.npz as a dict of arrays, with "meta" decoded from JSON ({} if absent)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d.get("meta", "{}")))
    return d
