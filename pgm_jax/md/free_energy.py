"""Free-energy estimators for lambda windows: TI, BAR and MBAR with uncertainties, statistical
inefficiency and equilibration detection from analysis/stats.py (numpy / scipy; no pymbar).

Conventions: reduced energies u = beta U (dimensionless); free energies f in units of kT unless a
function says kJ/mol.  u_kn[k, n] is the reduced energy of sample n in state k (samples of all
states pooled, N_k of them from state k, in order).

  bar(w_F, w_R)                    Bennett's acceptance ratio (Bennett, J. Comput. Phys. 22, 245
                                   (1976); Shirts, Bair, Hooker & Pande, PRL 91, 140601 (2003)),
                                   variance from Bennett's Eq. 10a (pymbar's default "BAR")
  mbar(u_kn, N_k)                  the MBAR equations (Shirts & Chodera, JCP 129, 124105 (2008))
                                   solved by Newton's method on their convex objective; asymptotic
                                   covariance Theta = W^T (I - W N W^T)^+ W by the SVD route
  ti(lambdas, means, sems)         trapezoid rule along the path of (lambda_elec, lambda_vdw) points
  estimate(samples, ...)           everything above for the samples of alchemy.FreeEnergyRun
                                   (prefix_fe.npz), with the gas-phase leg for a hydration free energy

Uncertainties are one standard error of the uncorrelated (subsampled) estimates."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

from ..analysis.stats import detect_equilibration, statistical_inefficiency, subsample
from ..units import KCAL


# ----------------------------------------------------------------------------- time series
# ----------------------------------------------------------------------------- BAR
def bar(w_F, w_R, tol: float = 1e-12) -> tuple:
    """(Delta f, standard error) in kT between two states 0 -> 1.  w_F = u_1(x) - u_0(x) for samples
    x of state 0, w_R = u_0(x) - u_1(x) for samples of state 1."""
    wF, wR = np.asarray(w_F, float).ravel(), np.asarray(w_R, float).ravel()
    nF, nR = len(wF), len(wR)
    if nF == 0 or nR == 0:
        raise ValueError("BAR needs samples from both states")
    M = math.log(nF / nR)

    def zero(df):
        """log sum_F f(M + w_F - df) - log sum_R f(-M + w_R + df), f(x) = 1 / (1 + e^x): increasing in
        df, zero at the BAR estimate (log space: no overflow for large works)."""
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
def _mbar_weights(u_kn, N_k, f):
    """log W_nk = f_k - u_kn - log sum_l N_l exp(f_l - u_ln) (K, N)."""
    with np.errstate(invalid="ignore"):
        a = f[:, None] - u_kn
    a = np.where(np.isfinite(a), a, -np.inf)
    with np.errstate(divide="ignore"):
        logden = logsumexp(a + np.log(N_k)[:, None], axis=0)
    return a - logden[None, :], logden


def mbar(u_kn, N_k, tol: float = 1e-10, max_iter: int = 500, f0=None) -> tuple:
    """(f (K,), Theta (K, K)): MBAR free energies (kT, f_0 = 0) and their asymptotic covariance.
    u_kn (K, N): reduced energies of every pooled sample in every state (+inf allowed: zero
    weight); N_k (K,): samples drawn from each state, in the order of the pooled samples (state 0
    must have samples; states without samples get f from the converged weights).

    The MBAR equations are the stationarity conditions of the convex function F(f) = sum_n log
    sum_k N_k exp(f_k - u_kn) - sum_k N_k f_k.  Start: neighbouring sampled states joined by the
    mean of the forward and reverse first-order estimates; then each iteration takes the Newton
    step or the self-consistent update, whichever leaves the smaller gradient (pymbar's
    'adaptive' scheme, which needs no objective comparisons: F is a large sum and its changes near
    convergence are below its rounding), until max|dF/df_k| / N_k < tol."""
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

    def grad(f):
        lw, _ = _mbar_weights(u, Nk, f)
        W = np.exp(lw)
        return W, Nk * (W.sum(axis=1) - 1.0)

    def gnorm(g):
        return float(np.max(np.abs(g[act] / Nk[act]))) if len(act) else 0.0

    def update(f, W):  # self-consistent MBAR equation
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


def _mbar_covariance(W, N_k):
    """Theta = W^T (I - W N W^T)^+ W (Shirts & Chodera 2008, Eq. 8) by the SVD of W (N, K):
    Theta = V S (I - S V^T N V S)^+ S V^T (pymbar's 'svd-ew')."""
    K = W.shape[1]
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    V = Vt.T
    S = np.diag(s)
    inner = np.eye(K) - S @ V.T @ np.diag(N_k) @ V @ S
    return V @ S @ np.linalg.pinv(inner) @ S @ V.T


def mbar_differences(f, Theta) -> tuple:
    """(Delta f_ij = f_j - f_i, its standard error) as (K, K) arrays."""
    f = np.asarray(f, float)
    d = np.diag(Theta)
    var = d[:, None] + d[None, :] - 2.0 * Theta
    return f[None, :] - f[:, None], np.sqrt(np.maximum(var, 0.0))


# ----------------------------------------------------------------------------- TI
def ti(lambdas, means, sems=None) -> tuple:
    """(Delta G, standard error) by the trapezoid rule along the path through the points
    lambdas (K, d) (e.g. (lambda_elec, lambda_vdw)), with the mean gradients means (K, d) of U
    (any units) and their standard errors sems (K, d), taken as independent between windows."""
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
def estimate(samples, discard_ps: float = 0.0, gas=None, stride: int = 1, end_ps: float | None = None) -> dict:
    """Free energy of switching the solute off (lambda = first window -> last window), by TI, BAR
    and MBAR, from the samples of alchemy.FreeEnergyRun (a dict or np.load(prefix_fe.npz)).

    Per window, samples after discard_ps (and up to end_ps, if given) are subsampled with the statistical
    inefficiency of the
    energy difference to the neighbouring window ('dE', for BAR / MBAR; alchemlyb's choice) and of
    dU/dlambda along the path (TI).  gas: optional gas-phase leg, {"delta_g": Delta G_gas(1 -> 0),
    "dudl": <dE_gas/dlambda_elec> at each window's lambda_elec} in kJ/mol, optionally with
    "delta_g_err" and "dudl_err"; delta_g and delta_g_err may be dicts by method ("ti", "bar",
    "mbar") for a sampled gas phase (flexible solute: estimate() of its own run).  A rigid solute's
    gas leg is exact (alchemy.GasPhaseLeg: E_gas(0) - E_gas(1), no error).  Then the result also
    has the hydration (solvation) free energy Delta G_hyd = Delta G_gas - Delta G_solv (errors
    added in quadrature) and TI of the gas-subtracted integrand (the solute's intramolecular
    electrostatics removed before the quadrature).  Energies in kJ/mol ('..._kcal': kcal/mol)."""
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

        def by(v, m):
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


def equilibration_times(samples) -> np.ndarray:
    """Per window, the equilibration time (ps) that maximises the number of uncorrelated samples of
    its energy difference to the neighbouring window (detect_equilibration): a check of discard_ps."""
    u = np.asarray(samples["u"], float)
    t = np.asarray(samples["time_ps"], float)
    K = u.shape[1]
    out = []
    for k in range(K):
        nb = k + 1 if k + 1 < K else k - 1
        t0, _, _ = detect_equilibration(u[:, nb, k] - u[:, k, k], nskip=max(1, len(t) // 100))
        out.append(t[t0] - t[0])
    return np.array(out)


def _overlap_min(u_kn, N_k, f) -> float:
    """Smallest off-diagonal neighbour element of the MBAR overlap matrix O = W^T W N (Klimovich,
    Shirts & Mobley, JCAMD 29, 397 (2015)); below ~0.03 neighbouring windows overlap poorly."""
    lw, _ = _mbar_weights(np.asarray(u_kn, float), np.asarray(N_k, float), np.asarray(f, float))
    W = np.exp(lw).T
    O = W.T @ W * np.asarray(N_k, float)[None, :]
    return float(min(O[k, k + 1] for k in range(len(N_k) - 1)))


def load(path: str) -> dict:
    """prefix_fe.npz as a dict (meta decoded from JSON)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d.get("meta", "{}")))
    return d
