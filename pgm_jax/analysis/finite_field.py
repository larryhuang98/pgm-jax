"""Analyse finite-field runs: dielectric constants from +-E replica pairs and zero-field fluctuations.

The runs are those of md/finite_field.FieldReplicas (docs/efield.md).  Contents: fluctuation_eps
(the fluctuation formula of one zero-field replica), analyse (all replicas of a run: single-sided,
+-E pairs, zero field), saturation_fit (eps(E) = eps0 - c E^2) and predicted_errors (the error
expected for a planned run).

With a uniform field E along e in tin-foil Ewald, eps = 1 + EPS_FACTOR <M.e> / (V |E|)
(md/efield.py); at constant displacement D, eps = D / (D - <M.e>/(eps0 V)) (finite_d_eps).  The
antisymmetric combination (<M.e>_{+E} - <M.e>_{-E}) / 2 of a +-E pair cancels the even-order
response and the spontaneous polarization of the box.

Units: M [e nm], V [nm^3], T [K], fields [V/nm], times [ps].
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..md.efield import EPS_FACTOR, finite_d_eps
from ..units import E_CHARGE_C, EPS0_SI, KB_SI
from .stats import block_mean, integrated_correlation_time, jackknife_error


def fluctuation_eps(
    M: ArrayLike, V: float, temperature: float, eps_inf: float = 1.0, nblocks: int = 10
) -> tuple[float, float]:
    """Return the static dielectric constant of one replica from its dipole fluctuations (tin-foil Ewald).

    Parameters
    ----------
    M : ArrayLike (F, 3)
        Cell dipole series [e nm].
    V : float
        Volume [nm^3].
    temperature : float
        Temperature [K].
    eps_inf : float
        High-frequency dielectric constant.
    nblocks : int
        Contiguous blocks of the jackknife (the first F mod nblocks samples are left out of the
        blocks).

    Returns
    -------
    (float, float)
        eps = eps_inf + (<M.M> - <M>.<M>) / (3 eps0 V kB T) and its jackknife error.
    """
    M = np.asarray(M, float)
    c = (E_CHARGE_C * 1e-9) ** 2 / (3 * EPS0_SI * V * 1e-27 * KB_SI * temperature)

    def est(m: np.ndarray) -> float:
        return eps_inf + c * (np.mean(np.sum(m * m, 1)) - np.sum(np.mean(m, 0) ** 2))

    n = len(M) // nblocks * nblocks
    Mb = M[len(M) - n :].reshape(nblocks, -1, 3)
    jk = np.array([est(np.concatenate([Mb[j] for j in range(nblocks) if j != i])) for i in range(nblocks)])
    return float(est(M)), float(jackknife_error(jk))


def analyse(
    meta: dict, data: dict, skip_ps: float = 50.0, nblocks: int = 10, eps_inf: float | None = None
) -> dict[str, Any]:
    """Return the dielectric constants of all replicas of a finite-field run.

    Parameters
    ----------
    meta : dict
        Run metadata: "volume_nm3", "temperature_K", "fields" (R, 3) [V/nm], optionally
        "field_kind" ("E" or "D": constant displacement, fields are D/eps0) and "eps_inf".
    data : dict
        "time_ps" (F,) and "M" (F, R, 3) cell dipoles [e nm] of the R replicas.
    skip_ps : float
        Equilibration time left out [ps].
    nblocks : int
        Blocks of the block-mean errors.
    eps_inf : float, optional
        eps_inf of the fluctuation estimate; None: meta["eps_inf"] or 1.

    Returns
    -------
    dict
        "volume_nm3", "temperature_K", "run_ps"; "single": per nonzero-field replica, eps from
        <M.e> with its block error, correlation time and the rms perpendicular dipole; "pairs": per
        +-E pair, eps from the antisymmetric combination; "zero": per zero-field replica (constant-E
        runs only), the fluctuation estimate and the correlation time of M_z; "fits":
        saturation_fit over the pairs up to each field strength.
    """
    disp = meta.get("field_kind", "E") == "D"

    def eps_of(m: float, err: float, mag: float) -> tuple[float, float]:
        """Return eps and its error from the mean <M.e> [e nm], its error and |E| (constant E or constant D)."""
        if not disp:
            return 1 + EPS_FACTOR * m / (V * mag), EPS_FACTOR * err / (V * mag)
        e = float(finite_d_eps(m, V, mag))
        return e, e * e * EPS_FACTOR * err / (V * mag)

    V, T = float(meta["volume_nm3"]), float(meta["temperature_K"])
    if eps_inf is None:
        eps_inf = float(meta.get("eps_inf", 1.0))
    sel = data["time_ps"] >= data["time_ps"][0] + skip_ps
    M = data["M"][sel]
    t = data["time_ps"][sel]
    dt = float(np.median(np.diff(t)))
    F = np.asarray(meta["fields"])
    out = {
        "volume_nm3": V,
        "temperature_K": T,
        "run_ps": float(t[-1] - t[0] + dt),
        "single": [],
        "pairs": [],
        "zero": [],
    }
    for k, E in enumerate(F):
        mag = float(np.linalg.norm(E))
        if mag == 0.0 and disp:
            continue
        if mag == 0.0:
            eps, err = fluctuation_eps(M[:, k], V, T, eps_inf, nblocks)
            out["zero"].append(
                {
                    "replica": k,
                    "eps": eps,
                    "err": err,
                    "tau_ps": integrated_correlation_time(M[:, k, 2], dt),
                    "M_mean": M[:, k].mean(0).tolist(),
                }
            )
            continue
        e = E / mag
        m, err = block_mean(M[:, k] @ e, nblocks)
        tau = integrated_correlation_time(M[:, k] @ e, dt)
        eps, eerr = eps_of(m, err, mag)
        out["single"].append(
            {
                "replica": k,
                "E": E.tolist(),
                "E_mag": mag,
                "M_par": m,
                "M_par_err": err,
                "eps": eps,
                "err": eerr,
                "tau_ps": tau,
                "M_perp_rms": float(np.sqrt(np.mean(np.sum((M[:, k] - np.outer(M[:, k] @ e, e)) ** 2, 1)))),
            }
        )
    used = set()
    for i, Ei in enumerate(F):
        for j, Ej in enumerate(F):
            if j <= i or i in used or j in used or np.linalg.norm(Ei) == 0 or not np.allclose(Ei, -Ej):
                continue
            used |= {i, j}
            mag = float(np.linalg.norm(Ei))
            e = Ei / mag
            d = 0.5 * (M[:, i] @ e - M[:, j] @ e)
            m, err = block_mean(d, nblocks)
            eps, eerr = eps_of(m, err, mag)
            out["pairs"].append(
                {
                    "replicas": (i, j),
                    "E_mag": mag,
                    "eps": eps,
                    "err": eerr,
                    "tau_ps": integrated_correlation_time(d, dt),
                }
            )
    out["fits"] = []
    mags = sorted({p["E_mag"] for p in out["pairs"]})
    for emax in mags[1:]:
        f = saturation_fit([p for p in out["pairs"] if p["E_mag"] <= emax + 1e-12])
        if f is not None:
            out["fits"].append(dict(f, E_max=emax))
    return out


def saturation_fit(pairs: list[dict]) -> dict[str, float] | None:
    """Fit eps(E) = eps0 - c E^2 over +-E pairs by weighted least squares (dielectric saturation).

    Parameters
    ----------
    pairs : list of dict
        Entries of analyse(...)["pairs"] ("E_mag" [V/nm], "eps", "err").

    Returns
    -------
    dict or None
        eps0, eps0_err, c [nm^2/V^2], c_err, chi2, n; None for fewer than two distinct fields.
    """
    E = np.array([p["E_mag"] for p in pairs])
    if len(np.unique(E)) < 2:
        return None
    y = np.array([p["eps"] for p in pairs])
    w = 1.0 / np.array([p["err"] for p in pairs]) ** 2
    A = np.stack([np.ones_like(E), -E * E], 1)
    C = np.linalg.inv(A.T @ (w[:, None] * A))
    b = C @ A.T @ (w * y)
    chi2 = float(np.sum(w * (y - A @ b) ** 2))
    return {
        "eps0": float(b[0]),
        "eps0_err": float(np.sqrt(C[0, 0])),
        "c": float(b[1]),
        "c_err": float(np.sqrt(C[1, 1])),
        "chi2": chi2,
        "n": int(len(E)),
    }


def predicted_errors(
    eps: float, eps_inf: float, V: float, temperature: float, field: float, tau_ps: float, run_ps: float
) -> dict:
    """Return the expected statistical errors of eps from a +-E pair and from zero-field fluctuations.

    Parameters
    ----------
    eps, eps_inf : float
        Static and high-frequency dielectric constants.
    V : float
        Volume [nm^3].
    temperature : float
        Temperature [K].
    field : float
        |E| of the pair [V/nm].
    tau_ps : float
        Integrated correlation time of M [ps].
    run_ps : float
        Length of each replica [ps] (the zero-field run gets 2 run_ps, the same cost).

    Returns
    -------
    dict
        sigma_ff (the +-E pair), sigma_fluct_same_cost and cost_ratio (fluctuation / finite-field
        variance at equal cost).

    Notes
    -----
    For a Gaussian M (analysis/dielectric.py): var(M_e) = (eps - eps_inf) eps0 V kB T, the error of
    a mean over a run of length T with correlation time tau is sqrt(var 2 tau / T), and the relative
    error of the fluctuation term is sqrt(2 tau / (3 T)).
    """
    var_Me = (
        (eps - eps_inf) * EPS0_SI * V * 1e-27 * KB_SI * temperature / (E_CHARGE_C * 1e-9) ** 2
    )  # (e nm)^2, one component
    s_mean = np.sqrt(var_Me * 2 * tau_ps / run_ps)  # error of <M_e> of one run
    s_ff = EPS_FACTOR / (V * field) * s_mean / np.sqrt(2)  # the +-E combination
    s_fl = (eps - eps_inf) * np.sqrt(2 * tau_ps / (3 * 2 * run_ps))
    return {"sigma_ff": float(s_ff), "sigma_fluct_same_cost": float(s_fl), "cost_ratio": float((s_fl / s_ff) ** 2)}
