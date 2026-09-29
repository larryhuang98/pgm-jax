"""Use free energies with parameter gradients as fitting targets: estimators, errors, chain rule.

The samples come from pgm_jax/md/fe_grad.ParameterGradients through
alchemy.FreeEnergyRun(param_grad=...); the identity d f_k/dtheta = <dU_k/dtheta>_k is explained
there.  Contents: gradient_estimate (free energies and gradients of a run), FEGradient (one free
energy with gradient and jackknife replicates), FreeEnergyTarget (a chi^2 term of a fit: value,
gradient, linear prediction), combine (linear combinations of independent results).

Estimators (gradient_estimate):

  "end":  the end windows' own samples, <dU_{K-1}/dtheta> over window K-1 minus <dU_0/dtheta> over
          window 0 (the identity above, directly);
  "mbar": the same two expectations with MBAR weights over the samples of every window,
          E_k[A] = sum_n W_nk A_k(x_n) [1]_: this is exactly the derivative of the MBAR estimate
          of f_k(theta) for states k(theta) reweighted from the sampled mixture, so it is
          consistent with the MBAR free energy.

Errors: block jackknife over time (n_blocks contiguous blocks, the same time block removed from
every window, which keeps the correlation between windows coupled by Hamiltonian exchange and
along each window's time series inside a block); the jackknife replicates are kept so that errors
of any linear combination of gradient entries (chain rule, scale directions) are exact.

Units: kJ/mol; gradients in kJ/mol per unit of each parameter-table entry.

References
----------
.. [1] M. R. Shirts, J. D. Chodera, J. Chem. Phys. 129, 124105 (2008).

See also docs/fe_gradients.md.
"""

from __future__ import annotations

import dataclasses as _dc
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..analysis import free_energy as fe
from ..analysis.stats import jackknife_error
from ..units import KCAL
from .params import ParameterSpace


# ----------------------------------------------------------------------------- estimators
@_dc.dataclass
class FEGradient:
    """A free energy with its gradient over the parameter entries, their errors and jackknife replicates.

    A mutable dataclass.  The replicates give exact errors of projections and chain-rule products.

    Parameters
    ----------
    value : float
        Free energy [kJ/mol].
    value_err : float
        Its jackknife standard error [kJ/mol].
    grad : np.ndarray (M,)
        Gradient over the entries `names` [kJ/mol per unit of each entry].
    grad_err : np.ndarray (M,)
        Its jackknife standard errors.
    names : list of str (M,)
        Parameter entries "quantity:key" (ParameterSpace.values names).
    jk_value : np.ndarray (B,)
        Delete-one-block replicates of the value.
    jk_grad : np.ndarray (B, M)
        Delete-one-block replicates of the gradient.
    estimator : {"mbar", "end"}
        Gradient estimator.
    """

    value: float
    value_err: float
    grad: np.ndarray
    grad_err: np.ndarray
    names: list
    jk_value: np.ndarray
    jk_grad: np.ndarray
    estimator: str = "mbar"

    def project(self, v: ArrayLike) -> tuple[float, float]:
        """Return (grad . v, its standard error) for a direction v (M,).

        E.g. v = ParameterSpace.scale_direction(...) gives dG/d ln s [kJ/mol].
        """
        v = np.asarray(v, float)
        return float(self.grad @ v), float(jackknife_error(self.jk_grad @ v))

    def chain(self, J: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return (dG/dtheta (n,), standard errors (n,)) for the Jacobian J = dP_flat/dtheta (M, n)."""
        J = np.asarray(J, float).reshape(len(self.names), -1)
        return self.grad @ J, jackknife_error(self.jk_grad @ J)

    def kcal(self) -> dict[str, float]:
        """Return {"value", "value_err"} in kcal/mol."""
        return {"value": self.value / KCAL, "value_err": self.value_err / KCAL}

    def named(self, indices: Sequence[int] | None = None) -> dict[str, tuple[float, float]]:
        """Return {name: (gradient, error)} for the entries `indices` (None: all)."""
        idx = range(len(self.names)) if indices is None else indices
        return {self.names[i]: (float(self.grad[i]), float(self.grad_err[i])) for i in idx}


def _subset(S: Mapping[str, Any], discard_ps: float, end_ps: float | None, stride: int) -> np.ndarray:
    """Return the sample indices with discard_ps < time <= end_ps [ps], every stride-th."""
    t = np.asarray(S["time_ps"], float)
    last = np.inf if end_ps is None else float(end_ps) + 1e-9
    keep = np.nonzero((t > float(discard_ps) + 1e-9) & (t <= last))[0][:: max(int(stride), 1)]
    return keep


def gradient_estimate(
    samples: Mapping[str, Any],
    discard_ps: float = 0.0,
    gas: dict | None = None,
    n_blocks: int = 10,
    end_ps: float | None = None,
    stride: int = 1,
) -> dict[str, Any]:
    """Return the free energies of the solution leg and of hydration with their parameter gradients.

    The solution leg is window 0 -> K-1.  Errors: block jackknife.

    Parameters
    ----------
    samples : mapping
        FreeEnergyRun samples with parameter gradients (a dict or free_energy.load of
        prefix_fe.npz): u (S, K, K), lambdas, kT [kJ/mol], time_ps, dudp (S, T, K, M) = dU_t/dP of
        each target state t on the samples of each window [kJ/mol per unit], dudp_targets (must
        contain 0 and K-1) and dudp_names (in the arrays or in meta).
    discard_ps : float
        Samples up to this time are discarded [ps].
    gas : dict, optional
        {"delta_g": Delta G_gas(1 -> 0) [kJ/mol], "grad": d Delta G_gas/dP (M,)} (a rigid solute's
        exact leg, gas_leg_gradient; zeros with intramolecular="keep", where the hydration free
        energy is -Delta G_solv); None: no hydration leg.
    n_blocks : int
        Contiguous time blocks of the jackknife.
    end_ps : float, optional
        Last time used [ps].
    stride : int
        Keep every stride-th sample.

    Returns
    -------
    dict
        "solv": {"mbar": FEGradient, "end": FEGradient}, "hyd": {...} (if gas), "names",
        "samples_per_window", "n_blocks", "kT", "end_means": (<dU_0/dP>_0, <dU_{K-1}/dP>_{K-1}) and
        "end_means_err".  The value of the "end" entries is the MBAR free energy too (the end-state
        estimator is for the gradient only).  Delta G_solv = kT (f_{K-1} - f_0); Delta G_hyd =
        Delta G_gas - Delta G_solv.

    Raises
    ------
    ValueError
        No parameter gradients in the samples, gradient targets without both end windows, or fewer
        than 2 n_blocks samples.

    Notes
    -----
    The MBAR solve of every jackknife replicate starts from the full-data free energies.
    """
    S = dict(samples)
    meta = S.get("meta", {})
    if isinstance(meta, str):
        meta = json.loads(meta)
    if "dudp" not in S:
        raise ValueError("the samples have no parameter gradients (FreeEnergyRun(param_grad=ParameterGradients(...)))")
    u = np.asarray(S["u"], float)
    G = np.asarray(S["dudp"], float)
    L = np.asarray(S["lambdas"], float)
    kT = float(S["kT"])
    K = len(L)
    targets = [int(x) for x in (S["dudp_targets"] if "dudp_targets" in S else meta["dudp_targets"])]
    names = list(S["dudp_names"] if "dudp_names" in S else meta["dudp_names"])
    if 0 not in targets or K - 1 not in targets:
        raise ValueError(f"gradient targets {targets} must include both end windows 0 and {K - 1}")
    i0, i1 = targets.index(0), targets.index(K - 1)
    keep = _subset(S, discard_ps, end_ps, stride)
    if len(keep) < 2 * n_blocks:
        raise ValueError(f"only {len(keep)} samples after {discard_ps} ps for {n_blocks} blocks")
    u, G = u[keep], G[keep]
    A0, A1 = G[:, i0], G[:, i1]  # (s, K, M)
    M = G.shape[-1]

    f_start = [None]  # MBAR starting point of the replicates (set after the full estimate)

    def estimate(idx: np.ndarray) -> dict[str, Any]:
        """Return MBAR and end-state estimates from the samples `idx` of every window.

        Keys: "dG" [kJ/mol], "mbar" and "end" gradients (M,), "e0", "e1" end-state means, "f" reduced
        free energies (K,).
        """
        s = len(idx)
        uu = u[idx]
        u_kn = uu.transpose(1, 2, 0).reshape(K, K * s)  # column (window n, sample)
        N_k = np.full(K, s)
        f, _ = fe.mbar(u_kn, N_k, f0=f_start[0])
        lw, _ = fe._mbar_weights(u_kn, N_k, f)
        W = np.exp(lw)
        a0 = A0[idx].transpose(1, 0, 2).reshape(K * s, M)
        a1 = A1[idx].transpose(1, 0, 2).reshape(K * s, M)
        e0, e1 = A0[idx, 0].mean(0), A1[idx, K - 1].mean(0)
        return {"dG": kT * (f[-1] - f[0]), "mbar": W[-1] @ a1 - W[0] @ a0, "end": e1 - e0, "e0": e0, "e1": e1, "f": f}

    full = estimate(np.arange(len(keep)))
    f_start[0] = full["f"]
    blocks = np.array_split(np.arange(len(keep)), int(n_blocks))
    reps = [estimate(np.concatenate([b for j, b in enumerate(blocks) if j != i])) for i in range(len(blocks))]
    jk_dG = np.array([r["dG"] for r in reps])
    out = {
        "names": names,
        "samples_per_window": len(keep),
        "n_blocks": int(n_blocks),
        "kT": kT,
        "end_means": (full["e0"], full["e1"]),
        "end_means_err": (jackknife_error([r["e0"] for r in reps]), jackknife_error([r["e1"] for r in reps])),
    }
    legs = {"solv": (1.0, 0.0, np.zeros(M))}
    if gas is not None:
        gg = np.zeros(M) if gas.get("grad") is None else np.asarray(gas["grad"], float).reshape(M)
        legs["hyd"] = (-1.0, float(gas["delta_g"]), gg)
    for leg, (sgn, c, cg) in legs.items():
        out[leg] = {}
        for est in ("mbar", "end"):
            jk_g = np.array([cg + sgn * r[est] for r in reps])
            jk_v = c + sgn * jk_dG
            out[leg][est] = FEGradient(
                value=c + sgn * full["dG"],
                value_err=float(jackknife_error(jk_v)),
                grad=cg + sgn * full[est],
                grad_err=jackknife_error(jk_g),
                names=names,
                jk_value=jk_v,
                jk_grad=jk_g,
                estimator=est,
            )
    return out


# ----------------------------------------------------------------------------- fitting targets
class FreeEnergyTarget:
    """A free energy (e.g. hydration) as a fitting target.

    Holds the value, gradient and statistical errors at the sampled parameters, the chain rule to
    any parameterization theta -> table, a linear prediction for nearby theta, and a chi^2 term
    against experiment.

        t = FreeEnergyTarget.from_npz("wat_fe.npz", discard_ps=200, experiment=-6.3 * 4.184, sigma=0.2 * 4.184)
        r = t.value_and_grad(theta_fn, theta)      # theta_fn(theta) -> parameter table of the run's system
        r["value"], r["grad"], r["value_err"], r["grad_err"], r["chi2"], r["dchi2"]

    theta_fn(theta) must reproduce the sampled table at the theta of the run (checked); combined
    with alchemical_map(sys0, sysA) it can act on the original (non-alchemical) table.

    Attributes
    ----------
    result : FEGradient
    p : np.ndarray (M,)
        The sampled parameters, flattened.
    names : list of str
    space : ParameterSpace
    experiment, sigma : float or None
        Experimental value and its uncertainty [kJ/mol].
    name : str
    """

    def __init__(
        self,
        result: FEGradient,
        params_flat: ArrayLike,
        space_names: Sequence[str],
        experiment: float | None = None,
        sigma: float | None = None,
        name: str = "",
        space: ParameterSpace | None = None,
    ) -> None:
        """Wrap one free-energy result as a fitting target.

        Parameters
        ----------
        result : FEGradient
            Value [kJ/mol], gradient and their errors at the sampled parameters.
        params_flat : array_like (M,)
            The sampled parameters, flattened by `space`.
        space_names : sequence of str
            Names of the M entries.
        experiment : float, optional
            Experimental value [kJ/mol] (for the chi^2 term).
        sigma : float, optional
            Its uncertainty [kJ/mol].
        name : str
            Label.
        space : ParameterSpace, optional
            The parameter layout (default: ParameterSpace.from_names(space_names)).
        """
        self.result, self.p = result, np.asarray(params_flat, float)
        self.names = list(space_names)
        self.space = space or ParameterSpace.from_names(self.names)
        self.experiment, self.sigma, self.name = experiment, sigma, name

    @classmethod
    def from_samples(
        cls,
        samples: Mapping[str, Any],
        discard_ps: float = 0.0,
        estimator: str = "mbar",
        leg: str | None = None,
        n_blocks: int = 10,
        experiment: float | None = None,
        sigma: float | None = None,
        name: str = "",
        gas: dict | None = None,
    ) -> FreeEnergyTarget:
        """Build a target from FreeEnergyRun samples with parameter gradients.

        Parameters
        ----------
        samples : mapping
            As gradient_estimate; meta must hold "params_flat" (and may hold "gas_delta_g", "gas_grad").
        discard_ps : float
            Samples up to this time are discarded [ps].
        estimator : {"mbar", "end"}
            Gradient estimator.
        leg : {"solv", "hyd"}, optional
            None: "hyd" if a gas leg is available, else "solv".
        n_blocks : int
            Jackknife blocks.
        experiment, sigma : float, optional
            Experimental value and uncertainty [kJ/mol].
        name : str
            Label.
        gas : dict, optional
            Gas leg as in gradient_estimate; None: from meta if present.

        Returns
        -------
        FreeEnergyTarget

        Notes
        -----
        meta must be a dict here (as from free_energy.load); gradient_estimate also accepts a JSON string.
        """
        S = dict(samples)
        meta = S.get("meta", {})
        if gas is None and meta.get("gas_grad") is not None:
            gas = {"delta_g": meta["gas_delta_g"], "grad": meta["gas_grad"]}
        r = gradient_estimate(S, discard_ps=discard_ps, gas=gas, n_blocks=n_blocks)
        leg = leg or ("hyd" if "hyd" in r else "solv")
        return cls(r[leg][estimator], meta["params_flat"], r["names"], experiment, sigma, name)

    @classmethod
    def from_npz(cls, path: str, **kw: Any) -> FreeEnergyTarget:
        """Build a target from prefix_fe.npz (analysis.free_energy.load); `**kw` as from_samples."""
        return cls.from_samples(fe.load(path), **kw)

    def _J(self, theta_fn: Callable[[jax.Array], dict], theta: ArrayLike, space: ParameterSpace) -> np.ndarray:
        """Return J = d space.flatten(theta_fn(theta)) / dtheta (M, n) by jax.jacfwd.

        Raises
        ------
        ValueError
            If theta_fn(theta) is not the sampled parameter table (rtol 1e-8, atol 1e-12).
        """

        def flat(th: jax.Array) -> jax.Array:
            return space.flatten(theta_fn(th))

        th = jnp.asarray(theta, jnp.float64)
        p = np.asarray(flat(th), float)
        if p.shape != self.p.shape or not np.allclose(p, self.p, rtol=1e-8, atol=1e-12):
            raise ValueError("theta_fn(theta) is not the parameter table of the sampled run")
        return np.asarray(jax.jacfwd(flat)(th), float).reshape(len(p), -1)

    def value_and_grad(
        self,
        theta_fn: Callable[[jax.Array], dict] | None = None,
        theta: ArrayLike | None = None,
        space: ParameterSpace | None = None,
    ) -> dict[str, Any]:
        """Return the value, gradient, their standard errors and, with an experiment, chi2 and its gradient.

        Parameters
        ----------
        theta_fn : callable, optional
            theta -> parameter table of the run's system; None: the gradient per table entry.
        theta : ArrayLike (n,), optional
            The fitted parameters (with theta_fn).
        space : ParameterSpace, optional
            Layout of the flat table (None: self.space).

        Returns
        -------
        dict
            "value", "value_err" [kJ/mol], "grad", "grad_err" (per theta or per entry), "name", and with
            an experiment "chi2" = ((value - experiment) / sigma)^2 and "dchi2" (sigma None: 1).

        Raises
        ------
        ValueError
            If no ParameterSpace is available for theta_fn.
        """
        r = self.result
        if theta_fn is None:
            g, ge = r.grad, r.grad_err
        else:
            space = space or self.space
            if space is None:
                raise ValueError("a ParameterSpace of the run's table is needed for theta_fn")
            g, ge = r.chain(self._J(theta_fn, theta, space))
        out = {"value": r.value, "value_err": r.value_err, "grad": g, "grad_err": ge, "name": self.name}
        if self.experiment is not None:
            s = self.sigma or 1.0
            out["chi2"] = ((r.value - self.experiment) / s) ** 2
            out["dchi2"] = 2.0 * (r.value - self.experiment) / s**2 * np.asarray(g)
        return out

    def estimate(
        self,
        theta_fn: Callable[[jax.Array], dict],
        theta: ArrayLike,
        space: ParameterSpace | None = None,
        unit: str = "kJ/mol",
    ) -> dict[str, Any]:
        """Return the target in the layout of a multi-target fit (one observable, n parameters).

        Parameters
        ----------
        theta_fn : callable
            theta -> parameter table of the run's system.
        theta : ArrayLike (n,)
            The fitted parameters.
        space : ParameterSpace, optional
            Layout of the flat table (None: self.space).
        unit : {"kJ/mol", "kcal/mol"}
            Unit of the returned values.

        Returns
        -------
        dict
            "names", y (1,), J (1, n), cov_y (1, 1), J_err (1, n), the delete-one-block replicates
            loo = {"y": (B, 1), "J": (B, 1, n)} (jackknife: cov = (B - 1)/B sum (x_b - mean)^2),
            "target" (1,) (nan without experiment), "tol" (1,) (sigma or 1), "unit".
        """
        c = 1.0 / KCAL if unit == "kcal/mol" else 1.0
        r = self.result
        J = self._J(theta_fn, theta, space or self.space)
        g = c * (r.grad @ J)
        jy = c * np.asarray(r.jk_value, float)[:, None]
        jJ = c * (np.asarray(r.jk_grad, float) @ J)[:, None, :]
        return {
            "names": [self.name or "dG"],
            "y": np.array([c * r.value]),
            "J": g[None, :],
            "cov_y": np.array([[(c * r.value_err) ** 2]]),
            "J_err": jackknife_error(jJ),
            "loo": {"y": jy, "J": jJ},
            "target": np.array([np.nan if self.experiment is None else c * self.experiment]),
            "tol": np.array([c * (self.sigma or 1.0)]),
            "unit": unit,
        }

    def predict(self, dp_flat: ArrayLike) -> float:
        """Return the first-order free energy [kJ/mol] after changing the flat table by dp_flat (M,)."""
        return float(self.result.value + self.result.grad @ np.asarray(dp_flat, float))


def combine(terms: Sequence[tuple[float, dict]]) -> dict[str, Any]:
    """Return the linear combination sum_i c_i X_i of results from independent runs.

    Relative free energies (c = +1, -1), transfer free energies between solvents, log P =
    -(DeltaG_solv(B) - DeltaG_solv(A)) / (RT ln 10).

    Parameters
    ----------
    terms : sequence of (float, dict)
        (c_i, value_and_grad result_i).

    Returns
    -------
    dict
        "value", "value_err", "grad", "grad_err"; errors added in quadrature (independent runs).
    """
    v = sum(c * r["value"] for c, r in terms)
    ve = math.sqrt(sum((c * r["value_err"]) ** 2 for c, r in terms))
    g = sum(c * np.asarray(r["grad"], float) for c, r in terms)
    ge = np.sqrt(sum((c * np.asarray(r["grad_err"], float)) ** 2 for c, r in terms))
    return {"value": float(v), "value_err": ve, "grad": g, "grad_err": ge}
