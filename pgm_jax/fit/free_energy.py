"""Free energies with parameter gradients as fitting targets: estimators, statistical errors,
chain rule to any parameterization (the samples come from pgm_jax/md/fe_grad.ParameterGradients
through alchemy.FreeEnergyRun(param_grad=...); the identity d f_k/dtheta = <dU_k/dtheta>_k is
explained there).

Estimators (gradient_estimate):
  "end":  the end windows' own samples, <dU_{K-1}/dtheta> over window K-1 minus <dU_0/dtheta> over
          window 0 (the identity above, directly);
  "mbar": the same two expectations with MBAR weights over the samples of every window,
          E_k[A] = sum_n W_nk A_k(x_n) (Shirts & Chodera, JCP 129, 124105 (2008)): this is exactly
          the derivative of the MBAR estimate of f_k(theta) for states k(theta) reweighted from
          the sampled mixture, so it is consistent with the MBAR free energy.
Errors: block jackknife over time (n_blocks contiguous blocks, the same time block removed from
every window, which keeps the correlation between windows coupled by Hamiltonian exchange and
along each window's time series inside a block); the jackknife replicates are kept so that errors
of any linear combination of gradient entries (chain rule, scale directions) are exact.

FEGradient holds one free energy with its gradient and jackknife replicates; FreeEnergyTarget
turns it into a chi^2 term of a fit (value, gradient, linear prediction); combine() sums several.

Units: kJ/mol; gradients in kJ/mol per unit of each parameter-table entry."""

from __future__ import annotations

import dataclasses as _dc
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

from ..analysis import free_energy as fe
from ..analysis.stats import jackknife_error
from ..units import KCAL
from .params import ParameterSpace


# ----------------------------------------------------------------------------- estimators
@_dc.dataclass
class FEGradient:
    """A free energy (kJ/mol), its gradient over the parameter entries `names` (kJ/mol per unit), their
    jackknife standard errors and the replicates (jk_value (B,), jk_grad (B, M)) for exact errors of
    projections and chain-rule products."""

    value: float
    value_err: float
    grad: np.ndarray
    grad_err: np.ndarray
    names: list
    jk_value: np.ndarray
    jk_grad: np.ndarray
    estimator: str = "mbar"

    def project(self, v) -> tuple:
        """(grad . v, standard error): e.g. v = ParameterSpace.scale_direction(...) gives dG/d ln s."""
        v = np.asarray(v, float)
        return float(self.grad @ v), float(jackknife_error(self.jk_grad @ v))

    def chain(self, J) -> tuple:
        """(dG/dtheta (n,), standard errors (n,)) for J = dP_flat/dtheta (M, n)."""
        J = np.asarray(J, float).reshape(len(self.names), -1)
        return self.grad @ J, jackknife_error(self.jk_grad @ J)

    def kcal(self) -> dict:
        return {"value": self.value / KCAL, "value_err": self.value_err / KCAL}

    def named(self, indices=None) -> dict:
        idx = range(len(self.names)) if indices is None else indices
        return {self.names[i]: (float(self.grad[i]), float(self.grad_err[i])) for i in idx}


def _subset(S, discard_ps, end_ps, stride):
    t = np.asarray(S["time_ps"], float)
    last = np.inf if end_ps is None else float(end_ps) + 1e-9
    keep = np.nonzero((t > float(discard_ps) + 1e-9) & (t <= last))[0][:: max(int(stride), 1)]
    return keep


def gradient_estimate(
    samples, discard_ps: float = 0.0, gas=None, n_blocks: int = 10, end_ps: float | None = None, stride: int = 1
) -> dict:
    """Free energy of the solution leg (window 0 -> K-1) and of hydration, with their parameter
    gradients, from FreeEnergyRun samples with parameter gradients (a dict or free_energy.load of
    prefix_fe.npz; needs 'dudp' (S, T, K, M), meta dudp_targets containing 0 and K-1, dudp_names).

    gas: None, or {"delta_g": Delta G_gas(1 -> 0), "grad": d Delta G_gas/dP (M,)} (kJ/mol; a rigid
    solute's exact leg, gas_leg_gradient; zeros with intramolecular="keep", where the hydration free
    energy is -Delta G_solv).  Returns {"solv": {"mbar": FEGradient, "end": FEGradient},
    "hyd": {...} (if gas), "names", "samples_per_window", "n_blocks", "end_means": (<dU_0/dP>_0,
    <dU_{K-1}/dP>_{K-1}) and "end_means_err"}.  The value of the "end" entries is the MBAR
    free energy too (the end-state estimator is for the gradient only).  Errors: block jackknife."""
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

    f_start = [None]

    def estimate(idx):
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
    """A free energy (e.g. hydration) as a fitting target: value, gradient and statistical errors at
    the sampled parameters, the chain rule to any parameterization theta -> table, a linear
    prediction for nearby theta, and a chi^2 term against experiment.

        t = FreeEnergyTarget.from_npz("wat_fe.npz", discard_ps=200, experiment=-6.3 * 4.184, sigma=0.2 * 4.184)
        r = t.value_and_grad(theta_fn, theta)      # theta_fn(theta) -> parameter table of the run's system
        r["value"], r["grad"], r["value_err"], r["grad_err"], r["chi2"], r["dchi2"]

    theta_fn(theta) must reproduce the sampled table at the theta of the run (checked); combined
    with alchemical_map(sys0, sysA) it can act on the original (non-alchemical) table."""

    def __init__(
        self,
        result: FEGradient,
        params_flat,
        space_names,
        experiment: float | None = None,
        sigma: float | None = None,
        name: str = "",
        space: ParameterSpace | None = None,
    ):
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
        samples,
        discard_ps=0.0,
        estimator="mbar",
        leg=None,
        n_blocks=10,
        experiment=None,
        sigma=None,
        name="",
        gas=None,
    ):
        S = dict(samples)
        meta = S.get("meta", {})
        if gas is None and meta.get("gas_grad") is not None:
            gas = {"delta_g": meta["gas_delta_g"], "grad": meta["gas_grad"]}
        r = gradient_estimate(S, discard_ps=discard_ps, gas=gas, n_blocks=n_blocks)
        leg = leg or ("hyd" if "hyd" in r else "solv")
        return cls(r[leg][estimator], meta["params_flat"], r["names"], experiment, sigma, name)

    @classmethod
    def from_npz(cls, path, **kw):
        return cls.from_samples(fe.load(path), **kw)

    def _J(self, theta_fn, theta, space):
        def flat(th):
            return space.flatten(theta_fn(th))

        th = jnp.asarray(theta, jnp.float64)
        p = np.asarray(flat(th), float)
        if p.shape != self.p.shape or not np.allclose(p, self.p, rtol=1e-8, atol=1e-12):
            raise ValueError("theta_fn(theta) is not the parameter table of the sampled run")
        return np.asarray(jax.jacfwd(flat)(th), float).reshape(len(p), -1)

    def value_and_grad(self, theta_fn=None, theta=None, space: ParameterSpace | None = None) -> dict:
        """Value, gradient (per theta, or per table entry without theta_fn), their standard errors
        and, with an experiment, chi2 = ((value - experiment) / sigma)^2 and its gradient."""
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

    def estimate(self, theta_fn, theta, space: ParameterSpace | None = None, unit: str = "kJ/mol") -> dict:
        """The target in the layout of a multi-target fit (one observable, n parameters): y (1,), J
        (1, n), cov_y (1, 1), J_err (1, n) and the delete-one-block replicates loo = {"y": (B, 1),
        "J": (B, 1, n)} (jackknife: cov = (B - 1)/B sum (x_b - mean)^2), in kJ/mol or kcal/mol."""
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

    def predict(self, dp_flat) -> float:
        """First-order prediction of the free energy after changing the table by dp_flat (kJ/mol)."""
        return float(self.result.value + self.result.grad @ np.asarray(dp_flat, float))


def combine(terms) -> dict:
    """Linear combination sum_i c_i X_i of results from independent runs (value_and_grad dicts):
    relative free energies (c = +1, -1), transfer free energies between solvents, log P =
    -(DeltaG_solv(B) - DeltaG_solv(A)) / (RT ln 10).  terms: [(c_i, result_i)]; errors in quadrature."""
    v = sum(c * r["value"] for c, r in terms)
    ve = math.sqrt(sum((c * r["value_err"]) ** 2 for c, r in terms))
    g = sum(c * np.asarray(r["grad"], float) for c, r in terms)
    ge = np.sqrt(sum((c * np.asarray(r["grad_err"], float)) ** 2 for c, r in terms))
    return {"value": float(v), "value_err": ve, "grad": g, "grad_err": ge}
