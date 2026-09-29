"""Fit parameters to several targets: objective, trust-region Levenberg-Marquardt steps, uncertainties.

Contents: Target (an observable with a target value and tolerance), Estimate (observables of one
run with Jacobian and errors), Objective (residuals, chi2, steps, covariance, propagation,
bootstrap).

Residuals.  For every fitted target i (a component of an observable),

    r_i = sqrt(w_i) (y_i(theta) - t_i) / s_i,     s_i^2 = sigma_tol,i^2 + sigma_stat,i^2,

with sigma_tol the tolerance given with the target (experimental / model error the fit may leave)
and sigma_stat the statistical error of y_i from the block jackknife of the current run (zero for the
exact gas-phase observables), plus a Gaussian prior (theta - theta_prior) / sigma_prior per
parameter (regularisation toward the initial values).  chi2 = |r|^2 + |prior|^2.

Step.  Gauss-Newton on the linearised residuals r + J_w d, Levenberg-Marquardt damped so that
|d / sigma_prior|_2 <= radius (the trust region, in units of the prior widths); the radius adapts
to the ratio of the achieved to the predicted chi2 decrease, measured by the next simulation.

Uncertainty.  At the fixed point the fitted parameters are a linear function of the measured
observables, d theta = -G J_w^T S dy with G = (J_w^T J_w + Sigma_prior^-1)^-1 and S = diag(sqrt(w)/s), so
the statistical covariance of the fitted parameters is

    C_theta = G J_w^T S Sigma_y S J_w G,

Sigma_y the jackknife covariance of the observables (correlated within a run).  When the weights are
the inverse statistical variances (no tolerances, no prior) this is (J^T Sigma_y^-1 J)^-1.  Any other
property p with Jacobian J_p is predicted with covariance J_p C_theta J_p^T.  `posterior_cov` = G is
the covariance if the tolerances are read as Gaussian errors of the targets (and the prior as a
prior), i.e. the full parameter uncertainty given model error; C_theta is the part due to
finite sampling only.  A block bootstrap (resampling the blocks of frames, recomputing y, J and the
step) checks C_theta including the noise of J.

Units: those of the observables (estimators.py) and of theta (params.py).

See also docs/liquid_fit.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..analysis.stats import jackknife_cov
from .estimators import GAS, LIQUID

if TYPE_CHECKING:
    from .estimators import GasPhase, LiquidSamples
    from .params import ParameterSpace


@dataclass
class Target:
    """An observable (estimators.LIQUID / GAS) with a target value and a tolerance.

    A mutable dataclass.

    Parameters
    ----------
    name : str
        Observable name (estimators.LIQUID or GAS).
    value : float or array, optional
        Target value in the observable's unit (an array of g(r) values for "rdf", on all bins or on
        the selected ones); None: evaluated and reported only.
    sigma : float or array
        Tolerance (same unit): the model / experimental error the fit may leave.
    weight : float
        Multiplies the squared residuals (e.g. 1 / number of rdf bins).
    fit : bool
        False: evaluated and propagated, not fitted.
    r_range : tuple of 2 float, optional
        For "rdf": the bins with r_range[0] <= r < r_range[1] [nm]; None: all bins.
    extra : dict
        Free-form annotations.
    """

    name: str
    value: object = None
    sigma: object = 1.0
    weight: float = 1.0
    fit: bool = True
    r_range: tuple | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Estimate:
    """Observables of one run at theta0 with their Jacobian and statistical errors (m components, n parameters).

    A mutable dataclass, made by Objective.estimate.

    Parameters
    ----------
    theta : np.ndarray (n,)
        Parameters of the run.
    names : list of str (m,)
        Component names (one per component; rdf bins as "rdf(<r>)").
    y : np.ndarray (m,)
        Observables (units of each observable).
    J : np.ndarray (m, n)
        dy/dtheta.
    cov_y : np.ndarray (m, m)
        Jackknife covariance of y.
    J_err : np.ndarray (m, n)
        Jackknife standard errors of J.
    target : np.ndarray (m,)
        Target values (nan where none).
    tol : np.ndarray (m,)
        Tolerances.
    weight : np.ndarray (m,)
        Weights.
    fit : np.ndarray (m,) bool
        Components that are fitted.
    loo : dict, optional
        Leave-one-block-out "y" (B, m) and "J" (B, m, n).
    """

    theta: np.ndarray
    names: list  # one per component
    y: np.ndarray
    J: np.ndarray  # (m, n)
    cov_y: np.ndarray  # (m, m) jackknife
    J_err: np.ndarray  # (m, n) jackknife standard errors of J
    target: np.ndarray  # (m,) nan where none
    tol: np.ndarray  # (m,)
    weight: np.ndarray  # (m,)
    fit: np.ndarray  # (m,) bool
    loo: dict = None  # leave-one-block-out y (B, m) and J (B, m, n)

    @property
    def err(self) -> np.ndarray:
        """Jackknife standard errors (m,) of y."""
        return np.sqrt(np.clip(np.diag(self.cov_y), 0.0, None))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (lists; None for missing targets; without `loo`)."""
        return {
            "theta": self.theta.tolist(),
            "names": self.names,
            "y": self.y.tolist(),
            "err": self.err.tolist(),
            "J": self.J.tolist(),
            "J_err": self.J_err.tolist(),
            "target": [None if not np.isfinite(t) else t for t in self.target.tolist()],
            "tol": self.tol.tolist(),
            "fit": self.fit.tolist(),
            "cov_y": self.cov_y.tolist(),
        }


class Objective:
    """Multi-target objective of a fit: residuals, chi2, trust-region steps and uncertainties.

    The formulas are in the module docstring.  Liquid observables come from LiquidSamples
    (reweighting), gas-phase ones from GasPhase (exact).  Not a pytree.

    Attributes
    ----------
    targets : list of Target
    space : ParameterSpace
    gas : GasPhase or None
    conservative : bool
        Inflate jackknife variances when halving the number of blocks increases them.
    prior_center : np.ndarray (n,)
        Centre of the Gaussian prior on theta.
    prior_sigma : np.ndarray (n,)
        Prior widths (from the space).
    rdf_r : np.ndarray or None
        Bin centres of the g(r) [nm].
    """

    def __init__(
        self,
        targets: list[Target],
        space: ParameterSpace,
        gas: GasPhase | None = None,
        prior_center: ArrayLike | None = None,
        rdf_r: ArrayLike | None = None,
        conservative: bool = True,
    ) -> None:
        """Set up the objective.

        Parameters
        ----------
        targets : list of Target
            The targets, in the order of the residuals.
        space : ParameterSpace
            The fitted parameters (its prior_sigma are the prior widths).
        gas : GasPhase, optional
            Gas-phase model; required for gas-phase targets and "hvap".
        prior_center : ArrayLike (n,), optional
            Centre of the prior; None: zeros (theta0 of a scale space).
        rdf_r : ArrayLike (bins,), optional
            Bin centres of the g(r) [nm] (RDFSpec.r); needed for rdf targets.
        conservative : bool
            Inflate jackknife variances (see estimate).

        Raises
        ------
        ValueError
            Gas-phase targets or hvap without `gas`.
        KeyError
            An unknown observable.
        """
        self.targets, self.space, self.gas = targets, space, gas
        self.conservative = bool(conservative)
        self.prior_center = np.zeros(space.n) if prior_center is None else np.asarray(prior_center, float)
        self.prior_sigma = np.asarray(space.prior_sigma, float)
        self.rdf_r = rdf_r
        if any(t.name in GAS or t.name == "hvap" for t in targets) and gas is None:
            raise ValueError("gas-phase targets and hvap need a GasPhase")
        for t in targets:
            if t.name not in LIQUID + GAS:
                raise KeyError(f"unknown observable {t.name!r}")

    # ------------------------------------------------------------------ model y(theta0 + delta)
    def _sel(self, t: Target) -> np.ndarray | None:
        """Return the selected rdf bin indices of a target, or None (not rdf, or no r_range)."""
        if t.name != "rdf" or t.r_range is None:
            return None
        r = np.asarray(self.rdf_r)
        return np.flatnonzero((r >= t.r_range[0]) & (r < t.r_range[1]))

    def model(
        self,
        samples: LiquidSamples | None,
        theta0: ArrayLike,
        delta: jax.Array,
        frame_weights: ArrayLike | None = None,
    ) -> jax.Array:
        """Return every target's observable at theta0 + delta, concatenated (m,).

        Liquid observables by linear-exponential reweighting of the frames of `samples` (None: no
        liquid targets), gas-phase ones exactly.  Differentiable in `delta`.

        Parameters
        ----------
        samples : LiquidSamples, optional
            The run's per-frame data.
        theta0 : ArrayLike (n,)
            Parameters of the run.
        delta : jax.Array (n,)
            Parameter step.
        frame_weights : ArrayLike (F,), optional
            Jackknife / bootstrap frame weights.
        """
        avg = samples.averages(delta, frame_weights) if samples is not None else None
        gas = self.gas(jnp.asarray(theta0) + delta) if self.gas is not None else None
        return self._assemble(samples, avg, gas)

    def _assemble(self, samples: LiquidSamples | None, avg: dict | None, gas: dict | None) -> jax.Array:
        """Return the concatenated target components from liquid averages `avg` and gas-phase values `gas`."""
        out = []
        for t in self.targets:
            if t.name in GAS:
                v = gas[t.name]
            else:
                v = samples.observable(t.name, avg, gas)
            v = jnp.atleast_1d(v)
            s = self._sel(t)
            out.append(v if s is None else v[s])
        return jnp.concatenate(out)

    def layout(
        self, samples: LiquidSamples | None = None
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the component names, targets, tolerances, weights and fit flags (m,) of the targets.

        A component without a target value is not fitted.  `samples` is not used.
        """
        names, tg, tol, w, fit = [], [], [], [], []
        for t in self.targets:
            if t.name == "rdf":
                s = self._sel(t)
                r = np.asarray(self.rdf_r) if s is None else np.asarray(self.rdf_r)[s]
                m = len(r)
                names += [f"rdf({x:.3f})" for x in r]
                val = np.full(m, np.nan) if t.value is None else np.asarray(t.value, float).reshape(-1)
                if t.value is not None and s is not None and len(val) == len(self.rdf_r):
                    val = val[s]
                tg += list(np.broadcast_to(val, (m,)))
                tol += list(np.broadcast_to(np.asarray(t.sigma, float), (m,)))
                w += [t.weight] * m
                fit += [t.fit] * m
            else:
                names.append(t.name)
                tg.append(np.nan if t.value is None else float(t.value))
                tol.append(float(t.sigma))
                w.append(float(t.weight))
                fit.append(bool(t.fit) and t.value is not None)
        return names, np.array(tg, float), np.array(tol, float), np.array(w, float), np.array(fit, bool)

    def estimate(self, samples: LiquidSamples | None, theta0: ArrayLike, keep_loo: bool = True) -> Estimate:
        """Return the observables of one run with their Jacobian and jackknife errors.

        Parameters
        ----------
        samples : LiquidSamples, optional
            The run (None: gas-phase targets only, zero errors).
        theta0 : ArrayLike (n,)
            Parameters of the run.
        keep_loo : bool
            Keep the leave-one-block-out values in Estimate.loo.

        Returns
        -------
        Estimate

        Notes
        -----
        y and J = jax.jacfwd of `model` at delta = 0; the covariance is the delete-one-block jackknife
        over samples.nblocks blocks.  With `conservative` and at least 8 blocks, the jackknife is
        repeated with half as many (twice as long) blocks; where a variance grows, the covariance rows
        and columns are scaled up by sqrt of the ratio (correlations kept), and J_err takes the larger
        of the two estimates.
        """
        theta0 = np.asarray(theta0, float)
        z = jnp.zeros(self.space.n)
        f = jax.jit(lambda d, fw: self.model(samples, theta0, d, fw))
        jf = jax.jit(jax.jacfwd(lambda d, fw: self.model(samples, theta0, d, fw)))
        ones = jnp.ones(samples.F) if samples is not None else None
        y, J = np.asarray(f(z, ones)), np.asarray(jf(z, ones))
        names, tg, tol, w, fit = self.layout()
        if samples is not None:
            W = samples.blocks_weights()
            yl = np.array([np.asarray(f(z, jnp.asarray(wb))) for wb in W])
            Jl = np.array([np.asarray(jf(z, jnp.asarray(wb))) for wb in W])
            B = len(W)
            cov = jackknife_cov(yl)
            J_err = np.sqrt((B - 1) / B * np.sum((Jl - Jl.mean(0)) ** 2, axis=0))
            if self.conservative and B >= 8:
                # blocks twice as long: if the variances grow, the blocks were not longer than the
                # correlation time (the dipole's ~10 ps): scale up, keeping the correlations
                Wh = samples.blocks_weights(B // 2)
                yh = np.array([np.asarray(f(z, jnp.asarray(wb))) for wb in Wh])
                Jh = np.array([np.asarray(jf(z, jnp.asarray(wb))) for wb in Wh])
                vh, v = np.diag(jackknife_cov(yh)), np.diag(cov)
                c = np.sqrt(np.maximum(1.0, np.where(v > 0, vh / np.where(v > 0, v, 1.0), 1.0)))
                cov = cov * c[:, None] * c[None, :]
                Bh = len(Wh)
                J_err = np.maximum(J_err, np.sqrt((Bh - 1) / Bh * np.sum((Jh - Jh.mean(0)) ** 2, axis=0)))
        else:
            yl, Jl = None, None
            cov, J_err = np.zeros((len(y), len(y))), np.zeros_like(J)
        return Estimate(
            theta0,
            names,
            y,
            J,
            cov,
            J_err,
            tg,
            tol,
            w,
            fit,
            {"y": yl, "J": Jl} if keep_loo and yl is not None else None,
        )

    # ------------------------------------------------------------------ fitting
    def scales(self, est: Estimate) -> tuple[np.ndarray, np.ndarray]:
        """Return s = sqrt(tol^2 + stat^2) (m,) and the residual row factors sqrt(w)/s (0 for unfitted components)."""
        s = np.sqrt(est.tol**2 + np.diag(est.cov_y))
        return s, np.where(est.fit, np.sqrt(est.weight) / s, 0.0)

    def chi2(self, y: ArrayLike, est: Estimate, theta: ArrayLike) -> tuple[float, float]:
        """Return (|r|^2, |prior|^2) at observables `y` (m,) and parameters `theta` (n,).

        r_i = sqrt(w_i) (y_i - t_i) / s_i on the fitted components; prior = (theta - prior_center) /
        prior_sigma.
        """
        _, f = self.scales(est)
        r = np.where(est.fit, (np.asarray(y) - np.nan_to_num(est.target)) * f, 0.0)
        p = (np.asarray(theta) - self.prior_center) / self.prior_sigma
        return float(r @ r), float(p @ p)

    def _lm(self, Jw: np.ndarray, c: np.ndarray, th: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
        """Solve the trust-region subproblem of the linearised residuals with the prior.

        Minimises |c + Jw x|^2 + |(th + x - prior)/sigma_prior|^2 subject to |x / sigma_prior| <= radius.

        Parameters
        ----------
        Jw : np.ndarray (m_fit, n)
            Weighted Jacobian of the fitted residuals.
        c : np.ndarray (m_fit,)
            Residuals at x = 0.
        th : np.ndarray (n,)
            Current parameters.
        radius : float
            Trust radius in units of the prior widths.

        Returns
        -------
        x : np.ndarray (n,)
            The step.
        lambda : float
            Levenberg-Marquardt damping (0 if the Gauss-Newton step is inside the region).

        Notes
        -----
        The damping is in the metric of the trust region, x(lambda) = -(A + lambda Sigma_prior^-1)^-1 g,
        and lambda is bracketed (upper bound x 4 until inside) then found by 60 bisections on
        |x(lambda)| = radius; the returned step is the one at the upper bracket (inside the region).
        """
        Pinv = np.diag(1.0 / self.prior_sigma**2)
        A = Jw.T @ Jw + Pinv
        g = Jw.T @ c + Pinv @ (th - self.prior_center)
        # damping in the metric of the trust region (|d / sigma_prior|): the exact solution of the
        # trust-region subproblem is d(lambda) = -(A + lambda Sigma_prior^-1)^-1 g (More-Sorensen); a
        # Marquardt diag(A) damping would push the step into the directions of small curvature
        D = Pinv

        def size(d: np.ndarray) -> float:
            return float(np.linalg.norm(d / self.prior_sigma))

        x = -np.linalg.solve(A, g)
        if size(x) <= radius:
            return x, 0.0
        lo, hi = 0.0, 1.0
        while size(-np.linalg.solve(A + hi * D, g)) > radius:
            hi *= 4.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            (lo, hi) = (mid, hi) if size(-np.linalg.solve(A + mid * D, g)) > radius else (lo, mid)
        return -np.linalg.solve(A + hi * D, g), hi

    def gas_rows(self, theta: ArrayLike) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the exact values (m,) and Jacobian (m, n) of the gas-phase components (layout order).

        Rows of other components are 0.  Returns None without a gas-phase model.  The value and
        Jacobian functions are jitted once and cached on the object.
        """
        if self.gas is None:
            return None
        if getattr(self, "_gas_jit", None) is None:

            def f(th: jax.Array) -> jax.Array:
                """Return the gas-phase component values in layout order (zeros for the other components)."""
                g = self.gas(th)
                out = []
                for t in self.targets:
                    v = jnp.atleast_1d(g[t.name]) if t.name in GAS else jnp.zeros(len(np.atleast_1d(self._n_of(t))))
                    out.append(v)
                return jnp.concatenate(out)

            self._gas_jit = (jax.jit(f), jax.jit(jax.jacfwd(f)))
        th = jnp.asarray(theta, float)
        return np.asarray(self._gas_jit[0](th)), np.asarray(self._gas_jit[1](th))

    def _n_of(self, t: Target) -> int | np.ndarray:
        """Return a placeholder whose length (via np.atleast_1d) is the number of components of target `t`."""
        if t.name != "rdf":
            return 1
        s = self._sel(t)
        return np.zeros(len(self.rdf_r) if s is None else len(s))

    def step(
        self,
        est: Estimate,
        radius: float = 1.0,
        J: np.ndarray | None = None,
        y: np.ndarray | None = None,
        exact_gas: bool = True,
        iters: int = 30,
    ) -> dict[str, Any]:
        """Return a Levenberg-Marquardt step within the trust region and its predictions.

        The liquid observables are linearised (y + J d); with exact_gas the gas-phase ones (exact,
        cheap) are kept nonlinear: Gauss-Newton iterations on the mixed model inside the trust region,
        until the step changes by less than 1e-8 (prior units) or `iters` iterations.

        Parameters
        ----------
        est : Estimate
            Observables of the current run.
        radius : float
            Trust radius, |d / sigma_prior| <= radius.
        J, y : np.ndarray, optional
            Jacobian (m, n) and values (m,) replacing est.J, est.y (bootstrap).
        exact_gas : bool
            Keep gas-phase observables nonlinear.
        iters : int
            Largest number of Gauss-Newton iterations with exact gas-phase rows.

        Returns
        -------
        dict
            "delta" (n,) the step; "lambda" the damping; "y_pred" (m,) predicted observables;
            "chi2_pred" (residual, prior) chi2 at theta + delta; "size" |delta / sigma_prior|;
            "at_boundary" whether the step is limited by the trust region.
        """
        J = est.J if J is None else J
        y = est.y if y is None else y
        _, f = self.scales(est)
        fit = est.fit
        th = est.theta
        t = np.nan_to_num(est.target)
        is_gas = np.array([n in GAS for n in est.names])
        use_gas = exact_gas and self.gas is not None and bool(np.any(is_gas & fit))
        d = np.zeros(len(th))
        lam = 0.0
        for _ in range(iters if use_gas else 1):
            Jm, ym = J, y + J @ d
            if use_gas:
                yg, Jg = self.gas_rows(th + d)
                ym = np.where(is_gas, yg, ym)
                Jm = np.where(is_gas[:, None], Jg, J)
            Jw = (Jm * f[:, None])[fit]
            c = ((ym - t) * f)[fit] - Jw @ d
            d_new, lam = self._lm(Jw, c, th, radius)
            done = np.linalg.norm((d_new - d) / self.prior_sigma) < 1e-8
            d = d_new
            if done:
                break
        y_pred = y + J @ d
        if use_gas:
            y_pred = np.where(is_gas, self.gas_rows(th + d)[0], y_pred)
        size = float(np.linalg.norm(d / self.prior_sigma))
        return {
            "delta": d,
            "lambda": lam,
            "y_pred": y_pred,
            "chi2_pred": self.chi2(y_pred, est, th + d),
            "size": size,
            "at_boundary": lam > 0.0,
        }

    def covariance(self, est: Estimate) -> dict[str, np.ndarray]:
        """Return the sampling covariance of the fitted parameters and the posterior-like covariance.

        Returns
        -------
        dict
            "C_theta" (n, n) = G J_w^T S Sigma_y S J_w G (sampling only); "G" (n, n) =
            (J_w^T J_w + Sigma_prior^-1)^-1; "theta_err" and "theta_err_posterior" (n,) their square-root
            diagonals (module docstring).
        """
        _, f = self.scales(est)
        fit = est.fit
        Jw = (est.J * f[:, None])[fit]
        S = f[fit]
        Sy = est.cov_y[np.ix_(fit, fit)]
        G = np.linalg.inv(Jw.T @ Jw + np.diag(1.0 / self.prior_sigma**2))
        C = G @ Jw.T @ (S[:, None] * Sy * S[None, :]) @ Jw @ G
        return {
            "C_theta": C,
            "G": G,
            "theta_err": np.sqrt(np.clip(np.diag(C), 0, None)),
            "theta_err_posterior": np.sqrt(np.diag(G)),
        }

    def propagate(self, est: Estimate, C: np.ndarray) -> np.ndarray:
        """Return the standard deviations (m,) of all components of est implied by a parameter covariance.

        sqrt(diag(J C J^T)) with C (n, n), e.g. C_theta or G of `covariance`.
        """
        return np.sqrt(np.clip(np.einsum("mi,ij,mj->m", est.J, C, est.J), 0.0, None))

    def bootstrap(
        self, samples: LiquidSamples, est: Estimate, radius: float | None, n: int = 200, seed: int = 0
    ) -> dict[str, Any]:
        """Return the block-bootstrap spread of the observables and of the fitted parameters theta + d.

        Each of `n` resamples of the blocks recomputes y, J and the step (radius None: no trust region).

        Returns
        -------
        dict
            "theta_sd" (n_par,), "theta_cov" (n_par, n_par), "y_sd" (m,) and "n".
        """
        rng = np.random.default_rng(seed)
        z = jnp.zeros(self.space.n)
        f = jax.jit(lambda d, fw: self.model(samples, est.theta, d, fw))
        jf = jax.jit(jax.jacfwd(lambda d, fw: self.model(samples, est.theta, d, fw)))
        th, ys = [], []
        for wb in samples.bootstrap_weights(rng, n):
            y = np.asarray(f(z, jnp.asarray(wb)))
            J = np.asarray(jf(z, jnp.asarray(wb)))
            st = self.step(est, radius=np.inf if radius is None else radius, J=J, y=y)
            th.append(est.theta + st["delta"])
            ys.append(y)
        th, ys = np.array(th), np.array(ys)
        return {
            "theta_sd": th.std(0, ddof=1),
            "theta_cov": np.cov(th.T).reshape(self.space.n, self.space.n),
            "y_sd": ys.std(0, ddof=1),
            "n": n,
        }
