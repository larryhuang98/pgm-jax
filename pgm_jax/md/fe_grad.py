"""Parameter gradients of alchemical free energies: hydration (solvation) free energies as
fitting targets, with statistical errors.

Thermodynamic identity.  For a Hamiltonian U_k(x; theta) sampled at fixed volume and temperature,
f_k(theta) = -kT ln Z_k(theta) has

    df_k / dtheta = < dU_k / dtheta >_k ,

so the free energy of the solution leg (window 0 = full coupling -> window K-1 = decoupled) has

    d DeltaG_solv / dtheta = < dU_{K-1}/dtheta >_{K-1} - < dU_0/dtheta >_0

and the hydration free energy DeltaG_hyd = DeltaG_gas - DeltaG_solv (alchemy.py) has
d DeltaG_hyd / dtheta = d DeltaG_gas / dtheta - d DeltaG_solv / dtheta.  The gas-phase leg of a
rigid solute is exact (E_gas(0) - E_gas(1), gradient by autodiff: gas_leg_gradient); with
intramolecular="keep" it is part of the Hamiltonian (the decoupled end state then still depends on
the solute's electrostatic parameters through the gas-phase correction, and the identity takes
care of it).  Only the two end states enter; the intermediate windows help only through MBAR.

dU_k/dtheta at a configuration is the partial derivative at the induced dipoles converged in
Hamiltonian k: the pGM energy is stationary in the dipoles (Hellmann-Feynman), so no derivative
of the solve is needed (ParamGradients: one dipole solve and one reverse-mode pass per target
Hamiltonian and configuration, batched over the windows with jax.vmap, on the device).  theta is
the parameter table (every entry of sys.table, flattened: ParamSpace); gradients with respect to
any other parameters theta' follow by the chain rule dG/dtheta' = (dP/dtheta')^T dG/dP
(FEGradient.chain, FreeEnergyTarget.value_and_grad).

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

Cost: per sample, 2 x K dipole re-solves and 2 x K reverse passes of the energy (K windows, two end
states); see docs/fe_gradients.md.

Units: kJ/mol, nm, e, e nm; gradients in kJ/mol per unit of each table entry."""

from __future__ import annotations

import dataclasses as _dc
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

from ..analysis.stats import jackknife_error
from ..system import QUANTITIES
from ..units import KCAL
from . import free_energy as fe
from .alchemy import PREFIX

# scale groups: parameter quantities scaled together and the exponent of the scale on each
# (lj_sqrt_eps carries sqrt(eps), so scaling eps by s scales it by s^1/2)
SCALE_GROUPS = {
    "charge": {"q": 1.0, "cov": 1.0},
    "eps": {"lj_sqrt_eps": 0.5},
    "rmin": {"lj_rmin_half": 1.0},
    "alpha": {"alpha": 1.0},
    "radius": {"radius": 1.0},
}


# ----------------------------------------------------------------------------- parameter space
class ParamSpace:
    """The parameter table as one flat vector: the entries of the listed quantities (default every
    quantity with at least one key), named 'quantity:key' (e.g. 'q:alch:WAT:OW')."""

    def __init__(self, table, quantities=None):
        qs = tuple(QUANTITIES if quantities is None else quantities)
        bad = [q for q in qs if q not in QUANTITIES]
        if bad:
            raise ValueError(f"unknown parameter quantities {bad}")
        self.quantities = [q for q in qs if len(table.keys[q])]
        self.keys = {q: list(table.keys[q]) for q in self.quantities}
        self._finish()

    @classmethod
    def from_names(cls, names):
        """The space of stored samples ('quantity:key' names in table order)."""
        self = cls.__new__(cls)
        self.quantities, self.keys = [], {}
        for nm in names:
            q, k = nm.split(":", 1)
            if q not in self.keys:
                self.quantities.append(q)
                self.keys[q] = []
            self.keys[q].append(k)
        self._finish()
        return self

    def _finish(self):
        self.names = [f"{q}:{k}" for q in self.quantities for k in self.keys[q]]
        self.n = len(self.names)
        off = np.cumsum([0] + [len(self.keys[q]) for q in self.quantities])
        self.slices = {q: slice(int(a), int(b)) for q, a, b in zip(self.quantities, off[:-1], off[1:])}

    def flatten(self, P):
        """(n,) float64 vector of the table P (a dict by quantity; JAX-differentiable)."""
        return jnp.concatenate([jnp.ravel(jnp.asarray(P[q], jnp.float64)) for q in self.quantities])

    def unflatten(self, v, like=None) -> dict:
        """Table dict from a flat vector (the quantities not in the space taken from `like`)."""
        out = dict(like) if like is not None else {}
        for q in self.quantities:
            out[q] = v[self.slices[q]]
        return out

    def index(self, name: str) -> int:
        return self.names.index(name)

    def select(self, quantities=None, solute: bool | None = None) -> np.ndarray:
        """Indices of the entries of the given quantities; solute True / False: only the alchemical
        solute's own keys ('alch:' prefix) / only the others (the environment)."""
        out = []
        for q in self.quantities:
            if quantities is not None and q not in quantities:
                continue
            for j, k in enumerate(self.keys[q]):
                own = k.startswith(PREFIX)
                if solute is None or own == bool(solute):
                    out.append(self.slices[q].start + j)
        return np.array(out, int)

    def scale_direction(self, p_flat, group: str, solute: bool | None = True) -> np.ndarray:
        """v with dG/d ln s = grad . v for scaling the parameters of `group` (SCALE_GROUPS: charge =
        charges and covalent dipoles, eps, rmin, alpha, radius) by s: v_i = e_i p_i, e_i the
        exponent of the scale on entry i (1/2 for sqrt(eps))."""
        v = np.zeros(self.n)
        p = np.asarray(p_flat, float)
        for q, e in SCALE_GROUPS[group].items():
            i = self.select((q,), solute)
            v[i] = e * p[i]
        return v


def scaled_params(space: ParamSpace, P, scales: dict, solute: bool | None = True) -> dict:
    """The table P with the parameters of each scale group multiplied (charge, rmin, alpha, radius by
    s; eps by s, i.e. sqrt(eps) by s^1/2): scales = {'charge': 1.05, 'eps': 0.9}; solute as in
    ParamSpace.select.  JAX-differentiable in the scales."""
    out = {q: jnp.asarray(P[q], jnp.float64) for q in P}
    for g, s in scales.items():
        if g not in SCALE_GROUPS:
            raise ValueError(f"unknown scale group {g!r} (one of {sorted(SCALE_GROUPS)})")
        for q, e in SCALE_GROUPS[g].items():
            if q not in space.slices:
                continue
            m = np.zeros(out[q].shape, bool)
            m[space.select((q,), solute) - space.slices[q].start] = True
            out[q] = out[q] * jnp.where(m, jnp.asarray(s, jnp.float64) ** e, 1.0)
    return out


def alchemical_map(sys0, sysA):
    """P0 -> PA: the parameter table of the original system (sys0.table) mapped onto the table of
    alchemical_system(sys0, k) (the solute's 'alch:' keys take the values of the keys they were
    copied from).  A JAX gather, so gradients of PA-functions with respect to P0 sum the solute's
    copy and the original key (e.g. water in water: the solute and the solvent share the model)."""
    idx = {}
    for q in QUANTITIES:
        pos = {k: i for i, k in enumerate(sys0.table.keys[q])}
        idx[q] = np.array([pos[k[len(PREFIX) :] if k.startswith(PREFIX) else k] for k in sysA.table.keys[q]], int)

    def f(P0):
        return {q: jnp.asarray(P0[q])[idx[q]] for q in QUANTITIES}

    return f


# ----------------------------------------------------------------------------- sampling
def _frame(windows, st):
    """(atom positions, candidate rows, overflow) of one window state (rigid or flexible engine)."""
    integ = windows.integ
    if hasattr(integ, "flex"):
        pos = st.dyn.position
        centers = integ.flex.list_centers(pos)
    else:
        pos = windows.sim.rigid.positions(st.dyn.position)
        centers = st.dyn.position.center
    cand, ovf = integ.nb.candidates(st.nbr, centers, st.box, pos)
    return pos, cand, ovf


class ParamGradients:
    """dU_k/dP of target Hamiltonians k (window indices, default the two end states 0 and K-1) at
    the configuration of every window, for the samples of alchemy.FreeEnergyRun(param_grad=...).

        run = FreeEnergyRun(windows, sample_every=500, exchange_every=500,
                            param_grad=ParamGradients(windows))

    sample() -> (T, K, M): [t, n] = dU_{targets[t]}/dP at the configuration of window n (kJ/mol per
    unit of each of the M entries of `space`), at the induced dipoles re-solved in Hamiltonian
    targets[t] from the configuration's own (Hellmann-Feynman).  Batched windows: one vmapped
    program over the windows (T target Hamiltonians by lax.map inside)."""

    def __init__(self, windows, targets=(0, -1), quantities=None):
        K = windows.n
        t = [int(x) % K for x in targets]
        if len(set(t)) != len(t) or not t:
            raise ValueError("targets: distinct window indices")
        self.windows, self.targets = windows, np.array(t, int)
        self.space = ParamSpace(windows.alchemy.sys.table, quantities)
        self._fns = {}

    @property
    def params(self):
        integ = self.windows.integ
        return integ.params if integ.params is not None else self.windows.alchemy.params0

    def _one(self, st, lam_t):
        w = self.windows
        ff, alch = w.sim.ff, w.alchemy
        params = self.params
        pos, cand, ovf0 = _frame(w, st)

        def one(lam):
            _, ind, it, ovf = alch.energy(ff, pos, st.box, cand, st.induction, params, lam)
            g = jax.grad(lambda P: alch.energy_fixed_mu(ff, pos, st.box, cand, ind.mu, P, lam))(params)
            return self.space.flatten(g), it, ovf

        G, it, ovf = jax.lax.map(one, lam_t)
        return G, jnp.max(it), jnp.any(ovf) | ovf0

    def _fn(self):
        w = self.windows
        key = (w._sizes(), jax.tree_util.tree_structure(w.S) if w.batched else None)
        if key not in self._fns:
            if w.batched:
                from .remd import _axes

                f = jax.vmap(self._one, in_axes=(_axes(w.S), None))
            else:
                f = self._one
            self._fns = {key: jax.jit(f)}
        return self._fns[key]

    def sample(self) -> np.ndarray:
        w = self.windows
        f = self._fn()
        lam_t = jnp.asarray(w.lambdas[self.targets], jnp.float64)
        if w.batched:
            G, it, ovf = f(w.S, lam_t)
        else:
            outs = [f(s, lam_t) for s in w.states]
            G, it, ovf = (jnp.stack([o[j] for o in outs]) for j in range(3))
        if bool(np.any(np.asarray(ovf))):
            raise RuntimeError("row capacity exceeded while sampling parameter gradients")
        return np.transpose(np.asarray(G, float), (1, 0, 2))

    def meta(self) -> dict:
        """What FreeEnergyRun stores with the samples: targets, entry names, the sampled parameters."""
        p = np.asarray(self.space.flatten(self.params), float)
        return {"dudp_targets": self.targets.tolist(), "dudp_names": self.space.names, "params_flat": p.tolist()}


def gas_leg_gradient(gas, params, space: ParamSpace) -> tuple:
    """(Delta G_gas(1 -> 0), d Delta G_gas / dP (M,)) of a rigid solute's gas-phase leg
    (alchemy.GasPhaseLeg: E_gas(0) - E_gas(1), exact), kJ/mol."""

    def f(P):
        return gas._e(jnp.asarray(0.0), P) - gas._e(jnp.asarray(1.0), P)

    v, g = jax.value_and_grad(f)(params)
    return float(v), np.asarray(space.flatten(g), float)


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
        """(grad . v, standard error): e.g. v = ParamSpace.scale_direction(...) gives dG/d ln s."""
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
        raise ValueError("the samples have no parameter gradients (FreeEnergyRun(param_grad=ParamGradients(...)))")
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
        space: ParamSpace | None = None,
    ):
        self.result, self.p = result, np.asarray(params_flat, float)
        self.names = list(space_names)
        self.space = space or ParamSpace.from_names(self.names)
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

    def value_and_grad(self, theta_fn=None, theta=None, space: ParamSpace | None = None) -> dict:
        """Value, gradient (per theta, or per table entry without theta_fn), their standard errors
        and, with an experiment, chi2 = ((value - experiment) / sigma)^2 and its gradient."""
        r = self.result
        if theta_fn is None:
            g, ge = r.grad, r.grad_err
        else:
            space = space or self.space
            if space is None:
                raise ValueError("a ParamSpace of the run's table is needed for theta_fn")
            g, ge = r.chain(self._J(theta_fn, theta, space))
        out = {"value": r.value, "value_err": r.value_err, "grad": g, "grad_err": ge, "name": self.name}
        if self.experiment is not None:
            s = self.sigma or 1.0
            out["chi2"] = ((r.value - self.experiment) / s) ** 2
            out["dchi2"] = 2.0 * (r.value - self.experiment) / s**2 * np.asarray(g)
        return out

    def estimate(self, theta_fn, theta, space: ParamSpace | None = None, unit: str = "kJ/mol") -> dict:
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
