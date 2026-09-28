"""Fitting parameters theta -> the parameter pytree of a ParamTable (system.py).

A ParameterSpace is a list of parameters, each acting on some entries (tying keys) of one quantity
of the table:

  scale   p[key] = p0[key] * exp(theta)        theta = ln s (a scale factor, as the Bayesian
                                               optimisation's scale_q, scale_p, scale_pol, scale_rad)
  shift   p[key] = p0[key] + theta             native units (e, e nm, nm, nm^3)

Scaling every charge of a neutral molecule keeps it neutral.  The Lennard-Jones well depth enters
the table as sqrt(eps) (lj_sqrt_eps): its "scale" multiplies eps by exp(theta), i.e. sqrt(eps) by
exp(theta / 2), so that theta is ln s_eps for both LJ quantities (as scripts/fit_liquid.py).

    space = ParameterSpace.scales(sys.table, ["q", "cov", "alpha", "radius", "lj_r", "lj_eps"])
    space.names          # ['ln s_q', 'ln s_cov', 'ln s_pol', 'ln s_rad', 'ln s_R', 'ln s_eps']
    P = space(theta)     # parameter pytree (differentiable in theta)
    space = ParameterSpace(sys.table, [Param("q", "scale", keys=["WAT:OW"]), ...])   # per key

Units of the table: nm, e, e nm, nm^3, sqrt(kJ/mol)."""
from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

# short names of the common scale factors -> table quantity
ALIASES = {"q": "q", "cov": "cov", "alpha": "alpha", "pol": "alpha", "radius": "radius", "rad": "radius",
           "lj_r": "lj_rmin_half", "lj_eps": "lj_sqrt_eps"}
LABELS = {"q": "q", "cov": "cov", "alpha": "pol", "radius": "rad", "lj_rmin_half": "R", "lj_sqrt_eps": "eps"}


@dataclass
class Param:
    """One fitting parameter: `quantity` of the table, `kind` "scale" (theta = ln s) or "shift"
    (native units), acting on the entries with the given tying keys (None: every entry with a
    nonzero value, for scales; every entry for shifts)."""
    quantity: str
    kind: str = "scale"
    keys: list | None = None
    name: str | None = None
    prior_sigma: float | None = None          # Gaussian prior width on theta (None: the space's default)
    extra: dict = field(default_factory=dict)


class ParameterSpace:
    def __init__(self, table, params: list[Param], p0=None, prior_sigma: float = 0.1):
        self.table = table
        self.p0 = {k: jnp.asarray(v) for k, v in (table.initial() if p0 is None else p0).items()}
        self.params = []
        self.index = []
        for p in params:
            q = ALIASES.get(p.quantity, p.quantity)
            if q not in self.p0:
                raise KeyError(f"unknown quantity {p.quantity!r}")
            if p.kind not in ("scale", "shift"):
                raise ValueError(f"kind must be 'scale' or 'shift', got {p.kind!r}")
            vals = np.asarray(self.p0[q])
            if p.keys is None:
                idx = np.flatnonzero(vals != 0.0) if p.kind == "scale" else np.arange(len(vals))
            else:
                idx = table.index(q, list(p.keys))
            if len(idx) == 0:
                raise ValueError(f"parameter on {q} acts on no entries (all zero?)")
            name = p.name or (f"ln s_{LABELS.get(q, q)}" if p.kind == "scale" else f"d {q}")
            if p.keys is not None and p.name is None:
                name += "[" + ",".join(p.keys) + "]"
            self.params.append(Param(q, p.kind, p.keys, name, p.prior_sigma, p.extra))
            self.index.append(np.asarray(idx, np.int32))
        self.names = [p.name for p in self.params]
        self.n = len(self.params)
        self.prior_sigma = np.array([prior_sigma if p.prior_sigma is None else p.prior_sigma for p in self.params])

    @classmethod
    def scales(cls, table, quantities, p0=None, prior_sigma: float = 0.1):
        """One global scale factor per quantity ("q", "cov", "alpha"/"pol", "radius"/"rad", "lj_r",
        "lj_eps", or any table quantity)."""
        return cls(table, [Param(q) for q in quantities], p0, prior_sigma)

    def __call__(self, theta, p0=None):
        """Parameter pytree at theta (differentiable)."""
        P = dict(self.p0 if p0 is None else p0)
        theta = jnp.asarray(theta)
        for j, (p, idx) in enumerate(zip(self.params, self.index)):
            t = theta[j]
            if p.kind == "scale":
                f = jnp.exp(0.5 * t) if p.quantity == "lj_sqrt_eps" else jnp.exp(t)
                P[p.quantity] = P[p.quantity] * jnp.ones(P[p.quantity].shape).at[idx].set(f)
            else:
                P[p.quantity] = P[p.quantity].at[idx].add(t)
        return P

    def zeros(self):
        return np.zeros(self.n)

    def describe(self, theta) -> str:
        theta = np.asarray(theta, float)
        out = []
        for p, t in zip(self.params, theta):
            out.append(f"{p.name} {t:+.4f}" + (f" (x{np.exp(t):.4f})" if p.kind == "scale" else ""))
        return ", ".join(out)

    def named_values(self, theta) -> dict:
        """{quantity: {key: value}} of the entries the space acts on, at theta."""
        P = self(np.asarray(theta, float))
        out = {}
        for p, idx in zip(self.params, self.index):
            keys = self.table.keys[p.quantity]
            out.setdefault(p.quantity, {}).update({keys[i]: float(P[p.quantity][i]) for i in idx})
        return out
