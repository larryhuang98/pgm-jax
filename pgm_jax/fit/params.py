"""Fitting parameters: a vector theta <-> the parameter pytree of a ParamTable (system.py).

One class, `ParameterSpace`, serves every fit and gradient in pgm_jax (liquid fits, QM fits,
parameter gradients of free energies).  A space is a list of `Param` blocks, each acting on some
entries (tying keys) of one quantity of the table, in one of three kinds:

  scale    one theta: p[key] = p0[key] * exp(theta)   (theta = ln s, a scale factor; the
           Lennard-Jones well depth is stored as sqrt(eps), so its scale multiplies sqrt(eps) by
           exp(theta / 2) and eps by exp(theta))
  shift    one theta: p[key] = p0[key] + theta          (table units)
  values   one theta per entry: p[key] = theta_key      (the values themselves, table units); for
           the charges optionally in the null space of neutrality constraints (theta = offsets
           from the starting charges along directions that keep every listed molecule's charge)

    space = ParameterSpace.scales(sys.table, ["q", "cov", "alpha", "radius", "lj_r", "lj_eps"])
    space.names        # ['ln s_q', 'ln s_cov', 'ln s_pol', 'ln s_rad', 'ln s_R', 'ln s_eps']
    P = space(theta)   # parameter pytree (differentiable in theta)
    space = ParameterSpace(sys.table, [Param("q", "scale", keys=["WAT:OW"]), ...])      # per key
    space = ParameterSpace.values(sys.table, {"q": "all", "alpha": ["OW"]}, neutral=[water])
    space = ParameterSpace.values(sys.table)   # every entry: flatten / unflatten / select for gradients

Units of the table: nm, e, e nm, nm^3, sqrt(kJ/mol) (lj_sqrt_eps).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

# short names of the common quantities -> table quantity
ALIASES = {
    "q": "q",
    "cov": "cov",
    "alpha": "alpha",
    "pol": "alpha",
    "radius": "radius",
    "rad": "radius",
    "lj_r": "lj_rmin_half",
    "lj_eps": "lj_sqrt_eps",
}
LABELS = {"q": "q", "cov": "cov", "alpha": "pol", "radius": "rad", "lj_rmin_half": "R", "lj_sqrt_eps": "eps"}
KINDS = ("scale", "shift", "values")

# scale groups (ParameterSpace.scale_direction): quantities scaled together and the exponent of the
# scale on each (lj_sqrt_eps carries sqrt(eps), so scaling eps by s scales it by s^1/2)
SCALE_GROUPS = {
    "charge": {"q": 1.0, "cov": 1.0},
    "eps": {"lj_sqrt_eps": 0.5},
    "rmin": {"lj_rmin_half": 1.0},
    "alpha": {"alpha": 1.0},
    "radius": {"radius": 1.0},
}


@dataclass
class Param:
    """One block of fitting parameters.

    Attributes
    ----------
    quantity : str
        Table quantity (or an alias of ALIASES: "lj_r", "lj_eps", "pol", "rad").
    kind : str
        "scale" (one theta = ln s), "shift" (one theta, table units) or "values" (one theta per
        entry, the values themselves).
    keys : list of str or None
        Tying keys acted on; None: every entry with a nonzero value (scale), every entry (shift,
        values).
    name : str or None
        Name of a scale / shift parameter (default "ln s_<label>" / "d <quantity>", with the keys).
    prior_sigma : float or None
        Gaussian prior width on theta (None: the space's default).
    bounds : tuple of 2 float or None
        "values": lower and upper bound of every entry (table units; None: unbounded).
    step : float or None
        "values": typical size of a change of an entry (table units), the optimizer's scaling and
        the unit of the ridge prior (None: 1).
    extra : dict
        Free-form annotations.
    """

    quantity: str
    kind: str = "scale"
    keys: list | None = None
    name: str | None = None
    prior_sigma: float | None = None
    bounds: tuple | None = None
    step: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class _Block:
    """A resolved Param: the table entries it acts on and its slice of theta."""

    param: Param
    idx: np.ndarray  # (m,) int32 table entries
    keys: list  # tying keys of the entries
    names: list  # names of its theta entries
    sl: slice  # its entries of theta
    full: bool = False  # "values" over every entry of the quantity, in table order
    null: object = None  # "values" of charges: (m, k) null-space basis of the neutrality constraints


class ParameterSpace:
    """Map between the fitted vector theta and the parameter pytree (see the module docstring)."""

    def __init__(self, table, params: list[Param], p0=None, prior_sigma: float = 0.1, neutral=None):
        """Resolve the parameter blocks on a table.

        Parameters
        ----------
        table : ParamTable or None
            The parameter table (None only for `from_names`).
        params : list of Param
            The blocks, in the order of theta.
        p0 : dict, optional
            Starting parameters {quantity: array} (default: table.initial()).
        prior_sigma : float
            Default Gaussian prior width on theta.
        neutral : list of Molecule, optional
            "values" blocks of the charges keep the total charge of each of these molecules: their
            theta are coordinates in the null space of the neutrality constraints (offsets from
            p0).

        Raises
        ------
        KeyError
            An unknown quantity.
        ValueError
            An unknown kind, or a block acting on no entries.
        """
        self.table = table
        self.p0 = None
        if table is not None:
            self.p0 = {k: jnp.asarray(v) for k, v in (table.initial() if p0 is None else p0).items()}
        self.blocks: list[_Block] = []
        off = 0
        for p in params:
            blk = self._resolve(p, off, neutral)
            self.blocks.append(blk)
            off = blk.sl.stop
        self._finish(prior_sigma)

    # ----------------------------------------------------------------- construction
    def _resolve(self, p: Param, off: int, neutral) -> _Block:
        """Resolve Param `p` into a _Block starting at theta entry `off`."""
        q = ALIASES.get(p.quantity, p.quantity)
        if q not in self.p0:
            raise KeyError(f"unknown quantity {p.quantity!r}")
        if p.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {p.kind!r}")
        vals = np.asarray(self.p0[q])
        all_keys = list(self.table.keys[q])
        if p.keys is None:
            idx = np.flatnonzero(vals != 0.0) if p.kind == "scale" else np.arange(len(vals))
        else:
            idx = self.table.index(q, list(p.keys))
        idx = np.asarray(idx, np.int32)
        if len(idx) == 0:
            raise ValueError(f"parameter on {q} acts on no entries (all zero?)")
        keys = [all_keys[i] for i in idx]
        full = p.kind == "values" and p.keys is None
        null = None
        if p.kind == "values" and q == "q" and neutral:
            null = self._neutral_basis(keys, neutral)
            names = [f"q:null{j}" for j in range(null.shape[1])]
        elif p.kind == "values":
            names = [f"{q}:{k}" for k in keys]
        else:
            name = p.name or (f"ln s_{LABELS.get(q, q)}" if p.kind == "scale" else f"d {q}")
            if p.keys is not None and p.name is None:
                name += "[" + ",".join(p.keys) + "]"
            names = [name]
        q_param = Param(q, p.kind, p.keys, names[0] if p.kind != "values" else p.name, p.prior_sigma, p.bounds, p.step)
        q_param.extra = p.extra
        return _Block(q_param, idx, keys, names, slice(off, off + len(names)), full, null)

    @staticmethod
    def _neutral_basis(keys: list, molecules) -> np.ndarray:
        """Return an orthonormal basis (m, k) of the charge offsets on `keys` that keep molecules neutral.

        The null space of one constraint row per molecule (the count of each key's atoms in it):
        adding any combination of the columns leaves every molecule's total charge unchanged.
        """
        from scipy.linalg import null_space

        rows = []
        for mol in {id(m): m for m in molecules}.values():
            kq = mol.tying_keys()["q"]
            row = np.array([sum(1.0 for kk in kq if kk == key) for key in keys])
            if row.any():
                rows.append(row)
        return jnp.asarray(null_space(np.array(rows)) if rows else np.eye(len(keys)))

    def _finish(self, prior_sigma: float) -> None:
        """Names, sizes, starting point, bounds, steps and priors of theta."""
        self.params = [b.param for b in self.blocks]
        self.names = [nm for b in self.blocks for nm in b.names]
        self.n = len(self.names)
        th0, lo, hi, step, prior = [], [], [], [], []
        for b in self.blocks:
            m = len(b.names)
            p = b.param
            sigma = prior_sigma if p.prior_sigma is None else p.prior_sigma
            prior += [sigma] * m
            if p.kind == "values" and b.null is None:
                lo_b, hi_b = p.bounds if p.bounds is not None else (-np.inf, np.inf)
                th0 += list(np.asarray(self.p0[p.quantity])[b.idx]) if self.p0 is not None else [0.0] * m
                lo += [lo_b] * m
                hi += [hi_b] * m
            else:
                th0 += [0.0] * m
                lo += [-np.inf] * m
                hi += [np.inf] * m
            step += [1.0 if p.step is None else p.step] * m
        self.lower, self.upper = np.array(lo, float), np.array(hi, float)
        self.theta0 = np.clip(np.array(th0, float), self.lower, self.upper)
        self.step = np.array(step, float)
        self.prior_sigma = np.array(prior, float)
        self.quantities = list(dict.fromkeys(b.param.quantity for b in self.blocks))
        self.keys = {}
        for b in self.blocks:
            self.keys.setdefault(b.param.quantity, []).extend(b.keys)
        self.slices = {b.param.quantity: b.sl for b in self.blocks}

    @classmethod
    def scales(cls, table, quantities, p0=None, prior_sigma: float = 0.1) -> ParameterSpace:
        """One global scale factor per quantity.

        Parameters
        ----------
        table : ParamTable
            The parameter table.
        quantities : iterable of str
            "q", "cov", "alpha" / "pol", "radius" / "rad", "lj_r", "lj_eps", or any table quantity.
        p0 : dict, optional
            Starting parameters.
        prior_sigma : float
            Gaussian prior width on each ln s.

        Returns
        -------
        ParameterSpace
        """
        return cls(table, [Param(q) for q in quantities], p0, prior_sigma)

    @classmethod
    def values(
        cls,
        table,
        free=None,
        p0=None,
        neutral=None,
        bounds: dict | None = None,
        steps: dict | None = None,
        prior_sigma: float = 0.1,
    ) -> ParameterSpace:
        """Build a space whose theta is the values of chosen table entries.

        Parameters
        ----------
        table : ParamTable
            The parameter table.
        free : dict, iterable of str or None
            {quantity: "all" | [keys]}, a list of quantities (all their keys), or None: every
            quantity that has keys (the flat table of parameter gradients, names "quantity:key").
        p0 : dict, optional
            Starting parameters (theta0 = their values).
        neutral : list of Molecule, optional
            Charges move in the null space of these molecules' neutrality constraints.
        bounds, steps : dict, optional
            {quantity: (lower, upper)} and {quantity: typical change} (table units).
        prior_sigma : float
            Gaussian prior width on each entry.

        Returns
        -------
        ParameterSpace
        """
        from ..system import QUANTITIES

        if free is None:
            free = {q: "all" for q in QUANTITIES if len(table.keys[q])}
        elif not isinstance(free, dict):
            bad = [q for q in free if q not in QUANTITIES]
            if bad:
                raise ValueError(f"unknown parameter quantities {bad}")
            free = {q: "all" for q in free if len(table.keys[q])}
        bounds, steps = bounds or {}, steps or {}
        params = [
            Param(q, "values", None if keys == "all" else list(keys), bounds=bounds.get(q), step=steps.get(q))
            for q, keys in free.items()
        ]
        return cls(table, params, p0, prior_sigma, neutral)

    @classmethod
    def from_names(cls, names) -> ParameterSpace:
        """Rebuild a "values" space from stored names "quantity:key" (in table order).

        The space has no table: flatten / unflatten / select / scale_direction work, evaluating
        theta needs p0.

        Parameters
        ----------
        names : iterable of str
            The names (e.g. the "dudp_names" of a free-energy run).

        Returns
        -------
        ParameterSpace
        """
        self = cls.__new__(cls)
        self.table, self.p0, self.blocks = None, None, []
        keys: dict = {}
        for nm in names:
            q, k = nm.split(":", 1)
            keys.setdefault(q, []).append(k)
        off = 0
        for q, ks in keys.items():
            blk = _Block(
                Param(q, "values"),
                np.arange(len(ks), dtype=np.int32),
                ks,
                [f"{q}:{k}" for k in ks],
                slice(off, off + len(ks)),
                full=True,
            )
            self.blocks.append(blk)
            off += len(ks)
        self._finish(0.1)
        return self

    # ----------------------------------------------------------------- theta -> parameters
    def __len__(self) -> int:
        """Return the number of entries of theta."""
        return self.n

    def __call__(self, theta, p0=None) -> dict:
        """Return the parameter pytree at theta (JAX-differentiable in theta).

        Parameters
        ----------
        theta : array (n,)
            Parameters (ln s for scales, table units for shifts and values).
        p0 : dict, optional
            Parameters the blocks act on (default: the space's p0).

        Returns
        -------
        dict
            {quantity: array} with every quantity of p0.
        """
        P = dict(self.p0 if p0 is None else p0)
        theta = jnp.asarray(theta)
        for b in self.blocks:
            q, kind = b.param.quantity, b.param.kind
            if kind == "scale":
                t = theta[b.sl.start]
                f = jnp.exp(0.5 * t) if q == "lj_sqrt_eps" else jnp.exp(t)
                P[q] = P[q] * jnp.ones(P[q].shape).at[b.idx].set(f)
            elif kind == "shift":
                P[q] = P[q].at[b.idx].add(theta[b.sl.start])
            elif b.null is None:
                P[q] = P[q].at[b.idx].set(theta[b.sl])
            else:
                P[q] = P[q].at[b.idx].add(b.null @ theta[b.sl])
        return P

    def zeros(self) -> np.ndarray:
        """Return theta = 0 (the starting parameters for scales and shifts)."""
        return np.zeros(self.n)

    def describe(self, theta) -> str:
        """Describe theta in one line: every parameter's name and value (and a scale's factor)."""
        theta = np.asarray(theta, float)
        out = []
        for b in self.blocks:
            for nm, t in zip(b.names, theta[b.sl]):
                out.append(f"{nm} {t:+.4f}" + (f" (x{np.exp(t):.4f})" if b.param.kind == "scale" else ""))
        return ", ".join(out)

    def named_values(self, theta) -> dict:
        """Return {quantity: {key: value}} of the table entries the space acts on, at theta."""
        P = self(np.asarray(theta, float))
        out = {}
        for b in self.blocks:
            q = b.param.quantity
            out.setdefault(q, {}).update({k: float(P[q][i]) for k, i in zip(b.keys, b.idx)})
        return out

    # ----------------------------------------------------------------- flat tables (values spaces)
    def _values_only(self) -> None:
        """Raise unless every block is a plain "values" block (flatten / unflatten)."""
        if any(b.param.kind != "values" or b.null is not None for b in self.blocks):
            raise ValueError("flatten / unflatten need a space of plain values (ParameterSpace.values)")

    def flatten(self, P):
        """Return the entries of the space as one float64 vector (JAX-differentiable).

        Parameters
        ----------
        P : dict
            Parameters {quantity: array} (e.g. a gradient with respect to the table).

        Returns
        -------
        jax.Array (n,)
        """
        self._values_only()
        parts = []
        for b in self.blocks:
            x = jnp.ravel(jnp.asarray(P[b.param.quantity], jnp.float64))
            parts.append(x if b.full else x[b.idx])
        return jnp.concatenate(parts)

    def unflatten(self, v, like=None) -> dict:
        """Return the parameter dict of a flat vector (inverse of flatten).

        Parameters
        ----------
        v : array (n,)
            The flat vector.
        like : dict, optional
            Parameters providing the quantities (and entries) the space does not cover.

        Returns
        -------
        dict
        """
        self._values_only()
        out = dict(like) if like is not None else {}
        for b in self.blocks:
            q = b.param.quantity
            out[q] = v[b.sl] if b.full else jnp.asarray(out[q]).at[b.idx].set(v[b.sl])
        return out

    def index(self, name: str) -> int:
        """Return the position of the parameter `name` in theta."""
        return self.names.index(name)

    def select(self, quantities=None, solute: bool | None = None) -> np.ndarray:
        """Return the positions in theta of the entries of some quantities ("values" spaces).

        Parameters
        ----------
        quantities : iterable of str, optional
            Quantities to include (default: all).
        solute : bool or None
            True / False: only the alchemical solute's own keys (prefix "alch:") / only the other
            (environment) keys; None: both.

        Returns
        -------
        np.ndarray of int
        """
        from ..md.alchemy import PREFIX  # tying-key prefix of the alchemical solute's own parameters

        out = []
        for b in self.blocks:
            q = b.param.quantity
            if quantities is not None and q not in quantities:
                continue
            for j, k in enumerate(b.keys):
                own = k.startswith(PREFIX)
                if solute is None or own == bool(solute):
                    out.append(b.sl.start + j)
        return np.array(out, int)

    def scale_direction(self, p_flat, group: str, solute: bool | None = True) -> np.ndarray:
        """Return the direction v with dG/d ln s = grad . v for scaling a group of parameters by s.

        Parameters
        ----------
        p_flat : array (n,)
            The flat parameters (flatten).
        group : str
            A key of SCALE_GROUPS: "charge" (charges and covalent dipoles), "eps", "rmin",
            "alpha", "radius".
        solute : bool or None
            As in `select`.

        Returns
        -------
        np.ndarray (n,)
            v_i = e_i p_i, e_i the exponent of the scale on entry i (1/2 for sqrt(eps)).
        """
        v = np.zeros(self.n)
        p = np.asarray(p_flat, float)
        for q, e in SCALE_GROUPS[group].items():
            i = self.select((q,), solute)
            v[i] = e * p[i]
        return v
