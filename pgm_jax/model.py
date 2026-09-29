"""Sum energy channels into a gas-phase force-field model.

Contents: Model, a list of channels (channels.ElecChannel, lj.LJChannel, vdw.GVDWChannel, ...)
with total energies, interaction energies (supermolecular: E_AB - E_A - E_B), n-body
decompositions and forces, for one geometry or a batch.

Parameters: the pytree of the System's ParamTable (`sys.table.initial()`), shared by every
channel and every subsystem; None means the initial values.  Energies are differentiable in
coordinates and parameters; compiled functions are cached by topology, so changing parameter
values never recompiles.  For periodic systems see periodic.PeriodicModel.

    model = Model([ElecChannel(), LJChannel()])
    e = model.batch_energy(system, coords)          # {"perm": (F,), "ind": (F,), "vdw": (F,), "total": (F,)}
    nb = model.nbody(system, coords, order=3)       # {"int": {...}, "nb2": {...}, "nb3": {...}}

Units: nm, kJ/mol.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .system import System


@dataclass
class Model:
    """Gas-phase model: the sum of the energies of a list of channels.

    A mutable dataclass (not a pytree).  Every channel returns a dict of energy components; the
    model merges them (a later channel's key replaces an earlier one) and adds "total", the sum of
    all components.

    Parameters
    ----------
    channels : list
        Channel objects (anything with energy(pos, sys, params) -> (dict, aux)) or factories
        sys -> channel, called by `build` for each System.
    """

    channels: list  # channel objects (with .energy(pos, sys, params)) or factories sys -> channel

    def __post_init__(self) -> None:
        """Create the cache of jitted batched energy functions (keyed by System.fingerprint)."""
        self._jit_cache = {}  # System.fingerprint() -> jitted batched energy

    def build(self, sys: System) -> list[Any]:
        """Return the channel objects for a System (factories are called with `sys`)."""
        return [c if hasattr(c, "energy") else c(sys) for c in self.channels]

    # ---------------------------------------------------------------- energies --
    def energy_fn(self, sys: System) -> Callable[..., dict[str, jax.Array]]:
        """Return the energy function of one System.

        Returns
        -------
        callable
            f(pos, params=None) -> {component: scalar} [kJ/mol], with pos (N, 3) [nm] and params the
            parameter pytree (None: initial values); includes "total".  Not jitted; differentiable.
        """
        chans = self.build(sys)

        def f(pos: jax.Array, params: Mapping[str, ArrayLike] | None = None) -> dict[str, jax.Array]:
            """Return the merged energy components of all channels and their "total" [kJ/mol]."""
            out = {}
            for ch in chans:
                e, _ = ch.energy(pos, sys, params)
                out.update(e)
            out["total"] = sum(v for k, v in out.items())
            return out

        return f

    def batch_energy(
        self, sys: System, coords: ArrayLike, params: Mapping[str, ArrayLike] | None = None, batch: int = 1024
    ) -> dict[str, np.ndarray]:
        """Return the energy components of a batch of geometries.

        Parameters
        ----------
        sys : System
            System.
        coords : ArrayLike (F, N, 3)
            Geometries [nm].
        params : Mapping of str to ArrayLike, optional
            Parameter pytree; None: the table's initial values.
        batch : int
            Geometries per jitted call.

        Returns
        -------
        dict of str to np.ndarray (F,)
            Energy components and "total" [kJ/mol] (host arrays).

        Notes
        -----
        The energy function is vmapped over the leading axis of `coords` (params shared) and jitted
        once per System.fingerprint(); a last batch of a different size compiles once more.  Returns
        host numpy arrays, so the result is not differentiable.
        """
        key = sys.fingerprint()
        if key not in self._jit_cache:
            self._jit_cache[key] = jax.jit(jax.vmap(self.energy_fn(sys), in_axes=(0, None)))
        f = self._jit_cache[key]
        outs = [f(jnp.asarray(coords[s : s + batch]), params) for s in range(0, len(coords), batch)]
        return {k: np.concatenate([np.asarray(o[k]) for o in outs]) for k in outs[0]}

    def nbody(
        self,
        sys: System,
        coords: ArrayLike,
        params: Mapping[str, ArrayLike] | None = None,
        order: int | None = None,
        batch: int = 1024,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return interaction and many-body energies by inclusion-exclusion over molecule subsets.

        Parameters
        ----------
        sys : System
            System; its molecules are the bodies.
        coords : ArrayLike (F, N, 3)
            Geometries [nm].
        params : Mapping of str to ArrayLike, optional
            Parameter pytree; None: the table's initial values.
        order : int, optional
            Highest n-body order returned; None: the number of molecules.
        batch : int
            Geometries per jitted call (batch_energy).

        Returns
        -------
        dict of str to dict of str to np.ndarray (F,)
            "int": E(all) - sum_m E(m) per component; "nb2", ..., "nb<order>": the total n-body
            energy per component, E_n = sum over the subsets S of n molecules of
            sum_{T subset S} (-1)^(|S|-|T|) E(T) [kJ/mol].

        Notes
        -----
        Evaluates every one of the 2^M - 1 subsets of the M molecules (independent of `order`), each
        as its own System (with the same ParamTable), so the cost grows as 2^M compiled calls.
        """
        nm = sys.nmol
        order = order or nm
        E = {}
        for k in range(1, nm + 1):
            for sub in itertools.combinations(range(nm), k):
                s, idx = sys.sub(sub)
                E[sub] = self.batch_energy(s, coords[:, idx, :], params, batch)
        comps = E[tuple(range(nm))].keys()
        res = {"int": {c: E[tuple(range(nm))][c] - sum(E[(m,)][c] for m in range(nm)) for c in comps}}
        # many-body terms: E_n(S) = sum_{T subset S} (-1)^{|S|-|T|} E(T)
        for n in range(2, min(order, nm) + 1):
            tot = {c: 0.0 for c in comps}
            for S in itertools.combinations(range(nm), n):
                for k in range(1, n + 1):
                    for T in itertools.combinations(S, k):
                        sgn = (-1) ** (n - k)
                        for c in comps:
                            tot[c] = tot[c] + sgn * E[T][c]
            res[f"nb{n}"] = tot
        return res

    def forces_fn(self, sys: System) -> Callable[[jax.Array, Mapping[str, ArrayLike] | None], jax.Array]:
        """Return a jitted force function f(pos, params) -> -dE_total/dpos (N, 3) [kJ/mol/nm].

        `pos` (N, 3) [nm], `params` the parameter pytree (None allowed: initial values).
        """
        f = self.energy_fn(sys)
        return jax.jit(lambda pos, params: -jax.grad(lambda x: f(x, params)["total"])(pos))
