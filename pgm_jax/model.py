"""A force-field model = a list of channels.  Gives total energies, interaction energies
(supermolecular: E_AB - E_A - E_B), n-body decompositions, and forces, for one geometry or
a batch.

Parameters: a dict {channel name: array}; channels without parameters get None.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .system import System


@dataclass
class Model:
    channels: list      # channel *factories*: callables sys -> channel object (so subsystems can be built)

    def __post_init__(self):
        self._jit_cache = {}                         # System signature -> jitted batched energy

    def build(self, sys: System):
        return [f(sys) for f in self.channels]

    # ---------------------------------------------------------------- energies --
    def energy_fn(self, sys: System):
        """Returns f(pos (n,3), params) -> dict of energy components (kJ/mol)."""
        chans = self.build(sys)

        def f(pos, params):
            out = {}
            for ch in chans:
                e, _ = ch.energy(pos, sys, None if params is None else params.get(ch.name))
                out.update(e)
            out["total"] = sum(v for k, v in out.items())
            return out

        return f

    def batch_energy(self, sys: System, coords, params=None, batch: int = 1024):
        key = sys.fingerprint()
        if key not in self._jit_cache:
            self._jit_cache[key] = jax.jit(jax.vmap(self.energy_fn(sys), in_axes=(0, None)))
        f = self._jit_cache[key]
        outs = [f(jnp.asarray(coords[s:s + batch]), params) for s in range(0, len(coords), batch)]
        return {k: np.concatenate([np.asarray(o[k]) for o in outs]) for k in outs[0]}

    def nbody(self, sys: System, coords, params=None, order: int | None = None, batch: int = 1024):
        """Interaction and many-body energies by inclusion-exclusion over molecule subsets.
        Returns {'int': {...components}, 'nb2': ..., 'nb3': ...} (n-body terms up to `order`)."""
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

    def forces_fn(self, sys: System):
        f = self.energy_fn(sys)
        return jax.jit(lambda pos, params: -jax.grad(lambda x: f(x, params)["total"])(pos))
