"""Run gas-phase Langevin dynamics of one flexible molecule (checks of fitted bonded terms).

Units: nm, ps, amu, K, kJ/mol.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ...units import KB


def langevin(
    efun: Callable[[jax.Array], jax.Array],
    X0: jax.Array,
    masses: ArrayLike,
    T_K: float,
    dt: float,
    nsteps: int,
    every: int,
    nrep: int,
    key: jax.Array,
    gamma: float = 2.0,
) -> tuple[jax.Array, jax.Array]:
    """Run `nrep` independent BAOAB Langevin trajectories of one molecule (jitted, vmapped over replicas).

    Parameters
    ----------
    efun : callable
        E(X) [kJ/mol] of positions X (n, 3) [nm] (differentiable).
    X0 : jax.Array (n, 3)
        Initial positions [nm] (all replicas).
    masses : ArrayLike (n,)
        Masses [amu].
    T_K : float
        Temperature [K].
    dt : float
        Time step [ps].
    nsteps : int
        Steps per trajectory (a multiple of `every`).
    every : int
        Steps between saved frames.
    nrep : int
        Number of replicas.
    key : jax.Array
        PRNG key.
    gamma : float
        Friction [1/ps].

    Returns
    -------
    X : jax.Array (nrep, nsteps / every, n, 3)
        Positions [nm].
    E : jax.Array (nrep, nsteps / every)
        Potential energies [kJ/mol].
    """
    m = jnp.asarray(masses)[:, None]
    kT = KB * T_K
    grad = jax.grad(efun)
    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt((1 - c1**2) * kT / m)  # (n, 1) velocity noise amplitude [nm/ps]

    def step(state: tuple, k: jax.Array) -> tuple[tuple, None]:
        """Take one BAOAB step (scan body; state (X, V, F), noise key k)."""
        X, V, F = state
        V = V + 0.5 * dt * F / m
        X = X + 0.5 * dt * V
        V = c1 * V + c2 * jax.random.normal(k, X.shape)
        X = X + 0.5 * dt * V
        F = -grad(X)
        V = V + 0.5 * dt * F / m
        return (X, V, F), None

    def block(state: tuple, ks: jax.Array) -> tuple[tuple, tuple[jax.Array, jax.Array]]:
        """Run `every` steps and output the positions and energy (scan body)."""
        state, _ = jax.lax.scan(step, state, ks)
        return state, (state[0], efun(state[0]))

    def one(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the frames and energies of one replica (Maxwell-Boltzmann start)."""
        k0, k1 = jax.random.split(key)
        V0 = jax.random.normal(k0, X0.shape) * jnp.sqrt(kT / m)
        ks = jax.random.split(k1, nsteps).reshape(nsteps // every, every, 2)
        _, (Xs, Es) = jax.lax.scan(block, (X0, V0, -grad(X0)), ks)
        return Xs, Es

    return jax.jit(jax.vmap(one))(jax.random.split(key, nrep))
