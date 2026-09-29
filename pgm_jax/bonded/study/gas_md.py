"""Gas-phase Langevin dynamics of one flexible molecule (checks of fitted bonded terms)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ...units import KB


def langevin(efun, X0, masses, T_K, dt, nsteps, every, nrep, key, gamma=2.0):
    """BAOAB; returns (nrep, nsteps/every, n, 3) positions and (nrep, nsteps/every) energies."""
    m = jnp.asarray(masses)[:, None]
    kT = KB * T_K
    grad = jax.grad(efun)
    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt((1 - c1**2) * kT / m)

    def step(state, k):
        X, V, F = state
        V = V + 0.5 * dt * F / m
        X = X + 0.5 * dt * V
        V = c1 * V + c2 * jax.random.normal(k, X.shape)
        X = X + 0.5 * dt * V
        F = -grad(X)
        V = V + 0.5 * dt * F / m
        return (X, V, F), None

    def block(state, ks):
        state, _ = jax.lax.scan(step, state, ks)
        return state, (state[0], efun(state[0]))

    def one(key):
        k0, k1 = jax.random.split(key)
        V0 = jax.random.normal(k0, X0.shape) * jnp.sqrt(kT / m)
        ks = jax.random.split(k1, nsteps).reshape(nsteps // every, every, 2)
        _, (Xs, Es) = jax.lax.scan(block, (X0, V0, -grad(X0)), ks)
        return Xs, Es

    return jax.jit(jax.vmap(one))(jax.random.split(key, nrep))
