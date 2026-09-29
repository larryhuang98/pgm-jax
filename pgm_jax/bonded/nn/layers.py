"""Small pure-JAX building blocks of the neural bonded model: two-layer MLPs and message passing
over the bond graph (parameters are plain dicts of arrays, so they are pytrees for the fitter)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def init_mlp(key, n_in: int, n_hid: int, n_out: int, zero_out: bool = False, scale_out: float = 1.0) -> dict:
    """Two-layer MLP; with zero_out the output layer starts at zero (the head predicts nothing)."""
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (n_in, n_hid)) * math.sqrt(2.0 / (n_in + n_hid))
    w2 = (
        jnp.zeros((n_hid, n_out))
        if zero_out
        else jax.random.normal(k2, (n_hid, n_out)) * math.sqrt(2.0 / (n_hid + n_out)) * scale_out
    )
    return {"w1": w1, "b1": jnp.zeros(n_hid), "w2": w2, "b2": jnp.zeros(n_out)}


def mlp(p: dict, x):
    return jax.nn.silu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def init_message_passing(keys: list, n_feat: int, width: int, layers: int, n_edge: int = 4) -> dict:
    """Embedding MLP and `layers` message-passing layers (keys are popped from the list)."""
    P = {"embed": init_mlp(keys.pop(), n_feat, width, width)}
    for l in range(layers):
        P[f"mp{l}"] = {
            "msg": init_mlp(keys.pop(), 2 * width + n_edge, width, width, scale_out=0.3),
            "upd": init_mlp(keys.pop(), 2 * width, width, width, scale_out=0.3),
        }
    return P


def embeddings(P: dict, X, src, dst, ef, n: int, layers: int):
    """Atom embeddings (n, W): h = MLP(x), then h += upd([h, sum_j msg([h_i, h_j, e_ij])])."""
    h = mlp(P["embed"], jnp.asarray(X))
    ef = jnp.asarray(ef)
    for l in range(layers):
        msg = mlp(P[f"mp{l}"]["msg"], jnp.concatenate([h[dst], h[src], ef], -1))
        agg = jax.ops.segment_sum(msg, dst, num_segments=n)
        h = h + mlp(P[f"mp{l}"]["upd"], jnp.concatenate([h, agg], -1))
    return h
