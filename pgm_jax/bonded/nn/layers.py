"""Provide the small pure-JAX building blocks of the neural bonded model: MLPs and message passing.

Two-layer MLPs (`init_mlp`, `mlp`) and message passing over the bond graph
(`init_message_passing`, `embeddings`).  Parameters are plain dicts of arrays, so they are
pytrees for the fitter.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def init_mlp(key: jax.Array, n_in: int, n_hid: int, n_out: int, zero_out: bool = False, scale_out: float = 1.0) -> dict:
    """Return the weights of a two-layer MLP (Glorot-normal initialisation, zero biases).

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    n_in, n_hid, n_out : int
        Input, hidden and output widths.
    zero_out : bool
        Start the output layer at zero (the head predicts nothing).
    scale_out : float
        Extra factor on the output layer's initial weights.

    Returns
    -------
    dict
        {"w1" (n_in, n_hid), "b1" (n_hid,), "w2" (n_hid, n_out), "b2" (n_out,)}.
    """
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (n_in, n_hid)) * math.sqrt(2.0 / (n_in + n_hid))
    w2 = (
        jnp.zeros((n_hid, n_out))
        if zero_out
        else jax.random.normal(k2, (n_hid, n_out)) * math.sqrt(2.0 / (n_hid + n_out)) * scale_out
    )
    return {"w1": w1, "b1": jnp.zeros(n_hid), "w2": w2, "b2": jnp.zeros(n_out)}


def mlp(p: dict, x: ArrayLike) -> jax.Array:
    """Return silu(x W1 + b1) W2 + b2 for inputs x (..., n_in)."""
    return jax.nn.silu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def init_message_passing(keys: list, n_feat: int, width: int, layers: int, n_edge: int = 4) -> dict:
    """Return the embedding MLP and `layers` message-passing layers (keys are popped from the list).

    Parameters
    ----------
    keys : list of jax.Array
        PRNG keys; 1 + 2 layers are consumed (popped from the end).
    n_feat : int
        Atom feature width.
    width : int
        Embedding width W.
    layers : int
        Number of message-passing layers.
    n_edge : int
        Edge feature width.

    Returns
    -------
    dict
        {"embed": MLP, "mp{l}": {"msg": MLP, "upd": MLP}}.
    """
    P = {"embed": init_mlp(keys.pop(), n_feat, width, width)}
    for l in range(layers):
        P[f"mp{l}"] = {
            "msg": init_mlp(keys.pop(), 2 * width + n_edge, width, width, scale_out=0.3),
            "upd": init_mlp(keys.pop(), 2 * width, width, width, scale_out=0.3),
        }
    return P


def embeddings(P: dict, X: ArrayLike, src: ArrayLike, dst: ArrayLike, ef: ArrayLike, n: int, layers: int) -> jax.Array:
    """Return the atom embeddings (n, W): h = MLP(x), then per layer h += upd([h, sum_j msg([h_i, h_j, e_ij])]).

    Parameters
    ----------
    P : dict
        Weights (`init_message_passing`).
    X : ArrayLike (n, n_feat)
        Atom features.
    src, dst : ArrayLike (2 nb,) int
        Directed edges j -> i (messages are summed at dst).
    ef : ArrayLike (2 nb, n_edge)
        Edge features.
    n : int
        Number of atoms.
    layers : int
        Number of message-passing layers.
    """
    h = mlp(P["embed"], jnp.asarray(X))
    ef = jnp.asarray(ef)
    for l in range(layers):
        msg = mlp(P[f"mp{l}"]["msg"], jnp.concatenate([h[dst], h[src], ef], -1))
        agg = jax.ops.segment_sum(msg, dst, num_segments=n)
        h = h + mlp(P[f"mp{l}"]["upd"], jnp.concatenate([h, agg], -1))
    return h
