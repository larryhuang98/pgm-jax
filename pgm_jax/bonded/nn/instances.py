"""Describe term instances as the network sees them: key decomposition and symmetric readouts.

A family's own `index(top, keyf)` is called with a recording key function, so every instance is
described by the components of its tying key (bond, angle, torsion, atom, ...) exactly as a typed
force field would key it; the literal rest of the key is its skeleton ("ba|X|X": a bond-angle
coupling = one angle and one atom).  `readout` turns a component into a vector that respects the
component's own symmetry (a bond read from either end gives the same vector), and the vectors
are concatenated in the order of the key, so oriented couplings stay oriented.

Contents: `decompose`, `typed_keys`, `readout`; SLOT (readout width of one component in units of
the embedding width) and CONTEXT_ATOMS (families with sequence context).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from .. import terms as T

if TYPE_CHECKING:
    import jax

    from ..model import MolSpec
    from ..topology import Topology

SLOT = 5  # readout width of one component, in units of the embedding width

# atoms whose residues give a family's instances their sequence context (residues i-1, i, i+1)
CONTEXT_ATOMS = {"cmap": lambda I: np.asarray(I["q"])[:, [0, 2, 4]]}


def decompose(family: str, top: Topology) -> dict:
    """Return the index arrays, components and skeletons of every instance of `family` in `top`.

    Returns
    -------
    dict
        "I" the family's index arrays; "n" instances; "n_slots" largest number of key components;
        "skel_of" skeleton per instance; "skeletons" sorted unique skeletons; "groups"
        {(slot, kind, n_atoms): (instance indices, atoms (k, n_atoms))} for batched readouts.
    """
    log = []

    def keyf(atoms: Sequence[int], kind: str) -> str:
        """Record the component (kind, atoms) and return a placeholder holding its log position."""
        log.append((kind, tuple(int(a) for a in atoms)))
        return f"\x00{len(log) - 1}\x00"  # placeholder: the component's position in the log

    idx, keys = T.REGISTRY[family].index(top, keyf)
    comps, skel_of = [], []
    for k in keys:
        parts = k.split("\x00")
        comps.append([log[int(i)] for i in parts[1::2]])
        skel_of.append("X".join(parts[0::2]))
    groups = {}  # (slot, kind, n_atoms) -> (instances, atoms)
    for inst, c in enumerate(comps):
        for slot, (kind, atoms) in enumerate(c):
            g = groups.setdefault((slot, kind, len(atoms)), ([], []))
            g[0].append(inst)
            g[1].append(atoms)
    return {
        "I": {k: np.asarray(v) for k, v in idx.items()},
        "n": len(keys),
        "n_slots": max([len(c) for c in comps] + [1]),
        "skel_of": skel_of,
        "skeletons": sorted(set(skel_of)),
        "groups": {k: (np.asarray(v[0], int), np.asarray(v[1], int)) for k, v in groups.items()},
    }


def typed_keys(spec: MolSpec, top: Topology, basis: Sequence[str], depth: int) -> dict:
    """Return the keys of the classical terms with atom environments to `depth` (0 = element-typed).

    Per family of `basis` and for the bond / angle reference values: {family: keys, "b0": bond keys,
    "th0": angle keys}, the rows of the typed tables.
    """
    from ..model import _classes

    cl = _classes(spec.elements, [tuple(b) for b in top.bonds], depth)

    def kt(atoms: Sequence[int], kind: str) -> str:
        """Return the typed key of a component (the atom environment for kind "atom")."""
        return top.key(atoms, kind, cl) if kind != "atom" else cl[atoms[0]]

    out = {f: list(T.REGISTRY[f].index(top, kt)[1]) for f in basis}
    out["b0"] = [kt(b, "bond") for b in top.bonds]
    out["th0"] = [kt(a, "angle") for a in top.angles]
    return out


def readout(kind: str, h: jax.Array, atoms: np.ndarray) -> jax.Array:
    """Return the description of one key component for k instances, jax.Array (k, <= SLOT * W).

    Parameters
    ----------
    kind : str
        Component kind ("atom", "bond", "pair", "angle", "torsion", "improper", "cmap", ...).
    h : jax.Array (n, W)
        Atom embeddings.
    atoms : np.ndarray (k, n_atoms) int
        Atoms of the component per instance.

    Notes
    -----
    Symmetric readouts (see bonded/nn/__init__.py): atom h_a; bond / pair [h_i + h_j, h_i * h_j];
    angle [h_j, h_i + h_k, h_i * h_k]; torsion [h_j + h_k, h_j * h_k, h_i + h_l, h_i * h_l,
    h_i * h_j + h_l * h_k]; improper [h_c, sum h_x, sum_{x<y} h_x * h_y]; cmap: the five embeddings
    concatenated in order (oriented); anything else [h_0, sum, sum of pair products].
    """
    H = [h[atoms[:, a]] for a in range(atoms.shape[1])]
    if kind == "cmap":  # oriented: C(i-1), N, CA, C, N(i+1)
        return jnp.concatenate(H, -1)
    if len(H) == 1:
        return H[0]
    if kind in ("bond", "pair") or len(H) == 2:
        return jnp.concatenate([H[0] + H[1], H[0] * H[1]], -1)
    if kind == "angle" and len(H) == 3:
        return jnp.concatenate([H[1], H[0] + H[2], H[0] * H[2]], -1)
    if kind == "torsion" and len(H) == 4:
        i, j, k, l = H
        return jnp.concatenate([j + k, j * k, i + l, i * l, i * j + l * k], -1)
    if kind == "improper" and len(H) == 4:
        c, a, b, d = H
        return jnp.concatenate([c, a + b + d, a * b + a * d + b * d], -1)
    S = sum(H)
    return jnp.concatenate([H[0], S, sum(x * y for p, x in enumerate(H) for y in H[p + 1 :])], -1)
