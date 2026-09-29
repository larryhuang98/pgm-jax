"""Core of the bonded term registry: internal coordinates of a frame, the Family interface and
the registry, and helpers shared by the families.

A family is an energy function of internal coordinates with its own index set, parameters and
tying keys (`index(top, keyf)` -> (arrays, keys), `params` {name: (per-key shape, init)},
`linear` (names entering the energy linearly), `energy(G, dev, I, p)`).  Every family reads the
geometry dict G of one frame and the deviations from the shared reference values
(db = b - b0, dc = cos - cos th0, dth = th - th0).  All energies kJ/mol, lengths nm, angles rad."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# paper takes experimental dissociation energies; only the anharmonicity depends on them)
_DE = {
    ("C", "H", 1): 413,
    ("C", "C", 1): 348,
    ("C", "C", 2): 614,
    ("C", "C", 1.5): 518,
    ("C", "N", 1): 305,
    ("C", "N", 2): 615,
    ("C", "N", 1.5): 540,
    ("C", "O", 1): 358,
    ("C", "O", 2): 745,
    ("C", "O", 1.5): 550,
    ("H", "O", 1): 463,
    ("H", "N", 1): 391,
    ("H", "S", 1): 363,
    ("C", "S", 1): 272,
    ("C", "F", 1): 485,
    ("C", "Cl", 1): 328,
    ("O", "P", 1): 335,
    ("O", "P", 2): 544,
}


def morse_depth(e1, e2, order):
    k = tuple(sorted((e1, e2))) + (float(order) if order in (1.5,) else int(round(order)),)
    if k in _DE:
        return float(_DE[k])
    k1 = tuple(sorted((e1, e2))) + (1,)
    return float(_DE.get(k1, 400.0))


# ------------------------------------------------------------------ geometry
def _dihedral(x0, x1, x2, x3):
    """Dihedral angle x0-x1-x2-x3 (rad) in the IUPAC sign convention (as Amber, RDKit): positive
    for a clockwise rotation of the front bond looking along x1 -> x2.  (Before the CMAP family the
    sign was reversed; every family except cmap is even in the dihedrals, so energies are unchanged.)"""
    b0, b1, b2 = x1 - x0, x2 - x1, x3 - x2
    n1, n2 = jnp.cross(b0, b1), jnp.cross(b1, b2)
    m1 = jnp.cross(n1, b1 / jnp.linalg.norm(b1, axis=-1, keepdims=True))
    return jnp.arctan2(-jnp.sum(m1 * n2, -1), jnp.sum(n1 * n2, -1))


def geometry(R, top):
    """Internal coordinates of one frame R (n, 3) nm (and R itself for vector-based families)."""
    G = {"R": R}
    b = top.bonds
    G["b"] = jnp.linalg.norm(R[b[:, 0]] - R[b[:, 1]], axis=-1)
    a = top.angles
    u, v = R[a[:, 0]] - R[a[:, 1]], R[a[:, 2]] - R[a[:, 1]]
    c = jnp.sum(u * v, -1) / (jnp.linalg.norm(u, axis=-1) * jnp.linalg.norm(v, axis=-1))
    G["cos"] = jnp.clip(c, -1.0 + 1e-12, 1.0 - 1e-12)
    G["th"] = jnp.arccos(G["cos"])
    t = top.propers
    G["phi"] = _dihedral(R[t[:, 0]], R[t[:, 1]], R[t[:, 2]], R[t[:, 3]]) if len(t) else jnp.zeros(0)
    im = top.impropers
    if len(im):
        c0, a0, b0, d0 = im.T
        G["imp"] = jnp.stack(
            [
                _dihedral(R[a0], R[b0], R[c0], R[d0]),
                _dihedral(R[b0], R[d0], R[c0], R[a0]),
                _dihedral(R[d0], R[a0], R[c0], R[b0]),
            ],
            -1,
        )
        th = []
        for p, q in ((a0, b0), (b0, d0), (a0, d0)):
            u, v = R[p] - R[c0], R[q] - R[c0]
            th.append(
                jnp.arccos(
                    jnp.clip(
                        jnp.sum(u * v, -1) / (jnp.linalg.norm(u, axis=-1) * jnp.linalg.norm(v, axis=-1)),
                        -1 + 1e-12,
                        1 - 1e-12,
                    )
                )
            )
        G["pyr"] = 2 * jnp.pi - (th[0] + th[1] + th[2])
    for name in ("pairs13", "pairs14", "pairs15"):
        p = getattr(top, name)
        if p is None:  # pairs15 is not built for large molecules
            continue
        G["r" + name[-2:]] = jnp.linalg.norm(R[p[:, 0]] - R[p[:, 1]], axis=-1) if len(p) else jnp.zeros(0)
    return G


# ------------------------------------------------------------------ families
class Family:
    name = ""
    params: dict = {}  # name -> (per-key shape, default init)
    linear: tuple = ()
    needs: tuple = ()  # families whose reference values it uses ("ref" always available)

    def index(self, top, keyf):
        raise NotImplementedError

    def energy(self, G, dev, I, p):
        raise NotImplementedError


REGISTRY: dict[str, Family] = {}


def register(cls):
    REGISTRY[cls.name] = cls()
    return cls


def _mask_n(rigid):
    """(nt, 4) periodicity mask: flexible torsions n = 1..4, rigid ones n = 2 only (paper)."""
    m = np.ones((len(rigid), 4))
    m[np.asarray(rigid, bool)] = [0, 1, 0, 0]
    return m


_N = jnp.arange(1, 5, dtype=float)


def _pair_index(pairs):
    return (
        {"u": np.asarray(pairs)[:, 0], "v": np.asarray(pairs)[:, 1]}
        if len(pairs)
        else {"u": np.zeros(0, int), "v": np.zeros(0, int)}
    )
