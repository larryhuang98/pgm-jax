"""Define the core of the bonded term registry: internal coordinates, the Family interface, helpers.

Contents: `geometry` (internal coordinates of one frame), `Family` (the interface of a term
family), `REGISTRY` and `register` (families by name), `morse_depth` (Morse well depths by bond
type), and helpers shared by the family modules (`_dihedral`, `_mask_n`, `_pair_index`, `_N`).

A family is an energy function of internal coordinates with its own index set, parameters and
tying keys: `index(top, keyf)` -> (index arrays, one key per instance), `params` {name:
(per-key shape, init)}, `linear` (names entering the energy linearly), `energy(G, dev, I, p)`.
Every family reads the geometry dict G of one frame and the deviations from the shared
reference values (db = b - b0, dc = cos th - cos th0, dth = th - th0), so couplings and
diagonal terms use one set of reference values, fitted jointly (bonded/model.py).

Units: energies kJ/mol, lengths nm, angles rad.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

if TYPE_CHECKING:
    import jax

    from ..topology import Topology

# Morse well depths De (kJ/mol) by (element, element, bond order), from bond dissociation energies (as the
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


def morse_depth(e1: str, e2: str, order: float) -> float:
    """Return the Morse well depth De [kJ/mol] of a bond from a table of dissociation energies.

    Parameters
    ----------
    e1, e2 : str
        Elements of the two atoms (any order).
    order : float
        Bond order (1, 1.5 aromatic, 2, 3).

    Returns
    -------
    float
        The table value for (elements, order); else the single-bond value of the element pair;
        else 400 kJ/mol.
    """
    k = tuple(sorted((e1, e2))) + (float(order) if order in (1.5,) else int(round(order)),)
    if k in _DE:
        return float(_DE[k])
    k1 = tuple(sorted((e1, e2))) + (1,)
    return float(_DE.get(k1, 400.0))


# ------------------------------------------------------------------ geometry
def _dihedral(x0: jax.Array, x1: jax.Array, x2: jax.Array, x3: jax.Array) -> jax.Array:
    """Return the dihedral angle x0-x1-x2-x3 [rad] in (-pi, pi] (arrays (..., 3) [nm], batched over leading axes).

    IUPAC sign convention (as Amber, RDKit): positive for a clockwise rotation of the front bond
    looking along x1 -> x2.  (Before the CMAP family the sign was reversed; every family except cmap
    is even in the dihedrals, so energies are unchanged.)
    """
    b0, b1, b2 = x1 - x0, x2 - x1, x3 - x2
    n1, n2 = jnp.cross(b0, b1), jnp.cross(b1, b2)
    m1 = jnp.cross(n1, b1 / jnp.linalg.norm(b1, axis=-1, keepdims=True))
    return jnp.arctan2(-jnp.sum(m1 * n2, -1), jnp.sum(n1 * n2, -1))


def geometry(R: jax.Array, top: Topology) -> dict[str, jax.Array]:
    """Return the internal coordinates of one frame (and the positions, for vector-based families).

    Parameters
    ----------
    R : jax.Array (n, 3)
        Atom positions [nm].
    top : Topology
        Valence topology of the molecule (bonded/topology.py).

    Returns
    -------
    dict
        "R" (n, 3) positions [nm]; "b" (nb,) bond lengths [nm]; "cos" (na,) angle cosines
        (clipped to +-(1 - 1e-12)); "th" (na,) angles [rad]; "phi" (nt,) proper dihedrals [rad];
        for molecules with impropers, "imp" (ni, 3) the three dihedrals a-b-c-d, b-d-c-a, d-a-c-b of
        each planar centre c with neighbours (a, b, d) [rad], and "pyr" (ni,) 2 pi minus the sum of
        the three angles at the centre [rad]; "r13", "r14", "r15" topological pair distances [nm]
        ("r15" only when the topology has pairs15).

    Notes
    -----
    Differentiable in R (used under jax.grad / jax.vmap by the fitter and the MD templates).
    """
    G = {"R": R}
    b = top.bonds
    G["b"] = jnp.linalg.norm(R[b[:, 0]] - R[b[:, 1]], axis=-1)
    a = top.angles
    u, v = R[a[:, 0]] - R[a[:, 1]], R[a[:, 2]] - R[a[:, 1]]
    c = jnp.sum(u * v, -1) / (jnp.linalg.norm(u, axis=-1) * jnp.linalg.norm(v, axis=-1))
    G["cos"] = jnp.clip(c, -1.0 + 1e-12, 1.0 - 1e-12)  # keeps arccos and its gradient finite at 0 and 180 deg
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
        G["pyr"] = 2 * jnp.pi - (th[0] + th[1] + th[2])  # 0 for a planar centre
    for name in ("pairs13", "pairs14", "pairs15"):
        p = getattr(top, name)
        if p is None:  # pairs15 is not built for large molecules
            continue
        G["r" + name[-2:]] = jnp.linalg.norm(R[p[:, 0]] - R[p[:, 1]], axis=-1) if len(p) else jnp.zeros(0)
    return G


# ------------------------------------------------------------------ families
class Family:
    """Interface of a bonded term family (subclasses are registered singletons, see `register`).

    A family defines `index` (its term instances in a topology and their tying keys) and `energy`
    (its energy for one frame).  Parameters are tied by key: all instances with the same key share
    one row of each parameter array (bonded/model.py `BondedTerms`), or get their own values from
    the neural bonded model (bonded/nn).

    Attributes
    ----------
    name : str
        Registry name (class attribute).
    params : dict
        {name: (per-key shape, init)}: shape () or e.g. (4,) for the four torsion multiplicities;
        init a float (the default value) or None (per-instance values from the reference geometry:
        `init_from_geometry`, or r0 of the pair families).
    linear : tuple of str
        Parameters that enter the energy linearly (for L1 penalties).
    needs : tuple of str
        Families whose reference values it uses ("ref", the bond / angle references, is always
        available).
    """

    name = ""
    params: dict = {}  # name -> (per-key shape, default init)
    linear: tuple = ()
    needs: tuple = ()  # families whose reference values it uses ("ref" always available)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the term instances of the family in a topology and their tying keys.

        Parameters
        ----------
        top : Topology
            Valence topology of the molecule.
        keyf : callable
            keyf(atoms, kind) -> str, the tying key of a component (kind "bond", "angle", "torsion",
            "improper", "pair", "atom", "cmap"); `Topology.key` with the atom classes of the typing.

        Returns
        -------
        index : dict of np.ndarray
            Index arrays of the instances (static for jit), read by `energy` as `I`.
        keys : list of str
            One tying key per instance.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return the family's energy of one frame [kJ/mol].

        Parameters
        ----------
        G : dict of jax.Array
            Internal coordinates of the frame (`geometry`).
        dev : dict of jax.Array
            Deviations from the reference values: "db" (nb,) [nm], "dc" (na,) cosine deviations
            (dimensionless), "dth" (na,) [rad].
        I : dict of np.ndarray
            The family's index arrays (`index`), plus "k" (key index per instance) and extras set by the
            model ("De" Morse depths [kJ/mol], "bij" Gaussian pair exponents [1/nm]).
        p : dict of jax.Array
            Parameters per instance: p[name] of shape (instances, *per-key shape).

        Returns
        -------
        jax.Array or float
            Energy [kJ/mol] (0.0 when the family has no instances).

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError


REGISTRY: dict[str, Family] = {}


def register(cls: type[Family]) -> type[Family]:
    """Register an instance of the family class `cls` in REGISTRY under `cls.name` (class decorator)."""
    REGISTRY[cls.name] = cls()
    return cls


def _mask_n(rigid: ArrayLike) -> np.ndarray:
    """Return the (nt, 4) torsion periodicity mask: flexible torsions n = 1..4, rigid ones n = 2 only (paper).

    `rigid` (nt,) bool marks torsions about a ring bond or a bond of order > 1
    (`Topology.rigid_torsion`).
    """
    m = np.ones((len(rigid), 4))
    m[np.asarray(rigid, bool)] = [0, 1, 0, 0]
    return m


_N = np.arange(1, 5, dtype=float)  # torsion multiplicities 1..4 (numpy: no JAX array at import time)


def _pair_index(pairs: ArrayLike) -> dict[str, np.ndarray]:
    """Return {"u": first column, "v": second column} of an (m, 2) index array (int arrays, empty if m = 0)."""
    return (
        {"u": np.asarray(pairs)[:, 0], "v": np.asarray(pairs)[:, 1]}
        if len(pairs)
        else {"u": np.zeros(0, int), "v": np.zeros(0, int)}
    )
