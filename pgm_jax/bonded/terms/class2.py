"""Define the class II coupling families [1]_ and the extended quadratic couplings (F3, F4).

Registered families: bond_bond, bond_angle, angle_angle, torsion_bond, torsion_angle, aat
(class II, terms.PAPER; their index sets are built by bonded/topology.py), torsion_mod (F3: the
torsion-bond / torsion-angle couplings factorised), bond_angle_x and angle_angle_x (F4: all
bond-angle and angle-angle pairs sharing an atom).  The couplings are products of the deviations
db = b - b0 [nm], dc = cos th - cos th0 and dth = th - th0 [rad] from the shared reference
values (terms/core.py).

Units: energies kJ/mol, lengths nm, angles rad.

References
----------
.. [1] A. S. Abdullah, Y. Wang, M. F. S. J. Menger, S. Sami, T. Head-Gordon, J. Chem. Theory
   Comput. 21, 11669 (2025). doi:10.1021/acs.jctc.5c01458
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from .core import _N, Family, _mask_n, _pair_index, register

if TYPE_CHECKING:
    import jax

    from ..topology import Topology


@register
class BondBond(Family):
    """Bond-bond coupling of the two bonds of every angle, E = K db_1 db_2, K [kJ/mol/nm^2]."""

    name = "bond_bond"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the bond pairs of every angle ("u", "v"), keyed by the angle ("bb|...")."""
        return _pair_index(top.bond_bond), ["bb|" + keyf(a, "angle") for a in top.angles]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K db_u db_v [kJ/mol]."""
        return jnp.sum(p["K"] * dev["db"][I["u"]] * dev["db"][I["v"]])


@register
class BondAngle(Family):
    """Bond-angle coupling of each arm of an angle with the angle, E = K db dc, K [kJ/mol/nm].

    The key is the angle and the bond's outer atom, so the two arms of an asymmetric angle get
    their own constants.
    """

    name = "bond_angle"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the (bond, angle) pairs ("u", "v"), keyed by angle and outer atom ("ba|...")."""
        keys = []
        for bi, ai in top.bond_angle:
            i, j, k = top.angles[ai]
            outer = [x for x in top.bonds[bi] if x != j][0]
            keys.append("ba|" + keyf(top.angles[ai], "angle") + "|" + keyf([outer], "atom"))
        return _pair_index(top.bond_angle), keys

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K db_u dc_v [kJ/mol]."""
        return jnp.sum(p["K"] * dev["db"][I["u"]] * dev["dc"][I["v"]])


def _aa_key(top: Topology, keyf: Callable[[Sequence[int], str], str], m1: int, m2: int) -> str:
    """Return the tying key of the angle pair (m1, m2).

    "aa|centre|shared arm|other arms" for two angles at the same centre sharing one arm, else
    "aax|" with the two sorted angle keys (the extended pairs of angle_angle_x).
    """
    a1, a2 = top.angles[m1], top.angles[m2]
    shared = sorted(set([a1[0], a1[2]]) & set([a2[0], a2[2]]))
    others = sorted(set([a1[0], a1[2], a2[0], a2[2]]) - set(shared))

    def cl(x: int) -> str:
        """Return the atom key of atom x."""
        return keyf([x], "atom")

    if len(set(a1) & set(a2)) >= 2 and a1[1] == a2[1] and shared:
        return "aa|" + cl(a1[1]) + "|" + cl(shared[0]) + "|" + "-".join(sorted(cl(o) for o in others))
    return "aax|" + "-".join(sorted([keyf(a1, "angle"), keyf(a2, "angle")]))


@register
class AngleAngle(Family):
    """Angle-angle coupling of angle pairs with the same centre sharing one arm, E = K dc_1 dc_2, K [kJ/mol]."""

    name = "angle_angle"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the angle pairs ("u", "v") of Topology.angle_angle with their `_aa_key` keys."""
        return _pair_index(top.angle_angle), [_aa_key(top, keyf, m1, m2) for m1, m2 in top.angle_angle]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K dc_u dc_v [kJ/mol]."""
        return jnp.sum(p["K"] * dev["dc"][I["u"]] * dev["dc"][I["v"]])


@register
class TorsionBond(Family):
    """Torsion-bond coupling, E = sum_n K_n db (1 + cos n phi) for the three bonds of every torsion.

    K [kJ/mol/nm] shape (4,), masked like `Torsion` (rigid torsions: n = 2 only).  The key
    distinguishes the middle bond ("mid") from an end bond ("end:" + bond key).
    """

    name = "torsion_bond"
    params = {"K": ((4,), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the (torsion, bond) pairs ("u", "v") with the periodicity mask and keys "tb|..."."""
        keys = []
        for t, bi in top.torsion_bond:
            i, j, k, l = top.propers[t]
            mid = set(top.bonds[bi]) == {j, k}
            keys.append(
                "tb|" + keyf(top.propers[t], "torsion") + "|" + ("mid" if mid else "end:" + keyf(top.bonds[bi], "bond"))
            )
        I = _pair_index(top.torsion_bond)
        I["mask"] = _mask_n(top.rigid_torsion)[I["u"]] if len(top.torsion_bond) else np.zeros((0, 4))
        return I, keys

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum mask_n K_n db_v (1 + cos n phi_u) [kJ/mol]."""
        phi = G["phi"][I["u"]]
        return jnp.sum(I["mask"] * p["K"] * dev["db"][I["v"]][:, None] * (1.0 + jnp.cos(_N[None] * phi[:, None])))


@register
class TorsionAngle(Family):
    """Torsion-angle coupling, E = sum_n K_n dc (1 + cos n phi) for the two angles of every torsion.

    K [kJ/mol] shape (4,), masked like `Torsion`.
    """

    name = "torsion_angle"
    params = {"K": ((4,), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the (torsion, angle) pairs ("u", "v") with the periodicity mask and keys "ta|..."."""
        keys = [
            "ta|" + keyf(top.propers[t], "torsion") + "|" + keyf(top.angles[a], "angle") for t, a in top.torsion_angle
        ]
        I = _pair_index(top.torsion_angle)
        I["mask"] = _mask_n(top.rigid_torsion)[I["u"]] if len(top.torsion_angle) else np.zeros((0, 4))
        return I, keys

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum mask_n K_n dc_v (1 + cos n phi_u) [kJ/mol]."""
        phi = G["phi"][I["u"]]
        return jnp.sum(I["mask"] * p["K"] * dev["dc"][I["v"]][:, None] * (1.0 + jnp.cos(_N[None] * phi[:, None])))


@register
class TorsionModulated(Family):
    """F3: torsion modulated by its bonds and angles, the torsion-bond / torsion-angle couplings factorised.

        E = sum_n K_n (1 + cos n phi) (1 + l_mid db_mid + l_end (db_end1 + db_end2) + l_ang (dc_1 + dc_2))

    with 3 coupling parameters per torsion type shared by all periodicities: K [kJ/mol] shape (4,)
    (masked like `Torsion`), l_mid and l_end [1/nm], l_ang (dimensionless).
    """

    name = "torsion_mod"
    params = {"K": ((4,), 0.0), "l_mid": ((), 0.0), "l_end": ((), 0.0), "l_ang": ((), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return per torsion: its index "i", its three bonds "b" (end, middle, end), its two angles "a", the mask."""
        nt = len(top.propers)
        tb = np.asarray(top.torsion_bond).reshape(nt, 3, 2)[:, :, 1] if nt else np.zeros((0, 3), int)
        ta = np.asarray(top.torsion_angle).reshape(nt, 2, 2)[:, :, 1] if nt else np.zeros((0, 2), int)
        return {"i": np.arange(nt), "b": tb, "a": ta, "mask": _mask_n(top.rigid_torsion)}, [
            keyf(t, "torsion") for t in top.propers
        ]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return the modulated torsion energy [kJ/mol] (see the class docstring)."""
        phi = G["phi"][I["i"]]
        db, dc = dev["db"][I["b"]], dev["dc"][I["a"]]
        amp = 1.0 + p["l_mid"] * db[:, 1] + p["l_end"] * (db[:, 0] + db[:, 2]) + p["l_ang"] * (dc[:, 0] + dc[:, 1])
        return jnp.sum(amp[:, None] * I["mask"] * p["K"] * (1.0 + jnp.cos(_N[None] * phi[:, None])))


@register
class AngleAngleTorsion(Family):
    """Angle-angle-torsion coupling (aat) for the two angles of every torsion.

    E = K dth_1 dth_2 cos phi, K [kJ/mol/rad^2].
    """

    name = "aat"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return per torsion its index "t" and its two angles "a1", "a2" (keys "aat|...")."""
        a = np.asarray(top.aat).reshape(-1, 3)
        return {"t": a[:, 0], "a1": a[:, 1], "a2": a[:, 2]}, ["aat|" + keyf(top.propers[t], "torsion") for t in a[:, 0]]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K dth_a1 dth_a2 cos phi_t [kJ/mol]."""
        return jnp.sum(p["K"] * dev["dth"][I["a1"]] * dev["dth"][I["a2"]] * jnp.cos(G["phi"][I["t"]]))


def _share_sets(top: Topology) -> tuple[np.ndarray, np.ndarray]:
    """Return the extended coupling sets (F4): bond-angle and angle-angle pairs sharing at least one atom.

    Bond-angle pairs exclude the angle's own arms (bond_angle); angle-angle pairs exclude the pairs
    with the same centre sharing one arm (angle_angle).  O(nb na + na^2).

    Returns
    -------
    ba : np.ndarray (k, 2) int
        (bond, angle) pairs.
    aa : np.ndarray (k', 2) int
        (angle, angle) pairs.
    """
    ba, aa = [], []
    for bi, b in enumerate(top.bonds):
        for ai, a in enumerate(top.angles):
            if set(b) & set(a) and not set(b) <= set(a):
                ba.append((bi, ai))
    for m1 in range(len(top.angles)):
        for m2 in range(m1 + 1, len(top.angles)):
            s = set(top.angles[m1]) & set(top.angles[m2])
            if s and not (top.angles[m1][1] == top.angles[m2][1] and len(s) == 2):
                aa.append((m1, m2))
    return np.array(ba, int).reshape(-1, 2), np.array(aa, int).reshape(-1, 2)


@register
class BondAngleX(Family):
    """F4: bond-angle coupling for every bond and angle sharing an atom (not an arm), E = K db dc, K [kJ/mol/nm]."""

    name = "bond_angle_x"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the (bond, angle) pairs of `_share_sets` ("u", "v"), keys "bax|bond|angle"."""
        ba, _ = _share_sets(top)
        return _pair_index(ba), [
            "bax|" + keyf(top.bonds[b], "bond") + "|" + keyf(top.angles[a], "angle") for b, a in ba
        ]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K db_u dc_v [kJ/mol]."""
        return jnp.sum(p["K"] * dev["db"][I["u"]] * dev["dc"][I["v"]])


@register
class AngleAngleX(Family):
    """F4: angle-angle coupling for every other angle pair sharing an atom, E = K dc_1 dc_2, K [kJ/mol]."""

    name = "angle_angle_x"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the (angle, angle) pairs of `_share_sets` ("u", "v") with their `_aa_key` keys."""
        _, aa = _share_sets(top)
        return _pair_index(aa), [_aa_key(top, keyf, m1, m2) for m1, m2 in aa]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K dc_u dc_v [kJ/mol]."""
        return jnp.sum(p["K"] * dev["dc"][I["u"]] * dev["dc"][I["v"]])


# The Amber / GAFF functional forms (harmonic bonds and angles, Fourier torsions, impropers) are in
# classical.py; use them with lj14_scale = 0.5 and typing = "amber" to tune GAFF-like parameters (bonded/amber.py)
