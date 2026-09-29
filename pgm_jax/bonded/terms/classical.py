"""Define the diagonal (class I) families and the Amber forms: bonds, angles, proper torsions, impropers.

Registered families: bond_morse, bond_harm, bond_quartic (bonds); angle_cos, angle_harm,
angle_cubic (angles); torsion, torsion_amber (proper torsions); improper, improper_amber
(out-of-plane).  bond_morse, angle_cos, torsion and improper belong to the class II set of
Abdullah et al. [1]_ (terms.PAPER); bond_harm, angle_harm, torsion_amber and improper_amber are
Amber's forms (terms.AMBER, bonded/amber.py).  The interface is described in terms/core.py
(`Family`); db, dc, dth are the deviations from the shared reference values b0, th0.

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

from .core import _N, Family, _dihedral, _mask_n, register

if TYPE_CHECKING:
    import jax

    from ..topology import Topology


@register
class BondMorse(Family):
    """Morse bond, E = De (1 - exp(-a db))^2 with a = sqrt(|Kb| / (2 De)).

    Kb [kJ/mol/nm^2] is the harmonic force constant 2 De a^2 at the minimum (fitted, per key); De
    [kJ/mol] is fixed per bond key from `morse_depth` (index extra "De"), so only the anharmonicity
    depends on the table.
    """

    name = "bond_morse"
    params = {"Kb": ((), 2.5e5)}  # kJ/mol/nm^2 (harmonic force constant 2 De a^2)
    linear = ()

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per bond (key kind "bond")."""
        return {"i": np.arange(len(top.bonds))}, [keyf(b, "bond") for b in top.bonds]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum De (1 - exp(-a db))^2 [kJ/mol] (see the class docstring)."""
        De = I["De"]
        a = jnp.sqrt(jnp.abs(p["Kb"]) / (2.0 * De))
        return jnp.sum(De * (1.0 - jnp.exp(-a * dev["db"][I["i"]])) ** 2)


@register
class BondHarm(Family):
    """Harmonic bond (Amber form), E = 0.5 Kb db^2, Kb [kJ/mol/nm^2] (Amber's K_b = Kb / 2)."""

    name = "bond_harm"
    params = {"Kb": ((), 2.5e5)}
    linear = ("Kb",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per bond (key kind "bond")."""
        return {"i": np.arange(len(top.bonds))}, [keyf(b, "bond") for b in top.bonds]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum 0.5 Kb db^2 [kJ/mol]."""
        return jnp.sum(0.5 * p["Kb"] * dev["db"][I["i"]] ** 2)


@register
class BondQuartic(Family):
    """Quartic bond, E = K2 db^2 / 2 + K3 db^3 + K4 db^4: a Morse bond to fourth order.

    As the O-H bonds of q-TIP4P/F; flexible enough to absorb the strong all-pair pGM electrostatics
    of a small molecule near its minimum (bounded below for K4 > 0).  K2 [kJ/mol/nm^2], K3
    [kJ/mol/nm^3], K4 [kJ/mol/nm^4].
    """

    name = "bond_quartic"
    params = {"K2": ((), 2.5e5), "K3": ((), 0.0), "K4": ((), 0.0)}  # kJ/mol/nm^2, /nm^3, /nm^4
    linear = ("K2", "K3", "K4")

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per bond (key kind "bond")."""
        return {"i": np.arange(len(top.bonds))}, [keyf(b, "bond") for b in top.bonds]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K2 db^2 / 2 + K3 db^3 + K4 db^4 [kJ/mol]."""
        db = dev["db"][I["i"]]
        return jnp.sum(0.5 * p["K2"] * db**2 + p["K3"] * db**3 + p["K4"] * db**4)


@register
class AngleCos(Family):
    """Cosine-harmonic angle (class II set), E = Ka (cos th - cos th0)^2, Ka [kJ/mol]."""

    name = "angle_cos"
    params = {"Ka": ((), 400.0)}  # kJ/mol
    linear = ("Ka",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per angle (key kind "angle")."""
        return {"i": np.arange(len(top.angles))}, [keyf(a, "angle") for a in top.angles]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum Ka dc^2 [kJ/mol]."""
        return jnp.sum(p["Ka"] * dev["dc"][I["i"]] ** 2)


@register
class AngleHarm(Family):
    """Amber angle, E = 0.5 Ka (th - th0)^2, Ka [kJ/mol/rad^2] (Amber's K_theta = Ka / 2)."""

    name = "angle_harm"
    params = {"Ka": ((), 400.0)}  # kJ/mol/rad^2
    linear = ("Ka",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per angle (key kind "angle")."""
        return {"i": np.arange(len(top.angles))}, [keyf(a, "angle") for a in top.angles]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum 0.5 Ka dth^2 [kJ/mol]."""
        return jnp.sum(0.5 * p["Ka"] * dev["dth"][I["i"]] ** 2)


@register
class AngleCubic(Family):
    """Cubic angle correction, E = Ka3 (cos th - cos th0)^3, Ka3 [kJ/mol] (F4)."""

    name = "angle_cubic"
    params = {"Ka3": ((), 0.0)}
    linear = ("Ka3",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per angle (key kind "angle")."""
        return {"i": np.arange(len(top.angles))}, [keyf(a, "angle") for a in top.angles]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum Ka3 dc^3 [kJ/mol]."""
        return jnp.sum(p["Ka3"] * dev["dc"][I["i"]] ** 3)


@register
class Torsion(Family):
    """Proper torsion (class II set), E = sum_n K_n (1 + cos n phi), n = 1..4, K [kJ/mol] shape (4,).

    Torsions about a ring bond or a bond of order > 1 keep only n = 2 (`_mask_n`, as the paper).
    """

    name = "torsion"
    params = {"K": ((4,), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per proper torsion (key kind "torsion") with the periodicity mask."""
        return {"i": np.arange(len(top.propers)), "mask": _mask_n(top.rigid_torsion)}, [
            keyf(t, "torsion") for t in top.propers
        ]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum mask_n K_n (1 + cos n phi) [kJ/mol]."""
        phi = G["phi"][I["i"]]
        return jnp.sum(I["mask"] * p["K"] * (1.0 + jnp.cos(_N[None] * phi[:, None])))


@register
class TorsionAmber(Family):
    """Amber proper torsion, E = sum_n K_n (1 + cos n phi), n = 1..4, for every torsion (no mask).

    Amber's PK with phase 0 is K_n; phase 180 deg is -K_n up to a constant.  K [kJ/mol] shape (4,).
    """

    name = "torsion_amber"
    params = {"K": ((4,), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per proper torsion (key kind "torsion")."""
        return {"i": np.arange(len(top.propers))}, [keyf(t, "torsion") for t in top.propers]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K_n (1 + cos n phi) [kJ/mol]."""
        phi = G["phi"][I["i"]]
        return jnp.sum(p["K"] * (1.0 + jnp.cos(_N[None] * phi[:, None])))


@register
class ImproperAmber(Family):
    """Amber improper, E = K (1 + cos(2 w - 180 deg)) = K (1 - cos 2w), K [kJ/mol] (Amber's PK).

    w is the dihedral a-b-c-d with the planar centre c third.  The quadruples come from
    `Topology.amber_impropers` (Amber's atom order, bonded/amber.with_amber_impropers) when set,
    else from the topology's planar centres (c, a, b, d) reordered to (a, b, c, d).
    """

    name = "improper_amber"
    params = {"K": ((), 5.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the improper quadruples ("q", centre third) with keys of kind "improper"."""
        quads = getattr(top, "amber_impropers", None)
        if quads is None:
            quads = [(a, b, c, d) for c, a, b, d in top.impropers]
        quads = np.asarray(quads, int).reshape(-1, 4)
        return {"q": quads}, [keyf((c, a, b, d), "improper") for a, b, c, d in quads]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K (1 - cos 2w) [kJ/mol] (0.0 without impropers)."""
        q = I["q"]
        if len(q) == 0:
            return 0.0
        R = G["R"]
        w = _dihedral(R[q[:, 0]], R[q[:, 1]], R[q[:, 2]], R[q[:, 3]])
        return jnp.sum(p["K"] * (1.0 - jnp.cos(2.0 * w)))


@register
class Improper(Family):
    """Out-of-plane term of the class II set, E = K sum_{3 dihedrals} sin^2(w), K [kJ/mol].

    Uses the three improper dihedrals of each planar 3-coordinated centre (`geometry` "imp");
    ~ K w^2 about 0 or 180 deg.
    """

    name = "improper"
    params = {"K": ((), 20.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per planar centre (key kind "improper")."""
        return {"i": np.arange(len(top.impropers))}, [keyf(m, "improper") for m in top.impropers]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K sin^2 over the three improper dihedrals [kJ/mol] (0.0 without impropers)."""
        if "imp" not in G:
            return 0.0
        return jnp.sum(p["K"][:, None] * jnp.sin(G["imp"][I["i"]]) ** 2)  # ~ K phi^2 about 0 or 180
