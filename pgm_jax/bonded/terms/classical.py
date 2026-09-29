"""Diagonal (class I) families and the Amber forms: bonds, angles, proper torsions, impropers."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .core import _N, Family, _dihedral, _mask_n, register


@register
class BondMorse(Family):
    name = "bond_morse"
    params = {"Kb": ((), 2.5e5)}               # kJ/mol/nm^2 (harmonic force constant 2 De a^2)
    linear = ()

    def index(self, top, keyf):
        return {"i": np.arange(len(top.bonds))}, [keyf(b, "bond") for b in top.bonds]

    def energy(self, G, dev, I, p):
        De = I["De"]
        a = jnp.sqrt(jnp.abs(p["Kb"]) / (2.0 * De))
        return jnp.sum(De * (1.0 - jnp.exp(-a * dev["db"][I["i"]])) ** 2)


@register
class BondHarm(Family):
    name = "bond_harm"
    params = {"Kb": ((), 2.5e5)}
    linear = ("Kb",)

    def index(self, top, keyf):
        return {"i": np.arange(len(top.bonds))}, [keyf(b, "bond") for b in top.bonds]

    def energy(self, G, dev, I, p):
        return jnp.sum(0.5 * p["Kb"] * dev["db"][I["i"]] ** 2)


@register
class BondQuartic(Family):
    """Quartic bond K2 db^2 / 2 + K3 db^3 + K4 db^4: a Morse bond to fourth order, as the O-H bonds of
    q-TIP4P/F; flexible enough to absorb the strong all-pair pGM electrostatics of a small molecule
    near its minimum (bounded below for K4 > 0)."""
    name = "bond_quartic"
    params = {"K2": ((), 2.5e5), "K3": ((), 0.0), "K4": ((), 0.0)}   # kJ/mol/nm^2, /nm^3, /nm^4
    linear = ("K2", "K3", "K4")

    def index(self, top, keyf):
        return {"i": np.arange(len(top.bonds))}, [keyf(b, "bond") for b in top.bonds]

    def energy(self, G, dev, I, p):
        db = dev["db"][I["i"]]
        return jnp.sum(0.5 * p["K2"] * db ** 2 + p["K3"] * db ** 3 + p["K4"] * db ** 4)


@register
class AngleCos(Family):
    name = "angle_cos"
    params = {"Ka": ((), 400.0)}                # kJ/mol
    linear = ("Ka",)

    def index(self, top, keyf):
        return {"i": np.arange(len(top.angles))}, [keyf(a, "angle") for a in top.angles]

    def energy(self, G, dev, I, p):
        return jnp.sum(p["Ka"] * dev["dc"][I["i"]] ** 2)


@register
class AngleHarm(Family):
    """Amber angle: 0.5 Ka (theta - theta0)^2 (Amber's K_theta = Ka / 2)."""
    name = "angle_harm"
    params = {"Ka": ((), 400.0)}                # kJ/mol/rad^2
    linear = ("Ka",)

    def index(self, top, keyf):
        return {"i": np.arange(len(top.angles))}, [keyf(a, "angle") for a in top.angles]

    def energy(self, G, dev, I, p):
        return jnp.sum(0.5 * p["Ka"] * dev["dth"][I["i"]] ** 2)


@register
class AngleCubic(Family):
    name = "angle_cubic"
    params = {"Ka3": ((), 0.0)}
    linear = ("Ka3",)

    def index(self, top, keyf):
        return {"i": np.arange(len(top.angles))}, [keyf(a, "angle") for a in top.angles]

    def energy(self, G, dev, I, p):
        return jnp.sum(p["Ka3"] * dev["dc"][I["i"]] ** 3)


@register
class Torsion(Family):
    name = "torsion"
    params = {"K": ((4,), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        return {"i": np.arange(len(top.propers)), "mask": _mask_n(top.rigid_torsion)}, \
               [keyf(t, "torsion") for t in top.propers]

    def energy(self, G, dev, I, p):
        phi = G["phi"][I["i"]]
        return jnp.sum(I["mask"] * p["K"] * (1.0 + jnp.cos(_N[None] * phi[:, None])))


@register
class TorsionAmber(Family):
    """Amber proper torsion: sum_n K_n (1 + cos n phi), n = 1..4, every torsion (Amber's PK with
    phase 0 is K_n; phase 180 is -K_n up to a constant)."""
    name = "torsion_amber"
    params = {"K": ((4,), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        return {"i": np.arange(len(top.propers))}, [keyf(t, "torsion") for t in top.propers]

    def energy(self, G, dev, I, p):
        phi = G["phi"][I["i"]]
        return jnp.sum(p["K"] * (1.0 + jnp.cos(_N[None] * phi[:, None])))


@register
class ImproperAmber(Family):
    """Amber improper: K (1 + cos(2 w - 180 deg)) = K (1 - cos 2w), w the dihedral a-b-c-d with the
    planar centre c third (Amber's PK is K)."""
    name = "improper_amber"
    params = {"K": ((), 5.0)}
    linear = ("K",)

    def index(self, top, keyf):
        quads = getattr(top, "amber_impropers", None)
        if quads is None:
            quads = [(a, b, c, d) for c, a, b, d in top.impropers]
        quads = np.asarray(quads, int).reshape(-1, 4)
        return {"q": quads}, [keyf((c, a, b, d), "improper") for a, b, c, d in quads]

    def energy(self, G, dev, I, p):
        q = I["q"]
        if len(q) == 0:
            return 0.0
        R = G["R"]
        w = _dihedral(R[q[:, 0]], R[q[:, 1]], R[q[:, 2]], R[q[:, 3]])
        return jnp.sum(p["K"] * (1.0 - jnp.cos(2.0 * w)))


@register
class Improper(Family):
    name = "improper"
    params = {"K": ((), 20.0)}
    linear = ("K",)

    def index(self, top, keyf):
        return {"i": np.arange(len(top.impropers))}, [keyf(m, "improper") for m in top.impropers]

    def energy(self, G, dev, I, p):
        if "imp" not in G:
            return 0.0
        return jnp.sum(p["K"][:, None] * jnp.sin(G["imp"][I["i"]]) ** 2)       # ~ K phi^2 about 0 or 180
