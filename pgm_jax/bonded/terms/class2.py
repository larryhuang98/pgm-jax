"""Class II couplings (Abdullah et al. 2025) and extended quadratic couplings (F3, F4)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .core import Family, _mask_n, _N, _pair_index, register


@register
class BondBond(Family):
    name = "bond_bond"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        return _pair_index(top.bond_bond), ["bb|" + keyf(a, "angle") for a in top.angles]

    def energy(self, G, dev, I, p):
        return jnp.sum(p["K"] * dev["db"][I["u"]] * dev["db"][I["v"]])


@register
class BondAngle(Family):
    name = "bond_angle"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        keys = []
        for bi, ai in top.bond_angle:
            i, j, k = top.angles[ai]
            outer = [x for x in top.bonds[bi] if x != j][0]
            keys.append("ba|" + keyf(top.angles[ai], "angle") + "|" + keyf([outer], "atom"))
        return _pair_index(top.bond_angle), keys

    def energy(self, G, dev, I, p):
        return jnp.sum(p["K"] * dev["db"][I["u"]] * dev["dc"][I["v"]])


def _aa_key(top, keyf, m1, m2):
    a1, a2 = top.angles[m1], top.angles[m2]
    shared = sorted(set([a1[0], a1[2]]) & set([a2[0], a2[2]]))
    others = sorted(set([a1[0], a1[2], a2[0], a2[2]]) - set(shared))
    cl = lambda x: keyf([x], "atom")
    if len(set(a1) & set(a2)) >= 2 and a1[1] == a2[1] and shared:
        return "aa|" + cl(a1[1]) + "|" + cl(shared[0]) + "|" + "-".join(sorted(cl(o) for o in others))
    return "aax|" + "-".join(sorted([keyf(a1, "angle"), keyf(a2, "angle")]))


@register
class AngleAngle(Family):
    name = "angle_angle"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        return _pair_index(top.angle_angle), [_aa_key(top, keyf, m1, m2) for m1, m2 in top.angle_angle]

    def energy(self, G, dev, I, p):
        return jnp.sum(p["K"] * dev["dc"][I["u"]] * dev["dc"][I["v"]])


@register
class TorsionBond(Family):
    name = "torsion_bond"
    params = {"K": ((4,), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        keys = []
        for t, bi in top.torsion_bond:
            i, j, k, l = top.propers[t]
            mid = set(top.bonds[bi]) == {j, k}
            keys.append("tb|" + keyf(top.propers[t], "torsion") + "|" + ("mid" if mid else "end:" + keyf(top.bonds[bi], "bond")))
        I = _pair_index(top.torsion_bond)
        I["mask"] = _mask_n(top.rigid_torsion)[I["u"]] if len(top.torsion_bond) else np.zeros((0, 4))
        return I, keys

    def energy(self, G, dev, I, p):
        phi = G["phi"][I["u"]]
        return jnp.sum(I["mask"] * p["K"] * dev["db"][I["v"]][:, None] * (1.0 + jnp.cos(_N[None] * phi[:, None])))


@register
class TorsionAngle(Family):
    name = "torsion_angle"
    params = {"K": ((4,), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        keys = ["ta|" + keyf(top.propers[t], "torsion") + "|" + keyf(top.angles[a], "angle") for t, a in top.torsion_angle]
        I = _pair_index(top.torsion_angle)
        I["mask"] = _mask_n(top.rigid_torsion)[I["u"]] if len(top.torsion_angle) else np.zeros((0, 4))
        return I, keys

    def energy(self, G, dev, I, p):
        phi = G["phi"][I["u"]]
        return jnp.sum(I["mask"] * p["K"] * dev["dc"][I["v"]][:, None] * (1.0 + jnp.cos(_N[None] * phi[:, None])))


@register
class TorsionModulated(Family):
    """F3: sum_n K_n (1 + cos n phi) x (1 + l_mid db_mid + l_end (db_end1 + db_end2) + l_ang (dc_1 + dc_2)):
    the torsion-bond and torsion-angle couplings factorised, 3 coupling parameters per torsion
    type shared by all periodicities."""
    name = "torsion_mod"
    params = {"K": ((4,), 0.0), "l_mid": ((), 0.0), "l_end": ((), 0.0), "l_ang": ((), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        nt = len(top.propers)
        tb = np.asarray(top.torsion_bond).reshape(nt, 3, 2)[:, :, 1] if nt else np.zeros((0, 3), int)
        ta = np.asarray(top.torsion_angle).reshape(nt, 2, 2)[:, :, 1] if nt else np.zeros((0, 2), int)
        return {"i": np.arange(nt), "b": tb, "a": ta, "mask": _mask_n(top.rigid_torsion)}, \
               [keyf(t, "torsion") for t in top.propers]

    def energy(self, G, dev, I, p):
        phi = G["phi"][I["i"]]
        db, dc = dev["db"][I["b"]], dev["dc"][I["a"]]
        amp = 1.0 + p["l_mid"] * db[:, 1] + p["l_end"] * (db[:, 0] + db[:, 2]) + p["l_ang"] * (dc[:, 0] + dc[:, 1])
        return jnp.sum(amp[:, None] * I["mask"] * p["K"] * (1.0 + jnp.cos(_N[None] * phi[:, None])))


@register
class AngleAngleTorsion(Family):
    name = "aat"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        a = np.asarray(top.aat).reshape(-1, 3)
        return {"t": a[:, 0], "a1": a[:, 1], "a2": a[:, 2]}, ["aat|" + keyf(top.propers[t], "torsion") for t in a[:, 0]]

    def energy(self, G, dev, I, p):
        return jnp.sum(p["K"] * dev["dth"][I["a1"]] * dev["dth"][I["a2"]] * jnp.cos(G["phi"][I["t"]]))


def _share_sets(top):
    """Extended coupling sets (F4): bond-angle and angle-angle pairs sharing at least one atom."""
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
    name = "bond_angle_x"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        ba, _ = _share_sets(top)
        return _pair_index(ba), ["bax|" + keyf(top.bonds[b], "bond") + "|" + keyf(top.angles[a], "angle") for b, a in ba]

    def energy(self, G, dev, I, p):
        return jnp.sum(p["K"] * dev["db"][I["u"]] * dev["dc"][I["v"]])


@register
class AngleAngleX(Family):
    name = "angle_angle_x"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        _, aa = _share_sets(top)
        return _pair_index(aa), [_aa_key(top, keyf, m1, m2) for m1, m2 in aa]

    def energy(self, G, dev, I, p):
        return jnp.sum(p["K"] * dev["dc"][I["u"]] * dev["dc"][I["v"]])


# Amber / GAFF functional forms (harmonic bonds and angles, Fourier torsions, impropers); use with
# lj14_scale = 0.5 and typing = "amber" to tune GAFF-like parameters (bonded/amber.py imports them)
