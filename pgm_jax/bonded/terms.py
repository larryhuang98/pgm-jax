"""Bonded term families: a registry of energy functions of internal coordinates, each with its
index set, parameters and tying keys.  All energies kJ/mol, lengths nm, angles rad.

Every family reads a geometry dict G of one frame (bond lengths, angle cosines and angles,
dihedrals, improper dihedrals, topological pair distances) and the deviations from the shared
reference values (db = b - b0, dc = cos - cos th0, dth = th - th0), so couplings and diagonal
terms use one set of reference values, fitted jointly.

Families (F-numbers of the plan):
  F1 class II (Abdullah et al. 2025):  bond_morse, angle_cos, bond_bond, bond_angle,
     angle_angle, torsion, torsion_bond, torsion_angle, aat, improper
  F2 topological pair potentials:      pair13_harm (Urey-Bradley), pair13_exp, pair14_exp
  F4 extended quadratic couplings:     bond_angle_x, angle_angle_x (all pairs sharing an atom),
     angle_cubic, bond_harm
  F7 out-of-plane alternatives:        pyramid (360 deg - sum of the three angles), improper
  F8 torsion x out-of-plane coupling:  torsion_oop
  F9 twist of 3-coordinated centres:   twist (Winkler-Dunitz twist angle, lone-pair aware)
To add a family: a Family with `index(top)` -> (arrays, keys), `params` {name: (shape, init)},
`linear` (names entering the energy linearly) and `energy(G, dev, I, p)`.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

# approximate average bond energies (kJ/mol) by element pair and bond order (Morse depths; the
# paper takes experimental dissociation energies; only the anharmonicity depends on them)
_DE = {("C", "H", 1): 413, ("C", "C", 1): 348, ("C", "C", 2): 614, ("C", "C", 1.5): 518,
       ("C", "N", 1): 305, ("C", "N", 2): 615, ("C", "N", 1.5): 540, ("C", "O", 1): 358,
       ("C", "O", 2): 745, ("C", "O", 1.5): 550, ("H", "O", 1): 463, ("H", "N", 1): 391,
       ("H", "S", 1): 363, ("C", "S", 1): 272, ("C", "F", 1): 485, ("C", "Cl", 1): 328,
       ("O", "P", 1): 335, ("O", "P", 2): 544}


def morse_depth(e1, e2, order):
    k = tuple(sorted((e1, e2))) + (float(order) if order in (1.5,) else int(round(order)),)
    if k in _DE:
        return float(_DE[k])
    k1 = tuple(sorted((e1, e2))) + (1,)
    return float(_DE.get(k1, 400.0))


# ------------------------------------------------------------------ geometry
def _dihedral(x0, x1, x2, x3):
    b0, b1, b2 = x1 - x0, x2 - x1, x3 - x2
    n1, n2 = jnp.cross(b0, b1), jnp.cross(b1, b2)
    m1 = jnp.cross(n1, b1 / jnp.linalg.norm(b1, axis=-1, keepdims=True))
    return jnp.arctan2(jnp.sum(m1 * n2, -1), jnp.sum(n1 * n2, -1))


def geometry(R, top):
    """Internal coordinates of one frame R (n, 3) nm."""
    G = {}
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
        G["imp"] = jnp.stack([_dihedral(R[a0], R[b0], R[c0], R[d0]), _dihedral(R[b0], R[d0], R[c0], R[a0]),
                              _dihedral(R[d0], R[a0], R[c0], R[b0])], -1)
        th = []
        for p, q in ((a0, b0), (b0, d0), (a0, d0)):
            u, v = R[p] - R[c0], R[q] - R[c0]
            th.append(jnp.arccos(jnp.clip(jnp.sum(u * v, -1) / (jnp.linalg.norm(u, axis=-1) * jnp.linalg.norm(v, axis=-1)), -1 + 1e-12, 1 - 1e-12)))
        G["pyr"] = 2 * jnp.pi - (th[0] + th[1] + th[2])
    for name in ("pairs13", "pairs14", "pairs15"):
        p = getattr(top, name)
        G["r" + name[-2:]] = jnp.linalg.norm(R[p[:, 0]] - R[p[:, 1]], axis=-1) if len(p) else jnp.zeros(0)
    return G


# ------------------------------------------------------------------ families
class Family:
    name = ""
    params: dict = {}        # name -> (per-key shape, default init)
    linear: tuple = ()
    needs: tuple = ()        # families whose reference values it uses ("ref" always available)

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
class AngleCos(Family):
    name = "angle_cos"
    params = {"Ka": ((), 400.0)}                # kJ/mol
    linear = ("Ka",)

    def index(self, top, keyf):
        return {"i": np.arange(len(top.angles))}, [keyf(a, "angle") for a in top.angles]

    def energy(self, G, dev, I, p):
        return jnp.sum(p["Ka"] * dev["dc"][I["i"]] ** 2)


@register
class AngleCubic(Family):
    name = "angle_cubic"
    params = {"Ka3": ((), 0.0)}
    linear = ("Ka3",)

    def index(self, top, keyf):
        return {"i": np.arange(len(top.angles))}, [keyf(a, "angle") for a in top.angles]

    def energy(self, G, dev, I, p):
        return jnp.sum(p["Ka3"] * dev["dc"][I["i"]] ** 3)


def _pair_index(pairs):
    return {"u": np.asarray(pairs)[:, 0], "v": np.asarray(pairs)[:, 1]} if len(pairs) else {"u": np.zeros(0, int), "v": np.zeros(0, int)}


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


@register
class TorsionOOP(Family):
    """F8: torsion x out-of-plane coupling, K cos(2 phi) sum sin^2(improper) for every proper torsion
    whose central atom is a planar (improper) centre.  Lets the out-of-plane stiffness of a
    conjugated centre (amide N, carbonyl C) fall as the torsion leaves planarity (resonance lost,
    centre pyramidalises), which a separable torsion + improper cannot do."""
    name = "torsion_oop"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        centre = {int(m[0]): i for i, m in enumerate(top.impropers)}
        t_, m_, keys = [], [], []
        for t, (i, j, k, l) in enumerate(top.propers):
            for c in (int(j), int(k)):
                if c in centre:
                    t_.append(t); m_.append(centre[c])
                    keys.append("toop|" + keyf(top.propers[t], "torsion") + "|" + keyf([c], "atom"))
        return {"t": np.array(t_, int), "m": np.array(m_, int)}, keys

    def energy(self, G, dev, I, p):
        if "imp" not in G or len(I["t"]) == 0:
            return 0.0
        s = jnp.sum(jnp.sin(G["imp"][I["m"]]) ** 2, -1)
        return jnp.sum(p["K"] * jnp.cos(2.0 * G["phi"][I["t"]]) * s)


def _twist_pairs(top):
    """Torsion pairs (t1, t2) that share three atoms i-j-k and differ in the last one, where the
    shared end atom is 3-coordinated: its two other substituents l1, l2 define one twist angle."""
    nb = {}
    for a, b in np.asarray(top.bonds):
        nb.setdefault(int(a), set()).add(int(b)); nb.setdefault(int(b), set()).add(int(a))
    groups = {}
    for t, (i, j, k, l) in enumerate(np.asarray(top.propers).tolist()):
        if len(nb[k]) == 3:
            groups.setdefault((i, j, k), []).append((l, t))
        if len(nb[j]) == 3:
            groups.setdefault((l, k, j), []).append((i, t))
    out = []
    for (i, j, k), v in groups.items():
        if len(v) == 2:
            (l1, t1), (l2, t2) = sorted(v)
            out.append((t1, t2, i, j, k, l1, l2))
    return out


@register
class Twist(Family):
    """F9: twist of a 3-coordinated centre about a bond (Winkler-Dunitz), tau = arg(e^{i phi1} -
    e^{i phi2}) from the two dihedrals i-j-k-l1, i-j-k-l2; sum_n K_n (1 + cos n tau).  For a planar
    centre tau = phi1; for a pyramidalised one (amide N at the rotation barrier) tau follows the
    lone pair / pi orbital, not the individual substituents, so the resonance barrier is not
    relieved by pyramidalisation as it is with separable dihedral terms.  Odd n only when the two
    substituents differ in type."""
    name = "twist"
    params = {"K": ((4,), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        tp = _twist_pairs(top)
        keys, mask = [], []
        for t1, t2, i, j, k, l1, l2 in tp:
            c1, c2 = keyf([l1], "atom"), keyf([l2], "atom")
            keys.append("tw|" + "-".join(keyf([x], "atom") for x in (i, j, k)) + "|" + "-".join(sorted([c1, c2])))
            mask.append([1, 1, 1, 1] if c1 != c2 else [0, 1, 0, 1])
        swap = np.array([keyf([l1], "atom") > keyf([l2], "atom") for *_, l1, l2 in tp], bool)
        t1 = np.array([x[0] for x in tp], int); t2 = np.array([x[1] for x in tp], int)
        a, b = np.where(swap, t2, t1), np.where(swap, t1, t2)
        return {"u": a, "v": b, "mask": np.array(mask, float).reshape(-1, 4)}, keys

    def energy(self, G, dev, I, p):
        if len(I["u"]) == 0:
            return 0.0
        p1, p2 = G["phi"][I["u"]], G["phi"][I["v"]]
        tau = jnp.arctan2(jnp.sin(p1) - jnp.sin(p2), jnp.cos(p1) - jnp.cos(p2))
        return jnp.sum(I["mask"] * p["K"] * (1.0 + jnp.cos(_N[None] * tau[:, None])))


@register
class Pyramid(Family):
    name = "pyramid"
    params = {"K": ((), 20.0)}
    linear = ("K",)

    def index(self, top, keyf):
        return {"i": np.arange(len(top.impropers))}, [keyf(m, "improper") for m in top.impropers]

    def energy(self, G, dev, I, p):
        if "pyr" not in G:
            return 0.0
        return jnp.sum(p["K"] * G["pyr"][I["i"]] ** 2)


class _PairFamily(Family):
    which = "13"

    def index(self, top, keyf):
        pairs = getattr(top, "pairs" + self.which)
        return {"i": np.arange(len(pairs))}, [keyf(pp, "pair") + "|" + self.which for pp in pairs]


@register
class Pair13Harm(_PairFamily):
    name = "pair13_harm"
    which = "13"
    params = {"K": ((), 0.0), "r0": ((), None)}         # r0 from the reference geometry
    linear = ("K",)

    def energy(self, G, dev, I, p):
        return jnp.sum(0.5 * p["K"] * (G["r13"][I["i"]] - p["r0"]) ** 2)


@register
class Pair13Exp(_PairFamily):
    name = "pair13_exp"
    which = "13"
    params = {"A": ((), 0.0), "B": ((), 30.0), "r0": ((), None)}
    linear = ("A",)

    def energy(self, G, dev, I, p):
        return jnp.sum(p["A"] * jnp.exp(-p["B"] * (G["r13"][I["i"]] - p["r0"])))


@register
class Pair14Exp(_PairFamily):
    name = "pair14_exp"
    which = "14"
    params = {"A": ((), 0.0), "B": ((), 30.0), "r0": ((), None)}
    linear = ("A",)

    def energy(self, G, dev, I, p):
        return jnp.sum(p["A"] * jnp.exp(-p["B"] * (G["r14"][I["i"]] - p["r0"])))


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


PAPER = ("bond_morse", "angle_cos", "bond_bond", "bond_angle", "angle_angle", "torsion", "torsion_bond",
         "torsion_angle", "aat", "improper")
