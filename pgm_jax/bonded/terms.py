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
  F12 electronic-structure-inspired:    conj (pi-axis conjugation), volume (signed-volume double well),
     hc_sigma / hc_lone (sigma->sigma*, n->sigma* hyperconjugation), angle_hyb / angle_hybsc
     (Coulson hybrid-orbital angles, fixed / self-consistent), pair13/14_tanh (distance-only),
     pair13/14_ovl (pGM Gaussian-overlap repulsion)
To add a family: a Family with `index(top)` -> (arrays, keys), `params` {name: (shape, init)},
`linear` (names entering the energy linearly) and `energy(G, dev, I, p)`.
"""
from __future__ import annotations

import jax
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


# ------------------------------------------------------------------ F12+: electronic-structure-inspired families
# Built from a few physical quantities (pi-orbital axes, donor/acceptor bond orbitals, hybridisation,
# signed volumes, Gaussian overlaps) instead of springs plus pairwise couplings.

def _unit(v, eps=1e-20):
    """Normalise along the last axis; finite (zero) for a zero vector, with finite gradients."""
    return v * jax.lax.rsqrt(jnp.sum(v * v, -1, keepdims=True) + eps)


def _neighbours(top):
    nb = [[] for _ in range(top.n)]
    for a, b in np.asarray(top.bonds):
        nb[int(a)].append(int(b)); nb[int(b)].append(int(a))
    return [sorted(x) for x in nb]


def _pad3(lst):
    return (list(lst) + [lst[-1]] * 3)[:3]


def pi_axes(R, c, nb3, deg):
    """pi-orbital axis a (unit, sign arbitrary) and its p fraction for centres c with 2 or 3
    neighbours nb3 (padded to 3).  3-coordinated: the normal of the plane through the tips of the
    three bond unit vectors (equal angles with all three bonds; POAV), p = (1 - 3x^2)/(1 - x^2) with
    x = a.u (Coulson orthogonality + s conservation; 1 planar, 3/4 tetrahedral).  2-coordinated: the
    normal of the bond plane, p = 1 (the p-type lone pair / pi orbital)."""
    u = _unit(R[nb3] - R[c][:, None, :])
    a3 = _unit(jnp.cross(u[:, 1] - u[:, 0], u[:, 2] - u[:, 0]))
    x = jnp.sum(a3 * u[:, 0], -1)
    p3 = jnp.clip((1.0 - 3.0 * x * x) / (1.0 - x * x), 0.0, 1.0)
    a2 = _unit(jnp.cross(u[:, 0], u[:, 1]))
    three = (jnp.asarray(deg) == 3)
    return jnp.where(three[:, None], a3, a2), jnp.where(three, p3, 1.0)


@register
class Conjugation(Family):
    """pi conjugation across a bond between two 2- or 3-coordinated centres:
    E = K (1 - (a_i . a_j)^2 p_i p_j).  One term gives the rotation barrier (cos^2 of the angle
    between the pi axes), the planarity of the centres (pyramidalisation lowers p) and their
    coupling (at the barrier the axes are perpendicular and pyramidalisation costs nothing)."""
    name = "conj"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        nb = _neighbours(top)
        rows, keys = [], []
        for i, j in np.asarray(top.bonds):
            i, j = int(i), int(j)
            if len(nb[i]) in (2, 3) and len(nb[j]) in (2, 3):
                rows.append((i, j, _pad3(nb[i]), _pad3(nb[j]), len(nb[i]), len(nb[j])))
                keys.append("conj|" + keyf([i, j], "bond"))
        z = lambda k, sh=(): np.array([r[k] for r in rows], int).reshape((-1,) + sh)
        return {"ci": z(0), "cj": z(1), "ni": z(2, (3,)), "nj": z(3, (3,)), "di": z(4), "dj": z(5)}, keys

    def energy(self, G, dev, I, p):
        if len(I["ci"]) == 0:
            return 0.0
        R = G["R"]
        ai, pi = pi_axes(R, I["ci"], I["ni"], I["di"])
        aj, pj = pi_axes(R, I["cj"], I["nj"], I["dj"])
        return jnp.sum(p["K"] * (1.0 - jnp.sum(ai * aj, -1) ** 2 * pi * pj))


@register
class Volume(Family):
    """Signed volume of a 3-coordinated centre, V = u1 . (u2 x u3) of the bond unit vectors:
    E = A V^2 + B V^4.  A > 0: planar centre; A < 0 < B: pyramidal double well whose barrier (the
    planar inversion state, A^2 / 4B) is fitted, e.g. amine inversion."""
    name = "volume"
    params = {"A": ((), 0.0), "B": ((), 0.0)}
    linear = ("A", "B")

    def index(self, top, keyf):
        nb = _neighbours(top)
        cs = [c for c in range(top.n) if len(nb[c]) == 3]
        return {"c": np.array(cs, int), "nb": np.array([nb[c] for c in cs], int).reshape(-1, 3)}, \
               ["vol|" + keyf([c], "atom") for c in cs]

    def energy(self, G, dev, I, p):
        if len(I["c"]) == 0:
            return 0.0
        R = G["R"]
        u = _unit(R[I["nb"]] - R[I["c"]][:, None, :])
        V = jnp.sum(u[:, 0] * jnp.cross(u[:, 1], u[:, 2]), -1)
        V2 = V * V
        return jnp.sum(p["A"] * V2 + p["B"] * V2 * V2)


@register
class HyperconjSigma(Family):
    """sigma -> sigma* hyperconjugation: every bond has a donor strength D and an acceptor strength A
    (tied by bond type); a torsion i-j-k-l gets E = -(D_ij A_kl + D_kl A_ij) ((1 - cos phi)/2)^2,
    largest antiperiplanar.  Torsion profiles of a new molecule follow from its bond types instead
    of per-torsion Fourier coefficients."""
    name = "hc_sigma"
    params = {"D": ((), 1.0), "A": ((), 0.0)}
    linear = ("A",)

    def index(self, top, keyf):
        bidx = {tuple(sorted(map(int, b))): k for k, b in enumerate(np.asarray(top.bonds))}
        t = np.asarray(top.propers).reshape(-1, 4)
        b1 = np.array([bidx[tuple(sorted((int(i), int(j))))] for i, j, _, _ in t], int)
        b2 = np.array([bidx[tuple(sorted((int(k), int(l))))] for _, _, k, l in t], int)
        return {"i": np.arange(len(top.bonds)), "t": np.arange(len(t)), "b1": b1, "b2": b2}, \
               ["hc|" + keyf(b, "bond") for b in top.bonds]

    def energy(self, G, dev, I, p):
        if len(I["t"]) == 0:
            return 0.0
        w = ((1.0 - jnp.cos(G["phi"][I["t"]])) / 2.0) ** 2
        D, A = p["D"], p["A"]
        return -jnp.sum((D[I["b1"]] * A[I["b2"]] + D[I["b2"]] * A[I["b1"]]) * w)


@register
class HyperconjLone(Family):
    """n -> sigma* (anomeric-type) hyperconjugation: the p-type lone pair of a 2- or 3-coordinated
    N, O or S (axis as for conj) donates into a bond k-l of an sp3 neighbour k:
    E = K (a_j . u_perp)^2, u_perp the k->l direction perpendicular to the j-k bond."""
    name = "hc_lone"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(self, top, keyf):
        nb = _neighbours(top)
        rows, keys = [], []
        for j in range(top.n):
            if top.elements[j] not in ("N", "O", "S") or len(nb[j]) not in (2, 3):
                continue
            for k in nb[j]:
                if len(nb[k]) != 4:
                    continue
                for l in nb[k]:
                    if l == j:
                        continue
                    rows.append((j, k, l, _pad3(nb[j]), len(nb[j])))
                    keys.append("nlp|" + keyf([j], "atom") + "|" + keyf([k, l], "bond"))
        z = lambda q, sh=(): np.array([r[q] for r in rows], int).reshape((-1,) + sh)
        return {"j": z(0), "k": z(1), "l": z(2), "nj": z(3, (3,)), "dj": z(4)}, keys

    def energy(self, G, dev, I, p):
        if len(I["j"]) == 0:
            return 0.0
        R = G["R"]
        a, _ = pi_axes(R, I["j"], I["nj"], I["dj"])
        e = _unit(R[I["k"]] - R[I["j"]])
        v = _unit(R[I["l"]] - R[I["k"]])
        vp = _unit(v - jnp.sum(v * e, -1, keepdims=True) * e)
        return jnp.sum(p["K"] * jnp.sum(a * vp, -1) ** 2)


def _half_bonds(top):
    """Directed bonds c->a (two per bond) and, per angle a-c-b, the indices of c->a and c->b."""
    hb = []
    for i, j in np.asarray(top.bonds):
        hb += [(int(i), int(j)), (int(j), int(i))]
    pos = {h: k for k, h in enumerate(hb)}
    ang = np.asarray(top.angles).reshape(-1, 3)
    return hb, np.array([pos[(int(c), int(a))] for a, c, b in ang], int), np.array([pos[(int(c), int(b))] for a, c, b in ang], int)


def _overlap_sq(lm1, lm2, cth):
    """Squared overlap of two sp^m hybrids on one atom at angle theta (Coulson):
    Delta = (1 + sqrt(m1 m2) cos theta) / sqrt((1 + m1)(1 + m2)), m = exp(lm)."""
    m1, m2 = jnp.exp(lm1), jnp.exp(lm2)
    return (1.0 + jnp.sqrt(m1 * m2) * cth) ** 2 / ((1.0 + m1) * (1.0 + m2))


def _hyb_init(G, top):
    """ln m per directed bond from the reference angles: ln m_a + ln m_b = -2 ln(-cos theta_ab)
    (least squares per molecule; angles <= 90 deg are skipped)."""
    hb, h1, h2 = _half_bonds(top)
    cth = np.asarray(G["cos"])
    ok = cth < -0.05
    M = np.zeros((int(ok.sum()), len(hb)))
    M[np.arange(len(M)), h1[ok]] = 1.0; M[np.arange(len(M)), h2[ok]] += 1.0
    y = -2.0 * np.log(-cth[ok])
    lm = np.linalg.lstsq(M, y, rcond=None)[0] if len(M) else np.zeros(len(hb))
    used = np.zeros(len(hb), bool); used[h1[ok]] = True; used[h2[ok]] = True
    return np.where(used, lm, np.log(3.0))


@register
class AngleHybrid(Family):
    """Bent/Coulson angle term: each directed bond c->a carries a hybridisation index m (sp^m,
    tied by centre and substituent type) and a stiffness k; an angle a-c-b costs
    (k_a + k_b)/2 Delta_ab^2, zero when the two hybrids are orthogonal, cos theta0 = -1/sqrt(m_a m_b).
    Reference angles of a new molecule follow from its substituents (Bent's rule) instead of one
    theta0 per angle type."""
    name = "angle_hyb"
    params = {"lm": ((), None), "k": ((), 500.0)}
    linear = ()

    def index(self, top, keyf):
        hb, h1, h2 = _half_bonds(top)
        return {"i": np.arange(len(hb)), "h1": h1, "h2": h2}, \
               ["hyb|" + keyf([c], "atom") + ">" + keyf([a], "atom") for c, a in hb]

    def init_from_geometry(self, G, I, top):
        return {"lm": _hyb_init(G, top)}

    def energy(self, G, dev, I, p):
        if len(I["h1"]) == 0:
            return 0.0
        d2 = _overlap_sq(p["lm"][I["h1"]], p["lm"][I["h2"]], G["cos"])
        return jnp.sum(0.5 * (jnp.abs(p["k"][I["h1"]]) + jnp.abs(p["k"][I["h2"]])) * d2)


@register
class AngleHybridSC(Family):
    """Self-consistent hybridisation: at each centre the ln m of its bonds relax,
    E(R) = min_z sum_ab k_ab Delta_ab(z, theta)^2 + kappa sum_a (z_a - z0_a)^2 (kappa >= 20 kJ/mol),
    solved by Gauss-Newton steps (gradients by the envelope theorem, as for induced dipoles).
    Rehybridisation couples all angles at a centre (angle-angle terms emerge with physical signs)."""
    name = "angle_hybsc"
    params = {"lm": ((), None), "k": ((), 500.0), "lkap": ((), 5.0)}
    linear = ()
    newton_steps = 12

    def index(self, top, keyf):
        hb, h1, h2 = _half_bonds(top)
        nb = _neighbours(top)
        centres = [c for c in range(top.n) if len(nb[c]) >= 2]
        pos = {h: k for k, h in enumerate(hb)}
        slot = np.full((len(centres), 4), -1, int)            # directed-bond index per centre slot
        for r, c in enumerate(centres):
            for s_, a in enumerate(nb[c][:4]):
                slot[r, s_] = pos[(c, a)]
        pairs = []                                               # (centre row, slot a, slot b, angle index)
        ang = np.asarray(top.angles).reshape(-1, 3)
        crow = {c: r for r, c in enumerate(centres)}
        for m_, (a, c, b) in enumerate(ang):
            r = crow[int(c)]
            pairs.append((r, nb[int(c)].index(int(a)), nb[int(c)].index(int(b)), m_))
        pr = np.array(pairs, int).reshape(-1, 4)
        # per-centre stiffness of the relaxation kappa is tied by centre type: stored on each directed
        # bond of the centre (one key per directed bond, the centre's value is the mean)
        return {"i": np.arange(len(hb)), "slot": slot, "pr": pr, "h1": h1, "h2": h2}, \
               ["hybsc|" + keyf([c], "atom") + ">" + keyf([a], "atom") for c, a in hb]

    def init_from_geometry(self, G, I, top):
        return {"lm": _hyb_init(G, top)}

    def energy(self, G, dev, I, p):
        if len(I["pr"]) == 0:
            return 0.0
        slot, pr = I["slot"], I["pr"]
        mask = slot >= 0
        sl = np.maximum(slot, 0)
        z0 = jnp.where(mask, p["lm"][sl], 0.0)
        kap = 20.0 + jnp.exp(jnp.sum(jnp.where(mask, p["lkap"][sl], 0.0), 1) / jnp.maximum(mask.sum(1), 1))
        kk = 0.5 * (jnp.abs(p["k"][I["h1"]]) + jnp.abs(p["k"][I["h2"]]))      # stiffness >= 0
        cth = G["cos"][pr[:, 3]]
        rows, sa, sb = pr[:, 0], pr[:, 1], pr[:, 2]
        fm = jnp.asarray(mask, float)

        def resid(zf):                       # E = |r|^2: Gauss-Newton normal matrix J^T J >= kappa I
            z = zf.reshape(z0.shape)
            m1, m2 = jnp.exp(z[rows, sa]), jnp.exp(z[rows, sb])
            d = (1.0 + jnp.sqrt(m1 * m2) * cth) / jnp.sqrt((1.0 + m1) * (1.0 + m2))
            return jnp.concatenate([jnp.sqrt(kk) * d, (jnp.sqrt(kap)[:, None] * fm * (z - z0)).reshape(-1)])

        jac = jax.jacfwd(resid)
        n = z0.size
        pad = jnp.diag(1.0 - fm.reshape(-1))

        def step(zf, _):
            zs = jax.lax.stop_gradient(zf)
            r, J = resid(zs), jac(zs)
            dz = jnp.linalg.solve(J.T @ J + pad + 1e-9 * jnp.eye(n), J.T @ r) * fm.reshape(-1)
            return zs - jnp.clip(dz, -0.5, 0.5), None

        zstar, _ = jax.lax.scan(step, jax.lax.stop_gradient(z0).reshape(-1), None, length=self.newton_steps)
        r = resid(jax.lax.stop_gradient(zstar))           # envelope theorem: no derivative through z*
        return jnp.sum(r * r)


class _TanhPair(Family):
    """Distance-only short-range term for topological pairs: E = sum_n C_n s^n, s = tanh((r - r0)/w),
    n = 1..4 (bounded at any distance)."""
    which = "13"
    width = 0.05
    params = {"C": ((4,), 0.0), "r0": ((), None)}
    linear = ("C",)

    def index(self, top, keyf):
        pairs = getattr(top, "pairs" + self.which)
        return {"i": np.arange(len(pairs))}, [keyf(pp, "pair") + "|t" + self.which for pp in pairs]

    def energy(self, G, dev, I, p):
        if len(I["i"]) == 0:
            return 0.0
        s = jnp.tanh((G["r" + self.which][I["i"]] - p["r0"]) / self.width)
        return jnp.sum(p["C"] * jnp.stack([s, s * s, s ** 3, s ** 4], -1))


@register
class Pair13Tanh(_TanhPair):
    name = "pair13_tanh"
    which = "13"
    width = 0.03


@register
class Pair14Tanh(_TanhPair):
    name = "pair14_tanh"
    which = "14"
    width = 0.06


class _OverlapPair(Family):
    """Exchange-type repulsion from the overlap of the pGM Gaussian densities of a topological pair:
    E = A exp(-(b_ij r)^2), b_ij = 1/sqrt(2 (R_i^2 + R_j^2)) with the pGM radii (the same widths as the
    electrostatics); one amplitude per pair type."""
    which = "13"
    needs_radius = True
    params = {"A": ((), 0.0)}
    linear = ("A",)

    def index(self, top, keyf):
        pairs = getattr(top, "pairs" + self.which)
        return {"i": np.arange(len(pairs))}, [keyf(pp, "pair") + "|o" + self.which for pp in pairs]

    def energy(self, G, dev, I, p):
        if len(I["i"]) == 0:
            return 0.0
        return jnp.sum(p["A"] * jnp.exp(-(I["bij"] * G["r" + self.which][I["i"]]) ** 2))


@register
class Pair13Ovl(_OverlapPair):
    name = "pair13_ovl"
    which = "13"


@register
class Pair14Ovl(_OverlapPair):
    name = "pair14_ovl"
    which = "14"
