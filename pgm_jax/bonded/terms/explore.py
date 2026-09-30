"""Define the families explored in the bonded study beyond the class II set.

Registered families: out-of-plane alternative `pyramid` (F7); torsion x out-of-plane coupling
`torsion_oop` (F8); twist of 3-coordinated centres `twist` (F9); topological pair potentials
`pair13_harm` (Urey-Bradley), `pair13_exp`, `pair14_exp` (F2); and the electronic-structure-
inspired families (F12+): pi conjugation `conj`, signed-volume double well `volume`,
hyperconjugation `hc_sigma` (sigma -> sigma*) and `hc_lone` (n -> sigma*), Coulson hybrid-orbital
angles `angle_hyb` (fixed) and `angle_hybsc` (self-consistent), distance-only pair terms
`pair13_tanh` / `pair14_tanh`, and pGM Gaussian-overlap repulsion `pair13_ovl` / `pair14_ovl`.
The F-numbers are those of the study plan (reports/bonded/README.md has the findings).  The
interface is described in terms/core.py (`Family`).

Units: energies kJ/mol, lengths nm, angles rad.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .core import _N, Family, register

if TYPE_CHECKING:
    from ..topology import Topology


@register
class TorsionOOP(Family):
    """F8: torsion x out-of-plane coupling, E = K cos(2 phi) sum sin^2(improper), K [kJ/mol].

    One instance per proper torsion and planar (improper) centre among its two central atoms.  Lets
    the out-of-plane stiffness of a conjugated centre (amide N, carbonyl C) fall as the torsion
    leaves planarity (resonance lost, centre pyramidalises), which a separable torsion + improper
    cannot do.
    """

    name = "torsion_oop"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return (torsion "t", improper centre "m") pairs, keys "toop|torsion|centre"."""
        centre = {int(m[0]): i for i, m in enumerate(top.impropers)}
        t_, m_, keys = [], [], []
        for t, (_i, j, k, _l) in enumerate(top.propers):
            for c in (int(j), int(k)):
                if c in centre:
                    t_.append(t)
                    m_.append(centre[c])
                    keys.append("toop|" + keyf(top.propers[t], "torsion") + "|" + keyf([c], "atom"))
        return {"t": np.array(t_, int), "m": np.array(m_, int)}, keys

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K cos(2 phi_t) sum_3 sin^2(imp_m) [kJ/mol] (0.0 without instances)."""
        if "imp" not in G or len(I["t"]) == 0:
            return 0.0
        s = jnp.sum(jnp.sin(G["imp"][I["m"]]) ** 2, -1)
        return jnp.sum(p["K"] * jnp.cos(2.0 * G["phi"][I["t"]]) * s)


def _twist_pairs(top: Topology) -> list[tuple[int, int, int, int, int, int, int]]:
    """Return the torsion pairs that define twist angles.

    Torsion pairs (t1, t2) share three atoms i-j-k and differ in the last one, where the shared end
    atom k is 3-coordinated: its two other substituents l1 < l2 define one twist angle.  Each torsion
    is considered in both directions.

    Returns
    -------
    list of tuple
        (t1, t2, i, j, k, l1, l2) per twist.
    """
    nb = {}
    for a, b in np.asarray(top.bonds):
        nb.setdefault(int(a), set()).add(int(b))
        nb.setdefault(int(b), set()).add(int(a))
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
    """F9: twist of a 3-coordinated centre about a bond (Winkler-Dunitz), E = sum_n K_n (1 + cos n tau).

    tau = arg(e^{i phi1} - e^{i phi2}) from the two dihedrals i-j-k-l1, i-j-k-l2; K [kJ/mol] shape
    (4,).  For a planar centre (phi2 = phi1 + pi) tau = phi1; for a pyramidalised one (amide N at
    the rotation barrier) tau follows the lone pair / pi orbital, not the individual substituents,
    so the resonance barrier is not relieved by pyramidalisation as it is with separable dihedral
    terms.  Odd n only when the two substituents differ in type.
    """

    name = "twist"
    params = {"K": ((4,), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the torsion pairs ("u", "v", ordered by substituent key) and the multiplicity mask of every twist."""
        tp = _twist_pairs(top)
        keys, mask = [], []
        for _t1, _t2, i, j, k, l1, l2 in tp:
            c1, c2 = keyf([l1], "atom"), keyf([l2], "atom")
            keys.append("tw|" + "-".join(keyf([x], "atom") for x in (i, j, k)) + "|" + "-".join(sorted([c1, c2])))
            mask.append([1, 1, 1, 1] if c1 != c2 else [0, 1, 0, 1])
        swap = np.array([keyf([l1], "atom") > keyf([l2], "atom") for *_, l1, l2 in tp], bool)
        t1 = np.array([x[0] for x in tp], int)
        t2 = np.array([x[1] for x in tp], int)
        a, b = np.where(swap, t2, t1), np.where(swap, t1, t2)
        return {"u": a, "v": b, "mask": np.array(mask, float).reshape(-1, 4)}, keys

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum mask_n K_n (1 + cos n tau) [kJ/mol] (0.0 without twists)."""
        if len(I["u"]) == 0:
            return 0.0
        p1, p2 = G["phi"][I["u"]], G["phi"][I["v"]]
        tau = jnp.arctan2(jnp.sin(p1) - jnp.sin(p2), jnp.cos(p1) - jnp.cos(p2))
        return jnp.sum(I["mask"] * p["K"] * (1.0 + jnp.cos(_N[None] * tau[:, None])))


@register
class Pyramid(Family):
    """F7: pyramidalisation of planar centres, E = K pyr^2, K [kJ/mol/rad^2].

    pyr = 2 pi - (sum of the three angles at the centre) (`geometry` "pyr").
    """

    name = "pyramid"
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
        """Return sum K pyr^2 [kJ/mol] (0.0 without impropers)."""
        if "pyr" not in G:
            return 0.0
        return jnp.sum(p["K"] * G["pyr"][I["i"]] ** 2)


class _PairFamily(Family):
    """Base of the topological pair families: one instance per 1-3 or 1-4 pair (`which` = "13" or "14")."""

    which = "13"

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per pair of Topology.pairs13 / pairs14 (keys "pair key|13" or "|14")."""
        pairs = getattr(top, "pairs" + self.which)
        return {"i": np.arange(len(pairs))}, [keyf(pp, "pair") + "|" + self.which for pp in pairs]


@register
class Pair13Harm(_PairFamily):
    """F2: Urey-Bradley 1-3 spring, E = 0.5 K (r13 - r0)^2.

    K [kJ/mol/nm^2]; r0 [nm] starts from the reference geometry.
    """

    name = "pair13_harm"
    which = "13"
    params = {"K": ((), 0.0), "r0": ((), None)}  # r0 from the reference geometry
    linear = ("K",)

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum 0.5 K (r13 - r0)^2 [kJ/mol]."""
        return jnp.sum(0.5 * p["K"] * (G["r13"][I["i"]] - p["r0"]) ** 2)


@register
class Pair13Exp(_PairFamily):
    """F2: exponential 1-3 pair term, E = A exp(-B (r13 - r0)).

    A [kJ/mol], B [1/nm]; r0 [nm] starts from the reference geometry.
    """

    name = "pair13_exp"
    which = "13"
    params = {"A": ((), 0.0), "B": ((), 30.0), "r0": ((), None)}
    linear = ("A",)

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum A exp(-B (r13 - r0)) [kJ/mol]."""
        return jnp.sum(p["A"] * jnp.exp(-p["B"] * (G["r13"][I["i"]] - p["r0"])))


@register
class Pair14Exp(_PairFamily):
    """F2: exponential 1-4 pair term, E = A exp(-B (r14 - r0)).

    A [kJ/mol], B [1/nm]; r0 [nm] starts from the reference geometry.
    """

    name = "pair14_exp"
    which = "14"
    params = {"A": ((), 0.0), "B": ((), 30.0), "r0": ((), None)}
    linear = ("A",)

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum A exp(-B (r14 - r0)) [kJ/mol]."""
        return jnp.sum(p["A"] * jnp.exp(-p["B"] * (G["r14"][I["i"]] - p["r0"])))


# ------------------------------------------------------------------ F12+: electronic-structure-inspired families
# Built from a few physical quantities (pi-orbital axes, donor/acceptor bond orbitals, hybridisation,
# signed volumes, Gaussian overlaps) instead of springs plus pairwise couplings.


def _unit(v: jax.Array, eps: float = 1e-20) -> jax.Array:
    """Return `v` normalised along the last axis; finite (zero) for a zero vector, with finite gradients."""
    return v * jax.lax.rsqrt(jnp.sum(v * v, -1, keepdims=True) + eps)


def _neighbours(top: Topology) -> list[list[int]]:
    """Return the sorted neighbour list of every atom of `top`."""
    nb = [[] for _ in range(top.n)]
    for a, b in np.asarray(top.bonds):
        nb[int(a)].append(int(b))
        nb[int(b)].append(int(a))
    return [sorted(x) for x in nb]


def _pad3(lst: Sequence[int]) -> list[int]:
    """Return the first three entries of `lst`, padded by repeating its last entry (for 2-coordinated centres)."""
    return (list(lst) + [lst[-1]] * 3)[:3]


def pi_axes(R: jax.Array, c: ArrayLike, nb3: ArrayLike, deg: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Return the pi-orbital axes (unit, sign arbitrary) and their p fractions for centres with 2 or 3 neighbours.

    Parameters
    ----------
    R : jax.Array (n, 3)
        Positions [nm].
    c : ArrayLike (k,)
        Centre atoms.
    nb3 : ArrayLike (k, 3)
        Their neighbours, padded to 3 (`_pad3`).
    deg : ArrayLike (k,)
        Number of neighbours (2 or 3).

    Returns
    -------
    a : jax.Array (k, 3)
        Axes.
    p : jax.Array (k,)
        p fractions in [0, 1].

    Notes
    -----
    3-coordinated: the normal of the plane through the tips of the three bond unit vectors (equal
    angles with all three bonds; POAV), p = (1 - 3x^2)/(1 - x^2) with x = a.u (Coulson orthogonality
    + s conservation; 1 planar, 3/4 tetrahedral).  2-coordinated: the normal of the bond plane,
    p = 1 (the p-type lone pair / pi orbital).
    """
    u = _unit(R[nb3] - R[c][:, None, :])
    a3 = _unit(jnp.cross(u[:, 1] - u[:, 0], u[:, 2] - u[:, 0]))
    x = jnp.sum(a3 * u[:, 0], -1)
    p3 = jnp.clip((1.0 - 3.0 * x * x) / (1.0 - x * x), 0.0, 1.0)  # p fraction of the pi hybrid
    a2 = _unit(jnp.cross(u[:, 0], u[:, 1]))
    three = jnp.asarray(deg) == 3
    return jnp.where(three[:, None], a3, a2), jnp.where(three, p3, 1.0)


@register
class Conjugation(Family):
    """pi conjugation across a bond between two 2- or 3-coordinated centres, E = K (1 - (a_i . a_j)^2 p_i p_j).

    K [kJ/mol]; a, p from `pi_axes`.  One term gives the rotation barrier (cos^2 of the angle between
    the pi axes), the planarity of the centres (pyramidalisation lowers p) and their coupling (at
    the barrier the axes are perpendicular and pyramidalisation costs nothing).
    """

    name = "conj"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return, per bond between 2- or 3-coordinated centres, the centres, their padded neighbours and degrees."""
        nb = _neighbours(top)
        rows, keys = [], []
        for i, j in np.asarray(top.bonds):
            i, j = int(i), int(j)
            if len(nb[i]) in (2, 3) and len(nb[j]) in (2, 3):
                rows.append((i, j, _pad3(nb[i]), _pad3(nb[j]), len(nb[i]), len(nb[j])))
                keys.append("conj|" + keyf([i, j], "bond"))

        def z(k: int, sh: tuple = ()) -> np.ndarray:
            """Return column k of the rows as an int array, reshaped to (-1, *sh)."""
            return np.array([r[k] for r in rows], int).reshape((-1,) + sh)

        return {"ci": z(0), "cj": z(1), "ni": z(2, (3,)), "nj": z(3, (3,)), "di": z(4), "dj": z(5)}, keys

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K (1 - (a_i . a_j)^2 p_i p_j) [kJ/mol] (0.0 without instances)."""
        if len(I["ci"]) == 0:
            return 0.0
        R = G["R"]
        ai, pi = pi_axes(R, I["ci"], I["ni"], I["di"])
        aj, pj = pi_axes(R, I["cj"], I["nj"], I["dj"])
        return jnp.sum(p["K"] * (1.0 - jnp.sum(ai * aj, -1) ** 2 * pi * pj))


@register
class Volume(Family):
    """Signed volume of a 3-coordinated centre, E = A V^2 + B V^4 with V = u1 . (u2 x u3).

    u are the bond unit vectors (V dimensionless); A, B [kJ/mol].  A > 0: planar centre; A < 0 < B:
    pyramidal double well whose barrier (the planar inversion state, A^2 / 4B) is fitted, e.g.
    amine inversion.
    """

    name = "volume"
    params = {"A": ((), 0.0), "B": ((), 0.0)}
    linear = ("A", "B")

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return every 3-coordinated centre "c" with its neighbours "nb" (keys "vol|atom")."""
        nb = _neighbours(top)
        cs = [c for c in range(top.n) if len(nb[c]) == 3]
        return {"c": np.array(cs, int), "nb": np.array([nb[c] for c in cs], int).reshape(-1, 3)}, [
            "vol|" + keyf([c], "atom") for c in cs
        ]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum A V^2 + B V^4 [kJ/mol] (0.0 without centres)."""
        if len(I["c"]) == 0:
            return 0.0
        R = G["R"]
        u = _unit(R[I["nb"]] - R[I["c"]][:, None, :])
        V = jnp.sum(u[:, 0] * jnp.cross(u[:, 1], u[:, 2]), -1)
        V2 = V * V
        return jnp.sum(p["A"] * V2 + p["B"] * V2 * V2)


@register
class HyperconjSigma(Family):
    """sigma -> sigma* hyperconjugation between the outer bonds of every torsion.

    Every bond has a donor strength D (dimensionless) and an acceptor strength A [kJ/mol] (tied by
    bond type); a torsion i-j-k-l gets E = -(D_ij A_kl + D_kl A_ij) ((1 - cos phi)/2)^2, largest
    antiperiplanar.  Torsion profiles of a new molecule follow from its bond types instead of
    per-torsion Fourier coefficients.  The instances (and keys) are the bonds; the torsions index
    them through "b1", "b2".
    """

    name = "hc_sigma"
    params = {"D": ((), 1.0), "A": ((), 0.0)}
    linear = ("A",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the bonds "i" (instances, keys "hc|bond"), the torsions "t" and their outer bonds "b1", "b2"."""
        bidx = {tuple(sorted(map(int, b))): k for k, b in enumerate(np.asarray(top.bonds))}
        t = np.asarray(top.propers).reshape(-1, 4)
        b1 = np.array([bidx[tuple(sorted((int(i), int(j))))] for i, j, _, _ in t], int)
        b2 = np.array([bidx[tuple(sorted((int(k), int(l))))] for _, _, k, l in t], int)
        return {"i": np.arange(len(top.bonds)), "t": np.arange(len(t)), "b1": b1, "b2": b2}, [
            "hc|" + keyf(b, "bond") for b in top.bonds
        ]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return -sum (D_b1 A_b2 + D_b2 A_b1) ((1 - cos phi)/2)^2 [kJ/mol] (0.0 without torsions)."""
        if len(I["t"]) == 0:
            return 0.0
        w = ((1.0 - jnp.cos(G["phi"][I["t"]])) / 2.0) ** 2
        D, A = p["D"], p["A"]
        return -jnp.sum((D[I["b1"]] * A[I["b2"]] + D[I["b2"]] * A[I["b1"]]) * w)


@register
class HyperconjLone(Family):
    """n -> sigma* (anomeric-type) hyperconjugation, E = K (a_j . u_perp)^2, K [kJ/mol].

    The p-type lone pair of a 2- or 3-coordinated N, O or S j (axis as for `Conjugation`) donates
    into a bond k-l of an sp3 (4-coordinated) neighbour k; u_perp is the k->l direction
    perpendicular to the j-k bond.
    """

    name = "hc_lone"
    params = {"K": ((), 0.0)}
    linear = ("K",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the (donor j, neighbour k, atom l) triples with j's padded neighbours and degree."""
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

        def z(q: int, sh: tuple = ()) -> np.ndarray:
            """Return column q of the rows as an int array, reshaped to (-1, *sh)."""
            return np.array([r[q] for r in rows], int).reshape((-1,) + sh)

        return {"j": z(0), "k": z(1), "l": z(2), "nj": z(3, (3,)), "dj": z(4)}, keys

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum K (a_j . u_perp)^2 [kJ/mol] (0.0 without instances)."""
        if len(I["j"]) == 0:
            return 0.0
        R = G["R"]
        a, _ = pi_axes(R, I["j"], I["nj"], I["dj"])
        e = _unit(R[I["k"]] - R[I["j"]])
        v = _unit(R[I["l"]] - R[I["k"]])
        vp = _unit(v - jnp.sum(v * e, -1, keepdims=True) * e)
        return jnp.sum(p["K"] * jnp.sum(a * vp, -1) ** 2)


def _half_bonds(top: Topology) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Return the directed bonds c->a (two per bond) and, per angle a-c-b, the indices of c->a and c->b.

    Returns
    -------
    hb : list of (int, int)
        Directed bonds (from, to).
    h1, h2 : np.ndarray (na,) int
        Directed-bond indices of the two arms of every angle.
    """
    hb = []
    for i, j in np.asarray(top.bonds):
        hb += [(int(i), int(j)), (int(j), int(i))]
    pos = {h: k for k, h in enumerate(hb)}
    ang = np.asarray(top.angles).reshape(-1, 3)
    return (
        hb,
        np.array([pos[(int(c), int(a))] for a, c, b in ang], int),
        np.array([pos[(int(c), int(b))] for a, c, b in ang], int),
    )


def _overlap_sq(lm1: jax.Array, lm2: jax.Array, cth: jax.Array) -> jax.Array:
    """Return the squared overlap of two sp^m hybrids on one atom at angle theta (Coulson).

        Delta^2 = (1 + sqrt(m1 m2) cos theta)^2 / ((1 + m1)(1 + m2)),  m = exp(lm)

    `lm1`, `lm2` are ln m of the two hybrids, `cth` = cos theta.
    """
    m1, m2 = jnp.exp(lm1), jnp.exp(lm2)
    return (1.0 + jnp.sqrt(m1 * m2) * cth) ** 2 / ((1.0 + m1) * (1.0 + m2))


def _hyb_init(G: dict[str, np.ndarray], top: Topology) -> np.ndarray:
    """Return ln m per directed bond from the reference angles (least squares per molecule).

    Orthogonal hybrids: ln m_a + ln m_b = -2 ln(-cos theta_ab).  Angles with cos theta >= -0.05 are
    skipped; directed bonds in no used angle get ln 3 (sp3).
    """
    hb, h1, h2 = _half_bonds(top)
    cth = np.asarray(G["cos"])
    ok = cth < -0.05  # angles above ~93 deg only: needs cos theta < 0 for the log
    M = np.zeros((int(ok.sum()), len(hb)))
    M[np.arange(len(M)), h1[ok]] = 1.0
    M[np.arange(len(M)), h2[ok]] += 1.0
    y = -2.0 * np.log(-cth[ok])
    lm = np.linalg.lstsq(M, y, rcond=None)[0] if len(M) else np.zeros(len(hb))
    used = np.zeros(len(hb), bool)
    used[h1[ok]] = True
    used[h2[ok]] = True
    return np.where(used, lm, np.log(3.0))


@register
class AngleHybrid(Family):
    """Bent/Coulson angle term from hybridisation indices of the directed bonds.

    Each directed bond c->a carries a hybridisation index m (sp^m, tied by centre and substituent
    type; parameter lm = ln m, initialised by `_hyb_init`) and a stiffness k [kJ/mol]; an angle a-c-b
    costs (|k_a| + |k_b|)/2 Delta_ab^2, zero when the two hybrids are orthogonal,
    cos theta0 = -1/sqrt(m_a m_b).  Reference angles of a new molecule follow from its substituents
    (Bent's rule) instead of one theta0 per angle type.
    """

    name = "angle_hyb"
    params = {"lm": ((), None), "k": ((), 500.0)}
    linear = ()

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the directed bonds "i" (keys "hyb|centre>substituent") and the angle arms "h1", "h2"."""
        hb, h1, h2 = _half_bonds(top)
        return {"i": np.arange(len(hb)), "h1": h1, "h2": h2}, [
            "hyb|" + keyf([c], "atom") + ">" + keyf([a], "atom") for c, a in hb
        ]

    def init_from_geometry(
        self, G: dict[str, np.ndarray], I: dict[str, np.ndarray], top: Topology
    ) -> dict[str, np.ndarray]:
        """Return the initial ln m per directed bond from the reference geometry (`_hyb_init`)."""
        return {"lm": _hyb_init(G, top)}

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum (|k_a| + |k_b|)/2 Delta_ab^2 [kJ/mol] (0.0 without angles)."""
        if len(I["h1"]) == 0:
            return 0.0
        d2 = _overlap_sq(p["lm"][I["h1"]], p["lm"][I["h2"]], G["cos"])
        return jnp.sum(0.5 * (jnp.abs(p["k"][I["h1"]]) + jnp.abs(p["k"][I["h2"]])) * d2)


@register
class AngleHybridSC(Family):
    """Self-consistent hybridisation: at each centre the ln m of its bonds relax to the geometry.

        E(R) = min_z sum_ab k_ab Delta_ab(z, theta)^2 + kappa sum_a (z_a - z0_a)^2

    with z = ln m of the directed bonds of a centre (z0 = lm, fitted), k_ab = (|k_a| + |k_b|)/2
    [kJ/mol] and kappa = 20 kJ/mol + exp(mean lkap of the centre's directed bonds).  Solved by
    `newton_steps` damped Gauss-Newton steps; the gradient with respect to R follows from the
    envelope theorem (no derivative through z*, as for induced dipoles), exact at the minimum and
    approximate to the extent z* is not converged.  Rehybridisation couples all angles at a centre
    (angle-angle terms emerge with physical signs).
    """

    name = "angle_hybsc"
    params = {"lm": ((), None), "k": ((), 500.0), "lkap": ((), 5.0)}
    linear = ()
    newton_steps = 12

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return the directed bonds with their per-centre slots and the angle rows.

        Index arrays: "i" directed bonds (instances, keys "hybsc|centre>substituent"), "slot" (C, 4)
        directed-bond index per centre and neighbour slot (-1 unused; only the first four neighbours
        get slots), "pr" (na, 4) (centre row, slot a, slot b, angle index), "h1", "h2" the angle arms.
        """
        hb, h1, h2 = _half_bonds(top)
        nb = _neighbours(top)
        centres = [c for c in range(top.n) if len(nb[c]) >= 2]
        pos = {h: k for k, h in enumerate(hb)}
        slot = np.full((len(centres), 4), -1, int)  # directed-bond index per centre slot
        for r, c in enumerate(centres):
            for s_, a in enumerate(nb[c][:4]):
                slot[r, s_] = pos[(c, a)]
        pairs = []  # (centre row, slot a, slot b, angle index)
        ang = np.asarray(top.angles).reshape(-1, 3)
        crow = {c: r for r, c in enumerate(centres)}
        for m_, (a, c, b) in enumerate(ang):
            r = crow[int(c)]
            pairs.append((r, nb[int(c)].index(int(a)), nb[int(c)].index(int(b)), m_))
        pr = np.array(pairs, int).reshape(-1, 4)
        # per-centre stiffness of the relaxation kappa is tied by centre type: stored on each directed
        # bond of the centre (one key per directed bond, the centre's value is the mean)
        return {"i": np.arange(len(hb)), "slot": slot, "pr": pr, "h1": h1, "h2": h2}, [
            "hybsc|" + keyf([c], "atom") + ">" + keyf([a], "atom") for c, a in hb
        ]

    def init_from_geometry(
        self, G: dict[str, np.ndarray], I: dict[str, np.ndarray], top: Topology
    ) -> dict[str, np.ndarray]:
        """Return the initial ln m per directed bond from the reference geometry (`_hyb_init`)."""
        return {"lm": _hyb_init(G, top)}

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return the relaxed hybridisation energy [kJ/mol] (0.0 without angles).

        Notes
        -----
        z starts at z0 and takes `newton_steps` steps dz = (J^T J)^-1 J^T r (clipped to +-0.5, unused
        slots held fixed) on the residual vector r whose squared norm is E; the scan runs on
        stop_gradient values, and E = |r(z*)|^2 is then evaluated with z* held constant.
        """
        if len(I["pr"]) == 0:
            return 0.0
        slot, pr = I["slot"], I["pr"]
        mask = slot >= 0
        sl = np.maximum(slot, 0)
        z0 = jnp.where(mask, p["lm"][sl], 0.0)
        # relaxation stiffness per centre: 20 kJ/mol + exp(mean ln-kappa of its directed bonds)
        kap = 20.0 + jnp.exp(jnp.sum(jnp.where(mask, p["lkap"][sl], 0.0), 1) / jnp.maximum(mask.sum(1), 1))
        kk = 0.5 * (jnp.abs(p["k"][I["h1"]]) + jnp.abs(p["k"][I["h2"]]))  # stiffness >= 0
        cth = G["cos"][pr[:, 3]]
        rows, sa, sb = pr[:, 0], pr[:, 1], pr[:, 2]
        fm = jnp.asarray(mask, float)

        def resid(zf: jax.Array) -> jax.Array:  # E = |r|^2: Gauss-Newton normal matrix J^T J >= kappa I
            """Return the residual vector r(z) whose squared norm is the energy (z flattened)."""
            z = zf.reshape(z0.shape)
            m1, m2 = jnp.exp(z[rows, sa]), jnp.exp(z[rows, sb])
            d = (1.0 + jnp.sqrt(m1 * m2) * cth) / jnp.sqrt((1.0 + m1) * (1.0 + m2))
            return jnp.concatenate([jnp.sqrt(kk) * d, (jnp.sqrt(kap)[:, None] * fm * (z - z0)).reshape(-1)])

        jac = jax.jacfwd(resid)
        n = z0.size
        pad = jnp.diag(1.0 - fm.reshape(-1))

        def step(zf: jax.Array, _: None) -> tuple[jax.Array, None]:
            """Return z after one damped Gauss-Newton step (scan body)."""
            zs = jax.lax.stop_gradient(zf)
            r, J = resid(zs), jac(zs)
            dz = jnp.linalg.solve(J.T @ J + pad + 1e-9 * jnp.eye(n), J.T @ r) * fm.reshape(-1)
            return zs - jnp.clip(dz, -0.5, 0.5), None  # damped Gauss-Newton step on ln m

        zstar, _ = jax.lax.scan(step, jax.lax.stop_gradient(z0).reshape(-1), None, length=self.newton_steps)
        r = resid(jax.lax.stop_gradient(zstar))  # envelope theorem: no derivative through z*
        return jnp.sum(r * r)


class _TanhPair(Family):
    """Distance-only short-range term for topological pairs, E = sum_n C_n s^n, s = tanh((r - r0)/w), n = 1..4.

    Bounded at any distance.  C [kJ/mol] shape (4,), r0 [nm] from the reference geometry, width w
    [nm] (class attribute: 0.03 for 1-3, 0.06 for 1-4 pairs).
    """

    which = "13"
    width = 0.05
    params = {"C": ((4,), 0.0), "r0": ((), None)}
    linear = ("C",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per 1-3 / 1-4 pair (keys "pair key|t13" or "|t14")."""
        pairs = getattr(top, "pairs" + self.which)
        return {"i": np.arange(len(pairs))}, [keyf(pp, "pair") + "|t" + self.which for pp in pairs]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum_n C_n s^n [kJ/mol] (0.0 without pairs)."""
        if len(I["i"]) == 0:
            return 0.0
        s = jnp.tanh((G["r" + self.which][I["i"]] - p["r0"]) / self.width)
        return jnp.sum(p["C"] * jnp.stack([s, s * s, s**3, s**4], -1))


@register
class Pair13Tanh(_TanhPair):
    """Distance-only 1-3 pair term (`_TanhPair`, width 0.03 nm)."""

    name = "pair13_tanh"
    which = "13"
    width = 0.03


@register
class Pair14Tanh(_TanhPair):
    """Distance-only 1-4 pair term (`_TanhPair`, width 0.06 nm)."""

    name = "pair14_tanh"
    which = "14"
    width = 0.06


class _OverlapPair(Family):
    """Exchange-type repulsion from the overlap of the pGM Gaussian densities of a topological pair.

    E = A exp(-(b_ij r)^2), b_ij = 1/sqrt(2 (R_i^2 + R_j^2)) [1/nm] with the pGM radii (the same
    widths as the electrostatics; index extra "bij" set by bonded/model.py, `needs_radius`); one
    amplitude A [kJ/mol] per pair type.
    """

    which = "13"
    needs_radius = True
    params = {"A": ((), 0.0)}
    linear = ("A",)

    def index(
        self, top: Topology, keyf: Callable[[Sequence[int], str], str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return one instance per 1-3 / 1-4 pair (keys "pair key|o13" or "|o14")."""
        pairs = getattr(top, "pairs" + self.which)
        return {"i": np.arange(len(pairs))}, [keyf(pp, "pair") + "|o" + self.which for pp in pairs]

    def energy(
        self, G: dict[str, jax.Array], dev: dict[str, jax.Array], I: dict[str, np.ndarray], p: dict[str, jax.Array]
    ) -> jax.Array | float:
        """Return sum A exp(-(b_ij r)^2) [kJ/mol] (0.0 without pairs)."""
        if len(I["i"]) == 0:
            return 0.0
        return jnp.sum(p["A"] * jnp.exp(-((I["bij"] * G["r" + self.which][I["i"]]) ** 2)))


@register
class Pair13Ovl(_OverlapPair):
    """Gaussian-overlap repulsion of 1-3 pairs (`_OverlapPair`)."""

    name = "pair13_ovl"
    which = "13"


@register
class Pair14Ovl(_OverlapPair):
    """Gaussian-overlap repulsion of 1-4 pairs (`_OverlapPair`)."""

    name = "pair14_ovl"
    which = "14"
