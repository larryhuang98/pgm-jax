"""Build the valence topology of one molecule from its bond graph.

Contents: `Topology` (internal coordinates, the coupling index sets of class II force fields,
topological pair classes, peptide backbone and residues, symmetry classes for tying) and
`build_topology`; graph helpers `near_pairs`, `atom_classes`, `peptide_backbone`.

All index arrays are numpy int arrays (static for jit).  Conventions:

    bonds (nb, 2); angles (na, 3) with the centre in the middle; propers (nt, 4) i-j-k-l about
    bond j-k; impropers (ni, 4) = (centre, a, b, c) for planar 3-coordinated centres;
    graph distance matrix `dist` (n, n) in bonds (1 = 1-2, 2 = 1-3, 3 = 1-4, ...), dense for
    molecules up to DENSE_MAX atoms (None above; pairs13/pairs14 are always built, pairs15 only
    with the dense matrix).

Couplings (Abdullah et al. [1]_, eqs 3, 5, 6, 8-10):

    bond_bond      the two bonds of every angle
    bond_angle     (bond, angle) for the two bonds of every angle
    angle_angle    angle pairs with the same centre sharing one arm
    torsion_bond   (torsion, bond) for the three bonds of every torsion
    torsion_angle  (torsion, angle) for the two angles of every torsion
    aat            (torsion, angle1, angle2): the two angles of every torsion

Peptides (found from the graph, no atom names needed; `peptide_backbone`):

    residue        (n,) residue index of every atom: the components left after cutting the
                   peptide bonds C(i-1)-N(i) (caps such as ACE / NME are residues of their own)
    cmaps          (k, 5) C(i-1), N, CA, C, N(i+1) for every residue with both backbone torsions
                   phi = C(i-1)-N-CA-C and psi = N-CA-C-N(i+1)

References
----------
.. [1] A. S. Abdullah, Y. Wang, M. F. S. J. Menger, S. Sami, T. Head-Gordon, J. Chem. Theory
   Comput. 21, 11669 (2025). doi:10.1021/acs.jctc.5c01458
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

DENSE_MAX = 1000  # atoms: dense graph-distance matrix and pairs15 up to this size


def _neighbours(n: int, bonds: ArrayLike) -> list[list[int]]:
    """Return the neighbour list of every atom (in bond order, unsorted)."""
    nbr = [[] for _ in range(n)]
    for i, j in bonds:
        nbr[int(i)].append(int(j))
        nbr[int(j)].append(int(i))
    return nbr


def _graph_dist(n: int, bonds: ArrayLike) -> tuple[np.ndarray, list[list[int]]]:
    """Return the dense graph distances (n, n) in bonds (99 = not connected) and the neighbour lists.

    Breadth-first search from every atom, O(n (n + nb)).
    """
    nbr = _neighbours(n, bonds)
    D = np.full((n, n), 99, int)
    for s in range(n):
        D[s, s] = 0
        front = [s]
        while front:
            nxt = []
            for u in front:
                for v in nbr[u]:
                    if D[s, v] == 99:
                        D[s, v] = D[s, u] + 1
                        nxt.append(v)
            front = nxt
    return D, nbr


def near_pairs(nbr: Sequence[Sequence[int]], depth: int = 3) -> dict:
    """Return {(i, j): graph distance} for i < j within `depth` bonds (breadth-first, linear in n).

    Parameters
    ----------
    nbr : sequence of sequence of int
        Neighbour lists.
    depth : int
        Largest graph distance kept [bonds].
    """
    out = {}
    for s in range(len(nbr)):
        seen = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            if seen[u] == depth:
                continue
            for v in nbr[u]:
                if v not in seen:
                    seen[v] = seen[u] + 1
                    q.append(v)
        for v, d in seen.items():
            if v > s:
                out[(s, v)] = d
    return out


def _ring_bonds(bonds: Sequence[tuple[int, int]], nbr: Sequence[Sequence[int]]) -> set[frozenset]:
    """Return the bonds in a ring: (i, j) with j reachable from i without the bond itself (as frozensets)."""
    ring = set()
    for i, j in bonds:
        seen, stack = {i}, [i]
        while stack:
            u = stack.pop()
            for v in nbr[u]:
                if {u, v} == {i, j} or v in seen:
                    continue
                seen.add(v)
                stack.append(v)
        if j in seen:
            ring.add(frozenset((i, j)))
    return ring


def atom_classes(elements: Sequence[str], bonds: Sequence[tuple[int, int]], rounds: int = 6) -> list[str]:
    """Return symmetry classes of the atoms by colour refinement on the bond graph.

    The element is the initial colour; each round appends the sorted colours of the neighbours and
    renames the colours "c0", "c1", ...  Stops when a round no longer splits a class, or after
    `rounds` rounds.

    Parameters
    ----------
    elements : sequence of str
        Element symbols.
    bonds : sequence of (int, int)
        Bonds.
    rounds : int
        Maximum number of refinement rounds.

    Returns
    -------
    list of str
        Class name per atom (symmetry-equivalent atoms share it).
    """
    n = len(elements)
    nbr = _neighbours(n, bonds)
    col = [str(e) for e in elements]
    for _ in range(rounds):
        new = [col[i] + "(" + ",".join(sorted(col[j] for j in nbr[i])) + ")" for i in range(n)]
        uniq = {c: k for k, c in enumerate(sorted(set(new)))}
        new = [f"c{uniq[c]}" for c in new]
        if len(set(new)) == len(set(col)):
            col = new
            break
        col = new
    return col


# ------------------------------------------------------------------ peptides
def _carbonyl_carbons(elements: Sequence[str], nbr: Sequence[Sequence[int]], order: dict) -> set[int]:
    """Return the carbons with a double (or resonance) bond to an oxygen.

    Without bond orders (`order` has no entry for the C-O bond): carbons with three neighbours of
    which one is a terminal oxygen.  `order` maps sorted atom pairs to bond orders.
    """
    out = set()
    for c, e in enumerate(elements):
        if e != "C":
            continue
        for o in nbr[c]:
            if elements[o] != "O":
                continue
            bo = order.get((min(c, o), max(c, o)))
            if (bo is not None and bo >= 1.5) or (bo is None and len(nbr[c]) == 3 and len(nbr[o]) == 1):
                out.add(c)
                break
    return out


def peptide_backbone(
    elements: Sequence[str], nbr: Sequence[Sequence[int]], order: dict | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the backbone units (N, CA, C) of a peptide graph, the phi/psi quintuples and the residues.

    A unit is N - CA - C with N a nitrogen, CA a tetrahedral carbon (not a carbonyl carbon) and C a
    carbonyl carbon; its previous carbonyl C(i-1) is the carbonyl carbon bonded to N (other than
    through CA), its next nitrogen N(i+1) the nitrogen bonded to C.  Peptide bonds are C(i-1)-N(i)
    and C(i)-N(i+1) bonds; cutting them splits the molecule into residues.

    Parameters
    ----------
    elements : sequence of str
        Element symbols.
    nbr : sequence of sequence of int
        Neighbour lists.
    order : dict, optional
        Bond orders by sorted atom pair; None: none known (carbonyls from the graph).

    Returns
    -------
    units : np.ndarray (k, 3) int
        Backbone units (N, CA, C), sorted.
    cmaps : np.ndarray (m, 5) int
        Quintuples C(i-1), N, CA, C, N(i+1) of units with both neighbours, sorted.
    residue : np.ndarray (n,) int
        Residue index of every atom, residues numbered by their lowest atom.
    """
    order = order or {}
    n = len(elements)
    carb = _carbonyl_carbons(elements, nbr, order)
    units, cmaps, peptide = [], [], set()
    for ca in range(n):
        if elements[ca] != "C" or len(nbr[ca]) != 4 or ca in carb:
            continue
        for nn in nbr[ca]:
            if elements[nn] != "N":
                continue
            for c in nbr[ca]:
                if c not in carb:
                    continue
                units.append((nn, ca, c))
                prev = [x for x in nbr[nn] if x != ca and x in carb]
                nxt = [x for x in nbr[c] if elements[x] == "N"]
                for p in prev:
                    peptide.add((min(p, nn), max(p, nn)))
                for x in nxt:
                    peptide.add((min(c, x), max(c, x)))
                if prev and nxt:
                    cmaps.append((prev[0], nn, ca, c, nxt[0]))
    # residues: connected components without the peptide bonds, numbered by their lowest atom
    residue = np.full(n, -1, int)
    r = 0
    for s in range(n):
        if residue[s] >= 0:
            continue
        residue[s] = r
        stack = [s]
        while stack:
            u = stack.pop()
            for v in nbr[u]:
                if residue[v] < 0 and (min(u, v), max(u, v)) not in peptide:
                    residue[v] = r
                    stack.append(v)
        r += 1
    return (np.array(sorted(set(units)), int).reshape(-1, 3), np.array(sorted(set(cmaps)), int).reshape(-1, 5), residue)


@dataclass
class Topology:
    """Valence topology of one molecule (a mutable dataclass of numpy index arrays; see the module docstring).

    Parameters
    ----------
    n : int
        Number of atoms.
    elements : list of str
        Element symbols.
    bonds : np.ndarray (nb, 2) int
        Bonds, sorted.
    angles : np.ndarray (na, 3) int
        Angles i-j-k, centre j.
    propers : np.ndarray (nt, 4) int
        Proper torsions i-j-k-l about bond j-k.
    impropers : np.ndarray (ni, 4) int
        (centre, a, b, c) of planar 3-coordinated centres.
    dist : np.ndarray (n, n) int or None
        Graph distances in bonds; None above DENSE_MAX atoms.
    rigid_torsion : np.ndarray (nt,) bool
        Central bond in a ring or of order > 1.
    classes : list of str
        Symmetry class per atom (base of the tying keys).
    bond_bond, bond_angle, angle_angle, torsion_bond, torsion_angle : np.ndarray (k, 2) int
        Coupling index pairs (see the module docstring).
    aat : np.ndarray (nt, 3) int
        (torsion, angle1, angle2).
    pairs13, pairs14 : np.ndarray (k, 2) int
        Atom pairs 2 and 3 bonds apart.
    pairs15 : np.ndarray (k, 2) int or None
        Pairs at least 4 bonds apart (and unconnected pairs); None above DENSE_MAX atoms.
    amber_impropers : np.ndarray (k, 4) int or None
        Amber-ordered impropers (centre third), set from a prmtop (bonded/amber.py).
    residue : np.ndarray (n,) int
        Residue index (`peptide_backbone`).
    backbone : np.ndarray (k, 3) int
        N, CA, C of every backbone unit.
    cmaps : np.ndarray (m, 5) int
        C(i-1), N, CA, C, N(i+1) per residue with both backbone torsions.
    near : dict
        {(i, j): d} for pairs i < j within 3 bonds.
    """

    n: int
    elements: list
    bonds: np.ndarray
    angles: np.ndarray
    propers: np.ndarray
    impropers: np.ndarray
    dist: np.ndarray  # (n, n) graph distances, or None above DENSE_MAX atoms
    rigid_torsion: np.ndarray  # (nt,) bool: central bond in a ring or of order > 1
    classes: list  # symmetry class per atom (tying key base)
    bond_bond: np.ndarray = field(default=None)
    bond_angle: np.ndarray = field(default=None)
    angle_angle: np.ndarray = field(default=None)
    torsion_bond: np.ndarray = field(default=None)
    torsion_angle: np.ndarray = field(default=None)
    aat: np.ndarray = field(default=None)
    pairs13: np.ndarray = field(default=None)
    pairs14: np.ndarray = field(default=None)
    pairs15: np.ndarray = field(default=None)  # None above DENSE_MAX atoms
    amber_impropers: np.ndarray = field(default=None)  # (k, 4) Amber-ordered impropers (centre third), from a prmtop
    residue: np.ndarray = field(default=None)  # (n,) residue index (peptide_backbone)
    backbone: np.ndarray = field(default=None)  # (k, 3) N, CA, C of every backbone unit
    cmaps: np.ndarray = field(default=None)  # (m, 5) C(i-1), N, CA, C, N(i+1)
    near: dict = field(default=None, repr=False)  # {(i, j): d} for pairs within 3 bonds

    # ------------------------------------------------------------------ keys
    def key(self, atoms: Sequence[int], kind: str, classes: Sequence[str] | None = None) -> str:
        """Return the tying key of a term instance, invariant under the term's own symmetry.

        Parameters
        ----------
        atoms : sequence of int
            Atoms of the instance.
        kind : str
            "bond", "pair", "torsion" (read in either direction), "angle" (outer atoms interchangeable),
            "improper" (outer atoms in any order), or any other kind (order kept, e.g. "cmap").
        classes : sequence of str, optional
            Atom classes to use; None: `self.classes`.

        Returns
        -------
        str
            kind + ":" + the classes joined by "-".
        """
        c = [(classes or self.classes)[a] for a in atoms]
        if kind in ("bond", "pair", "torsion"):
            c = min(c, c[::-1])
        elif kind == "angle":
            c = [min(c[0], c[2]), c[1], max(c[0], c[2])]
        elif kind == "improper":
            c = [c[0]] + sorted(c[1:])
        return kind + ":" + "-".join(c)

    def graph_distance(self, i: int, j: int) -> int:
        """Return the graph distance of atoms i, j in bonds (4 stands for "at least 4" without the dense matrix)."""
        if self.dist is not None:
            return int(self.dist[i, j])
        if i == j:
            return 0
        return self.near.get((min(i, j), max(i, j)), 4)


def build_topology(
    elements: Sequence[str],
    bonds: Sequence[tuple[int, int]],
    bond_orders: tuple[Sequence[tuple[int, int]], Sequence[float]] | None = None,
    xyz: ArrayLike | None = None,
    classes: list[str] | None = None,
    dense: bool | None = None,
) -> Topology:
    """Build the Topology of a molecule from its elements and bonds.

    Parameters
    ----------
    elements : sequence of str
        Element symbols.
    bonds : sequence of (int, int)
        Bonds (any order; sorted here).
    bond_orders : (pairs, orders), optional
        Bond orders of the listed atom pairs (1, 1.5, 2, 3); None: all single.  Used for rigid
        torsions and carbonyl detection.
    xyz : ArrayLike (n, 3), optional
        Coordinates (any length unit) to decide which 3-coordinated centres are planar (sum of the
        three angles > 350 deg); None: every 3-coordinated centre gets an improper.
    classes : list of str, optional
        Atom classes for the tying keys; None: `atom_classes` (colour refinement).
    dense : bool, optional
        Build the dense distance matrix and pairs15; None: for up to DENSE_MAX atoms.

    Returns
    -------
    Topology
        The topology with all index sets and the peptide backbone.
    """
    n = len(elements)
    bonds = np.array(sorted(tuple(sorted(b)) for b in bonds), int).reshape(-1, 2)
    order = {}
    if bond_orders is not None:
        raw = [tuple(sorted(b)) for b in bond_orders[0]]
        order = dict(zip(raw, bond_orders[1]))
    dense = n <= DENSE_MAX if dense is None else dense
    nbr = _neighbours(n, bonds)
    D = _graph_dist(n, bonds)[0] if dense else None
    near = near_pairs(nbr, 3)
    ring = _ring_bonds([tuple(b) for b in bonds], nbr)
    bidx = {tuple(b): k for k, b in enumerate(bonds)}

    def bond_of(i: int, j: int) -> int:
        """Return the index of bond i-j."""
        return bidx[tuple(sorted((i, j)))]

    angles = [(i, j, k) for j in range(n) for a, i in enumerate(sorted(nbr[j])) for k in sorted(nbr[j])[a + 1 :]]
    angles = np.array(angles, int).reshape(-1, 3)
    aidx = {}
    for m, (i, j, k) in enumerate(angles):
        aidx[(i, j, k)] = m
        aidx[(k, j, i)] = m

    propers, rigid = [], []  # rigid: central bond in a ring or of order > 1
    for j, k in bonds:
        for i in sorted(nbr[j]):
            if i == k:
                continue
            for l in sorted(nbr[k]):
                if l in (j, i):
                    continue
                propers.append((i, j, k, l))
                rigid.append(frozenset((j, k)) in ring or order.get((min(j, k), max(j, k)), 1.0) > 1.0)
    propers = np.array(propers, int).reshape(-1, 4)

    impropers = []
    for c in range(n):
        if len(nbr[c]) != 3:
            continue
        a, b, d = sorted(nbr[c])
        planar = True
        if xyz is not None:  # sum of the three angles at the centre ~ 360 degrees
            x = np.asarray(xyz)
            ang = 0.0
            for p, q in ((a, b), (b, d), (a, d)):
                u, v = x[p] - x[c], x[q] - x[c]
                ang += np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
            planar = ang > 350.0  # degrees: within 10 deg of planar
        if planar:
            impropers.append((c, a, b, d))
    impropers = np.array(impropers, int).reshape(-1, 4)

    top = Topology(
        n=n,
        elements=list(elements),
        bonds=bonds,
        angles=angles,
        propers=propers,
        impropers=impropers,
        dist=D,
        rigid_torsion=np.array(rigid, bool),
        classes=classes or atom_classes(elements, [tuple(b) for b in bonds]),
        near=near,
    )
    top.bond_bond = np.array([(bond_of(i, j), bond_of(j, k)) for i, j, k in angles], int).reshape(-1, 2)
    top.bond_angle = np.array(
        [(bond_of(a_[0], a_[1]), m) for m, (i, j, k) in enumerate(angles) for a_ in ((i, j), (j, k))], int
    ).reshape(-1, 2)
    by_centre = {}
    for m, (_i, j, _k) in enumerate(angles):
        by_centre.setdefault(int(j), []).append(m)
    aa = []
    for ms in by_centre.values():  # angle pairs share their centre
        for p, m1 in enumerate(ms):
            i1, _, k1 = angles[m1]
            for m2 in ms[p + 1 :]:
                i2, _, k2 = angles[m2]
                if len({i1, k1} & {i2, k2}) == 1:
                    aa.append((m1, m2))
    top.angle_angle = np.array(sorted(aa), int).reshape(-1, 2)
    top.torsion_bond = np.array(
        [(t, bond_of(*p)) for t, (i, j, k, l) in enumerate(propers) for p in ((i, j), (j, k), (k, l))], int
    ).reshape(-1, 2)
    top.torsion_angle = np.array(
        [(t, aidx[a_]) for t, (i, j, k, l) in enumerate(propers) for a_ in ((i, j, k), (j, k, l))], int
    ).reshape(-1, 2)
    top.aat = np.array([(t, aidx[(i, j, k)], aidx[(j, k, l)]) for t, (i, j, k, l) in enumerate(propers)], int).reshape(
        -1, 3
    )
    for name, d in (("pairs13", 2), ("pairs14", 3)):
        pp = sorted(p for p, dd in near.items() if dd == d)
        setattr(top, name, np.array(pp, int).reshape(-1, 2))
    if dense:
        iu = np.triu_indices(n, 1)
        sel = D[iu] >= 4
        top.pairs15 = np.stack([iu[0][sel], iu[1][sel]], 1).astype(int).reshape(-1, 2)
    top.backbone, top.cmaps, top.residue = peptide_backbone(elements, nbr, order)
    return top
