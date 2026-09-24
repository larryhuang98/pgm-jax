"""Valence topology of one molecule from its bond graph: internal coordinates, the coupling index
sets of class II force fields, topological pair classes, and symmetry classes for tying.

All index arrays are numpy int arrays (static for jit).  Conventions:
  bonds (nb, 2); angles (na, 3) with the centre in the middle; propers (nt, 4) i-j-k-l about
  bond j-k; impropers (ni, 4) = (centre, a, b, c) for planar 3-coordinated centres;
  graph distance matrix `dist` (n, n) in bonds (1 = 1-2, 2 = 1-3, 3 = 1-4, ...).
Couplings (Abdullah et al. 2025, eqs 3, 5, 6, 8-10):
  bond_bond      the two bonds of every angle
  bond_angle     (bond, angle) for the two bonds of every angle
  angle_angle    angle pairs with the same centre sharing one arm
  torsion_bond   (torsion, bond) for the three bonds of every torsion
  torsion_angle  (torsion, angle) for the two angles of every torsion
  aat            (torsion, angle1, angle2): the two angles of every torsion
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _graph_dist(n, bonds):
    D = np.full((n, n), 99, int)
    nbr = [[] for _ in range(n)]
    for i, j in bonds:
        nbr[i].append(j); nbr[j].append(i)
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


def _ring_bonds(bonds, nbr):
    ring = set()
    for i, j in bonds:
        seen, stack = {i}, [i]
        while stack:
            u = stack.pop()
            for v in nbr[u]:
                if {u, v} == {i, j} or v in seen:
                    continue
                seen.add(v); stack.append(v)
        if j in seen:
            ring.add(frozenset((i, j)))
    return ring


def atom_classes(elements, bonds, rounds: int = 6):
    """Symmetry classes by colour refinement on the bond graph (element as initial colour)."""
    n = len(elements)
    _, nbr = _graph_dist(n, bonds)
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


@dataclass
class Topology:
    n: int
    elements: list
    bonds: np.ndarray
    angles: np.ndarray
    propers: np.ndarray
    impropers: np.ndarray
    dist: np.ndarray
    rigid_torsion: np.ndarray            # (nt,) bool: central bond in a ring or of order > 1
    classes: list                         # symmetry class per atom (tying key base)
    bond_bond: np.ndarray = field(default=None)
    bond_angle: np.ndarray = field(default=None)
    angle_angle: np.ndarray = field(default=None)
    torsion_bond: np.ndarray = field(default=None)
    torsion_angle: np.ndarray = field(default=None)
    aat: np.ndarray = field(default=None)
    pairs13: np.ndarray = field(default=None)
    pairs14: np.ndarray = field(default=None)
    pairs15: np.ndarray = field(default=None)

    # ------------------------------------------------------------------ keys
    def key(self, atoms, kind: str, classes=None) -> str:
        """Tying key of a term instance, invariant under the term's own symmetry."""
        c = [(classes or self.classes)[a] for a in atoms]
        if kind in ("bond", "pair", "torsion"):
            c = min(c, c[::-1])
        elif kind == "angle":
            c = [min(c[0], c[2]), c[1], max(c[0], c[2])]
        elif kind == "improper":
            c = [c[0]] + sorted(c[1:])
        return kind + ":" + "-".join(c)


def build_topology(elements, bonds, bond_orders=None, xyz=None, classes=None) -> Topology:
    n = len(elements)
    bonds = np.array(sorted(tuple(sorted(b)) for b in bonds), int).reshape(-1, 2)
    order = {}
    if bond_orders is not None:
        raw = [tuple(sorted(b)) for b in bond_orders[0]]
        order = dict(zip(raw, bond_orders[1]))
    D, nbr = _graph_dist(n, bonds)
    ring = _ring_bonds([tuple(b) for b in bonds], nbr)
    bidx = {tuple(b): k for k, b in enumerate(bonds)}
    bond_of = lambda i, j: bidx[tuple(sorted((i, j)))]

    angles = [(i, j, k) for j in range(n) for a, i in enumerate(sorted(nbr[j])) for k in sorted(nbr[j])[a + 1:]]
    angles = np.array(angles, int).reshape(-1, 3)
    aidx = {}
    for m, (i, j, k) in enumerate(angles):
        aidx[(i, j, k)] = m; aidx[(k, j, i)] = m

    propers, rigid = [], []
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
        if xyz is not None:            # sum of the three angles at the centre ~ 360 degrees
            x = np.asarray(xyz)
            ang = 0.0
            for p, q in ((a, b), (b, d), (a, d)):
                u, v = x[p] - x[c], x[q] - x[c]
                ang += np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
            planar = ang > 350.0
        if planar:
            impropers.append((c, a, b, d))
    impropers = np.array(impropers, int).reshape(-1, 4)

    top = Topology(n=n, elements=list(elements), bonds=bonds, angles=angles, propers=propers,
                   impropers=impropers, dist=D, rigid_torsion=np.array(rigid, bool),
                   classes=classes or atom_classes(elements, [tuple(b) for b in bonds]))
    top.bond_bond = np.array([(bond_of(i, j), bond_of(j, k)) for i, j, k in angles], int).reshape(-1, 2)
    top.bond_angle = np.array([(bond_of(a_[0], a_[1]), m) for m, (i, j, k) in enumerate(angles)
                               for a_ in ((i, j), (j, k))], int).reshape(-1, 2)
    aa = []
    for m1, (i1, j1, k1) in enumerate(angles):
        for m2 in range(m1 + 1, len(angles)):
            i2, j2, k2 = angles[m2]
            if j1 == j2 and len({i1, k1} & {i2, k2}) == 1:
                aa.append((m1, m2))
    top.angle_angle = np.array(aa, int).reshape(-1, 2)
    top.torsion_bond = np.array([(t, bond_of(*p)) for t, (i, j, k, l) in enumerate(propers)
                                 for p in ((i, j), (j, k), (k, l))], int).reshape(-1, 2)
    top.torsion_angle = np.array([(t, aidx[a_]) for t, (i, j, k, l) in enumerate(propers)
                                  for a_ in ((i, j, k), (j, k, l))], int).reshape(-1, 2)
    top.aat = np.array([(t, aidx[(i, j, k)], aidx[(j, k, l)]) for t, (i, j, k, l) in enumerate(propers)], int).reshape(-1, 3)
    iu = np.triu_indices(n, 1)
    for name, sel in (("pairs13", D[iu] == 2), ("pairs14", D[iu] == 3), ("pairs15", D[iu] >= 4)):
        setattr(top, name, np.stack([iu[0][sel], iu[1][sel]], 1).astype(int).reshape(-1, 2))
    return top
