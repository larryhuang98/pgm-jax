"""Pair topology of an MD system: neighbour-list groups, special pairs and constraints.

Contents: `MoleculeRule` (how the engine treats one molecule type), `MDTopology` (the per-atom
tables, built by `MDTopology.build` or `MDTopology.rigid`) and `heavy_atom_groups`.

pGM electrostatics has no exclusions, but the van der Waals term does, and molecules can be large
(proteins), so the pair rows of the force field are organised as

  groups          units of the neighbour list: a small molecule (up to `max_single` atoms, e.g.
                  water) is one group; a larger one is split into heavy-atom groups (a heavy atom
                  with its hydrogens, at most 5 atoms), so that the list stays fine-grained
  special groups  for each group, the groups with an atom within `depth` bonds of one of its atoms
                  (itself included; a symmetric relation)
  special pairs   each atom's partners in its special groups: they come from a fixed table (exact
                  float32 displacements, per-pair van der Waals weight), never from the neighbour
                  list; every other pair comes from the list with weight 1
  vdW weights     rigid molecules: 0 for every intramolecular pair (Amber's rigid water);
                  flexible ones: (d >= lj_min_sep) + lj14_scale (d == 3) with d the graph distance,
                  the model the bonded terms were fitted with
  constraints     distance constraints (i, j, d0): rigid molecules by constraints (water: three
                  distances), X-H bonds of flexible ones (md/constraints.py)
  virtual sites   (Molecule.vsites, md/vsites.py) are part of their host atom: same group, the
                  host's graph distances (weight 0 to the host itself and to what the host is
                  excluded from, as Amber's extra points); bonds to sites are not graph bonds;
                  sites cannot be constrained

Here depth = max(3, lj_min_sep - 1): every pair farther apart in the graph has weight 1 and may
come from the neighbour list.  MDTopology.rigid(sys) reproduces the rigid-body engine (groups =
molecules, every intramolecular pair special with weight 0, no constraints).  All tables are host
numpy arrays, built once.

Units: nm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

from ..bonded.topology import near_pairs

if TYPE_CHECKING:
    from ..system import Molecule, System

MAX_SINGLE = 6  # molecules up to this many atoms are one neighbour-list group


@dataclass
class MoleculeRule:
    """How the MD engine treats one molecule type.

    A mutable dataclass; rules are compared by identity in `MDTopology.build` (molecules sharing a
    Molecule and a rule object are processed once).

    Parameters
    ----------
    bonds : list of tuple of int
        Bonds (i, j) of the molecular graph, in the molecule's atom order (bonds to virtual sites
        are ignored).
    vdw : {"none", "graph"}
        "none": no intramolecular van der Waals (weight 0 for every intramolecular pair); "graph":
        weights from graph distances.
    lj_min_sep : int
        Smallest graph distance (in bonds) with full van der Waals weight ("graph" only).
    lj14_scale : float
        Weight of the pairs three bonds apart (added to the full weight when lj_min_sep <= 3).
    constraints : tuple of (int, int, float)
        Distance constraints (i, j, d0) in the molecule's atom order, d0 [nm].
    """

    bonds: list
    vdw: str = "none"
    lj_min_sep: int = 4
    lj14_scale: float = 0.0
    constraints: tuple = ()


def heavy_atom_groups(elements: Sequence[str], bonds: Sequence[tuple[int, int]]) -> np.ndarray:
    """Return the group index of every atom: each heavy atom with the hydrogens bonded to it.

    Hydrogens bonded to nothing heavy (e.g. H2) join the group of their first grouped neighbour,
    or form a group of their own.

    Parameters
    ----------
    elements : Sequence[str] (n,)
        Element symbols ("H" for hydrogen).
    bonds : Sequence[tuple[int, int]]
        Bonds (i, j) in local atom indices.

    Returns
    -------
    np.ndarray (n,) int
        Group index per atom, 0 .. n_groups-1 (heavy atoms numbered in atom order first).
    """
    n = len(elements)
    g = np.full(n, -1, int)
    k = 0
    for i, e in enumerate(elements):
        if e != "H":
            g[i] = k
            k += 1
    for i, j in bonds:
        for h, x in ((i, j), (j, i)):
            if elements[h] == "H" and g[h] < 0 and elements[x] != "H":
                g[h] = g[x]
    for i in range(n):
        if g[i] < 0:
            nb = [x for a, b in bonds for x in (a, b) if i in (a, b) and x != i and g[x] >= 0]
            g[i] = g[nb[0]] if nb else k
            if not nb:
                k += 1
    return g


@dataclass
class MDTopology:
    """Pair topology tables of a system (module docstring); a mutable dataclass of host arrays.

        top = MDTopology.build(system, rules)          # one MoleculeRule per molecule
        top = MDTopology.rigid(system)                 # the rigid-body engine

    Parameters
    ----------
    n : int
        Number of atoms N.
    mol : np.ndarray (N,) int
        Molecule of every atom.
    group : np.ndarray (N,) int
        Neighbour-list group of every atom.
    n_group : int
        Number of groups.
    special : np.ndarray (N, S) int32
        Special partners of every atom (padding N).
    special_w : np.ndarray (N, S)
        Van der Waals weights of the special pairs (0 for padding), dimensionless.
    special_groups : np.ndarray (N, Gs) int32
        Special groups of each atom's group (padding -1).
    constraints : np.ndarray (nc, 2) int
        Constrained atom pairs (global indices).
    constraint_d0 : np.ndarray (nc,)
        Constraint distances [nm].
    """

    n: int
    mol: np.ndarray  # (N,) molecule of every atom
    group: np.ndarray  # (N,) neighbour-list group of every atom
    n_group: int
    special: np.ndarray  # (N, S) special partners (padding N)
    special_w: np.ndarray  # (N, S) van der Waals weights of the special pairs
    special_groups: np.ndarray  # (N, Gs) special groups of each atom's group (padding -1)
    constraints: np.ndarray  # (nc, 2) atom pairs
    constraint_d0: np.ndarray  # (nc,) nm

    # ------------------------------------------------------------------ builders
    @classmethod
    def rigid(cls, sys: System) -> MDTopology:
        """Return the topology of the rigid-body engine.

        Groups = molecules (no size limit), every intramolecular pair special with weight 0, no
        constraints.

        Parameters
        ----------
        sys : System
            The system.

        Returns
        -------
        MDTopology
        """
        rule = MoleculeRule(bonds=[], vdw="none")
        return cls.build(sys, [rule] * sys.nmol, max_single=10**9)

    @classmethod
    def build(cls, sys: System, rules: Sequence[MoleculeRule], max_single: int = MAX_SINGLE) -> MDTopology:
        """Build the topology from one rule per molecule.

        Parameters
        ----------
        sys : System
            The system.
        rules : Sequence[MoleculeRule] (nmol,)
            rules[k] is the MoleculeRule of sys.molecules[k]; molecules that share their Molecule and
            rule objects are processed once.
        max_single : int
            Molecules with up to this many atoms (or without bonds) are one neighbour-list group.

        Returns
        -------
        MDTopology
            S and Gs are the largest numbers of special partners / special groups of any atom.

        Raises
        ------
        ValueError
            If len(rules) != sys.nmol, or from `_molecule` (constraints on virtual sites, a large
            molecule with vdw="none").
        """
        if len(rules) != sys.nmol:
            raise ValueError("one rule per molecule")
        N = sys.n
        group = np.zeros(N, int)
        sp_rows, sgs, cons, d0s = [None] * N, [None] * N, [], []
        cache = {}
        g0 = 0
        for k, (m, rule) in enumerate(zip(sys.molecules, rules)):
            off = int(sys.offsets[k])
            key = (id(m), id(rule))  # identical molecules share their Molecule and rule objects
            if key not in cache:
                cache[key] = cls._molecule(m, rule, max_single)
            lg, n_lg, sp, sg, c = cache[key]
            group[off : off + m.n] = lg + g0
            for a in range(m.n):
                sp_rows[off + a] = [(off + b, w) for b, w in sp[a]]
                sgs[off + a] = [g + g0 for g in sg[lg[a]]]
            for i, j, d in c:
                cons.append((off + i, off + j))
                d0s.append(d)
            g0 += n_lg
        S = max([len(r) for r in sp_rows] + [1])
        Gs = max([len(r) for r in sgs] + [1])
        special = np.full((N, S), N, np.int32)  # padding: index N (a dummy atom)
        special_w = np.zeros((N, S))
        special_groups = np.full((N, Gs), -1, np.int32)
        for a in range(N):
            for s, (b, w) in enumerate(sp_rows[a]):
                special[a, s], special_w[a, s] = b, w
            special_groups[a, : len(sgs[a])] = sgs[a]
        return cls(
            n=N,
            mol=np.asarray(sys.mol),
            group=group,
            n_group=g0,
            special=special,
            special_w=special_w,
            special_groups=special_groups,
            constraints=np.array(cons, int).reshape(-1, 2),
            constraint_d0=np.array(d0s, float),
        )

    @staticmethod
    def _molecule(
        m: Molecule, rule: MoleculeRule, max_single: int
    ) -> tuple[np.ndarray, int, list[list[tuple[int, float]]], list[list[int]], list[tuple[int, int, float]]]:
        """Return the local groups, special pairs and constraints of one molecule.

        Parameters
        ----------
        m : Molecule
            The molecule (its `vsites`, `elements`, `name`, `n` are used).
        rule : MoleculeRule
            Its rule.
        max_single : int
            Molecules with up to this many atoms (or without bonds) are one group.

        Returns
        -------
        lg : np.ndarray (n,) int
            Local group of every atom (virtual sites in their host's group).
        n_lg : int
            Number of local groups.
        sp : list of list of (int, float)
            Special partners (local atom, van der Waals weight) of every atom, sorted by atom.
        sg : list of list of int
            Special groups of every local group, sorted.
        constraints : list of (int, int, float)
            The rule's constraints (local indices, d0 [nm]).

        Raises
        ------
        ValueError
            If a constraint involves a virtual site, or a molecule of more than `max_single` atoms
            with bonds has vdw="none" (it must be one group).
        """
        n = m.n
        host = {vs.site: vs.host for vs in (getattr(m, "vsites", None) or ())}  # site -> host atom
        bonds = [(int(i), int(j)) for i, j in rule.bonds if int(i) not in host and int(j) not in host]
        bad = [(i, j) for i, j, _ in rule.constraints if i in host or j in host]
        if bad:
            raise ValueError(f"{m.name}: constraints {bad} involve virtual sites (sites are placed, not constrained)")
        if n <= max_single or not bonds:
            lg = np.zeros(n, int)
        else:
            if rule.vdw == "none":
                raise ValueError(
                    f"{m.name}: a molecule without intramolecular van der Waals must be one group "
                    f"({n} atoms > {max_single})"
                )
            lg = heavy_atom_groups(list(m.elements), bonds)
            if host:  # a site joins its host's group (renumbered, no empty groups)
                for a, h in host.items():
                    lg[a] = lg[h]
                lg = np.unique(lg, return_inverse=True)[1].reshape(-1)
        n_lg = int(lg.max()) + 1 if n else 0
        depth = max(3, int(rule.lj_min_sep) - 1)  # pairs beyond it have full weight: not special
        nbr = [[] for _ in range(n)]
        for i, j in bonds:
            nbr[i].append(j)
            nbr[j].append(i)
        near = near_pairs(nbr, depth) if bonds else {}
        sg = [{g} for g in range(n_lg)]
        for i, j in near:
            sg[lg[i]].add(lg[j])
            sg[lg[j]].add(lg[i])
        members = [np.nonzero(lg == g)[0] for g in range(n_lg)]

        def weight(i: int, j: int) -> float:
            """Return the van der Waals weight of the local pair (i, j) (sites take their host's place)."""
            if rule.vdw == "none":
                return 0.0
            i, j = host.get(i, i), host.get(j, j)  # a site takes its host's place in the graph
            if i == j:
                return 0.0
            d = near.get((min(i, j), max(i, j)), depth + 1)
            return float(d >= rule.lj_min_sep) + float(rule.lj14_scale) * (d == 3)

        sp = []
        for a in range(n):
            part = sorted(int(b) for g in sg[lg[a]] for b in members[g] if b != a)
            sp.append([(b, weight(a, b)) for b in part])
        return lg, n_lg, sp, [sorted(s) for s in sg], list(rule.constraints)

    # ------------------------------------------------------------------ helpers
    @property
    def n_constraints(self) -> int:
        """Number of distance constraints."""
        return len(self.constraints)

    def group_radius(self, pos: ArrayLike, masses: ArrayLike) -> float:
        """Return the largest atom-to-group-centre distance (host).

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Positions [nm], molecules whole.
        masses : ArrayLike (N,)
            Masses [amu] (the group centres are mass-weighted; groups must have a nonzero mass).

        Returns
        -------
        float
            Largest distance of an atom from the centre of mass of its group [nm] (0 for no atoms).
        """
        pos = np.asarray(pos, float)
        m = np.asarray(masses, float)
        c = np.zeros((self.n_group, 3))
        np.add.at(c, self.group, m[:, None] * pos)
        c /= np.bincount(self.group, weights=m, minlength=self.n_group)[:, None]
        return float(np.max(np.linalg.norm(pos - c[self.group], axis=1))) if self.n else 0.0
