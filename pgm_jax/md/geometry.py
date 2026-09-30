"""Rigid-molecule geometry from the prmtop: rebuild coordinates whose rigid triangles disagree with it.

pmemd's SHAKE pulls every constrained molecule to the prmtop's bond lengths at the first step; the rigid-body
engine keeps the geometry of the coordinates.  `conform_rigid_geometry` gives the coordinates the prmtop's
geometry (three-atom molecules with three bonds: water), keeping the first atom and the orientation, so a
restart written with another geometry (e.g. TIP3P coordinates under a DEGAUSS-3p prmtop) simulates the
model the prmtop defines.

Contents: conform_rigid_geometry.

Units: nm (coordinates), Angstrom in the prmtop.
"""

from __future__ import annotations

import numpy as np

from ..prmtop import Prmtop


def conform_rigid_geometry(prmtop: str, pos: np.ndarray, tol: float = 1e-4) -> tuple[np.ndarray, int]:
    """Return the coordinates with every three-atom, three-bond molecule at the prmtop's bond lengths.

    The first atom stays where it is; the other two are placed in the molecule's plane at the prmtop's
    distances (bisector kept for equal bonds, so the orientation is that of the input).

    Parameters
    ----------
    prmtop : str
        Amber prmtop (bonds with hydrogen and without, BOND_EQUIL_VALUE in Angstrom).
    pos : np.ndarray (N, 3)
        Atom positions [nm].
    tol : float
        Bond-length disagreement [nm] above which a molecule is rebuilt.

    Returns
    -------
    positions : np.ndarray (N, 3)
        New positions [nm] (a copy).
    n_changed : int
        Number of molecules rebuilt.
    """
    top = Prmtop.read(prmtop)
    eq = np.asarray(top.get("BOND_EQUIL_VALUE")) * 0.1
    bonds = {}
    for name in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        b = np.asarray(top.get(name, [])).reshape(-1, 3)
        for i3, j3, t in b:
            bonds[(int(i3) // 3, int(j3) // 3)] = eq[int(t) - 1]
    natom = int(top.pointers["NATOM"])
    mol_of = np.repeat(np.arange(len(top.get("ATOMS_PER_MOLECULE"))), np.asarray(top.get("ATOMS_PER_MOLECULE"), int))
    assert len(mol_of) == natom
    members: dict[int, list[int]] = {}
    for a, m in enumerate(mol_of):
        members.setdefault(int(m), []).append(a)
    out = np.array(pos, float)
    changed = 0
    length: dict[int, dict[tuple[int, int], float]] = {}
    for (i, j), r in bonds.items():
        length.setdefault(int(mol_of[i]), {})[(min(i, j), max(i, j))] = r
    for m, atoms in members.items():
        if len(atoms) != 3 or len(length.get(m, {})) != 3:
            continue
        a0, a1, a2 = sorted(atoms)
        L = length[m]
        r01, r02, r12 = L[(a0, a1)], L[(a0, a2)], L[(a1, a2)]
        p = out[[a0, a1, a2]]
        cur = (np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0]), np.linalg.norm(p[2] - p[1]))
        if max(abs(cur[0] - r01), abs(cur[1] - r02), abs(cur[2] - r12)) <= tol:
            continue
        e1 = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
        e2 = (p[2] - p[0]) / np.linalg.norm(p[2] - p[0])
        phi = np.arccos(np.clip((r01**2 + r02**2 - r12**2) / (2 * r01 * r02), -1.0, 1.0))
        u = (e1 + e2) / np.linalg.norm(e1 + e2)  # bisector of the input keeps the orientation
        w = e1 - np.dot(e1, u) * u
        w /= np.linalg.norm(w)
        out[a1] = p[0] + r01 * (np.cos(phi / 2) * u + np.sin(phi / 2) * w)
        out[a2] = p[0] + r02 * (np.cos(phi / 2) * u - np.sin(phi / 2) * w)
        changed += 1
    return out, changed
