"""Molecular systems: atoms, molecules, and their pGM parameters.

A `Molecule` is a template (elements, atom types, covalent-dipole topology, pGM
parameters).  A `System` is an ordered list of molecules; it flattens everything into
static index arrays that the energy code closes over.  Coordinates are NOT part of the
system: they are passed separately as (n, 3) or (B, n, 3) arrays, so one System serves a
whole batch of geometries (all 8 points of an S66x8 curve, all 45k water trimers, ...).

Units everywhere: nm, e, e*nm, nm^3, kJ/mol.

pGM conventions (Wei et al. JCP 2020; Wang et al. JCTC 2019):
  * each atom: Gaussian charge q, Gaussian radius R (the prmtop POL_GAUSS_RADII value),
    isotropic polarizability alpha;
  * permanent dipoles are covalent dipoles along covalent basis vectors (CBV):
    p_i = sum_k c_ik * unit(r_j(k) - r_i), with j(k) a bonded or virtually bonded atom;
  * all pairs interact, no 1-2/1-3 masking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Molecule:
    name: str
    elements: list[str]                       # e.g. ["O", "H", "H"]
    types: list[str]                          # force-field atom types, e.g. ["ow", "hw", "hw"]
    q: np.ndarray                             # (m,) e
    radius: np.ndarray                        # (m,) nm, pGM Gaussian radius
    alpha: np.ndarray                         # (m,) nm^3
    cov: list[tuple[int, int, float]] = field(default_factory=list)   # (i, j, c) local indices, c in e*nm
    extra: dict = field(default_factory=dict)  # per-atom arrays for later channels (C6, widths, ...)

    @property
    def n(self) -> int:
        return len(self.elements)

    @property
    def charge(self) -> float:
        return float(np.sum(self.q))


class System:
    """Flattened, static description of a set of molecules (topology only)."""

    def __init__(self, molecules: list[Molecule]):
        self.molecules = molecules
        offs = np.cumsum([0] + [m.n for m in molecules])
        self.offsets = offs
        self.n = int(offs[-1])
        self.nmol = len(molecules)
        self.mol = np.concatenate([np.full(m.n, k, dtype=np.int32) for k, m in enumerate(molecules)])
        self.elements = [e for m in molecules for e in m.elements]
        self.types = [t for m in molecules for t in m.types]
        self.q = np.concatenate([m.q for m in molecules]).astype(float)
        self.radius = np.concatenate([m.radius for m in molecules]).astype(float)
        self.alpha = np.concatenate([m.alpha for m in molecules]).astype(float)
        ci, cj, cc = [], [], []
        for k, m in enumerate(molecules):
            for i, j, c in m.cov:
                ci.append(offs[k] + i); cj.append(offs[k] + j); cc.append(c)
        self.cov_i = np.array(ci, dtype=np.int32)
        self.cov_j = np.array(cj, dtype=np.int32)
        self.cov_c = np.array(cc, dtype=float)
        ii, jj = np.triu_indices(self.n, k=1)
        self.pair_i, self.pair_j = ii, jj
        self.pair_inter = (self.mol[ii] != self.mol[jj])
        extra_keys = set().union(*[m.extra.keys() for m in molecules]) if molecules else set()
        self.extra = {k: np.concatenate([np.asarray(m.extra[k], float) for m in molecules]) for k in extra_keys
                      if all(k in m.extra for m in molecules)}

    def atom_slice(self, k: int) -> slice:
        return slice(int(self.offsets[k]), int(self.offsets[k + 1]))

    def sub(self, mols: tuple[int, ...]) -> tuple["System", np.ndarray]:
        """Subsystem of the given molecules and the atom index map into this system."""
        idx = np.concatenate([np.arange(self.offsets[k], self.offsets[k + 1]) for k in mols])
        return System([self.molecules[k] for k in mols]), idx

    def fingerprint(self) -> str:
        """Hash of topology and all parameters (safe cache key for compiled energy functions)."""
        import hashlib
        h = hashlib.sha1()
        for a in (self.q, self.radius, self.alpha, self.cov_i, self.cov_j, self.cov_c, self.mol):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update("|".join(self.elements).encode())
        for k in sorted(self.extra):
            h.update(k.encode()); h.update(np.ascontiguousarray(self.extra[k]).tobytes())
        return h.hexdigest()

    def signature(self) -> str:
        return "+".join(m.name for m in self.molecules)

    def __repr__(self) -> str:
        return f"System({self.signature()}, n={self.n})"
