"""Neighbour lists for MD from JAX-MD, as candidate rows for the force field.

Both list types give `candidates(nb, centers, H)` -> (rows, overflow): an (N, C) array whose row i
holds atoms that may lie within the cutoff of atom i (padding = N).  Rows are supersets; the force
field masks the pairs outside the cutoff.

  MoleculeNeighbors (default for rigid molecules): a JAX-MD list of molecular centres of mass
    within cutoff + 2 r_max + skin (r_max: largest atom-to-centre distance).  Every step each atom
    keeps the neighbouring molecules whose centre is within cutoff + r_max of the atom (the only
    ones that can have an atom inside its cutoff), expanded to their atoms.  Rotations never invalidate it and there are
    ~n_atoms_per_mol^2 fewer centre pairs to search, so rebuilds are rare and cheap (the atom list
    of water is rebuilt every ~10 steps because of the hydrogens' rotation).
  AtomNeighbors: a JAX-MD list of atoms within cutoff + skin (boxes too small for the centre list).

Lists are built by JAX-MD from wrapped fractional coordinates in float32 (cell list in the unit
cube when the box holds at least three cells per side, all pairs otherwise; float64 is up to
~100x slower on workstation GPUs, and a list only needs distances to a fraction of the skin), with
the exact minimum image of box.py (JAX-MD's fractional rounding is not exact in skewed cells such
as the truncated octahedron).  JAX-MD rebuilds a list when a point has moved more than skin/2.

Overflows are flagged, never silent: Simulation reallocates and repeats the block of steps."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._jaxmd import partition
from .box import check_box, max_cutoff, min_image, wrap_fractional

_MARGIN = 1e-3  # nm: float32 rounding of list distances


def _jaxmd_list(H, r_cutoff, skin, capacity_multiplier):
    H0 = jnp.asarray(H, jnp.float32)

    def displacement(Ra, Rb, box=None, **_):
        Hc = H0 if box is None else jnp.transpose(box)
        return min_image(jnp.matmul(Ra - Rb, Hc, precision=jax.lax.Precision.HIGHEST), Hc)

    return partition.neighbor_list(
        displacement,
        H0.T,
        r_cutoff,
        dr_threshold=skin,
        capacity_multiplier=float(capacity_multiplier),
        fractional_coordinates=True,
        format=partition.NeighborListFormat.Dense,
        mask_self=True,
    )


def _failed(nb) -> bool:
    """Dense-list or cell-list overflow, or cells too small after a box change.  (JAX-MD 0.2.29
    raises its MALFORMED_BOX bit for every *valid* box matrix -- inverted predicate in partition.py --
    so that bit is ignored; boxes are checked by box.check_box instead.)"""
    PEC = partition.PartitionErrorCode
    bad = PEC.NEIGHBOR_LIST_OVERFLOW | PEC.CELL_LIST_OVERFLOW | PEC.CELL_SIZE_TOO_SMALL
    return bool(int(nb.error.code) & int(bad))


def _allocate(fn, x, H):
    H = jnp.asarray(H, jnp.float64)
    u = wrap_fractional(jnp.asarray(x, jnp.float64), H).astype(jnp.float32)
    nb = fn.allocate(u, box=H.T.astype(jnp.float32))
    if nb.cell_size is not None:  # static field: must be hashable for jit caching
        nb = nb.set(cell_size=float(np.asarray(nb.cell_size).reshape(-1)[0]))
    return nb


def _update(nb, x, H, force_rebuild):
    H = jnp.asarray(H, jnp.float64)
    u = wrap_fractional(x, H).astype(jnp.float32)
    ref = jnp.where(force_rebuild, nb.reference_position + 0.5, nb.reference_position)
    return nb.set(reference_position=ref).update(u, box=H.T.astype(jnp.float32))


class AtomNeighbors:
    kind = "atom"

    def __init__(self, n_atoms: int, H, cutoff: float, skin: float, capacity_multiplier: float = 1.25):
        self.n = int(n_atoms)
        self.cutoff, self.skin = float(cutoff), float(skin)
        self.rlist = self.cutoff + self.skin + _MARGIN  # JAX-MD lists pairs within r_cutoff + dr_threshold
        check_box(H, self.rlist)
        self._fn = _jaxmd_list(H, self.cutoff + _MARGIN, self.skin, capacity_multiplier)

    def allocate(self, pos, centers, H):
        check_box(np.asarray(H), self.rlist)
        return _allocate(self._fn, pos, H)

    def update(self, nb, pos, centers, H, force_rebuild=False):
        return _update(nb, pos, H, force_rebuild)

    def candidates(self, nb, centers=None, H=None, pos=None):
        return nb.idx, jnp.zeros((), bool)

    def size(self, nb, centers, H, pos, factor=1.2):
        return None

    failed = staticmethod(_failed)


class MoleculeNeighbors:
    kind = "molecule"

    def __init__(self, mol, n_mol: int, r_max: float, H, cutoff: float, skin: float, capacity_multiplier: float = 1.25):
        mol = np.asarray(mol)
        self.n, self.nmol = len(mol), int(n_mol)
        self.cutoff, self.skin, self.r_max = float(cutoff), float(skin), float(r_max)
        self.rlist = self.cutoff + 2.0 * self.r_max + self.skin + _MARGIN
        check_box(H, self.rlist)
        counts = np.bincount(mol, minlength=self.nmol)
        nmax = int(counts.max())
        table = np.full((self.nmol + 1, nmax), self.n, np.int32)  # last row: padding molecule
        fill = np.zeros(self.nmol, int)
        for a, m in enumerate(mol):
            table[m, fill[m]] = a
            fill[m] += 1
        self.table = jnp.asarray(table)
        self.mol = jnp.asarray(mol)
        self.ratom = self.cutoff + self.r_max + _MARGIN  # atom-to-centre distance kept
        self._fn = _jaxmd_list(H, self.cutoff + 2.0 * self.r_max + _MARGIN, self.skin, capacity_multiplier)
        self.cap = None  # molecules kept per atom (set by size())

    @staticmethod
    def fits(H, cutoff, skin, r_max) -> bool:
        return cutoff + 2.0 * r_max + skin + _MARGIN <= max_cutoff(H)

    def allocate(self, pos, centers, H):
        check_box(np.asarray(H), self.rlist)
        return _allocate(self._fn, centers, H)

    def update(self, nb, pos, centers, H, force_rebuild=False):
        return _update(nb, centers, H, force_rebuild)

    def _within(self, nb, centers, H, pos):
        """(N, Mm) molecule candidates of each atom and the mask of those within cutoff + r_max."""
        idx = nb.idx[self.mol]  # neighbour molecules of each atom's molecule
        valid = idx < self.nmol
        k = jnp.where(valid, idx, 0)
        Hc = jnp.asarray(H, jnp.float32)
        d = min_image(pos.astype(jnp.float32)[:, None, :] - centers.astype(jnp.float32)[k], Hc)
        return idx, valid & (jnp.sum(d * d, -1) < self.ratom**2)

    def size(self, nb, centers, H, pos, factor=1.2):
        """Molecules kept per atom: 20 % above the current maximum, multiple of 4."""
        cmax = int(jnp.max(jnp.sum(self._within(nb, centers, H, pos)[1], axis=1)))
        self.cap = min(int(np.ceil((cmax * factor + 4) / 4.0) * 4), int(nb.idx.shape[1]))
        return self.cap

    def candidates(self, nb, centers, H, pos):
        """Atoms of the molecules whose centre is within cutoff + r_max of each atom:
        (N, cap * max_atoms_per_molecule), and an overflow flag (more such molecules than cap)."""
        idx, w = self._within(nb, centers, H, pos)
        slot = jnp.cumsum(w, axis=1) - 1
        count = slot[:, -1] + 1
        cap = self.cap
        tgt = jnp.where(w & (slot < cap), slot, cap)
        rows = jnp.broadcast_to(jnp.arange(self.n)[:, None], idx.shape)
        kept = jnp.full((self.n, cap + 1), self.nmol, idx.dtype).at[rows, tgt].set(idx)[:, :cap]
        return self.table[kept].reshape(self.n, -1), jnp.max(count) > cap

    failed = staticmethod(_failed)


# backwards-compatible name
Neighbors = AtomNeighbors
