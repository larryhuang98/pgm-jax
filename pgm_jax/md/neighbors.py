"""Neighbour lists for MD from JAX-MD, as candidate rows for the force field.

Contents: `AtomNeighbors` and `MoleculeNeighbors`, two list types with the same interface
(`allocate`, `update`, `candidates`, `size`, `failed`), and the JAX-MD glue (`_jaxmd_list`,
`_allocate`, `_update`, `_failed`).

Both list types give `candidates(nb, centers, H, pos)` -> (rows, overflow): an (N, C) array whose
row i holds atoms that may lie within the cutoff of atom i (padding = N).  Rows are supersets;
the force field masks the pairs outside the cutoff.

  MoleculeNeighbors (default for rigid molecules): a JAX-MD list of molecular centres of mass
    within cutoff + 2 r_max + skin (r_max: largest atom-to-centre distance).  Every step each atom
    keeps the neighbouring molecules whose centre is within cutoff + r_max of the atom (the only
    ones that can have an atom inside its cutoff), expanded to their atoms.  Rotations never
    invalidate it and there are ~n_atoms_per_mol^2 fewer centre pairs to search, so rebuilds are
    rare and cheap (the atom list of water is rebuilt every ~10 steps because of the hydrogens'
    rotation).  The flexible and path-integral engines use it with the neighbour-list groups of
    md/topology.py (and their centres) in place of molecules.
  AtomNeighbors: a JAX-MD list of atoms within cutoff + skin (boxes too small for the centre list).

Lists are built by JAX-MD from wrapped fractional coordinates in float32 (cell list in the unit
cube when the box holds at least three cells per side, all pairs otherwise; float64 is up to
~100x slower on workstation GPUs, and a list only needs distances to a fraction of the skin), with
the exact minimum image of box.py (JAX-MD's fractional rounding is not exact in skewed cells such
as the truncated octahedron).  JAX-MD rebuilds a list when a point has moved more than skin/2.
Each list distance gets a margin of 1e-3 nm (`_MARGIN`) for the float32 rounding.

Overflows are flagged, never silent: Simulation reallocates and repeats the block of steps.
`allocate`, `size` and `failed` are host-side (not traceable); `update` and `candidates` are
traced inside the compiled steps.

Units: nm.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ._jaxmd import partition
from .box import check_box, max_cutoff, min_image, wrap_fractional

_MARGIN = 1e-3  # nm: float32 rounding of list distances


def _jaxmd_list(H: ArrayLike, r_cutoff: float, skin: float, capacity_multiplier: float) -> partition.NeighborListFns:
    """Return the JAX-MD dense neighbour-list functions for fractional coordinates.

    Parameters
    ----------
    H : ArrayLike (3, 3)
        Box, lattice vectors as rows [nm] (the default box of the displacement; the list calls pass
        the current box).
    r_cutoff : float
        List cutoff without the skin [nm]; JAX-MD lists pairs within r_cutoff + skin.
    skin : float
        Skin [nm] (JAX-MD's dr_threshold; rebuild after a move of skin/2).
    capacity_multiplier : float
        Spare capacity of the dense list (and the cell list) on allocation.

    Returns
    -------
    partition.NeighborListFns
        allocate / update functions of a float32 list in the unit cube (self pairs masked).
    """
    H0 = jnp.asarray(H, jnp.float32)

    def displacement(Ra: jax.Array, Rb: jax.Array, box: jax.Array | None = None, **_) -> jax.Array:
        # fractional Ra - Rb -> Cartesian (rows of H), then the exact minimum image of box.py
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


def _failed(nb: partition.NeighborList) -> bool:
    """Return whether a list overflowed or its cells became too small after a box change (host).

    Checks the dense-list overflow, cell-list overflow and cell-size-too-small bits of
    `nb.error`.  JAX-MD 0.2.29 raises its MALFORMED_BOX bit for every *valid* box matrix (inverted
    predicate in partition.py), so that bit is ignored; boxes are checked by box.check_box instead.

    Parameters
    ----------
    nb : partition.NeighborList
        The list (concrete, not traced: the error code is read on the host).

    Returns
    -------
    bool
        True if the list must be reallocated.
    """
    PEC = partition.PartitionErrorCode
    bad = PEC.NEIGHBOR_LIST_OVERFLOW | PEC.CELL_LIST_OVERFLOW | PEC.CELL_SIZE_TOO_SMALL
    return bool(int(nb.error.code) & int(bad))


def _allocate(fn: partition.NeighborListFns, x: ArrayLike, H: ArrayLike) -> partition.NeighborList:
    """Allocate a list for points x in box H (host).

    Parameters
    ----------
    fn : partition.NeighborListFns
        From `_jaxmd_list`.
    x : ArrayLike (n, 3)
        Points (atoms or centres) [nm].
    H : ArrayLike (3, 3)
        Box, lattice vectors as rows [nm].

    Returns
    -------
    partition.NeighborList
        The list, built from wrapped float32 fractional coordinates; its cell size is stored as a
        Python float (a static field must be hashable for the jit cache).
    """
    H = jnp.asarray(H, jnp.float64)
    u = wrap_fractional(jnp.asarray(x, jnp.float64), H).astype(jnp.float32)
    nb = fn.allocate(u, box=H.T.astype(jnp.float32))
    if nb.cell_size is not None:  # static field: must be hashable for jit caching
        nb = nb.set(cell_size=float(np.asarray(nb.cell_size).reshape(-1)[0]))
    return nb


def _update(
    nb: partition.NeighborList, x: jax.Array, H: ArrayLike, force_rebuild: bool | jax.Array
) -> partition.NeighborList:
    """Update a list for points x in box H (traceable).

    Parameters
    ----------
    nb : partition.NeighborList
        The list.
    x : jax.Array (n, 3)
        Points [nm].
    H : ArrayLike (3, 3)
        Box, lattice vectors as rows [nm].
    force_rebuild : bool or jax.Array () bool
        Rebuild even if no point moved more than skin/2 (after a volume move).

    Returns
    -------
    partition.NeighborList
        The updated list (JAX-MD rebuilds it if needed; overflows are flagged in its error code).
    """
    H = jnp.asarray(H, jnp.float64)
    u = wrap_fractional(x, H).astype(jnp.float32)
    # a reference shifted by half the (unit) cell exceeds any skin: JAX-MD then rebuilds the list
    ref = jnp.where(force_rebuild, nb.reference_position + 0.5, nb.reference_position)
    return nb.set(reference_position=ref).update(u, box=H.T.astype(jnp.float32))


class AtomNeighbors:
    """Neighbour list of atoms within cutoff + skin (module docstring).

    Mutable host object holding the JAX-MD list functions; the list itself (a
    partition.NeighborList) is part of the integrator state.

    Attributes
    ----------
    kind : str
        "atom".
    n : int
        Number of atoms N.
    cutoff : float
        Pair cutoff [nm].
    skin : float
        Skin [nm].
    rlist : float
        Largest listed distance, cutoff + skin + margin [nm] (checked against the box).
    """

    kind = "atom"

    def __init__(
        self, n_atoms: int, H: ArrayLike, cutoff: float, skin: float, capacity_multiplier: float = 1.25
    ) -> None:
        """Set up the list functions and check the box.

        Parameters
        ----------
        n_atoms : int
            Number of atoms N.
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].
        cutoff : float
            Pair cutoff [nm].
        skin : float
            Skin [nm].
        capacity_multiplier : float
            Spare capacity of the dense list on allocation.

        Raises
        ------
        ValueError
            From box.check_box: H not reduced, or cutoff + skin + margin above half the box height.
        """
        self.n = int(n_atoms)
        self.cutoff, self.skin = float(cutoff), float(skin)
        self.rlist = self.cutoff + self.skin + _MARGIN  # JAX-MD lists pairs within r_cutoff + dr_threshold
        check_box(H, self.rlist)
        self._fn = _jaxmd_list(H, self.cutoff + _MARGIN, self.skin, capacity_multiplier)

    def allocate(self, pos: ArrayLike, centers: ArrayLike | None, H: ArrayLike) -> partition.NeighborList:
        """Allocate the list for positions `pos` in box H (host; `centers` is not used).

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Positions [nm].
        centers : ArrayLike or None
            Not used (same interface as MoleculeNeighbors).
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].

        Returns
        -------
        partition.NeighborList

        Raises
        ------
        ValueError
            From box.check_box (box not reduced or too small).
        """
        check_box(np.asarray(H), self.rlist)
        return _allocate(self._fn, pos, H)

    def update(
        self,
        nb: partition.NeighborList,
        pos: jax.Array,
        centers: jax.Array | None,
        H: ArrayLike,
        force_rebuild: bool | jax.Array = False,
    ) -> partition.NeighborList:
        """Update the list for positions `pos` (traceable; `centers` is not used).

        Parameters
        ----------
        nb : partition.NeighborList
            The list.
        pos : jax.Array (N, 3)
            Positions [nm].
        centers : jax.Array or None
            Not used.
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].
        force_rebuild : bool or jax.Array () bool
            Rebuild unconditionally.

        Returns
        -------
        partition.NeighborList
        """
        return _update(nb, pos, H, force_rebuild)

    def candidates(
        self,
        nb: partition.NeighborList,
        centers: jax.Array | None = None,
        H: ArrayLike | None = None,
        pos: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return the candidate rows (the dense list itself) and a False overflow flag.

        Parameters
        ----------
        nb : partition.NeighborList
            The list.
        centers, H, pos : optional
            Not used (same interface as MoleculeNeighbors).

        Returns
        -------
        rows : jax.Array (N, C) int32
            Candidate partners of every atom (padding N).
        overflow : jax.Array () bool
            Always False (list overflows are in `nb.error`).
        """
        return nb.idx, jnp.zeros((), bool)

    def size(
        self, nb: partition.NeighborList, centers: jax.Array | None, H: ArrayLike, pos: jax.Array, factor: float = 1.2
    ) -> None:
        """Return None: an atom list has no per-atom capacity to size (same interface as MoleculeNeighbors)."""
        return None

    failed = staticmethod(_failed)


class MoleculeNeighbors:
    """Neighbour list of molecular centres, expanded to atom rows every step (module docstring).

    The "molecules" are any partition of the atoms with centres: molecules for the rigid engine,
    neighbour-list groups (md/topology.py) for the flexible and path-integral engines.  Mutable
    host object: `size` sets the per-atom capacity `cap`, which fixes the shape of the candidate
    rows (a change recompiles the step).

    Attributes
    ----------
    kind : str
        "molecule".
    n : int
        Number of atoms N.
    nmol : int
        Number of molecules (groups) M.
    cutoff, skin, r_max : float
        Pair cutoff, skin and largest atom-to-centre distance [nm].
    rlist : float
        Largest listed centre distance, cutoff + 2 r_max + skin + margin [nm].
    ratom : float
        Atom-to-centre distance kept by `candidates`, cutoff + r_max + margin [nm].
    table : jax.Array (M + 1, nmax) int32
        Atoms of every molecule (padding N); the last row is the padding molecule.
    mol : jax.Array (N,) int
        Molecule of every atom.
    cap : int or None
        Molecules kept per atom (None until `size` is called).
    """

    kind = "molecule"

    def __init__(
        self,
        mol: ArrayLike,
        n_mol: int,
        r_max: float,
        H: ArrayLike,
        cutoff: float,
        skin: float,
        capacity_multiplier: float = 1.25,
    ) -> None:
        """Set up the atom table and the centre-list functions and check the box.

        Parameters
        ----------
        mol : ArrayLike (N,) int
            Molecule (or group) of every atom, 0 .. n_mol-1.
        n_mol : int
            Number of molecules M.
        r_max : float
            Largest distance of an atom from its molecule's centre [nm] (with a margin for flexible
            molecules; the caller's choice).
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].
        cutoff : float
            Pair cutoff [nm].
        skin : float
            Skin of the centre list [nm].
        capacity_multiplier : float
            Spare capacity of the dense list on allocation.

        Raises
        ------
        ValueError
            From box.check_box: H not reduced, or `rlist` above half the box height.
        """
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
    def fits(H: ArrayLike, cutoff: float, skin: float, r_max: float) -> bool:
        """Return whether a centre list with these settings fits the box (cutoff + 2 r_max + skin + margin).

        Parameters
        ----------
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].
        cutoff : float
            Pair cutoff [nm].
        skin : float
            Skin [nm].
        r_max : float
            Largest atom-to-centre distance [nm].

        Returns
        -------
        bool
            True if the list distance is at most half the smallest box height (box.max_cutoff).
        """
        return cutoff + 2.0 * r_max + skin + _MARGIN <= max_cutoff(H)

    def allocate(self, pos: ArrayLike, centers: ArrayLike, H: ArrayLike) -> partition.NeighborList:
        """Allocate the centre list for `centers` in box H (host; `pos` is not used).

        Parameters
        ----------
        pos : ArrayLike
            Not used (same interface as AtomNeighbors).
        centers : ArrayLike (M, 3)
            Molecule (group) centres [nm].
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].

        Returns
        -------
        partition.NeighborList

        Raises
        ------
        ValueError
            From box.check_box (box not reduced or too small).
        """
        check_box(np.asarray(H), self.rlist)
        return _allocate(self._fn, centers, H)

    def update(
        self,
        nb: partition.NeighborList,
        pos: jax.Array,
        centers: jax.Array,
        H: ArrayLike,
        force_rebuild: bool | jax.Array = False,
    ) -> partition.NeighborList:
        """Update the centre list for `centers` (traceable; `pos` is not used).

        Parameters
        ----------
        nb : partition.NeighborList
            The centre list.
        pos : jax.Array
            Not used.
        centers : jax.Array (M, 3)
            Molecule (group) centres [nm].
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].
        force_rebuild : bool or jax.Array () bool
            Rebuild unconditionally.

        Returns
        -------
        partition.NeighborList
        """
        return _update(nb, centers, H, force_rebuild)

    def _within(
        self, nb: partition.NeighborList, centers: jax.Array, H: ArrayLike, pos: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return the molecule candidates of each atom and the mask of those within cutoff + r_max.

        Parameters
        ----------
        nb : partition.NeighborList
            The centre list.
        centers : jax.Array (M, 3)
            Centres [nm].
        H : ArrayLike (3, 3)
            Box [nm].
        pos : jax.Array (N, 3)
            Positions [nm].

        Returns
        -------
        idx : jax.Array (N, Mm) int
            Neighbour molecules of each atom's molecule (padding M).
        within : jax.Array (N, Mm) bool
            Real candidates whose centre is within `ratom` of the atom (float32 minimum image).
        """
        idx = nb.idx[self.mol]  # neighbour molecules of each atom's molecule
        valid = idx < self.nmol
        k = jnp.where(valid, idx, 0)
        Hc = jnp.asarray(H, jnp.float32)
        d = min_image(pos.astype(jnp.float32)[:, None, :] - centers.astype(jnp.float32)[k], Hc)
        return idx, valid & (jnp.sum(d * d, -1) < self.ratom**2)

    def size(
        self, nb: partition.NeighborList, centers: jax.Array, H: ArrayLike, pos: jax.Array, factor: float = 1.2
    ) -> int:
        """Set and return `cap`, the molecules kept per atom (host).

        Parameters
        ----------
        nb : partition.NeighborList
            The centre list (concrete).
        centers : jax.Array (M, 3)
            Centres [nm].
        H : ArrayLike (3, 3)
            Box [nm].
        pos : jax.Array (N, 3)
            Positions [nm].
        factor : float
            Headroom over the current maximum count.

        Returns
        -------
        int
            ceil((cmax factor + 4) / 4) * 4, a multiple of 4 at least 4 above cmax factor, capped at
            the width of the centre list.
        """
        cmax = int(jnp.max(jnp.sum(self._within(nb, centers, H, pos)[1], axis=1)))
        self.cap = min(int(np.ceil((cmax * factor + 4) / 4.0) * 4), int(nb.idx.shape[1]))
        return self.cap

    def candidates(
        self, nb: partition.NeighborList, centers: jax.Array, H: ArrayLike, pos: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return the atoms of the molecules whose centre is within cutoff + r_max of each atom.

        Parameters
        ----------
        nb : partition.NeighborList
            The centre list.
        centers : jax.Array (M, 3)
            Centres [nm].
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].
        pos : jax.Array (N, 3)
            Positions [nm].

        Returns
        -------
        rows : jax.Array (N, cap * nmax) int32
            Candidate atoms of every atom (padding N).  The atom's own molecule is not listed
            (the centre list masks self pairs): its pairs come from the special-pair table
            (md/topology.py), and the force field drops list pairs that are special.
        overflow : jax.Array () bool
            True if an atom has more than `cap` such molecules (the extra ones are dropped: the caller
            must resize and repeat).
        """
        idx, w = self._within(nb, centers, H, pos)
        slot = jnp.cumsum(w, axis=1) - 1  # compaction: position of each kept molecule in its row
        count = slot[:, -1] + 1
        cap = self.cap
        tgt = jnp.where(w & (slot < cap), slot, cap)  # dropped / overflowing ones go to the extra column cap
        rows = jnp.broadcast_to(jnp.arange(self.n)[:, None], idx.shape)
        kept = jnp.full((self.n, cap + 1), self.nmol, idx.dtype).at[rows, tgt].set(idx)[:, :cap]
        return self.table[kept].reshape(self.n, -1), jnp.max(count) > cap

    failed = staticmethod(_failed)
