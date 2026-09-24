"""Neighbour lists for MD from JAX-MD.

JAX-MD builds a dense, full neighbour list idx (N, max_neighbours): row i holds every atom within
cutoff + skin of atom i (padding = N), built from wrapped fractional coordinates (cell list in
the unit cube when the box holds at least three cells per side, all pairs otherwise).  The metric
is the exact minimum image of box.py (JAX-MD's fractional rounding is not exact in skewed cells
such as the truncated octahedron).  JAX-MD rebuilds the list when an atom has moved more than
skin/2.  The list is built in float32 (fractional coordinates in [0, 1), float32 box): a
neighbour list only needs distances to within a fraction of the skin, and float64 is up to ~100x
slower on consumer and workstation GPUs.  The force field compacts each row to the pairs inside
the cutoff every step and works on the rows directly (per-atom sums, no scatter-adds).

Overflows (dense list, cell list) are flagged, never silent: Simulation reallocates and repeats
the block of steps."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._jaxmd import partition
from .box import check_box, min_image, wrap_fractional


class Neighbors:
    def __init__(self, n_atoms: int, H, cutoff: float, skin: float, capacity_multiplier: float = 1.25):
        self.n = int(n_atoms)
        self.cutoff, self.skin = float(cutoff), float(skin)
        margin = 1e-3                                 # nm: float32 rounding of list distances
        self.rlist = self.cutoff + self.skin + margin  # JAX-MD lists pairs within r_cutoff + dr_threshold
        check_box(H, self.rlist)
        H0 = jnp.asarray(H, jnp.float32)

        def displacement(Ra, Rb, box=None, **_):
            Hc = H0 if box is None else jnp.transpose(box)
            return min_image((Ra - Rb) @ Hc, Hc)

        self._fn = partition.neighbor_list(displacement, H0.T, self.cutoff + margin, dr_threshold=self.skin,
                                           capacity_multiplier=float(capacity_multiplier), fractional_coordinates=True,
                                           format=partition.NeighborListFormat.Dense, mask_self=True)

    def allocate(self, pos, H):
        """Host-side allocation (not jittable)."""
        check_box(np.asarray(H), self.rlist)
        H = jnp.asarray(H, jnp.float64)
        u = wrap_fractional(jnp.asarray(pos, jnp.float64), H).astype(jnp.float32)
        nb = self._fn.allocate(u, box=H.T.astype(jnp.float32))
        if nb.cell_size is not None:          # static field: must be hashable for jit caching
            nb = nb.set(cell_size=float(np.asarray(nb.cell_size).reshape(-1)[0]))
        return nb

    def update(self, nb, pos, H, force_rebuild=False):
        """Jittable update; `force_rebuild` (traced bool), e.g. after a box change."""
        H = jnp.asarray(H, jnp.float64)
        u = wrap_fractional(pos, H).astype(jnp.float32)
        ref = jnp.where(force_rebuild, nb.reference_position + 0.5, nb.reference_position)
        return nb.set(reference_position=ref).update(u, box=H.T.astype(jnp.float32))

    @staticmethod
    def failed(nb) -> bool:
        """Host check after a block of steps: dense-list or cell-list overflow, or cells too small
        after a box change.  (JAX-MD 0.2.29 raises its MALFORMED_BOX bit for every *valid* box
        matrix -- inverted predicate in partition.py -- so that bit is ignored; boxes are checked
        by box.check_box instead.)"""
        PEC = partition.PartitionErrorCode
        bad = PEC.NEIGHBOR_LIST_OVERFLOW | PEC.CELL_LIST_OVERFLOW | PEC.CELL_SIZE_TOO_SMALL
        return bool(int(nb.error.code) & int(bad))
