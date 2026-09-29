"""pGM energies, forces, virials and dipoles for external MD codes: one jitted call per configuration.

External drivers (ASE, i-PI, OpenMM; pgm_jax/interfaces/) hand over positions and a cell and want
the energy and forces back; they integrate the equations of motion themselves.  `PGMEngine` keeps
everything that is expensive to rebuild on the device between calls:

  * the jitted force field of the MD engine (PGMForceField: smooth PME, pair rows, pmemd-pgm's
    induction solver with the mu4 predictor) plus, for flexible molecules, their bonded terms
    (FlexibleTemplate, as FlexibleSimulation), charge flux and the intramolecular van der Waals;
  * the neighbour list (molecular-centre list of the MD engine, or an atom list for small boxes),
    updated inside the jitted call (JAX-MD rebuilds it when something moved more than skin/2);
  * the induced dipoles and their predictor history (InductionState), so successive MD steps start
    the CG from the extrapolated dipoles exactly as the native integrator does.  Several `slots`
    keep separate histories for interleaved configurations (i-PI sends the P beads of a ring polymer
    to one client one after the other): the first calls fill the slots in turn, later calls pick the
    slot whose last configuration is closest (or the slot named by the caller).  A configuration
    far from the slot's last one (a jump larger than `jump` nm, e.g. a new structure) restarts that
    slot's predictor;
  * for batches of close structures that share the cell (ring-polymer beads, i-PI's batched
    requests), compute_batch evaluates them in one vmapped call: P slots with stacked dipole
    histories, one neighbour list of their mean, every structure matched to its own slot.

Per call the host sends positions (float64) and the cell and receives one packed float64 array
(energy terms, forces, flags); induced dipoles, the cell dipole and the virial stay on the device
until asked for.  Overflows of the row or list capacities, a box that changed by more than 10 % in
volume, or an atom further from its list-group centre than the list radius are detected after the
call; the engine then resizes or rebuilds and repeats the call (never silent), as the native driver
does with its blocks.

Positions may come wrapped atom by atom (ASE, OpenMM) or never wrapped (i-PI): inside the call each
molecule is made whole along its bond tree (pointer doubling, O(N log depth)) and whole molecules
are shifted into the primary cell by their centres of mass.  Energies and forces do not depend on
either.  Cells may be any right-handed cell: a general cell (ASE) is rotated to the reduced lower
triangular form of the engine (a along x, b in the xy plane), and forces and virials are rotated
back.

Virial W = dE/d eps (3 x 3, kJ/mol) at the converged dipoles (the energy is variational in them):
  stress="atomic"     every atom scaled with the box (x -> (1 + eps) x), the derivative external
                      codes expect for flexible molecules (ASE stress = W / V, i-PI virial = -W);
                      the default with flexible templates;
  stress="molecular"  molecular centres of mass scaled, molecules translated rigidly (the native
                      pressure; for rigid molecules held by constraints, whose atomic virial would
                      miss the constraint forces); the default for the rigid-molecule model.
Both include, with MDSettings.lj_lrc, the long-range correction's impulse term, exactly as
Simulation.pressure(): W = PGMForceField.strain_derivative (+ the bonded terms for "atomic").

Units: nm, kJ/mol, kJ/mol/nm, e nm, amu (conversions in the driver modules)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..md.box import check_box, inv3, max_cutoff, min_image, reduce_box
from ..md.forcefield import MDSettings, PGMForceField
from ..md.neighbors import AtomNeighbors, MoleculeNeighbors
from ..md.topology import MDTopology
from ..system import System

__all__ = ["PGMEngine", "GasPhaseEngine", "EngineResult", "standard_cell"]

# neighbour-list error bits that make a list invalid (as md/neighbors._failed; JAX-MD's MALFORMED_BOX
# bit is set for valid boxes by JAX-MD 0.2.29 and is ignored there too)
from ..md._jaxmd import partition as _partition  # noqa: E402

_PEC = _partition.PartitionErrorCode
_LIST_ERRORS = int(_PEC.NEIGHBOR_LIST_OVERFLOW | _PEC.CELL_LIST_OVERFLOW | _PEC.CELL_SIZE_TOO_SMALL)


# ----------------------------------------------------------------------------- cells
def standard_cell(cell):
    """(H, Q) for a right-handed cell (rows are lattice vectors, nm): H = reduce_box(cell @ Q) is
    lower triangular and reduced, Q a rotation (Q = I when the cell is already lower triangular).
    Positions map as x -> x @ Q, forces back as f -> f @ Q.T, a virial W -> Q W Q.T."""
    C = np.asarray(cell, float)
    if not np.all(np.isfinite(C)) or abs(np.linalg.det(C)) < 1e-12:
        raise ValueError("the cell must be periodic in three dimensions (non-singular)")
    if np.linalg.det(C) < 0:
        raise ValueError("left-handed cell: reorder the lattice vectors")
    if np.all(np.abs(np.triu(C, 1)) <= 1e-12 * np.abs(C).max()) and np.all(np.diag(C) > 0):
        return reduce_box(np.tril(C)), None
    Qr, Rr = np.linalg.qr(C.T)  # C = Rr^T Qr^T  ->  C Qr = Rr^T (lower triangular)
    s = np.sign(np.diag(Rr))
    Qr = Qr * s[None, :]  # positive diagonal; det(Qr) = +1 for a right-handed cell
    L = C @ Qr
    return reduce_box(np.tril(L)), Qr


def _bond_tree(sys: System, bonds_of=None):
    """Parent of every atom along a spanning tree of its molecule's bond graph (roots: the first atom
    of each molecule; atoms not connected to it hang from the root) and the tree depth."""
    N = sys.n
    parent = np.arange(N)
    depth = 0
    for k, m in enumerate(sys.molecules):
        off = int(sys.offsets[k])
        bonds = bonds_of(k) if bonds_of is not None else list(getattr(m, "bonds", []) or [])
        nbr = [[] for _ in range(m.n)]
        for i, j in bonds:
            nbr[int(i)].append(int(j))
            nbr[int(j)].append(int(i))
        seen = np.zeros(m.n, bool)
        d = np.zeros(m.n, int)
        for root in range(m.n):  # BFS from atom 0, then any unreached atom
            if seen[root]:
                continue
            seen[root] = True
            if root != 0:
                parent[off + root] = off  # disconnected: hang from the molecule's first atom
                d[root] = 1
            queue = [root]
            while queue:
                a = queue.pop(0)
                for b in nbr[a]:
                    if not seen[b]:
                        seen[b] = True
                        parent[off + b] = off + a
                        d[b] = d[a] + 1
                        queue.append(b)
        depth = max(depth, int(d.max()) if m.n else 0)
    return parent, depth


def match_previous(X, prev):
    """perm with prev[perm[k]] the previous structure closest to X[k] (a one-to-one assignment
    minimising the summed squared displacements: i-PI does not keep the order of the beads in its
    batches).  X: (n, N, 3), prev: (P, N, 3), n <= P."""
    B = X.shape[0]
    a, b = X.reshape(B, -1), prev.reshape(prev.shape[0], -1)
    C = np.sum(a * a, 1)[:, None] + np.sum(b * b, 1)[None, :] - 2.0 * a @ b.T
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(C)
        perm = np.empty(B, int)
        perm[rows] = cols
        return perm
    except ImportError:  # greedy
        perm, used = np.full(B, -1), set()
        for k in np.argsort(C.min(1)):
            j = next(int(j) for j in np.argsort(C[k]) if int(j) not in used)
            perm[k] = j
            used.add(j)
        return perm


@dataclass
class EngineResult:
    """One evaluation, host arrays in engine units (kJ/mol, kJ/mol/nm), in the caller's frame."""

    energy: float
    forces: np.ndarray
    terms: dict  # elec, vdw, bonded (kJ/mol)
    iterations: int  # CG iterations of the dipole solve
    _dev: dict = field(default_factory=dict, repr=False)  # device arrays: mu, dipole parts, virial
    _Q: np.ndarray | None = None

    def _rot_vec(self, v):
        v = np.asarray(v, float)
        return v if self._Q is None else v @ self._Q.T

    @property
    def induced_dipoles(self) -> np.ndarray:
        """(N, 3) e nm, the converged induced dipoles."""
        if "mu" not in self._dev:
            self._dev["mu"] = self._dev.pop("mu_fn")()
        return self._rot_vec(self._dev["mu"])

    @property
    def dipole_components(self) -> np.ndarray:
        """(3, 3) e nm: rows M_q (charges, molecules whole, about their centres of mass), M_perm
        (covalent dipoles), M_ind (induced dipoles); md/dipoles.py conventions.  Computed in the
        engine call (engine.with_dipole) or on first access."""
        if "dip" not in self._dev:
            self._dev["dip"] = self._dev.pop("dip_fn")()
        return self._rot_vec(self._dev["dip"])

    @property
    def dipole(self) -> np.ndarray:
        """Total dipole of the cell (e nm), M_q + M_perm + M_ind."""
        return self.dipole_components.sum(0)

    @property
    def virial(self) -> np.ndarray | None:
        """dE/d eps (3, 3) kJ/mol, or None when not computed."""
        if "W" not in self._dev:
            return None
        W = np.asarray(self._dev["W"], float)
        return W if self._Q is None else self._Q @ W @ self._Q.T


class _Slot:
    def __init__(self):
        self.x = None  # last positions (host, standard frame)
        self.ind = None  # InductionState (device)
        self.nbr = None  # neighbour list (device)


class PGMEngine:
    """Periodic pGM (+ Lennard-Jones / GVDW, + bonded terms of flexible templates) for external
    drivers.  Build it like a native simulation:

        eng = PGMEngine(system, pos_nm, H_nm, MDSettings(...))                # rigid-molecule model
        eng = PGMEngine(system, pos_nm, H_nm, settings, templates=[tpl] * n)  # flexible molecules
        eng = PGMEngine.from_amber("water.prmtop", "water.rst7", settings=...)
        res = eng.compute(pos_nm, cell_nm, virial=True)     # EngineResult (kJ/mol, kJ/mol/nm)

    templates=None gives the model of the rigid-molecule engine (Simulation): pGM with every pair,
    no intramolecular van der Waals, no bonded terms; the external code must then hold the
    molecules rigid (ASE FixBondLengths, OpenMM constraints).  With templates (FlexibleTemplate /
    RigidTemplate, one per molecule) the model is FlexibleSimulation's.  params: parameter pytree
    (None: the system's values).  Virtual sites and alchemical regions are not supported here."""

    def __init__(
        self,
        sys: System,
        pos,
        H,
        settings: MDSettings = MDSettings(),
        templates=None,
        params=None,
        stress: str | None = None,
        slots: int = 1,
        r_margin: float = 0.05,
        neighbor_list: str = "auto",
        jump: float = 0.05,
        restraints=None,
        bead_margin: float = 0.08,
    ):
        from ..md.vsites import VirtualSites

        if VirtualSites.of(sys) is not None:
            raise NotImplementedError("virtual sites are not supported by the external-code interfaces yet")
        if getattr(settings, "iel", "none") != "none":
            raise NotImplementedError(
                "extended-Lagrangian dipoles (MDSettings.iel) need the native integrator's "
                "sequence of steps; external codes call the engine with SCF dipoles (iel='none')"
            )
        if stress is None:  # rigid-molecule model: molecular virial (as the native pressure)
            stress = "molecular" if templates is None else "atomic"
        if stress not in ("atomic", "molecular"):
            raise ValueError("stress must be 'atomic' or 'molecular'")
        H0, Q = standard_cell(H)
        pos = np.asarray(pos, float) if Q is None else np.asarray(pos, float) @ Q
        self.sys, self.settings, self.params = sys, settings, params
        self.stress_mode, self.jump, self.bead_margin = stress, float(jump), float(bead_margin)
        self.with_dipole = False  # cell dipole inside every call (i-PI extras); else on first access
        self.n = sys.n
        self.masses = np.asarray(sys.masses, float)
        self.flex = None
        if templates is None:
            self.topology = MDTopology.rigid(sys)
            self.ff = PGMForceField(sys, H0, settings, topology=self.topology)
            bonds_of = None
        else:
            from ..md.flexible import FlexibleMolecules
            from ..md.flux import ChargeFlux

            templates = list(templates)
            uniq = {id(t): t for t in templates}.values()
            for tpl in uniq:
                tpl.check_settings(settings)
            rules = {id(t): t.md_rule("none") for t in uniq}
            self.topology = MDTopology.build(sys, [rules[id(t)] for t in templates])
            self.flex = FlexibleMolecules(sys, pos, H0, templates, self.topology, self.masses)
            self.ff = PGMForceField(
                sys, H0, settings, topology=self.topology, flux=ChargeFlux.from_templates(sys, templates)
            )
            self.ff.masses = jnp.asarray(self.masses)
            tb = [
                t.terms.mols[t.index].top.bonds if t.has_bonded else sys.molecules[k].bonds
                for k, t in enumerate(templates)
            ]
            bonds_of = lambda k: tb[k]
        from ..md.restraints import as_restraints

        self.restraints = as_restraints(restraints)
        if self.restraints is not None:
            self.restraints.check(sys.n)
        parent, depth = _bond_tree(sys, bonds_of)
        self._parent = jnp.asarray(parent)
        self._nhop = max(1, int(math.ceil(math.log2(depth + 1)))) if depth > 0 else 0
        self._mol = jnp.asarray(sys.mol)
        self._nmol = sys.nmol
        mm = np.bincount(np.asarray(sys.mol), weights=self.masses, minlength=sys.nmol)
        self._wmol = jnp.asarray(self.masses / mm[np.asarray(sys.mol)])
        self._group = jnp.asarray(self.topology.group)
        self._ngroup = int(self.topology.n_group)
        mg = np.bincount(np.asarray(self.topology.group), weights=self.masses, minlength=self._ngroup)
        self._wgroup = jnp.asarray(self.masses / mg[np.asarray(self.topology.group)])
        from ..md.dipoles import CellDipole

        self._celldip = CellDipole(self.ff)
        # whole molecules at the start: list radius and sizes
        x0 = np.asarray(self._whole_jit()(jnp.asarray(pos), jnp.asarray(H0)))
        self.initial_positions, self.initial_box = x0, H0  # standard frame (OpenMM's box form)
        self.r_margin = float(r_margin)
        self.r_list = self.topology.group_radius(x0, self.masses) + self.r_margin
        self._nb_mode = neighbor_list
        self._make_neighbors(H0)
        self._size(x0, H0)
        self.slots = [_Slot() for _ in range(max(1, int(slots)))]
        self.stats = {"calls": 0, "repeats": 0, "rebuilds": 0, "cg": 0, "resets": 0, "time": 0.0}
        self._compile()

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_amber(cls, prmtop: str, coords: str, charges: str = "pgm", **kw) -> PGMEngine:
        """Rigid-molecule model from a pGM prmtop and coordinates (as Simulation.from_amber)."""
        from ..md.io import box_from_cell, read_coordinates
        from ..md.simulation import _dedupe
        from ..param import read_prmtop_pgm

        mols = _dedupe(read_prmtop_pgm(prmtop, first_residue_only=False, charges=charges))
        sys = System(mols)
        xyz, _, box = read_coordinates(coords)
        if box is None:
            raise ValueError("coordinates have no periodic box")
        return cls(sys, xyz * 0.1, box_from_cell(*box) * 0.1, **kw)

    @classmethod
    def from_simulation(cls, sim, **kw) -> PGMEngine:
        """The model of a native Simulation / FlexibleSimulation at its current state (same system,
        settings, parameters and templates; restraints included)."""
        templates = kw.pop("templates", None)
        if templates is None and hasattr(sim, "flex"):
            raise ValueError("FlexibleSimulation: pass its templates, PGMEngine.from_simulation(sim, templates=...)")
        if getattr(sim.integ, "efield", None) is not None or getattr(sim.integ, "bias", None) is not None:
            raise NotImplementedError(
                "the external-code interfaces do not carry the external field (efield=) or "
                "the biases (bias=) of a simulation; use the native engine"
            )
        kw.setdefault("params", sim.integ.params)
        kw.setdefault("restraints", sim.integ.restraints)
        return cls(sim.sys, sim.positions_nm(), np.asarray(sim.state.box), sim.settings, templates=templates, **kw)

    # ------------------------------------------------------------------ neighbour lists
    def _make_neighbors(self, H):
        s = self.settings
        mode = self._nb_mode
        if mode == "auto":
            mode = "molecule" if MoleculeNeighbors.fits(H, s.pair_cutoff, s.skin, self.r_list) else "atom"
        if mode == "molecule":
            self.nb = MoleculeNeighbors(self.topology.group, self._ngroup, self.r_list, H, s.pair_cutoff, s.skin)
        else:
            self.nb = AtomNeighbors(self.n, H, s.pair_cutoff, s.skin)
        self._nb_volume = float(np.linalg.det(np.asarray(H)))

    def _centers(self, x):
        return jax.ops.segment_sum(self._wgroup[:, None] * x, self._group, self._ngroup)

    def _size(self, x, H, factor: float = 1.2, nbr=None):
        x, H = jnp.asarray(x), jnp.asarray(H)
        c = self._centers(x)
        nbr = self.nb.allocate(x, c, H) if nbr is None else nbr
        if self.nb.kind == "molecule":
            self.nb.size(nbr, c, H, x, factor)
        idx = self.nb.candidates(nbr, c, H, x)[0]
        self.ff.size_rows(x, H, idx, factor)
        return nbr

    # ------------------------------------------------------------------ jitted pieces
    def _whole(self, x, H):
        """Molecules made whole along their bond trees, then shifted by lattice vectors so that
        their centres of mass lie in the primary cell."""
        if self._nhop:
            s = min_image(x - x[self._parent], H)
            a = self._parent
            for _ in range(self._nhop + 1):
                s = s + s[a]
                a = a[a]
            x = x[a] + s
        com = jax.ops.segment_sum(self._wmol[:, None] * x, self._mol, self._nmol)
        hi = jax.lax.Precision.HIGHEST
        f = jnp.matmul(com, inv3(H), precision=hi)
        return x - jnp.matmul(jnp.floor(f), H, precision=hi)[self._mol]

    def _whole_jit(self):
        if getattr(self, "_whole_c", None) is None:
            self._whole_c = jax.jit(self._whole)
        return self._whole_c

    def _list_fits(self, H) -> bool:
        s = self.settings
        if self.nb.kind == "molecule":
            return MoleculeNeighbors.fits(H, s.pair_cutoff, s.skin, self.r_list)
        return self.nb.rlist <= max_cutoff(H)

    def _eval(self, x, H, ind, nbr, virial: bool, dipole: bool = True):
        x = self._whole(x, H)
        c = self._centers(x)
        nbr = self.nb.update(nbr, x, c, H)
        cand, ovf = self.nb.candidates(nbr, c, H, x)
        res = self.ff.compute(x, H, cand, ind, self.params)
        E, F = res.energy["total"], res.forces
        eb = jnp.zeros((), jnp.float64)
        gb = None
        if self.flex is not None:
            eb, gb = jax.value_and_grad(self.flex.energy)(x)
            F = F - gb
        er = jnp.zeros((), jnp.float64)
        if self.restraints is not None:
            er, gr = jax.value_and_grad(self.restraints.energy)(x, H)
            F = F - gr
        ext = jnp.max(jnp.sqrt(jnp.sum((x - c[self._group]) ** 2, axis=1)))
        head = jnp.stack(
            [
                E + eb + er,
                res.energy["elec"],
                res.energy["vdw"],
                eb,
                er,
                res.iterations.astype(jnp.float64),
                res.residual.astype(jnp.float64),
                (res.overflow | ovf).astype(jnp.float64),
                nbr.error.code.astype(jnp.float64),
                ext,
            ]
        )
        packed = jnp.concatenate([head, F.reshape(-1)])
        dev = {"mu": res.induction.mu}
        if dipole:
            dev["dip"] = self._celldip.components(x, H, res.induction.mu, self.params)
        if virial:
            dev["W"] = self._virial(x, H, cand, res.induction.mu, gb)
        return packed, res.induction, nbr, dev

    def _dipole_only(self, x, H, mu):
        return self._celldip.components(self._whole(x, H), H, mu, self.params)

    def _virial(self, x, H, cand, mu, gb):
        molecular = self.stress_mode == "molecular"
        W = self.ff.strain_derivative(x, H, cand, mu, self.params, molecular=molecular)
        # strain_derivative returns the full tensor (its lower components from rotation invariance);
        # external codes get the symmetric tensor built from the upper triangle (for the atomic
        # virial the tensor itself; for the molecular one its antisymmetric torque part is dropped)
        W = jnp.triu(W) + jnp.triu(W, 1).T
        if gb is not None and not molecular:
            W = W + gb.T @ x  # bonded energy under x -> x (1 + eps)^T: sum_i g_i (x) x_i
        if self.restraints is not None:
            W = W + (
                self.restraints.strain_derivative(x, H, self.ff.mol, self.ff.masses, self._nmol)
                if molecular
                else self._restraint_strain_atomic(x, H)
            )
        return W

    def _restraint_strain_atomic(self, x, H):
        def e(eps):
            F = jnp.eye(3) + eps
            return self.restraints.energy(x @ F.T, H @ F.T)

        return jax.grad(e)(jnp.zeros((3, 3)))

    def _compile(self):
        self._fn = jax.jit(self._eval, static_argnames=("virial", "dipole"))
        self._dip_fn = jax.jit(self._dipole_only)
        self._vir_fn = jax.jit(self._virial_only)

    def _virial_only(self, x, H, ind, nbr):
        x = self._whole(x, H)
        c = self._centers(x)
        cand, _ = self.nb.candidates(nbr, c, H, x)
        gb = jax.grad(self.flex.energy)(x) if self.flex is not None else None
        return self._virial(x, H, cand, ind.mu, gb)

    # ------------------------------------------------------------------ host side
    @staticmethod
    def _displacement(a, b, H) -> float:
        """Largest displacement between two configurations (nm), each atom by its minimum image:
        drivers may wrap atoms or molecules back into the cell between calls."""
        d = b - a
        m = float(np.max(np.abs(d)))
        if m < 0.25 * float(np.min(np.diag(H))):  # no lattice jump possible: the common case
            return m
        f = d @ np.linalg.inv(H)
        return float(np.max(np.abs(d - np.round(f) @ H)))

    def _slot_for(self, x, H) -> _Slot:
        """An unused slot while there is one (the first calls of interleaved configurations fill the
        slots in turn), then the slot whose last configuration is closest to x."""
        if len(self.slots) == 1:
            return self.slots[0]
        for s in self.slots:
            if s.x is None:
                return s
        d = [self._displacement(s.x, x, H) for s in self.slots]
        return self.slots[int(np.argmin(d))]

    def batch_slots(self, positions, cell) -> np.ndarray:
        """Slots for a batch of structures evaluated one by one: the one-to-one assignment to the
        slots' last configurations (first batches: slot k for structure k)."""
        X = np.asarray(positions, float)
        B = X.shape[0]
        if B > len(self.slots) or any(self.slots[k].x is None for k in range(B)):
            return np.arange(B) % len(self.slots)
        _, Q = standard_cell(cell)
        Xs = X if Q is None else X @ Q
        return match_previous(Xs, np.stack([self.slots[k].x for k in range(B)]))

    def reset(self):
        """Forget the dipole histories (the next call starts every slot from scratch)."""
        for s in self.slots:
            s.x = s.ind = s.nbr = None

    def compute(self, pos, cell, virial: bool = False, slot: int | None = None) -> EngineResult:
        """Energy and forces (and the virial if asked) at positions pos (N, 3) nm and cell (3, 3)
        nm, rows = lattice vectors; any right-handed cell, atoms wrapped or not.  slot: the
        induced-dipole history to use (default: the closest one)."""
        t0 = time.perf_counter()
        cell = np.asarray(cell, float)
        x = np.array(pos, float)  # a copy: kept as the slot's last configuration
        if x.shape != (self.n, 3):
            raise ValueError(f"expected positions of shape ({self.n}, 3), got {x.shape}")
        cache = getattr(self, "_cell_cache", None)
        if cache is not None and np.array_equal(cache[0], cell):
            H, Q, Hd = cache[1:]  # same cell as the last call (NVE / NVT)
        else:
            H, Q = standard_cell(cell)
            check_box(H, self.settings.pair_cutoff + self.settings.skin)
            if abs(float(np.linalg.det(H)) / self._nb_volume - 1.0) > 0.10 or not self._list_fits(H):
                self._rebuild(x if Q is None else x @ Q, H)
            Hd = jnp.asarray(H)
            self._cell_cache = (cell.copy(), H, Q, Hd)
        if Q is not None:
            x = x @ Q
        slot = self._slot_for(x, H) if slot is None else self.slots[int(slot)]
        if slot.x is not None and self._displacement(slot.x, x, H) > self.jump:
            slot.ind = None  # new configuration: restart the predictor
            self.stats["resets"] += 1
        xd = jnp.asarray(x)
        for attempt in range(8):
            if slot.nbr is None:
                xw = self._whole_c(xd, Hd)
                slot.nbr = self.nb.allocate(xw, self._centers(xw), Hd)
            ind = self.ff.init_induction() if slot.ind is None else slot.ind
            packed, ind_new, nbr_new, dev = self._fn(
                xd, Hd, ind, slot.nbr, virial=bool(virial), dipole=bool(self.with_dipole)
            )
            out = np.asarray(packed)
            ovf, code, ext = bool(out[7]), int(out[8]), float(out[9])
            nb_bad = bool(code & _LIST_ERRORS)  # = nb.failed(nbr_new), without another transfer
            far = self.nb.kind == "molecule" and ext > self.r_list
            if not (ovf or nb_bad or far) and np.isfinite(out[0]):
                break
            if not np.isfinite(out[0]) and not (ovf or nb_bad or far):
                raise FloatingPointError("pGM energy is not finite")
            self.stats["repeats"] += 1
            if far:
                self.r_margin = max(2.0 * self.r_margin, ext - self.r_list + self.r_margin + 0.02)
                self._rebuild(x, H)
            else:
                old = (self.ff.capacity, getattr(self.nb, "cap", None))
                xw = self._whole_c(xd, Hd)
                try:
                    nbr = self._size(xw, Hd, 1.3, None if nb_bad else slot.nbr)
                except ValueError:
                    self._rebuild(x, H)
                    continue
                if ovf:
                    self.ff.grow_rows(old[0])
                    if getattr(self.nb, "cap", None) is not None and old[1] is not None:
                        self.nb.cap = max(self.nb.cap, old[1] + 4)
                for s in self.slots:
                    s.nbr = None
                slot.nbr = nbr
                self._compile()
        else:
            raise RuntimeError("neighbour list / row capacity keeps overflowing")
        slot.x, slot.ind, slot.nbr = x, ind_new, nbr_new
        n = self.n
        E = float(out[0])
        F = out[10 : 10 + 3 * n].reshape(n, 3)
        if Q is not None:
            F = F @ Q.T
        self._last = (xd, Hd, slot)
        if "dip" not in dev:
            mu = dev["mu"]
            dev["dip_fn"] = lambda: self._dip_fn(xd, Hd, mu)
        self.stats["calls"] += 1
        self.stats["cg"] += int(out[5])
        self.stats["time"] += time.perf_counter() - t0
        return EngineResult(
            E,
            F,
            {"elec": float(out[1]), "vdw": float(out[2]), "bonded": float(out[3]), "restraint": float(out[4])},
            int(out[5]),
            dev,
            Q,
        )

    def virial_of_last(self, res: EngineResult) -> np.ndarray:
        """The virial (kJ/mol, caller's frame) of the last computed configuration without a new
        dipole solve (e.g. when a driver asks for the stress after the forces)."""
        xd, Hd, slot = self._last
        res._dev["W"] = self._vir_fn(xd, Hd, slot.ind, slot.nbr)
        return res.virial

    def _rebuild(self, x, H):
        """New neighbour-list object for box H (large volume change, list radius exceeded)."""
        self.stats["rebuilds"] += 1
        xw = np.asarray(self._whole_jit()(jnp.asarray(x), jnp.asarray(H)))
        self.r_list = max(self.r_list, self.topology.group_radius(xw, self.masses) + self.r_margin)
        self._make_neighbors(H)
        nbr = self._size(xw, H)
        for s in self.slots:
            s.nbr = None
        self._compile()
        return nbr

    # ------------------------------------------------------------------ batches (ring-polymer beads)
    def _whole_batch(self, X, H):
        """Every structure made whole and wrapped (as _whole), then each molecule of structure k
        shifted by the lattice vector that puts its centre of mass nearest to that of structure 0,
        so that the batch mean (the centroid of a ring polymer) is meaningful."""
        Xw = jax.vmap(self._whole, in_axes=(0, None))(X, H)
        com = jax.vmap(lambda x: jax.ops.segment_sum(self._wmol[:, None] * x, self._mol, self._nmol))(Xw)
        d = min_image(com - com[0][None], H)
        return Xw + (com[0][None] + d - com)[:, self._mol]

    def _eval_batch(self, X, H, ind, nbr, idx, virial: bool, chunk, dipole: bool = True):
        """X: (P, N, 3) the slots' configurations (the list is built on their mean); idx: (n,) the
        slots evaluated (repeats allowed: identical inputs give identical results)."""
        X = self._whole_batch(X, H)
        qc = jnp.mean(X, 0)
        c = self._centers(qc)
        nb = self._nbb
        nbr = nb.update(nbr, qc, c, H)
        ff, params = self.ff, self.params

        def one(xk, indk):
            cand, ovf = nb.candidates(nbr, c, H, xk)
            res = ff.compute(xk, H, cand, indk, params)
            E, F = res.energy["total"], res.forces
            eb, gb = jnp.zeros((), jnp.float64), None
            if self.flex is not None:
                eb, gb = jax.value_and_grad(self.flex.energy)(xk)
                F = F - gb
            er = jnp.zeros((), jnp.float64)
            if self.restraints is not None:
                er, gr = jax.value_and_grad(self.restraints.energy)(xk, H)
                F = F - gr
            head = jnp.stack(
                [
                    E + eb + er,
                    res.energy["elec"],
                    res.energy["vdw"],
                    eb,
                    er,
                    res.iterations.astype(jnp.float64),
                    res.residual.astype(jnp.float64),
                    (res.overflow | ovf).astype(jnp.float64),
                ]
            )
            dev = {"mu": res.induction.mu}
            if dipole:
                dev["dip"] = self._celldip.components(xk, H, res.induction.mu, params)
            if virial:
                dev["W"] = self._virial(xk, H, cand, res.induction.mu, gb)
            return jnp.concatenate([head, F.reshape(-1)]), res.induction, dev

        ind_full = ind
        ind = jax.tree_util.tree_map(lambda a: a[idx], ind_full.set(count=None)).set(count=ind_full.count)
        Xall, X = X, X[idx]
        ax = jax.tree_util.tree_map(lambda _: 0, ind.set(count=None))
        vf = jax.vmap(one, in_axes=(0, ax), out_axes=(0, ax, 0))
        B = X.shape[0]
        if chunk is None or chunk >= B or B % chunk:
            packed, ind_new, dev = vf(X, ind)
        else:  # chunks of vmapped structures, one after the other
            count = ind.count
            split = lambda a: a.reshape((B // chunk, chunk) + a.shape[1:])  # noqa: E731
            merge = lambda a: a.reshape((B,) + a.shape[2:])  # noqa: E731

            def body(args):
                xc, ic = args
                p, i2, d = vf(xc, ic.set(count=count))
                return (p, i2.set(count=None), d), i2.count

            (packed, ind_new, dev), counts = jax.lax.map(
                body, (split(X), jax.tree_util.tree_map(split, ind.set(count=None)))
            )
            packed, dev = merge(packed), jax.tree_util.tree_map(merge, dev)
            ind_new = jax.tree_util.tree_map(merge, ind_new).set(count=counts[0])
        ref = c[self._group] if nb.kind == "molecule" else qc
        ext = jnp.max(jnp.sqrt(jnp.sum((Xall - ref[None]) ** 2, axis=-1)))
        flags = jnp.stack([nbr.error.code.astype(jnp.float64), ext])
        put = lambda full, new: full.at[idx].set(new)  # noqa: E731
        ind_new = jax.tree_util.tree_map(put, ind_full.set(count=None), ind_new.set(count=None)).set(
            count=ind_new.count
        )
        return packed, flags, ind_new, nbr, dev

    def _batch_setup(self, B, X, H):
        """Batch state: a molecular-centre neighbour list of the batch mean with the list radius
        enlarged by bead_margin, stacked dipole histories (the predictor's step counter shared)."""
        s = self.settings
        r = self.r_list + self.bead_margin
        if MoleculeNeighbors.fits(H, s.pair_cutoff, s.skin, r):
            self._nbb = MoleculeNeighbors(self.topology.group, self._ngroup, r, H, s.pair_cutoff, s.skin)
            self._bfar = r  # atom to its centroid group's centre
        else:  # atom list of the centroid: pairs of structure atoms within cutoff + 2 margins
            rc = s.pair_cutoff + 2.0 * self.bead_margin
            skin = min(s.skin, max_cutoff(H) - rc - 0.002)
            if skin < 0.02:
                return False
            self._nbb = AtomNeighbors(self.n, H, rc, skin)
            self._bfar = self.bead_margin  # atom to its centroid atom
        Xw = self._whole_batch_c(jnp.asarray(X), jnp.asarray(H))
        qc = jnp.mean(Xw, 0)
        c = self._centers(qc)
        nbr = self._nbb.allocate(qc, c, jnp.asarray(H))
        if self._nbb.kind == "molecule":
            self._nbb.cap = max(self._nbb.size(nbr, c, jnp.asarray(H), Xw[k], 1.2) for k in range(B))
        ind0 = self.ff.init_induction()
        ind = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (B,) + jnp.shape(a)), ind0.set(count=None))
        self._bstate = {
            "P": B,
            "ind": ind.set(count=ind0.count),
            "nbr": nbr,
            "H": np.asarray(H).copy(),
            "vol": float(np.linalg.det(H)),
            "x": np.asarray(X, float),
            "filled": 0,
            "seen": np.zeros(B, bool),
        }
        self._bfn = jax.jit(self._eval_batch, static_argnames=("virial", "chunk", "dipole"))
        return True

    def compute_batch(self, positions, cell, virial: bool = False, chunk="auto") -> list:
        """Structures that share one cell and stay close to each other (the beads of a ring polymer:
        i-PI's batched requests) in one vmapped call; positions (B, N, 3) nm.

        The engine keeps P = B (first batch) slots, each with its own induced-dipole history and
        last configuration; one neighbour list of the slots' mean (radius + bead_margin) serves them
        all.  Every batch is matched to the slots: exact duplicates (i-PI pads partial batches with
        copies of the last structure) are evaluated once, the distinct structures are assigned
        one-to-one to the slots with the closest last configurations (i-PI does not keep the order
        of the beads), and the vmapped call runs over these slots only (padded to P/4, P/2 or P
        structures: one compiled program per size); the list is built on the mean of all slots'
        last configurations.  chunk: structures per vmapped chunk ("auto": 8 when that divides
        more than 8).  Falls back to one compute() per structure when the
        slots cannot share a list.  Returns one EngineResult per input structure."""
        X = np.array(positions, float)
        B = X.shape[0]
        cell = np.asarray(cell, float)
        H, Q = standard_cell(cell)
        if Q is not None:
            X = X @ Q
        check_box(H, self.settings.pair_cutoff + self.settings.skin)
        keys, uniq, where = {}, [], []
        for k in range(B):  # exact duplicates: i-PI's padding
            key = X[k].tobytes()
            if key not in keys:
                keys[key] = len(uniq)
                uniq.append(k)
            where.append(keys[key])
        U = X[uniq]
        n = len(U)
        if getattr(self, "_whole_batch_c", None) is None:
            self._whole_batch_c = jax.jit(self._whole_batch)
        st = getattr(self, "_bstate", None)
        if st is None or n > st["P"] or abs(float(np.linalg.det(H)) / st["vol"] - 1.0) > 0.10:
            P = max(B, n) if st is None else max(st["P"], n)
            full0 = np.stack([U[k % n] for k in range(P)])
            if not self._batch_setup(P, full0, H):
                back = (lambda y: y) if Q is None else (lambda y: y @ Q.T)
                return [self.compute(back(X[k]), cell, virial, slot=k % len(self.slots)) for k in range(B)]
            st = self._bstate
            st["x"] = full0
        P = st["P"]
        if st["filled"] < P and st["filled"] + n <= P:  # first batches: fill the slots in order
            slot_of = np.arange(st["filled"], st["filled"] + n)
            st["filled"] += n
        else:
            slot_of = match_previous(U, st["x"])
            st["filled"] = P
        full = st["x"].copy()
        full[slot_of] = U
        present = np.zeros(P, bool)
        present[slot_of] = True
        # evaluate the slots of this batch only, padded to P/4, P/2 or P structures (one compiled
        # program per size); i-PI often splits the beads of a step over two batches
        m = next((b for b in (max(1, P // 4), max(1, P // 2)) if b >= n and P % b == 0), P)
        idx = np.concatenate([slot_of, np.full(m - n, slot_of[0])]).astype(np.int32)
        if chunk == "auto":
            chunk = 8 if (m > 8 and m % 8 == 0) else None
        if st["seen"].all() and self._displacement(st["x"][slot_of], U, H) > self.jump:
            ind0 = self.ff.init_induction()  # a new configuration: restart the predictors
            st["ind"] = jax.tree_util.tree_map(
                lambda a: jnp.broadcast_to(a, (P,) + jnp.shape(a)), ind0.set(count=None)
            ).set(count=ind0.count)
            self.stats["resets"] += 1
        st["seen"] |= present
        t0 = time.perf_counter()
        Xd, Hd, md = jnp.asarray(full), jnp.asarray(H), jnp.asarray(idx)
        for attempt in range(8):
            packed, flags, ind_new, nbr_new, dev = self._bfn(
                Xd, Hd, st["ind"], st["nbr"], md, virial=bool(virial), chunk=chunk, dipole=bool(self.with_dipole)
            )
            out, fl = np.asarray(packed), np.asarray(flags)
            ovf = bool(np.any(out[:, 7]))
            nb_bad = bool(int(fl[0]) & _LIST_ERRORS)
            far = float(fl[1]) > self._bfar
            if not (ovf or nb_bad or far):
                break
            self.stats["repeats"] += 1
            if far:
                self.bead_margin *= 1.5
                keep = (st["ind"], st["filled"], st["seen"])
                if not self._batch_setup(P, full, H):
                    self._bstate = None
                    back = (lambda y: y) if Q is None else (lambda y: y @ Q.T)
                    return [self.compute(back(X[k]), cell, virial, slot=k % len(self.slots)) for k in range(B)]
                st = self._bstate
                st["ind"], st["filled"], st["seen"], st["x"] = keep[0], keep[1], keep[2], full
            else:
                Xw = self._whole_batch_c(Xd, Hd)
                old = self.ff.capacity
                qc = jnp.mean(Xw, 0)
                c = self._centers(qc)
                nbr = self._nbb.allocate(qc, c, Hd) if nb_bad else st["nbr"]
                if self._nbb.kind == "molecule":
                    self._nbb.cap = max(self._nbb.size(nbr, c, Hd, Xw[k], 1.3) for k in range(P))
                caps = []
                for k in range(P):
                    self.ff.size_rows(Xw[k], Hd, self._nbb.candidates(nbr, c, Hd, Xw[k])[0], 1.3)
                    caps.append(self.ff.capacity)
                self.ff.fit_rows(caps)
                if ovf:
                    self.ff.grow_rows(old)
                st["nbr"] = nbr
                self._compile()
                for sl in self.slots:
                    sl.nbr = None
                self._bfn = jax.jit(self._eval_batch, static_argnames=("virial", "chunk", "dipole"))
        else:
            raise RuntimeError("neighbour list / row capacity keeps overflowing (batch)")
        st["ind"], st["nbr"], st["x"] = ind_new, nbr_new, full
        nat = self.n
        res_u = []
        host = {key: np.asarray(v) for key, v in dev.items() if key in ("dip", "W")}  # one transfer each
        mu_all = dev["mu"]
        for j in range(n):
            k = j  # row j of the evaluated slots (idx[j] = slot_of[j])
            F = out[k, 8 : 8 + 3 * nat].reshape(nat, 3)
            if Q is not None:
                F = F @ Q.T
            dk = {key: v[k] for key, v in host.items()}
            dk["mu_fn"] = lambda k=k: np.asarray(mu_all[k])
            if "dip" not in dk:
                dk["dip_fn"] = lambda k=k, sl=int(slot_of[j]): self._dip_fn(Xd[sl], Hd, mu_all[k])
            res_u.append(
                EngineResult(
                    float(out[k, 0]),
                    F,
                    {
                        "elec": float(out[k, 1]),
                        "vdw": float(out[k, 2]),
                        "bonded": float(out[k, 3]),
                        "restraint": float(out[k, 4]),
                    },
                    int(out[k, 5]),
                    dk,
                    Q,
                )
            )
            self.stats["cg"] += int(out[k, 5])
        self.stats["calls"] += n
        self.stats["batches"] = self.stats.get("batches", 0) + 1
        self.stats["slot_evaluations"] = self.stats.get("slot_evaluations", 0) + m
        self.stats["time"] += time.perf_counter() - t0
        return [res_u[where[k]] for k in range(B)]

    def describe(self) -> str:
        s = self.settings
        return (
            f"pgm_jax engine: {self.sys.nmol} molecules, {self.n} atoms, "
            f"{'flexible templates' if self.flex is not None else 'rigid-molecule model'}, "
            f"{s.precision} precision, PME grid {self.ff.pme.K} order {s.pme_order}, {s.describe_cutoffs()}, "
            f"{self.nb.kind} neighbour list, dipole tol {s.dipole_tol:g}, {len(self.slots)} dipole slot(s), "
            f"stress {self.stress_mode}, device {jax.devices()[0]}"
        )


# ----------------------------------------------------------------------------- gas phase
class GasPhaseEngine:
    """Gas-phase pGM (pgm_jax.Model: every pair, dense induction solve) for external drivers:
    energy, forces, induced dipoles and the total dipole of a cluster or molecule, no cell.

        eng = GasPhaseEngine(Model([ElecChannel(), LJChannel()]), system, params=None)
        res = eng.compute(pos_nm)"""

    def __init__(self, model, sys: System, params=None):
        self.model, self.sys, self.params, self.n = model, sys, params, sys.n
        chans = model.build(sys)

        def f(pos, params):
            out, aux = {}, {}
            for ch in chans:
                e, a = ch.energy(pos, sys, params)
                out.update(e)
                if isinstance(a, dict):
                    aux.update(a)
            total = sum(out.values())
            return total, (out, aux)

        def run(pos, params):
            (E, (terms, aux)), g = jax.value_and_grad(f, has_aux=True)(pos, params)
            P = sys.expand(params)
            mu = aux.get("mu", jnp.zeros((sys.n, 3)))
            p = aux.get("p", jnp.zeros((sys.n, 3)))
            com = jnp.sum(jnp.asarray(sys.masses)[:, None] * pos, 0) / jnp.sum(jnp.asarray(sys.masses))
            dip = jnp.stack([jnp.sum(P["q"][:, None] * (pos - com), 0), jnp.sum(p, 0), jnp.sum(mu, 0)])
            return E, -g, terms, mu, dip

        self._fn = jax.jit(run)
        self.stats = {"calls": 0, "time": 0.0}

    def compute(self, pos, cell=None, virial: bool = False) -> EngineResult:
        if virial:
            raise ValueError("no virial in the gas phase")
        t0 = time.perf_counter()
        E, F, terms, mu, dip = self._fn(jnp.asarray(pos, jnp.float64), self.params)
        F = np.asarray(F)
        res = EngineResult(float(E), F, {k: float(v) for k, v in terms.items()}, 0, {"mu": mu, "dip": dip}, None)
        self.stats["calls"] += 1
        self.stats["time"] += time.perf_counter() - t0
        return res
