"""Provide an ASE calculator for pGM (pgm_jax.interfaces.engine), periodic or gas phase.

Contents: `PGMCalculator` (ASE Calculator around a PGMEngine or GasPhaseEngine),
`atoms_from_system`, and `FixRigidMolecules` / `rigid_constraints` / `rigid_blocks`
(vectorised SHAKE / RATTLE for small rigid molecules, the rigid-molecule model).

    from pgm_jax.interfaces.ase import PGMCalculator, atoms_from_system, rigid_constraints
    eng = PGMEngine.from_amber("water.prmtop", "water.rst7", settings=MDSettings())
    atoms = atoms_from_system(eng.sys, pos_nm, H_nm)          # symbols, pGM masses, cell, pbc
    atoms.set_constraint(rigid_constraints(eng.sys))          # rigid-molecule model: waters rigid (SHAKE/RATTLE)
    atoms.calc = PGMCalculator(eng)
    atoms.get_potential_energy(); atoms.get_forces(); atoms.get_stress()
    atoms.calc.results["induced_dipoles"]                      # e Angstrom, per atom

Properties (ASE units: eV, Angstrom, e): energy / free_energy, forces, stress (Voigt, periodic
engines; the virial W = dE/d eps of the engine over the volume, "atomic" or "molecular" scaling as
chosen when the engine was built), dipole (the cell dipole M_q + M_perm + M_ind, molecules whole,
e Angstrom; the three parts in results["dipole_components"], rows M_q, M_perm, M_ind) and
induced_dipoles (N, 3, e Angstrom).  Extra results: "energy_terms" (eV: elec, vdw, bonded,
restraint) and "cg_iterations".

A stress asked for after the forces of the same configuration costs one strain derivative at the
converged dipoles (no second dipole solve).  Positions wrapped atom by atom are fine.

Units: ASE's (eV, Angstrom, amu, e) at the interface; KJMOL and NM convert from the engine's.

See also docs/interfaces.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixConstraint
from ase.stress import full_3x3_to_voigt_6_stress

from .engine import GasPhaseEngine

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ..system import System
    from .engine import PGMEngine

KJMOL = units.kJ / units.mol  # eV per kJ/mol
NM = 10.0  # Angstrom per nm


class PGMCalculator(Calculator):
    """ASE Calculator for a PGMEngine (periodic) or GasPhaseEngine (see the module docstring).

    Attributes
    ----------
    engine : PGMEngine or GasPhaseEngine
        The engine.
    periodic : bool
        False for a GasPhaseEngine (no stress).
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "dipole", "induced_dipoles"]
    nolabel = True

    def __init__(self, engine: PGMEngine | GasPhaseEngine, **kwargs: object) -> None:
        """Set up the calculator.

        Parameters
        ----------
        engine : PGMEngine or GasPhaseEngine
            The engine (its atom order is the Atoms' order).
        **kwargs
            Passed to ase.calculators.calculator.Calculator.
        """
        super().__init__(**kwargs)
        self.engine = engine
        self.periodic = not isinstance(engine, GasPhaseEngine)
        self._res = None

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        """Compute the requested properties into `self.results` (ASE Calculator interface).

        A new engine call is made only when the system changed; a stress asked for later uses
        `PGMEngine.virial_of_last` (no second dipole solve).

        Parameters
        ----------
        atoms : Atoms, optional
            The configuration (copied without its constraints); None: the last one.
        properties : sequence of str
            Requested properties.
        system_changes : sequence of str
            What changed since the last call (ASE).

        Raises
        ------
        ValueError
            If the atom counts differ, a stress is asked of a gas-phase engine, or a periodic engine
            gets an Atoms object without pbc in all three directions.
        """
        if atoms is not None:  # Calculator.calculate: self.atoms = atoms.copy(), but without
            self.atoms = _light_copy(atoms)  # deep-copying the constraints (most of ASE's cost per step)
        a = self.atoms
        if len(a) != self.engine.n:
            raise ValueError(f"the engine has {self.engine.n} atoms, the Atoms object {len(a)}")
        want_stress = "stress" in properties
        if want_stress and not self.periodic:
            raise ValueError("stress needs a periodic engine")
        if system_changes or self._res is None:
            pos = a.get_positions() / NM
            if self.periodic:
                if not all(a.pbc):
                    raise ValueError("a periodic pGM engine needs pbc=True in all three directions")
                res = self.engine.compute(pos, np.asarray(a.get_cell()) / NM, virial=want_stress)
            else:
                res = self.engine.compute(pos)
            self._res = res
            r = self.results
            r["energy"] = r["free_energy"] = res.energy * KJMOL
            r["forces"] = res.forces * (KJMOL / NM)
            r["energy_terms"] = {k: v * KJMOL for k, v in res.terms.items()}
            r["cg_iterations"] = res.iterations
        res = self._res
        r = self.results
        if want_stress and "stress" not in r:
            W = res.virial if res.virial is not None else self.engine.virial_of_last(res)
            W = 0.5 * (W + W.T)
            # ASE stress = (1/V) dE/d eps (positive under tension), eV/A^3
            V = abs(np.linalg.det(np.asarray(self.atoms.get_cell()))) / NM**3
            r["stress"] = full_3x3_to_voigt_6_stress(W / V) * (KJMOL / NM**3)
        if any(p in properties for p in ("dipole", "induced_dipoles")):
            self._dipoles()

    def _dipoles(self) -> None:
        """Store the cell dipole, its components and the induced dipoles [e Angstrom] in the results."""
        res, r = self._res, self.results
        r["dipole_components"] = res.dipole_components * NM
        r["dipole"] = r["dipole_components"].sum(0)
        r["induced_dipoles"] = res.induced_dipoles * NM

    def get_induced_dipoles(self, atoms: Atoms | None = None) -> np.ndarray:
        """Return the induced dipoles, np.ndarray (N, 3) [e Angstrom]."""
        return self.get_property("induced_dipoles", atoms)


# ----------------------------------------------------------------------------- helpers
def _light_copy(atoms: Atoms) -> Atoms:
    """Return atoms.copy() without the constraints (enough for the calculator's change checks)."""
    new = atoms.__class__(cell=atoms.cell, pbc=atoms.pbc, info=atoms.info, celldisp=atoms._celldisp.copy())
    new.arrays = {k: v.copy() for k, v in atoms.arrays.items()}
    return new


def atoms_from_system(sys: System, pos_nm: ArrayLike, H_nm: ArrayLike | None = None) -> Atoms:
    """Return ase.Atoms for a pgm_jax System: element symbols, the system's masses and positions.

    Parameters
    ----------
    sys : System
        The system.
    pos_nm : ArrayLike (N, 3)
        Positions [nm] (stored in Angstrom).
    H_nm : ArrayLike (3, 3), optional
        Cell, lattice vectors as rows [nm]: sets the cell and pbc; None: no cell.
    """
    symbols = [e for m in sys.molecules for e in m.elements]
    a = Atoms(symbols=symbols, positions=np.asarray(pos_nm, float) * NM, masses=np.asarray(sys.masses, float))
    if H_nm is not None:
        a.set_cell(np.asarray(H_nm, float) * NM)
        a.set_pbc(True)
    return a


class FixRigidMolecules(FixConstraint):
    """Distance constraints of small rigid molecules (water: O-H, O-H, H-H), solved exactly and vectorised.

    SHAKE for positions by Newton iterations on the (up to 3) Lagrange multipliers of each
    molecule, RATTLE for momenta by one batched linear solve, both vectorised over molecules.  ASE's
    FixBondLengths does the same pair by pair in Python loops (about 1 ms per water per step).
    Molecules are grouped by their number of constraints; differences use the minimum image of the
    cell when any direction is periodic.

    Attributes
    ----------
    pairs : dict
        {c: (M, c, 2) atom pairs}: molecules grouped by their number c of constraints.
    bondlengths : dict or None
        {c: (M, c)} target distances [Angstrom]; None until the first configuration sets them.
    tolerance : float
        Relative tolerance on |u|^2 - d^2.
    maxiter : int
        Newton iterations of SHAKE.
    constraint_forces : np.ndarray (N, 3)
        Forces removed by the last adjust_forces [eV/Angstrom].
    """

    def __init__(
        self,
        blocks: Sequence[Sequence[tuple[int, int]]],
        bondlengths: Sequence[Sequence[float]] | None = None,
        tolerance: float = 1e-13,
        maxiter: int = 100,
    ) -> None:
        """Set up the constraints.

        Parameters
        ----------
        blocks : sequence of sequence of (int, int)
            One list of atom pairs per molecule (molecules with different numbers of pairs are grouped
            automatically).
        bondlengths : sequence of sequence of float, optional
            One list of distances [Angstrom] per block; None: from the first configuration.
        tolerance : float
            Relative convergence tolerance of SHAKE.
        maxiter : int
            Newton iterations of SHAKE.
        """
        # molecules grouped by their number of constraints: {c: (M, c, 2) atom pairs} (a few arrays, so
        # that ASE's deep copies of the constraints stay cheap)
        groups, lengths = {}, {}
        for k, b in enumerate(blocks):
            b = np.asarray(b, int).reshape(-1, 2)
            if len(b):
                groups.setdefault(len(b), []).append(b)
                if bondlengths is not None:
                    lengths.setdefault(len(b), []).append(np.asarray(bondlengths[k], float))
        self.pairs = {c: np.array(v) for c, v in groups.items()}
        self.bondlengths = None if bondlengths is None else {c: np.array(v) for c, v in lengths.items()}
        self.tolerance, self.maxiter = float(tolerance), int(maxiter)
        self._groups = None

    @property
    def blocks(self) -> list[np.ndarray]:
        """The atom-pair lists of every molecule, grouped by size."""
        return [b for c in sorted(self.pairs) for b in self.pairs[c]]

    def get_removed_dof(self, atoms: Atoms) -> int:
        """Return the number of constraints (degrees of freedom removed)."""
        return int(sum(v.shape[0] * v.shape[1] for v in self.pairs.values()))

    def get_indices(self) -> np.ndarray:
        """Return the constrained atoms (unique, sorted)."""
        return np.unique(np.concatenate([v.ravel() for v in self.pairs.values()])) if self.pairs else np.zeros(0, int)

    def todict(self) -> dict:
        """Return the ASE dictionary form (blocks and tolerance)."""
        return {
            "name": "FixRigidMolecules",
            "kwargs": {"blocks": [b.tolist() for b in self.blocks], "tolerance": self.tolerance},
        }

    def index_shuffle(self, atoms: Atoms, ind: Sequence[int]) -> None:
        """Refuse slicing (ASE interface).

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("FixRigidMolecules does not support slicing")

    @staticmethod
    def _mic(d: np.ndarray, cell: ArrayLike, pbc: ArrayLike) -> np.ndarray:
        """Return displacements d (..., 3) by the minimum image of the cell (unchanged without pbc)."""
        if not np.any(pbc):
            return d
        C = np.asarray(cell)
        f = d @ np.linalg.inv(C)
        return d - np.round(f) @ C

    def _setup(self, atoms: Atoms) -> None:
        """Set the target lengths (if needed) and per group the constraint coupling matrices from the masses.

        C_kl = e(a_k, l) / m_{a_k} - e(b_k, l) / m_{b_k} with e(i, l) = [i == a_l] - [i == b_l]: how
        multiplier l moves bond vector k.
        """
        if self.bondlengths is None:
            x = atoms.positions
            self.bondlengths = {
                c: np.linalg.norm(self._mic(x[P[..., 0]] - x[P[..., 1]], atoms.cell, atoms.pbc), axis=-1)
                for c, P in self.pairs.items()
            }
        m = atoms.get_masses()
        self._groups = []
        for c, P in self.pairs.items():  # P: (M, c, 2)
            ds = self.bondlengths[c]
            a, b = P[..., 0], P[..., 1]

            # C_kl = e(a_k, l) / m_a_k - e(b_k, l) / m_b_k,  e(i, l) = [i == a_l] - [i == b_l]
            def e(i: np.ndarray) -> np.ndarray:
                """Return e(i, l) = [i == a_l] - [i == b_l] for atoms i (M, c)."""
                return (i[:, :, None] == a[:, None, :]).astype(float) - (i[:, :, None] == b[:, None, :])

            Cm = e(a) / m[a][:, :, None] - e(b) / m[b][:, :, None]
            self._groups.append((a, b, np.array(ds), Cm))
        self._masses = m

    def _check(self, atoms: Atoms) -> None:
        """Rebuild the setup when the masses (or the number of atoms) changed."""
        if self._groups is None or len(self._masses) != len(atoms) or np.any(self._masses != atoms.get_masses()):
            self._setup(atoms)

    def _spread(
        self, n: int, a: np.ndarray, b: np.ndarray, coef: np.ndarray, vec: np.ndarray, inv_m: bool
    ) -> np.ndarray:
        """Return the per-atom sum of +-coef_l vec_l (divided by the masses if inv_m), np.ndarray (n, 3).

        Uses bincount, not np.add.at (which is ~50x slower).
        """
        w = (coef[..., None] * vec).reshape(-1, 3)
        ia, ib = a.ravel(), b.ravel()
        out = np.stack([np.bincount(ia, w[:, j], n) - np.bincount(ib, w[:, j], n) for j in range(3)], 1)
        return out / self._masses[:, None] if inv_m else out

    def adjust_positions(self, atoms: Atoms, new: np.ndarray) -> None:
        """Adjust `new` positions in place so that every constrained distance is met (SHAKE).

        The corrections are along the old bond vectors s0: new += sum_l lam_l s0_l / m (+ for a_l,
        - for b_l), with lam from Newton iterations on |u_k(lam)|^2 = d_k^2.

        Raises
        ------
        RuntimeError
            If SHAKE does not converge in `maxiter` iterations.
        """
        self._check(atoms)
        old = atoms.positions
        n = len(atoms)
        for a, b, d, Cm in self._groups:
            r0 = old[a] - old[b]
            s0 = self._mic(r0, atoms.cell, atoms.pbc)  # (M, c, 3) constrained directions
            u0 = new[a] - new[b] - r0 + s0  # new bond vectors, minimum image as the old ones
            lam = np.zeros(d.shape)
            G = Cm[..., None] * s0[:, None, :, :]  # (M, c, c, 3): d u_k / d lam_l
            for _it in range(self.maxiter):
                u = u0 + np.einsum("mkl,mklx->mkx", np.broadcast_to(lam[:, None, :], Cm.shape), G)
                f = np.sum(u * u, -1) - d * d
                if np.max(np.abs(f) / (d * d)) < self.tolerance:
                    break
                J = 2.0 * np.einsum("mkx,mklx->mkl", u, G)
                lam = lam - np.linalg.solve(J, f[..., None])[..., 0]  # Newton step on the multipliers
            else:
                raise RuntimeError("FixRigidMolecules: SHAKE did not converge")
            new += self._spread(n, a, b, lam, s0, True)

    def adjust_momenta(self, atoms: Atoms, p: np.ndarray) -> None:
        """Remove the momentum components along the constraints in place (RATTLE: d/dt |r_ab|^2 = 0)."""
        self._check(atoms)
        x = atoms.positions
        m = self._masses
        v = p / m[:, None]
        n = len(atoms)
        for a, b, _d, Cm in self._groups:
            s = self._mic(x[a] - x[b], atoms.cell, atoms.pbc)
            dv = v[a] - v[b]
            # RATTLE: s_k . (dv_k + sum_l mu_l C_kl s_l) = 0 for every constraint k
            A = Cm * np.einsum("mkx,mlx->mkl", s, s)
            mu = np.linalg.solve(A, -np.sum(s * dv, -1)[..., None])[..., 0]
            p += self._spread(n, a, b, mu, s, False)

    def adjust_forces(self, atoms: Atoms, forces: np.ndarray) -> None:
        """Project the forces in place like momenta (no force along the constraints) and store the removed part."""
        self.constraint_forces = -forces.copy()
        self.adjust_momenta(atoms, forces)
        self.constraint_forces += forces


def rigid_blocks(sys: System) -> list[list[tuple[int, int]]]:
    """Return one list of atom pairs per molecule of up to three atoms (all intramolecular pairs).

    The constraints that hold the rigid-molecule model rigid.

    Raises
    ------
    ValueError
        For a molecule of more than three atoms.
    """
    blocks = []
    for k, m in enumerate(sys.molecules):
        off = int(sys.offsets[k])
        if m.n > 3:
            raise ValueError(
                f"molecule {k} ({m.name}) has {m.n} atoms: rigid molecules of more than three atoms "
                "cannot be held by distance constraints here; use flexible templates"
            )
        blocks.append([(off + i, off + j) for i in range(m.n) for j in range(i + 1, m.n)])
    return blocks


def rigid_constraints(sys: System, tolerance: float = 1e-13) -> FixRigidMolecules:
    """Return a constraint that keeps the molecules of `sys` (up to three atoms, e.g. water) rigid.

    At their current geometry (FixRigidMolecules: vectorised SHAKE / RATTLE).
    """
    return FixRigidMolecules(rigid_blocks(sys), tolerance=tolerance)
