"""ASE calculator for pGM (pgm_jax.interfaces.engine), periodic or gas phase.

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
induced_dipoles (N, 3, e Angstrom).  Extra results: "energy_terms" (eV:
elec, vdw, bonded, restraint) and "cg_iterations".

A stress asked for after the forces of the same configuration costs one strain derivative at the
converged dipoles (no second dipole solve).  Positions wrapped atom by atom are fine."""

from __future__ import annotations

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixConstraint
from ase.stress import full_3x3_to_voigt_6_stress

from .engine import GasPhaseEngine

KJMOL = units.kJ / units.mol  # eV per kJ/mol
NM = 10.0  # Angstrom per nm


class PGMCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress", "dipole", "induced_dipoles"]
    nolabel = True

    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.periodic = not isinstance(engine, GasPhaseEngine)
        self._res = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
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
            V = abs(np.linalg.det(np.asarray(self.atoms.get_cell()))) / NM**3
            r["stress"] = full_3x3_to_voigt_6_stress(W / V) * (KJMOL / NM**3)
        if any(p in properties for p in ("dipole", "induced_dipoles")):
            self._dipoles()

    def _dipoles(self):
        res, r = self._res, self.results
        r["dipole_components"] = res.dipole_components * NM
        r["dipole"] = r["dipole_components"].sum(0)
        r["induced_dipoles"] = res.induced_dipoles * NM

    def get_induced_dipoles(self, atoms=None):
        """(N, 3) induced dipoles, e Angstrom."""
        return self.get_property("induced_dipoles", atoms)


# ----------------------------------------------------------------------------- helpers
def _light_copy(atoms):
    """atoms.copy() without the constraints (enough for the calculator's change checks)."""
    new = atoms.__class__(cell=atoms.cell, pbc=atoms.pbc, info=atoms.info, celldisp=atoms._celldisp.copy())
    new.arrays = {k: v.copy() for k, v in atoms.arrays.items()}
    return new


def atoms_from_system(sys, pos_nm, H_nm=None) -> Atoms:
    """ase.Atoms for a pgm_jax System: element symbols, the system's masses (amu), positions
    (Angstrom) and, with H_nm (rows = lattice vectors), the cell with pbc."""
    symbols = [e for m in sys.molecules for e in m.elements]
    a = Atoms(symbols=symbols, positions=np.asarray(pos_nm, float) * NM, masses=np.asarray(sys.masses, float))
    if H_nm is not None:
        a.set_cell(np.asarray(H_nm, float) * NM)
        a.set_pbc(True)
    return a


class FixRigidMolecules(FixConstraint):
    """Distance constraints of small rigid molecules (water: O-H, O-H, H-H), solved exactly and
    vectorised over molecules: SHAKE for positions by Newton iterations on the (up to 3) Lagrange
    multipliers of each molecule, RATTLE for momenta by one batched linear solve.  ASE's
    FixBondLengths does the same pair by pair in Python loops (about 1 ms per water per step).

    blocks: one list of atom pairs per molecule (all molecules of a group must have the same number
    of pairs; mixed sizes are grouped automatically).  bondlengths: None (from the first
    configuration) or one list per block."""

    def __init__(self, blocks, bondlengths=None, tolerance: float = 1e-13, maxiter: int = 100):
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
    def blocks(self):
        return [b for c in sorted(self.pairs) for b in self.pairs[c]]

    def get_removed_dof(self, atoms):
        return int(sum(v.shape[0] * v.shape[1] for v in self.pairs.values()))

    def get_indices(self):
        return np.unique(np.concatenate([v.ravel() for v in self.pairs.values()])) if self.pairs else np.zeros(0, int)

    def todict(self):
        return {
            "name": "FixRigidMolecules",
            "kwargs": {"blocks": [b.tolist() for b in self.blocks], "tolerance": self.tolerance},
        }

    def index_shuffle(self, atoms, ind):
        raise NotImplementedError("FixRigidMolecules does not support slicing")

    @staticmethod
    def _mic(d, cell, pbc):
        if not np.any(pbc):
            return d
        C = np.asarray(cell)
        f = d @ np.linalg.inv(C)
        return d - np.round(f) @ C

    def _setup(self, atoms):
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
            def e(i):
                return (i[:, :, None] == a[:, None, :]).astype(float) - (i[:, :, None] == b[:, None, :])

            Cm = e(a) / m[a][:, :, None] - e(b) / m[b][:, :, None]
            self._groups.append((a, b, np.array(ds), Cm))
        self._masses = m

    def _check(self, atoms):
        if self._groups is None or len(self._masses) != len(atoms) or np.any(self._masses != atoms.get_masses()):
            self._setup(atoms)

    def _spread(self, n, a, b, coef, vec, inv_m):
        """Per-atom sum of +-coef_l vec_l (divided by the masses if inv_m); bincount, not np.add.at
        (which is ~50x slower)."""
        w = (coef[..., None] * vec).reshape(-1, 3)
        ia, ib = a.ravel(), b.ravel()
        out = np.stack([np.bincount(ia, w[:, j], n) - np.bincount(ib, w[:, j], n) for j in range(3)], 1)
        return out / self._masses[:, None] if inv_m else out

    def adjust_positions(self, atoms, new):
        self._check(atoms)
        old = atoms.positions
        n = len(atoms)
        for a, b, d, Cm in self._groups:
            r0 = old[a] - old[b]
            s0 = self._mic(r0, atoms.cell, atoms.pbc)  # (M, c, 3) constrained directions
            u0 = new[a] - new[b] - r0 + s0
            lam = np.zeros(d.shape)
            G = Cm[..., None] * s0[:, None, :, :]  # (M, c, c, 3): d u_k / d lam_l
            for _it in range(self.maxiter):
                u = u0 + np.einsum("mkl,mklx->mkx", np.broadcast_to(lam[:, None, :], Cm.shape), G)
                f = np.sum(u * u, -1) - d * d
                if np.max(np.abs(f) / (d * d)) < self.tolerance:
                    break
                J = 2.0 * np.einsum("mkx,mklx->mkl", u, G)
                lam = lam - np.linalg.solve(J, f[..., None])[..., 0]
            else:
                raise RuntimeError("FixRigidMolecules: SHAKE did not converge")
            new += self._spread(n, a, b, lam, s0, True)

    def adjust_momenta(self, atoms, p):
        self._check(atoms)
        x = atoms.positions
        m = self._masses
        v = p / m[:, None]
        n = len(atoms)
        for a, b, _d, Cm in self._groups:
            s = self._mic(x[a] - x[b], atoms.cell, atoms.pbc)
            dv = v[a] - v[b]
            A = Cm * np.einsum("mkx,mlx->mkl", s, s)
            mu = np.linalg.solve(A, -np.sum(s * dv, -1)[..., None])[..., 0]
            p += self._spread(n, a, b, mu, s, False)

    def adjust_forces(self, atoms, forces):
        self.constraint_forces = -forces.copy()
        self.adjust_momenta(atoms, forces)
        self.constraint_forces += forces


def rigid_blocks(sys):
    """One list of atom pairs per molecule of up to three atoms (all intramolecular pairs): the
    constraints that hold the rigid-molecule model rigid; larger molecules raise ValueError."""
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


def rigid_constraints(sys, tolerance: float = 1e-13) -> FixRigidMolecules:
    """Constraint that keeps the molecules of `sys` (up to three atoms, e.g. water) rigid at their
    current geometry (FixRigidMolecules: vectorised SHAKE / RATTLE)."""
    return FixRigidMolecules(rigid_blocks(sys), tolerance=tolerance)
