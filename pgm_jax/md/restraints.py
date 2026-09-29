"""Restraints for MD: positional, distance, angle, dihedral and centre-of-mass distance terms
added to the force field in both MD drivers (`Simulation`: rigid bodies; `FlexibleSimulation`:
atoms).

    from pgm_jax.md.restraints import (Restraints, PositionRestraint, DistanceRestraint,
                                       DihedralRestraint, COMDistanceRestraint, harmonic, KCAL_A2)
    rs = Restraints([
        PositionRestraint(heavy, pos[heavy], k=1.0 * KCAL_A2, scaling="com", box=H),
        DistanceRestraint([[i, j]], bounds=(0.0, 0.25, 0.30, 0.50), k=2000.0),        # NOE-like
        DihedralRestraint([[a, b, c, d]], bounds=harmonic(np.radians(-60.0)), k=100.0),
        COMDistanceRestraint(ligand, pocket, bounds=harmonic(1.2), k=500.0, masses=sys.masses),  # umbrella
    ])
    sim = FlexibleSimulation(..., restraints=rs)        # or Simulation(..., restraints=rs)
    sim.observables()["erestraint"], sim.restraint_energies()      # total; by kind
    asys.position_restraints(k, "backbone", sim.positions_nm(), sim.state.box)   # proteins (protein/amber.py)

Force constants follow Amber (restraint_wt, NMR rk2 / rk3): E = k x^2, with no factor 1/2; a spring
constant K of E = K x^2 / 2 is k = K / 2.  Units kJ/mol/nm^2 (kJ/mol/rad^2 for angles and
dihedrals): 1 kcal/mol/A^2 = 418.4 kJ/mol/nm^2 (KCAL_A2), 1 kcal/mol/rad^2 = 4.184 kJ/mol/rad^2
(KCAL_RAD2).  Angles in rad (np.radians for Amber's degrees).

Forms:
  positional  E = k (|d| - r0)^2 for |d| > r0, else 0, with d = r - r_ref(H) (minimum image) and r0
              the flat-bottom radius (0: harmonic, Amber ntr = 1).
  distance, angle, dihedral, centre-of-mass distance: Amber's NMR flat-bottom form in the
              coordinate x (nm or rad), r1 <= r2 <= r3 <= r4, constants k2 (lower wall) and k3
              (upper wall):
                  x < r1          k2 [(r2 - r1)^2 + 2 (r2 - r1)(r1 - x)]   (linear, slope continuous)
                  r1 <= x < r2    k2 (x - r2)^2
                  r2 <= x <= r3   0
                  r3 < x <= r4    k3 (x - r3)^2
                  x > r4          k3 [(r4 - r3)^2 + 2 (r4 - r3)(x - r4)]
              r1 = -inf / r4 = inf leave out the linear parts; harmonic(x0) = (-inf, x0, x0, inf)
              gives E = k (x - x0)^2.  r1 = r2 removes the lower wall (as in Amber), r3 = r4 the upper.

Coordinates:
  - distances and the bond vectors of angles and dihedrals are minimum images in the box H
    (box.min_image: exact below half the smallest box height);
  - angles in [0, pi];
  - dihedrals in the IUPAC sign convention of pgm_jax.bonded.terms (as Amber): positive for a
    clockwise rotation of the front bond looking along j -> k.  They are periodic: phi is taken in
    [c - pi, c + pi), c = (r2 + r3) / 2, the image nearest the flat bottom, so a window may cross
    +-pi (r2 = 170 deg, r3 = 190 deg).  The energy jumps at the antipode c + pi unless both walls
    give the same energy there;
  - group centres (COMDistanceRestraint) are weighted means of the minimum-image displacements
    from the group's first atom, so a group must be smaller than half the box height.

Positional references and the box.  The Monte Carlo barostat scales the box and the molecular
centres; molecules are translated rigidly.  A reference given at the box H0 (`box`) follows the
box according to `scaling`:
  "none"        fixed Cartesian reference (GROMACS refcoord-scaling = no).  The restraint then
                pins absolute positions, and volume moves, which scale about the origin, pay for
                moving restrained molecules;
  "fractional"  every reference point in fractional coordinates, r_ref(H) = r_ref H0^-1 H (affine;
                GROMACS "all").  For single atoms (ions) and small molecules;
  "com"         the centroid c of the reference points (weights `weights`, default equal) in
                fractional coordinates, the points rigidly attached: r_ref(H) = r_ref - c + c H0^-1 H
                (GROMACS "com").  The choice for a macromolecule: a volume move shifts the molecule
                by (s - 1) R_com and its reference by (s - 1) c, so a molecule at its reference
                stays there (exactly when c is its centre of mass, to ~1e-5 nm otherwise).
Every Monte Carlo trial energy includes the restraints at the scaled positions and box (so at
the scaled references): the acceptance is exact for the box-dependent potential.  The pressure
(`Simulation.pressure`) adds the restraints' molecular strain derivative (`strain_derivative`).

Energies are float64 functions of (pos, H), jit-able and differentiable; the drivers take the
forces by autodiff (for rigid bodies, the atomic restraint forces are mapped to centre forces and
torques together with the force-field forces).  Parameters are compile-time constants: changing
them (`Simulation.set_restraints`) recompiles the step.  Units nm, rad, kJ/mol."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..units import KCAL
from .box import centers_of_mass, min_image

KCAL_A2 = 418.4  # kJ/mol/nm^2 per kcal/mol/A^2
KCAL_RAD2 = KCAL  # kJ/mol/rad^2 per kcal/mol/rad^2
_HI = jax.lax.Precision.HIGHEST
_TINY = 1e-60


def harmonic(x0):
    """Bounds (r1, r2, r3, r4) of a harmonic restraint E = k (x - x0)^2."""
    return (-np.inf, x0, x0, np.inf)


def _wall(d, w):
    """0 for d <= 0, d^2 for 0 < d <= w, then linear with a continuous slope, w^2 + 2 w (d - w).
    Written with a clipped d, so w = inf (no linear part) keeps values and gradients finite."""
    c = jnp.clip(d, 0.0, w)
    return c * c + 2.0 * c * (d - c)


def nmr_energy(x, r1, r2, r3, r4, k2, k3):
    """Amber's NMR flat-bottom restraint energy of the coordinate(s) x (see the module docstring)."""
    return k2 * _wall(r2 - x, r2 - r1) + k3 * _wall(x - r3, r4 - r3)


def _norm(v):
    """|v| over the last axis with a zero gradient at v = 0 (instead of NaN)."""
    s = jnp.sum(v * v, -1)
    return jnp.sqrt(jnp.maximum(s, _TINY))


def dihedral(x0, x1, x2, x3):
    """Dihedral angle x0-x1-x2-x3 (rad, [-pi, pi]) in the IUPAC sign convention (the formula of
    pgm_jax.bonded.terms.core._dihedral)."""
    return dihedral_from_bonds(x1 - x0, x2 - x1, x3 - x2)


def dihedral_from_bonds(b0, b1, b2):
    """Dihedral angle from the bond vectors b0 = x1 - x0, b1 = x2 - x1, b2 = x3 - x2."""
    n1, n2 = jnp.cross(b0, b1), jnp.cross(b1, b2)
    m1 = jnp.cross(n1, b1 / _norm(b1)[..., None])
    return jnp.arctan2(-jnp.sum(m1 * n2, -1), jnp.sum(n1 * n2, -1))


def _per_item(x, m, name):
    a = np.asarray(x, float)
    if a.ndim > 1 or (a.ndim == 1 and len(a) != m):
        raise ValueError(f"{name}: a scalar or one value per restraint ({m})")
    return np.broadcast_to(a, (m,)).copy()


# ----------------------------------------------------------------------------- terms
class Restraint:
    """A set of restraints of one kind.  energies(pos, H) -> (m,) kJ/mol, values(pos, H) -> (m,)
    the restrained coordinate, energy = sum of energies, atoms() every atom index used."""

    kind = ""

    def energies(self, pos, H):
        raise NotImplementedError

    def values(self, pos, H):
        raise NotImplementedError

    def atoms(self) -> np.ndarray:
        raise NotImplementedError

    def energy(self, pos, H):
        return jnp.sum(self.energies(pos, H))

    def describe(self) -> str:
        return f"{self.kind} {len(self)}"


class PositionRestraint(Restraint):
    """E = k (|d| - r0)^2 beyond the flat-bottom radius r0, d the minimum image of r_i - r_ref,i(H).
    atoms (n,), ref (n, 3) nm at the box `box` (needed unless scaling == "none"); k (kJ/mol/nm^2)
    and r0 (nm): scalars or per atom; scaling "none" | "fractional" | "com" (module docstring);
    weights: of the reference centroid for "com" (default equal)."""

    kind = "position"

    def __init__(self, atoms, ref, k, r0=0.0, scaling: str = "none", box=None, weights=None):
        self.idx = np.asarray(atoms, int).reshape(-1)
        n = len(self.idx)
        ref = np.asarray(ref, float)
        if ref.shape != (n, 3):
            raise ValueError(f"ref must be ({n}, 3) nm, got {ref.shape}")
        self.k, self.r0 = _per_item(k, n, "k"), _per_item(r0, n, "r0")
        if np.any(self.k < 0) or np.any(self.r0 < 0):
            raise ValueError("k and r0 must be >= 0")
        if scaling not in ("none", "fractional", "com"):
            raise ValueError("scaling: 'none' | 'fractional' | 'com'")
        self.scaling = scaling
        self.ref = ref
        if scaling != "none":
            if box is None:
                raise ValueError(f"scaling={scaling!r} needs the box of the reference (box=H0)")
            Hinv = np.linalg.inv(np.asarray(box, float))
            if scaling == "fractional":
                self.frac = ref @ Hinv
            else:
                w = np.ones(n) if weights is None else np.asarray(weights, float).reshape(-1)
                if w.shape != (n,) or np.any(w < 0) or w.sum() <= 0:
                    raise ValueError("weights: one non-negative value per atom, not all zero")
                c = (w[:, None] * ref).sum(0) / w.sum()
                self.offset, self.frac = ref - c, c @ Hinv

    def reference(self, H):
        """Reference positions (n, 3) nm at the box H."""
        if self.scaling == "none":
            return jnp.asarray(self.ref)
        f = jnp.matmul(jnp.asarray(self.frac), jnp.asarray(H, jnp.float64), precision=_HI)
        return f if self.scaling == "fractional" else jnp.asarray(self.offset) + f

    def displacements(self, pos, H):
        H = jnp.asarray(H, jnp.float64)
        return min_image(jnp.asarray(pos, jnp.float64)[self.idx] - self.reference(H), H)

    def values(self, pos, H):
        return _norm(self.displacements(pos, H))

    def energies(self, pos, H):
        return self.k * jnp.maximum(self.values(pos, H) - self.r0, 0.0) ** 2

    def atoms(self):
        return self.idx

    def __len__(self):
        return len(self.idx)

    def describe(self) -> str:
        k = f"{self.k[0]:g}" if np.all(self.k == self.k[0]) else f"{self.k.min():g}-{self.k.max():g}"
        return f"position {len(self)} atoms (k {k}, {self.scaling})"


class _NMRRestraint(Restraint):
    """Restraints of a coordinate of `n_atoms` atoms in Amber's NMR flat-bottom form.
    idx (m, n_atoms); bounds (r1, r2, r3, r4), each a scalar or one value per restraint; either k
    (both walls) or k2 (lower) and k3 (upper)."""

    n_atoms = 2
    periodic = False

    def __init__(self, idx, bounds, k=None, k2=None, k3=None):
        idx = np.asarray(idx, int)
        if idx.ndim == 1:
            idx = idx[None]
        if idx.ndim != 2 or idx.shape[1] != self.n_atoms:
            raise ValueError(f"{self.kind} restraints need {self.n_atoms} atoms each")
        self.idx = idx
        m = len(idx)
        if len(bounds) != 4:
            raise ValueError("bounds: (r1, r2, r3, r4)")
        r = [_per_item(b, m, f"r{i + 1}") for i, b in enumerate(bounds)]
        if k is not None:
            if k2 is not None or k3 is not None:
                raise ValueError("give k (both walls) or k2 and k3")
            k2 = k3 = k
        if k2 is None or k3 is None:
            raise ValueError("give k (both walls) or k2 and k3")
        self.k2, self.k3 = _per_item(k2, m, "k2"), _per_item(k3, m, "k3")
        if np.any(self.k2 < 0) or np.any(self.k3 < 0):
            raise ValueError("force constants must be >= 0")
        if not (np.all(np.isfinite(r[1])) and np.all(np.isfinite(r[2]))):
            raise ValueError("r2 and r3 must be finite (r1 = r2 or r3 = r4 remove a wall)")
        if not (np.all(r[0] <= r[1]) and np.all(r[1] <= r[2]) and np.all(r[2] <= r[3])):
            raise ValueError("bounds must satisfy r1 <= r2 <= r3 <= r4")
        if self.periodic and np.any(r[2] - r[1] > 2 * np.pi):
            raise ValueError("dihedral window r3 - r2 exceeds 2 pi")
        self.r = r

    def coordinate(self, x, H):
        raise NotImplementedError

    def values(self, pos, H):
        H = jnp.asarray(H, jnp.float64)
        x = jnp.asarray(pos, jnp.float64)
        v = self.coordinate([x[self.idx[:, a]] for a in range(self.n_atoms)], H)
        if self.periodic:  # the image in [c - pi, c + pi)
            c = 0.5 * (self.r[1] + self.r[2])
            v = c + jnp.mod(v - c + jnp.pi, 2 * jnp.pi) - jnp.pi
        return v

    def energies(self, pos, H):
        return nmr_energy(self.values(pos, H), *self.r, self.k2, self.k3)

    def atoms(self):
        return self.idx.reshape(-1)

    def __len__(self):
        return len(self.idx)


class DistanceRestraint(_NMRRestraint):
    """Distance |r_j - r_i| (minimum image, nm) of each pair (i, j)."""

    kind = "distance"
    n_atoms = 2

    def coordinate(self, x, H):
        return _norm(min_image(x[1] - x[0], H))


class AngleRestraint(_NMRRestraint):
    """Angle i-j-k (rad, [0, pi]) of each triple."""

    kind = "angle"
    n_atoms = 3

    def coordinate(self, x, H):
        u, v = min_image(x[0] - x[1], H), min_image(x[2] - x[1], H)
        return jnp.arctan2(_norm(jnp.cross(u, v)), jnp.sum(u * v, -1))


class DihedralRestraint(_NMRRestraint):
    """Dihedral i-j-k-l (rad, IUPAC sign, periodic about the window centre) of each quadruple."""

    kind = "dihedral"
    n_atoms = 4
    periodic = True

    def coordinate(self, x, H):
        return dihedral_from_bonds(min_image(x[1] - x[0], H), min_image(x[2] - x[1], H), min_image(x[3] - x[2], H))


class COMDistanceRestraint(_NMRRestraint):
    """Distance (nm) between the weighted centres of two atom groups (umbrella sampling); weights
    from `masses` (per-atom masses of the whole system, e.g. System.masses) or equal."""

    kind = "com_distance"

    def __init__(self, group_a, group_b, bounds, k=None, k2=None, k3=None, masses=None):
        self.ga, self.gb = (np.asarray(g, int).reshape(-1) for g in (group_a, group_b))
        if not len(self.ga) or not len(self.gb):
            raise ValueError("empty group")
        super().__init__(np.zeros((1, 2), int), bounds, k, k2, k3)
        w = (lambda g: np.ones(len(g))) if masses is None else (lambda g: np.asarray(masses, float)[g])
        self.wa, self.wb = w(self.ga), w(self.gb)
        if np.any(self.wa < 0) or np.any(self.wb < 0) or self.wa.sum() <= 0 or self.wb.sum() <= 0:
            raise ValueError("group weights must be non-negative and not all zero")

    @staticmethod
    def _centre(x, g, w, H):
        d = min_image(x[g] - x[g[0]], H)
        return x[g[0]] + jnp.sum(jnp.asarray(w)[:, None] * d, 0) / float(np.sum(w))

    def values(self, pos, H):
        H = jnp.asarray(H, jnp.float64)
        x = jnp.asarray(pos, jnp.float64)
        ca, cb = self._centre(x, self.ga, self.wa, H), self._centre(x, self.gb, self.wb, H)
        return _norm(min_image(cb - ca, H))[None]

    def atoms(self):
        return np.concatenate([self.ga, self.gb])

    def __len__(self):
        return 1

    def describe(self) -> str:
        return f"com_distance ({len(self.ga)} - {len(self.gb)} atoms)"


# ----------------------------------------------------------------------------- container
class Restraints:
    """Restraint terms applied together: energy(pos, H) (float64, jit-able), energies(pos, H) by
    kind, strain_derivative for the pressure.  `terms`: Restraint objects."""

    def __init__(self, terms=()):
        self.terms = []
        for t in terms:
            self.add(t)

    def add(self, term: Restraint) -> Restraints:
        if not isinstance(term, Restraint):
            raise TypeError(f"not a restraint: {term!r}")
        self.terms.append(term)
        return self

    def __len__(self):
        return len(self.terms)

    def __iter__(self):
        return iter(self.terms)

    def check(self, n_atoms: int) -> None:
        """Every atom index must exist in a system of n_atoms atoms."""
        for t in self.terms:
            a = t.atoms()
            if a.size and (a.min() < 0 or a.max() >= n_atoms):
                raise ValueError(f"{t.describe()}: atom index out of range for {n_atoms} atoms")

    @property
    def kinds(self) -> list:
        return list(dict.fromkeys(t.kind for t in self.terms))

    def energy(self, pos, H):
        """Total restraint energy (kJ/mol) at positions pos (N, 3) and box H (nm)."""
        e = jnp.zeros((), jnp.float64)
        for t in self.terms:
            e = e + t.energy(pos, H)
        return e

    def energies(self, pos, H) -> dict:
        """Restraint energy by kind (kJ/mol)."""
        out = {}
        for t in self.terms:
            out[t.kind] = out.get(t.kind, jnp.zeros((), jnp.float64)) + t.energy(pos, H)
        return out

    def forces(self, pos, H):
        """Atomic restraint forces (N, 3) kJ/mol/nm."""
        return -jax.grad(self.energy)(jnp.asarray(pos, jnp.float64), H)

    def strain_derivative(self, pos, H, mol, masses, nmol: int):
        """dE/d eps (3, 3) under the barostat's molecular scaling: molecular centres of mass
        (masses, molecule index mol per atom) and the box deformed by (1 + eps), molecules
        translated rigidly (as PGMForceField.strain_derivative)."""
        return molecular_strain(self.energy, pos, H, mol, masses, nmol)

    def describe(self) -> str:
        return ", ".join(t.describe() for t in self.terms)


def molecular_strain(energy, pos, H, mol, masses, nmol: int):
    """dE/d eps (3, 3) of energy(pos, H) under molecular scaling (see Restraints.strain_derivative)."""
    pos = jnp.asarray(pos, jnp.float64)
    H = jnp.asarray(H, jnp.float64)
    w = jnp.asarray(masses, jnp.float64)
    com = centers_of_mass(pos, w, mol, nmol)

    def e(eps):
        F = jnp.eye(3) + eps
        return energy(pos + jnp.matmul(com, eps.T, precision=_HI)[mol], jnp.matmul(H, F.T, precision=_HI))

    return jax.grad(e)(jnp.zeros((3, 3)))


def as_restraints(x) -> Restraints | None:
    """None, a Restraints container, one Restraint or a sequence of them -> Restraints (or None)."""
    if x is None or isinstance(x, Restraints):
        return x
    if isinstance(x, Restraint):
        return Restraints([x])
    return Restraints(list(x))
