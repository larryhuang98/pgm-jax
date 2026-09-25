"""Molecules, systems and their parameters.

Topology and parameters are kept apart so that energies can be differentiated with respect to
coordinates, parameters and the box alike:

  Molecule    template: elements, atom types, covalent-dipole topology, bonds, masses, initial
              parameter values, and a tying key for every parameter (defaults below).
  ParamTable  the free parameters: for each quantity an ordered list of keys, one value per key.
              Atoms (or covalent dipoles) with the same key share one value and one gradient.
              `table.initial()` is the parameter pytree (dict of arrays) that energy functions
              take and JAX differentiates.
  System      an ordered list of molecules flattened into static index arrays, including, per
              quantity, the position of every atom's value in the table.
              `sys.expand(params)` gives per-atom arrays (a differentiable gather).

Quantities (units nm, e, kJ/mol):
  q             Gaussian charge (e)                                            per atom
  radius        pGM Gaussian radius R (nm; prmtop POL_GAUSS_RADII)              per atom
  alpha         isotropic polarizability (nm^3)                                per atom
  cov           covalent dipole strength c (e nm): p_i += c unit(r_j - r_i)    per covalent dipole
  lj_rmin_half  Lennard-Jones R* = r_min / 2 (nm)                               per atom
  lj_sqrt_eps   square root of the LJ well depth (sqrt(kJ/mol))               per atom
  gvdw_sqrt_a   GVDW: square root of the repulsion amplitude A (sqrt(kJ/mol))  per atom
  gvdw_sqrt_c6  GVDW: square root of the dispersion C6 (sqrt(kJ/mol nm^6))    per atom
  gvdw_b        GVDW: repulsion exponent scale b (dimensionless)              per atom
  quad          Gaussian quadrupole strength t (e nm^2) of one term of the     per quadrupole term
                covalent quadrupole basis: Theta_i += t S(u_ij, u_ik)
                (multipole.py; j == k: uniaxial along the covalent vector)
LJ pairs combine as r_min = R*_i + R*_j and eps = sqrt(eps_i) sqrt(eps_j) (Lorentz-Berthelot,
Amber's rule).  The square root is the parameter so that gradients stay finite at eps = 0.
GVDW pairs combine as A_ij = a_i a_j, C6_ij = c_i c_j (a, c the square roots) and b_ij =
(b_i + b_j) / 2, with the Gaussian pair exponent of the electrostatics (vdw.py).

Default tying (pGM practice): radius, alpha and LJ by atom type (as the pGM-pol table and Amber
LJ types); q and covalent dipoles per molecule, with symmetry-equivalent atoms tied (as py_resp
with equivalencing).  Equivalent atoms come from colour refinement on the graph of bonds and
covalent-dipole pairs, with element and type as initial colours; the classes do not depend on
the atom order, so every copy or reordering of a molecule gets the same keys.  Override any
quantity with `Molecule(keys={quantity: [key per atom or per covalent dipole]})`.

pGM conventions (Wei et al. JCP 2020; Wang et al. JCTC 2019): all atom pairs interact in the
electrostatics (no 1-2/1-3 masking); permanent dipoles are covalent dipoles along covalent
basis vectors to bonded or virtually bonded atoms.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

ATOM_QUANTITIES = ("q", "radius", "alpha", "lj_rmin_half", "lj_sqrt_eps", "gvdw_sqrt_a", "gvdw_sqrt_c6", "gvdw_b")
TERM_QUANTITIES = ("cov", "quad")                                   # one value per covalent dipole / quadrupole term
QUANTITIES = ATOM_QUANTITIES + TERM_QUANTITIES
BY_TYPE = ("radius", "alpha", "lj_rmin_half", "lj_sqrt_eps", "gvdw_sqrt_a", "gvdw_sqrt_c6", "gvdw_b")   # default: tied by atom type
BY_MOLECULE = ("q",)                                                 # default: per molecule, symmetry-tied

MASSES = {"H": 1.008, "Li": 6.94, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990,
          "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Br": 79.904, "Rb": 85.468, "I": 126.90,
          "Cs": 132.91}


@dataclass
class Molecule:
    name: str
    elements: list[str]                       # e.g. ["O", "H", "H"]
    types: list[str]                          # atom types, e.g. ["OW", "HW", "HW"]
    q: np.ndarray                             # (m,) e                      initial values
    radius: np.ndarray                        # (m,) nm
    alpha: np.ndarray                         # (m,) nm^3
    cov: list[tuple[int, int, float]] = field(default_factory=list)    # (i, j, c) local indices, c in e nm
    lj_rmin_half: np.ndarray | None = None    # (m,) nm              None -> 0 (no LJ)
    lj_sqrt_eps: np.ndarray | None = None     # (m,) sqrt(kJ/mol)    None -> 0
    bonds: list[tuple[int, int]] = field(default_factory=list)
    gvdw_sqrt_a: np.ndarray | None = None     # (m,) sqrt(kJ/mol)          None -> 0 (no GVDW)
    gvdw_sqrt_c6: np.ndarray | None = None    # (m,) sqrt(kJ/mol nm^6)     None -> 0
    gvdw_b: np.ndarray | None = None          # (m,) dimensionless          None -> 1
    quad: list[tuple[int, int, int, float]] = field(default_factory=list)   # (i, j, k, t), t in e nm^2
    masses: np.ndarray | None = None          # (m,) amu             None -> element masses
    keys: dict[str, list[str]] = field(default_factory=dict)          # tying-key overrides per quantity
    extra: dict = field(default_factory=dict)  # per-atom arrays for later channels

    def __post_init__(self):
        m = len(self.elements)
        self.elements, self.types = list(self.elements), list(self.types)
        self.q = np.asarray(self.q, float).reshape(m)
        self.radius = np.asarray(self.radius, float).reshape(m)
        self.alpha = np.asarray(self.alpha, float).reshape(m)
        self.lj_rmin_half = np.zeros(m) if self.lj_rmin_half is None else np.asarray(self.lj_rmin_half, float).reshape(m)
        self.lj_sqrt_eps = np.zeros(m) if self.lj_sqrt_eps is None else np.asarray(self.lj_sqrt_eps, float).reshape(m)
        self.gvdw_sqrt_a = np.zeros(m) if self.gvdw_sqrt_a is None else np.asarray(self.gvdw_sqrt_a, float).reshape(m)
        self.gvdw_sqrt_c6 = np.zeros(m) if self.gvdw_sqrt_c6 is None else np.asarray(self.gvdw_sqrt_c6, float).reshape(m)
        self.gvdw_b = np.ones(m) if self.gvdw_b is None else np.asarray(self.gvdw_b, float).reshape(m)
        self.quad = [(int(i), int(j), int(k), float(t)) for i, j, k, t in self.quad]
        self.masses = (np.array([MASSES[e] for e in self.elements]) if self.masses is None
                       else np.asarray(self.masses, float).reshape(m))
        self.cov = [(int(i), int(j), float(c)) for i, j, c in self.cov]
        self.bonds = [(int(i), int(j)) for i, j in self.bonds]
        assert len(self.types) == m

    def __setstate__(self, state):
        # molecules pickled before the GVDW and quadrupole quantities existed
        self.__dict__.update(state)
        m = len(self.elements)
        for name, fill in (("gvdw_sqrt_a", 0.0), ("gvdw_sqrt_c6", 0.0), ("gvdw_b", 1.0)):
            if getattr(self, name, None) is None:
                setattr(self, name, np.full(m, fill))
        if getattr(self, "quad", None) is None:
            self.quad = []

    @property
    def n(self) -> int:
        return len(self.elements)

    @property
    def charge(self) -> float:
        return float(np.sum(self.q))

    def values(self, quantity: str) -> np.ndarray:
        """Initial values of one quantity: per atom, or per covalent dipole for 'cov'."""
        if quantity == "cov":
            return np.array([c for _, _, c in self.cov], float)
        if quantity == "quad":
            return np.array([t for *_, t in self.quad], float)
        return getattr(self, quantity)

    def n_terms(self, quantity: str) -> int:
        return {"cov": len(self.cov), "quad": len(self.quad)}.get(quantity, self.n)

    def symmetry_classes(self) -> np.ndarray:
        """Canonical class id per atom (colour refinement; independent of the atom order)."""
        adj = [set() for _ in range(self.n)]
        for i, j in list(self.bonds) + [(i, j) for i, j, _ in self.cov]:
            if i != j:
                adj[i].add(j)
                adj[j].add(i)

        def relabel(sig):
            order = {s: k for k, s in enumerate(sorted(set(sig)))}
            return [order[s] for s in sig]

        colour = relabel([f"{e}|{t}" for e, t in zip(self.elements, self.types)])
        while True:
            new = relabel([(colour[i], tuple(sorted(colour[k] for k in adj[i]))) for i in range(self.n)])
            if len(set(new)) == len(set(colour)):
                return np.array(colour)
            colour = new

    def tying_keys(self) -> dict[str, list[str]]:
        """Key per atom (per covalent dipole for 'cov') for every quantity."""
        cls = self.symmetry_classes()
        label = {}
        for t in set(self.types):                         # classes of one type, in canonical order
            cs = sorted({int(c) for c, tt in zip(cls, self.types) if tt == t})
            for k, c in enumerate(cs):
                label[c] = t if len(cs) == 1 else f"{t}.{k}"
        atom = [label[int(c)] for c in cls]
        keys = {qn: list(self.types) for qn in BY_TYPE}
        keys.update({qn: [f"{self.name}:{a}" for a in atom] for qn in BY_MOLECULE})
        keys["cov"] = [f"{self.name}:{atom[i]}>{atom[j]}" for i, j, _ in self.cov]
        keys["quad"] = [f"{self.name}:Q:{atom[i]}>{atom[j]}" if j == k else
                        f"{self.name}:Q:{atom[i]}>" + "|".join(sorted((atom[j], atom[k]))) for i, j, k, _ in self.quad]
        for qn, ks in self.keys.items():
            if qn not in QUANTITIES:
                raise KeyError(f"unknown quantity {qn!r}")
            if len(ks) != self.n_terms(qn):
                raise ValueError(f"{self.name}: {len(ks)} keys for {qn}")
            keys[qn] = list(ks)
        return keys


class ParamTable:
    """Tied parameters: for each quantity an ordered list of keys and an initial value per key
    (the mean over the atoms that share the key; a spread larger than `tol` is reported)."""

    def __init__(self, molecules: list[Molecule], tol: float = 1e-6):
        members = {qn: {} for qn in QUANTITIES}
        seen = set()
        for m in molecules:
            if id(m) in seen:
                continue
            seen.add(id(m))
            keys = m.tying_keys()
            for qn in QUANTITIES:
                for k, v in zip(keys[qn], m.values(qn)):
                    members[qn].setdefault(k, []).append(float(v))
        self.keys = {qn: list(members[qn]) for qn in QUANTITIES}
        self._pos = {qn: {k: i for i, k in enumerate(self.keys[qn])} for qn in QUANTITIES}
        self.values0 = {qn: np.array([np.mean(members[qn][k]) for k in self.keys[qn]], float) for qn in QUANTITIES}
        self.spread = {qn: np.array([np.ptp(members[qn][k]) for k in self.keys[qn]], float) for qn in QUANTITIES}
        bad = [(qn, k, s) for qn in QUANTITIES for k, s, v in zip(self.keys[qn], self.spread[qn], self.values0[qn])
               if s > tol * max(1.0, abs(v))]
        if bad:
            warnings.warn("tied values differ (quantity, key, spread): " + ", ".join(f"{a} {b} {c:.3g}" for a, b, c in bad[:10]))

    def initial(self) -> dict[str, jnp.ndarray]:
        return {qn: jnp.asarray(v) for qn, v in self.values0.items()}

    def index(self, quantity: str, keys: list[str]) -> np.ndarray:
        try:
            return np.array([self._pos[quantity][k] for k in keys], dtype=np.int32)
        except KeyError as e:
            raise KeyError(f"{quantity} key {e} is not in this ParamTable (build it from all molecules)") from None

    def named(self, params) -> dict[str, dict[str, float]]:
        """{quantity: {key: value}} for inspection."""
        return {qn: dict(zip(self.keys[qn], np.asarray(params[qn]).tolist())) for qn in QUANTITIES}

    def sizes(self) -> dict[str, int]:
        return {qn: len(self.keys[qn]) for qn in QUANTITIES}

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for qn in QUANTITIES:
            h.update("|".join(self.keys[qn]).encode())
            h.update(np.ascontiguousarray(self.values0[qn]).tobytes())
        return h.hexdigest()


class System:
    """Flattened, static description of a set of molecules (topology and parameter indices).
    Coordinates, parameters and box are passed to the energy functions, never stored here."""

    def __init__(self, molecules: list[Molecule], table: ParamTable | None = None):
        self.molecules = molecules
        self.table = ParamTable(molecules) if table is None else table
        offs = np.cumsum([0] + [m.n for m in molecules])
        self.offsets = offs
        self.n = int(offs[-1])
        self.nmol = len(molecules)
        self.mol = np.concatenate([np.full(m.n, k, dtype=np.int32) for k, m in enumerate(molecules)])
        self.elements = [e for m in molecules for e in m.elements]
        self.types = [t for m in molecules for t in m.types]
        self.masses = np.concatenate([m.masses for m in molecules])
        keys = {}
        for m in molecules:                                  # tying keys are computed once per template
            if id(m) not in keys:
                keys[id(m)] = m.tying_keys()
        self.idx = {qn: np.concatenate([self.table.index(qn, keys[id(m)][qn]) for m in molecules]).astype(np.int32)
                    for qn in QUANTITIES}
        ci, cj = [], []
        for k, m in enumerate(molecules):
            for i, j, _ in m.cov:
                ci.append(offs[k] + i)
                cj.append(offs[k] + j)
        self.cov_i = np.array(ci, dtype=np.int32)
        self.cov_j = np.array(cj, dtype=np.int32)
        qt = [(offs[k] + i, offs[k] + j, offs[k] + l) for k, m in enumerate(molecules) for i, j, l, _ in m.quad]
        self.quad_ijk = np.array(qt, dtype=np.int32).reshape(-1, 3)
        self._pairs = None                                    # all pairs: built on first use (gas phase)
        extra_keys = set().union(*[m.extra.keys() for m in molecules]) if molecules else set()
        self.extra = {k: np.concatenate([np.asarray(m.extra[k], float) for m in molecules]) for k in extra_keys
                      if all(k in m.extra for m in molecules)}
        self.params0 = self.table.initial()

    # ------------------------------------------------------------------ all pairs (gas-phase models)
    def _all_pairs(self):
        if getattr(self, "_pairs", None) is None:
            ii, jj = np.triu_indices(self.n, k=1)
            self._pairs = (ii, jj, self.mol[ii] != self.mol[jj])
        return self._pairs

    @property
    def pair_i(self) -> np.ndarray:
        return self._all_pairs()[0]

    @property
    def pair_j(self) -> np.ndarray:
        return self._all_pairs()[1]

    @property
    def pair_inter(self) -> np.ndarray:
        return self._all_pairs()[2]

    def expand(self, params=None) -> dict[str, jnp.ndarray]:
        """Per-atom (and per-covalent-dipole) parameter arrays from the tied tables; `None` gives
        the initial values.  Differentiable in `params`."""
        P = self.params0 if params is None else params
        return {qn: jnp.asarray(P[qn])[self.idx[qn]] for qn in QUANTITIES}

    def atom_slice(self, k: int) -> slice:
        return slice(int(self.offsets[k]), int(self.offsets[k + 1]))

    def sub(self, mols: tuple[int, ...]) -> tuple["System", np.ndarray]:
        """Subsystem of the given molecules (same ParamTable) and the atom index map into this system."""
        idx = np.concatenate([np.arange(self.offsets[k], self.offsets[k + 1]) for k in mols])
        return System([self.molecules[k] for k in mols], table=self.table), idx

    def fingerprint(self) -> str:
        """Hash of topology, parameter indices and the table (cache key for compiled functions;
        parameter *values* passed at call time are not part of it)."""
        h = hashlib.sha1()
        for a in (self.mol, self.cov_i, self.cov_j, self.quad_ijk, *[self.idx[qn] for qn in QUANTITIES]):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update("|".join(self.elements).encode())
        h.update(self.table.fingerprint().encode())
        for k in sorted(self.extra):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.extra[k]).tobytes())
        return h.hexdigest()

    def signature(self) -> str:
        return "+".join(m.name for m in self.molecules)

    def __repr__(self) -> str:
        return f"System({self.signature()}, n={self.n})"
