"""Describe molecules, systems and their tied parameters.

Contents: Molecule (a molecule template with initial parameter values), ParamTable (the tied
free parameters), System (molecules flattened into static index arrays), and the quantity
tables ATOM_QUANTITIES, TERM_QUANTITIES, QUANTITIES, BY_TYPE, BY_MOLECULE, MASSES.

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

pGM conventions ([1]_, [2]_): all atom pairs interact in the
electrostatics (no 1-2/1-3 masking); permanent dipoles are covalent dipoles along covalent
basis vectors to bonded or virtually bonded atoms.

Virtual sites (md/vsites.py): massless atoms of a molecule (element "EP", Amber's extra points,
gets mass 0) whose positions are functions of other atoms, listed in `Molecule.vsites`; they carry
parameters like any atom.  Gas-phase and Ewald models take their positions as given; the MD
engines place them and spread their forces.

    water = Molecule("wat", ["O", "H", "H"], ["OW", "HW", "HW"], q, radius, alpha, cov=[(0, 1, c), ...])
    system = System([water] * 512)             # one template, 512 copies, one ParamTable
    params = system.table.initial()            # {quantity: (K,) array}, the differentiable pytree
    per_atom = system.expand(params)           # {quantity: (N,) or (terms,) array}

Units: nm, e, e nm, e nm^2, nm^3, kJ/mol, amu (see the quantity table).

References
----------
.. [1] H. Wei, R. Qi, J. Wang, P. Cieplak, Y. Duan, R. Luo, J. Chem. Phys. 153, 114116 (2020).
.. [2] J. Wang, P. Cieplak, R. Luo, Y. Duan, J. Chem. Theory Comput. 15, 1146 (2019).
       doi:10.1021/acs.jctc.8b00603

See also docs/model_options.md, docs/virtual_sites.md.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

ATOM_QUANTITIES = ("q", "radius", "alpha", "lj_rmin_half", "lj_sqrt_eps", "gvdw_sqrt_a", "gvdw_sqrt_c6", "gvdw_b")
TERM_QUANTITIES = ("cov", "quad")  # one value per covalent dipole / quadrupole term
QUANTITIES = ATOM_QUANTITIES + TERM_QUANTITIES
BY_TYPE = (
    "radius",
    "alpha",
    "lj_rmin_half",
    "lj_sqrt_eps",
    "gvdw_sqrt_a",
    "gvdw_sqrt_c6",
    "gvdw_b",
)  # default: tied by atom type
BY_MOLECULE = ("q",)  # default: per molecule, symmetry-tied

MASSES = {
    "H": 1.008,
    "Li": 6.94,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.990,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.098,
    "Br": 79.904,
    "Rb": 85.468,
    "I": 126.90,
    "Cs": 132.91,
    "EP": 0.0,
}  # EP: extra point (virtual site), massless


@dataclass
class Molecule:
    """Molecule template: topology, initial parameter values and parameter tying keys.

    A Molecule is a mutable dataclass (not a pytree); systems refer to templates by identity, so
    512 copies of one Molecule object in a System share one set of tying keys and one template in
    the MD engines.  `__post_init__` normalises the arrays; per-atom quantities have length m (the
    number of atoms), per-term quantities one value per covalent dipole or quadrupole term.

    Parameters
    ----------
    name : str
        Molecule name; prefixes the per-molecule tying keys (q, cov, quad).
    elements : list of str (m,)
        Element symbols (keys of MASSES for default masses; "EP" for a massless extra point).
    types : list of str (m,)
        Atom types (tying keys of the by-type quantities).
    q : ArrayLike (m,)
        Initial Gaussian charges [e].
    radius : ArrayLike (m,)
        Initial pGM Gaussian radii R [nm].
    alpha : ArrayLike (m,)
        Initial isotropic polarizabilities [nm^3].
    cov : list of (int, int, float)
        Covalent dipoles (i, j, c): a dipole c unit(r_j - r_i) on atom i, local indices, c [e nm].
    lj_rmin_half : ArrayLike (m,), optional
        Lennard-Jones R* = r_min / 2 [nm]; None: zeros (no LJ).
    lj_sqrt_eps : ArrayLike (m,), optional
        Square root of the LJ well depth [sqrt(kJ/mol)]; None: zeros.
    bonds : list of (int, int)
        Bonds (local indices); used for symmetry classes and topology, not for electrostatic masking.
    gvdw_sqrt_a : ArrayLike (m,), optional
        GVDW: square root of the repulsion amplitude [sqrt(kJ/mol)]; None: zeros (no GVDW).
    gvdw_sqrt_c6 : ArrayLike (m,), optional
        GVDW: square root of C6 [sqrt(kJ/mol nm^6)]; None: zeros.
    gvdw_b : ArrayLike (m,), optional
        GVDW repulsion exponent scale (dimensionless); None: ones.
    quad : list of (int, int, int, float)
        Covalent quadrupole terms (i, j, k, t), t [e nm^2] (multipole.py).
    masses : ArrayLike (m,), optional
        Atomic masses [amu]; None: element masses from MASSES.
    keys : dict of str to list of str
        Tying-key overrides {quantity: key per atom, or per term for "cov" / "quad"}.
    extra : dict
        Per-atom arrays for later channels (concatenated by System when every molecule has them).
    vsites : list
        Virtual sites (md/vsites.py VirtualSite objects with local indices).
    """

    name: str
    elements: list[str]  # e.g. ["O", "H", "H"]
    types: list[str]  # atom types, e.g. ["OW", "HW", "HW"]
    q: np.ndarray  # (m,) e                      initial values
    radius: np.ndarray  # (m,) nm
    alpha: np.ndarray  # (m,) nm^3
    cov: list[tuple[int, int, float]] = field(default_factory=list)  # (i, j, c) local indices, c in e nm
    lj_rmin_half: np.ndarray | None = None  # (m,) nm              None -> 0 (no LJ)
    lj_sqrt_eps: np.ndarray | None = None  # (m,) sqrt(kJ/mol)    None -> 0
    bonds: list[tuple[int, int]] = field(default_factory=list)
    gvdw_sqrt_a: np.ndarray | None = None  # (m,) sqrt(kJ/mol)          None -> 0 (no GVDW)
    gvdw_sqrt_c6: np.ndarray | None = None  # (m,) sqrt(kJ/mol nm^6)     None -> 0
    gvdw_b: np.ndarray | None = None  # (m,) dimensionless          None -> 1
    quad: list[tuple[int, int, int, float]] = field(default_factory=list)  # (i, j, k, t), t in e nm^2
    masses: np.ndarray | None = None  # (m,) amu             None -> element masses
    keys: dict[str, list[str]] = field(default_factory=dict)  # tying-key overrides per quantity
    extra: dict = field(default_factory=dict)  # per-atom arrays for later channels
    vsites: list = field(default_factory=list)  # virtual sites (md/vsites.py VirtualSite, local indices)

    def __post_init__(self) -> None:
        """Normalise the fields: lists, float arrays of length m, defaults for None, int/float tuples.

        Raises
        ------
        AssertionError
            If `types` does not have one entry per element.
        """
        m = len(self.elements)
        self.elements, self.types = list(self.elements), list(self.types)
        self.q = np.asarray(self.q, float).reshape(m)
        self.radius = np.asarray(self.radius, float).reshape(m)
        self.alpha = np.asarray(self.alpha, float).reshape(m)
        self.lj_rmin_half = (
            np.zeros(m) if self.lj_rmin_half is None else np.asarray(self.lj_rmin_half, float).reshape(m)
        )
        self.lj_sqrt_eps = np.zeros(m) if self.lj_sqrt_eps is None else np.asarray(self.lj_sqrt_eps, float).reshape(m)
        self.gvdw_sqrt_a = np.zeros(m) if self.gvdw_sqrt_a is None else np.asarray(self.gvdw_sqrt_a, float).reshape(m)
        self.gvdw_sqrt_c6 = (
            np.zeros(m) if self.gvdw_sqrt_c6 is None else np.asarray(self.gvdw_sqrt_c6, float).reshape(m)
        )
        self.gvdw_b = np.ones(m) if self.gvdw_b is None else np.asarray(self.gvdw_b, float).reshape(m)
        self.quad = [(int(i), int(j), int(k), float(t)) for i, j, k, t in self.quad]
        self.masses = (
            np.array([MASSES[e] for e in self.elements])
            if self.masses is None
            else np.asarray(self.masses, float).reshape(m)
        )
        self.cov = [(int(i), int(j), float(c)) for i, j, c in self.cov]
        self.bonds = [(int(i), int(j)) for i, j in self.bonds]
        self.vsites = list(self.vsites or [])
        assert len(self.types) == m

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled Molecule, filling the GVDW, quadrupole and virtual-site fields of old pickles."""
        # molecules pickled before the GVDW and quadrupole quantities existed
        self.__dict__.update(state)
        m = len(self.elements)
        for name, fill in (("gvdw_sqrt_a", 0.0), ("gvdw_sqrt_c6", 0.0), ("gvdw_b", 1.0)):
            if getattr(self, name, None) is None:
                setattr(self, name, np.full(m, fill))
        if getattr(self, "quad", None) is None:
            self.quad = []
        if getattr(self, "vsites", None) is None:
            self.vsites = []

    @property
    def n(self) -> int:
        """Number of atoms (virtual sites included)."""
        return len(self.elements)

    @property
    def charge(self) -> float:
        """Net initial charge [e]."""
        return float(np.sum(self.q))

    def values(self, quantity: str) -> np.ndarray:
        """Return the initial values of one quantity.

        Parameters
        ----------
        quantity : str
            A name of QUANTITIES.

        Returns
        -------
        np.ndarray
            Per atom (m,), per covalent dipole for "cov", per quadrupole term for "quad"; units of the
            quantity (module docstring).
        """
        if quantity == "cov":
            return np.array([c for _, _, c in self.cov], float)
        if quantity == "quad":
            return np.array([t for *_, t in self.quad], float)
        return getattr(self, quantity)

    def n_terms(self, quantity: str) -> int:
        """Return the number of values of a quantity: covalent dipoles, quadrupole terms, or atoms."""
        return {"cov": len(self.cov), "quad": len(self.quad)}.get(quantity, self.n)

    def symmetry_classes(self) -> np.ndarray:
        """Return a canonical symmetry-class id per atom by colour refinement.

        Returns
        -------
        np.ndarray (m,) int
            Class id per atom (0, 1, ...).

        Notes
        -----
        The graph has the bonds, the covalent-dipole pairs and the edges between each virtual site and
        its parent atoms.  Initial colours are "element|type"; each round replaces an atom's colour by
        (colour, sorted colours of its neighbours) and relabels the signatures in sorted order, until
        the number of classes stops growing (1-dimensional Weisfeiler-Lehman refinement).  Because the
        labels come from sorted signatures, the ids do not depend on the atom order.  Refinement may
        put atoms that are not truly symmetry-equivalent into one class for highly regular graphs.
        """
        adj = [set() for _ in range(self.n)]
        site_edges = [(vs.site, a) for vs in self.vsites for a in vs.atoms]  # a virtual site and its parents
        for i, j in list(self.bonds) + [(i, j) for i, j, _ in self.cov] + site_edges:
            if i != j:
                adj[i].add(j)
                adj[j].add(i)

        def relabel(sig: list) -> list[int]:
            order = {s: k for k, s in enumerate(sorted(set(sig)))}
            return [order[s] for s in sig]

        colour = relabel([f"{e}|{t}" for e, t in zip(self.elements, self.types)])
        while True:
            new = relabel([(colour[i], tuple(sorted(colour[k] for k in adj[i]))) for i in range(self.n)])
            if len(set(new)) == len(set(colour)):
                return np.array(colour)
            colour = new

    def tying_keys(self) -> dict[str, list[str]]:
        """Return the tying key of every atom (every term for "cov" and "quad") for every quantity.

        Atoms of one type in one symmetry class get the label "<type>" (or "<type>.<k>" when the type
        has several classes).  By-type quantities (BY_TYPE) use the atom type; the charge uses
        "<name>:<label>"; covalent dipoles "<name>:<label_i>><label_j>"; quadrupole terms
        "<name>:Q:<label_i>><label_j>" (axial) or "<name>:Q:<label_i>><label_j>|<label_k>" (sorted).
        Overrides in `self.keys` replace the defaults.

        Returns
        -------
        dict of str to list of str
            {quantity: keys}, for every name in QUANTITIES.

        Raises
        ------
        KeyError
            If `self.keys` names an unknown quantity.
        ValueError
            If an override has the wrong number of keys.
        """
        cls = self.symmetry_classes()
        label = {}
        for t in set(self.types):  # classes of one type, in canonical order
            cs = sorted({int(c) for c, tt in zip(cls, self.types) if tt == t})
            for k, c in enumerate(cs):
                label[c] = t if len(cs) == 1 else f"{t}.{k}"
        atom = [label[int(c)] for c in cls]
        keys = {qn: list(self.types) for qn in BY_TYPE}
        keys.update({qn: [f"{self.name}:{a}" for a in atom] for qn in BY_MOLECULE})
        keys["cov"] = [f"{self.name}:{atom[i]}>{atom[j]}" for i, j, _ in self.cov]
        keys["quad"] = [
            f"{self.name}:Q:{atom[i]}>{atom[j]}"
            if j == k
            else f"{self.name}:Q:{atom[i]}>" + "|".join(sorted((atom[j], atom[k])))
            for i, j, k, _ in self.quad
        ]
        for qn, ks in self.keys.items():
            if qn not in QUANTITIES:
                raise KeyError(f"unknown quantity {qn!r}")
            if len(ks) != self.n_terms(qn):
                raise ValueError(f"{self.name}: {len(ks)} keys for {qn}")
            keys[qn] = list(ks)
        return keys


class ParamTable:
    """Tied free parameters: for each quantity an ordered list of keys and an initial value per key.

    Atoms (or covalent dipoles, quadrupole terms) with the same tying key share one value; the
    parameter pytree is {quantity: (K,) array} with K keys of that quantity (`initial()`).  The
    table is built once and is not a pytree itself; parameter values are passed to the energy
    functions separately.

    Attributes
    ----------
    keys : dict of str to list of str
        Ordered keys per quantity (first occurrence order over the molecules).
    values0 : dict of str to np.ndarray (K,)
        Initial value per key: the mean over the atoms / terms that share it (quantity units).
    spread : dict of str to np.ndarray (K,)
        Peak-to-peak spread of the tied initial values per key.
    """

    def __init__(self, molecules: list[Molecule], tol: float = 1e-6) -> None:
        """Build the table from molecule templates.

        Each template (by identity) is counted once.  A spread of tied initial values larger than
        `tol * max(1, |value|)` is reported with warnings.warn (the first ten), not raised.

        Parameters
        ----------
        molecules : list of Molecule
            All molecules whose keys the table must contain.
        tol : float
            Relative tolerance on the spread of tied initial values (dimensionless).
        """
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
        bad = [
            (qn, k, s)
            for qn in QUANTITIES
            for k, s, v in zip(self.keys[qn], self.spread[qn], self.values0[qn])
            if s > tol * max(1.0, abs(v))
        ]
        if bad:
            warnings.warn(
                "tied values differ (quantity, key, spread): " + ", ".join(f"{a} {b} {c:.3g}" for a, b, c in bad[:10]),
                stacklevel=2,
            )

    def initial(self) -> dict[str, jnp.ndarray]:
        """Return the initial parameter pytree {quantity: jax.Array (K,)} (units of the quantity)."""
        return {qn: jnp.asarray(v) for qn, v in self.values0.items()}

    def index(self, quantity: str, keys: list[str]) -> np.ndarray:
        """Return the table positions of a list of keys of one quantity.

        Parameters
        ----------
        quantity : str
            A name of QUANTITIES.
        keys : list of str
            Tying keys.

        Returns
        -------
        np.ndarray int32
            Position of each key in `self.keys[quantity]`.

        Raises
        ------
        KeyError
            If a key is not in the table (the table was built without that molecule).
        """
        try:
            return np.array([self._pos[quantity][k] for k in keys], dtype=np.int32)
        except KeyError as e:
            raise KeyError(f"{quantity} key {e} is not in this ParamTable (build it from all molecules)") from None

    def named(self, params: Mapping[str, ArrayLike]) -> dict[str, dict[str, float]]:
        """Return {quantity: {key: value}} of a parameter pytree, for inspection."""
        return {qn: dict(zip(self.keys[qn], np.asarray(params[qn]).tolist())) for qn in QUANTITIES}

    def sizes(self) -> dict[str, int]:
        """Return the number of keys per quantity."""
        return {qn: len(self.keys[qn]) for qn in QUANTITIES}

    def fingerprint(self) -> str:
        """Return a SHA-1 hex digest of the keys and initial values (not of the current parameter values)."""
        h = hashlib.sha1()
        for qn in QUANTITIES:
            h.update("|".join(self.keys[qn]).encode())
            h.update(np.ascontiguousarray(self.values0[qn]).tobytes())
        return h.hexdigest()


class System:
    """Flattened, static description of a set of molecules (topology and parameter indices).

    Coordinates, parameters and box are passed to the energy functions, never stored here, so one
    System serves every configuration and parameter vector; its index arrays are host numpy arrays
    (static under jit).  Not a pytree.

        system = System([water] * 512)
        per_atom = system.expand(params)           # {"q": (N,), ..., "cov": (C,), "quad": (T,)}

    Attributes
    ----------
    molecules : list of Molecule
        The molecules in order (repeated templates are the same object).
    table : ParamTable
        The parameter table (shared with subsystems).
    n, nmol : int
        Number of atoms N and of molecules.
    offsets : np.ndarray (nmol + 1,) int
        First atom of each molecule, and N.
    mol : np.ndarray (N,) int32
        Molecule index of every atom.
    elements, types : list of str (N,)
        Element and atom type of every atom.
    masses : np.ndarray (N,)
        Atomic masses [amu].
    idx : dict of str to np.ndarray int32
        Per quantity, the table position of every atom's (or term's) value.
    cov_i, cov_j : np.ndarray (C,) int32
        Global atom indices of the covalent dipoles (dipole on cov_i along cov_j - cov_i).
    quad_ijk : np.ndarray (T, 3) int32
        Global atom indices (i, j, k) of the quadrupole terms.
    extra : dict of str to np.ndarray (N,)
        Per-atom extra arrays present in every molecule.
    params0 : dict of str to jax.Array
        The table's initial parameter pytree.
    """

    def __init__(self, molecules: list[Molecule], table: ParamTable | None = None) -> None:
        """Flatten the molecules into index arrays.

        Parameters
        ----------
        molecules : list of Molecule
            Molecules in order; repeat one object for identical molecules (keys are computed once per
            object).
        table : ParamTable, optional
            Parameter table to index into (must contain every key); None builds one from `molecules`.

        Raises
        ------
        KeyError
            If `table` lacks a key of one of the molecules.
        """
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
        for m in molecules:  # tying keys are computed once per template
            if id(m) not in keys:
                keys[id(m)] = m.tying_keys()
        self.idx = {
            qn: np.concatenate([self.table.index(qn, keys[id(m)][qn]) for m in molecules]).astype(np.int32)
            for qn in QUANTITIES
        }
        ci, cj = [], []
        for k, m in enumerate(molecules):
            for i, j, _ in m.cov:
                ci.append(offs[k] + i)
                cj.append(offs[k] + j)
        self.cov_i = np.array(ci, dtype=np.int32)
        self.cov_j = np.array(cj, dtype=np.int32)
        qt = [(offs[k] + i, offs[k] + j, offs[k] + l) for k, m in enumerate(molecules) for i, j, l, _ in m.quad]
        self.quad_ijk = np.array(qt, dtype=np.int32).reshape(-1, 3)
        self._pairs = None  # all pairs: built on first use (gas phase)
        extra_keys = set().union(*[m.extra.keys() for m in molecules]) if molecules else set()
        self.extra = {
            k: np.concatenate([np.asarray(m.extra[k], float) for m in molecules])
            for k in extra_keys
            if all(k in m.extra for m in molecules)
        }
        self.params0 = self.table.initial()

    # ------------------------------------------------------------------ all pairs (gas-phase models)
    def _all_pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (i, j, intermolecular) of all atom pairs i < j, built on first use and cached (gas phase)."""
        if getattr(self, "_pairs", None) is None:
            ii, jj = np.triu_indices(self.n, k=1)
            self._pairs = (ii, jj, self.mol[ii] != self.mol[jj])
        return self._pairs

    @property
    def pair_i(self) -> np.ndarray:
        """First atom i of every pair i < j (N (N - 1) / 2,)."""
        return self._all_pairs()[0]

    @property
    def pair_j(self) -> np.ndarray:
        """Second atom j of every pair i < j (N (N - 1) / 2,)."""
        return self._all_pairs()[1]

    @property
    def pair_inter(self) -> np.ndarray:
        """Whether each pair i < j is intermolecular (bool array)."""
        return self._all_pairs()[2]

    def expand(self, params: Mapping[str, ArrayLike] | None = None) -> dict[str, jnp.ndarray]:
        """Return per-atom (and per-term) parameter arrays gathered from the tied table.

        Parameters
        ----------
        params : Mapping of str to ArrayLike (K,), optional
            Parameter pytree {quantity: value per key}; None uses the table's initial values.

        Returns
        -------
        dict of str to jax.Array
            {quantity: (N,)} for the atom quantities, (C,) for "cov", (T,) for "quad"; units of the
            quantity.  Differentiable in `params` (a gather).
        """
        P = self.params0 if params is None else params
        return {qn: jnp.asarray(P[qn])[self.idx[qn]] for qn in QUANTITIES}

    def atom_slice(self, k: int) -> slice:
        """Return the slice of the atoms of molecule `k`."""
        return slice(int(self.offsets[k]), int(self.offsets[k + 1]))

    @classmethod
    def from_prmtop(cls, path: str, charges: str = "pgm") -> System:
        """Build the System of every molecule of an Amber prmtop.

        Identical molecules share one Molecule (param.read_prmtop_molecules), so that MD builds one
        template per kind.

        Parameters
        ----------
        path : str
            pGM prmtop file.
        charges : str
            Which charges to read, passed to param.read_prmtop_molecules ("pgm": the pGM charges).

        Returns
        -------
        System
        """
        from .param import read_prmtop_molecules

        return cls(read_prmtop_molecules(path, charges=charges))

    def sub(self, mols: tuple[int, ...]) -> tuple[System, np.ndarray]:
        """Return the subsystem of the given molecules (same ParamTable) and its atom indices in this system.

        Parameters
        ----------
        mols : tuple of int
            Molecule indices, in the order of the subsystem.

        Returns
        -------
        System
            The subsystem.
        np.ndarray (n_sub,) int
            Index of every subsystem atom in this system.
        """
        idx = np.concatenate([np.arange(self.offsets[k], self.offsets[k + 1]) for k in mols])
        return System([self.molecules[k] for k in mols], table=self.table), idx

    def fingerprint(self) -> str:
        """Return a SHA-1 hex digest of topology, parameter indices, elements, table and extra arrays.

        Used as the cache key of compiled functions: parameter *values* passed at call time are not
        part of it (the table's initial values are).  Masses and virtual sites are not hashed.
        """
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
        """Return the molecule names joined with "+"."""
        return "+".join(m.name for m in self.molecules)

    def __repr__(self) -> str:
        """Return "System(<signature>, n=<atoms>)"."""
        return f"System({self.signature()}, n={self.n})"
