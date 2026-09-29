"""Flexible molecules for pGM MD: bonded terms on the periodic pGM force field, constraints.

Contents: the templates `FlexibleTemplate` (fitted bonded terms) and `RigidTemplate` (rigid
molecules of up to three atoms held by constraints), `FlexibleMolecules` (per-atom molecule
representation), `FlexibleIntegrator` (constrained BAOAB on atoms), the engine
`FlexibleSimulation`, and `liquid_box` (a starting box of one template).  Rigid molecules by
constraints and macromolecules run in the same engine.

A molecule type is described by a template:
  FlexibleTemplate  pGM electrostatics (the `Molecule` the force field uses) and bonded parameters
                    fitted with `pgm_jax.bonded` (typed families or frozen neural bonded terms);
  RigidTemplate     a rigid molecule of up to three atoms (water, ions) held by distance
                    constraints, no bonded terms, no intramolecular van der Waals.
The force field treats every pair electrostatically (pGM has no exclusions); the van der Waals
term of the intramolecular pairs follows the fitted model,
    E_vdW,intra = sum_{pairs d_ij >= lj_min_sep} U_ij + lj14_scale sum_{1-4 pairs} U_ij,
through the special pairs of md/topology.py (inside the cutoff, as every other pair), so the
extra energy of a flexible molecule is its bonded energy.  Templates fitted with charge flux
(BondedSettings.flux) bring it along (md/flux.py: charges and covalent dipoles follow the bond
lengths).  Large molecules are split into heavy-atom groups for the neighbour list.  Optional
distance constraints (X-H bonds at their reference lengths; rigid templates always) are applied
with SHAKE / RATTLE in g-BAOAB order (md/constraints.py); with hydrogen mass repartitioning
(`hmr`) that allows 2 fs.

    tpl = FlexibleTemplate.from_fit(model, P)                 # after fitting pgm_jax.bonded
    tpl.save("methanol.flex")
    pos, H = liquid_box(tpl, 256, density=0.75)
    sim = FlexibleSimulation(System([tpl.pgm] * 256), [tpl] * 256, pos, H, MDSettings(), dt=0.0005,
                             temperature=298.0, thermostat=Bussi(1.0), barostat=MonteCarloBarostat())
    sim.run(20000, report_every=2000)

    # a protein in rigid water, X-H bonds constrained, 2 fs
    sim = FlexibleSimulation(sys, [protein] + [RigidTemplate(water, xyz)] * n_wat, pos, H,
                             MDSettings(), dt=0.002, constraints="h-bonds", hmr=3.024)

Molecules are kept whole: positions are never wrapped atom by atom, only whole molecules are
shifted by lattice vectors.  Virtual sites (Molecule.vsites, md/vsites.py) are not integrated: they
are rebuilt from their parents every step and their forces are spread to their parents.

Units: nm, ps, amu, kJ/mol, K.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, TextIO

import jax
import jax.numpy as jnp
import numpy as np

from ..system import System
from ..units import AMU_NM3_TO_G_CM3, KB
from ._jaxmd import simulate
from .barostats import MonteCarloBarostat
from .box import check_box, inv3, reduce_box, volume
from .constraints import Constraints, hmr_masses
from .engine import MDEngine
from .flux import ChargeFlux
from .forcefield import MDSettings, PGMForceField
from .integrate import Dynamics, Integrator, MDState, field_state
from .rigid import _unwrap
from .thermostats import Bussi, Thermostat
from .topology import MDTopology, MoleculeRule
from .vsites import VirtualSites

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import ArrayLike

    from ..bonded.model import BondedModel, BondedTerms, MolSpec
    from ..bonded.nn.model import NNBonded
    from ..system import Molecule
    from .alchemy import Alchemy
    from .efield import ExternalField
    from .forcefield import Result
    from .mts import MTS
    from .restraints import Restraint, Restraints


# ----------------------------------------------------------------------------- templates
class FlexibleTemplate:
    """Bonded parameters of one molecule type, fitted with pgm_jax.bonded, plus its pGM molecule.

    The MD engine treats electrostatics with all pairs, so the fit must have used pGM
    electrostatics without exclusions or refitted charges.  Charge flux (BondedSettings.flux) runs
    as fitted (md/flux.py; FlexibleSimulation builds it from the templates).  Molecules sharing a
    template object share its compiled bonded energy (vmapped over the copies).

    Attributes
    ----------
    specs : list of MolSpec
        Molecule specifications of the fit (topologies dropped; rebuilt by `terms`).
    settings : dict
        BondedSettings of the fit (as a dict).
    P : dict
        Fitted parameters (numpy leaves).
    index : int
        Which molecule of the fit this template is.
    has_bonded : bool
        True (class attribute; RigidTemplate: False).
    """

    def __init__(self, specs: Sequence[MolSpec], settings: dict, P: dict, index: int = 0) -> None:
        """Store a fit and check that it is compatible with the MD engine.

        Parameters
        ----------
        specs : Sequence of MolSpec
            The molecules of the fit.
        settings : dict
            BondedSettings of the fit as a dict (dataclasses.asdict).
        P : dict
            Fitted parameters.
        index : int
            Molecule of the fit this template describes.

        Raises
        ------
        ValueError
            A fit with electrostatic or induction exclusions, fitted charges, learned pair scales
            or quadrupoles (the MD engine uses pGM with all pairs).
        """
        from ..bonded.model import BondedSettings

        self.specs = [replace(s, top=None) for s in specs]
        self.settings = dict(settings)
        self.P = jax.tree_util.tree_map(np.asarray, P)
        self.index = int(index)
        st = BondedSettings(**self.settings)
        bad = []
        if st.elec_exclude != 0 or st.elec14_scale != 1.0:
            bad.append("electrostatic exclusions")
        if st.qfit >= 0 or st.qbci >= 0:
            bad.append("fitted charges")
        if st.escale:
            bad.append("learned pair scales")
        if st.ind_exclude not in (-1, 0):
            bad.append("induction exclusions")
        if st.quadrupoles:
            bad.append("quadrupoles (not in the MD engine yet)")
        if bad:
            raise ValueError("the MD engine uses pGM with all pairs; this fit used " + ", ".join(bad))
        self._model = self._terms = None

    @classmethod
    def from_fit(cls, model: BondedModel, P: dict, index: int = 0) -> FlexibleTemplate:
        """Return the template of molecule `index` of a fitted BondedModel and its parameters.

        Neural bonded terms ("nnb") are frozen: their stage-1 coefficients are evaluated once here.
        """
        if getattr(model, "nnb", None) is not None and "coef" not in P["nnb"]:
            P = dict(P)
            P["nnb"] = model.nnb.freeze(P["nnb"])
        return cls(model.mols, asdict(model.s), P, index)

    @classmethod
    def from_network(cls, net: NNBonded, P: dict, spec: MolSpec, **settings: Any) -> FlexibleTemplate:
        """Return the template of any molecule from a trained neural bonded model.

        net and P: bonded.nn.NNBonded and its parameters (e.g. NNBonded.load); stage 1 is
        evaluated for this molecule and frozen (its topology is built if missing).  **settings:
        the BondedSettings the network was trained with (lj14_scale, lj_min_sep, elec, ...).
        """
        from ..bonded.model import BondedSettings
        from ..bonded.topology import build_topology

        if spec.top is None:
            spec.top = build_topology(spec.elements, spec.bonds, (spec.bonds, spec.bond_orders), spec.ref_xyz * 10.0)
        C = net.coefficients(P, net.prepare(spec))
        c = net.config
        st = BondedSettings(
            families=("nnb",),
            nn_width=c.width,
            nn_layers=c.layers,
            nn_ref=c.ref,
            nn_basis=c.basis,
            nn_b_span=c.b_span,
            nn_th_span=c.th_span,
            nn_out_scale=c.out_scale,
            nn_pgm_features=c.pgm_features,
            nn_table_depth=c.table_depth,
            nn_context=c.context,
            **settings,
        )
        return cls([spec], asdict(st), {"nnb": {"coef": [C]}}, 0)

    @property
    def spec(self) -> MolSpec:
        """The MolSpec of this molecule."""
        return self.specs[self.index]

    @property
    def pgm(self) -> Molecule:
        """The pGM molecule (charges, radii, polarizabilities, covalent dipoles, van der Waals)."""
        return self.spec.pgm

    @property
    def name(self) -> str:
        """Molecule name."""
        return self.spec.name

    @property
    def terms(self) -> BondedTerms:
        """The bonded terms (BondedTerms: no gas-phase nonbonded setup, any molecule size)."""
        if self._terms is None:
            from ..bonded.model import BondedSettings, BondedTerms

            self._terms = BondedTerms([replace(s) for s in self.specs], BondedSettings(**self.settings))
        return self._terms

    @property
    def model(self) -> BondedModel:
        """The full gas-phase model the bonded terms were fitted with (built on first use).

        BondedModel: bonded terms + pGM and intramolecular van der Waals, for reference energies
        of isolated molecules.
        """
        if self._model is None:
            from ..bonded.model import BondedModel, BondedSettings

            self._model = BondedModel([replace(s) for s in self.specs], BondedSettings(**self.settings))
        return self._model

    @property
    def n(self) -> int:
        """Number of atoms."""
        return len(self.spec.elements)

    def lj_pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the intramolecular van der Waals pairs (i, j, weight).

        Weight 1 for graph distance >= lj_min_sep, lj14_scale for 1-4 pairs (added if both).  The
        MD engine takes them from md/topology.py's special pairs; this is for reference.
        """
        top = self.terms.mols[self.index].top
        i, j = np.triu_indices(self.n, 1)
        d = top.dist[i, j] if top.dist is not None else np.array([top.graph_distance(a, b) for a, b in zip(i, j)])
        w = (d >= self.settings.get("lj_min_sep", 4)).astype(float)
        w = w + float(self.settings.get("lj14_scale", 0.0)) * (d == 3)
        keep = w > 0
        return i[keep], j[keep], w[keep]

    def bond_lengths(self) -> np.ndarray:
        """Return the reference bond lengths [nm] of the fitted model, in the topology's bond order.

        The lengths X-H constraints hold (typed families: P["ref"]["b0"]; frozen neural terms:
        their b0 coefficients).
        """
        terms, P = self.terms, jax.tree_util.tree_map(np.asarray, self.P)
        if terms.fams:
            return np.asarray(P["ref"]["b0"])[terms.I[self.index]["bond"]]
        return np.asarray(P["nnb"]["coef"][self.index]["b0"])

    def md_rule(self, constraints: str = "none") -> MoleculeRule:
        """Return how the MD engine treats this molecule (md/topology.py).

        Intramolecular van der Waals by graph distance (lj_min_sep, lj14_scale of the fit);
        constraints "none", "h-bonds" (X-H bonds at the model's reference lengths) or "all-bonds"
        (every bond at its reference length).

        Raises
        ------
        ValueError
            An unknown constraints option.
        """
        top = self.terms.mols[self.index].top
        cons = []
        if constraints in ("h-bonds", "all-bonds"):
            el = self.spec.elements
            for (i, j), b0 in zip(top.bonds, self.bond_lengths()):
                if constraints == "all-bonds" or (el[i] == "H") != (el[j] == "H"):
                    cons.append((int(i), int(j), float(b0)))
        elif constraints != "none":
            raise ValueError("constraints: 'none' | 'h-bonds' | 'all-bonds'")
        return MoleculeRule(
            bonds=[tuple(int(x) for x in b) for b in top.bonds],
            vdw="graph",
            lj_min_sep=int(self.settings.get("lj_min_sep", 4)),
            lj14_scale=float(self.settings.get("lj14_scale", 0.0)),
            constraints=tuple(cons),
        )

    has_bonded = True

    def check_settings(self, settings: MDSettings) -> None:
        """Check that the MD model is the model the bonded terms were fitted with.

        Parameters
        ----------
        settings : MDSettings
            The MD settings (their terms: elec, vdw, gvdw_rep).

        Raises
        ------
        ValueError
            A term of the MD settings differs from the fit's (gvdw_rep only matters for GVDW).
        """
        st, md = self.terms.s, settings.terms
        for name in ("elec", "vdw", "gvdw_rep"):
            if getattr(st, name) != getattr(md, name) and not (name == "gvdw_rep" and st.vdw != "gvdw"):
                raise ValueError(
                    f"template {self.name} was fitted with {name}={getattr(st, name)!r}, "
                    f"the MD settings have {getattr(md, name)!r}"
                )

    def bonded_energy(self, R: jax.Array, P: dict | None = None) -> jax.Array:
        """Return the bonded energy [kJ/mol] of one copy at positions R (n, 3) [nm] (P: None = fitted)."""
        return self.terms.bonded_energy(self.index, R, jax.tree_util.tree_map(jnp.asarray, self.P if P is None else P))

    def save(self, path: str) -> None:
        """Write the template to a pickle file (specs, settings, parameters, index)."""
        with open(path, "wb") as fh:
            pickle.dump({"specs": self.specs, "settings": self.settings, "P": self.P, "index": self.index}, fh)

    @classmethod
    def load(cls, path: str) -> FlexibleTemplate:
        """Return a template read from a `save` file (a pickle: read only files you trust)."""
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["specs"], d["settings"], d["P"], d["index"])


class RigidTemplate:
    """A rigid molecule of up to three atoms (water, ions) for FlexibleSimulation.

    Every distance is held by a constraint (from the geometry `xyz`), no bonded terms, no
    intramolecular van der Waals (Amber's rigid water).  Virtual sites of the molecule (TIP4P's M
    site, TIP5P's lone pairs) come on top of the three atoms: they are placed, not constrained.
    Implements the template interface of FlexibleTemplate that the engine uses.

    Attributes
    ----------
    pgm : Molecule
        The pGM molecule.
    name : str
        Name.
    n : int
        Number of atoms (sites included).
    real_atoms : list of int
        The atoms that are not virtual sites (at most 3).
    xyz : np.ndarray (n, 3)
        Geometry [nm].
    has_bonded : bool
        False (class attribute).
    """

    has_bonded = False

    def __init__(self, molecule: Molecule, xyz: ArrayLike | None = None, name: str | None = None) -> None:
        """Set up the template.

        Parameters
        ----------
        molecule : Molecule
            The pGM molecule.
        xyz : ArrayLike (n, 3), optional
            Geometry [nm] (None: molecule.extra["xyz"]; not needed for one atom).
        name : str, optional
            Name (None: the molecule's).

        Raises
        ------
        ValueError
            More than three real atoms.
        """
        self.pgm = molecule
        self.name = name or molecule.name
        self.n = molecule.n
        sites = {vs.site for vs in molecule.vsites}
        self.real_atoms = [a for a in range(self.n) if a not in sites]
        if len(self.real_atoms) > 3:
            raise ValueError(
                "RigidTemplate holds up to three atoms (plus virtual sites) by distances; use a FlexibleTemplate"
            )
        x = np.asarray(xyz if xyz is not None else molecule.extra.get("xyz"), float) if self.n > 1 else np.zeros((1, 3))
        self.xyz = x.reshape(self.n, 3)

    @property
    def spec(self) -> RigidTemplate:
        """The template itself (it provides `elements` like a MolSpec)."""
        return self

    @property
    def elements(self) -> list[str]:
        """Element symbols."""
        return list(self.pgm.elements)

    def md_rule(self, constraints: str = "none") -> MoleculeRule:
        """Return the rule: every distance of the real atoms constrained, no van der Waals (any option)."""
        ra = self.real_atoms
        pairs = [(i, j) for k, i in enumerate(ra) for j in ra[k + 1 :]]
        cons = tuple((i, j, float(np.linalg.norm(self.xyz[i] - self.xyz[j]))) for i, j in pairs)
        return MoleculeRule(bonds=[], vdw="none", constraints=cons)

    def check_settings(self, settings: MDSettings) -> None:
        """Accept any settings (a rigid molecule has no fitted terms)."""
        return None

    def bonded_energy(self, R: jax.Array, P: dict | None = None) -> float:
        """Return 0.0 (no bonded terms)."""
        return 0.0


def _unwrap_bonded(x: np.ndarray, H: np.ndarray, bonds: Sequence[Sequence[int]]) -> np.ndarray:
    """Return a molecule made whole along its bonds (host).

    Works for any size, unlike the minimum image of the first atom (rigid._unwrap, used without
    bonds): a depth-first walk puts every atom at the minimum image of the atom it is reached
    from.  x (n, 3) and the result [nm]; H (3, 3) [nm]; bonds local.
    """
    n = len(x)
    if n <= 1 or not len(bonds):
        return _unwrap(x, H)
    Hinv = np.linalg.inv(H)
    nbr = [[] for _ in range(n)]
    for i, j in bonds:
        nbr[int(i)].append(int(j))
        nbr[int(j)].append(int(i))
    out = np.array(x, float)
    seen = np.zeros(n, bool)
    for s0 in range(n):
        if seen[s0]:
            continue
        seen[s0] = True
        stack = [s0]
        while stack:
            u = stack.pop()
            for v in nbr[u]:
                if not seen[v]:
                    d = x[v] - out[u]
                    for _ in range(2):
                        d = d - np.round(d @ Hinv) @ H
                    out[v] = out[u] + d
                    seen[v] = True
                    stack.append(v)
    return out


# ----------------------------------------------------------------------------- molecules
class FlexibleMolecules:
    """Per-atom molecule representation: bonded energies, centres, whole-molecule wrapping.

    Bonded energies are grouped by template (one vmap per template), with molecular and
    neighbour-list-group centres and whole-molecule wrapping; it provides the `positions` / `wrap`
    interface of rigid.RigidMolecules to the engine.  `templates[k]` belongs to `sys.molecules[k]`
    (same atom order); `topology` is the system's MDTopology.

    Attributes
    ----------
    masses : jax.Array (N,)
        Physical masses [amu] (after mass repartitioning).
    mass : jax.Array (N, 1)
        The integrator's masses [amu] (1 at virtual sites, whose momenta are held at 0).
    real : jax.Array (N, 1)
        1 for real atoms, 0 for sites (with virtual sites only).
    mmol, mgroup : jax.Array
        Masses of the molecules and of the list groups [amu].
    groups : list of (template, jax.Array (copies, n))
        Atom rows of the copies of every flexible template.
    pos0 : jax.Array (N, 3)
        Initial positions, molecules whole, sites placed [nm].
    r_max : float
        Largest atom-to-group-centre distance at pos0 [nm].
    dof_correction : int
        0 (the rigid-body correction does not apply).
    """

    def __init__(
        self,
        sys: System,
        pos: ArrayLike,
        H: ArrayLike,
        templates: Sequence[FlexibleTemplate | RigidTemplate],
        topology: MDTopology,
        masses: ArrayLike | None = None,
        vsites: VirtualSites | None = None,
    ) -> None:
        """Check the templates, make the molecules whole and group the flexible copies.

        Parameters
        ----------
        sys : System
            The system.
        pos : ArrayLike (N, 3)
            Positions [nm].
        H : ArrayLike (3, 3)
            Box [nm].
        templates : Sequence of templates (nmol,)
            Template of every molecule.
        topology : MDTopology
            The system's pair topology.
        masses : ArrayLike (N,), optional
            Masses [amu] (None: the system's).
        vsites : VirtualSites, optional
            The virtual sites.

        Raises
        ------
        ValueError
            Not one template per molecule, a template that does not match its molecule, or bonded
            terms on virtual sites.
        """
        if len(templates) != sys.nmol:
            raise ValueError("one template per molecule")
        pos, H = np.asarray(pos, float), np.asarray(H, float)
        self.sys, self.nmol, self.n = sys, sys.nmol, sys.n
        self.topology = topology
        self.mol = jnp.asarray(sys.mol)
        self.group = jnp.asarray(topology.group)
        self.n_group = topology.n_group
        m = np.asarray(sys.masses if masses is None else masses, float)
        self.masses = jnp.asarray(m)
        self.vsites = vsites
        if vsites is None:
            self.mass = jnp.asarray(m)[:, None]
        else:  # the integrator's masses: 1 at the sites, whose momenta are held at 0 (`real`)
            self.real = jnp.asarray(vsites.real, jnp.float64)[:, None]
            self.mass = jnp.asarray(np.where(vsites.real, m, 1.0))[:, None]
        self.mmol = jax.ops.segment_sum(self.masses, self.mol, self.nmol)
        self.mgroup = jax.ops.segment_sum(self.masses, self.group, self.n_group)
        whole = np.zeros_like(pos)
        groups = {}
        for k, (molk, tpl) in enumerate(zip(sys.molecules, templates)):
            sl = sys.atom_slice(k)
            if tpl.n != molk.n or list(tpl.spec.elements) != list(molk.elements):
                raise ValueError(f"template {tpl.name} does not match molecule {k} ({molk.name})")
            bonds = tpl.terms.mols[tpl.index].top.bonds if tpl.has_bonded else molk.bonds
            whole[sl] = _unwrap_bonded(pos[sl], H, bonds)
            if tpl.has_bonded and molk.vsites:
                sites = {vs.site for vs in molk.vsites}
                if any(int(a) in sites for b in bonds for a in b):
                    raise ValueError(
                        f"template {tpl.name}: bonded terms involve virtual sites; sites carry no bonded terms"
                    )
            if tpl.has_bonded:
                groups.setdefault(id(tpl), (tpl, []))[1].append(np.arange(sl.start, sl.stop))
        self.groups = [(tpl, jnp.asarray(np.array(rows))) for tpl, rows in groups.values()]
        if vsites is not None:
            whole = np.asarray(vsites.place(whole, H))
            vsites.check(whole, H, (sys.cov_i, sys.cov_j))
        self.pos0 = jnp.asarray(whole)
        self.r_max = topology.group_radius(whole, m)
        self.dof_correction = 0

    def centers(self, pos: jax.Array) -> jax.Array:
        """Return the molecular centres of mass (M, 3) [nm] (wrapping, barostat, molecular virial)."""
        return jax.ops.segment_sum(self.masses[:, None] * pos, self.mol, self.nmol) / self.mmol[:, None]

    def list_centers(self, pos: jax.Array) -> jax.Array:
        """Return the centres of mass (G, 3) [nm] of the neighbour-list groups."""
        return jax.ops.segment_sum(self.masses[:, None] * pos, self.group, self.n_group) / self.mgroup[:, None]

    def energy(self, pos: jax.Array, P_atoms: Any = None) -> jax.Array | float:
        """Return the bonded energy [kJ/mol] of every flexible molecule at positions pos (N, 3) [nm].

        The intramolecular van der Waals pairs are in the force field's special pairs.  P_atoms is
        not used.
        """
        e = 0.0
        for tpl, rows in self.groups:
            e = e + jnp.sum(jax.vmap(tpl.bonded_energy)(pos[rows]))
        return e

    def positions(self, pos: jax.Array) -> jax.Array:
        """Return pos (the integrator's positions are the atom positions)."""
        return pos

    def wrap(self, pos: jax.Array, H: ArrayLike) -> jax.Array:
        """Return the positions with whole molecules shifted so that their centres lie in the cell."""
        H = jnp.asarray(H)
        hi = jax.lax.Precision.HIGHEST
        f = jnp.matmul(self.centers(pos), inv3(H), precision=hi)
        return pos - jnp.matmul(jnp.floor(f), H, precision=hi)[self.mol]

    def extent(self, pos: jax.Array) -> jax.Array:
        """Return the largest atom-to-group-centre distance [nm] (check of the neighbour-list margin)."""
        return jnp.max(jnp.linalg.norm(pos - self.list_centers(pos)[self.group], axis=1))


# ----------------------------------------------------------------------------- integrator
class FlexibleIntegrator(Integrator):
    """Velocity Verlet (NVE) / BAOAB (NVT: Langevin, Bussi or GLE) on atoms, with constraints.

    Constraints in g-BAOAB order (SHAKE after every drift, RATTLE after every kick and thermostat
    step; GLE auxiliaries are projected too); NPT adds the Monte Carlo barostat with molecular
    scaling (centres of mass scaled, molecules translated rigidly, which keeps the constraints).
    Degrees of freedom: 3 per real atom minus one per constraint, minus 3 when the total momentum
    is conserved (NVE, Bussi; drawn momenta then have no net momentum, given ones are kept).

    Attributes
    ----------
    flex : FlexibleMolecules
        The molecules.
    vsites : VirtualSites or None
        Virtual sites.
    cons : Constraints or None
        Constraints (None without any).
    n_real : int
        Real (non-site) atoms.
    momentum_conserved : bool
        NVE or Bussi.

    Other attributes as Integrator.
    """

    def __init__(
        self,
        ff: PGMForceField,
        flex: FlexibleMolecules,
        neighbors: Any,
        dt: float = 0.0005,
        constraints: Constraints | None = None,
        **kw: Any,
    ) -> None:
        """Set up the integrator (arguments as Integrator; `constraints`: SHAKE / RATTLE or None)."""
        self.flex = flex
        self.vsites = getattr(flex, "vsites", None)
        self.cons = constraints if (constraints is not None and constraints.nc) else None
        super().__init__(ff, flex, neighbors, dt, **kw)
        nc = self.cons.nc if self.cons is not None else 0
        self.n_real = flex.n - (0 if self.vsites is None else self.vsites.n_sites)
        # NVE and Bussi rescaling conserve the total momentum (constraint forces are internal)
        self.momentum_conserved = self.ensemble == "nve" or isinstance(self.thermostat, Bussi)
        self.dof = 3 * self.n_real - nc - (3 if self.momentum_conserved else 0)

    def _forces(
        self,
        pos: jax.Array,
        box: jax.Array,
        induction: Any,
        nbr: Any,
        force_rebuild: bool | jax.Array = False,
        lam: jax.Array | None = None,
        bias: Any = None,
        field: tuple | None = None,
    ) -> tuple[jax.Array, Result, Any]:
        """Return atomic forces, the result and the list as Integrator._forces, for atoms.

        Adds the bonded energy and forces of the flexible molecules; site forces are spread to
        the parents.  pos (N, 3) [nm]; forces (N, 3) [kJ/mol/nm].
        """
        centers = self.flex.list_centers(pos)
        nbr = self.nb.update(nbr, pos, centers, box, force_rebuild)
        cand, ovf = self.nb.candidates(nbr, centers, box, pos)
        if self.alchemy is None:
            res = self.ff.compute(
                pos, box, cand, induction, self.params, keep_geometry=self.keep_geometry, efield=field
            )
        else:  # Hamiltonian at the state's coupling lam (alchemy.py)
            res = self.alchemy.compute(self.ff, pos, box, cand, induction, self.params, lam)
        e_in, g_in = jax.value_and_grad(self.flex.energy)(pos)
        energy = dict(res.energy)
        energy["total"] = res.energy["total"] + e_in
        res = self._add_restraints(res._replace(energy=energy, overflow=res.overflow | ovf), pos, box, bias)
        if self.vsites is not None:  # site forces to the parents
            return self.vsites.spread(pos, box, res.forces - g_in), res, nbr
        return res.forces - g_in, res, nbr

    def _bias_atoms(self, x: jax.Array) -> jax.Array:
        """Return x (the positions are the atoms)."""
        return x

    def _map_atom_forces(self, x: jax.Array, box: jax.Array, F: jax.Array) -> jax.Array:
        """Return F with site forces spread to the parents."""
        return F if self.vsites is None else self.vsites.spread(x, box, F)

    def place(self, pos: jax.Array, box: jax.Array) -> jax.Array:
        """Positions with the virtual sites rebuilt from their parents (identity without sites)."""
        return pos if self.vsites is None else self.vsites.place(pos, box)

    def init(
        self, pos: ArrayLike, box: ArrayLike, key: jax.Array, momentum: ArrayLike | None = None, bias: Any = None
    ) -> MDState:
        """Return a new state (host): positions on the constraints, momenta drawn or set, forces.

        Drawn momenta (sites excluded) have zero total momentum when it is conserved; given
        momenta [amu nm/ps] are kept (zeroed at sites); RATTLE projects them.  Arguments as
        Integrator.init with atom positions pos (N, 3) [nm].
        """
        box = jnp.asarray(box, jnp.float64)
        pos = jnp.asarray(pos, jnp.float64)
        if self.cons is not None:  # start on the constraint surface
            for _ in range(3):  # three SHAKE passes from the start geometry
                pos = self.cons.positions(pos, pos)
        pos = self.place(pos, box)
        nbr = self.nb.allocate(pos, self.flex.list_centers(pos), box)
        key, split = jax.random.split(key)
        zero = jnp.zeros_like(pos)
        dyn = Dynamics(pos, zero, zero, self.flex.mass, key)
        if momentum is None and self.vsites is not None:  # real atoms only, zero total momentum
            real = self.flex.real
            p = jnp.sqrt(self.flex.mass * self.kT) * jax.random.normal(split, pos.shape, jnp.float64) * real
            dyn = dyn.set(momentum=(p - jnp.sum(p, 0) / self.n_real) * real)
        elif momentum is None:
            dyn = simulate.initialize_momenta(dyn, split, self.kT)
        else:
            p = jnp.asarray(momentum, jnp.float64)
            dyn = dyn.set(momentum=p if self.vsites is None else p * self.flex.real)
        if self.momentum_conserved and momentum is None:  # drawn momenta: the 3 centre-of-mass dof carry no energy
            m = self.flex.masses[:, None]
            dyn = dyn.set(momentum=dyn.momentum - m * jnp.sum(dyn.momentum, 0) / jnp.sum(m))
        if self.cons is not None:
            dyn = dyn.set(momentum=self.cons.momenta(pos, dyn.momentum, self.flex.masses))
        z = jnp.zeros((), jnp.float64)
        zi = jnp.zeros((), jnp.int32)
        dyn, aux = self._init_aux(dyn)
        st = MDState(
            dyn=dyn,
            box=box,
            induction=self.ff.init_induction(),
            nbr=nbr,
            epot=z,
            elec=z,
            vdw=z,
            iters=zi,
            max_iters=zi,
            resid=z,
            step=zi,
            mc=jnp.zeros(4, jnp.int32),
            mc_dv=jnp.asarray(0.01 * float(volume(box)), jnp.float64),
            overflow=jnp.zeros((), bool),
            aux=aux,
            heat=z,
            cg_total=z,
            bias=self._init_bias(bias),
        )
        return self.forces(field_state(st, self.efield), False)

    def _scaled(self, dyn: Dynamics) -> tuple[jax.Array, jax.Array | None, Callable, Callable]:
        """Return the mass-scaled atomic momenta, the site mask, the inverse map and the projection.

        The projection (RATTLE in mass-scaled form) keeps the momenta and the thermostat
        auxiliaries on the constraint tangent space.
        """
        sm = jnp.sqrt(dyn.mass)
        q = dyn.position
        if self.cons is None:

            def project(u: jax.Array) -> jax.Array:
                return u
        else:

            def project(u: jax.Array) -> jax.Array:
                return self.cons.momenta(q, u * sm, self.flex.masses) / sm

        mask = None if self.vsites is None else jnp.broadcast_to(self.flex.real, dyn.momentum.shape)  # sites: no noise
        return dyn.momentum / sm, mask, (lambda v: dyn.set(momentum=v * sm)), project

    # ------------------------------------------------------------------ constrained steps (g-BAOAB)
    def _kick(self, dyn: Dynamics, h: float) -> Dynamics:
        """Return dyn after a kick p += h F [ps], projected by RATTLE."""
        p = dyn.momentum + h * dyn.force
        return dyn.set(momentum=p if self.cons is None else self.cons.momenta(dyn.position, p, self.flex.masses))

    def _drift(self, dyn: Dynamics, h: float, project: bool = True) -> Dynamics:
        """Return dyn after a drift by h [ps]: SHAKE positions, momenta of the constrained move.

        p = m (q1 - q) / h, then RATTLE.  project=False leaves the RATTLE projection to the next
        kick, which projects at the same positions (projection is linear:
        P(p + h F) = P(P p + h F)).  Virtual sites have zero momentum, so they stay where they
        are; `_step` rebuilds them once per step, before the forces.
        """
        q = dyn.position
        q1 = q + h * dyn.momentum / dyn.mass
        if self.cons is not None:
            q1 = self.cons.positions(q1, q)
        p = dyn.mass * (q1 - q) / h
        if self.cons is not None and project:
            p = self.cons.momenta(q1, p, self.flex.masses)
        return dyn.set(position=q1, momentum=p)

    def _step(self, st: MDState) -> MDState:
        """Advance one constrained step (g-BAOAB; plain Integrator._step without constraints or sites)."""
        if self.cons is None and self.vsites is None:
            return super()._step(st)
        dt = self.dt
        aux, heat = st.aux, st.heat
        dyn = self._kick(st.dyn, dt / 2)
        # the last drift's momenta are projected by the closing kick (same positions)
        if self.ensemble == "nve":
            dyn = self._drift(dyn, dt, project=False)
        else:
            dyn = self._drift(dyn, dt / 2)  # projected: the O step books the heat of P p
            dyn, aux, heat = self._o_step(dyn, aux, heat, dt, self.thermostat_kT(st))
            dyn = self._drift(dyn, dt / 2, project=False)
        if self.vsites is not None:
            dyn = dyn.set(position=self.vsites.place(dyn.position, st.box))
        F, res, nbr = self._forces(
            dyn.position, st.box, st.induction, st.nbr, lam=st.lam, bias=st.bias, field=self.field_at(st, st.step + 1)
        )
        st = self._with_result(st.set(dyn=dyn, aux=aux, heat=heat), F, res, nbr)
        st = st.set(dyn=self._kick(st.dyn, dt / 2), step=st.step + 1)
        if self.ensemble == "npt":
            st = jax.lax.cond(st.step % self.interval == 0, self._barostat, lambda s: s, st)
        return st

    def _barostat(self, st: MDState) -> MDState:
        """Try one Monte Carlo volume move as Integrator._barostat, molecules translated rigidly.

        Every atom moves with its molecule's centre of mass, r -> r + (s - 1) R_k, which keeps the
        constraints; the trial energy includes the bonded energy.
        """
        key, k1, k2 = jax.random.split(st.dyn.rng, 3)
        H = st.box
        V = volume(H)
        dV = (2.0 * jax.random.uniform(k1, dtype=jnp.float64) - 1.0) * st.mc_dv
        Vn = V + dV
        s = jnp.cbrt(jnp.maximum(Vn, 1e-12) / V)
        pos = st.dyn.position
        pos_n = pos + ((s - 1.0) * self.flex.centers(pos))[self.flex.mol]
        Hn = H * s
        c_n = self.flex.list_centers(pos_n)
        nbr_n = self.nb.update(st.nbr, pos_n, c_n, Hn, True)
        cand, ovf0 = self.nb.candidates(nbr_n, c_n, Hn, pos_n)
        field = self.field_at(st, st.step)
        if self.alchemy is None:
            e_n, ind_n, _, ovf = self.ff.energy(pos_n, Hn, cand, st.induction, self.params, efield=field)
        else:
            e_n, ind_n, _, ovf = self.alchemy.energy(self.ff, pos_n, Hn, cand, st.induction, self.params, st.lam)
        e_n = e_n + self.flex.energy(pos_n) + self._restraint_energy(pos_n, Hn, st.bias)
        ovf = ovf | ovf0
        kT = self.thermostat_kT(st)
        e_0 = st.epot
        if self.ff.shadow:  # iEL/0-SCF: converged energies at both volumes
            c0 = self.flex.list_centers(pos)
            e_0 = (
                self.ff.energy(
                    pos, H, self.nb.candidates(st.nbr, c0, H, pos)[0], st.induction, self.params, efield=field
                )[0]
                + self.flex.energy(pos)
                + self._restraint_energy(pos, H, st.bias)
            )
        w = (e_n - e_0) + self.pressure * dV - self.nmol * kT * jnp.log(jnp.maximum(Vn, 1e-12) / V)
        accept = (Vn > 0) & (jnp.log(jax.random.uniform(k2, dtype=jnp.float64)) < -w / kT)
        st = st.set(dyn=st.dyn.set(rng=key), overflow=st.overflow | ovf)

        def acc(st: MDState) -> MDState:  # accepted: move to the trial box and re-evaluate the forces
            st = st.set(dyn=st.dyn.set(position=pos_n), box=Hn, induction=ind_n)
            F, res, nbr = self._forces(pos_n, Hn, ind_n, nbr_n, lam=st.lam, bias=st.bias, field=field)
            return self._with_result(st, F, res, nbr)

        st = jax.lax.cond(accept, acc, lambda s: s, st)
        mc = st.mc + jnp.array([1, 0, 1, 0], jnp.int32) + accept.astype(jnp.int32) * jnp.array([0, 1, 0, 1], jnp.int32)
        adapt = mc[2] >= 10
        rate = mc[3] / jnp.maximum(mc[2], 1)
        dv = jnp.where(
            adapt & (rate < 0.25), st.mc_dv / 1.1, jnp.where(adapt & (rate > 0.75), st.mc_dv * 1.1, st.mc_dv)
        )
        dv = jnp.minimum(dv, 0.3 * volume(st.box))
        mc = jnp.where(adapt, mc.at[2].set(0).at[3].set(0), mc)
        return st.set(mc=mc, mc_dv=dv)

    def kinetic(self, st: MDState) -> tuple[jax.Array, jax.Array]:
        """Return the (total, centre-of-mass translational) kinetic energy [kJ/mol]."""
        p = st.dyn.momentum
        ke = 0.5 * jnp.sum(p * p / self.flex.mass)
        Pm = jax.ops.segment_sum(p, self.flex.mol, self.flex.nmol)
        return ke, 0.5 * jnp.sum(Pm * Pm / self.flex.mmol[:, None])

    def temperatures(self, st: MDState) -> tuple[jax.Array, jax.Array]:
        """Return the (centre-of-mass translational, internal) temperatures [K]."""
        ke, ke_t = self.kinetic(st)
        n_t = 3 * self.nmol - (3 if self.momentum_conserved else 0)
        nc = self.cons.nc if self.cons is not None else 0
        n_i = 3 * self.n_real - 3 * self.nmol - nc
        return 2.0 * ke_t / (n_t * KB), 2.0 * (ke - ke_t) / (max(n_i, 1) * KB)


# ----------------------------------------------------------------------------- driver
class FlexibleSimulation(MDEngine):
    """Molecular dynamics of flexible molecules (atoms integrated individually).

    Bonded terms, X-H or all-bond constraints, virtual sites rebuilt every step, charge flux.  The
    shared machinery (blocks, observables, run loop, checkpoints) is MDEngine's (md/engine.py);
    the integrator is FlexibleIntegrator (mts.MTSFlexibleIntegrator with `mts`).

    Attributes
    ----------
    topology : MDTopology
        Pair topology (groups, special pairs, constraints).
    vsites : VirtualSites or None
        Virtual sites.
    flex : FlexibleMolecules
        The molecules (also `rigid`, for the base driver).
    constraints : Constraints
        The constraints (possibly none).
    r_list : float
        Neighbour-list group radius (largest group radius + r_margin) [nm].

    Other attributes as MDEngine.
    """

    def __init__(
        self,
        system: System,
        templates: Sequence[FlexibleTemplate | RigidTemplate],
        positions: ArrayLike,
        box: ArrayLike,
        settings: MDSettings = MDSettings(),
        *,
        dt: float = 0.0005,
        temperature: float = 298.0,
        thermostat: Thermostat | str | None = "langevin",
        barostat: MonteCarloBarostat | None = None,
        velocities: ArrayLike | None = None,
        seed: int = 0,
        params: dict | None = None,
        restraints: Restraints | Restraint | list | None = None,
        alchemy: Alchemy | None = None,
        mts: MTS | None = None,
        bias: Any = None,
        efield: ExternalField | ArrayLike | None = None,
        constraints: str = "none",
        hmr: float | Sequence[float | None] | None = None,
        constraint_options: dict | None = None,
        r_margin: float = 0.05,
        log: TextIO | None = None,
    ) -> None:
        """Set up MD of flexible molecules.

        Parameters
        ----------
        system : System
            Molecules (the pGM part of every template).
        templates : Sequence of FlexibleTemplate or RigidTemplate
            FlexibleTemplate or RigidTemplate per molecule of `system` (identical objects are
            shared).
        positions : ArrayLike (N, 3)
            Atom positions [nm] (molecules whole; virtual sites are rebuilt from their parents).
        box : ArrayLike (3, 3)
            Box [nm], lattice vectors as rows.
        settings : MDSettings
            Force-field, cutoff, PME and solver settings.
        dt : float
            Time step [ps] (the outer step with mts).
        temperature : float
            Temperature [K] of the thermostat, the barostat and drawn momenta.
        thermostat : Thermostat, str or None
            Langevin(friction), Bussi(tau), GLE..., a name for the default settings of a kind
            ("langevin" = Langevin(1/ps), the default), or None for NVE.
        barostat : MonteCarloBarostat or None
            Isotropic Monte Carlo barostat with molecular scaling (None: constant volume).
        velocities : ArrayLike (N, 3), optional
            Atom velocities [nm/ps]; default: drawn at the temperature.
        seed : int
            Seed of the random stream.
        params : dict, optional
            Force-field parameters (default: those of the system).
        restraints, alchemy, mts, bias, efield : optional
            As for Simulation.
        constraints : {"none", "h-bonds", "all-bonds"}
            "none", "h-bonds" (X-H bonds of the flexible templates) or "all-bonds" (every bond);
            rigid templates are always constrained (md/constraints.py, docs/shake.md).
        hmr : float, sequence or None
            Hydrogen mass [amu] for mass repartitioning (taken from the bonded heavy atom), one
            value (or None) per molecule, e.g. AmberSystem.hmr({"water": 4.0, "protein": 3.024}),
            or None (constraints.hmr_masses).
        constraint_options : dict, optional
            Keywords of md/constraints.Constraints (n_iter, dense_max, tol, bucket) and
            "max_single" (largest molecule solved as one block, md/topology.py).
        r_margin : float
            Margin [nm] added to the largest group radius for the molecular neighbour list.
        log : text stream or None
            Receives the rows of the log table of `run` too (diagnostics go to the logger
            "pgm_jax.md.flexible").

        Raises
        ------
        ValueError
            A box too small for the cutoff, templates that do not fit the settings, invalid
            options or combinations.
        """
        H = reduce_box(box)
        check_box(H, settings.pair_cutoff + settings.neighbors.skin)
        self.sys, self.settings, self.log = system, settings, log
        uniq = {id(t): t for t in templates}.values()
        for tpl in uniq:
            tpl.check_settings(settings)
        rules = {id(t): t.md_rule(constraints) for t in uniq}
        copts = dict(constraint_options or {})
        kw = {} if copts.get("max_single") is None else {"max_single": copts.pop("max_single")}
        copts.pop("max_single", None)
        self.topology = MDTopology.build(system, [rules[id(t)] for t in templates], **kw)
        masses = hmr_masses(system, hmr)
        self.vsites = VirtualSites.of(system)
        self.flex = FlexibleMolecules(system, positions, H, templates, self.topology, masses, self.vsites)
        self.rigid = self.flex  # wrap() / positions() used by the base driver
        self.ff = PGMForceField(
            system, H, settings, topology=self.topology, flux=ChargeFlux.from_templates(system, templates)
        )
        self.ff.masses = jnp.asarray(masses)
        self.constraints = Constraints(self.topology.constraints, self.topology.constraint_d0, masses, **copts)
        self.r_list = self._r_list = self.flex.r_max + r_margin
        self._make_neighbors(H)
        pos0 = self.flex.pos0
        self._size_lists(pos0, H)
        integ, extra = FlexibleIntegrator, {}
        if mts is not None:  # multiple time stepping: dt is the outer step
            from .mts import MTSFlexibleIntegrator

            integ, extra = MTSFlexibleIntegrator, {"mts": mts}
        self.integ = integ(
            self.ff,
            self.flex,
            self.nb,
            dt,
            constraints=self.constraints,
            temperature=temperature,
            thermostat=thermostat,
            barostat=barostat,
            params=params,
            restraints=restraints,
            alchemy=alchemy,
            bias=bias,
            efield=efield,
            **extra,
        )
        self.dt, self.ensemble, self.T0 = dt, self.integ.ensemble, temperature
        mom = None if velocities is None else self.flex.mass * jnp.asarray(velocities)
        self.state = self.integ.init(pos0, H, jax.random.PRNGKey(seed), mom)
        self.time_ps = 0.0
        nflex = sum(1 for t in templates if t.has_bonded)
        self._log.info(
            f"pgm_jax MD: {system.nmol} molecules ({nflex} flexible), {system.n} atoms"
            f"{'' if self.vsites is None else f' ({self.vsites.n_sites} virtual sites)'}, "
            f"{self.topology.n_group} list groups, {self.constraints.nc} constraints, {self._describe_coupling()}, "
            f"dt {dt * 1000:g} fs, {settings.precision} precision, PME grid {self.ff.pme.K} order "
            f"{settings.pme.order}, {settings.describe_cutoffs()}, {self.nb.kind} neighbour list (group radius "
            f"{self.r_list:.3f} nm), {settings.describe_induction()}, device {jax.devices()[0]}"
        )
        if self.constraints.nc:
            self._log.info(
                f"constraints ({constraints}): {self.constraints.describe()}; {self.integ.dof} degrees of freedom"
            )
        self._describe_options(alchemy, mts)

    def minimize(self, steps: int = 500, max_step: float = 0.01, ftol: float = 50.0, seed: int = 1) -> dict:
        """Minimise the energy by steepest descent, then draw new velocities at the temperature.

        Adaptive step (the atom with the largest force moves h <= max_step; h grows by 1.2 after an
        accepted step and halves after a rejected one), constraints kept by SHAKE and virtual
        sites placed, until the largest force is below ftol or `steps` evaluations.  Relaxes
        clashes of built structures (hydrogens added by tleap, solvent boxes) before dynamics.
        The bias state and a field amplitude set with set_field are kept.

        Parameters
        ----------
        steps : int
            Largest number of force evaluations.
        max_step : float
            Largest displacement of an atom per step [nm].
        ftol : float
            Stop when the largest atomic force is below this [kJ/mol/nm].
        seed : int
            Seed of the new velocities.

        Returns
        -------
        dict
            "steps" (evaluations), "accepted", "energy" [kJ/mol], "fmax" [kJ/mol/nm].
        """
        integ, st = self.integ, self.state
        cons = integ.cons

        @jax.jit
        def trial(pos: jax.Array, box: jax.Array, induction: Any, nbr: Any, h: float, F: jax.Array) -> tuple:
            """Return the trial positions h F / |F|_max from pos (SHAKE, sites) and their forces."""
            fmax = jnp.max(jnp.linalg.norm(F, axis=1))
            new = pos + h * F / jnp.maximum(fmax, 1e-12)
            if cons is not None:
                new = cons.positions(new, pos)
            new = integ.place(new, box)
            F1, res, nbr1 = integ._forces(new, box, induction, nbr, field=integ.field_at(st, st.step))
            return new, F1, res, nbr1

        pos, F, box = st.dyn.position, st.dyn.force, st.box
        induction, nbr, E = st.induction, st.nbr, float(st.epot)
        h, n_acc = float(max_step), 0
        for it in range(int(steps)):  # noqa: B007  (it + 1 = evaluations used, reported below)
            fmax = float(jnp.max(jnp.linalg.norm(F, axis=1)))
            if fmax < ftol:
                break
            new, F1, res, nbr1 = trial(pos, box, induction, nbr, h, F)
            E1 = float(res.energy["total"])
            if np.isfinite(E1) and E1 < E and not bool(res.overflow):
                pos, F, induction, nbr, E = new, F1, res.induction, nbr1, E1
                h = min(h * 1.2, max_step)
                n_acc += 1
            else:
                h *= 0.5
                if h < 1e-7:
                    break
        self._size_lists(pos, box)
        self.integ.compile()
        self.state = self.integ.init(pos, box, jax.random.PRNGKey(seed), bias=st.bias)
        if st.efield is not None:  # keep a field amplitude set with set_field
            self.state = self.integ.forces(self.state.set(efield=st.efield), False)
        out = {"steps": it + 1, "accepted": n_acc, "energy": E, "fmax": float(jnp.max(jnp.linalg.norm(F, axis=1)))}
        self._log.info(f"minimised: {out}")
        return out

    # ----------------------------------------------------------------- MDEngine hooks
    checkpoint_kind = "md-flexible"

    def _list_groups(self) -> tuple[np.ndarray, int]:
        """Return the groups of the molecular neighbour list (md/topology.py splits large molecules)."""
        return self.topology.group, self.topology.n_group

    def _list_centers(self, dynpos: jax.Array) -> jax.Array:
        """Return the centres of mass of the neighbour-list groups at atom positions `dynpos` [nm]."""
        return self.flex.list_centers(dynpos)

    def _after_block(self) -> None:
        """Check that every atom stays within the list radius of its group's centre.

        Raises
        ------
        RuntimeError
            An atom beyond the radius (increase r_margin).
        """
        ext = float(self.flex.extent(self.state.dyn.position))
        if self.nb.kind == "molecule" and ext > self.r_list:
            raise RuntimeError(
                f"an atom is {ext:.3f} nm from its group's centre, beyond the neighbour-list "
                f"radius {self.r_list:.3f} nm; increase r_margin"
            )

    def half_step_kinetic(self, st: MDState | None = None) -> float | None:
        """Return the kinetic energy [kJ/mol] as the mean of the two half-step values around a step.

        (K(p - h F / 2) + K(p + h F / 2)) / 2 = K(p) + h^2/8 (P F) M^-1 (P F), with P F the force
        projected onto the constraint tangent space and h the time step.  At full steps BAOAB and
        velocity Verlet under-estimate the kinetic energy of a mode of frequency w by the factor
        1 - (w h)^2 / 4 (harmonic limit; e.g. 3 % of the total for methanol at 2 fs with X-H
        constraints), while their configurational sampling is accurate; the half-step mean is exact
        in that limit (Amber's leapfrog reports the same average).  Not defined with multiple time
        stepping (returns None).  st: the state (None: the current one).
        """
        if getattr(self.integ, "mts", None) is not None:
            return None
        if not hasattr(self, "_half_ke_jit"):
            integ, cons, flex = self.integ, self.constraints, self.flex

            def f(st: MDState) -> jax.Array:
                """Return K(p) + dt^2/8 (P F) M^-1 (P F) (sites excluded)."""
                F = st.dyn.force
                if cons.nc:
                    F = cons.momenta(st.dyn.position, F, flex.masses)
                if integ.vsites is not None:
                    F = F * flex.real
                return integ.kinetic(st)[0] + integ.dt**2 / 8.0 * jnp.sum(F * F / flex.mass)

            self._half_ke_jit = jax.jit(f)
        return float(self._half_ke_jit(self.state if st is None else st))

    def observables(self) -> dict:
        """Return MDEngine.observables with the flexible engine's entries.

        temp_com / temp_internal (instead of temp_trans / temp_rot), temp_half (from
        `half_step_kinetic`) [K], and with constraints shake_err (largest relative length error)
        and rattle_err (Constraints.velocity_violation).
        """
        out = super().observables()
        out["temp_com"] = out.pop("temp_trans")
        out["temp_internal"] = out.pop("temp_rot")
        kh = self.half_step_kinetic()
        if kh is not None:  # at large dt the better kinetic temperature
            out["temp_half"] = 2.0 * kh / (self.integ.dof * KB)
        if self.constraints.nc:
            st = self.state
            out["shake_err"] = float(self.constraints.violation(st.dyn.position))
            out["rattle_err"] = float(
                self.constraints.velocity_violation(st.dyn.position, st.dyn.momentum, self.flex.masses)
            )
        return out

    def positions(self) -> np.ndarray:
        """Return the atom positions (N, 3) [nm] of the current state (molecules whole)."""
        return np.asarray(self.state.dyn.position)

    def velocities(self) -> np.ndarray:
        """Return the atom velocities (N, 3) [nm/ps] of the current state."""
        return np.asarray(self.state.dyn.momentum / self.flex.mass)


# ----------------------------------------------------------------------------- building a box
def liquid_box(
    tpl: FlexibleTemplate, n_mol: int, density: float, seed: int = 0, min_dist: float = 0.22, tries: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Return a cubic box of randomly rotated copies of the template's reference geometry.

    The copies sit on a cubic lattice (random sites of it when n_mol is not a cube) at the given
    density (start below the liquid density and let NPT compress); each copy is re-drawn until no
    two atoms of different molecules are closer than `min_dist`.

    Parameters
    ----------
    tpl : FlexibleTemplate
        The template.
    n_mol : int
        Number of molecules.
    density : float
        Density [g/cm^3].
    seed : int
        Random seed.
    min_dist : float
        Smallest intermolecular atom distance [nm].
    tries : int
        Rotations tried per molecule.

    Returns
    -------
    positions : np.ndarray (n_mol * n_atoms, 3)
        Positions [nm].
    box : np.ndarray (3, 3)
        Cubic box [nm].

    Raises
    ------
    RuntimeError
        A molecule could not be placed.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(tpl.spec.ref_xyz, float)
    m = np.asarray(tpl.pgm.masses, float)
    x0 = x0 - (m[:, None] * x0).sum(0) / m.sum()
    L = (n_mol * m.sum() * AMU_NM3_TO_G_CM3 / density) ** (1.0 / 3.0)
    k = int(np.ceil(n_mol ** (1.0 / 3.0)))
    a = L / k
    sites = np.array([(i, j, l) for i in range(k) for j in range(k) for l in range(k)], float) * a + a / 2
    if n_mol < k**3:  # spread over the whole box, not the first n_mol sites
        sites = sites[np.sort(rng.choice(k**3, n_mol, replace=False))]
    H = np.eye(3) * L
    placed = []

    def rot() -> np.ndarray:
        """Return a uniformly random rotation matrix (from a normalized Gaussian quaternion)."""
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    for s in sites:
        for _ in range(tries):
            y = x0 @ rot().T + s
            ok = True
            for other in placed:
                d = y[:, None, :] - other[None, :, :]
                d -= np.round(d / L) * L
                if np.min(np.sum(d * d, -1)) < min_dist**2:
                    ok = False
                    break
            if ok:
                break
        else:
            raise RuntimeError("could not place a molecule; lower the density")
        placed.append(y)
    return np.concatenate(placed), H
