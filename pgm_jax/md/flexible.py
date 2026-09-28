"""Flexible molecules for pGM MD: bonded terms on top of the periodic pGM force field, rigid
molecules by constraints, macromolecules.

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
    sim = FlexibleSimulation(System([tpl.pgm] * 256), [tpl] * 256, pos, H, MDSettings(),
                             dt=0.0005, ensemble="npt", temperature=298.0)
    sim.run(20000, report=2000)

    # a protein in rigid water, X-H bonds constrained, 2 fs
    sim = FlexibleSimulation(sys, [protein] + [RigidTemplate(water, xyz)] * n_wat, pos, H,
                             MDSettings(), dt=0.002, constraints="h-bonds", hmr=3.024)

Molecules are kept whole: positions are never wrapped atom by atom, only whole molecules are
shifted by lattice vectors.  Virtual sites (Molecule.vsites, md/vsites.py) are not integrated: they
are rebuilt from their parents every step and their forces are spread to their parents.
Units: nm, ps, amu, kJ/mol, K."""
from __future__ import annotations

import pickle
from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import numpy as np

from ..system import System
from ._jaxmd import simulate
from .box import check_box, inv3, reduce_box, volume
from .constraints import Constraints, hmr_masses
from .flux import ChargeFlux
from .forcefield import MDSettings, PGMForceField
from .integrate import KB, Dynamics, Integrator, MDState, upgrade_state
from .neighbors import AtomNeighbors, MoleculeNeighbors
from .rigid import _unwrap
from .simulation import Simulation
from .topology import MDTopology, MoleculeRule
from .vsites import VirtualSites


# ----------------------------------------------------------------------------- templates
class FlexibleTemplate:
    """Bonded parameters of one molecule type, fitted with pgm_jax.bonded, plus its pGM molecule.
    The MD engine treats electrostatics with all pairs, so the fit must have used pGM electrostatics
    without exclusions or refitted charges.  Charge flux (BondedSettings.flux) runs as fitted
    (md/flux.py; FlexibleSimulation builds it from the templates)."""

    def __init__(self, specs, settings: dict, P: dict, index: int = 0):
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
    def from_fit(cls, model, P, index: int = 0) -> "FlexibleTemplate":
        """From a fitted BondedModel and its parameters (molecule `index` of the model).  Neural
        bonded terms ("nnb") are frozen: their stage-1 coefficients are evaluated once here."""
        if getattr(model, "nnb", None) is not None and "coef" not in P["nnb"]:
            P = dict(P)
            P["nnb"] = model.nnb.freeze(P["nnb"])
        return cls(model.mols, asdict(model.s), P, index)

    @classmethod
    def from_network(cls, net, P, spec, **settings) -> "FlexibleTemplate":
        """Template of any molecule from a trained neural bonded model (bonded.nn.NNBonded and its
        parameters, e.g. NNBonded.load): stage 1 is evaluated for this molecule and frozen.
        settings: the BondedSettings the network was trained with (lj14_scale, lj_min_sep, elec, ...)."""
        from ..bonded.model import BondedSettings
        from ..bonded.topology import build_topology
        if spec.top is None:
            spec.top = build_topology(spec.elements, spec.bonds, (spec.bonds, spec.bond_orders), spec.ref_xyz * 10.0)
        C = net.coefficients(P, net.prepare(spec))
        c = net.config
        st = BondedSettings(families=("nnb",), nn_width=c.width, nn_layers=c.layers, nn_ref=c.ref, nn_basis=c.basis,
                            nn_b_span=c.b_span, nn_th_span=c.th_span, nn_out_scale=c.out_scale,
                            nn_pgm_features=c.pgm_features, nn_table_depth=c.table_depth, nn_context=c.context, **settings)
        return cls([spec], asdict(st), {"nnb": {"coef": [C]}}, 0)

    @property
    def spec(self):
        return self.specs[self.index]

    @property
    def pgm(self):
        return self.spec.pgm

    @property
    def name(self):
        return self.spec.name

    @property
    def terms(self):
        """The bonded terms (BondedTerms: no gas-phase nonbonded setup, any molecule size)."""
        if self._terms is None:
            from ..bonded.model import BondedSettings, BondedTerms
            self._terms = BondedTerms([replace(s) for s in self.specs], BondedSettings(**self.settings))
        return self._terms

    @property
    def model(self):
        """The full gas-phase model the bonded terms were fitted with (BondedModel: + pGM and
        intramolecular van der Waals), for reference energies of isolated molecules."""
        if self._model is None:
            from ..bonded.model import BondedModel, BondedSettings
            self._model = BondedModel([replace(s) for s in self.specs], BondedSettings(**self.settings))
        return self._model

    @property
    def n(self) -> int:
        return len(self.spec.elements)

    def lj_pairs(self):
        """Intramolecular van der Waals pairs (i, j, weight): graph distance >= lj_min_sep (1), 1-4
        (lj14_scale).  (The MD engine takes them from md/topology.py's special pairs.)"""
        top = self.terms.mols[self.index].top
        i, j = np.triu_indices(self.n, 1)
        d = top.dist[i, j] if top.dist is not None else np.array([top.graph_distance(a, b) for a, b in zip(i, j)])
        w = (d >= self.settings.get("lj_min_sep", 4)).astype(float)
        w = w + float(self.settings.get("lj14_scale", 0.0)) * (d == 3)
        keep = w > 0
        return i[keep], j[keep], w[keep]

    def bond_lengths(self) -> np.ndarray:
        """Reference bond lengths (nm) of the fitted model, in the order of the topology's bonds
        (the lengths X-H constraints hold)."""
        terms, P = self.terms, jax.tree_util.tree_map(np.asarray, self.P)
        if terms.fams:
            return np.asarray(P["ref"]["b0"])[terms.I[self.index]["bond"]]
        return np.asarray(P["nnb"]["coef"][self.index]["b0"])

    def md_rule(self, constraints: str = "none") -> MoleculeRule:
        """How the MD engine treats this molecule: intramolecular van der Waals by graph distance
        (lj_min_sep, lj14_scale of the fit); constraints "none" or "h-bonds" (X-H bonds at the
        model's reference lengths)."""
        top = self.terms.mols[self.index].top
        cons = []
        if constraints == "h-bonds":
            el = self.spec.elements
            for (i, j), b0 in zip(top.bonds, self.bond_lengths()):
                if (el[i] == "H") != (el[j] == "H"):
                    cons.append((int(i), int(j), float(b0)))
        elif constraints != "none":
            raise ValueError("constraints: 'none' | 'h-bonds'")
        return MoleculeRule(bonds=[tuple(int(x) for x in b) for b in top.bonds], vdw="graph",
                            lj_min_sep=int(self.settings.get("lj_min_sep", 4)),
                            lj14_scale=float(self.settings.get("lj14_scale", 0.0)), constraints=tuple(cons))

    has_bonded = True

    def check_settings(self, settings):
        """The MD model must be the model the bonded terms were fitted with."""
        st = self.terms.s
        for name in ("elec", "vdw", "gvdw_rep"):
            if getattr(st, name) != getattr(settings, name) and not (name == "gvdw_rep" and st.vdw != "gvdw"):
                raise ValueError(f"template {self.name} was fitted with {name}={getattr(st, name)!r}, "
                                 f"the MD settings have {getattr(settings, name)!r}")

    def bonded_energy(self, R, P=None):
        return self.terms.bonded_energy(self.index, R, jax.tree_util.tree_map(jnp.asarray, self.P if P is None else P))

    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump({"specs": self.specs, "settings": self.settings, "P": self.P, "index": self.index}, fh)

    @classmethod
    def load(cls, path: str) -> "FlexibleTemplate":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["specs"], d["settings"], d["P"], d["index"])


class RigidTemplate:
    """A rigid molecule of up to three atoms (water, ions) for FlexibleSimulation: every distance
    held by a constraint (from the geometry `xyz`, nm), no bonded terms, no intramolecular van der
    Waals (Amber's rigid water).  Virtual sites of the molecule (TIP4P's M site, TIP5P's lone pairs)
    come on top of the three atoms: they are placed, not constrained."""
    has_bonded = False

    def __init__(self, molecule, xyz=None, name: str | None = None):
        self.pgm = molecule
        self.name = name or molecule.name
        self.n = molecule.n
        sites = {vs.site for vs in molecule.vsites}
        self.real_atoms = [a for a in range(self.n) if a not in sites]
        if len(self.real_atoms) > 3:
            raise ValueError("RigidTemplate holds up to three atoms (plus virtual sites) by distances; "
                             "use a FlexibleTemplate")
        x = np.asarray(xyz if xyz is not None else molecule.extra.get("xyz"), float) if self.n > 1 else np.zeros((1, 3))
        self.xyz = x.reshape(self.n, 3)

    @property
    def spec(self):
        return self

    @property
    def elements(self):
        return list(self.pgm.elements)

    def md_rule(self, constraints: str = "none") -> MoleculeRule:
        ra = self.real_atoms
        pairs = [(i, j) for k, i in enumerate(ra) for j in ra[k + 1:]]
        cons = tuple((i, j, float(np.linalg.norm(self.xyz[i] - self.xyz[j]))) for i, j in pairs)
        return MoleculeRule(bonds=[], vdw="none", constraints=cons)

    def check_settings(self, settings):
        return None

    def bonded_energy(self, R, P=None):
        return 0.0


def _unwrap_bonded(x, H, bonds):
    """Make a molecule whole along its bonds (any size, unlike the minimum image of the first
    atom): every atom at the minimum image of the atom it is reached from."""
    n = len(x)
    if n <= 1 or not len(bonds):
        return _unwrap(x, H)
    Hinv = np.linalg.inv(H)
    nbr = [[] for _ in range(n)]
    for i, j in bonds:
        nbr[int(i)].append(int(j)); nbr[int(j)].append(int(i))
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
    """Per-atom dynamics for every molecule: bonded energies grouped by template, molecular and
    neighbour-list-group centres, whole-molecule wrapping.  `templates[k]` belongs to
    `sys.molecules[k]` (same atom order); `topology` is the system's MDTopology."""

    def __init__(self, sys: System, pos, H, templates, topology: MDTopology, masses=None,
                 vsites: VirtualSites | None = None):
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
        else:                  # the integrator's masses: 1 at the sites, whose momenta are held at 0 (`real`)
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
                    raise ValueError(f"template {tpl.name}: bonded terms involve virtual sites; "
                                     "sites carry no bonded terms")
            if tpl.has_bonded:
                groups.setdefault(id(tpl), (tpl, []))[1].append(np.arange(sl.start, sl.stop))
        self.groups = [(tpl, jnp.asarray(np.array(rows))) for tpl, rows in groups.values()]
        if vsites is not None:
            whole = np.asarray(vsites.place(whole, H))
            vsites.check(whole, H, (sys.cov_i, sys.cov_j))
        self.pos0 = jnp.asarray(whole)
        self.r_max = topology.group_radius(whole, m)
        self.dof_correction = 0

    def centers(self, pos):
        """Molecular centres of mass (wrapping, barostat scaling, molecular virial)."""
        return jax.ops.segment_sum(self.masses[:, None] * pos, self.mol, self.nmol) / self.mmol[:, None]

    def list_centers(self, pos):
        """Centres of mass of the neighbour-list groups."""
        return jax.ops.segment_sum(self.masses[:, None] * pos, self.group, self.n_group) / self.mgroup[:, None]

    def energy(self, pos, P_atoms=None):
        """Bonded energy (kJ/mol) of every flexible molecule (the intramolecular van der Waals pairs
        are in the force field's special pairs)."""
        e = 0.0
        for tpl, rows in self.groups:
            e = e + jnp.sum(jax.vmap(tpl.bonded_energy)(pos[rows]))
        return e

    def positions(self, pos):
        return pos

    def wrap(self, pos, H):
        """Whole molecules shifted so that their centres of mass lie in the primary cell."""
        H = jnp.asarray(H)
        hi = jax.lax.Precision.HIGHEST
        f = jnp.matmul(self.centers(pos), inv3(H), precision=hi)
        return pos - jnp.matmul(jnp.floor(f), H, precision=hi)[self.mol]

    def extent(self, pos):
        """Largest atom-to-group-centre distance (nm), to check the neighbour-list margin."""
        return jnp.max(jnp.linalg.norm(pos - self.list_centers(pos)[self.group], axis=1))


# ----------------------------------------------------------------------------- integrator
class FlexibleIntegrator(Integrator):
    """Velocity Verlet (NVE) / BAOAB (NVT, thermostats.py: Langevin, Bussi or GLE) on atoms, with
    constraints in g-BAOAB order (SHAKE after every drift, RATTLE after every kick and thermostat
    step; GLE auxiliaries are projected too); NPT adds the Monte Carlo barostat with molecular
    scaling (centres of mass scaled, molecules translated rigidly, which keeps the constraints)."""

    def __init__(self, ff: PGMForceField, flex: FlexibleMolecules, neighbors, dt: float = 0.0005,
                 constraints: Constraints | None = None, **kw):
        self.flex = flex
        self.vsites = getattr(flex, "vsites", None)
        self.cons = constraints if (constraints is not None and constraints.nc) else None
        super().__init__(ff, flex, neighbors, dt, **kw)
        nc = self.cons.nc if self.cons is not None else 0
        self.n_real = flex.n - (0 if self.vsites is None else self.vsites.n_sites)
        self.dof = 3 * self.n_real - nc - (3 if self.ensemble == "nve" else 0)

    def _forces(self, pos, box, induction, nbr, force_rebuild=False, lam=None, bias=None):
        centers = self.flex.list_centers(pos)
        nbr = self.nb.update(nbr, pos, centers, box, force_rebuild)
        cand, ovf = self.nb.candidates(nbr, centers, box, pos)
        if self.alchemy is None:
            res = self.ff.compute(pos, box, cand, induction, self.params, keep_geometry=self.keep_geometry)
        else:                                      # Hamiltonian at the state's coupling lam (alchemy.py)
            res = self.alchemy.compute(self.ff, pos, box, cand, induction, self.params, lam)
        e_in, g_in = jax.value_and_grad(self.flex.energy)(pos)
        energy = dict(res.energy)
        energy["total"] = res.energy["total"] + e_in
        res = self._add_restraints(res._replace(energy=energy, overflow=res.overflow | ovf), pos, box, bias)
        if self.vsites is not None:                          # site forces to the parents
            return self.vsites.spread(pos, box, res.forces - g_in), res, nbr
        return res.forces - g_in, res, nbr

    def _bias_atoms(self, x):
        return x

    def _map_atom_forces(self, x, box, F):
        return F if self.vsites is None else self.vsites.spread(x, box, F)

    def place(self, pos, box):
        """Positions with the virtual sites rebuilt from their parents (identity without sites)."""
        return pos if self.vsites is None else self.vsites.place(pos, box)

    def init(self, pos, box, key, momentum=None, bias=None) -> MDState:
        box = jnp.asarray(box, jnp.float64)
        pos = jnp.asarray(pos, jnp.float64)
        if self.cons is not None:                          # start on the constraint surface
            for _ in range(3):
                pos = self.cons.positions(pos, pos)
        pos = self.place(pos, box)
        nbr = self.nb.allocate(pos, self.flex.list_centers(pos), box)
        key, split = jax.random.split(key)
        zero = jnp.zeros_like(pos)
        dyn = Dynamics(pos, zero, zero, self.flex.mass, key)
        if momentum is None and self.vsites is not None:     # real atoms only, zero total momentum
            real = self.flex.real
            p = jnp.sqrt(self.flex.mass * self.kT) * jax.random.normal(split, pos.shape, jnp.float64) * real
            dyn = dyn.set(momentum=(p - jnp.sum(p, 0) / self.n_real) * real)
        elif momentum is None:
            dyn = simulate.initialize_momenta(dyn, split, self.kT)
        else:
            p = jnp.asarray(momentum, jnp.float64)
            dyn = dyn.set(momentum=p if self.vsites is None else p * self.flex.real)
        if self.cons is not None:
            dyn = dyn.set(momentum=self.cons.momenta(pos, dyn.momentum, self.flex.masses))
        z = jnp.zeros((), jnp.float64)
        zi = jnp.zeros((), jnp.int32)
        dyn, aux = self._init_aux(dyn)
        st = MDState(dyn=dyn, box=box, induction=self.ff.init_induction(), nbr=nbr, epot=z, elec=z, vdw=z,
                     iters=zi, max_iters=zi, resid=z, step=zi, mc=jnp.zeros(4, jnp.int32),
                     mc_dv=jnp.asarray(0.01 * float(volume(box)), jnp.float64), overflow=jnp.zeros((), bool),
                     aux=aux, heat=z, cg_total=z, bias=self._init_bias(bias))
        return self.forces(st, False)

    def _scaled(self, dyn: Dynamics):
        """Mass-scaled atomic momenta; the projection keeps them (and the thermostat
        auxiliaries) on the constraint tangent space."""
        sm = jnp.sqrt(dyn.mass)
        q = dyn.position
        if self.cons is None:
            project = lambda u: u                                      # noqa: E731
        else:
            project = lambda u: self.cons.momenta(q, u * sm, self.flex.masses) / sm   # noqa: E731
        mask = None if self.vsites is None else jnp.broadcast_to(self.flex.real, dyn.momentum.shape)   # sites: no noise
        return dyn.momentum / sm, mask, (lambda v: dyn.set(momentum=v * sm)), project

    # ------------------------------------------------------------------ constrained steps (g-BAOAB)
    def _kick(self, dyn: Dynamics, h: float) -> Dynamics:
        p = dyn.momentum + h * dyn.force
        return dyn.set(momentum=p if self.cons is None else self.cons.momenta(dyn.position, p, self.flex.masses))

    def _drift(self, dyn: Dynamics, h: float) -> Dynamics:
        """Positions advanced by h (SHAKE) and momenta consistent with the constrained move (RATTLE).
        Virtual sites have zero momentum, so they stay where they are; `_step` rebuilds them once
        per step, before the forces."""
        q = dyn.position
        q1 = q + h * dyn.momentum / dyn.mass
        if self.cons is not None:
            q1 = self.cons.positions(q1, q)
        p = dyn.mass * (q1 - q) / h
        return dyn.set(position=q1, momentum=p if self.cons is None else self.cons.momenta(q1, p, self.flex.masses))

    def _step(self, st: MDState) -> MDState:
        if self.cons is None and self.vsites is None:
            return super()._step(st)
        dt = self.dt
        aux, heat = st.aux, st.heat
        dyn = self._kick(st.dyn, dt / 2)
        if self.ensemble == "nve":
            dyn = self._drift(dyn, dt)
        else:
            dyn = self._drift(dyn, dt / 2)
            dyn, aux, heat = self._o_step(dyn, aux, heat, dt, self.thermostat_kT(st))
            dyn = self._drift(dyn, dt / 2)
        if self.vsites is not None:
            dyn = dyn.set(position=self.vsites.place(dyn.position, st.box))
        F, res, nbr = self._forces(dyn.position, st.box, st.induction, st.nbr, lam=st.lam, bias=st.bias)
        st = self._with_result(st.set(dyn=dyn, aux=aux, heat=heat), F, res, nbr)
        st = st.set(dyn=self._kick(st.dyn, dt / 2), step=st.step + 1)
        if self.ensemble == "npt":
            st = jax.lax.cond(st.step % self.interval == 0, self._barostat, lambda s: s, st)
        return st

    def _barostat(self, st: MDState) -> MDState:
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
        if self.alchemy is None:
            e_n, ind_n, _, ovf = self.ff.energy(pos_n, Hn, cand, st.induction, self.params)
        else:
            e_n, ind_n, _, ovf = self.alchemy.energy(self.ff, pos_n, Hn, cand, st.induction, self.params, st.lam)
        e_n = e_n + self.flex.energy(pos_n) + self._restraint_energy(pos_n, Hn, st.bias)
        ovf = ovf | ovf0
        kT = self.thermostat_kT(st)
        w = (e_n - st.epot) + self.pressure * dV - self.nmol * kT * jnp.log(jnp.maximum(Vn, 1e-12) / V)
        accept = (Vn > 0) & (jnp.log(jax.random.uniform(k2, dtype=jnp.float64)) < -w / kT)
        st = st.set(dyn=st.dyn.set(rng=key), overflow=st.overflow | ovf)

        def acc(st):
            st = st.set(dyn=st.dyn.set(position=pos_n), box=Hn, induction=ind_n)
            F, res, nbr = self._forces(pos_n, Hn, ind_n, nbr_n, lam=st.lam, bias=st.bias)
            return self._with_result(st, F, res, nbr)

        st = jax.lax.cond(accept, acc, lambda s: s, st)
        mc = st.mc + jnp.array([1, 0, 1, 0], jnp.int32) + accept.astype(jnp.int32) * jnp.array([0, 1, 0, 1], jnp.int32)
        adapt = mc[2] >= 10
        rate = mc[3] / jnp.maximum(mc[2], 1)
        dv = jnp.where(adapt & (rate < 0.25), st.mc_dv / 1.1, jnp.where(adapt & (rate > 0.75), st.mc_dv * 1.1, st.mc_dv))
        dv = jnp.minimum(dv, 0.3 * volume(st.box))
        mc = jnp.where(adapt, mc.at[2].set(0).at[3].set(0), mc)
        return st.set(mc=mc, mc_dv=dv)

    def kinetic(self, st: MDState):
        """(total, centre-of-mass translational) kinetic energy, kJ/mol."""
        p = st.dyn.momentum
        ke = 0.5 * jnp.sum(p * p / self.flex.mass)
        Pm = jax.ops.segment_sum(p, self.flex.mol, self.flex.nmol)
        return ke, 0.5 * jnp.sum(Pm * Pm / self.flex.mmol[:, None])

    def temperatures(self, st: MDState):
        """(centre-of-mass translational, internal) temperatures, K."""
        ke, ke_t = self.kinetic(st)
        n_t = 3 * self.nmol - (3 if self.ensemble == "nve" else 0)
        nc = self.cons.nc if self.cons is not None else 0
        n_i = 3 * self.n_real - 3 * self.nmol - nc
        return 2.0 * ke_t / (n_t * KB), 2.0 * (ke - ke_t) / (max(n_i, 1) * KB)


# ----------------------------------------------------------------------------- driver
class FlexibleSimulation(Simulation):
    """Simulation driver for flexible molecules (same reporting, trajectories and checkpoints as
    `Simulation`); `templates[k]` (FlexibleTemplate or RigidTemplate) belongs to `sys.molecules[k]`.
    constraints: "none" | "h-bonds" (X-H bonds of the flexible templates; rigid templates are
    always constrained); hmr: hydrogen mass (amu) for mass repartitioning (the mass comes from the
    bonded heavy atom), None, or one value (or None) per molecule, e.g. AmberSystem.hmr({"water":
    4.0, "protein": 3.024}) (constraints.hmr_masses); restraints: md/restraints.py; alchemy: an
    alchemical region (md/alchemy.py); mts: multiple time stepping (md/mts.py: MTS settings; dt is
    then the outer step); bias: biases on collective variables (pgm_jax.bias)."""

    def __init__(self, sys: System, templates, pos_nm, H_nm, settings: MDSettings = MDSettings(),
                 dt: float = 0.0005, ensemble: str = "nvt", temperature: float = 298.0, gamma: float = 1.0,
                 pressure: float = 1.0, barostat_interval: int = 100, seed: int = 0, vel_nm_ps=None,
                 params=None, log=None, neighbor_list: str = "auto", r_margin: float = 0.05,
                 constraints: str = "none", hmr=None, max_single: int | None = None,
                 thermostat="langevin", tau_t: float = 1.0, restraints=None, alchemy=None, mts=None, bias=None):
        H = reduce_box(H_nm)
        check_box(H, settings.pair_cutoff + settings.skin)
        self.sys, self.settings, self.log = sys, settings, log
        uniq = {id(t): t for t in templates}.values()
        for tpl in uniq:
            tpl.check_settings(settings)
        rules = {id(t): t.md_rule(constraints) for t in uniq}
        kw = {} if max_single is None else {"max_single": max_single}
        self.topology = MDTopology.build(sys, [rules[id(t)] for t in templates], **kw)
        masses = hmr_masses(sys, hmr)
        self.vsites = VirtualSites.of(sys)
        self.flex = FlexibleMolecules(sys, pos_nm, H, templates, self.topology, masses, self.vsites)
        self.rigid = self.flex                                   # wrap() / positions() used by the base driver
        self.ff = PGMForceField(sys, H, settings, topology=self.topology, flux=ChargeFlux.from_templates(sys, templates))
        self.ff.masses = jnp.asarray(masses)
        self.constraints = Constraints(self.topology.constraints, self.topology.constraint_d0, masses)
        self.r_list = self._r_list = self.flex.r_max + r_margin
        self._nb_mode = neighbor_list
        self._make_neighbors(H)
        pos0 = self.flex.pos0
        self._size_lists(pos0, H)
        integ, extra = FlexibleIntegrator, {}
        if mts is not None:                                  # multiple time stepping: dt is the outer step
            from .mts import MTSFlexibleIntegrator
            integ, extra = MTSFlexibleIntegrator, {"mts": mts}
        self.integ = integ(self.ff, self.flex, self.nb, dt, constraints=self.constraints, ensemble=ensemble,
                           temperature=temperature, gamma=gamma, pressure=pressure,
                           barostat_interval=barostat_interval, params=params,
                           thermostat=thermostat, tau_t=tau_t, restraints=restraints, alchemy=alchemy, bias=bias,
                           **extra)
        self.dt, self.ensemble, self.T0 = dt, ensemble, temperature
        mom = None if vel_nm_ps is None else self.flex.mass * jnp.asarray(vel_nm_ps)
        self.state = self.integ.init(pos0, H, jax.random.PRNGKey(seed), mom)
        self.time_ps = 0.0
        nflex = sum(1 for t in templates if t.has_bonded)
        self._print(f"# pgm_jax MD: {sys.nmol} molecules ({nflex} flexible), {sys.n} atoms"
                    f"{'' if self.vsites is None else f' ({self.vsites.n_sites} virtual sites)'}, "
                    f"{self.topology.n_group} list groups, {self.constraints.nc} constraints, {ensemble.upper()}"
                    f"{'' if self.integ.thermostat is None else ' (' + self.integ.thermostat.describe() + ')'}, "
                    f"dt {dt * 1000:g} fs, {settings.precision} precision, PME grid {self.ff.pme.K} order "
                    f"{settings.pme_order}, {settings.describe_cutoffs()}, {self.nb.kind} neighbour list (group radius "
                    f"{self.r_list:.3f} nm), dipole tol {settings.dipole_tol:g}, device {jax.devices()[0]}")
        if self.integ.restraints is not None:
            self._print(f"# restraints: {self.integ.restraints.describe()}")
        if self.ff.flux is not None:
            self._print(f"# {self.ff.flux.describe()}")
        if alchemy is not None:
            self._print(f"# alchemical region: {alchemy.describe()}")
        if mts is not None:
            self._print(f"# {self.integ.describe_mts()}")
        self._describe_bias()

    def minimize(self, steps: int = 500, max_step: float = 0.01, ftol: float = 50.0, seed: int = 1) -> dict:
        """Steepest descent (adaptive step, at most max_step nm per atom, constraints kept by SHAKE)
        until the largest force is below ftol (kJ/mol/nm) or `steps` evaluations; then new
        velocities are drawn at the target temperature.  Relaxes clashes of built structures
        (hydrogens added by tleap, solvent boxes) before dynamics."""
        integ, st = self.integ, self.state
        cons = integ.cons

        @jax.jit
        def trial(pos, box, induction, nbr, h, F):
            fmax = jnp.max(jnp.linalg.norm(F, axis=1))
            new = pos + h * F / jnp.maximum(fmax, 1e-12)
            if cons is not None:
                new = cons.positions(new, pos)
            new = integ.place(new, box)
            F1, res, nbr1 = integ._forces(new, box, induction, nbr)
            return new, F1, res, nbr1

        pos, F, box = st.dyn.position, st.dyn.force, st.box
        induction, nbr, E = st.induction, st.nbr, float(st.epot)
        h, n_acc = float(max_step), 0
        for it in range(int(steps)):
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
        out = {"steps": it + 1, "accepted": n_acc, "energy": E, "fmax": float(jnp.max(jnp.linalg.norm(F, axis=1)))}
        self._print(f"# minimised: {out}")
        return out

    def _make_neighbors(self, H):
        s = self.settings
        mode = self._nb_mode
        if mode == "auto":
            mode = "molecule" if MoleculeNeighbors.fits(H, s.pair_cutoff, s.skin, self._r_list) else "atom"
        if mode == "molecule":
            self.nb = MoleculeNeighbors(self.topology.group, self.topology.n_group, self._r_list, H, s.pair_cutoff,
                                        s.skin)
        else:
            self.nb = AtomNeighbors(self.sys.n, H, s.pair_cutoff, s.skin)
        self._nb_volume = float(volume(jnp.asarray(H)))

    def _size_lists(self, pos, H, factor: float = 1.2, nbr=None):
        c = self.flex.list_centers(pos)
        nbr = self.nb.allocate(pos, c, H) if nbr is None else nbr
        if self.nb.kind == "molecule":
            self.nb.size(nbr, c, H, pos, factor)
        idx = self.nb.candidates(nbr, c, H, pos)[0]
        self.ff.size_rows(pos, H, idx, factor)
        return nbr

    def _advance(self, n: int):
        super()._advance(n)
        ext = float(self.flex.extent(self.state.dyn.position))
        if self.nb.kind == "molecule" and ext > self.r_list:
            raise RuntimeError(f"an atom is {ext:.3f} nm from its group's centre, beyond the neighbour-list "
                               f"radius {self.r_list:.3f} nm; increase r_margin")

    def observables(self) -> dict:
        out = super().observables()
        out["temp_com"] = out.pop("temp_trans")
        out["temp_internal"] = out.pop("temp_rot")
        if self.constraints.nc:
            out["shake_err"] = float(self.constraints.violation(self.state.dyn.position))
        return out

    def _pressure(self, st):
        pos = st.dyn.position
        c = self.flex.list_centers(pos)
        idx = self.nb.candidates(st.nbr, c, st.box, pos)[0]
        if self.integ.alchemy is None:
            W = self.ff.strain_derivative(pos, st.box, idx, st.induction.mu, self.integ.params)
        else:
            W = self.integ.alchemy.strain_derivative(self.ff, pos, st.box, idx, st.induction.mu, self.integ.params, st.lam)
        W = W + self.integ.restraint_strain(pos, st.box, st.bias)
        ke_t = self.integ.kinetic(st)[1]
        return (2.0 * ke_t - jnp.trace(W)) / (3.0 * volume(st.box)) * 16.605390671738466

    def positions_nm(self):
        return np.asarray(self.state.dyn.position)

    def velocities_nm_ps(self):
        return np.asarray(self.state.dyn.momentum / self.flex.mass)

    def load(self, path: str):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        st = self._bias_of_checkpoint(upgrade_state(jax.tree_util.tree_map(jnp.asarray, d["state"]), self.state.aux))
        pos = st.dyn.position
        self.state = st.set(nbr=self.nb.allocate(pos, self.flex.list_centers(pos), st.box))
        self.time_ps = d["time_ps"]


# ----------------------------------------------------------------------------- building a box
def liquid_box(tpl: FlexibleTemplate, n_mol: int, density: float, seed: int = 0, min_dist: float = 0.22,
               tries: int = 200):
    """n_mol copies of the template's reference geometry, randomly rotated, on a cubic lattice of
    the given density (g/cm^3; start below the liquid density and let NPT compress).  Returns
    positions (n_mol * n_atoms, 3) nm and the box H (3, 3) nm; copies are re-drawn until no two
    atoms of different molecules are closer than `min_dist` nm."""
    rng = np.random.default_rng(seed)
    x0 = np.asarray(tpl.spec.ref_xyz, float)
    m = np.asarray(tpl.pgm.masses, float)
    x0 = x0 - (m[:, None] * x0).sum(0) / m.sum()
    L = (n_mol * m.sum() * 1.66053906660e-3 / density) ** (1.0 / 3.0)
    k = int(np.ceil(n_mol ** (1.0 / 3.0)))
    a = L / k
    sites = np.array([(i, j, l) for i in range(k) for j in range(k) for l in range(k)], float) * a + a / 2
    if n_mol < k ** 3:                                  # spread over the whole box, not the first n_mol sites
        sites = sites[np.sort(rng.choice(k ** 3, n_mol, replace=False))]
    H = np.eye(3) * L
    placed = []

    def rot():
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    for s in sites:
        for _ in range(tries):
            y = x0 @ rot().T + s
            ok = True
            for other in placed:
                d = y[:, None, :] - other[None, :, :]
                d -= np.round(d / L) * L
                if np.min(np.sum(d * d, -1)) < min_dist ** 2:
                    ok = False
                    break
            if ok:
                break
        else:
            raise RuntimeError("could not place a molecule; lower the density")
        placed.append(y)
    return np.concatenate(placed), H
