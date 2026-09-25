"""Flexible molecules for pGM MD: bonded terms and intramolecular Lennard-Jones on top of the
periodic pGM force field.

A molecule type is described by a `FlexibleTemplate`: its pGM electrostatics (the `Molecule` the
force field uses), and bonded parameters fitted with `pgm_jax.bonded` (families, reference values,
force constants).  The force field already treats every intramolecular pair electrostatically (pGM
has no exclusions), so the extra energy of a flexible molecule is

    E_intra = E_bonded(R) + sum_{pairs d_ij >= lj_min_sep} LJ_ij + lj14_scale sum_{1-4 pairs} LJ_ij,

exactly the model the bonded terms were fitted with (`BondedModel.energy` in the gas phase).
Molecules are kept whole: positions are never wrapped atom by atom, only whole molecules are
shifted by lattice vectors.

    tpl = FlexibleTemplate.from_fit(model, P)                 # after fitting pgm_jax.bonded
    tpl.save("methanol.flex")
    pos, H = liquid_box(tpl, 256, density=0.75)
    sim = FlexibleSimulation(System([tpl.pgm] * 256), [tpl] * 256, pos, H, MDSettings(),
                             dt=0.0005, ensemble="npt", temperature=298.0)
    sim.run(20000, report=2000)

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
from .forcefield import MDSettings, PGMForceField
from .integrate import KB, Dynamics, Integrator, MDState
from .neighbors import AtomNeighbors, MoleculeNeighbors
from .rigid import _unwrap
from .simulation import Simulation


# ----------------------------------------------------------------------------- templates
class FlexibleTemplate:
    """Bonded parameters of one molecule type, fitted with pgm_jax.bonded, plus its pGM molecule.
    The MD engine treats electrostatics with all pairs, so the fit must have used pGM electrostatics
    without exclusions, charge flux or refitted charges."""

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
        if st.flux:
            bad.append("charge flux")
        if st.qfit >= 0 or st.qbci >= 0:
            bad.append("fitted charges")
        if st.escale:
            bad.append("learned pair scales")
        if st.ind_exclude not in (-1, 0):
            bad.append("induction exclusions")
        if bad:
            raise ValueError("the MD engine uses pGM with all pairs; this fit used " + ", ".join(bad))
        self._model = None

    @classmethod
    def from_fit(cls, model, P, index: int = 0) -> "FlexibleTemplate":
        """From a fitted BondedModel and its parameters (molecule `index` of the model)."""
        return cls(model.mols, asdict(model.s), P, index)

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
    def model(self):
        if self._model is None:
            from ..bonded.model import BondedModel, BondedSettings
            self._model = BondedModel([replace(s) for s in self.specs], BondedSettings(**self.settings))
        return self._model

    @property
    def n(self) -> int:
        return len(self.spec.elements)

    def lj_pairs(self):
        """Intramolecular LJ pairs (i, j, weight): graph distance >= lj_min_sep (1), 1-4 (lj14_scale)."""
        D = self.model.mols[self.index].top.dist
        i, j = np.triu_indices(self.n, 1)
        d = D[i, j]
        w = (d >= self.settings.get("lj_min_sep", 4)).astype(float)
        w = w + float(self.settings.get("lj14_scale", 0.0)) * (d == 3)
        keep = w > 0
        return i[keep], j[keep], w[keep]

    def bonded_energy(self, R, P=None):
        return self.model.bonded_energy(self.index, R, jax.tree_util.tree_map(jnp.asarray, self.P if P is None else P))

    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump({"specs": self.specs, "settings": self.settings, "P": self.P, "index": self.index}, fh)

    @classmethod
    def load(cls, path: str) -> "FlexibleTemplate":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["specs"], d["settings"], d["P"], d["index"])


# ----------------------------------------------------------------------------- molecules
class FlexibleMolecules:
    """Per-atom dynamics for every molecule, grouped by template for the intramolecular energy.
    `templates[k]` belongs to `sys.molecules[k]` (same atom order)."""

    def __init__(self, sys: System, pos, H, templates):
        if len(templates) != sys.nmol:
            raise ValueError("one template per molecule")
        pos, H = np.asarray(pos, float), np.asarray(H, float)
        self.sys, self.nmol, self.n = sys, sys.nmol, sys.n
        self.mol = jnp.asarray(sys.mol)
        m = np.asarray(sys.masses, float)
        self.masses = jnp.asarray(m)
        self.mass = jnp.asarray(m)[:, None]
        self.mmol = jax.ops.segment_sum(self.masses, self.mol, self.nmol)
        whole = np.zeros_like(pos)
        groups = {}
        for k, (molk, tpl) in enumerate(zip(sys.molecules, templates)):
            sl = sys.atom_slice(k)
            if len(tpl.spec.elements) != molk.n or list(tpl.spec.elements) != list(molk.elements):
                raise ValueError(f"template {tpl.name} does not match molecule {k} ({molk.name})")
            whole[sl] = _unwrap(pos[sl], H)
            groups.setdefault(id(tpl), (tpl, []))[1].append(np.arange(sl.start, sl.stop))
        self.groups = []
        for tpl, rows in groups.values():
            i, j, w = tpl.lj_pairs()
            self.groups.append((tpl, jnp.asarray(np.array(rows)), (jnp.asarray(i), jnp.asarray(j), jnp.asarray(w))))
        self.pos0 = jnp.asarray(whole)
        c = self.centers(self.pos0)
        self.r_max = float(jnp.max(jnp.linalg.norm(self.pos0 - c[self.mol], axis=1)))
        self.dof_correction = 0

    def centers(self, pos):
        return jax.ops.segment_sum(self.masses[:, None] * pos, self.mol, self.nmol) / self.mmol[:, None]

    def energy(self, pos, P_atoms):
        """Intramolecular energy (kJ/mol): bonded terms + intramolecular LJ with the force field's
        per-atom LJ parameters (so LJ parameter gradients include the intramolecular pairs)."""
        e = 0.0
        rh, se = P_atoms["lj_rmin_half"], P_atoms["lj_sqrt_eps"]
        for tpl, rows, (i, j, w) in self.groups:
            X = pos[rows]                                              # (nmol_t, nat, 3)
            e = e + jnp.sum(jax.vmap(tpl.bonded_energy)(X))
            if len(i):
                r = jnp.linalg.norm(X[:, i] - X[:, j], axis=-1)
                s6 = ((rh[rows][:, i] + rh[rows][:, j]) / r) ** 6
                e = e + jnp.sum(w * se[rows][:, i] * se[rows][:, j] * (s6 * s6 - 2.0 * s6))
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
        """Largest atom-to-centre distance (nm), to check the molecule neighbour list margin."""
        return jnp.max(jnp.linalg.norm(pos - self.centers(pos)[self.mol], axis=1))


# ----------------------------------------------------------------------------- integrator
class FlexibleIntegrator(Integrator):
    """Velocity Verlet (NVE) / BAOAB Langevin (NVT) on atoms; NPT adds the Monte Carlo barostat
    with molecular scaling (centres of mass scaled, molecules translated rigidly)."""

    def __init__(self, ff: PGMForceField, flex: FlexibleMolecules, neighbors, dt: float = 0.0005, **kw):
        self.flex = flex
        super().__init__(ff, flex, neighbors, dt, **kw)
        self.dof = 3 * flex.n - (3 if self.ensemble == "nve" else 0)

    def _forces(self, pos, box, induction, nbr, force_rebuild=False):
        centers = self.flex.centers(pos)
        nbr = self.nb.update(nbr, pos, centers, box, force_rebuild)
        cand, ovf = self.nb.candidates(nbr, centers, box, pos)
        res = self.ff.compute(pos, box, cand, induction, self.params)
        e_in, g_in = jax.value_and_grad(self.flex.energy)(pos, self.ff._atoms(self.params))
        energy = dict(res.energy)
        energy["total"] = res.energy["total"] + e_in
        res = res._replace(energy=energy, overflow=res.overflow | ovf)
        return res.forces - g_in, res, nbr

    def init(self, pos, box, key, momentum=None) -> MDState:
        box = jnp.asarray(box, jnp.float64)
        pos = jnp.asarray(pos, jnp.float64)
        nbr = self.nb.allocate(pos, self.flex.centers(pos), box)
        key, split = jax.random.split(key)
        zero = jnp.zeros_like(pos)
        dyn = Dynamics(pos, zero, zero, self.flex.mass, key)
        if momentum is None:
            dyn = simulate.initialize_momenta(dyn, split, self.kT)
        else:
            dyn = dyn.set(momentum=jnp.asarray(momentum, jnp.float64))
        z = jnp.zeros((), jnp.float64)
        zi = jnp.zeros((), jnp.int32)
        st = MDState(dyn=dyn, box=box, induction=self.ff.init_induction(), nbr=nbr, epot=z, elec=z, vdw=z,
                     iters=zi, max_iters=zi, resid=z, step=zi, mc=jnp.zeros(4, jnp.int32),
                     mc_dv=jnp.asarray(0.01 * float(volume(box)), jnp.float64), overflow=jnp.zeros((), bool))
        return self.forces(st, False)

    def _ou_step(self, dyn: Dynamics, dt: float) -> Dynamics:
        key, k1 = jax.random.split(dyn.rng)
        c = jnp.exp(-self.gamma_value * dt)
        s = jnp.sqrt(self.kT * (1.0 - c * c))
        P = c * dyn.momentum + s * jnp.sqrt(dyn.mass) * jax.random.normal(k1, dyn.momentum.shape, dyn.momentum.dtype)
        return dyn.set(momentum=P, rng=key)

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
        c_n = self.flex.centers(pos_n)
        nbr_n = self.nb.update(st.nbr, pos_n, c_n, Hn, True)
        cand, ovf0 = self.nb.candidates(nbr_n, c_n, Hn, pos_n)
        e_n, ind_n, _, ovf = self.ff.energy(pos_n, Hn, cand, st.induction, self.params)
        e_n = e_n + self.flex.energy(pos_n, self.ff._atoms(self.params))
        ovf = ovf | ovf0
        w = (e_n - st.epot) + self.pressure * dV - self.nmol * self.kT * jnp.log(jnp.maximum(Vn, 1e-12) / V)
        accept = (Vn > 0) & (jnp.log(jax.random.uniform(k2, dtype=jnp.float64)) < -w / self.kT)
        st = st.set(dyn=st.dyn.set(rng=key), overflow=st.overflow | ovf)

        def acc(st):
            st = st.set(dyn=st.dyn.set(position=pos_n), box=Hn, induction=ind_n)
            F, res, nbr = self._forces(pos_n, Hn, ind_n, nbr_n)
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
        n_i = 3 * self.flex.n - 3 * self.nmol
        return 2.0 * ke_t / (n_t * KB), 2.0 * (ke - ke_t) / (max(n_i, 1) * KB)


# ----------------------------------------------------------------------------- driver
class FlexibleSimulation(Simulation):
    """Simulation driver for flexible molecules (same reporting, trajectories and checkpoints as
    `Simulation`); `templates[k]` gives the bonded parameters of `sys.molecules[k]`."""

    def __init__(self, sys: System, templates, pos_nm, H_nm, settings: MDSettings = MDSettings(),
                 dt: float = 0.0005, ensemble: str = "nvt", temperature: float = 298.0, gamma: float = 1.0,
                 pressure: float = 1.0, barostat_interval: int = 100, seed: int = 0, vel_nm_ps=None,
                 params=None, log=None, neighbor_list: str = "auto", r_margin: float = 0.05):
        H = reduce_box(H_nm)
        check_box(H, settings.cutoff + settings.skin)
        self.sys, self.settings, self.log = sys, settings, log
        self.flex = FlexibleMolecules(sys, pos_nm, H, templates)
        self.rigid = self.flex                                   # wrap() / positions() used by the base driver
        self.ff = PGMForceField(sys, H, settings)
        self.r_list = self._r_list = self.flex.r_max + r_margin
        self._nb_mode = neighbor_list
        self._make_neighbors(H)
        pos0 = self.flex.pos0
        self._size_lists(pos0, H)
        self.integ = FlexibleIntegrator(self.ff, self.flex, self.nb, dt, ensemble=ensemble, temperature=temperature,
                                        gamma=gamma, pressure=pressure, barostat_interval=barostat_interval,
                                        params=params)
        self.dt, self.ensemble, self.T0 = dt, ensemble, temperature
        mom = None if vel_nm_ps is None else self.flex.mass * jnp.asarray(vel_nm_ps)
        self.state = self.integ.init(pos0, H, jax.random.PRNGKey(seed), mom)
        self.time_ps = 0.0
        self._print(f"# pgm_jax MD: {sys.nmol} flexible molecules, {sys.n} atoms, {ensemble.upper()}, dt {dt * 1000:g} fs, "
                    f"{settings.precision} precision, PME grid {self.ff.pme.K} order {settings.pme_order}, "
                    f"cutoff {settings.cutoff} nm, {self.nb.kind} neighbour list (molecule radius {self.r_list:.3f} nm), "
                    f"dipole tol {settings.dipole_tol:g}, device {jax.devices()[0]}")

    def _size_lists(self, pos, H, factor: float = 1.2, nbr=None):
        c = self.flex.centers(pos)
        nbr = self.nb.allocate(pos, c, H) if nbr is None else nbr
        if self.nb.kind == "molecule":
            self.nb.size(nbr, c, H, pos, factor)
        idx = self.nb.candidates(nbr, c, H, pos)[0]
        self.ff.mc = None
        cmax = int(jax.jit(self.ff.row_counts)(jnp.asarray(pos), jnp.asarray(H), idx))
        width = int(idx.shape[1]) + int(self.ff.intra.shape[1])
        self.ff.mc = min(int(np.ceil((cmax * (1.0 + 0.5 * (factor - 1.0)) + 8) / 8.0) * 8), width)
        return nbr

    def _advance(self, n: int):
        super()._advance(n)
        ext = float(self.flex.extent(self.state.dyn.position))
        if self.nb.kind == "molecule" and ext > self.r_list:
            raise RuntimeError(f"an atom is {ext:.3f} nm from its molecule's centre, beyond the neighbour-list "
                               f"radius {self.r_list:.3f} nm; increase r_margin")

    def observables(self) -> dict:
        out = super().observables()
        out["temp_com"] = out.pop("temp_trans")
        out["temp_internal"] = out.pop("temp_rot")
        return out

    def _pressure(self, st):
        pos = st.dyn.position
        c = self.flex.centers(pos)
        W = self.ff.strain_derivative(pos, st.box, self.nb.candidates(st.nbr, c, st.box, pos)[0],
                                      st.induction.mu, self.integ.params)
        ke_t = self.integ.kinetic(st)[1]
        return (2.0 * ke_t - jnp.trace(W)) / (3.0 * volume(st.box)) * 16.605390671738466

    def positions_nm(self):
        return np.asarray(self.state.dyn.position)

    def velocities_nm_ps(self):
        return np.asarray(self.state.dyn.momentum / self.flex.mass)

    def load(self, path: str):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        st = jax.tree_util.tree_map(jnp.asarray, d["state"])
        pos = st.dyn.position
        self.state = st.set(nbr=self.nb.allocate(pos, self.flex.centers(pos), st.box))
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
