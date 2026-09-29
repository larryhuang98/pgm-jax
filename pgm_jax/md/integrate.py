"""Integrators for rigid-molecule pGM MD built from JAX-MD's simulation primitives.

  NVE : velocity Verlet with the NO_SQUISH free-rotor splitting for quaternions (JAX-MD
        simulate.momentum_step / position_step, rigid_body registrations; Miller et al. 2002).
  NVT : BAOAB with the O step of a thermostat from thermostats.py (Langevin friction `gamma`,
        Bussi rescaling `tau_t`, or a smooth GLE).  For rigid bodies the thermostat acts on the
        mass-scaled centre-of-mass momenta and body-frame angular momenta, L_l / sqrt(I_l), mapped
        to and from the quaternion conjugate momenta.  (JAX-MD 0.2.29's rigid-body stochastic_step
        draws the quaternion-momentum noise with a diagonal covariance instead of
        sum_l s_l^2 P_l P_l^T, which under-heats the rotations: rigid water settled 15-25 K below
        the target.)  The heat exchanged in the O steps is booked in MDState.heat, so
        E_tot + |aux|^2/2 - heat is conserved up to integration and induction errors.
  NPT : NVT plus an isotropic Monte Carlo barostat every `barostat_interval` steps (Amber
        barostat = 2, OpenMM MonteCarloBarostat): centres of mass and box are scaled,
        orientations and momenta are kept, acceptance on dE + P dV - N kT ln(V'/V); the
        maximum volume change adapts to 25-75 % acceptance.
Restraints (restraints.py, `restraints=`) add their energy to the potential energy and their
atomic forces to the force-field forces before these are mapped to the bodies; the barostat's trial
energy includes them at the scaled positions and box.
Biases on collective variables (pgm_jax/bias, `bias=`: metadynamics, OPES, static biases) add
V(s(x)) like the restraints, with a state MDState.bias (hills, kernels, COLVAR buffer) that is
updated after the steps at which a bias deposits, inside the compiled loop; the forces of the new
bias replace those of the old one at once (so the next kick uses them) and the change of V at
fixed positions is booked as heat (econs stays conserved) and in BiasState.work.
An alchemical region (alchemy.py, `alchemy=`) makes the Hamiltonian depend on the state's coupling
MDState.lam = (lambda_elec, lambda_vdw), a traced value like kT, so lambda windows share one
compiled step (batched with jax.vmap); without one the step is unchanged.
An external electric field (efield.py, `efield=`: an ExternalField or three numbers in V/nm) adds
-E(t) . M to the potential energy and q E to the forces, and enters the induction equations.  Its
amplitude is the state variable MDState.efield (V/nm; Simulation.set_field changes it without
recompiling), modulated by cos(omega t + phase) for a time-dependent field, whose explicit time
dependence is booked in MDState.heat.  Without a field the step is unchanged.

Units: nm, ps, amu, kJ/mol, K."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._jaxmd import dataclasses, rigid_body, simulate, space
from .box import volume
from .forcefield import InductionState, PGMForceField
from .efield import as_field
from .restraints import as_restraints, molecular_strain
from .rigid import RigidBody, RigidMolecules
from .thermostats import Thermostat, make_thermostat

KB = 0.0083144626181532                  # kJ/mol/K
BAR = 1.0 / 16.605390671738466           # kJ/mol/nm^3 per bar


@dataclasses.dataclass
class Dynamics:
    """The part of the state JAX-MD's primitives act on."""
    position: RigidBody
    momentum: RigidBody
    force: RigidBody
    mass: RigidBody
    rng: jnp.ndarray


@dataclasses.dataclass
class MDState:
    dyn: Dynamics
    box: jnp.ndarray                  # H, lattice vectors as rows (nm)
    induction: InductionState
    nbr: object                       # jax_md.partition.NeighborList (dense)
    epot: jnp.ndarray                 # kJ/mol (float64)
    elec: jnp.ndarray
    vdw: jnp.ndarray
    iters: jnp.ndarray                # CG iterations of the last solve
    max_iters: jnp.ndarray            # largest since the last report
    resid: jnp.ndarray                # largest final residual since the last report
    step: jnp.ndarray
    mc: jnp.ndarray                   # (tries, accepts, window tries, window accepts) int32
    mc_dv: jnp.ndarray                # current maximum volume change (nm^3)
    overflow: jnp.ndarray             # a force evaluation exceeded the row capacity (block must be repeated)
    aux: jnp.ndarray = None           # thermostat auxiliary momenta (mass-scaled), (n_aux,) + momenta shape
    heat: jnp.ndarray = None          # heat taken up in the thermostat steps since the start (kJ/mol)
    cg_total: jnp.ndarray = None      # CG iterations summed over all force evaluations (float64)
    kT: jnp.ndarray = None            # thermostat kB T (kJ/mol) as a state variable (replica exchange, remd.py:
                                      # one compiled step for every temperature); None: Integrator.kT
    lam: jnp.ndarray = None           # (2,) alchemical coupling (lambda_elec, lambda_vdw) of the state (alchemy.py:
                                      # lambda windows share one compiled step); None: the Alchemy's default
    mts: object = None                # multiple time stepping: forces of each level, short-range list (mts.MTSState)
    bias: object = None               # state of the biases (pgm_jax.bias.BiasState) or None
    efield: jnp.ndarray = None        # (3,) V/nm amplitude of the external field (efield.py); None: no field
    fshift: jnp.ndarray = None        # (3,) e nm dipole of the re-wrapped charged molecules (itinerant charges)
    fdip: jnp.ndarray = None          # (3,) e nm M of the last force evaluation (the dipole the field acts on)


def upgrade_state(st: MDState, aux) -> MDState:
    """States pickled before the thermostat fields existed: add them."""
    if getattr(st, "aux", None) is None:
        st = st.set(aux=aux)
    if getattr(st, "heat", None) is None:
        st = st.set(heat=jnp.zeros((), jnp.float64))
    if getattr(st, "cg_total", None) is None:
        st = st.set(cg_total=jnp.zeros((), jnp.float64))
    return st


def strided_loop(st, n, step, post, stride: int):
    """n steps with post(state) after every step whose counter is a multiple of `stride`, written
    as plain inner loops of `stride` steps, so the steps in between carry no conditional (on a GPU
    each lax.cond reads its predicate on the host: two per step cost ~0.1 ms)."""
    s0 = st.step if jnp.ndim(st.step) == 0 else st.step.reshape(-1)[0]
    head = (stride - s0 % stride) % stride
    h = jnp.minimum(head, n)
    st = jax.lax.fori_loop(0, h, lambda _, s: step(s), st)
    st = jax.lax.cond((h == head) & (head > 0), post, lambda s: s, st)
    rem = n - h
    nch = rem // stride
    inner = lambda s: jax.lax.fori_loop(0, stride, lambda _, t: step(t), s)          # noqa: E731
    st = jax.lax.fori_loop(0, nch, lambda _, s: post(inner(s)), st)
    return jax.lax.fori_loop(0, rem - nch * stride, lambda _, s: step(s), st)


def field_state(st: MDState, field) -> MDState:
    """The external-field fields of a state for an integrator with `field` (an ExternalField or
    None): amplitude (kept if the state has one), offset and dipole, or all None without a field."""
    if field is None:
        return st.set(efield=None, fshift=None, fdip=None)
    z = jnp.zeros(3, jnp.float64)
    E = getattr(st, "efield", None)
    return st.set(efield=jnp.asarray(field.E0, jnp.float64) if E is None else jnp.asarray(E, jnp.float64),
                  fshift=z if getattr(st, "fshift", None) is None else st.fshift,
                  fdip=z if getattr(st, "fdip", None) is None else st.fdip)


class Integrator:
    keep_geometry = False             # ask the force field for its row geometry (multiple time stepping, mts.py)

    def __init__(self, ff: PGMForceField, rigid: RigidMolecules, neighbors, dt: float = 0.001,
                 ensemble: str = "nvt", temperature: float = 298.0, gamma: float = 1.0,
                 pressure: float = 1.0, barostat_interval: int = 100, params=None,
                 thermostat: str | Thermostat = "langevin", tau_t: float = 1.0, restraints=None, alchemy=None,
                 bias=None, efield=None):
        ensemble = ensemble.lower()
        if ensemble not in ("nve", "nvt", "npt"):
            raise ValueError("ensemble must be nve, nvt or npt")
        self.ff, self.rigid, self.nb = ff, rigid, neighbors
        self.dt, self.ensemble = float(dt), ensemble
        self.kT = KB * float(temperature)
        self.gamma_value = float(gamma)
        self.thermostat = None if ensemble == "nve" else make_thermostat(thermostat, gamma, tau_t)
        self.pressure = float(pressure) * BAR
        self.interval = int(barostat_interval)
        self.params = params
        self.shift = space.free()[1]
        self.nmol = rigid.nmol
        self.dof = 6 * rigid.nmol - rigid.dof_correction - (3 if ensemble == "nve" else 0)
        self.restraints = as_restraints(restraints)
        if self.restraints is not None:
            self.restraints.check(rigid.sys.n)
        from ..bias.core import as_bias_set
        self.bias = as_bias_set(bias, colvar=100)   # pgm_jax.bias.BiasSet or None
        if self.bias is not None:
            self.bias.bind(float(temperature))
            self.bias.check(rigid.sys.n)
        self.alchemy = alchemy                     # alchemy.Alchemy or None (then every hook below is inactive)
        if alchemy is not None:
            alchemy.check(ff)
            if ff.iel:
                raise NotImplementedError("extended-Lagrangian dipoles (iel) with an alchemical region")
        self.efield = as_field(efield)             # efield.ExternalField or None (then the step is unchanged)
        if self.efield is not None:
            if alchemy is not None:
                raise NotImplementedError("an external field with an alchemical region (the field would act on the "
                                          "unscaled solute charges)")
            Q = np.bincount(np.asarray(ff.sys.mol), weights=np.asarray(ff._atoms(params)["q"]), minlength=ff.sys.nmol)
            self.field_charged = bool(np.any(np.abs(Q) > 1e-6))
            if ensemble == "npt" and self.field_charged:
                raise NotImplementedError("NPT with an external field and charged molecules: the field energy of the "
                                          "ions is not invariant under the barostat's scaling (md/efield.py); run NVT")
        self.compile()

    def compile(self):
        """(Re)create the jit-compiled entry points (after changing static sizes such as ff.mc)."""
        self.run = jax.jit(self._run)
        self.forces = jax.jit(self._state_forces)

    # --------------------------------------------------------------------- forces
    def field_at(self, st: MDState, step):
        """(E (3,) V/nm at time step * dt, dipole offset) for the force field, or None without a field."""
        if self.efield is None:
            return None
        v = self.efield.value(st.efield, step * self.dt)
        return (v, st.fshift) if self.efield.kind == "E" else (v, st.fshift, "D")

    def _book_field(self, old: MDState, new: MDState) -> MDState:
        """Explicit time dependence of the field: the energy it supplies over the step, the trapezoid
        (dt/2) (dH/dt|_n + dH/dt|_n+1) with dH/dt = -dE/dt . M (efield.py), is booked as heat, so
        that econs stays conserved (the shadow energy of the time-extended velocity Verlet)."""
        if self.efield is None or not self.efield.time_dependent:
            return new
        f = self.efield
        t0, t1 = old.step * self.dt, new.step * self.dt
        w = 0.5 * (t1 - t0) * (f.dHdt(old.efield, t0, old.fdip, volume(old.box))
                               + f.dHdt(new.efield, t1, new.fdip, volume(new.box)))
        return new.set(heat=new.heat + w)

    def _forces(self, body, box, induction, nbr, force_rebuild=False, lam=None, bias=None, field=None):
        pos = self.rigid.positions(body)
        nbr = self.nb.update(nbr, pos, body.center, box, force_rebuild)
        cand, ovf = self.nb.candidates(nbr, body.center, box, pos)
        if self.alchemy is None:
            res = self.ff.compute(pos, box, cand, induction, self.params, keep_geometry=self.keep_geometry,
                                  efield=field)
        else:                                      # Hamiltonian at the state's coupling lam
            res = self.alchemy.compute(self.ff, pos, box, cand, induction, self.params, lam)
        res = self._add_restraints(res._replace(overflow=res.overflow | ovf), pos, box, bias)
        return self.rigid.forces(body, res.forces), res, nbr

    def _has_extra(self, bias) -> bool:
        return self.restraints is not None or (self.bias is not None and bias is not None)

    def _extra_energy(self, pos, box, bias=None):
        """Restraint energy + bias energy (with the bias state `bias`; none if None), kJ/mol."""
        e = jnp.zeros((), jnp.float64)
        if self.restraints is not None:
            e = e + self.restraints.energy(pos, box)
        if self.bias is not None and bias is not None:
            e = e + self.bias.energy(bias, pos, box)
        return e

    def _add_restraints(self, res, pos, box, bias=None):
        """Restraint and bias energy and atomic forces added to a force-field result."""
        if not self._has_extra(bias):
            return res
        e, g = jax.value_and_grad(self._extra_energy)(pos, box, bias)
        return res._replace(energy=dict(res.energy, total=res.energy["total"] + e), forces=res.forces - g)

    def _restraint_energy(self, pos, box, bias=None):
        return self._extra_energy(pos, box, bias) if self._has_extra(bias) else 0.0

    def restraint_strain(self, pos, box, bias=None):
        """d(E_restraint + E_bias) / d eps (3, 3) under molecular scaling (for the pressure)."""
        if not self._has_extra(bias):
            return jnp.zeros((3, 3))
        return molecular_strain(lambda p, h: self._extra_energy(p, h, bias), pos, box, self.ff.mol, self.ff.masses,
                                self.nmol)

    # --------------------------------------------------------------------- biases (pgm_jax.bias)
    def _bias_atoms(self, x):
        """Atom positions of the engine's position variable."""
        return self.rigid.positions(x)

    def _map_atom_forces(self, x, box, F):
        """Atomic forces -> the engine's force variable (linear)."""
        return self.rigid.forces(x, F)

    def _bias_post(self, st: MDState) -> MDState:
        """After a step: the COLVAR row, then the bias updates due at this step.  The forces and
        epot of the state are corrected to the new bias at the same positions (a bias-only
        evaluation), and the energy change is booked as heat and as bias work."""
        if self.bias is None or st.bias is None:
            return st
        x, box = st.dyn.position, st.box
        pos = self._bias_atoms(x)
        st = st.set(bias=self.bias.record(st.bias, pos, box, st.step))
        if not self.bias.dynamic:
            return st

        def dep(st):
            old = st.bias
            new = self.bias.deposit(old, pos, box, st.step)
            e0, g0 = jax.value_and_grad(self.bias.energy, argnums=1)(old, pos, box)
            e1, g1 = jax.value_and_grad(self.bias.energy, argnums=1)(new, pos, box)
            dF = self._map_atom_forces(x, box, g0 - g1)
            de = e1 - e0
            add = lambda a, b: jax.tree_util.tree_map(jnp.add, a, b)          # noqa: E731
            st = st.set(bias=new._replace(work=new.work + de), dyn=st.dyn.set(force=add(st.dyn.force, dF)),
                        epot=st.epot + de, heat=st.heat + de)
            m = getattr(st, "mts", None)
            if m is not None:                          # multiple time stepping: the bias is in the slow group
                st = st.set(mts=m.set(forces=(add(m.forces[0], dF),) + tuple(m.forces[1:])))
            return st

        return jax.lax.cond(self.bias.due(st.step), dep, lambda s: s, st)

    def _with_result(self, st: MDState, F, res, nbr) -> MDState:
        st = st.set(dyn=st.dyn.set(force=F), nbr=nbr, induction=res.induction, epot=res.energy["total"],
                    elec=res.energy["elec"], vdw=res.energy["vdw"], iters=res.iterations,
                    max_iters=jnp.maximum(st.max_iters, res.iterations), resid=jnp.maximum(st.resid, res.residual),
                    overflow=st.overflow | res.overflow, cg_total=st.cg_total + res.iterations)
        return st if res.dipole is None else st.set(fdip=res.dipole)

    def check_block(self, st: MDState) -> None:
        """Host-side check of a finished block of steps, before the driver's overflow handling (a
        no-op here; multiple time stepping grows its short-range pair list, mts.py)."""
        return None

    def _state_forces(self, st: MDState, force_rebuild=True) -> MDState:
        F, res, nbr = self._forces(st.dyn.position, st.box, st.induction, st.nbr, force_rebuild, lam=st.lam,
                                   bias=st.bias, field=self.field_at(st, st.step))
        return self._with_result(st, F, res, nbr)

    # --------------------------------------------------------------------- setup
    def init(self, body, box, key, momentum=None, bias=None) -> MDState:
        """Host-side: allocate the neighbour list, compute forces, draw or set momenta."""
        box = jnp.asarray(box, jnp.float64)
        nbr = self.nb.allocate(self.rigid.positions(body), body.center, box)
        zero = jax.tree_util.tree_map(jnp.zeros_like, body)
        key, split = jax.random.split(key)
        dyn = simulate.canonicalize_mass(Dynamics(body, zero, zero, self.rigid.mass, key))
        if momentum is None:
            dyn = simulate.initialize_momenta(dyn, split, self.kT)
        else:
            dyn = dyn.set(momentum=momentum)
        z = jnp.zeros((), jnp.float64)
        zi = jnp.zeros((), jnp.int32)
        dyn, aux = self._init_aux(dyn)
        st = MDState(dyn=dyn, box=box, induction=self.ff.init_induction(), nbr=nbr, epot=z, elec=z, vdw=z,
                     iters=zi, max_iters=zi, resid=z, step=zi, mc=jnp.zeros(4, jnp.int32),
                     mc_dv=jnp.asarray(0.01 * float(volume(box)), jnp.float64), overflow=jnp.zeros((), bool),
                     aux=aux, heat=z, cg_total=z, bias=self._init_bias(bias))
        return self.forces(field_state(st, self.efield), False)

    def _init_bias(self, bias=None):
        """The bias state of a new MD state: `bias` (e.g. kept across minimisation) or a fresh one."""
        if self.bias is None:
            return None
        return self.bias.init() if bias is None else bias

    # --------------------------------------------------------------------- thermostat
    def _scaled(self, dyn: Dynamics):
        """Mass-scaled momenta of the thermostatted degrees of freedom, a mask (0 for rotations about
        axes with zero moment of inertia), the inverse map and the constraint projection."""
        P, M = dyn.momentum.center, dyn.mass.center                       # M: (nmol, 1)
        q = dyn.position.orientation
        I = dyn.mass.orientation                                         # (nmol, 3) principal moments
        L = rigid_body.conjugate_momentum_to_angular_momentum(q, dyn.momentum.orientation)
        has = I > 0
        v = jnp.stack([P / jnp.sqrt(M), jnp.where(has, L / jnp.sqrt(jnp.where(has, I, 1.0)), 0.0)])
        mask = jnp.stack([jnp.ones_like(P), has.astype(P.dtype)])

        def unpack(v):
            Pq = rigid_body.angular_momentum_to_conjugate_momentum(q, v[1] * jnp.sqrt(I))
            return dyn.set(momentum=RigidBody(v[0] * jnp.sqrt(M), Pq))
        return v, mask, unpack, (lambda u: u)

    def _init_aux(self, dyn: Dynamics):
        v, mask, _, project = self._scaled(dyn)
        if self.thermostat is None or self.thermostat.n_aux == 0:
            return dyn, jnp.zeros((0,) + v.shape, jnp.float64)
        key, k = jax.random.split(dyn.rng)
        aux = jax.vmap(project)(self.thermostat.init_aux(k, v.shape, self.kT))
        if mask is not None:
            aux = aux * mask[None]
        return dyn.set(rng=key), aux

    def thermostat_kT(self, st: MDState):
        """kB T (kJ/mol) the thermostat and barostat use: the state's own (replica exchange) or the
        integrator's."""
        return self.kT if st.kT is None else st.kT

    def _o_step(self, dyn: Dynamics, aux, heat, h: float, kT=None):
        """Thermostat step at fixed positions; returns dyn, auxiliaries and the updated heat."""
        v, mask, unpack, project = self._scaled(dyn)
        key, k = jax.random.split(dyn.rng)
        e0 = 0.5 * (jnp.sum(v * v) + jnp.sum(aux * aux))
        v, aux = self.thermostat.apply(v, aux, k, h, self.kT if kT is None else kT, float(self.dof), project, mask)
        e1 = 0.5 * (jnp.sum(v * v) + jnp.sum(aux * aux))
        return unpack(v).set(rng=key), aux, heat + (e1 - e0)

    # --------------------------------------------------------------------- one step
    def _step(self, st: MDState) -> MDState:
        dt = self.dt
        aux, heat = st.aux, st.heat
        dyn = simulate.momentum_step(st.dyn, dt / 2)
        if self.ensemble == "nve":
            dyn = simulate.position_step(dyn, self.shift, dt)
        else:
            dyn = simulate.position_step(dyn, self.shift, dt / 2)
            dyn, aux, heat = self._o_step(dyn, aux, heat, dt, self.thermostat_kT(st))
            dyn = simulate.position_step(dyn, self.shift, dt / 2)
        F, res, nbr = self._forces(dyn.position, st.box, st.induction, st.nbr, lam=st.lam, bias=st.bias,
                                   field=self.field_at(st, st.step + 1))
        st = self._with_result(st.set(dyn=dyn, aux=aux, heat=heat), F, res, nbr)
        st = st.set(dyn=simulate.momentum_step(st.dyn, dt / 2), step=st.step + 1)
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
        body = st.dyn.position
        body_n = RigidBody(body.center * s, body.orientation)
        Hn = H * s
        pos_n = self.rigid.positions(body_n)
        nbr_n = self.nb.update(st.nbr, pos_n, body_n.center, Hn, True)
        cand, ovf0 = self.nb.candidates(nbr_n, body_n.center, Hn, pos_n)
        field = self.field_at(st, st.step)
        if self.alchemy is None:
            e_n, ind_n, _, ovf = self.ff.energy(pos_n, Hn, cand, st.induction, self.params, efield=field)
        else:
            e_n, ind_n, _, ovf = self.alchemy.energy(self.ff, pos_n, Hn, cand, st.induction, self.params, st.lam)
        e_n = e_n + self._restraint_energy(pos_n, Hn, st.bias)
        ovf = ovf | ovf0
        kT = self.thermostat_kT(st)
        e_0 = st.epot
        if self.ff.shadow:                         # iEL/0-SCF: converged energies at both volumes
            pos0 = self.rigid.positions(body)
            e_0 = self.ff.energy(pos0, H, self.nb.candidates(st.nbr, body.center, H, pos0)[0], st.induction,
                                 self.params, efield=field)[0] + self._restraint_energy(pos0, H, st.bias)
        w = (e_n - e_0) + self.pressure * dV - self.nmol * kT * jnp.log(jnp.maximum(Vn, 1e-12) / V)
        accept = (Vn > 0) & (jnp.log(jax.random.uniform(k2, dtype=jnp.float64)) < -w / kT)
        st = st.set(dyn=st.dyn.set(rng=key), overflow=st.overflow | ovf)

        def acc(st):
            st = st.set(dyn=st.dyn.set(position=body_n), box=Hn, induction=ind_n)
            F, res, nbr = self._forces(body_n, Hn, ind_n, nbr_n, lam=st.lam, bias=st.bias, field=field)
            return self._with_result(st, F, res, nbr)

        st = jax.lax.cond(accept, acc, lambda s: s, st)
        mc = st.mc + jnp.array([1, 0, 1, 0], jnp.int32) + accept.astype(jnp.int32) * jnp.array([0, 1, 0, 1], jnp.int32)
        adapt = mc[2] >= 10
        rate = mc[3] / jnp.maximum(mc[2], 1)
        dv = jnp.where(adapt & (rate < 0.25), st.mc_dv / 1.1, jnp.where(adapt & (rate > 0.75), st.mc_dv * 1.1, st.mc_dv))
        dv = jnp.minimum(dv, 0.3 * volume(st.box))
        mc = jnp.where(adapt, mc.at[2].set(0).at[3].set(0), mc)
        return st.set(mc=mc, mc_dv=dv)

    def _run(self, st: MDState, n) -> MDState:
        st = st.set(max_iters=jnp.zeros((), jnp.int32), resid=jnp.zeros((), jnp.float64), overflow=jnp.zeros((), bool))
        step = lambda s: self._book_field(s, self._step(s))                   # noqa: E731
        if self.bias is None or self.bias.stride == 0:
            return jax.lax.fori_loop(0, n, lambda _, s: step(s), st)
        return strided_loop(st, n, step, self._bias_post, self.bias.stride)

    # --------------------------------------------------------------------- observables
    def kinetic(self, st: MDState):
        """(total, translational) kinetic energy, kJ/mol."""
        ke = simulate.kinetic_energy(st.dyn)
        p, m = st.dyn.momentum.center, st.dyn.mass.center
        return ke, 0.5 * jnp.sum(p * p / m)

    def temperature(self, st: MDState):
        return 2.0 * self.kinetic(st)[0] / (self.dof * KB)

    def temperatures(self, st: MDState):
        """(translational, rotational) temperatures, K (equipartition check)."""
        ke, ke_t = self.kinetic(st)
        n_t = 3 * self.nmol - (3 if self.ensemble == "nve" else 0)
        n_r = 3 * self.nmol - self.rigid.dof_correction
        return 2.0 * ke_t / (n_t * KB), 2.0 * (ke - ke_t) / (n_r * KB)
