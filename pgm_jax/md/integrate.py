"""Integrators for rigid-molecule pGM MD built from JAX-MD's simulation primitives.

  NVE : velocity Verlet with the NO_SQUISH free-rotor splitting for quaternions (JAX-MD
        simulate.momentum_step / position_step, rigid_body registrations; Miller et al. 2002).
  NVT : BAOAB Langevin, friction `gamma` (1/ps).  The O step is an exact Ornstein-Uhlenbeck update
        of the centre-of-mass momenta and of the body-frame angular momenta,
        L_l <- c L_l + sqrt(I_l kT (1 - c^2)) xi_l, c = exp(-gamma dt), mapped to and from the
        quaternion conjugate momenta.  (JAX-MD 0.2.29's rigid-body stochastic_step draws the
        quaternion-momentum noise with a diagonal covariance instead of sum_l s_l^2 P_l P_l^T, which
        under-heats the rotations: rigid water settled 15-25 K below the target.)
  NPT : NVT plus an isotropic Monte Carlo barostat every `barostat_interval` steps (Amber
        barostat = 2, OpenMM MonteCarloBarostat): centres of mass and box are scaled,
        orientations and momenta are kept, acceptance on dE + P dV - N kT ln(V'/V); the
        maximum volume change adapts to 25-75 % acceptance.

Units: nm, ps, amu, kJ/mol, K."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._jaxmd import dataclasses, rigid_body, simulate, space
from .box import volume
from .forcefield import InductionState, PGMForceField
from .rigid import RigidBody, RigidMolecules

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


class Integrator:
    def __init__(self, ff: PGMForceField, rigid: RigidMolecules, neighbors, dt: float = 0.001,
                 ensemble: str = "nvt", temperature: float = 298.0, gamma: float = 1.0,
                 pressure: float = 1.0, barostat_interval: int = 100, params=None):
        ensemble = ensemble.lower()
        if ensemble not in ("nve", "nvt", "npt"):
            raise ValueError("ensemble must be nve, nvt or npt")
        self.ff, self.rigid, self.nb = ff, rigid, neighbors
        self.dt, self.ensemble = float(dt), ensemble
        self.kT = KB * float(temperature)
        self.gamma_value = float(gamma)
        self.pressure = float(pressure) * BAR
        self.interval = int(barostat_interval)
        self.params = params
        self.shift = space.free()[1]
        self.nmol = rigid.nmol
        self.dof = 6 * rigid.nmol - rigid.dof_correction - (3 if ensemble == "nve" else 0)
        self.compile()

    def compile(self):
        """(Re)create the jit-compiled entry points (after changing static sizes such as ff.mc)."""
        self.run = jax.jit(self._run)
        self.forces = jax.jit(self._state_forces)

    # --------------------------------------------------------------------- forces
    def _forces(self, body, box, induction, nbr, force_rebuild=False):
        pos = self.rigid.positions(body)
        nbr = self.nb.update(nbr, pos, body.center, box, force_rebuild)
        cand, ovf = self.nb.candidates(nbr, body.center, box, pos)
        res = self.ff.compute(pos, box, cand, induction, self.params)
        res = res._replace(overflow=res.overflow | ovf)
        return self.rigid.forces(body, res.forces), res, nbr

    def _with_result(self, st: MDState, F, res, nbr) -> MDState:
        return st.set(dyn=st.dyn.set(force=F), nbr=nbr, induction=res.induction, epot=res.energy["total"],
                      elec=res.energy["elec"], vdw=res.energy["vdw"], iters=res.iterations,
                      max_iters=jnp.maximum(st.max_iters, res.iterations), resid=jnp.maximum(st.resid, res.residual),
                      overflow=st.overflow | res.overflow)

    def _state_forces(self, st: MDState, force_rebuild=True) -> MDState:
        F, res, nbr = self._forces(st.dyn.position, st.box, st.induction, st.nbr, force_rebuild)
        return self._with_result(st, F, res, nbr)

    # --------------------------------------------------------------------- setup
    def init(self, body, box, key, momentum=None) -> MDState:
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
        st = MDState(dyn=dyn, box=box, induction=self.ff.init_induction(), nbr=nbr, epot=z, elec=z, vdw=z,
                     iters=zi, max_iters=zi, resid=z, step=zi, mc=jnp.zeros(4, jnp.int32),
                     mc_dv=jnp.asarray(0.01 * float(volume(box)), jnp.float64), overflow=jnp.zeros((), bool))
        return self.forces(st, False)

    # --------------------------------------------------------------------- one step
    def _step(self, st: MDState) -> MDState:
        dt = self.dt
        dyn = simulate.momentum_step(st.dyn, dt / 2)
        if self.ensemble == "nve":
            dyn = simulate.position_step(dyn, self.shift, dt)
        else:
            dyn = simulate.position_step(dyn, self.shift, dt / 2)
            dyn = self._ou_step(dyn, dt)
            dyn = simulate.position_step(dyn, self.shift, dt / 2)
        F, res, nbr = self._forces(dyn.position, st.box, st.induction, st.nbr)
        st = self._with_result(st.set(dyn=dyn), F, res, nbr)
        st = st.set(dyn=simulate.momentum_step(st.dyn, dt / 2), step=st.step + 1)
        if self.ensemble == "npt":
            st = jax.lax.cond(st.step % self.interval == 0, self._barostat, lambda s: s, st)
        return st

    def _ou_step(self, dyn: Dynamics, dt: float) -> Dynamics:
        """Exact Ornstein-Uhlenbeck step for centre-of-mass and body-frame angular momenta."""
        key, k1, k2 = jax.random.split(dyn.rng, 3)
        c = jnp.exp(-self.gamma_value * dt)
        s = jnp.sqrt(self.kT * (1.0 - c * c))
        P, M = dyn.momentum.center, dyn.mass.center                       # M: (nmol, 1)
        P = c * P + s * jnp.sqrt(M) * jax.random.normal(k1, P.shape, P.dtype)
        q = dyn.position.orientation
        I = dyn.mass.orientation                                         # (nmol, 3) principal moments
        L = rigid_body.conjugate_momentum_to_angular_momentum(q, dyn.momentum.orientation)
        L = c * L + s * jnp.sqrt(I) * jax.random.normal(k2, L.shape, L.dtype)
        Pq = rigid_body.angular_momentum_to_conjugate_momentum(q, L)
        return dyn.set(momentum=RigidBody(P, Pq), rng=key)

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
        e_n, ind_n, _, ovf = self.ff.energy(pos_n, Hn, cand, st.induction, self.params)
        ovf = ovf | ovf0
        w = (e_n - st.epot) + self.pressure * dV - self.nmol * self.kT * jnp.log(jnp.maximum(Vn, 1e-12) / V)
        accept = (Vn > 0) & (jnp.log(jax.random.uniform(k2, dtype=jnp.float64)) < -w / self.kT)
        st = st.set(dyn=st.dyn.set(rng=key), overflow=st.overflow | ovf)

        def acc(st):
            st = st.set(dyn=st.dyn.set(position=body_n), box=Hn, induction=ind_n)
            F, res, nbr = self._forces(body_n, Hn, ind_n, nbr_n)
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
        return jax.lax.fori_loop(0, n, lambda _, s: self._step(s), st)

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
