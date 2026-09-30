"""Integrators for rigid-molecule pGM MD built from JAX-MD's simulation primitives.

Contents: the state pytrees `Dynamics` and `MDState`, the rigid-body `Integrator` (also the base
of flexible.FlexibleIntegrator and the multiple-time-step integrators of md/mts.py), and helpers
for states (`upgrade_state`, `field_state`) and compiled loops (`strided_loop`).

  NVE : velocity Verlet with the NO_SQUISH free-rotor splitting for quaternions (JAX-MD
        simulate.momentum_step / position_step, rigid_body registrations [1]_).
  NVT : BAOAB with the O step of a thermostat from thermostats.py (Langevin(friction),
        Bussi(tau) rescaling, or a smooth GLE).  For rigid bodies the thermostat acts on the
        mass-scaled centre-of-mass momenta and body-frame angular momenta, L_l / sqrt(I_l), mapped
        to and from the quaternion conjugate momenta.  (JAX-MD 0.2.29's rigid-body stochastic_step
        draws the quaternion-momentum noise with a diagonal covariance instead of
        sum_l s_l^2 P_l P_l^T, which under-heats the rotations: rigid water settled 15-25 K below
        the target.)  The heat exchanged in the O steps is booked in MDState.heat, so
        E_tot + |aux|^2/2 - heat is conserved up to integration and induction errors.
  NPT : NVT plus an isotropic Monte Carlo barostat every `barostat.every` steps (Amber
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

One step (BAOAB; NVE: velocity Verlet): half kick, half drift, O step of length dt (thermostat),
half drift, forces at the new positions, half kick, then (NPT) the barostat move every
`interval` steps (lax.cond).  Blocks of n steps are one jitted lax.fori_loop (`run`).

Units: nm, ps, amu, kJ/mol, K.

References
----------
.. [1] T. F. Miller III, M. Eleftheriou, P. Pattnaik, A. Ndirango, D. Newns, G. J. Martyna,
   J. Chem. Phys. 116, 8649 (2002).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..units import KB, KJMOL_NM3_PER_BAR
from ._jaxmd import dataclasses, rigid_body, simulate, space
from .barostats import MonteCarloBarostat, ensemble_name
from .box import volume
from .efield import as_field
from .forcefield import InductionState, PGMForceField
from .restraints import as_restraints, molecular_strain
from .rigid import RigidBody, RigidMolecules
from .thermostats import Thermostat, make_thermostat

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from .alchemy import Alchemy
    from .efield import ExternalField
    from .forcefield import Result
    from .restraints import Restraint, Restraints


@dataclasses.dataclass
class Dynamics:
    """The part of the state JAX-MD's primitives act on (a JAX-MD dataclass, i.e. a pytree).

    Parameters
    ----------
    position : RigidBody
        center (M, 3) [nm], orientation Quaternion (M, 4); flexible engine: atom positions (N, 3).
    momentum : RigidBody
        center (M, 3) [amu nm/ps], orientation: quaternion conjugate momenta (M, 4).
    force : RigidBody
        Centre forces (M, 3) [kJ/mol/nm] and quaternion forces (M, 4).
    mass : RigidBody
        center (M, 1) [amu] (after simulate.canonicalize_mass), orientation: principal moments
        (M, 3) [amu nm^2].
    rng : jax.Array
        PRNG key of the thermostat and barostat.
    """

    position: RigidBody
    momentum: RigidBody
    force: RigidBody
    mass: RigidBody
    rng: jnp.ndarray


@dataclasses.dataclass
class MDState:
    """State of one MD trajectory (a JAX-MD dataclass, i.e. a pytree; every field a leaf or subtree).

    Optional fields are None when their option is off; None changes the tree structure, so
    switching an option recompiles the step.  Scalars are float64 unless noted.

    Parameters
    ----------
    dyn : Dynamics
        Positions, momenta, forces, masses, random key.
    box : jax.Array (3, 3)
        H, lattice vectors as rows [nm].
    induction : InductionState
        Induced dipoles and predictor history (md/forcefield.py).
    nbr : partition.NeighborList
        JAX-MD dense neighbour list (None in checkpoints).
    epot : jax.Array ()
        Potential energy [kJ/mol] (restraints, biases and field included).
    elec, vdw : jax.Array ()
        Electrostatic and van der Waals parts [kJ/mol].
    iters : jax.Array () int32
        CG iterations of the last solve.
    max_iters : jax.Array () int32
        Largest CG iteration count since the last block start.
    resid : jax.Array ()
        Largest final CG residual since the last block start.
    step : jax.Array () int32
        Step counter.
    mc : jax.Array (4,) int32
        Barostat (tries, accepts, window tries, window accepts).
    mc_dv : jax.Array ()
        Current maximum volume change [nm^3].
    overflow : jax.Array () bool
        A force evaluation exceeded the row capacity (the block must be repeated).
    aux : jax.Array
        Thermostat auxiliary momenta (mass-scaled), (n_aux,) + the shape of the scaled momenta
        [sqrt(kJ/mol)].
    heat : jax.Array ()
        Heat taken up in the thermostat steps (and from time-dependent fields and bias updates)
        since the start [kJ/mol].
    cg_total : jax.Array ()
        CG iterations summed over all force evaluations.
    kT : jax.Array (), optional
        Thermostat kB T [kJ/mol] as a state variable (replica exchange: one compiled step for every
        temperature); None: Integrator.kT.
    lam : jax.Array (2,), optional
        Alchemical coupling (lambda_elec, lambda_vdw) (lambda windows share one compiled step);
        None: the Alchemy's default.
    mts : MTSState, optional
        Multiple time stepping: forces of each level, short-range list (md/mts.py).
    bias : BiasState, optional
        State of the biases (pgm_jax.bias).
    efield : jax.Array (3,), optional
        Amplitude of the external field [V/nm] (md/efield.py); None: no field.
    fshift : jax.Array (3,), optional
        Dipole of the re-wrapped charged molecules (itinerant charges) [e nm].
    fdip : jax.Array (3,), optional
        Cell dipole M of the last force evaluation (the dipole the field acts on) [e nm].
    """

    dyn: Dynamics
    box: jnp.ndarray  # H, lattice vectors as rows (nm)
    induction: InductionState
    nbr: object  # jax_md.partition.NeighborList (dense)
    epot: jnp.ndarray  # kJ/mol (float64)
    elec: jnp.ndarray
    vdw: jnp.ndarray
    iters: jnp.ndarray  # CG iterations of the last solve
    max_iters: jnp.ndarray  # largest since the last report
    resid: jnp.ndarray  # largest final residual since the last report
    step: jnp.ndarray
    mc: jnp.ndarray  # (tries, accepts, window tries, window accepts) int32
    mc_dv: jnp.ndarray  # current maximum volume change (nm^3)
    overflow: jnp.ndarray  # a force evaluation exceeded the row capacity (block must be repeated)
    aux: jnp.ndarray = None  # thermostat auxiliary momenta (mass-scaled), (n_aux,) + momenta shape
    heat: jnp.ndarray = None  # heat taken up in the thermostat steps since the start (kJ/mol)
    cg_total: jnp.ndarray = None  # CG iterations summed over all force evaluations (float64)
    kT: jnp.ndarray = None  # thermostat kB T (kJ/mol) as a state variable (replica exchange, remd.py:
    # one compiled step for every temperature); None: Integrator.kT
    lam: jnp.ndarray = None  # (2,) alchemical coupling (lambda_elec, lambda_vdw) of the state (alchemy.py:
    # lambda windows share one compiled step); None: the Alchemy's default
    mts: object = None  # multiple time stepping: forces of each level, short-range list (mts.MTSState)
    bias: object = None  # state of the biases (pgm_jax.bias.BiasState) or None
    efield: jnp.ndarray = None  # (3,) V/nm amplitude of the external field (efield.py); None: no field
    fshift: jnp.ndarray = None  # (3,) e nm dipole of the re-wrapped charged molecules (itinerant charges)
    fdip: jnp.ndarray = None  # (3,) e nm M of the last force evaluation (the dipole the field acts on)


def upgrade_state(st: MDState, aux: jax.Array) -> MDState:
    """Return a state pickled before the thermostat fields existed with them added.

    aux (the current auxiliaries' template), zero heat and zero CG total; states that have them
    are returned unchanged.
    """
    if getattr(st, "aux", None) is None:
        st = st.set(aux=aux)
    if getattr(st, "heat", None) is None:
        st = st.set(heat=jnp.zeros((), jnp.float64))
    if getattr(st, "cg_total", None) is None:
        st = st.set(cg_total=jnp.zeros((), jnp.float64))
    return st


def strided_loop(
    st: MDState, n: int | jax.Array, step: Callable[[MDState], MDState], post: Callable[[MDState], MDState], stride: int
) -> MDState:
    """Advance n steps with post(state) after every step whose counter is a multiple of `stride`.

    Written as plain inner loops of `stride` steps (a head up to the next multiple, full chunks,
    a tail), so the steps in between carry no conditional (on a GPU each lax.cond reads its
    predicate on the host: two per step cost ~0.1 ms).  Traceable (lax.fori_loop); n may be
    traced.

    Parameters
    ----------
    st : MDState
        Start state (its step counter may be batched; the first entry is used).
    n : int or jax.Array ()
        Steps.
    step : Callable[[MDState], MDState]
        One step.
    post : Callable[[MDState], MDState]
        Called after every step whose counter is a multiple of `stride`.
    stride : int
        Interval of `post` [steps] (static).

    Returns
    -------
    MDState
    """
    s0 = st.step if jnp.ndim(st.step) == 0 else st.step.reshape(-1)[0]
    head = (stride - s0 % stride) % stride
    h = jnp.minimum(head, n)
    st = jax.lax.fori_loop(0, h, lambda _, s: step(s), st)
    st = jax.lax.cond((h == head) & (head > 0), post, lambda s: s, st)
    rem = n - h
    nch = rem // stride

    def inner(s: MDState) -> MDState:  # `stride` steps without a conditional
        return jax.lax.fori_loop(0, stride, lambda _, t: step(t), s)

    st = jax.lax.fori_loop(0, nch, lambda _, s: post(inner(s)), st)
    return jax.lax.fori_loop(0, rem - nch * stride, lambda _, s: step(s), st)


def field_state(st: MDState, field: ExternalField | None) -> MDState:
    """Return the state with its external-field fields set for an integrator with `field`.

    With a field: the amplitude (kept if the state has one, else field.E0), offset and dipole
    (zeros if missing); without one (None): all three None.
    """
    if field is None:
        return st.set(efield=None, fshift=None, fdip=None)
    z = jnp.zeros(3, jnp.float64)
    E = getattr(st, "efield", None)
    return st.set(
        efield=jnp.asarray(field.E0, jnp.float64) if E is None else jnp.asarray(E, jnp.float64),
        fshift=z if getattr(st, "fshift", None) is None else st.fshift,
        fdip=z if getattr(st, "fdip", None) is None else st.fdip,
    )


class Integrator:
    """Rigid-body integrator: NVE velocity Verlet, NVT BAOAB, NPT with the Monte Carlo barostat.

    See the module docstring for the scheme and the options.  `run(state, n)` and
    `forces(state, force_rebuild)` are jitted entry points, re-created by `compile` whenever a
    static size (row capacities, list capacity, restraints) changes.  The settings (dt, kT,
    thermostat, barostat, restraints, ...) are baked into the compiled step.

        integ = Integrator(ff, rigid, nb, dt=0.002, temperature=298.0, thermostat="bussi")
        st = integ.init(rigid.body0, H, jax.random.PRNGKey(0))
        st = integ.run(st, 1000)

    Attributes
    ----------
    ff : PGMForceField
        Force field.
    rigid : RigidMolecules
        Rigid bodies (flexible.FlexibleMolecules for the subclass).
    nb : AtomNeighbors or MoleculeNeighbors
        Neighbour-list object.
    thermostat : Thermostat or None
        Thermostat.
    barostat : MonteCarloBarostat or None
        Barostat.
    ensemble : {"nve", "nvt", "npt"}
        Ensemble.
    dt : float
        Time step [ps].
    kT : float
        kB T of the bath [kJ/mol].
    pressure : float
        Barostat target [kJ/mol/nm^3] (1 bar without a barostat).
    interval : int
        Steps between barostat moves.
    params : dict or None
        Force-field parameters (None: the system's).
    nmol : int
        Number of molecules.
    dof : int
        Degrees of freedom 6 M - dof_correction (- 3 in NVE: the total momentum is conserved).
    restraints : Restraints or None
        Restraints.
    bias : BiasSet or None
        Biases.
    alchemy : Alchemy or None
        Alchemical region.
    efield : ExternalField or None
        External field.
    field_charged : bool
        Some molecule is charged (with a field only).
    keep_geometry : bool
        Ask the force field for its row geometry (multiple time stepping; class attribute).
    """

    keep_geometry = False  # ask the force field for its row geometry (multiple time stepping, mts.py)

    def __init__(
        self,
        ff: PGMForceField,
        rigid: RigidMolecules,
        neighbors: Any,
        dt: float = 0.001,
        temperature: float = 298.0,
        thermostat: str | Thermostat | None = "langevin",
        barostat: MonteCarloBarostat | None = None,
        params: dict | None = None,
        restraints: Restraints | Restraint | list | None = None,
        alchemy: Alchemy | None = None,
        bias: Any = None,
        efield: ExternalField | ArrayLike | None = None,
    ) -> None:
        """Build the integrator and compile its entry points.

        Parameters
        ----------
        ff : PGMForceField
            The force field.
        rigid : RigidMolecules
            The rigid bodies (FlexibleMolecules for the subclass).
        neighbors : MoleculeNeighbors or AtomNeighbors
            The neighbour-list object.
        dt : float
            Time step [ps].
        temperature : float
            Bath temperature [K] (also the temperature of drawn momenta).
        thermostat : Thermostat, str or None
            The thermostat (thermostats.make_thermostat: an object, a name for the default
            settings of a kind, or None for NVE).
        barostat : MonteCarloBarostat or None
            The barostat (None: constant volume); needs a thermostat.
        params : dict, optional
            Force-field parameters (default: those of the system).
        restraints, alchemy, bias, efield : optional
            Restraints (restraints.py), an alchemical region (alchemy.py), biases (pgm_jax.bias)
            and an external field (efield.py); see the module docstring.

        Raises
        ------
        ValueError
            A barostat without a thermostat, invalid restraints or biases.
        NotImplementedError
            Unsupported combinations (iEL or a field with an alchemical region, NPT with a field
            and charged molecules).
        """
        self.ff, self.rigid, self.nb = ff, rigid, neighbors
        self.thermostat = make_thermostat(thermostat)
        self.barostat = barostat
        ensemble = ensemble_name(self.thermostat, barostat)
        self.dt, self.ensemble = float(dt), ensemble
        self.kT = KB * float(temperature)
        # the barostat's target in kJ/mol/nm^3 and its interval (defaults when there is none)
        self.pressure = float(barostat.pressure if barostat is not None else 1.0) * KJMOL_NM3_PER_BAR
        self.interval = int(barostat.every if barostat is not None else 100)
        self.params = params
        self.shift = space.free()[1]
        self.nmol = rigid.nmol
        self.dof = 6 * rigid.nmol - rigid.dof_correction - (3 if ensemble == "nve" else 0)
        self.restraints = as_restraints(restraints)
        if self.restraints is not None:
            self.restraints.check(rigid.sys.n)
        from ..bias.core import as_bias_set

        self.bias = as_bias_set(bias, colvar=100)  # pgm_jax.bias.BiasSet or None
        if self.bias is not None:
            self.bias.bind(float(temperature))
            self.bias.check(rigid.sys.n)
        self.alchemy = alchemy  # alchemy.Alchemy or None (then every hook below is inactive)
        if alchemy is not None:
            alchemy.check(ff)
            if ff.iel:
                raise NotImplementedError("extended-Lagrangian dipoles (iel) with an alchemical region")
        self.efield = as_field(efield)  # efield.ExternalField or None (then the step is unchanged)
        if self.efield is not None:
            if alchemy is not None:
                raise NotImplementedError(
                    "an external field with an alchemical region (the field would act on the unscaled solute charges)"
                )
            Q = np.bincount(np.asarray(ff.sys.mol), weights=np.asarray(ff._atoms(params)["q"]), minlength=ff.sys.nmol)
            self.field_charged = bool(np.any(np.abs(Q) > 1e-6))
            if ensemble == "npt" and self.field_charged:
                raise NotImplementedError(
                    "NPT with an external field and charged molecules: the field energy of the "
                    "ions is not invariant under the barostat's scaling (md/efield.py); run NVT"
                )
        self.compile()

    def compile(self) -> None:
        """(Re)create the jit-compiled entry points (after changing static sizes such as ff.mc)."""
        self.run = jax.jit(self._run)
        self.forces = jax.jit(self._state_forces)

    # --------------------------------------------------------------------- forces
    def field_at(self, st: MDState, step: int | jax.Array) -> tuple | None:
        """Return the field argument of the force field at time step * dt, or None without a field.

        (E (3,) [V/nm], dipole offset fshift [e nm]) for a constant-E field, with a third entry
        "D" for a constant displacement.
        """
        if self.efield is None:
            return None
        v = self.efield.value(st.efield, step * self.dt)
        return (v, st.fshift) if self.efield.kind == "E" else (v, st.fshift, "D")

    def _book_field(self, old: MDState, new: MDState) -> MDState:
        """Return `new` with the energy a time-dependent field supplied over the step booked as heat.

        The trapezoid (dt/2) (dH/dt|_n + dH/dt|_n+1) with dH/dt = -dE/dt . M (efield.py), so that
        econs stays conserved (the shadow energy of the time-extended velocity Verlet).  No change
        without a time-dependent field.
        """
        if self.efield is None or not self.efield.time_dependent:
            return new
        f = self.efield
        t0, t1 = old.step * self.dt, new.step * self.dt
        w = (
            0.5
            * (t1 - t0)
            * (f.dHdt(old.efield, t0, old.fdip, volume(old.box)) + f.dHdt(new.efield, t1, new.fdip, volume(new.box)))
        )
        return new.set(heat=new.heat + w)

    def _forces(
        self,
        body: RigidBody,
        box: jax.Array,
        induction: InductionState,
        nbr: Any,
        force_rebuild: bool | jax.Array = False,
        lam: jax.Array | None = None,
        bias: Any = None,
        field: tuple | None = None,
    ) -> tuple[RigidBody, Result, Any]:
        """Return the body forces, the force-field result and the updated list at one configuration.

        Updates the neighbour list, evaluates the force field (or the alchemical Hamiltonian at
        lam) with the induction solve, adds restraints and biases, and maps the atomic forces to
        the bodies.

        Parameters
        ----------
        body : RigidBody
            Body positions.
        box : jax.Array (3, 3)
            Box [nm].
        induction : InductionState
            Induction state of the previous evaluation.
        nbr : partition.NeighborList
            Neighbour list.
        force_rebuild : bool or jax.Array () bool
            Rebuild the list unconditionally.
        lam : jax.Array (2,), optional
            Alchemical coupling.
        bias : BiasState, optional
            Bias state.
        field : tuple, optional
            `field_at` output.

        Returns
        -------
        forces : RigidBody
            Centre forces [kJ/mol/nm] and quaternion forces.
        result : forcefield.Result
            Energies [kJ/mol], atomic forces, induction, CG statistics, overflow (list or rows).
        nbr : partition.NeighborList
            The updated list.
        """
        pos = self.rigid.positions(body)
        nbr = self.nb.update(nbr, pos, body.center, box, force_rebuild)
        cand, ovf = self.nb.candidates(nbr, body.center, box, pos)
        if self.alchemy is None:
            res = self.ff.compute(
                pos, box, cand, induction, self.params, keep_geometry=self.keep_geometry, efield=field
            )
        else:  # Hamiltonian at the state's coupling lam
            res = self.alchemy.compute(self.ff, pos, box, cand, induction, self.params, lam)
        res = self._add_restraints(res._replace(overflow=res.overflow | ovf), pos, box, bias)
        return self.rigid.forces(body, res.forces), res, nbr

    def _has_extra(self, bias: Any) -> bool:
        """Return whether restraints or (with a bias state) biases add to the energy."""
        return self.restraints is not None or (self.bias is not None and bias is not None)

    def _extra_energy(self, pos: jax.Array, box: jax.Array, bias: Any = None) -> jax.Array:
        """Return the restraint energy + bias energy [kJ/mol] (bias state `bias`; no bias if None)."""
        e = jnp.zeros((), jnp.float64)
        if self.restraints is not None:
            e = e + self.restraints.energy(pos, box)
        if self.bias is not None and bias is not None:
            e = e + self.bias.energy(bias, pos, box)
        return e

    def _add_restraints(self, res: Result, pos: jax.Array, box: jax.Array, bias: Any = None) -> Result:
        """Return a force-field result with the restraint and bias energy and atomic forces added."""
        if not self._has_extra(bias):
            return res
        e, g = jax.value_and_grad(self._extra_energy)(pos, box, bias)
        return res._replace(energy=dict(res.energy, total=res.energy["total"] + e), forces=res.forces - g)

    def _restraint_energy(self, pos: jax.Array, box: jax.Array, bias: Any = None) -> jax.Array | float:
        """Return `_extra_energy`, or 0.0 when there are neither restraints nor biases."""
        return self._extra_energy(pos, box, bias) if self._has_extra(bias) else 0.0

    def restraint_strain(self, pos: jax.Array, box: jax.Array, bias: Any = None) -> jax.Array:
        """Return d(E_restraint + E_bias) / d eps (3, 3) [kJ/mol] under molecular scaling (pressure)."""
        if not self._has_extra(bias):
            return jnp.zeros((3, 3))
        return molecular_strain(
            lambda p, h: self._extra_energy(p, h, bias), pos, box, self.ff.mol, self.ff.masses, self.nmol
        )

    # --------------------------------------------------------------------- biases (pgm_jax.bias)
    def _bias_atoms(self, x: Any) -> jax.Array:
        """Return the atom positions (N, 3) [nm] of the engine's position variable."""
        return self.rigid.positions(x)

    def _map_atom_forces(self, x: Any, box: jax.Array, F: jax.Array) -> Any:
        """Return atomic forces F (N, 3) mapped to the engine's force variable (linear map)."""
        return self.rigid.forces(x, F)

    def _bias_post(self, st: MDState) -> MDState:
        """Record the COLVAR row after a step, then apply the bias updates due at this step.

        The forces and epot of the state are corrected to the new bias at the same positions (a
        bias-only evaluation), and the energy change is booked as heat and as bias work.  The
        deposit is a lax.cond on BiasSet.due.
        """
        if self.bias is None or st.bias is None:
            return st
        x, box = st.dyn.position, st.box
        pos = self._bias_atoms(x)
        st = st.set(bias=self.bias.record(st.bias, pos, box, st.step))
        if not self.bias.dynamic:
            return st

        def dep(st: MDState) -> MDState:
            """Deposit, and correct forces, epot, heat and work to the new bias."""
            old = st.bias
            new = self.bias.deposit(old, pos, box, st.step)
            e0, g0 = jax.value_and_grad(self.bias.energy, argnums=1)(old, pos, box)
            e1, g1 = jax.value_and_grad(self.bias.energy, argnums=1)(new, pos, box)
            dF = self._map_atom_forces(x, box, g0 - g1)
            de = e1 - e0

            def add(a: Any, b: Any) -> Any:  # pytree sum (RigidBody forces)
                return jax.tree_util.tree_map(jnp.add, a, b)

            st = st.set(
                bias=new._replace(work=new.work + de),
                dyn=st.dyn.set(force=add(st.dyn.force, dF)),
                epot=st.epot + de,
                heat=st.heat + de,
            )
            m = getattr(st, "mts", None)
            if m is not None:  # multiple time stepping: the bias is in the slow group
                st = st.set(mts=m.set(forces=(add(m.forces[0], dF),) + tuple(m.forces[1:])))
            return st

        return jax.lax.cond(self.bias.due(st.step), dep, lambda s: s, st)

    def _with_result(self, st: MDState, F: Any, res: Result, nbr: Any) -> MDState:
        """Return the state with the forces F, the list and the result's energies and statistics."""
        st = st.set(
            dyn=st.dyn.set(force=F),
            nbr=nbr,
            induction=res.induction,
            epot=res.energy["total"],
            elec=res.energy["elec"],
            vdw=res.energy["vdw"],
            iters=res.iterations,
            max_iters=jnp.maximum(st.max_iters, res.iterations),
            resid=jnp.maximum(st.resid, res.residual),
            overflow=st.overflow | res.overflow,
            cg_total=st.cg_total + res.iterations,
        )
        return st if res.dipole is None else st.set(fdip=res.dipole)

    def check_block(self, st: MDState) -> None:
        """Check a finished block of steps on the host, before the driver's overflow handling.

        A no-op here; multiple time stepping grows its short-range pair list (mts.py).
        """
        return None

    def _state_forces(self, st: MDState, force_rebuild: bool = True) -> MDState:
        """Return the state with forces, energies and dipoles evaluated at its positions (jitted as `forces`)."""
        F, res, nbr = self._forces(
            st.dyn.position,
            st.box,
            st.induction,
            st.nbr,
            force_rebuild,
            lam=st.lam,
            bias=st.bias,
            field=self.field_at(st, st.step),
        )
        return self._with_result(st, F, res, nbr)

    # --------------------------------------------------------------------- setup
    def init(
        self, body: RigidBody, box: ArrayLike, key: jax.Array, momentum: RigidBody | None = None, bias: Any = None
    ) -> MDState:
        """Return a new state: allocate the neighbour list, draw or set momenta, compute forces (host).

        Parameters
        ----------
        body : RigidBody
            Body positions (flexible engine: atom positions (N, 3) [nm]).
        box : ArrayLike (3, 3)
            Box [nm].
        key : jax.Array
            PRNG key (momenta, then the state's stream).
        momentum : RigidBody, optional
            Momenta (None: drawn at kT).
        bias : BiasState, optional
            Bias state to keep (None: a fresh one).

        Returns
        -------
        MDState
            Step 0, thermostat auxiliaries drawn, the barostat step size 1 % of the volume.
        """
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

    def _init_bias(self, bias: Any = None) -> Any:
        """Return the bias state of a new MD state: `bias` (e.g. kept across minimisation) or a fresh one."""
        if self.bias is None:
            return None
        return self.bias.init() if bias is None else bias

    # --------------------------------------------------------------------- thermostat
    def _scaled(self, dyn: Dynamics) -> tuple[jax.Array, jax.Array | None, Callable, Callable]:
        """Return the mass-scaled momenta, their mask, the inverse map and the constraint projection.

        Rigid bodies: v = (P / sqrt(M), L / sqrt(I)) stacked (2, M, 3), L the body-frame angular
        momentum from the quaternion momenta; the mask is 0 for rotations about axes with zero
        moment of inertia; unpack(v) sets the momenta back; the projection is the identity.
        """
        P, M = dyn.momentum.center, dyn.mass.center  # M: (nmol, 1)
        q = dyn.position.orientation
        I = dyn.mass.orientation  # (nmol, 3) principal moments
        L = rigid_body.conjugate_momentum_to_angular_momentum(q, dyn.momentum.orientation)
        has = I > 0
        v = jnp.stack([P / jnp.sqrt(M), jnp.where(has, L / jnp.sqrt(jnp.where(has, I, 1.0)), 0.0)])
        mask = jnp.stack([jnp.ones_like(P), has.astype(P.dtype)])

        def unpack(v: jax.Array) -> Dynamics:  # scaled momenta -> centre and quaternion momenta
            Pq = rigid_body.angular_momentum_to_conjugate_momentum(q, v[1] * jnp.sqrt(I))
            return dyn.set(momentum=RigidBody(v[0] * jnp.sqrt(M), Pq))

        return v, mask, unpack, (lambda u: u)

    def _init_aux(self, dyn: Dynamics) -> tuple[Dynamics, jax.Array]:
        """Return dyn (key advanced) and thermostat auxiliaries drawn at kT (empty without any)."""
        v, mask, _, project = self._scaled(dyn)
        if self.thermostat is None or self.thermostat.n_aux == 0:
            return dyn, jnp.zeros((0,) + v.shape, jnp.float64)
        key, k = jax.random.split(dyn.rng)
        aux = jax.vmap(project)(self.thermostat.init_aux(k, v.shape, self.kT))
        if mask is not None:
            aux = aux * mask[None]
        return dyn.set(rng=key), aux

    def thermostat_kT(self, st: MDState) -> float | jax.Array:
        """Return the kB T [kJ/mol] of the thermostat and barostat: the state's own or the integrator's.

        The state's own is set by replica exchange (MDState.kT).
        """
        return self.kT if st.kT is None else st.kT

    def _o_step(
        self, dyn: Dynamics, aux: jax.Array, heat: jax.Array, h: float, kT: float | jax.Array | None = None
    ) -> tuple[Dynamics, jax.Array, jax.Array]:
        """Apply the thermostat step of length h [ps] at fixed positions; return dyn, aux and heat.

        The heat increases by the change of |v|^2/2 + |aux|^2/2 [kJ/mol].
        """
        v, mask, unpack, project = self._scaled(dyn)
        key, k = jax.random.split(dyn.rng)
        e0 = 0.5 * (jnp.sum(v * v) + jnp.sum(aux * aux))
        v, aux = self.thermostat.apply(v, aux, k, h, self.kT if kT is None else kT, float(self.dof), project, mask)
        e1 = 0.5 * (jnp.sum(v * v) + jnp.sum(aux * aux))
        return unpack(v).set(rng=key), aux, heat + (e1 - e0)

    # --------------------------------------------------------------------- one step
    def _step(self, st: MDState) -> MDState:
        """Advance one step (BAOAB, or velocity Verlet in NVE; barostat every `interval` steps)."""
        dt = self.dt
        aux, heat = st.aux, st.heat
        dyn = simulate.momentum_step(st.dyn, dt / 2)
        if self.ensemble == "nve":
            dyn = simulate.position_step(dyn, self.shift, dt)
        else:
            dyn = simulate.position_step(dyn, self.shift, dt / 2)
            dyn, aux, heat = self._o_step(dyn, aux, heat, dt, self.thermostat_kT(st))
            dyn = simulate.position_step(dyn, self.shift, dt / 2)
        F, res, nbr = self._forces(
            dyn.position, st.box, st.induction, st.nbr, lam=st.lam, bias=st.bias, field=self.field_at(st, st.step + 1)
        )
        st = self._with_result(st.set(dyn=dyn, aux=aux, heat=heat), F, res, nbr)
        st = st.set(dyn=simulate.momentum_step(st.dyn, dt / 2), step=st.step + 1)
        if self.ensemble == "npt":
            st = jax.lax.cond(st.step % self.interval == 0, self._barostat, lambda s: s, st)
        return st

    def _barostat(self, st: MDState) -> MDState:
        """Try one isotropic Monte Carlo volume move (molecular scaling) and adapt the step size.

        dV uniform in [-mc_dv, mc_dv]; centres and box scaled by s = (V'/V)^(1/3), orientations
        and momenta kept; the trial energy (dipoles re-solved, restraints and biases included) is
        accepted with probability min(1, exp(-w / kT)),
        w = dE + P dV - M kT ln(V'/V) (M molecules).  With iEL/0-SCF shadow dynamics both energies
        are converged ones.  After every 10 tries of a window the step size is divided by 1.1
        below 25 % acceptance and multiplied by 1.1 above 75 % (at most 0.3 V).
        """
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
        if self.ff.shadow:  # iEL/0-SCF: converged energies at both volumes
            pos0 = self.rigid.positions(body)
            e_0 = self.ff.energy(
                pos0, H, self.nb.candidates(st.nbr, body.center, H, pos0)[0], st.induction, self.params, efield=field
            )[0] + self._restraint_energy(pos0, H, st.bias)
        w = (e_n - e_0) + self.pressure * dV - self.nmol * kT * jnp.log(jnp.maximum(Vn, 1e-12) / V)
        accept = (Vn > 0) & (jnp.log(jax.random.uniform(k2, dtype=jnp.float64)) < -w / kT)
        st = st.set(dyn=st.dyn.set(rng=key), overflow=st.overflow | ovf)

        def acc(st: MDState) -> MDState:  # accepted: move to the trial box and re-evaluate the forces
            st = st.set(dyn=st.dyn.set(position=body_n), box=Hn, induction=ind_n)
            F, res, nbr = self._forces(body_n, Hn, ind_n, nbr_n, lam=st.lam, bias=st.bias, field=field)
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

    def _run(self, st: MDState, n: int | jax.Array) -> MDState:
        """Advance n steps (jitted as `run`); reset the block maxima (CG iterations, residual, overflow).

        With a strided bias the loop is `strided_loop` with `_bias_post`; otherwise one
        lax.fori_loop of `_step` (with the field's heat booking).
        """
        st = st.set(max_iters=jnp.zeros((), jnp.int32), resid=jnp.zeros((), jnp.float64), overflow=jnp.zeros((), bool))

        def step(s: MDState) -> MDState:  # one step with the time-dependent field's heat
            return self._book_field(s, self._step(s))

        if self.bias is None or self.bias.stride == 0:
            return jax.lax.fori_loop(0, n, lambda _, s: step(s), st)
        return strided_loop(st, n, step, self._bias_post, self.bias.stride)

    # --------------------------------------------------------------------- observables
    def kinetic(self, st: MDState) -> tuple[jax.Array, jax.Array]:
        """Return the (total, centre-of-mass translational) kinetic energy [kJ/mol]."""
        ke = simulate.kinetic_energy(st.dyn)
        p, m = st.dyn.momentum.center, st.dyn.mass.center
        return ke, 0.5 * jnp.sum(p * p / m)

    def temperature(self, st: MDState) -> jax.Array:
        """Return the kinetic temperature 2 K / (dof kB) [K]."""
        return 2.0 * self.kinetic(st)[0] / (self.dof * KB)

    def temperatures(self, st: MDState) -> tuple[jax.Array, jax.Array]:
        """Return the (translational, rotational) temperatures [K] (equipartition check)."""
        ke, ke_t = self.kinetic(st)
        n_t = 3 * self.nmol - (3 if self.ensemble == "nve" else 0)
        n_r = 3 * self.nmol - self.rigid.dof_correction
        return 2.0 * ke_t / (n_t * KB), 2.0 * (ke - ke_t) / (n_r * KB)
