"""Temperature replica exchange (parallel tempering) for both MD engines.

Contents: the exchange logic (`geometric_ladder`, `exchange_pairs`, `metropolis`,
`temperature_reduced_energies`, `ExchangeStatistics`, `read_exchange_log`), helpers for stacked
(batched) states (`_stack`, `_take`, `_axes`, ...), the replica engine `MDReplicas` (also the
base of alchemy.LambdaWindows and finite_field.FieldReplicas) and the driver `ReplicaExchange`.

R copies (replicas) of one system run at the temperatures T_0 < T_1 < ... < T_{R-1} of a ladder
(`geometric_ladder`).  Every `exchange_every` steps neighbouring temperatures try to swap their
configurations, alternately the even pairs (0,1), (2,3), ... and the odd pairs (1,2), (3,4), ...
(deterministic even/odd [2]_; it gives faster round trips than random pair choices [3]_).  A swap
of the configurations x_i, x_j of states i, j is accepted with the Metropolis probability

    P = min(1, exp(-Delta)),   Delta = u_i(x_j) + u_j(x_i) - u_i(x_i) - u_j(x_j),

with u_k(x) the reduced (dimensionless) energy of configuration x in thermodynamic state k.  For
temperature exchange u_k(x) = beta_k [U(x) + P V(x)] (P V at constant pressure only), so that
Delta = (beta_i - beta_j) [U(x_j) - U(x_i) + P (V_j - V_i)] ([1]_; [2]_ for NPT).  The momenta of
an exchanged configuration are rescaled by sqrt(T_new / T_old), which removes the kinetic
energies from Delta; the thermostat auxiliaries (GLE: mass-scaled momenta with variance kT) are
rescaled in the same way.

Hamiltonian exchange.  Only reduced energies enter the criterion (`metropolis`).  Replicas with
different parameters, restraints or scaled terms need u_k(x_j) = beta_k U_k(x_j), i.e. the energy
of each configuration under the neighbouring Hamiltonians (one extra energy evaluation per pair and
exchange: override `ReplicaExchange.reduced_energies`), and an accepted swap must re-evaluate the
forces and induced dipoles in the new Hamiltonian (`MDReplicas.permute` moves them with the
configuration, which is exact only when every replica has the same Hamiltonian).

What moves and what stays.  The temperatures are fixed slots and configurations move between
them.  A swap moves the positions, box, forces, energies, induction state (induced dipoles, the
predictor history and its normaliser), neighbour list, and the rescaled momenta and thermostat
auxiliaries.  The temperature (MDState.kT), the random stream of the thermostat (one independent
stream per slot), the step counter, the barostat counters and step size, and the statistics stay
with the slot.  The energy a slot receives in an exchange is booked as heat, so each slot's
"econs" (E_tot + |aux|^2/2 - heat) keeps showing the integration drift across exchanges (each
accepted swap brings a trajectory with its own fluctuation of the integration error: a random walk
of about +-50 kJ/mol in 4 ns for the peptide below, no systematic drift).  After a swap, the
dipole predictor extrapolates a history recorded at the old speed; the CG absorbs this at the
same tolerance (peptide at 300 K: 7.40 CG iterations per step against 7.36 in plain MD).

Engines (`MDReplicas`).  One Simulation or FlexibleSimulation serves every replica: kB T is a
state variable (MDState.kT), so one compiled step runs every temperature.  Two modes:

  batched=True    the replicas are one stacked state advanced by jax.vmap of the step, one program
                  for all of them.  Small systems use the GPU much better: ACE-(ALA)3-NME in 580
                  waters (1,782 atoms) takes 0.96 ms/step alone and 2.64 ms/step for 8 replicas
                  (2.9x the throughput; 4 replicas already reach 2.6x).  NVT only: under vmap, a
                  lax.cond with a per-replica predicate runs both branches, which would evaluate
                  the Monte Carlo barostat's trial energy every step.  The same effect makes every
                  step rebuild the (cheap, molecular-centre) neighbour list; the predictor's
                  fused/unfused switch stays a real branch because the induction step counter,
                  identical in all replicas, is kept unbatched (10 % faster).  Static sizes
                  (row capacity, neighbour-list capacities) are shared: an overflow in any
                  replica resizes all of them and repeats the block for all.
  batched=False   the replicas are advanced one after the other through the engine's own driver
                  (Simulation.advance: overflow handling, rebuilds of the neighbour lists when an
                  NPT volume drifts).  For NPT, and for systems that fill the GPU on their own.
                  NPT ladders whose volumes differ by more than ~10 % make the shared neighbour-list
                  layout rebuild (and recompile) as replicas alternate.

Both modes give the same trajectories and exchanges (to floating-point summation order).

Checks: tests/test_remd.py (Metropolis statistics with known energies, harmonic oscillators at
every temperature, swaps, batched = sequential, restarts); scripts/protein/remd_peptide.py
(ACE-(ALA)3-NME in water, 8 replicas at 300-400 K, 4 ns each: acceptance 0.34-0.39 for every
pair, 200 round trips, energy and backbone populations at 300 K as in plain MD of the same length).

Outputs of `ReplicaExchange.run(prefix=...)`, per temperature slot k (two digits):
  prefix_Tkk.log   observables as in Simulation.run, plus the replica at that temperature;
  prefix_Tkk.nc    Amber NetCDF trajectory at T_k (per temperature, i.e. demultiplexed by
                   temperature); prefix_Tkk.rst7 Amber restart;
and for the whole run: prefix_remd.log (one line per exchange: the replica at each temperature
from then on and the outcome of each neighbour pair), prefix_remd.json (ladder, acceptance
matrix, round trips, speed) and prefix.remd.chk (the complete state: every replica, the replica
map, statistics and the exchange random state; `ReplicaExchange.load_checkpoint`).  Frames and log lines at
a step are written after that step's exchange, so the replica of a frame is the one in the
exchange-log line of that step or the last line before it (`read_exchange_log`).

Units: K, kJ/mol, nm, ps.

References
----------
.. [1] Y. Sugita, Y. Okamoto, Chem. Phys. Lett. 314, 141 (1999).
.. [2] T. Okabe, M. Kawata, Y. Okamoto, M. Mikami, Chem. Phys. Lett. 335, 435 (2001).
.. [3] S. Syed, A. Bouchard-Cote, G. Deligiannidis, A. Doucet, J. R. Stat. Soc. B 84, 321
   (2022).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..units import KB
from .driver import (
    LogTable,
    Stopwatch,
    block_length,
    device_tree,
    host_tree,
    read_checkpoint,
    retry_block,
    write_checkpoint,
)
from .engine import OPTIONAL_STATE
from .io import NetCDFTrajectory, write_restart

if TYPE_CHECKING:
    from typing import TextIO

    from .integrate import MDState

LEGACY_FORMAT = "pgm_jax remd 1"  # the "format" entry of legacy pickle checkpoints
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- exchange logic
def geometric_ladder(t_min: float, t_max: float, n: int) -> np.ndarray:
    """Return n temperatures from t_min to t_max in constant ratio.

    T_k = t_min (t_max / t_min)^(k / (n - 1)).  When the heat capacity changes little over the
    range, neighbouring energy distributions then overlap equally, so every pair has about the same
    acceptance.

    Parameters
    ----------
    t_min, t_max : float
        Lowest and highest temperature [K].
    n : int
        Number of temperatures (>= 2).

    Returns
    -------
    np.ndarray (n,)
        Temperatures [K], increasing.

    Raises
    ------
    ValueError
        Unless n >= 2 and 0 < t_min < t_max.

    Examples
    --------
    >>> geometric_ladder(300.0, 1200.0, 3)
    array([ 300.,  600., 1200.])
    """
    if int(n) < 2 or not (0.0 < t_min < t_max):
        raise ValueError("need n >= 2 and 0 < t_min < t_max")
    return float(t_min) * (float(t_max) / float(t_min)) ** (np.arange(int(n)) / (int(n) - 1.0))


def exchange_pairs(n: int, parity: int) -> list[tuple[int, int]]:
    """Return the neighbour pairs tried together: (0,1), (2,3), ... for even parity, (1,2), ... for odd.

    Parameters
    ----------
    n : int
        Number of states.
    parity : int
        Exchange counter (only its parity is used).

    Returns
    -------
    list of tuple of int
        Disjoint neighbour pairs (i, i + 1).

    Examples
    --------
    >>> exchange_pairs(5, 1)
    [(1, 2), (3, 4)]
    """
    return [(i, i + 1) for i in range(int(parity) % 2, int(n) - 1, 2)]


def metropolis(u: ArrayLike, pairs: Sequence[tuple[int, int]], uniforms: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Accept or reject configuration swaps between disjoint pairs of states.

    A pair (i, j) is accepted with probability min(1, exp(-Delta)),
    Delta = u[i, j] + u[j, i] - u[i, i] - u[j, j] (module docstring).

    Parameters
    ----------
    u : ArrayLike (K, K)
        u[i, j] = u_i(x_j): reduced (dimensionless) energy, in state i, of the configuration now in
        state j.
    pairs : Sequence[tuple[int, int]]
        Disjoint pairs of states.
    uniforms : ArrayLike (len(pairs),)
        One U(0, 1) number per pair.

    Returns
    -------
    accepted : np.ndarray (len(pairs),) bool
        Outcome per pair.
    src : np.ndarray (K,) int
        After the swaps state i holds the configuration previously in state src[i].

    Raises
    ------
    ValueError
        If the pairs are not disjoint.
    FloatingPointError
        If a Delta is not finite.
    """
    u = np.asarray(u, float)
    src = np.arange(len(u))
    acc = np.zeros(len(pairs), bool)
    seen = set()
    for k, (i, j) in enumerate(pairs):
        if i in seen or j in seen or i == j:
            raise ValueError("exchange pairs must be disjoint")
        seen.update((i, j))
        delta = u[i, j] + u[j, i] - u[i, i] - u[j, j]
        if not np.isfinite(delta):
            raise FloatingPointError(f"exchange {i}-{j}: reduced energy difference {delta}")
        acc[k] = delta <= 0.0 or uniforms[k] < np.exp(-delta)
        if acc[k]:
            src[i], src[j] = src[j], src[i]
    return acc, src


def temperature_reduced_energies(
    temperatures: ArrayLike, U: ArrayLike, V: ArrayLike | None = None, pressure: float | None = None
) -> np.ndarray:
    """Return u[i, j] = beta_i (U_j + P V_j), the reduced energies of a temperature ladder.

    Parameters
    ----------
    temperatures : ArrayLike (K,)
        Temperatures of the slots [K].
    U : ArrayLike (K,)
        Potential energy of the configuration in each slot [kJ/mol].
    V : ArrayLike (K,), optional
        Volume of each configuration [nm^3] (needed with `pressure`).
    pressure : float, optional
        Pressure [kJ/mol/nm^3] (None: constant volume, no P V term).

    Returns
    -------
    np.ndarray (K, K)
        Reduced energies (dimensionless).
    """
    beta = 1.0 / (KB * np.asarray(temperatures, float))
    H = np.asarray(U, float)
    if pressure is not None:
        H = H + float(pressure) * np.asarray(V, float)
    return beta[:, None] * H[None, :]


class ExchangeStatistics:
    """Attempts and acceptances per pair of states, the replica at each state, and round trips.

    A replica (walker) completes a round trip when it has gone from the lowest temperature to the
    highest and back; `transits` counts one-way trips between the two ends.  Mutable host object;
    `to_dict` / `from_dict` serialize it for checkpoints.

    Attributes
    ----------
    n : int
        Number of states (temperature or lambda slots).
    attempts, accepts : np.ndarray (n, n) int
        Swap attempts and acceptances per pair (symmetric).
    replica : np.ndarray (n,) int
        Replica (walker) at each state.
    last_end : np.ndarray (n,) int
        Per replica: last end visited (0 lowest, 1 highest, -1 none).
    phase : np.ndarray (n,) int
        Per replica: 1 after the lowest end, 2 after the lowest then the highest.
    round_trips, transits : np.ndarray (n,) int
        Per replica: completed round trips and one-way transits.
    n_exchanges : int
        Exchange attempts (rounds of pairs) so far.
    """

    def __init__(self, n: int) -> None:
        """Set up empty statistics for n states, replica k at state k.

        Parameters
        ----------
        n : int
            Number of states.
        """
        self.n = int(n)
        self.attempts = np.zeros((n, n), int)
        self.accepts = np.zeros((n, n), int)
        self.replica = np.arange(n)  # replica (walker) at each temperature
        self.last_end = np.full(n, -1)  # per replica: last end visited (0 lowest, 1 highest)
        self.phase = np.zeros(n, int)  # per replica: 1 after T_0, 2 after T_0 then T_max
        self.round_trips = np.zeros(n, int)
        self.transits = np.zeros(n, int)
        self.n_exchanges = 0
        self._ends()

    def _ends(self) -> None:
        """Update the transit and round-trip counters for the replicas now at the two end states."""
        lo, hi = self.replica[0], self.replica[-1]
        if self.last_end[lo] == 1:
            self.transits[lo] += 1
        if self.phase[lo] == 2:
            self.round_trips[lo] += 1
        self.last_end[lo], self.phase[lo] = 0, 1
        if self.last_end[hi] == 0:
            self.transits[hi] += 1
        if self.phase[hi] == 1:
            self.phase[hi] = 2
        self.last_end[hi] = 1

    def record(self, pairs: Sequence[tuple[int, int]], accepted: Sequence[bool], src: ArrayLike) -> None:
        """Record one exchange round.

        Parameters
        ----------
        pairs : Sequence[tuple[int, int]]
            The pairs tried.
        accepted : Sequence[bool]
            Outcome per pair.
        src : ArrayLike (n,) int
            The permutation of `metropolis`: state i now holds the configuration of state src[i].
        """
        for (i, j), a in zip(pairs, accepted):
            self.attempts[i, j] += 1
            self.attempts[j, i] += 1
            self.accepts[i, j] += int(a)
            self.accepts[j, i] += int(a)
        self.replica = self.replica[np.asarray(src)]
        self.n_exchanges += 1
        self._ends()

    def acceptance(self) -> np.ndarray:
        """Return the (n, n) acceptance ratio accepted / attempted swaps (nan where never attempted)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.attempts > 0, self.accepts / np.maximum(self.attempts, 1), np.nan)

    def neighbour_acceptance(self) -> np.ndarray:
        """Return the acceptance ratio of the n - 1 neighbour pairs (i, i + 1) (nan where never attempted)."""
        a = self.acceptance()
        return np.array([a[i, i + 1] for i in range(self.n - 1)])

    def to_dict(self) -> dict:
        """Return every attribute (arrays copied), for a checkpoint."""
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExchangeStatistics:
        """Return statistics rebuilt from `to_dict` output (without calling __init__)."""
        out = cls.__new__(cls)
        for k, v in d.items():
            setattr(out, k, np.array(v) if isinstance(v, np.ndarray) else v)
        return out


def read_exchange_log(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the exchange log prefix_remd.log of `ReplicaExchange.run`.

    Parameters
    ----------
    path : str
        The log file.

    Returns
    -------
    steps : np.ndarray (E,) int
        Step of every exchange.
    replicas : np.ndarray (E, R) int
        Replica at each temperature from that step on.
    outcomes : np.ndarray (E, R-1) str
        Outcome per neighbour pair: "+" accepted, "." rejected, "-" not tried.
    """
    steps, reps, outs = [], [], []
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        steps.append(int(f[0]))
        reps.append([int(x) for x in f[1:-1]])
        outs.append(list(f[-1]))
    return np.array(steps, int), np.array(reps, int), np.array(outs)


# ----------------------------------------------------------------------------- stacked states
def _nocount(st: MDState) -> MDState:
    """Return the state without its induction step counter (kept unbatched in stacked states)."""
    return st.set(induction=st.induction.set(count=None))


def _stack(states: Sequence[MDState]) -> MDState:
    """Stack states along a new leading axis (every leaf (R, ...)), the induction counter unbatched.

    Parameters
    ----------
    states : Sequence[MDState]
        States of identical structure and static sizes.

    Returns
    -------
    MDState
        The stacked state; `induction.count` is the common scalar counter.

    Raises
    ------
    ValueError
        If the induction step counters differ.
    """
    counts = {int(s.induction.count) for s in states}
    if len(counts) != 1:
        raise ValueError(f"batched replicas need equal induction step counters, got {sorted(counts)}")
    S = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[_nocount(s) for s in states])
    return S.set(induction=S.induction.set(count=states[0].induction.count))


def _take(S: MDState, index: int | jax.Array) -> MDState:
    """Return slot `index` (an int, or an index array for a gather) of a stacked state."""
    T = jax.tree_util.tree_map(lambda x: x[index], _nocount(S))
    return T.set(induction=T.induction.set(count=S.induction.count))


def _axes(S: MDState) -> MDState:
    """Return the vmap axes of a stacked state (a pytree of the state's structure).

    0 for every leaf, None (unbatched) for the induction step counter, so the predictor's fused /
    unfused lax.cond stays a real branch under vmap.
    """
    return jax.tree_util.tree_map(lambda _: 0, _nocount(S))


def _broadcast(tree: Any, n: int) -> Any:
    """Return `tree` with every leaf broadcast to a new leading axis of length n."""
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), tree)


# ----------------------------------------------------------------------------- MD replicas
class MDReplicas:
    """The replicas of one Simulation or FlexibleSimulation at the temperatures of a ladder.

    The engine of ReplicaExchange (and the base of alchemy.LambdaWindows and
    finite_field.FieldReplicas).  `sim` provides the system, force field, integrator and neighbour
    lists (its `state` is used as scratch space); its thermostat is used at every temperature.  The
    replicas start from the current configuration of `sim` with momenta (and thermostat
    auxiliaries) drawn at their own temperatures from independent random streams (`seed`).

    The interface ReplicaExchange uses (a test engine can provide the same): temperatures, n, dt,
    pressure (kJ/mol/nm^3 or None), time_ps, advance(n), potentials(), volumes(), permute(src),
    observables(k), frames(), write_restarts(prefix), state_dict(), load_state_dict(d).

    Batched mode keeps one stacked MDState `S` (every leaf with a leading replica axis, the
    induction step counter unbatched; `_axes`) and vmapped, jitted entry points built by `_build`
    for the current static sizes; sequential mode keeps a list `states` and uses the engine's own
    driver.

    Attributes
    ----------
    sim : Simulation or FlexibleSimulation
        The engine.
    integ : Integrator
        Its integrator.
    batched : bool
        One vmapped program for all replicas (NVT only).
    temperatures : np.ndarray (R,)
        Temperature of each slot [K].
    n : int
        Number of replicas R.
    dt : float
        Time step [ps].
    pressure : float or None
        Barostat pressure [kJ/mol/nm^3] (as Integrator.pressure) under NPT, else None.
    time_ps : float
        Simulation time [ps].
    S : MDState
        Stacked state (batched mode).
    states : list of MDState
        Replica states (sequential mode).
    """

    def __init__(self, sim: Any, temperatures: ArrayLike, batched: bool = True, seed: int = 0) -> None:
        """Build the replica states from the current configuration of `sim`.

        Parameters
        ----------
        sim : Simulation or FlexibleSimulation
            The engine, with a thermostat.
        temperatures : ArrayLike (R,)
            Temperatures [K], positive and increasing (R >= 2).
        batched : bool
            One vmapped program for all replicas (NVT only) or one after the other.
        seed : int
            Seed of the replicas' momenta and thermostat streams.

        Raises
        ------
        ValueError
            No thermostat, multiple time stepping, batched NPT, or an invalid ladder.
        NotImplementedError
            A time-dependent bias (metadynamics, OPES).

        Notes
        -----
        Every state is initialized by the integrator at its own kT (momenta drawn at the engine's
        temperature, then scaled by sqrt(kB T_k / kT) together with the thermostat auxiliaries), with
        MDState.kT = kB T_k and the neighbour list of `sim`'s state.
        """
        integ = sim.integ
        if integ.thermostat is None:
            raise ValueError("replica exchange needs a thermostat (create the simulation with thermostat=...)")
        if getattr(integ, "mts", None) is not None:
            raise ValueError("replica exchange with multiple time stepping (mts=) is not supported yet")
        if getattr(integ, "bias", None) is not None and integ.bias.dynamic:
            raise NotImplementedError(
                "replica exchange with a time-dependent bias (metadynamics / OPES: the bias "
                "would have to stay with its slot and enter the criterion); static biases "
                "are part of every replica's energy and work"
            )
        if batched and sim.ensemble == "npt":
            raise ValueError(
                "batched replicas run NVT only (under vmap the barostat's trial energy would be "
                "evaluated every step); use batched=False for NPT"
            )
        T = np.asarray(temperatures, float)
        if T.ndim != 1 or len(T) < 2 or np.any(np.diff(T) <= 0) or T[0] <= 0:
            raise ValueError("temperatures: at least two, positive and increasing")
        self.sim, self.integ, self.batched = sim, integ, bool(batched)
        self.temperatures, self.n, self.dt = T, len(T), float(sim.dt)
        self.pressure = float(integ.pressure) if sim.ensemble == "npt" else None
        self.time_ps = 0.0
        base = sim.state
        self._template = base.nbr
        states = []
        for Tk, key in zip(T, jax.random.split(jax.random.PRNGKey(int(seed)), self.n)):
            st = integ.init(base.dyn.position, base.box, key)  # momenta drawn at the engine's kT
            s = float(np.sqrt(KB * Tk / integ.kT))
            st = st.set(
                dyn=st.dyn.set(momentum=jax.tree_util.tree_map(lambda p: p * s, st.dyn.momentum)),
                aux=st.aux * s,
                kT=jnp.asarray(KB * Tk, jnp.float64),
                nbr=base.nbr,
            )
            states.append(st)
        self._exchange_seq = jax.jit(self._exchange_one)
        if self.batched:
            self.S = _stack(states)
            self._build()
        else:
            self.states = states

    # ------------------------------------------------------------------ compiled pieces
    def _build(self) -> None:
        """Build the vmapped, jitted entry points for the current static sizes and neighbour-list layout.

        `_run` (a block of steps), `_forces` (forces of every slot), `_exchange` (the configuration
        swap), `_wrap`, `_positions` and, for flexible molecules, `_extent` (largest atom-to-centre
        distance).  Rebuilt whenever the stacked state's sizes or layout change.
        """
        integ, sim = self.integ, self.sim
        ax = _axes(self.S)
        self._run = jax.jit(jax.vmap(integ._run, in_axes=(ax, None), out_axes=ax))
        self._forces = jax.jit(jax.vmap(lambda s: integ._state_forces(s, True), in_axes=(ax,), out_axes=ax))
        self._exchange = jax.jit(jax.vmap(self._exchange_one, in_axes=(ax, ax, 0), out_axes=ax))
        self._wrap = jax.jit(jax.vmap(sim.rigid.wrap))
        self._positions = jax.jit(jax.vmap(sim.rigid.positions))
        flex = getattr(sim, "flex", None)
        self._extent = jax.jit(lambda P: jnp.max(jax.vmap(flex.extent)(P))) if flex is not None else None

    def _exchange_one(self, dst: MDState, src: MDState, s: float | jax.Array) -> MDState:
        """Return dst's slot with src's configuration.

        Positions, box, forces, induction state, neighbour list and energies come from `src`; the
        momenta and thermostat auxiliaries of `src` are scaled by s = sqrt(T_dst / T_src).  The slot's
        kT, random stream, step and barostat counters stay; the change of its total energy
        (kinetic + potential + |aux|^2/2) is booked as heat.

        Parameters
        ----------
        dst : MDState
            State of the receiving slot.
        src : MDState
            State whose configuration moves in.
        s : float or jax.Array ()
            Momentum scale factor (dimensionless).

        Returns
        -------
        MDState
        """

        def ke(st):
            return self.integ.kinetic(st)[0]

        e0 = ke(dst) + dst.epot + 0.5 * jnp.sum(dst.aux * dst.aux)
        mom = jax.tree_util.tree_map(lambda p: p * s, src.dyn.momentum)
        new = dst.set(
            dyn=dst.dyn.set(position=src.dyn.position, momentum=mom, force=src.dyn.force),
            box=src.box,
            induction=src.induction,
            nbr=src.nbr,
            epot=src.epot,
            elec=src.elec,
            vdw=src.vdw,
            iters=src.iters,
            aux=src.aux * s,
        )
        e1 = ke(new) + new.epot + 0.5 * jnp.sum(new.aux * new.aux)
        return new.set(heat=dst.heat + (e1 - e0))

    # ------------------------------------------------------------------ access
    def _slot(self, S: MDState, k: int) -> MDState:
        """Return the MDState of slot k of the stacked state S."""
        return _take(S, k)

    def state(self, k: int) -> MDState:
        """Return the MDState of slot k."""
        return self._slot(self.S, k) if self.batched else self.states[k]

    def state_template(self) -> MDState:
        """Return a state with the structure of the checkpointed replica states.

        Slot 0 without its neighbour list: the template driver.read_checkpoint rebuilds the states on.
        """
        return self.state(0).set(nbr=None)

    def potentials(self) -> np.ndarray:
        """Return the potential energy of the configuration in each slot [kJ/mol], shape (R,)."""
        if self.batched:
            return np.asarray(self.S.epot, float)
        return np.array([float(s.epot) for s in self.states])

    def volumes(self) -> np.ndarray:
        """Return the box volume of each slot [nm^3], shape (R,)."""
        boxes = np.asarray(self.S.box) if self.batched else np.array([np.asarray(s.box) for s in self.states])
        return np.abs(np.linalg.det(boxes))

    def _on(self, k: int) -> Any:
        """Point the engine's simulation object at slot k (for its observables and writers) and return it."""
        self.sim.state, self.sim.time_ps = self.state(k), self.time_ps
        return self.sim

    def observables(self, k: int) -> dict[str, Any]:
        """Return the observables of slot k (Simulation.observables of that state)."""
        return self._on(k).observables()

    def positions(self, k: int) -> np.ndarray:
        """Return the atom positions (N, 3) [nm] of slot k."""
        return self._on(k).positions()

    def frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the positions and boxes of every slot, in Angstrom (for the NetCDF trajectories).

        Returns
        -------
        xyz : np.ndarray (R, N, 3)
            Atom positions [Angstrom].
        boxes : np.ndarray (R, 3, 3)
            Boxes, lattice vectors as rows [Angstrom].
        """
        if self.batched:
            return np.asarray(self._positions(self.S.dyn.position)) * 10.0, np.asarray(self.S.box) * 10.0
        X = [self.positions(k) for k in range(self.n)]
        return np.array(X) * 10.0, np.array([np.asarray(s.box) for s in self.states]) * 10.0

    def write_restarts(self, prefix: str) -> None:
        """Write an Amber NetCDF restart of every slot: prefix_Tkk.rst7."""
        for k in range(self.n):
            sim = self._on(k)
            write_restart(
                f"{prefix}_T{k:02d}.rst7",
                sim.positions() * 10.0,
                sim.velocities() * 10.0,
                np.asarray(sim.state.box) * 10.0,
                self.time_ps,
                title=f"pgm_jax REMD T={self.temperatures[k]:g} K",
            )

    # ------------------------------------------------------------------ dynamics
    def advance(self, n: int) -> None:
        """Advance every replica by n steps (and the time by n dt)."""
        if self.batched:
            self._advance_batched(int(n))
        else:
            self._advance_sequential(int(n))
        self.time_ps += int(n) * self.dt

    def _sizes(self) -> tuple[Any, int | None]:
        """Return the static sizes shared by the replicas: row capacities (ff.capacity), molecule-list cap."""
        return self.sim.ff.capacity, getattr(self.sim.nb, "cap", None)

    def _fit(self, sizes: Sequence[tuple[Any, int | None]], grow_rows: Any = None, grow_cap: int | None = None) -> None:
        """Set static sizes that fit every entry of `sizes` (the largest of each part); re-jit if changed.

        Parameters
        ----------
        sizes : Sequence of tuple
            Entries of `_sizes`.
        grow_rows : optional
            Row capacities that overflowed: the rows grow beyond them (PGMForceField.grow_rows).
        grow_cap : int, optional
            Molecule-list cap that overflowed: the new cap is at least 4 above it.
        """
        sim, now = self.sim, self._sizes()
        sim.ff.fit_rows([s[0] for s in sizes])
        if grow_rows is not None:
            sim.ff.grow_rows(grow_rows)
        caps = [s[1] for s in sizes if s[1] is not None]
        if caps and getattr(sim.nb, "cap", None) is not None:
            sim.nb.cap = max(caps) if grow_cap is None else max(max(caps), grow_cap + 4)
        if self._sizes() != now:
            self.integ.compile()

    def _advance_sequential(self, n: int) -> None:
        """Advance the replicas one after the other by n steps through the engine's driver.

        A replica may re-size the shared static sizes from its own configuration; they never drop
        below what another replica needed (no resize ping-pong).
        """
        sim = self.sim
        for k in range(self.n):
            before = self._sizes()
            sim.state, sim.time_ps = self.states[k], self.time_ps
            sim.advance(n)
            self.states[k] = sim.state
            # a replica may re-size the shared static sizes from its own configuration: never
            # below what another replica needed (no resize ping-pong)
            self._fit([before, self._sizes()])

    def _nb_failed(self, S: MDState) -> bool:
        """Return whether any replica's neighbour list failed (the engine's test on each error code)."""
        codes = np.asarray(S.nbr.error.code).reshape(-1)
        return any(self.sim.nb.failed(SimpleNamespace(error=SimpleNamespace(code=c))) for c in codes)

    def _run_block(self, start: MDState, n: int) -> MDState:
        """Return the stacked state after one compiled block of n steps of every replica (blocks until done)."""
        new = self._run(start, n)
        jax.block_until_ready(new.epot)
        return new

    def _advance_batched(self, n: int) -> None:
        """Advance every replica by n steps as one vmapped block.

        Overflows of any replica's neighbour list or pair rows re-size the shared static sizes for
        all and repeat the block (driver.retry_block); then the molecules are re-wrapped into the
        box, the energies checked and, for flexible molecules, their extent checked against the
        list radius.

        Parameters
        ----------
        n : int
            Steps.

        Raises
        ------
        FloatingPointError
            A replica's energy is not finite.
        RuntimeError
            A block keeps overflowing, or an atom beyond the neighbour-list radius (r_margin).
        """
        sim = self.sim

        def resize(start: MDState, nb_bad: bool, row_bad: bool) -> MDState:
            """Resize (`_resize`) after an overflow and log a line."""
            step0 = int(np.asarray(start.step)[0])
            start = self._resize(start, nb_bad, row_bad)
            logger.info(
                f"{'neighbour list' if nb_bad else 'row capacity'} overflow in steps {step0}-{step0 + n} "
                f"(replicas): resized (rows {sim.ff.mc or sim.nb.cap}, list "
                f"{self._template.idx.shape[1]}), repeating"
            )
            return start

        new = retry_block(
            lambda s: self._run_block(s, n),
            self.S,
            lambda s: (self._nb_failed(s), bool(np.any(np.asarray(s.overflow)))),
            resize,
        )
        new = new.set(dyn=new.dyn.set(position=self._wrap(new.dyn.position, new.box)))
        e = np.asarray(new.epot)
        if not np.all(np.isfinite(e)):
            raise FloatingPointError(
                f"energy is not finite at step {int(np.asarray(new.step)[0])} "
                f"(replicas {np.nonzero(~np.isfinite(e))[0].tolist()})"
            )
        self.S = new
        if self._extent is not None and sim.nb.kind == "molecule":
            ext = float(self._extent(new.dyn.position))
            if ext > sim.r_list:
                raise RuntimeError(
                    f"an atom is {ext:.3f} nm from its group's centre, beyond the neighbour-list "
                    f"radius {sim.r_list:.3f} nm; increase r_margin"
                )

    def _resize(self, start: MDState, nb_bad: bool, row_bad: bool) -> MDState:
        """Set static sizes and one neighbour-list layout that fit every replica; return forces at the start.

        The sizes are the largest of each part over the replicas, never below what overflowed; after a
        list overflow the largest list becomes the shared layout (`_template`), filled with each
        replica's neighbours by the force evaluation.

        Parameters
        ----------
        start : MDState
            Stacked state at the start of the failed block.
        nb_bad : bool
            A neighbour list overflowed.
        row_bad : bool
            The pair rows overflowed.

        Returns
        -------
        MDState
            `start` with forces (and lists) re-evaluated at the new sizes, its induction state kept.
        """
        sim = self.sim
        old = self._sizes()
        sizes, lists = [], []
        for k in range(self.n):
            st = self._slot(start, k)
            lists.append(sim._size_lists(st.dyn.position, st.box, 1.3, None if nb_bad else st.nbr))
            sizes.append(self._sizes())
        # never below what overflowed
        self._fit(sizes, old[0] if row_bad else None, old[1] if row_bad else None)
        if nb_bad:
            self._template = max(lists, key=lambda b: b.max_occupancy)
        self.integ.compile()
        S = start.set(nbr=_broadcast(self._template, self.n)) if nb_bad else start
        self.S = S
        self._build()
        return self._forces(S).set(induction=start.induction)

    def permute(self, src: ArrayLike) -> None:
        """Move the configuration of slot src[k] to slot k (momenta and auxiliaries rescaled).

        Parameters
        ----------
        src : ArrayLike (R,) int
            Permutation from `metropolis`.
        """
        src = np.asarray(src, int)
        if np.all(src == np.arange(self.n)):
            return
        scale = np.sqrt(self.temperatures / self.temperatures[src])
        if self.batched:
            self.S = self._exchange(self.S, _take(self.S, jnp.asarray(src)), jnp.asarray(scale))
        else:
            old = list(self.states)
            self.states = [
                old[k] if src[k] == k else self._exchange_seq(old[k], old[src[k]], scale[k]) for k in range(self.n)
            ]

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict:
        """Return the replicas' content for a checkpoint.

        Returns
        -------
        dict
            temperatures [K], time_ps, batched, and the state of every slot (host arrays, no
            neighbour lists).
        """
        return {
            "temperatures": self.temperatures.copy(),
            "time_ps": self.time_ps,
            "batched": self.batched,
            "states": [host_tree(self.state(k).set(nbr=None)) for k in range(self.n)],
        }

    def load_state_dict(self, d: dict) -> None:
        """Load the replica states from a checkpoint (written in either mode).

        Neighbour lists are rebuilt and the static sizes fitted to the loaded configurations.

        Parameters
        ----------
        d : dict
            Output of `state_dict` (as read by driver.read_checkpoint).

        Raises
        ------
        ValueError
            If the checkpoint temperatures differ.
        """
        if len(d["temperatures"]) != self.n or not np.allclose(d["temperatures"], self.temperatures, rtol=1e-12):
            raise ValueError(f"checkpoint temperatures {list(d['temperatures'])} differ from {list(self.temperatures)}")
        sim = self.sim
        states = [device_tree(s) for s in d["states"]]
        sizes, lists = [], []
        for st in states:
            lists.append(sim._size_lists(st.dyn.position, st.box))
            sizes.append(self._sizes())
        self._fit(sizes)
        self.integ.compile()
        if self.batched:
            self._template = max(lists, key=lambda b: b.max_occupancy)
            S = _stack([s.set(nbr=self._template) for s in states])
            self.S = S
            self._build()
            self.S = S.set(nbr=self._forces(S).nbr)  # every replica's own list, one layout
        else:
            self.states = [s.set(nbr=nbr) for s, nbr in zip(states, lists)]
        self.time_ps = float(d["time_ps"])


# ----------------------------------------------------------------------------- driver
class ReplicaExchange:
    """Temperature replica exchange driver: advance, exchange, statistics, files and checkpoints.

        sim = FlexibleSimulation(..., thermostat="bussi", dt=0.002, constraints="h-bonds", hmr=3.024)
        sim.minimize(300); sim.run(25000)                       # equilibrate at the lowest temperature
        rex = ReplicaExchange(sim, geometric_ladder(300.0, 400.0, 8), exchange_every=500)
        rex.run(500000, prefix="ala3", report_every=5000, traj_every=500, checkpoint_every=50000)

    `sim`: a Simulation / FlexibleSimulation (replicas built by MDReplicas with `batched` and
    `seed`), or a ready replica engine with the MDReplicas interface (then `temperatures` is not
    given).  Exchanges are attempted every `exchange_every` steps.

    Attributes
    ----------
    replicas : MDReplicas or replica engine
        The replicas.
    T : np.ndarray (R,)
        Temperatures [K].
    n : int
        Number of replicas R.
    every : int
        Steps between exchange attempts.
    rng : np.random.Generator
        Exchange random numbers (PCG64, seeded from `seed`).
    stats : ExchangeStatistics
        Exchange statistics.
    step : int
        Steps done.
    """

    def __init__(
        self,
        sim: Any,
        temperatures: ArrayLike | None = None,
        exchange_every: int = 500,
        batched: bool = True,
        seed: int = 0,
        log: TextIO | None = None,
    ) -> None:
        """Set up the replicas and the exchange statistics.

        Parameters
        ----------
        sim : Simulation, FlexibleSimulation or replica engine
            An MD engine (replicas built by MDReplicas with `batched` and `seed`), or a ready
            replica engine with the MDReplicas interface (then `temperatures` is not given).
        temperatures : array (R,), optional
            Temperature ladder [K], increasing.
        exchange_every : int
            Steps between exchange attempts.
        batched : bool
            One vmapped program for all replicas (NVT only) or one after the other.
        seed : int
            Seed of the replicas' streams and of the exchange random numbers.
        log : text stream or None
            Receives one progress line per report (acceptance, round trips, speed); the setup
            is logged to the logger "pgm_jax.md.remd".

        Raises
        ------
        ValueError
            Missing or superfluous temperatures, exchange_every < 1.
        """
        if hasattr(sim, "integ"):
            if temperatures is None:
                raise ValueError("give the temperatures of the replicas")
            self.replicas = MDReplicas(sim, temperatures, batched=batched, seed=seed)
        else:
            if temperatures is not None:
                raise ValueError("a replica engine brings its own temperatures")
            self.replicas = sim
        self.T = np.asarray(self.replicas.temperatures, float)
        self.n = len(self.T)
        self.every = int(exchange_every)
        if self.every < 1:
            raise ValueError("exchange_every must be >= 1")
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 0x5EED])))
        self.stats = ExchangeStatistics(self.n)
        self.step = 0
        self.log = log
        mode = "batched" if getattr(self.replicas, "batched", False) else "sequential"
        logger.info(
            f"replica exchange: {self.n} replicas ({mode}), T = "
            f"{' '.join(f'{t:.2f}' for t in self.T)} K, exchanges every {self.every} steps "
            f"({self.every * self.replicas.dt:g} ps)"
            f"{'' if self.replicas.pressure is None else ', NPT (P V in the criterion)'}"
        )

    # ------------------------------------------------------------------ exchanges
    def reduced_energies(self) -> np.ndarray:
        """Return u[i, j] = u_i(x_j) (R, R) for the configurations now at each temperature.

        For a temperature ladder beta_i (U_j + P V_j) (dimensionless).  Hamiltonian exchange
        overrides this with energies of each configuration evaluated in the neighbouring states.
        """
        rep = self.replicas
        V = rep.volumes() if rep.pressure is not None else None
        return temperature_reduced_energies(self.T, rep.potentials(), V, rep.pressure)

    def exchange(self) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Attempt one exchange round (even or odd pairs, alternating) and apply the accepted swaps.

        Returns
        -------
        pairs : list of tuple of int
            The pairs tried.
        accepted : np.ndarray bool
            Outcome per pair.
        """
        pairs = exchange_pairs(self.n, self.stats.n_exchanges)
        uniforms = self.rng.random(len(pairs))
        acc, src = metropolis(self.reduced_energies(), pairs, uniforms)
        self.replicas.permute(src)
        self.stats.record(pairs, acc, src)
        return pairs, acc

    def _outcome(self, pairs: Sequence[tuple[int, int]], acc: Sequence[bool]) -> str:
        """Return the outcome string of one round for the exchange log ("+", "." or "-" per neighbour pair)."""
        out = ["-"] * (self.n - 1)
        for (i, _), a in zip(pairs, acc):
            out[i] = "+" if a else "."
        return "".join(out)

    # ------------------------------------------------------------------ running
    def run(
        self,
        nsteps: int,
        *,
        prefix: str | None = "remd",
        report_every: int = 1000,
        traj_every: int = 0,
        checkpoint_every: int = 0,
        append: bool = False,
    ) -> dict:
        """Advance every replica nsteps with exchanges every `exchange_every` steps.

        Parameters
        ----------
        nsteps : int
            Steps.
        prefix : str or None
            Path prefix of the files (None: no files): prefix_Tkk.log (log table per temperature),
            prefix_Tkk.nc, prefix_remd.log (exchanges), prefix_remd.json (summary),
            prefix.remd.chk and prefix_Tkk.rst7.
        report_every : int
            Steps between rows of the per-temperature logs and progress lines (0: none).
        traj_every : int
            Steps between trajectory frames of every temperature (0: none).
        checkpoint_every : int
            Steps between checkpoints (0: none; with checkpoints also at the end).
        append : bool
            Append to existing files (a continuation).

        Returns
        -------
        dict
            `summary` of the run.
        """
        report, traj, restart = report_every, traj_every, checkpoint_every
        rep, n = self.replicas, self.n
        block = block_length(nsteps, self.every, report, traj, restart)
        files = prefix is not None
        logs = (
            [LogTable(f"{prefix}_T{k:02d}.log", append=append, title=[f"T = {self.T[k]:.4f} K"]) for k in range(n)]
            if (files and report)
            else None
        )
        trajs = (
            [NetCDFTrajectory(f"{prefix}_T{k:02d}.nc", rep.frames()[0].shape[1], append=append) for k in range(n)]
            if (files and traj)
            else None
        )
        xlog = open(f"{prefix}_remd.log", "a" if append else "w") if files else None
        if xlog is not None and not append:
            xlog.write(
                f"# replica exchange, T (K) = {' '.join(f'{t:.4f}' for t in self.T)}\n"
                f"# step, replica at T_0 .. T_{n - 1} from this step on, outcome per pair (i, i+1): "
                f"+ accepted, . rejected, - not tried\n"
            )
        clock = Stopwatch(self.step, rep.dt)
        done = 0
        while done < nsteps:
            m = min(block, nsteps - done)
            rep.advance(m)
            done += m
            self.step += m
            if self.step % self.every == 0:
                pairs, acc = self.exchange()
                if xlog is not None:
                    xlog.write(
                        f"{self.step:12d} "
                        + " ".join(f"{r:3d}" for r in self.stats.replica)
                        + f"  {self._outcome(pairs, acc)}\n"
                    )
                    xlog.flush()
            speed = clock.ns_per_day(self.step)
            if report and self.step % report == 0:
                if logs is not None:
                    for k in range(n):
                        obs = rep.observables(k)
                        obs["replica"] = int(self.stats.replica[k])
                        obs["ns_per_day"] = speed
                        logs[k].write(obs)
                acc = self.stats.neighbour_acceptance()
                self._progress(
                    f"step {self.step} ({rep.time_ps:.1f} ps): acceptance "
                    f"{' '.join('  -  ' if np.isnan(a) else f'{a:.3f}' for a in acc)}, round trips "
                    f"{int(self.stats.round_trips.sum())}, {speed:.1f} ns/day per replica"
                )
            if trajs is not None and self.step % traj == 0:
                X, B = rep.frames()
                for k in range(n):
                    trajs[k].write(rep.time_ps, X[k], B[k])
            if files and restart and self.step % restart == 0:
                self._write_checkpoint_files(prefix)
        speed = clock.ns_per_day(self.step)
        for t in logs or []:
            t.close()
        if xlog is not None:
            xlog.close()
        if files and restart:
            self._write_checkpoint_files(prefix)
        summary = self.summary(ns_per_day=speed)
        if files:
            with open(f"{prefix}_remd.json", "w") as fh:
                json.dump(summary, fh, indent=1)
        return summary

    def _progress(self, line: str) -> None:
        """One progress line to the log stream (if any)."""
        if self.log is not None:
            print(line, file=self.log, flush=True)

    def summary(self, ns_per_day: float | None = None) -> dict:
        """Ladder, acceptance, round trips and (optionally) speed of the run so far.

        Parameters
        ----------
        ns_per_day : float, optional
            Speed per replica [ns/day] to include (with the aggregate speed).

        Returns
        -------
        dict
            JSON-serializable summary (the content of prefix_remd.json).
        """
        st = self.stats
        acc = st.acceptance()
        out = {
            "temperatures_K": self.T.tolist(),
            "exchange_every": self.every,
            "steps": self.step,
            "time_ps": self.replicas.time_ps,
            "exchanges": st.n_exchanges,
            "neighbour_acceptance": [None if np.isnan(a) else float(a) for a in st.neighbour_acceptance()],
            "acceptance_matrix": [[None if np.isnan(a) else float(a) for a in row] for row in acc],
            "attempts": st.attempts.tolist(),
            "accepts": st.accepts.tolist(),
            "round_trips": st.round_trips.tolist(),
            "round_trips_total": int(st.round_trips.sum()),
            "transits": st.transits.tolist(),
            "replica_at_temperature": st.replica.tolist(),
            "batched": bool(getattr(self.replicas, "batched", False)),
        }
        if ns_per_day is not None:
            out["ns_per_day_per_replica"] = ns_per_day
            out["ns_per_day_aggregate"] = ns_per_day * self.n
        return out

    # ------------------------------------------------------------------ checkpoints
    def _write_checkpoint_files(self, prefix: str) -> None:
        """Write the files of run's checkpoints: prefix.remd.chk and Amber restarts prefix_Tkk.rst7.

        The restarts are written only when the replica engine has `write_restarts`.
        """
        self.save_checkpoint(prefix + ".remd.chk")
        if hasattr(self.replicas, "write_restarts"):
            self.replicas.write_restarts(prefix)

    def save_checkpoint(self, path: str) -> None:
        """Write a checkpoint of the whole run (driver.write_checkpoint, kind "remd").

        Parameters
        ----------
        path : str
            The file (prefix.remd.chk in `run`).

        Notes
        -----
        The checkpoint holds every replica state (`state_dict` of the engine), the replica map
        and statistics, the step and the state of the exchange random-number generator;
        `load_checkpoint` continues the run bitwise on the CPU.
        """
        content = {
            "temperatures": self.T,
            "exchange_every": self.every,
            "step": self.step,
            "rng": self.rng.bit_generator.state,
            "stats": self.stats.to_dict(),
            "replicas": self.replicas.state_dict(),
        }
        write_checkpoint(path, "remd", content)

    def load_checkpoint(self, path: str) -> None:
        """Continue from a checkpoint written by `save_checkpoint` (or a legacy pickle .remd.chk).

        Legacy pickle checkpoints are those of pgm_jax up to commit e72c57c.

        Parameters
        ----------
        path : str
            The checkpoint (same system, settings and temperatures; the replica engine may be
            batched or sequential).

        Raises
        ------
        ValueError
            Another kind of checkpoint or other temperatures.
        """
        template = getattr(self.replicas, "state_template", None)
        d = read_checkpoint(path, "remd", template() if template else None, OPTIONAL_STATE, legacy_format=LEGACY_FORMAT)
        if len(d["temperatures"]) != self.n or not np.allclose(d["temperatures"], self.T, rtol=1e-12):
            raise ValueError(f"checkpoint temperatures {list(d['temperatures'])} differ from {list(self.T)}")
        self.replicas.load_state_dict(d["replicas"])
        self.step = int(d["step"])
        self.rng.bit_generator.state = d["rng"]
        self.stats = ExchangeStatistics.from_dict(d["stats"])
