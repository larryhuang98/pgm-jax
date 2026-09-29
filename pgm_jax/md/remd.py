"""Temperature replica exchange (parallel tempering) for both MD engines.

R copies (replicas) of one system run at the temperatures T_0 < T_1 < ... < T_{R-1} of a ladder
(`geometric_ladder`).  Every `exchange_every` steps neighbouring temperatures try to swap their
configurations, alternately the even pairs (0,1), (2,3), ... and the odd pairs (1,2), (3,4), ...
(deterministic even/odd, Okabe et al., CPL 335, 435 (2001); it gives faster round trips than
random pair choices: Syed et al., JRSSB 84, 321 (2022)).  A swap of the configurations x_i, x_j
of states i, j is accepted with the Metropolis probability

    P = min(1, exp(-Delta)),   Delta = u_i(x_j) + u_j(x_i) - u_i(x_i) - u_j(x_j),

with u_k(x) the reduced (dimensionless) energy of configuration x in thermodynamic state k.  For
temperature exchange u_k(x) = beta_k [U(x) + P V(x)] (P V at constant pressure only), so that
Delta = (beta_i - beta_j) [U(x_j) - U(x_i) + P (V_j - V_i)] (Sugita & Okamoto, CPL 314, 141
(1999); Okabe et al. 2001 for NPT).  The momenta of an exchanged configuration are rescaled by
sqrt(T_new / T_old), which removes the kinetic energies from Delta; the thermostat auxiliaries
(GLE: mass-scaled momenta with variance kT) are rescaled in the same way.

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
                  (Simulation._advance: overflow handling, rebuilds of the neighbour lists when an
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
map, statistics and the exchange random state; `ReplicaExchange.load`).  Frames and log lines at
a step are written after that step's exchange, so the replica of a frame is the one in the
exchange-log line of that step or the last line before it (`read_exchange_log`).

Units: K, kJ/mol, nm, ps."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

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

LEGACY_FORMAT = "pgm_jax remd 1"  # the "format" entry of legacy pickle checkpoints


# ----------------------------------------------------------------------------- exchange logic
def geometric_ladder(t_min: float, t_max: float, n: int) -> np.ndarray:
    """n temperatures (K) from t_min to t_max in constant ratio, T_k = t_min (t_max/t_min)^(k/(n-1)).
    When the heat capacity changes little over the range, neighbouring energy distributions then
    overlap equally, so every pair has about the same acceptance."""
    if int(n) < 2 or not (0.0 < t_min < t_max):
        raise ValueError("need n >= 2 and 0 < t_min < t_max")
    return float(t_min) * (float(t_max) / float(t_min)) ** (np.arange(int(n)) / (int(n) - 1.0))


def exchange_pairs(n: int, parity: int) -> list:
    """Neighbour pairs tried together: (0,1), (2,3), ... for even parity, (1,2), (3,4), ... for odd."""
    return [(i, i + 1) for i in range(int(parity) % 2, int(n) - 1, 2)]


def metropolis(u, pairs, uniforms):
    """Accept or reject configuration swaps between disjoint pairs of states.

    u[i, j] = u_i(x_j): reduced energy, in state i, of the configuration now in state j;
    uniforms: one U(0, 1) number per pair.  Returns (accepted: bool per pair, src): after the swaps
    state i holds the configuration previously in state src[i]."""
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


def temperature_reduced_energies(temperatures, U, V=None, pressure=None) -> np.ndarray:
    """u[i, j] = beta_i (U_j + P V_j) for a temperature ladder: U (kJ/mol) and V (nm^3) of the
    configuration in each slot, P in kJ/mol/nm^3 (None at constant volume)."""
    beta = 1.0 / (KB * np.asarray(temperatures, float))
    H = np.asarray(U, float)
    if pressure is not None:
        H = H + float(pressure) * np.asarray(V, float)
    return beta[:, None] * H[None, :]


class ExchangeStatistics:
    """Attempts and acceptances per pair of temperatures, the replica at each temperature, and
    round trips: a replica completes one when it has gone from the lowest temperature to the
    highest and back; `transits` counts one-way trips between the two ends."""

    def __init__(self, n: int):
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

    def _ends(self):
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

    def record(self, pairs, accepted, src):
        for (i, j), a in zip(pairs, accepted):
            self.attempts[i, j] += 1
            self.attempts[j, i] += 1
            self.accepts[i, j] += int(a)
            self.accepts[j, i] += int(a)
        self.replica = self.replica[np.asarray(src)]
        self.n_exchanges += 1
        self._ends()

    def acceptance(self) -> np.ndarray:
        """(n, n) accepted / attempted swaps (nan where never attempted)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.attempts > 0, self.accepts / np.maximum(self.attempts, 1), np.nan)

    def neighbour_acceptance(self) -> np.ndarray:
        a = self.acceptance()
        return np.array([a[i, i + 1] for i in range(self.n - 1)])

    def to_dict(self) -> dict:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> ExchangeStatistics:
        out = cls.__new__(cls)
        for k, v in d.items():
            setattr(out, k, np.array(v) if isinstance(v, np.ndarray) else v)
        return out


def read_exchange_log(path: str):
    """prefix_remd.log -> (steps (E,), replica at each temperature from that step on (E, R),
    outcome per neighbour pair (E, R-1) chars: '+' accepted, '.' rejected, '-' not tried)."""
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
def _nocount(st):
    """The state without its induction step counter (kept unbatched in stacked states)."""
    return st.set(induction=st.induction.set(count=None))


def _stack(states):
    counts = {int(s.induction.count) for s in states}
    if len(counts) != 1:
        raise ValueError(f"batched replicas need equal induction step counters, got {sorted(counts)}")
    S = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[_nocount(s) for s in states])
    return S.set(induction=S.induction.set(count=states[0].induction.count))


def _take(S, index):
    """Slot `index` (an int, or an index array for a gather) of a stacked state."""
    T = jax.tree_util.tree_map(lambda x: x[index], _nocount(S))
    return T.set(induction=T.induction.set(count=S.induction.count))


def _axes(S):
    """vmap axes of a stacked state: 0 for every leaf, None (unbatched) for the induction step
    counter, so the predictor's fused / unfused lax.cond stays a branch under vmap."""
    return jax.tree_util.tree_map(lambda _: 0, _nocount(S))


def _broadcast(tree, n: int):
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), tree)


# ----------------------------------------------------------------------------- MD replicas
class MDReplicas:
    """The replicas of one Simulation or FlexibleSimulation at the temperatures of a ladder: the
    engine of ReplicaExchange.  `sim` provides the system, force field, integrator and neighbour
    lists (its `state` is used as scratch space); its thermostat is used at every temperature.
    The replicas start from the current configuration of `sim` with momenta (and thermostat
    auxiliaries) drawn at their own temperatures from independent random streams (`seed`).

    The interface ReplicaExchange uses (a test engine can provide the same): temperatures, n, dt,
    pressure (kJ/mol/nm^3 or None), time_ps, advance(n), potentials(), volumes(), permute(src),
    observables(k), frames(), write_restarts(prefix), state_dict(), load_state_dict(d)."""

    def __init__(self, sim, temperatures, batched: bool = True, seed: int = 0):
        integ = sim.integ
        if integ.thermostat is None:
            raise ValueError("replica exchange needs a thermostat (ensemble nvt or npt)")
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
    def _build(self):
        """vmapped entry points for the current static sizes and neighbour-list layout."""
        integ, sim = self.integ, self.sim
        ax = _axes(self.S)
        self._run = jax.jit(jax.vmap(integ._run, in_axes=(ax, None), out_axes=ax))
        self._forces = jax.jit(jax.vmap(lambda s: integ._state_forces(s, True), in_axes=(ax,), out_axes=ax))
        self._exchange = jax.jit(jax.vmap(self._exchange_one, in_axes=(ax, ax, 0), out_axes=ax))
        self._wrap = jax.jit(jax.vmap(sim.rigid.wrap))
        self._positions = jax.jit(jax.vmap(sim.rigid.positions))
        flex = getattr(sim, "flex", None)
        self._extent = jax.jit(lambda P: jnp.max(jax.vmap(flex.extent)(P))) if flex is not None else None

    def _exchange_one(self, dst, src, s):
        """dst's slot with src's configuration: momenta and thermostat auxiliaries scaled by
        s = sqrt(T_dst / T_src); the energy change of the slot is booked as heat."""

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
    def _slot(self, S, k: int):
        """The MDState of slot k of the stacked state S."""
        return _take(S, k)

    def state(self, k: int):
        """MDState of temperature slot k."""
        return self._slot(self.S, k) if self.batched else self.states[k]

    def state_template(self):
        """A state with the structure of the replica states in a checkpoint (slot 0 without its
        neighbour list): the template driver.read_checkpoint rebuilds them on."""
        return self.state(0).set(nbr=None)

    def potentials(self) -> np.ndarray:
        """Potential energy (kJ/mol) of the configuration at each temperature."""
        if self.batched:
            return np.asarray(self.S.epot, float)
        return np.array([float(s.epot) for s in self.states])

    def volumes(self) -> np.ndarray:
        boxes = np.asarray(self.S.box) if self.batched else np.array([np.asarray(s.box) for s in self.states])
        return np.abs(np.linalg.det(boxes))

    def _on(self, k: int):
        """Point the engine's simulation object at slot k (for its observables and file writers)."""
        self.sim.state, self.sim.time_ps = self.state(k), self.time_ps
        return self.sim

    def observables(self, k: int) -> dict:
        return self._on(k).observables()

    def positions_nm(self, k: int) -> np.ndarray:
        """Atom positions (N, 3) of temperature slot k, nm."""
        return self._on(k).positions_nm()

    def frames(self):
        """Positions (R, N, 3) and boxes (R, 3, 3) of every slot, A."""
        if self.batched:
            return np.asarray(self._positions(self.S.dyn.position)) * 10.0, np.asarray(self.S.box) * 10.0
        X = [self.positions_nm(k) for k in range(self.n)]
        return np.array(X) * 10.0, np.array([np.asarray(s.box) for s in self.states]) * 10.0

    def write_restarts(self, prefix: str):
        """Amber NetCDF restart of every slot: prefix_Tkk.rst7."""
        for k in range(self.n):
            sim = self._on(k)
            write_restart(
                f"{prefix}_T{k:02d}.rst7",
                sim.positions_nm() * 10.0,
                sim.velocities_nm_ps() * 10.0,
                np.asarray(sim.state.box) * 10.0,
                self.time_ps,
                title=f"pgm_jax REMD T={self.temperatures[k]:g} K",
            )

    # ------------------------------------------------------------------ dynamics
    def advance(self, n: int):
        """Every replica n steps."""
        if self.batched:
            self._advance_batched(int(n))
        else:
            self._advance_sequential(int(n))
        self.time_ps += int(n) * self.dt

    def _sizes(self):
        """Static sizes shared by the replicas: row capacities (ff.capacity) and molecule-list width."""
        return self.sim.ff.capacity, getattr(self.sim.nb, "cap", None)

    def _fit(self, sizes, grow_rows=None, grow_cap=None):
        """Static sizes that fit every entry of `sizes` (the largest of each part), at least 8 rows /
        4 molecules above `grow_*` (sizes that overflowed); re-jit if anything changed."""
        sim, now = self.sim, self._sizes()
        sim.ff.fit_rows([s[0] for s in sizes])
        if grow_rows is not None:
            sim.ff.grow_rows(grow_rows)
        caps = [s[1] for s in sizes if s[1] is not None]
        if caps and getattr(sim.nb, "cap", None) is not None:
            sim.nb.cap = max(caps) if grow_cap is None else max(max(caps), grow_cap + 4)
        if self._sizes() != now:
            self.integ.compile()

    def _advance_sequential(self, n: int):
        sim = self.sim
        for k in range(self.n):
            before = self._sizes()
            sim.state, sim.time_ps = self.states[k], self.time_ps
            sim._advance(n)
            self.states[k] = sim.state
            # a replica may re-size the shared static sizes from its own configuration: never
            # below what another replica needed (no resize ping-pong)
            self._fit([before, self._sizes()])

    def _nb_failed(self, S) -> bool:
        """Any replica's neighbour list failed (the engine's test, on each replica's error code)."""
        codes = np.asarray(S.nbr.error.code).reshape(-1)
        return any(self.sim.nb.failed(SimpleNamespace(error=SimpleNamespace(code=c))) for c in codes)

    def _run_block(self, start, n: int):
        """One compiled block of n steps of every replica from the stacked state `start`."""
        new = self._run(start, n)
        jax.block_until_ready(new.epot)
        return new

    def _advance_batched(self, n: int) -> None:
        """n steps of every replica as one vmapped block.

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

        def resize(start, nb_bad: bool, row_bad: bool):
            """_resize and a log line."""
            step0 = int(np.asarray(start.step)[0])
            start = self._resize(start, nb_bad, row_bad)
            sim._print(
                f"# {'neighbour list' if nb_bad else 'row capacity'} overflow in steps {step0}-{step0 + n} "
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

    def _resize(self, start, nb_bad: bool, row_bad: bool):
        """Static sizes that fit every replica (the largest of each), one neighbour-list layout for
        all (the largest list, filled with each replica's neighbours), then forces at the start."""
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

    def permute(self, src):
        """Slot k receives the configuration of slot src[k] (momenta and auxiliaries rescaled)."""
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
        """The replicas' content for a checkpoint.

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

    def load_state_dict(self, d: dict):
        """Replica states from a checkpoint (either mode); neighbour lists are rebuilt and the static
        sizes fitted to the loaded configurations."""
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
    """Temperature replica exchange.

        sim = FlexibleSimulation(..., ensemble="nvt", thermostat="bussi", dt=0.002, constraints="h-bonds", hmr=3.024)
        sim.minimize(300); sim.run(25000)                       # equilibrate at the lowest temperature
        rex = ReplicaExchange(sim, geometric_ladder(300.0, 400.0, 8), exchange_every=500)
        rex.run(500000, report=5000, traj=500, restart=50000, prefix="ala3")

    `sim`: a Simulation / FlexibleSimulation (replicas built by MDReplicas with `batched` and
    `seed`), or a ready replica engine with the MDReplicas interface (then `temperatures` is not
    given).  Exchanges are attempted every `exchange_every` steps."""

    def __init__(
        self, sim, temperatures=None, exchange_every: int = 500, batched: bool = True, seed: int = 0, log=sys.stdout
    ):
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
        self._print(
            f"# replica exchange: {self.n} replicas ({mode}), T = "
            f"{' '.join(f'{t:.2f}' for t in self.T)} K, exchanges every {self.every} steps "
            f"({self.every * self.replicas.dt:g} ps)"
            f"{'' if self.replicas.pressure is None else ', NPT (P V in the criterion)'}"
        )

    def _print(self, s):
        if self.log is not None:
            print(s, file=self.log, flush=True)

    # ------------------------------------------------------------------ exchanges
    def reduced_energies(self) -> np.ndarray:
        """u[i, j] = u_i(x_j) for the configurations now at each temperature (temperature ladder:
        beta_i (U_j + P V_j)).  Hamiltonian exchange overrides this with energies of each
        configuration evaluated in the neighbouring states."""
        rep = self.replicas
        V = rep.volumes() if rep.pressure is not None else None
        return temperature_reduced_energies(self.T, rep.potentials(), V, rep.pressure)

    def exchange(self):
        """One exchange attempt (even or odd pairs, alternating); returns (pairs, accepted)."""
        pairs = exchange_pairs(self.n, self.stats.n_exchanges)
        uniforms = self.rng.random(len(pairs))
        acc, src = metropolis(self.reduced_energies(), pairs, uniforms)
        self.replicas.permute(src)
        self.stats.record(pairs, acc, src)
        return pairs, acc

    def _outcome(self, pairs, acc) -> str:
        out = ["-"] * (self.n - 1)
        for (i, _), a in zip(pairs, acc):
            out[i] = "+" if a else "."
        return "".join(out)

    # ------------------------------------------------------------------ running
    def run(
        self,
        nsteps: int,
        report: int = 1000,
        traj: int = 0,
        restart: int = 0,
        prefix: str | None = "remd",
        append: bool = False,
    ):
        """nsteps steps of every replica, exchanges every `exchange_every` steps; per-temperature
        logs every `report` steps, trajectories every `traj`, checkpoints every `restart` (0: off;
        prefix None: no files)."""
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
                self._print(
                    f"# step {self.step} ({rep.time_ps:.1f} ps): acceptance "
                    f"{' '.join('  -  ' if np.isnan(a) else f'{a:.3f}' for a in acc)}, round trips "
                    f"{int(self.stats.round_trips.sum())}, {speed:.1f} ns/day per replica"
                )
            if trajs is not None and self.step % traj == 0:
                X, B = rep.frames()
                for k in range(n):
                    trajs[k].write(rep.time_ps, X[k], B[k])
            if files and restart and self.step % restart == 0:
                self.save(prefix)
        speed = clock.ns_per_day(self.step)
        for t in logs or []:
            t.close()
        if xlog is not None:
            xlog.close()
        if files and restart:
            self.save(prefix)
        summary = self.summary(ns_per_day=speed)
        if files:
            with open(f"{prefix}_remd.json", "w") as fh:
                json.dump(summary, fh, indent=1)
        return summary

    def summary(self, ns_per_day: float | None = None) -> dict:
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
    def save(self, prefix: str) -> None:
        """Write a checkpoint prefix.remd.chk (and prefix_Tkk.rst7 Amber restarts when the replica
        engine writes them).

        Parameters
        ----------
        prefix : str
            Path prefix of the files.

        Notes
        -----
        The checkpoint (driver.write_checkpoint, kind "remd") holds every replica state
        (`state_dict` of the engine), the replica map and statistics, the step and the state of
        the exchange random-number generator; `load` continues the run bitwise on the CPU.
        """
        content = {
            "temperatures": self.T,
            "exchange_every": self.every,
            "step": self.step,
            "rng": self.rng.bit_generator.state,
            "stats": self.stats.to_dict(),
            "replicas": self.replicas.state_dict(),
        }
        write_checkpoint(prefix + ".remd.chk", "remd", content)
        if hasattr(self.replicas, "write_restarts"):
            self.replicas.write_restarts(prefix)

    def load(self, path: str) -> None:
        """Continue from a checkpoint written by `save`, or from a legacy pickle ``.remd.chk`` of
        pgm_jax up to commit e72c57c.

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
