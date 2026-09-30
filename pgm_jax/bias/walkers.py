"""Run several biased simulations of one system in one compiled program (jax.vmap).

Contents: `Walkers`, a subclass of the replica engine `MDReplicas` of md/remd.py (stacked
states, one layout of the neighbour lists, overflows resized for all), for either MD engine
(Simulation, FlexibleSimulation).  Small systems leave most of a GPU idle, so W walkers cost
little more than one.

    from pgm_jax.bias.walkers import Walkers
    sim = FlexibleSimulation(..., bias=BiasSet([MetaD([phi, psi], ...)], colvar=100))
    sim.minimize(200)
    w = Walkers(sim, 8, shared=True)              # multiple-walker metadynamics: one bias, 8 depositors
    w = Walkers(sim, 8)                           # 8 independent runs (each its own bias), e.g. for error bars
    w = Walkers(sim, 24, bias_states=[...])       # e.g. umbrella windows: one Harmonic centre per walker
    w.run(500000, prefix="ala2", report_every=5000, checkpoint_every=50000)

Two modes:

    shared=False  every walker carries its own bias state (BiasState batched with the MD state);
                  the MD step and the bias hook of md/integrate.py are vmapped.
    shared=True   one bias state for all walkers (multiple walkers [1]_): after each step every
                  walker's COLVAR row is recorded and, when an update is due, the walkers deposit
                  one after the other (walker 0 first) into the shared bias; then the forces of
                  all walkers are corrected to the new bias at their positions (the change of V
                  booked as heat per walker, and its sum as bias work).

Walkers start from the simulation's current configuration with momenta drawn from independent
random streams.  They run NVT only, without multiple time stepping.  Outputs: prefix_wNN.colvar
per walker; hills in prefix.hills (shared) or prefix_wNN.hills; prefix_walkers.log (per report:
mean temperature, epot and bias of the walkers); checkpoint prefix.walkers.chk (all states;
`load_checkpoint`).

Units: steps, ps, K, kJ/mol (as the MD engines).

References
----------
.. [1] P. Raiteri, A. Laio, F. L. Gervasio, C. Micheletti, M. Parrinello, J. Phys. Chem. B 110,
   3533 (2006).

See also docs/enhanced_sampling.md; md/remd.py (MDReplicas), bias/core.py (BiasSet).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TextIO

import jax
import jax.numpy as jnp
import numpy as np

from ..md.driver import LogTable, Stopwatch, block_length, device_tree, host_tree, read_checkpoint, write_checkpoint
from ..md.engine import OPTIONAL_STATE
from ..md.remd import MDReplicas, _nocount, _stack
from .io import BiasOutput

if TYPE_CHECKING:
    from ..md.flexible import FlexibleSimulation
    from ..md.integrate import MDState
    from ..md.simulation import Simulation
    from .core import BiasState


def _unbatched_bias(S: MDState) -> MDState:
    """Return the vmap axes of a stacked state with a shared bias.

    0 for every leaf except the induction step counter and the bias state (None: unbatched), so the
    shared bias is broadcast to every walker.
    """
    ax = jax.tree_util.tree_map(lambda _: 0, _nocount(S).set(bias=None))
    return ax.set(induction=ax.induction.set(count=None), bias=None)


class Walkers(MDReplicas):
    """Several biased copies of one simulation advanced together (see the module docstring).

    Attributes
    ----------
    sim : Simulation or FlexibleSimulation
        The simulation the walkers were created from (system, force field, integrator).
    integ : object
        `sim.integ`, the integrator whose step and bias hook are vmapped.
    shared : bool
        One bias state for all walkers.
    n : int
        Number of walkers W.
    S : MDState
        The stacked walker states (leading axis W; the bias state unbatched if shared).
    log : text stream or None
        Echo of the log table rows.
    """

    def __init__(
        self,
        sim: Simulation | FlexibleSimulation,
        walkers: int,
        shared: bool = False,
        bias_states: Sequence[BiasState] | None = None,
        seed: int = 0,
        log: TextIO | None = None,
    ) -> None:
        """Set up W walkers from `sim`'s current configuration.

        Parameters
        ----------
        sim : Simulation or FlexibleSimulation
            Created with bias=... and a thermostat, NVT, without multiple time stepping.
        walkers : int
            Number of walkers W.
        shared : bool
            One bias state deposited into by every walker (multiple-walker metadynamics / OPES)
            instead of one per walker.
        bias_states : sequence of BiasState, optional
            One bias state per walker (e.g. umbrella centres; not with shared=True); None: every
            walker starts from the simulation's bias state.
        seed : int
            Seed of the walkers' momenta and thermostat streams.
        log : text stream, optional
            Receives the rows of the log table of `run` too; None: file only.

        Raises
        ------
        ValueError
            A simulation without a bias or thermostat, NPT, MTS, bias_states with shared=True, or a
            number of bias states other than W.
        """
        self.log = log
        n = walkers
        integ = sim.integ
        if integ.bias is None:
            raise ValueError("the simulation has no bias (bias=...)")
        if integ.thermostat is None:
            raise ValueError("walkers need a thermostat (create the simulation with thermostat=...)")
        if sim.ensemble == "npt":
            raise ValueError("batched walkers run NVT only")
        if getattr(integ, "mts", None) is not None:
            raise ValueError("walkers with multiple time stepping are not supported")
        if shared and bias_states is not None:
            raise ValueError("a shared bias has one state (set it on the simulation)")
        self.sim, self.integ, self.batched, self.shared = sim, integ, True, bool(shared)
        self.n = int(n)
        self.temperatures = np.full(self.n, float(sim.T0))
        self.dt, self.pressure, self.time_ps = float(sim.dt), None, 0.0
        base = sim.state
        self._template = base.nbr
        if bias_states is not None and len(bias_states) != self.n:
            raise ValueError("one bias state per walker")
        states = []
        for k, key in enumerate(jax.random.split(jax.random.PRNGKey(int(seed)), self.n)):
            b = base.bias if bias_states is None else bias_states[k]
            st = integ.init(base.dyn.position, base.box, key, bias=b)
            states.append(st.set(nbr=base.nbr))
        if self.shared:
            self.S = _stack([s.set(bias=None) for s in states]).set(bias=base.bias)
        else:
            self.S = _stack(states)
        self._build()
        self.S = self._forces(self.S).set(induction=self.S.induction)
        self._rows = [[] for _ in range(self.n)]

    # ------------------------------------------------------------------ compiled pieces
    def _axes(self) -> MDState:
        """Return the vmap axes of the stacked state (the shared bias unbatched when shared=True)."""
        from ..md.remd import _axes

        return _unbatched_bias(self.S) if self.shared else _axes(self.S)

    def _build(self) -> None:
        """Compile the vmapped pieces: the block runner, forces, wrapping, positions, extent, bias energies.

        Replaces `MDReplicas._build`: `_run(S, n)` is `_run_shared` or `_run_independent`, jitted with
        the step count `n` static (a new block length recompiles).
        """
        integ, sim = self.integ, self.sim
        ax = self._axes()
        if self.shared:
            self._run = jax.jit(self._run_shared, static_argnums=1)
        else:
            self._run = jax.jit(self._run_independent, static_argnums=1)
        self._forces = jax.jit(jax.vmap(lambda s: integ._state_forces(s, True), in_axes=(ax,), out_axes=ax))
        self._wrap = jax.jit(jax.vmap(sim.rigid.wrap))
        self._positions = jax.jit(jax.vmap(sim.rigid.positions))
        flex = getattr(sim, "flex", None)
        self._extent = jax.jit(lambda P: jnp.max(jax.vmap(flex.extent)(P))) if flex is not None else None
        bias = integ.bias
        if self.shared:
            self._energies = jax.jit(
                jax.vmap(lambda x, box, bs: bias.energies(bs, integ._bias_atoms(x), box), in_axes=(0, 0, None))
            )
        else:
            self._energies = jax.jit(jax.vmap(lambda x, box, bs: bias.energies(bs, integ._bias_atoms(x), box)))

    def _run_independent(self, S: MDState, n: int) -> MDState:
        """Run `n` steps of walkers that each have their own bias (traced; jitted by `_build`).

        The step and the bias hook (`integ._bias_post`) are vmapped over the walkers; the loop over
        steps is outside vmap (the walkers share the step counter), so the steps between bias events
        carry no conditional (md/integrate.strided_loop with the bias set's stride).

        Parameters
        ----------
        S : MDState
            Stacked walker states (leading axis W).
        n : int
            Number of steps (static).

        Returns
        -------
        MDState
            The stacked states after n steps.
        """
        from ..md.integrate import strided_loop

        integ = self.integ
        ax = self._axes()
        W = self.n
        # per-walker solver diagnostics, reset for the block
        S = S.set(max_iters=jnp.zeros(W, jnp.int32), resid=jnp.zeros(W), overflow=jnp.zeros(W, bool))
        step = jax.vmap(
            lambda s: integ._book_field(s, integ._step(s)), in_axes=(ax,), out_axes=ax
        )  # E(t) work (efield.py)
        if integ.bias.stride == 0:
            return jax.lax.fori_loop(0, n, lambda _, s: step(s), S)
        post = jax.vmap(integ._bias_post, in_axes=(ax,), out_axes=ax)
        return strided_loop(S, n, step, post, integ.bias.stride)

    def _run_shared(self, S: MDState, n: int) -> MDState:
        """Run `n` steps of walkers that share one bias state (traced; jitted by `_build`).

        After every (vmapped) step, `post` records one COLVAR row per walker and, when an update is
        due, deposits the walkers' updates in order into the shared bias and corrects every walker's
        forces, epot and heat to the new bias at its positions.

        Parameters
        ----------
        S : MDState
            Stacked walker states with an unbatched bias state.
        n : int
            Number of steps (static).

        Returns
        -------
        MDState
            The states after n steps.

        Notes
        -----
        Unlike `_run_independent`, the bias hook runs after every step (one lax.cond per walker for
        the COLVAR row and one for the update), not in strides.
        """
        integ, bias = self.integ, self.integ.bias
        ax = self._axes()
        W = self.n
        z = jnp.zeros(W)
        S = S.set(max_iters=jnp.zeros(W, jnp.int32), resid=z, overflow=jnp.zeros(W, bool))
        step = jax.vmap(
            lambda s: integ._book_field(s, integ._step(s)), in_axes=(ax,), out_axes=ax
        )  # E(t) work (efield.py)
        # V and dV/dpos of every walker under one (shared) bias state
        grad = jax.vmap(jax.value_and_grad(bias.energy, argnums=1), in_axes=(None, 0, 0))
        to_engine = jax.vmap(integ._map_atom_forces)
        atoms = jax.vmap(integ._bias_atoms)

        def post(S: MDState) -> MDState:
            """Record the COLVAR rows of all walkers, then apply the shared-bias update if one is due."""
            pos = atoms(S.dyn.position)
            bs = S.bias
            t = S.step[0]  # the walkers share the step counter
            for w in range(W):
                bs = bias.record(bs, pos[w], S.box[w], t)
            S = S.set(bias=bs)
            if not bias.dynamic:
                return S

            def dep(S: MDState) -> MDState:
                """Deposit every walker's update in turn, then move all walkers to the new bias.

                The forces, epot and heat of each walker change by the bias change at its positions; the sum
                of the energy changes is added to the bias work.
                """
                old = S.bias
                new = old
                for w in range(W):
                    new = bias.deposit(new, pos[w], S.box[w], t)
                e0, g0 = grad(old, pos, S.box)
                e1, g1 = grad(new, pos, S.box)
                dF = to_engine(S.dyn.position, S.box, g0 - g1)  # forces of the new bias at the same positions
                de = e1 - e0  # (W,) change of V at fixed x: booked as heat and bias work
                F = jax.tree_util.tree_map(jnp.add, S.dyn.force, dF)
                return S.set(
                    bias=new._replace(work=new.work + jnp.sum(de)),
                    dyn=S.dyn.set(force=F),
                    epot=S.epot + de,
                    heat=S.heat + de,
                )

            return jax.lax.cond(bias.due(t), dep, lambda s: s, S)

        return jax.lax.fori_loop(0, n, lambda _, s: post(step(s)), S)

    # ------------------------------------------------------------------ bias bookkeeping (host)
    def _reserve(self, n: int) -> None:
        """Grow the bias buffers for `n` steps of every walker (host; n * W updates for a shared bias)."""
        bias = self.integ.bias
        if self.shared:
            self.S = self.S.set(bias=bias.reserve(self.S.bias, n * self.n))
        else:
            per = bias.reserve_many([jax.tree_util.tree_map(lambda a: a[w], self.S.bias) for w in range(self.n)], n)
            self.S = self.S.set(bias=jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per))

    def _drain(self) -> None:
        """Move the COLVAR rows of the device buffer(s) to the per-walker host lists and empty the buffer(s)."""
        bias = self.integ.bias
        if self.shared:
            rows, bs = bias.drain(self.S.bias)
            for w in range(self.n):
                self._rows[w].append(rows[w :: self.n])  # rows are recorded walker by walker at each COLVAR step
            self.S = self.S.set(bias=bs)
        else:
            log, nlog = np.asarray(self.S.bias.log), np.asarray(self.S.bias.nlog)
            for w in range(self.n):
                self._rows[w].append(log[w, : nlog[w]])
            self.S = self.S.set(bias=self.S.bias._replace(nlog=jnp.zeros(self.n, jnp.int32)))

    def advance(self, n: int) -> None:
        """Advance every walker `n` steps: reserve the bias buffers, run (`MDReplicas.advance`), collect the rows."""
        self._reserve(int(n))
        super().advance(int(n))
        self._drain()

    def bias_state(self, w: int = 0) -> BiasState:
        """Return the bias state of walker `w` (the shared one for shared=True)."""
        if self.shared:
            return self.S.bias
        return jax.tree_util.tree_map(lambda a: a[w], self.S.bias)

    def bias_energies(self) -> np.ndarray:
        """Return the bias energies of every walker, np.ndarray (W, n_bias) [kJ/mol]."""
        return np.asarray(self._energies(self.S.dyn.position, self.S.box, self.S.bias))

    def rows(self, w: int) -> np.ndarray:
        """Return the COLVAR rows of walker `w` collected since the last call, np.ndarray (n, ncol)."""
        r = self._rows[w]
        self._rows[w] = []
        return np.concatenate(r) if r else np.zeros((0, self.integ.bias.ncol))

    def _slot(self, S: MDState, k: int) -> MDState:
        """Return the MDState of walker `k` of the stacked state `S`, with its bias state (the shared one if shared)."""
        T = jax.tree_util.tree_map(lambda x: x[k], _nocount(S.set(bias=None)))
        T = T.set(induction=T.induction.set(count=S.induction.count))
        return T.set(bias=S.bias if self.shared else jax.tree_util.tree_map(lambda a: a[k], S.bias))

    def state(self, k: int) -> MDState:
        """Return the MDState of walker `k` (with its bias state)."""
        return self._slot(self.S, k)

    # ------------------------------------------------------------------ driver
    def run(
        self,
        nsteps: int,
        *,
        prefix: str = "walkers",
        report_every: int = 1000,
        checkpoint_every: int = 0,
        append: bool = False,
    ) -> None:
        """Advance every walker `nsteps` steps with output files.

        Parameters
        ----------
        nsteps : int
            Steps [steps].
        prefix : str
            Path prefix of the files: COLVAR rows prefix_wNN.colvar per walker, hills in
            prefix.hills (shared) or prefix_wNN.hills.
        report_every : int
            Steps between rows of prefix_walkers.log (mean temperature [K] of up to 64 walkers, mean
            potential and bias energies and total bias work [kJ/mol], aggregate ns/day); 0: none.
        checkpoint_every : int
            Steps between checkpoints prefix.walkers.chk (0: none; with checkpoints also at the end).
        append : bool
            Append to existing files (a continuation).
        """
        report, restart = report_every, checkpoint_every
        bias, sim = self.integ.bias, self.sim
        block = block_length(nsteps, report, restart)
        outs = []
        for w in range(self.n):
            st = self.bias_state(w)
            outs.append(
                BiasOutput(bias, f"{prefix}_w{w:02d}", self.dt, sim.T0, append=append, state=st, hills=not self.shared)
            )
        shared_out = None
        if self.shared:
            shared_out = BiasOutput(bias, prefix, self.dt, sim.T0, append=append, state=self.S.bias, colvar=False)
        table = LogTable(prefix + "_walkers.log", append=append, echo=self.log)
        clock = Stopwatch(0, self.dt)
        done = 0
        for w in range(self.n):
            self.rows(w)
        while done < nsteps:
            k = min(block, nsteps - done)
            self.advance(k)
            done += k
            for w in range(self.n):
                outs[w].write(self.rows(w), self.bias_state(w))
            if shared_out is not None:
                shared_out.write([], self.S.bias)
            step = int(np.asarray(self.S.step)[0])
            if report and step % report == 0:
                T = [float(self.integ.temperature(self.state(w))) for w in range(min(self.n, 64))]
                eb = self.bias_energies().sum(1)
                table.write(
                    {
                        "step": step,
                        "time_ps": step * self.dt,
                        "temp_mean": float(np.mean(T)),
                        "epot_mean": float(np.mean(np.asarray(self.S.epot))),
                        "ebias_mean": float(eb.mean()),
                        "bias_work": float(np.sum(np.asarray(self.S.bias.work))),
                        "ns_day_agg": clock.ns_per_day(done) * self.n,
                    }
                )
            if restart and step % restart == 0:
                self.save_checkpoint(prefix + ".walkers.chk")
        table.close()
        if restart:
            self.save_checkpoint(prefix + ".walkers.chk")

    def state_dict(self) -> dict:
        """Return the walkers' content for a checkpoint.

        Returns
        -------
        dict
            n, shared, time_ps and the state of every walker (host arrays with its bias state, no
            neighbour lists).
        """
        return {
            "n": self.n,
            "shared": self.shared,
            "time_ps": self.time_ps,
            "states": [host_tree(self.state(k).set(nbr=None)) for k in range(self.n)],
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore the walker states from `state_dict` content; neighbour lists are rebuilt in one layout.

        Parameters
        ----------
        d : dict
            Content of a checkpoint of the same walker set.

        Raises
        ------
        ValueError
            A checkpoint of another number of walkers or another bias mode.
        """
        if int(d["n"]) != self.n or bool(d["shared"]) != self.shared:
            raise ValueError("checkpoint of a different walker set")
        sim = self.sim
        states = [device_tree(s) for s in d["states"]]
        lists = []
        for st in states:
            lists.append(sim._size_lists(st.dyn.position, st.box))
        self.integ.compile()
        self._template = max(lists, key=lambda b: b.max_occupancy)
        if self.shared:
            S = _stack([s.set(bias=None, nbr=self._template) for s in states]).set(bias=states[0].bias)
        else:
            S = _stack([s.set(nbr=self._template) for s in states])
        self.S = S
        self._build()
        self.S = S.set(nbr=self._forces(S).nbr)
        self.time_ps = float(d["time_ps"])

    def save_checkpoint(self, path: str) -> None:
        """Write a checkpoint of every walker (driver.write_checkpoint, kind "walkers").

        Parameters
        ----------
        path : str
            The file (prefix.walkers.chk in `run`).
        """
        write_checkpoint(path, "walkers", self.state_dict())

    def load_checkpoint(self, path: str) -> None:
        """Continue from a checkpoint written by `save_checkpoint`.

        Legacy pickle ``.walkers.chk`` files of pgm_jax up to commit e72c57c are read too (same
        system, walker count and mode).

        Parameters
        ----------
        path : str
            The checkpoint file.

        Raises
        ------
        ValueError
            Another kind of checkpoint or another walker set.
        """
        self.load_state_dict(read_checkpoint(path, "walkers", self.state_template(), OPTIONAL_STATE))
