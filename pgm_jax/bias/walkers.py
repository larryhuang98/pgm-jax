"""Several biased simulations of one system in one compiled program (jax.vmap), for either MD
engine (Simulation, FlexibleSimulation).

    from pgm_jax.bias.walkers import Walkers
    sim = FlexibleSimulation(..., bias=BiasSet([MetaD([phi, psi], ...)], colvar=100))
    sim.minimize(200)
    w = Walkers(sim, 8, shared=True)              # multiple-walker metadynamics: one bias, 8 depositors
    w = Walkers(sim, 8)                           # 8 independent runs (each its own bias), e.g. for error bars
    w = Walkers(sim, 24, bias_states=[...])       # e.g. umbrella windows: one Harmonic centre per walker
    w.run(500000, report=5000, restart=50000, prefix="ala2")

Small systems leave most of a GPU idle, so W walkers cost little more than one (the replica
engine of md/remd.py: stacked states, one layout of the neighbour lists, overflows resized for all).

shared=False  every walker carries its own bias state (BiasState batched with the MD state).
shared=True   one bias state for all walkers (multiple walkers, Raiteri et al., JPCB 110, 3533
              (2006)): after each step every walker's COLVAR row is recorded and, when an update is
              due, the walkers deposit one after the other (walker 0 first) into the shared bias;
              then the forces of all walkers are corrected to the new bias at their positions (the
              change of V booked as heat and bias work, per walker).
Walkers start from the simulation's current configuration with momenta drawn from independent
random streams.  Outputs: prefix_wNN.colvar per walker; hills in prefix.hills (shared) or
prefix_wNN.hills; prefix_walkers.log (per report: mean temperature, epot and bias of the walkers);
checkpoint prefix.walkers.chk (all states; `load`)."""
from __future__ import annotations

import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..md.remd import MDReplicas, _broadcast, _nocount, _stack, _take
from .io import BiasOutput


def _unbatched_bias(S):
    """vmap axes: 0 everywhere except the induction step counter and the (shared) bias state."""
    ax = jax.tree_util.tree_map(lambda _: 0, _nocount(S).set(bias=None))
    return ax.set(induction=ax.induction.set(count=None), bias=None)


class Walkers(MDReplicas):
    def __init__(self, sim, n: int, shared: bool = False, bias_states=None, seed: int = 0):
        integ = sim.integ
        if integ.bias is None:
            raise ValueError("the simulation has no bias (bias=...)")
        if integ.thermostat is None:
            raise ValueError("walkers need a thermostat (nvt)")
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
            self.bias_state = base.bias
            self.S = _stack([s.set(bias=None) for s in states]).set(bias=base.bias)
        else:
            self.S = _stack(states)
        self._build()
        self.S = self._forces(self.S).set(induction=self.S.induction)
        self._rows = [[] for _ in range(self.n)]

    # ------------------------------------------------------------------ compiled pieces
    def _axes(self):
        from ..md.remd import _axes
        return _unbatched_bias(self.S) if self.shared else _axes(self.S)

    def _build(self):
        integ, sim = self.integ, self.sim
        ax = self._axes()
        if self.shared:
            self._run = jax.jit(self._run_shared, static_argnums=1)
        else:
            self._run = jax.jit(jax.vmap(integ._run, in_axes=(ax, None), out_axes=ax))
        self._forces = jax.jit(jax.vmap(lambda s: integ._state_forces(s, True), in_axes=(ax,), out_axes=ax))
        self._wrap = jax.jit(jax.vmap(sim.rigid.wrap))
        self._positions = jax.jit(jax.vmap(sim.rigid.positions))
        flex = getattr(sim, "flex", None)
        self._extent = jax.jit(lambda P: jnp.max(jax.vmap(flex.extent)(P))) if flex is not None else None
        bias = integ.bias
        if self.shared:
            self._energies = jax.jit(jax.vmap(lambda x, box, bs: bias.energies(bs, integ._bias_atoms(x), box),
                                              in_axes=(0, 0, None)))
        else:
            self._energies = jax.jit(jax.vmap(lambda x, box, bs: bias.energies(bs, integ._bias_atoms(x), box)))

    def _run_shared(self, S, n):
        integ, bias = self.integ, self.integ.bias
        ax = self._axes()
        W = self.n
        z = jnp.zeros(W)
        S = S.set(max_iters=jnp.zeros(W, jnp.int32), resid=z, overflow=jnp.zeros(W, bool))
        step = jax.vmap(integ._step, in_axes=(ax,), out_axes=ax)
        grad = jax.vmap(jax.value_and_grad(bias.energy, argnums=1), in_axes=(None, 0, 0))
        to_engine = jax.vmap(integ._map_atom_forces)
        atoms = jax.vmap(integ._bias_atoms)

        def post(S):
            pos = atoms(S.dyn.position)
            bs = S.bias
            t = S.step[0]
            for w in range(W):
                bs = bias.record(bs, pos[w], S.box[w], t)
            S = S.set(bias=bs)
            if not bias.dynamic:
                return S

            def dep(S):
                old = S.bias
                new = old
                for w in range(W):
                    new = bias.deposit(new, pos[w], S.box[w], t)
                e0, g0 = grad(old, pos, S.box)
                e1, g1 = grad(new, pos, S.box)
                dF = to_engine(S.dyn.position, S.box, g0 - g1)
                de = e1 - e0
                F = jax.tree_util.tree_map(jnp.add, S.dyn.force, dF)
                return S.set(bias=new._replace(work=new.work + jnp.sum(de)), dyn=S.dyn.set(force=F),
                             epot=S.epot + de, heat=S.heat + de)

            return jax.lax.cond(bias.due(t), dep, lambda s: s, S)

        return jax.lax.fori_loop(0, n, lambda _, s: post(step(s)), S)

    # ------------------------------------------------------------------ bias bookkeeping (host)
    def _resize(self, start, nb_bad: bool, row_bad: bool):
        """MDReplicas._resize with the walkers' own state access (a shared bias is not stacked)."""
        sim = self.sim
        old = self._sizes()
        sizes, lists = [], []
        self.S = start
        for k in range(self.n):
            st = self.state(k)
            lists.append(sim._size_lists(st.dyn.position, st.box, 1.3, None if nb_bad else st.nbr))
            sizes.append(self._sizes())
        self._fit(sizes, old[0] if row_bad else None, old[1] if row_bad else None)
        if nb_bad:
            self._template = max(lists, key=lambda b: b.max_occupancy)
        self.integ.compile()
        S = start.set(nbr=_broadcast(self._template, self.n)) if nb_bad else start
        self.S = S
        self._build()
        return self._forces(S).set(induction=start.induction)

    def _reserve(self, n: int):
        bias = self.integ.bias
        if self.shared:
            self.S = self.S.set(bias=bias.reserve(self.S.bias, n * self.n))
        else:
            per = [bias.reserve(jax.tree_util.tree_map(lambda a: a[w], self.S.bias), n) for w in range(self.n)]
            self.S = self.S.set(bias=jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per))

    def _drain(self):
        bias = self.integ.bias
        if self.shared:
            rows, bs = bias.drain(self.S.bias)
            for w in range(self.n):
                self._rows[w].append(rows[w::self.n])
            self.S = self.S.set(bias=bs)
        else:
            log, nlog = np.asarray(self.S.bias.log), np.asarray(self.S.bias.nlog)
            for w in range(self.n):
                self._rows[w].append(log[w, :nlog[w]])
            self.S = self.S.set(bias=self.S.bias._replace(nlog=jnp.zeros(self.n, jnp.int32)))

    def advance(self, n: int):
        self._reserve(int(n))
        super().advance(int(n))
        self._drain()

    def bias_state(self, w: int = 0):
        """The bias state of walker w (the shared one for shared=True)."""
        if self.shared:
            return self.S.bias
        return jax.tree_util.tree_map(lambda a: a[w], self.S.bias)

    def bias_energies(self) -> np.ndarray:
        """(W, n_bias) bias energies of every walker."""
        return np.asarray(self._energies(self.S.dyn.position, self.S.box, self.S.bias))

    def rows(self, w: int) -> np.ndarray:
        """COLVAR rows of walker w collected since the last call."""
        r = self._rows[w]
        self._rows[w] = []
        return np.concatenate(r) if r else np.zeros((0, self.integ.bias.ncol))

    def state(self, k: int):
        T = jax.tree_util.tree_map(lambda x: x[k], _nocount(self.S.set(bias=None)))
        T = T.set(induction=T.induction.set(count=self.S.induction.count))
        return T.set(bias=self.bias_state(k))

    # ------------------------------------------------------------------ driver
    def run(self, nsteps: int, report: int = 1000, restart: int = 0, prefix: str = "walkers", append: bool = False,
            log=sys.stdout):
        """Advance every walker nsteps; files as in the module docstring."""
        bias, sim = self.integ.bias, self.sim
        block = int(np.gcd.reduce([x for x in (report, restart, nsteps) if x > 0]))
        outs = []
        for w in range(self.n):
            st = self.bias_state(w)
            outs.append(BiasOutput(bias, f"{prefix}_w{w:02d}", self.dt, sim.T0, append=append, state=st,
                                   hills=not self.shared))
        shared_out = None
        if self.shared:
            shared_out = BiasOutput(bias, prefix, self.dt, sim.T0, append=append, state=self.S.bias, colvar=False)
        lf = open(prefix + "_walkers.log", "a" if append else "w")
        if not append:
            lf.write("#       step       time_ps     temp_mean     epot_mean     ebias_mean      bias_work    ns_day_agg\n")
        t0, done = time.time(), 0
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
                work = float(np.sum(np.asarray(self.S.bias.work)))
                el = max(time.time() - t0, 1e-9)
                nsd = done * self.dt / 1000.0 / el * 86400.0 * self.n
                line = (f"{step:12d} {step * self.dt:13.4f} {np.mean(T):13.3f} {float(np.mean(np.asarray(self.S.epot))):13.3f} "
                        f"{float(eb.mean()):14.4f} {work:14.4f} {nsd:13.2f}")
                lf.write(line + "\n")
                lf.flush()
                if log is not None:
                    print(line, file=log, flush=True)
            if restart and step % restart == 0:
                self.save(prefix + ".walkers.chk")
        lf.close()
        if restart:
            self.save(prefix + ".walkers.chk")

    def save(self, path: str):
        d = {"n": self.n, "shared": self.shared, "time_ps": self.time_ps,
             "states": [jax.tree_util.tree_map(np.asarray, self.state(k).set(nbr=None)) for k in range(self.n)]}
        with open(path, "wb") as fh:
            pickle.dump(d, fh)

    def load(self, path: str):
        """Continue from `save` (same system, walker count and mode)."""
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if d["n"] != self.n or d["shared"] != self.shared:
            raise ValueError("checkpoint of a different walker set")
        sim = self.sim
        states = [jax.tree_util.tree_map(jnp.asarray, s) for s in d["states"]]
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
