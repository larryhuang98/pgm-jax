"""Replica exchange (md/remd.py): the criterion, the bookkeeping, and swaps of MD states.

What is checked, and against what: the Metropolis criterion and neighbour pairing; the exchange
chain on replicas with fixed energies against the exact stationary distribution over permutations
(total variation below 0.01) and the exact acceptance per pair; sampled distributions of harmonic
oscillators at every temperature against Gamma(d/2, kT) (a broken criterion fails the same check);
thermostats with a traced temperature; swaps of MD states (what moves with the configuration, what
stays with the slot, rescaled momenta and auxiliaries, energy booked as heat); batched (vmap) and
sequential engines giving the same run, overflow handling, and restarts from checkpoints.
"""

import itertools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import md_settings, small_box, water, water_lattice

from pgm_jax import System
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.remd import (
    ReplicaExchange,
    _broadcast,
    exchange_pairs,
    geometric_ladder,
    metropolis,
    read_exchange_log,
    temperature_reduced_energies,
)
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import GLE, Bussi, Langevin
from pgm_jax.units import KB, KJMOL_NM3_PER_BAR


# ----------------------------------------------------------------------------- toy replica engines
class FixedEnergies:
    """Configurations with fixed potential energies and no dynamics."""

    dt, pressure, time_ps = 0.001, None, 0.0

    def __init__(self, temperatures, energies):
        """Set up configurations with the given potential energies [kJ/mol] at the given temperatures [K]."""
        self.temperatures = np.asarray(temperatures, float)
        self.n = len(self.temperatures)
        self.E = np.asarray(energies, float)
        self.config = np.arange(self.n)  # configuration at each temperature

    def advance(self, n):
        """Advance the clock by n steps (the configurations do not move)."""
        self.time_ps += n * self.dt

    def potentials(self):
        """Return the potential energy at each temperature slot [kJ/mol]."""
        return self.E[self.config]

    def permute(self, src):
        """Move configurations: slot k receives the configuration of slot src[k]."""
        self.config = self.config[np.asarray(src)]


class Harmonic:
    """Replica engine of harmonic oscillators sampled by an exact Ornstein-Uhlenbeck step.

    One overdamped particle in U = k |x|^2 / 2 (d dimensions) per temperature, sampled by the exact
    Ornstein-Uhlenbeck step x -> c x + sqrt((1 - c^2) kT / k) xi (slow for c near 1, so that the
    exchanges carry much of the sampling).
    """

    dt, pressure = 0.001, None

    def __init__(self, temperatures, d=10, k=1.0, c=0.95, seed=0):
        """Set up one d-dimensional particle per temperature, drawn from its Boltzmann distribution.

        Parameters
        ----------
        temperatures : sequence of float
            Temperatures [K].
        d : int
            Dimension.
        k : float
            Force constant [kJ/mol/nm^2].
        c : float
            Memory of the Ornstein-Uhlenbeck step (0: independent samples).
        seed : int
            Seed of the generator.
        """
        self.temperatures = np.asarray(temperatures, float)
        self.n, self.k, self.c, self.time_ps = len(self.temperatures), k, c, 0.0
        kT = KB * self.temperatures
        self.rng = np.random.default_rng(seed)
        self.x = self.rng.normal(size=(self.n, d)) * np.sqrt(kT / k)[:, None]
        self.s = np.sqrt((1.0 - c * c) * kT / k)

    def advance(self, n):
        """Take n Ornstein-Uhlenbeck steps at every temperature."""
        for _ in range(n):
            self.x = self.c * self.x + self.s[:, None] * self.rng.normal(size=self.x.shape)
        self.time_ps += n * self.dt

    def potentials(self):
        """Return U = k |x|^2 / 2 at each temperature slot [kJ/mol]."""
        return 0.5 * self.k * np.sum(self.x**2, axis=1)

    def permute(self, src):
        """Move configurations: slot k receives the configuration of slot src[k]."""
        self.x = self.x[np.asarray(src)]

    def observables(self, k):
        """Return the time and potential energy of slot k."""
        return {"time_ps": self.time_ps, "epot": float(self.potentials()[k])}

    def state_dict(self):
        """Return the state for a checkpoint (positions, generator state, time)."""
        return {"x": self.x.copy(), "rng": self.rng.bit_generator.state, "time_ps": self.time_ps}

    def load_state_dict(self, d):
        """Restore the state of state_dict."""
        self.x, self.time_ps = d["x"].copy(), d["time_ps"]
        self.rng.bit_generator.state = d["rng"]


def _sample(rex, steps):
    """Alternate one step and one exchange attempt `steps` times; return the potentials (steps, K)."""
    U = []
    for _ in range(steps):
        rex.replicas.advance(1)
        rex.exchange()
        U.append(rex.replicas.potentials())
    return np.array(U)


# ----------------------------------------------------------------------------- exchange logic
def test_ladder_pairs_and_criterion():
    """Geometric ladders, alternating neighbour pairs and the Metropolis criterion (also with P V)."""
    T = geometric_ladder(300.0, 450.0, 5)
    assert np.isclose(T[0], 300.0) and np.isclose(T[-1], 450.0) and np.allclose(T[1:] / T[:-1], 1.5**0.25)
    assert exchange_pairs(5, 0) == [(0, 1), (2, 3)] and exchange_pairs(5, 1) == [(1, 2), (3, 4)]
    assert exchange_pairs(4, 3) == [(1, 2)]
    # a higher-energy configuration moving down in temperature: P = exp(-(b0 - b1)(U1 - U0))
    U = np.array([-100.0, -90.0])
    u = temperature_reduced_energies([300.0, 330.0], U)
    p = np.exp(-(1.0 / (KB * 300.0) - 1.0 / (KB * 330.0)) * (U[1] - U[0]))
    assert 0.0 < p < 1.0
    assert metropolis(u, [(0, 1)], [p * 0.999])[0][0] and not metropolis(u, [(0, 1)], [p * 1.001])[0][0]
    acc, src = metropolis(temperature_reduced_energies([300.0, 330.0], U[::-1]), [(0, 1)], [0.9999])
    assert acc[0] and list(src) == [1, 0]  # downhill: always
    # constant pressure: enthalpies U + P V
    V, P = np.array([30.0, 31.0]), 1.0 * KJMOL_NM3_PER_BAR
    uP = temperature_reduced_energies([300.0, 330.0], U, V, P)
    assert np.allclose(uP, temperature_reduced_energies([300.0, 330.0], U + P * V))
    with pytest.raises(ValueError):
        metropolis(u, [(0, 1), (1, 0)], [0.5, 0.5])


def test_metropolis_statistics_known_energies():
    """The exchange chain samples the exact distribution over replica permutations.

    Exchange-only chain over replica permutations: the visited permutations follow
    pi(sigma) ~ exp(-sum_k beta_k E_sigma(k)) and each pair's acceptance is E_pi[min(1, e^-Delta)].
    """
    T = geometric_ladder(300.0, 420.0, 4)
    E = np.array([0.0, 25.0, 60.0, 110.0])
    rex = ReplicaExchange(FixedEnergies(T, E), exchange_every=1, seed=3, log=None)
    perms = list(itertools.permutations(range(4)))
    index = {p: i for i, p in enumerate(perms)}
    counts = np.zeros(len(perms))
    n = 60000
    for _ in range(n):
        rex.exchange()
        counts[index[tuple(rex.replicas.config)]] += 1
    beta = 1.0 / (KB * T)
    w = np.array([np.exp(-np.sum(beta * E[list(p)])) for p in perms])
    w /= w.sum()
    tv = 0.5 * np.abs(counts / n - w).sum()
    assert tv < 0.01, tv
    acc = rex.stats.neighbour_acceptance()
    for i in range(3):
        exact = sum(
            wp * min(1.0, np.exp(-(beta[i] - beta[i + 1]) * (E[p[i + 1]] - E[p[i]]))) for p, wp in zip(perms, w)
        )
        assert abs(acc[i] - exact) < 0.01, (i, acc[i], exact)
        assert rex.stats.attempts[i, i + 1] == n // 2
    assert np.array_equal(rex.stats.replica, rex.replicas.config)  # replica map follows the configurations
    assert rex.stats.round_trips.sum() > 100 and np.all(rex.stats.transits >= 2 * rex.stats.round_trips)


def test_harmonic_distributions_at_every_temperature():
    """REMD samples the canonical energy distribution at every temperature.

    Detailed balance: U of a d-dimensional oscillator is Gamma(d/2, kT) at every temperature of
    the ladder, although much of the sampling at each temperature comes from exchanges.  Accepting
    every swap (no energy criterion) fails the same check.
    """
    from scipy import stats

    T, d = geometric_ladder(300.0, 600.0, 4), 10
    kT = KB * T
    rex = ReplicaExchange(Harmonic(T, d, seed=2), exchange_every=1, seed=1, log=None)
    U = _sample(rex, 40000)[2000:]
    mean_err = U.mean(0) / (0.5 * d * kT) - 1.0
    var_err = U.var(0) / (0.5 * d * kT**2) - 1.0
    ks = [stats.kstest(U[::10, k], stats.gamma(0.5 * d, scale=kT[k]).cdf).statistic for k in range(4)]
    assert np.abs(mean_err).max() < 0.03 and np.abs(var_err).max() < 0.12 and max(ks) < 0.04, (mean_err, var_err, ks)
    acc = rex.stats.neighbour_acceptance()
    assert np.all((acc > 0.2) & (acc < 0.8)), acc

    class AlwaysSwap(ReplicaExchange):
        """A broken exchange that accepts every swap."""

        def reduced_energies(self):
            """Return zero reduced energies (every swap accepted)."""
            return np.zeros((self.n, self.n))

    bad = AlwaysSwap(Harmonic(T, d, seed=2), exchange_every=1, seed=1, log=None)
    Ub = _sample(bad, 40000)[2000:]
    assert Ub.mean(0)[0] / (0.5 * d * kT[0]) - 1.0 > 0.2 and Ub.mean(0)[-1] / (0.5 * d * kT[-1]) - 1.0 < -0.15


def test_driver_restart_reproduces_toy_run(tmp_path):
    """A run restarted from a checkpoint equals the uninterrupted run; the exchange log is written."""
    T = geometric_ladder(300.0, 500.0, 5)
    ref = ReplicaExchange(Harmonic(T, seed=4), exchange_every=3, seed=9, log=None)
    ref.run(60, report_every=6, prefix=None)
    x_ref, rep_ref = ref.replicas.x.copy(), ref.stats.replica.copy()
    a = ReplicaExchange(Harmonic(T, seed=4), exchange_every=3, seed=9, log=None)
    a.run(30, report_every=6, checkpoint_every=30, prefix=str(tmp_path / "toy"))
    b = ReplicaExchange(Harmonic(T, seed=77), exchange_every=3, seed=0, log=None)
    b.load_checkpoint(str(tmp_path / "toy.remd.chk"))
    b.run(30, report_every=6, prefix=str(tmp_path / "toy2"))
    assert b.step == 60 and np.array_equal(b.stats.replica, rep_ref) and np.array_equal(b.replicas.x, x_ref)
    assert b.stats.n_exchanges == ref.stats.n_exchanges and np.array_equal(b.stats.accepts, ref.stats.accepts)
    steps, reps, outs = read_exchange_log(str(tmp_path / "toy_remd.log"))
    assert list(steps) == list(range(3, 31, 3)) and reps.shape == (10, 5) and outs.shape == (10, 4)
    assert set(outs[0]) <= {"+", ".", "-"} and all(c == "-" for c in outs[0][1::2])  # even pairs first


@pytest.mark.parametrize("th", [Langevin(2.0), Bussi(0.2), GLE.band()], ids=["langevin", "bussi", "gle"])
def test_thermostats_take_traced_temperature(th):
    """Thermostat O steps accept kB T as a traced value (jit result = eager, 1e-12).

    kB T enters the O steps as a traced value (one per replica in a batched run).
    """
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (40, 3))
    aux = th.init_aux(jax.random.PRNGKey(1), v.shape, 2.5)

    def f(kT):
        """Return the O step at thermal energy kT."""
        return th.apply(v, aux, jax.random.PRNGKey(2), 0.002, kT, 120.0, lambda u: u, None)

    a, b = f(2.5), jax.jit(f)(jnp.asarray(2.5))
    assert np.allclose(a[0], b[0], rtol=1e-12, atol=1e-12) and np.allclose(a[1], b[1], rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------------- MD engines
def _econs(integ, st):
    """Return the conserved energy K + U + |aux|^2 / 2 - heat of an MD state [kJ/mol]."""
    return float(integ.kinetic(st)[0] + st.epot + 0.5 * jnp.sum(st.aux * st.aux) - st.heat)


def test_md_swap_and_batched_equals_sequential():
    """Batched and sequential MD replicas agree; a swap moves the right parts of the state.

    Rigid water by constraints with the GLE thermostat (auxiliaries): batched and sequential
    replica engines give the same exchanges and trajectories; a swap moves the configuration (with
    dipoles, predictor history, forces and neighbour list) and the rescaled momenta and auxiliaries,
    keeps temperature, random stream and step with the slot and books the energy as heat.
    """
    pos, H, w = water_lattice()
    wat = water()
    sys = System([wat] * (len(pos) // 3))
    s = MDSettings().replace(precision="double", dipole_tol=1e-9, cutoff=0.55, skin=0.05)
    sim = FlexibleSimulation(
        sys,
        [RigidTemplate(wat, w)] * sys.nmol,
        pos,
        H,
        s,
        dt=0.002,
        temperature=300.0,
        thermostat="gle",
        log=None,
        seed=1,
    )
    T = np.array([300.0, 304.0, 308.0])
    runs = {}
    for batched in (True, False):
        rex = ReplicaExchange(sim, T, exchange_every=10, batched=batched, seed=5, log=None)
        trace = []
        for _ in range(4):
            rex.replicas.advance(10)
            rex.exchange()
            trace.append(rex.stats.replica.copy())
        runs[batched] = rex, np.array(trace)
    (rb, tb), (rs, ts) = runs[True], runs[False]
    assert np.array_equal(tb, ts) and rb.stats.accepts.sum() >= 2, (tb, rb.stats.accepts)
    for k in range(3):
        a, b = rb.replicas.state(k), rs.replicas.state(k)
        assert np.abs(np.asarray(a.dyn.position) - np.asarray(b.dyn.position)).max() < 1e-9
        assert np.abs(np.asarray(a.dyn.momentum) - np.asarray(b.dyn.momentum)).max() < 1e-8
        assert abs(float(a.heat) - float(b.heat)) < 1e-6 and float(a.kT) == KB * T[k]
    # an explicit swap of slots 0 and 1 in the batched engine
    integ = sim.integ
    s0, s1, s2 = (rb.replicas.state(k) for k in range(3))
    e0, e1 = _econs(integ, s0), _econs(integ, s1)
    rb.replicas.permute([1, 0, 2])
    n0, n1, n2 = (rb.replicas.state(k) for k in range(3))
    f = np.sqrt(T[0] / T[1])

    def same(x, y):
        """Return whether x and y are bitwise equal."""
        return np.array_equal(np.asarray(x), np.asarray(y))

    for moved, src in ((n0, s1), (n1, s0)):
        for x, y in (
            (moved.dyn.position, src.dyn.position),
            (moved.dyn.force, src.dyn.force),
            (moved.box, src.box),
            (moved.induction.mu, src.induction.mu),
            (moved.induction.hist, src.induction.hist),
            (moved.nbr.idx, src.nbr.idx),
            (moved.epot, src.epot),
        ):
            assert same(x, y)
    assert np.allclose(np.asarray(n0.dyn.momentum), np.asarray(s1.dyn.momentum) * f, rtol=1e-14, atol=0)
    assert np.allclose(np.asarray(n1.aux), np.asarray(s0.aux) / f, rtol=1e-14, atol=0) and n1.aux.size > 0
    for kept, dst in ((n0, s0), (n1, s1)):
        for x, y in ((kept.kT, dst.kT), (kept.dyn.rng, dst.dyn.rng), (kept.step, dst.step), (kept.mc_dv, dst.mc_dv)):
            assert same(x, y)
    assert abs(_econs(integ, n0) - e0) < 1e-8 and abs(_econs(integ, n1) - e1) < 1e-8  # exchange booked as heat
    assert abs(float(integ.kinetic(n0)[0]) / float(integ.kinetic(s1)[0]) - T[0] / T[1]) < 1e-12
    assert all(
        same(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(n2.set(nbr=None)), jax.tree_util.tree_leaves(s2.set(nbr=None)))
    )
    rb.replicas.permute([1, 0, 2])  # and back
    b0 = rb.replicas.state(0)
    assert same(b0.dyn.position, s0.dyn.position) and np.allclose(
        np.asarray(b0.dyn.momentum), np.asarray(s0.dyn.momentum), rtol=1e-14
    )
    assert abs(float(b0.heat) - float(s0.heat)) < 1e-9
    # overflows in the batched engine (row capacity, then the neighbour list) are resized for every
    # replica and the block repeated: the run continues exactly as the sequential one
    rep = rb.replicas
    sim.ff.mc = 16
    rep._build()
    t = rep._template
    small = t.set(idx=t.idx[:, :4], max_occupancy=4)
    for shrink in ("rows", "list"):
        if shrink == "list":
            S = rep.S.set(nbr=_broadcast(small, 3))
            rep.S = S
            rep._build()
            rep.S = rep._forces(S).set(induction=S.induction)  # rebuilt into 4 slots: overflow
            assert rep._nb_failed(rep.S)
        rep.advance(10)
        rs.replicas.advance(10)
        for k in range(3):
            d = np.abs(np.asarray(rep.state(k).dyn.position) - np.asarray(rs.replicas.state(k).dyn.position)).max()
            assert d < 1e-9, (shrink, k, d)
    assert sim.ff.mc > 16 and rep._template.max_occupancy > 4 and not rep._nb_failed(rep.S)


def test_md_npt_sequential_restart(tmp_path):
    """NPT replica exchange (sequential engine) writes its files and restarts exactly.

    Rigid-body water, NPT (P V in the criterion), sequential engine: files, and a restart from the
    checkpoint reproduces the continued run.
    """
    pos, H, _ = water_lattice()
    sys = System([water()] * (len(pos) // 3))
    s = MDSettings().replace(precision="double", dipole_tol=1e-9, cutoff=0.55, skin=0.05)
    sim = Simulation(
        sys, pos, H, s, dt=0.001, thermostat=Bussi(0.1), barostat=MonteCarloBarostat(every=5), log=None, seed=2
    )
    with pytest.raises(ValueError):
        ReplicaExchange(sim, [300.0, 310.0], batched=True, log=None)  # NPT needs the sequential engine
    T = [300.0, 303.0, 306.0]
    rex = ReplicaExchange(sim, T, exchange_every=5, batched=False, seed=3, log=None)
    p = str(tmp_path / "w")
    rex.run(20, report_every=10, traj_every=10, checkpoint_every=20, prefix=p)
    rex.run(20, report_every=10, prefix=str(tmp_path / "w_cont"))
    X_ref = [rex.replicas.positions(k) for k in range(3)]
    B_ref = [np.asarray(rex.replicas.state(k).box) for k in range(3)]
    trace_ref, acc_ref = rex.stats.replica.copy(), rex.stats.accepts.copy()
    for suffix in ("_T00.log", "_T02.nc", "_T01.rst7", "_remd.log", "_remd.json", ".remd.chk"):
        assert os.path.exists(p + suffix), suffix
    rex.load_checkpoint(p + ".remd.chk")
    assert rex.step == 20
    rex.run(20, report_every=10, prefix=str(tmp_path / "w_again"))
    assert np.array_equal(rex.stats.replica, trace_ref) and np.array_equal(rex.stats.accepts, acc_ref)
    for k in range(3):
        assert np.abs(rex.replicas.positions(k) - X_ref[k]).max() < 1e-7
        assert np.abs(np.asarray(rex.replicas.state(k).box) - B_ref[k]).max() < 1e-9
    log = open(p + "_T01.log").read().splitlines()
    assert log[0].startswith("# T = 303") and "replica" in log[1] and len(log) == 4


def test_md_rigid_batched_checkpoint_continues_sequentially(tmp_path):
    """A batched REMD checkpoint continues in the sequential engine as in the batched one.

    Rigid-body water, NVT, batched engine: a checkpoint loads into the sequential engine, which
    continues exactly as the batched run does.
    """
    pos, H, _ = water_lattice()
    sys = System([water()] * (len(pos) // 3))
    s = MDSettings().replace(precision="double", dipole_tol=1e-9, cutoff=0.55, skin=0.05)
    sim = Simulation(sys, pos, H, s, dt=0.001, thermostat=Langevin(5.0), log=None, seed=4)
    T = [300.0, 304.0, 308.0, 312.0]
    rb = ReplicaExchange(sim, T, exchange_every=5, batched=True, seed=6, log=None)
    rs = ReplicaExchange(sim, T, exchange_every=5, batched=False, seed=0, log=None)
    p = str(tmp_path / "r")
    rb.run(20, report_every=0, checkpoint_every=20, prefix=p)
    rb.run(20, report_every=0, prefix=None)
    rs.load_checkpoint(p + ".remd.chk")
    rs.run(20, report_every=0, prefix=None)
    assert rs.step == rb.step == 40 and np.array_equal(rs.stats.replica, rb.stats.replica)
    assert np.array_equal(rs.stats.accepts, rb.stats.accepts) and rb.stats.accepts.sum() >= 2
    for k in range(4):
        assert np.abs(rs.replicas.positions(k) - rb.replicas.positions(k)).max() < 1e-8
        assert abs(rs.replicas.observables(k)["econs"] - rb.replicas.observables(k)["econs"]) < 1e-6


def test_md_replicas_split_rows_fit_every_part():
    """Batched replicas with split rows size both parts and recover from an overflow.

    Split rows (elec_cutoff < cutoff): shared capacities fit each part of the rows for every
    replica, and an overflow of the electrostatic part in the batched engine re-sizes both parts
    and repeats the block (same run as replicas that never overflowed).
    """
    sys, pos, H = small_box(4)
    s = md_settings(cutoff=0.6, elec_cutoff=0.45, dipole_tol=1e-9, max_iter=100)

    def make():
        """Build the small-box simulation with split rows (Bussi NVT)."""
        return Simulation(sys, pos, H, s, dt=0.001, thermostat="bussi", temperature=300.0, log=None, seed=2)

    sim = make()
    mc, mc_e = sim.ff.capacity
    tail = mc - mc_e
    sim.ff.fit_rows([(mc, mc_e), (mc_e + 4 + tail, mc_e + 4), (mc_e + tail + 16, mc_e)])
    assert sim.ff.capacity == (mc_e + 4 + tail + 16, mc_e + 4)
    T = [300.0, 320.0]
    ref = ReplicaExchange(make(), T, exchange_every=10, batched=True, seed=3, log=None).replicas
    rex = ReplicaExchange(make(), T, exchange_every=10, batched=True, seed=3, log=None).replicas
    ff = rex.sim.ff
    small = mc_e // 2
    ff.mc, ff.mc_e = small + tail, small  # electrostatic part too small
    rex.integ.compile()
    rex._build()
    rex.advance(20)
    ref.advance(20)
    assert ff.mc_e >= small + 8 and ff.mc - ff.mc_e >= tail
    for k in range(2):
        a, b = rex.state(k), ref.state(k)
        xa, xb = (np.asarray(rex.sim.rigid.positions(st.dyn.position)) for st in (a, b))
        assert np.abs(xa - xb).max() < 1e-9
