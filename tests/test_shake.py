"""Holonomic constraints (md/constraints.py) in the flexible engine: the dense (Newton per cluster)
and iterative (matrix-free) solvers against each other, the RATTLE projection, degrees of freedom,
energy conservation at 1 and 2 fs, NPT and multiple time stepping with constraints."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_flexible import template

from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.constraints import Constraints
from pgm_jax.md.flexible import FlexibleSimulation, liquid_box
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.thermostats import Bussi, Langevin
from pgm_jax.system import System
from pgm_jax.units import KB


def _clusters():
    """Water (3 constraints), CH3 (3), OH (1), methanol with every bond (5), a six-ring with its
    bonds and hydrogens (12) and a 20-bond chain (above dense_max: the iterative solver)."""
    rng = np.random.default_rng(3)
    x, pairs, m = [], [], []

    def add(xyz, bonds, masses, at):
        o = len(np.concatenate(x)) if x else 0
        x.append(np.asarray(xyz, float) + at)
        pairs.extend((o + i, o + j) for i, j in bonds)
        m.extend(masses)

    add([[0, 0, 0], [0.0957, 0, 0], [-0.024, 0.0927, 0]], [(0, 1), (0, 2), (1, 2)], [16.0, 1.0, 1.0], 0.0)
    add(
        [[0, 0, 0], [0.109, 0, 0], [-0.036, 0.103, 0], [-0.036, -0.052, 0.089]],
        [(0, 1), (0, 2), (0, 3)],
        [12.0, 1.0, 1.0, 1.0],
        1.0,
    )
    add([[0, 0, 0], [0.096, 0, 0]], [(0, 1)], [16.0, 1.0], 2.0)
    tpl, xm = template()
    add(xm, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)], np.asarray(tpl.pgm.masses), 3.0)
    ang = np.arange(6) * np.pi / 3
    ring = np.stack([0.14 * np.cos(ang), 0.14 * np.sin(ang), np.zeros(6)], 1)
    hs = ring * (0.248 / 0.14)
    add(
        np.concatenate([ring, hs]),
        [(k, (k + 1) % 6) for k in range(6)] + [(k, k + 6) for k in range(6)],
        [12.0] * 6 + [1.0] * 6,
        4.0,
    )
    chain = np.cumsum(np.stack([0.153 * np.ones(21), 0.03 * rng.normal(size=21), 0.03 * rng.normal(size=21)], 1), 0)
    chain *= 0.153 / np.linalg.norm(np.diff(chain, axis=0), axis=1).mean()
    add(chain, [(k, k + 1) for k in range(20)], [12.0, 14.0, 12.0] * 7, 5.0)
    x = np.concatenate(x)
    pairs = np.array(pairs)
    d0 = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
    return x, pairs, d0, np.asarray(m, float), rng


def test_solvers_agree_and_project():
    x, pairs, d0, m, rng = _clusters()
    y = x + 0.004 * rng.normal(size=x.shape)  # an MD step's displacement
    p = rng.normal(size=x.shape) * np.sqrt(m)[:, None]
    solvers = {
        "blocks": Constraints(pairs, d0, m),
        "one block": Constraints(pairs, d0, m, bucket=False, dense_max=40),
        "iterative": Constraints(pairs, d0, m, dense_max=0, tol=1e-13, rattle_tol=1e-14),
    }
    assert [b.kind for b in solvers["blocks"].blocks] == ["dense"] * 3 + ["sparse"]  # <= 3, 5, 12; the chain
    out = {}
    for name, C in solvers.items():
        z = np.asarray(jax.jit(C.positions)(jnp.asarray(y), jnp.asarray(x)))
        r = np.linalg.norm(z[pairs[:, 0]] - z[pairs[:, 1]], axis=1)
        tol = 1e-10 if name == "blocks" else 1e-12  # the chain: iterative solver at tol 1e-10
        assert np.abs(r / d0 - 1).max() < tol, (name, np.abs(r / d0 - 1).max())
        assert abs(float(C.violation(jnp.asarray(z))) - np.abs(r / d0 - 1).max()) < 1e-14
        assert np.allclose((m[:, None] * (z - y)).sum(0), 0.0, atol=1e-12)  # internal forces
        q = np.asarray(C.momenta(jnp.asarray(z), jnp.asarray(p), m))
        v = q / m[:, None]
        rv = np.einsum("cx,cx->c", z[pairs[:, 0]] - z[pairs[:, 1]], v[pairs[:, 0]] - v[pairs[:, 1]])
        vtol = 1e-10 if name == "blocks" else 1e-12  # iterative RATTLE at rattle_tol 1e-11
        assert np.abs(rv).max() < vtol * np.abs(v).max()
        assert float(C.velocity_violation(jnp.asarray(z), jnp.asarray(q), m)) < vtol
        assert np.allclose(q.sum(0), p.sum(0), atol=1e-12)  # no net momentum change
        q2 = np.asarray(C.momenta(jnp.asarray(z), jnp.asarray(q), m))
        assert np.allclose(q2, q, atol=1e-10)  # a projection
        # mass-weighted orthogonal: the removed part is M^-1-orthogonal to the kept one
        assert abs(np.sum((p - q) * q / m[:, None])) < 1e-10 * np.sum(p * p / m[:, None])
        out[name] = (z, q)
    for name in ("one block", "iterative"):
        assert np.abs(out[name][0] - out["blocks"][0]).max() < 1e-10, name
        assert np.abs(out[name][1] - out["blocks"][1]).max() < 1e-9 * np.abs(p).max(), name


def test_rattle_velocity_is_tangent_finite_difference():
    """After RATTLE the constraint lengths change only at second order along the velocities:
    |r(x + e v)| - d0 = O(e^2) (halving e divides it by 4), without RATTLE O(e)."""
    x, pairs, d0, m, rng = _clusters()
    C = Constraints(pairs, d0, m)
    p = rng.normal(size=x.shape) * np.sqrt(m)[:, None]
    q = np.asarray(C.momenta(jnp.asarray(x), jnp.asarray(p), m))

    def dev(mom, e):
        z = x + e * mom / m[:, None]
        return np.abs(np.linalg.norm(z[pairs[:, 0]] - z[pairs[:, 1]], axis=1) - d0).max()

    e = 1e-3
    assert 3.5 < dev(q, e) / dev(q, e / 2) < 4.5
    assert 1.8 < dev(p, e) / dev(p, e / 2) < 2.2


def _methanol_box(n=32, seed=0):
    tpl, _ = template()
    pos, H = liquid_box(tpl, n, 0.55, seed=seed, min_dist=0.18)
    return tpl, System([tpl.pgm] * n), pos, H


def _cluster():
    """Eight methanols (a 2 x 2 x 2 lattice at liquid density) in a 3.6 nm box with a 1.7 nm
    cutoff: no pair crosses the cutoff, so NVE conserves the energy to the integration error."""
    tpl, _ = template()
    pos, H = liquid_box(tpl, 8, 0.75, seed=2, min_dist=0.2)
    return tpl, System([tpl.pgm] * 8), pos + 1.4, np.eye(3) * 3.6


SCL = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=1.7, skin=0.05, lj_lrc=False)


def test_rules_and_degrees_of_freedom():
    tpl, sys_, pos, H = _cluster()
    assert len(tpl.md_rule("h-bonds").constraints) == 4 and len(tpl.md_rule("all-bonds").constraints) == 5
    assert all(abs(d - b) < 1e-12 for (i, j, d), b in zip(tpl.md_rule("all-bonds").constraints, tpl.bond_lengths()))
    with pytest.raises(ValueError):
        tpl.md_rule("h-angles")
    n = sys_.n
    for cons, nc in (("none", 0), ("h-bonds", 32), ("all-bonds", 40)):
        for th, sub in ((None, 3), ("langevin", 0), ("bussi", 3)):
            sim = FlexibleSimulation(sys_, [tpl] * 8, pos, H, SCL, dt=0.001, thermostat=th, constraints=cons, log=None)
            assert sim.constraints.nc == nc and sim.integ.dof == 3 * n - nc - sub, (cons, th)
            if sub:  # total momentum removed at the start
                assert np.abs(np.asarray(sim.state.dyn.momentum).sum(0)).max() < 1e-10


def test_constraints_every_step_and_nve():
    """X-H bonds at 0.5, 1 and 2 fs, every bond at 2 fs: every step on the constraint surface
    (1e-10) with tangent velocities; the energy fluctuation of velocity Verlet + RATTLE grows as
    dt^2 and does not drift."""
    tpl, sys_, pos, H = _cluster()
    eq = FlexibleSimulation(
        sys_,
        [tpl] * 8,
        pos,
        H,
        SCL,
        dt=0.001,
        thermostat=Langevin(20.0),
        temperature=298.0,
        constraints="h-bonds",
        log=None,
    )
    eq.advance(400)
    x0, v0, H0 = eq.positions(), eq.velocities(), np.asarray(eq.state.box)
    fluct = {}
    for cons, dt in (
        ("none", 0.0005),
        ("h-bonds", 0.0005),
        ("h-bonds", 0.001),
        ("h-bonds", 0.002),
        ("all-bonds", 0.002),
    ):
        sim = FlexibleSimulation(
            sys_, [tpl] * 8, x0, H0, SCL, dt=dt, thermostat=None, velocities=v0, constraints=cons, log=None
        )
        E = []
        for k in range(int(round(0.2 / dt))):  # 0.2 ps, checked every step
            sim.advance(1)
            o = sim.observables()
            if cons != "none":
                assert o["shake_err"] < 1e-10 and o["rattle_err"] < 1e-10, (
                    cons,
                    dt,
                    k,
                    o["shake_err"],
                    o["rattle_err"],
                )
            E.append(o["etot"])
        E = np.asarray(E)
        fluct[(cons, dt)] = np.std(E)
        assert abs(np.mean(E[-20:]) - np.mean(E[:20])) < 3.0 * np.std(E), (cons, dt)
    r1 = fluct[("h-bonds", 0.001)] / fluct[("h-bonds", 0.0005)]
    r2 = fluct[("h-bonds", 0.002)] / fluct[("h-bonds", 0.001)]
    assert 2.5 < r1 < 6.0 and 2.5 < r2 < 6.0, (r1, r2, fluct)
    assert fluct[("all-bonds", 0.002)] < 1.5 * fluct[("h-bonds", 0.002)], fluct


def test_skipped_projection_is_exact():
    """The drift before a kick leaves RATTLE to the kick (same positions): identical trajectories to
    projecting after every drift (NVE and Langevin)."""
    from pgm_jax.md import flexible as F

    tpl, sys_, pos, H = _cluster()
    for th in (None, "langevin"):
        a = FlexibleSimulation(sys_, [tpl] * 8, pos, H, SCL, dt=0.002, thermostat=th, constraints="h-bonds", seed=4)
        a.advance(20)
        orig = F.FlexibleIntegrator._drift
        try:
            F.FlexibleIntegrator._drift = lambda self, dyn, h, project=True: orig(self, dyn, h, True)
            b = FlexibleSimulation(sys_, [tpl] * 8, pos, H, SCL, dt=0.002, thermostat=th, constraints="h-bonds", seed=4)
            b.advance(20)
        finally:
            F.FlexibleIntegrator._drift = orig
        assert np.abs(a.positions() - b.positions()).max() < 1e-11
        assert np.abs(np.asarray(a.state.dyn.momentum - b.state.dyn.momentum)).max() < 1e-9


def test_npt_bussi_and_mts_keep_constraints():
    """Monte Carlo volume moves (molecular scaling) keep the constraints; Bussi keeps the total
    momentum at zero; multiple time stepping with every bond constrained."""
    from pgm_jax.md.mts import MTS

    tpl, sys_, pos, H = _methanol_box()
    s = MDSettings().replace(precision="mixed", dipole_tol=1e-5, cutoff=0.6, skin=0.05)
    sim = FlexibleSimulation(
        sys_,
        [tpl] * 32,
        pos,
        H,
        s,
        dt=0.002,
        thermostat=Bussi(0.2),
        barostat=MonteCarloBarostat(2000.0, 5),
        constraints="h-bonds",
        log=None,
    )
    V0 = sim.observables()["volume_nm3"]
    for _ in range(5):
        sim.advance(100)
        o = sim.observables()
        assert o["shake_err"] < 1e-10 and o["rattle_err"] < 1e-10
    assert int(sim.state.mc[0]) >= 50 and int(sim.state.mc[1]) >= 1 and o["volume_nm3"] != V0
    P = np.asarray(sim.state.dyn.momentum).sum(0)  # Bussi scales it; only PME's float32 net force moves it
    assert 0.5 * np.sum(P * P) / float(np.sum(sim.flex.masses)) < 1e-4 * KB * 298.0
    tpl, sys_, pos, H = _cluster()
    m = FlexibleSimulation(
        sys_,
        [tpl] * 8,
        pos,
        H,
        SCL,
        dt=0.003,
        thermostat=None,
        constraints="all-bonds",
        mts=MTS(inner=3, split="special"),
        log=None,
    )
    e0 = m.observables()["etot"]
    for _ in range(4):
        m.advance(25)
        o = m.observables()
        assert o["shake_err"] < 1e-10 and o["rattle_err"] < 1e-10
    assert abs(o["etot"] - e0) < 2e-2 * 0.5 * m.integ.dof * KB * 298.0


def test_iterative_solver_in_md():
    """A 29-atom peptide with every bond constrained (above dense_max: one iterative block), NVE at
    2 fs in vacuum-like conditions: constraints every step, energy conserved."""
    from test_md_macro import _peptide_template

    tpl, model, P, x = _peptide_template()
    s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.05, lj_lrc=False)
    sim = FlexibleSimulation(
        System([tpl.pgm]),
        [tpl],
        x + 1.5,
        np.eye(3) * 3.0,
        s,
        dt=0.002,
        thermostat=None,
        temperature=300.0,
        constraints="all-bonds",
        log=None,
    )
    assert [b.kind for b in sim.constraints.blocks] == ["sparse"] and sim.constraints.nc == len(tpl.spec.bonds)
    E = []
    for _ in range(50):
        sim.advance(2)
        o = sim.observables()
        assert o["shake_err"] < 1e-9 and o["rattle_err"] < 1e-10, (o["shake_err"], o["rattle_err"])
        E.append(o["etot"])
    ke = 0.5 * sim.integ.dof * KB * 300.0
    assert np.std(E) < 1e-2 * ke, np.std(E) / ke


def test_thermostats_with_constraints():
    """Langevin and GLE (its auxiliary momenta projected onto the constraint tangent space at every
    O step) with every X-H bond constrained: constraints and tangent velocities every block; the effective energy
    econs = E_tot + |aux|^2/2 - heat stays conserved to the integration error."""
    tpl, sys_, pos, H = _cluster()
    for th in ("langevin", "gle"):
        sim = FlexibleSimulation(
            sys_,
            [tpl] * 8,
            pos,
            H,
            SCL,
            dt=0.001,
            thermostat=Langevin(5.0) if th == "langevin" else th,
            temperature=298.0,
            constraints="h-bonds",
            seed=2,
            log=None,
        )
        e0 = sim.observables()["econs"]
        ec = []
        for _ in range(10):
            sim.advance(20)
            o = sim.observables()
            assert o["shake_err"] < 1e-10 and o["rattle_err"] < 1e-10, (th, o["shake_err"], o["rattle_err"])
            ec.append(o["econs"])
        ke = 0.5 * sim.integ.dof * KB * 298.0
        assert np.abs(np.asarray(ec) - e0).max() < 1e-2 * ke, (th, np.abs(np.asarray(ec) - e0).max() / ke)


def test_half_step_kinetic_energy():
    """half_step_kinetic() is the mean kinetic energy of the RATTLE-projected half-step momenta
    p -+ h F / 2 (the leapfrog average), reported as temp_half."""
    tpl, sys_, pos, H = _cluster()
    sim = FlexibleSimulation(sys_, [tpl] * 8, pos, H, SCL, dt=0.002, constraints="h-bonds", log=None)
    sim.advance(10)
    st = sim.state
    q, p, F = st.dyn.position, st.dyn.momentum, st.dyn.force
    m, M = sim.flex.masses, np.asarray(sim.flex.mass)
    ke = [0.5 * np.sum(np.asarray(sim.constraints.momenta(q, p + s * 0.001 * F, m)) ** 2 / M) for s in (-1.0, 1.0)]
    assert abs(sim.half_step_kinetic() - np.mean(ke)) < 1e-9 * np.mean(ke)
    o = sim.observables()
    assert abs(o["temp_half"] - 2.0 * np.mean(ke) / (sim.integ.dof * KB)) < 1e-8
