"""Alchemical free energies (md/alchemy.py, md/free_energy.py): the lambda-dependent Hamiltonian
(original at lambda = 1, decoupled end state = the box without the solute, dU/dlambda against
finite differences with the dipoles re-solved, soft core finite at overlap, pressure), batched
lambda windows (= sequential, samples, Hamiltonian exchange), the gas-phase leg, the estimators
(MBAR, BAR, TI, statistical inefficiency) on harmonic oscillators with analytic free energies,
and the driver's outputs and restarts."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import alch_box, alch_frame, alch_settings, alch_sim, flex_solute_box, water, water_geometry

from pgm_jax import System
from pgm_jax.analysis import free_energy as fe
from pgm_jax.analysis import stats
from pgm_jax.md.alchemy import (
    Alchemy,
    FreeEnergyRun,
    GasPhaseLeg,
    LambdaWindows,
    alchemical_system,
    standard_schedule,
)
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.box import min_image
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.simulation import Simulation


# ----------------------------------------------------------------------------- Hamiltonian
def test_full_coupling_is_the_original_hamiltonian():
    """lambda = (1, 1): energy, forces, pressure and the Monte Carlo trial energy of the original
    system (the solute's van der Waals moves from the ordinary rows to the soft-core rows)."""
    sim, alch, P, (pos, H, sys0) = alch_sim()
    plain = Simulation(sys0, pos, H, alch_settings(), dt=0.001, log=None)
    assert abs(float(sim.state.epot) - float(plain.state.epot)) < 1e-9 * abs(float(plain.state.epot))
    assert abs(float(sim.state.vdw) - float(plain.state.vdw)) < 1e-9
    F0, F1 = plain.state.dyn.force, sim.state.dyn.force
    assert np.allclose(F0.center, F1.center, atol=1e-8) and np.allclose(
        F0.orientation.vec, F1.orientation.vec, atol=1e-8
    )
    assert abs(sim.pressure() - plain.pressure()) < 1e-6 * max(1.0, abs(plain.pressure()))
    X, Hb, cand = alch_frame(sim)
    e1 = alch.energy(sim.ff, X, Hb, cand, sim.ff.init_induction(), P, None)[0]
    e0 = plain.ff.energy(X, Hb, cand, plain.ff.init_induction())[0]
    assert abs(float(e1) - float(e0)) < 1e-9 * abs(float(e0))


def test_npt_with_an_alchemical_region_follows_the_plain_run():
    """The barostat's trial energy goes through the alchemical Hamiltonian: at lambda = (1, 1) an
    NPT run follows the plain run (same seed and start velocities; the solute copy has its own
    rigid-body frame, so momenta drawn in body frames would differ) step by step."""
    vel = np.random.default_rng(4).normal(size=(192, 3)) * 0.5
    kw = dict(thermostat="bussi", barostat=MonteCarloBarostat(every=4), seed=3, velocities=vel)
    sim, _, _, (pos, H, sys0) = alch_sim(alch_settings(dipole_tol=1e-10), **kw)
    plain = Simulation(sys0, pos, H, alch_settings(dipole_tol=1e-10), dt=0.001, log=None, **kw)
    sim.advance(40)
    plain.advance(40)
    assert int(sim.state.mc[0]) == 10 and int(sim.state.mc[1]) == int(plain.state.mc[1]) > 0
    assert np.allclose(np.asarray(sim.state.box), np.asarray(plain.state.box), rtol=1e-10)
    assert abs(float(sim.state.epot) - float(plain.state.epot)) < 1e-7 * abs(float(plain.state.epot))


@pytest.mark.parametrize("lam", [(0.6, 1.0), (0.3, 0.7), (0.0, 0.4), (0.0, 0.05)])
def test_dudl_matches_finite_differences_with_resolved_dipoles(lam):
    """Hellmann-Feynman: dU/dlambda at the converged dipoles (autodiff at fixed mu) equals the
    central difference of the energy with the dipoles re-solved at every lambda."""
    sim, alch, P, _ = alch_sim()
    X, Hb, cand = alch_frame(sim)
    ff = sim.ff
    U = jax.jit(lambda l: alch.energy(ff, X, Hb, cand, ff.init_induction(), P, l))
    lam = jnp.asarray(lam)
    e, ind, _, _ = U(lam)
    g = np.asarray(alch.dudl(ff, X, Hb, cand, ind.mu, P, lam))
    h = 1e-4
    for j in range(2):
        d = jnp.zeros(2).at[j].set(h)
        fd = (float(U(lam + d)[0]) - float(U(lam - d)[0])) / (2 * h)
        assert abs(fd - g[j]) < 1e-6 * max(1.0, abs(g[j])), (lam, j, fd, g[j])


def test_dudl_at_the_ends():
    """The endpoints lambda_elec = 0 (polarizability at its floor) and 1: one-sided differences
    (second order, from the three points lambda, lambda +- h, 2h) agree with the analytic value."""
    sim, alch, P, _ = alch_sim()
    X, Hb, cand = alch_frame(sim)
    ff = sim.ff
    U = jax.jit(lambda l: alch.energy(ff, X, Hb, cand, ff.init_induction(), P, l)[:2])
    h = 1e-4
    for lam, sgn in (((0.0, 1.0), 1.0), ((1.0, 1.0), -1.0), ((0.0, 0.0), 1.0)):
        lam = jnp.asarray(lam)
        e0, ind = U(lam)
        g = np.asarray(alch.dudl(ff, X, Hb, cand, ind.mu, P, lam))
        j = 0 if lam[1] == 1.0 else 1
        d = jnp.zeros(2).at[j].set(sgn * h)
        fd = sgn * (-3 * float(e0) + 4 * float(U(lam + d)[0]) - float(U(lam + 2 * d)[0])) / (2 * h)
        assert abs(fd - g[j]) < 1e-5 * max(1.0, abs(g[j])), (lam, fd, g[j])
    # lambda_elec = 0: the solute has no charges or dipoles, so its only lambda_elec derivative is the
    # induction limit -(1 - eps) alpha |E|^2 / 2 plus the linear permanent terms: finite
    assert np.isfinite(g).all()


def test_decoupled_end_state_is_the_box_without_the_solute():
    """lambda = (0, 0): the environment's energy and forces are those of the box without the
    solute (up to the polarizability floor), and the solute feels no force."""
    sim, alch, P, (pos, H, sys0) = alch_sim()
    X, Hb, cand = alch_frame(sim)
    ff = sim.ff
    lam = jnp.zeros(2)
    res = jax.jit(lambda: alch.compute(ff, X, Hb, cand, ff.init_induction(), P, lam))()
    env = Simulation(System([water()] * (sys0.nmol - 1)), pos[3:], H, alch_settings(), dt=0.001, log=None)
    ref = jax.jit(env.ff.compute)(X[3:], Hb, env.ff.rows_for(X[3:], Hb), env.ff.init_induction())
    assert abs(float(res.energy["total"]) - float(ref.energy["total"])) < 1e-8 * abs(float(ref.energy["total"]))
    assert abs(float(res.energy["vdw"]) - float(ref.energy["vdw"])) < 1e-9
    assert np.allclose(res.forces[3:], ref.forces, atol=1e-7)
    assert float(jnp.max(jnp.abs(res.forces[:3]))) < 1e-5  # the floor leaves ~1e-8 of the induction


def test_softcore_is_finite_at_overlap_and_lennard_jones_at_one():
    """Soft core: finite energy and zero force at r = 0 for lambda_vdw < 1, U(0) = lambda eps (1/w^2 -
    2/w) with w = sc_alpha (1 - lambda) / 2; Lennard-Jones at lambda_vdw = 1; the long-range
    correction scales with lambda_vdw."""
    sim, alch, P, _ = alch_sim(alch_settings(lj_lrc=True))
    X, Hb, cand = alch_frame(sim)
    Pa = sim.sys.expand(P)
    N = sim.sys.n
    tail = float(alch._tail(Pa, Hb))
    assert tail < 0.0
    one = jnp.full(cand.shape, N, cand.dtype).at[0, 0].set(9)  # the pair (solute O, oxygen 9) only
    Xo = X.at[9:12].add(X[0] - X[9])  # that water's oxygen on the solute's
    E = jax.jit(lambda x, c, lv: alch.softcore_energy(x, Hb, c, P, lv))
    eps = float(Pa["lj_sqrt_eps"][0] * Pa["lj_sqrt_eps"][9])
    for lv in (0.0, 0.2, 0.5, 0.9):
        e, g = jax.value_and_grad(E)(Xo, one, lv)
        w = 0.5 * alch.sc_alpha * (1.0 - lv)
        assert abs(float(e) - lv * (tail + eps * (1.0 / w**2 - 2.0 / w))) < 1e-9 * max(1.0, abs(float(e))), lv
        assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g).max()) < 1e-12
    e, g = jax.value_and_grad(E)(Xo, cand, 0.5)  # with the whole row: finite too
    assert np.isfinite(float(e)) and np.all(np.isfinite(np.asarray(g)))
    # lambda_vdw = 1: plain LJ (Amber form) over the same pairs, plus the full tail
    kk = np.asarray(cand[0])
    kk = kk[(kk < N) & (kk >= 3)]
    x = min_image(X[kk] - X[0], Hb)
    r = np.linalg.norm(np.asarray(x), axis=1)
    rmin = np.asarray(Pa["lj_rmin_half"][0] + Pa["lj_rmin_half"][kk])
    ep = np.asarray(Pa["lj_sqrt_eps"][0] * Pa["lj_sqrt_eps"][kk])
    on = (r < 0.55) & (ep > 0)
    lj = np.sum(ep[on] * ((rmin[on] / r[on]) ** 12 - 2 * (rmin[on] / r[on]) ** 6))
    assert abs(float(E(X, cand, 1.0)) - (lj + tail)) < 1e-9 * max(1.0, abs(lj))
    assert float(E(X, cand, 0.0)) == 0.0


def test_pressure_at_intermediate_lambda_matches_volume_derivative():
    """Alchemy.strain_derivative (molecular virial of the lambda Hamiltonian, no tail) against a
    finite difference of the fixed-mu energy under molecular scaling of centres and box."""
    sim, alch, P, _ = alch_sim(alch_settings(lj_lrc=False))
    X, Hb, cand = alch_frame(sim)
    ff = sim.ff
    lam = jnp.asarray([0.4, 0.6])
    _, ind, _, _ = alch.energy(ff, X, Hb, cand, ff.init_induction(), P, lam)
    W = alch.strain_derivative(ff, X, Hb, cand, ind.mu, P, lam)
    m = ff.masses
    com = (
        jax.ops.segment_sum(m[:, None] * X, ff.mol, sim.sys.nmol)
        / jax.ops.segment_sum(m, ff.mol, sim.sys.nmol)[:, None]
    )

    def e(s):
        return float(alch.energy_fixed_mu(ff, X + ((s - 1.0) * com)[ff.mol], Hb * s, cand, ind.mu, P, lam))

    h = 1e-6
    fd = (e(1 + h) - e(1 - h)) / (2 * h)
    assert abs(fd - float(jnp.trace(W))) < 1e-6 * max(1.0, abs(fd)), (fd, float(jnp.trace(W)))


# ----------------------------------------------------------------------------- windows
def test_windows_batched_equal_sequential_and_exchange():
    sim, alch, P, _ = alch_sim(alch_settings(dipole_tol=1e-9), thermostat="bussi")
    L = standard_schedule(3, [0.5, 0.0])
    wb = LambdaWindows(sim, L, batched=True, seed=1)
    ws = LambdaWindows(sim, L, batched=False, seed=1)
    for k in range(len(L)):
        assert np.allclose(np.asarray(wb.state(k).lam), L[k])
    wb.advance(10)
    ws.advance(10)
    assert np.allclose(wb.potentials(), ws.potentials(), rtol=1e-10)
    ub, gb, _ = wb.sample()
    us, gs, _ = ws.sample()
    assert np.allclose(ub, us, atol=1e-8) and np.allclose(gb, gs, atol=1e-7)
    beta = 1.0 / float(wb.integ.kT)
    assert np.allclose(np.diag(ub), beta * wb.potentials(), atol=1e-7)  # u_k(x_k) = beta U at the window's own lambda
    # dU/dlambda_vdw from the samples: the soft-core term alone
    X, Hb, cand = alch_frame(wb._on(3))
    gv = jax.grad(lambda lv: alch.softcore_energy(X, Hb, cand, P, lv))(L[3, 1])
    assert abs(float(gv) - gb[3, 1]) < 1e-7 * max(1.0, abs(float(gv)))
    # Hamiltonian exchange: slots 1 and 2 swap; each re-evaluated in its own Hamiltonian, energy booked as heat
    src = np.array([0, 2, 1, 3, 4])
    econs = [wb.observables(k)["econs"] for k in range(5)]
    wb.permute(src)
    ws.permute(src)
    e = wb.potentials()
    assert abs(e[1] - ub[1, 2] / beta) < 1e-6 and abs(e[2] - ub[2, 1] / beta) < 1e-6
    assert np.allclose(e, ws.potentials(), rtol=1e-10)
    for k in (0, 3, 4):  # untouched slots
        assert wb.observables(k)["econs"] == econs[k]
    assert np.allclose(np.asarray(wb.state(1).lam), L[1]) and np.allclose(np.asarray(wb.state(2).lam), L[2])
    assert np.allclose(np.asarray(wb.state(1).induction.hist[3]), np.asarray(wb.state(1).induction.mu))


def test_free_energy_run_outputs_and_restart(tmp_path):
    sim, alch, P, _ = alch_sim(alch_settings(dipole_tol=1e-8), thermostat="bussi")
    L = standard_schedule(3, [0.5, 0.0])
    prefix = str(tmp_path / "w")
    run = FreeEnergyRun(LambdaWindows(sim, L, seed=2), sample_every=5, exchange_every=10, log=None, meta={"x": 1})
    s = run.run(40, prefix=prefix, report_every=20, checkpoint_every=20)
    d = fe.load(prefix + "_fe.npz")
    assert d["u"].shape == (8, 5, 5) and d["dudl"].shape == (8, 5, 2) and d["meta"]["x"] == 1
    assert s["exchanges"] == 4 and os.path.exists(prefix + ".fe.chk") and os.path.exists(prefix + "_L03.rst7")
    # continue from the 40-step checkpoint in a new driver: same windows state, samples appended
    run2 = FreeEnergyRun(LambdaWindows(sim, L, seed=5), sample_every=5, exchange_every=10, log=None)
    run2.load_checkpoint(prefix + ".fe.chk")
    assert run2.step == 40 and len(run2.samples["u"]) == 8
    assert np.allclose(run2.windows.potentials(), run.windows.potentials())
    run2.run(10, prefix=prefix, report_every=0)
    assert fe.load(prefix + "_fe.npz")["u"].shape == (10, 5, 5)
    r = fe.estimate(fe.load(prefix + "_fe.npz"), discard_ps=0.0, gas={"delta_g": 1.0, "dudl": np.zeros(5)})
    for m in ("ti", "bar", "mbar"):
        assert np.isfinite(r[m]) and np.isfinite(r[f"dG_hyd_{m}"])


# ----------------------------------------------------------------------------- gas-phase leg
def test_gas_phase_leg_matches_a_lone_molecule_in_a_large_box():
    """E_gas(lambda) of the gas-phase model equals the MD engine's energy of the lone solute in a
    large periodic box (image and PME errors < 1e-3 kJ/mol here); dE_gas/dlambda by autodiff."""
    w = water()
    sysA, P = alchemical_system(System([w]), 0)
    alch = Alchemy(sysA, 0)
    rng = np.random.default_rng(3)
    xyz = water_geometry()
    xyz = xyz @ np.linalg.qr(rng.normal(size=(3, 3)))[0].T + 2.1
    H = np.eye(3) * 4.2
    s = MDSettings().replace(
        precision="double",
        cutoff=1.2,
        skin=0.0,
        ewald_beta=3.0,
        pme_grid=(64, 64, 64),
        pme_order=8,
        dipole_tol=1e-12,
        max_iter=200,
        peek=0.0,
        lj_lrc=False,
    )
    ff = PGMForceField(sysA, H, s)
    alch.check(ff)
    idx = ff.rows_for(xyz, H)
    gas = GasPhaseLeg(alch, xyz, "qpi")
    for le in (1.0, 0.5, 0.0):
        e = float(
            alch.energy(ff, jnp.asarray(xyz), jnp.asarray(H), idx, ff.init_induction(), P, jnp.array([le, 1.0]))[0]
        )
        assert abs(e - gas.energy(le, P)) < 1e-3, (le, e, gas.energy(le, P))
    assert abs(gas.delta_g(P) - (gas.energy(0.0, P) - gas.energy(1.0, P))) < 1e-12 and abs(gas.energy(0.0, P)) < 1e-6
    h = 1e-5
    assert abs(gas.dudl(0.5, P) - (gas.energy(0.5 + h, P) - gas.energy(0.5 - h, P)) / (2 * h)) < 1e-6


# ----------------------------------------------------------------------------- setup errors
def test_refused_setups():
    pos, H, sys0 = alch_box()
    with pytest.raises(ValueError, match="alchemical_system"):
        Alchemy(sys0, 0)  # shares its keys with every other water
    w = water()
    ion = System([w.__class__("ION", ["Na"], ["Na"], [1.0], [0.05], [1e-4])] + [w] * 3)
    sysI, _ = alchemical_system(ion, 0)
    with pytest.raises(NotImplementedError, match="net charge"):
        Alchemy(sysI, 0)
    sysA, P = alchemical_system(sys0, 0)
    with pytest.raises(NotImplementedError, match="vdw"):
        Simulation(sysA, pos, H, alch_settings(vdw="gvdw"), log=None, alchemy=Alchemy(sysA, 0))
    with pytest.raises(ValueError):
        Alchemy(sysA, 0, lam=(1.2, 0.0))
    sim = Simulation(sysA, pos, H, alch_settings(), log=None, params=P, alchemy=Alchemy(sysA, 0), thermostat=None)
    with pytest.raises(ValueError, match="thermostat"):
        LambdaWindows(sim, standard_schedule(2, [0.0]))


# ----------------------------------------------------------------------------- flexible solutes
def intra_lj(tpl, Pa, Y):
    """The template's intramolecular Lennard-Jones (weighted pairs) at positions Y (solute first)."""
    i, j, w = tpl.lj_pairs()
    r = np.linalg.norm(Y[j] - Y[i], axis=1)
    rmin = np.asarray(Pa["lj_rmin_half"])[i] + np.asarray(Pa["lj_rmin_half"])[j]
    eps = np.asarray(Pa["lj_sqrt_eps"])[i] * np.asarray(Pa["lj_sqrt_eps"])[j]
    return float(np.sum(w * eps * ((rmin / r) ** 12 - 2 * (rmin / r) ** 6)))


def test_flexible_solute_hamiltonian():
    """A flexible solute in the flexible engine: lambda = (1, 1) is the original Hamiltonian (its
    intramolecular 1-4 van der Waals moved out of the ordinary rows and back); at lambda = (0, 0)
    the energy is the waters' alone plus the solute's bonded and intramolecular van der Waals energy,
    and the waters feel the forces of the box without the solute."""
    from pgm_jax.md.flexible import FlexibleSimulation

    tpl, sys0, tpls, X, H = flex_solute_box()
    s = alch_settings()
    sysA, P = alchemical_system(sys0, 0)
    alch = Alchemy(sysA, 0)
    plain = FlexibleSimulation(sys0, tpls, X, H, s, log=None, constraints="h-bonds")
    sim = FlexibleSimulation(sysA, tpls, X, H, s, log=None, params=P, alchemy=alch, constraints="h-bonds")
    assert alch._intra is not None and len(alch._intra[0]) == 3  # the three scaled H-C-O-H pairs
    assert abs(float(sim.state.epot) - float(plain.state.epot)) < 1e-9 * abs(float(plain.state.epot))
    assert np.allclose(np.asarray(sim.state.dyn.force), np.asarray(plain.state.dyn.force), atol=1e-7)
    assert abs(sim.pressure() - plain.pressure()) < 1e-6 * max(1.0, abs(plain.pressure()))
    off = sim.integ.forces(sim.state.set(lam=jnp.zeros(2)), False)
    Y = np.asarray(off.dyn.position)
    env = FlexibleSimulation(System(sys0.molecules[1:]), tpls[1:], Y[6:], H, s, log=None, constraints="h-bonds")
    ref = float(env.state.epot) + float(tpl.bonded_energy(jnp.asarray(Y[:6]))) + intra_lj(tpl, sysA.expand(P), Y)
    assert abs(float(off.epot) - ref) < 1e-8 * abs(ref), (float(off.epot), ref)
    assert np.allclose(np.asarray(off.dyn.force)[6:], np.asarray(env.state.dyn.force), atol=1e-6)


def test_keep_intramolecular_rigid_equals_annihilation_plus_gas_leg():
    """intramolecular="keep" on a rigid solute: U_keep(lambda) - U_annihilate(lambda) = E_gas(1) -
    E_gas(lambda) (a constant of the configuration), and dU/dlambda_elec differs by dE_gas/dlambda."""
    sim, ann, P, _ = alch_sim()
    X, Hb, cand = alch_frame(sim)
    ff = sim.ff
    keep = Alchemy(sim.sys, 0, intramolecular="keep")
    keep.check(ff)
    gas = GasPhaseLeg(ann, X[:3], "qpi")
    for lam in ((1.0, 1.0), (0.6, 1.0), (0.0, 0.5)):
        lam = jnp.asarray(lam)
        ea, ind, _, _ = ann.energy(ff, X, Hb, cand, ff.init_induction(), P, lam)
        ek, _, _, _ = keep.energy(ff, X, Hb, cand, ff.init_induction(), P, lam)
        assert abs((float(ek) - float(ea)) - (gas.energy(1.0, P) - gas.energy(float(lam[0]), P))) < 1e-7
        ga = np.asarray(ann.dudl(ff, X, Hb, cand, ind.mu, P, lam))
        gk = np.asarray(keep.dudl(ff, X, Hb, cand, ind.mu, P, lam))
        assert abs(gk[0] - (ga[0] - gas.dudl(float(lam[0]), P))) < 1e-6 and abs(gk[1] - ga[1]) < 1e-9


def test_keep_intramolecular_flexible_solute():
    """A flexible solute with intramolecular="keep": the original Hamiltonian at (1, 1); at (0, 0) the
    waters' energy plus the solute's bonded, intramolecular van der Waals and whole gas-phase
    electrostatic energy (the decoupled state is the gas-phase molecule); dU/dlambda against
    finite differences with the dipoles re-solved."""
    from pgm_jax.md.flexible import FlexibleSimulation

    tpl, sys0, tpls, X, H = flex_solute_box()
    s = alch_settings()
    sysA, P = alchemical_system(sys0, 0)
    alch = Alchemy(sysA, 0, intramolecular="keep")
    plain = FlexibleSimulation(sys0, tpls, X, H, s, log=None, constraints="h-bonds")
    sim = FlexibleSimulation(sysA, tpls, X, H, s, log=None, params=P, alchemy=alch, constraints="h-bonds")
    assert abs(float(sim.state.epot) - float(plain.state.epot)) < 1e-9 * abs(float(plain.state.epot))
    assert np.allclose(np.asarray(sim.state.dyn.force), np.asarray(plain.state.dyn.force), atol=1e-7)
    off = sim.integ.forces(sim.state.set(lam=jnp.zeros(2)), False)
    Y = np.asarray(off.dyn.position)
    env = FlexibleSimulation(System(sys0.molecules[1:]), tpls[1:], Y[6:], H, s, log=None, constraints="h-bonds")
    gas = GasPhaseLeg(alch, Y[:6], "qpi")
    ref = (
        float(env.state.epot)
        + float(tpl.bonded_energy(jnp.asarray(Y[:6])))
        + intra_lj(tpl, sysA.expand(P), Y)
        + gas.energy(1.0, P)
    )
    assert abs(float(off.epot) - ref) < 1e-8 * abs(ref), (float(off.epot), ref)
    ff = sim.ff
    cand = sim.integ.nb.candidates(off.nbr, sim.flex.list_centers(jnp.asarray(Y)), off.box, jnp.asarray(Y))[0]
    U = jax.jit(lambda l: alch.energy(ff, jnp.asarray(Y), off.box, cand, ff.init_induction(), P, l))
    lam = jnp.asarray([0.5, 0.7])
    _, ind, _, _ = U(lam)
    g = np.asarray(alch.dudl(ff, jnp.asarray(Y), off.box, cand, ind.mu, P, lam))
    h = 1e-4
    for j in range(2):
        d = jnp.zeros(2).at[j].set(h)
        fd = (float(U(lam + d)[0]) - float(U(lam - d)[0])) / (2 * h)
        assert abs(fd - g[j]) < 1e-6 * max(1.0, abs(g[j])), (j, fd, g[j])


def test_flexible_windows_and_lone_solute_gas_leg():
    """Batched windows on the flexible engine (= sequential; u_k(x_k) = beta U), and the gas-phase leg
    of a flexible solute: the lone molecule in a 4.2 nm box (lone_solute) has the electrostatic
    energy of GasPhaseLeg at every lambda_elec (bonded and intramolecular terms cancel in
    E(lambda) - E(0))."""
    from pgm_jax.md.alchemy import lone_solute
    from pgm_jax.md.flexible import FlexibleSimulation

    tpl, sys0, tpls, X, H = flex_solute_box()
    sysA, P = alchemical_system(sys0, 0)
    sim = FlexibleSimulation(
        sysA,
        tpls,
        X,
        H,
        alch_settings(dipole_tol=1e-9),
        dt=0.001,
        log=None,
        params=P,
        alchemy=Alchemy(sysA, 0),
        constraints="h-bonds",
        thermostat="bussi",
    )
    L = standard_schedule(2, [0.4, 0.0])
    wb, ws = LambdaWindows(sim, L, seed=1), LambdaWindows(sim, L, batched=False, seed=1)
    wb.advance(6)
    ws.advance(6)
    ub, gb, _ = wb.sample()
    us, gs, _ = ws.sample()
    assert np.allclose(ub, us, atol=1e-7) and np.allclose(gb, gs, atol=1e-6)
    assert np.allclose(np.diag(ub), wb.potentials() / float(wb.integ.kT), atol=1e-7)
    sub, x, Hg = lone_solute(sysA, 0, np.asarray(wb.state(0).dyn.position), 4.2)
    sg = MDSettings().replace(
        precision="double",
        cutoff=1.2,
        skin=0.0,
        ewald_beta=3.0,
        pme_grid=(64, 64, 64),
        pme_order=8,
        dipole_tol=1e-12,
        max_iter=200,
        peek=0.0,
        lj_lrc=False,
        neighbor_list="atom",
    )
    alch_g = Alchemy(sub, 0)
    gsim = FlexibleSimulation(sub, [tpl], x, Hg, sg, log=None, params=P, alchemy=alch_g, thermostat=None)

    def E(le):
        return float(gsim.integ.forces(gsim.state.set(lam=jnp.array([le, 1.0])), False).epot)

    gas = GasPhaseLeg(alch_g, np.asarray(gsim.state.dyn.position), "qpi")
    e0 = E(0.0)
    for le in (1.0, 0.4):
        assert abs((E(le) - e0) - (gas.energy(le, P) - gas.energy(0.0, P))) < 2e-3, le


# ----------------------------------------------------------------------------- estimators
def _harmonic(rng, K, x0, n):
    xs = [x0[k] + rng.normal(size=n) / np.sqrt(K[k]) for k in range(len(K))]
    X = np.concatenate(xs)
    return xs, 0.5 * K[:, None] * (X[None, :] - x0[:, None]) ** 2


def test_mbar_bar_ti_on_harmonic_oscillators():
    """States u_k = K_k (x - x0_k)^2 / 2 (beta = 1): f_k = ln(K_k / K_0) / 2.  MBAR and BAR within
    their error bars, the error bars match the spread over independent repeats, TI along
    K(lambda) = K_0 + lambda (K_1 - K_0) converges to the same answer."""
    K = np.array([1.0, 1.6, 2.6, 4.2, 6.8])
    x0 = np.array([0.0, 0.1, -0.1, 0.2, 0.0])
    exact = 0.5 * np.log(K / K[0])
    rng = np.random.default_rng(11)
    est, errs, bars, bar_errs = [], [], [], []
    for _rep in range(40):
        xs, u = _harmonic(rng, K, x0, 400)
        f, Th = fe.mbar(u, np.full(len(K), 400))
        D, dD = fe.mbar_differences(f, Th)
        est.append(D[0, -1])
        errs.append(dD[0, -1])
        wF = 0.5 * K[1] * (xs[0] - x0[1]) ** 2 - 0.5 * K[0] * (xs[0] - x0[0]) ** 2
        wR = 0.5 * K[0] * (xs[1] - x0[0]) ** 2 - 0.5 * K[1] * (xs[1] - x0[1]) ** 2
        b, db = fe.bar(wF, wR)
        bars.append(b)
        bar_errs.append(db)
    est, bars = np.array(est), np.array(bars)
    assert abs(est.mean() - exact[-1]) < 3 * est.std() / np.sqrt(len(est))
    assert 0.7 < np.mean(errs) / est.std() < 1.4, (np.mean(errs), est.std())
    assert abs(bars.mean() - exact[1]) < 3 * bars.std() / np.sqrt(len(bars))
    assert 0.7 < np.mean(bar_errs) / bars.std() < 1.4, (np.mean(bar_errs), bars.std())
    # MBAR with an unsampled intermediate state
    xs, u = _harmonic(rng, K, x0, 4000)
    keep = np.concatenate([np.arange(4000), np.arange(8000, 20000)])
    f, _ = fe.mbar(u[:, keep], np.array([4000, 0, 4000, 4000, 4000]))
    assert abs(f[1] - exact[1]) < 0.02
    # TI: <dU/dlambda> = (K_1 - K_0) <x^2> / 2 = (K_1 - K_0) / (2 K(lambda)), exact integral ln(K_1/K_0)/2
    lam = np.linspace(0.0, 1.0, 41)
    Kl = K[0] + lam * (K[-1] - K[0])
    means = (K[-1] - K[0]) / (2.0 * Kl)
    v, e = fe.ti(lam[:, None], means[:, None])
    assert abs(v - exact[-1]) < 2e-3 and e == 0.0
    # two-dimensional paths: the segments add up
    L2 = np.array([[1.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 0.0]])
    G2 = np.array([[2.0, 7.0], [1.0, 7.0], [0.0, 3.0], [5.0, 1.0]])
    assert abs(fe.ti(L2, G2)[0] - (-(0.5 * 1.5 + 0.5 * 0.5) - 0.5 * (3.0 + 1.0))) < 1e-12


def test_statistical_inefficiency_and_equilibration():
    rng = np.random.default_rng(5)
    for phi in (0.0, 0.8, 0.95):
        a = np.zeros(200000)
        e = rng.normal(size=a.size)
        for i in range(1, a.size):
            a[i] = phi * a[i - 1] + e[i]
        g = stats.statistical_inefficiency(a)
        exact = (1 + phi) / (1 - phi)
        assert abs(g - exact) < 0.1 * exact, (phi, g, exact)
    a = np.concatenate([np.linspace(8.0, 0.0, 300), rng.normal(size=3000)])
    t0, g, neff = stats.detect_equilibration(a, nskip=10)
    assert 200 <= t0 <= 400 and neff > 2000
    assert list(stats.subsample(10, 2.3)) == [0, 3, 6, 9]
