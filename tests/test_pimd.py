"""Path-integral MD (md/pimd.py): normal modes, contraction, exact free ring-polymer propagation,
estimators against the exact discretised harmonic oscillator, thermostat mode temperatures, RPMD
energy conservation, the pGM bead engine (vmapped beads = beads one by one, contraction forces =
-dU/dq, contraction to P beads = no contraction) and the flexible-water fit."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_grad import water  # noqa: E402

from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec  # noqa: E402
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.md.pimd import (  # noqa: E402
    HBAR,
    WATER_FAMILIES,
    PIMDIntegrator,
    PIMDSimulation,
    PotentialEngine,
    RingPolymer,
    contraction_matrix,
    flexible_water,
    harmonic_frequencies,
    normal_modes,
    qtip4pf_intra,
    water_geometry,
)
from pgm_jax.system import System  # noqa: E402

T = 300.0


def exact_ho(P, w, T=T):
    """<V> = <K> per degree of freedom of the P-bead discretised harmonic oscillator (kJ/mol)."""
    r = RingPolymer(P, T)
    return 0.5 * KB * T * float(np.sum(w**2 / (r.omega**2 + w**2)))


def test_normal_modes_and_contraction():
    for P in (1, 2, 5, 8):
        C, idx = normal_modes(P)
        assert np.allclose(C.T @ C, np.eye(P), atol=1e-12)
        S = 2 * np.eye(P) - np.roll(np.eye(P), 1, 0) - np.roll(np.eye(P), -1, 0)  # sum (q_k - q_k+1)^2
        lam = np.diag(C.T @ S @ C)
        assert np.allclose(C.T @ S @ C, np.diag(lam), atol=1e-12)
        assert np.allclose(lam, 4 * np.sin(np.pi * idx / P) ** 2, atol=1e-12)
        r = RingPolymer(P, T)
        x = jnp.asarray(np.random.default_rng(P).normal(size=(P, 4, 3)))
        assert np.allclose(r.from_nm(r.to_nm(x)), x, atol=1e-13)
        assert np.allclose(
            np.asarray(r.spring(x, jnp.ones((4, 1)))).sum(),
            0.5 * r.omega_P**2 * float(jnp.sum((x - jnp.roll(x, -1, 0)) ** 2)),
        )
    for P, Pc in ((8, 1), (8, 3), (8, 4), (16, 5), (6, 6)):
        Tm = contraction_matrix(P, Pc)
        assert np.allclose(Tm @ Tm.T, Pc / P * np.eye(Pc), atol=1e-12)
        assert np.allclose(Tm.sum(1), 1.0)  # a rigid shift stays a shift
        if Pc == P:
            assert np.allclose(Tm, np.eye(P))
    j = np.arange(16)
    q = 0.3 + np.cos(2 * np.pi * j / 16) - 0.5 * np.sin(2 * np.pi * j / 16)  # a smooth (l <= 1) path
    jc = np.arange(5)
    assert np.allclose(
        contraction_matrix(16, 5) @ q, 0.3 + np.cos(2 * np.pi * jc / 5) - 0.5 * np.sin(2 * np.pi * jc / 5)
    )


def test_potential_engine_contraction():
    """Contracted soft potential: forces are -dU/dq, P' = P is no contraction, P' = 1 gives every bead
    the centroid force (the model of scripts/pimd_openmm.py, checked there against OpenMM)."""
    P, n = 8, 5
    stiff = lambda x, box: jnp.sum(1e4 * x[:, 0] ** 2 + 3e5 * x[:, 0] ** 4)  # noqa: E731
    soft = lambda x, box: jnp.sum(50.0 * jnp.sum(x * x, -1) + 400.0 * x[:, 1] ** 3)  # noqa: E731
    q = 0.05 * jax.random.normal(jax.random.PRNGKey(0), (P, n, 3))
    box = jnp.eye(3)
    full = PotentialEngine(lambda x, b: stiff(x, b) + soft(x, b)).compute(q, box, None)
    same = PotentialEngine(stiff, soft=soft, contract=P).compute(q, box, None)
    assert np.allclose(full[0], same[0]) and abs(float(full[1] - same[1])) < 1e-10
    for Pc in (1, 3, 4):
        eng = PotentialEngine(stiff, soft=soft, contract=Pc)
        f, U, _ = eng.compute(q, box, None)
        g = jax.grad(lambda y: eng.compute(y, box, None)[1])(q)
        assert np.allclose(f, -g, atol=1e-9)
    f1 = PotentialEngine(lambda x, b: 0.0 * jnp.sum(x), soft=soft, contract=1).compute(q, box, None)[0]
    fc = -jax.grad(lambda x: soft(x, box))(jnp.mean(q, 0))
    assert np.allclose(f1, jnp.broadcast_to(fc, f1.shape))


@pytest.mark.parametrize("kind", ["exact", "cayley"])
def test_free_ring_polymer_propagation(kind):
    """V = 0, no thermostat: 'exact' reproduces the analytic normal-mode motion; both conserve the
    energy of every mode exactly."""
    P, m, dt, n = 8, 2.0, 0.001, 25
    eng = PotentialEngine(lambda x, box: 0.0 * jnp.sum(x))
    integ = PIMDIntegrator(eng, [m], P, T, dt, mode="rpmd", propagator=kind)
    st = integ.init(jnp.zeros((1, 3)), jnp.eye(3), jax.random.PRNGKey(3))
    r = integ.ring
    q0, p0 = r.to_nm(st.q), r.to_nm(st.p)
    st1 = integ.run(st, n)
    q1, p1 = r.to_nm(st1.q), r.to_nm(st1.p)
    w = r.omega[:, None, None]
    e = lambda q, p: 0.5 * p * p / m + 0.5 * m * w * w * q * q  # noqa: E731
    assert np.allclose(e(q1, p1), e(q0, p0), rtol=1e-10, atol=1e-12)
    if kind == "exact":
        t = n * dt
        safe = np.where(w > 0, w, 1.0)
        qa = q0 * np.cos(w * t) + np.where(w > 0, p0 * np.sin(w * t) / (m * safe), p0 * t / m)
        assert np.allclose(q1, qa, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("thermostat", ["pile-l", "pile-g"])
def test_harmonic_oscillator_estimators(thermostat):
    """<V>, primitive and centroid-virial <K> of 3D harmonic oscillators (beta hbar omega = 2.5)
    against the exact P-bead values; the exact quantum limit is approached as 1/P^2."""
    w, m, n = 100.0, 1.0, 64
    for P in (1, 8):
        eng = PotentialEngine(lambda x, box: 0.5 * m * w**2 * jnp.sum(x * x))
        integ = PIMDIntegrator(eng, np.full(n, m), P, T, 0.0005, "pimd", thermostat, tau0=0.02)
        st = integ.run(integ.init(jnp.zeros((n, 3)), jnp.eye(3), jax.random.PRNGKey(P)), 2000)
        est = jax.jit(integ.estimators)
        V, Kp, Kc = [], [], []

        def body(st, _):
            st = integ._run(st, 25)
            e = integ.estimators(st)
            return st, jnp.stack([e["epot"], e["prim"].sum(), e["cv"].sum()])

        st, X = jax.jit(lambda s: jax.lax.scan(body, s, None, length=400))(st)
        X = np.asarray(X) / (3 * n)
        ref = exact_ho(P, w)
        for k in range(3):
            assert abs(X[:, k].mean() / ref - 1) < 0.025, (P, k, X[:, k].mean(), ref)
        if P == 1:
            assert np.allclose(X[:, 1:], 0.5 * KB * T)  # both estimators are kT/2 exactly
        del est
    q = 0.25 * HBAR * w / math.tanh(HBAR * w / (2 * KB * T))
    assert abs(exact_ho(64, w) - q) < 0.02 * abs(exact_ho(8, w) - q)  # 1/P^2 convergence


def test_free_particle_mode_temperatures():
    """V = 0: PILE-L gives every normal mode (centroid included) the temperature T_P and the
    internal modes the free ring-polymer spread kT_P / (m omega_l^2); TRPMD leaves the centroid
    momentum alone."""
    P, n, m = 8, 128, 1.008
    eng = PotentialEngine(lambda x, box: 0.0 * jnp.sum(x))
    integ = PIMDIntegrator(eng, np.full(n, m), P, T, 0.0005, "pimd", "pile-l", tau0=0.05)
    st = integ.run(integ.init(jnp.zeros((n, 3)), jnp.eye(3), jax.random.PRNGKey(0)), 500)
    r = integ.ring

    def body(st, _):
        st = integ._run(st, 20)
        e = integ.estimators(st)
        qn = r.to_nm(st.q)
        return st, (e["t_modes"], jnp.mean(qn * qn, axis=(1, 2)))

    st, (tm, q2) = jax.jit(lambda s: jax.lax.scan(body, s, None, length=300))(st)
    tm, q2 = np.asarray(tm).mean(0), np.asarray(q2).mean(0)
    assert np.all(np.abs(tm / T - 1) < 0.03), tm
    assert np.all(np.abs(q2[1:] / (r.kT_P / (m * r.omega[1:] ** 2)) - 1) < 0.05), q2
    integ.set_thermostat("trpmd")
    st1 = integ.run(st, 200)
    assert np.allclose(np.asarray(st1.p).sum(0), np.asarray(st.p).sum(0), atol=1e-10)  # centroid momentum


def test_rpmd_energy_conservation():
    """NVE ring polymer in an anharmonic potential: H_P conserved to O(dt^2), no drift."""
    P, n = 8, 16

    def V(x, box):
        return jnp.sum(0.5 * 1e4 * x * x + 2e5 * x**4)

    integ = PIMDIntegrator(PotentialEngine(V), np.full(n, 1.008), P, T, 0.0002, "rpmd")
    st = integ.init(0.01 * jax.random.normal(jax.random.PRNGKey(1), (n, 3)), jnp.eye(3), jax.random.PRNGKey(2))
    est = jax.jit(integ.estimators)
    E = []
    for _ in range(50):
        st = integ.run(st, 100)
        E.append(float(est(st)["econs"]))
    E = np.array(E)
    ke = 1.5 * n * P * integ.ring.kT_P
    assert E.std() < 2e-4 * ke and abs(E[-1] - E[0]) < 5e-4 * ke, (E.std() / ke, (E[-1] - E[0]) / ke)


# ----------------------------------------------------------------------------- pGM engine
def _water_template():
    m = water()
    t = math.radians(104.5)
    x = np.array([[0, 0, 0], [0.0957, 0, 0], [0.0957 * math.cos(t), 0.0957 * math.sin(t), 0]])
    spec = MolSpec("WAT", ["O", "H", "H"], [(0, 1), (0, 2)], [1, 1], 0, x, m)
    model = BondedModel([spec], BondedSettings(families=WATER_FAMILIES))
    P = model.init_params()
    P["bond_quartic"]["K2"] = jnp.full_like(P["bond_quartic"]["K2"], 4.5e5)
    P["angle_harm"]["Ka"] = jnp.full_like(P["angle_harm"]["Ka"], 350.0)
    return FlexibleTemplate.from_fit(model, P), x


def _water_box(n_side=2, L=1.5, seed=0):
    tpl, x = _water_template()
    rng = np.random.default_rng(seed)
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                c = (np.array([i, j, k]) + 0.5) * L / n_side + rng.normal(scale=0.02, size=3)
                pos.append((x - x.mean(0)) @ R.T + c)
    n = n_side**3
    return tpl, System([tpl.pgm] * n), np.concatenate(pos), np.eye(3) * L


def _sim(**kw):
    tpl, sys, pos, H = _water_box()
    s = MDSettings(precision="double", dipole_tol=1e-10, cutoff=0.5, skin=0.05, lj_lrc=False, max_iter=200)
    return FlexibleSimulation(
        sys, [tpl] * sys.nmol, pos, H, s, dt=0.0002, ensemble="nvt", temperature=T, thermostat="bussi", log=None, **kw
    )


def test_pgm_beads_match_single_evaluations():
    sim = _sim()
    P = 4
    pi = PIMDSimulation(sim, beads=P, log=None, seed=1)
    st = pi.state
    ind = sim.ff.init_induction()
    for k in range(P):
        F, res, _ = sim.integ._forces(st.q[k], st.box, ind, sim.state.nbr, True)
        assert np.abs(np.asarray(F) - np.asarray(st.f[k])).max() < 1e-7 * np.abs(np.asarray(F)).max()
    U = sum(float(sim.integ._forces(st.q[k], st.box, ind, sim.state.nbr, True)[1].energy["total"]) for k in range(P))
    assert abs(U - float(st.upot)) < 1e-8 * abs(U)


def test_bead_chunks_match_vmap():
    """Beads evaluated in lax.map chunks of vmapped beads = all beads vmapped at once."""
    sim = _sim()
    a = PIMDSimulation(sim, beads=4, log=None, seed=5)
    b = PIMDSimulation(sim, beads=4, log=None, seed=5, bead_chunk=2)
    assert np.allclose(a.state.f, b.state.f, atol=1e-9) and abs(float(a.state.upot - b.state.upot)) < 1e-8
    a._advance(20)
    b._advance(20)
    assert np.allclose(a.state.q, b.state.q, atol=1e-9)
    assert int(a.state.eng.induction.count) == int(b.state.eng.induction.count) == 21


def test_contraction_forces_and_identity():
    sim = _sim()
    P = 4
    full = PIMDSimulation(sim, beads=P, log=None, seed=2)
    same = PIMDSimulation(sim, beads=P, contract=P, log=None, seed=2)  # P' = P: no contraction
    assert same.engine.Pc is None and np.allclose(same.state.f, full.state.f)
    for Pc in (1, 2):
        rpc = PIMDSimulation(sim, beads=P, contract=Pc, log=None, seed=2)
        st = rpc.state
        eng = rpc.engine
        f = lambda q: eng.compute(q, st.box, st.eng)  # noqa: E731
        d = jnp.asarray(np.random.default_rng(Pc).normal(size=st.q.shape))
        h = 1e-5
        dU = (float(f(st.q + h * d)[1]) - float(f(st.q - h * d)[1])) / (2 * h)
        assert abs(dU + float(jnp.sum(st.f * d))) < 2e-6 * float(jnp.sum(jnp.abs(st.f * d))), (Pc, dU)
        # the monomer reference makes the contracted U close to the full one for a compact polymer
        assert abs(float(st.upot) - float(full.state.upot)) < 0.02 * abs(float(full.state.upot))


def test_pgm_rpmd_conserves_ring_polymer_energy():
    """RPMD of pGM water (NVE ring polymer): the fluctuation of H_P is a second-order integration
    error (a quarter when dt is halved) and small at dt = 0.05 fs."""
    sd = []
    for dt in (1e-4, 5e-5):
        pi = PIMDSimulation(_sim(), beads=4, mode="rpmd", log=None, seed=3, dt=dt)
        E = []
        for _ in range(4):
            pi._advance(int(round(4e-3 / dt / 4)))
            E.append(pi.observables()["econs"])
        sd.append(np.std(E))
    ke = 1.5 * pi.sim.sys.n * 4 * 4 * KB * T
    assert 3.0 < sd[0] / sd[1] < 5.5 and sd[1] < 5e-4 * ke, (sd, sd[1] / ke)
    assert np.isfinite(pi.pressure()) and pi.observables()["ke_H_cv_meV"] > 0


def test_pgm_npt_barostat():
    """Monte Carlo trial energy = the U of the force evaluation at the same state; at 3 kbar the box
    of a dilute water system shrinks, with accepted moves."""
    sim = _sim()
    pi = PIMDSimulation(
        sim,
        beads=2,
        log=None,
        seed=4,
        ensemble="npt",
        pressure=3000.0,
        barostat_interval=5,
        thermostat="pile-g",
        tau0=0.05,
        bead_margin=0.05,
    )
    st = pi.state
    U, _ = jax.jit(pi.engine.energy)(st.q, st.box, st.eng)
    assert abs(float(U) - float(st.upot)) < 1e-7 * abs(float(st.upot))
    q2 = pi.engine.scale(st.q, 1.01)
    qc, qc2 = np.asarray(jnp.mean(st.q, 0)), np.asarray(jnp.mean(q2, 0))
    X, X2 = qc.reshape(-1, 3, 3), qc2.reshape(-1, 3, 3)
    assert np.allclose(np.linalg.norm(X[:, 1] - X[:, 0], axis=1), np.linalg.norm(X2[:, 1] - X2[:, 0], axis=1))
    V0 = pi.observables()["volume_nm3"]
    pi._advance(400)
    o = pi.observables()
    assert o["mc_accept"] > 0 and o["volume_nm3"] < V0, (o["mc_accept"], V0, o["volume_nm3"])


def test_flexible_water_fit_reproduces_target():
    tpl, rep = flexible_water(water(), n_samples=300)
    assert rep["rms_energy_kJmol"] < 1.0
    assert np.allclose(rep["freq_fit_cm"], rep["freq_target_cm"], rtol=0.03)
    x = water_geometry(0.096, 0.093, 104.0)
    assert abs(float(tpl.model.energy(0, jnp.asarray(x), jax.tree_util.tree_map(jnp.asarray, tpl.P))[0])) < 1e6
    f = harmonic_frequencies(qtip4pf_intra, water_geometry(0.09419, 0.09419, 107.4), [16.0, 1.008, 1.008])
    assert 1500 < f[0] < 1650 and 3800 < f[1] < 3950
