"""Extended-Lagrangian induced dipoles (MDSettings.iel, docs/iel.md): exact shadow forces of
iEL/0-SCF (finite differences), second-order shadow energy error, time reversibility of the
auxiliary-dipole propagation, agreement with converged SCF along a trajectory, energy conservation
in a tiny box, iEL/SCF, the barostat path and the flexible engine."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_grad import water  # noqa: E402
from test_md import small_box  # noqa: E402
from test_md_macro import _water_box  # noqa: E402

from pgm_jax import System  # noqa: E402
from pgm_jax.md.forcefield import MDSettings, PGMForceField  # noqa: E402
from pgm_jax.md.iel import spectral_radius  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.md.neighbors import Neighbors  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402


def _settings(**kw):
    base = dict(cutoff=0.55, skin=0.05, pme_grid=(24, 24, 24), pme_order=6, precision="double", dipole_tol=1e-10,
                max_iter=200)
    base.update(kw)
    return MDSettings(**base)


def _water(**kw):
    pos, H, _ = _water_box()
    return System([water()] * (len(pos) // 3)), pos, H


def test_niklasson_recurrence_is_stable_on_the_pgm_water_spectrum():
    # eigenvalues of alpha A for 512 pGM waters: 0.68 - 1.85 (docs/iel.md)
    for K in (0, 3, 4, 5, 6, 7):
        lam = np.linspace(0.68, 1.85, 40)
        rho = [spectral_radius(l, K) for l in lam]
        assert max(rho) < 1.0 + 1e-9, (K, max(rho))
        if K > 0:
            assert max(rho) < 0.9999
    assert spectral_radius(1.0, 5) < 0.92
    assert spectral_radius(2.2, 5) > 1.0                       # kappa lam beyond the stability limit


def _ff_state(settings, seed=1):
    sys, pos, H = small_box(seed)
    ff = PGMForceField(sys, H, settings)
    idx = Neighbors(sys.n, H, settings.cutoff, settings.skin).allocate(pos, None, H).idx
    return ff, jnp.asarray(pos), jnp.asarray(H), idx


@pytest.mark.parametrize("omega,precond", [(1.0, "jacobi"), (0.8, "jacobi"), (1.0, "block"), (0.9, "block")])
def test_shadow_forces_are_exact_and_energy_error_second_order(omega, precond):
    s = _settings(cutoff=0.6, pme_grid=(48, 48, 48), pme_order=8, ewald_beta=6.0, iel="0scf", iel_omega=omega,
                  iel_precond=precond)
    ff, pos, H, idx = _ff_state(s)
    comp = jax.jit(ff.compute)
    ref = comp(pos, H, idx, ff.init_induction())               # warm-up step: converged dipoles
    mu_star, e_star = ref.induction.mu, float(ref.energy["total"])
    assert int(ref.iterations) > 5
    rng = np.random.default_rng(0)
    noise = jnp.asarray(rng.normal(size=mu_star.shape)) * float(jnp.sqrt(jnp.mean(mu_star ** 2)))

    def shadow(eps):
        ind = ref.induction.set(count=jnp.asarray(100, jnp.int32), xl=ref.induction.xl.at[0].set(mu_star + eps * noise))
        return ind

    ind = shadow(5e-2)
    res = comp(pos, H, idx, ind)
    assert int(res.iterations) == 0
    P = ff._atoms(None)
    _, F_hf = ff._energy_forces(pos, H, res.induction.mu, ff.geometry(pos, H, idx, P, forces=True), P)
    E = jax.jit(lambda y: ff.compute(y, H, idx, ind).energy["total"])
    h = 3e-6                                                   # FD error ~ 1e-4 (h^2)
    for k in range(3):
        v = jnp.asarray(rng.normal(size=pos.shape))
        fd = (float(E(pos + h * v)) - float(E(pos - h * v))) / (2 * h)
        an = -float(jnp.sum(res.forces * v))
        hf = -float(jnp.sum(F_hf * v))                         # fixed-dipole forces at mu = x + delta alone
        assert abs(fd - an) < 1e-7 * abs(an) + 1e-3, (fd, an)
        assert abs(fd - hf) > 10 * abs(fd - an), (fd, an, hf)
    # U~ - U* is second order in the error of x (and below the fixed-dipole energy at x)
    d1 = float(comp(pos, H, idx, shadow(1e-2)).energy["total"]) - e_star
    d2 = float(comp(pos, H, idx, shadow(5e-3)).energy["total"]) - e_star
    assert abs(d1) > 1e-6 and 3.0 < d1 / d2 < 5.0, (d1, d2)
    # the dipoles after the Jacobi step are closer to mu* than x
    err_x = float(jnp.linalg.norm(5e-2 * noise))
    err_mu = float(jnp.linalg.norm(res.induction.mu - mu_star))
    assert err_mu < 0.9 * err_x, (err_mu, err_x)


def _reverse(sim):
    st = sim.state
    X = st.induction.xl
    Xr = X.at[0].set(X[2]).at[2].set(X[0])
    dyn = st.dyn.set(momentum=jax.tree_util.tree_map(lambda p: -p, st.dyn.momentum))
    sim.state = st.set(dyn=dyn, induction=st.induction.set(xl=Xr))


@pytest.mark.parametrize("mode", ["0scf", "scf"])
def test_time_reversibility_without_dissipation(mode):
    sys, pos, H = _water()
    s = _settings(iel=mode, iel_order=0, iel_iter=2)
    sim = Simulation(sys, pos, H, s, dt=0.0005, ensemble="nve", log=None, seed=3)
    sim._advance(10)                                           # past the warm-up
    x0 = sim.positions_nm()
    sim._advance(60)
    moved = np.abs(sim.positions_nm() - x0).max()
    _reverse(sim)
    sim._advance(60)
    back = np.abs(sim.positions_nm() - x0).max()
    assert moved > 1e-3 and back < 1e-9 * max(moved, 1.0) * 1e3, (moved, back)


def test_dissipation_breaks_reversibility_only_slightly():
    sys, pos, H = _water()
    s = _settings(iel="0scf", iel_order=5)
    sim = Simulation(sys, pos, H, s, dt=0.0005, ensemble="nve", log=None, seed=3)
    sim._advance(10)
    x0 = sim.positions_nm()
    sim._advance(60)
    _reverse(sim)
    sim._advance(60)
    assert np.abs(sim.positions_nm() - x0).max() < 1e-4


def _trajectory(settings, n=8, block=25, dt=0.001, seed=5, ensemble="nve"):
    sys, pos, H = _water()
    sim = Simulation(sys, pos, H, settings, dt=dt, ensemble=ensemble, log=None, seed=seed, temperature=300.0,
                     thermostat="bussi", tau_t=0.1)
    return sim


def test_dipoles_and_energy_follow_the_converged_solution():
    s = _settings(iel="0scf")
    sim = _trajectory(s)
    ref = PGMForceField(sim.sys, np.asarray(sim.state.box), dataclasses.replace(s, iel="none"))
    ref.mc = sim.ff.mc
    solve = jax.jit(lambda pos, H, idx: ref.compute(pos, H, idx, ref.init_induction()))
    rel, de = [], []
    sim._advance(20)
    for _ in range(8):
        sim._advance(25)
        st = sim.state
        pos = sim.rigid.positions(st.dyn.position)
        idx = sim.nb.candidates(st.nbr, st.dyn.position.center, st.box, pos)[0]
        r = solve(pos, st.box, idx)
        mu = st.induction.mu
        rel.append(float(jnp.sqrt(jnp.mean((mu - r.induction.mu) ** 2) / jnp.mean(r.induction.mu ** 2))))
        de.append(float(st.epot - r.energy["total"]))
    assert int(sim.state.iters) == 0
    assert max(rel) < 2e-3, rel
    assert max(abs(x) for x in de) < 0.05, de                   # kJ/mol for 64 waters (|U| ~ 2500)


def _drift_and_noise(settings, dt=0.001, n=12, block=50):
    sim = _trajectory(settings, dt=dt, ensemble="nvt")
    sim._advance(100)                                          # relax the lattice start (Bussi 0.1 ps)
    nve = Simulation(sim.sys, sim.positions_nm(), np.asarray(sim.state.box), settings, dt=dt, ensemble="nve",
                     log=None, vel_nm_ps=sim.velocities_nm_ps())
    E = []
    nve._advance(20)
    for _ in range(n):
        nve._advance(block)
        E.append(nve.observables()["etot"])
    E = np.array(E)
    t = np.arange(n) * block * dt
    slope = np.polyfit(t, E, 1)[0] * 1000.0 / nve.integ.dof / (KB * 300.0)      # kT / ns / dof
    return slope, np.std(E - np.polyval(np.polyfit(t, E, 1), t)) / (0.5 * nve.integ.dof * KB * 300.0)


def test_energy_conservation_in_a_tiny_box():
    base = _settings()
    d_scf, n_scf = _drift_and_noise(base)
    d_iel, n_iel = _drift_and_noise(dataclasses.replace(base, iel="0scf"))
    # shadow Hamiltonian conserved as well as the converged one (both dominated by the 1 fs integrator)
    assert abs(d_iel) < max(3 * abs(d_scf), 0.05), (d_iel, d_scf)
    assert n_iel < 3 * n_scf + 1e-5, (n_iel, n_scf)


def test_iel_scf_modes():
    base = _settings(dipole_tol=1e-8)
    sim = _trajectory(dataclasses.replace(base, iel="scf", iel_iter=0))       # CG from x to tolerance
    sim._advance(40)
    it_x = float(sim.state.cg_total) / 40
    ref = _trajectory(base)
    ref._advance(40)
    it_mu4 = float(ref.state.cg_total) / 40
    assert it_x < it_mu4 + 3, (it_x, it_mu4)
    sim2 = _trajectory(dataclasses.replace(base, iel="scf", iel_iter=2))
    sim2._advance(40)
    assert int(sim2.state.iters) == 2 and np.isfinite(sim2.observables()["etot"])


def test_barostat_and_flexible_engine():
    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    sys, pos, H = _water()
    s = _settings(iel="0scf", dipole_tol=1e-8)
    sim = Simulation(sys, pos, H * 1.02, s, dt=0.001, ensemble="npt", barostat_interval=5, log=None,
                     thermostat="bussi", tau_t=0.1, seed=2)
    sim._advance(100)
    o = sim.observables()
    assert int(sim.state.mc[0]) == 20 and np.isfinite(o["epot"]) and 0.5 < o["density_g_cm3"] < 1.5
    _, _, w = _water_box()
    tpl = RigidTemplate(water(), w)
    fl = FlexibleSimulation(sys, [tpl] * sys.nmol, pos, H, s, dt=0.001, ensemble="nve", log=None)
    rg = Simulation(sys, pos, H, s, dt=0.001, ensemble="nve", log=None)
    assert abs(float(fl.state.epot) - float(rg.state.epot)) < 1e-8 * abs(float(rg.state.epot))
    fl._advance(30)
    assert int(fl.state.iters) == 0 and np.isfinite(fl.observables()["etot"])


def test_refuses_unsupported_combinations():
    with pytest.raises(ValueError):
        PGMForceField(*_water()[:1], np.eye(3) * 1.24, _settings(iel="xl"))
    with pytest.raises(ValueError):
        PGMForceField(*_water()[:1], np.eye(3) * 1.24, _settings(iel="0scf", differentiable=True))
    with pytest.raises(ValueError):
        PGMForceField(*_water()[:1], np.eye(3) * 1.24, _settings(iel="0scf", iel_order=2))


def test_response_spectrum_and_positive_auxiliary_energy():
    from pgm_jax.md.iel import response_spectrum
    s = _settings(cutoff=0.6, pme_grid=(48, 48, 48), pme_order=8, ewald_beta=6.0, iel="0scf")
    ff, pos, H, idx = _ff_state(s)
    lo_j, hi_j = response_spectrum(ff, pos, H, idx, precond="jacobi")
    lo_b, hi_b = response_spectrum(ff, pos, H, idx, precond="block")
    assert 0.0 < lo_j < 1.0 < hi_j < 2.5 and 0.0 < lo_b < 1.0 < hi_b < hi_j, (lo_j, hi_j, lo_b, hi_b)
    # omega lambda_max < 1: every auxiliary mode has positive energy; without dissipation (time
    # reversible) E_kin + U~ is conserved at 1 fs as with converged dipoles
    sys, p, Hw = _water()
    dev = {}
    for name, st in (("iel", _settings(iel="0scf", iel_omega=0.5, iel_order=0, iel_kappa=1.82)), ("scf", _settings())):
        sim = Simulation(sys, p, Hw, st, dt=0.001, ensemble="nve", log=None, seed=3)
        sim._advance(20)
        e0 = sim.observables()["etot"]
        E = []
        for _ in range(8):
            sim._advance(50)
            E.append(sim.observables()["etot"] - e0)
        dev[name] = (max(abs(e) for e in E), sim.observables()["ekin"])
    assert dev["iel"][0] < 3.0 * dev["scf"][0] + 0.002 * dev["scf"][1], dev
