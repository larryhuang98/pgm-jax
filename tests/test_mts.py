"""Multiple time stepping (md/mts.py): one fast step per outer step is the ordinary integrator, the
force groups sum to the full force and the fast forces are the gradient of the fast energy (short-
range and special-pair splits, every fast induction model), the list rebuild, the NVE step is
time-reversible, energy conservation, per-group kinetic temperatures with thermostats (two and
three levels), NPT + restraints + checkpoints, and the settings that must be refused."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_flexible import _box  # noqa: E402
from test_grad import water  # noqa: E402
from test_md_macro import _water_box  # noqa: E402

from pgm_jax import System  # noqa: E402
from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.md.mts import MTS  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402

S_WATER = MDSettings(precision="double", dipole_tol=1e-12, max_iter=300, cutoff=0.55, skin=0.05)


def water_sim(engine, mts=None, dt=0.002, ensemble="nve", thermostat="langevin", seed=3, settings=S_WATER, **kw):
    """64 pGM-like waters (rigid bodies or constraints), 0.55 nm cutoff, float64."""
    pos, H, w = _water_box(n_side=4, spacing=0.31)
    wat = water()
    sys = System([wat] * (len(pos) // 3))
    if engine == "rigid":
        return Simulation(sys, pos, H, settings, dt=dt, ensemble=ensemble, thermostat=thermostat, log=None, seed=seed,
                          mts=mts, **kw)
    return FlexibleSimulation(sys, [RigidTemplate(wat, w)] * sys.nmol, pos, H, settings, dt=dt, ensemble=ensemble,
                              thermostat=thermostat, log=None, seed=seed, mts=mts, **kw)


def methanol_sim(mts=None, dt=0.001, ensemble="nve", **kw):
    """32 flexible methanols (class II bonded terms, 1-4 scaled LJ), 0.6 nm cutoff, float64."""
    tpl, sys, pos, H = _box(n=32, density=0.55)
    s = MDSettings(precision="double", dipole_tol=1e-10, max_iter=300, cutoff=0.6, skin=0.05, lj_lrc=False)
    return FlexibleSimulation(sys, [tpl] * sys.nmol, pos, H, s, dt=dt, ensemble=ensemble, log=None, mts=mts, **kw)


MTS1 = MTS(inner=1, r_short=0.4, buffer=0.1, anchor=False)


@pytest.mark.parametrize("engine", ["rigid", "constraints"])
@pytest.mark.parametrize("thermo", ["nve", "bussi", "langevin", "gle", "langevin-inner"])
def test_one_fast_step_is_the_ordinary_integrator(engine, thermo):
    """inner = 1: B_slow B_fast A O A B_fast B_slow is BAOAB with F = F_slow + F_fast (the O step at
    the outer or the inner level is the same step)."""
    ens = "nve" if thermo == "nve" else "nvt"
    th = "langevin" if thermo in ("nve", "langevin-inner") else thermo
    mts = MTS(inner=1, r_short=0.4, buffer=0.1, anchor=False, o_step="inner" if thermo.endswith("inner") else "outer")
    out = []
    for m in (None, mts):
        sim = water_sim(engine, m, ensemble=ens, thermostat=th)
        sim._advance(15)
        out.append((sim.positions_nm(), sim.velocities_nm_ps(), sim.observables()["econs"]))
    assert np.abs(out[0][0] - out[1][0]).max() < 1e-11
    assert np.abs(out[0][1] - out[1][1]).max() < 1e-9
    assert abs(out[0][2] - out[1][2]) < 1e-8 * abs(out[0][2])


def test_groups_sum_to_the_full_force_and_fast_forces_are_gradients():
    sim = methanol_sim(MTS(inner=2, r_short=0.4, buffer=0.1), dt=0.001, ensemble="nvt", gamma=10.0)
    sim._advance(100)
    st, integ = sim.state, sim.integ
    F = np.asarray(st.dyn.force)
    rms = np.sqrt(np.mean(F ** 2))
    assert np.abs(sum(np.asarray(f) for f in st.mts.forces) - F).max() < 1e-12 * rms
    # the full force is the ordinary one at the same positions
    ref = methanol_sim(None, dt=0.001)
    ref.state = ref.integ.forces(ref.state.set(dyn=ref.state.dyn.set(position=st.dyn.position), box=st.box), True)
    assert np.abs(np.asarray(ref.state.dyn.force) - F).max() < 1e-7 * rms
    # fast level = -grad(fast nonbonded energy + bonded energy), here and after a displacement
    e = jax.jit(jax.grad(lambda y, s: integ.short_energy(y, s.box, s) + integ.flex.energy(y)))
    x = st.dyn.position
    assert np.abs(np.asarray(st.mts.forces[1]) + np.asarray(e(x, st))).max() < 1e-10 * rms
    y = x + 0.01 * jax.random.normal(jax.random.PRNGKey(1), x.shape)
    st2 = integ._eval_fast(st.set(dyn=st.dyn.set(position=y)), 1)
    assert np.abs(np.asarray(st2.mts.forces[1]) + np.asarray(e(y, st2))).max() < 1e-10 * rms
    # two atoms moved by 0.06 nm (more than the buffer together): the list is rebuilt from the
    # neighbour list, with the forces of a list built from scratch there
    z = x.at[0, 0].add(0.06).at[40, 1].add(-0.06)
    st3 = integ._eval_fast(st.set(dyn=st.dyn.set(position=z)), 1)
    assert int(st3.mts.rebuilds) == int(st.mts.rebuilds) + 1 and not bool(st3.mts.overflow)
    fresh = integ.forces(st.set(dyn=st.dyn.set(position=z)), True)
    assert np.abs(np.asarray(st3.mts.forces[1]) - np.asarray(fresh.mts.forces[1])).max() < 1e-9 * rms


@pytest.mark.parametrize("pol", ["none", "direct", "mutual"])
def test_special_pair_split(pol):
    """split="special": the fast level (bonded terms + the pGM and van der Waals interactions of the
    special pairs, unswitched) is the gradient of its energy, and one fast step per outer step is the
    ordinary integrator."""
    sim = methanol_sim(MTS(inner=2, split="special", polarization=pol), dt=0.001, ensemble="nvt", gamma=10.0)
    sim._advance(40)
    st, integ = sim.state, sim.integ
    F1 = np.asarray(st.mts.forces[1])
    g = jax.grad(lambda y: integ.short_energy(y, st.box, st) + integ.flex.energy(y))(st.dyn.position)
    assert np.abs(F1 + np.asarray(g)).max() < 1e-10 * np.sqrt(np.mean(F1 ** 2))
    out = []
    for m in (None, MTS(inner=1, split="special", polarization=pol, anchor=False)):
        sim = methanol_sim(m, dt=0.0005, ensemble="nvt", thermostat="bussi")
        sim._advance(15)
        out.append(sim.positions_nm())
    assert np.abs(out[0] - out[1]).max() < 1e-11


def _reverse(integ, st, n):
    flip = lambda s: s.set(dyn=s.dyn.set(momentum=jax.tree_util.tree_map(lambda p: -p, s.dyn.momentum)))  # noqa: E731
    back = integ.run(flip(integ.run(st, n)), n)
    return st, back


@pytest.mark.parametrize("case", ["rigid", "constraints", "flexible-3-levels"])
def test_nve_step_is_time_reversible(case):
    if case == "flexible-3-levels":
        sim = methanol_sim(MTS(inner=2, bonded=2, r_short=0.4, buffer=0.1), dt=0.002)
    else:
        sim = water_sim(case, MTS(inner=3, r_short=0.4, buffer=0.1), dt=0.004)
    st0, back = _reverse(sim.integ, sim.state, 25)
    x0 = np.asarray(sim.rigid.positions(st0.dyn.position))
    x1 = np.asarray(sim.rigid.positions(back.dyn.position))
    assert np.abs(x1 - x0).max() < 1e-8, np.abs(x1 - x0).max()


def test_energy_conservation():
    """Flexible methanol, NVE: MTS with outer 2h / fast h conserves the energy nearly as well as the
    ordinary integrator at h, much better than at 2h; no drift."""
    base = methanol_sim(MTS(inner=2, r_short=0.4, buffer=0.1), dt=0.001, ensemble="nvt", gamma=10.0)
    base._advance(200)
    x, H, v = base.positions_nm(), np.asarray(base.state.box), base.velocities_nm_ps()
    res = {}
    for name, dt, m in (("h", 0.001, None), ("2h", 0.002, None), ("mts", 0.002, MTS(inner=2, r_short=0.4, buffer=0.1))):
        tpl, sys, _, _ = _box(n=32, density=0.55)
        s = base.settings
        sim = FlexibleSimulation(sys, [tpl] * sys.nmol, x, H, s, dt=dt, ensemble="nve", vel_nm_ps=v, log=None, mts=m)
        E = []
        n = int(round(0.3 / dt / 10))
        for _ in range(10):
            sim._advance(n)
            E.append(sim.observables()["etot"])
        ke = 0.5 * sim.integ.dof * KB * 298.0
        res[name] = (np.std(E) / ke, (E[-1] - E[0]) / ke)
    assert res["mts"][0] < 0.6 * res["2h"][0], res
    assert abs(res["mts"][1]) < 0.01, res


@pytest.mark.parametrize("engine,thermostat,o_step", [("rigid", "bussi", "outer"), ("rigid", "langevin", "outer"),
                                                    ("constraints", "langevin", "outer"),
                                                    ("constraints", "langevin", "inner")])
def test_group_temperatures(engine, thermostat, o_step):
    """NVT with MTS (outer 4 fs, fast 2 fs): translational and rotational (rigid bodies) or centre-of-
    mass and internal (constraints) temperatures at the target."""
    s = MDSettings(precision="mixed", dipole_tol=1e-5, cutoff=0.55, skin=0.05)
    sim = water_sim(engine, MTS(inner=2, r_short=0.4, buffer=0.1, o_step=o_step), dt=0.004, ensemble="nvt",
                    thermostat=thermostat, settings=s, temperature=300.0, gamma=5.0, tau_t=0.1)
    sim._advance(250)
    T = []
    for _ in range(150):
        sim._advance(10)
        o = sim.observables()
        T.append((o["temp_K"], o.get("temp_trans", o.get("temp_com")), o.get("temp_rot", o.get("temp_internal"))))
    T = np.mean(T, axis=0)
    assert np.all(np.abs(T - 300.0) < 15.0), T


def test_three_levels_thermostat():
    """Flexible methanol with three levels (slow 2 fs, short-range 1 fs, bonded 0.5 fs) and the O step
    in the middle of the outer step: centre-of-mass and internal temperatures at the target."""
    sim = methanol_sim(MTS(inner=2, bonded=2, r_short=0.4, buffer=0.1), dt=0.002, ensemble="nvt",
                       thermostat="langevin", gamma=5.0, temperature=300.0)
    sim._advance(300)
    T = []
    for _ in range(120):
        sim._advance(10)
        o = sim.observables()
        T.append((o["temp_K"], o["temp_com"], o["temp_internal"]))
    T = np.mean(T, axis=0)
    assert np.all(np.abs(T - 300.0) < 20.0), T


def test_npt_restraints_and_checkpoint(tmp_path):
    """NPT with the barostat at outer steps and a positional restraint in the fast group; a checkpoint
    continues the run exactly."""
    from pgm_jax.md.restraints import PositionRestraint
    pos, H, w = _water_box(n_side=4, spacing=0.31)
    rest = PositionRestraint([0, 3], pos[[0, 3]], k=500.0)
    s = MDSettings(precision="double", dipole_tol=1e-9, cutoff=0.55, skin=0.05)
    mk = lambda: water_sim("constraints", MTS(inner=2, r_short=0.4, buffer=0.1), dt=0.004, ensemble="npt",  # noqa: E731
                           thermostat="bussi", settings=s, barostat_interval=5, restraints=rest)
    sim = mk()
    sim._advance(100)
    assert int(sim.state.mc[0]) == 20 and np.isfinite(sim.observables()["econs"])
    assert sim.observables()["erestraint"] > 0.0
    sim.save(str(tmp_path / "a"))
    sim._advance(20)
    sim2 = mk()
    sim2.load(str(tmp_path / "a.chk"))
    sim2._advance(20)
    assert np.abs(sim.positions_nm() - sim2.positions_nm()).max() < 1e-8


def test_refused_settings():
    with pytest.raises(ValueError, match="bonded"):
        water_sim("rigid", MTS(inner=2, split="bonded"))
    with pytest.raises(ValueError, match="cutoff"):
        water_sim("constraints", MTS(inner=2, r_short=0.5, buffer=0.1))           # list beyond 0.55 nm
    with pytest.raises(ValueError, match="direct"):
        water_sim("constraints", MTS(inner=2, r_short=0.4, polarization="direct"),
                  settings=MDSettings(precision="double", cutoff=0.55, skin=0.05, elec="qp"))
    with pytest.raises(ValueError, match="predictor"):
        water_sim("constraints", MTS(inner=2, r_short=0.4, anchor=True),
                  settings=MDSettings(precision="double", cutoff=0.55, skin=0.05, predictor="ls"))
    from pgm_jax.md.remd import MDReplicas
    sim = water_sim("constraints", MTS(inner=2, r_short=0.4), ensemble="nvt", thermostat="bussi")
    with pytest.raises(ValueError, match="multiple time stepping"):
        MDReplicas(sim, [300.0, 310.0])
