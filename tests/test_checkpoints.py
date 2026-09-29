"""Checkpoints of every MD driver: legacy pickle checkpoints of pgm_jax commit e72c57c
(tests/data/legacy_checkpoints, written by make_legacy_checkpoints.py there with that code) load
and continue as the old code continued them, and the current npz format continues a run exactly.

The systems below are those of make_legacy_checkpoints.py, built with the current API; real_npt.chk
is a checkpoint of a real run of the old code (test_real_old_checkpoint)."""

import os

import jax
import numpy as np
import pytest

from pgm_jax.bias import BiasSet, MetaD, cv
from pgm_jax.bias.walkers import Walkers
from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec
from pgm_jax.md.alchemy import Alchemy, FreeEnergyRun, LambdaWindows, alchemical_system
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.driver import is_legacy_checkpoint, read_checkpoint, read_legacy_checkpoint
from pgm_jax.md.finite_field import FieldReplicas
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, RigidTemplate
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.pimd import PILE, PIMDSimulation
from pgm_jax.md.remd import ReplicaExchange
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import Bussi, Langevin
from pgm_jax.system import Molecule, System

LEGACY = os.path.join(os.path.dirname(__file__), "data", "legacy_checkpoints")
S = MDSettings().replace(precision="double", dipole_tol=1e-10, max_iter=300, cutoff=0.55, skin=0.05)
LAMBDAS = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
FIELDS = [(0.0, 0.0, 0.5), (0.0, 0.0, -0.5)]


@pytest.fixture(scope="module")
def expected():
    """The states the code of e72c57c reached 10 steps after loading each checkpoint."""
    with np.load(os.path.join(LEGACY, "expected.npz")) as z:
        return dict(z)


def close(a, b):
    """Same continuation up to summation order (bitwise on the machine that wrote it)."""
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-10)


def same(a, b):
    """Bitwise equal pytrees / arrays."""
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# ----------------------------------------------------------------------------- systems
def water():
    """The toy pGM water of pgm_jax.models.toy."""
    return Molecule(
        "WAT",
        ["O", "H", "H"],
        ["OW", "HW", "HW"],
        np.array([-0.8, 0.4, 0.4]),
        np.array([0.06, 0.05, 0.05]),
        np.array([1.0e-3, 0.3e-3, 0.3e-3]),
        cov=[(0, 1, -0.02), (0, 2, -0.02), (1, 0, 0.008), (2, 0, 0.008)],
        lj_rmin_half=[0.178, 0.0, 0.0],
        lj_sqrt_eps=[0.80, 0.0, 0.0],
        bonds=[(0, 1), (0, 2)],
    )


def water_lattice(n_side=4, spacing=0.31, seed=0):
    """Randomly oriented rigid waters on a cubic lattice: positions, box, molecule geometry [nm]."""
    rng = np.random.default_rng(seed)
    t = np.radians(104.52 / 2)
    w = np.array(
        [[0, 0, 0], [0.09572 * np.sin(t), 0.09572 * np.cos(t), 0], [-0.09572 * np.sin(t), 0.09572 * np.cos(t), 0]]
    )
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                pos.append(w @ Q.T + (np.array([i, j, k]) + 0.5) * spacing)
    return np.concatenate(pos), np.eye(3) * n_side * spacing, w


def rigid():
    """64 rigid waters, NVT (Bussi)."""
    pos, H, _ = water_lattice()
    return Simulation(System([water()] * 64), pos, H, S, dt=0.001, thermostat=Bussi(0.1), seed=3, log=None)


def rigid_metad():
    """8 rigid waters with a two-CV metadynamics bias, Langevin NVT."""
    pos, _, _ = water_lattice(2)
    s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
    m = MetaD(
        [cv.Distance(0, 9), cv.Dihedral(1, 0, 9, 10)],
        sigma=[0.03, 0.4],
        height=1.0,
        pace=5,
        biasfactor=5.0,
        temperature=300.0,
    )
    return Simulation(
        System([water()] * 8),
        pos + 1.2,
        np.eye(3) * 3.0,
        s,
        dt=0.0005,
        thermostat=Langevin(5.0),
        temperature=300.0,
        seed=2,
        log=None,
        bias=BiasSet([m], colvar=5),
    )


def flexible(**kw):
    """64 waters in the flexible engine (SETTLE-like rigid templates), GLE NPT by default."""
    pos, H, w = water_lattice()
    opts = dict(thermostat="gle", barostat=MonteCarloBarostat(every=5), seed=1) | kw
    return FlexibleSimulation(
        System([water()] * 64), [RigidTemplate(water(), w)] * 64, pos, H, S, dt=0.002, log=None, **opts
    )


def alchemy_sim():
    """64 waters, the first one alchemical (PME), Bussi NVT."""
    pos, H, _ = water_lattice()
    sysA, P = alchemical_system(System([water()] * 64), 0)
    s = MDSettings().replace(
        precision="double",
        dipole_tol=1e-9,
        max_iter=400,
        cutoff=0.55,
        skin=0.05,
        ewald_beta=5.0,
        pme_grid=(32, 32, 32),
        peek=0.0,
    )
    return Simulation(sysA, pos, H, s, dt=0.001, log=None, params=P, alchemy=Alchemy(sysA, 0), thermostat="bussi")


def field_sim():
    """64 rigid waters in an external field (amplitude set per replica), Bussi NVT."""
    pos, H, _ = water_lattice()
    return Simulation(System([water()] * 64), pos, H, S, dt=0.001, thermostat="bussi", log=None, efield=(0.0, 0.0, 0.0))


def pimd_sim():
    """8 flexible waters (harmonic-quartic intramolecular terms) for 4-bead PIMD."""
    m = water()
    t = np.radians(104.5)
    x = np.array([[0, 0, 0], [0.0957, 0, 0], [0.0957 * np.cos(t), 0.0957 * np.sin(t), 0]])
    families = ("bond_quartic", "angle_harm", "angle_cubic", "bond_bond", "bond_angle")
    model = BondedModel(
        [MolSpec("WAT", ["O", "H", "H"], [(0, 1), (0, 2)], [1, 1], 0, x, m)], BondedSettings(families=families)
    )
    P = model.init_params()
    P["bond_quartic"]["K2"] = P["bond_quartic"]["K2"] * 0.0 + 4.5e5
    P["angle_harm"]["Ka"] = P["angle_harm"]["Ka"] * 0.0 + 350.0
    tpl = FlexibleTemplate.from_fit(model, P)
    rng = np.random.default_rng(0)
    pos = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                c = (np.array([i, j, k]) + 0.5) * 1.5 / 2 + rng.normal(scale=0.02, size=3)
                pos.append((x - x.mean(0)) @ R.T + c)
    s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=0.5, skin=0.05, lj_lrc=False, max_iter=200)
    return FlexibleSimulation(
        System([tpl.pgm] * 8),
        [tpl] * 8,
        np.concatenate(pos),
        np.eye(3) * 1.5,
        s,
        dt=0.0002,
        thermostat="bussi",
        temperature=300.0,
        log=None,
    )


def pimd(seed):
    """4-bead NPT PIMD of pimd_sim."""
    return PIMDSimulation(
        pimd_sim(), beads=4, seed=seed, barostat=MonteCarloBarostat(1000.0, every=5), thermostat=PILE("g")
    )


# ----------------------------------------------------------------------------- tests
@pytest.mark.parametrize("name", ["rigid", "rigid_metad", "flexible"])
def test_engine_checkpoints(name, expected, tmp_path):
    """Simulation / FlexibleSimulation: a legacy .chk continues as with the old code; saved again
    (npz format) and loaded, the continuation is bitwise the same."""
    sim = {"rigid": rigid, "rigid_metad": rigid_metad, "flexible": flexible}[name]()
    path = os.path.join(LEGACY, name + ".chk")
    assert is_legacy_checkpoint(path)
    sim.load_checkpoint(path)
    new = str(tmp_path / "new")
    sim.save_checkpoint(new + ".chk")
    assert not is_legacy_checkpoint(new + ".chk")
    sim.advance(10)
    close(sim.positions(), expected[f"{name}.pos"])
    close(sim.velocities(), expected[f"{name}.vel"])
    close(sim.state.epot, expected[f"{name}.epot"])
    first = sim.state.set(nbr=None)
    sim.load_checkpoint(new + ".chk")
    sim.advance(10)
    same(sim.state.set(nbr=None), first)
    with pytest.raises(ValueError, match="not a 'pimd' one"):
        read_checkpoint(new + ".chk", "pimd")


def test_replica_exchange_checkpoint(expected, tmp_path):
    """Batched temperature REMD: legacy .remd.chk continuation; npz round trip bitwise."""
    T = np.array([300.0, 304.0, 308.0])
    rex = ReplicaExchange(flexible(temperature=300.0, barostat=None), T, exchange_every=10, seed=5)
    rex.load_checkpoint(os.path.join(LEGACY, "remd.remd.chk"))
    rex.save_checkpoint(str(tmp_path / "new.remd.chk"))
    rex.replicas.advance(10)
    rex.exchange()
    close(np.stack([rex.replicas.state(k).dyn.position for k in range(3)]), expected["remd.pos"])
    assert np.array_equal(rex.stats.replica, expected["remd.replica"])
    first = [rex.replicas.state(k).set(nbr=None) for k in range(3)]
    rex.load_checkpoint(str(tmp_path / "new.remd.chk"))
    rex.replicas.advance(10)
    rex.exchange()
    same([rex.replicas.state(k).set(nbr=None) for k in range(3)], first)


def test_free_energy_checkpoint(expected, tmp_path):
    """Lambda windows with samples and exchanges: legacy .fe.chk continuation; npz round trip."""
    run = FreeEnergyRun(LambdaWindows(alchemy_sim(), LAMBDAS, seed=7), sample_every=5, exchange_every=10)
    run.load_checkpoint(os.path.join(LEGACY, "fe.fe.chk"))
    run.save_checkpoint(str(tmp_path / "new.fe.chk"))
    run.run(10, prefix=None)
    close(run.arrays()["u"], expected["fe.u"])
    first = run.arrays()
    run.load_checkpoint(str(tmp_path / "new.fe.chk"))
    run.run(10, prefix=None)
    for k, v in run.arrays().items():
        assert np.array_equal(np.asarray(v), np.asarray(first[k])), k


def test_field_replicas_checkpoint(expected, tmp_path):
    """Finite-field replicas: legacy .ffchk continuation; npz round trip bitwise."""
    rep = FieldReplicas(field_sim(), FIELDS, seed=9)
    rep.load_checkpoint(os.path.join(LEGACY, "ff.ffchk"))
    rep.save_checkpoint(str(tmp_path / "new.ffchk"))
    rep.advance(10)
    close(np.stack([rep.state(k).dyn.position.center for k in range(2)]), expected["ff.center"])
    first = rep.S.set(nbr=None)
    rep.load_checkpoint(str(tmp_path / "new.ffchk"))
    rep.advance(10)
    same(rep.S.set(nbr=None), first)


def test_pimd_checkpoint(expected, tmp_path):
    """NPT PIMD: legacy .pimd.chk continuation; npz round trip bitwise."""
    pi = pimd(8)
    pi.load_checkpoint(os.path.join(LEGACY, "pimd.pimd.chk"))
    pi.save_checkpoint(str(tmp_path / "new.pimd.chk"))
    pi.advance(10)
    close(pi.state.q, expected["pimd.q"])
    close(pi.state.box, expected["pimd.box"])
    first = pi.state.set(eng=pi.state.eng.set(nbr=None))
    pi.load_checkpoint(str(tmp_path / "new.pimd.chk"))
    pi.advance(10)
    same(pi.state.set(eng=pi.state.eng.set(nbr=None)), first)


def test_walkers_checkpoint(expected, tmp_path):
    """Shared-bias walkers: legacy .walkers.chk continuation; npz round trip bitwise."""
    wk = Walkers(rigid_metad(), 3, shared=True, seed=9)
    wk.load_checkpoint(os.path.join(LEGACY, "wk.walkers.chk"))
    wk.save_checkpoint(str(tmp_path / "new.walkers.chk"))
    wk.advance(10)
    close(wk.S.dyn.position.center, expected["walkers.center"])
    close(wk.bias_energies(), expected["walkers.ebias"])
    first = wk.S.set(nbr=None)
    wk.load_checkpoint(str(tmp_path / "new.walkers.chk"))
    wk.advance(10)
    same(wk.S.set(nbr=None), first)


def test_real_old_checkpoint(tmp_path):
    """A checkpoint of a real run of the old code (512 rigid pGM waters, NPT; runs/npt_check/npt.chk
    of the main repository, 2026-09-24, older than e72c57c: MDState without the thermostat fields)
    loads into a 512-water rigid simulation (the state is upgraded), and after conversion to the npz
    format the continuation is bitwise the same."""
    path = os.path.join(LEGACY, "real_npt.chk")
    old = read_legacy_checkpoint(path)["state"]
    H = np.asarray(old.box)
    pos, _, _ = water_lattice(8, spacing=float(H[0, 0]) / 8)  # replaced by the checkpoint's positions
    sim = Simulation(
        System([water()] * 512), pos, H, MDSettings(), dt=0.001, thermostat="bussi", barostat=MonteCarloBarostat()
    )
    sim.load_checkpoint(path)
    assert np.array_equal(np.asarray(sim.state.dyn.position.center), np.asarray(old.dyn.position.center))
    assert np.array_equal(np.asarray(sim.state.induction.mu), np.asarray(old.induction.mu))
    sim.advance(5)
    new = str(tmp_path / "new.chk")
    sim.save_checkpoint(new)
    sim.advance(5)
    first = sim.state.set(nbr=None)
    sim.load_checkpoint(new)
    sim.advance(5)
    same(sim.state.set(nbr=None), first)
    assert np.isfinite(float(sim.state.epot))
