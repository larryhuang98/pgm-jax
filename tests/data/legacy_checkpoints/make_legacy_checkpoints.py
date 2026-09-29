"""Write checkpoints in the legacy (pickle) format of pgm_jax commit e72c57c, for
tests/test_checkpoints.py: every driver's checkpoint of a small system, plus the state reached by
continuing 10 steps from each checkpoint with that code (expected.npz).

Run once with the code of e72c57c on the path (not the current package):

    git archive e72c57c | tar x -C /tmp/pgm_e72c57c
    PYTHONPATH=/tmp/pgm_e72c57c JAX_PLATFORMS=cpu python tests/data/legacy_checkpoints/make_legacy_checkpoints.py

The systems are built from plain numpy here (the same numbers as pgm_jax.models.toy) and the old
constructors, so the current code can rebuild them exactly.
"""

import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from pgm_jax.bias import BiasSet, MetaD, cv  # noqa: E402
from pgm_jax.bias.walkers import Walkers  # noqa: E402
from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec  # noqa: E402
from pgm_jax.md.alchemy import Alchemy, FreeEnergyRun, LambdaWindows, alchemical_system  # noqa: E402
from pgm_jax.md.finite_field import FieldReplicas  # noqa: E402
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, RigidTemplate  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.pimd import PIMDSimulation  # noqa: E402
from pgm_jax.md.remd import ReplicaExchange  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402
from pgm_jax.system import Molecule, System  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


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
    """pgm_jax.models.toy.water_lattice."""
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


S = MDSettings(precision="double", dipole_tol=1e-10, max_iter=300, cutoff=0.55, skin=0.05)
expected = {}


def rigid():
    pos, H, _ = water_lattice()
    return Simulation(
        System([water()] * 64), pos, H, S, dt=0.001, ensemble="nvt", thermostat="bussi", tau_t=0.1, seed=3, log=None
    )


def rigid_metad():
    pos, _, _ = water_lattice(2)
    s = MDSettings(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
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
        ensemble="nvt",
        thermostat="langevin",
        gamma=5.0,
        temperature=300.0,
        seed=2,
        log=None,
        bias=BiasSet([m], colvar=5),
    )


def flexible():
    pos, H, w = water_lattice()
    return FlexibleSimulation(
        System([water()] * 64),
        [RigidTemplate(water(), w)] * 64,
        pos,
        H,
        S,
        dt=0.002,
        ensemble="npt",
        thermostat="gle",
        barostat_interval=5,
        seed=1,
        log=None,
    )


def alchemy_sim():
    pos, H, _ = water_lattice()
    sysA, P = alchemical_system(System([water()] * 64), 0)
    s = MDSettings(
        precision="double",
        dipole_tol=1e-9,
        max_iter=400,
        cutoff=0.55,
        skin=0.05,
        ewald_beta=5.0,
        pme_grid=(32, 32, 32),
        peek=0.0,
    )
    return Simulation(
        sysA, pos, H, s, dt=0.001, log=None, params=P, alchemy=Alchemy(sysA, 0), thermostat="bussi", ensemble="nvt"
    )


def field_sim():
    pos, H, _ = water_lattice()
    return Simulation(
        System([water()] * 64),
        pos,
        H,
        S,
        dt=0.001,
        ensemble="nvt",
        thermostat="bussi",
        log=None,
        efield=(0.0, 0.0, 0.0),
    )


def main():
    # single engines: checkpoint after 20 steps, then 10 more steps in a fresh object from the checkpoint
    for name, make in (("rigid", rigid), ("rigid_metad", rigid_metad), ("flexible", flexible)):
        sim = make()
        sim._advance(20)
        sim.save(os.path.join(OUT, name))
        os.remove(os.path.join(OUT, name + ".rst7"))
        if os.path.exists(os.path.join(OUT, name + ".bias")):
            os.remove(os.path.join(OUT, name + ".bias"))
        again = make()
        again.load(os.path.join(OUT, name + ".chk"))
        again._advance(10)
        expected[f"{name}.pos"] = again.positions_nm()
        expected[f"{name}.vel"] = again.velocities_nm_ps()
        expected[f"{name}.epot"] = np.asarray(again.state.epot)
    # replica exchange (batched, 3 slots)
    rex = ReplicaExchange(flexible_nvt(), np.array([300.0, 304.0, 308.0]), exchange_every=10, seed=5, log=None)
    rex.replicas.advance(10)
    rex.exchange()
    rex.save(os.path.join(OUT, "remd"))
    for k in range(3):
        os.remove(os.path.join(OUT, f"remd_T{k:02d}.rst7"))
    rex2 = ReplicaExchange(flexible_nvt(), np.array([300.0, 304.0, 308.0]), exchange_every=10, seed=5, log=None)
    rex2.load(os.path.join(OUT, "remd.remd.chk"))
    rex2.replicas.advance(10)
    rex2.exchange()
    expected["remd.pos"] = np.stack([np.asarray(rex2.replicas.state(k).dyn.position) for k in range(3)])
    expected["remd.replica"] = np.asarray(rex2.stats.replica)
    # lambda windows with samples and exchanges
    lams = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    run = FreeEnergyRun(LambdaWindows(alchemy_sim(), lams, seed=2), sample_every=5, exchange_every=10, log=None)
    run.run(20, prefix=os.path.join(OUT, "fe"))
    for f in ("fe_fe.npz", "fe_fe.json") + tuple(f"fe_L{k:02d}.rst7" for k in range(3)):
        os.remove(os.path.join(OUT, f))
    run2 = FreeEnergyRun(LambdaWindows(alchemy_sim(), lams, seed=7), sample_every=5, exchange_every=10, log=None)
    run2.load(os.path.join(OUT, "fe.fe.chk"))
    run2.run(10, prefix=None)
    expected["fe.u"] = np.asarray(run2.arrays()["u"])
    # finite-field replicas
    rep = FieldReplicas(field_sim(), [(0.0, 0.0, 0.5), (0.0, 0.0, -0.5)], seed=1)
    rep.advance(10)
    rep.save(os.path.join(OUT, "ff.ffchk"))
    rep2 = FieldReplicas(field_sim(), [(0.0, 0.0, 0.5), (0.0, 0.0, -0.5)], seed=9)
    rep2.load(os.path.join(OUT, "ff.ffchk"))
    rep2.advance(10)
    expected["ff.center"] = np.stack([np.asarray(rep2.state(k).dyn.position.center) for k in range(2)])
    # path integrals (4 beads, NPT) and shared-bias walkers
    pi = PIMDSimulation(
        pimd_sim(), beads=4, seed=1, ensemble="npt", pressure=1000.0, barostat_interval=5, thermostat="pile-g", log=None
    )
    pi._advance(10)
    pi.save(os.path.join(OUT, "pimd"))
    os.remove(os.path.join(OUT, "pimd.rst7"))
    pi2 = PIMDSimulation(
        pimd_sim(), beads=4, seed=8, ensemble="npt", pressure=1000.0, barostat_interval=5, thermostat="pile-g", log=None
    )
    pi2.load(os.path.join(OUT, "pimd.pimd.chk"))
    pi2._advance(10)
    expected["pimd.q"] = np.asarray(pi2.state.q)
    expected["pimd.box"] = np.asarray(pi2.state.box)
    wk = Walkers(rigid_metad(), 3, shared=True, seed=4)
    wk.advance(20)
    wk.save(os.path.join(OUT, "wk.walkers.chk"))
    wk2 = Walkers(rigid_metad(), 3, shared=True, seed=9)
    wk2.load(os.path.join(OUT, "wk.walkers.chk"))
    wk2.advance(10)
    expected["walkers.center"] = np.asarray(wk2.S.dyn.position.center)
    expected["walkers.ebias"] = wk2.bias_energies()
    np.savez(os.path.join(OUT, "expected.npz"), **expected)
    print("written:", sorted(os.listdir(OUT)))


def pimd_sim():
    """8 flexible waters (the harmonic-quartic water of tests/test_pimd.py) for 4-bead PIMD."""
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
    s = MDSettings(precision="double", dipole_tol=1e-10, cutoff=0.5, skin=0.05, lj_lrc=False, max_iter=200)
    return FlexibleSimulation(
        System([tpl.pgm] * 8),
        [tpl] * 8,
        np.concatenate(pos),
        np.eye(3) * 1.5,
        s,
        dt=0.0002,
        ensemble="nvt",
        temperature=300.0,
        thermostat="bussi",
        log=None,
    )


def flexible_nvt():
    pos, H, w = water_lattice()
    return FlexibleSimulation(
        System([water()] * 64),
        [RigidTemplate(water(), w)] * 64,
        pos,
        H,
        S,
        dt=0.002,
        temperature=300.0,
        thermostat="gle",
        seed=1,
        log=None,
    )


if __name__ == "__main__":
    sys.exit(main())
