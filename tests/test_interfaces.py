"""Interfaces to other MD codes (pgm_jax.interfaces): the device-resident engine against the native
force field and MD engines, the ASE calculator (energy, forces, stress vs finite differences, rigid
water constraints, NVE), the i-PI socket client (protocol and units against an in-process server,
single and batched requests; a real i-PI run if IPI_ROOT or i-pi is available) and the OpenMM
PythonForce (if OpenMM >= 8.4 is importable)."""

import os
import socket
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_md import small_box

from pgm_jax.interfaces import GasPhaseEngine, PGMEngine
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.units import KB


def settings(**kw):
    base = dict(
        cutoff=0.6,
        skin=0.05,
        ewald_beta=6.0,
        pme_grid=(32, 32, 32),
        pme_order=6,
        lj_lrc=False,
        dipole_tol=1e-10,
        max_iter=500,
        precision="double",
    )
    base.update(kw)
    return MDSettings(**base)


def water_box(seed=0):
    return small_box(seed, nw=30, nm=0)


def rotation(seed):
    R = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))[0]
    return R if np.linalg.det(R) > 0 else -R


@pytest.fixture(scope="module")
def box():
    sysm, pos, H = small_box(0)
    s = settings()
    return sysm, pos, H, s, PGMEngine(sysm, pos, H, s, stress="atomic")


def test_engine_matches_native_force_field(box):
    sysm, pos, H, s, eng = box
    ff = PGMForceField(sysm, H, s)
    idx = ff.rows_for(pos, H)
    ref = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    r = eng.compute(pos, H, virial=True)
    assert abs(r.energy - float(ref.energy["total"])) < 1e-9 * abs(r.energy)
    assert np.abs(r.forces - np.asarray(ref.forces)).max() < 1e-8
    assert np.abs(r.induced_dipoles - np.asarray(ref.induction.mu)).max() < 1e-10
    W = np.asarray(ff.strain_derivative(pos, H, idx, ref.induction.mu, molecular=False))
    assert np.allclose(np.diag(r.virial), np.diag(W), rtol=1e-9, atol=1e-8)
    assert np.allclose(r.virial, r.virial.T, atol=1e-8)
    # general (rotated) cell, atoms wrapped one by one: same energy, rotated forces / virial / dipole
    R = rotation(1)
    C = H @ R.T
    f = (pos @ R.T) @ np.linalg.inv(C)
    r2 = eng.compute((f - np.floor(f)) @ C, C, virial=True)
    assert abs(r2.energy - r.energy) < 1e-9 * abs(r.energy)
    assert np.abs(r2.forces - r.forces @ R.T).max() < 1e-7
    assert np.abs(r2.virial - R @ r.virial @ R.T).max() < 1e-7
    assert np.abs(r2.dipole - r.dipole @ R.T).max() < 1e-10


def test_virial_matches_finite_differences(box):
    """Atomic strain derivative (all nine components) vs differences of the energy of strained
    configurations (general cells: the engine rotates them to its box form)."""
    sysm, pos, H, s, eng = box
    W = eng.compute(pos, H, virial=True).virial
    h = 1e-5
    for a, b in ((0, 0), (1, 0), (0, 1), (2, 1), (0, 2)):
        e = []
        for sg in (1, -1):
            F = np.eye(3)
            F[a, b] += sg * h
            e.append(eng.compute(pos @ F.T, H @ F.T).energy)
        assert abs((e[0] - e[1]) / (2 * h) - W[a, b]) < 2e-6 * np.abs(W).max(), (a, b)


def test_molecular_virial_trace_matches_native(box):
    sysm, pos, H, s, _ = box
    ff = PGMForceField(sysm, H, s)
    idx = ff.rows_for(pos, H)
    mu = jax.jit(ff.compute)(pos, H, idx, ff.init_induction()).induction.mu
    W = np.asarray(ff.strain_derivative(pos, H, idx, mu))
    r = PGMEngine(sysm, pos, H, s, stress="molecular").compute(pos, H, virial=True)
    assert abs(np.trace(r.virial) - np.trace(W)) < 1e-8 * np.abs(W).max()


def test_flexible_templates_match_flexible_simulation():
    from test_flexible import _box

    from pgm_jax.md.flexible import FlexibleSimulation

    tpl, sysm, pos, H = _box()
    s = MDSettings(precision="double", dipole_tol=1e-10, cutoff=0.6, skin=0.05, lj_lrc=False)
    sim = FlexibleSimulation(sysm, [tpl] * sysm.nmol, pos, H, s, thermostat=None, log=None)
    eng = PGMEngine(sysm, sim.positions(), H, s, templates=[tpl] * sysm.nmol)
    r = eng.compute(sim.positions(), H, virial=True)
    F = np.asarray(sim.state.dyn.force)
    assert abs(r.energy - float(sim.state.epot)) < 1e-8 * abs(r.energy)
    assert np.abs(r.forces - F).max() < 1e-6 * np.abs(F).max()
    assert r.terms["bonded"] > 0
    # atomic virial of the bonded terms: finite differences of the total energy
    x = sim.positions()
    h = 1e-5
    for a, b in ((0, 0), (0, 1)):
        e = []
        for sg in (1, -1):
            Fm = np.eye(3)
            Fm[a, b] += sg * h
            e.append(eng.compute(x @ Fm.T, np.asarray(H) @ Fm.T).energy)
        assert abs((e[0] - e[1]) / (2 * h) - r.virial[a, b]) < 2e-6 * np.abs(r.virial).max()


def test_slots_and_resizing():
    """Two interleaved configurations in two slots give the results of separate engines (the dipole
    history of each slot); an engine whose row capacity is too small resizes and repeats."""
    sysm, pos, H = water_box(1)
    s = settings(dipole_tol=1e-5)
    rng = np.random.default_rng(0)
    confs = [pos + 0.002 * k * rng.normal(size=pos.shape) for k in range(4)]
    other = [c + 0.03 for c in confs]  # a rigid shift: a different "bead"
    eng2 = PGMEngine(sysm, pos, H, s, slots=2)
    a, b = PGMEngine(sysm, pos, H, s), PGMEngine(sysm, pos + 0.03, H, s)
    for x, y in zip(confs, other):
        r1, r2 = eng2.compute(x, H), eng2.compute(y, H)
        ra, rb = a.compute(x, H), b.compute(y, H)
        assert r1.iterations == ra.iterations and r2.iterations == rb.iterations
        assert abs(r1.energy - ra.energy) < 1e-9 * abs(ra.energy) and abs(r2.energy - rb.energy) < 1e-9 * abs(rb.energy)
    eng = PGMEngine(sysm, pos, H, settings())
    ref = eng.compute(pos, H).energy
    eng.ff.mc = 8  # far too small: overflow, resize, repeat
    eng._compile()
    for sl in eng.slots:
        sl.nbr = None
    assert abs(eng.compute(pos, H).energy - ref) < 1e-9 * abs(ref)
    assert eng.stats["repeats"] >= 1 and eng.ff.mc > 8


def test_compute_batch_matches_single_structures():
    """Ring-polymer-like batches in one vmapped call (shared list of the batch mean, stacked dipole
    histories, optionally in chunks) = one engine call per structure; molecules of one bead may
    sit in another periodic image."""
    sysm, pos, H = water_box(5)
    s = settings(dipole_tol=1e-10, cutoff=0.4)  # the box holds the molecule list of the batch mean
    rng = np.random.default_rng(1)
    X = np.stack([pos + 0.004 * rng.normal(size=pos.shape) for _ in range(4)])
    X[2, :3] += np.asarray(H)[0]  # a whole molecule of bead 2 one cell over
    for chunk, sc in ((None, s), (2, settings(dipole_tol=1e-10))):  # molecule list, then atom list
        ref = [PGMEngine(sysm, pos, H, sc, stress="atomic").compute(x, H, virial=True) for x in X]
        eng = PGMEngine(sysm, pos, H, sc, stress="atomic", bead_margin=0.02)
        out = eng.compute_batch(X, H, virial=True, chunk=chunk)
        for r, q in zip(out, ref):
            assert abs(r.energy - q.energy) < 1e-9 * abs(q.energy)
            assert np.abs(r.forces - q.forces).max() < 1e-7
            assert np.abs(r.virial - q.virial).max() < 1e-7
            assert np.abs(r.dipole - q.dipole).max() < 1e-9
        # i-PI: beads in another order, partial batches padded with copies of the last structure
        Y = X + 2e-4
        part = eng.compute_batch(np.stack([Y[3], Y[1], Y[1]]), H)
        assert len(part) == 3 and part[1].energy == part[2].energy
        full = eng.compute_batch(Y[::-1] + 1e-4, H)
        for r, x in zip(full + part[:2], list(Y[::-1] + 1e-4) + [Y[3], Y[1]]):
            q = PGMEngine(sysm, pos, H, sc).compute(x, H)
            assert abs(r.energy - q.energy) < 1e-9 * abs(q.energy)
        st = eng.stats
        assert st["batches"] == 3 and st["calls"] == 10 and st["slot_evaluations"] == 10 and st["resets"] == 0
        assert eng._nbb.kind == ("molecule" if chunk is None else "atom")


def test_gas_phase_engine():
    from test_grad import cluster

    from pgm_jax import ElecChannel, LJChannel, Model

    sysm, pos = cluster(np.random.default_rng(0))
    model = Model([ElecChannel(), LJChannel()])
    eng = GasPhaseEngine(model, sysm)
    r = eng.compute(pos)
    E = model.energy_fn(sysm)(jnp.asarray(pos))["total"]
    F = model.forces_fn(sysm)(jnp.asarray(pos), None)
    assert abs(r.energy - float(E)) < 1e-10 * abs(float(E))
    assert np.abs(r.forces - np.asarray(F)).max() < 1e-8
    assert np.abs(r.induced_dipoles).max() > 0


# ----------------------------------------------------------------------------- ASE
ase = pytest.importorskip("ase")


def test_ase_calculator_units_stress_and_dipoles(box):
    from ase import units

    from pgm_jax.interfaces.ase import PGMCalculator, atoms_from_system

    KJMOL_EV = units.kJ / units.mol  # ASE's constants (CODATA 2014)
    sysm, pos, H, s, eng = box
    r = eng.compute(pos, H, virial=True)
    atoms = atoms_from_system(sysm, pos, H)
    atoms.calc = PGMCalculator(eng)
    assert abs(atoms.get_potential_energy() - r.energy * KJMOL_EV) < 1e-9 * abs(r.energy * KJMOL_EV)
    assert np.abs(atoms.get_forces() - r.forces * KJMOL_EV / 10).max() < 1e-9
    st = atoms.get_stress(voigt=False)
    V = atoms.get_volume()
    assert np.allclose(st * V, r.virial * KJMOL_EV, atol=1e-9)
    # stress by finite differences of ASE energies (atomic scaling)
    cell0, p0 = atoms.get_cell().copy(), atoms.get_positions().copy()
    h = 1e-5
    for a, b in ((1, 1), (2, 0)):
        e = []
        for sg in (1, -1):
            F = np.eye(3)
            F[a, b] += sg * h
            atoms.set_cell(cell0 @ F.T, scale_atoms=False)
            atoms.set_positions(p0 @ F.T)
            e.append(atoms.get_potential_energy())
        fd = (e[0] - e[1]) / (2 * h) / V
        assert abs(fd - st[a, b]) < 1e-5 * np.abs(st).max()
    atoms.set_cell(cell0)
    atoms.set_positions(p0)
    assert np.allclose(atoms.calc.get_property("dipole", atoms), r.dipole * 10, atol=1e-10)
    assert np.allclose(atoms.calc.get_induced_dipoles(atoms), r.induced_dipoles * 10, atol=1e-10)


def test_ase_rigid_water_nve():
    """NVE with ASE's VelocityVerlet and FixRigidMolecules conserves the energy; waters stay rigid."""
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    from pgm_jax.interfaces.ase import PGMCalculator, atoms_from_system, rigid_constraints

    sysm, pos, H = water_box(0)
    s = settings(dipole_tol=1e-8)
    atoms = atoms_from_system(sysm, pos, H)
    atoms.calc = PGMCalculator(PGMEngine(sysm, pos, H, s))
    atoms.set_constraint(rigid_constraints(sysm))
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(0))
    dyn = VelocityVerlet(atoms, 0.5 * units.fs)
    E = []
    for _ in range(20):
        dyn.run(5)
        E.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
    E = np.asarray(E)
    ke = 0.5 * (3 * 90 - 90 - 3) * units.kB * 300
    assert np.std(E) < 2e-3 * ke and abs(E[-1] - E[0]) < 2e-3 * ke
    x = atoms.get_positions().reshape(-1, 3, 3)
    d = np.linalg.norm(x[:, 1] - x[:, 0], axis=1)
    assert np.abs(d - d[0]).max() < 1e-9


def test_fix_rigid_molecules_equals_fix_bond_lengths():
    from ase.constraints import FixBondLengths

    from pgm_jax.interfaces.ase import atoms_from_system, rigid_blocks, rigid_constraints

    sysm, pos, H = water_box(2)
    rng = np.random.default_rng(3)
    a1, a2 = atoms_from_system(sysm, pos, H), atoms_from_system(sysm, pos, H)
    a1.set_constraint(rigid_constraints(sysm))
    a2.set_constraint(FixBondLengths([p for b in rigid_blocks(sysm) for p in b]))
    new = a1.get_positions() + 0.01 * rng.normal(size=(len(a1), 3))
    a1.set_positions(new)
    a2.set_positions(new)
    assert np.abs(a1.get_positions() - a2.get_positions()).max() < 1e-9
    p = rng.normal(size=(len(a1), 3))
    a1.set_momenta(p)
    a2.set_momenta(p)
    assert np.abs(a1.get_momenta() - a2.get_momenta()).max() < 1e-9


# ----------------------------------------------------------------------------- i-PI
class FakeIPI:
    """The server side of the i-PI socket protocol (as i-PI's interfaces/sockets.py sends it)."""

    def __init__(self, path):
        self.path = path
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        self.srv.listen(1)

    def accept(self):
        self.c, _ = self.srv.accept()

    def recv(self, n):
        b = b""
        while len(b) < n:
            b += self.c.recv(n - len(b))
        return b

    def msg(self, s):
        return s.upper().ljust(12).encode()

    def status(self):
        self.c.sendall(self.msg("STATUS"))
        return self.recv(12).decode().strip()

    def init(self, text=""):
        t = text.encode()
        self.c.sendall(self.msg("INIT") + np.int32(0).tobytes() + np.int32(len(t)).tobytes() + t)

    def posdata(self, h, pos):
        self.c.sendall(
            self.msg("POSDATA")
            + h.tobytes()
            + np.linalg.inv(h).tobytes()
            + np.int32(len(pos)).tobytes()
            + pos.tobytes()
        )

    def posdata_batch(self, hs, poss):
        body = b"".join(h.tobytes() + np.linalg.inv(h).tobytes() for h in hs)
        self.c.sendall(
            self.msg("POSDATA")
            + np.int32(len(poss[0])).tobytes()
            + body
            + np.concatenate([p.reshape(-1) for p in poss]).tobytes()
        )

    def getforce(self, batch=1):
        self.c.sendall(self.msg("GETFORCE"))
        assert self.recv(12).decode().strip() == "FORCEREADY"
        if batch == 1:
            E = np.frombuffer(self.recv(8), np.float64)[0]
            n = int(np.frombuffer(self.recv(4), np.int32)[0])
            F = np.frombuffer(self.recv(24 * n), np.float64).reshape(n, 3)
            vir = np.frombuffer(self.recv(72), np.float64).reshape(3, 3)
            k = int(np.frombuffer(self.recv(4), np.int32)[0])
            return [(E, F, vir, self.recv(k).decode())]
        E = np.frombuffer(self.recv(8 * batch), np.float64)
        n = int(np.frombuffer(self.recv(4), np.int32)[0])
        F = np.frombuffer(self.recv(24 * n * batch), np.float64).reshape(batch, n, 3)
        vir = np.frombuffer(self.recv(72 * batch), np.float64).reshape(batch, 3, 3)
        ex = []
        for _ in range(batch):
            k = int(np.frombuffer(self.recv(4), np.int32)[0])
            ex.append(self.recv(k).decode())
        return [(E[i], F[i], vir[i], ex[i]) for i in range(batch)]


def test_ipi_client_protocol_and_units(tmp_path):
    import json

    from pgm_jax.interfaces.ipi import IPIClient
    from pgm_jax.units import BOHR_NM_CODATA2022, HARTREE_KJMOL

    sysm, pos, H = water_box(3)
    s = settings()
    ref = PGMEngine(sysm, pos, H, s)
    name = f"pgmtest{os.getpid()}"
    prefix = str(tmp_path) + "/ipi_"
    srv = FakeIPI(prefix + name)
    client = IPIClient(lambda p, c: PGMEngine(sysm, p, c, s, slots=2), name, unix=True, sockets_prefix=prefix, log=None)
    th = threading.Thread(target=client.run, daemon=True)
    th.start()
    srv.accept()
    h = np.asarray(H).T / BOHR_NM_CODATA2022  # i-PI's cell: lattice vectors as columns, Bohr
    x = pos / BOHR_NM_CODATA2022
    assert srv.status() == "NEEDINIT"
    srv.init("")
    assert srv.status() == "READY"
    srv.posdata(np.ascontiguousarray(h), np.ascontiguousarray(x))
    assert srv.status() == "HAVEDATA"
    ((E, F, vir, ex),) = srv.getforce()
    r = ref.compute(pos, H, virial=True)
    assert abs(E * HARTREE_KJMOL - r.energy) < 1e-9 * abs(r.energy)
    assert np.abs(F * HARTREE_KJMOL / BOHR_NM_CODATA2022 - r.forces).max() < 1e-7
    assert np.abs(-vir * HARTREE_KJMOL - r.virial).max() < 1e-7
    assert np.allclose(np.asarray(json.loads(ex)["dipole"]) * BOHR_NM_CODATA2022, r.dipole, atol=1e-10)
    # batched request (INIT announces batch_size): two structures, one per slot
    srv.init("batch_size:2")
    y = pos + 0.001
    srv.posdata_batch(
        [np.ascontiguousarray(h)] * 2, [np.ascontiguousarray(x), np.ascontiguousarray(y / BOHR_NM_CODATA2022)]
    )
    assert srv.status() == "HAVEDATA"
    out = srv.getforce(batch=2)
    r2 = ref.compute(y, H)
    assert abs(out[1][0] * HARTREE_KJMOL - r2.energy) < 1e-9 * abs(r2.energy)
    srv.c.sendall(srv.msg("EXIT"))
    th.join(timeout=30)
    assert not th.is_alive() and client.stats["structures"] == 3


def _ipi_available():
    try:
        from pgm_jax.interfaces import ipi_tools

        ipi_tools.ipi_command()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ipi_available(), reason="i-PI not available (set IPI_ROOT)")
def test_ipi_real_server_short_nvt(tmp_path):
    """A real i-PI server (classical NVT, then 2 beads batched) driven by the pgm_jax client: the
    step-0 potential is the engine's energy; the conserved quantity is conserved."""
    from test_flexible import _box

    from pgm_jax.interfaces import ipi_tools as T
    from pgm_jax.interfaces.ipi import IPIClient

    tpl, sysm, pos, H = _box()
    s = MDSettings(precision="double", dipole_tol=1e-8, cutoff=0.6, skin=0.05, lj_lrc=False)
    tpls = [tpl] * sysm.nmol
    E0 = PGMEngine(sysm, pos, H, s, templates=tpls).compute(pos, H).energy
    symbols = [e for m in sysm.molecules for e in m.elements]
    for nb in (1, 2):
        name = f"pgmreal{os.getpid()}_{nb}"
        wd = str(tmp_path / f"b{nb}")
        T.write_input(
            wd, symbols, pos, H, sysm.masses, nbeads=nb, steps=20, dt_fs=0.25, stride=1, address=name, batch_size=nb
        )
        client, st, props, wall = T.run(
            wd,
            name,
            lambda: IPIClient(
                lambda p, c: PGMEngine(sysm, p, c, s, templates=tpls, slots=nb), name, unix=True, log=None
            ),
        )
        assert abs(props["potential"][0] - E0) < 1e-6 * abs(E0)
        cons = props["conserved"]
        assert np.abs(cons - cons[0]).max() < 0.05 * abs(props["kinetic_md"][0])


# ----------------------------------------------------------------------------- OpenMM
def _openmm():
    try:
        import openmm

        return openmm if hasattr(openmm, "PythonForce") else None
    except ImportError:
        return None


@pytest.mark.skipif(_openmm() is None, reason="OpenMM >= 8.4 (PythonForce) not available")
def test_openmm_pythonforce_energy_forces_and_nve():
    import openmm
    from openmm import unit

    from pgm_jax.interfaces.openmm import PGMOpenMM

    sysm, pos, H = water_box(4)
    s = settings(dipole_tol=1e-8)
    eng = PGMEngine(sysm, pos, H, s)
    r = eng.compute(pos, H)
    om = PGMOpenMM(eng)
    system = om.system(rigid=True)
    assert system.getNumConstraints() == 90
    integ = openmm.VerletIntegrator(0.0005 * unit.picoseconds)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPeriodicBoxVectors(*[openmm.Vec3(*v) for v in om.box()])
    ctx.setPositions(om.positions())
    st = ctx.getState(getEnergy=True, getForces=True)
    assert abs(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) - r.energy) < 1e-8 * abs(r.energy)
    F = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    assert np.abs(F - r.forces).max() < 1e-6
    ctx.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    ctx.applyVelocityConstraints(1e-10)
    E = []
    for _ in range(10):
        integ.step(10)
        st = ctx.getState(getEnergy=True)
        E.append((st.getPotentialEnergy() + st.getKineticEnergy()).value_in_unit(unit.kilojoule_per_mole))
    ke = 0.5 * (3 * 90 - 90 - 3) * KB * 300
    assert np.std(E) < 3e-3 * ke
