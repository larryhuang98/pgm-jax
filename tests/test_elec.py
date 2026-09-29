"""Electrostatics channel: parity with sander/pmemd-pgm, forces, many-body structure."""

import os

import numpy as np
import pytest

from pgm_jax.channels import ElecChannel
from pgm_jax.lj import LJChannel
from pgm_jax.model import Model
from pgm_jax.param import Molecule, read_prmtop_pgm
from pgm_jax.system import System
from pgm_jax.units import KCAL

AMBER_TEST = os.path.expanduser("~/amber25/test/pgm_4wat")
PGM3P25_TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")


def _water_4wat() -> Molecule:
    """Parameters of the Amber pgm_4wat regression test (w4_pgm.top)."""
    return read_prmtop_pgm(os.path.join(AMBER_TEST, "w4_pgm.top"))[0]


def _restart_coords(path):
    lines = open(path).read().split("\n")
    vals = [float(x) for l in lines[2:8] for x in l.split()]
    return np.array(vals[:36]).reshape(12, 3) * 0.1


@pytest.mark.skipif(not os.path.exists(AMBER_TEST), reason="Amber pgm_4wat test not available")
def test_sander_parity_4wat():
    """sander pGM (ipgm=1, gas phase, dipole_scf_tol=1e-7): EELEC = -2164.4829, VDWAALS = 6.7727 kcal/mol."""
    w = _water_4wat()
    sys = System([w] * 4)
    pos = _restart_coords(os.path.join(AMBER_TEST, "restrt0"))
    e = Model([lambda s: ElecChannel()]).energy_fn(sys)(pos, None)
    kcal = float(e["total"]) / KCAL
    assert abs(kcal - (-2164.4829)) / 2164.4829 < 1e-4, kcal
    vdw = float(Model([LJChannel()]).energy_fn(sys)(pos)["total"]) / KCAL  # same run: VDWAALS = 6.7727
    assert abs(vdw - 6.7727) < 1e-4, vdw


@pytest.mark.skipif(not os.path.exists(PGM3P25_TOP), reason="pGM3P-25 topology not available")
def test_prmtop_reader_pgm3p25():
    w = read_prmtop_pgm(PGM3P25_TOP)[0]
    assert w.elements == ["O", "H", "H"]
    assert np.isclose(w.q[0], -2.0405622) and np.isclose(w.radius[0], 0.0605150752)
    assert len(w.cov) == 4 and np.isclose(w.cov[0][2], -0.0191201350)
    # LJ from ACOEF/BCOEF: O only (A = 5.81935564e5 kcal/mol A^12, B = 5.94825035e2 kcal/mol A^6)
    A, B = 5.81935564e5, 5.94825035e2
    rmin_A, eps_kcal = (2 * A / B) ** (1 / 6), B * B / (4 * A)
    assert np.isclose(w.lj_rmin_half[0], rmin_A / 20) and np.isclose(w.lj_sqrt_eps[0] ** 2, eps_kcal * KCAL)
    assert w.lj_rmin_half[1] == 0 and w.lj_sqrt_eps[1] == 0
    assert sorted(w.bonds) == [(0, 1), (0, 2), (1, 2)] and np.isclose(w.masses[0], 16.0)


def _water_generic():
    return Molecule(
        "WAT",
        ["O", "H", "H"],
        ["ow", "hw", "hw"],
        np.array([-0.8, 0.4, 0.4]),
        np.array([0.06, 0.05, 0.05]),
        np.array([1.0e-3, 0.3e-3, 0.3e-3]),
        cov=[(0, 1, -0.02), (0, 2, -0.02), (1, 0, 0.008), (2, 0, 0.008)],
    )


def _monomer(shift, rot):
    t = np.radians(104.52 / 2)
    m = np.array(
        [[0, 0, 0], [0.09572 * np.sin(t), 0.09572 * np.cos(t), 0], [-0.09572 * np.sin(t), 0.09572 * np.cos(t), 0]]
    )
    return m @ rot.T + shift


def _trimer(rng):
    def R():
        return np.linalg.qr(rng.normal(size=(3, 3)))[0]

    return np.concatenate(
        [
            _monomer(np.zeros(3), np.eye(3)),
            _monomer(np.array([0.29, 0.05, 0]), R()),
            _monomer(np.array([0.1, 0.28, 0.05]), R()),
        ]
    )


def test_forces_match_finite_difference():
    rng = np.random.default_rng(0)
    sys = System([_water_generic()] * 3)
    pos = _trimer(rng)
    model = Model([lambda s: ElecChannel()])
    f = model.energy_fn(sys)
    F = np.asarray(model.forces_fn(sys)(pos, None))
    h = 1e-6
    for a, k in [(0, 0), (4, 1), (8, 2)]:
        d = np.zeros_like(pos)
        d[a, k] = h
        fd = -(float(f(pos + d, None)["total"]) - float(f(pos - d, None)["total"])) / (2 * h)
        assert abs(fd - F[a, k]) < 1e-5 * max(1.0, abs(fd)), (a, k, fd, F[a, k])


def test_permanent_is_pairwise_and_induction_is_not():
    rng = np.random.default_rng(1)
    sys = System([_water_generic()] * 3)
    coords = _trimer(rng)[None]
    res = Model([lambda s: ElecChannel()]).nbody(sys, coords)
    assert abs(res["nb3"]["perm"][0]) < 1e-9
    assert abs(res["nb3"]["ind"][0]) > 1e-3
    # consistency: interaction = sum of 2-body + 3-body
    tot = res["int"]["total"][0]
    assert abs(tot - (res["nb2"]["total"][0] + res["nb3"]["total"][0])) < 1e-9


def test_invariances_and_net_force_torque():
    """Energy invariant under rigid translation/rotation; net force and net torque vanish (isolated cluster)."""
    rng = np.random.default_rng(3)
    sys = System([_water_generic()] * 3)
    pos = _trimer(rng)
    model = Model([lambda s: ElecChannel()])
    f = model.energy_fn(sys)
    e0 = float(f(pos, None)["total"])
    Rm = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    assert abs(float(f(pos @ Rm.T + np.array([0.7, -0.2, 1.3]), None)["total"]) - e0) < 1e-8 * abs(e0)
    F = np.asarray(model.forces_fn(sys)(pos, None))
    assert np.abs(F.sum(0)).max() < 1e-8
    assert np.abs(np.cross(pos, F).sum(0)).max() < 1e-8


def test_ewald_matches_gas_phase_in_a_large_box():
    """Two waters in a 6 nm cubic box: periodic energy -> isolated energy (image dipole terms ~1e-3 kJ/mol);
    and independent of the Ewald splitting parameter."""
    from pgm_jax.ewald import PeriodicPGM

    rng = np.random.default_rng(4)
    sys = System([_water_generic()] * 2)
    pos = _trimer(rng)[:6] + 3.0
    H = np.eye(3) * 6.0
    e_gas = float(Model([lambda s: ElecChannel()]).energy_fn(sys)(pos, None)["total"])
    e1 = float(PeriodicPGM(sys, H, pos, b0=1.2, rc=2.9).energy(pos)[0]["total"])
    e2 = float(PeriodicPGM(sys, H, pos, b0=1.0, rc=2.9, k_tol=1e-10).energy(pos)[0]["total"])
    assert abs(e1 - e2) < 1e-4, (e1, e2)
    assert abs(e1 - e_gas) < 5e-3, (e1, e_gas)


def test_elec_decomposition():
    """elst + ind = supermolecular interaction; ind <= 0; a monomer's elst/ind are zero."""
    from pgm_jax.channels import elec_decomposition

    w = _water_generic()
    rng = np.random.default_rng(3)
    pos = _trimer(rng)
    sys = System([w] * 3)
    d = elec_decomposition(pos, sys)
    e = Model([lambda s: ElecChannel()]).nbody(sys, pos[None])["int"]
    assert abs(float(d["elec"]) - float(e["perm"][0] + e["ind"][0])) < 1e-8
    assert abs(float(d["elst"] + d["ind"] - d["elec"])) < 1e-10
    assert float(d["ind"]) < 0
    d1 = elec_decomposition(pos[:3], System([w]))
    assert abs(float(d1["elst"])) < 1e-10 and abs(float(d1["ind"])) < 1e-10
