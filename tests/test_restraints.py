"""Restraints (md/restraints.py): Amber's flat-bottom forms in both MD drivers.

What is checked, and against what: the flat-bottom form against Amber's NMR restraint piece by
piece (1e-13) and its continuity; forces and the molecular strain derivative of every kind
against central differences (1e-6 relative, h = 1e-6); distances as true minimum images (27
images); dihedral sign and periodicity through +-180 deg; positional references under box scaling
("none", "fractional", "com"); both MD drivers (NVE with energy moving through the restraints,
rigid-body force mapping, the pressure contribution, a restrained atom held, Monte Carlo trials
that include the restraint energy); protein selections and position restraints.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import random_atoms_box, rigid_water_sim, water, water_cluster_box, water_lattice

from pgm_jax import System
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.restraints import (
    AngleRestraint,
    COMDistanceRestraint,
    DihedralRestraint,
    DistanceRestraint,
    PositionRestraint,
    Restraints,
    dihedral,
    harmonic,
    nmr_energy,
)
from pgm_jax.md.thermostats import Bussi, Langevin
from pgm_jax.units import BAR_PER_KJMOL_NM3, KB

DATA = os.path.join(os.path.dirname(__file__), "data")
PRM, CRD = os.path.join(DATA, "pep_wat.prmtop"), os.path.join(DATA, "pep_wat.inpcrd")


def _amber_nmr(x, r1, r2, r3, r4, k2, k3):
    """Return Amber's NMR restraint energy at x, piece by piece (reference implementation).

    Amber's NMR restraint, piece by piece.
    """
    if x < r1:
        return k2 * (r1 - r2) ** 2 + 2 * k2 * (r1 - r2) * (x - r1)
    if x < r2:
        return k2 * (x - r2) ** 2
    if x <= r3:
        return 0.0
    if x <= r4:
        return k3 * (x - r3) ** 2
    return k3 * (r4 - r3) ** 2 + 2 * k3 * (r4 - r3) * (x - r4)


def test_flat_bottom_form():
    """nmr_energy equals Amber's piecewise form, is C1 at the knots, and validates its bounds."""
    b, k2, k3 = (0.1, 0.3, 0.4, 0.7), 150.0, 250.0
    xs = np.linspace(-0.3, 1.2, 61)
    e = np.asarray(nmr_energy(jnp.asarray(xs), *b, k2, k3))
    assert np.allclose(e, [_amber_nmr(x, *b, k2, k3) for x in xs], rtol=1e-13, atol=1e-13)
    g = jax.grad(lambda x: nmr_energy(x, *b, k2, k3))
    for x in b:  # value and slope continuous at the knots
        h = 1e-7
        assert abs(float(nmr_energy(x + h, *b, k2, k3)) - float(nmr_energy(x - h, *b, k2, k3))) < 1e-4
        assert abs(float(g(x + h)) - float(g(x - h))) < 1e-4
    # harmonic: E = k (x - x0)^2 with infinite outer bounds, finite gradients everywhere
    gh = jax.vmap(jax.grad(lambda x: nmr_energy(x, *harmonic(0.5), 10.0, 10.0)))(jnp.asarray(xs))
    assert np.all(np.isfinite(gh)) and np.allclose(gh, 20.0 * (xs - 0.5))
    # r1 = r2 removes the lower wall (Amber)
    assert float(nmr_energy(-5.0, 0.2, 0.2, 0.4, 0.6, 100.0, 100.0)) == 0.0
    for bad in ((0.3, 0.2, 0.4, 0.5), (0.0, -np.inf, 0.4, 0.5), (0.1, 0.2, 0.4, 0.3)):
        with pytest.raises(ValueError):
            DistanceRestraint([0, 1], bad, k=1.0)
    with pytest.raises(ValueError):
        DistanceRestraint([0, 1], (0.1, 0.2, 0.3, 0.4), k=1.0, k2=2.0)


def _all_kinds(pos, H):
    """Return one restraint of every kind on the random-atom system.

    One restraint set of every kind, parameters chosen so that every region of the forms is
    visited (some restraints inside the flat bottom, some on the linear walls).
    """
    m = np.arange(1.0, len(pos) + 1)
    return Restraints(
        [
            PositionRestraint(
                [0, 1, 2, 3],
                pos[:4] + [[0.05, 0, 0], [0, 0.3, 0], [0.01, 0, 0], [0, 0, -0.9]],
                k=[400.0, 300.0, 500.0, 200.0],
                r0=[0.0, 0.1, 0.02, 0.0],
            ),
            PositionRestraint([4, 5], pos[4:6] + 0.04, k=600.0, scaling="fractional", box=H),
            PositionRestraint([6, 7, 8], pos[6:9] - 0.03, k=800.0, r0=0.01, scaling="com", box=H, weights=m[6:9]),
            DistanceRestraint(
                [[0, 9], [2, 11], [3, 12], [5, 13]],
                ([0.0, 0.1, 0.1, 0.15], 0.2, [0.3, 0.3, 0.3, 0.6], [0.5, 0.5, 0.5, 0.8]),
                k2=[100.0, 50.0, 80.0, 120.0],
                k3=[200.0, 70.0, 90.0, 60.0],
            ),
            AngleRestraint([[0, 4, 9], [1, 5, 10], [2, 6, 11]], (0.3, 1.0, 1.5, 2.2), k2=40.0, k3=30.0),
            DihedralRestraint([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]], (-2.5, -1.0, 0.5, 2.0), k=25.0),
            DihedralRestraint([[3, 5, 7, 13]], harmonic(np.pi), k=15.0),
            COMDistanceRestraint([0, 1, 2], [9, 10, 11, 12], (0.1, 0.25, 0.3, 0.4), k=300.0, masses=m),
        ]
    )


def test_forces_and_strain_derivative_match_finite_differences():
    """Restraint forces and the molecular strain derivative match central differences."""
    pos, H = random_atoms_box()
    rs = _all_kinds(pos, H)
    E = jax.jit(rs.energy)
    assert set(rs.energies(pos, H)) == {"position", "distance", "angle", "dihedral", "com_distance"}
    F = np.asarray(rs.forces(pos, H))
    h = 1e-6
    for a in range(len(pos)):
        for k in range(3):
            d = np.zeros_like(pos)
            d[a, k] = h
            fd = -(float(E(pos + d, H)) - float(E(pos - d, H))) / (2 * h)
            assert abs(fd - F[a, k]) < 1e-6 * max(1.0, abs(fd)), (a, k, fd, F[a, k])
    # every term contributes, and the forms visit several regions
    for t in rs:
        assert float(t.energy(pos, H)) > 0.0, t.describe()
    # distances are true minimum images (27 images)
    dr = rs.terms[3]
    imgs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) @ H
    ref = [np.min(np.linalg.norm(pos[j] - pos[i] + imgs, axis=1)) for i, j in dr.idx]
    assert np.allclose(dr.values(pos, H), ref, rtol=1e-12)
    # molecular strain derivative: three "molecules", centres scaled, molecules translated rigidly
    mol = np.repeat(np.arange(3), [5, 5, 4])
    w = np.arange(1.0, 15.0)
    W = np.asarray(rs.strain_derivative(pos, H, jnp.asarray(mol), w, 3))
    com = np.array([np.average(pos[mol == k], 0, weights=w[mol == k]) for k in range(3)])
    for i in range(3):
        for j in range(3):
            eps = np.zeros((3, 3))
            eps[i, j] = h

            def ep(s):
                """Return the energy with box and molecular centres strained by s eps."""
                return float(E(pos + (com @ (s * eps).T)[mol], H @ (np.eye(3) + s * eps).T))

            fd = (ep(1.0) - ep(-1.0)) / (2 * h)
            assert abs(fd - W[i, j]) < 1e-6 * max(1.0, abs(fd)), (i, j, fd, W[i, j])


def test_dihedral_sign_and_periodicity():
    """Dihedral restraints: the bonded code's sign, windows across +-180 deg, periodicity, smoothness."""
    from pgm_jax.bonded.terms.core import _dihedral

    x = jnp.asarray(np.random.default_rng(1).normal(size=(50, 4, 3)))
    assert np.allclose(dihedral(*(x[:, a] for a in range(4))), _dihedral(*(x[:, a] for a in range(4))), atol=1e-14)

    def quad(phi):  # dihedral 0-1-2-3 equal to phi (checked)
        """Return four atoms whose dihedral 0-1-2-3 is phi [rad]."""
        return (
            np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.15], [0.1 * np.cos(phi), 0.1 * np.sin(phi), 0.15]])
            + 0.5
        )

    H = np.eye(3) * 3.0
    for phi in np.radians([-179.0, -60.0, 0.0, 45.0, 120.0, 179.0]):
        assert abs(float(dihedral(*quad(phi))) - phi) < 1e-12
    k2, k3 = 30.0, 50.0
    dr = DihedralRestraint([0, 1, 2, 3], np.radians([150.0, 170.0, 190.0, 210.0]), k2=k2, k3=k3)

    def E(deg):
        """Return the restraint energy at a dihedral of deg degrees."""
        return float(dr.energy(quad(np.radians(deg)), H))

    assert E(175.0) == 0.0 and E(-175.0) == 0.0 and E(-170.0) == 0.0  # the window crosses +-180
    assert abs(E(-160.0) - k3 * np.radians(10.0) ** 2) < 1e-12  # -160 = 200 deg: upper wall
    assert abs(E(160.0) - k2 * np.radians(10.0) ** 2) < 1e-12
    assert abs(E(-120.0) - k3 * (np.radians(20.0) ** 2 + 2 * np.radians(20.0) * np.radians(30.0))) < 1e-12
    assert abs(E(35.0 + 360.0) - E(35.0)) < 1e-12
    # smooth through the crossing
    g = jax.grad(lambda x: dr.energy(x, H))
    for deg in (179.9, -179.9, 180.0):
        assert np.all(np.isfinite(g(jnp.asarray(quad(np.radians(deg))))))
    # a harmonic restraint at 180 deg is even about 180
    hr = DihedralRestraint([0, 1, 2, 3], harmonic(np.pi), k=10.0)
    for deg in (170.0, 100.0, 30.0):
        assert abs(float(hr.energy(quad(np.radians(deg)), H)) - float(hr.energy(quad(np.radians(-deg)), H))) < 1e-12


def test_reference_scaling():
    """Position references follow the box as "none", "fractional" or "com" prescribes."""
    pos, H = random_atoms_box(2)
    ref = pos[:5] + 0.02
    w = np.array([1.0, 12.0, 16.0, 1.0, 14.0])
    F = np.eye(3) + np.array([[0.02, 0.0, 0.0], [0.01, -0.015, 0.0], [0.005, 0.003, 0.01]])  # deformation
    Hn = H @ F.T
    r_none = PositionRestraint(range(5), ref, 100.0)
    r_frac = PositionRestraint(range(5), ref, 100.0, scaling="fractional", box=H)
    r_com = PositionRestraint(range(5), ref, 100.0, scaling="com", box=H, weights=w)
    c = (w[:, None] * ref).sum(0) / w.sum()
    assert np.allclose(r_none.reference(Hn), ref, atol=1e-14)
    assert np.allclose(r_frac.reference(Hn), ref @ F.T, atol=1e-14)
    assert np.allclose(r_com.reference(Hn), ref - c + c @ F.T, atol=1e-14)
    assert np.allclose(r_com.reference(H), ref, atol=1e-14) and np.allclose(r_frac.reference(H), ref, atol=1e-14)
    # atoms deformed with the box keep their fractional displacements ("fractional"); atoms moved
    # rigidly with the scaled centroid (the Monte Carlo barostat's molecular move) keep their
    # displacements and energy ("com")
    d = np.asarray(r_frac.displacements(pos, H))
    assert np.allclose(r_frac.displacements(pos @ F.T, Hn), d @ F.T, atol=1e-14)
    e0c = float(r_com.energy(pos, H))
    assert abs(float(r_com.energy(pos + c @ (F.T - np.eye(3)), Hn)) - e0c) < 1e-10 * e0c
    assert abs(float(r_none.energy(pos + c @ (F.T - np.eye(3)), Hn)) - float(r_none.energy(pos, H))) > 1e-3
    with pytest.raises(ValueError):
        PositionRestraint([0], ref[:1], 1.0, scaling="com")  # needs the reference box


def _cluster_restraints(pos, masses):
    """Return pulling restraints of every kind between the water cluster's oxygens.

    Restraints of every kind between the cluster's waters (oxygens 3k), pulling: ~50 kJ/mol
    move between the restraints and the molecules within a short run.
    """

    def oxygen(k):
        """Return the atom index of the oxygen of water k."""
        return 3 * k

    return Restraints(
        [
            PositionRestraint([oxygen(7)], pos[[oxygen(7)]] + [0.1, 0.0, 0.0], k=1500.0),
            DistanceRestraint([[oxygen(0), oxygen(1)]], harmonic(0.40), k=2000.0),
            AngleRestraint([[oxygen(2), oxygen(3), oxygen(7)]], harmonic(np.radians(70.0)), k=100.0),
            DihedralRestraint([[oxygen(4), oxygen(5), oxygen(6), oxygen(0)]], harmonic(np.radians(30.0)), k=10.0),
            COMDistanceRestraint(range(12, 15), range(18, 21), (0.0, 0.15, 0.2, 0.25), k=800.0, masses=masses),
        ]
    )


@pytest.mark.parametrize("engine", ["rigid", "atoms"])
@pytest.mark.slow
def test_nve_with_restraints_and_force_mapping(engine):
    """NVE with restraints conserves energy; restraint forces and virial map correctly.

    Both engines (rigid bodies; atoms with SHAKE / RATTLE): NVE conserves the energy while
    ~40 kJ/mol move between the restraints and the molecules (the error is the integrator's,
    4x smaller at half the step); removing the restraints on the same state changes the forces by
    exactly the restraint forces (for rigid bodies mapped to centre forces and torques as the force
    field's) and epot by erestraint.
    """
    pos, H, w = water_cluster_box()
    masses = np.asarray(System([water()]).masses)
    rs = _cluster_restraints(pos, np.tile(masses, len(pos) // 3))
    s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
    sim = rigid_water_sim(engine, pos, H, w, s, dt=0.001, thermostat=None, restraints=rs)
    o = sim.observables()
    assert abs(o["erestraint"] - float(rs.energy(sim.positions(), H))) < 1e-10 and o["erestraint"] > 40.0
    assert set(sim.restraint_energies()) == {"position", "distance", "angle", "dihedral", "com_distance"}
    E, R = [o["etot"]], [o["erestraint"]]
    for _ in range(8):
        sim.advance(40)
        o = sim.observables()
        E.append(o["etot"])
        R.append(o["erestraint"])
    assert max(R) - min(R) > 20.0, R
    assert np.max(np.abs(np.array(E) - E[0])) < 1e-3 * (max(R) - min(R)), (E, R)
    st = sim.state
    x = sim.rigid.positions(st.dyn.position)
    F_r = rs.forces(x, st.box)
    mapped = sim.rigid.forces(st.dyn.position, F_r) if engine == "rigid" else F_r
    W = rs.strain_derivative(x, st.box, sim.ff.mol, sim.ff.masses, sim.sys.nmol)
    dP = -float(jnp.trace(W)) / (3.0 * float(jnp.linalg.det(st.box))) * BAR_PER_KJMOL_NM3
    p_with = sim.pressure()
    sim.set_restraints(None)
    assert abs(p_with - sim.pressure() - dP) < 1e-6 * max(1.0, abs(dP)), (p_with, sim.pressure(), dP)
    assert "erestraint" not in sim.observables() and sim.restraint_energies() == {}
    for a, b, c in zip(
        jax.tree_util.tree_leaves(st.dyn.force),
        jax.tree_util.tree_leaves(sim.state.dyn.force),
        jax.tree_util.tree_leaves(mapped),
    ):
        c = np.asarray(c)
        assert np.allclose(np.asarray(a) - np.asarray(b), c, rtol=0, atol=1e-7 * np.abs(c).max())
    assert abs((o["epot"] - sim.observables()["epot"]) - o["erestraint"]) < 1e-6


@pytest.mark.parametrize("engine", ["rigid", "atoms"])
@pytest.mark.slow
def test_barostat_trials_include_restraints(engine):
    """Monte Carlo volume trials include the restraint energy (fixed vs com reference).

    Monte Carlo volume moves every step (tiny time step, zero initial velocities: only the
    barostat moves the atoms) with a stiff restraint on a water far from the origin.  With a fixed
    reference ("none") the trials pay the restraint energy, so it stays at a few kT and fewer
    moves are accepted; with a "com" reference (mass-weighted centroid = the centre of mass that
    the barostat scales) the reference moves with the molecule, the restraint energy stays zero
    and the volume moves freely.  (A trial energy without the restraints would let the fixed
    reference be dragged: ~0.02 nm, hundreds of kJ/mol.)
    """
    pos, H, w = water_lattice()
    m = np.tile(np.asarray(System([water()]).masses), len(pos) // 3)
    far = np.arange(3 * 63, 3 * 64)  # centre near (1.1, 1.1, 1.1) nm
    s = MDSettings().replace(precision="double", dipole_tol=1e-8, cutoff=0.55, skin=0.05)
    out = {}
    for scaling in ("none", "com"):
        r = PositionRestraint(far, pos[far], k=1e6, scaling=scaling, box=H, weights=m[far])
        sim = rigid_water_sim(
            engine,
            pos,
            H,
            w,
            s,
            dt=1e-5,
            thermostat=Langevin(1.0),
            barostat=MonteCarloBarostat(every=1),
            temperature=300.0,
            velocities=np.zeros_like(pos),
            restraints=r,
            seed=2,
        )
        V0 = sim.observables()["volume_nm3"]
        sim.advance(120)
        o = sim.observables()
        out[scaling] = (o["erestraint"], o["mc_accept"], abs(o["volume_nm3"] / V0 - 1.0))
    kT = KB * 300.0
    e_none, acc_none, dv_none = out["none"]
    e_com, acc_com, dv_com = out["com"]
    assert e_none < 5 * kT and e_com < 0.05 * kT, out
    assert acc_com > 0.2 and dv_com > 0.01 and acc_none < 0.5 * acc_com, out


def test_flexible_engine_restrained_atom_held():
    """A restrained oxygen moves to its reference and stays there (atom engine).

    Atom engine (rigid water by constraints, 2 fs, Bussi): an oxygen restrained 0.2 nm away
    from its start moves to its reference and stays there.
    """
    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate

    pos, H, w = water_lattice()
    wat = water()
    nmol = len(pos) // 3
    sys = System([wat] * nmol)
    ref = pos[[0]] + [[0.0, 0.2, 0.0]]
    rs = PositionRestraint([0], ref, k=3000.0, r0=0.01)
    s = MDSettings().replace(precision="double", dipole_tol=1e-8, cutoff=0.55, skin=0.05)
    sim = FlexibleSimulation(
        sys,
        [RigidTemplate(wat, w)] * nmol,
        pos,
        H,
        s,
        dt=0.002,
        temperature=300.0,
        thermostat=Bussi(0.1),
        restraints=rs,
        log=None,
    )
    L = H[0, 0]

    def dist():
        """Return the minimum-image distance of oxygen 0 from its reference [nm]."""
        return float(np.linalg.norm((lambda v: v - np.round(v / L) * L)(sim.positions()[0] - ref[0])))

    d = [dist()]
    for _ in range(6):
        sim.advance(50)
        o = sim.observables()
        d.append(dist())
        assert o["shake_err"] < 1e-9 and np.isfinite(o["econs"])
    assert d[0] > 0.19 and max(d[3:]) < 0.1, d
    assert abs(o["erestraint"] - 3000.0 * max(d[-1] - 0.01, 0.0) ** 2) < 1e-8


def test_protein_selections_and_position_restraints():
    """Protein selections (heavy, backbone, ca) and backbone position restraints with a com reference."""
    from pgm_jax.protein import load_amber

    asys = load_amber(PRM, CRD)
    sys = asys.system()
    prot = asys.molecules[0]
    heavy, bb, ca = asys.select("heavy"), asys.select("backbone"), asys.select("ca")
    el = np.array(sys.elements)
    assert len(heavy) == int(np.sum(el[: prot.n] != "H")) and np.all(el[heavy] != "H")
    assert {prot.atom_names[a] for a in bb} == {"N", "CA", "C", "O"} and len(ca) == 2  # ALA, SER
    assert len(asys.select("heavy", kinds=("water",))) == sum(m.kind == "water" for m in asys.molecules)
    pos = asys.system_positions()
    rs = asys.position_restraints(418.4, "backbone")
    assert len(rs) == 1 and rs.terms[0].scaling == "com" and np.array_equal(rs.terms[0].idx, bb)
    assert float(rs.energy(pos, asys.box)) < 1e-20
    # "com": the reference moves with the protein's centre (mass-weighted) under box scaling
    H2 = asys.box * 1.03
    t = rs.terms[0]
    c = np.average(pos[bb], 0, weights=np.asarray(prot.molecule.masses)[bb])
    assert float(rs.energy(pos + 0.03 * c, H2)) < 1e-20
    with pytest.raises(ValueError):
        asys.select("sidechain")
    assert t.describe().startswith("position")
