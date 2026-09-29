"""External electric fields (pgm_jax/md/efield.py): constant E, constant D and time-dependent fields.

What is checked, and against what:

- units: 1 V/nm on 1 e is 96.485 kJ/mol/nm (Faraday constant), the finite-field dielectric
  constant eps - 1 = 4 pi M / (V E) and its SI factor e / (eps0 nm);
- the gas-phase response: the induced dipole change is alpha_mol E exactly and the energy is
  E(0) - E.M0 - E.alpha.E / 2; net force zero, torque M x E, forces = -grad E;
- MD forces against autodiff at fixed dipoles and central differences with re-solved dipoles, for
  every electrostatics level; q E exactly for point charges;
- the periodic cell's linear response alpha_cell, dU/dE = -M and dM/dE = alpha_cell through the
  differentiable solve; constant D: E = D / eps0 - M / (eps0 V) and its energy, derivative and
  virial;
- zero field = the field-free engine; NVE with static, time-dependent (work booked as heat) and
  constant-D fields; charged molecules re-wrapped across the cell; set_field and checkpoints;
  finite-field replicas; MTS(inner=1) with a field; charge flux in a field.

Tolerances: identities 1e-9 to 1e-12 (float64, dipole tolerance 1e-12 of md_settings);
finite differences 1e-6 to 2e-6 relative (h = 1e-5 nm); energy conservation relative to the
kinetic energy scale 0.5 dof kB T of the small box.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import cluster, flux_settings, flux_template, md_settings, methanol, small_box, water_sim

from pgm_jax import ElecChannel, Model, Molecule, System
from pgm_jax.channels import molecular_polarizability
from pgm_jax.md import efield as EF
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.dipoles import CellDipole
from pgm_jax.md.forcefield import PGMForceField
from pgm_jax.md.simulation import Simulation
from pgm_jax.units import E_CHARGE_C, EPS0_SI, KB, KE

E1 = np.array([0.3, -0.5, 0.8])  # V/nm, deliberately strong and oblique


def test_field_unit_conversions():
    """Field units: V/nm to model units, the finite-field eps formula, the period of 1000 cm^-1."""
    assert abs(EF.VNM_TO_INTERNAL * KE - 96.48533212) < 1e-6  # 1 V/nm on 1 e: 96.485 kJ/mol/nm
    # eps - 1 = 4 pi M / (V F) in model units (F in e/nm^2) = EPS_FACTOR M / (V E) with E in V/nm
    M, V, E = 2.7, 15.4, 0.1
    assert abs(EF.finite_field_eps(M, V, E) - 1.0 - 4 * np.pi * M / (V * E * EF.VNM_TO_INTERNAL)) < 1e-6
    assert abs(EF.EPS_FACTOR - 18.0951) < 1e-4
    assert abs(EF.EPS_FACTOR / (E_CHARGE_C / (EPS0_SI * 1e-9)) - 1) < 1e-8  # = e / (eps0 nm), SI
    f = EF.ExternalField.from_wavenumber((0, 0, 1.0), 1000.0)
    assert abs(2 * np.pi / f.omega - 0.0333564) < 1e-6  # 1000 cm^-1: period 33.36 fs
    assert EF.as_field([0, 0, 1]).E0 == (0.0, 0.0, 1.0) and not EF.as_field([0, 0, 1]).time_dependent


@pytest.mark.parametrize("which", ["cluster", "methanol"])
def test_gas_phase_response_is_the_molecular_polarizability(which):
    """In the gas phase the field induces alpha_mol E and the energy is quadratic in E.

    sum mu(E) - sum mu(0) = alpha_mol E exactly, E(E) = E(0) - E.M0 - E.alpha.E / 2, and forces
    in the field = -grad E (the energy is variational in mu).
    """
    if which == "cluster":
        sys, x = cluster(np.random.default_rng(0))
    else:
        m, x = methanol()
        sys = System([m])
    x = jnp.asarray(x)
    P = sys.expand(None)
    e0, a0 = ElecChannel().energy(x, sys, None)
    e1, a1 = ElecChannel(efield=tuple(E1)).energy(x, sys, None)
    alpha = np.asarray(molecular_polarizability(x, sys))  # nm^3
    Ei = E1 * EF.VNM_TO_INTERNAL
    dmu = np.asarray(a1["mu"]).sum(0) - np.asarray(a0["mu"]).sum(0)
    assert np.allclose(dmu, alpha @ Ei, rtol=1e-9, atol=1e-14), (dmu, alpha @ Ei)
    M0 = (np.asarray(P["q"])[:, None] * np.asarray(x)).sum(0) + np.asarray(a0["p"]).sum(0) + np.asarray(a0["mu"]).sum(0)

    def tot(e):
        """Return the total of an energy dict [kJ/mol]."""
        return float(sum(e.values()))

    expect = tot(e0) - KE * (Ei @ M0 + 0.5 * Ei @ alpha @ Ei)
    assert abs(tot(e1) - expect) < 1e-9 * abs(expect), (tot(e1), expect)
    M1 = M0 + dmu
    assert abs(float(e1["field"]) + KE * Ei @ M1) < 1e-9
    # forces with the field: autodiff total vs central finite differences of the energy
    f = Model([ElecChannel(efield=tuple(E1))]).energy_fn(sys)
    F = -jax.grad(lambda y: f(y)["total"])(x)
    rng = np.random.default_rng(1)
    for _ in range(4):
        i, c = rng.integers(sys.n), rng.integers(3)
        h = 1e-5
        d = jnp.zeros_like(x).at[i, c].set(h)
        fd = -(float(f(x + d)["total"]) - float(f(x - d)["total"])) / (2 * h)
        assert abs(fd - float(F[i, c])) < 1e-6 * max(1.0, abs(fd)), (fd, float(F[i, c]))
    # the net force of a neutral system in a uniform field is zero; its torque is M x E
    assert np.abs(np.asarray(F).sum(0)).max() < 1e-9
    torque = np.cross(np.asarray(x), np.asarray(F)).sum(0)
    tau = KE * np.cross(M1, Ei)
    assert np.allclose(torque, tau, atol=1e-8 * max(1, np.abs(tau).max())), (torque, tau)


def _ff(sys, pos, H, **kw):
    """Return a PGMForceField (md_settings(**kw)) of the system and its neighbour rows at pos."""
    ff = PGMForceField(sys, H, md_settings(**kw))
    return ff, ff.rows_for(pos, H)


@pytest.mark.parametrize("elec", ["qpi", "qp", "q"])
def test_md_forces_equal_autodiff_and_finite_differences(elec):
    """MD forces in a field equal autodiff at fixed mu and central differences with mu re-solved.

    For elec = qpi, qp and q: forces vs autodiff 1e-9 of the largest force; the field energy is
    -E.M with M the cell dipole of CellDipole; central differences 2e-6 relative (h = 1e-5 nm); for
    point charges the field adds exactly q E to every force.
    """
    sys, pos, H = small_box(1)
    pos = jnp.asarray(pos)
    ff, idx = _ff(sys, pos, H, elec=elec)
    fld = (jnp.asarray(E1), None)
    res = jax.jit(lambda y: ff.compute(y, H, idx, ff.init_induction(), efield=fld))(pos)
    P = ff._atoms(None)
    F_ad = -jax.grad(lambda y: ff.energy_fixed_mu(y, H, res.induction.mu, idx, P, fld)[0])(jnp.asarray(pos))
    scale = float(jnp.abs(F_ad).max())
    assert float(jnp.abs(res.forces - F_ad).max()) < 1e-9 * scale
    e_fixed = ff.energy_fixed_mu(pos, H, res.induction.mu, idx, P, fld)
    assert abs(float(e_fixed[0]) - float(res.energy["total"])) < 1e-8 * abs(float(res.energy["total"]))
    assert abs(float(e_fixed[1]["field"]) - float(res.energy["field"])) < 1e-9
    # field term = -E.M with M the cell dipole of md/dipoles.py (neutral molecules)
    M = np.asarray(CellDipole(ff).components(pos, H, res.induction.mu)).sum(0)
    assert np.allclose(np.asarray(res.dipole), M, atol=1e-10)
    assert abs(float(res.energy["field"]) + EF.FARADAY_KJ * E1 @ M) < 1e-8
    # finite differences with the dipoles re-solved
    E = jax.jit(lambda y: ff.compute(y, H, idx, ff.init_induction(), efield=fld).energy["total"])
    rng = np.random.default_rng(2)
    for _ in range(3):
        i, c = rng.integers(sys.n), rng.integers(3)
        h = 1e-5
        d = jnp.zeros_like(jnp.asarray(pos)).at[i, c].set(h)
        fd = -(float(E(pos + d)) - float(E(pos - d))) / (2 * h)
        assert abs(fd - float(res.forces[i, c])) < 2e-6 * max(1.0, abs(fd)), (fd, float(res.forces[i, c]))
    # the field adds exactly q E to the forces of a charges-only model
    if elec == "q":
        r0 = ff.compute(pos, H, idx, ff.init_induction())
        dF = np.asarray(res.forces - r0.forces)
        assert np.allclose(dF, np.asarray(P["q"])[:, None] * E1[None, :] * EF.FARADAY_KJ, atol=1e-9)


def test_field_adds_no_molecular_virial():
    """A uniform field adds no molecular strain derivative for neutral molecules (1e-8 relative)."""
    sys, pos, H = small_box(2)
    pos = jnp.asarray(pos)
    ff, idx = _ff(sys, pos, H)
    fld = (jnp.asarray(E1), None)
    res = ff.compute(pos, H, idx, ff.init_induction(), efield=fld)
    W0 = ff.strain_derivative(pos, H, idx, res.induction.mu)
    W1 = ff.strain_derivative(pos, H, idx, res.induction.mu, efield=fld)
    assert float(jnp.abs(W1 - W0).max()) < 1e-8 * float(jnp.abs(W0).max())


def test_periodic_linear_response_is_the_cell_polarizability():
    """At fixed nuclei the cell's induced dipole responds as alpha_cell E and U is quadratic in E.

    At fixed nuclei: M_ind(E) - M_ind(0) = alpha_cell E (Ewald dipole couplings), and the energy
    is quadratic: U(E) = U(0) - E.M0 - E.alpha_cell.E / 2.
    """
    sys, pos, H = small_box(3)
    pos = jnp.asarray(pos)
    ff, idx = _ff(sys, pos, H)
    r0 = ff.compute(pos, H, idx, ff.init_induction())
    fld = (jnp.asarray(E1), None)
    r1 = ff.compute(pos, H, idx, ff.init_induction(), efield=fld)
    A = np.asarray(CellDipole(ff).polarizability(pos, H, idx, tol=1e-12))
    Ei = E1 * EF.VNM_TO_INTERNAL
    dM = np.asarray(r1.induction.mu).sum(0) - np.asarray(r0.induction.mu).sum(0)
    assert np.allclose(dM, A @ Ei, rtol=1e-8, atol=1e-12), (dM, A @ Ei)
    M0 = np.asarray(CellDipole(ff).components(pos, H, r0.induction.mu)).sum(0)
    expect = float(r0.energy["total"]) - KE * (Ei @ M0 + 0.5 * Ei @ A @ Ei)
    assert abs(float(r1.energy["total"]) - expect) < 1e-9 * abs(expect)


def test_field_derivatives_through_the_differentiable_solve():
    """dU/dE = -M (Hellmann-Feynman) and dM/dE = alpha_cell through the custom_vjp of the solve."""
    sys, pos, H = small_box(4)
    pos = jnp.asarray(pos)
    ff, idx = _ff(sys, pos, H, differentiable=True, adjoint_tol=1e-12)

    def f(E):
        """Return the force-field result at field E [V/nm]."""
        return ff.compute(pos, H, idx, ff.init_induction(), efield=(E, None))

    E = jnp.asarray(E1)
    res = f(E)
    g = jax.grad(lambda E: f(E).energy["total"])(E)
    assert np.allclose(np.asarray(g), -EF.FARADAY_KJ * np.asarray(res.dipole), rtol=1e-8)
    J = np.asarray(jax.jacrev(lambda E: f(E).dipole)(E))  # e nm per V/nm
    A = np.asarray(CellDipole(ff).polarizability(pos, H, idx, tol=1e-12)) * EF.VNM_TO_INTERNAL
    assert np.allclose(J, A, rtol=1e-7, atol=1e-12), (J, A)


def test_zero_field_is_the_field_free_engine():
    """A zero field gives the field-free energy, forces and trajectory (1e-12)."""
    sys, pos, H = small_box(5)
    pos = jnp.asarray(pos)
    ff, idx = _ff(sys, pos, H)
    r0 = ff.compute(pos, H, idx, ff.init_induction())
    rz = ff.compute(pos, H, idx, ff.init_induction(), efield=(jnp.zeros(3), None))
    assert r0.dipole is None and "field" not in r0.energy
    assert float(rz.energy["field"]) == 0.0
    assert abs(float(rz.energy["total"]) - float(r0.energy["total"])) < 1e-12 * abs(float(r0.energy["total"]))
    assert float(jnp.abs(rz.forces - r0.forces).max()) < 1e-12 * float(jnp.abs(r0.forces).max())
    kw = dict(settings=md_settings(cutoff=0.6, dipole_tol=1e-8), dt=0.001, thermostat=None, log=None, seed=3)
    a = Simulation(sys, pos, H, **kw)
    b = Simulation(sys, pos, H, efield=(0.0, 0.0, 0.0), **kw)
    a.advance(20)
    b.advance(20)
    assert np.abs(a.positions() - b.positions()).max() < 1e-12
    assert abs(a.observables()["econs"] - b.observables()["econs"]) < 1e-9
    assert b.state.fdip is not None and a.state.efield is None


def _econs_drift(sim, blocks=8, n=50):
    """Advance `blocks` blocks of n steps and return econs after each [kJ/mol]."""
    e = []
    for _ in range(blocks):
        sim.advance(n)
        e.append(sim.observables()["econs"])
    return np.asarray(e)


@pytest.mark.parametrize("engine", ["rigid", "flexible"])
def test_nve_conserves_energy_in_a_static_field(engine):
    """NVE in a static 1 V/nm field conserves the energy while the molecules turn in the field.

    Both engines, waters only, no van der Waals term (its truncation would dominate the drift in this
    tiny box): std and drift of E_tot below 2e-4 / 4e-4 of 0.5 dof kB T while the field energy
    changes by more than 0.1 kJ/mol; the field energy is -F E_z M_z.
    """
    sys, pos, H = small_box(6, nm=0)  # waters only (RigidTemplate: up to three atoms)
    pos = jnp.asarray(pos)
    # no van der Waals term: its truncation at the cutoff (not the field) would dominate the drift of this tiny box
    kw = dict(
        settings=md_settings(cutoff=0.6, dipole_tol=1e-8, vdw="none"),
        dt=0.001,
        thermostat=None,
        log=None,
        seed=1,
        efield=(0.0, 0.0, 1.0),
    )
    if engine == "rigid":
        sim = Simulation(sys, pos, H, **kw)
    else:
        from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate

        tpl = [RigidTemplate(m, pos[sys.atom_slice(k)]) for k, m in enumerate(sys.molecules)]
        sim = FlexibleSimulation(sys, tpl, pos, H, **kw)
    ef0 = sim.observables()["field_energy"]
    e = _econs_drift(sim)
    obs = sim.observables()
    ke_scale = 0.5 * sim.integ.dof * KB * 298
    assert abs(obs["field_energy"] - ef0) > 0.1, (obs["field_energy"], ef0)  # the molecules turn in the field
    assert np.std(e) < 2e-4 * ke_scale and abs(e[-1] - e[0]) < 4e-4 * ke_scale, (np.std(e), e[-1] - e[0], ke_scale)
    assert abs(obs["field_energy"] + EF.FARADAY_KJ * obs["Mz"]) < 1e-6


def test_time_dependent_field_work_is_booked():
    """E(t) = E0 cos(w t): E_tot changes by the field's work, econs is conserved."""
    sys, pos, H = small_box(7)
    pos = jnp.asarray(pos)
    fld = EF.ExternalField((0.0, 0.0, 1.5), omega=2 * np.pi / 0.2)  # 200 fs period
    sim = Simulation(
        sys,
        pos,
        H,
        settings=md_settings(cutoff=0.6, dipole_tol=1e-8, vdw="none"),
        dt=0.0005,
        thermostat=None,
        log=None,
        seed=2,
        efield=fld,
    )
    et, ec = [], []
    for _ in range(8):
        sim.advance(50)
        o = sim.observables()
        et.append(o["etot"])
        ec.append(o["econs"])
    assert abs(o["efield"] - 1.5 * abs(np.cos(fld.omega * o["time_ps"]))) < 1e-9
    assert np.ptp(et) > 10 * np.ptp(ec), (np.ptp(et), np.ptp(ec))
    ke_scale = 0.5 * sim.integ.dof * KB * 298
    assert np.ptp(ec) < 5e-4 * ke_scale, (np.ptp(ec), ke_scale)


def test_set_field_and_checkpoint(tmp_path):
    """set_field changes the field of a running simulation, and a checkpoint keeps it."""
    sys, pos, H = small_box(8)
    pos = jnp.asarray(pos)
    sim = Simulation(sys, pos, H, settings=md_settings(cutoff=0.6), dt=0.001, log=None, efield=(0.0, 0.0, 0.2))
    sim.set_field((0.0, 0.1, 0.0))
    assert np.allclose(np.asarray(sim.state.efield), [0, 0.1, 0])
    sim.advance(10)
    sim.save_checkpoint(str(tmp_path / "c.chk"))
    sim2 = Simulation(sys, pos, H, settings=md_settings(cutoff=0.6), dt=0.001, log=None, efield=(0.0, 0.0, 0.2))
    sim2.load_checkpoint(str(tmp_path / "c.chk"))
    assert np.allclose(np.asarray(sim2.state.efield), [0, 0.1, 0])


def _ions_box():
    """Return a water / methanol box with one Na+ / Cl- pair (single-atom rigid bodies) on free lattice points.

    Returns
    -------
    system : System
    positions : np.ndarray (N, 3)
        Positions [nm].
    box : np.ndarray (3, 3)
        Box [nm].
    """
    sys, pos, H = small_box(9, nw=28, nm=2)
    grid = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(3)], float)
    grid = (grid + 0.5) / np.array([4, 4, 3])
    free = grid[np.random.default_rng(9).permutation(len(grid))][30:32]  # lattice points small_box left empty
    na = Molecule(
        "NA",
        ["Na"],
        ["Na+"],
        np.array([1.0]),
        np.array([0.05]),
        np.array([0.2e-3]),
        lj_rmin_half=[0.18],
        lj_sqrt_eps=[0.3],
    )
    cl = Molecule(
        "CL",
        ["Cl"],
        ["Cl-"],
        np.array([-1.0]),
        np.array([0.08]),
        np.array([3.0e-3]),
        lj_rmin_half=[0.22],
        lj_sqrt_eps=[0.4],
    )
    x = free @ H
    return System(sys.molecules + [na, cl]), np.concatenate([pos, x]), H


def test_charged_molecules_energy_continuous_across_rewrapping():
    """Charged molecules driven across the cell by a field: the re-wrapping keeps econs continuous.

    Moving the Na+ by a lattice vector changes epot by -Q E.L (the unwrapped energy, 1e-6); during
    the run the driver books whole lattice vectors Q L in fshift and econs varies by less than 2e-3 of
    the kinetic scale; the barostat with a field is refused.
    """
    sys, pos, H = _ions_box()
    kw = dict(settings=md_settings(cutoff=0.6, dipole_tol=1e-8), dt=0.001, log=None, seed=4)
    sim = Simulation(sys, pos, H, thermostat=None, efield=(0.0, 0.0, 2.0), **kw)
    st = sim.state
    # shift the Na+ by a lattice vector: the wrap in the driver books Q L in fshift, epot unchanged
    body = st.dyn.position
    k = sys.nmol - 2
    moved = body.set(center=body.center.at[k].add(jnp.asarray(H[2])))
    sim.state = sim.integ.forces(st.set(dyn=st.dyn.set(position=moved)), False)
    e_moved = float(sim.state.epot)
    assert abs(e_moved - float(st.epot) + EF.FARADAY_KJ * 2.0 * H[2, 2]) < 1e-6  # unwrapped: -Q E.L
    e = _econs_drift(sim, blocks=6, n=50)  # the driver re-wraps the ions (the field drives them across the cell)
    n = np.asarray(sim.state.fshift) @ np.linalg.inv(np.asarray(sim.state.box))  # Q L: whole lattice vectors
    assert np.allclose(n, np.round(n), atol=1e-9) and np.abs(np.round(n)).sum() >= 1, n
    ke_scale = 0.5 * sim.integ.dof * KB * 298
    assert np.ptp(e) < 2e-3 * ke_scale, np.ptp(e)
    with pytest.raises(NotImplementedError):
        Simulation(sys, pos, H, barostat=MonteCarloBarostat(), efield=(0.0, 0.0, 0.1), **kw)


# ----------------------------------------------------------------------------- constant displacement
DD = np.array([1.0, -2.0, 3.0])  # D / eps0, V/nm


@pytest.mark.parametrize("elec", ["qpi", "q"])
def test_constant_d_forces_equal_autodiff_and_finite_differences(elec):
    """Constant-D forces equal autodiff and central differences; the D energy term is V eps0 E^2 / 2.

    E = D / eps0 - M / (eps0 V) (the Maxwell field); forces vs autodiff 1e-9, vs central differences
    2e-6 relative; for point charges the field adds q E(M) exactly and the net force is zero.
    """
    sys, pos, H = small_box(10)
    pos = jnp.asarray(pos)
    ff, idx = _ff(sys, pos, H, elec=elec)
    fld = (jnp.asarray(DD), None, "D")
    res = jax.jit(lambda y: ff.compute(y, H, idx, ff.init_induction(), efield=fld))(pos)
    P = ff._atoms(None)
    F_ad = -jax.grad(lambda y: ff.energy_fixed_mu(y, H, res.induction.mu, idx, P, fld)[0])(pos)
    assert float(jnp.abs(res.forces - F_ad).max()) < 1e-9 * float(jnp.abs(F_ad).max())
    # the energy term is V eps0 |E|^2 / 2 with E = D/eps0 - M/(eps0 V)
    V = abs(np.linalg.det(H))
    Emac = DD - EF.EPS_FACTOR * np.asarray(res.dipole) / V
    e_expect = KE * V / (8 * np.pi) * np.sum((Emac * EF.VNM_TO_INTERNAL) ** 2)
    assert abs(float(res.energy["field"]) - e_expect) < 1e-9 * e_expect, (float(res.energy["field"]), e_expect)
    E = jax.jit(lambda y: ff.compute(y, H, idx, ff.init_induction(), efield=fld).energy["total"])
    rng = np.random.default_rng(3)
    for _ in range(3):
        i, c = rng.integers(sys.n), rng.integers(3)
        h = 1e-5
        d = jnp.zeros_like(pos).at[i, c].set(h)
        fd = -(float(E(pos + d)) - float(E(pos - d))) / (2 * h)
        assert abs(fd - float(res.forces[i, c])) < 2e-6 * max(1.0, abs(fd)), (fd, float(res.forces[i, c]))
    # the net force is Q_tot E = 0 and the ions feel the Maxwell field: the field adds q E(M) exactly
    if elec == "q":
        r0 = ff.compute(pos, H, idx, ff.init_induction())
        dF = np.asarray(res.forces - r0.forces)
        assert np.allclose(dF, np.asarray(P["q"])[:, None] * Emac[None, :] * EF.FARADAY_KJ, atol=1e-9)


def test_constant_d_linear_response_derivative_and_virial():
    """Constant D: the linear response, dU/dD and the strain derivative with the D term.

    At fixed nuclei dM_ind = (1 + kappa alpha_cell)^-1 alpha_cell dD (kappa = 4 pi / V); dU/dD =
    V eps0 E(M) (through the differentiable solve); the strain derivative includes the volume dependence
    of the D term (vs finite differences of the molecular scaling at fixed mu).
    """
    sys, pos, H = small_box(11)
    pos = jnp.asarray(pos)
    ff, idx = _ff(sys, pos, H, differentiable=True, adjoint_tol=1e-12)

    def f(D):
        """Return the force-field result at displacement field D [V/nm]."""
        return ff.compute(pos, H, idx, ff.init_induction(), efield=(D, None, "D"))

    D1, D2 = jnp.asarray(DD), jnp.asarray(DD) + jnp.asarray([0.3, 0.2, -0.5])
    r1, r2 = f(D1), f(D2)
    V = abs(np.linalg.det(H))
    A = np.asarray(CellDipole(ff).polarizability(pos, H, idx, tol=1e-12))
    kap = 4 * np.pi / V
    dM = np.asarray(r2.dipole - r1.dipole)
    expect = np.linalg.solve(np.eye(3) + kap * A, A @ (np.asarray(D2 - D1) * EF.VNM_TO_INTERNAL))
    assert np.allclose(dM, expect, rtol=1e-8, atol=1e-12), (dM, expect)
    g = np.asarray(jax.grad(lambda D: f(D).energy["total"])(D1))
    Emac = np.asarray(D1) - EF.EPS_FACTOR * np.asarray(r1.dipole) / V
    assert np.allclose(g, KE * V / (4 * np.pi) * EF.VNM_TO_INTERNAL**2 * Emac, rtol=1e-8)
    # virial: d/d eps of E at fixed mu, molecules translated with their centres of mass
    ffd, _ = _ff(sys, pos, H)
    fld = (D1, None, "D")
    mu = r1.induction.mu
    W = np.asarray(ffd.strain_derivative(pos, H, idx, mu, efield=fld))
    P = ffd._atoms(None)
    m = np.asarray(sys.masses)
    mol = np.asarray(sys.mol)
    com = np.zeros((sys.nmol, 3))
    np.add.at(com, mol, m[:, None] * np.asarray(pos))
    com /= np.bincount(mol, weights=m)[:, None]

    def e(t):
        """Return the fixed-mu energy with box and molecular centres strained by t."""
        return float(ffd.energy_fixed_mu(pos + jnp.asarray(t * com[mol]), H * (1 + t), mu, idx, P, fld)[0])

    h = 1e-6
    fd = (e(h) - e(-h)) / (2 * h)
    tail = float(ffd._vdw_tail(P, H))
    assert abs(np.trace(W) + 3 * tail - fd) < 1e-5 * max(1.0, abs(fd)), (np.trace(W), fd)


def test_nve_conserves_energy_at_constant_displacement():
    """NVE at constant D conserves the energy and reports E_z = D_z - F M_z / V."""
    sys, pos, H = small_box(12, nm=0)
    sim = Simulation(
        sys,
        pos,
        H,
        settings=md_settings(cutoff=0.6, dipole_tol=1e-8, vdw="none"),
        dt=0.001,
        thermostat=None,
        log=None,
        seed=5,
        efield=EF.displacement((0.0, 0.0, 2.0)),
    )
    e = _econs_drift(sim)
    o = sim.observables()
    V = abs(np.linalg.det(np.asarray(sim.state.box)))
    assert abs(o["Emac_z"] - (2.0 - EF.EPS_FACTOR * o["Mz"] / V)) < 1e-9
    ke_scale = 0.5 * sim.integ.dof * KB * 298
    assert np.std(e) < 2e-4 * ke_scale and abs(e[-1] - e[0]) < 4e-4 * ke_scale, (np.std(e), e[-1] - e[0])


def test_field_replicas_batched_run_and_analysis(tmp_path):
    """Batched finite-field replicas record each replica's cell dipole and field; the analysis pairs them."""
    from pgm_jax.analysis.finite_field import analyse
    from pgm_jax.md.finite_field import FieldReplicas, read_series

    sys, pos, H = small_box(13, nm=0)
    sim = Simulation(
        sys,
        pos,
        H,
        settings=md_settings(cutoff=0.6, dipole_tol=1e-6),
        dt=0.001,
        thermostat="bussi",
        log=None,
        efield=(0.0, 0.0, 0.0),
    )
    rep = FieldReplicas(sim, [(0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.0, 0.0, 0.0)], seed=1)
    rep.run(40, sample_every=10, prefix=str(tmp_path / "ff"), report_every=20)
    meta, d = read_series(str(tmp_path / "ff.ffd"))
    assert d["M"].shape == (4, 3, 3) and np.allclose(meta["fields"][1], [0, 0, -0.5])
    for k in range(3):  # the recorded M is the cell dipole of each replica
        st = rep.state(k)
        p = sim.rigid.positions(st.dyn.position)
        M = np.asarray(CellDipole(sim.ff).components(p, st.box, st.induction.mu)).sum(0)
        assert np.allclose(d["M"][-1, k], M, atol=1e-6), (d["M"][-1, k], M)
        assert np.allclose(np.asarray(st.efield), meta["fields"][k])
    res = analyse(meta, d, skip_ps=0.0, nblocks=2)
    assert len(res["pairs"]) == 1 and len(res["zero"]) == 1 and len(res["single"]) == 2


@pytest.mark.parametrize("engine", ["rigid", "constraints"])
def test_mts_with_a_field_is_the_ordinary_integrator_at_one_fast_step(engine):
    """MTS with one fast step and a time-dependent field is the ordinary integrator.

    MTS(inner=1) with a time-dependent field = the ordinary step with it (field at the outer
    evaluations, the work booked the same way).
    """
    from pgm_jax.md.mts import MTS

    fld = EF.ExternalField((0.2, 0.0, 0.8), omega=30.0)
    out = []
    for m in (None, MTS(inner=1, r_short=0.4, buffer=0.1, anchor=False)):
        sim = water_sim(engine, m, thermostat=None, efield=fld)
        sim.advance(15)
        o = sim.observables()
        out.append((sim.positions(), o["econs"], float(sim.state.heat)))
    assert np.abs(out[0][0] - out[1][0]).max() < 1e-11
    assert abs(out[0][1] - out[1][1]) < 1e-8 * abs(out[0][1])
    assert abs(out[0][2] - out[1][2]) < 1e-9 and abs(out[0][2]) > 1e-4


@pytest.mark.parametrize("kind", ["E", "D"])
def test_charge_flux_with_a_field(kind):
    """Charge flux in an E or D field: forces match autodiff and central differences.

    Charge flux (q(R), c(R)): the field's potential -E . r enters the charge pull-back; forces vs
    autodiff at fixed mu and vs differences of the energy with the dipoles re-solved.
    """
    from pgm_jax.md.flexible import FlexibleSimulation, liquid_box

    tpl, _ = flux_template()
    n = 16
    pos, H = liquid_box(tpl, n, 0.55, seed=0, min_dist=0.18)
    pos = pos + 0.004 * np.random.default_rng(1).normal(size=pos.shape)
    sys = System([tpl.pgm] * n)
    sim = FlexibleSimulation(sys, [tpl] * n, pos, H, flux_settings(), thermostat=None, log=None)
    pos, H = jnp.asarray(sim.flex.pos0), jnp.asarray(H)
    ff = PGMForceField(sys, H, flux_settings(), topology=sim.topology, flux=sim.ff.flux)
    idx = ff.rows_for(pos, H)
    fld = (jnp.asarray(E1), None) if kind == "E" else (jnp.asarray(DD), None, "D")
    res = jax.jit(lambda y: ff.compute(y, H, idx, ff.init_induction(), efield=fld))(pos)
    P = ff._atoms(None)
    F_ad = -jax.grad(lambda y: ff.energy_fixed_mu(y, H, res.induction.mu, idx, P, fld)[0])(pos)
    assert float(jnp.abs(res.forces - F_ad).max()) < 1e-9 * float(jnp.abs(F_ad).max())
    e = jax.jit(lambda y: ff.energy(y, H, idx, res.induction, efield=fld)[0])
    h = 1e-6
    for a, k in [(0, 0), (1, 1), (5, 2), (40, 0)]:
        d = jnp.zeros_like(pos).at[a, k].set(h)
        fd = -(float(e(pos + d)) - float(e(pos - d))) / (2 * h)
        assert abs(fd - float(res.forces[a, k])) < 1e-6 * max(1.0, abs(fd)), (a, k, fd, float(res.forces[a, k]))
