"""Cross-feature checks of the merged feature branches (docs/CHANGES_2026-09.md): extended-Lagrangian
dipoles in an external field (constant E and constant D: exact shadow forces, the field-polarized
solution), biases together with a field and with iEL, walkers with a time-dependent field,
multiple time stepping in FlexibleSimulation.minimize, and the combinations that are refused."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_grad import water
from test_hmr import _cluster
from test_md import settings, small_box

from pgm_jax import System
from pgm_jax.bias import BiasSet, Harmonic, cv
from pgm_jax.md import efield as EF
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.simulation import Simulation
from pgm_jax.units import KB

E1 = np.array([0.3, -0.5, 0.8])  # V/nm
DD = np.array([1.0, -2.0, 3.0])  # D / eps0, V/nm


def _field(kind):
    return (jnp.asarray(E1), None) if kind == "E" else (jnp.asarray(DD), None, "D")


def _rel(a, b):
    return float(jnp.sqrt(jnp.sum((a - b) ** 2) / jnp.sum(b**2)))


# ----------------------------------------------------------------------------- iEL in a field
@pytest.mark.parametrize(
    "kind,precond,omega", [("E", "jacobi", 1.0), ("E", "block", 0.9), ("D", "jacobi", 1.0), ("D", "block", 0.9)]
)
def test_iel_shadow_forces_are_exact_in_a_field(kind, precond, omega):
    """iEL/0-SCF with an external field: the warm-up gives the field-polarized SCF dipoles, x at that
    solution has zero residual (the field is on the right-hand side of the auxiliary-dipole step),
    and the shadow forces match finite differences of the shadow energy (at constant D including the
    kappa |sum delta|^2 / 2 term, which the preconditioner does not contain)."""
    sys, pos, H = small_box(1)
    pos = jnp.asarray(pos)
    fld = _field(kind)
    ff = PGMForceField(sys, H, settings(iel="0scf", iel_precond=precond, iel_omega=omega))
    idx = ff.rows_for(pos, H)
    comp = jax.jit(lambda y, ind: ff.compute(y, H, idx, ind, efield=fld))
    ref = comp(pos, ff.init_induction())
    mu_star, e_star = ref.induction.mu, float(ref.energy["total"])
    scf = PGMForceField(sys, H, settings())
    r_scf = scf.compute(pos, H, idx, scf.init_induction(), efield=fld)
    assert _rel(mu_star, r_scf.induction.mu) < 1e-9
    assert abs(e_star - float(r_scf.energy["total"])) < 1e-9 * abs(e_star)
    assert abs(float(ref.energy["field"]) - float(r_scf.energy["field"])) < 1e-8
    # without the field in the auxiliary equations the residual at mu* would be the field itself
    at = ref.induction.set(count=jnp.asarray(100, jnp.int32), xl=ref.induction.xl.at[0].set(mu_star))
    r0 = comp(pos, at)
    assert int(r0.iterations) == 0
    assert _rel(r0.induction.mu, mu_star) < 1e-9 and abs(float(r0.energy["total"]) - e_star) < 1e-8 * abs(e_star)
    rng = np.random.default_rng(0)
    noise = jnp.asarray(rng.normal(size=mu_star.shape)) * float(jnp.sqrt(jnp.mean(mu_star**2)))

    def shadow(eps):
        return ref.induction.set(
            count=jnp.asarray(100, jnp.int32), xl=ref.induction.xl.at[0].set(mu_star + eps * noise)
        )

    ind = shadow(5e-2)
    res = comp(pos, ind)
    P = ff._atoms(None)
    ext = ff._ext(fld, H)
    M = ff.field_dipole(pos, P["q"], ff.perm_dipoles(pos, H, P["cov"]) + res.induction.mu, ext[1])
    _, F_hf = ff._energy_forces(pos, H, res.induction.mu, ff.geometry(pos, H, idx, P, forces=True), P, None, ext, M)
    E = jax.jit(lambda y: ff.compute(y, H, idx, ind, efield=fld).energy["total"])
    h = 3e-6
    for _ in range(3):
        v = jnp.asarray(rng.normal(size=pos.shape))
        fd = (float(E(pos + h * v)) - float(E(pos - h * v))) / (2 * h)
        an = -float(jnp.sum(res.forces * v))
        hf = -float(jnp.sum(F_hf * v))  # fixed-dipole forces at mu alone
        assert abs(fd - an) < 1e-7 * abs(an) + 1e-3, (fd, an)
        assert abs(fd - hf) > 10 * abs(fd - an), (fd, an, hf)
    # U~ - U* second order in the error of x
    d1 = float(comp(pos, shadow(1e-2)).energy["total"]) - e_star
    d2 = float(comp(pos, shadow(5e-3)).energy["total"]) - e_star
    assert abs(d1) > 1e-6 and 3.0 < d1 / d2 < 5.0, (d1, d2)


@pytest.mark.parametrize("kind", ["E", "D"])
def test_iel_scf_step_converges_to_the_field_polarized_dipoles(kind):
    """iEL/SCF (CG from x to tolerance) in a field: the operator and the right-hand side include the
    field (and at constant D its kappa term), so a step from a perturbed x ends at the SCF dipoles."""
    sys, pos, H = small_box(3)
    pos = jnp.asarray(pos)
    fld = _field(kind)
    ff = PGMForceField(sys, H, settings(iel="scf", iel_iter=0, dipole_tol=1e-11))
    idx = ff.rows_for(pos, H)
    ref = ff.compute(pos, H, idx, ff.init_induction(), efield=fld)
    mu = ref.induction.mu
    x = mu * 1.05
    res = ff.compute(
        pos, H, idx, ref.induction.set(count=jnp.asarray(100, jnp.int32), xl=ref.induction.xl.at[0].set(x)), efield=fld
    )
    assert int(res.iterations) > 0 and _rel(res.induction.mu, mu) < 1e-9
    assert abs(float(res.energy["total"]) - float(ref.energy["total"])) < 1e-9 * abs(float(ref.energy["total"]))


def _econs(sim, blocks=8, n=50):
    e = []
    for _ in range(blocks):
        sim.advance(n)
        e.append(sim.observables()["econs"])
    return np.asarray(e)


@pytest.mark.parametrize("field", [(0.0, 0.0, 1.0), "D"])
def test_iel_nve_conserves_energy_in_a_field(field):
    sys, pos, H = small_box(6, nm=0)
    fld = EF.displacement((0.0, 0.0, 2.0)) if field == "D" else field
    s = settings(cutoff=0.6, dipole_tol=1e-8, vdw="none", iel="0scf")
    sim = Simulation(sys, pos, H, settings=s, dt=0.001, thermostat=None, log=None, seed=1, efield=fld)
    ef0 = sim.observables()["field_energy"]
    e = _econs(sim)
    o = sim.observables()
    ke = 0.5 * sim.integ.dof * KB * 298
    assert int(sim.state.iters) == 0 and abs(o["field_energy"] - ef0) > 0.1
    assert np.std(e) < 4e-4 * ke and abs(e[-1] - e[0]) < 8e-4 * ke, (np.std(e) / ke, (e[-1] - e[0]) / ke)


# ----------------------------------------------------------------------------- biases
@pytest.mark.parametrize("iel", ["none", "0scf"])
def test_bias_with_field_and_iel(iel):
    """A static umbrella on an O-O distance together with an external field (and iEL/0-SCF): the
    bias forces add to the field forces, and NVE conserves econs."""
    pos, H, w = _cluster()
    sys = System([water()] * (len(pos) // 3))
    s = MDSettings(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False, iel=iel)
    d = cv.Distance(0, 9)
    d0 = float(d(jnp.asarray(pos), jnp.asarray(H)))
    kw = dict(dt=0.0005, thermostat=None, log=None, seed=2, efield=(0.0, 0.4, 0.8))
    sim = Simulation(sys, pos, H, s, bias=BiasSet([Harmonic([d], at=[d0 + 0.05], kappa=[2000.0])]), **kw)
    ref = Simulation(sys, pos, H, s, **kw)
    x = sim.rigid.positions(sim.state.dyn.position)
    g = jax.grad(lambda p: sim.integ.bias.energy(sim.state.bias, p, jnp.asarray(H)))(x)
    mapped = sim.rigid.forces(sim.state.dyn.position, -g)
    for a, b_, c in zip(
        jax.tree_util.tree_leaves(sim.state.dyn.force),
        jax.tree_util.tree_leaves(ref.state.dyn.force),
        jax.tree_util.tree_leaves(mapped),
    ):
        assert np.allclose(np.asarray(a) - np.asarray(b_), np.asarray(c), atol=1e-7 * np.abs(np.asarray(c)).max())
    o = sim.observables()
    assert abs(o["epot"] - ref.observables()["epot"] - o["ebias"]) < 1e-6 and o["ebias"] > 1.0
    assert abs(o["field_energy"] - ref.observables()["field_energy"]) < 1e-9
    E, B = [], []
    for _ in range(8):
        sim.advance(40)
        o = sim.observables()
        E.append(o["econs"])
        B.append(o["ebias"])
    assert max(B) - min(B) > 0.5, B
    assert np.ptp(E) < 0.05 * (max(B) - min(B)) + 2e-3, (np.ptp(E), B)


def test_walkers_book_the_work_of_a_time_dependent_field():
    """Independent walkers step through the same compiled step as a single simulation, including the
    heat booked for the explicit time dependence of E(t): walker 0 reproduces the single run."""
    from pgm_jax.bias.walkers import Walkers

    pos, H, w = _cluster()
    sys = System([water()] * (len(pos) // 3))
    s = MDSettings(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
    d = cv.Distance(0, 9)
    fld = EF.ExternalField((0.0, 0.0, 1.5), omega=2 * np.pi / 0.1)

    def mk():
        return Simulation(
            sys,
            pos,
            H,
            s,
            dt=0.001,
            thermostat="bussi",
            temperature=300.0,
            log=None,
            bias=BiasSet([Harmonic([d], at=[0.6], kappa=[500.0])], colvar=5),
            efield=fld,
        )

    sim, one = mk(), mk()
    wk = Walkers(sim, 2, shared=False, seed=4)
    one.state = wk.state(0).set(nbr=one.state.nbr)
    one.advance(60)
    wk.advance(60)
    h1, hw = float(one.state.heat), float(np.asarray(wk.S.heat)[0])
    assert abs(float(one.state.epot) - float(np.asarray(wk.S.epot)[0])) < 1e-6
    assert abs(h1 - hw) < 1e-6 * max(1.0, abs(h1)), (h1, hw)


# ----------------------------------------------------------------------------- other paths
def test_flexible_minimize_with_mts():
    from test_md_macro import _water_box

    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    from pgm_jax.md.mts import MTS

    pos, H, w = _water_box(n_side=4, spacing=0.31)
    wat = water()
    sys = System([wat] * (len(pos) // 3))
    s = MDSettings(precision="double", dipole_tol=1e-10, max_iter=300, cutoff=0.55, skin=0.05)
    sim = FlexibleSimulation(
        sys,
        [RigidTemplate(wat, w)] * sys.nmol,
        pos,
        H,
        s,
        dt=0.004,
        log=None,
        mts=MTS(inner=2, r_short=0.4, buffer=0.1),
    )
    out = sim.minimize(steps=3)
    assert out["steps"] >= 1
    sim.advance(4)
    assert np.isfinite(sim.observables()["epot"])


def test_refused_combinations():
    from test_pimd import T, _water_box

    from pgm_jax.interfaces.engine import PGMEngine
    from pgm_jax.md.flexible import FlexibleSimulation
    from pgm_jax.md.pimd import PIMDSimulation

    tpl, sys, pos, H = _water_box()
    s = MDSettings(precision="double", dipole_tol=1e-10, cutoff=0.5, skin=0.05, lj_lrc=False, max_iter=200)

    def flex(settings=s, **kw):
        return FlexibleSimulation(
            sys, [tpl] * sys.nmol, pos, H, settings, dt=0.0002, thermostat="bussi", temperature=T, log=None, **kw
        )

    ub = BiasSet([Harmonic([cv.Distance(0, 3)], at=[0.3], kappa=[100.0])])
    for kw in (
        dict(bias=ub),
        dict(efield=(0.0, 0.0, 0.1)),
        dict(constraints="h-bonds"),
        dict(settings=MDSettings(precision="double", cutoff=0.5, skin=0.05, lj_lrc=False, iel="0scf")),
    ):
        with pytest.raises((NotImplementedError, ValueError)):
            PIMDSimulation(flex(**kw), beads=4, log=None)
    for kw in (dict(bias=ub), dict(efield=(0.0, 0.0, 0.1))):
        with pytest.raises(NotImplementedError):
            PGMEngine.from_simulation(flex(**kw), templates=[tpl] * sys.nmol)
    with pytest.raises(NotImplementedError):
        PGMEngine(sys, pos, H, MDSettings(cutoff=0.5, skin=0.05, iel="0scf"), templates=[tpl] * sys.nmol)


# ----------------------------------------------------------------------------- full virial tensor
def _rotated_energy(ff, pos, H, mu, P, eps, com, efield=None):
    """Energy at fixed mu of the strained configuration, rotated back to a lower-triangular box
    (everything rotates: positions, box, induced dipoles and the field), where the engine is exact."""
    F = np.eye(3) + eps
    Hs = H @ F.T
    x = pos + ((com @ eps.T)[np.asarray(ff.mol)] if com is not None else pos @ eps.T)
    Q, R = np.linalg.qr(Hs.T)
    Q = Q @ np.diag(np.sign(np.diag(R)))  # Hs Q lower triangular, positive diagonal
    L = Hs @ Q
    assert np.abs(np.triu(L, 1)).max() < 1e-12
    x2 = jnp.asarray(x @ Q)
    fld = None if efield is None else (jnp.asarray(np.asarray(efield[0]) @ Q),) + tuple(efield[1:])
    idx = ff.rows_for(x2, jnp.asarray(L))
    return float(ff.energy_fixed_mu(x2, jnp.asarray(L), jnp.asarray(np.asarray(mu) @ Q), idx, P, fld)[0])


@pytest.mark.parametrize("molecular,field", [(True, None), (False, None), (False, "E"), (True, "D")])
def test_strain_derivative_full_tensor_matches_finite_differences(molecular, field):
    """PGMForceField.strain_derivative returns the whole tensor: every component (including the lower
    off-diagonal ones, which take the box out of lower-triangular form) against central differences of
    the energy of strained configurations rotated back to a lower-triangular box."""
    sys, pos, H = small_box(4)
    ff = PGMForceField(sys, H, settings(lj_lrc=True))
    idx = ff.rows_for(jnp.asarray(pos), H)
    fld = None if field is None else _field(field)
    mu = ff.compute(jnp.asarray(pos), H, idx, ff.init_induction(), efield=fld).induction.mu
    P = ff._atoms(None)
    W = np.asarray(ff.strain_derivative(jnp.asarray(pos), H, idx, mu, molecular=molecular, efield=fld))
    m = np.asarray(ff.masses)
    mol = np.asarray(ff.mol)
    com = None
    if molecular:
        com = (
            np.stack([np.bincount(mol, weights=m * pos[:, c]) for c in range(3)], 1)
            / np.bincount(mol, weights=m)[:, None]
        )
    h = 1e-5
    fd = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            e = np.zeros((3, 3))
            e[a, b] = h
            fd[a, b] = (
                _rotated_energy(ff, pos, np.asarray(H), mu, P, e, com, fld)
                - _rotated_energy(ff, pos, np.asarray(H), mu, P, -e, com, fld)
            ) / (2 * h)
    fd = fd - float(ff._vdw_tail(P, H)) * np.eye(3)
    scale = np.abs(fd).max()
    assert np.abs(W - fd).max() < 1e-6 * scale, (W - fd) / scale
    if not molecular and field is None:
        assert np.abs(W - W.T).max() < 1e-6 * scale
