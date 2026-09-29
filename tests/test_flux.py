"""Charge flux in MD (pgm_jax/md/flux.py): q(R) and covalent dipoles c(R) in the MD engine.

What is checked, and against what: the flux of a fitted template equals the bonded model's
(charges, covalent dipoles, gas-phase energy and forces); forces, molecular and atomic strain
derivatives and the differentiable path against autodiff and central differences with the dipoles
re-solved; the cell dipole with q(R); rigid molecules (constant shift); the gather tables of the
flux map; refusals (pmemd-pgm export, stray flux parameters); NVE energy conservation and flux-free
constrained bonds.

Tolerances: identities 1e-10 to 1e-15 (float64, dipoles to 1e-12); finite differences 1e-6 to
1e-7 relative (h = 1e-6).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import METHANOL_BONDS, flux_settings, flux_template, methanol

from pgm_jax.bonded import terms as T
from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec
from pgm_jax.channels import ElecChannel, perm_dipoles
from pgm_jax.md.box import lower_triangular_frame
from pgm_jax.md.dipoles import CellDipole
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box
from pgm_jax.md.flux import ChargeFlux, molecule_at
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.thermostats import Langevin
from pgm_jax.protein import write_pgm_prmtop
from pgm_jax.system import System
from pgm_jax.units import KB


@pytest.fixture(scope="module")
def box():
    """Return the 16-methanol flux box (fixture, module scope): template, System, sim, positions, box.

    16 flexible methanols with flux, bonds perturbed off their references (1.15 nm box).
    """
    tpl, _ = flux_template()
    n = 16
    pos, H = liquid_box(tpl, n, 0.55, seed=0, min_dist=0.18)
    pos = pos + 0.004 * np.random.default_rng(1).normal(size=pos.shape)
    sys_ = System([tpl.pgm] * n)
    sim = FlexibleSimulation(sys_, [tpl] * n, pos, H, flux_settings(), thermostat=None, log=None)
    return tpl, sys_, sim, np.asarray(sim.flex.pos0), jnp.asarray(H)


def test_flux_equals_bonded_model():
    """The MD engine's charge flux equals the bonded model's, and so do its forces.

    Charges and covalent dipoles as BondedModel._flux; the bonded model's gas-phase pGM energy is
    the gas-phase Model's at those charges; in MD, flux equals the flux-free engine run on the
    charges of the geometry, and the forces are the gas-phase model's gradient (one molecule).
    """
    tpl, x = flux_template()
    y = x + 0.004 * np.random.default_rng(0).normal(size=x.shape)
    sys1 = System([tpl.pgm])
    fl = ChargeFlux.from_templates(sys1, [tpl])
    assert fl.n_bonds == 5 and set(fl.params) == {"jb", "jc", "jc2"}
    assert np.array_equal(fl.cov_bond >= 0, np.ones(len(tpl.pgm.cov), bool))
    Q = sys1.expand()
    P = jax.tree_util.tree_map(jnp.asarray, tpl.P)
    q, c = fl.charges(jnp.asarray(y), jnp.eye(3) * 5.0, Q["q"], Q["cov"], fl.theta())
    qb, cb = tpl.model._flux(0, jnp.asarray(y), P, Q["q"], Q["cov"])
    assert float(jnp.abs(q - qb).max()) < 1e-15 and float(jnp.abs(c - cb).max()) < 1e-15
    assert float(jnp.abs(q - Q["q"]).max()) > 1e-3 and abs(float(jnp.sum(q) - jnp.sum(Q["q"]))) < 1e-14
    # gas phase: the bonded model's pGM energy = Model(ElecChannel) of the molecule at those charges
    mol_y = molecule_at(tpl, y)
    assert np.allclose(mol_y.q, q, rtol=0, atol=1e-15)
    model, d = tpl.model, tpl.model.nb[0]
    e_nb, _, st = model.nonbonded(0, jnp.asarray(y), P, state=True)
    ii, jj = sys1.pair_i, sys1.pair_j
    e_lj = model._vdw(jnp.asarray(y), Q, d["lj"], ii, jj, model._bij(Q["radius"][ii], Q["radius"][jj]))
    out, aux = ElecChannel().energy(jnp.asarray(y), System([mol_y]))
    assert abs(float(e_nb - e_lj - out["perm"] - out["ind"])) < 1e-10 * abs(float(out["perm"]))
    # MD: flux == no flux at the same charges; forces == gas-phase gradient (periodic images aside)
    s = flux_settings(cutoff=1.8, pme_grid=None, ewald_beta=4.0, pme_order=6)
    sim = FlexibleSimulation(sys1, [tpl], y + 2.0, np.eye(3) * 4.0, s, thermostat=None, log=None)
    H = jnp.eye(3) * 4.0
    xb = sim.flex.pos0
    idx = sim.ff.rows_for(xb, H)
    ff0 = PGMForceField(System([mol_y]), H, s, topology=sim.topology)
    r1 = sim.ff.compute(xb, H, idx, sim.ff.init_induction())
    r0 = ff0.compute(xb, H, idx, ff0.init_induction())
    assert abs(float(r1.energy["total"] - r0.energy["total"])) < 1e-11 * abs(float(r0.energy["total"]))
    assert np.allclose(r1.induction.mu, r0.induction.mu, rtol=0, atol=1e-12 * float(jnp.abs(r0.induction.mu).max()))
    F = np.asarray(sim.state.dyn.force)
    g = np.asarray(jax.grad(lambda R: model.energy(0, R, P)[0])(jnp.asarray(y)))
    P0 = dict(P, flux={k: 0.0 * v for k, v in P["flux"].items()})
    g0 = np.asarray(jax.grad(lambda R: model.energy(0, R, P0)[0])(jnp.asarray(y)))
    rms = np.sqrt(np.mean(g**2))
    assert np.abs(g - g0).max() > 0.1 * rms  # the flux forces matter
    assert np.abs(F + g).max() < 1e-3 * rms, (np.abs(F + g).max(), rms)


def test_flux_forces_and_strain_derivatives(box):
    """Flux forces and strain derivatives match autodiff and central differences.

    Forces = -dE/dR with q(R), c(R) (autodiff at fixed dipoles, and central differences with the
    dipoles re-solved); molecular and atomic strain derivatives vs differences of the energy.
    """
    tpl, sys_, sim, pos, H = box
    ff = PGMForceField(sys_, H, flux_settings(), topology=sim.topology, flux=sim.ff.flux)
    idx = ff.rows_for(pos, H)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    mu = res.induction.mu
    P = ff._atoms(None)
    F_ad = -jax.grad(lambda y: ff.energy_fixed_mu(y, H, mu, idx, P)[0])(jnp.asarray(pos))
    assert np.allclose(res.forces, F_ad, rtol=0, atol=1e-10 * float(jnp.abs(F_ad).max()))
    plain = PGMForceField(sys_, H, flux_settings(), topology=sim.topology)
    F0 = plain.compute(pos, H, idx, plain.init_induction()).forces
    assert float(jnp.abs(res.forces - F0).max()) > 0.05 * float(jnp.sqrt(jnp.mean(F0**2)))
    e = jax.jit(lambda y, h: ff.energy(y, h, idx, res.induction)[0])
    h = 1e-6
    for a, k in [(0, 0), (1, 1), (5, 2), (40, 0), (77, 1)]:
        d = np.zeros_like(pos)
        d[a, k] = h
        fd = -(float(e(pos + d, H)) - float(e(pos - d, H))) / (2 * h)
        assert abs(fd - float(res.forces[a, k])) < 1e-6 * max(1.0, abs(fd)), (a, k, fd, float(res.forces[a, k]))
    m, mol = np.asarray(sys_.masses), np.asarray(sys_.mol)
    com = np.array([np.average(pos[mol == k], 0, weights=m[mol == k]) for k in range(sys_.nmol)])
    W = ff.strain_derivative(pos, H, idx, mu)
    es = jax.jit(lambda t: ff.energy(pos + (t * com)[mol], H * (1 + t), idx, res.induction)[0])
    fd = (float(es(1e-6)) - float(es(-1e-6))) / 2e-6
    assert abs(fd - float(jnp.trace(W))) < 1e-7 * abs(fd), (fd, float(jnp.trace(W)))
    Wa = ff.strain_derivative(pos, H, idx, mu, molecular=False)  # bonds stretch: flux active
    # the engine assumes a lower-triangular box: strained boxes are rotated back (box.lower_triangular_frame)
    ea = jax.jit(lambda x, h: ff.energy(x, h, idx, res.induction)[0])

    def eas(s):
        """Return the energy after the atomic strain s, rotated back to a lower-triangular box."""
        return float(
            ea(*lower_triangular_frame(np.asarray(pos) @ (np.eye(3) + s).T, np.asarray(H) @ (np.eye(3) + s).T))
        )

    for a, b in ((0, 0), (2, 1), (1, 2)):
        E = np.zeros((3, 3))
        E[a, b] = 1e-6
        fd = (eas(E) - eas(-E)) / 2e-6
        assert abs(fd - float(Wa[a, b])) < 1e-6 * max(1.0, abs(fd)), (a, b, fd, float(Wa[a, b]))


def test_flux_differentiable_path(box):
    """The differentiable path with flux parameters matches central differences.

    settings.differentiable: gradients of forces and dipoles with respect to jb, jc, jc2 (and
    positions) and dE/djb against central differences with the dipoles re-solved.
    """
    tpl, sys_, sim, pos, H = box
    ff = PGMForceField(
        sys_, H, flux_settings(differentiable=True, adjoint_tol=1e-12), topology=sim.topology, flux=sim.ff.flux
    )
    idx = ff.rows_for(pos, H)
    rng = np.random.default_rng(2)
    wF, wmu = rng.normal(size=pos.shape), rng.normal(size=pos.shape)

    def loss(theta, x):
        """Return a weighted sum of forces and induced dipoles (the test functional)."""
        r = ff.compute(x, H, idx, ff.init_induction(), theta)
        return jnp.sum(wF * r.forces) + 1e3 * jnp.sum(wmu * r.induction.mu)

    theta0 = {**sys_.params0, "flux": {k: jnp.asarray(v) for k, v in ff.flux.params.items()}}
    L = jax.jit(loss)
    g_th, g_x = jax.jit(jax.grad(loss, argnums=(0, 1)))(theta0, jnp.asarray(pos))
    for trial in range(2):
        v = {
            k: jnp.asarray(rng.normal(size=a.shape) * np.maximum(np.abs(np.asarray(a)), 1e-2))
            for k, a in theta0["flux"].items()
        }
        h = 1e-6
        plus = {**theta0, "flux": {k: a + h * v[k] for k, a in theta0["flux"].items()}}
        minus = {**theta0, "flux": {k: a - h * v[k] for k, a in theta0["flux"].items()}}
        fd = (float(L(plus, pos)) - float(L(minus, pos))) / (2 * h)
        ad = sum(float(jnp.sum(g_th["flux"][k] * v[k])) for k in v)
        assert abs(fd - ad) < 1e-6 * max(1.0, abs(fd)), (trial, fd, ad)
    d = rng.normal(size=pos.shape) * 1e-3
    h = 1e-4
    fd = (float(L(theta0, pos + h * d)) - float(L(theta0, pos - h * d))) / (2 * h)
    assert abs(fd - float(jnp.sum(g_x * d))) < 1e-6 * max(1.0, abs(fd)), (fd, float(jnp.sum(g_x * d)))
    E = jax.jit(lambda th: ff.compute(pos, H, idx, ff.init_induction(), th).energy["total"])
    gE = jax.grad(E)(theta0)["flux"]["jb"]
    for j in range(len(gE)):
        h = 1e-4
        up = {**theta0, "flux": {**theta0["flux"], "jb": theta0["flux"]["jb"].at[j].add(h)}}
        dn = {**theta0, "flux": {**theta0["flux"], "jb": theta0["flux"]["jb"].at[j].add(-h)}}
        fd = (float(E(up)) - float(E(dn))) / (2 * h)
        assert abs(fd - float(gE[j])) < 1e-6 * max(1.0, abs(fd)), (j, fd, float(gE[j]))
    # reweighting / liquid fits (scripts/fit_liquid.py): dU/dtheta at the converged dipoles, Hellmann-Feynman
    mu = ff.compute(pos, H, idx, ff.init_induction(), theta0).induction.mu
    gU = jax.grad(lambda th: ff.energy_fixed_mu(pos, H, mu, idx, ff._atoms(th))[0])(theta0)["flux"]["jb"]
    assert np.allclose(gU, gE, rtol=1e-8, atol=1e-8), (gU, gE)


def test_flux_cell_dipole_and_rigid_molecules(box):
    """The cell dipole uses q(R), c(R); frozen charges of each geometry give the same energy.

    The cell dipole takes q(R), c(R); a molecule held rigid gets the constant shift of its
    geometry (molecule_at), which reproduces the flux engine's energy at that geometry.
    """
    tpl, sys_, sim, pos, H = box
    ff = sim.ff
    idx = ff.rows_for(pos, H)
    res = ff.compute(pos, H, idx, ff.init_induction())
    C = np.asarray(CellDipole(ff).components(pos, H, res.induction.mu))
    P = ff.charges_at(pos, H, ff._atoms(None))
    assert np.allclose(C[0], (np.asarray(P["q"])[:, None] * pos).sum(0), atol=1e-12)
    assert np.allclose(C[1], np.asarray(perm_dipoles(jnp.asarray(pos), sys_, P["cov"])).sum(0), atol=1e-12)
    q0 = np.asarray(sys_.expand()["q"])
    assert np.abs(C[0] - (q0[:, None] * pos).sum(0)).max() > 1e-4  # base charges would differ
    # rigid: each molecule's charges frozen at its own geometry give the same energy
    mols = [molecule_at(tpl, pos[sys_.atom_slice(k)], name=f"m{k}") for k in range(sys_.nmol)]
    ffr = PGMForceField(System(mols), H, flux_settings(), topology=sim.topology)
    rr = ffr.compute(pos, H, idx, ffr.init_induction())
    assert abs(float(rr.energy["total"] - res.energy["total"])) < 1e-10 * abs(float(res.energy["total"]))
    with pytest.raises(ValueError):
        molecule_at(FlexibleTemplate.from_fit(*_no_flux_fit()), None)


def _no_flux_fit():
    """Return (BondedModel, parameters) of methanol without charge flux."""
    m, x = methanol()
    model = BondedModel(
        [MolSpec("methanol", list(m.elements), METHANOL_BONDS, [1] * len(METHANOL_BONDS), 0, x, m)],
        BondedSettings(families=T.PAPER, lj14_scale=0.5),
    )
    return model, model.init_params()


def test_flux_map_tables():
    """The gather tables of the flux map: padding, minimum image and the VJP.

    The gather tables of the flux map: molecules without flux (padding) keep their charges,
    bonds across the box boundary are taken at the minimum image, and the map's vector-Jacobian
    product (the force pull-back) matches central differences.
    """
    tpl, x = flux_template()
    ntpl = FlexibleTemplate.from_fit(*_no_flux_fit())
    sys_ = System([tpl.pgm, ntpl.pgm, tpl.pgm])
    fl = ChargeFlux.from_templates(sys_, [tpl, ntpl, tpl])
    rng = np.random.default_rng(4)
    H = jnp.eye(3) * 1.2
    pos = jnp.asarray(
        np.concatenate(
            [x + rng.normal(scale=0.004, size=x.shape) + s for s in ([0.0, 0, 0], [0.4, 0, 0], [1.17, 0.5, 0.5])]
        )
    )
    Q, th = sys_.expand(), fl.theta()

    def f(y):
        """Return (q, c) of the flux map at positions y."""
        return fl.charges(y, H, Q["q"], Q["cov"], th)

    q, c = f(pos)
    assert np.array_equal(q[6:12], Q["q"][6:12]) and float(jnp.abs(q[12:] - Q["q"][12:]).max()) > 1e-3
    assert np.array_equal(c[10:20], Q["cov"][10:20])
    y = pos.at[13].add(-H[0])  # an atom wrapped by one box vector
    assert np.allclose(f(y)[0], q, rtol=0, atol=1e-14)
    phi, gc = jnp.asarray(rng.normal(size=sys_.n)), jnp.asarray(rng.normal(size=len(sys_.cov_i)))
    g = jax.vjp(f, pos)[1]((phi, gc))[0]

    def L(y):
        """Return a random linear functional of (q, c) at positions y."""
        return float(jnp.sum(phi * f(y)[0]) + jnp.sum(gc * f(y)[1]))

    for a, k in [(0, 0), (1, 2), (5, 1), (13, 0), (17, 2)]:
        d = jnp.zeros_like(pos).at[a, k].set(1e-6)
        fd = (L(pos + d) - L(pos - d)) / 2e-6
        assert abs(fd - float(g[a, k])) < 1e-7 * max(1.0, abs(fd)), (a, k, fd, float(g[a, k]))


def test_flux_refusals_and_options():
    """Flux order 1 has no jc2; stray, mis-shaped or invalid flux parameters and pmemd export are refused."""
    tpl, x = flux_template(order=1)
    fl = ChargeFlux.from_templates(System([tpl.pgm]), [tpl])
    assert set(fl.params) == {"jb", "jc"}
    ntpl = FlexibleTemplate.from_fit(*_no_flux_fit())
    assert ChargeFlux.from_templates(System([ntpl.pgm] * 2), [ntpl] * 2) is None
    mixed = ChargeFlux.from_templates(System([tpl.pgm, ntpl.pgm]), [tpl, ntpl])  # one molecule with flux
    assert mixed.n_bonds == 5 and np.all(mixed.cov_bond[len(tpl.pgm.cov) :] == -1)
    with pytest.raises(ValueError, match="pmemd-pgm has no charge flux"):
        write_pgm_prmtop(None, "unused.prmtop", templates=[tpl])
    sys1 = System([ntpl.pgm])
    ff = PGMForceField(sys1, np.eye(3) * 3.0, flux_settings())
    with pytest.raises(ValueError, match="no charge flux"):
        ff._atoms({**sys1.params0, "flux": {"jb": jnp.zeros(3), "jc": jnp.zeros(3)}})
    with pytest.raises(ValueError, match="shapes"):
        fl.theta({"flux": {"jb": jnp.zeros(2), "jc": jnp.zeros(3)}})
    with pytest.raises(ValueError):
        ChargeFlux([(0, 1)], [0.1], [0], [2.0], [], {"jb": [1.0], "jc": [0.0]}, 2)  # sign must be -1, 0, 1
    with pytest.raises(ValueError):
        PGMForceField(
            sys1,
            np.eye(3) * 3.0,
            flux_settings(),
            flux=fl.__class__([(0, 1)], [0.1], [0], [1.0], [], {"jb": [1.0], "jc": [0.0]}, 6),
        )


def test_flux_nve_and_constraints():
    """NVE with flux conserves the energy; constrained X-H bonds carry no flux.

    NVE conservation with flux (double precision, tight dipoles), and X-H constraints at the
    reference lengths leave those bonds without flux.
    """
    tpl, _ = flux_template()
    n = 32  # as test_flexible's NVE test (same box and cutoff)
    pos, H = liquid_box(tpl, n, 0.55, seed=0, min_dist=0.18)
    sys_ = System([tpl.pgm] * n)
    # beta 6 / nm: small real-space terms at the cutoff, so the check sees the integration (flux or not,
    # the hard-cutoff noise at the default 4 / nm is 8e-4 kT per degree of freedom here)
    s = MDSettings().replace(
        precision="double", dipole_tol=1e-8, cutoff=0.6, skin=0.05, lj_lrc=False, ewald_beta=6.0, pme_spacing=0.05
    )
    sim = FlexibleSimulation(
        sys_, [tpl] * n, pos, H, s, dt=0.0005, thermostat=Langevin(10.0), temperature=298.0, log=None
    )
    sim.advance(1000)
    sim2 = FlexibleSimulation(
        sys_,
        [tpl] * n,
        sim.positions(),
        np.asarray(sim.state.box),
        s,
        dt=0.0005,
        thermostat=None,
        velocities=sim.velocities(),
        log=None,
    )
    E = []
    for _ in range(10):
        sim2.advance(100)
        E.append(sim2.observables()["etot"])
    ke = 0.5 * sim2.integ.dof * KB * 298.0
    assert np.std(E) < 1e-3 * ke and abs(E[-1] - E[0]) < 2e-3 * ke, (np.std(E) / ke, (E[-1] - E[0]) / ke)
    simc = FlexibleSimulation(sys_, [tpl] * n, pos, H, s, dt=0.001, constraints="h-bonds", log=None)
    simc.advance(50)
    x = simc.state.dyn.position
    db = np.asarray(simc.ff.flux.deviations(x, simc.state.box)).reshape(n, 5)
    assert np.abs(db[:, 1:]).max() < 1e-7 and np.abs(db[:, 0]).max() > 1e-4  # C-H, O-H held; C-O free
