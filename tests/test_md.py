"""MD engine: PME vs exact Ewald, forces and virial vs finite differences, neighbour lists,
rigid bodies, thermostat, energy conservation, I/O and exact restarts."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pgm_jax import PeriodicPGM, System  # noqa: E402
from pgm_jax.md.box import check_box, min_image, reduce_box  # noqa: E402
from pgm_jax.md.forcefield import MDSettings, PGMForceField  # noqa: E402
from pgm_jax.md.integrate import KB, Integrator  # noqa: E402
from pgm_jax.md.io import NetCDFTrajectory, read_coordinates, write_restart  # noqa: E402
from pgm_jax.md.neighbors import Neighbors  # noqa: E402
from pgm_jax.md.rigid import RigidMolecules, matrix_to_quaternion  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402
from test_grad import methanol, water  # noqa: E402

TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
RST = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")
W = water()
need_water_box = pytest.mark.skipif(not (os.path.exists(TOP) and os.path.exists(RST)), reason="pGM3P-25 box not available")


def small_box(seed=0, nw=30, nm=4):
    """Waters and methanols on a jittered lattice in a skewed (reduced) triclinic box, nm."""
    rng = np.random.default_rng(seed)
    H = reduce_box(np.array([[1.75, 0.0, 0.0], [0.45, 1.70, 0.0], [-0.40, 0.50, 1.65]]))
    t = np.radians(104.52 / 2)
    w = np.array([[0, 0, 0], [0.09572 * np.sin(t), 0.09572 * np.cos(t), 0], [-0.09572 * np.sin(t), 0.09572 * np.cos(t), 0]])
    m, xm = methanol()
    xm = xm - xm.mean(0)
    mols, pos = [], []
    grid = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(3)], float)
    grid = (grid + 0.5) / np.array([4, 4, 3])
    for n, f in enumerate(grid[rng.permutation(len(grid))][:nw + nm]):
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        c = f @ H + rng.normal(scale=0.01, size=3)
        if n < nm:
            mols.append(m); pos.append(xm @ R.T + c)
        else:
            mols.append(W); pos.append((w - w.mean(0)) @ R.T + c)
    return System(mols), np.concatenate(pos), H





def settings(**kw):
    base = dict(cutoff=0.6, skin=0.05, ewald_beta=6.0, pme_grid=(48, 48, 48), pme_order=8, lj_lrc=False,
                dipole_tol=1e-12, max_iter=500, peek=0.0, extrap_order=0, precision="double")
    base.update(kw)
    return MDSettings(**base)


def ff_and_list(sys, pos, H, **kw):
    s = settings(**kw)
    ff = PGMForceField(sys, H, s)
    nb = Neighbors(sys.n, H, s.cutoff, s.skin)
    return ff, nb.allocate(pos, None, H).idx


def test_pme_matches_exact_ewald():
    sys, pos, H = small_box()
    ff, idx = ff_and_list(sys, pos, H)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    ew = PeriodicPGM(sys, H, pos, b0=6.0, rc=0.6).energy(pos)[0]["total"]
    P = ff._atoms(None)
    e_elec = res.energy["elec"]
    assert abs(float(e_elec) - float(ew)) < 2e-6 * abs(float(ew)), (float(e_elec), float(ew))
    mu_ew = PeriodicPGM(sys, H, pos, b0=6.0, rc=0.6).induced_dipoles(pos)
    assert np.allclose(res.induction.mu, mu_ew, atol=1e-6 * float(jnp.abs(mu_ew).max()))


def test_row_gradient_forces_equal_autodiff_and_finite_differences():
    sys, pos, H = small_box(1)
    ff, idx = ff_and_list(sys, pos, H)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    P = ff._atoms(None)
    F_ad = -jax.grad(lambda x: ff.energy_fixed_mu(x, H, res.induction.mu, idx, P)[0])(jnp.asarray(pos))
    assert np.allclose(res.forces, F_ad, atol=1e-9 * float(jnp.abs(F_ad).max()))
    e = jax.jit(lambda x: ff.energy(x, H, idx, ff.init_induction())[0])
    h = 1e-6
    for a, k in [(0, 0), (5, 1), (40, 2), (77, 0)]:
        d = np.zeros_like(pos); d[a, k] = h
        fd = -(float(e(pos + d)) - float(e(pos - d))) / (2 * h)
        assert abs(fd - float(res.forces[a, k])) < 1e-5 * max(1.0, abs(fd)), (a, k, fd, float(res.forces[a, k]))


def test_differentiable_forces_and_dipoles():
    """settings.differentiable: gradients of forces and induced dipoles (implicit differentiation of
    the dipole solve) against central differences with the dipoles re-solved at every point."""
    sys, pos, H = small_box(6)
    s = settings(differentiable=True, adjoint_tol=1e-12)
    ff = PGMForceField(sys, H, s)
    idx = ff.rows_for(pos, H)
    rng = np.random.default_rng(7)
    wF, wmu = rng.normal(size=pos.shape), rng.normal(size=pos.shape)

    def loss(theta, x):
        res = ff.compute(x, H, idx, ff.init_induction(), theta)
        return jnp.sum(wF * res.forces) + 1e3 * jnp.sum(wmu * res.induction.mu)

    theta0 = sys.params0
    x0 = jnp.asarray(pos)
    L = jax.jit(loss)
    g_th, g_x = jax.jit(jax.grad(loss, argnums=(0, 1)))(theta0, x0)
    # forward values do not depend on the option
    plain = PGMForceField(sys, H, settings())
    r0, r1 = jax.jit(plain.compute)(x0, H, idx, plain.init_induction()), jax.jit(ff.compute)(x0, H, idx, ff.init_induction())
    assert np.allclose(r0.forces, r1.forces, rtol=0, atol=1e-10) and np.allclose(r0.induction.mu, r1.induction.mu, rtol=0, atol=1e-14)
    # parameters: random directions scaled by each leaf
    leaves, tree = jax.tree_util.tree_flatten(theta0)
    for trial in range(3):
        v = [rng.normal(size=np.shape(l)) * np.maximum(np.abs(np.asarray(l)), 1e-3) for l in leaves]
        vt = jax.tree_util.tree_unflatten(tree, [jnp.asarray(a) for a in v])
        h = 1e-6
        plus = jax.tree_util.tree_map(lambda a, b: a + h * b, theta0, vt)
        minus = jax.tree_util.tree_map(lambda a, b: a - h * b, theta0, vt)
        fd = (float(L(plus, x0)) - float(L(minus, x0))) / (2 * h)
        ad = sum(float(jnp.sum(a * b)) for a, b in zip(jax.tree_util.tree_leaves(g_th), jax.tree_util.tree_leaves(vt)))
        assert abs(fd - ad) < 1e-6 * max(1.0, abs(fd)), (trial, fd, ad)
    # positions
    for trial in range(2):
        d = rng.normal(size=pos.shape) * 1e-3
        h = 1e-4
        fd = (float(L(theta0, x0 + h * d)) - float(L(theta0, x0 - h * d))) / (2 * h)
        ad = float(jnp.sum(g_x * d))
        assert abs(fd - ad) < 1e-6 * max(1.0, abs(fd)), (trial, fd, ad)


def test_molecular_strain_derivative():
    sys, pos, H = small_box(2)
    ff, idx = ff_and_list(sys, pos, H, lj_lrc=True)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    W = ff.strain_derivative(pos, H, idx, res.induction.mu)
    P = ff._atoms(None)
    from pgm_jax.lj import lj_long_range
    m = np.asarray(sys.masses)
    com = np.array([np.average(pos[sys.mol == k], 0, weights=m[sys.mol == k]) for k in range(sys.nmol)])
    e = jax.jit(lambda s: ff.energy(pos + (s * com)[sys.mol], H * (1 + s), idx, ff.init_induction())[0])
    h = 1e-6
    fd = (float(e(h)) - float(e(-h))) / (2 * h)
    tail = -3 * float(lj_long_range(P, abs(np.linalg.det(H)), 0.6))           # impulse term added to W
    assert abs(fd + tail - float(jnp.trace(W))) < 1e-5 * max(1.0, abs(fd)), (fd, float(jnp.trace(W)))


def test_neighbor_list_is_complete_in_skewed_box():
    rng = np.random.default_rng(3)
    H = reduce_box(np.array([[2.72, 0, 0], [-0.9067, 2.5645, 0], [-0.9067, -1.2823, 2.2210]]))   # truncated octahedron
    pos = rng.uniform(size=(600, 3)) @ H
    nb = Neighbors(600, H, 0.9, 0.1)
    idx = np.asarray(nb.allocate(pos, None, H).idx)
    got = {(i, int(j)) for i in range(600) for j in idx[i] if j < 600}
    d = np.asarray(min_image(jnp.asarray(pos[:, None] - pos[None]), jnp.asarray(H)))
    r = np.linalg.norm(d, axis=-1)
    brute = {(i, j) for i, j in zip(*np.nonzero(r < 1.0)) if i != j}
    # brute-force check of the minimum image itself: 27 images
    imgs = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]) @ H
    r27 = np.min(np.linalg.norm((pos[:5, None, None] - pos[None, :, None]) + imgs[None, None], axis=-1), axis=-1)
    assert np.allclose(np.minimum(r[:5], 1.0), np.minimum(r27, 1.0))
    assert brute <= got


def test_rigid_bodies_roundtrip():
    sys, pos, H = small_box(4)
    rig = RigidMolecules(sys, pos, H)
    assert rig.fit_rmsd < 1e-10
    assert np.allclose(rig.positions(rig.body0), pos, atol=1e-10)
    rng = np.random.default_rng(5)
    Rm = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    Rm = Rm * np.sign(np.linalg.det(Rm))
    from pgm_jax.md._jaxmd import rigid_body
    q = rigid_body.Quaternion(jnp.asarray(matrix_to_quaternion(Rm)))
    v = rng.normal(size=3)
    assert np.allclose(rigid_body.quaternion_rotate(q, jnp.asarray(v)), Rm @ v)
    # rigid velocity field -> momenta -> velocities
    body = rig.body0
    vc = rng.normal(size=(sys.nmol, 3))
    w = rng.normal(size=(sys.nmol, 3))
    rel = pos - np.asarray(body.center)[sys.mol]
    vel = vc[sys.mol] + np.cross(w[sys.mol], rel)
    mom = rig.momenta_from_velocities(body, jnp.asarray(pos), jnp.asarray(vel))
    assert np.allclose(rig.atom_velocities(body, mom), vel, atol=1e-9)


def test_langevin_equipartition():
    sys, pos, H = small_box(6)
    ff, _ = ff_and_list(sys, pos, H)
    rig = RigidMolecules(sys, pos, H)
    integ = Integrator(ff, rig, Neighbors(sys.n, H, 0.6, 0.05), dt=0.002, ensemble="nvt", temperature=300.0, gamma=5.0)
    from pgm_jax.md.integrate import Dynamics
    from pgm_jax.md._jaxmd import simulate
    zero = jax.tree_util.tree_map(jnp.zeros_like, rig.body0)
    dyn = simulate.canonicalize_mass(Dynamics(rig.body0, zero, zero, rig.mass, jax.random.PRNGKey(0)))
    step = jax.jit(lambda d: integ._ou_step(d, 0.002))
    kt, kr = [], []
    for i in range(3000):
        dyn = step(dyn)
        if i > 500:
            ke = simulate.kinetic_energy(dyn)
            ket = 0.5 * jnp.sum(dyn.momentum.center ** 2 / dyn.mass.center)
            kt.append(float(ket)); kr.append(float(ke - ket))
    n = sys.nmol * 3
    assert abs(np.mean(kt) / (0.5 * n * KB * 300) - 1) < 0.05
    assert abs(np.mean(kr) / (0.5 * n * KB * 300) - 1) < 0.05


def test_netcdf_trajectory_and_restart(tmp_path):
    from scipy.io import netcdf_file
    n = 7
    tr = NetCDFTrajectory(str(tmp_path / "t.nc"), n)
    H = np.array([[20.0, 0, 0], [3.0, 19.0, 0], [-2.0, 4.0, 18.0]])
    xs = [np.random.default_rng(k).normal(size=(n, 3)) for k in range(3)]
    for k, x in enumerate(xs):
        tr.write(k * 0.5, x, H)
    f = netcdf_file(str(tmp_path / "t.nc"), "r", mmap=False)
    assert f.variables["coordinates"].shape == (3, n, 3)
    assert np.allclose(np.array(f.variables["coordinates"][2]), xs[2], atol=1e-5)
    assert np.allclose(np.array(f.variables["time"][:]), [0, 0.5, 1.0])
    f.close()
    tr2 = NetCDFTrajectory(str(tmp_path / "t.nc"), n, append=True)
    tr2.write(1.5, xs[0], H)
    assert netcdf_file(str(tmp_path / "t.nc"), "r", mmap=False).variables["coordinates"].shape[0] == 4
    v = np.random.default_rng(9).normal(size=(n, 3))
    write_restart(str(tmp_path / "r.rst7"), xs[1], v, H, 12.5)
    x2, v2, box = read_coordinates(str(tmp_path / "r.rst7"))
    assert np.allclose(x2, xs[1]) and np.allclose(v2, v)


@need_water_box
def test_nve_energy_conservation_and_exact_restart(tmp_path):
    s = MDSettings(cutoff=0.8, skin=0.1, pme_grid=(48, 48, 48), dipole_tol=1e-6, precision="mixed")
    sim = Simulation.from_amber(TOP, RST, settings=s, ensemble="nve", dt=0.001, log=None)
    E = []
    for _ in range(10):
        sim._advance(100)
        E.append(sim.observables()["etot"])
    ke_scale = 0.5 * sim.integ.dof * KB * 298
    assert np.std(E) < 2e-4 * ke_scale and abs(E[-1] - E[0]) < 5e-4 * ke_scale, (np.std(E), E[-1] - E[0])
    # continuation from a checkpoint: the same trajectory up to floating-point summation order
    # (PME spreading uses atomic adds on the GPU, so runs are not bitwise reproducible)
    sim.save(str(tmp_path / "c"))
    sim._advance(20)
    x_cont, e_cont = sim.positions_nm(), sim.observables()["etot"]
    sim2 = Simulation.from_amber(TOP, RST, settings=s, ensemble="nve", dt=0.001, log=None)
    sim2.load(str(tmp_path / "c.chk"))
    assert int(sim2.state.step) == 1000 and abs(sim2.time_ps - 1.0) < 1e-12
    sim2._advance(20)
    assert np.abs(sim2.positions_nm() - x_cont).max() < 1e-4
    assert abs(sim2.observables()["etot"] - e_cont) < 0.1
