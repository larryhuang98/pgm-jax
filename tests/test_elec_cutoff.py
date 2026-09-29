"""Separate real-space electrostatics cutoff (MDSettings.elec_cutoff): unchanged engine when it
equals the cutoff, exact decomposition of split rows into electrostatics at elec_cutoff and van der
Waals at cutoff, forces / virial / differentiable path against finite differences, exact special
pairs in both parts of the rows, row-capacity overflows of each part, the Ewald coefficient rule
and an accuracy bound for the recommended settings."""
import io

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from test_md import settings, small_box  # noqa: E402

from pgm_jax.md.forcefield import DSUM_TOL, PGMForceField, elec_cutoff_settings, ewald_beta_for  # noqa: E402
from pgm_jax.md.neighbors import Neighbors  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402
from pgm_jax.md.topology import MDTopology, MoleculeRule  # noqa: E402

RC_E, RC_V = 0.45, 0.6


def _setup(sys, pos, H, s, capacity=True, topology=None):
    """Force field, candidate rows (pair cutoff + skin) and, with `capacity`, sized (compacted) rows."""
    ff = PGMForceField(sys, H, s, topology=topology)
    idx = Neighbors(sys.n, H, s.pair_cutoff, s.skin).allocate(pos, None, H).idx
    if capacity:
        ff.size_rows(jnp.asarray(pos), jnp.asarray(H), idx)
    return ff, idx


def _evaluate(ff, idx, pos, H):
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    W = ff.strain_derivative(pos, H, idx, res.induction.mu)
    e_mc = ff.energy(pos, H, idx, ff.init_induction())[0]
    assert not bool(res.overflow)
    return res, W, e_mc


def test_elec_cutoff_equal_to_cutoff_is_the_single_cutoff_engine():
    sys, pos, H = small_box(1)
    for capacity in (False, True):
        ff0, idx = _setup(sys, pos, H, settings(), capacity)
        ff1, _ = _setup(sys, pos, H, settings(elec_cutoff=0.6), capacity)
        assert not ff1.split and ff1.capacity == ff0.capacity
        (r0, W0, e0), (r1, W1, e1) = _evaluate(ff0, idx, pos, H), _evaluate(ff1, idx, pos, H)
        assert np.array_equal(r0.forces, r1.forces) and np.array_equal(r0.induction.mu, r1.induction.mu)
        assert all(float(r0.energy[k]) == float(r1.energy[k]) for k in r0.energy)
        assert np.array_equal(W0, W1) and float(e0) == float(e1)


def test_split_rows_are_elec_at_elec_cutoff_plus_vdw_at_cutoff():
    """E, forces, dipoles, virial and the Monte Carlo energy of split rows (electrostatics at RC_E,
    LJ at RC_V) against single-cutoff runs: elec(RC_E, no vdW) + [full(RC_V) - elec(RC_V, no vdW)]."""
    sys, pos, H = small_box(1)
    e_only, idx_e = _setup(sys, pos, H, settings(cutoff=RC_E, vdw="none"))
    full, idx_v = _setup(sys, pos, H, settings(cutoff=RC_V, lj_lrc=True))
    full0, _ = _setup(sys, pos, H, settings(cutoff=RC_V, vdw="none"))
    (re_, We, ee), (rv, Wv, ev), (rv0, Wv0, ev0) = (_evaluate(e_only, idx_e, pos, H), _evaluate(full, idx_v, pos, H),
                                                    _evaluate(full0, idx_v, pos, H))
    F_ref = re_.forces + rv.forces - rv0.forces
    W_ref = We + Wv - Wv0
    for capacity in (False, True):
        ff, idx = _setup(sys, pos, H, settings(cutoff=RC_V, elec_cutoff=RC_E, lj_lrc=True), capacity)
        assert ff.split and (ff.mc_e is not None) == capacity
        r, W, e = _evaluate(ff, idx, pos, H)
        assert abs(float(r.energy["elec"] - re_.energy["elec"])) < 1e-10 * abs(float(re_.energy["elec"]))
        assert abs(float(r.energy["vdw"] - rv.energy["vdw"])) < 1e-10 * abs(float(rv.energy["vdw"]))
        assert np.allclose(r.forces, F_ref, rtol=0, atol=1e-10 * float(jnp.abs(F_ref).max()))
        assert np.allclose(r.induction.mu, re_.induction.mu, rtol=0, atol=1e-12 * float(jnp.abs(re_.induction.mu).max()))
        assert np.allclose(W, W_ref, rtol=0, atol=1e-10 * float(jnp.abs(W_ref).max()))
        assert abs(float(e - (ee + ev - ev0))) < 1e-10 * abs(float(ee))


def test_elec_cutoff_longer_than_cutoff():
    """elec_cutoff > cutoff: rows to elec_cutoff, van der Waals weights masked beyond the cutoff."""
    sys, pos, H = small_box(2)
    e_only, idx_e = _setup(sys, pos, H, settings(cutoff=RC_V, vdw="none"))
    full, idx_v = _setup(sys, pos, H, settings(cutoff=RC_E))
    full0, _ = _setup(sys, pos, H, settings(cutoff=RC_E, vdw="none"))
    ff, idx = _setup(sys, pos, H, settings(cutoff=RC_E, elec_cutoff=RC_V))
    assert not ff.split and ff.rc_pair == RC_V
    r, re_, rv, rv0 = (jax.jit(f.compute)(pos, H, i, f.init_induction())
                       for f, i in ((ff, idx), (e_only, idx_e), (full, idx_v), (full0, idx_v)))
    F_ref = re_.forces + rv.forces - rv0.forces
    assert np.allclose(r.forces, F_ref, rtol=0, atol=1e-10 * float(jnp.abs(F_ref).max()))
    assert abs(float(r.energy["vdw"] - rv.energy["vdw"])) < 1e-10 * abs(float(rv.energy["vdw"]))
    assert abs(float(r.energy["elec"] - re_.energy["elec"])) < 1e-10 * abs(float(re_.energy["elec"]))


def test_split_rows_forces_and_virial_match_autodiff_and_finite_differences():
    sys, pos, H = small_box(2)
    ff, idx = _setup(sys, pos, H, settings(cutoff=RC_V, elec_cutoff=RC_E, lj_lrc=True))
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    P = ff._atoms(None)
    F_ad = -jax.grad(lambda x: ff.energy_fixed_mu(x, H, res.induction.mu, idx, P)[0])(jnp.asarray(pos))
    assert np.allclose(res.forces, F_ad, atol=1e-9 * float(jnp.abs(F_ad).max()))
    e = jax.jit(lambda x, h: ff.energy(x, h, idx, ff.init_induction())[0])
    h = 1e-6
    for a, k in [(0, 0), (7, 1), (40, 2), (77, 0)]:
        d = np.zeros_like(pos); d[a, k] = h
        fd = -(float(e(pos + d, H)) - float(e(pos - d, H))) / (2 * h)
        assert abs(fd - float(res.forces[a, k])) < 1e-5 * max(1.0, abs(fd)), (a, k, fd, float(res.forces[a, k]))
    # molecular virial: isotropic strain of the box and the molecular centres
    W = ff.strain_derivative(pos, H, idx, res.induction.mu)
    from pgm_jax.lj import lj_long_range
    m = np.asarray(sys.masses)
    com = np.array([np.average(pos[sys.mol == k], 0, weights=m[sys.mol == k]) for k in range(sys.nmol)])
    fd = (float(e(pos + (h * com)[sys.mol], H * (1 + h))) - float(e(pos - (h * com)[sys.mol], H * (1 - h)))) / (2 * h)
    tail = -3 * float(lj_long_range(P, abs(np.linalg.det(H)), RC_V))
    assert abs(fd + tail - float(jnp.trace(W))) < 1e-5 * max(1.0, abs(fd)), (fd, float(jnp.trace(W)))


def test_split_rows_differentiable_forces_and_dipoles():
    """settings.differentiable with split rows: gradients of forces and dipoles in the parameters
    and positions against central differences with the dipoles re-solved."""
    sys, pos, H = small_box(6)
    ff, idx = _setup(sys, pos, H, settings(cutoff=RC_V, elec_cutoff=RC_E, differentiable=True, adjoint_tol=1e-12))
    rng = np.random.default_rng(3)
    wF, wmu = rng.normal(size=pos.shape), rng.normal(size=pos.shape)

    def loss(theta, x):
        res = ff.compute(x, H, idx, ff.init_induction(), theta)
        return jnp.sum(wF * res.forces) + 1e3 * jnp.sum(wmu * res.induction.mu)

    theta0, x0 = sys.params0, jnp.asarray(pos)
    L = jax.jit(loss)
    g_th, g_x = jax.jit(jax.grad(loss, argnums=(0, 1)))(theta0, x0)
    leaves, tree = jax.tree_util.tree_flatten(theta0)
    v = [rng.normal(size=np.shape(l)) * np.maximum(np.abs(np.asarray(l)), 1e-3) for l in leaves]
    vt = jax.tree_util.tree_unflatten(tree, [jnp.asarray(a) for a in v])
    h = 1e-6
    fd = (float(L(jax.tree_util.tree_map(lambda a, b: a + h * b, theta0, vt), x0))
          - float(L(jax.tree_util.tree_map(lambda a, b: a - h * b, theta0, vt), x0))) / (2 * h)
    ad = sum(float(jnp.sum(a * b)) for a, b in zip(jax.tree_util.tree_leaves(g_th), jax.tree_util.tree_leaves(vt)))
    assert abs(fd - ad) < 1e-6 * max(1.0, abs(fd)), (fd, ad)
    d = rng.normal(size=pos.shape) * 1e-3
    h = 1e-4
    fd = (float(L(theta0, x0 + h * d)) - float(L(theta0, x0 - h * d))) / (2 * h)
    assert abs(fd - float(jnp.sum(g_x * d))) < 1e-6 * max(1.0, abs(fd)), (fd, float(jnp.sum(g_x * d)))


def _graph_topology(sys):
    """Methanol with intramolecular van der Waals (1-4 pairs at 0.5), water without."""
    rules = [MoleculeRule(bonds=m.bonds, vdw="graph", lj_min_sep=4, lj14_scale=0.5) if m.name == "MeOH"
             else MoleculeRule(bonds=[], vdw="none") for m in sys.molecules]
    return MDTopology.build(sys, rules)


def test_special_pairs_exact_in_both_parts_of_split_rows():
    """Special pairs keep exact (offset-difference) float32 displacements in the electrostatic and in
    the van der Waals rows; with elec_cutoff 0.2 nm methanol's weighted 1-4 pairs fall in the latter.
    Far from the origin, differences of float32 coordinates would be off by ~1e-6 nm."""
    sys, pos, H = small_box(3)
    pos = pos + 20.0                                              # molecules stay whole
    top = _graph_topology(sys)
    ff, idx = _setup(sys, pos, H, settings(cutoff=RC_V, elec_cutoff=0.2, precision="mixed"), topology=top)
    k, x, within, wv, over, vrows = jax.jit(ff._rows)(jnp.asarray(pos), jnp.asarray(H), idx)
    assert not bool(over)
    special = {(a, int(b)) for a in range(sys.n) for b in top.special[a] if b < sys.n}
    p32 = pos.astype(np.float32).astype(np.float64)
    for part, (kk, xx, ww) in enumerate(((k, x, within), vrows[:3])):
        kk, ww = np.asarray(kk), np.asarray(ww)
        xx = np.stack([np.asarray(c, np.float64) for c in xx], -1)
        a, c = np.nonzero(ww)
        b = kk[a, c]
        sp = np.array([(i, int(j)) in special for i, j in zip(a, b)])
        assert sp.sum() > 0, part                                  # special pairs in both parts
        exact = pos[a[sp]] - pos[b[sp]]                            # intramolecular: no image shift
        assert np.abs(xx[a[sp], c[sp]] - exact).max() < 1e-7, (part, np.abs(xx[a[sp], c[sp]] - exact).max())
        assert np.abs((p32[a[sp]] - p32[b[sp]]) - exact).max() > 1e-7      # the test can tell
    assert float(jnp.abs(vrows[3]).max()) > 0                      # weighted pairs beyond elec_cutoff


def test_split_rows_with_special_pair_weights_decompose():
    """Graph van der Waals weights of special pairs survive the split (float64)."""
    sys, pos, H = small_box(3)
    top = _graph_topology(sys)
    ff, idx = _setup(sys, pos, H, settings(cutoff=RC_V, elec_cutoff=0.2), topology=top)
    ref, idx_v = _setup(sys, pos, H, settings(cutoff=RC_V), topology=top)
    r, rref = jax.jit(ff.compute)(pos, H, idx, ff.init_induction()), jax.jit(ref.compute)(pos, H, idx_v, ref.init_induction())
    assert abs(float(r.energy["vdw"] - rref.energy["vdw"])) < 1e-12 * abs(float(rref.energy["vdw"]))


def test_row_capacity_overflow_of_each_part():
    sys, pos, H = small_box(4)
    ff, idx = _setup(sys, pos, H, settings(cutoff=RC_V, elec_cutoff=RC_E))
    mc, mc_e = ff.capacity
    ce, cv = (int(c) for c in ff.pair_counts(jnp.asarray(pos), jnp.asarray(H), idx))
    assert ce <= mc_e and cv <= mc - mc_e and int(ff.row_counts(jnp.asarray(pos), jnp.asarray(H), idx)) <= mc
    ok = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    assert not bool(ok.overflow)
    for small in ((mc - mc_e + ce - 1, ce - 1), (mc_e + cv - 1, mc_e)):           # electrostatic part, then vdW part
        ff.mc, ff.mc_e = small
        assert bool(jax.jit(ff.compute)(pos, H, idx, ff.init_induction()).overflow), small
    # the driver finds the overflow, re-sizes both parts and repeats the block
    log = io.StringIO()
    s = settings(cutoff=RC_V, elec_cutoff=RC_E, dipole_tol=1e-8, max_iter=100)
    ref = Simulation(sys, pos, H, s, dt=0.001, ensemble="nve", log=None)
    sim = Simulation(sys, pos, H, s, dt=0.001, ensemble="nve", log=log)
    tail = sim.ff.mc - sim.ff.mc_e
    sim.ff.mc, sim.ff.mc_e = ce - 1 + tail, ce - 1               # electrostatic part too small
    sim.integ.compile()
    sim._advance(20)
    ref._advance(20)
    assert "row capacity overflow" in log.getvalue()
    assert sim.ff.mc_e >= ce + 7 and sim.ff.mc - sim.ff.mc_e >= tail
    assert np.allclose(sim.positions_nm(), ref.positions_nm(), rtol=0, atol=1e-9)


def test_ewald_beta_rule_and_recommended_settings():
    # Amber's dsum_tol convention (erfc(beta rc)/rc with rc in A): ew_coeff of Amber outputs at 1e-5
    for rc, b in ((0.8, 3.4864), (0.9, 3.0768), (1.0, 2.7511)):
        assert abs(ewald_beta_for(rc, 1e-5) - b) < 1e-3, (rc, ewald_beta_for(rc, 1e-5))
    assert abs(ewald_beta_for(0.9) - 4.0) < 1e-3                  # DSUM_TOL: the default 0.9 nm / 4.0 nm^-1
    kw = elec_cutoff_settings(0.7)
    assert kw["elec_cutoff"] == 0.7 and 5.1 < kw["ewald_beta"] < 5.3 and DSUM_TOL < 1e-7
    assert abs(kw["pme_spacing"] - 0.08 * (4.0 / kw["ewald_beta"]) ** 1.6) < 1e-12
    assert abs(elec_cutoff_settings(0.7, exponent=1.0)["pme_spacing"] * kw["ewald_beta"] - 0.32) < 1e-12


def test_short_elec_cutoff_accuracy_bound():
    """elec_cutoff_settings at 0.45 and 0.55 nm with LJ at 0.7 nm against a tight reference (0.7 nm,
    erfc(beta rc) ~ 1e-10, fine grid, order 8), float64; the LJ parts are identical, so the
    differences are electrostatic.  The grid rule keeps the force error nearly independent of the
    cutoff (measured 1.0e-4 and 7e-5 here, 5e-5 at 0.7 nm; beta x spacing fixed: 5e-4 and 3e-4)."""
    sys, pos, H = small_box(5)
    base = dict(cutoff=0.7, dipole_tol=1e-10, pme_order=6, pme_grid=None)
    tight = dict(ewald_beta=6.6, pme_spacing=0.02, pme_order=8)
    ff, idx = _setup(sys, pos, H, settings(**{**base, **tight}), capacity=False)
    ref = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    ff0, _ = _setup(sys, pos, H, settings(**{**base, **tight, "vdw": "none"}), capacity=False)
    rms = float(jnp.sqrt(jnp.mean(jax.jit(ff0.compute)(pos, H, idx, ff0.init_induction()).forces ** 2)))
    for rc in (0.45, 0.55):
        ff, idx = _setup(sys, pos, H, settings(**{**base, **elec_cutoff_settings(rc)}), capacity=False)
        assert ff.split
        r = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
        ferr = float(jnp.sqrt(jnp.mean((r.forces - ref.forces) ** 2))) / rms
        eerr = abs(float(r.energy["elec"] - ref.energy["elec"])) / abs(float(ref.energy["elec"]))
        assert abs(float(r.energy["vdw"] - ref.energy["vdw"])) < 1e-12 * abs(float(ref.energy["vdw"]))
        assert ferr < 2e-4 and eerr < 1e-6, (rc, ferr, eerr)
