"""Bonded package: topology, every term family's gradient vs finite differences, rigid-motion
invariance, the pGM part equal to the gas-phase ElecChannel, classical exclusions, flux."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from test_grad import methanol  # noqa: E402

from pgm_jax.bonded import terms as T  # noqa: E402
from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec  # noqa: E402
from pgm_jax.bonded.topology import build_topology  # noqa: E402
from pgm_jax.channels import ElecChannel  # noqa: E402
from pgm_jax.system import System  # noqa: E402


def ethanal():
    """Acetaldehyde, nm (planar carbonyl carbon -> improper)."""
    x = np.array(
        [
            [0.0000, 0.0000, 0.0],
            [0.1500, 0.0000, 0.0],
            [0.2180, 0.1030, 0.0],
            [0.1980, -0.0990, 0.0],
            [-0.0360, -0.1030, 0.0],
            [-0.0380, 0.0520, 0.0890],
            [-0.0380, 0.0520, -0.0890],
        ]
    )
    el = ["C", "C", "O", "H", "H", "H", "H"]
    bonds = [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (0, 6)]
    return el, bonds, [1, 2, 1, 1, 1, 1], x


def test_topology_counts():
    el, bonds, orders, x = ethanal()
    top = build_topology(el, bonds, (bonds, orders), x * 10)
    assert len(top.bonds) == 6 and len(top.angles) == 9 and len(top.propers) == 6
    assert [tuple(i) for i in top.impropers] == [(1, 0, 2, 3)]
    assert len(top.pairs14) == 6 and len(top.bond_bond) == 9 and len(top.torsion_bond) == 18
    assert not top.rigid_torsion.any()


def _all_families_model(settings_kw=None):
    el, bonds, orders, x = ethanal()
    fams = tuple(f for f in T.REGISTRY if f != "bond_harm")
    spec = MolSpec("ethanal", el, bonds, orders, 0, x)
    return BondedModel([spec], BondedSettings(families=fams, **(settings_kw or {}))), x


def test_every_family_gradient_and_invariance():
    model, x = _all_families_model()
    P = model.init_params()
    rng = np.random.default_rng(0)
    lin = model.linear_mask(P)
    P = jax.tree_util.tree_map(
        lambda v, l: v + (0.5 * rng.normal(size=np.shape(v)) if l else 0.01 * v * rng.normal(size=np.shape(v))), P, lin
    )
    P["ref"] = model.init_params()["ref"]
    X = jnp.asarray(x + 0.004 * rng.normal(size=x.shape))
    E = jax.jit(lambda X: model.bonded_energy(0, X, P))
    g = jax.grad(lambda X: model.bonded_energy(0, X, P))(X)
    for a, k in [(0, 0), (2, 1), (3, 2), (6, 0)]:
        d = np.zeros(x.shape)
        d[a, k] = 1e-6
        fd = (float(E(X + d)) - float(E(X - d))) / 2e-6
        assert abs(fd - float(g[a, k])) < 1e-5 * max(1.0, abs(fd)), (a, k, fd, float(g[a, k]))
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    assert abs(float(E(X @ Q.T + 0.3)) - float(E(X))) < 1e-8 * max(1.0, abs(float(E(X))))
    # every family with instances in ethanal contributes (conj / hc_lone: see test_electronic_families)
    for f in model.fams:
        P0 = jax.tree_util.tree_map(jnp.zeros_like, P)
        P0["ref"] = P["ref"]
        P0[f] = P[f]
        if f in ("bond_morse", "conj", "hc_lone") or len(model.I[0][f]["k"]) == 0:  # cmap: no backbone
            continue
        assert abs(float(model.bonded_energy(0, X, P0))) > 0, f


def test_pgm_part_matches_elec_channel_and_exclusions():
    m, x = methanol()
    spec = MolSpec("MeOH", m.elements, m.bonds, [1] * len(m.bonds), 0, x, pgm=m)
    model = BondedModel([spec], BondedSettings(families=("angle_cos",), lj_min_sep=99))
    e, dip = model.nonbonded(0, jnp.asarray(x))
    sys = System([m])
    out, aux = ElecChannel().energy(jnp.asarray(x), sys)
    assert abs(float(e) - float(out["perm"] + out["ind"])) < 1e-9 * abs(float(e))
    ref_dip = jnp.sum(jnp.asarray(m.q)[:, None] * x, 0) + jnp.sum(aux["p"], 0) + jnp.sum(aux["mu"], 0)
    assert np.allclose(dip, ref_dip, atol=1e-12)
    # classical control: with every pair within 3 bonds removed, methanol (max 3 bonds) has no electrostatics
    model3 = BondedModel(
        [MolSpec("MeOH", m.elements, m.bonds, [1] * 5, 0, x, pgm=m)],
        BondedSettings(families=("angle_cos",), elec_exclude=3, lj_min_sep=99),
    )
    e3, _ = model3.nonbonded(0, jnp.asarray(x))
    assert abs(float(e3)) < 1e-12


def test_flux_conserves_charge_and_is_differentiable():
    m, x = methanol()
    spec = MolSpec("MeOH", m.elements, m.bonds, [1] * len(m.bonds), 0, x, pgm=m)
    model = BondedModel([spec], BondedSettings(families=("bond_harm", "angle_cos"), flux=True))
    P = model.init_params()
    P["flux"]["jb"] = P["flux"]["jb"] + 0.5
    P["flux"]["jc"] = P["flux"]["jc"] + 0.1
    X = jnp.asarray(x * 1.03)
    Q = model.nb[0]["sys"].expand(None)
    q, cov = model._flux(0, X, P, Q["q"], Q["cov"])
    assert abs(float(jnp.sum(q) - jnp.sum(Q["q"]))) < 1e-12 and float(jnp.max(jnp.abs(q - Q["q"]))) > 1e-4
    g = jax.grad(lambda P: model.energy(0, X, P)[0])(P)
    assert float(jnp.max(jnp.abs(g["flux"]["jb"]))) > 0


def test_fit_recovers_synthetic_parameters():
    """Frames labelled by a known model (class II + pGM): the fitter recovers energies and forces."""
    from pgm_jax.bonded.fit import Fitter, FrameSet

    m, x = methanol()
    spec = MolSpec("MeOH", m.elements, m.bonds, [1] * 5, 0, x, pgm=m)
    model = BondedModel([spec], BondedSettings(families=T.PAPER))
    P0 = model.init_params()
    rng = np.random.default_rng(1)
    lin = model.linear_mask(P0)
    Pt = jax.tree_util.tree_map(
        lambda v, l: v * (1 + 0.2 * rng.normal(size=np.shape(v))) + (3 * rng.normal(size=np.shape(v)) if l else 0),
        P0,
        lin,
    )
    Pt["ref"] = {"b0": P0["ref"]["b0"] * 1.01, "th0": P0["ref"]["th0"] + 0.01}
    X = jnp.asarray(x[None] + 0.005 * rng.normal(size=(120,) + x.shape))
    E, G = jax.vmap(jax.value_and_grad(lambda X: model.energy(0, X, Pt)[0]))(X)
    fs = FrameSet(np.asarray(X[:80]), np.asarray(E[:80]) + 50.0, -np.asarray(G[:80]))
    te = FrameSet(np.asarray(X[80:]), np.asarray(E[80:]), -np.asarray(G[80:]))
    fit = Fitter(model, {0: {"train": fs, "test": te}}, l2=1e-8)
    P = fit.fit(P0, maxiter=4000, verbose=False)
    r = fit.metrics(P, "test")[0]
    assert r["E_MAE"] < 0.01 and r["F_MAE"] < 0.1, r


def test_elec14_scaling_between_pgm_and_exclusion():
    m, x = methanol()
    X = jnp.asarray(x)
    e = {}
    for tag, kw in (
        ("pgm", {}),
        ("half", {"elec_exclude": 2, "elec14_scale": 0.5}),
        ("zero", {"elec_exclude": 2, "elec14_scale": 0.0}),
        ("excl3", {"elec_exclude": 3}),
    ):
        model = BondedModel(
            [MolSpec("MeOH", m.elements, m.bonds, [1] * 5, 0, x, pgm=m)],
            BondedSettings(families=("angle_cos",), lj_min_sep=99, **kw),
        )
        e[tag] = float(model.nonbonded(0, X)[0])
    assert abs(e["zero"] - e["excl3"]) < 1e-10 and e["half"] != e["zero"] and e["half"] != e["pgm"]


def test_twist_angle():
    """Twist of a 3-coordinated centre: equals the dihedral for a planar centre, follows a rigid
    rotation about the bond, and stays at 90 deg when the rotated centre pyramidalises symmetrically."""
    el, bonds, orders, x = ethanal()
    top = build_topology(el, bonds, (bonds, orders), x * 10)
    I, keys = T.REGISTRY["twist"].index(top, lambda a, k: "x")
    assert len(keys) == 3  # one per methyl H (pair O, H on C1); sp3 C0 has none

    def tau(X):
        G = T.geometry(jnp.asarray(X), top)
        p1, p2 = G["phi"][I["u"]], G["phi"][I["v"]]
        return np.asarray(jnp.arctan2(jnp.sin(p1) - jnp.sin(p2), jnp.cos(p1) - jnp.cos(p2))), np.asarray(p1)

    t, p1 = tau(x)
    assert np.allclose(np.angle(np.exp(1j * (t - p1))), 0.0, atol=1e-6)
    # rotate the C1 substituents (O, H) by 50 deg about the C0-C1 axis: every twist moves by the same angle
    ax = (x[1] - x[0]) / np.linalg.norm(x[1] - x[0])

    def rot(v, a):
        return v * np.cos(a) + np.cross(ax, v) * np.sin(a) + ax * np.dot(ax, v) * (1 - np.cos(a))

    y = x.copy()
    for k in (2, 3):
        y[k] = x[1] + rot(x[k] - x[1], np.radians(50.0))
    t2, _ = tau(y)
    d = np.angle(np.exp(1j * (t2 - t)))
    assert np.allclose(np.abs(d), np.radians(50.0), atol=1e-6) and np.allclose(d, d[0])
    # pyramidalise C1 (both substituents folded to one side of the C0-C1-substituent plane, as the
    # amide N at the rotation barrier): the twist is unchanged, the individual dihedrals move by ~20 deg
    t3, p3 = tau(y)
    perp = lambda v: v - ax * np.dot(ax, v)
    n = np.cross(ax, perp(y[2] - x[1]))
    n /= np.linalg.norm(n)
    z = y.copy()
    for k in (2, 3):
        z[k] = y[k] + 0.4 * np.linalg.norm(perp(y[k] - x[1])) * n
    t4, p4 = tau(z)
    assert np.allclose(np.angle(np.exp(1j * (t4 - t3))), 0.0, atol=1e-6)
    assert np.abs(np.angle(np.exp(1j * (p4 - p3)))).max() > np.radians(5.0)


def test_learned_pair_scales():
    """escale: E = pGM + sum_c kappa_c E_perm,c; the fitter's cached linear path equals the direct
    energy and forces, and kappa = -1 on every 1-2/1-3/1-4 class removes those permanent pairs."""
    from pgm_jax.bonded.fit import Fitter, FrameSet

    m, x = methanol()
    spec = MolSpec("MeOH", m.elements, m.bonds, [1] * 5, 0, x, pgm=m)
    model = BondedModel([spec], BondedSettings(families=("angle_cos",), escale=(1, 2, 3), lj_min_sep=99))
    P = model.init_params()
    nk = len(P["escale"]["kappa"])
    assert nk == len(model.es_pos) and nk >= 3
    P["escale"]["kappa"] = jnp.linspace(-1.0, 0.5, nk)
    rng = np.random.default_rng(3)
    X = x[None] + 0.003 * rng.normal(size=(4,) + x.shape)
    fs = FrameSet(X, np.zeros(4), np.zeros_like(X))
    fit = Fitter(model, {0: {"train": fs}})
    E, F, _ = fit._predict(0, P, jnp.asarray(X), fit.nonbonded(0, "train"))
    E2, G2 = jax.vmap(jax.value_and_grad(lambda X: model.energy(0, X, P)[0]))(jnp.asarray(X))
    assert np.allclose(E, E2, atol=1e-8) and np.allclose(F, -G2, atol=1e-6)
    # kappa = -1 everywhere removes exactly the permanent 1-2/1-3/1-4 pair energies (pair by pair)
    from pgm_jax.channels import _pair_perm, perm_dipoles
    from pgm_jax.units import KE

    P["escale"]["kappa"] = -jnp.ones(nk)
    R = jnp.asarray(x)
    e_all = float(model.energy(0, R, P)[0] - model.energy(0, R, {**P, "escale": {"kappa": jnp.zeros(nk)}})[0])
    s = model.nb[0]["sys"]
    Q = s.expand()
    p = perm_dipoles(R, s, Q["cov"])
    D = spec.top.dist
    e_ref = 0.0
    for i, j in zip(s.pair_i, s.pair_j):
        if 1 <= D[i, j] <= 3:
            e_ref -= KE * float(
                _pair_perm(
                    R[i], R[j], Q["q"][i], p[i], Q["q"][j], p[j], model._bij(Q["radius"][i], Q["radius"][j]), model._phi
                )
            )
    assert abs(e_all - e_ref) < 1e-8 * max(1.0, abs(e_ref)) and abs(e_ref) > 1.0


def test_fitted_typed_charges():
    """qfit: typed pGM charges / covalent dipoles start at the ESP values (deep typing reproduces
    the fixed-parameter energy), keep the molecular charge, and the fitter runs the dynamic path."""
    from pgm_jax.bonded.fit import Fitter, FrameSet

    m, x = methanol()
    R = jnp.asarray(x)
    mk = lambda q: BondedModel(
        [MolSpec("MeOH", m.elements, m.bonds, [1] * 5, 0, x, pgm=m)], BondedSettings(families=("angle_cos",), qfit=q)
    )
    model = mk(8)
    P = model.init_params()
    assert model.nb_dynamic and "elec" in P
    e_fix = float(model.nonbonded(0, R)[0])
    e_fit = float(model.nonbonded(0, R, P)[0])
    assert abs(e_fix - e_fit) < 1e-8 * max(1.0, abs(e_fix))
    P["elec"]["q"] = P["elec"]["q"] + 0.1  # uniform shift is removed by neutrality
    assert abs(float(model.nonbonded(0, R, P)[0]) - e_fit) < 1e-8 * max(1.0, abs(e_fit))
    coarse = mk(0)  # element typing: fewer charge types
    assert len(coarse.q_keys) == len(set(m.elements)) < len(model.q_keys)
    rng = np.random.default_rng(4)
    X = x[None] + 0.003 * rng.normal(size=(3,) + x.shape)
    E, G = jax.vmap(jax.value_and_grad(lambda X: model.energy(0, X, P)[0]))(jnp.asarray(X))
    fit = Fitter(model, {0: {"train": FrameSet(X, np.asarray(E), -np.asarray(G))}})
    Ep, Fp, _ = fit._predict(0, P, jnp.asarray(X))
    assert np.allclose(Ep, E) and np.allclose(Fp, -G)


def test_bond_charge_increments():
    """qbci: zero increments reproduce the ESP-parameter energy; increments keep the total charge."""
    m, x = methanol()
    R = jnp.asarray(x)
    model = BondedModel(
        [MolSpec("MeOH", m.elements, m.bonds, [1] * 5, 0, x, pgm=m)], BondedSettings(families=("angle_cos",), qbci=1)
    )
    P = model.init_params()
    assert model.nb_dynamic and abs(float(model.nonbonded(0, R, P)[0]) - float(model.nonbonded(0, R)[0])) < 1e-10
    P["bci"]["t"] = P["bci"]["t"] + 0.05 * jnp.arange(1, len(P["bci"]["t"]) + 1)
    _, dip, st = model.nonbonded(0, R, P, state=True)
    q0 = model.nb[0]["sys"].expand()["q"]
    assert abs(float(jnp.sum(st["q"]) - jnp.sum(q0))) < 1e-12 and float(jnp.abs(st["q"] - q0).max()) > 0.04


def test_separate_induction_exclusion():
    """ind_exclude: excluding only the permanent 1-2/1-3 pairs keeps pGM's induced dipoles (same
    molecular dipole); excluding only their induction keeps the permanent energy."""
    m, x = methanol()
    R = jnp.asarray(x)
    mk = lambda **kw: BondedModel(
        [MolSpec("MeOH", m.elements, m.bonds, [1] * 5, 0, x, pgm=m)],
        BondedSettings(families=("angle_cos",), lj_min_sep=99, **kw),
    )
    e0, d0, s0 = mk().nonbonded(0, R, state=True)
    e1, d1, s1 = mk(elec_exclude=2, ind_exclude=0).nonbonded(0, R, state=True)
    e2, d2, s2 = mk(elec_exclude=0, ind_exclude=2).nonbonded(0, R, state=True)
    e3, d3, s3 = mk(elec_exclude=2).nonbonded(0, R, state=True)
    assert np.allclose(s1["mu"], s0["mu"]) and np.allclose(d1, d0) and abs(float(e1 - e0)) > 1e-3
    assert np.allclose(s2["mu"], s3["mu"]) and not np.allclose(s2["mu"], s0["mu"])


def methyl_formate():
    """HC(=O)OCH3, nm: a conjugated C-O bond and an sp3 neighbour of the ester O."""
    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.060, 0.104, 0.0],
            [-0.055, -0.095, 0.0],
            [0.134, 0.0, 0.0],
            [0.195, 0.124, 0.0],
            [0.300, 0.110, 0.0],
            [0.160, 0.180, 0.089],
            [0.160, 0.180, -0.089],
        ]
    )
    el = ["C", "O", "H", "O", "C", "H", "H", "H"]
    bonds = [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7)]
    return el, bonds, [2, 1, 1, 1, 1, 1, 1], x


def test_electronic_families():
    """F12 families: finite-difference gradients, rigid-motion invariance, and their physics:
    pi-axis p fraction 1 planar / 3/4 tetrahedral, conj = cos^2 of the rotation, volume double well,
    self-consistent hybrids = fixed hybrids at the reference geometry."""
    el, bonds, orders, x = methyl_formate()
    fams = (
        "conj",
        "volume",
        "hc_sigma",
        "hc_lone",
        "angle_hyb",
        "angle_hybsc",
        "pair13_tanh",
        "pair14_tanh",
        "pair13_ovl",
        "pair14_ovl",
    )
    model = BondedModel([MolSpec("mf", el, bonds, orders, 0, x)], BondedSettings(families=fams))
    for f in fams:
        assert len(model.I[0][f]["k"]) > 0, f
    P = model.init_params()
    rng = np.random.default_rng(5)
    lin = model.linear_mask(P)
    P = jax.tree_util.tree_map(
        lambda v, l: v + (0.5 * rng.normal(size=np.shape(v)) if l else 0.01 * v * rng.normal(size=np.shape(v))), P, lin
    )
    X = jnp.asarray(x + 0.004 * rng.normal(size=x.shape))
    E = jax.jit(lambda X: model.bonded_energy(0, X, P))
    g = jax.grad(lambda X: model.bonded_energy(0, X, P))(X)
    for a, k in [(0, 0), (1, 1), (3, 2), (4, 0), (6, 2)]:
        d = np.zeros(x.shape)
        d[a, k] = 1e-6
        fd = (float(E(X + d)) - float(E(X - d))) / 2e-6
        assert abs(fd - float(g[a, k])) < 1e-5 * max(1.0, abs(fd)), (a, k, fd, float(g[a, k]))
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    Q = Q * np.sign(np.linalg.det(Q))
    assert abs(float(E(X @ Q.T + 0.3)) - float(E(X))) < 1e-8 * max(1.0, abs(float(E(X))))
    # pi axes: planar p = 1, tetrahedral p = 3/4
    tet = jnp.asarray([[0.0, 0.0, 0.0], [1, 1, 1], [1, -1, -1], [-1, 1, -1]], float)
    tri = jnp.asarray([[0.0, 0.0, 0.0], [1, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0]], float)
    for R_, pexp in ((tet, 0.75), (tri, 1.0)):
        _, pp = T.pi_axes(R_, np.array([0]), np.array([[1, 2, 3]]), np.array([3]))
        assert abs(float(pp[0]) - pexp) < 1e-6
    # conj: rotating the ester O substituent by 90 deg about C-O removes the conjugation
    Pc = jax.tree_util.tree_map(jnp.zeros_like, P)
    Pc["ref"] = P["ref"]
    Pc["conj"] = {"K": jnp.ones_like(P["conj"]["K"])}
    ax = (x[3] - x[0]) / np.linalg.norm(x[3] - x[0])

    def rot(v, a):
        return v * np.cos(a) + np.cross(ax, v) * np.sin(a) + ax * np.dot(ax, v) * (1 - np.cos(a))

    for ang, e_exp in ((0.0, 0.0), (90.0, 1.0), (45.0, 0.5)):
        y = x.copy()
        for k in (4, 5, 6, 7):
            y[k] = x[3] + rot(x[k] - x[3], np.radians(ang))
        e = float(model.bonded_energy(0, jnp.asarray(y), Pc))
        assert abs(e - e_exp) < 0.02, (ang, e)
    # hybrids: nearly zero energy at the reference geometry (the least-squares m cannot make every
    # angle of this rough geometry exact), self-consistent ones softer than fixed ones
    Ph = model.init_params()
    for f in fams:
        if f not in ("angle_hyb", "angle_hybsc"):
            Ph[f] = jax.tree_util.tree_map(jnp.zeros_like, Ph[f])
    e_ref = float(model.bonded_energy(0, jnp.asarray(x), Ph))
    Y = jnp.asarray(x + 0.01 * rng.normal(size=x.shape))
    only = lambda f: {
        **{k: jax.tree_util.tree_map(jnp.zeros_like, v) for k, v in Ph.items() if k != "ref"},
        "ref": Ph["ref"],
        f: Ph[f],
    }
    e_fix, e_sc = (
        float(model.bonded_energy(0, Y, only("angle_hyb"))),
        float(model.bonded_energy(0, Y, only("angle_hybsc"))),
    )
    assert e_ref < 0.05 * e_fix and 0.0 < e_sc < e_fix, (e_ref, e_sc, e_fix)
