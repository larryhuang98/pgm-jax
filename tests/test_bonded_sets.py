"""The three bonded term sets: Amber forms (typing by atom type, gradients) and the fast neural
bonded terms (symmetry, rotation invariance, gradients, freezing for MD)."""

import jax
import jax.numpy as jnp
import numpy as np
from test_bonded import ethanal
from test_grad import methanol

from pgm_jax.bonded import terms as T
from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec

BONDS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)]


def _methanol_spec():
    m, x = methanol()
    return MolSpec("methanol", list(m.elements), BONDS, [1] * 5, 0, x, m), x


def _fd_check(E, x, rng, h=1e-6):
    g = jax.grad(E)(jnp.asarray(x))
    for _ in range(3):
        d = rng.normal(size=x.shape)
        fd = (E(jnp.asarray(x + h * d)) - E(jnp.asarray(x - h * d))) / (2 * h)
        assert abs(float(fd) - float(jnp.sum(g * d))) < 1e-6 * max(1.0, abs(float(fd)))


def test_amber_set_typed_by_atom_type_and_gradients():
    spec, x = _methanol_spec()
    model = BondedModel([spec], BondedSettings(families=T.SETS["amber"], typing="amber", lj14_scale=0.5))
    assert model.keys["bond_harm"] == sorted(set(model.keys["bond_harm"]), key=model.keys["bond_harm"].index)
    assert "bond:c3-h1" in model.keys["bond_harm"] and len(model.keys["bond_harm"]) == 3  # c3-oh, c3-h1, ho-oh
    P = model.init_params()
    P["torsion_amber"]["K"] = P["torsion_amber"]["K"] + 1.0
    rng = np.random.default_rng(0)
    y = x + 0.005 * rng.normal(size=x.shape)
    _fd_check(lambda R: model.energy(0, R, P)[0], y, rng)
    el, bonds, orders, xe = ethanal()
    from test_bonded import ethanal as _e  # noqa: F401

    m2 = BondedModel([MolSpec("ethanal", el, bonds, orders, 0, xe)], BondedSettings(families=("improper_amber",)))
    P2 = m2.init_params()
    _fd_check(lambda R: m2.bonded_energy(0, R, P2), xe + 0.01 * rng.normal(size=xe.shape), rng)


def test_nnb_symmetry_invariance_gradients_and_freeze():
    spec, x = _methanol_spec()
    model = BondedModel([spec], BondedSettings(families=T.SETS["nn"]))
    P = model.init_params()
    rng = np.random.default_rng(1)
    # random network weights (the output layers start at zero)
    P["nnb"] = jax.tree_util.tree_map(lambda v: v + 0.1 * jnp.asarray(rng.normal(size=v.shape)), P["nnb"])
    C = model.nnb.coefficients(P["nnb"], 0)
    top = spec.top
    ch = [k for k, (i, j) in enumerate(top.bonds) if {spec.elements[i], spec.elements[j]} == {"C", "H"}]
    Kb = np.asarray(C["bond_morse"]["Kb"])
    assert len(ch) == 3 and np.allclose(Kb[ch], Kb[ch[0]], rtol=1e-12) and np.ptp(Kb) > 0
    y = jnp.asarray(x + 0.004 * rng.normal(size=x.shape))

    def E(R):
        return model.bonded_energy(0, R, P)

    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    assert abs(float(E(y @ Q.T)) - float(E(y))) < 1e-9 * max(1.0, abs(float(E(y))))
    _fd_check(E, np.asarray(y), rng)
    Pf = dict(P)
    Pf["nnb"] = model.nnb.freeze(P["nnb"])
    assert abs(float(model.bonded_energy(0, y, Pf)) - float(E(y))) < 1e-10 * max(1.0, abs(float(E(y))))
    # torsions related by the molecule's symmetry get the same parameters
    V = np.asarray(C["torsion"]["K"])
    hcoh = [k for k, (i, j, kk, l) in enumerate(top.propers) if spec.elements[i] == "H" and spec.elements[l] == "H"]
    assert len(hcoh) == 3 and np.allclose(V[hcoh], V[hcoh[0]], rtol=1e-12)


def test_nnb_template_md_consistency():
    from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate
    from pgm_jax.md.forcefield import MDSettings
    from pgm_jax.system import System

    spec, x = _methanol_spec()
    model = BondedModel([spec], BondedSettings(families=T.SETS["nn"]))
    P = model.init_params()
    rng = np.random.default_rng(2)
    P["nnb"] = jax.tree_util.tree_map(lambda v: v + 0.05 * jnp.asarray(rng.normal(size=v.shape)), P["nnb"])
    tpl = FlexibleTemplate.from_fit(model, P)
    assert "coef" in tpl.P["nnb"]
    y = x + 0.004 * rng.normal(size=x.shape)
    s = MDSettings().replace(precision="double", dipole_tol=1e-9, cutoff=1.8, skin=0.05, lj_lrc=False)
    sim = FlexibleSimulation(System([tpl.pgm]), [tpl], y + 2.0, np.eye(3) * 4.0, s, thermostat=None, log=None)
    F = np.asarray(sim.state.dyn.force)
    g = np.asarray(jax.grad(lambda R: model.energy(0, R, P)[0])(jnp.asarray(y)))
    assert np.abs(F + g).max() < 1e-3 * np.sqrt(np.mean(g**2))


def test_nnb_typed_table_and_residual_penalty():
    spec, x = _methanol_spec()
    model = BondedModel([spec], BondedSettings(families=T.SETS["nn"], nn_table_depth=0, nn_resid_l2=1.0))
    nn = model.nnb
    P = model.init_params()
    assert set(nn.tvoc) == set(T.PAPER) | {"b0", "th0"}
    # element-typed keys: one bond key per element pair
    pairs = {tuple(sorted((spec.elements[i], spec.elements[j]))) for i, j in spec.top.bonds}
    assert len(nn.tvoc["bond_morse"]) == len(pairs) == len(nn.tvoc["b0"])
    rng = np.random.default_rng(5)
    Q = dict(P["nnb"])
    for k in Q:
        if k.startswith("tab_"):
            Q[k] = jnp.asarray(rng.normal(size=Q[k].shape))
    # network residual zero (output layers start at zero): the typed table alone, no penalty
    C = nn.coefficients(Q, 0)
    el = spec.elements
    Kb = np.asarray(C["bond_morse"]["Kb"])
    key = [tuple(sorted((el[i], el[j]))) for i, j in spec.top.bonds]
    for a in range(len(key)):
        for b in range(len(key)):
            if key[a] == key[b]:
                assert Kb[a] == Kb[b]
    assert float(nn.penalty(Q, [0])) == 0.0
    # random network: a positive penalty with a gradient on the heads only through the residual
    Q = jax.tree_util.tree_map(lambda v: v + 0.1 * jnp.asarray(rng.normal(size=v.shape)), Q)
    pen = float(nn.penalty(Q, [0]))
    g = jax.grad(lambda q: nn.penalty(q, [0]))(Q)
    assert pen > 0 and float(jnp.abs(g["head_bond_morse"]["w2"]).max()) > 0
    assert float(jnp.abs(g["tab_bond_morse"]).max()) == 0.0
