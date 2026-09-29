"""Reweighted ensemble averages and their gradients (top-down refinement)."""

import jax
import jax.numpy as jnp
import numpy as np
from test_protein_bonded import ACE_ALA_GLY_NME, peptide_spec

from pgm_jax.bonded import terms as T
from pgm_jax.bonded.model import BondedSettings, BondedTerms
from pgm_jax.ensemble import ALPHA_BOX, KARPLUS, KB, Reweighting, backbone_torsions, in_region, karplus


def test_karplus_and_regions():
    A, B, C, d = KARPLUS["3J_HNHA_Vogeli2007"]
    assert abs(float(karplus(jnp.asarray(-d), A, B, C, d)) - (A + B + C)) < 1e-12
    phi = jnp.radians(jnp.asarray([-65.0, -120.0, 60.0]))
    psi = jnp.radians(jnp.asarray([-40.0, 130.0, 40.0]))
    assert np.array_equal(np.asarray(in_region(phi, psi, *ALPHA_BOX)), [1.0, 0.0, 0.0])


def test_reweighting_gradient_is_the_covariance_formula():
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.normal(size=(200, 4, 3)))

    def g(R):
        return jnp.stack([jnp.sum(jnp.cos(R)), jnp.sum(R[0] * R[1])])

    def f(th, R):
        return th @ g(R)

    th0 = jnp.asarray([0.3, -0.2])
    rw = Reweighting(f, th0, X, temperature=300.0)
    O = jnp.asarray(rng.normal(size=200)) + jnp.sum(X[:, 2], -1)
    assert abs(float(rw.average(th0, O)) - float(O.mean())) < 1e-12
    assert abs(float(rw.n_eff(th0)) - 200.0) < 1e-9
    G = jax.vmap(g)(X)
    beta = 1.0 / (KB * 300.0)
    cov = -beta * (jnp.mean(O[:, None] * G, 0) - O.mean() * G.mean(0))
    grad = jax.grad(lambda th: rw.average(th, O))(th0)
    assert np.allclose(grad, cov, rtol=1e-10, atol=1e-12)
    # away from theta0: finite differences
    th = th0 + jnp.asarray([0.05, 0.02])
    gr = jax.grad(lambda t: rw.average(t, O))(th)
    for k in range(2):
        e = jnp.zeros(2).at[k].set(1e-6)
        fd = (rw.average(th + e, O) - rw.average(th - e, O)) / 2e-6
        assert abs(float(fd) - float(gr[k])) < 1e-6 * max(1.0, abs(float(fd)))
    assert float(rw.n_eff(th)) < 200.0


def test_cmap_refinement_gradient_on_peptide_frames():
    s = peptide_spec(ACE_ALA_GLY_NME)
    terms = BondedTerms([s], BondedSettings(families=T.PROTEIN, lj14_scale=0.5))
    P = terms.init_params()
    rng = np.random.default_rng(2)
    X = s.ref_xyz[None] + 0.01 * rng.normal(size=(80,) + s.ref_xyz.shape)

    def energy(th, R):
        Q = dict(P)
        Q["cmap"] = {"cm": th}
        return terms.bonded_energy(0, R, Q)

    th0 = P["cmap"]["cm"]
    rw = Reweighting(energy, th0, X, temperature=298.0)
    phi, psi = backbone_torsions(X, terms.mols[0].top)
    J = karplus(phi, *KARPLUS["3J_HNHA_Vogeli2007"])
    obs = [(J, np.asarray(J.mean(0)) + 0.3, 0.5)]
    th = th0 + 0.5 * jnp.asarray(rng.normal(size=th0.shape))
    val, grad = rw.chi2_and_grad(th, obs)
    d = jnp.asarray(rng.normal(size=th0.shape))
    fd = (rw.chi2(th + 1e-6 * d, obs) - rw.chi2(th - 1e-6 * d, obs)) / 2e-6
    assert abs(float(fd) - float(jnp.sum(grad * d))) < 1e-6 * max(1.0, abs(float(fd)))
    assert np.isfinite(float(val)) and 1.0 <= float(rw.n_eff(th)) <= 80.0
