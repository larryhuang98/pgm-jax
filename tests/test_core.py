"""Kernels, solvers and parameter plumbing (no Amber files needed)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pgm_jax.channels import ElecChannel  # noqa: E402
from pgm_jax.kernels import gauss_bij, gauss_coulomb, gauss_overlap, gd6_jax, tt6_jax  # noqa: E402
from pgm_jax.param import molecule_from_dict, molecule_to_dict  # noqa: E402
from pgm_jax.solver import minimize_newton, solve_linear_induction  # noqa: E402
from pgm_jax.system import Molecule, System  # noqa: E402


def _methanol():
    x = np.array(
        [
            [-0.0467, 0.6590, 0.0],
            [-0.0467, -0.7598, 0.0],
            [-1.0830, 0.9930, 0.0],
            [0.4406, 1.0735, 0.8902],
            [0.4406, 1.0735, -0.8902],
            [0.8785, -1.0591, 0.0],
        ]
    )  # Angstrom
    m = Molecule(
        "MeOH",
        ["C", "O", "H", "H", "H", "H"],
        ["c3", "oh", "h1", "h1", "h1", "ho"],
        np.array([0.12, -0.62, 0.02, 0.02, 0.02, 0.44]),
        np.array([0.07, 0.06, 0.05, 0.05, 0.05, 0.05]),
        np.array([1.2e-3, 0.8e-3, 0.4e-3, 0.4e-3, 0.4e-3, 0.3e-3]),
        cov=[(0, 1, 0.01), (1, 0, -0.01), (1, 5, -0.02), (5, 1, 0.005)]
        + [c for h in (2, 3, 4) for c in ((0, h, 0.002), (h, 0, -0.002))],
    )  # symmetric in the 3 methyl H
    m.bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)]
    return m, x


def test_gauss_coulomb_continuous_at_small_r():
    b = gauss_bij(0.06, 0.05)
    r = jnp.array([0.0, 0.5e-4 / b, 0.99e-4 / b, 1.01e-4 / b, 0.1])
    v = np.asarray(gauss_coulomb(r, b))
    assert np.isclose(v[0], 2 * b / np.sqrt(np.pi))
    assert abs(v[2] - v[3]) < 1e-9 * v[2]
    assert np.isclose(v[4], float(jax.scipy.special.erf(b * 0.1) / 0.1))


def test_gauss_overlap_is_normalised():
    """Integral over all separations of the overlap of two unit clouds is 1."""
    b = float(gauss_bij(0.06, 0.05))
    r = np.linspace(0, 12 / b, 20001)
    trap = getattr(np, "trapezoid", None) or np.trapz
    integral = trap(4 * np.pi * r**2 * np.asarray(gauss_overlap(jnp.asarray(r), b)), r)
    assert abs(integral - 1) < 1e-8


def test_damping_functions_limits():
    """Both dampings rise monotonically to 1; tt6 is continuous (to 1e-6) where it switches to its series.
    (gd6 switches to its leading term 8x^6/(9 pi) at x = 0.15, which is ~4 % off there: inherited
    from evoff, not relied on by the electrostatics.)"""
    x = jnp.array([1e-3, 0.1, 0.2, 0.49, 0.51, 1.0, 2.0, 4.0, 30.0])
    g, t = np.asarray(gd6_jax(x)), np.asarray(tt6_jax(x))
    assert np.all(np.diff(g) > 0) and np.all(np.diff(t) > 0)
    assert abs(g[-1] - 1) < 1e-12 and abs(t[-1] - 1) < 1e-6
    assert np.isclose(g[0], 8 * 1e-18 / (9 * np.pi))
    xs = jnp.array([0.5 - 1e-9, 0.5 + 1e-9])
    ts = np.asarray(tt6_jax(xs))
    assert abs(ts[0] - ts[1]) < 1e-6 * ts[1]  # series truncated after x^11: 3.5e-7 relative at x = 0.5


def test_newton_solver_matches_linear_induction_and_its_gradient():
    """minimize_newton on the quadratic induction functional = direct linear solve, and the implicit
    derivative of the minimum w.r.t. the field matches the linear-solve derivative."""
    rng = np.random.default_rng(0)
    n = 4
    A = rng.normal(size=(3 * n, 3 * n)) * 0.05
    T = jnp.asarray((A + A.T).reshape(n, 3, n, 3).transpose(0, 2, 1, 3))
    T = T * (1 - jnp.eye(n))[:, :, None, None]
    alpha = jnp.array([1.0, 0.8, 1.2, 0.9])
    F = jnp.asarray(rng.normal(size=(n, 3)))
    Tm = T.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    G = lambda m, f: jnp.sum(m.reshape(n, 3) ** 2 / (2 * alpha[:, None])) - m @ f.reshape(-1) + 0.5 * m @ Tm @ m
    mu_lin = solve_linear_induction(T, alpha, F).reshape(-1)
    mu_new = minimize_newton(G, jnp.zeros(3 * n), F)
    assert np.allclose(mu_new, mu_lin, atol=1e-10)
    e_lin = lambda f: G(solve_linear_induction(T, alpha, f).reshape(-1), f)
    e_new = lambda f: G(minimize_newton(G, jnp.zeros(3 * n), f), f)
    assert np.allclose(jax.grad(e_new)(F), jax.grad(e_lin)(F), atol=1e-9)


def test_molecule_json_roundtrip():
    m, _ = _methanol()
    m.lj_rmin_half[:] = 0.15
    m.keys = {"alpha": [f"a{k}" for k in range(m.n)]}
    m2 = molecule_from_dict(molecule_to_dict(m))
    for k in ("q", "radius", "alpha", "lj_rmin_half", "lj_sqrt_eps", "masses"):
        assert np.array_equal(getattr(m, k), getattr(m2, k))
    assert m2.cov == m.cov and m2.elements == m.elements and m2.types == m.types
    assert m2.bonds == m.bonds and m2.keys == m.keys and m2.tying_keys() == m.tying_keys()
    old = {
        k: v
        for k, v in molecule_to_dict(m).items()
        if k in ("name", "elements", "types", "q", "radius_nm", "alpha_nm3", "cov")
    }
    m3 = molecule_from_dict(old)  # evoff's format
    assert np.array_equal(m3.q, m.q) and np.all(m3.lj_sqrt_eps == 0) and m3.bonds == []


def test_atom_mapping_permutation_invariance():
    """Shuffle the atom order of a geometry, map parameters onto it by bond graph: same pGM energy."""
    pytest.importorskip("networkx")
    from pgm_jax.param import map_atoms, reorder

    m, x = _methanol()
    order = np.random.default_rng(0).permutation(m.n)
    x2, el2 = x[order], [m.elements[k] for k in order]
    m2 = reorder(m, map_atoms(m.elements, x, el2, x2))
    assert m2.elements == el2
    e1 = ElecChannel().energy(jnp.asarray(x * 0.1), System([m]))[0]
    e2 = ElecChannel().energy(jnp.asarray(x2 * 0.1), System([m2]))[0]
    for k in e1:
        assert abs(float(e1[k]) - float(e2[k])) < 1e-8 * max(1.0, abs(float(e1[k])))
