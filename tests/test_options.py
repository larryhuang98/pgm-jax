"""Model options: electrostatics levels, Gaussian quadrupoles (analytic kernels vs automatic
derivatives of the operator form, covalent quadrupole basis), GVDW kernel, old pickles."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import erf
from test_grad import cluster, methanol, water

from pgm_jax import System
from pgm_jax.channels import ElecChannel
from pgm_jax.multipole import (
    S_tensor,
    multipole_field,
    multipole_pair_energy,
    quadrupoles,
    with_quadrupoles,
)
from pgm_jax.vdw import C0, GVDWChannel, gvdw_G, gvdw_pair, set_gvdw


def _rand_quad(rng, n):
    A = rng.normal(size=(n, 3, 3))
    A = 0.5 * (A + np.swapaxes(A, 1, 2))
    return A - np.trace(A, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3


def _operator_energy(x, a, qi, pi, Ti, qj, pj, Tj):
    """E = O_i O_j phi by nested automatic derivatives (the definition)."""

    def f(v):
        return erf(a * jnp.linalg.norm(v)) / jnp.linalg.norm(v)

    g, H = jax.grad(f), jax.hessian(f)
    T3 = jax.jacfwd(H)
    T4 = jax.jacfwd(T3)
    e = qi * qj * f(x) + (qj * pi - qi * pj) @ g(x) - pi @ H(x) @ pj
    e += (jnp.sum((qi * Tj + qj * Ti) * H(x))) / 3
    e += (jnp.einsum("a,bc,abc->", pi, Tj, T3(x)) - jnp.einsum("a,bc,abc->", pj, Ti, T3(x))) / 3
    e += jnp.einsum("ab,cd,abcd->", Ti, Tj, T4(x)) / 9
    return e


def test_multipole_kernels_match_operator_form():
    rng = np.random.default_rng(0)
    for scale in (0.05, 0.3):  # a r < 1 (series) and > 1 (closed form)
        x = rng.normal(size=3) * scale
        a = 7.0
        qi, qj = rng.normal(size=2)
        pi, pj = rng.normal(size=(2, 3)) * 0.02
        Ti, Tj = _rand_quad(rng, 2) * 0.003
        ref = _operator_energy(
            jnp.asarray(x), a, qi, jnp.asarray(pi), jnp.asarray(Ti), qj, jnp.asarray(pj), jnp.asarray(Tj)
        )
        got = multipole_pair_energy(
            jnp.asarray(x)[None],
            jnp.asarray([a]),
            jnp.asarray([qi]),
            jnp.asarray(pi)[None],
            jnp.asarray(Ti)[None],
            jnp.asarray([qj]),
            jnp.asarray(pj)[None],
            jnp.asarray(Tj)[None],
        )[0]
        assert abs(float(got) - float(ref)) < 1e-9 * max(1.0, abs(float(ref))), (scale, got, ref)

        # field at i = -grad_x of the potential of j
        def V(v):
            return _operator_energy(v, a, 1.0, jnp.zeros(3), jnp.zeros((3, 3)), qj, jnp.asarray(pj), jnp.asarray(Tj))

        E_ref = -jax.grad(V)(jnp.asarray(x))
        E = multipole_field(
            jnp.asarray(x)[None], jnp.asarray([a]), jnp.asarray([qj]), jnp.asarray(pj)[None], jnp.asarray(Tj)[None]
        )[0]
        assert np.allclose(E, E_ref, rtol=1e-9, atol=1e-9 * float(jnp.max(jnp.abs(E_ref))))


def test_quadrupole_basis_is_traceless_and_rotates():
    m, x = methanol()
    mq = with_quadrupoles(m)
    assert len(mq.quad) > 0 and all(len(t) == 4 for t in mq.quad)
    # terminal H atoms get 1-3 partners
    assert any(i == 5 and j != k for i, j, k, _ in mq.quad)
    rng = np.random.default_rng(1)
    t = jnp.asarray(rng.normal(size=len(mq.quad)) * 1e-3)
    sys = System([mq])
    Th = quadrupoles(jnp.asarray(x), sys, t)
    assert np.allclose(np.trace(np.asarray(Th), axis1=1, axis2=2), 0.0, atol=1e-15)
    assert np.allclose(Th, np.swapaxes(Th, 1, 2))
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    Th2 = quadrupoles(jnp.asarray(x @ Q.T), sys, t)
    assert np.allclose(Th2, np.einsum("ab,nbc,dc->nad", Q, Th, Q), atol=1e-14)
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    S = np.asarray(S_tensor(jnp.asarray(u), jnp.asarray(u)))
    assert np.allclose(S @ u, u) and abs(np.trace(S)) < 1e-15  # Theta_zz = 1 along u


def test_electrostatics_levels():
    sys, pos = cluster(np.random.default_rng(3))
    pos = jnp.asarray(pos)
    P = sys.expand()
    full = ElecChannel().energy(pos, sys)[0]
    assert float(full["perm"] + full["ind"]) == pytest.approx(
        float(sum(ElecChannel.level("qpi").energy(pos, sys)[0].values())), rel=1e-12
    )
    qp = ElecChannel.level("qp").energy(pos, sys)[0]
    assert "ind" not in qp and float(qp["perm"]) == pytest.approx(float(full["perm"]), rel=1e-12)
    q_only = ElecChannel.level("q").energy(pos, sys)[0]["perm"]
    from pgm_jax.densities import gauss_bij, gauss_coulomb
    from pgm_jax.units import KE

    ii, jj = sys.pair_i, sys.pair_j
    r = jnp.linalg.norm(pos[ii] - pos[jj], axis=-1)
    ref = KE * jnp.sum(P["q"][ii] * P["q"][jj] * gauss_coulomb(r, gauss_bij(P["radius"][ii], P["radius"][jj])))
    assert float(q_only) == pytest.approx(float(ref), rel=1e-12)
    qi = ElecChannel.level("qi").energy(pos, sys)[0]
    assert float(qi["ind"]) < 0.0


def test_quadrupoles_zero_strength_and_forces():
    m, x = methanol()
    mq = with_quadrupoles(m)
    sys = System([mq, water()])
    rng = np.random.default_rng(4)
    w = np.array([[0, 0, 0], [0.0957, 0, 0], [-0.024, 0.0927, 0]]) + [0.3, 0.1, 0.05]
    pos = jnp.asarray(np.concatenate([x, w]))
    ch = ElecChannel(quadrupoles=True)
    e0 = ch.energy(pos, sys)[0]
    e_ref = ElecChannel().energy(pos, sys)[0]
    assert abs(float(e0["perm"] + e0["ind"] - e_ref["perm"] - e_ref["ind"])) < 1e-9
    P = dict(sys.params0)
    P["quad"] = jnp.asarray(rng.normal(size=P["quad"].shape) * 2e-3)

    def E(y):
        return sum(ch.energy(y, sys, P)[0].values())

    F = -jax.grad(E)(pos)
    d = np.zeros(pos.shape)
    d[2, 1] = 1e-6
    fd = -(E(pos + d) - E(pos - d)) / 2e-6
    assert abs(float(F[2, 1]) - float(fd)) < 1e-5 * max(1.0, abs(float(fd)))
    assert abs(float(E(pos)) - float(e0["perm"] + e0["ind"])) > 1e-3  # the quadrupoles do something
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    assert float(E(pos @ Q.T)) == pytest.approx(float(E(pos)), rel=1e-10)


def test_gvdw_kernel():
    y = jnp.linspace(1e-3, 6.0, 4001)
    G, H = gvdw_G(y)
    # continuity across the series / closed-form switch and the y -> 0 limit
    lo, hi = gvdw_G(jnp.asarray([0.6 - 1e-9])), gvdw_G(jnp.asarray([0.6 + 1e-9]))  # 2e-9 apart
    assert abs(float(lo[0][0] - hi[0][0])) < 3e-9 * abs(float(hi[1][0])) * 0.6 + 1e-13
    assert abs(float(lo[1][0] - hi[1][0])) < 1e-7 * abs(float(hi[1][0]))
    assert float(gvdw_G(jnp.asarray([1e-8]))[0][0]) == pytest.approx(C0, rel=1e-12)
    # H = G'/y
    dG = jax.vmap(jax.grad(lambda t: gvdw_G(t[None])[0][0]))(y)
    assert np.allclose(H * y, dG, rtol=1e-8, atol=1e-12)
    # pmemd-pgm's expressions (pairs_calc_PGM.i), numpy, y > 0.01
    yy = np.asarray(y)
    e = np.exp(-(yy**2))
    from scipy.special import erf as serf

    Bx = serf(yy) - 2 / np.sqrt(np.pi) * yy * e
    gx = 4 / 3 / np.sqrt(np.pi) * yy**3 * e
    Fx = (Bx - gx) ** 2 + 0.5 * gx**2
    assert np.allclose(np.asarray(G) * yy**6, Fx, rtol=1e-9, atol=1e-15)
    # pair energy and radial derivative
    r = jnp.linspace(0.05, 1.2, 200)
    for rep in ("gauss", "slater"):
        e, dr = gvdw_pair(r, 5.0, 3000.0, 2e-3, 1.3, rep, grad=True)
        de = jax.vmap(jax.grad(lambda t: gvdw_pair(t, 5.0, 3000.0, 2e-3, 1.3, rep)))(r)
        assert np.allclose(dr * r, de, rtol=1e-9, atol=1e-9)


def test_gvdw_channel_and_old_pickles():
    w = set_gvdw(water(), {"OW": (100.0, 0.05, 4.5)})
    sys = System([w, w])
    pos = jnp.asarray(
        np.array(
            [[0, 0, 0], [0.0957, 0, 0], [-0.024, 0.0927, 0], [0.29, 0.02, 0.01], [0.38, 0.03, 0.0], [0.27, 0.11, 0.0]]
        )
    )
    e = GVDWChannel(rep="slater").energy(pos, sys)[0]["vdw"]
    P = sys.expand()
    r = float(jnp.linalg.norm(pos[0] - pos[3]))
    beta = 1 / np.sqrt(2 * 2 * float(P["radius"][0]) ** 2)
    ref = gvdw_pair(r, beta, 100.0**2, 0.05**2, 4.5, "slater")
    assert float(e) == pytest.approx(float(ref), rel=1e-12)  # only the O-O pair
    # a Molecule pickled before GVDW / quadrupoles existed
    m, _ = methanol()
    st = dict(m.__dict__)
    for k in ("gvdw_sqrt_a", "gvdw_sqrt_c6", "gvdw_b", "quad"):
        st.pop(k)
    m2 = object.__new__(type(m))
    m2.__setstate__(st)
    m3 = pickle.loads(pickle.dumps(m2))
    assert np.all(m3.gvdw_b == 1.0) and m3.quad == [] and System([m3]).table.sizes()["quad"] == 0
