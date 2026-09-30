"""Double-exponential (DE) van der Waals of DEGAUSS: pair function, tail, MD force field, periodic model.

What is checked, and against what: the closed-form limits of the pair energy (finite value at r = 0,
minimum -eps at r = rm, decay to zero); the analytic (1/r) dU/dr against central differences (1e-7
relative); the continuum tail and its cutoff impulse against numerical quadrature of the pair
function (1e-6 relative); the row forces of the MD force field against autodiff at fixed dipoles
(1e-9) and against central differences of the energy (1e-5 relative); the periodic model
(`PeriodicModel(vdw="de")`) against the MD force field (van der Waals energy, 1e-9 relative); the
molecular strain derivative against isotropic scaling, including the tail impulse (1e-5 relative);
and that the Lennard-Jones and DE forms differ (the DE form is really selected).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import small_box
from test_md import ff_and_list

from pgm_jax import PeriodicModel
from pgm_jax.de import (
    DE_ALPHA,
    DE_BETA,
    DEChannel,
    de_groups,
    de_long_range,
    de_pair,
    de_pair_grad,
    de_tail_impulse,
)
from pgm_jax.options import check_vdw

RM, EPS = 0.35, 0.8


def test_pair_limits():
    """u(0) is finite, the minimum is -eps at r = rm, u tends to zero, and the value matches Eq. 5 of the paper."""
    a, b = DE_ALPHA, DE_BETA
    u0 = EPS / (a - b) * (b * np.exp(a) - a * np.exp(b))
    assert abs(float(de_pair(0.0, RM, EPS)) - u0) < 1e-9 * abs(u0)
    assert np.isfinite(u0) and u0 > 0
    assert abs(float(de_pair(RM, RM, EPS)) + EPS) < 1e-12
    r = np.linspace(0.05, 1.5, 4001)
    e = np.asarray(de_pair(r, RM, EPS))
    assert abs(e.min() + EPS) < 1e-4 and abs(r[e.argmin()] - RM) < 1e-3
    assert abs(float(de_pair(5.0, RM, EPS))) < 1e-4 * EPS
    assert np.all(np.diff(e[r < RM]) < 0) and np.all(np.diff(e[r > RM]) > 0)


def test_pair_derivative_matches_finite_differences():
    """(1/r) dU/dr of de_pair_grad equals central differences of de_pair (1e-7 relative, h = 1e-6 nm)."""
    r = np.array([0.08, 0.2, 0.3, 0.35, 0.5, 0.9])
    e, d = de_pair_grad(r, RM, EPS)
    assert np.allclose(e, de_pair(r, RM, EPS), rtol=0, atol=0)
    h = 1e-6
    fd = (np.asarray(de_pair(r + h, RM, EPS)) - np.asarray(de_pair(r - h, RM, EPS))) / (2 * h) / r
    assert np.allclose(np.asarray(d), fd, rtol=1e-7, atol=1e-7)


def _quadrature(f, rc, rmax=40.0, n=400001):
    """Return int_rc^rmax f(r) dr by the trapezoid rule on a fine grid."""
    r = np.linspace(rc, rmax, n)
    return np.trapezoid(f(r), r)


def test_tail_and_impulse_match_quadrature():
    """de_long_range and de_tail_impulse equal 2 pi/V sum int r^2 u dr and 2 pi rc^3/(3V) sum u(rc) numerically.

    Also the identity that the strain derivative of the tail, -E_lrc - X per diagonal element, equals
    the continuum virial (2 pi / 3V) sum int r^3 u' dr (1e-6 relative).
    """
    sys, pos, H = small_box(0)
    P = sys.expand(None)
    groups = de_groups(P["lj_rmin_half"], P["lj_sqrt_eps"])
    V, rc = abs(np.linalg.det(H)), 0.6
    E = float(de_long_range(P, V, rc, groups))
    X = float(de_tail_impulse(P, V, rc, groups))
    rep, n = groups
    R, s = np.asarray(P["lj_rmin_half"])[rep], np.asarray(P["lj_sqrt_eps"])[rep]
    E_ref = X_ref = W_ref = 0.0
    for a in range(len(rep)):
        for b in range(len(rep)):
            rm, eps, w = R[a] + R[b], s[a] * s[b], n[a] * n[b]
            if rm <= 0 or eps == 0:
                continue
            E_ref += w * _quadrature(lambda r: r * r * np.asarray(de_pair(r, rm, eps)), rc)
            X_ref += w * float(de_pair(rc, rm, eps))
            h = 1e-7
            W_ref += w * _quadrature(
                lambda r: r**3 * (np.asarray(de_pair(r + h, rm, eps)) - np.asarray(de_pair(r - h, rm, eps))) / (2 * h),
                rc,
            )
    E_ref, X_ref, W_ref = 2 * np.pi * E_ref / V, 2 * np.pi * rc**3 * X_ref / (3 * V), 2 * np.pi * W_ref / (3 * V)
    assert abs(E - E_ref) < 1e-6 * abs(E_ref), (E, E_ref)
    assert abs(X - X_ref) < 1e-12 * max(1.0, abs(X_ref)) + 1e-9 * abs(X_ref), (X, X_ref)
    assert abs(-E - X - W_ref) < 1e-5 * abs(W_ref), (-E - X, W_ref)
    assert E < 0  # an attractive continuum


def test_row_forces_equal_autodiff_and_finite_differences():
    """MD forces with vdw="de" equal autodiff at fixed dipoles (1e-9) and central differences (1e-5 relative)."""
    sys, pos, H = small_box(1)
    ff, idx = ff_and_list(sys, pos, H, vdw="de", lj_lrc=True)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    P = ff._atoms(None)
    F_ad = -jax.grad(lambda x: ff.energy_fixed_mu(x, H, res.induction.mu, idx, P)[0])(jnp.asarray(pos))
    assert np.allclose(res.forces, F_ad, atol=1e-9 * float(jnp.abs(F_ad).max()))
    e = jax.jit(lambda x: ff.energy(x, H, idx, ff.init_induction())[0])
    h = 1e-6
    for a, k in [(0, 0), (5, 1), (40, 2), (77, 0)]:
        d = np.zeros_like(pos)
        d[a, k] = h
        fd = -(float(e(pos + d)) - float(e(pos - d))) / (2 * h)
        assert abs(fd - float(res.forces[a, k])) < 1e-5 * max(1.0, abs(fd)), (a, k, fd, float(res.forces[a, k]))


def test_de_differs_from_lj_and_matches_periodic_model():
    """The DE energy of the force field equals PeriodicModel(vdw="de") and differs from Lennard-Jones."""
    sys, pos, H = small_box(2)
    out = {}
    for form in ("lj", "de"):
        ff, idx = ff_and_list(sys, pos, H, vdw=form, lj_lrc=True)
        res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
        out[form] = float(res.energy["vdw"])
    model = PeriodicModel(sys, H, pos, ewald_beta=6.0, cutoff=0.6, vdw="de", lj_lrc=True)
    e_model = float(model.energy(pos)["vdw"])
    assert abs(out["de"] - e_model) < 1e-9 * max(1.0, abs(e_model)), (out["de"], e_model)
    assert abs(out["de"] - out["lj"]) > 1e-3 * abs(out["lj"]), out


def test_molecular_strain_derivative_includes_tail_impulse():
    """The molecular strain derivative matches isotropic scaling of box and centres plus the DE tail impulse."""
    sys, pos, H = small_box(2)
    ff, idx = ff_and_list(sys, pos, H, vdw="de", lj_lrc=True)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    W = ff.strain_derivative(pos, H, idx, res.induction.mu)
    P = ff._atoms(None)
    m = np.asarray(sys.masses)
    com = np.array([np.average(pos[sys.mol == k], 0, weights=m[sys.mol == k]) for k in range(sys.nmol)])
    e = jax.jit(lambda s: ff.energy(pos + (s * com)[sys.mol], H * (1 + s), idx, ff.init_induction())[0])
    h = 1e-6
    fd = (float(e(h)) - float(e(-h))) / (2 * h)
    impulse = -3 * float(ff._vdw_tail_impulse(P, jnp.asarray(H)))
    assert abs(fd + impulse - float(jnp.trace(W))) < 1e-5 * max(1.0, abs(fd)), (fd, impulse, float(jnp.trace(W)))


def test_gas_phase_channel_and_options():
    """DEChannel sums de_pair over the intermolecular pairs; check_vdw accepts "de" and rejects other names."""
    sys, pos, H = small_box(3)
    out = DEChannel().energy(jnp.asarray(pos), sys)[0]["vdw"]
    P = sys.expand(None)
    inter = sys.pair_inter
    i, j = sys.pair_i[inter], sys.pair_j[inter]
    r = np.linalg.norm(pos[i] - pos[j], axis=-1)
    rm = np.asarray(P["lj_rmin_half"])[i] + np.asarray(P["lj_rmin_half"])[j]
    eps = np.asarray(P["lj_sqrt_eps"])[i] * np.asarray(P["lj_sqrt_eps"])[j]
    assert abs(float(out) - float(np.sum(de_pair(r, rm, eps)))) < 1e-10 * max(1.0, abs(float(out)))
    check_vdw("de")
    with pytest.raises(ValueError):
        check_vdw("morse")
