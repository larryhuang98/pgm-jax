"""Isotropic periodic sum (IPS): coefficients, pair functions, and the MD force field with long_range="ips".

What is checked, and against what: the order-4 Gaussian coefficients against sander's closed formulas
(aipseg, 1e-13) and the point-charge ones against AIPSE; that the pair function and its first N - 1
derivatives vanish at the cutoff for N = 3..8 (1e-9); the polynomial kernels G_n against autodiff (1e-12);
the LJ and DE IPS pair forces against autodiff of their energies, and the zero DE force at the cutoff; the
electrostatic energy of the MD force field (charges, permanent and arbitrary induced dipoles, IPS orders 4
and 6) against an independent brute-force sum built from the pair function with autodiff derivatives
(1e-10 relative); the row forces against autodiff at fixed dipoles (1e-9) and central differences of the
energy (1e-5 relative), for LJ and DE; and that the default (PME) energies are unchanged by the new
setting.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import small_box
from jax.scipy.special import erf
from test_md import ff_and_list

from pgm_jax import ips
from pgm_jax.md.box import min_image
from pgm_jax.units import KE


def _aipseg(g):
    """Return sander's closed-form order-4 Gaussian IPS coefficients (degauss_setup_system_ips)."""
    g2 = g * g
    e, er, rp = math.exp(-g2), math.erf(g), math.sqrt(math.pi)
    return np.array(
        [
            (2 * e * g * (57 + g2 * (22 + 4 * g2)) / rp - 105 * er) / 48,
            (105 * er - 6 * e * g * (35 + g2 * (18 + 4 * g2)) / rp) / 48,
            (6 * e * g * (21 + g2 * (14 + 4 * g2)) / rp - 63 * er) / 48,
            (15 * er - e * g * (30 + g2 * (20 + 8 * g2)) / rp) / 48,
        ]
    )


def test_order4_coefficients_are_sanders():
    """Order 4 reproduces aipseg for Gaussians and AIPSE for the point charge."""
    for g in (0.7, 1.5, 3.0, 5.0):
        assert np.abs(np.asarray(ips.elec_coefficients(jnp.asarray(g), 4)) - _aipseg(g)).max() < 1e-13
    assert np.allclose(ips.point_coefficients(4) * 16, [-35, 35, -21, 5], atol=1e-12)
    assert np.abs(np.asarray(ips.elec_coefficients(jnp.asarray(14.0), 4)) - ips.point_coefficients(4)).max() < 1e-10


@pytest.mark.parametrize("order", [3, 4, 5, 6, 8])
def test_pair_function_and_derivatives_vanish_at_cutoff(order):
    """Phi(rc) and its first order - 1 derivatives are zero."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    g = 2.5
    c = [mp.mpf(float(v)) for v in np.asarray(ips.elec_coefficients(jnp.asarray(g), order))]

    def Phi(u):
        return mp.erf(g * u) / u + sum(c[k] * u ** (2 * k) for k in range(order))

    assert max(abs(mp.diff(Phi, 1, m)) for m in range(order)) < 1e-9


def test_polynomial_kernels_match_autodiff():
    """G_n of the polynomial equal (-(1/r) d/dr)^n applied by autodiff."""
    rc = 0.9
    c = ips.elec_coefficients(jnp.asarray(2.0), 5)

    def f(r):
        return sum(c[k] * (r / rc) ** (2 * k) for k in range(5)) / rc

    g1 = lambda r: -jax.grad(f)(r) / r  # noqa: E731
    g2 = lambda r: -jax.grad(g1)(r) / r  # noqa: E731
    g3 = lambda r: -jax.grad(g2)(r) / r  # noqa: E731
    r0 = 0.37
    G = ips.poly_kernels(c, jnp.asarray(r0), rc, 4)
    ref = [f(r0), g1(r0), g2(r0), g3(r0)]
    assert np.allclose([float(x) for x in G], [float(x) for x in ref], rtol=1e-12)


def test_lj_and_de_pair_forces():
    """(1/r) dU/dr equals autodiff; the DE IPS force and the LJ IPS energy-shift behave at the cutoff."""
    rmin, eps, rc = 0.35, 0.6, 0.9
    for r in (0.2, 0.31, 0.5, 0.85):
        r = jnp.asarray(r)
        e, d = ips.lj_ips_pair(r, rmin, eps, rc)
        assert abs(float(d) - float(jax.grad(lambda x: ips.lj_ips_pair(x, rmin, eps, rc)[0])(r) / r)) < 1e-9 * abs(float(d))
        e, d = ips.de_ips_pair(r, rmin, eps, 18.2, 3.6, rc)
        ref = jax.grad(lambda x: ips.de_ips_pair(x, rmin, eps, 18.2, 3.6, rc)[0])(r) / r
        assert abs(float(d) - float(ref)) < 1e-9 * abs(float(d))
    assert abs(float(ips.de_ips_pair(jnp.asarray(rc), rmin, eps, 18.2, 3.6, rc)[1])) < 1e-12
    assert abs(float(ips.lj_ips_pair(jnp.asarray(rc), rmin, eps, rc)[0])) < 1e-12


def _brute_elec(ff, pos, H, P, mu, order):
    """Return the IPS electrostatic energy [kJ/mol] from the pair function with autodiff derivatives."""
    rc = ff.rc_e
    q, R = np.asarray(P["q"]), np.asarray(P["radius"])
    d = np.asarray(ff.perm_dipoles(jnp.asarray(pos), H, P["cov"]) + mu)
    n = len(q)
    e = 0.0
    for i in range(n):
        for k in range(i + 1, n):
            x = np.asarray(min_image(jnp.asarray(pos[i] - pos[k]), H))
            r = np.linalg.norm(x)
            if r >= rc:
                continue
            a = 1.0 / math.sqrt(2.0 * (R[i] ** 2 + R[k] ** 2))
            c = ips.elec_coefficients(jnp.asarray(a * rc), order)

            def Phi(v, a=a, c=c):
                rr = jnp.linalg.norm(v)
                u = rr / rc
                return erf(a * rr) / rr + sum(c[m] * u ** (2 * m) for m in range(order)) / rc

            xv = jnp.asarray(x)
            gr, hs = jax.grad(Phi)(xv), jax.hessian(Phi)(xv)
            e += float(
                q[i] * q[k] * Phi(xv) - q[i] * d[k] @ gr + q[k] * d[i] @ gr - d[i] @ hs @ d[k]
            )
    for i in range(n):  # self images: 1/2 q^2 c0 / rc and -c1 |d|^2 / rc^3
        c = ips.elec_coefficients(jnp.asarray(rc / (2.0 * R[i])), order)
        e += 0.5 * q[i] ** 2 * float(c[0]) / rc - float(c[1]) * float(d[i] @ d[i]) / rc**3
    alpha = np.asarray(P["alpha"])
    pol = sum(float(mu[i] @ mu[i]) / (2 * alpha[i]) for i in range(n) if alpha[i] > 0)
    return KE * (e + pol)


@pytest.mark.parametrize("order", [4, 6])
def test_md_electrostatics_equal_brute_force(order):
    """The force field's IPS electrostatic energy equals the brute-force pair-function sum (1e-10 relative)."""
    sys, pos, H = small_box(1)
    ff, idx = ff_and_list(sys, pos, H, long_range="ips", ips_order=order, vdw="none", cutoff=0.6)
    P = ff._atoms(None)
    rng = np.random.default_rng(7)
    mu = jnp.asarray(1e-3 * rng.standard_normal((sys.n, 3)))
    e, parts = ff.energy_fixed_mu(jnp.asarray(pos), H, mu, idx, P)
    ref = _brute_elec(ff, pos, H, P, mu, order)
    assert abs(float(parts["elec"]) - ref) < 1e-10 * abs(ref), (float(parts["elec"]), ref)


@pytest.mark.parametrize("form", ["lj", "de"])
def test_md_forces_equal_autodiff_and_finite_differences(form):
    """IPS row forces (charges, dipoles, van der Waals) equal autodiff (1e-9) and central differences (1e-5)."""
    sys, pos, H = small_box(1)
    ff, idx = ff_and_list(sys, pos, H, long_range="ips", vdw=form)
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


def test_default_is_pme_and_ips_differs():
    """long_range defaults to "pme" (energies unchanged); IPS gives a different, finite energy."""
    sys, pos, H = small_box(1)
    out = {}
    for lr in ("pme", "ips"):
        ff, idx = ff_and_list(sys, pos, H, long_range=lr)
        out[lr] = float(jax.jit(ff.compute)(pos, H, idx, ff.init_induction()).energy["total"])
    ff0, idx0 = ff_and_list(sys, pos, H)
    assert float(jax.jit(ff0.compute)(pos, H, idx0, ff0.init_induction()).energy["total"]) == out["pme"]
    assert np.isfinite(out["ips"]) and out["ips"] != out["pme"]


def test_de_boundary_constant():
    """The pmemd boundary energy is the pair function at rc times the pair fraction: negative for water."""
    rm, eps, rc = 0.35366, 0.636, 0.9
    phi = float(ips.de_ips_pair(jnp.asarray(rc), jnp.asarray(rm), jnp.asarray(eps), 18.17, 3.65, rc)[0])
    assert phi < 0.0
    assert abs(phi / 4.184 + 1.447622e-3) < 1e-6
