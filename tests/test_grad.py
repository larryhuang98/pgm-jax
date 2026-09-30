"""Differentiability of the gas-phase and periodic models, and parameter tying.

What is checked, and against what: gradients with respect to parameters, box and positions
against central differences, including second derivatives through the induction solve (force
matching functionals) and the derivative of the induced dipoles; the default tying keys and the
tied gradient as the sum of per-atom gradients; no recompilation when parameters change; the
neutralising background of a charged periodic system (independent of the Ewald splitting).

Tolerances: parameter gradients rtol 2e-6 / atol 1e-8 by default (h = 1e-6 relative steps),
looser (1e-5 to 1e-4) for second derivatives through the iterative CG solve of the periodic model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import cluster, methanol

from pgm_jax.channels import ElecChannel, molecular_polarizability
from pgm_jax.ewald import PeriodicPGM
from pgm_jax.lj import LJChannel
from pgm_jax.model import Model
from pgm_jax.periodic import PeriodicModel
from pgm_jax.system import QUANTITIES, Molecule, System


def perturbed(params, rng, scale=0.05):
    """Return parameters moved off their initial values by a random relative scale (and q, cov shifts).

    Parameters moved off the initial values so no gradient is accidentally zero by symmetry.
    """
    return {
        k: v * (1 + scale * rng.uniform(-1, 1, size=v.shape))
        + (0.001 * rng.uniform(-1, 1, size=v.shape) if k in ("q", "cov") else 0)
        for k, v in params.items()
    }


def check_param_grad(f, params, h=1e-6, rtol=2e-6, atol=1e-8):
    """Assert that jax.grad(f) matches central differences for every parameter entry.

    Parameters
    ----------
    f : callable
        Scalar function of the parameter dict [kJ/mol].
    params : dict
        Parameters (dict of arrays by quantity).
    h : float
        Relative step (at least h in absolute terms).
    rtol, atol : float
        Tolerance atol + rtol max(1, |fd|) on each entry.
    """
    g = jax.jit(jax.grad(f))(params)
    f = jax.jit(f)
    for qn in QUANTITIES:
        for k in range(len(params[qn])):
            step = h * max(1.0, abs(float(params[qn][k])))
            up = {**params, qn: params[qn].at[k].add(step)}
            dn = {**params, qn: params[qn].at[k].add(-step)}
            fd = (float(f(up)) - float(f(dn))) / (2 * step)
            assert abs(fd - float(g[qn][k])) < atol + rtol * max(1.0, abs(fd)), (qn, k, fd, float(g[qn][k]))


# ------------------------------------------------------------------------------ tying --


def test_default_tying_keys():
    """The default tying keys are per molecule for charges and per type for alpha, independent of atom order."""
    sys, _ = cluster(np.random.default_rng(0))
    t = sys.table
    assert t.keys["q"] == ["WAT:OW", "WAT:HW", "MeOH:c3", "MeOH:oh", "MeOH:h1", "MeOH:ho"]
    assert t.keys["alpha"] == ["OW", "HW", "c3", "oh", "h1", "ho"]
    assert set(t.keys["cov"]) == {
        "WAT:OW>HW",
        "WAT:HW>OW",
        "MeOH:c3>oh",
        "MeOH:oh>c3",
        "MeOH:oh>ho",
        "MeOH:ho>oh",
        "MeOH:c3>h1",
        "MeOH:h1>c3",
    }
    # symmetry classes do not depend on atom order
    m, _ = methanol()
    perm = [5, 3, 1, 0, 4, 2]
    inv = np.argsort(perm)
    m2 = Molecule(
        m.name,
        [m.elements[k] for k in perm],
        [m.types[k] for k in perm],
        m.q[perm],
        m.radius[perm],
        m.alpha[perm],
        cov=[(int(inv[i]), int(inv[j]), c) for i, j, c in m.cov],
        bonds=[(int(inv[i]), int(inv[j])) for i, j in m.bonds],
    )
    k1, k2 = m.tying_keys(), m2.tying_keys()
    assert [k1["q"][k] for k in perm] == k2["q"]


def test_tied_gradient_is_sum_of_atom_gradients():
    """The gradient of a tied parameter is the sum of the per-atom (untied) gradients (1e-10)."""
    sys, pos = cluster(np.random.default_rng(1))
    untied = [
        Molecule(
            **{**m.__dict__, "keys": {qn: [f"{id(m)}:{qn}{k}" for k in range(m.n_terms(qn))] for qn in QUANTITIES}}
        )
        for m in sys.molecules
    ]
    sys_u = System(untied)

    def f(s):
        """Return the energy of system s as a function of its parameters."""
        return lambda P: Model([ElecChannel(), LJChannel()]).energy_fn(s)(jnp.asarray(pos), P)["total"]

    g_t = jax.grad(f(sys))(sys.params0)
    g_u = jax.grad(f(sys_u))(sys_u.params0)
    for qn in QUANTITIES:
        summed = np.zeros(len(sys.table.keys[qn]))
        np.add.at(summed, sys.idx[qn], np.asarray(g_u[qn])[sys_u.idx[qn]])
        assert np.allclose(summed, g_t[qn], rtol=1e-10, atol=1e-12), qn


def test_no_recompile_when_parameters_change():
    """Changing parameter values reuses the compiled batch energy (one jit cache entry)."""
    sys, pos = cluster(np.random.default_rng(2))
    model = Model([ElecChannel(), LJChannel()])
    P0 = sys.params0
    P1 = perturbed(P0, np.random.default_rng(3))
    e0 = model.batch_energy(sys, pos[None], P0)["total"]
    e1 = model.batch_energy(sys, pos[None], P1)["total"]
    assert len(model._jit_cache) == 1 and abs(e0[0] - e1[0]) > 1e-3
    fn = list(model._jit_cache.values())[0]
    assert getattr(fn, "_cache_size", lambda: 1)() == 1


# ---------------------------------------------------------------------------- gas phase --


def test_gas_parameter_gradients():
    """Gas-phase parameter gradients of the energy match central differences."""
    rng = np.random.default_rng(4)
    sys, pos = cluster(rng)
    P = perturbed(sys.params0, rng)
    f = Model([ElecChannel(), LJChannel()]).energy_fn(sys)
    check_param_grad(lambda p: f(jnp.asarray(pos), p)["total"], P)


def test_gas_force_parameter_gradients():
    """Parameter gradients of a force functional match central differences (rtol 1e-5).

    d/dparams of a force-matching-like functional (second derivatives through the solve).
    """
    rng = np.random.default_rng(5)
    sys, pos = cluster(rng)
    P = perturbed(sys.params0, rng)
    w = rng.normal(size=pos.shape)
    fo = Model([ElecChannel(), LJChannel()]).forces_fn(sys)
    check_param_grad(lambda p: jnp.sum(fo(jnp.asarray(pos), p) * w), P, rtol=1e-5, atol=1e-6)


def test_polarizability_parameter_gradients():
    """Parameter gradients of the molecular polarizability trace match central differences."""
    rng = np.random.default_rng(6)
    m, x = methanol()
    sys = System([m])
    P = perturbed(sys.params0, rng)
    check_param_grad(lambda p: jnp.trace(molecular_polarizability(jnp.asarray(x), sys, p)) * 1e3, P)


# ----------------------------------------------------------------------------- periodic --


@pytest.fixture(scope="module")
def periodic():
    """Return a perturbed cluster in a triclinic box with its PeriodicModel (fixture, module scope)."""
    rng = np.random.default_rng(7)
    sys, pos = cluster(rng)
    H = np.array([[1.45, 0.0, 0.0], [0.15, 1.40, 0.0], [-0.10, 0.20, 1.35]])  # triclinic, nm
    pos = pos + np.array([0.5, 0.5, 0.5])
    model = PeriodicModel(sys, H, pos, cutoff=0.6, ewald_beta=6.5, skin=0.05, k_tol=1e-10, dipole_tol=1e-13)
    P = perturbed(sys.params0, rng)
    return model, sys, pos, H, P


def test_periodic_forces(periodic):
    """Periodic forces match central differences of the energy (1e-5 relative, h = 1e-6 nm)."""
    model, sys, pos, H, P = periodic
    F = np.asarray(jax.jit(model.forces)(pos, P))
    ej = jax.jit(lambda x: model.energy(x, P)["total"])

    def e(x):
        """Return the periodic energy at x [kJ/mol]."""
        return float(ej(x))

    h = 1e-6
    for a, k in [(0, 0), (4, 1), (7, 2), (10, 0)]:
        d = np.zeros_like(pos)
        d[a, k] = h
        fd = -(e(pos + d) - e(pos - d)) / (2 * h)
        assert abs(fd - F[a, k]) < 1e-5 * max(1.0, abs(fd)), (a, k, fd, F[a, k])


def test_periodic_parameter_gradients(periodic):
    """Periodic parameter gradients match central differences (rtol 1e-5)."""
    model, sys, pos, H, P = periodic
    check_param_grad(lambda p: model.energy(pos, p)["total"], P, rtol=1e-5, atol=1e-7)


def test_periodic_force_parameter_gradients(periodic):
    """Second derivatives through the periodic CG solve match central differences (rtol 1e-4).

    Second derivatives through the CG solve (custom_jvp + custom_linear_solve).
    """
    model, sys, pos, H, P = periodic
    w = np.random.default_rng(8).normal(size=pos.shape)
    check_param_grad(lambda p: jnp.sum(model.forces(pos, p) * w), P, rtol=1e-4, atol=1e-5)


def test_periodic_box_gradient(periodic):
    """The gradient with respect to the box matrix matches central differences (1e-5 relative)."""
    model, sys, pos, H, P = periodic
    ej = jax.jit(lambda h: model.energy(pos, P, h)["total"])
    g = np.asarray(jax.grad(ej)(jnp.asarray(H)))

    def e(h):
        """Return the periodic energy for box h [kJ/mol]."""
        return float(ej(h))

    step = 1e-6
    for a in range(3):
        for b in range(3):
            d = np.zeros((3, 3))
            d[a, b] = step
            fd = (e(H + d) - e(H - d)) / (2 * step)
            assert abs(fd - g[a, b]) < 1e-5 * max(1.0, abs(fd)), (a, b, fd, g[a, b])


def test_periodic_pressure_is_minus_dE_dV(periodic):
    """The molecular strain derivative matches isotropic scaling of box and centres.

    Molecular strain derivative vs finite differences of isotropic scaling of box and centres of mass.
    """
    model, sys, pos, H, P = periodic
    W = np.asarray(jax.jit(model.strain_derivative)(pos, P))
    w = sys.masses
    com = np.array([np.average(pos[sys.mol == k], axis=0, weights=w[sys.mol == k]) for k in range(sys.nmol)])

    ej = jax.jit(lambda s: model.energy(pos + (s * com)[sys.mol], P, H * (1 + s))["total"])

    def e(s):
        """Return the energy with box and molecular centres scaled by 1 + s."""
        return float(ej(s))

    h = 1e-6
    fd = (e(h) - e(-h)) / (2 * h)
    assert abs(fd - np.trace(W)) < 1e-5 * max(1.0, abs(fd)), (fd, np.trace(W))


def test_induced_dipole_derivative(periodic):
    """The JVP of the induced dipoles in alpha matches central differences (1e-5 relative)."""
    model, sys, pos, H, P = periodic
    mu = jax.jit(lambda p: model.elec.induced_dipoles(pos, p))
    k = 0
    t = {qn: jnp.zeros_like(v) for qn, v in P.items()}
    t["alpha"] = t["alpha"].at[k].set(1.0)
    _, dmu = jax.jvp(mu, (P,), (t,))
    h = 1e-9
    fd = (mu({**P, "alpha": P["alpha"].at[k].add(h)}) - mu({**P, "alpha": P["alpha"].at[k].add(-h)})) / (2 * h)
    assert np.allclose(dmu, fd, rtol=1e-5, atol=1e-6 * float(jnp.abs(fd).max()))


def test_charged_system_independent_of_ewald_splitting():
    """A charged periodic system's energy does not depend on the Ewald splitting (1e-4 kJ/mol).

    With a net charge the neutralising-background term keeps the energy independent of b0.
    """
    rng = np.random.default_rng(9)
    sys, pos = cluster(rng)
    P = dict(sys.params0)
    P["q"] = P["q"].at[0].add(0.5)  # net charge
    pos = pos + 3.0
    H = np.eye(3) * 6.0
    e1 = float(PeriodicPGM(sys, H, pos, ewald_beta=1.2, cutoff=2.9).energy(pos, P)[0]["total"])
    e2 = float(PeriodicPGM(sys, H, pos, ewald_beta=1.0, cutoff=2.9, k_tol=1e-10).energy(pos, P)[0]["total"])
    assert abs(e1 - e2) < 1e-4, (e1, e2)
