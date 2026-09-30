"""Thermostats (md/thermostats.py): O steps, Bussi, GLE, and constrained NVT runs.

What is checked, and against what: exact O steps keep the Maxwell distribution (<v^2> = kT for
unit masses, and the auxiliaries' N(0, kT)) and relax a hot start (3 %, 20,000 x 3 samples);
Bussi's kinetic-energy fluctuations are canonical (mean 3 %, variance 15 %, statistical); GLE
kernels and the fluctuation-dissipation check; constrained NVT runs conserve the effective energy
E_tot + |aux|^2 / 2 - heat about as well as NVE conserves E_tot; the thermostat / barostat
objects and the command-line helper.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import water, water_lattice

from pgm_jax.md.thermostats import GLE, Bussi, Langevin, make_thermostat

KT = 2.4777


def IDENT(u):
    """Return u unchanged (the identity projection of unconstrained velocities)."""
    return u


def _run(th, v, aux, n, h=0.002, dof=None, seed=0):
    """Apply n O steps of thermostat th and return (v, aux, per-step [<v^2>, <aux^2>, K]).

    Parameters
    ----------
    th : object
        Thermostat (Langevin, Bussi, GLE).
    v : jax.Array
        Velocities (unit masses, so <v^2> = kT at equilibrium).
    aux : jax.Array
        Auxiliary momenta of the thermostat.
    n : int
        Number of steps.
    h : float
        Time step [ps].
    dof : float, optional
        Degrees of freedom (None: v.size).
    seed : int
        Seed of the random keys.
    """
    dof = float(v.size if dof is None else dof)
    step = jax.jit(lambda v, a, k: th.apply(v, a, k, h, KT, dof, IDENT, None))
    key = jax.random.PRNGKey(seed)
    out = []
    for _ in range(n):
        key, k = jax.random.split(key)
        v, aux = step(v, aux, k)
        a2 = float(jnp.mean(aux * aux)) if aux.size else KT
        out.append((float(jnp.mean(v * v)), a2, float(0.5 * jnp.sum(v * v))))
    return v, aux, np.array(out)


@pytest.mark.parametrize(
    "th",
    [Langevin(5.0), Bussi(0.1), GLE.band(), GLE.lowpass(5.0, 50.0)],
    ids=["langevin", "bussi", "gle-band", "gle-lowpass"],
)
def test_o_step_keeps_and_reaches_maxwell(th):
    """Every O step keeps the Maxwell distribution and relaxes a hot start to kT (3 %).

    The slow-band GLE, which acts without forces only through its small zero-frequency floor, is
    checked through its drift matrix instead: every eigenvalue has a positive real part.
    """
    key = jax.random.PRNGKey(1)
    shape = (20000, 3)
    v = jnp.sqrt(KT) * jax.random.normal(key, shape)
    aux = th.init_aux(jax.random.PRNGKey(2), shape, KT)
    _, _, r = _run(th, v, aux, 200)
    assert abs(r[:, 0].mean() / KT - 1) < 0.03 and abs(r[:, 1].mean() / KT - 1) < 0.03, r[:, :2].mean(0) / KT
    # hot start (2x kinetic energy) relaxes to kT.  Without forces the slow-band GLE acts only
    # through its small zero-frequency floor, so it is checked through its drift matrix instead:
    # every eigenvalue has a positive real part (no conserved combination, ergodic O step).
    if isinstance(th, GLE) and th.n_aux == 2:
        assert np.linalg.eigvals(th.A).real.min() > 0
        return
    _, _, r = _run(th, v * np.sqrt(2.0), aux, 3000)
    assert abs(r[-500:, 0].mean() / KT - 1) < 0.03, r[-500:, 0].mean() / KT


def test_bussi_canonical_kinetic_energy_fluctuations():
    """Bussi gives the canonical kinetic-energy mean and variance (n_f kT^2 / 2) of 30 dof."""
    n_f = 30
    v = jnp.sqrt(KT) * jax.random.normal(jax.random.PRNGKey(3), (n_f,))
    _, _, r = _run(Bussi(0.01), v, jnp.zeros((0, n_f)), 20000, h=0.002)
    K = r[2000:, 2]
    assert abs(K.mean() / (0.5 * n_f * KT) - 1) < 0.03
    assert abs(K.var() / (0.5 * n_f * KT**2) - 1) < 0.15, K.var() / (0.5 * n_f * KT**2)


def test_gle_kernels_and_fdt_check():
    """GLE kernels have the requested shapes; non-positive A + A^T and unknown names are refused."""
    band = GLE.band(peak=3.0, center=20.0, width=30.0, floor=0.1)
    K = band.kernel([0.0, 20.0, 1000.0])
    assert abs(K[0] - 0.1) < 1e-12 and abs(K[1] / 3.0 - 1) < 0.03 and K[2] < 0.01
    pure = GLE.band(floor=0.0)
    assert abs(pure.kernel([0.0])[0]) < 1e-12 and abs(np.linalg.det(pure.A)) < 1e-9  # conserved mode
    low = GLE.lowpass(2.0, 40.0)
    assert np.allclose(low.kernel([0.0, 40.0]), [2.0, 1.0])
    with pytest.raises(ValueError):
        GLE([[0.0, 1.0], [-1.0, -0.5]])  # A + A^T not positive semidefinite
    assert isinstance(make_thermostat("bussi"), Bussi) and make_thermostat("gle").n_aux == 2
    assert make_thermostat(None) is None and make_thermostat(Bussi(0.5)).tau == 0.5
    with pytest.raises(ValueError, match="unknown name"):
        make_thermostat("berendsen")


@pytest.mark.slow
def test_constrained_nvt_runs_conserve_effective_energy():
    """Constrained NVT with every thermostat conserves econs and keeps T near the target.

    Rigid water by constraints, 2 fs: every thermostat holds the constraints, keeps T near the
    target and conserves E_tot + |aux|^2/2 - heat as well as NVE conserves E_tot.
    """
    from pgm_jax import System
    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    from pgm_jax.md.forcefield import MDSettings

    pos, H, w = water_lattice()
    wat = water()
    sys = System([wat] * (len(pos) // 3))
    s = MDSettings().replace(precision="double", dipole_tol=1e-9, cutoff=0.55, skin=0.05)
    tpl = RigidTemplate(wat, w)
    res = {}
    for name in ("nve", "langevin", "bussi", "gle"):
        sim = FlexibleSimulation(
            sys,
            [tpl] * sys.nmol,
            pos,
            H,
            s,
            dt=0.002,
            temperature=300.0,
            log=None,
            thermostat={"nve": None, "bussi": Bussi(0.2)}.get(name, name),
            seed=4,
        )
        sim.advance(100)
        e0 = sim.observables()["econs"]
        dev, T = 0.0, []
        for _ in range(6):
            sim.advance(50)
            o = sim.observables()
            dev = max(dev, abs(o["econs"] - e0))
            T.append(o["temp_K"])
            assert o["shake_err"] < 1e-9
        res[name] = (dev, np.mean(T), o["ekin"])
    for name, (dev, T, ek) in res.items():
        assert dev < 3.0 * res["nve"][0] + 0.005 * ek, (name, res)
        if name != "nve":
            assert 200.0 < T < 400.0, (name, T)


def test_coupling_objects():
    """Thermostat / barostat objects of the engines, the command-line helper, PILE settings."""
    from pgm_jax.cli.args import make_coupling
    from pgm_jax.md.barostats import MonteCarloBarostat, ensemble_name
    from pgm_jax.md.pimd import PILE, as_pile

    th, b = make_coupling("bussi", tau=0.5, barostat="mc", pressure=2.0, barostat_every=10)
    assert isinstance(th, Bussi) and th.tau == 0.5 and b == MonteCarloBarostat(2.0, 10)
    assert make_coupling("none") == (None, None)
    th, b = make_coupling("langevin", friction=5.0)
    assert isinstance(th, Langevin) and th.friction == 5.0 and b is None
    assert ensemble_name(None, None) == "nve" and ensemble_name(Bussi(), None) == "nvt"
    assert ensemble_name(Bussi(), MonteCarloBarostat()) == "npt"
    for bad in (lambda: ensemble_name(None, MonteCarloBarostat()), lambda: MonteCarloBarostat(every=0)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError, match="friction"):
        Langevin(-1.0)
    assert as_pile("pile-g") == PILE("g") and PILE("pile-l").kind == "l" and PILE().tau_centroid == 0.2
    with pytest.raises(ValueError):
        PILE("x")
