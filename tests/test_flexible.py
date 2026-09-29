"""Flexible-molecule MD: templates, the single-molecule limit (MD forces = gas-phase model),
NVE energy conservation and an NPT run that compresses a dilute box (neighbour-list rebuilds)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import methanol_liquid, methanol_template

from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.thermostats import Langevin
from pgm_jax.system import System
from pgm_jax.units import KB


def test_template_roundtrip_and_pgm_only(tmp_path):
    tpl, x = methanol_template()
    i, j, w = tpl.lj_pairs()
    assert len(i) == 3 and np.allclose(w, 0.5)  # the three H-C-O-H pairs
    tpl.save(str(tmp_path / "m.flex"))
    t2 = FlexibleTemplate.load(str(tmp_path / "m.flex"))
    y = jnp.asarray(x + 0.004 * np.random.default_rng(1).normal(size=x.shape))
    assert abs(float(t2.bonded_energy(y)) - float(tpl.bonded_energy(y))) < 1e-10
    assert float(tpl.bonded_energy(y)) > 0.0
    for bad in (dict(elec_exclude=3), dict(escale=(1,))):  # (charge flux runs: test_flux.py)
        with pytest.raises(ValueError):
            methanol_template(**bad)


def test_single_molecule_matches_gas_phase_model():
    """One molecule in a large box: MD forces (PME pGM + bonded + intramolecular LJ) equal the
    gradient of the gas-phase model the bonded terms are fitted with."""
    tpl, x = methanol_template()
    y = x + 0.004 * np.random.default_rng(0).normal(size=x.shape)
    s = MDSettings().replace(precision="double", dipole_tol=1e-9, cutoff=1.8, skin=0.05, lj_lrc=False)
    sim = FlexibleSimulation(System([tpl.pgm]), [tpl], y + 2.0, np.eye(3) * 4.0, s, thermostat=None, log=None)
    F = np.asarray(sim.state.dyn.force)
    P = jax.tree_util.tree_map(jnp.asarray, tpl.P)
    g = np.asarray(jax.grad(lambda R: tpl.model.energy(0, R, P)[0])(jnp.asarray(y)))
    rms = np.sqrt(np.mean(g**2))
    assert np.abs(F + g).max() < 1e-3 * rms, (np.abs(F + g).max(), rms)


def test_nve_energy_conservation():
    tpl, sys, pos, H = methanol_liquid()
    s = MDSettings().replace(precision="double", dipole_tol=1e-8, cutoff=0.6, skin=0.05, lj_lrc=False)
    sim = FlexibleSimulation(
        sys, [tpl] * sys.nmol, pos, H, s, dt=0.0005, thermostat=Langevin(10.0), temperature=298.0, log=None
    )
    sim.advance(1000)
    sim2 = FlexibleSimulation(
        sys,
        [tpl] * sys.nmol,
        sim.positions(),
        np.asarray(sim.state.box),
        s,
        dt=0.0005,
        thermostat=None,
        velocities=sim.velocities(),
        log=None,
    )
    E = []
    for _ in range(10):
        sim2.advance(100)
        E.append(sim2.observables()["etot"])
    ke = 0.5 * sim2.integ.dof * KB * 298.0
    assert np.std(E) < 1e-3 * ke and abs(E[-1] - E[0]) < 2e-3 * ke, (np.std(E) / ke, (E[-1] - E[0]) / ke)
    # molecules stay whole and bonded
    X = sim2.positions().reshape(sys.nmol, tpl.n, 3)
    b = np.linalg.norm(X[:, [0, 1]] - X[:, [1, 5]], axis=-1)
    assert b.max() < 0.16 and b.min() > 0.08


def test_npt_compresses_dilute_box():
    """NPT at 2 kbar from a dilute box: the box shrinks by far more than the neighbour lists were
    built for, so the driver must rebuild them on the way (and the molecules must stay whole)."""
    tpl, sys, pos, H = methanol_liquid(density=0.45)
    s = MDSettings().replace(precision="mixed", dipole_tol=1e-5, cutoff=0.5, skin=0.05)
    sim = FlexibleSimulation(
        sys,
        [tpl] * sys.nmol,
        pos,
        H,
        s,
        dt=0.0005,
        thermostat=Langevin(5.0),
        barostat=MonteCarloBarostat(2000.0, 5),
        temperature=298.0,
        log=None,
    )
    rho0 = sim.observables()["density_g_cm3"]
    for _ in range(4):
        sim.advance(1000)
    o = sim.observables()
    assert np.isfinite(o["etot"]) and o["density_g_cm3"] > rho0 * 1.3, (rho0, o["density_g_cm3"])
    assert getattr(sim, "n_rebuilds", 0) >= 1
    X = sim.positions().reshape(sys.nmol, tpl.n, 3)
    assert np.linalg.norm(X[:, 0] - X[:, 1], axis=-1).max() < 0.16
