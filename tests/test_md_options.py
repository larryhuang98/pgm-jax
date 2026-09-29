"""MD engine options: electrostatics levels (PME vs exact Ewald), GVDW (pair energies vs the
periodic reference, analytic row forces vs autodiff and finite differences, dispersion tail),
flexible molecules with GVDW, and the template/settings consistency check."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_md import settings, small_box  # noqa: E402

from pgm_jax import PeriodicModel, PeriodicPGM, System, set_gvdw  # noqa: E402
from pgm_jax.md.forcefield import PGMForceField  # noqa: E402
from pgm_jax.md.neighbors import Neighbors  # noqa: E402

GV = {"OW": (60.0, 0.05, 4.0), "c3": (40.0, 0.06, 3.5), "oh": (55.0, 0.05, 4.2), "h1": (8.0, 0.01, 3.0)}


def _gvdw_box(seed=0):
    sys, pos, H = small_box(seed)
    mols = {id(m): set_gvdw(m, GV) for m in sys.molecules}
    return System([mols[id(m)] for m in sys.molecules]), pos, H


def _list(sys, pos, H, s):
    return Neighbors(sys.n, H, s.cutoff, s.skin).allocate(pos, None, H).idx


@pytest.mark.parametrize("elec", ["q", "qp", "qi"])
def test_elec_levels_match_ewald(elec):
    sys, pos, H = small_box(2)
    s = settings(elec=elec)
    ff = PGMForceField(sys, H, s)
    res = jax.jit(ff.compute)(pos, H, _list(sys, pos, H, s), ff.init_induction())
    ew = PeriodicPGM(sys, H, pos, b0=6.0, rc=0.6, elec=elec).energy(pos)[0]["total"]
    assert abs(float(res.energy["elec"]) - float(ew)) < 2e-6 * abs(float(ew)), (
        elec,
        float(res.energy["elec"]),
        float(ew),
    )
    if elec in ("q", "qp"):
        assert float(jnp.abs(res.induction.mu).max()) == 0.0
    full = jax.jit(PGMForceField(sys, H, settings()).compute)(pos, H, _list(sys, pos, H, s), ff.init_induction())
    assert abs(float(full.energy["elec"]) - float(res.energy["elec"])) > 1.0  # the levels differ


@pytest.mark.parametrize("rep", ["gauss", "slater"])
def test_gvdw_md_matches_periodic_and_forces(rep):
    sys, pos, H = _gvdw_box(1)
    s = settings(vdw="gvdw", gvdw_rep=rep)
    ff = PGMForceField(sys, H, s)
    idx = _list(sys, pos, H, s)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    ref = PeriodicModel(sys, H, pos, rc=0.6, b0=6.0, vdw="gvdw", gvdw_rep=rep).energy(pos)["vdw"]
    assert abs(float(res.energy["vdw"]) - float(ref)) < 1e-9 * max(1.0, abs(float(ref))), (
        float(res.energy["vdw"]),
        float(ref),
    )
    assert abs(float(ref)) > 1e-2
    P = ff._atoms(None)
    F_ad = -jax.grad(lambda x: ff.energy_fixed_mu(x, H, res.induction.mu, idx, P)[0])(jnp.asarray(pos))
    assert np.allclose(res.forces, F_ad, atol=1e-9 * float(jnp.abs(F_ad).max()))
    e = jax.jit(lambda x: ff.energy(x, H, idx, ff.init_induction())[0])
    h = 1e-6
    for a, k in [(0, 0), (5, 1), (40, 2)]:
        d = np.zeros_like(pos)
        d[a, k] = h
        fd = -(float(e(pos + d)) - float(e(pos - d))) / (2 * h)
        assert abs(fd - float(res.forces[a, k])) < 1e-5 * max(1.0, abs(fd))


def test_gvdw_tail_and_virial():
    sys, pos, H = _gvdw_box(3)
    s = settings(vdw="gvdw", lj_lrc=True)
    ff = PGMForceField(sys, H, s)
    idx = _list(sys, pos, H, s)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    pm = PeriodicModel(sys, H, pos, rc=0.6, b0=6.0, vdw="gvdw", lj_lrc=True)
    assert abs(float(res.energy["vdw"]) - float(pm.energy(pos)["vdw"])) < 1e-9 * abs(float(pm.energy(pos)["vdw"]))
    W = ff.strain_derivative(pos, H, idx, res.induction.mu)
    W_ref = pm.virial_derivative(pos)
    # the MD engine's minimum image assumes a reduced (lower-triangular) box, so only strains that
    # keep that form are compared: the diagonal and the upper triangle (the isotropic barostat
    # needs the trace)
    iu = np.triu_indices(3)
    assert np.allclose(np.asarray(W)[iu], np.asarray(W_ref)[iu], rtol=2e-6, atol=2e-6 * float(jnp.abs(W_ref).max()))


def test_flexible_gvdw_single_molecule_and_settings_check():
    from test_flexible import template

    from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate
    from pgm_jax.md.forcefield import MDSettings

    tpl0, x = template(vdw="gvdw")
    spec = tpl0.specs[0]
    from dataclasses import replace

    spec = replace(spec, pgm=set_gvdw(spec.pgm, GV))
    from pgm_jax.bonded import terms as T
    from pgm_jax.bonded.model import BondedModel, BondedSettings

    model = BondedModel([spec], BondedSettings(families=T.PAPER, lj14_scale=0.5, vdw="gvdw"))
    tpl = FlexibleTemplate.from_fit(model, model.init_params())
    assert len(tpl.lj_pairs()[0]) == 3
    y = x + 0.004 * np.random.default_rng(0).normal(size=x.shape)
    s = MDSettings(precision="double", dipole_tol=1e-9, cutoff=1.8, skin=0.05, lj_lrc=False, vdw="gvdw")
    sim = FlexibleSimulation(System([tpl.pgm]), [tpl], y + 2.0, np.eye(3) * 4.0, s, ensemble="nve", log=None)
    F = np.asarray(sim.state.dyn.force)
    P = jax.tree_util.tree_map(jnp.asarray, tpl.P)
    g = np.asarray(jax.grad(lambda R: tpl.model.energy(0, R, P)[0])(jnp.asarray(y)))
    assert np.abs(F + g).max() < 1e-3 * np.sqrt(np.mean(g**2))
    with pytest.raises(ValueError):
        FlexibleSimulation(
            System([tpl.pgm]),
            [tpl],
            y + 2.0,
            np.eye(3) * 4.0,
            MDSettings(cutoff=1.8, skin=0.05),
            ensemble="nve",
            log=None,
        )
