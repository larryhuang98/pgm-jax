"""Hydrogen mass repartitioning per molecule (md/constraints.py hmr_masses, AmberSystem.hmr,
FlexibleSimulation(hmr=...)): which atoms change, total masses conserved, no silent defaults."""
import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pgm_jax import System  # noqa: E402
from pgm_jax.md.constraints import hmr_masses, repartition_masses  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from test_grad import water  # noqa: E402
from test_md_macro import _water_box  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
PRM, CRD = os.path.join(DATA, "pep_wat.prmtop"), os.path.join(DATA, "pep_wat.inpcrd")   # ACE-ALA-SER-NME, TIP3P, NaCl


def _cluster():
    """Eight waters (2 x 2 x 2 lattice) in a 3 nm box with a 1.2 nm cutoff: no pair crosses the
    cutoff, so NVE conserves the energy to the integration error (a hard cutoff in a small box
    makes the energy jump by several kJ/mol)."""
    pos, _, w = _water_box(n_side=2, spacing=0.31)
    return pos + 1.2, np.eye(3) * 3.0, w


def test_per_molecule_hydrogen_masses():
    """Solvated peptide: protein hydrogens 3.024, water hydrogens 4.0, ions untouched; each
    molecule keeps its mass and only hydrogens and the heavy atoms bonded to them change."""
    from pgm_jax.protein import load_amber
    asys = load_amber(PRM, CRD)
    sys = asys.system()
    hmr = asys.hmr({"protein": 3.024, "water": 4.0, "ion": None})
    kinds = [m.kind for m in asys.molecules]
    assert [h for h, k in zip(hmr, kinds) if k == "ion"] == [None, None]
    m0, m = np.asarray(sys.masses), hmr_masses(sys, hmr)
    el = np.array(sys.elements)
    for k, mol in enumerate(asys.molecules):
        sl = sys.atom_slice(k)
        assert abs(m[sl].sum() - m0[sl].sum()) < 1e-9                     # per molecule
        if mol.kind == "ion":
            assert np.array_equal(m[sl], m0[sl])
        else:
            h = el[sl] == "H"
            assert np.allclose(m[sl][h], hmr[k])
            changed = ~np.isclose(m[sl], m0[sl])
            assert np.all(changed[h])                                    # every hydrogen
            bonded = {int(x) for i, j in mol.molecule.bonds for x, y in ((i, j), (j, i)) if el[sl][y] == "H" and el[sl][x] != "H"}
            assert {int(a) for a in np.nonzero(changed & ~h)[0]} == bonded   # and only their heavy partners
    wk = kinds.index("water")
    o = sys.atom_slice(wk).start
    assert abs(m[o] - (m0[o] - 2 * (4.0 - m0[o + 1]))) < 1e-9
    # a single value is the old uniform repartitioning; None leaves the masses alone
    assert np.allclose(hmr_masses(sys, 3.024)[el == "H"], 3.024)
    assert np.array_equal(hmr_masses(sys, None), m0)
    bonds = np.concatenate([np.asarray(mm.bonds, int).reshape(-1, 2) + sys.offsets[k] for k, mm in enumerate(sys.molecules)])
    assert np.allclose(hmr_masses(sys, 3.024), repartition_masses(m0, sys.elements, bonds, 3.024))
    with pytest.raises(KeyError):
        asys.hmr({"protein": 3.024})                                   # water has hydrogens
    with pytest.raises(ValueError):
        asys.hmr({"protein": 3.024, "waters": 4.0})
    with pytest.raises(TypeError):
        hmr_masses(sys, {"water": 4.0})
    with pytest.raises(ValueError):
        hmr_masses(sys, [3.024, 4.0])
    bad = [None] * sys.nmol
    bad[wk] = 10.0                                                     # the water oxygen would go negative
    with pytest.raises(ValueError):
        hmr_masses(sys, bad)
    # CH3 carbons cannot give 3 x 3 amu to their hydrogens at 4.0 and stay heavier than ~3 amu
    mp = hmr_masses(sys, [4.0 if k == "protein" else None for k in kinds])
    assert mp[el != "H"].min() < 3.1



def test_flexible_simulation_per_molecule_hmr():
    """FlexibleSimulation(hmr=[per molecule]): dynamics, constraints and the molecular centres use
    the repartitioned masses; a water cluster with half the waters at 4 amu hydrogens conserves
    the energy at 2 fs (NVE, no pair crosses the cutoff)."""
    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    pos, H, w = _cluster()
    wat = water()
    nmol = len(pos) // 3
    sys = System([wat] * nmol)
    hmr = [4.0, None] * (nmol // 2)
    s = MDSettings(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
    sim = FlexibleSimulation(sys, [RigidTemplate(wat, w)] * nmol, pos, H, s, dt=0.002, ensemble="nve", hmr=hmr,
                             temperature=300.0, log=None)
    m, m0 = np.asarray(sim.flex.masses), np.asarray(sys.masses)
    assert abs(m.sum() - m0.sum()) < 1e-9
    for k in range(nmol):
        sl = sys.atom_slice(k)
        if hmr[k] is None:
            assert np.array_equal(m[sl], m0[sl])
        else:
            assert np.allclose(m[sl], [m0[sl][0] - 2 * (4.0 - m0[sl][1]), 4.0, 4.0])
    assert np.allclose(sim.ff.masses, m) and np.allclose(sim.flex.mass[:, 0], m)
    atoms = np.asarray(sim.constraints.atoms)
    real = atoms < sys.n                                              # (padding: a dummy atom)
    assert np.allclose(np.asarray(sim.constraints.invm)[real], 1.0 / m[atoms[real]])
    e0 = sim.observables()["etot"]
    dev = 0.0
    for _ in range(5):
        sim._advance(40)
        o = sim.observables()
        dev = max(dev, abs(o["etot"] - e0))
        assert o["shake_err"] < 1e-9
    assert dev < 1e-3 * o["ekin"], (dev, o["ekin"])
