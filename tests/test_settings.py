"""MDSettings in groups (terms, cutoffs, neighbors, pme, induction with iEL) and replace(**flat).

Checked against the flat defaults of pgm_jax up to commit e72c57c (OLD_DEFAULTS): every flat name
has a place in the groups with its old default; replace accepts flat names and whole groups and
returns frozen, hashable settings that compare by value.
"""

import pytest

from pgm_jax.md.forcefield import (
    FLAT_SETTINGS,
    Cutoffs,
    ExtendedLagrangian,
    Induction,
    MDSettings,
    NeighborList,
    Terms,
)

# the defaults of the flat MDSettings of pgm_jax up to commit e72c57c
OLD_DEFAULTS = {
    "cutoff": 0.9,
    "elec_cutoff": None,
    "skin": 0.1,
    "ewald_beta": 4.0,
    "pme_grid": None,
    "pme_spacing": 0.08,
    "pme_order": 6,
    "lj_lrc": True,
    "dipole_tol": 1e-5,
    "max_iter": 50,
    "predictor": "mu4",
    "fused": True,
    "norm_refresh": 1000,
    "local_cut": 0.3,
    "local_niter": 0,
    "peek": 0.65,
    "extrap_order": 3,
    "extrap_steps": 2,
    "elec": "qpi",
    "vdw": "lj",
    "gvdw_rep": "gauss",
    "iel": "none",
    "iel_iter": 1,
    "iel_order": 7,
    "iel_kappa": None,
    "iel_alpha": None,
    "iel_precond": "block",
    "iel_omega": 1.0,
    "iel_shadow": True,
}


def get(settings, path):
    """Return the value at a path of nested fields."""
    for name in path:
        settings = getattr(settings, name)
    return settings


def test_defaults_are_the_old_ones():
    """Every flat setting of the old class has a place in the groups, with the old default.

    The flat names added later are the engines' neighbor_list keyword and the DE exponents of
    the double-exponential van der Waals form (18.17 and 3.65, Paper I of DEGAUSS).
    """
    s = MDSettings()
    added = {"neighbor_list": "auto", "de_alpha": 18.17, "de_beta": 3.65}
    assert set(FLAT_SETTINGS) == set(OLD_DEFAULTS) | set(added)
    for k, v in (OLD_DEFAULTS | added).items():
        assert get(s, FLAT_SETTINGS[k]) == v, k
    assert (s.precision, s.differentiable, s.adjoint_tol) == ("mixed", False, 1e-6)
    assert s.neighbors.mode == "auto" and s.pair_cutoff == 0.9 and s.has_induction and s.perm_dipoles


def test_replace_flat_and_groups():
    """replace() takes flat names and whole groups; results are frozen, hashable, compared by value.

    replace takes flat names and whole groups; the result is frozen, hashable and compares by
    value (compiled functions are cached on it).
    """
    s = MDSettings().replace(dipole_tol=1e-8, cutoff=0.8, skin=0.05, iel="0scf", iel_omega=0.9, neighbor_list="atom")
    assert s.induction.tol == 1e-8 and s.cutoffs.cutoff == 0.8 and s.neighbors == NeighborList(0.05, "atom")
    assert s.induction.iel == ExtendedLagrangian(scheme="0scf", omega=0.9) and s.induction.max_iter == 50
    t = MDSettings(
        cutoffs=Cutoffs(cutoff=0.8),
        neighbors=NeighborList(skin=0.05, mode="atom"),
        induction=Induction(tol=1e-8, iel=ExtendedLagrangian("0scf", omega=0.9)),
    )
    assert s == t and hash(s) == hash(t)
    u = s.replace(terms=Terms(elec="q"), precision="double")
    assert u.terms.elec == "q" and u.precision == "double" and u.induction == s.induction
    assert not u.has_induction
    with pytest.raises(TypeError, match="unknown setting"):
        MDSettings().replace(tolerance=1e-3)
    with pytest.raises(AttributeError):
        s.cutoff  # noqa: B018  (flat attributes are gone: hard break)
