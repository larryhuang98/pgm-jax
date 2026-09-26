"""Combinations of MD features that are not implemented together are refused with a clear error
(no silent fallback): multiple time stepping with an alchemical region, with virtual sites in the
flexible engine or with charge flux in a fast pair model; an alchemical region with charge flux."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pgm_jax import System  # noqa: E402
from pgm_jax.md.alchemy import Alchemy, alchemical_system  # noqa: E402
from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate, liquid_box  # noqa: E402
from pgm_jax.md.forcefield import MDSettings, ewald_beta_for  # noqa: E402
from pgm_jax.md.mts import MTS  # noqa: E402
from test_alchemy import alch_sim  # noqa: E402
from test_flux import flux_template  # noqa: E402
from test_vsites import _tip4pew_ideal  # noqa: E402

FLUX_SETTINGS = MDSettings(cutoff=0.5, skin=0.05, ewald_beta=6.0, pme_grid=(32, 32, 32), lj_lrc=False,
                           precision="double")


def _flux_box(n=16):
    tpl, _ = flux_template()
    pos, H = liquid_box(tpl, n, 0.55, seed=0, min_dist=0.18)
    return tpl, System([tpl.pgm] * n), pos, H


def test_mts_refuses_an_alchemical_region():
    with pytest.raises(NotImplementedError, match="alchemical"):
        alch_sim(mts=MTS(inner=2, r_short=0.4, buffer=0.1))


def test_mts_refuses_sites_in_the_flexible_engine():
    sys, pos, H = _tip4pew_ideal()
    s = MDSettings(elec="q", cutoff=0.65, skin=0.05, ewald_beta=ewald_beta_for(0.65), pme_spacing=0.06,
                   precision="double")
    tpl = RigidTemplate(sys.molecules[0], pos[:4])
    with pytest.raises(NotImplementedError, match="virtual sites"):
        FlexibleSimulation(sys, [tpl] * sys.nmol, pos, H, s, dt=0.002, log=None, mts=MTS(inner=2, split="bonded"))


def test_charge_flux_with_mts_pairs_and_alchemy_is_refused():
    tpl, sys_, pos, H = _flux_box()
    with pytest.raises(NotImplementedError, match="charge flux"):
        FlexibleSimulation(sys_, [tpl] * sys_.nmol, pos, H, FLUX_SETTINGS, dt=0.001, log=None,
                           mts=MTS(inner=2, split="special"))
    sysA, P = alchemical_system(sys_, 0)
    with pytest.raises(NotImplementedError, match="charge flux"):
        FlexibleSimulation(sysA, [tpl] * sys_.nmol, pos, H, FLUX_SETTINGS, dt=0.001, log=None, params=P,
                           alchemy=Alchemy(sysA, 0))
