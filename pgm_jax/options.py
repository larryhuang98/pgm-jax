"""Model options shared by the gas-phase, periodic, bonded-fitting and MD code.

Electrostatics (`elec`), all with Gaussian distributions and every atom pair interacting:
  "q"    Gaussian charges only (a fixed-charge Gaussian model)
  "qp"   + permanent covalent dipoles (pGM without polarization)
  "qi"   charges + induced dipoles
  "qpi"  + permanent and induced dipoles: pGM (default)
`quadrupoles=True` adds permanent Gaussian quadrupoles from the covalent quadrupole basis
(multipole.py): gas phase and bonded fitting only for now, not yet in PME / MD.

Van der Waals (`vdw`): "lj" (Amber form, Lorentz-Berthelot), "gvdw" (Gaussian-density vdW,
vdw.py; `gvdw_rep` "gauss" | "slater"), "none".
"""
ELEC_LEVELS = {"q": (False, False), "qp": (True, False), "qi": (False, True), "qpi": (True, True)}
VDW_FORMS = ("lj", "gvdw", "none")
GVDW_REP = ("gauss", "slater")


def elec_flags(level: str) -> tuple[bool, bool]:
    """(permanent dipoles, induction) of an electrostatics level."""
    if level not in ELEC_LEVELS:
        raise ValueError(f"unknown electrostatics level {level!r}: {', '.join(ELEC_LEVELS)}")
    return ELEC_LEVELS[level]


def check_vdw(vdw: str, rep: str = "gauss"):
    if vdw not in VDW_FORMS:
        raise ValueError(f"unknown vdw {vdw!r}: {', '.join(VDW_FORMS)}")
    if rep not in GVDW_REP:
        raise ValueError(f"unknown GVDW repulsion {rep!r}: {', '.join(GVDW_REP)}")
