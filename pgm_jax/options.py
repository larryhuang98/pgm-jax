"""Model options shared by the gas-phase, periodic, bonded-fitting and MD code.

Contents: the option tables ELEC_LEVELS, VDW_FORMS, GVDW_REP and their checks elec_flags and
check_vdw.

Electrostatics (`elec`), all with Gaussian distributions and every atom pair interacting:

    "q"    Gaussian charges only (a fixed-charge Gaussian model)
    "qp"   + permanent covalent dipoles (pGM without polarization)
    "qi"   charges + induced dipoles
    "qpi"  + permanent and induced dipoles: pGM (default)

`quadrupoles=True` adds permanent Gaussian quadrupoles from the covalent quadrupole basis
(multipole.py): gas phase and bonded fitting only for now, not yet in PME / MD.

Van der Waals (`vdw`): "lj" (Amber form, Lorentz-Berthelot), "gvdw" (Gaussian-density vdW,
vdw.py; `gvdw_rep` "gauss" | "slater"), "none".

See also docs/model_options.md.
"""

ELEC_LEVELS = {"q": (False, False), "qp": (True, False), "qi": (False, True), "qpi": (True, True)}
VDW_FORMS = ("lj", "gvdw", "none")
GVDW_REP = ("gauss", "slater")


def elec_flags(level: str) -> tuple[bool, bool]:
    """Return the (permanent dipoles, induction) switches of an electrostatics level.

    Parameters
    ----------
    level : {"q", "qp", "qi", "qpi"}
        Electrostatics level (see the module docstring).

    Returns
    -------
    tuple of (bool, bool)
        (has permanent covalent dipoles, has induced dipoles).

    Raises
    ------
    ValueError
        If `level` is not a key of ELEC_LEVELS.
    """
    if level not in ELEC_LEVELS:
        raise ValueError(f"unknown electrostatics level {level!r}: {', '.join(ELEC_LEVELS)}")
    return ELEC_LEVELS[level]


def check_vdw(vdw: str, rep: str = "gauss") -> None:
    """Check a van der Waals form and GVDW repulsion name.

    Parameters
    ----------
    vdw : {"lj", "gvdw", "none"}
        Van der Waals form.
    rep : {"gauss", "slater"}
        GVDW repulsion form (checked even when `vdw` is not "gvdw").

    Raises
    ------
    ValueError
        If `vdw` is not in VDW_FORMS or `rep` is not in GVDW_REP.
    """
    if vdw not in VDW_FORMS:
        raise ValueError(f"unknown vdw {vdw!r}: {', '.join(VDW_FORMS)}")
    if rep not in GVDW_REP:
        raise ValueError(f"unknown GVDW repulsion {rep!r}: {', '.join(GVDW_REP)}")
