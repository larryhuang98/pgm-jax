"""Named sets of bonded term families of the bonded study (scripts/bonded/experiments.py, examples)."""

from __future__ import annotations

from .. import terms as T

FAMILY_SETS = {
    "protein": T.PROTEIN,
    "paper": T.PAPER,
    "explore": T.PAPER,
    "amber": T.AMBER,
    "nn": ("nnb",),
    "diag": ("bond_morse", "angle_cos", "torsion", "improper"),
    "diag+p14": ("bond_morse", "angle_cos", "torsion", "improper", "pair14_exp"),
    "diag+ub": ("bond_morse", "angle_cos", "torsion", "improper", "pair13_harm", "pair14_exp"),
    "pair": ("bond_morse", "pair13_harm", "pair14_exp", "improper"),
    "pair+tors": ("bond_morse", "pair13_harm", "pair14_exp", "torsion", "improper"),
    "pair+ang": ("bond_morse", "angle_cos", "pair13_harm", "pair14_exp", "torsion", "improper"),
    "paper+pair14": T.PAPER + ("pair14_exp",),
    "tmod": ("bond_morse", "angle_cos", "bond_bond", "bond_angle", "angle_angle", "torsion_mod", "aat", "improper"),
    "paper+x": T.PAPER + ("bond_angle_x", "angle_angle_x", "angle_cubic"),
    "paper+oop": T.PAPER + ("torsion_oop",),
    "paper+tw": T.PAPER + ("twist",),
    "diag+tw": ("bond_morse", "angle_cos", "torsion", "improper", "twist"),
    "diag+oop": ("bond_morse", "angle_cos", "torsion", "improper", "torsion_oop"),
    "paper-pyr": tuple(f for f in T.PAPER if f != "improper") + ("pyramid",),
    "all": T.PAPER + ("bond_angle_x", "angle_angle_x", "angle_cubic", "pair13_harm", "pair14_exp"),
    "all+tw": T.PAPER + ("bond_angle_x", "angle_angle_x", "angle_cubic", "pair13_harm", "pair14_exp", "twist"),
    "diag+ub+tw": ("bond_morse", "angle_cos", "torsion", "improper", "pair13_harm", "pair14_exp", "twist"),
    # F12: electronic-structure-inspired families
    "diag+conj": ("bond_morse", "angle_cos", "torsion", "improper", "conj"),
    "paper+conj": T.PAPER + ("conj",),
    "diag+hc": ("bond_morse", "angle_cos", "torsion", "improper", "hc_sigma", "hc_lone"),
    "diag+vol": ("bond_morse", "angle_cos", "torsion", "volume"),
    "diag+new": ("bond_morse", "angle_cos", "torsion", "volume", "conj", "hc_sigma", "hc_lone"),
    "hyb": ("bond_morse", "angle_hyb", "torsion", "improper"),
    "hybsc": ("bond_morse", "angle_hybsc", "torsion", "improper"),
    "diag+ovl": ("bond_morse", "angle_cos", "torsion", "improper", "pair13_ovl", "pair14_ovl"),
    "chem": ("bond_morse", "angle_cos", "volume", "conj", "hc_sigma", "hc_lone", "pair14_exp"),
    "chem+hyb": ("bond_morse", "angle_hybsc", "volume", "conj", "hc_sigma", "hc_lone", "pair14_exp"),
    "dist": ("bond_morse", "pair13_tanh", "pair14_tanh", "volume"),
    "dist+chem": ("bond_morse", "pair13_tanh", "pair14_tanh", "volume", "conj", "hc_sigma", "hc_lone"),
}


def families_of(spec: str) -> tuple:
    """A FAMILY_SETS name, or families and set names joined by '+' ("amber+cmap", "paper+twist")."""
    if spec in FAMILY_SETS:
        return tuple(FAMILY_SETS[spec])
    out = []
    for tok in spec.split("+"):
        out += list(FAMILY_SETS[tok]) if tok in FAMILY_SETS else [tok]
    return tuple(dict.fromkeys(out))
