"""Collect the bonded term families: a registry of energy functions of internal coordinates.

Each family has its index set, parameters and tying keys (terms/core.py, `Family`).  Every
family reads a geometry dict G of one frame (bond lengths, angle cosines and angles, dihedrals,
improper dihedrals, topological pair distances) and the deviations from the shared reference
values (db = b - b0, dc = cos - cos th0, dth = th - th0), so couplings and diagonal terms use
one set of reference values, fitted jointly.

Families (F-numbers of the study plan):

    F1 class II [1]_:                    bond_morse, angle_cos, bond_bond, bond_angle,
                                         angle_angle, torsion, torsion_bond, torsion_angle, aat, improper
    F2 topological pair potentials:      pair13_harm (Urey-Bradley), pair13_exp, pair14_exp
    F3 modulated torsion:                torsion_mod
    F4 extended quadratic couplings:     bond_angle_x, angle_angle_x (all pairs sharing an atom),
                                         angle_cubic, bond_harm
    F7 out-of-plane alternatives:        pyramid (360 deg - sum of the three angles), improper
    F8 torsion x out-of-plane coupling:  torsion_oop
    F9 twist of 3-coordinated centres:   twist (Winkler-Dunitz twist angle, lone-pair aware)
    F12 electronic-structure-inspired:   conj (pi-axis conjugation), volume (signed-volume double well),
                                         hc_sigma / hc_lone (sigma->sigma*, n->sigma* hyperconjugation),
                                         angle_hyb / angle_hybsc (Coulson hybrid-orbital angles, fixed /
                                         self-consistent), pair13/14_tanh (distance-only),
                                         pair13/14_ovl (pGM Gaussian-overlap repulsion)
    Amber forms:                         bond_harm, angle_harm, torsion_amber, improper_amber, bond_quartic
    backbone maps:                       cmap, cmap6 (Fourier CMAP)

Modules: core (geometry, Family, REGISTRY), classical (diagonal and Amber forms), class2
(couplings), explore (F2, F7-F12), cmap (backbone phi/psi correction).  Importing the modules
registers their families.  Named sets: AMBER, PAPER, PROTEIN and SETS (by set name).

To add a family: a Family with `index(top, keyf)` -> (arrays, keys), `params` {name: (shape,
init)}, `linear` (names entering the energy linearly) and `energy(G, dev, I, p)`, decorated with
`register`; tests/test_bonded.py checks every registered family against finite differences.

Units: energies kJ/mol, lengths nm, angles rad.

References
----------
.. [1] A. S. Abdullah, Y. Wang, M. F. S. J. Menger, S. Sami, T. Head-Gordon, J. Chem. Theory
   Comput. 21, 11669 (2025). doi:10.1021/acs.jctc.5c01458

See also docs/howto_bonded.md.
"""

from __future__ import annotations

from .class2 import (  # noqa: F401
    AngleAngle,
    AngleAngleTorsion,
    AngleAngleX,
    BondAngle,
    BondAngleX,
    BondBond,
    TorsionAngle,
    TorsionBond,
    TorsionModulated,
)
from .classical import (  # noqa: F401
    AngleCos,
    AngleCubic,
    AngleHarm,
    BondHarm,
    BondMorse,
    BondQuartic,
    Improper,
    ImproperAmber,
    Torsion,
    TorsionAmber,
)
from .cmap import CMAPFourier, CMAPFourier6, cmap_basis, cmap_grid, phi_psi  # noqa: F401

# importing the modules registers their families; the names stay available as terms.<name>
from .core import _N, REGISTRY, Family, _dihedral, _mask_n, _pair_index, geometry, morse_depth, register  # noqa: F401
from .explore import (  # noqa: F401
    AngleHybrid,
    AngleHybridSC,
    Conjugation,
    HyperconjLone,
    HyperconjSigma,
    Pair13Exp,
    Pair13Harm,
    Pair13Ovl,
    Pair13Tanh,
    Pair14Exp,
    Pair14Ovl,
    Pair14Tanh,
    Pyramid,
    TorsionOOP,
    Twist,
    Volume,
    pi_axes,
)

AMBER = ("bond_harm", "angle_harm", "torsion_amber", "improper_amber")  # Amber / GAFF forms
PAPER = (  # the class II set of Abdullah et al. (2025)
    "bond_morse",
    "angle_cos",
    "bond_bond",
    "bond_angle",
    "angle_angle",
    "torsion",
    "torsion_bond",
    "torsion_angle",
    "aat",
    "improper",
)
# the three bonded term sets: Amber forms (for tuning GAFF-like parameters), the explored families
# (class II set of the bonded study; any REGISTRY families can be added), and the fast neural
# bonded terms (bonded/nn/; "nnb" is handled by BondedModel, not by this registry)
# proteins: the Amber forms plus the backbone phi/psi correction map (bonded/terms/cmap.py)
PROTEIN = AMBER + ("cmap",)
SETS = {"amber": AMBER, "explore": PAPER, "nn": ("nnb",), "protein": PROTEIN}
