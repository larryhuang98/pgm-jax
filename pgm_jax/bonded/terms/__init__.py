"""Bonded term families: a registry of energy functions of internal coordinates, each with its
index set, parameters and tying keys.  All energies kJ/mol, lengths nm, angles rad.

Every family reads a geometry dict G of one frame (bond lengths, angle cosines and angles,
dihedrals, improper dihedrals, topological pair distances) and the deviations from the shared
reference values (db = b - b0, dc = cos - cos th0, dth = th - th0), so couplings and diagonal
terms use one set of reference values, fitted jointly.

Families (F-numbers of the plan):
  F1 class II (Abdullah et al. 2025):  bond_morse, angle_cos, bond_bond, bond_angle,
     angle_angle, torsion, torsion_bond, torsion_angle, aat, improper
  F2 topological pair potentials:      pair13_harm (Urey-Bradley), pair13_exp, pair14_exp
  F4 extended quadratic couplings:     bond_angle_x, angle_angle_x (all pairs sharing an atom),
     angle_cubic, bond_harm
  F7 out-of-plane alternatives:        pyramid (360 deg - sum of the three angles), improper
  F8 torsion x out-of-plane coupling:  torsion_oop
  F9 twist of 3-coordinated centres:   twist (Winkler-Dunitz twist angle, lone-pair aware)
  F12 electronic-structure-inspired:    conj (pi-axis conjugation), volume (signed-volume double well),
     hc_sigma / hc_lone (sigma->sigma*, n->sigma* hyperconjugation), angle_hyb / angle_hybsc
     (Coulson hybrid-orbital angles, fixed / self-consistent), pair13/14_tanh (distance-only),
     pair13/14_ovl (pGM Gaussian-overlap repulsion)
Modules: core (geometry, Family, REGISTRY), classical (diagonal and Amber forms), class2
(couplings), explore (F2, F7-F12), cmap (backbone phi/psi correction).
To add a family: a Family with `index(top)` -> (arrays, keys), `params` {name: (shape, init)},
`linear` (names entering the energy linearly) and `energy(G, dev, I, p)`.
"""
from __future__ import annotations

# importing the modules registers their families; the names stay available as terms.<name>
from .core import REGISTRY, Family, _dihedral, _mask_n, _N, _pair_index, geometry, morse_depth, register  # noqa: F401
from .classical import (AngleCos, AngleCubic, AngleHarm, BondHarm, BondMorse, Improper, ImproperAmber,  # noqa: F401
                        Torsion, TorsionAmber)
from .class2 import (AngleAngle, AngleAngleTorsion, AngleAngleX, BondAngle, BondAngleX, BondBond,  # noqa: F401
                     TorsionAngle, TorsionBond, TorsionModulated)
from .explore import (AngleHybrid, AngleHybridSC, Conjugation, HyperconjLone, HyperconjSigma, Pair13Exp,  # noqa: F401
                      Pair13Harm, Pair13Ovl, Pair13Tanh, Pair14Exp, Pair14Ovl, Pair14Tanh, Pyramid, TorsionOOP,
                      Twist, Volume, pi_axes)
from .cmap import CMAPFourier, CMAPFourier6, cmap_basis, cmap_grid, phi_psi  # noqa: F401

AMBER = ("bond_harm", "angle_harm", "torsion_amber", "improper_amber")
PAPER = ("bond_morse", "angle_cos", "bond_bond", "bond_angle", "angle_angle", "torsion", "torsion_bond",
         "torsion_angle", "aat", "improper")
# the three bonded term sets: Amber forms (for tuning GAFF-like parameters), the explored families
# (class II set of the bonded study; any REGISTRY families can be added), and the fast neural
# bonded terms (bonded/nn.py; "nnb" is handled by BondedModel, not by this registry)
# proteins: the Amber forms plus the backbone phi/psi correction map (bonded/terms/cmap.py)
PROTEIN = AMBER + ("cmap",)
SETS = {"amber": AMBER, "explore": PAPER, "nn": ("nnb",), "protein": PROTEIN}

