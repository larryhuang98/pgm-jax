"""pgm_jax: the polarizable Gaussian multipole (pGM) model with Lennard-Jones, in JAX.

Fixed functional form, differentiable everything: energies, forces, induced dipoles,
polarizabilities and virials are JAX functions of the coordinates, of the parameters (tied
tables, see system.py) and, for periodic systems, of the box.

Gaussian charges + covalent permanent dipoles + induced Gaussian dipoles, all pairs, no masking;
LJ between molecules; gas phase (Model) and periodic (PeriodicModel, Ewald).  Validated against
Amber sander / pmemd-pgm (scripts/validate_amber.py).

Units: nm, e, e nm, nm^3, kJ/mol (see units.py).  Call
    jax.config.update("jax_enable_x64", True)
before use; everything is validated in float64.
"""

from .channels import ElecChannel, elec_decomposition, molecular_polarizability, perm_dipoles
from .ewald import PeriodicPGM, neighbor_list
from .lj import LJChannel, PeriodicLJ
from .md.box import box_from_cell
from .model import Model
from .param import load_molecule, read_prmtop_pgm, save_molecule
from .periodic import PeriodicModel, pressure_bar, strain_derivative
from .system import Molecule, ParamTable, System
from .vdw import GVDWChannel, PeriodicGVDW, set_gvdw

__all__ = [
    "ElecChannel",
    "elec_decomposition",
    "molecular_polarizability",
    "perm_dipoles",
    "PeriodicPGM",
    "box_from_cell",
    "neighbor_list",
    "LJChannel",
    "PeriodicLJ",
    "Model",
    "load_molecule",
    "read_prmtop_pgm",
    "save_molecule",
    "PeriodicModel",
    "pressure_bar",
    "strain_derivative",
    "Molecule",
    "ParamTable",
    "System",
    "GVDWChannel",
    "PeriodicGVDW",
    "set_gvdw",
]
