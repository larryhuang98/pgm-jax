"""Polarizable Gaussian multipole (pGM) force field with Lennard-Jones or GVDW, in JAX.

Fixed functional form, differentiable everything: energies, forces, induced dipoles,
polarizabilities and virials are JAX functions of the coordinates, of the parameters (tied
tables, see system.py) and, for periodic systems, of the box.

Gaussian charges + covalent permanent dipoles + induced Gaussian dipoles, all pairs, no masking;
LJ (or GVDW) between molecules; gas phase (Model) and periodic (PeriodicModel, Ewald).  Validated
against Amber sander / pmemd-pgm (scripts/validate_amber.py).

Contents (re-exported here): the topology and parameter tables (Molecule, ParamTable, System in
system.py), the energy channels (ElecChannel, LJChannel, GVDWChannel, and their periodic
counterparts PeriodicPGM, PeriodicLJ, PeriodicGVDW), the models that sum them (Model,
PeriodicModel), analysis helpers (elec_decomposition, molecular_polarizability, perm_dipoles,
pressure_bar, strain_derivative, box_from_cell, neighbor_list) and molecule I/O (load_molecule,
save_molecule, read_prmtop_pgm, set_gvdw).  Molecular dynamics lives in pgm_jax.md, parameter
fitting in pgm_jax.fit.

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
