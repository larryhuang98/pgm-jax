"""pgm_jax: the polarizable Gaussian multipole (pGM) model in JAX.

Gaussian charges + covalent permanent dipoles + induced Gaussian dipoles, all pairs, no
masking; gas phase and periodic (Ewald); energies, forces (autodiff), induced dipoles,
molecular polarizabilities, n-body decompositions, batched over geometries.
Validated against Amber sander / pmemd-pgm (scripts/validate_amber.py).

Units: nm, e, e nm, nm^3, kJ/mol (see units.py).  Call
    jax.config.update("jax_enable_x64", True)
before use; everything is validated in float64.
"""
from .channels import ElecChannel, elec_decomposition, molecular_polarizability, perm_dipoles
from .ewald import PeriodicPGM, box_matrix
from .model import Model
from .param import load_molecule, read_prmtop_pgm, save_molecule
from .system import Molecule, System

__all__ = ["ElecChannel", "elec_decomposition", "molecular_polarizability", "perm_dipoles",
           "PeriodicPGM", "box_matrix", "Model", "load_molecule", "read_prmtop_pgm", "save_molecule",
           "Molecule", "System"]
