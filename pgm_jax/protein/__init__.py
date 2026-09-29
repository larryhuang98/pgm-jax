"""Proteins with pGM and fitted (or neural) bonded terms.

  residues   bond orders and names that Amber topologies do not carry (carbonyls, carboxylates,
             aromatic rings; water and ion names)
  library    ResidueLibrary: pGM parameters by residue and atom name (JSON), placeholder from
             Amber charges, or assembled from fitted fragments
  amber      load_amber: a tleap system (protein, water, ions) as pgm_jax molecules and bonded
             model inputs; amber_template: Amber-form bonded terms + CMAP from the prmtop
  pmemd      write_pgm_prmtop: the engine's model as a pmemd-pgm prmtop (production MD with
             pmemd.pgm.cuda); pmemd_mdin, pmemd_grid: the matching nonbonded settings

The route from a structure to MD:
    pdb4amber / tleap (protein.pdb -> protein.prmtop, protein.inpcrd; solvent, ions)
    asys = load_amber(prmtop, inpcrd, electrostatics=library)          # pGM molecules
    tpl = FlexibleTemplate.from_network(net, P, asys.molecules[0].spec)  # neural bonded terms
          (or amber_template(asys.molecules[0], prmtop): ff19SB-form bonded terms)
    FlexibleSimulation(asys.system(), asys.templates({0: tpl}), asys.system_positions(), asys.box,
                       settings, dt=0.002, constraints="h-bonds", hmr=3.024)
"""

from .amber import AmberSystem, LoadedMolecule, amber_template, load_amber  # noqa: F401
from .library import ResidueLibrary  # noqa: F401
from .pmemd import pmemd_grid, pmemd_mdin, write_pgm_prmtop  # noqa: F401
