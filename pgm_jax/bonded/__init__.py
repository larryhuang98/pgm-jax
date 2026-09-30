"""Provide bonded (valence) terms for flexible pGM molecules: topology, term registry, model, fitting.

Modules: topology.py (valence topology from the bond graph), terms/ (the registry of term
families), model.py (BondedSettings, MolSpec, BondedTerms, BondedModel: bonded terms with the
intramolecular pGM model), fit.py (Fitter: fit to QM energies and forces), amber.py (Amber / GAFF
parameters in and out), nn/ (neural bonded terms), study/ (the bonded-term study: molecules,
data, benchmarks).

The bonded terms carry only what the all-pair pGM electrostatics and the Lennard-Jones from 1-5
pairs on do not, so they are fitted on top of that nonbonded model.  See docs/howto_bonded.md.
"""

from __future__ import annotations
