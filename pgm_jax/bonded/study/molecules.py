"""Molecule set for the bonded-term study (after Abdullah et al., arXiv 2504.14398, Fig. 1).

Each entry: SMILES (explicit charge), total charge and the subset it belongs to:
  A1 flexible neutral, A2 strong Coulombic 1-4 (the paper's worst cases for fixed-charge force
  fields), A3 charged, A4 rigid (rings, double bonds), B alanine dipeptide.
`build(name)` gives elements, bonds and a few low-energy conformers (Angstrom) from RDKit
(ETKDG + MMFF); geometries are refined later at the sampling level.
"""

from __future__ import annotations

MOLECULES = {
    # A1 flexible, neutral
    "ethane": ("CC", 0, "A1"),
    "methanol": ("CO", 0, "A1"),
    "methylamine": ("CN", 0, "A1"),
    "methanethiol": ("CS", 0, "A1"),
    "acetaldehyde": ("CC=O", 0, "A1"),
    "formic_acid": ("OC=O", 0, "A1"),
    "formamide": ("NC=O", 0, "A1"),
    "fluorochloroethane": ("FCCCl", 0, "A1"),
    # A2 strong Coulombic 1-4
    "chloroformic_acid": ("OC(=O)Cl", 0, "A2"),
    "chloromethanol": ("OCCl", 0, "A2"),
    # A3 charged
    "acetate": ("CC(=O)[O-]", -1, "A3"),
    "methylammonium": ("C[NH3+]", 1, "A3"),
    "hydrogen_phosphate": ("OP(=O)([O-])[O-]", -2, "A3"),
    # A4 rigid
    "ethene": ("C=C", 0, "A4"),
    "benzene": ("c1ccccc1", 0, "A4"),
    "pyridine": ("c1ccncc1", 0, "A4"),
    "cyclopentane": ("C1CCCC1", 0, "A4"),
    # B
    "alanine_dipeptide": ("CC(=O)N[C@@H](C)C(=O)NC", 0, "B"),
}


def build(name: str, n_conf: int = 30, keep: int = 4, rms: float = 0.3, seed: int = 7):
    """-> dict(name, smiles, charge, subset, elements, bonds, conformers [list of (n,3) A], mmff_E)."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smi, charge, subset = MOLECULES[name]
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert Chem.GetFormalCharge(mol) == charge, name
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params))
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    energies = [e for _, e in res]
    order = np.argsort(energies)
    kept = []
    heavy = [a.GetIdx() for a in mol.GetAtoms()]
    for i in order:
        cid = cids[int(i)]
        if all(AllChem.GetConformerRMS(mol, cid, cids[int(j)], atomIds=heavy) > rms for j in kept):
            kept.append(int(i))
        if len(kept) == keep:
            break
    confs = [mol.GetConformer(cids[i]).GetPositions().tolist() for i in kept]
    return {
        "name": name,
        "smiles": smi,
        "charge": charge,
        "subset": subset,
        "elements": [a.GetSymbol() for a in mol.GetAtoms()],
        "bonds": [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
        "bond_orders": [b.GetBondTypeAsDouble() for b in mol.GetBonds()],
        "conformers": confs,
        "mmff_E": [float(energies[i]) for i in kept],
    }
