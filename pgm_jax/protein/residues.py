"""Residue chemistry of proteins that the bond graph of an Amber topology does not carry: bond
orders (carbonyls, carboxylates, amides, guanidinium, aromatic rings), water and ion names.

Bond orders enter the neural bonded model (atom and edge features) and the backbone detection
(carbonyl carbons); they are resonance-averaged where Amber's residue has one protonation state
for several Lewis structures (1.5 for carboxylates, guanidinium and aromatic rings)."""

from __future__ import annotations

WATER = {"WAT", "HOH", "TIP3", "TP3", "SOL", "T3P", "OPC", "SPC", "PGM"}
IONS = {
    "NA",
    "Na+",
    "CL",
    "Cl-",
    "K",
    "K+",
    "MG",
    "Mg2+",
    "CA",
    "Ca2+",
    "ZN",
    "Zn2+",
    "Li+",
    "LI",
    "Rb+",
    "Cs+",
    "F-",
    "Br-",
    "I-",
}

_RING6 = [("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"), ("CZ", "CE2"), ("CE2", "CD2"), ("CD2", "CG")]
_TRP = [
    ("CG", "CD1"),
    ("CD1", "NE1"),
    ("NE1", "CE2"),
    ("CE2", "CD2"),
    ("CD2", "CG"),
    ("CE2", "CZ2"),
    ("CZ2", "CH2"),
    ("CH2", "CZ3"),
    ("CZ3", "CE3"),
    ("CE3", "CD2"),
]
_HIS = [("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")]

# side-chain bond orders by residue name (backbone and termini: backbone_order)
SIDE_CHAIN = {
    "ASP": {("CG", "OD1"): 1.5, ("CG", "OD2"): 1.5},
    "ASH": {("CG", "OD1"): 2.0},
    "GLU": {("CD", "OE1"): 1.5, ("CD", "OE2"): 1.5},
    "GLH": {("CD", "OE1"): 2.0},
    "ASN": {("CG", "OD1"): 2.0},
    "GLN": {("CD", "OE1"): 2.0},
    "ARG": {("CZ", "NH1"): 1.5, ("CZ", "NH2"): 1.5, ("NE", "CZ"): 1.5},
    "PHE": {b: 1.5 for b in _RING6},
    "TYR": {b: 1.5 for b in _RING6},
    "TRP": {b: 1.5 for b in _TRP},
    **{h: {b: 1.5 for b in _HIS} for h in ("HIS", "HID", "HIE", "HIP")},
}


def base_name(resname: str) -> str:
    """Residue name without the terminal prefix of Amber's terminal libraries (NALA -> ALA)."""
    r = resname.strip()
    if len(r) == 4 and r[0] in "NC" and (r[1:] in SIDE_CHAIN or r[1:] in _PLAIN):
        return r[1:]
    return r


_PLAIN = {
    "ALA",
    "GLY",
    "SER",
    "THR",
    "CYS",
    "CYX",
    "CYM",
    "VAL",
    "LEU",
    "ILE",
    "MET",
    "PRO",
    "LYS",
    "LYN",
    "ACE",
    "NME",
    "NHE",
    "HYP",
}


def bond_order(
    res_a: str, name_a: str, res_b: str, name_b: str, same_residue: bool, terminal_carboxylate: bool = False
) -> float:
    """Order of the bond between atoms (residue name, atom name); 1 unless listed.  The backbone
    carbonyl C=O is 2 (1.5 for both C-O bonds of a C-terminal carboxylate)."""
    a, b = name_a.strip(), name_b.strip()
    if not same_residue:
        return 1.0  # peptide C-N, disulfide S-S
    if {a, b} in ({"C", "O"}, {"C", "OXT"}):
        return 1.5 if terminal_carboxylate else 2.0
    table = SIDE_CHAIN.get(base_name(res_a), {})
    return table.get((a, b), table.get((b, a), 1.0))


def residue_key(label: str, atom_names) -> str:
    """Library key of a residue: Amber's terminal library names for termini (NMET, CGLY), which
    prmtops label with the plain name; detected from the atoms (H1/H2/H3 on the N-terminus, OXT
    on the C-terminus)."""
    names = set(n.strip() for n in atom_names)
    lab = label.strip()
    if lab in WATER or lab in IONS or len(lab) == 4:
        return lab
    if "OXT" in names:
        return "C" + lab
    if {"H1", "H2"} <= names and "N" in names:
        return "N" + lab
    return lab
