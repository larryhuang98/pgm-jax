"""Proteins from Amber topologies: tleap builds the system (residues, hydrogens, termini,
solvent, ions) and this module turns it into pgm_jax molecules.

    asys = load_amber("protein.prmtop", "protein.inpcrd", electrostatics=ResidueLibrary.load("pgm_residues.json"))
    prot = asys.molecules[0]                        # kind "protein": .molecule (pGM), .spec (bonded model input)
    tpl = amber_template(prot, "protein.prmtop")    # Amber-form bonded terms + CMAP from the prmtop (ff19SB)
    # or: FlexibleTemplate.from_network(net, P, prot.spec) with a trained neural bonded model
    sim = FlexibleSimulation(asys.system(), asys.templates({0: tpl}), asys.positions, asys.box,
                             MDSettings(), dt=0.002, constraints="h-bonds", hmr=3.024)

Molecules are the connected components of the bond graph; kind "water" (a 3-atom residue with a
water name), "ion" (one atom), "protein" (anything with a peptide backbone) or "other".  pGM
electrostatics come from a ResidueLibrary, from the prmtop itself when it is a pGM prmtop
(POL_GAUSS_* sections; electrostatics="prmtop"), or from `ResidueLibrary.placeholder(prmtop)`.
Water can be replaced by a given pGM water model (`water=`: Molecule, atoms in the prmtop's order).
Units nm, e."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..bonded.model import MolSpec
from ..md.io import box_from_cell, read_coordinates
from ..param import _prmtop_lj, _prmtop_sections
from ..prmtop import Prmtop
from ..system import Molecule, System
from .library import ResidueLibrary
from .residues import IONS, WATER, bond_order, residue_key

_EL = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg", 15: "P", 16: "S", 17: "Cl", 19: "K",
       20: "Ca", 30: "Zn", 35: "Br", 37: "Rb", 53: "I", 55: "Cs"}


@dataclass
class LoadedMolecule:
    kind: str                       # "protein" | "water" | "ion" | "other"
    atoms: np.ndarray               # indices in the prmtop
    molecule: Molecule              # pGM electrostatics + van der Waals, local atom order
    spec: MolSpec | None            # input of the bonded models (flexible molecules)
    atom_names: list
    residue_names: list             # per atom
    residue_index: np.ndarray       # per atom (prmtop residue numbering)

    @property
    def n(self) -> int:
        return len(self.atoms)


@dataclass
class AmberSystem:
    molecules: list
    positions: np.ndarray           # (N, 3) nm, the prmtop's atom order
    box: np.ndarray | None          # (3, 3) nm
    prmtop: str
    order: np.ndarray = field(default=None)   # prmtop atom index of every atom of system() (molecules concatenated)

    def system(self) -> System:
        return System([m.molecule for m in self.molecules])

    def system_positions(self) -> np.ndarray:
        return self.positions[self.order]

    def templates(self, flexible: dict | None = None) -> list:
        """One MD template per molecule: RigidTemplate for water and ions (geometry of the first
        instance), `flexible[k]` for the others (FlexibleTemplate)."""
        from ..md.flexible import RigidTemplate
        flexible = flexible or {}
        out, rigid = [], {}
        for k, m in enumerate(self.molecules):
            if k in flexible:
                out.append(flexible[k])
            elif m.kind in ("water", "ion"):
                key = (m.kind, tuple(m.atom_names))
                if key not in rigid:
                    rigid[key] = RigidTemplate(m.molecule, self.positions[m.atoms], name=m.residue_names[0])
                out.append(rigid[key])
            else:
                raise ValueError(f"molecule {k} ({m.kind}, {m.n} atoms) needs a flexible template")
        return out


def _components(n, bonds):
    nbr = [[] for _ in range(n)]
    for i, j in bonds:
        nbr[i].append(j); nbr[j].append(i)
    comp = np.full(n, -1)
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        comp[s] = c
        stack = [s]
        while stack:
            u = stack.pop()
            for v in nbr[u]:
                if comp[v] < 0:
                    comp[v] = c
                    stack.append(v)
        c += 1
    return comp, c


def load_amber(prmtop: str, inpcrd: str, electrostatics="placeholder", water: Molecule | None = None) -> AmberSystem:
    """Molecules of an Amber topology with pGM electrostatics (ResidueLibrary, "prmtop" for a pGM
    prmtop, or "placeholder") and the prmtop's Lennard-Jones parameters."""
    pt = Prmtop.read(prmtop)
    s = _prmtop_sections(prmtop)
    names = pt.get("ATOM_NAME")
    n = len(names)
    Z = pt.get("ATOMIC_NUMBER")
    el = [_EL[int(z)] for z in Z]
    types = pt.get("AMBER_ATOM_TYPE")
    mass = pt.get("MASS")
    labels = pt.get("RESIDUE_LABEL")
    ptr = list(pt.get("RESIDUE_POINTER") - 1) + [n]
    resi = np.zeros(n, int)
    for r in range(len(labels)):
        resi[ptr[r]:ptr[r + 1]] = r
    resn = [labels[r] for r in resi]
    B = np.concatenate([pt.get("BONDS_INC_HYDROGEN"), pt.get("BONDS_WITHOUT_HYDROGEN")]).reshape(-1, 3)[:, :2] // 3
    bonds = [tuple(sorted((int(a), int(b)))) for a, b in B]
    rh, se = _prmtop_lj(s)
    xyz, _, box = read_coordinates(inpcrd)
    pos = xyz * 0.1
    H = box_from_cell(*box) * 0.1 if box is not None else None
    # electrostatics per atom
    if isinstance(electrostatics, str) and electrostatics == "placeholder":
        electrostatics = ResidueLibrary.placeholder(prmtop)
    cov_global = []
    if isinstance(electrostatics, str) and electrostatics == "prmtop":
        q = np.asarray(pt.get("POL_GAUSS_MONOPOLES_LIST"), float)
        rad = np.asarray(pt.get("POL_GAUSS_RADII_LIST"), float) * 0.1
        alp = np.asarray(pt.get("POL_GAUSS_POLARIZABILITY_LIST"), float) * 1e-3
        nptr = pt.get("POL_GAUSS_COVALENT_POINTERS_LIST").astype(int)
        catm = pt.get("POL_GAUSS_COVALENT_ATOMS_LIST").astype(int) - 1
        cdip = np.asarray(pt.get("POL_GAUSS_COVALENT_DIPOLES_LIST"), float) * 0.1
        start = np.concatenate([[0], np.cumsum(nptr)])
        cov_global = [(i, int(catm[k]), float(cdip[k])) for i in range(n) for k in range(start[i], start[i + 1])]
    elif isinstance(electrostatics, ResidueLibrary):
        lib = electrostatics
        q, alp, rad = np.zeros(n), np.zeros(n), np.zeros(n)
        rkey = [residue_key(labels[r], names[ptr[r]:ptr[r + 1]]) for r in range(len(labels))]
        for a in range(n):
            d = lib.atom(rkey[resi[a]], names[a])
            q[a], alp[a], rad[a] = d["q"], d["alpha_nm3"], d["radius_nm"]
        nbr = [[] for _ in range(n)]
        for i, j in bonds:
            nbr[i].append(j); nbr[j].append(i)
        for r in range(len(labels)):
            atoms_r = {names[a]: a for a in range(ptr[r], ptr[r + 1])}
            for an, pn, c in lib.cov(rkey[r]):
                i = atoms_r.get(an)
                if i is None:
                    raise KeyError(f"residue {labels[r]} {r + 1} has no atom {an}")
                if pn[0] in "+-":
                    near = {v for u in range(ptr[r], ptr[r + 1]) for v in nbr[u] if resi[v] != r}
                    near |= {w for v in list(near) for w in nbr[v]}
                    cand = [v for v in near if names[v] == pn[1:] and ((resi[v] < r) if pn[0] == "-" else (resi[v] > r))]
                    if not cand:
                        continue                                   # chain end: the partner does not exist
                    j = cand[0]
                else:
                    j = atoms_r[pn]
                cov_global.append((i, j, float(c)))
    else:
        raise ValueError("electrostatics: a ResidueLibrary, 'prmtop' or 'placeholder'")
    # molecules
    comp, nc = _components(n, bonds)
    mols, order, shared = [], [], {}
    for c in range(nc):
        atoms = np.nonzero(comp == c)[0]
        loc = {int(a): k for k, a in enumerate(atoms)}
        lbonds = [(loc[i], loc[j]) for i, j in bonds if i in loc]
        rn = [resn[a] for a in atoms]
        if len(atoms) == 1 and (rn[0] in IONS or el[atoms[0]] not in ("H", "C", "N", "O", "S")):
            kind = "ion"
        elif len(atoms) == 3 and rn[0] in WATER:
            kind = "water"
        else:
            kind = "other"
        ri = resi[atoms]
        key = (kind, tuple(rn), tuple(names[a] for a in atoms), tuple(np.round(q[atoms], 8)))
        if kind == "water" and water is not None:
            mol = water
        elif kind in ("water", "ion") and key in shared:
            mol = shared[key]                                  # identical solvent molecules share one Molecule
        else:
            cov = [(loc[i], loc[j], cc) for i, j, cc in cov_global if i in loc]
            mol = Molecule(name=rn[0] if kind in ("water", "ion") else f"mol{c}", elements=[el[a] for a in atoms],
                           types=[types[a] for a in atoms], q=q[atoms], radius=rad[atoms], alpha=alp[atoms], cov=cov,
                           lj_rmin_half=rh[atoms], lj_sqrt_eps=se[atoms], bonds=lbonds, masses=mass[atoms])
            if kind in ("water", "ion"):
                shared[key] = mol
        spec = None
        if kind == "other":
            oxt = {r for r in set(ri) if any(names[a] == "OXT" for a in atoms if resi[a] == r)}
            orders = [bond_order(resn[atoms[i]], names[atoms[i]], resn[atoms[j]], names[atoms[j]],
                                 ri[i] == ri[j], terminal_carboxylate=ri[i] in oxt) for i, j in lbonds]
            spec = MolSpec(mol.name, [el[a] for a in atoms], lbonds, orders, int(round(float(np.sum(mol.q)))),
                           pos[atoms], mol, atom_names=[names[a] for a in atoms], residue_names=rn)
            from ..bonded.topology import build_topology
            spec.top = build_topology(spec.elements, spec.bonds, (spec.bonds, spec.bond_orders), spec.ref_xyz * 10.0)
            if len(spec.top.cmaps) or len(spec.top.backbone):
                kind = "protein"
        mols.append(LoadedMolecule(kind, atoms, mol, spec, [names[a] for a in atoms], rn, ri))
        order.append(atoms)
    return AmberSystem(mols, pos, H, prmtop, order=np.concatenate(order))


def amber_template(loaded: LoadedMolecule, prmtop: str, families=None, lj14_scale: float = 0.5, **settings):
    """FlexibleTemplate of a protein (or any molecule) with the Amber-form bonded terms and CMAP
    initialised from the prmtop (e.g. ff19SB): the classical bonded model on top of pGM, the
    starting point for refitting."""
    from ..bonded import terms as T
    from ..bonded.amber import init_from_prmtop, with_amber_impropers
    from ..bonded.model import BondedSettings, BondedTerms
    from ..md.flexible import FlexibleTemplate
    spec = loaded.spec
    offset = int(loaded.atoms[0])
    if not np.array_equal(loaded.atoms, np.arange(offset, offset + loaded.n)):
        raise ValueError("the molecule's atoms are not contiguous in the prmtop")
    with_amber_impropers(spec, prmtop, offset=offset)
    terms = BondedTerms([spec], BondedSettings(families=tuple(families or T.PROTEIN), lj14_scale=lj14_scale, **settings))
    P = init_from_prmtop(terms, terms.init_params(), {0: (prmtop, offset)})
    return FlexibleTemplate.from_fit(terms, P)
