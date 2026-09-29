"""Store pGM parameters of residues by residue and atom name, to assemble the pGM molecule of any protein.

Contents: ResidueLibrary (the library, JSON I/O, placeholder and from_fits builders) and FORMAT.
Residue keys are Amber's names with the terminal
variants of its terminal libraries (NMET, CGLY: residues.residue_key).

    {"format": "pgm_jax residue library 1",
     "residues": {"ALA": {"atoms": {"N": {"q": -0.41, "alpha_nm3": 1.1e-3, "radius_nm": 0.051}, ...},
                          "cov":   [["N", "H", 0.0012], ["N", "CA", -0.004], ["N", "-C", 0.002],
                                    ["C", "+N", 0.003], ...]}, ...}}

Covalent-dipole partners are atom names of the same residue, "-X" (atom X of the preceding residue
bonded to this one: the peptide C of residue i-1) or "+X" (the following residue); c in e nm.

Sources:

  `ResidueLibrary.placeholder(prmtop)`  Amber charges + pGM polarizabilities and radii by atom type,
      no covalent dipoles: runs the whole pipeline before a pGM protein library exists;
  `ResidueLibrary.from_fits(...)`       residue parameters taken from fitted fragments (capped
      dipeptides fitted with py_resp / pgm_jax), averaged over the instances of each residue.
Residue names are Amber's (terminal variants NALA, CALA, ...; HID / HIE / HIP, ASH, GLH, LYN, CYX).

Units: e, nm, nm^3, e nm (the pGM-pol table is converted from bohr with CODATA 2018, units.py).
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np

from ..param import read_pol_table
from ..prmtop import Prmtop
from ..units import BOHR_NM_CODATA2018
from .residues import residue_key

FORMAT = "pgm_jax residue library 1"
# pGM-pol table types used for atoms whose own type is not in the table, by element
_GENERIC = {"H": "hc", "C": "c3", "N": "n", "O": "o", "S": "s"}
# rough monatomic-ion values (bohr^3, bohr) for the placeholder only: free-ion polarizabilities
_ION = {
    "Na": (1.0, 1.0),
    "K": (5.5, 1.3),
    "Li": (0.2, 0.8),
    "Cl": (23.6, 1.8),
    "Br": (32.0, 1.9),
    "Mg": (0.5, 0.9),
    "Ca": (3.2, 1.1),
}
# atomic number -> element (placeholder)
_ZEL = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O", 11: "Na", 12: "Mg", 16: "S", 17: "Cl", 19: "K", 20: "Ca", 35: "Br"}


class ResidueLibrary:
    """pGM parameters by residue key and atom name (module docstring for the JSON layout).

    Attributes
    ----------
    residues : dict
        {residue key: {"atoms": {name: {"q" [e], "alpha_nm3", "radius_nm"}}, "cov": [[atom,
        partner, c [e nm]], ...]}}.
    note : str
        Provenance.
    """

    def __init__(self, residues: dict | None = None, note: str = "") -> None:
        """Build a library from a residues dict (None: empty) and a provenance note."""
        self.residues = residues or {}
        self.note = note

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        """Write the library as JSON ({"format": FORMAT, "note", "residues"})."""
        json.dump({"format": FORMAT, "note": self.note, "residues": self.residues}, open(path, "w"), indent=1)

    @classmethod
    def load(cls, path: str) -> ResidueLibrary:
        """Read a library written by save.

        Raises
        ------
        ValueError
            If the file's "format" is not FORMAT.
        """
        d = json.load(open(path))
        if d.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT!r} file")
        return cls(d["residues"], d.get("note", ""))

    # ------------------------------------------------------------------ lookups
    def atom(self, resname: str, name: str) -> dict:
        """Return the parameters {"q", "alpha_nm3", "radius_nm"} of one atom of a residue.

        Raises
        ------
        KeyError
            If the residue or the atom is not in the library.
        """
        try:
            return self.residues[resname]["atoms"][name]
        except KeyError:
            raise KeyError(f"residue library has no atom {name} of {resname}") from None

    def cov(self, resname: str) -> list:
        """Return the covalent dipoles [[atom, partner, c [e nm]], ...] of a residue (empty if unknown)."""
        return self.residues.get(resname, {}).get("cov", [])

    # ------------------------------------------------------------------ builders
    @classmethod
    def placeholder(cls, prmtop: str, pol_table: dict | None = None) -> ResidueLibrary:
        """Build a placeholder library from an Amber prmtop: Amber charges, pGM-pol polarizabilities and radii.

        Polarizabilities and radii come by atom type (GAFF equivalents, element defaults for unknown
        types; rough free-ion values for monatomic ions), with no covalent dipoles; one entry per residue
        key of the prmtop (first instance).  Amber extra points (type EP: virtual sites) keep their
        charge as a point charge (md/vsites.py POINT_RADIUS) and get no polarizability.

        Parameters
        ----------
        prmtop : str
            Amber prmtop (classical charges in CHARGE).
        pol_table : dict, optional
            pGM-pol table (param.read_pol_table); None: the AmberTools one.

        Returns
        -------
        ResidueLibrary
        """
        from ..md.vsites import AMBER_EP_TYPE, POINT_RADIUS

        pt = Prmtop.read(prmtop)
        tab = pol_table or read_pol_table()
        names, types = pt.get("ATOM_NAME"), pt.get("AMBER_ATOM_TYPE")
        q = pt.get("CHARGE") / 18.2223
        Z = pt.get("ATOMIC_NUMBER")
        ptr = list(pt.get("RESIDUE_POINTER") - 1) + [len(names)]
        res = {}
        for r, lab in enumerate(pt.get("RESIDUE_LABEL")):
            key = residue_key(lab, names[ptr[r] : ptr[r + 1]])
            if key in res:
                continue
            atoms = {}
            for a in range(ptr[r], ptr[r + 1]):
                e = _ZEL.get(int(Z[a]), "C")
                t = types[a].lower()
                if types[a].strip() == AMBER_EP_TYPE:
                    atoms[names[a]] = {"q": float(q[a]), "alpha_nm3": 0.0, "radius_nm": POINT_RADIUS}
                    continue
                if ptr[r + 1] - ptr[r] == 1 and e in _ION:
                    alpha, rad = _ION[e]
                else:
                    alpha, rad = tab[t if t in tab else _GENERIC.get(e, "c3")]
                atoms[names[a]] = {
                    "q": float(q[a]),
                    "alpha_nm3": alpha * BOHR_NM_CODATA2018**3,
                    "radius_nm": rad * BOHR_NM_CODATA2018,
                }
            res[key] = {"atoms": atoms, "cov": []}
        return cls(res, note=f"placeholder from {prmtop}: Amber charges, pGM-pol polarizabilities, no covalent dipoles")

    @classmethod
    def from_fits(cls, fits: Iterable[tuple]) -> ResidueLibrary:
        """Build a library from fitted fragments, averaging each residue over its instances.

        Parameters
        ----------
        fits : iterable of tuple
            (Molecule, atom names, residue key per atom, residue index per atom, residue keys to take)
            per fitted fragment (e.g. capped dipeptides).

        Returns
        -------
        ResidueLibrary
            Every taken residue's atoms get the mean q / alpha / radius over its instances and the
            mean covalent-dipole strength per (atom, partner) name pair; partners in the previous /
            next residue are written "-X" / "+X".  Averaged charges need not sum to an integer.
        """
        acc, cov = {}, {}
        for mol, names, resn, resi, take in fits:
            resi = np.asarray(resi)
            for a in range(mol.n):
                if resn[a] not in take:
                    continue
                d = acc.setdefault(resn[a], {}).setdefault(names[a], [])
                d.append((float(mol.q[a]), float(mol.alpha[a]), float(mol.radius[a])))
            for i, j, c in mol.cov:
                if resn[i] not in take:
                    continue
                if resi[j] == resi[i]:
                    pn = names[j]
                else:
                    pn = ("-" if resi[j] < resi[i] else "+") + names[j]
                cov.setdefault(resn[i], {}).setdefault((names[i], pn), []).append(float(c))
        res = {}
        for rn, atoms in acc.items():
            res[rn] = {
                "atoms": {
                    n: {
                        "q": float(np.mean([v[0] for v in vals])),
                        "alpha_nm3": float(np.mean([v[1] for v in vals])),
                        "radius_nm": float(np.mean([v[2] for v in vals])),
                    }
                    for n, vals in atoms.items()
                },
                "cov": [[a, b, float(np.mean(v))] for (a, b), v in sorted(cov.get(rn, {}).items())],
            }
        return cls(res, note="from fitted fragments")
