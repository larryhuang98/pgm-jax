"""Build pGM Molecules from parameter files: Amber prmtops, JSON caches and py_resp fits.

Contents:

  * `read_prmtop_pgm` / `read_prmtop_molecules` - molecules straight from an Amber pGM prmtop
    (POL_GAUSS_* sections, LJ tables, bonds, masses, extra points), e.g. pGM3P-25 water from
    rayl_512_v2.prmtop; `share_identical` merges identical residues into one template;
  * `save_molecule` / `load_molecule` (`molecule_to_dict` / `molecule_from_dict`) - JSON cache
    under data/params/;
  * `molecule_from_pyresp` - a py_resp (ipol=5, pGM-perm) fit: charges and covalent dipoles
    from the .chg file (`read_pyresp_chg`), polarizabilities and radii from the pGM-pol table
    (`read_pol_table`; evoff's scripts/param_s66.py runs the whole chain: Psi4 -> antechamber
    -> py_resp);
  * `bonds_from_geometry`, `bond_graph` / `map_atoms` - atom correspondence between two
    geometries of one molecule (graph isomorphism, networkx), so one parameter file serves every
    geometry; `reorder` applies the mapping.

Units: Molecule values are in nm / e / e nm / nm^3 / sqrt(kJ/mol); prmtop units are Angstrom / e
(pGM sections; CHARGE is e x 18.2223) / e Angstrom / Angstrom^3 / kcal/mol; py_resp files are
in atomic units (bohr).  Functions and arguments in Angstrom end in `_A`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .paths import resource
from .prmtop import Prmtop
from .system import Molecule
from .units import ANG_NM, BOHR_NM, KCAL

if TYPE_CHECKING:
    import networkx

    from .md.vsites import VirtualSite

Z2EL = {
    1: "H",
    3: "Li",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    11: "Na",
    15: "P",
    16: "S",
    17: "Cl",
    19: "K",
    35: "Br",
    37: "Rb",
    53: "I",
    55: "Cs",
}


AMBER_CHARGE = 18.2223  # prmtop CHARGE unit: e -> sqrt(kcal A / mol)


def prmtop_extra_points(s: Prmtop) -> dict[int, VirtualSite]:
    """Return the Amber extra points of a parsed prmtop as {atom: VirtualSite} (global 0-based indices).

    The frames follow Amber's rules from the bond graph and bond lengths (md/vsites.py
    amber_extra_points).

    Parameters
    ----------
    s : Prmtop
        Parsed prmtop.

    Returns
    -------
    dict of int to VirtualSite
        Extra points (empty if there are none).

    Raises
    ------
    NotImplementedError
        If the prmtop has VIRTUAL_SITE_FRAMES (pmemd custom frames are not read).
    ValueError
        If an extra point has a nonzero mass.
    """
    from .md.vsites import amber_extra_points

    if "VIRTUAL_SITE_FRAMES" in s:
        raise NotImplementedError(
            "prmtop has VIRTUAL_SITE_FRAMES (pmemd custom extra-point frames): not supported; "
            "define the sites with Molecule.vsites (md/vsites.py)"
        )

    def trip(sec: str) -> np.ndarray:
        return np.array([int(x) for x in s.get(sec, [])], int).reshape(-1, 3)

    bh, bx = trip("BONDS_INC_HYDROGEN"), trip("BONDS_WITHOUT_HYDROGEN")
    req = [float(x) for x in s.get("BOND_EQUIL_VALUE", [])]

    def as_list(b: np.ndarray) -> list[tuple[int, int, int]]:
        return [(i // 3, j // 3, t - 1) for i, j, t in b]

    eps = amber_extra_points(s["AMBER_ATOM_TYPE"], as_list(bh), as_list(bx), req)
    mass = np.array([float(x) for x in s["MASS"]])
    for e in eps:
        if mass[e] != 0.0:
            raise ValueError(f"extra point {e + 1} has mass {mass[e]:g}; Amber extra points are massless")
    return eps


def read_prmtop_pgm(
    path: str, first_residue_only: bool = True, charges: str = "pgm", point_radius: float | None = None
) -> list[Molecule]:
    """Return the molecules (one per residue) of an Amber pGM prmtop.

    Reads pGM multipoles, radii and polarizabilities, covalent dipoles, LJ from the type-pair
    tables (converted to per-type R* and sqrt(eps); NBFIX-style pairs that break Lorentz-Berthelot
    raise), bonds and masses.  Molecules of several residues (proteins; covalent dipoles across
    residues raise) are read by protein.load_amber(prmtop, coords, electrostatics="prmtop").
    Extra points (atom type EP, mass 0) become virtual sites (Molecule.vsites) with Amber's frames
    (md/vsites.py); their element is "EP".

    Parameters
    ----------
    path : str
        Amber prmtop.
    first_residue_only : bool
        Return only the first residue (e.g. one water of a water box).
    charges : {"pgm", "amber"}
        "pgm": the POL_GAUSS_* sections.  "amber": a classical prmtop, point charges CHARGE /
        18.2223 with Gaussian radius `point_radius`, no polarizability, no covalent dipoles (run with
        MDSettings().replace(elec="q")).
    point_radius : float, optional
        Gaussian radius of the point charges for charges="amber" [nm]; None:
        md.vsites.POINT_RADIUS (1e-4 nm).

    Returns
    -------
    list of Molecule
        One per residue in prmtop order (a new object each; see share_identical), named by the
        residue label.

    Raises
    ------
    ValueError
        If the prmtop has no pGM sections (charges="pgm"), `charges` is unknown, a covalent dipole
        or an extra-point frame crosses residues, or the LJ tables are not Lorentz-Berthelot.

    Notes
    -----
    Elements come from ATOMIC_NUMBER, or else from the first letter of the atom name (digits
    removed), which fails for two-letter elements.
    """
    s = Prmtop.read(path)
    names = s["ATOM_NAME"]
    types = s["AMBER_ATOM_TYPE"]
    res_ptr = [int(x) - 1 for x in s["RESIDUE_POINTER"]] + [len(names)]
    res_lab = s["RESIDUE_LABEL"]
    if charges == "pgm":
        if "POL_GAUSS_MONOPOLES_LIST" not in s:
            raise ValueError(f"{path} has no pGM sections (POL_GAUSS_*): pass charges='amber' for its point charges")
        q = np.array([float(x) for x in s["POL_GAUSS_MONOPOLES_LIST"]])
        rad = np.array([float(x) for x in s["POL_GAUSS_RADII_LIST"]])
        alp = np.array([float(x) for x in s["POL_GAUSS_POLARIZABILITY_LIST"]])
        nptr = [int(x) for x in s["POL_GAUSS_COVALENT_POINTERS_LIST"]]
        catm = [int(x) - 1 for x in s["POL_GAUSS_COVALENT_ATOMS_LIST"]]
        cdip = [float(x) for x in s["POL_GAUSS_COVALENT_DIPOLES_LIST"]]
    elif charges == "amber":
        from .md.vsites import POINT_RADIUS

        q = np.array([float(x) for x in s["CHARGE"]]) / AMBER_CHARGE
        rad = np.full(len(names), (POINT_RADIUS if point_radius is None else float(point_radius)) / ANG_NM)
        alp = np.zeros(len(names))
        nptr, catm, cdip = [0] * len(names), [], []
    else:
        raise ValueError("charges: 'pgm' (POL_GAUSS sections) or 'amber' (point charges from CHARGE)")
    start = np.concatenate([[0], np.cumsum(nptr)])  # first covalent dipole of every atom
    mass = np.array([float(x) for x in s["MASS"]])
    rh, se = _prmtop_lj(s)
    bonds = [
        (int(a) // 3, int(b) // 3)
        for sec in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN")
        for a, b in zip(s.get(sec, [])[0::3], s.get(sec, [])[1::3])
    ]
    eps = prmtop_extra_points(s)
    mols = []
    nres = len(res_lab) if not first_residue_only else 1
    for r in range(nres):
        a0, a1 = res_ptr[r], res_ptr[r + 1]
        cov = []
        for i in range(a0, a1):
            for k in range(start[i], start[i + 1]):
                if not a0 <= catm[k] < a1:
                    raise ValueError(
                        f"atom {i + 1} has a covalent dipole to atom {catm[k] + 1} of another residue: "
                        "read multi-residue molecules with protein.load_amber(electrostatics='prmtop')"
                    )
                cov.append((i - a0, catm[k] - a0, cdip[k] * ANG_NM))
        if "ATOMIC_NUMBER" in s:
            el = [Z2EL[int(z)] if a not in eps else "EP" for a, z in zip(range(a0, a1), s["ATOMIC_NUMBER"][a0:a1])]
        else:
            el = [
                re.sub(r"\d+", "", n)[:1].upper() if a not in eps else "EP" for a, n in zip(range(a0, a1), names[a0:a1])
            ]
        bd = [(i - a0, j - a0) for i, j in bonds if a0 <= i < a1 and a0 <= j < a1]
        vs = []
        for e in range(a0, a1):
            if e in eps:
                if not all(a0 <= a < a1 for a in eps[e].atoms):
                    raise ValueError(
                        f"extra point {e + 1}: frame atoms {[a + 1 for a in eps[e].atoms]} outside its residue"
                    )
                vs.append(eps[e].shifted(-a0))
        mols.append(
            Molecule(
                name=res_lab[r],
                elements=el,
                types=types[a0:a1],
                q=q[a0:a1].copy(),
                radius=rad[a0:a1] * ANG_NM,
                alpha=alp[a0:a1] * ANG_NM**3,
                cov=cov,
                lj_rmin_half=rh[a0:a1],
                lj_sqrt_eps=se[a0:a1],
                bonds=bd,
                masses=mass[a0:a1],
                vsites=vs,
            )
        )
    return mols


def share_identical(mols: list[Molecule]) -> list[Molecule]:
    """Return the molecules with identical ones replaced by the first of them.

    Identical means same name, elements, types, charges, radii, polarizabilities, covalent dipoles,
    LJ parameters, bonds and virtual sites (masses, GVDW, quadrupoles, keys and extra arrays are
    not compared), so that the MD engines build one template per kind of molecule.
    """
    seen, out = {}, []
    for m in mols:
        key = (
            m.name,
            tuple(m.elements),
            tuple(m.types),
            m.q.tobytes(),
            m.radius.tobytes(),
            m.alpha.tobytes(),
            tuple(m.cov),
            m.lj_rmin_half.tobytes(),
            m.lj_sqrt_eps.tobytes(),
            tuple(m.bonds),
            tuple(m.vsites),
        )
        out.append(seen.setdefault(key, m))
    return out


def read_prmtop_molecules(path: str, charges: str = "pgm", point_radius: float | None = None) -> list[Molecule]:
    """Every molecule (residue) of a prmtop, identical ones shared (share_identical).

    Parameters
    ----------
    path : str
        Amber prmtop (a pGM prmtop, or a classical one with charges="amber").
    charges, point_radius
        As in read_prmtop_pgm.

    Returns
    -------
    list of Molecule
        One entry per residue, in prmtop order.
    """
    return share_identical(read_prmtop_pgm(path, first_residue_only=False, charges=charges, point_radius=point_radius))


def _prmtop_lj(s) -> tuple[np.ndarray, np.ndarray]:
    """Return per-atom LJ R* [nm] and sqrt(eps) [sqrt(kJ/mol)] from the prmtop's ACOEF/BCOEF tables.

    Amber stores A = eps r_min^12 [kcal/mol A^12] and B = 2 eps r_min^6 [kcal/mol A^6] per type
    pair; the per-type values come from the diagonal (r_min = (2A/B)^(1/6), eps = B^2/(4A); types
    with A or B = 0 get zeros).

    Raises
    ------
    ValueError
        If an off-diagonal pair is not the Lorentz-Berthelot combination of the diagonals (NBFIX).
    """
    ntypes = int(s["POINTERS"][1])
    ti = np.array([int(x) - 1 for x in s["ATOM_TYPE_INDEX"]])
    nbi = np.array([int(x) - 1 for x in s["NONBONDED_PARM_INDEX"]]).reshape(ntypes, ntypes)
    A = np.array([float(x) for x in s["LENNARD_JONES_ACOEF"]])[nbi]
    B = np.array([float(x) for x in s["LENNARD_JONES_BCOEF"]])[nbi]
    a, b = np.diag(A), np.diag(B)
    ok = (a > 0) & (b > 0)
    rmin = np.where(ok, (2 * np.where(ok, a, 1) / np.where(ok, b, 1)) ** (1 / 6), 0.0)  # A
    eps = np.where(ok, b**2 / (4 * np.where(ok, a, 1)), 0.0)  # kcal/mol
    rm = rmin[:, None] / 2 + rmin[None, :] / 2
    ee = np.sqrt(eps[:, None] * eps[None, :])
    A_lb, B_lb = ee * rm**12, 2 * ee * rm**6
    if not (np.allclose(A, A_lb, rtol=1e-6, atol=1e-8) and np.allclose(B, B_lb, rtol=1e-6, atol=1e-8)):
        raise ValueError("prmtop LJ pairs are not Lorentz-Berthelot combinations of the type diagonals (NBFIX?)")
    return (rmin / 2 * ANG_NM)[ti], np.sqrt(eps * KCAL)[ti]


def molecule_to_dict(m: Molecule) -> dict[str, Any]:
    """Return a JSON-serialisable dict of a Molecule (the format of save_molecule).

    Keys carry their unit where it is not e: "radius_nm", "alpha_nm3", "lj_rmin_half_nm"; "cov"
    [e nm], "lj_sqrt_eps" [sqrt(kJ/mol)], "masses" [amu], GVDW and "quad" in library units;
    virtual sites via VirtualSite.to_list.
    """
    return {
        "name": m.name,
        "elements": m.elements,
        "types": m.types,
        "q": m.q.tolist(),
        "radius_nm": m.radius.tolist(),
        "alpha_nm3": m.alpha.tolist(),
        "cov": [[int(i), int(j), float(c)] for i, j, c in m.cov],
        "lj_rmin_half_nm": m.lj_rmin_half.tolist(),
        "lj_sqrt_eps": m.lj_sqrt_eps.tolist(),
        "bonds": [[int(i), int(j)] for i, j in m.bonds],
        "masses": m.masses.tolist(),
        "keys": m.keys,
        "gvdw_sqrt_a": m.gvdw_sqrt_a.tolist(),
        "gvdw_sqrt_c6": m.gvdw_sqrt_c6.tolist(),
        "gvdw_b": m.gvdw_b.tolist(),
        "quad": [[int(i), int(j), int(k), float(t)] for i, j, k, t in m.quad],
        "extra": {k: np.asarray(v).tolist() for k, v in m.extra.items()},
        "vsites": [vs.to_list() for vs in m.vsites],
    }


def molecule_from_dict(d: dict[str, Any]) -> Molecule:
    """Return the Molecule of a molecule_to_dict dict.

    Also reads the older format (no LJ, bonds, masses, keys, GVDW, quadrupoles, extra arrays or
    virtual sites: Molecule defaults are used).
    """
    from .md.vsites import VirtualSite

    return Molecule(
        name=d["name"],
        elements=d["elements"],
        types=d["types"],
        q=np.array(d["q"]),
        radius=np.array(d["radius_nm"]),
        alpha=np.array(d["alpha_nm3"]),
        cov=[(int(i), int(j), float(c)) for i, j, c in d["cov"]],
        lj_rmin_half=d.get("lj_rmin_half_nm"),
        lj_sqrt_eps=d.get("lj_sqrt_eps"),
        bonds=[(int(i), int(j)) for i, j in d.get("bonds", [])],
        masses=d.get("masses"),
        keys=d.get("keys", {}),
        extra={k: np.array(v) for k, v in d.get("extra", {}).items()},
        gvdw_sqrt_a=d.get("gvdw_sqrt_a"),
        gvdw_sqrt_c6=d.get("gvdw_sqrt_c6"),
        gvdw_b=d.get("gvdw_b"),
        quad=[(int(i), int(j), int(k), float(t)) for i, j, k, t in d.get("quad", [])],
        vsites=[VirtualSite.from_list(v) for v in d.get("vsites", [])],
    )


def save_molecule(m: Molecule, path: str) -> None:
    """Write a Molecule as JSON (molecule_to_dict), creating the directory.

    `path` must have a directory part (os.makedirs of "" fails).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(molecule_to_dict(m), open(path, "w"), indent=1)


def load_molecule(path: str) -> Molecule:
    """Read a Molecule written by save_molecule."""
    return molecule_from_dict(json.load(open(path)))


# ------------------------------------------------------------------ py_resp / pGM-pol --
# the pGM-pol table shipped with AmberTools' PyRESP examples
PGM_POL_TABLE = resource("amberhome", "AmberTools/examples/PyRESP/polarizability/pGM-pol-2016-09-01")


def read_pol_table(path: str = PGM_POL_TABLE) -> dict[str, tuple[float, float]]:
    """Return the pGM-pol table {gaff type: (alpha [bohr^3], radius [bohr])}.

    Parsed exactly as py_resp.read_pol_dict: the header line is skipped, type lines are read until
    a line starting with "a", then the "EQ" lines give further types the values of their first
    type.  Types are lower-cased.
    """
    tab, lines = {}, open(path).read().splitlines()[1:]
    k = 0
    while lines[k].split()[0] != "a":
        t = lines[k].split()
        tab[t[0].lower()] = (float(t[1]), float(t[2]))
        k += 1
    for ln in lines[k + 1 :]:
        t = ln.split()
        if not t or t[0] != "EQ":
            break
        for x in t[2:]:
            tab[x.lower()] = tab[t[1].lower()]
    return tab


def read_pyresp_chg(path: str) -> dict[str, Any]:
    """Parse a py_resp .chg file (atomic units).

    Returns
    -------
    dict
        "crd" (n, 3) coordinates [bohr]; "q" (n,) charges [e]; "Z" atomic numbers; "cov" list of
        (i, j, c) covalent dipoles, 0-based, c [e bohr]; and, if present, "p_global" and
        "mu_global" (n, 3) permanent and induced dipoles [e bohr].
    """
    sec, cur = {}, None
    for ln in open(path):
        if ln.startswith("%FLAG"):
            cur = ln[6:].split(":")[0].strip()
            sec[cur] = []
        elif cur and ln.strip() and not ln.split()[0].isalpha() and ln.split()[0] not in ("atm.no", "dip.no"):
            sec[cur].append(ln.split())
    out = {
        "crd": np.array([[float(x) for x in r[1:4]] for r in sec["ATOM CRD"]]),
        "q": np.array([float(r[3]) for r in sec["ATOM CHRG"]]),
        "Z": [int(r[1]) for r in sec["ATOM CHRG"]],
        "cov": [(int(r[1]) - 1, int(r[2]) - 1, float(r[4])) for r in sec.get("PERM DIP LOCAL", [])],
    }
    for k, key in (("PERM DIP GLOBAL", "p_global"), ("IND DIP GLOBAL", "mu_global")):
        if k in sec:
            out[key] = np.array([[float(x) for x in r[1:4]] for r in sec[k]])
    return out


def molecule_from_pyresp(
    name: str,
    elements: list[str],
    types: list[str],
    chg_path: str,
    table: dict[str, tuple[float, float]] | None = None,
    n_atoms: int | None = None,
) -> Molecule:
    """Return a pGM Molecule from a py_resp fit.

    The covalent dipole convention is the same as ours: p_i = sum_k c_k unit(r_ref(k) - r_i).
    Bonds come from the fit geometry; no LJ (zeros).

    Parameters
    ----------
    name : str
        Molecule name.
    elements : list of str
        Element symbols in the order of the .chg file.
    types : list of str
        GAFF atom types (keys of the pGM-pol table).
    chg_path : str
        py_resp .chg file.
    table : dict, optional
        pGM-pol table (read_pol_table); None reads PGM_POL_TABLE.
    n_atoms : int, optional
        Multi-conformer fits list every conformer; keep the first `n_atoms` atoms (the conformers
        are equivalenced).  None keeps all.

    Returns
    -------
    Molecule
        Charges [e], covalent dipoles [e nm], radii [nm] and polarizabilities [nm^3] (BOHR_NM,
        CODATA 2014), bonds.
    """
    table = table or read_pol_table()
    c = read_pyresp_chg(chg_path)
    if n_atoms is not None and len(c["q"]) > n_atoms:
        c = {
            "crd": c["crd"][:n_atoms],
            "q": c["q"][:n_atoms],
            "Z": c["Z"][:n_atoms],
            "cov": [x for x in c["cov"] if x[0] < n_atoms and x[1] < n_atoms],
        }
    al = np.array([table[t.lower()][0] for t in types]) * BOHR_NM**3
    rad = np.array([table[t.lower()][1] for t in types]) * BOHR_NM
    cov = [(i, j, p * BOHR_NM) for i, j, p in c["cov"]]
    bonds = bonds_from_geometry(elements, c["crd"] * BOHR_NM * 10.0)
    return Molecule(
        name=name, elements=list(elements), types=list(types), q=c["q"], radius=rad, alpha=al, cov=cov, bonds=bonds
    )


# ------------------------------------------------------------------- atom mapping --
# covalent radii [Angstrom] for bond detection
COV_RADII_A = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "S": 1.05,
    "Cl": 1.02,
    "P": 1.07,
    "Br": 1.20,
    "I": 1.39,
}


def bonds_from_geometry(elements: Sequence[str], xyz_A: np.ndarray, scale: float = 1.2) -> list[tuple[int, int]]:
    """Return the bonds (i < j) whose length is below `scale` x the sum of covalent radii.

    Parameters
    ----------
    elements : sequence of str
        Element symbols (keys of COV_RADII_A).
    xyz_A : np.ndarray (n, 3)
        Coordinates [Angstrom].
    scale : float
        Tolerance factor on the sum of covalent radii (dimensionless).

    Returns
    -------
    list of (int, int)
    """
    x = np.asarray(xyz_A)
    r = np.array([COV_RADII_A[e] for e in elements])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    i, j = np.nonzero(np.triu(d < scale * (r[:, None] + r[None, :]), k=1))
    return [(int(a), int(b)) for a, b in zip(i, j)]


def bond_graph(elements: Sequence[str], xyz_A: np.ndarray, scale: float = 1.2) -> networkx.Graph:
    """Return the networkx bond graph of a geometry (nodes carry the element as "el").

    Arguments as in bonds_from_geometry (coordinates in Angstrom).  Needs networkx.
    """
    import networkx as nx

    g = nx.Graph()
    for k, e in enumerate(elements):
        g.add_node(k, el=e)
    g.add_edges_from(bonds_from_geometry(elements, xyz_A, scale))
    return g


def map_atoms(ref_el: Sequence[str], ref_xyz_A: np.ndarray, el: Sequence[str], xyz_A: np.ndarray) -> np.ndarray:
    """Return perm with perm[k] = index in the second geometry of reference atom k.

    The mapping is a bond-graph isomorphism that preserves elements; among isomorphisms, the first
    one found.  Parameters are symmetric under automorphisms (checked in evoff's
    scripts/param_s66.py check), so the choice does not matter.  Coordinates in Angstrom; needs
    networkx.

    Raises
    ------
    ValueError
        If the bond graphs are not isomorphic.
    """
    from networkx.algorithms import isomorphism as iso

    g0, g1 = bond_graph(ref_el, ref_xyz_A), bond_graph(el, xyz_A)
    gm = iso.GraphMatcher(g0, g1, node_match=lambda a, b: a["el"] == b["el"])
    m = next(gm.isomorphisms_iter(), None)
    if m is None:
        raise ValueError("bond graphs are not isomorphic")
    return np.array([m[k] for k in range(len(ref_el))])


def reorder(m: Molecule, perm: np.ndarray) -> Molecule:
    """Return the Molecule with atoms in the order of another geometry: new atom perm[k] = old atom k.

    Parameters
    ----------
    m : Molecule
        Molecule in the reference order.
    perm : np.ndarray (n,) int
        Mapping from map_atoms.

    Returns
    -------
    Molecule
        Charges, radii, polarizabilities, LJ, masses, keys and extra arrays permuted, covalent
        dipoles and bonds renumbered.  GVDW parameters and quadrupole terms are not carried over
        (Molecule defaults).

    Raises
    ------
    NotImplementedError
        If the molecule has virtual sites.
    """
    n = m.n
    inv = np.empty(n, dtype=int)
    inv[perm] = np.arange(n)
    keys = {qn: (ks if qn == "cov" else [ks[k] for k in inv]) for qn, ks in m.keys.items()}
    if m.vsites:
        raise NotImplementedError(f"{m.name}: reorder of a molecule with virtual sites")
    return Molecule(
        name=m.name,
        elements=[m.elements[k] for k in inv],
        types=[m.types[k] for k in inv],
        q=m.q[inv],
        radius=m.radius[inv],
        alpha=m.alpha[inv],
        cov=[(int(perm[i]), int(perm[j]), c) for i, j, c in m.cov],
        lj_rmin_half=m.lj_rmin_half[inv],
        lj_sqrt_eps=m.lj_sqrt_eps[inv],
        bonds=[(int(perm[i]), int(perm[j])) for i, j in m.bonds],
        masses=m.masses[inv],
        keys=keys,
        extra={k: np.asarray(v)[inv] for k, v in m.extra.items()},
    )
