"""Where pGM parameters come from.

  * `read_prmtop_pgm`  – molecules straight from an Amber pGM prmtop (POL_GAUSS_* sections),
                         e.g. pGM3P-25 water from rayl_512_v2.prmtop;
  * `save_molecule` / `load_molecule` – JSON cache under data/params/;
  * `molecule_from_pyresp` – a py_resp (ipol=5, pGM-perm) fit: charges and covalent dipoles
                         from the .chg file, polarizabilities and radii from the pGM-pol table
                         (evoff's scripts/param_s66.py runs the whole chain: Psi4 -> antechamber -> py_resp);
  * `bond_graph` / `map_atoms` – atom correspondence between two geometries of one molecule
                         (graph isomorphism), so one parameter file serves every geometry;
                         `reorder` applies the mapping (needs networkx).
Units in Molecule are nm / e / e nm / nm^3; prmtop units are Angstrom / e / e Angstrom / Angstrom^3.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from .system import Molecule

ANG = 0.1
Z2EL = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 15: "P", 16: "S", 17: "Cl", 19: "K", 35: "Br", 37: "Rb", 53: "I", 55: "Cs"}


def _prmtop_sections(path: str) -> dict[str, list[str]]:
    sec, cur, fmt = {}, None, {}
    for line in open(path):
        if line.startswith("%FLAG"):
            cur = line.split()[1]
            sec[cur] = []
        elif line.startswith("%FORMAT"):
            fmt[cur] = re.search(r"\((.*)\)", line).group(1)
        elif line.startswith("%COMMENT") or cur is None:
            continue
        else:
            sec[cur].append(line.rstrip("\n"))
    out = {}
    for k, lines in sec.items():
        f = fmt.get(k, "")
        m = re.match(r"(\d+)([aAiIeEfF])(\d+)", f)
        if not m:
            out[k] = lines
            continue
        width = int(m.group(3))
        toks = []
        for ln in lines:
            for s in range(0, len(ln), width):
                t = ln[s:s + width].strip()
                if t:
                    toks.append(t)
        out[k] = toks
    return out


def read_prmtop_pgm(path: str, first_residue_only: bool = True) -> list[Molecule]:
    s = _prmtop_sections(path)
    names = s["ATOM_NAME"]
    types = s["AMBER_ATOM_TYPE"]
    res_ptr = [int(x) - 1 for x in s["RESIDUE_POINTER"]] + [len(names)]
    res_lab = s["RESIDUE_LABEL"]
    q = np.array([float(x) for x in s["POL_GAUSS_MONOPOLES_LIST"]])
    rad = np.array([float(x) for x in s["POL_GAUSS_RADII_LIST"]])
    alp = np.array([float(x) for x in s["POL_GAUSS_POLARIZABILITY_LIST"]])
    nptr = [int(x) for x in s["POL_GAUSS_COVALENT_POINTERS_LIST"]]
    catm = [int(x) - 1 for x in s["POL_GAUSS_COVALENT_ATOMS_LIST"]]
    cdip = [float(x) for x in s["POL_GAUSS_COVALENT_DIPOLES_LIST"]]
    start = np.concatenate([[0], np.cumsum(nptr)])
    mols = []
    nres = len(res_lab) if not first_residue_only else 1
    for r in range(nres):
        a0, a1 = res_ptr[r], res_ptr[r + 1]
        cov = []
        for i in range(a0, a1):
            for k in range(start[i], start[i + 1]):
                cov.append((i - a0, catm[k] - a0, cdip[k] * ANG))
        if "ATOMIC_NUMBER" in s:
            el = [Z2EL[int(z)] for z in s["ATOMIC_NUMBER"][a0:a1]]
        else:
            el = [re.sub(r"\d+", "", n)[:1].upper() for n in names[a0:a1]]
        mols.append(Molecule(name=res_lab[r], elements=el, types=types[a0:a1], q=q[a0:a1].copy(),
                             radius=rad[a0:a1] * ANG, alpha=alp[a0:a1] * ANG ** 3, cov=cov))
    return mols


def molecule_to_dict(m: Molecule) -> dict:
    return {"name": m.name, "elements": m.elements, "types": m.types, "q": m.q.tolist(), "radius_nm": m.radius.tolist(),
            "alpha_nm3": m.alpha.tolist(), "cov": [[int(i), int(j), float(c)] for i, j, c in m.cov],
            "extra": {k: np.asarray(v).tolist() for k, v in m.extra.items()}}


def molecule_from_dict(d: dict) -> Molecule:
    return Molecule(name=d["name"], elements=d["elements"], types=d["types"], q=np.array(d["q"]),
                    radius=np.array(d["radius_nm"]), alpha=np.array(d["alpha_nm3"]),
                    cov=[(int(i), int(j), float(c)) for i, j, c in d["cov"]], extra={k: np.array(v) for k, v in d.get("extra", {}).items()})


def save_molecule(m: Molecule, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(molecule_to_dict(m), open(path, "w"), indent=1)


def load_molecule(path: str) -> Molecule:
    return molecule_from_dict(json.load(open(path)))


# ------------------------------------------------------------------ py_resp / pGM-pol --
BOHR_NM = 0.052917721067
PGM_POL_TABLE = os.path.expanduser("~/amber25/AmberTools/examples/PyRESP/polarizability/pGM-pol-2016-09-01")


def read_pol_table(path: str = PGM_POL_TABLE) -> dict[str, tuple[float, float]]:
    """{gaff type: (alpha bohr^3, radius bohr)}, EQ lines expanded, exactly as py_resp.read_pol_dict."""
    tab, lines = {}, open(path).read().splitlines()[1:]
    k = 0
    while lines[k].split()[0] != "a":
        t = lines[k].split()
        tab[t[0].lower()] = (float(t[1]), float(t[2]))
        k += 1
    for ln in lines[k + 1:]:
        t = ln.split()
        if not t or t[0] != "EQ":
            break
        for x in t[2:]:
            tab[x.lower()] = tab[t[1].lower()]
    return tab


def read_pyresp_chg(path: str) -> dict:
    """Parse a py_resp .chg file (atomic units)."""
    sec, cur = {}, None
    for ln in open(path):
        if ln.startswith("%FLAG"):
            cur = ln[6:].split(":")[0].strip()
            sec[cur] = []
        elif cur and ln.strip() and not ln.split()[0].isalpha() and ln.split()[0] not in ("atm.no", "dip.no"):
            sec[cur].append(ln.split())
    out = {"crd": np.array([[float(x) for x in r[1:4]] for r in sec["ATOM CRD"]]),
           "q": np.array([float(r[3]) for r in sec["ATOM CHRG"]]),
           "Z": [int(r[1]) for r in sec["ATOM CHRG"]],
           "cov": [(int(r[1]) - 1, int(r[2]) - 1, float(r[4])) for r in sec.get("PERM DIP LOCAL", [])]}
    for k, key in (("PERM DIP GLOBAL", "p_global"), ("IND DIP GLOBAL", "mu_global")):
        if k in sec:
            out[key] = np.array([[float(x) for x in r[1:4]] for r in sec[k]])
    return out


def molecule_from_pyresp(name: str, elements: list[str], types: list[str], chg_path: str,
                         table: dict | None = None) -> Molecule:
    """pGM molecule from a py_resp fit.  The covalent dipole convention is the same as ours:
    p_i = sum_k c_k unit(r_ref(k) - r_i)."""
    table = table or read_pol_table()
    c = read_pyresp_chg(chg_path)
    al = np.array([table[t.lower()][0] for t in types]) * BOHR_NM ** 3
    rad = np.array([table[t.lower()][1] for t in types]) * BOHR_NM
    cov = [(i, j, p * BOHR_NM) for i, j, p in c["cov"]]
    return Molecule(name=name, elements=list(elements), types=list(types), q=c["q"], radius=rad, alpha=al, cov=cov)


# ------------------------------------------------------------------- atom mapping --
COV_RADII_A = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05, "Cl": 1.02, "P": 1.07, "Br": 1.20, "I": 1.39}


def bond_graph(elements, xyz_A, scale: float = 1.2):
    import networkx as nx
    x = np.asarray(xyz_A)
    g = nx.Graph()
    for k, e in enumerate(elements):
        g.add_node(k, el=e)
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if np.linalg.norm(x[i] - x[j]) < scale * (COV_RADII_A[elements[i]] + COV_RADII_A[elements[j]]):
                g.add_edge(i, j)
    return g


def map_atoms(ref_el, ref_xyz_A, el, xyz_A) -> np.ndarray:
    """perm with perm[k] = index in the second geometry of reference atom k (bond-graph isomorphism).
    Among isomorphisms, the first one found; parameters are symmetric under automorphisms
    (checked in evoff's scripts/param_s66.py check), so the choice does not matter."""
    from networkx.algorithms import isomorphism as iso
    g0, g1 = bond_graph(ref_el, ref_xyz_A), bond_graph(el, xyz_A)
    gm = iso.GraphMatcher(g0, g1, node_match=lambda a, b: a["el"] == b["el"])
    m = next(gm.isomorphisms_iter(), None)
    if m is None:
        raise ValueError("bond graphs are not isomorphic")
    return np.array([m[k] for k in range(len(ref_el))])


def reorder(m: Molecule, perm: np.ndarray) -> Molecule:
    """Molecule with atoms in the order of another geometry: new atom perm[k] = old atom k."""
    n = m.n
    inv = np.empty(n, dtype=int)
    inv[perm] = np.arange(n)
    return Molecule(name=m.name, elements=[m.elements[k] for k in inv], types=[m.types[k] for k in inv],
                    q=m.q[inv], radius=m.radius[inv], alpha=m.alpha[inv],
                    cov=[(int(perm[i]), int(perm[j]), c) for i, j, c in m.cov],
                    extra={k: np.asarray(v)[inv] for k, v in m.extra.items()})
