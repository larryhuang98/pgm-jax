"""pmemd-pgm topologies of pgm_jax systems: the tleap prmtop of a system rewritten so that
pmemd.pgm / pmemd.pgm.cuda run the model of the MD engine (pGM electrostatics, Lennard-Jones,
exclusions, 1-4 pairs, bonded terms, masses), for production MD of large systems.

    asys = load_amber("sys.prmtop", "sys.inpcrd", electrostatics=library)        # or "placeholder"
    k = [i for i, m in enumerate(asys.molecules) if m.kind == "protein"][0]
    templates = asys.templates({k: amber_template(asys.molecules[k], "sys.prmtop")})
    write_pgm_prmtop(asys, "sys_pgm.prmtop", templates)      # FlexibleSimulation(asys.system(), templates, ...)
    open("md.in", "w").write(pmemd_mdin(MDSettings(), asys.box, nstlim=500000))

The written file is the input prmtop with the sections below replaced or added; the rest
(residues, names, atom types, box, solvent pointers, GB radii) is kept.  Atom order: the prmtop's.

pGM (pmemd-pgm finds them by name: pGM_multipoles.F90, pGM_induced.F90; Amber units)
  POL_GAUSS_FORCEFIELD              1; its presence requires ipgm = 1 in the mdin, and vice versa
  POL_GAUSS_COVALENT_POINTERS_LIST  number of covalent dipoles of each atom
  POL_GAUSS_COVALENT_ATOMS_LIST     their partners j (1-based), atom after atom
  POL_GAUSS_COVALENT_DIPOLES_LIST   strengths c (e A): p_i += c unit(r_j - r_i), as Molecule.cov
  POL_GAUSS_MONOPOLES_LIST          Gaussian charges q (e)
  POL_GAUSS_RADII_LIST              Gaussian radii R (A); pair exponent 1/sqrt(2 (R_i^2 + R_j^2)) in both codes
  POL_GAUSS_POLARIZABILITY_LIST     alpha (A^3); pmemd-pgm treats alpha <= 1e-6 A^3 as non-polarizable
  CHARGE                            18.2223 q: pmemd-pgm does not use it (no separate 1-4
                                    electrostatics under pGM), analysis tools read it
The values are the engine's, `system.expand(params)` (tied parameter tables, fitted or initial),
not the Molecules' initial values.

Van der Waals (md/topology.py is the engine's definition)
  LJ tables       ATOM_TYPE_INDEX, NONBONDED_PARM_INDEX, LENNARD_JONES_ACOEF / BCOEF: one type per
                  distinct (R*, eps) of the engine, Lorentz-Berthelot pairs, A = eps r_min^12,
                  B = 2 eps r_min^6 (kcal/mol, A); no 10-12 terms.
  exclusions      NUMBER_EXCLUDED_ATOMS / EXCLUDED_ATOMS_LIST: the pairs without regular van der
                  Waals in the engine: every pair of a rigid molecule (water, ions), pairs of a
                  flexible molecule fewer than lj_min_sep bonds apart.  pmemd-pgm keeps excluded
                  pairs in its pGM electrostatics (nb_pairlist.F90), so, as here, they only lose
                  their van der Waals term.
  1-4 pairs       atoms 3 bonds apart (shortest path, as the engine: a pair that is also 1-3 in a
                  five-membered ring is not 1-4) get exactly one proper dihedral with the 1-4 flag
                  (positive third index); every other dihedral has it off.  pmemd-pgm drops the
                  1-4 electrostatics and hard-codes SCNB = 1 (prmtop_dat.F90), so SCNB is written
                  as 1 and lj14_scale != 1 goes into the 1-4 tables pmemd reads for CHARMM
                  topologies: LENNARD_JONES_14_ACOEF / BCOEF = lj14_scale (A, B), FORCE_FIELD_TYPE
                  naming CHARMM, and empty Urey-Bradley and CHARMM-improper sections.  The CHARMM
                  flag has no other effect here (it also sets chngmask = 0, which only matters
                  for extra points).
Bonded terms: the FlexibleTemplates' (bonded/amber.export_bonded: bonds, angles, torsions,
impropers, CMAP tabulated on Amber's 24 x 24 grid), or with templates=None the input's (e.g.
ff19SB with its CMAP grids, which the engine's Fourier maps only approximate: amber_template).
The bonds of rigid molecules get their RigidTemplate's distances (SHAKE / SETTLE lengths: a
template takes the geometry of the first instance, which in tleap's water boxes differs from
TIP3P's 0.9572 / 1.5136 A by up to 3e-4 A).  Masses: optionally repartitioned as
FlexibleSimulation(hmr=...) does (every bond, water included).

What a prmtop cannot carry, measured by scripts/protein/check_pgm_prmtop.py (docs/protein_ff.md):
  * pmemd-pgm's Coulomb constant (Tinker's 332.05382 kcal A/mol) is 2.98e-5 below the engine's
    (CODATA, units.KE): its electrostatic energies and forces are the engine's x KE_AMBER_PGM / KE;
  * pmemd's PME influence function has an extra factor (factor_lambda, pme_recip_dat.F90): 2-5e-7
    of the electrostatic energy at 0.8 A grid spacing and order 6, nothing at 0.4 A and order 8;
  * pmemd interpolates the tabulated CMAP bicubically: Trp-cage's CMAP energy differs by 3e-3
    kcal/mol, backbone forces by up to 0.05 kcal/mol/A (Amber's format fixes 24 x 24 grids);
  * one lj14_scale for all flexible molecules; LJ and full pGM (elec "qpi") only, no GVDW,
    quadrupoles or charge flux (md/flux.py; refused); Amber's number formats (9 significant
    digits, CMAP grids to 1e-5 kcal/mol).
Everything else agrees to pmemd's print precision (1e-4 kcal/mol) and the forces to 2e-5 kcal/mol/A.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from ..bonded.topology import near_pairs
from ..md.constraints import repartition_masses
from ..md.topology import MoleculeRule
from ..prmtop import Prmtop
from ..units import KCAL

AMBER_CHARGE = 18.2223  # e -> Amber's charge unit (sqrt(kcal/mol A))
CHARMM_TAG = "CHARMM-form 1-4 LJ tables only (pgm_jax pGM, lj14_scale {:g})"
CHARMM_SECTIONS = (
    "FORCE_FIELD_TYPE",
    "LENNARD_JONES_14_ACOEF",
    "LENNARD_JONES_14_BCOEF",
    "CHARMM_UREY_BRADLEY_COUNT",
    "CHARMM_UREY_BRADLEY",
    "CHARMM_UREY_BRADLEY_FORCE_CONSTANT",
    "CHARMM_UREY_BRADLEY_EQUIL_VALUE",
    "CHARMM_NUM_IMPROPERS",
    "CHARMM_IMPROPERS",
    "CHARMM_NUM_IMPR_TYPES",
    "CHARMM_IMPROPER_FORCE_CONSTANT",
    "CHARMM_IMPROPER_PHASE",
)
POL_GAUSS = (
    "POL_GAUSS_FORCEFIELD",
    "POL_GAUSS_COVALENT_POINTERS_LIST",
    "POL_GAUSS_COVALENT_ATOMS_LIST",
    "POL_GAUSS_COVALENT_DIPOLES_LIST",
    "POL_GAUSS_MONOPOLES_LIST",
    "POL_GAUSS_RADII_LIST",
    "POL_GAUSS_POLARIZABILITY_LIST",
)


def molecule_rules(asys, templates=None, lj14_scale: float = 0.5, lj_min_sep: int = 4) -> list:
    """The engine's MoleculeRule of every molecule of an AmberSystem: templates[k].md_rule() (as
    FlexibleSimulation), or without templates: water and ions rigid, every other molecule flexible
    with Amber's pairs (lj_min_sep 4, lj14_scale 1/2) and its bonded terms left to the prmtop."""
    if templates is not None:
        if len(templates) != len(asys.molecules):
            raise ValueError("one template per molecule of the AmberSystem")
        return [t.md_rule("none") for t in templates]
    out = []
    for m in asys.molecules:
        if m.kind in ("water", "ion"):
            out.append(MoleculeRule(bonds=[], vdw="none"))
        else:
            out.append(
                MoleculeRule(
                    bonds=list(m.molecule.bonds), vdw="graph", lj_min_sep=int(lj_min_sep), lj14_scale=float(lj14_scale)
                )
            )
    return out


def pair_classes(n: int, rule: MoleculeRule):
    """(excluded pairs, 1-4 pairs) of one molecule, local indices i < j."""
    if rule.vdw == "none":
        return [(i, j) for i in range(n) for j in range(i + 1, n)], []
    if rule.vdw != "graph":
        raise ValueError(f"unknown van der Waals rule {rule.vdw!r}")
    nbr = [[] for _ in range(n)]
    for i, j in rule.bonds:
        nbr[i].append(j)
        nbr[j].append(i)
    near = near_pairs(nbr, max(3, int(rule.lj_min_sep) - 1))
    excl = sorted(p for p, d in near.items() if d < rule.lj_min_sep)
    return excl, sorted(p for p, d in near.items() if d == 3)


def _lj_tables(rh_nm, se):
    """LJ types (1-based, per atom) and Amber tables from per-atom R* (nm) and sqrt(eps) (sqrt(kJ/mol))."""
    types, ti = {}, []
    for key in zip(np.asarray(rh_nm, float).tolist(), np.asarray(se, float).tolist()):
        ti.append(types.setdefault(key, len(types)) + 1)
    nt = len(types)
    R = np.array([k[0] for k in types]) * 10.0  # A
    E = np.array([k[1] for k in types])
    ico = np.zeros((nt, nt), int)
    A, B = [], []
    for j in range(nt):
        for i in range(j + 1):
            eps = E[i] * E[j] / KCAL
            rmin = R[i] + R[j]
            A.append(eps * rmin**12)
            B.append(2.0 * eps * rmin**6)
            ico[i, j] = ico[j, i] = len(A)
    return ti, nt, ico.ravel(), np.array(A), np.array(B)


def _dihedral_rows(pt):
    rows = []
    for sec in ("DIHEDRALS_INC_HYDROGEN", "DIHEDRALS_WITHOUT_HYDROGEN"):
        rows += [tuple(int(x) for x in r) for r in pt.get(sec).reshape(-1, 5)]
    return rows


def _set_14_flags(pt, pairs14: set, isH) -> int:
    """Exactly one proper dihedral with the 1-4 flag per pair in pairs14 (0-based atoms, i < j),
    none for any other pair; returns the number of flagged dihedrals."""
    per = pt.get("DIHEDRAL_PERIODICITY")
    done, rows_h, rows_n = set(), [], []
    for a, b, c, d, t in _dihedral_rows(pt):
        improper = d < 0
        i, l = a // 3, abs(d) // 3
        pair = (min(i, l), max(i, l))
        want = not improper and pair in pairs14 and pair not in done and per[t - 1] > 0
        if want:
            done.add(pair)
            c = abs(c)
        elif c == 0 and not improper:  # atom 0 third: -0 does not exist, reverse
            a, b, c, d = d, c, b, a
            c = -abs(c)
        else:
            c = -abs(c)  # (pmemd never takes 1-4 pairs from impropers)
        row = (a, b, c, d, t)
        (rows_h if any(isH[abs(x) // 3] for x in row[:4]) else rows_n).extend(row)
    missing = pairs14 - done
    if missing:
        i, j = sorted(missing)[0]
        raise ValueError(
            f"{len(missing)} pairs 3 bonds apart (e.g. atoms {i + 1}, {j + 1}) have no proper "
            "dihedral with a nonzero periodicity to carry their 1-4 interaction"
        )
    pt.set("DIHEDRALS_INC_HYDROGEN", rows_h)
    pt.set("DIHEDRALS_WITHOUT_HYDROGEN", rows_n)
    return len(done)


def _input_lj14_scale(pt) -> float:
    """1 / SCNB of the dihedrals that carry 1-4 pairs in an Amber prmtop (2 when the section is
    absent, Amber's default); several values raise (pass lj14_scale)."""
    rows = [r for r in _dihedral_rows(pt) if r[2] >= 0 and r[3] >= 0]
    if not rows:
        return 0.5
    scnb = pt.get("SCNB_SCALE_FACTOR") if "SCNB_SCALE_FACTOR" in pt else np.full(pt.pointers["NPTRA"], 2.0)
    vals = sorted({round(float(scnb[r[4] - 1]), 6) for r in rows})
    if len(vals) != 1 or vals[0] <= 0:
        raise ValueError(f"the prmtop's 1-4 pairs have SCNB {vals}: pass lj14_scale (one value per file)")
    return 1.0 / vals[0]


def _rigid_bonds(pt, asys, templates, isH) -> float:
    """Bond lengths of rigid molecules = their templates' distances (the engine's constraints; SHAKE
    and SETTLE read them from the bonds).  Every pair of a rigid molecule must be a prmtop bond.
    Bond types are rebuilt (deduplicated); returns the largest change of a length (A)."""
    B = np.concatenate([pt.get("BONDS_INC_HYDROGEN"), pt.get("BONDS_WITHOUT_HYDROGEN")]).reshape(-1, 3)
    K, r0 = pt.get("BOND_FORCE_CONSTANT"), pt.get("BOND_EQUIL_VALUE")
    params = {tuple(sorted((int(a) // 3, int(b) // 3))): [K[t - 1], r0[t - 1]] for a, b, t in B}
    change = 0.0
    for m, tpl in zip(asys.molecules, templates):
        if tpl.has_bonded or m.n < 2:
            continue
        for i, j, d0 in tpl.md_rule("none").constraints:
            pair = tuple(sorted((int(m.atoms[i]), int(m.atoms[j]))))
            if pair not in params:
                raise ValueError(
                    f"rigid molecule at atom {m.atoms[0] + 1}: no prmtop bond between atoms "
                    f"{pair[0] + 1} and {pair[1] + 1} (pmemd would not hold it rigid)"
                )
            change = max(change, abs(params[pair][1] - 10.0 * d0))
            params[pair][1] = 10.0 * d0
    table, rows_h, rows_n = {}, [], []
    for (a, b), (k, r) in params.items():
        t = table.setdefault((round(float(k), 8), round(float(r), 8)), len(table) + 1)
        (rows_h if isH[a] or isH[b] else rows_n).extend([3 * a, 3 * b, t])
    pt.set("BOND_FORCE_CONSTANT", [k for k, _ in table])
    pt.set("BOND_EQUIL_VALUE", [r for _, r in table])
    pt.set("BONDS_INC_HYDROGEN", rows_h)
    pt.set("BONDS_WITHOUT_HYDROGEN", rows_n)
    pt.set_pointers(NBONH=len(rows_h) // 3, MBONA=len(rows_n) // 3, NBONA=len(rows_n) // 3, NUMBND=len(table))
    return change


def write_pgm_prmtop(
    asys,
    out: str,
    templates=None,
    params=None,
    system=None,
    lj14_scale: float | None = None,
    lj_min_sep: int | None = None,
    hmr: float | None = None,
) -> dict:
    """Write `out`: asys.prmtop (the tleap topology load_amber read) with the pGM model of the MD engine.

    asys       AmberSystem (protein.load_amber): molecules, their prmtop atoms and pGM parameters
    templates  one per molecule, as passed to FlexibleSimulation (asys.templates({k: tpl})): the
               FlexibleTemplates' bonded terms are exported and their lj14_scale / lj_min_sep
               used; the RigidTemplates' distances become the lengths of their bonds.
               None: the prmtop's bonded terms are kept, water and ions rigid, other molecules
               flexible with lj14_scale (default: 1 / the prmtop's SCNB, 1/2 for ff19SB) and
               lj_min_sep (default 4)
    params     the parameter pytree the engine runs with (default: system.params0)
    system     the System of those parameters (default: asys.system(), molecules in asys order)
    hmr        hydrogen mass (amu) for mass repartitioning, as FlexibleSimulation(hmr=...)
    Returns a summary (counts, 1-4 mode, bonded export counts)."""
    from ..bonded.amber import export_bonded
    from ..md.flux import template_flux_order

    flux = [t.name for t in (templates or []) if t is not None and template_flux_order(t)]
    if flux or (isinstance(params, dict) and "flux" in params):
        raise ValueError(
            f"charge flux ({', '.join(flux) or 'flux parameters'}): pmemd-pgm has no charge flux, so the "
            "model cannot be written as a pmemd-pgm prmtop"
        )
    pt0 = Prmtop.read(asys.prmtop)
    if "CTITLE" in pt0 or ("CHARMM_UREY_BRADLEY_COUNT" in pt0 and np.any(pt0.get("CHARMM_UREY_BRADLEY_COUNT"))):
        raise ValueError(f"{asys.prmtop} is a CHARMM (chamber) topology; only Amber topologies are converted")
    if any(m.molecule.vsites for m in asys.molecules):
        # pmemd.pgm (CPU) spreads extra-point forces in its pGM branch (pme_force.F90: orient_frc), but
        # pmemd.pgm.cuda's pGM force path (cuda/pgm_gpu.cpp) never calls kOrientForces: on the GPU the
        # forces on extra points would not reach their frames, so this topology would not run the model
        raise NotImplementedError(
            "write_pgm_prmtop: systems with virtual sites (Amber extra points) are not "
            "supported: pmemd.pgm.cuda's pGM force path does not spread extra-point forces"
        )
    n = int(pt0.pointers["NATOM"])
    order = np.asarray(asys.order)
    if len(order) != n or sorted(order.tolist()) != list(range(n)):
        raise ValueError("the AmberSystem does not cover the prmtop's atoms")
    if templates is not None and (lj14_scale is not None or lj_min_sep is not None):
        raise ValueError("lj14_scale / lj_min_sep come from the templates")
    if templates is None and lj14_scale is None:
        lj14_scale = _input_lj14_scale(pt0)
    rules = molecule_rules(asys, templates, lj14_scale, 4 if lj_min_sep is None else lj_min_sep)
    flex = [r for r in rules if r.vdw == "graph"]
    scales = sorted({float(r.lj14_scale) for r in flex})
    if len(scales) > 1:
        raise ValueError(f"flexible molecules with different lj14_scale {scales}: one 1-4 table per file")
    s14 = scales[0] if scales else 1.0

    # ---------------------------------------------------------------- bonded terms (export_bonded)
    exported = {}
    with tempfile.TemporaryDirectory() as tmp:
        cur = asys.prmtop
        for k, (m, tpl) in enumerate(zip(asys.molecules, templates or [None] * len(rules))):
            if tpl is None or not tpl.has_bonded:
                continue
            off = int(m.atoms[0])
            if not np.array_equal(m.atoms, np.arange(off, off + m.n)):
                raise ValueError(f"molecule {k}: atoms not contiguous in the prmtop")
            nxt = os.path.join(tmp, f"export_{k}.prmtop")
            exported[k] = export_bonded(cur, nxt, tpl.terms, tpl.P, m=tpl.index, offset=off, scnb=1.0)
            cur = nxt
        pt = Prmtop.read(cur)
    rigid_change = _rigid_bonds(pt, asys, templates, pt.get("ATOMIC_NUMBER") == 1) if templates is not None else None

    # ---------------------------------------------------------------- per-atom parameters, prmtop order
    sys_ = asys.system() if system is None else system
    if sys_.n != n:
        raise ValueError("system and prmtop atom counts differ")
    P = {k: np.asarray(v) for k, v in sys_.expand(params).items()}
    per_atom = {}
    for qn in ("q", "radius", "alpha", "lj_rmin_half", "lj_sqrt_eps"):
        v = np.empty(n)
        v[order] = P[qn]
        per_atom[qn] = v
    Z = pt.get("ATOMIC_NUMBER")
    el_sys = np.array(sys_.elements)
    el_prm = np.empty(n, object)
    el_prm[order] = el_sys
    Zel = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
    bad = [a for a in range(n) if int(Z[a]) in Zel and Zel[int(Z[a])] != el_prm[a]]
    if bad:
        raise ValueError(f"system and prmtop elements differ (atom {bad[0] + 1})")
    ci, cj, cc = order[sys_.cov_i], order[sys_.cov_j], P["cov"] * 10.0
    cov = [[] for _ in range(n)]
    for i, j, c in zip(ci.tolist(), cj.tolist(), cc.tolist()):
        cov[i].append((j, c))

    # ---------------------------------------------------------------- pGM sections
    pt.set("CHARGE", per_atom["q"] * AMBER_CHARGE)
    for name in POL_GAUSS:
        pt.remove(name)
    if "IPOL" not in pt:
        pt.set("IPOL", [0], fmt="1I8")
    pt.set(
        "POL_GAUSS_FORCEFIELD",
        [1],
        fmt="i5",
        after="IPOL",
        comments=[
            "This indicates that this parm file is specific to pGM force field",
            "This must be present if ipgm (in mdin) is 1",
            "This must NOT be present if ipgm is 0",
            "written by pgm_jax.protein.pmemd.write_pgm_prmtop",
        ],
    )
    pt.set(
        "POL_GAUSS_COVALENT_POINTERS_LIST",
        [len(c) for c in cov],
        fmt="10I8",
        after="POL_GAUSS_FORCEFIELD",
        comments=["number of covalent dipoles per atom", f"  dimension = {n}"],
    )
    ncov = sum(len(c) for c in cov)
    pt.set(
        "POL_GAUSS_COVALENT_ATOMS_LIST",
        [j + 1 for c in cov for j, _ in c],
        fmt="10I8",
        after="POL_GAUSS_COVALENT_POINTERS_LIST",
        comments=[f"  dimension = {ncov}"],
    )
    pt.set(
        "POL_GAUSS_COVALENT_DIPOLES_LIST",
        [x for c in cov for _, x in c],
        fmt="5E16.8",
        after="POL_GAUSS_COVALENT_ATOMS_LIST",
        comments=["  unit: e-Angstrom", f"  dimension = {ncov}"],
    )
    after = "POL_GAUSS_COVALENT_DIPOLES_LIST"
    for name, v, unit in (
        ("POL_GAUSS_MONOPOLES_LIST", per_atom["q"], "e"),
        ("POL_GAUSS_RADII_LIST", per_atom["radius"] * 10.0, "Angstrom"),
        ("POL_GAUSS_POLARIZABILITY_LIST", per_atom["alpha"] * 1e3, "Angstrom**3"),
    ):
        pt.set(name, v, fmt="5E16.8", after=after, comments=[f"  unit: {unit}", f"  dimension = {n}"])
        after = name

    # ---------------------------------------------------------------- Lennard-Jones types and tables
    ti, nt, ico, A, B = _lj_tables(per_atom["lj_rmin_half"], per_atom["lj_sqrt_eps"])
    pt.set("ATOM_TYPE_INDEX", ti)
    pt.set("NONBONDED_PARM_INDEX", ico)
    pt.set("LENNARD_JONES_ACOEF", A)
    pt.set("LENNARD_JONES_BCOEF", B)
    for name in ("HBOND_ACOEF", "HBOND_BCOEF", "HBCUT"):
        pt.set(name, [], fmt="5E16.8", after="EXCLUDED_ATOMS_LIST")

    # ---------------------------------------------------------------- exclusions and 1-4 pairs
    excl = [[] for _ in range(n)]
    pairs14 = set()
    cache = {}
    for m, rule in zip(asys.molecules, rules):
        key = (id(rule), m.n)
        if key not in cache:
            cache[key] = pair_classes(m.n, rule)
        ex, p14 = cache[key]
        a = m.atoms
        for i, j in ex:
            u, v = sorted((int(a[i]), int(a[j])))
            excl[u].append(v)
        pairs14.update(tuple(sorted((int(a[i]), int(a[j])))) for i, j in p14)
    counts = [len(e) if e else 1 for e in excl]
    pt.set("NUMBER_EXCLUDED_ATOMS", counts)
    pt.set("EXCLUDED_ATOMS_LIST", [x for e in excl for x in ([v + 1 for v in sorted(e)] if e else [0])])
    n14 = _set_14_flags(pt, pairs14, Z == 1)
    ntypes_d = len(pt.get("DIHEDRAL_PERIODICITY"))
    pt.set("SCNB_SCALE_FACTOR", np.ones(ntypes_d), fmt="5E16.8", after="SCEE_SCALE_FACTOR")
    for name in CHARMM_SECTIONS:
        pt.remove(name)
    if s14 != 1.0:
        _charmm_14(pt, s14, A, B)
    rows_h = len(pt.get("DIHEDRALS_INC_HYDROGEN")) // 5
    rows_n = len(pt.get("DIHEDRALS_WITHOUT_HYDROGEN")) // 5
    pt.set_pointers(NTYPES=nt, NNB=sum(counts), NPHB=0, NPHIH=rows_h, MPHIA=rows_n, NPHIA=rows_n)

    # ---------------------------------------------------------------- masses
    if hmr is not None:
        bonds = (
            np.concatenate([pt.get("BONDS_INC_HYDROGEN"), pt.get("BONDS_WITHOUT_HYDROGEN")]).reshape(-1, 3)[:, :2] // 3
        )
        pt.set("MASS", repartition_masses(pt.get("MASS"), list(el_prm), bonds, hmr))
    pt.write(out)
    return {
        "atoms": n,
        "lj_types": nt,
        "excluded_pairs": int(sum(len(e) for e in excl)),
        "pairs_14": len(pairs14),
        "dihedrals_14": n14,
        "lj14_scale": s14,
        "lj14_mode": "SCNB 1" if s14 == 1.0 else "CHARMM 1-4 tables",
        "covalent_dipoles": ncov,
        "exported": exported,
        "rigid_length_change_A": rigid_change,
        "hmr": hmr,
    }


def _charmm_14(pt, scale: float, A, B):
    """1-4 LJ = scale x LJ through the CHARMM 1-4 tables (see the module docstring)."""
    after = "LENNARD_JONES_BCOEF"
    pt.set(
        "FORCE_FIELD_TYPE",
        [f"{1:2d}{CHARMM_TAG.format(scale):<78.78s}"],
        fmt="i2,a78",
        after="POINTERS",
        comments=["pmemd reads LENNARD_JONES_14_* when this names CHARMM: pmemd-pgm fixes SCNB at 1"],
    )
    for name, v in (
        ("LENNARD_JONES_14_ACOEF", scale * np.asarray(A)),
        ("LENNARD_JONES_14_BCOEF", scale * np.asarray(B)),
    ):
        pt.set(name, v, fmt="5E16.8", after=after, comments=[f"{scale:g} x LENNARD_JONES_*COEF (pgm_jax lj14_scale)"])
        after = name
    for name, fmt, v in (
        ("CHARMM_UREY_BRADLEY_COUNT", "2I8", [0, 0]),
        ("CHARMM_UREY_BRADLEY", "10I8", []),
        ("CHARMM_UREY_BRADLEY_FORCE_CONSTANT", "5E16.8", []),
        ("CHARMM_UREY_BRADLEY_EQUIL_VALUE", "5E16.8", []),
        ("CHARMM_NUM_IMPROPERS", "10I8", [0]),
        ("CHARMM_IMPROPERS", "10I8", []),
        ("CHARMM_NUM_IMPR_TYPES", "1I8", [0]),
        ("CHARMM_IMPROPER_FORCE_CONSTANT", "5E16.8", []),
        ("CHARMM_IMPROPER_PHASE", "5E16.8", []),
    ):
        pt.set(name, v, fmt=fmt, after=after)
        after = name


# ------------------------------------------------------------------ mdin
def _pmemd_fft_ok(n: int) -> bool:
    if n % 4:
        return False
    for f in (2, 3, 5):
        while n % f == 0:
            n //= f
    return n == 1


def pmemd_grid(H_nm, spacing: float = 0.08) -> tuple:
    """Smallest PME grid with at most `spacing` nm between planes that pmemd.pgm and
    pmemd.pgm.cuda both accept (multiples of 4, prime factors 2, 3, 5): run the engine with
    MDSettings().replace(pme_grid=pmemd_grid(H)) so that both codes use the same grid."""
    H = np.asarray(H_nm, float)
    V = abs(np.linalg.det(H))
    heights = [V / np.linalg.norm(np.cross(H[(i + 1) % 3], H[(i + 2) % 3])) for i in range(3)]
    out = []
    for h in heights:
        n = max(4, int(np.ceil(h / spacing - 1e-9)))
        while not _pmemd_fft_ok(n):
            n += 1
        out.append(n)
    return tuple(out)


def pmemd_mdin(
    settings,
    H_nm,
    nstlim: int = 0,
    dt: float = 0.002,
    ensemble: str = "nvt",
    temperature: float = 298.0,
    thermostat: str = "langevin",
    gamma: float = 1.0,
    tau_t: float = 1.0,
    pressure: float = 1.0,
    constraints: str = "h-bonds",
    irest: int = 0,
    ntpr: int = 1000,
    ntwx: int = 0,
    ntwr: int = 0,
    ntwf: int = 0,
    ig: int = -1,
    maxcyc: int = 0,
    tempi: float | None = None,
    title: str = "pgm_jax model",
) -> str:
    """pmemd-pgm mdin with the nonbonded model of MDSettings: cut = ee_dsum_cut = cutoff (one
    cutoff for LJ and the pGM direct sum), ew_coeff = ewald_beta, the PME grid (pme_grid, or from
    pme_spacing and the box H_nm; pmemd needs pmemd_grid's sizes) and order, vdwmeth =
    lj_lrc, dipole_scf_tol = dipole_tol.  pmemd's influence function has an extra factor
    (factor_lambda in pme_recip_dat.F90) that the engine's Euler-spline moduli do not: at 0.8 A
    grid spacing and order 6 the electrostatic energies differ by ~3e-7 relative, at 0.4 A and
    order 8 by 2e-9 (validation/check_pgm_prmtop.json).
    Dynamics: dt (ps), ensemble "nve" | "nvt" | "npt" (Monte Carlo barostat), thermostat
    "langevin" (ntt=3, gamma 1/ps) | "bussi" (ntt=11, tau_t ps), tempi (initial velocities, K;
    default: temperature; pmemd draws them for every degree of freedom before SHAKE, so a
    constrained system starts ~1.5x hotter); constraints "h-bonds" (ntc=2,
    ntf=2: SHAKE on X-H bonds, rigid water by SETTLE; the engine's constraints="h-bonds") or "none"
    (ntc=1, ntf=1: every bond flexible, water too, unlike the engine, which always keeps water
    rigid; for single points).  maxcyc > 0: an energy minimisation instead (imin=1, 100 steepest-descent
    steps then conjugate gradients, no constraints), e.g. for tleap structures with clashes.  The
    induced-dipole solver is left at pmemd-pgm's defaults (its PCG with local preconditioner); the
    GPU predictor is set by the environment (PGM_GPU_PRED)."""
    from ..md.pme import grid_size

    H = np.asarray(H_nm, float)
    grid = tuple(settings.pme.grid) if settings.pme.grid is not None else tuple(grid_size(H, settings.pme.spacing))
    if not all(_pmemd_fft_ok(int(k)) for k in grid):
        raise ValueError(
            f"PME grid {grid}: pmemd.pgm(.cuda) needs multiples of 4 with prime factors 2, 3, 5; run "
            f"the engine with MDSettings().replace(pme_grid=pmemd_grid(H, spacing)), e.g. "
            f"{pmemd_grid(H, settings.pme.spacing)}"
        )
    cut = 10.0 * settings.cutoffs.cutoff
    if settings.terms.vdw != "lj" or settings.terms.elec != "qpi":
        raise ValueError("pmemd_mdin writes LJ + full pGM (vdw='lj', elec='qpi')")
    ntc, ntf = {"h-bonds": (2, 2), "none": (1, 1)}[constraints]
    if maxcyc > 0:
        head = f"   imin=1, maxcyc={maxcyc}, ncyc={min(100, maxcyc)}, ntmin=1, ntx=1, ipgm=1,\n   ntb=1,\n"
        ntc = ntf = 1
    ntb, extra = 1, ""
    if ensemble == "npt":
        ntb, extra = 2, f"ntp=1, barostat=2, pres0={pressure:g}, mcbarint=100,\n   "
    t0 = temperature if tempi is None else tempi
    if ensemble == "nve":
        ntt = "ntt=0,"
    elif thermostat == "langevin":
        ntt = f"ntt=3, gamma_ln={gamma:g}, temp0={temperature:g}, tempi={t0:g}, ig={ig},"
    elif thermostat == "bussi":
        ntt = f"ntt=11, tautp={tau_t:g}, temp0={temperature:g}, tempi={t0:g}, ig={ig},"
    else:
        raise ValueError(f"thermostat {thermostat!r}: pmemd_mdin writes langevin | bussi")
    if maxcyc <= 0:
        head = (
            f"   imin=0, nstlim={nstlim}, dt={dt:g}, irest={irest}, ntx={5 if irest else 1}, ipgm=1,\n"
            f"   ntb={ntb}, {extra}{ntt}\n"
        )
    return (
        f" {title}\n &cntrl\n{head}"
        f"   ntc={ntc}, ntf={ntf}, tol=0.0000001, cut={cut:g},\n"
        f"   ntpr={ntpr}, ntwx={ntwx}, ntwr={ntwr}, ntwf={ntwf}, ioutfm=1,\n /\n"
        f" &ewald\n   nfft1={grid[0]}, nfft2={grid[1]}, nfft3={grid[2]}, order={settings.pme.order},"
        f" ew_coeff={settings.pme.ewald_beta / 10.0:g},\n   skinnb={10.0 * settings.neighbors.skin:g},"
        f" vdwmeth={1 if settings.terms.lj_lrc else 0},\n /\n"
        f" &pol_gauss\n   ee_dsum_cut={cut:g}, dipole_scf_tol={settings.induction.tol:g},"
        f" scf_cg_niter={settings.induction.max_iter},\n /\n"
    )
