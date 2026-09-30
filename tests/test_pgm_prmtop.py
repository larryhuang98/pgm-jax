"""pmemd-pgm topologies of the engine's model (protein/pmemd.py).

What is checked, and against what: the written pGM sections read back as the molecules (q,
alpha, radius, covalent dipoles, LJ; 1e-8); the exclusions and 1-4 pairs are the engine's pair
rules; the LJ and 1-4 tables are the engine's Lorentz-Berthelot pairs times lj14_scale; masses
with HMR; without templates the prmtop's own terms are kept; the mdin has the engine's nonbonded
settings.  With the pGM3P-25 files, a water round trip; with pmemd.pgm installed, its
single-point energies and forces on the written file against the engine's.
"""

import importlib.util
import os

import numpy as np
import pytest

from pgm_jax.md.constraints import repartition_masses
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.topology import MDTopology
from pgm_jax.param import read_prmtop_pgm
from pgm_jax.paths import resource
from pgm_jax.prmtop import Prmtop
from pgm_jax.protein import (
    ResidueLibrary,
    amber_template,
    load_amber,
    pmemd_grid,
    pmemd_mdin,
    write_pgm_prmtop,
)
from pgm_jax.units import KCAL

DATA = os.path.join(os.path.dirname(__file__), "data")
PRM, CRD = os.path.join(DATA, "pep_wat.prmtop"), os.path.join(DATA, "pep_wat.inpcrd")  # ACE-ALA-SER-NME, TIP3P, NaCl
WATER_TOP = resource("gvdw_data", "topology/rayl_512_v2.prmtop")
WATER_RST = resource("gvdw_data", "inputs/lj/inpcrd.restrt")
PMEMD = resource("pmemd_pgm_bin", "pmemd.pgm")


def _library():
    """Return a placeholder residue library with covalent dipoles, some to neighbouring residues."""
    lib = ResidueLibrary.placeholder(PRM)
    lib.residues["ALA"]["cov"] = [
        ["N", "H", 0.0012],
        ["N", "CA", -0.004],
        ["C", "+N", 0.002],
        ["N", "-C", -0.003],
        ["C", "O", 0.005],
    ]
    lib.residues["SER"]["cov"] = [["OG", "HG", 0.003], ["N", "-C", -0.002]]
    return lib


def _pairs_by_weight(sys_, templates, order):
    """Return the engine's special pairs in prmtop numbering, {(i, j): van der Waals weight}.

    Engine pair rules (md/topology.py) in prmtop numbering: {(i, j): van der Waals weight} of the special pairs.
    """
    topo = MDTopology.build(sys_, [t.md_rule("none") for t in templates])
    out = {}
    for a in range(sys_.n):
        for b, w in zip(topo.special[a], topo.special_w[a]):
            if b < sys_.n:
                i, j = sorted((int(order[a]), int(order[b])))
                out[(i, j)] = float(w)
    return out


def test_protein_sections_round_trip(tmp_path):
    """A written pmemd-pgm topology of the solvated peptide reads back as the engine's model.

    pGM sections and parameters (1e-8), cross-residue covalent dipoles, CHARGE = monopoles x 18.2223,
    exclusions and flagged 1-4 dihedrals = the engine's pair weights, LJ and 1-4 tables (5e-9
    relative), HMR masses, rigid-water bond lengths; without templates tleap's sections are kept; the
    mdin carries the engine's cutoff, Ewald coefficient, grid and tolerance.
    """
    asys = load_amber(PRM, CRD, electrostatics=_library())
    k = [i for i, m in enumerate(asys.molecules) if m.kind == "protein"][0]
    templates = asys.templates({k: amber_template(asys.molecules[k], PRM)})
    out = str(tmp_path / "pep_pgm.prmtop")
    info = write_pgm_prmtop(asys, out, templates, hmr=3.024)
    assert info["lj14_mode"] == "CHARMM 1-4 tables" and info["covalent_dipoles"] > 0
    pt = Prmtop.read(out)
    n = pt.pointers["NATOM"]
    # pGM sections: consistent lengths; read back, every molecule has its parameters
    ptr = pt.get("POL_GAUSS_COVALENT_POINTERS_LIST")
    assert len(ptr) == n and ptr.sum() == len(pt.get("POL_GAUSS_COVALENT_ATOMS_LIST")) == len(
        pt.get("POL_GAUSS_COVALENT_DIPOLES_LIST")
    )
    back = load_amber(out, CRD, electrostatics="prmtop")
    for m0, m1 in zip(asys.molecules, back.molecules):
        assert np.array_equal(m0.atoms, m1.atoms) and m0.kind == m1.kind
        for qn in ("q", "alpha", "radius", "lj_rmin_half", "lj_sqrt_eps"):
            np.testing.assert_allclose(getattr(m1.molecule, qn), getattr(m0.molecule, qn), rtol=1e-8, atol=1e-14)
        c0 = {(i, j): c for i, j, c in m0.molecule.cov}
        c1 = {(i, j): c for i, j, c in m1.molecule.cov}
        assert c0.keys() == c1.keys() and all(abs(c0[p] - c1[p]) <= 1e-8 * abs(c0[p]) for p in c0)
    assert any(abs(a - b) > 1 for m in back.molecules for a, b, _ in [(i, j, c) for i, j, c in m.molecule.cov])
    with pytest.raises(ValueError, match="another residue"):
        read_prmtop_pgm(out, first_residue_only=False)  # cross-residue dipoles need load_amber
    np.testing.assert_allclose(pt.get("CHARGE"), pt.get("POL_GAUSS_MONOPOLES_LIST") * 18.2223, rtol=1e-8)
    # exclusions and 1-4 pairs are the engine's: weight 0 or lj14_scale <-> excluded; 1-4 once each
    sys_ = asys.system()
    w = _pairs_by_weight(sys_, templates, asys.order)
    nex, lst = pt.get("NUMBER_EXCLUDED_ATOMS"), pt.get("EXCLUDED_ATOMS_LIST")
    start = np.concatenate([[0], np.cumsum(nex)])
    excl = {(i, int(j) - 1) for i in range(n) for j in lst[start[i] : start[i + 1]] if j > 0}
    assert excl == {p for p, x in w.items() if x != 1.0}
    D = np.concatenate([pt.get("DIHEDRALS_INC_HYDROGEN"), pt.get("DIHEDRALS_WITHOUT_HYDROGEN")]).reshape(-1, 5)
    flagged = [tuple(sorted((a // 3, d // 3))) for a, b, c, d, t in D if c >= 0 and d >= 0]
    assert len(flagged) == len(set(flagged)) and set(flagged) == {p for p, x in w.items() if x == 0.5}
    assert np.all(pt.get("SCNB_SCALE_FACTOR") == 1.0)
    # LJ tables: every type pair is the engine's Lorentz-Berthelot pair; 1-4 tables = lj14_scale x
    nt = pt.pointers["NTYPES"]
    ti = pt.get("ATOM_TYPE_INDEX") - 1
    ico = pt.get("NONBONDED_PARM_INDEX").reshape(nt, nt) - 1
    A, B = pt.get("LENNARD_JONES_ACOEF"), pt.get("LENNARD_JONES_BCOEF")
    P = {q: np.empty(n) for q in ("lj_rmin_half", "lj_sqrt_eps")}
    for q in P:
        P[q][asys.order] = np.asarray(sys_.expand()[q])
    first = {t: int(np.nonzero(ti == t)[0][0]) for t in range(nt)}
    for a in range(nt):
        for b in range(nt):
            i, j = first[a], first[b]
            eps = P["lj_sqrt_eps"][i] * P["lj_sqrt_eps"][j] / KCAL
            rmin = 10.0 * (P["lj_rmin_half"][i] + P["lj_rmin_half"][j])
            np.testing.assert_allclose([A[ico[a, b]], B[ico[a, b]]], [eps * rmin**12, 2 * eps * rmin**6], rtol=5e-9)
    np.testing.assert_allclose(pt.get("LENNARD_JONES_14_ACOEF"), 0.5 * A)
    np.testing.assert_allclose(pt.get("LENNARD_JONES_14_BCOEF"), 0.5 * B)
    assert "CHARMM" in pt.get("FORCE_FIELD_TYPE")[0] and pt.get("CHARMM_NUM_IMPROPERS")[0] == 0
    # masses as FlexibleSimulation(hmr=3.024); rigid-water bonds at the template's geometry
    bonds = np.concatenate(
        [np.asarray(m.bonds, int).reshape(-1, 2) + sys_.offsets[q] for q, m in enumerate(sys_.molecules)]
    )
    m_sys = repartition_masses(np.asarray(sys_.masses), sys_.elements, bonds, 3.024)
    np.testing.assert_allclose(pt.get("MASS")[asys.order], m_sys, rtol=1e-8)
    wat = [(m, t) for m, t in zip(asys.molecules, templates) if m.kind == "water"][0]
    B3 = np.concatenate([pt.get("BONDS_INC_HYDROGEN"), pt.get("BONDS_WITHOUT_HYDROGEN")]).reshape(-1, 3)
    r0 = {tuple(sorted((a // 3, b // 3))): pt.get("BOND_EQUIL_VALUE")[t - 1] for a, b, t in B3}
    for i, j, d0 in wat[1].md_rule().constraints:
        assert abs(r0[tuple(sorted((int(wat[0].atoms[i]), int(wat[0].atoms[j]))))] - 10 * d0) < 1e-7
    # without templates: the prmtop's bonded terms, tleap's exclusions and 1-4 pairs (Amber's 1-2, 1-3, 1-4)
    out2 = str(tmp_path / "pep_kept.prmtop")
    assert write_pgm_prmtop(asys, out2, lj14_scale=1.0)["lj14_mode"] == "SCNB 1"
    p2, p0 = Prmtop.read(out2), Prmtop.read(PRM)
    for name in (
        "EXCLUDED_ATOMS_LIST",
        "NUMBER_EXCLUDED_ATOMS",
        "BOND_FORCE_CONSTANT",
        "CMAP_INDEX",
        "CMAP_PARAMETER_01",
    ):
        np.testing.assert_array_equal(p2.get(name), p0.get(name))
    assert "FORCE_FIELD_TYPE" not in p2 and "LENNARD_JONES_14_ACOEF" not in p2
    D0 = np.concatenate([p0.get("DIHEDRALS_INC_HYDROGEN"), p0.get("DIHEDRALS_WITHOUT_HYDROGEN")]).reshape(-1, 5)
    D2 = np.concatenate([p2.get("DIHEDRALS_INC_HYDROGEN"), p2.get("DIHEDRALS_WITHOUT_HYDROGEN")]).reshape(-1, 5)
    assert sorted(map(tuple, D0)) == sorted(map(tuple, D2))
    # mdin: the engine's nonbonded settings
    st = MDSettings().replace(
        cutoff=0.9, ewald_beta=3.5, pme_grid=pmemd_grid(asys.box), pme_order=8, lj_lrc=False, dipole_tol=1e-6
    )
    txt = pmemd_mdin(st, asys.box, nstlim=100)
    for s in (
        "cut=9,",
        "ee_dsum_cut=9,",
        "ew_coeff=0.35",
        "order=8",
        "vdwmeth=0",
        "dipole_scf_tol=1e-06",
        "ipgm=1",
        f"nfft1={st.pme.grid[0]},",
    ):
        assert s in txt, s
    with pytest.raises(ValueError, match="multiples of 4"):
        pmemd_mdin(MDSettings().replace(pme_grid=(33, 36, 36)), asys.box)


@pytest.mark.skipif(not os.path.exists(WATER_TOP), reason="pGM3P-25 water prmtop not available")
@pytest.mark.needs_data
def test_pgm_water_round_trip(tmp_path):
    """The pGM3P-25 water prmtop survives read / write (parameters 1e-8, sections exact)."""
    asys = load_amber(WATER_TOP, WATER_RST, electrostatics="prmtop")
    out = str(tmp_path / "w.prmtop")
    info = write_pgm_prmtop(asys, out, asys.templates())
    assert info["lj14_mode"] == "SCNB 1" and info["rigid_length_change_A"] < 1e-5
    w0, w1 = read_prmtop_pgm(WATER_TOP)[0], read_prmtop_pgm(out)[0]
    for qn in ("q", "alpha", "radius", "lj_rmin_half", "lj_sqrt_eps", "masses"):
        np.testing.assert_allclose(getattr(w1, qn), getattr(w0, qn), rtol=1e-8, atol=1e-14)
    assert [(i, j) for i, j, _ in w0.cov] == [(i, j) for i, j, _ in w1.cov]
    np.testing.assert_allclose([c for *_, c in w1.cov], [c for *_, c in w0.cov], rtol=1e-8)
    p0, p1 = Prmtop.read(WATER_TOP), Prmtop.read(out)
    for name in ("POL_GAUSS_COVALENT_ATOMS_LIST", "EXCLUDED_ATOMS_LIST", "ATOMS_PER_MOLECULE"):
        np.testing.assert_array_equal(p1.get(name), p0.get(name))
    np.testing.assert_allclose(p1.get("LENNARD_JONES_ACOEF"), p0.get("LENNARD_JONES_ACOEF"), rtol=1e-8)


@pytest.mark.skipif(not os.path.exists(PMEMD), reason="pmemd.pgm not installed")
@pytest.mark.needs_external
def test_pmemd_single_point_matches_engine(tmp_path, monkeypatch):
    """pmemd.pgm single-point energies and forces equal the engine's on the written topology.

    pmemd.pgm (CPU, float64) on the written solvated peptide against the engine: bonded, 1-4,
    van der Waals and pGM electrostatic energies and the forces (the engine's PME with pmemd's
    influence-function factor, the one PME convention in which they differ).
    """
    spec = importlib.util.spec_from_file_location(
        "check_pgm_prmtop",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts/protein/check_pgm_prmtop.py"),
    )
    ck = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ck)
    import pgm_jax.md.pme as pme

    plain = pme.bspline_moduli
    monkeypatch.setattr(pme, "bspline_moduli", lambda K, order: plain(K, order) / ck.amber_lambda(K, order))
    asys, templates = ck.model("pep")
    prm = str(tmp_path / "pep.prmtop")
    write_pgm_prmtop(asys, prm, templates)
    st = ck.settings_for(asys.box)
    ck.run_pmemd("cpu", str(tmp_path / "cpu"), prm, CRD, ck.sp_mdin(st, asys.box))
    E = ck.read_energies(str(tmp_path / "cpu/mdout"))
    F = ck.read_forces(str(tmp_path / "cpu/mdfrc"))
    e_rb, f_rb = ck.rigid_bond_terms(prm, asys, templates, asys.positions * 10.0)
    E["BOND"] -= e_rb
    eng, F_eng = ck.engine(asys, templates, st)
    for term in ("BOND", "ANGLE", "DIHED", "CMAP", "1-4 NB", "VDWAALS"):
        assert abs(E[term] - eng[term]) < 2e-4, (term, E[term], eng[term])
    assert E["1-4 EEL"] == 0.0 and abs(E["EELEC"] - eng["EELEC"]) < 1e-3, (E["EELEC"], eng["EELEC"])
    dF = F - f_rb - F_eng
    other = np.setdiff1d(np.arange(len(F)), ck.cmap_atoms(asys, templates))
    assert np.abs(dF[other]).max() < 2e-5 and np.abs(dF).max() < 5e-3  # CMAP: bicubic grid vs Fourier map
