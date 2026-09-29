"""Proteins from Amber topologies: molecules, backbone and residues, pGM electrostatics from a
residue library, Amber-form bonded terms from ff19SB, a solvated peptide in MD with constraints."""

import os

import numpy as np

from pgm_jax.bonded.amber import read_bonded
from pgm_jax.md.flexible import FlexibleSimulation
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.protein import ResidueLibrary, amber_template, load_amber

DATA = os.path.join(os.path.dirname(__file__), "data")
PRM, CRD = (
    os.path.join(DATA, "pep_wat.prmtop"),
    os.path.join(DATA, "pep_wat.inpcrd"),
)  # ACE-ALA-SER-NME, TIP3P, NaCl (ff19SB)


def test_load_molecules_and_library(tmp_path):
    asys = load_amber(PRM, CRD)
    kinds = [m.kind for m in asys.molecules]
    assert kinds[0] == "protein" and kinds.count("ion") == 2 and kinds.count("water") == len(kinds) - 3
    prot = asys.molecules[0]
    top = prot.spec.top
    assert top.cmaps.shape == (2, 5) and int(top.residue.max()) + 1 == 4
    names = prot.atom_names
    for q in top.cmaps:  # C(i-1) N CA C N(i+1) by Amber's names
        assert [names[a] for a in q] == ["C", "N", "CA", "C", "N"]
    assert abs(float(np.sum(prot.molecule.q))) < 1e-4  # neutral peptide
    # waters share one Molecule; the system keeps the prmtop order
    wat = [m.molecule for m in asys.molecules if m.kind == "water"]
    assert all(w is wat[0] for w in wat)
    assert np.array_equal(asys.order, np.arange(len(asys.positions)))
    # a residue library with covalent dipoles, including a partner in the next residue
    lib = ResidueLibrary.placeholder(PRM)
    lib.residues["ALA"]["cov"] = [["N", "H", 0.001], ["C", "+N", 0.002], ["N", "-C", -0.003]]
    lib.save(str(tmp_path / "lib.json"))
    asys2 = load_amber(PRM, CRD, electrostatics=ResidueLibrary.load(str(tmp_path / "lib.json")))
    cov = asys2.molecules[0].molecule.cov
    n = names
    res = asys2.molecules[0].residue_names
    got = {(n[i], res[i], n[j], res[j]) for i, j, c in cov}
    assert ("C", "ALA", "N", "SER") in got and ("N", "ALA", "C", "ACE") in got and ("N", "ALA", "H", "ALA") in got


def test_amber_template_matches_prmtop_terms():
    asys = load_amber(PRM, CRD)
    prot = asys.molecules[0]
    tpl = amber_template(prot, PRM)
    amb = read_bonded(PRM)
    nb = sum(1 for e in amb["bonds"] if e[0] < prot.n)
    assert nb == len(prot.spec.bonds)
    # the typed parameters reproduce the ff19SB bond and angle constants of every instance
    P = tpl.P
    Im = tpl.terms.I[0]
    Kb = np.asarray(P["bond_harm"]["Kb"])[Im["bond_harm"]["k"]] / (2 * 4.184 * 100)
    ref = {tuple(sorted(e[:2])): e[2] for e in amb["bonds"] if e[0] < prot.n}
    for (i, j), k in zip(tpl.terms.mols[0].top.bonds, Kb):
        assert abs(ref[(int(i), int(j))] - k) < 1e-6 * ref[(int(i), int(j))]


def test_solvated_peptide_md_with_constraints():
    asys = load_amber(PRM, CRD)
    tpl = amber_template(asys.molecules[0], PRM)
    s = MDSettings(precision="mixed", cutoff=0.8, skin=0.1, dipole_tol=1e-5)
    sim = FlexibleSimulation(
        asys.system(),
        asys.templates({0: tpl}),
        asys.system_positions(),
        asys.box,
        s,
        dt=0.002,
        ensemble="nvt",
        constraints="h-bonds",
        hmr=3.024,
        log=None,
    )
    n_h_prot = sum(e == "H" for e in asys.molecules[0].spec.elements)
    n_w = sum(m.kind == "water" for m in asys.molecules)
    assert sim.constraints.nc == n_h_prot + 3 * n_w
    sim._advance(100)
    obs = sim.observables()
    assert np.isfinite(obs["etot"]) and obs["shake_err"] < 1e-6 and 150 < obs["temp_K"] < 450, obs
