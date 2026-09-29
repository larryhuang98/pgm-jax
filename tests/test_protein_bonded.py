"""Protein bonded terms: backbone and residues from the bond graph, the CMAP family (Fourier
phi/psi correction), the protein term set."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_bonded_sets import _fd_check  # noqa: E402

from pgm_jax.bonded import terms as T  # noqa: E402
from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec  # noqa: E402
from pgm_jax.bonded.topology import build_topology  # noqa: E402

ACE_ALA_NME = "CC(=O)N[C@@H](C)C(=O)NC"
ACE_ALA_GLY_NME = "CC(=O)N[C@@H](C)C(=O)NCC(=O)NC"


def peptide_spec(smiles, name="peptide", seed=7):
    """MolSpec of a small peptide from SMILES (RDKit, ETKDG geometry, nm); no pGM parameters."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    x = m.GetConformer().GetPositions() * 0.1
    el = [a.GetSymbol() for a in m.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    orders = [b.GetBondTypeAsDouble() for b in m.GetBonds()]
    return MolSpec(name, el, bonds, orders, 0, x)


def _top(spec):
    return build_topology(spec.elements, spec.bonds, (spec.bonds, spec.bond_orders), spec.ref_xyz * 10.0)


def test_backbone_and_residues_from_graph():
    s = peptide_spec(ACE_ALA_GLY_NME)
    top = _top(s)
    assert top.cmaps.shape == (2, 5)  # Ala and Gly have both torsions; the caps none
    assert int(top.residue.max()) + 1 == 4  # ACE, ALA, GLY, NME
    el = s.elements
    for cp, n, ca, c, nn in top.cmaps:
        assert (el[cp], el[n], el[ca], el[c], el[nn]) == ("C", "N", "C", "C", "N")
        assert top.residue[n] == top.residue[ca] == top.residue[c]
        assert top.residue[cp] != top.residue[n] and top.residue[nn] != top.residue[c]
    # every atom of a residue is connected inside it; peptide bonds join consecutive residues
    assert len(top.backbone) == 2
    # small molecules without a backbone
    t2 = build_topology(["C", "O", "H", "H", "H", "H"], [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)])
    assert len(t2.cmaps) == 0 and set(t2.residue) == {0}


def test_large_molecule_topology_is_sparse():
    s = peptide_spec(ACE_ALA_GLY_NME)
    dense = _top(s)
    sparse = build_topology(s.elements, s.bonds, (s.bonds, s.bond_orders), s.ref_xyz * 10.0, dense=False)
    assert sparse.dist is None and sparse.pairs15 is None
    for f in ("bonds", "angles", "propers", "impropers", "pairs13", "pairs14", "angle_angle", "cmaps", "residue"):
        assert np.array_equal(getattr(dense, f), getattr(sparse, f)), f
    i, j = np.triu_indices(s.ref_xyz.shape[0], 1)
    for a, b in zip(i[:200], j[:200]):
        assert min(dense.dist[a, b], 4) == sparse.graph_distance(a, b)


def test_cmap_basis_grid_and_torsions():
    order = T.cmap.ORDER if hasattr(T, "cmap") else 3
    nb = (2 * order + 1) ** 2 - 1
    rng = np.random.default_rng(0)
    phi, psi = rng.uniform(-np.pi, np.pi, 50), rng.uniform(-np.pi, np.pi, 50)
    B = np.asarray(T.cmap_basis(jnp.asarray(phi), jnp.asarray(psi)))
    assert B.shape == (50, nb)
    assert np.allclose(
        B, np.asarray(T.cmap_basis(jnp.asarray(phi + 2 * np.pi), jnp.asarray(psi - 2 * np.pi))), atol=1e-12
    )
    # orthogonal on the full grid (distinct Fourier modes)
    g = -np.pi + 2 * np.pi * np.arange(24) / 24
    P, S = np.meshgrid(g, g, indexing="ij")
    Bg = np.asarray(T.cmap_basis(jnp.asarray(P.ravel()), jnp.asarray(S.ravel())))
    C = Bg.T @ Bg
    assert np.allclose(C - np.diag(np.diag(C)), 0.0, atol=1e-9)
    c = rng.normal(size=nb)
    assert np.allclose(T.cmap_grid(c), (Bg @ c).reshape(24, 24))
    # phi/psi of the quintuples are the proper torsions of the topology
    s = peptide_spec(ACE_ALA_NME)
    top = _top(s)
    G = T.geometry(jnp.asarray(s.ref_xyz), top)
    ph, ps = T.phi_psi(jnp.asarray(s.ref_xyz), top.cmaps)
    props = {tuple(t): k for k, t in enumerate(top.propers.tolist())}
    for q, a, b in zip(top.cmaps.tolist(), np.asarray(ph), np.asarray(ps)):
        for t, v in ((tuple(q[:4]), a), (tuple(q[1:]), b)):
            k = props.get(t, props.get(t[::-1]))
            assert k is not None and abs(float(G["phi"][k]) - v) < 1e-12


def test_protein_set_energy_gradients_and_invariance():
    s = peptide_spec(ACE_ALA_GLY_NME)
    model = BondedModel([s], BondedSettings(families=T.SETS["protein"], lj14_scale=0.5))
    assert "cmap" in model.fams and len(model.keys["cmap"]) == 2
    P = model.init_params()
    rng = np.random.default_rng(3)
    P["cmap"]["cm"] = jnp.asarray(rng.normal(size=P["cmap"]["cm"].shape))
    x = s.ref_xyz + 0.003 * rng.normal(size=s.ref_xyz.shape)
    E = lambda R: model.bonded_energy(0, R, P)
    _fd_check(E, x, rng)
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    Q = Q * np.sign(np.linalg.det(Q))  # proper rotation (phi/psi change sign under reflection)
    assert abs(float(E(jnp.asarray(x @ Q.T))) - float(E(jnp.asarray(x)))) < 1e-9
    # zero map, zero energy
    P0 = model.init_params()
    e_amber = BondedModel([s], BondedSettings(families=T.AMBER, lj14_scale=0.5)).bonded_energy(0, jnp.asarray(x), P0)
    assert abs(float(model.bonded_energy(0, jnp.asarray(x), P0)) - float(e_amber)) < 1e-9


def _nn_net(specs, **kw):
    from pgm_jax.bonded.nn import NNBConfig, NNBonded

    for s in specs:
        if s.top is None:
            s.top = _top(s)
    return NNBonded.for_molecules(specs, NNBConfig(basis=T.PROTEIN, **kw))


def _randomise(P, rng, scale=0.1):
    return jax.tree_util.tree_map(lambda v: v + scale * jnp.asarray(rng.normal(size=v.shape)), P)


def test_nnb_protein_basis_context_reuse_and_persistence(tmp_path):
    from pgm_jax.bonded.nn import NNBonded

    train = [peptide_spec(ACE_ALA_NME, "ala"), peptide_spec(ACE_ALA_GLY_NME, "alagly")]
    net = _nn_net(train)
    assert net.vocab.skeletons["cmap"] == ["X"] and net.vocab.slots["cmap"] == 1
    rng = np.random.default_rng(0)
    P = _randomise(net.init_params(), rng)
    assert net.vocab.frozen and "res" in P
    # a molecule that was not in the training set: same parameter shapes
    new = peptide_spec("CC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)NC", "alaala")
    new.top = _top(new)
    C = net.coefficients(P, net.prepare(new))
    assert C["cmap"]["cm"].shape == (2, (2 * 3 + 1) ** 2 - 1)
    # the residue context reaches the map: changing only the residue MLP changes the cmap coefficients
    P2 = dict(P)
    P2["res"] = _randomise(P["res"], rng, 0.5)
    C2 = net.coefficients(P2, net.prepare(new))
    assert float(jnp.max(jnp.abs(C2["cmap"]["cm"] - C["cmap"]["cm"]))) > 1e-6
    assert float(jnp.max(jnp.abs(C2["bond_harm"]["Kb"] - C["bond_harm"]["Kb"]))) == 0.0
    # stage 2 gradients
    tab = net.prepare(new)
    x = new.ref_xyz + 0.003 * rng.normal(size=new.ref_xyz.shape)
    _fd_check(lambda R: net.energy_from(C, tab, R), x, rng)
    # save / load: same coefficients, frozen vocabulary
    net.save(str(tmp_path / "nnb.pkl"), P)
    net2, P3 = NNBonded.load(str(tmp_path / "nnb.pkl"))
    C3 = net2.coefficients(P3, net2.prepare(new))
    assert all(
        np.allclose(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(C), jax.tree_util.tree_leaves(C3))
    )
    # families the training set never saw are refused, not silently mispredicted
    small = _nn_net(
        [
            MolSpec(
                "meoh",
                ["C", "O", "H", "H", "H", "H"],
                [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)],
                [1] * 5,
                0,
                np.array(
                    [
                        [0, 0, 0],
                        [0.143, 0, 0],
                        [-0.036, 0.1, 0],
                        [-0.036, -0.05, 0.087],
                        [-0.036, -0.05, -0.087],
                        [0.175, 0.09, 0],
                    ]
                ),
            )
        ]
    )
    small.init_params()
    with pytest.raises(ValueError):
        small.prepare(new)


def test_nnb_context_off_matches_head_width():
    train = [peptide_spec(ACE_ALA_NME, "ala")]
    on, off = _nn_net(train), _nn_net([peptide_spec(ACE_ALA_NME, "ala")], context=False)
    Pon, Poff = on.init_params(), off.init_params()
    W = on.config.width
    assert Pon["head_cmap"]["w1"].shape[0] - Poff["head_cmap"]["w1"].shape[0] == 3 * W
    assert "res" not in Poff
    # identical weights where the shapes agree (same seed and key order)
    assert np.allclose(Pon["head_bond_harm"]["w1"], Poff["head_bond_harm"]["w1"])


def test_dihedral_sign_is_iupac():
    """The sign of the torsions (it matters for the CMAP sine terms) is RDKit's / Amber's."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem, rdMolTransforms

    m = Chem.AddHs(Chem.MolFromSmiles(ACE_ALA_NME))
    AllChem.EmbedMolecule(m, randomSeed=3)
    X = m.GetConformer().GetPositions()
    for q in [(1, 3, 4, 6), (3, 4, 6, 8), (0, 1, 3, 4), (2, 1, 3, 4)]:
        ref = rdMolTransforms.GetDihedralRad(m.GetConformer(), *q)
        got = float(T.geometry.__globals__["_dihedral"](*[jnp.asarray(X[i]) for i in q]))
        assert abs(np.angle(np.exp(1j * (got - ref)))) < 1e-9


def _minimal_prmtop(path, elements):
    """A prmtop with only what the bonded export / import reads (not a runnable topology)."""
    from pgm_jax.prmtop import POINTER_NAMES, Prmtop, Section

    Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
    secs = [
        Section("POINTERS", "10I8", [len(elements)] + [0] * (len(POINTER_NAMES) - 1)),
        Section("ATOMIC_NUMBER", "10I8", [Z[e] for e in elements]),
    ]
    for name, fmt in (
        ("BOND_FORCE_CONSTANT", "5E16.8"),
        ("BOND_EQUIL_VALUE", "5E16.8"),
        ("ANGLE_FORCE_CONSTANT", "5E16.8"),
        ("ANGLE_EQUIL_VALUE", "5E16.8"),
        ("DIHEDRAL_FORCE_CONSTANT", "5E16.8"),
        ("DIHEDRAL_PERIODICITY", "5E16.8"),
        ("DIHEDRAL_PHASE", "5E16.8"),
        ("SCEE_SCALE_FACTOR", "5E16.8"),
        ("SCNB_SCALE_FACTOR", "5E16.8"),
        ("BONDS_INC_HYDROGEN", "10I8"),
        ("BONDS_WITHOUT_HYDROGEN", "10I8"),
        ("ANGLES_INC_HYDROGEN", "10I8"),
        ("ANGLES_WITHOUT_HYDROGEN", "10I8"),
        ("DIHEDRALS_INC_HYDROGEN", "10I8"),
        ("DIHEDRALS_WITHOUT_HYDROGEN", "10I8"),
    ):
        secs.append(Section(name, fmt, []))
    Prmtop("%VERSION  VERSION_STAMP = V0001.000", secs).write(path)


def test_prmtop_export_import_round_trip(tmp_path):
    from pgm_jax.bonded.amber import export_bonded, init_from_prmtop, read_bonded
    from pgm_jax.bonded.model import BondedTerms
    from pgm_jax.prmtop import Prmtop

    s = peptide_spec(ACE_ALA_GLY_NME)
    terms = BondedTerms([s], BondedSettings(families=T.PROTEIN))
    rng = np.random.default_rng(4)
    P = jax.tree_util.tree_map(np.asarray, terms.init_params())
    P["bond_harm"]["Kb"] = P["bond_harm"]["Kb"] * rng.uniform(0.5, 1.5, P["bond_harm"]["Kb"].shape)
    P["angle_harm"]["Ka"] = P["angle_harm"]["Ka"] * rng.uniform(0.5, 1.5, P["angle_harm"]["Ka"].shape)
    P["torsion_amber"]["K"] = rng.normal(size=P["torsion_amber"]["K"].shape) * 3.0
    P["improper_amber"]["K"] = rng.normal(size=P["improper_amber"]["K"].shape) * 10.0
    P["cmap"]["cm"] = rng.normal(size=P["cmap"]["cm"].shape)
    P["ref"]["th0"] = P["ref"]["th0"] + 0.05 * rng.normal(size=P["ref"]["th0"].shape)
    src, out = str(tmp_path / "in.prmtop"), str(tmp_path / "out.prmtop")
    _minimal_prmtop(src, s.elements)
    counts = export_bonded(src, out, terms, P, scnb=2.0)
    assert counts["cmap"][0] == 2
    ptr = Prmtop.read(out).pointers
    amb = read_bonded(out)
    assert ptr["NBONH"] + ptr["MBONA"] == len(s.bonds) == len(amb["bonds"])
    # one dihedral per 1-4 pair carries the 1-4 interactions
    raw = np.concatenate(
        [Prmtop.read(out).get("DIHEDRALS_INC_HYDROGEN"), Prmtop.read(out).get("DIHEDRALS_WITHOUT_HYDROGEN")]
    ).reshape(-1, 5)
    with14 = {tuple(sorted((a // 3, d // 3))) for a, b, c, d, t in raw if c > 0 and d > 0}
    assert with14 == {tuple(p) for p in terms.mols[0].top.pairs14.tolist()}
    # importing the exported file gives the parameters back
    fresh = BondedTerms([s], BondedSettings(families=T.PROTEIN))
    Q = init_from_prmtop(fresh, fresh.init_params(), {0: out})
    for f, name in (
        ("bond_harm", "Kb"),
        ("angle_harm", "Ka"),
        ("torsion_amber", "K"),
        ("improper_amber", "K"),
        ("cmap", "cm"),
    ):
        tol = 1e-4 if f == "cmap" else 1e-6  # CMAP grids are written with 5 decimals (kcal/mol)
        assert np.allclose(np.asarray(Q[f][name]), P[f][name], rtol=tol, atol=tol), f
    for name in ("b0", "th0"):
        assert np.allclose(np.asarray(Q["ref"][name]), P["ref"][name], atol=1e-7), name
    x = jnp.asarray(s.ref_xyz + 0.003 * rng.normal(size=s.ref_xyz.shape))
    e1 = terms.bonded_energy(0, x, jax.tree_util.tree_map(jnp.asarray, P))
    e2 = fresh.bonded_energy(0, x, Q)
    assert abs(float(e1) - float(e2)) < 1e-6 * max(1.0, abs(float(e1)))
