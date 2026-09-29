"""Macromolecules in the MD engine: neighbour-list groups and special pairs (md/topology.py),
constraints (md/constraints.py), rigid water by constraints, a flexible peptide against the
gas-phase model it was fitted with."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_grad import water

from pgm_jax import System
from pgm_jax.bonded import terms as T
from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec
from pgm_jax.md.constraints import Constraints, repartition_masses
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, RigidTemplate
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.topology import MDTopology, MoleculeRule, heavy_atom_groups
from pgm_jax.system import Molecule

PEPTIDE = "CC(=O)N[C@@H](C)C(=O)NCC(=O)NC"  # Ace-Ala-Gly-Nme, 29 atoms
_RAD = {"H": 0.05, "C": 0.07, "N": 0.065, "O": 0.06}
_ALP = {"H": 0.4e-3, "C": 1.2e-3, "N": 1.0e-3, "O": 0.8e-3}
_RMH = {"H": 0.13, "C": 0.19, "N": 0.18, "O": 0.17}
_SEP = {"H": 0.12, "C": 0.33, "N": 0.40, "O": 0.45}


def peptide(smiles=PEPTIDE, seed=5):
    """(MolSpec with a pGM molecule of plausible parameters, geometry nm)."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    AllChem.ComputeGasteigerCharges(m)
    x = m.GetConformer().GetPositions() * 0.1
    el = [a.GetSymbol() for a in m.GetAtoms()]
    q = np.array([a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()])
    q -= q.mean()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    orders = [b.GetBondTypeAsDouble() for b in m.GetBonds()]
    rng = np.random.default_rng(seed)
    cov = [c for i, j in bonds for c in ((i, j, 0.004 * rng.normal()), (j, i, 0.004 * rng.normal()))]
    mol = Molecule(
        "pep",
        el,
        el,
        q,
        [_RAD[e] for e in el],
        [_ALP[e] for e in el],
        cov=cov,
        lj_rmin_half=[_RMH[e] for e in el],
        lj_sqrt_eps=[_SEP[e] for e in el],
        bonds=bonds,
    )
    return MolSpec("pep", el, bonds, orders, 0, x, mol), x


def test_groups_and_special_pairs():
    s, x = peptide()
    top = MDTopology.build(System([s.pgm]), [MoleculeRule(bonds=s.bonds, vdw="graph", lj_min_sep=4, lj14_scale=0.5)])
    g = top.group
    assert top.n_group == sum(e != "H" for e in s.elements)
    assert np.array_equal(g, heavy_atom_groups(s.elements, s.bonds))
    from pgm_jax.bonded.topology import build_topology

    bt = build_topology(s.elements, s.bonds)
    pairs = {}
    for a in range(top.n):
        for b, w in zip(top.special[a], top.special_w[a]):
            if b < top.n:
                pairs[(a, int(b))] = w
    for (a, b), w in pairs.items():  # symmetric, weights from graph distances
        assert pairs[(b, a)] == w
        d = bt.dist[a, b]
        assert w == float(d >= 4) + 0.5 * (d == 3)
    for a in range(top.n):  # every pair within three bonds is special
        for b in range(top.n):
            if a != b and bt.dist[a, b] <= 3:
                assert (a, b) in pairs
    # rigid molecules: groups = molecules, every intramolecular pair special with weight 0
    rt = MDTopology.rigid(System([water(), water()]))
    assert rt.n_group == 2 and np.all(rt.special_w == 0) and rt.special.shape[1] == 2


def test_constraints_shake_rattle():
    rng = np.random.default_rng(0)
    # water triangle (0-2), CH3 (3-6), OH (7-8)
    x = np.array(
        [
            [0, 0, 0],
            [0.0957, 0, 0],
            [-0.024, 0.0927, 0],
            [1, 1, 1],
            [1.109, 1, 1],
            [0.964, 1.103, 1],
            [0.964, 0.948, 1.089],
            [2, 0, 0],
            [2.096, 0, 0],
        ],
        float,
    )
    pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (3, 6), (7, 8)]
    d0 = [np.linalg.norm(x[i] - x[j]) for i, j in pairs]
    m = np.array([16.0, 1.0, 1.0, 12.0, 1.0, 1.0, 1.0, 16.0, 1.0])
    C = Constraints(pairs, d0, m)
    y = x + 0.01 * rng.normal(size=x.shape)
    z = np.asarray(C.positions(jnp.asarray(y), jnp.asarray(x)))
    for (i, j), d in zip(pairs, d0):
        assert abs(np.linalg.norm(z[i] - z[j]) - d) < 1e-12
    assert np.allclose((m[:, None] * (z - y)).sum(0), 0.0, atol=1e-12)  # internal forces: no net momentum
    p = rng.normal(size=x.shape)
    q = np.asarray(C.momenta(jnp.asarray(z), jnp.asarray(p), m))
    v = q / m[:, None]
    for i, j in pairs:
        assert abs(np.dot(z[i] - z[j], v[i] - v[j])) < 1e-12
    assert np.allclose(np.asarray(C.momenta(jnp.asarray(z), jnp.asarray(q), m)), q, atol=1e-12)
    assert np.allclose(q.sum(0), p.sum(0), atol=1e-12)
    mh = repartition_masses(
        m, ["O", "H", "H", "C", "H", "H", "H", "O", "H"], [(0, 1), (0, 2), (3, 4), (3, 5), (3, 6), (7, 8)]
    )
    assert abs(mh.sum() - m.sum()) < 1e-12 and np.allclose(mh[[1, 2, 4, 5, 6, 8]], 3.024)


def _peptide_template():
    s, x = peptide()
    model = BondedModel([s], BondedSettings(families=T.PROTEIN, lj14_scale=0.5))
    P = model.init_params()
    rng = np.random.default_rng(1)
    P["cmap"]["cm"] = jnp.asarray(rng.normal(size=P["cmap"]["cm"].shape))
    P["torsion_amber"]["K"] = jnp.asarray(rng.normal(size=P["torsion_amber"]["K"].shape))
    return FlexibleTemplate.from_fit(model, P), model, P, x


def test_peptide_forces_match_gas_phase_model():
    """A 29-atom peptide split into heavy-atom groups: MD forces (PME pGM + bonded + intramolecular
    van der Waals from the special pairs, 1-4 scaled) equal the gradient of the gas-phase model."""
    tpl, model, P, x = _peptide_template()
    y = x + 0.003 * np.random.default_rng(0).normal(size=x.shape)
    s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=2.2, skin=0.05, lj_lrc=False)
    sim = FlexibleSimulation(System([tpl.pgm]), [tpl], y + 3.0, np.eye(3) * 6.0, s, thermostat=None, log=None)
    assert sim.topology.n_group > 1
    F = np.asarray(sim.state.dyn.force)
    g = np.asarray(jax.grad(lambda R: model.energy(0, R, P)[0])(jnp.asarray(y)))
    rms = np.sqrt(np.mean(g**2))
    assert np.abs(F + g).max() < 2e-3 * rms, (np.abs(F + g).max(), rms)


def _water_box(n_side=4, spacing=0.31, seed=0):
    rng = np.random.default_rng(seed)
    t = np.radians(104.52 / 2)
    w = np.array(
        [[0, 0, 0], [0.09572 * np.sin(t), 0.09572 * np.cos(t), 0], [-0.09572 * np.sin(t), 0.09572 * np.cos(t), 0]]
    )
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                pos.append(w @ Q.T + (np.array([i, j, k]) + 0.5) * spacing)
    return np.concatenate(pos), np.eye(3) * n_side * spacing, w


def test_rigid_water_by_constraints_matches_rigid_bodies():
    from pgm_jax.md.simulation import Simulation

    pos, H, w = _water_box()
    wat = water()
    sys = System([wat] * (len(pos) // 3))
    s = MDSettings().replace(precision="double", dipole_tol=1e-9, cutoff=0.55, skin=0.05)
    rig = Simulation(sys, pos, H, s, dt=0.001, thermostat=None, log=None)
    tpl = RigidTemplate(wat, w)
    flex = FlexibleSimulation(sys, [tpl] * sys.nmol, pos, H, s, dt=0.002, thermostat=None, log=None)
    assert flex.constraints.nc == 3 * sys.nmol
    e_r, e_f = float(rig.state.epot), float(flex.state.epot)
    assert abs(e_r - e_f) < 1e-8 * abs(e_r), (e_r, e_f)
    # NVE with SHAKE / RATTLE at 2 fs: constraints held; the energy fluctuates (hard cutoff in a
    # small box, lattice start) no more than with the rigid-body integrator
    rig2 = Simulation(sys, pos, H, s, dt=0.002, thermostat=None, log=None)
    dev = {}
    for name, sim in (("constraints", flex), ("rigid", rig2)):
        e0, d = sim.observables()["etot"], 0.0
        for _ in range(5):
            sim.advance(40)
            d = max(d, abs(sim.observables()["etot"] - e0))
        dev[name] = d
    assert flex.observables()["shake_err"] < 1e-10
    assert dev["constraints"] < 2.0 * dev["rigid"] + 1.0, dev


def test_peptide_in_water_hbond_constraints_hmr():
    """Flexible peptide + rigid water, X-H constraints and hydrogen mass repartitioning: runs at
    2 fs with the constraints held and a bounded energy drift."""
    tpl, model, P, x = _peptide_template()
    pos_w, H, w = _water_box(n_side=6, spacing=0.31)
    c = H.diagonal() / 2
    xp = x - x.mean(0) + c
    keep = [
        k
        for k in range(len(pos_w) // 3)
        if np.min(np.linalg.norm(pos_w[3 * k : 3 * k + 3, None] - xp[None], axis=-1)) > 0.25
    ]
    waters = np.concatenate([pos_w[3 * k : 3 * k + 3] for k in keep])
    wat = water()
    sys = System([tpl.pgm] + [wat] * len(keep))
    s = MDSettings().replace(precision="double", dipole_tol=1e-8, cutoff=0.7, skin=0.08)
    sim = FlexibleSimulation(
        sys,
        [tpl] + [RigidTemplate(wat, w)] * len(keep),
        np.concatenate([xp, waters]),
        H,
        s,
        dt=0.002,
        thermostat=None,
        constraints="h-bonds",
        hmr=3.024,
        log=None,
        r_margin=0.08,
    )
    n_h = sum(e == "H" for e in tpl.spec.elements)
    assert sim.constraints.nc == n_h + 3 * len(keep)
    e0 = sim.observables()["etot"]
    sim.advance(100)
    obs = sim.observables()
    assert obs["shake_err"] < 1e-9
    # bounded (the hard cutoff and the lattice start make the energy fluctuate by ~0.5 % of KE)
    assert np.isfinite(obs["etot"]) and abs(obs["etot"] - e0) < 0.02 * obs["ekin"], (e0, obs["etot"], obs["ekin"])
