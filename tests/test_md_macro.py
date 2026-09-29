"""Macromolecules in the MD engine: groups and special pairs, constraints, a flexible peptide.

What is checked, and against what: heavy-atom neighbour-list groups and special pairs with weights
from graph distances (md/topology.py); SHAKE / RATTLE on water, CH3 and OH clusters (constraint
lengths and velocities to 1e-12, no net momentum) and mass repartitioning; rigid water by
constraints against rigid bodies (energy 1e-8 relative, comparable NVE fluctuation); a flexible
29-atom peptide against the gas-phase model it was fitted with (forces 2e-3 of the RMS force); a
solvated peptide with X-H constraints and HMR at 2 fs.  The peptides need RDKit.
"""

import jax
import jax.numpy as jnp
import numpy as np
from _systems import peptide, peptide_template, requires, water, water_lattice

from pgm_jax import System
from pgm_jax.md.constraints import Constraints, repartition_masses
from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.topology import MDTopology, MoleculeRule, heavy_atom_groups


@requires("rdkit")
def test_groups_and_special_pairs():
    """Topology groups are the heavy atoms and special pairs carry graph-distance weights.

    Every pair within three bonds is special, weights are symmetric (0 for 1-2 / 1-3, 0.5 for 1-4, 1
    beyond); rigid molecules form one group each with every intramolecular pair excluded.
    """
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
    """SHAKE / RATTLE hold bond lengths and tangent velocities; mass repartitioning keeps the total mass."""
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


@requires("rdkit")
def test_peptide_forces_match_gas_phase_model():
    """MD forces of a flexible peptide equal the gradient of its gas-phase model.

    A 29-atom peptide split into heavy-atom groups: MD forces (PME pGM + bonded + intramolecular
    van der Waals from the special pairs, 1-4 scaled) equal the gradient of the gas-phase model.
    """
    tpl, model, P, x = peptide_template()
    y = x + 0.003 * np.random.default_rng(0).normal(size=x.shape)
    s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=2.2, skin=0.05, lj_lrc=False)
    sim = FlexibleSimulation(System([tpl.pgm]), [tpl], y + 3.0, np.eye(3) * 6.0, s, thermostat=None, log=None)
    assert sim.topology.n_group > 1
    F = np.asarray(sim.state.dyn.force)
    g = np.asarray(jax.grad(lambda R: model.energy(0, R, P)[0])(jnp.asarray(y)))
    rms = np.sqrt(np.mean(g**2))
    assert np.abs(F + g).max() < 2e-3 * rms, (np.abs(F + g).max(), rms)


def test_rigid_water_by_constraints_matches_rigid_bodies():
    """Rigid water by constraints equals rigid bodies and conserves energy as well at 2 fs."""
    from pgm_jax.md.simulation import Simulation

    pos, H, w = water_lattice()
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


@requires("rdkit")
def test_peptide_in_water_hbond_constraints_hmr():
    """A solvated flexible peptide with X-H constraints and HMR runs stably at 2 fs.

    Flexible peptide + rigid water, X-H constraints and hydrogen mass repartitioning: runs at
    2 fs with the constraints held and a bounded energy drift.
    """
    tpl, model, P, x = peptide_template()
    pos_w, H, w = water_lattice(n_side=6, spacing=0.31)
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
