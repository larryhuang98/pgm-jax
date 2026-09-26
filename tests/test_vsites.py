"""Virtual sites (md/vsites.py): constructions against the OpenMM / Amber formulas, force spreading
(the transposed Jacobian) against finite differences, Amber extra points from tleap topologies,
the pair topology, pGM with charged and polarizable sites (forces and strain derivative against
finite differences with the dipoles re-solved), atoms with zero polarizability, and both MD
engines (energies, body forces, NVE, degrees of freedom)."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pgm_jax.md.box import reduce_box  # noqa: E402
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, RigidTemplate, liquid_box  # noqa: E402
from pgm_jax.md.forcefield import MDSettings, PGMForceField  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.md.io import box_from_cell, read_coordinates  # noqa: E402
from pgm_jax.md.simulation import Simulation, _dedupe  # noqa: E402
from pgm_jax.md.topology import MDTopology, MoleculeRule  # noqa: E402
from pgm_jax.md.vsites import VirtualSite, VirtualSites, amber_extra_points  # noqa: E402
from pgm_jax.param import molecule_from_dict, molecule_to_dict, read_prmtop_pgm  # noqa: E402
from pgm_jax.system import Molecule, System  # noqa: E402
from test_grad import methanol, water  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
TET = np.radians(54.735)


# ----------------------------------------------------------------------------- molecules
def methanol_sites():
    """Methanol with one site of every kind (charged; some polarizable, some not), a covalent
    dipole to a site, and positions (nm) with the sites placed."""
    m, x = methanol()                       # C O H H H HO
    vs = [VirtualSite.average2(6, 0, 1, 0.3, 0.7),
          VirtualSite.average3(7, 1, 0, 5, 0.5, 0.3, 0.2),
          VirtualSite.out_of_plane(8, 1, 0, 5, 0.2, 0.3, 4.0),
          VirtualSite.local(9, (1, 0, 5, 2), (0.6, 0.4, 0.0, 0.0), (-1.0, 1.0, 0.0, 0.0), (-1.0, 0.0, 1.0, 0.0),
                            (0.02, 0.015, -0.01)),
          VirtualSite.amber(10, 1, 0, 5, (0.0, 0.03 * np.sin(TET), 0.03 * np.cos(TET))),
          VirtualSite.amber(11, 0, 2, 3, (0.01, -0.02, 0.025))]
    ns = len(vs)
    q = np.concatenate([m.q, [0.05, -0.08, 0.04, -0.03, -0.06, 0.02]])
    q[1] -= q.sum()                          # neutral
    mol = Molecule("MeOHVS", m.elements + ["EP"] * ns, m.types + [f"EP{k}" for k in range(ns)], q,
                   np.concatenate([m.radius, [0.04, 0.05, 0.03, 0.04, 0.035, 0.03]]),
                   np.concatenate([m.alpha, [0.2e-3, 0.0, 0.1e-3, 0.0, 0.15e-3, 0.0]]),
                   cov=m.cov + [(1, 10, 0.004), (10, 1, -0.002), (0, 6, 0.003)],
                   lj_rmin_half=np.concatenate([m.lj_rmin_half, np.zeros(ns)]),
                   lj_sqrt_eps=np.concatenate([m.lj_sqrt_eps, np.zeros(ns)]), bonds=m.bonds, vsites=vs)
    xs = np.asarray(VirtualSites.of(System([mol])).place(np.concatenate([x, np.zeros((ns, 3))])))
    return mol, xs


def reference_sites(x):
    """The sites of methanol_sites from the published formulas, absolute coordinates, numpy."""
    C, O, H1, H2, H3, HO = x[:6]
    unit = lambda v: v / np.linalg.norm(v)                                  # noqa: E731
    out = [0.3 * C + 0.7 * O,                                                # TwoParticleAverageSite
           0.5 * O + 0.3 * C + 0.2 * HO]                                     # ThreeParticleAverageSite
    r12, r13 = C - O, HO - O
    out.append(O + 0.2 * r12 + 0.3 * r13 + 4.0 * np.cross(r12, r13))         # OutOfPlaneSite
    o = 0.6 * O + 0.4 * C                                                    # LocalCoordinatesSite
    xd, yd = C - O, HO - O
    ez = unit(np.cross(xd, yd)); ex = unit(xd); ey = np.cross(ez, ex)
    out.append(o + 0.02 * ex + 0.015 * ey - 0.01 * ez)
    for B, A, Cc, p in ((O, C, HO, (0.0, 0.03 * np.sin(TET), 0.03 * np.cos(TET))), (C, H1, H2, (0.01, -0.02, 0.025))):
        u, v = unit(A - B), unit(Cc - B)                                      # sander do_local_global
        ave, diff = (u + v) / 2, (v - u) / 2
        f3, f1 = -ave / np.linalg.norm(ave), diff / np.linalg.norm(diff)
        f2 = np.cross(f3, f1)
        out.append(B + p[0] * f1 + p[1] * f2 + p[2] * f3)
    return np.array(out)


def lattice_box(mol, x, n_side=3, spacing=0.5, seed=0):
    """n_side^3 randomly rotated copies on a jittered cubic lattice (nm)."""
    rng = np.random.default_rng(seed)
    m = mol.masses
    x0 = x - (m[:, None] * x).sum(0) / m.sum()
    pos = []
    for i in range(n_side ** 3):
        g = np.array([i // n_side ** 2, (i // n_side) % n_side, i % n_side]) + 0.5
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        pos.append(x0 @ R.T + g * spacing + rng.normal(scale=0.01, size=3))
    return np.concatenate(pos), np.eye(3) * spacing * n_side


def settings(**kw):
    base = dict(cutoff=0.6, skin=0.05, ewald_beta=6.0, pme_grid=(40, 40, 40), pme_order=8, lj_lrc=True,
                dipole_tol=1e-12, max_iter=500, peek=0.0, precision="double")
    base.update(kw)
    return MDSettings(**base)


# ----------------------------------------------------------------------------- constructions
def test_constructions_match_published_formulas_and_minimum_image():
    mol, x = methanol_sites()
    vs = VirtualSites.of(System([mol]))
    assert vs.n_sites == 6 and vs.kinds == ("amber", "average2", "average3", "local", "outofplane")
    rng = np.random.default_rng(1)
    y = x.copy()
    y[:6] += rng.normal(scale=0.005, size=(6, 3))                          # a distorted geometry
    ref = reference_sites(y[:6])
    got = np.asarray(vs.place(y))
    assert np.abs(got[6:] - ref).max() < 1e-14 and np.array_equal(got[:6], y[:6])
    # a molecule straddling a triclinic box: parents in other images give the same site (relative to the host)
    H = reduce_box(np.array([[1.2, 0, 0], [0.3, 1.1, 0], [-0.2, 0.4, 1.0]]))
    shift = np.array([[0, 0, 0], [1, 0, 0], [0, -1, 1], [0, 0, 0], [1, 1, 0], [-1, 0, 0]]) @ H
    got = np.asarray(vs.place(np.concatenate([y[:6] + shift, np.zeros((6, 3))]), H))
    hosts = [0, 1, 1, 1, 1, 0]
    assert np.abs(got[6:] - got[hosts] - (ref - y[hosts])).max() < 1e-12
    # TIP4P M site from the geometry: on the bisector at d_OM
    w = VirtualSite.tip4p(3, 0, 1, 2, 0.0125)
    assert abs(w.params[1] - 0.10667672) < 1e-8 and abs(sum(w.params) - 1.0) < 1e-15


def test_spread_is_the_transposed_jacobian():
    mol, x = methanol_sites()
    vs = VirtualSites.of(System([mol]))
    rng = np.random.default_rng(2)
    F = rng.normal(size=x.shape)
    Fs = np.asarray(vs.spread(x, None, F))
    assert np.all(Fs[6:] == 0.0)
    # every kind is equivariant under rigid motions: total force and torque are conserved
    assert np.abs(Fs.sum(0) - F.sum(0)).max() < 1e-12
    assert np.abs(np.cross(x, Fs).sum(0) - np.cross(x, F).sum(0)).max() < 1e-12
    # work: F . d(place(x))/dx along random directions of the real atoms, by central differences
    E = lambda y: float(jnp.sum(jnp.asarray(F) * vs.place(y)))             # noqa: E731
    for _ in range(3):
        d = np.zeros_like(x)
        d[:6] = rng.normal(size=(6, 3))
        h = 1e-6
        fd = (E(x + h * d) - E(x - h * d)) / (2 * h)
        assert abs(fd - float(np.sum(Fs * d))) < 1e-7 * max(1.0, abs(fd))


def test_definitions_are_validated():
    with pytest.raises(ValueError):
        VirtualSite.average3(3, 0, 1, 2, 0.5, 0.3, 0.3)                 # weights do not sum to 1
    with pytest.raises(ValueError):
        VirtualSite.local(3, (0, 1, 2), (1, 0, 0), (1, 0, 0), (-1, 0, 1), (0, 0, 0))   # x weights must sum to 0
    with pytest.raises(ValueError):
        VirtualSite.average2(1, 1, 0, 0.5, 0.5)                          # a site among its parents
    w = water()
    bad = Molecule("W", w.elements + ["EP"], w.types + ["EP"], np.r_[w.q, 0.0], np.r_[w.radius, 0.01],
                   np.r_[w.alpha, 0.0], masses=np.r_[w.masses, 1.0], vsites=[VirtualSite.tip4p(3, 0, 1, 2, 0.015)])
    with pytest.raises(ValueError, match="massless"):
        VirtualSites.of(System([bad]))
    ok = Molecule("W", w.elements + ["EP"], w.types + ["EP"], np.r_[w.q, 0.0], np.r_[w.radius, 0.01],
                  np.r_[w.alpha, 0.0], vsites=[VirtualSite.tip4p(3, 0, 1, 2, 0.015)])
    assert ok.masses[3] == 0.0                                           # element EP is massless
    back = molecule_from_dict(molecule_to_dict(ok))
    assert back.vsites == ok.vsites
    plain = Molecule("W", w.elements + ["EP"], w.types + ["EP"], np.r_[w.q, 0.0], np.r_[w.radius, 0.01],
                     np.r_[w.alpha, 0.0], vsites=[VirtualSite.tip4p(3, 0, 1, 2, 0.02)])
    assert len({id(m) for m in _dedupe([ok, back, plain])}) == 2         # sites are part of the identity
    # degenerate frames are refused at setup (a local frame with parallel x and y directions)
    loc = Molecule("W", w.elements + ["EP"], w.types + ["EP"], np.r_[w.q, 0.0], np.r_[w.radius, 0.01], np.r_[w.alpha, 0.0],
                   vsites=[VirtualSite.local(3, (0, 1, 2), (1, 0, 0), (-1, 1, 0), (-2, 2, 0), (0.01, 0, 0))])
    x = np.array([[0, 0, 0], [0.1, 0, 0], [-0.03, 0.09, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="degenerate"):
        VirtualSites.of(System([loc])).check(x)


# ----------------------------------------------------------------------------- Amber extra points
def _amber_ep_reference(x, center, first, third, p):
    """sander do_local_global, frame type 1."""
    u = (x[first] - x[center]) / np.linalg.norm(x[first] - x[center])
    v = (x[third] - x[center]) / np.linalg.norm(x[third] - x[center])
    f3 = -(u + v) / np.linalg.norm(u + v)
    f1 = (v - u) / np.linalg.norm(v - u)
    return x[center] + p[0] * f1 + p[1] * np.cross(f3, f1) + p[2] * f3


@pytest.mark.parametrize("name,nep,req", [("tip4pew_small", 1, 0.0125), ("tip5p_small", 2, 0.07)])
def test_amber_extra_points_from_tleap(name, nep, req):
    mols = read_prmtop_pgm(os.path.join(DATA, name + ".prmtop"), first_residue_only=False, charges="amber")
    xyz, _, box = read_coordinates(os.path.join(DATA, name + ".inpcrd"))
    H = box_from_cell(*box) * 0.1
    sys = System(mols)
    vs = VirtualSites.of(sys)
    m = mols[0]
    assert m.elements == ["O", "H", "H"] + ["EP"] * nep and np.all(m.masses[3:] == 0) and vs.n_sites == nep * len(mols)
    assert abs(m.q.sum()) < 1e-6 and np.all(m.alpha == 0.0)
    pos = np.asarray(vs.place(xyz * 0.1, H))
    X = pos.reshape(len(mols), 3 + nep, 3)
    for k in range(3):
        for e in range(nep):
            p = m.vsites[e].params[2]
            assert np.abs(X[k, 3 + e] - _amber_ep_reference(X[k], 0, 1, 2, p)).max() < 1e-14
    d = np.linalg.norm(X[:, 3:] - X[:, :1], axis=-1)
    assert np.abs(d - req).max() < 1e-12
    u = X[:, 1:3] - X[:, :1]
    bis = (u / np.linalg.norm(u, axis=-1, keepdims=True)).sum(1)
    if nep == 1:                        # TIP4P: on the bisector, toward the hydrogens
        cos = np.sum((X[:, 3] - X[:, 0]) * bis, -1) / (d[:, 0] * np.linalg.norm(bis, axis=-1))
        assert cos.min() > 1 - 1e-12
    else:                               # TIP5P: tetrahedral lone pairs opposite the hydrogens
        lp = X[:, 3:] - X[:, :1]
        ang = np.degrees(np.arccos(np.sum(lp[:, 0] * lp[:, 1], -1) / (req * req)))
        assert np.abs(ang - 2 * 54.735).max() < 1e-9
        assert np.all(np.sum(lp.sum(1) * bis, -1) < 0)
    # tleap's own placement of the box waters, to its precision (TIP5P: tleap may list the two lone pairs the
    # other way round; the first water, tleap's library monomer, has its own lone-pair geometry, which sander
    # replaces at the start as we do)
    T = (xyz * 0.1).reshape(X.shape)
    dev = np.abs(X[:, 3:] - T[:, 3:]).max(axis=(1, 2))
    if nep == 2:
        dev = np.minimum(dev, np.abs(X[:, 3:] - T[:, 3:][:, ::-1]).max(axis=(1, 2)))
    assert dev[1:].max() < 5e-3 and np.array_equal(X[:, :3], T[:, :3])


def test_amber_frame_rules():
    # carbonyl oxygen (frame type 2): C1(=O2)(C3)N4 with two EPs on O
    types = ["C", "O", "CT", "N", "EP", "EP"]
    heavy = [(0, 1, 0), (0, 2, 0), (0, 3, 0), (1, 4, 1), (1, 5, 1)]
    eps = amber_extra_points(types, [], heavy, [1.5, 0.35])
    assert set(eps) == {4, 5} and eps[4].atoms == (1, 2, 0, 3) and eps[4].kind == "amber"
    s60 = np.sin(np.radians(60.0))
    assert np.allclose(eps[4].params[2], (s60 * 0.035, 0, 0.5 * 0.035)) and np.allclose(eps[5].params[2], (-s60 * 0.035, 0, 0.5 * 0.035))
    x = np.array([[0, 0, 0], [0, 0.123, 0], [0.13, -0.07, 0], [-0.13, -0.07, 0], [0, 0, 0], [0, 0, 0]], float)
    sys = System([Molecule("CO", ["C", "O", "C", "N", "EP", "EP"], types, np.zeros(6), np.full(6, 0.05), np.zeros(6),
                           vsites=[eps[4], eps[5]])])
    y = np.asarray(VirtualSites.of(sys).place(x))
    A, C = (x[2] + x[0]) / 2, (x[3] + x[0]) / 2                           # bond midpoints of the carbon
    ref = _amber_ep_reference(np.array([A, x[1], C]), 1, 0, 2, eps[4].params[2])
    assert np.abs(y[4] - ref).max() < 1e-14 and abs(np.linalg.norm(y[4] - x[1]) - 0.035) < 1e-14
    # sulfur: EPs along +-y; a heavy + hydrogen centre; errors where Amber stops
    eps = amber_extra_points(["S", "CT", "CT", "EP", "EP"], [], [(0, 1, 0), (0, 2, 0), (0, 3, 1), (0, 4, 1)], [1.8, 0.7])
    assert np.allclose(eps[3].params[2], (0, 0.07, 0)) and np.allclose(eps[4].params[2], (0, -0.07, 0))
    eps = amber_extra_points(["OH", "CT", "HO", "EP"], [(0, 2, 0)], [(0, 1, 0), (0, 3, 1)], [1.0, 0.5])
    assert eps[3].atoms == (0, 1, 2) and np.allclose(eps[3].params[2], (0, 0, 0.05))
    with pytest.raises(ValueError, match="too many"):
        amber_extra_points(["N", "C", "C", "C", "EP"], [], [(0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 1)], [1.4, 0.5])
    with pytest.raises(ValueError, match="bonded to no atom"):
        amber_extra_points(["OW", "HW", "HW", "EP"], [(0, 1, 0), (0, 2, 0)], [], [1.0])


# ----------------------------------------------------------------------------- pair topology
def test_topology_sites_belong_to_their_host():
    mol, x = methanol_sites()
    rule = MoleculeRule(bonds=list(mol.bonds) + [(1, 10)], vdw="graph", lj_min_sep=4, lj14_scale=0.5)
    top = MDTopology.build(System([mol]), [rule], max_single=4)          # force heavy-atom groups
    g = top.group
    assert g[6] == g[0] and g[11] == g[0] and all(g[k] == g[1] for k in (7, 8, 9, 10))
    assert top.n_group == len(set(g.tolist())) == 2                     # no empty groups
    w = {(int(a), int(b)): float(wt) for a in range(top.n) for b, wt in zip(top.special[a], top.special_w[a]) if b < top.n}
    assert w[(10, 1)] == 0.0 and w[(10, 7)] == 0.0                     # the host and a site of the same host
    assert w[(10, 2)] == 0.0                                            # host O - H1 is 1-3: excluded
    assert w[(5, 2)] == 0.5 and w[(10, 2)] == w[(1, 2)]                  # HO-C-O... 1-4; sites inherit
    assert w[(11, 5)] == w[(0, 5)] == 0.0                               # C-HO is 1-3
    with pytest.raises(ValueError, match="virtual sites"):
        MDTopology.build(System([mol]), [MoleculeRule(bonds=mol.bonds, constraints=((1, 10, 0.03),))])


# ----------------------------------------------------------------------------- pGM with sites
def _pgm_box():
    mol, x = methanol_sites()
    pos, H = lattice_box(mol, x, 3, 0.5)
    sys = System([mol] * 27)
    return sys, np.asarray(VirtualSites.of(sys).place(pos, H)), H


def test_pgm_forces_and_strain_derivative_with_sites():
    """Charged, polarizable and non-polarizable sites, a covalent dipole to a site: spread forces
    and the molecular strain derivative against central differences with the dipoles re-solved."""
    sys, pos, H = _pgm_box()
    vs = VirtualSites.of(sys)
    ff = PGMForceField(sys, H, settings())
    idx = ff.rows_for(pos, H)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    alpha = np.asarray(ff._atoms(None)["alpha"])
    mu = np.asarray(res.induction.mu)
    assert np.all(mu[alpha == 0] == 0.0) and np.abs(mu[alpha > 0]).min() > 0.0
    assert np.isfinite(float(res.energy["total"]))
    F = np.asarray(vs.spread(pos, H, res.forces))
    e = jax.jit(lambda y, h: ff.energy(vs.place(y, h), h, idx, ff.init_induction())[0])
    h = 1e-6
    for a, k in [(0, 0), (1, 1), (5, 2), (13, 0), (12 * 13 + 1, 2)]:
        d = np.zeros_like(pos); d[a, k] = h
        fd = -(float(e(pos + d, H)) - float(e(pos - d, H))) / (2 * h)
        assert abs(fd - F[a, k]) < 1e-6 * max(1.0, abs(fd)), (a, k, fd, F[a, k])
    W = np.asarray(ff.strain_derivative(pos, H, idx, res.induction.mu))
    m = np.asarray(sys.masses)
    com = np.array([np.average(pos[sys.mol == k], 0, weights=m[sys.mol == k]) for k in range(sys.nmol)])
    from pgm_jax.lj import lj_long_range
    tail = float(lj_long_range(ff._atoms(None), abs(np.linalg.det(H)), 0.6))
    for (i, j) in [(0, 0), (1, 2), (2, 0)]:
        eps = np.zeros((3, 3)); eps[i, j] = h
        ee = lambda s: float(e(pos + (com @ (s * eps).T)[sys.mol], H @ (np.eye(3) + s * eps).T))   # noqa: E731
        fd = (ee(1.0) - ee(-1.0)) / (2 * h) - (tail if i == j else 0.0)
        assert abs(fd - W[i, j]) < 1e-5 * max(1.0, abs(fd)), (i, j, fd, W[i, j])
    with pytest.raises(NotImplementedError):
        ff.strain_derivative(pos, H, idx, res.induction.mu, molecular=False)


def test_zero_polarizability_atoms():
    """alpha = 0 atoms (real atoms and sites): mu stays 0 with every predictor, energies and
    parameter gradients are finite (differentiable solve), and removing them from the
    polarizable set is the same as a limit alpha -> 0."""
    sys, pos, H = _pgm_box()
    P = dict(sys.params0)
    keys = sys.table.keys["alpha"]
    P["alpha"] = P["alpha"].at[keys.index("ho")].set(0.0)               # a real atom type without polarizability
    ff = PGMForceField(sys, H, settings(differentiable=True, adjoint_tol=1e-12))
    idx = ff.rows_for(pos, H)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction(), P)
    a = np.asarray(ff._atoms(P)["alpha"])
    assert np.all(np.asarray(res.induction.mu)[a == 0] == 0.0)
    g = jax.jit(jax.grad(lambda th: ff.compute(pos, H, idx, ff.init_induction(), th).energy["total"]))(P)
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves(g))
    # the mask is the limit of a vanishing polarizability
    P2 = dict(P); P2["alpha"] = P["alpha"].at[keys.index("ho")].set(1e-12)
    r2 = jax.jit(ff.compute)(pos, H, idx, ff.init_induction(), P2)
    assert abs(float(r2.energy["total"]) - float(res.energy["total"])) < 1e-8 * abs(float(res.energy["total"]))
    # every predictor path (fused mu4, ls, none) keeps them at 0 over a few steps
    for pred in ("mu4", "ls", "none"):
        f2 = PGMForceField(sys, H, settings(predictor=pred, dipole_tol=1e-6))
        ind = f2.init_induction()
        for step in range(5):
            r = jax.jit(f2.compute)(pos + 1e-4 * step, H, idx, ind, P)
            ind = r.induction
            assert np.all(np.asarray(ind.mu)[a == 0] == 0.0) and np.isfinite(float(r.energy["total"]))


# ----------------------------------------------------------------------------- MD engines
def _tip4pew_ideal():
    """The small tleap TIP4P-Ew box with every water at the model geometry (Amber's SHAKE lengths
    0.9572 / 1.5136 A) and the extra points placed."""
    prm = os.path.join(DATA, "tip4pew_small.prmtop")
    mols = _dedupe(read_prmtop_pgm(prm, first_residue_only=False, charges="amber"))
    xyz, _, box = read_coordinates(os.path.join(DATA, "tip4pew_small.inpcrd"))
    H = box_from_cell(*box) * 0.1
    X = (xyz * 0.1).reshape(-1, 4, 3)
    r, hh = 0.09572, 0.15136
    t = np.arcsin(hh / 2 / r)
    ideal = np.array([[0, 0, 0], [r * np.sin(t), r * np.cos(t), 0], [-r * np.sin(t), r * np.cos(t), 0]])
    for k in range(len(X)):
        y = X[k, :3] - X[k, :3].mean(0)
        c = ideal - ideal.mean(0)
        U, _, Vt = np.linalg.svd(c.T @ y)
        R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
        X[k, :3] = c @ R.T + X[k, :3].mean(0)
    sys = System(mols)
    return sys, np.asarray(VirtualSites.of(sys).place(X.reshape(-1, 3), H)), H


def test_engines_with_sites_agree_and_conserve_energy():
    """TIP4P-Ew (point charges, Amber's EP frame) in the rigid engine and as constrained water with a
    placed site in the flexible engine: same energies and body forces at the same state; NVE."""
    from pgm_jax.md.forcefield import ewald_beta_for
    sys, pos, H = _tip4pew_ideal()
    s = MDSettings(elec="q", cutoff=0.65, skin=0.05, ewald_beta=ewald_beta_for(0.65), pme_spacing=0.06,
                   precision="double")
    eq = Simulation(sys, pos, H, s, dt=0.001, ensemble="nvt", thermostat="bussi", tau_t=0.1, log=None, seed=1)
    eq._advance(500)                                      # relax tleap's voids a little
    pos, vel, H = eq.positions_nm(), eq.velocities_nm_ps(), np.asarray(eq.state.box)
    rig = Simulation(sys, pos, H, s, dt=0.001, ensemble="nve", log=None, vel_nm_ps=vel)
    tpl = RigidTemplate(sys.molecules[0], pos[:4])
    flx = FlexibleSimulation(sys, [tpl] * sys.nmol, pos, H, s, dt=0.001, ensemble="nve", log=None, vel_nm_ps=vel)
    n = sys.nmol
    assert rig.integ.dof == flx.integ.dof == 6 * n - 3 and flx.constraints.nc == 3 * n
    assert abs(rig.observables()["epot"] - flx.observables()["epot"]) < 1e-9 * abs(rig.observables()["epot"])
    assert abs(rig.observables()["ekin"] - flx.observables()["ekin"]) < 1e-9 * rig.observables()["ekin"]
    # the flexible engine's spread forces act on the bodies as the rigid engine's site forces do
    Fb = rig.rigid.forces(rig.state.dyn.position, jnp.asarray(flx.state.dyn.force))
    Fr = rig.state.dyn.force
    scale = float(jnp.abs(Fr.center).max())
    assert float(jnp.abs(Fb.center - Fr.center).max()) < 1e-9 * scale
    assert float(jnp.abs(Fb.orientation.vec - Fr.orientation.vec).max()) < 1e-9 * scale
    assert np.all(np.asarray(flx.state.dyn.force)[3::4] == 0.0)
    trace = []
    for sim in (rig, flx):
        E = []
        for _ in range(6):
            sim._advance(50)
            E.append(sim.observables()["etot"])
        ke = 0.5 * sim.integ.dof * KB * 300.0              # hard cutoffs in a small box: crossings dominate
        assert np.std(E) < 2e-3 * ke and abs(E[-1] - E[0]) < 3e-3 * ke, (type(sim).__name__, np.std(E) / ke)
        trace.append(np.array(E) - E[0])
    # NO_SQUISH rigid bodies and RATTLE + placed sites integrate the same dynamics
    assert np.abs(trace[0] - trace[1]).max() < 2e-4 * ke, trace
    X = flx.positions_nm().reshape(-1, 4, 3)
    assert np.abs(np.linalg.norm(X[:, 3] - X[:, 0], axis=1) - 0.0125).max() < 1e-12
    assert np.all(np.asarray(flx.state.dyn.momentum)[3::4] == 0.0)
    assert np.all(flx.velocities_nm_ps()[3::4] == 0.0)


def test_flexible_molecule_with_sites_nvt_nve_and_hmr():
    """Flexible methanol with one site of every kind: thermostat and constraints leave the sites
    alone, the temperature counts real atoms only, and NVE conserves energy."""
    from pgm_jax.bonded import terms as T
    from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec
    mol, x = methanol_sites()
    spec = MolSpec("meohvs", list(mol.elements), [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)], [1] * 5, 0, x, mol)
    model = BondedModel([spec], BondedSettings(families=T.PAPER, lj14_scale=0.5))
    tpl = FlexibleTemplate.from_fit(model, model.init_params())
    pos, H = liquid_box(tpl, 27, 0.45, seed=0, min_dist=0.18)
    sys = System([mol] * 27)
    s = MDSettings(precision="double", dipole_tol=1e-9, cutoff=0.6, skin=0.05, lj_lrc=False)
    sim = FlexibleSimulation(sys, [tpl] * 27, pos, H, s, dt=0.001, ensemble="nvt", temperature=300.0, thermostat="langevin",
                             gamma=20.0, constraints="h-bonds", hmr=3.0, log=None)
    m = np.asarray(sim.flex.masses)
    assert np.all(m[6::12] == 0.0) and abs(m.sum() - sys.masses.sum()) < 1e-9          # sites stay massless
    assert sim.integ.dof == 3 * 6 * 27 - 4 * 27
    sim._advance(300)
    o = sim.observables()
    assert 150.0 < o["temp_K"] < 450.0 and o["shake_err"] < 1e-9
    p = np.asarray(sim.state.dyn.momentum).reshape(27, 12, 3)
    assert np.all(p[:, 6:] == 0.0)
    y = sim.positions_nm().reshape(27, 12, 3)
    assert np.abs(y[:, 6:] - np.asarray(VirtualSites.of(sys).place(y.reshape(-1, 3), sim.state.box)).reshape(27, 12, 3)[:, 6:]).max() < 1e-12
    nve = FlexibleSimulation(sys, [tpl] * 27, sim.positions_nm(), np.asarray(sim.state.box), s, dt=0.0005, ensemble="nve",
                             vel_nm_ps=sim.velocities_nm_ps(), constraints="h-bonds", hmr=3.0, log=None)
    E = []
    for _ in range(8):
        nve._advance(50)
        E.append(nve.observables()["etot"])
    ke = 0.5 * nve.integ.dof * KB * 300.0
    assert np.std(E) < 1e-3 * ke and abs(E[-1] - E[0]) < 2e-3 * ke, (np.std(E) / ke, (E[-1] - E[0]) / ke)


def test_load_amber_protein_in_tip4pew():
    from pgm_jax.protein.amber import load_amber, amber_template
    from pgm_jax.protein.pmemd import write_pgm_prmtop
    prm, crd = os.path.join(DATA, "pep_tip4pew.prmtop"), os.path.join(DATA, "pep_tip4pew.inpcrd")
    asys = load_amber(prm, crd)
    kinds = [m.kind for m in asys.molecules]
    assert kinds.count("protein") == 1 and kinds.count("water") == len(kinds) - 1
    wat = next(m for m in asys.molecules if m.kind == "water").molecule
    assert wat.elements == ["O", "H", "H", "EP"] and wat.alpha[3] == 0.0 and len(wat.vsites) == 1
    k = kinds.index("protein")
    tpls = asys.templates({k: amber_template(asys.molecules[k], prm)})
    sim = FlexibleSimulation(asys.system(), tpls, asys.system_positions(), asys.box,
                             MDSettings(cutoff=0.8, skin=0.05, pme_spacing=0.1), dt=0.002, ensemble="nvt",
                             constraints="h-bonds", hmr=asys.hmr({"water": 4.0, "protein": 3.024}),
                             thermostat="bussi", log=None)
    n_wat = kinds.count("water")
    assert sim.vsites.n_sites == n_wat and sim.integ.dof == 3 * (asys.system().n - n_wat) - sim.constraints.nc
    out = sim.minimize(10)                                  # steepest descent keeps the sites placed
    assert out["accepted"] > 0
    sim._advance(20)
    assert np.isfinite(sim.observables()["etot"])
    X = sim.positions_nm()
    assert np.abs(np.linalg.norm(X[sim.vsites.site] - X[sim.vsites.host], axis=1) - 0.0125).max() < 1e-12
    with pytest.raises(NotImplementedError, match="virtual sites"):
        write_pgm_prmtop(asys, "/nonexistent/x.prmtop", tpls)
