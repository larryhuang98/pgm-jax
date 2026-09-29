"""Enhanced sampling (pgm_jax.bias): collective variables and bias forces against finite
differences, metadynamics hills and grids, OPES against a plain re-implementation of PLUMED's
OPES_METAD, the MD hook in both engines (forces, NVE with a static and a growing bias, deposition
inside the compiled loop, files, checkpoints, pressure), the model-potential engine with walkers,
and the analysis tools (c(t), WHAM, histograms)."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_grad import water
from test_hmr import _cluster

from pgm_jax import System
from pgm_jax.bias import OPES, BiasSet, Harmonic, LowerWall, MetaD, StaticBias, UpperWall, cv
from pgm_jax.bias import analysis as A
from pgm_jax.bias.toy import ToyLangevin, double_well, ring
from pgm_jax.md.box import reduce_box
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.units import BAR_PER_KJMOL_NM3, KB

DATA = os.path.join(os.path.dirname(__file__), "data")


def _system(seed=0, n=14):
    rng = np.random.default_rng(seed)
    H = reduce_box(np.array([[1.6, 0.0, 0.0], [0.5, 1.5, 0.0], [-0.4, 0.6, 1.45]]))
    return rng.uniform(size=(n, 3)) @ H, H


def _fd_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, k] += h
            xm[i, k] -= h
            g[i, k] = (float(f(xp)) - float(f(xm))) / (2 * h)
    return g


def _cvs(pos, H):
    return [
        cv.Distance(0, 9),
        cv.Angle(1, 2, 3),
        cv.Dihedral(4, 5, 6, 7),
        cv.Coordination([0, 1, 2, 3], [4, 5, 6, 7, 8], r0=0.5),
        cv.COMDistance([0, 1, 2], [10, 11], masses=np.arange(1.0, 15)),
        cv.RMSD([3, 4, 5, 6, 7], pos[3:8] + 0.05 * np.random.default_rng(1).normal(size=(5, 3))),
        cv.RMSD([3, 4, 5, 6, 7], pos[3:8] + 0.02, align=False),
        cv.Linear([cv.Distance(0, 9), cv.Angle(1, 2, 3)], [1.0, -0.3], offset=0.2),
        cv.Custom(lambda x, H: jnp.sum(x[:3, 0] ** 2), name="custom"),
    ]


def test_cv_gradients_match_finite_differences():
    pos, H = _system()
    for c in _cvs(pos, H):
        g = np.asarray(c.grad(pos, H))
        fd = _fd_grad(lambda x: c(x, H), pos)
        assert np.all(np.isfinite(g))
        assert np.allclose(g, fd, atol=2e-7 * max(1.0, np.abs(fd).max())), (c.name, np.abs(g - fd).max())


def test_cv_values():
    from pgm_jax.md.restraints import dihedral

    pos, H = _system()
    x = np.asarray(pos)
    assert abs(float(cv.Dihedral(4, 5, 6, 7)(pos, None)) - float(dihedral(*x[[4, 5, 6, 7]]))) < 1e-14
    # rational switching: the limit at r = r0 and the value away from it
    s = cv.switching(jnp.asarray([0.25, 0.5, 0.5 + 1e-6, 1.0]), 0.5)
    assert abs(float(s[1]) - 0.5) < 1e-12 and abs(float(s[2]) - 0.5) < 1e-5
    assert abs(float(s[0]) - (1 - 0.5**6) / (1 - 0.5**12)) < 1e-12
    # RMSD after the optimal rotation: a rotated, translated copy has RMSD 0; Kabsch reference
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(6, 3)) * 0.3
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    Q = Q * np.sign(np.linalg.det(Q))
    moved = ref @ Q.T + [0.3, -0.2, 0.1]
    r = cv.RMSD(np.arange(6), ref)
    assert float(r(moved, None)) < 1e-6
    noisy = moved + 0.03 * rng.normal(size=moved.shape)
    a, b = noisy - noisy.mean(0), ref - ref.mean(0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    dd = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, dd]) @ Vt
    kabsch = np.sqrt(np.mean(np.sum((a - b @ R.T) ** 2, 1)))
    assert abs(float(r(noisy, None)) - kabsch) < 1e-10


def test_static_biases_and_walls():
    pos, H = _system()
    d, phi = cv.Distance(0, 9), cv.Dihedral(4, 5, 6, 7)
    s = np.array([float(d(pos, H)), float(phi(pos, H))])
    h = Harmonic([d, phi], at=[0.3, s[1] + 2 * np.pi - 0.1], kappa=[100.0, 20.0], temperature=300.0)
    assert abs(float(h.energy(h.init(), pos, H)) - (50.0 * (s[0] - 0.3) ** 2 + 10.0 * 0.1**2)) < 1e-10
    assert abs(float(h.energy(h.state(at=[s[0], s[1]]), pos, H))) < 1e-20
    uw = UpperWall(d, s[0] - 0.1, 400.0, exp=2, eps=0.5)
    lw = LowerWall(d, s[0] - 0.1, 400.0)
    assert abs(float(uw.energy((), pos, H)) - 400.0 * 0.2**2) < 1e-10 and float(lw.energy((), pos, H)) == 0.0
    st = StaticBias(phi, lambda x: 3.0 * jnp.cos(3.0 * x[0]))
    assert abs(float(st.energy((), pos, H)) - 3.0 * np.cos(3 * s[1])) < 1e-12


def _deposit(b, s_list, step0=1):
    st = b.init()
    for i, s in enumerate(s_list):
        st = b.update(st, jnp.asarray(s, jnp.float64), step0 + i)
    return st


def test_metad_hills_heights_and_periodicity():
    kT = KB * 300.0
    phi, d = cv.Dihedral(0, 1, 2, 3), cv.Distance(0, 3)
    b = MetaD([d, phi], sigma=[0.05, 0.4], height=1.5, pace=10, biasfactor=6.0, temperature=300.0, capacity=4)
    rng = np.random.default_rng(0)
    S = np.stack([0.3 + 0.1 * rng.normal(size=12), rng.uniform(-np.pi, np.pi, 12)], 1)
    st = b.init()
    st = b.reserve(st, 12)
    assert st.heights.shape[0] >= 12

    def V(C, H, s):
        return sum(
            h
            * np.exp(
                -0.5 * (((s[0] - c[0]) / 0.05) ** 2 + ((np.mod(s[1] - c[1] + np.pi, 2 * np.pi) - np.pi) / 0.4) ** 2)
            )
            for c, h in zip(C, H)
        )

    C, Hs = [], []
    for i, s in enumerate(S):
        w = 1.5 * np.exp(-V(C, Hs, s) / (kT * 5.0))
        st = b.update(st, jnp.asarray(s), 10 * (i + 1))
        C.append(s)
        Hs.append(w)
        assert abs(float(st.heights[i]) - w) < 1e-12
    for s in S[:4] + [0.01, 0.2]:
        assert abs(float(b.potential(st, jnp.asarray(s))) - V(C, Hs, s)) < 1e-12
    # a hill at +pi - 0.05 acts at -pi + 0.05 as at +pi - 0.15
    one = b.update(b.init(), jnp.asarray([0.3, np.pi - 0.05]), 10)
    assert (
        abs(
            float(b.potential(one, jnp.asarray([0.3, -np.pi + 0.05])))
            - float(b.potential(one, jnp.asarray([0.3, np.pi - 0.15])))
        )
        < 1e-12
    )
    h = b.hills(st)
    assert list(h["step"]) == list(range(10, 130, 10)) and h["center"].shape == (12, 2)
    # standard metadynamics: constant heights
    b0 = MetaD(d, sigma=0.05, height=1.5, pace=10, biasfactor=None, temperature=300.0)
    st0 = _deposit(b0, [[0.3], [0.3]])
    assert np.allclose(np.asarray(st0.heights[:2]), 1.5)


@pytest.mark.parametrize("periodic", [False, True])
def test_metad_grid_matches_hill_sum(periodic):
    """Hermite-interpolated grid vs the exact hill sum: 1D and 2D, values and gradients, and C1
    continuity of the interpolated bias (forces continuous across grid lines)."""
    c1 = cv.Dihedral(0, 1, 2, 3) if periodic else cv.Distance(0, 1)
    c2 = cv.Dihedral(1, 2, 3, 4) if periodic else cv.Angle(0, 1, 2)
    lo, hi = (-np.pi, np.pi) if periodic else (0.0, 3.0)
    rng = np.random.default_rng(1)
    for cvs, sigma, bins in (([c1], [0.3], 160), ([c1, c2], [0.3, 0.35], [120, 120])):
        b = MetaD(cvs, sigma=sigma, height=2.0, pace=1, biasfactor=8.0, temperature=300.0, grid=(lo, hi, bins))
        d = len(cvs)
        S = rng.uniform(lo + 0.5, hi - 0.5, size=(40, d))
        st = _deposit(b, S)
        pts = rng.uniform(lo + 0.3, hi - 0.3, size=(200, d))
        if periodic:
            pts = rng.uniform(-np.pi - 1, np.pi + 1, size=(200, d))
        vg = jax.vmap(lambda s: b.potential(st, s))(jnp.asarray(pts))
        vh = jax.vmap(lambda s: b.hill_sum(st, s))(jnp.asarray(pts))
        assert float(jnp.max(jnp.abs(vg - vh))) < 2e-3 * float(jnp.max(vh)), float(jnp.max(jnp.abs(vg - vh)))
        gg = jax.vmap(jax.grad(lambda s: b.potential(st, s)))(jnp.asarray(pts))
        gh = jax.vmap(jax.grad(lambda s: b.hill_sum(st, s)))(jnp.asarray(pts))
        assert float(jnp.max(jnp.abs(gg - gh))) < 2e-2 * float(jnp.max(jnp.abs(gh)))
        # continuity of the gradient across a node
        node = b.grid.lo + b.grid.dx * 37
        e = np.zeros(d)
        e[0] = 1e-9
        g = jax.grad(lambda s: b.potential(st, s))
        p = np.full(d, node[0] if d == 1 else 0.0)
        p[0] = node[0]
        assert float(jnp.max(jnp.abs(g(jnp.asarray(p + e)) - g(jnp.asarray(p - e))))) < 1e-5


def _opes_reference(S, sigma0, barrier, kT, periods, compression=1.0, fixed_sigma=False, recursive=True):
    """PLUMED OPES_METAD (update() and calculate()) written out with lists, one CV set per call."""
    d = len(sigma0)
    g = barrier / kT
    pref = 1 - 1 / g
    eps = np.exp(-barrier / pref / kT)
    cut2 = 2 * barrier / pref / kT
    vac = np.exp(-0.5 * cut2)
    P = np.asarray(periods, float)

    def diff(a, b):
        x = np.asarray(a) - np.asarray(b)
        return np.where(P > 0, x - P * np.round(x / np.where(P > 0, P, 1)), x)

    K = []  # [height, center, sigma]
    sw = eps**pref
    sw2 = sw * sw
    Z = 1.0

    def kern(k, s):
        n2 = np.sum((diff(s, k[1]) / k[2]) ** 2)
        return k[0] * (np.exp(-0.5 * n2) - vac) if n2 < cut2 else 0.0

    def bias(s):
        return kT * pref * np.log(sum(kern(k, s) for k in K) / sw / Z + eps)

    out = []
    for s in S:
        s = np.asarray(s, float)
        V = bias(s)
        out.append(V)
        w = np.exp(V / kT)
        sw += w
        sw2 += w * w
        neff = (1 + sw) ** 2 / (1 + sw2)
        sig = np.array(sigma0, float)
        if not fixed_sigma:
            sig = sig * (neff * (d + 2) / 4) ** (-1 / (4 + d))
        h = w * np.prod(np.asarray(sigma0) / sig)

        def merge(t, g):  # g merged into t (moments about t's centre)
            dc = diff(g[1], t[1])
            hm = t[0] + g[0]
            c = t[1] + g[0] / hm * dc
            s2 = (t[0] * t[2] ** 2 + g[0] * (g[2] ** 2 + dc**2)) / hm - (g[0] / hm * dc) ** 2
            return [hm, np.where(P > 0, c - P * np.round(c / np.where(P > 0, P, 1)), c), np.sqrt(s2)]

        def mergeable(center, skip):
            best, bn = None, compression**2
            for i, k in enumerate(K):
                if i == skip:
                    continue
                n2 = np.sum((diff(center, k[1]) / k[2]) ** 2)
                if n2 < bn:
                    best, bn = i, n2
            return best

        best = mergeable(s, None) if compression > 0 else None
        if best is None:
            K.append([h, s.copy(), sig])
        else:
            K[best] = merge(K[best], [h, s, sig])
            if recursive:
                g = best
                t = mergeable(K[g][1], g)
                while t is not None:
                    K[t] = merge(K[t], K[g])
                    del K[g]
                    if t > g:
                        t -= 1
                    g = t
                    t = mergeable(K[g][1], g)
        Z = np.mean([sum(kern(j, k[1]) for j in K) / sw for k in K])
    return out, K, Z, sw


@pytest.mark.parametrize("fixed,recursive", [(False, True), (True, True), (False, False)])
def test_opes_matches_reference_algorithm(fixed, recursive):
    kT = KB * 300.0
    phi, d = cv.Dihedral(0, 1, 2, 3), cv.Distance(0, 3)
    b = OPES(
        [d, phi],
        sigma=[0.05, 0.3],
        pace=1,
        barrier=30.0,
        temperature=300.0,
        capacity=8,
        fixed_sigma=fixed,
        recursive=recursive,
    )
    rng = np.random.default_rng(2)
    S = np.stack(
        [0.3 + 0.08 * rng.normal(size=60), np.mod(2.0 + 0.9 * rng.normal(size=60) + np.pi, 2 * np.pi) - np.pi], 1
    )
    st = b.init()
    assert abs(float(b.potential(st, jnp.asarray(S[0]))) + 30.0) < 1e-10  # V = -barrier at the start
    V = []
    for i, s in enumerate(S):
        st = b.reserve(st, 1)
        V.append(float(b.potential(st, jnp.asarray(s))))
        st = b.update(st, jnp.asarray(s), i + 1)
    Vr, Kr, Zr, swr = _opes_reference(
        S, [0.05, 0.3], 30.0, kT, [0.0, 2 * np.pi], fixed_sigma=fixed, recursive=recursive
    )
    assert np.allclose(V, Vr, atol=1e-9, rtol=0), np.max(np.abs(np.array(V) - Vr))
    assert int(st.nk) == len(Kr) < 60 and (recursive or int(st.merged) == 60 - len(Kr))
    # the same kernels (in another order after recursive deletions)
    kr = np.array(sorted([[k[0], *k[1], *k[2]] for k in Kr]))
    kj = np.array(
        sorted(
            np.concatenate(
                [
                    np.asarray(st.heights[: int(st.nk)])[:, None],
                    np.asarray(st.centers[: int(st.nk)]),
                    np.asarray(st.sigmas[: int(st.nk)]),
                ],
                1,
            ).tolist()
        )
    )
    assert np.allclose(kr, kj, rtol=1e-9, atol=1e-12)
    assert abs(float(st.zed) - Zr) < 1e-10 * Zr and abs(float(st.sum_w) - swr) < 1e-9 * swr
    for s in S[:5]:  # the final bias too
        assert (
            abs(
                float(b.potential(st, jnp.asarray(s)))
                - _opes_reference(
                    np.vstack([S, s]), [0.05, 0.3], 30.0, kT, [0.0, 2 * np.pi], fixed_sigma=fixed, recursive=recursive
                )[0][-1]
            )
            < 1e-9
        )
    assert b.info(st)["kernels"] == len(Kr)
    # a full buffer: new kernels merge into their nearest neighbour instead of being lost
    small = OPES([d, phi], sigma=[0.05, 0.3], pace=1, barrier=30.0, temperature=300.0, capacity=4, compression=0.0)
    st = _deposit(small, S[:10])
    assert int(st.nk) == 4 and int(st.forced) == 6 and np.all(np.asarray(st.heights) > 0)


def test_bias_set_state_io(tmp_path):
    d = cv.Distance(0, 1)
    bs = BiasSet(
        [MetaD(d, 0.05, 1.0, 5, temperature=300.0), OPES(d, 0.05, 5, 20.0, temperature=300.0), Harmonic(d, 0.3, 10.0)],
        colvar=1,
    )
    st = bs.init(log_rows=2)
    st = bs.reserve(st, 50)
    assert st.log.shape[0] >= 51
    x = np.zeros((2, 3))
    for k in range(1, 21):
        x[1, 0] = 0.3 + 0.01 * np.sin(k)
        st = bs.record(st, jnp.asarray(x), None, k)
        st = bs.deposit(st, jnp.asarray(x), None, k)
    rows, st2 = bs.drain(st)
    assert rows.shape == (20, 1 + 3 + 3) and int(st2.nlog) == 0 and list(rows[:3, 0]) == [1, 2, 3]
    assert int(st.parts[0].n) == 4 and int(st.parts[1].counter) == 4
    bs.save(st, str(tmp_path / "b.bias"))
    st3 = bs.load(str(tmp_path / "b.bias"))
    assert abs(float(bs.energy(st3, jnp.asarray(x))) - float(bs.energy(st, jnp.asarray(x)))) < 1e-14


# ----------------------------------------------------------------------------- MD engines
def _water_sim(engine, pos, H, w, settings, **kw):
    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    from pgm_jax.md.simulation import Simulation

    wat = water()
    sys = System([wat] * (len(pos) // 3))
    if engine == "rigid":
        return Simulation(sys, pos, H, settings, log=None, **kw)
    return FlexibleSimulation(sys, [RigidTemplate(wat, w)] * sys.nmol, pos, H, settings, log=None, **kw)


def _cluster_bias(pace=0, height=2.0):
    d, phi = cv.Distance(0, 9), cv.Dihedral(1, 0, 9, 10)
    m = MetaD([d, phi], sigma=[0.03, 0.4], height=height, pace=max(pace, 1), biasfactor=5.0, temperature=300.0)
    if pace == 0:
        m.pace = 0  # static: never deposits
    return m, d, phi


@pytest.mark.parametrize("engine", ["rigid", "atoms"])
def test_md_bias_forces_and_static_nve(engine):
    """A metadynamics bias with pre-deposited hills, held static: the state's forces minus the
    unbiased ones equal -dV/dx (mapped to the bodies for rigid molecules) and the bias forces match
    finite differences of V(s(x)); NVE conserves E_tot while ~20 kJ/mol move in and out of the bias."""
    pos, H, w = _cluster()
    s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
    m, d, phi = _cluster_bias(pace=0)
    s0 = np.array([float(d(pos, H)), float(phi(pos, H))])
    st = m.init()
    rng = np.random.default_rng(0)
    for k in range(30):
        st = m.update(st, jnp.asarray(s0 + [0.03 * rng.normal(), 0.5 * rng.normal()]), k)
    bs = BiasSet([m, UpperWall(d, 0.6, 1000.0)], colvar=10)
    sim = _water_sim(engine, pos, H, w, s, dt=0.0005, thermostat=None, bias=bs)
    sim.set_bias_state(bs.init()._replace(parts=(st, bs.biases[1].init())))
    ref = _water_sim(engine, pos, H, w, s, dt=0.0005, thermostat=None)
    x = sim.rigid.positions(sim.state.dyn.position)
    g = jax.grad(lambda p: bs.energy(sim.state.bias, p, H))(x)
    fd = _fd_grad(lambda p: bs.energy(sim.state.bias, jnp.asarray(p), H), np.asarray(x), h=1e-6)
    assert np.allclose(np.asarray(g), fd, atol=1e-6 * np.abs(fd).max()) and np.abs(fd).max() > 10.0
    mapped = sim.rigid.forces(sim.state.dyn.position, -g) if engine == "rigid" else -g
    for a, b_, c in zip(
        jax.tree_util.tree_leaves(sim.state.dyn.force),
        jax.tree_util.tree_leaves(ref.state.dyn.force),
        jax.tree_util.tree_leaves(mapped),
    ):
        assert np.allclose(np.asarray(a) - np.asarray(b_), np.asarray(c), atol=1e-7 * np.abs(np.asarray(c)).max())
    o = sim.observables()
    assert abs(o["epot"] - ref.observables()["epot"] - o["ebias"]) < 1e-6 and o["ebias"] > 5.0
    E, B = [o["etot"]], [o["ebias"]]
    for _ in range(10):
        sim.advance(40)
        o = sim.observables()
        E.append(o["etot"])
        B.append(o["ebias"])
    assert o["hills"] == 30 and o["bias_work"] == 0.0
    assert max(B) - min(B) > 5.0, B
    assert np.max(np.abs(np.array(E) - E[0])) < 2e-3 * (max(B) - min(B)), (E, B)
    rows = sim.bias_rows()
    assert rows.shape == (40, 1 + 3 + 2) and np.all(np.diff(rows[:, 0]) == 10)


@pytest.mark.parametrize("engine", ["rigid", "atoms"])
def test_md_deposition_in_loop(engine, tmp_path):
    """Hills deposited inside the compiled loop every `pace` steps; after a block the stored forces
    and epot equal a fresh evaluation with the grown bias; NVE: econs (E_tot - work of the updates,
    booked as heat) stays constant while the bias pumps in tens of kJ/mol; COLVAR / HILLS files,
    checkpoint and continuation."""
    from pgm_jax.bias.io import read_table

    pos, H, w = _cluster()
    s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
    m, d, phi = _cluster_bias(pace=10, height=1.0)
    bs = BiasSet([m, Harmonic(d, 0.30, 2000.0)], colvar=5)
    sim = _water_sim(engine, pos, H, w, s, dt=0.0005, thermostat=None, bias=bs)
    o0 = sim.observables()
    prefix = str(tmp_path / "md")
    sim.run(200, report_every=100, checkpoint_every=100, prefix=prefix)
    o = sim.observables()
    assert o["hills"] == 20 and o["bias_work"] > 5.0
    assert abs(o["econs"] - o0["econs"]) < 2e-3 * o["bias_work"], (o0["econs"], o["econs"], o["bias_work"])
    assert abs(o["etot"] - o0["etot"] - o["bias_work"]) < 2e-3 * o["bias_work"]
    st = sim.state
    fresh = sim.integ.forces(st, False)
    for a, b_ in zip(jax.tree_util.tree_leaves(st.dyn.force), jax.tree_util.tree_leaves(fresh.dyn.force)):
        assert np.allclose(np.asarray(a), np.asarray(b_), atol=1e-7 * np.abs(np.asarray(b_)).max())
    assert abs(float(st.epot - fresh.epot)) < 1e-8
    meta, cvs = read_table(prefix + ".colvar")
    assert len(cvs["step"]) == 40 and cvs["step"][0] == 5 and "bias0_metad" in cvs
    _, hills = read_table(prefix + ".hills")
    assert np.array_equal(hills["step"], np.arange(10, 210, 10))
    assert np.allclose(hills["height"], np.asarray(st.bias.parts[0].heights[:20]))
    # the bias of step t is recorded before the hill of step t
    assert cvs["bias0_metad"][1] == 0.0 and cvs["bias0_metad"][2] > 0.0  # step 10: before the first hill
    # continuation from the checkpoint reproduces the next block
    a = sim.integ.run(sim.state, 20)
    sim2 = _water_sim(
        engine,
        pos,
        H,
        w,
        s,
        dt=0.0005,
        thermostat=None,
        bias=BiasSet([_cluster_bias(10, 1.0)[0], Harmonic(d, 0.30, 2000.0)], colvar=5),
    )
    sim2.load_checkpoint(prefix + ".chk")
    b_ = sim2.integ.run(sim2.state, 20)
    assert int(a.bias.parts[0].n) == int(b_.bias.parts[0].n) == 22
    assert abs(float(a.epot) - float(b_.epot)) < 1e-6
    st3 = bs.load(prefix + ".bias")
    assert int(st3.parts[0].n) == 20


def test_md_opes_nvt_pressure_and_mts():
    """OPES in NVT (Bussi) through the rigid engine and a bias with multiple time stepping (bias in the
    slow group): runs, deposits, and the pressure includes the bias's strain derivative."""
    from pgm_jax.md.mts import MTS
    from pgm_jax.md.restraints import molecular_strain

    pos, H, w = _cluster()
    s = MDSettings().replace(precision="double", dipole_tol=1e-8, cutoff=1.2, skin=0.1, lj_lrc=False)
    d = cv.Distance(0, 9)
    op = OPES(d, sigma=0.02, pace=20, barrier=15.0)
    sim = _water_sim(
        "rigid", pos, H, w, s, dt=0.001, thermostat="bussi", temperature=300.0, bias=[op, UpperWall(d, 0.7, 500.0)]
    )
    sim.advance(200)
    o = sim.observables()
    assert o["kernels"] >= 1 and o["neff"] > 1.0
    st = sim.state
    x = sim.rigid.positions(st.dyn.position)
    W = molecular_strain(
        lambda p, h: sim.integ.bias.energy(st.bias, p, h), x, st.box, sim.ff.mol, sim.ff.masses, sim.sys.nmol
    )
    p_with = sim.pressure()
    ref = _water_sim("rigid", pos, H, w, s, dt=0.001, temperature=300.0)
    ref.state = st.set(bias=None)
    dP = -float(jnp.trace(W)) / (3.0 * float(jnp.linalg.det(st.box))) * BAR_PER_KJMOL_NM3
    assert abs(p_with - ref.pressure() - dP) < 1e-6 * max(1.0, abs(dP))
    # multiple time stepping: the bias is part of the slow force
    m, _, _ = _cluster_bias(pace=5, height=1.0)
    sim = _water_sim("atoms", pos, H, w, s, dt=0.002, thermostat=None, bias=m, mts=MTS(inner=2, split="bonded"))
    E0 = sim.observables()["econs"]
    sim.advance(40)
    o = sim.observables()
    assert o["hills"] == 8 and abs(o["econs"] - E0) < 0.05 * max(o["bias_work"], 1.0), (E0, o)


def test_flexible_peptide_dihedral_bias():
    """Solvated peptide (flexible, h-bond constraints): metadynamics on its backbone phi/psi (grid)
    inside the atom engine; the CV follows ensemble.backbone_torsions."""
    from pgm_jax.fit.reweighting import backbone_torsions
    from pgm_jax.md.flexible import FlexibleSimulation
    from pgm_jax.protein import amber_template, load_amber

    prm, crd = os.path.join(DATA, "pep_wat.prmtop"), os.path.join(DATA, "pep_wat.inpcrd")
    asys = load_amber(prm, crd)
    prot = asys.molecules[0]
    top = prot.spec.top
    tpl = amber_template(prot, prm)
    q = np.asarray(top.cmaps)[0]  # C-N-CA-C-N of residue 1
    phi, psi = cv.Dihedral(*q[:4]), cv.Dihedral(*q[1:])
    m = MetaD([phi, psi], sigma=0.35, height=1.0, pace=5, biasfactor=6.0, grid=(-np.pi, np.pi, 72))
    s = MDSettings().replace(dipole_tol=1e-5, cutoff=0.8, skin=0.1)
    sim = FlexibleSimulation(
        asys.system(),
        asys.templates({0: tpl}),
        asys.system_positions(),
        asys.box,
        s,
        dt=0.001,
        temperature=300.0,
        thermostat="bussi",
        constraints="h-bonds",
        bias=m,
        log=None,
    )
    sim.advance(20)
    o = sim.observables()
    assert o["hills"] == 4 and np.isfinite(o["econs"])
    x = sim.positions()[: prot.n][None]
    ph, ps = backbone_torsions(x, top)
    v = sim.cv_values()[0]
    assert (
        abs(np.cos(v[0]) - np.cos(float(np.asarray(ph)[0, 0]))) < 1e-9
        and abs(np.cos(v[1]) - np.cos(float(np.asarray(ps)[0, 0]))) < 1e-9
    )


def test_remd_refuses_dynamic_bias():
    from pgm_jax.md.remd import ReplicaExchange

    pos, H, w = _cluster()
    s = MDSettings().replace(precision="double", dipole_tol=1e-8, cutoff=1.2, skin=0.1, lj_lrc=False)
    m, _, _ = _cluster_bias(pace=10)
    sim = _water_sim("rigid", pos, H, w, s, dt=0.001, temperature=300.0, bias=m)
    with pytest.raises(NotImplementedError):
        ReplicaExchange(sim, [300.0, 320.0], exchange_every=10)


# ----------------------------------------------------------------------------- model engine and analysis
def test_toy_walkers_and_analysis():
    """Double well: shared-bias walkers deposit W hills per pace; independent walkers each their own;
    c(t) against a direct evaluation; the reweighted histogram of the biased run and the bias
    estimate agree with the exact FES to a few kJ/mol after a short run."""
    U = double_well(barrier=15.0)
    x = cv.Component(0, 0)
    b = MetaD(x, sigma=0.1, height=1.0, pace=50, biasfactor=8.0, grid=(-2.5, 2.5, 250))
    sh = ToyLangevin(
        U, [[-1.0, 0, 0]], mass=10.0, temperature=300.0, dt=0.005, gamma=1.0, bias=b, walkers=4, shared=True, seed=1
    )
    out = sh.run(1000, sample=50)
    assert int(sh.state.bias.parts[0].n) == 4 * 20 and out["cv"].shape == (20, 4, 1)
    ind = ToyLangevin(
        U,
        [[-1.0, 0, 0]],
        mass=10.0,
        temperature=300.0,
        dt=0.005,
        gamma=1.0,
        bias=MetaD(x, sigma=0.1, height=1.0, pace=50, biasfactor=8.0, grid=(-2.5, 2.5, 250)),
        walkers=3,
        shared=False,
        seed=2,
    )
    out = ind.run(60000, sample=50)
    assert np.all(np.asarray(ind.state.bias.parts[0].n) == 1200)
    kT = KB * 300.0
    ax = np.linspace(-1.5, 1.5, 61)
    Fex = U.fes(ax)
    for wk in range(3):
        st = jax.tree_util.tree_map(lambda a: a[wk], ind.state.bias.parts[0])
        hills = ind.bias.biases[0].hills(st)
        steps, ct = A.metad_ct(hills, 8.0, kT, [0.0], np.linspace(-2.5, 2.5, 501)[:, None])
        # direct c(t) of the last hill
        V = np.asarray(A.bias_on_grid(ind.bias.biases[0], st, np.linspace(-2.5, 2.5, 501)[:, None]))
        direct = kT * np.log(np.sum(np.exp(8 / 7 * V / kT)) / np.sum(np.exp(V / 7 / kT)))
        assert abs(ct[-1] - direct) < 2e-3 * abs(direct)
        lw = A.ct_weights(out["step"], out["bias"][:, wk, 0], steps, ct, kT)
        Fh = A.histogram_fes(out["cv"][:, wk, 0], lw, [ax], kT)
        Fb = A.fes_from_bias(ind.bias.biases[0], st, ax[:, None])
        m = Fex < 12.0
        assert A.align_rmsd(Fh, Fex, m)[0] < 4.0 and A.align_rmsd(Fb, Fex, m)[0] < 4.0


def test_toy_ring_periodic_opes():
    """A particle on a ring (periodic CV theta): OPES flattens the angular barriers (both minima
    visited) and its FES estimate is within a few kJ/mol of the exact one after a short run."""
    U = ring()
    th = cv.Custom(lambda x, H: jnp.arctan2(x[0, 1], x[0, 0]), period=2 * np.pi, name="theta")
    b = OPES(th, sigma=0.2, pace=50, barrier=20.0)
    sim = ToyLangevin(U, [[1.0, 0, 0]], mass=10.0, temperature=300.0, dt=0.004, gamma=1.0, bias=b, walkers=2, seed=3)
    out = sim.run(50000, sample=50)
    ax = A.periodic_axis(60)
    Fex = U.fes(ax)
    Fex -= Fex.min()
    for wk in range(2):
        st = jax.tree_util.tree_map(lambda a: a[wk], sim.state.bias.parts[0])
        F = A.fes_from_bias(b, st, ax[:, None])
        assert A.align_rmsd(F, Fex, Fex < 15.0)[0] < 3.0
        Fh = A.histogram_fes(
            out["cv"][:, wk, 0],
            A.opes_weights(out["bias"][:, wk, 0], KB * 300.0),
            [ax],
            KB * 300.0,
            periods=[2 * np.pi],
        )
        assert A.align_rmsd(Fh, Fex, Fex < 15.0)[0] < 4.0


def test_wham_and_histogram():
    """WHAM recovers a known 1D FES from exact samples of harmonic umbrella windows."""
    rng = np.random.default_rng(0)
    kT = 2.5
    ax = np.linspace(-1.5, 1.5, 61)
    F = 8.0 * (ax**2 - 1) ** 2
    samples, centers = [], np.linspace(-1.4, 1.4, 15)
    fine = np.linspace(-2.0, 2.0, 4001)
    Ff = 8.0 * (fine**2 - 1) ** 2
    for c in centers:
        p = np.exp(-(Ff + 0.5 * 300.0 * (fine - c) ** 2) / kT)
        samples.append(rng.choice(fine, size=20000, p=p / p.sum()))
    Fw, f = A.wham(samples, centers, np.full(15, 300.0), ax, kT)
    assert A.align_rmsd(Fw, F, F < 10)[0] < 0.3
    # an unweighted histogram of Boltzmann samples
    p = np.exp(-Ff / kT)
    Fh = A.histogram_fes(rng.choice(fine, size=400000, p=p / p.sum()), None, [ax], kT)
    assert A.align_rmsd(Fh, F, F < 6)[0] < 0.2


@pytest.mark.parametrize("shared", [False, True])
def test_walkers(shared, tmp_path):
    """Walkers of the rigid engine in one vmapped program: independent (each its own bias; walker 0
    reproduces a single simulation with the same start) or sharing one bias (W hills per pace, the
    state's forces equal a fresh evaluation with the shared bias); files and checkpoint."""
    from pgm_jax.bias.io import read_table
    from pgm_jax.bias.walkers import Walkers

    pos, H, w = _cluster()
    s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
    m, d, phi = _cluster_bias(pace=10, height=1.0)
    sim = _water_sim(
        "rigid", pos, H, w, s, dt=0.001, thermostat="bussi", temperature=300.0, bias=BiasSet([m], colvar=5)
    )
    W = 3
    wk = Walkers(sim, W, shared=shared, seed=4)
    prefix = str(tmp_path / "wk")
    wk.run(100, report_every=50, checkpoint_every=100, prefix=prefix)
    if shared:
        assert int(wk.S.bias.parts[0].n) == W * 10
        _, hills = read_table(prefix + ".hills")
        assert len(hills["step"]) == W * 10
        for k in range(W):
            st = wk.state(k)
            fresh = sim.integ.forces(st, False)
            for a, b_ in zip(jax.tree_util.tree_leaves(st.dyn.force), jax.tree_util.tree_leaves(fresh.dyn.force)):
                assert np.allclose(np.asarray(a), np.asarray(b_), atol=1e-7 * np.abs(np.asarray(b_)).max())
    else:
        assert np.all(np.asarray(wk.S.bias.parts[0].n) == 10)
        # walker 0 vs the same state advanced alone
        one = _water_sim(
            "rigid",
            pos,
            H,
            w,
            s,
            dt=0.001,
            thermostat="bussi",
            temperature=300.0,
            bias=BiasSet([_cluster_bias(pace=10, height=1.0)[0]], colvar=5),
        )
        wk2 = Walkers(sim, W, shared=False, seed=4)
        st0 = wk2.state(0)
        one.state = st0.set(nbr=one.state.nbr)
        one.advance(100)
        wk2.advance(100)
        assert abs(float(one.state.epot) - float(np.asarray(wk2.S.epot)[0])) < 1e-6
    for k in range(W):
        _, cvs = read_table(f"{prefix}_w{k:02d}.colvar")
        assert np.array_equal(cvs["step"], np.arange(5, 105, 5))
    E = wk.bias_energies()
    wk3 = Walkers(sim, W, shared=shared, seed=9)
    wk3.load_checkpoint(prefix + ".walkers.chk")
    assert np.allclose(wk3.bias_energies(), E, atol=1e-10)


def test_reserve_many_equal_shapes():
    """Independent walkers whose buffers grow differently are padded to common sizes (stackable)."""
    d = cv.Distance(0, 1)
    bs = BiasSet(
        [
            OPES(d, 0.01, 1, 20.0, temperature=300.0, capacity=4, compression=0.0),
            MetaD(d, 0.05, 1.0, 1, temperature=300.0, capacity=4),
        ],
        colvar=1,
    )
    a = bs.init(log_rows=2)
    b = bs.init(log_rows=2)
    x = np.zeros((2, 3))
    for k in range(1, 4):
        x[1, 0] = 0.3 + 0.1 * k
        a = bs.deposit(a, jnp.asarray(x), None, k)
    out = bs.reserve_many([a, b], 10)
    la, lb = jax.tree_util.tree_leaves(out[0]), jax.tree_util.tree_leaves(out[1])
    assert all(p.shape == q.shape for p, q in zip(la, lb))
    assert out[0].parts[0].heights.shape[0] > 4 and int(out[0].parts[0].nk) == 3
    stacked = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *out)
    assert (
        abs(
            float(bs.energy(jax.tree_util.tree_map(lambda v: v[0], stacked), jnp.asarray(x)))
            - float(bs.energy(a, jnp.asarray(x)))
        )
        < 1e-12
    )
