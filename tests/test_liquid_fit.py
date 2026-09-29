"""Ensemble-gradient fitting (pgm_jax/fit): per-frame derivatives against finite differences with
the induced dipoles re-solved, the fluctuation formulas, gas-phase properties, the LM step and the
parameter covariance, and one iteration of LiquidFit end to end."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_grad import water  # noqa: E402
from test_md import settings, small_box  # noqa: E402

from pgm_jax import ElecChannel, System  # noqa: E402
from pgm_jax.fit import (  # noqa: E402
    FrameAnalyzer,
    GasPhase,
    LiquidSamples,
    Objective,
    Param,
    ParameterSpace,
    RDFSpec,
    Target,
)
from pgm_jax.fit.estimators import KB, KCAL  # noqa: E402
from pgm_jax.fit.optimize import Estimate  # noqa: E402
from pgm_jax.md.dipoles import CellDipole  # noqa: E402
from pgm_jax.md.forcefield import PGMForceField  # noqa: E402
from pgm_jax.units import DEBYE_E_NM, KE  # noqa: E402

QTY = ["q", "cov", "alpha", "radius", "lj_r", "lj_eps"]


def _analyzer(tol=1e-12, **kw):
    sys, pos, H = small_box(3)
    space = ParameterSpace.scales(sys.table, QTY)
    st = settings(pme_grid=(32, 32, 32), pme_order=6, **kw)
    an = FrameAnalyzer(sys, H, st, space, rdf=RDFSpec.by_type(sys, "OW", rmax=0.8, nbins=40), tol=tol, chunk=2)
    return sys, pos, H, space, an


def test_parameter_space_scales_and_keys():
    sys, pos, H = small_box(0)
    sp = ParameterSpace.scales(sys.table, QTY)
    p0 = sys.table.initial()
    th = np.array([0.1, -0.2, 0.05, 0.02, -0.01, 0.3])
    P = sp(th)
    assert np.allclose(P["q"], p0["q"] * np.exp(0.1))
    assert np.allclose(P["cov"], p0["cov"] * np.exp(-0.2))
    assert np.allclose(P["lj_sqrt_eps"] ** 2, p0["lj_sqrt_eps"] ** 2 * np.exp(0.3))
    assert np.allclose(P["lj_rmin_half"], p0["lj_rmin_half"] * np.exp(-0.01))
    sp2 = ParameterSpace(sys.table, [Param("alpha", keys=["OW"]), Param("q", "shift", keys=["WAT:OW"])])
    P2 = sp2(np.array([np.log(2.0), 0.01]))
    ia = sys.table.index("alpha", ["OW"])[0]
    iq = sys.table.index("q", ["WAT:OW"])[0]
    assert np.isclose(P2["alpha"][ia], 2 * p0["alpha"][ia]) and np.isclose(P2["q"][iq], p0["q"][iq] + 0.01)
    others = np.delete(np.arange(len(p0["alpha"])), ia)
    assert np.allclose(P2["alpha"][others], p0["alpha"][others])


def test_frame_values_match_the_md_force_field():
    """U, M and alpha_cell of FrameAnalyzer equal PGMForceField / CellDipole at the same frame."""
    sys, pos, H, space, an = _analyzer()
    th = np.zeros(space.n)
    out = an.frame(th, pos, H)
    ff = PGMForceField(sys, H, settings(pme_grid=(32, 32, 32), pme_order=6))
    idx = ff.rows_for(pos, H)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    assert abs(out["U"] - float(res.energy["total"])) < 1e-8 * abs(float(res.energy["total"]))
    cd = CellDipole(ff)
    M = np.asarray(cd.components(pos, H, res.induction.mu)).sum(0)
    assert np.allclose(out["M"], M, atol=1e-10)
    a = np.trace(np.asarray(cd.polarizability(pos, H, idx, tol=1e-12))) / 3.0
    assert abs(out["alpha"] - a) < 1e-9 * a
    D = np.linalg.norm(np.asarray(cd.molecular(pos, H, res.induction.mu)), axis=1).mean()
    assert abs(out["D"] - D) < 1e-10


def test_frame_derivatives_against_finite_differences():
    """dU, dM, d alpha_cell, dD (explicit, at fixed nuclei) vs central differences of the
    re-solved values, float64, all six scale factors."""
    sys, pos, H, space, an = _analyzer()
    th = np.array([0.02, -0.03, 0.04, 0.01, 0.005, -0.02])
    out = an.frame(th, pos, H)
    h = 1e-5
    for j in range(space.n):
        e = np.zeros(space.n)
        e[j] = h
        fp, fm = an.frame(th + e, pos, H, grad=False), an.frame(th - e, pos, H, grad=False)
        for key, dkey, sl in (("U", "dU", ()), ("M", "dM", (slice(None),)), ("alpha", "dalpha", ()), ("D", "dD", ())):
            fd = (np.asarray(fp[key]) - np.asarray(fm[key])) / (2 * h)
            an_ = np.asarray(out[dkey])[sl + (j,)]
            scale = max(np.max(np.abs(fd)), 1e-12 * (1 + np.max(np.abs(np.asarray(out[key])))))
            assert np.max(np.abs(an_ - fd)) <= 2e-6 * scale + 1e-10, (space.names[j], key, an_, fd)


def test_mixed_precision_derivatives_close_to_double():
    sys, pos, H, space, an = _analyzer()
    _, _, _, _, am = _analyzer(tol=1e-6, precision="mixed")
    th = np.zeros(space.n)
    a, b = an.frame(th, pos, H), am.frame(th, pos, H)
    for k in ("dM", "dalpha", "dD"):
        assert np.max(np.abs(a[k] - b[k])) < 2e-4 * np.max(np.abs(a[k])), k
    assert np.max(np.abs(a["dU"] - b["dU"])) < 1e-4 * np.max(np.abs(a["dU"]))


def test_batched_frames_equal_single_frames():
    sys, pos, H, space, an = _analyzer()
    rng = np.random.default_rng(0)
    frames = [(pos + 0.002 * rng.normal(size=pos.shape), H, None) for _ in range(3)]
    th = np.zeros(space.n)
    out = an.analyze(th, frames)
    for k, f in enumerate(frames):
        one = an.frame(th, *f)
        for key in ("U", "M", "dU", "dM", "alpha", "rdf"):
            assert np.allclose(out[key][k], one[key], rtol=1e-9, atol=1e-10), key
    assert out["rdf"].shape == (3, 40) and np.all(out["converged"])


def _synthetic_samples(F=400, n=3, seed=0, T=300.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(F, n))
    fr = {
        "U": -1000 + 30 * x[:, 0] + rng.normal(size=F),
        "dU": 50 * x + rng.normal(size=(F, n)),
        "V": 15.0 + 0.1 * rng.normal(size=F) + 0.05 * x[:, 1],
        "M": rng.normal(size=(F, 3)) * 0.3 + 0.1 * x[:, :1],
        "dM": rng.normal(size=(F, 3, n)) * 0.1,
        "alpha": 0.6 + 0.01 * rng.normal(size=F),
        "dalpha": 0.02 + 0.001 * rng.normal(size=(F, n)),
        "D": 0.05 + 0.001 * rng.normal(size=F),
        "dD": 0.01 + 0.001 * rng.normal(size=(F, n)),
    }
    return fr, LiquidSamples(fr, T, 512, 9000.0, nblocks=8)


def test_jacobians_are_the_fluctuation_formulas():
    fr, s = _synthetic_samples()
    beta = 1.0 / (KB * s.T)
    th0 = np.zeros(3)
    gas = lambda th: {"gas_energy": 5.0 + 2.0 * th[0], "gas_dipole": 1.8 + th[1], "gas_polarizability": 1.4 + th[2]}
    obj = Objective(
        [
            Target("density", 1.0),
            Target("hvap", 10.0),
            Target("eps", 70.0),
            Target("liquid_dipole", fit=False),
            Target("gas_dipole", 1.855),
        ],
        ParameterSpace(_space_table(), [Param("q"), Param("cov"), Param("alpha")]),
        gas=gas,
    )
    est = obj.estimate(s, th0)
    dU = fr["dU"]
    cov = lambda a: np.mean(
        (a - a.mean(0))[:, None] * (dU - dU.mean(0))
        if a.ndim == 1
        else (a - a.mean(0))[:, :, None] * (dU - dU.mean(0))[:, None, :],
        axis=0,
    )
    V, M = fr["V"], fr["M"]
    rho = 9000.0 / V * 1.66053906660e-3
    d_rho = -beta * cov(rho)
    d_U = dU.mean(0) - beta * cov(fr["U"])
    d_h = (np.array([2.0, 0, 0]) - d_U / 512) / KCAL
    M2 = np.sum(M * M, 1)
    dM2 = 2 * np.einsum("fc,fcn->fn", M, fr["dM"])
    aV = fr["alpha"] / V
    kT = KB * s.T
    c = 4 * np.pi * KE / (3 * kT)
    dM2avg = dM2.mean(0) - beta * cov(M2)
    dMavg = fr["dM"].mean(0) - beta * cov(M)
    dV = -beta * cov(V)
    fl_num = M2.mean() - np.sum(M.mean(0) ** 2)
    d_eps = 4 * np.pi * ((fr["dalpha"] / V[:, None]).mean(0) - beta * cov(aV)) + c * (
        (dM2avg - 2 * M.mean(0) @ dMavg) / V.mean() - fl_num / V.mean() ** 2 * dV
    )
    eps = 1 + 4 * np.pi * aV.mean() + c * fl_num / V.mean()
    d_D = (fr["dD"].mean(0) - beta * cov(fr["D"])) / DEBYE_E_NM
    assert np.isclose(est.y[2], eps) and np.isclose(est.y[0], rho.mean())
    for row, ref in ((0, d_rho), (1, d_h), (2, d_eps), (3, d_D), (4, np.array([0, 1.0, 0]))):
        assert np.allclose(est.J[row], ref, rtol=1e-8, atol=1e-12), (row, est.J[row], ref)
    assert est.cov_y.shape == (5, 5) and est.err[4] < 1e-10 and np.all(est.err[:4] > 0)


def _space_table():
    return System([water()]).table


def test_linear_reweighting_prediction_and_n_eff():
    fr, s = _synthetic_samples()
    d = np.array([0.001, 0.0, 0.0])
    avg = s.averages(jnp.asarray(d))
    w = np.exp(-s.beta * (np.asarray(s.dU) @ d))
    w /= w.sum()
    assert np.isclose(float(avg["U"]), np.sum(w * (fr["U"] + fr["dU"] @ d)))
    assert np.isclose(s.n_eff(d), 1.0 / np.sum(w * w))
    assert s.n_eff(np.zeros(3)) == pytest.approx(s.F)


def test_gas_phase_properties_and_the_md_monomer_energy():
    """GasPhase energy = the MD engine's energy of one molecule in a large box (images of a
    neutral molecule add ~1e-4 kJ/mol); gradients of energy, dipole and polarizability vs finite
    differences; values against the gas-phase ElecChannel."""
    m = water()
    t = np.radians(104.52 / 2)
    x = np.array(
        [[0, 0, 0], [0.09572 * np.sin(t), 0.09572 * np.cos(t), 0], [-0.09572 * np.sin(t), 0.09572 * np.cos(t), 0]]
    )
    sys = System([m])
    space = ParameterSpace.scales(sys.table, ["q", "cov", "alpha", "radius"])
    gp = GasPhase(m, x, sys.table, space)
    th = np.array([0.01, -0.02, 0.03, 0.01])
    v = gp.values(th)
    e, aux = ElecChannel().energy(jnp.asarray(x - x.mean(0)), sys, space(th))
    assert np.isclose(v["gas_energy"], float(sum(e.values())))
    Hb = np.eye(3) * 6.0
    ff = PGMForceField(sys, Hb, settings(cutoff=2.5, skin=0.0, ewald_beta=1.6, pme_grid=(48, 48, 48)))
    y = x - x.mean(0) + 3.0
    res = jax.jit(ff.compute)(y, Hb, ff.rows_for(y, Hb), ff.init_induction(), space(th))
    assert abs(float(res.energy["total"]) - v["gas_energy"]) < 2e-3
    J = gp.jacobian(th)
    h = 1e-6
    for j in range(4):
        e_ = np.zeros(4)
        e_[j] = h
        vp, vm = gp.values(th + e_), gp.values(th - e_)
        for k in J:
            assert abs(J[k][j] - (vp[k] - vm[k]) / (2 * h)) < 1e-6 * (1 + abs(J[k][j])), (k, j)
    assert v["gas_dipole"] > 0.1 and v["gas_polarizability"] > 0


def _linear_estimate(obj, theta, y, J, cov, target, tol):
    m = len(y)
    return Estimate(
        np.asarray(theta, float),
        [f"y{i}" for i in range(m)],
        np.asarray(y, float),
        np.asarray(J, float),
        np.asarray(cov, float),
        np.zeros_like(J),
        np.asarray(target, float),
        np.asarray(tol, float),
        np.ones(m),
        np.ones(m, bool),
    )


def test_lm_step_trust_region_and_covariance_calibration():
    """Linear model y = y0 + J theta + noise: the unconstrained step lands on the weighted least-
    squares solution; the trust region bounds |d / sigma_prior|; C_theta matches the spread of fits
    over noise realisations (and equals (J^T S^-1 J)^-1 without tolerances and with a wide prior)."""
    table = _space_table()
    space = ParameterSpace.scales(table, ["q", "alpha"], prior_sigma=1e3)
    obj = Objective([], space)
    rng = np.random.default_rng(3)
    J = np.array([[1.0, 0.3], [0.2, -2.0], [0.5, 0.5]])
    sig = np.array([0.1, 0.2, 0.05])
    Sy = np.diag(sig**2)
    Sy[0, 2] = Sy[2, 0] = 0.3 * sig[0] * sig[2]
    true = np.array([0.03, -0.02])
    t = J @ true
    L = np.linalg.cholesky(Sy)
    ths = []
    for _ in range(4000):
        y = L @ rng.normal(size=3)  # measured at theta = 0
        est = _linear_estimate(obj, np.zeros(2), y, J, Sy, t, np.zeros(3))
        ths.append(obj.step(est, radius=np.inf)["delta"])
    ths = np.array(ths)
    C = obj.covariance(est)["C_theta"]
    W = np.diag(1 / sig**2)
    C_ref = np.linalg.inv(J.T @ W @ J) @ J.T @ W @ Sy @ W @ J @ np.linalg.inv(J.T @ W @ J)
    assert np.allclose(C, C_ref, rtol=1e-4)
    assert np.allclose(np.cov(ths.T), C, rtol=0.1, atol=1e-9)
    assert np.allclose(ths.mean(0), true, atol=4 * np.sqrt(np.diag(C) / len(ths)))
    st = obj.step(est, radius=1e-5)
    assert st["size"] <= 1e-5 * (1 + 1e-6) and st["at_boundary"]
    est0 = _linear_estimate(obj, np.zeros(2), np.zeros(3), J, np.diag(sig**2), t, np.zeros(3))
    assert np.allclose(obj.covariance(est0)["C_theta"], np.linalg.inv(J.T @ W @ J), rtol=1e-4)


def test_one_iteration_of_liquid_fit(tmp_path):
    """End to end on the small box (CPU, a few frames): JSON record with observables, Jacobian,
    step, predictions and uncertainties; resume picks up the next parameters."""
    from pgm_jax.fit.liquid import LiquidFit

    sys, pos, H = small_box(1)
    space = ParameterSpace.scales(sys.table, ["q", "lj_eps"])
    k = [i for i, m in enumerate(sys.molecules) if m.name == "WAT"][0]
    gas = GasPhase(sys.molecules[k], pos[sys.atom_slice(k)], sys.table, space)
    rdf = RDFSpec.by_type(sys, "OW", rmax=0.8, nbins=16)
    obj = Objective(
        [
            Target("density", 1.0, 0.01),
            Target("hvap", 10.0, 0.1),
            Target("eps", 70.0, 5.0),
            Target("gas_dipole", 2.0, 0.01),
            Target("rdf", None, fit=False),
        ],
        space,
        gas=gas,
        rdf_r=rdf.r,
    )
    st = settings(
        cutoff=0.6,
        skin=0.1,
        pme_grid=(16, 16, 16),
        pme_order=6,
        ewald_beta=5.0,
        dipole_tol=1e-6,
        precision="double",
        peek=0.65,
        max_iter=100,
    )
    fit = LiquidFit(
        sys,
        pos,
        H,
        space,
        obj,
        settings=st,
        dt=0.001,
        equil_ps=0.02,
        prod_ps=0.08,
        every_ps=0.01,
        chunk=4,
        nblocks=4,
        rdf=rdf,
        prefix=str(tmp_path / "fit"),
        exact_every=2,
        bootstrap=5,
        log=None,
        tol=1e-8,
    )
    th = fit.run(np.zeros(2), 1)
    d = json.load(open(tmp_path / "fit.json"))
    r = d["records"][0]
    assert r["info"]["frames"] == 8 and len(r["estimate"]["y"]) == 4 + 16
    assert np.allclose(th, r["next_theta"]) and np.all(np.isfinite(r["uq"]["theta_err"]))
    assert r["step"]["n_eff_exact"] > 0 and len(r["step"]["y_exact"]) == 20
    fit2 = LiquidFit(sys, pos, H, space, obj, settings=st, dt=0.001, rdf=rdf, prefix=str(tmp_path / "fit"), log=None)
    assert np.allclose(fit2.resume(), th)


def test_nvt_replicas_are_ordered_by_replica(tmp_path):
    """NVT with batched replicas: frames of each replica contiguous (blocks never mix replicas),
    replicas independent (different trajectories), estimates finite."""
    from pgm_jax.fit.liquid import LiquidFit

    sys, pos, H = small_box(2)
    space = ParameterSpace.scales(sys.table, ["q"])
    obj = Objective([Target("energy", None, fit=False), Target("eps", None, fit=False)], space)
    st = settings(
        cutoff=0.6,
        skin=0.1,
        pme_grid=(16, 16, 16),
        pme_order=6,
        ewald_beta=5.0,
        dipole_tol=1e-6,
        precision="double",
        peek=0.65,
        max_iter=100,
    )
    fit = LiquidFit(
        sys,
        pos,
        H,
        space,
        obj,
        settings=st,
        dt=0.001,
        equil_ps=0.01,
        prod_ps=0.03,
        every_ps=0.01,
        chunk=2,
        nblocks=2,
        prefix=str(tmp_path / "rep"),
        bootstrap=0,
        log=None,
        tol=1e-8,
        ensemble="nvt",
        replicas=2,
        equil_rep_ps=0.01,
        fixed=True,
    )
    frames, _, info = fit.simulate(np.zeros(1), seed=3)
    assert info["frames"] == 6 and frames["U"].shape == (6,)
    assert np.allclose(frames["V"], frames["V"][0])  # NVT
    a, b = frames["U"][:3], frames["U"][3:]
    assert not np.allclose(a, b)  # independent replicas
    assert np.all(np.abs(np.diff(a)) > 0)


def test_rdf_histogram_matches_numpy():
    sys, pos, H, space, an = _analyzer()
    out = an.frame(np.zeros(space.n), pos, H, grad=False)
    s = an.rdf
    o = pos[s.a]
    d = o[:, None, :] - o[None, :, :]
    Hn = np.asarray(H)
    for c in (2, 1, 0):  # minimum image, reduced box
        d = d - np.round(d[..., c : c + 1] / Hn[c, c]) * Hn[c]
    r = np.linalg.norm(d, axis=-1)[np.triu_indices(len(o), 1)]
    cnt, edges = np.histogram(r, bins=s.nbins, range=(0, s.rmax))
    V = abs(np.linalg.det(Hn))
    g = cnt * V / (len(o) * (len(o) - 1) / 2) / (4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3))
    assert np.allclose(out["rdf"], g, atol=1e-9)


def test_thermal_expansion_and_compressibility_gradients():
    fr, s = _synthetic_samples()
    obj = Objective(
        [Target("alpha_p", None, fit=False), Target("kappa_t", None, fit=False)],
        ParameterSpace(_space_table(), [Param("q"), Param("cov"), Param("alpha")]),
    )
    est = obj.estimate(s, np.zeros(3))
    beta, kT, p = s.beta, KB * s.T, 1.0 / 16.605390671738466
    V, U, dU = fr["V"], fr["U"], fr["dU"]
    Hh = U + p * V
    m = lambda a: a.mean(0)
    cov = lambda a: m((a - m(a))[:, None] * (dU - m(dU)))
    a_p = (m(V * Hh) - m(V) * m(Hh)) / (KB * s.T**2 * m(V))
    k_t = (m(V * V) - m(V) ** 2) / (kT * m(V)) / 16.605390671738466
    dVH = m(V[:, None] * dU) - beta * cov(V * Hh)
    dV = -beta * cov(V)
    dH = m(dU) - beta * cov(Hh)
    da = (dVH - dV * m(Hh) - m(V) * dH) / (KB * s.T**2 * m(V)) - a_p * dV / m(V)
    dV2 = -beta * cov(V * V)
    dk = ((dV2 - 2 * m(V) * dV) / (kT * m(V)) - (m(V * V) - m(V) ** 2) / (kT * m(V) ** 2) * dV) / 16.605390671738466
    assert np.isclose(est.y[0], a_p) and np.isclose(est.y[1], k_t)
    assert np.allclose(est.J[0], da, rtol=1e-8) and np.allclose(est.J[1], dk, rtol=1e-8)
