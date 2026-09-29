"""Parameter gradients of alchemical free energies (md/fe_grad.py): dU/dP at converged dipoles
against finite differences with the dipoles re-solved (annihilate and keep), the sampler (batched =
sequential = direct), the estimators as exact derivatives of the reweighted free energies (end
states: exponential averaging; MBAR with the sampled mixture) on stored frames, the gas-phase leg,
an exactly known case (a lone rigid solute: zero variance, the gradient of E_gas(0) - E_gas(1)),
harmonic oscillators with analytic df/dtheta and calibrated jackknife errors, the driver's outputs
and the fitting-target API."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_alchemy import alch_sim, frame, settings  # noqa: E402
from test_grad import water  # noqa: E402

from pgm_jax import System  # noqa: E402
from pgm_jax.md import fe_grad as fg  # noqa: E402
from pgm_jax.md import free_energy as fe  # noqa: E402
from pgm_jax.md.alchemy import (  # noqa: E402
    Alchemy,
    FreeEnergyRun,
    GasPhaseLeg,
    LambdaWindows,
    alchemical_system,
    standard_schedule,
)
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402


def direction(space, P, seed=0):
    """A random direction in the flat table, relative to each entry (zero where the entry is zero)."""
    p = np.asarray(space.flatten(P))
    return p * np.random.default_rng(seed).normal(size=p.size)


# ----------------------------------------------------------------------------- dU/dP
@pytest.mark.parametrize("mode,lam", [("annihilate", (1.0, 1.0)), ("annihilate", (0.5, 1.0)),
                                      ("annihilate", (0.0, 0.4)), ("keep", (0.4, 1.0)), ("keep", (0.0, 0.0))])
def test_dudp_matches_finite_differences_with_resolved_dipoles(mode, lam):
    """Hellmann-Feynman for parameters: dU/dP at the converged dipoles (autodiff at fixed mu) equals the
    central difference of the energy with the dipoles re-solved, along a random direction of the
    whole table (charges, covalent dipoles, radii, polarizabilities, LJ of solute and solvent)."""
    sim, alch, P, _ = alch_sim()
    X, Hb, cand = frame(sim)
    ff = sim.ff
    if mode == "keep":
        alch = Alchemy(sim.sys, 0, intramolecular="keep")
        alch.check(ff)
    space = fg.ParamSpace(sim.sys.table)
    lam = jnp.asarray(lam)
    U = jax.jit(lambda p: alch.energy(ff, X, Hb, cand, ff.init_induction(), space.unflatten(p, P), lam)[:2])
    p0 = space.flatten(P)
    _, ind = U(p0)
    g = np.asarray(space.flatten(jax.grad(lambda Q: alch.energy_fixed_mu(ff, X, Hb, cand, ind.mu, Q, lam))(P)))
    for seed in range(3):
        v = direction(space, P, seed)
        h = 1e-5
        fd = (float(U(p0 + h * v)[0]) - float(U(p0 - h * v)[0])) / (2 * h)
        print(f"[fe_grad] HF {mode} {tuple(np.asarray(lam).tolist())}: FD {fd:.8f} autodiff {g @ v:.8f} rel {abs(fd - g @ v) / abs(fd):.1e}")
        assert abs(fd - g @ v) < 1e-6 * max(1.0, abs(fd)), (mode, lam, fd, g @ v)
    # parameters the Hamiltonian does not see there have zero derivative
    solq = space.select(("q", "cov"), solute=True)
    if float(lam[0]) == 0.0 and mode == "annihilate":
        assert np.abs(g[solq]).max() < 1e-12


# ----------------------------------------------------------------------------- sampler
def windows(batched=True, seed=1, mode="annihilate"):
    sim, alch, P, _ = alch_sim(settings(dipole_tol=1e-9), thermostat="bussi", ensemble="nvt")
    if mode == "keep":
        pos = sim.positions_nm()
        alch = Alchemy(sim.sys, 0, intramolecular="keep")
        sim = Simulation(sim.sys, pos, np.asarray(sim.state.box), settings(dipole_tol=1e-9), dt=0.001, log=None,
                         params=P, alchemy=alch, thermostat="bussi", ensemble="nvt")
    L = standard_schedule(3, [0.5, 0.0])
    return LambdaWindows(sim, L, batched=batched, seed=seed), P


def test_sampler_batched_equals_sequential_and_direct():
    wb, P = windows(True)
    ws, _ = windows(False)
    wb.advance(8)
    ws.advance(8)
    gb, gs = fg.ParamGradients(wb), fg.ParamGradients(ws)
    Gb, Gs = gb.sample(), gs.sample()
    K, M = wb.n, gb.space.n
    assert Gb.shape == (2, K, M) and list(gb.targets) == [0, K - 1] and len(gb.space.names) == M
    assert np.allclose(Gb, Gs, rtol=1e-8, atol=1e-7)
    ff, alch = wb.sim.ff, wb.alchemy
    for t, k in enumerate((0, K - 1)):
        for n in (0, 2, K - 1):
            st = wb.state(n)
            X, cand, _ = fg._frame(wb, st)
            lam = jnp.asarray(wb.lambdas[k])
            _, ind, _, _ = alch.energy(ff, X, st.box, cand, ff.init_induction(), P, lam)
            g = gb.space.flatten(jax.grad(lambda Q: alch.energy_fixed_mu(ff, X, st.box, cand, ind.mu, Q, lam))(P))
            assert np.allclose(Gb[t, n], np.asarray(g), rtol=1e-7, atol=1e-6)
    # the decoupled end state does not see the solute's parameters (annihilation: charges, LJ)
    sol = gb.space.select(solute=True)
    assert np.abs(Gb[1][:, sol]).max() < 1e-6 * np.abs(Gb[0][:, sol]).max()


@pytest.mark.parametrize("mode", ["annihilate", "keep"])
def test_estimators_are_derivatives_of_reweighted_free_energies(mode):
    """On stored frames the estimators are exact derivatives: 'end' of the exponential-averaging
    (Zwanzig) free energies of the perturbed end states from their own windows, 'mbar' of the MBAR
    free energies of the perturbed end states from the sampled mixture.  Central differences in the
    parameters (energies with the dipoles re-solved) against the analytic gradient . v."""
    w, P = windows(mode=mode)
    pg = fg.ParamGradients(w)
    space = pg.space
    ff, alch = w.sim.ff, w.alchemy
    K = w.n
    E = jax.jit(lambda st, p, lam: alch.energy(ff, *(lambda f: (f[0], st.box, f[1]))(fg._frame(w, st)),
                                               st.induction, space.unflatten(p, P), lam)[0])
    v = direction(space, P, 7)
    h = 3e-5                                  # truncation (h^2: 5e-4 at h = 1e-4) vs float64 roundoff / h
    p0 = space.flatten(P)
    us, gs, dU = [], [], []                   # dU[s, sign, t, n]: U_t(x_n; p0 + sign h v)
    for _ in range(8):
        w.advance(5)
        u, _, _ = w.sample()
        us.append(u.T)                        # (n, k) -> stored as [k, n] like FreeEnergyRun: u[k, n]
        gs.append(pg.sample())
        dU.append([[[float(E(w.state(n), p0 + sg * h * v, jnp.asarray(w.lambdas[k]))) for n in range(K)]
                    for k in (0, K - 1)] for sg in (1.0, -1.0)])
    u = np.array([x.T for x in us])           # (S, K, K): u[s, k, n]
    G = np.array(gs)
    dU = np.array(dU)
    kT = float(w.integ.kT)
    S = {"u": u, "dudp": G, "lambdas": w.lambdas, "kT": kT, "time_ps": np.arange(1, 9, dtype=float),
         "meta": {"dudp_targets": [0, K - 1], "dudp_names": space.names}}
    r = fg.gradient_estimate(S, n_blocks=2)
    # end states, exponential averaging from the own window's samples
    U0 = u * kT
    zw = lambda a: -kT * np.log(np.mean(np.exp(-a / kT)))                      # noqa: E731

    def end(sg):
        i = 0 if sg > 0 else 1
        return zw(dU[:, i, 1, K - 1] - U0[:, K - 1, K - 1]) - zw(dU[:, i, 0, 0] - U0[:, 0, 0])
    fd_end = (end(1) - end(-1)) / (2 * h)
    print(f"[fe_grad] reweighting {mode}: end FD {fd_end:.8f} estimator {r['solv']['end'].grad @ v:.8f}")
    assert abs(fd_end - r["solv"]["end"].grad @ v) < 1e-5 * max(100.0, abs(fd_end)), (fd_end, r["solv"]["end"].grad @ v)
    # MBAR: the perturbed end states as unsampled states of the mixture sampled at p0
    s = u.shape[0]
    u_kn = u.transpose(1, 2, 0).reshape(K, K * s)
    f, _ = fe.mbar(u_kn, np.full(K, s))
    _, logden = fe._mbar_weights(u_kn, np.full(K, s), f)
    from scipy.special import logsumexp

    def mb(sg):
        i = 0 if sg > 0 else 1
        fk = [-logsumexp(-dU[:, i, t, :].T.reshape(-1) / kT - logden) for t in (0, 1)]
        return kT * (fk[1] - fk[0])
    fd_mbar = (mb(1) - mb(-1)) / (2 * h)
    print(f"[fe_grad] reweighting {mode}: MBAR FD {fd_mbar:.8f} estimator {r['solv']['mbar'].grad @ v:.8f}")
    assert abs(fd_mbar - r["solv"]["mbar"].grad @ v) < 1e-5 * max(100.0, abs(fd_mbar)), (fd_mbar, r["solv"]["mbar"].grad @ v)
    assert abs(r["solv"]["mbar"].value - kT * (f[-1] - f[0])) < 1e-8


# ----------------------------------------------------------------------------- gas phase, exact case
def test_gas_leg_gradient_and_exact_sampled_case():
    """The rigid solute's gas-phase leg: d(E_gas(0) - E_gas(1))/dP by autodiff = central differences.
    The same free energy sampled with the MD engine (the lone rigid water in a 4.2 nm box, windows
    lambda_elec = 1, 0.5, 0): its energy does not depend on the configuration, so both estimators
    must return the exact gradient with zero variance (up to the box's image and PME error)."""
    w = water()
    sysA, P = alchemical_system(System([w]), 0)
    alch = Alchemy(sysA, 0)
    t = np.radians(104.52 / 2)
    xyz = np.array([[0, 0, 0], [0.09572 * np.sin(t), 0.09572 * np.cos(t), 0],
                    [-0.09572 * np.sin(t), 0.09572 * np.cos(t), 0]]) + 2.1
    gas = GasPhaseLeg(alch, xyz, "qpi")
    space = fg.ParamSpace(sysA.table)
    dg, gg = fg.gas_leg_gradient(gas, P, space)
    assert abs(dg - gas.delta_g(P)) < 1e-10
    v = direction(space, P, 3)
    h = 1e-6
    Pp, Pm = (space.unflatten(space.flatten(P) + s * h * v, P) for s in (1.0, -1.0))
    fd = (gas.delta_g(Pp) - gas.delta_g(Pm)) / (2 * h)
    assert abs(fd - gg @ v) < 1e-6 * max(1.0, abs(fd))
    H = np.eye(3) * 4.2
    s = MDSettings(precision="double", cutoff=1.2, skin=0.1, ewald_beta=3.0, pme_grid=(64, 64, 64), pme_order=8,
                   dipole_tol=1e-12, max_iter=200, peek=0.0, lj_lrc=False)
    sim = Simulation(sysA, xyz, H, s, dt=0.001, log=None, params=P, alchemy=alch, thermostat="bussi",
                     ensemble="nvt", neighbor_list="atom", temperature=298.0)
    L = np.array([[1.0, 1.0], [0.5, 1.0], [0.0, 1.0]])
    win = LambdaWindows(sim, L, seed=2)
    run = FreeEnergyRun(win, sample_every=5, exchange_every=5, log=None, param_grad=fg.ParamGradients(win))
    run.run(60, prefix=None)
    r = fg.gradient_estimate(run.arrays(), gas={"delta_g": dg, "grad": gg}, n_blocks=4)
    for est in ("end", "mbar"):
        solv, hyd = r["solv"][est], r["hyd"][est]
        # the solution "leg" here is the same molecule in vacuum: DeltaG_solv = DeltaG_gas, hydration 0
        assert abs(solv.value - dg) < 2e-3 and abs(hyd.value) < 2e-3
        scale = np.abs(gg).max()
        print(f"[fe_grad] lone solute {est}: DeltaG {solv.value:.6f} exact {dg:.6f}; max |grad - exact| "
              f"{np.abs(solv.grad - gg).max():.2e} of max |exact| {scale:.1f}; max grad error bar {np.abs(solv.grad_err).max():.1e}")
        assert np.abs(solv.grad - gg).max() < 1e-3 * scale, (est, np.abs(solv.grad - gg).max(), scale)
        assert np.abs(solv.grad_err).max() < 1e-6 * scale and solv.value_err < 1e-6


# ----------------------------------------------------------------------------- analytic toy
def test_harmonic_oscillators_analytic_gradient_and_calibrated_errors():
    """States u_k(x; theta) = K_k exp(c_k theta) x^2 / 2 (beta = 1) at theta = 0: f_k = ln K_k(theta) / 2,
    d(f_{K-1} - f_0)/dtheta = (c_{K-1} - c_0) / 2 exactly.  Correlated samples (AR(1), phi = 0.9) of
    every state; both estimators unbiased within 3 standard errors over 40 repeats, their jackknife
    errors match the spread of the estimates (ratio 0.7-1.4), the value too."""
    Kk = np.array([1.0, 1.7, 3.0, 5.0])
    c = np.array([0.8, 0.3, -0.2, -0.6])
    exact_g = 0.5 * (c[-1] - c[0])
    exact_f = 0.5 * np.log(Kk[-1] / Kk[0])
    rng = np.random.default_rng(1)
    n, phi, K = 2000, 0.9, len(Kk)
    res = {"end": [], "mbar": []}
    errs = {"end": [], "mbar": []}
    vals, verrs = [], []
    for rep in range(40):
        X = np.zeros((n, K))
        e = rng.normal(size=(n, K)) / np.sqrt(Kk)
        X[0] = e[0]
        for i in range(1, n):
            X[i] = phi * X[i - 1] + np.sqrt(1 - phi * phi) * e[i]
        u = 0.5 * Kk[None, :, None] * X[:, None, :] ** 2                   # u[s, k, n]
        du = 0.5 * (c * Kk)[None, :, None] * X[:, None, :] ** 2
        S = {"u": u, "dudp": du[:, [0, K - 1], :, None], "lambdas": np.stack([np.linspace(1, 0, K)] * 2, 1),
             "kT": 1.0, "time_ps": np.arange(1, n + 1, dtype=float),
             "meta": {"dudp_targets": [0, K - 1], "dudp_names": ["q:theta"]}}
        r = fg.gradient_estimate(S, n_blocks=10)
        for est in res:
            res[est].append(r["solv"][est].grad[0])
            errs[est].append(r["solv"][est].grad_err[0])
        vals.append(r["solv"]["mbar"].value)
        verrs.append(r["solv"]["mbar"].value_err)
    for est in res:
        a = np.array(res[est])
        print(f"[fe_grad] harmonic {est}: mean {a.mean():.5f} +- {a.std() / np.sqrt(len(a)):.5f} exact {exact_g:.5f}; "
              f"mean error bar / spread {np.mean(errs[est]) / a.std():.3f}")
        assert abs(a.mean() - exact_g) < 3 * a.std() / np.sqrt(len(a)), (est, a.mean(), exact_g)
        assert 0.7 < np.mean(errs[est]) / a.std() < 1.4, (est, np.mean(errs[est]), a.std())
    vals = np.array(vals)
    print(f"[fe_grad] harmonic value: mean {vals.mean():.5f} +- {vals.std() / np.sqrt(len(vals)):.5f} exact {exact_f:.5f}; "
          f"error bar / spread {np.mean(verrs) / vals.std():.3f}")
    assert abs(vals.mean() - exact_f) < 3 * vals.std() / np.sqrt(len(vals))
    assert 0.7 < np.mean(verrs) / vals.std() < 1.4


# ----------------------------------------------------------------------------- driver and targets
def test_run_outputs_restart_and_fitting_target(tmp_path):
    w, P = windows()
    pg = fg.ParamGradients(w)
    prefix = str(tmp_path / "g")
    run = FreeEnergyRun(w, sample_every=5, exchange_every=5, log=None, param_grad=pg)
    run.run(60, prefix=prefix, restart=30)
    d = fe.load(prefix + "_fe.npz")
    K, M = w.n, pg.space.n
    assert d["dudp"].shape == (12, 2, K, M) and d["meta"]["dudp_names"] == pg.space.names
    assert np.allclose(d["meta"]["params_flat"], np.asarray(pg.space.flatten(P)))
    # continue from the checkpoint with gradients; refused without samples of them
    run2 = FreeEnergyRun(w, sample_every=5, exchange_every=5, log=None, param_grad=pg)
    run2.load(prefix + ".fe.chk")
    run2.run(10, prefix=prefix)
    assert fe.load(prefix + "_fe.npz")["dudp"].shape == (14, 2, K, M)
    plain = FreeEnergyRun(w, sample_every=5, log=None)
    plain.run(10, prefix=str(tmp_path / "p"))
    run3 = FreeEnergyRun(w, sample_every=5, log=None, param_grad=pg)
    with pytest.raises(ValueError, match="no parameter gradients"):
        run3.load(str(tmp_path / "p.fe.chk"))
    run3.load_windows(str(tmp_path / "p.fe.chk"))
    assert run3.step == 0 and w.time_ps == 0.0 and not run3.samples["u"]
    # the target: a scale of the solute's charges theta -> table, chain rule = projection
    space = pg.space
    t = fg.FreeEnergyTarget.from_npz(prefix + "_fe.npz", n_blocks=2, experiment=-20.0, sigma=2.0)
    p0 = np.asarray(space.flatten(P))
    vq = space.scale_direction(p0, "charge")

    def theta_fn(th):
        return fg.scaled_params(space, P, {"charge": jnp.exp(th[0])})
    with pytest.raises(ValueError):
        t.value_and_grad(theta_fn, jnp.array([0.1]))
    r = t.value_and_grad(theta_fn, jnp.array([0.0]))
    proj, err = t.result.project(vq)
    assert abs(r["grad"][0] - proj) < 1e-8 * max(1.0, abs(proj)) and abs(r["grad_err"][0] - err) < 1e-8 * max(1.0, err)
    assert abs(r["dchi2"][0] - 2 * (r["value"] + 20.0) / 4.0 * proj) < 1e-8 * max(1.0, abs(proj))
    est = t.estimate(theta_fn, jnp.array([0.0]), unit="kcal/mol")
    assert est["J"].shape == (1, 1) and est["loo"]["J"].shape == (2, 1, 1) and est["loo"]["y"].shape == (2, 1)
    assert abs(est["J"][0, 0] - proj / 4.184) < 1e-8 * max(1.0, abs(proj)) and abs(est["target"][0] + 20.0 / 4.184) < 1e-12
    # original (non-alchemical) table: the solute's copy and the solvent's key both follow P0
    sys0 = alch_sim()[3][2]
    amap = fg.alchemical_map(sys0, w.alchemy.sys)
    P0 = sys0.params0
    PA = amap(P0)
    assert np.allclose(np.asarray(space.flatten(PA)), p0)
    J = np.asarray(jax.jacfwd(lambda q: space.flatten(amap({**P0, "q": q})))(jnp.asarray(P0["q"])))
    gq, _ = t.result.chain(J)
    iq = [space.index("q:WAT:OW"), space.index("q:alch:WAT:OW")] if "q:WAT:OW" in space.names else None
    if iq is not None:
        j = list(sys0.table.keys["q"]).index("WAT:OW")
        assert abs(gq[j] - t.result.grad[iq].sum()) < 1e-9 * max(1.0, abs(gq[j]))
    # combinations (relative free energies): value and gradient add, errors in quadrature
    a = {"value": 1.0, "value_err": 0.3, "grad": np.array([1.0, 2.0]), "grad_err": np.array([0.1, 0.2])}
    b = {"value": 4.0, "value_err": 0.4, "grad": np.array([0.5, 1.0]), "grad_err": np.array([0.1, 0.0])}
    cmb = fg.combine([(1.0, b), (-1.0, a)])
    assert cmb["value"] == 3.0 and abs(cmb["value_err"] - 0.5) < 1e-12 and np.allclose(cmb["grad"], [-0.5, -1.0])


def test_param_space_and_scaled_params():
    sim, alch, P, (pos, H, sys0) = alch_sim()
    space = fg.ParamSpace(sim.sys.table)
    p = space.flatten(P)
    back = space.unflatten(p, P)
    for q in space.quantities:
        assert np.allclose(np.asarray(back[q]), np.asarray(P[q]))
    assert fg.ParamSpace.from_names(space.names).names == space.names
    Q = fg.scaled_params(space, P, {"charge": 1.1, "eps": 0.81})
    sol, env = space.select(("q",), True), space.select(("q",), False)
    pq = np.asarray(space.flatten(Q))
    assert np.allclose(pq[sol], 1.1 * np.asarray(p)[sol]) and np.allclose(pq[env], np.asarray(p)[env])
    se = space.select(("lj_sqrt_eps",), True)
    assert np.allclose(pq[se], 0.9 * np.asarray(p)[se])
    v = space.scale_direction(p, "eps")
    assert np.allclose(v[se], 0.5 * np.asarray(p)[se]) and np.count_nonzero(v) == np.count_nonzero(np.asarray(p)[se])


def test_flexible_solute_keep_sampler():
    """A flexible solute with intramolecular="keep" (the gas-phase correction depends on the solute's
    electrostatic parameters at every lambda): batched = sequential = direct; the decoupled end
    state still depends on the solute's charges (its gas-phase electrostatics) and its LJ (its
    intramolecular pairs), not on the solute-water coupling."""
    from test_alchemy import flex_box

    from pgm_jax.md.flexible import FlexibleSimulation
    tpl, sys0, tpls, X, H = flex_box()
    sysA, P = alchemical_system(sys0, 0)
    mk = lambda: FlexibleSimulation(sysA, tpls, X, H, settings(dipole_tol=1e-9), dt=0.001, log=None, params=P,   # noqa: E731
                                    alchemy=Alchemy(sysA, 0, intramolecular="keep"), constraints="h-bonds",
                                    thermostat="bussi")
    L = standard_schedule(2, [0.4, 0.0])
    wb, ws = LambdaWindows(mk(), L, seed=1), LambdaWindows(mk(), L, batched=False, seed=1)
    wb.advance(6)
    ws.advance(6)
    gb, gs = fg.ParamGradients(wb), fg.ParamGradients(ws)
    Gb, Gs = gb.sample(), gs.sample()
    assert np.allclose(Gb, Gs, rtol=1e-7, atol=1e-6)
    ff, alch = wb.sim.ff, wb.alchemy
    K = wb.n
    st = wb.state(1)
    Y, cand, _ = fg._frame(wb, st)
    lam = jnp.zeros(2)
    _, ind, _, _ = alch.energy(ff, Y, st.box, cand, ff.init_induction(), P, lam)
    g = np.asarray(gb.space.flatten(jax.grad(lambda Q: alch.energy_fixed_mu(ff, Y, st.box, cand, ind.mu, Q, lam))(P)))
    assert np.allclose(Gb[1, 1], g, rtol=1e-7, atol=1e-6)
    sq = gb.space.select(("q",), solute=True)
    se = gb.space.select(("lj_sqrt_eps",), solute=True)
    assert np.abs(Gb[1][:, sq]).max() > 1.0 and np.abs(Gb[1][:, se]).max() > 1e-3     # gas-phase elec, intra LJ
    # the gas-phase part at (0, 0) equals dE_gas/dq of the lone solute at its geometry
    gas = GasPhaseLeg(alch, np.asarray(Y)[:6], "qpi")
    gg = np.asarray(gb.space.flatten(jax.grad(lambda Q: gas._e(jnp.asarray(1.0), Q))(P)))
    assert np.allclose(Gb[1, 1][sq], gg[sq], rtol=1e-6, atol=1e-6)
