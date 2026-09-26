"""Charge flux in liquid MD (pgm_jax/md/flux.py): flexible methanol with and without flux, on a GPU.

    python examples/fit_bonded_template.py methanol --wmu 1 --maxiter 4000 --out runs/flux/methanol_noflux.flex
    python examples/fit_bonded_template.py methanol --flux 1 --wmu 1 --maxiter 4000 --out runs/flux/methanol_flux1.flex
    python scripts/flux_md.py liquid runs/flux/methanol_flux1.flex     # NVT 2 ps + NPT 100 + 100 ps
    python scripts/flux_md.py nve runs/flux/methanol_flux1.flex        # NVE from the liquid state
    python scripts/flux_md.py speed runs/flux/methanol_flux1.flex --replicate 2
    python scripts/flux_md.py gas runs/flux/methanol_flux1.flex        # isolated molecule: <U_gas>, <|mu|>

liquid: 216 molecules from a dilute lattice (0.55 g/cm^3), NVT 2 ps, NPT 298 K / 1 bar (dt 0.5 fs,
Langevin 1/ps, barostat every 25 steps, MDSettings() defaults: mixed precision, 0.9 nm, dipole tol
1e-5) 100 ps of equilibration and 100 ps sampled every 0.5 ps: density, potential energy per
molecule (bonded + van der Waals + electrostatics), mean |molecular dipole| (charges, covalent and
induced dipoles; md/dipoles.py) and the mean charge shift of the flux.  Writes
runs/flux/<template>_liquid.json and the final state (<template>_liquid.npz).
nve: NVE from that state, --nve_ps (50) ps mixed (dt 0.5 fs, tol 1e-5) and --double_ps (a fifth
of it) double (tol 1e-8); drift of E_tot in kT per ns per degree of freedom (linear fit).
speed: NVT from that state (replicated n x n x n; --thermostat, --fixed_iter n: exactly n CG
iterations per step, which isolates the cost of the flux terms from the iteration count), ms per
step and CG iterations per step of the template with its flux and of the same template with the
flux switched off (the flux-free code path), alternating, so the only difference is the flux.
gas: the isolated molecule with the gas-phase model the template was fitted with (BondedModel:
bonded terms, pGM with every pair, intramolecular van der Waals, the flux), 256 independent copies
(vmap), BAOAB Langevin 5/ps, dt 0.5 fs, 20 ps + 100 ps sampled every 50 fs: <U_gas> and <|mu|>;
with the liquid's <U>/N it gives the heat of vaporization <U_gas> - <U_liq>/N + RT."""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pgm_jax.md.dipoles import CellDipole
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.integrate import KB
from pgm_jax.system import System
from pgm_jax.units import DEBYE_E_NM

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=("liquid", "nve", "speed", "gas"))
ap.add_argument("template")
ap.add_argument("--n", type=int, default=216)
ap.add_argument("--temp", type=float, default=298.0)
ap.add_argument("--equil_ps", type=float, default=100.0)
ap.add_argument("--prod_ps", type=float, default=100.0)
ap.add_argument("--replicate", type=int, default=1)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--nve_ps", type=float, default=50.0, help="nve: mixed-precision run length (ps; 0: skip)")
ap.add_argument("--double_ps", type=float, default=None, help="nve: double-precision run length (default nve_ps / 5)")
ap.add_argument("--thermostat", default="langevin", help="speed: langevin (1/ps) | bussi (1 ps)")
ap.add_argument("--fixed_iter", type=int, default=0, help="speed: exactly this many CG iterations per step")
a = ap.parse_args()

tpl = FlexibleTemplate.load(a.template)
stem = os.path.splitext(a.template)[0]
T, dt = a.temp, 0.0005


def no_flux(t):
    """The same template with the flux switched off (the engine's flux-free path)."""
    return FlexibleTemplate(t.specs, {**t.settings, "flux": 0}, {k: v for k, v in t.P.items() if k != "flux"}, t.index)


def mol_dipole_fn(sim):
    cd = CellDipole(sim.ff)
    return jax.jit(lambda x, H, mu: jnp.mean(jnp.linalg.norm(cd.molecular(x, H, mu, sim.integ.params), axis=1)))


if a.cmd == "liquid":
    N = a.n
    pos, H = liquid_box(tpl, N, 0.55, seed=1, min_dist=0.18)
    sys_ = System([tpl.pgm] * N)
    st = MDSettings()
    nvt = FlexibleSimulation(sys_, [tpl] * N, pos, H, st, dt=dt, ensemble="nvt", temperature=T, gamma=5.0, log=None)
    nvt._advance(4000)
    sim = FlexibleSimulation(sys_, [tpl] * N, nvt.positions_nm(), np.asarray(nvt.state.box), st, dt=dt,
                             ensemble="npt", temperature=T, gamma=1.0, barostat_interval=25,
                             vel_nm_ps=nvt.velocities_nm_ps(), log=sys.stdout)
    for _ in range(int(round(a.equil_ps / 0.5))):         # 0.5 ps blocks (an overflow repeats one block)
        sim._advance(1000)
    dip = mol_dipole_fn(sim)
    fl = sim.ff.flux
    qshift = None
    if fl is not None:
        P0 = sim.ff._atoms(None)
        qshift = jax.jit(lambda x, H: jnp.mean(jnp.abs(sim.ff.charges_at(x, H, P0)["q"] - P0["q"])))
        dbond = jax.jit(lambda x, H: fl.deviations(x, H))
    rec = {k: [] for k in ("time_ps", "density", "epot_per_mol", "mol_dipole_D", "temp_K", "cg_iter", "dq_mean", "db_mean")}
    every = int(round(0.5 / dt))
    t0, s0, cg0 = time.time(), int(sim.state.step), float(sim.state.cg_total)
    for _ in range(int(round(a.prod_ps / 0.5))):
        sim._advance(every)
        o = sim.observables()
        x, Hb, mu = sim.state.dyn.position, sim.state.box, sim.state.induction.mu
        rec["time_ps"].append(round(o["time_ps"], 3)); rec["density"].append(o["density_g_cm3"])
        rec["epot_per_mol"].append(o["epot"] / N); rec["temp_K"].append(o["temp_K"])
        rec["mol_dipole_D"].append(float(dip(x, Hb, mu)) / DEBYE_E_NM); rec["cg_iter"].append(o["cg_iter"])
        rec["dq_mean"].append(float(qshift(x, Hb)) if qshift else 0.0)
        rec["db_mean"].append(float(jnp.mean(dbond(x, Hb))) if qshift else 0.0)
    wall = time.time() - t0
    steps = int(sim.state.step) - s0

    def mean_se(v, nb=5):
        v = np.asarray(v)
        b = np.array([x.mean() for x in np.array_split(v, nb)])
        return float(v.mean()), float(b.std(ddof=1) / np.sqrt(nb))

    out = {"template": a.template, "flux": tpl.settings.get("flux", 0), "n_mol": N, "T": T, "dt_ps": dt,
           "equil_ps": a.equil_ps, "prod_ps": a.prod_ps, "record": rec,
           "density": mean_se(rec["density"]), "epot_per_mol": mean_se(rec["epot_per_mol"]),
           "mol_dipole_D": mean_se(rec["mol_dipole_D"]), "temp_K": mean_se(rec["temp_K"]),
           "dq_mean_e": float(np.mean(rec["dq_mean"])), "db_mean_nm": float(np.mean(rec["db_mean"])),
           "cg_per_step": (float(sim.state.cg_total) - cg0) / steps, "ns_per_day": steps * dt / 1000.0 / (wall / 86400.0),
           "device": str(jax.devices()[0])}
    print({k: v for k, v in out.items() if k != "record"}, flush=True)
    json.dump(out, open(stem + "_liquid.json", "w"), indent=1)
    np.savez(stem + "_liquid.npz", pos=sim.positions_nm(), vel=sim.velocities_nm_ps(), box=np.asarray(sim.state.box))

elif a.cmd == "nve":
    z = np.load(stem + "_liquid.npz")
    N = len(z["pos"]) // tpl.n
    sys_ = System([tpl.pgm] * N)
    out = {"template": a.template, "flux": tpl.settings.get("flux", 0), "nve": []}
    runs = (("0.5 fs, mixed, tol 1e-5", "mixed", 1e-5, a.nve_ps),
            ("0.5 fs, double, tol 1e-8", "double", 1e-8, a.nve_ps / 5 if a.double_ps is None else a.double_ps))
    for label, prec, tol, ps in [r for r in runs if r[3] > 0]:
        s = FlexibleSimulation(sys_, [tpl] * N, z["pos"], z["box"], MDSettings(precision=prec, dipole_tol=tol), dt=dt,
                               ensemble="nve", vel_nm_ps=z["vel"], log=None)
        every = int(round(0.1 / dt))
        t, E = [], []
        for _ in range(int(round(ps / 0.1))):
            s._advance(every)
            o = s.observables()
            t.append(o["time_ps"]); E.append(o["etot"])
        t, E = np.array(t), np.array(E)
        fit = np.polyfit(t, E, 1)
        r = {"label": label, "dof": s.integ.dof, "drift_kT_per_ns_per_dof": float(fit[0] * 1000.0 / (KB * T) / s.integ.dof),
             "rms_fluct_kT": float(np.std(E - np.polyval(fit, t)) / (KB * T)), "cg_per_step": float(s.state.cg_total) / int(s.state.step),
             "time_ps": t.tolist(), "etot": E.tolist()}
        out["nve"].append(r)
        print(label, {k: v for k, v in r.items() if k not in ("time_ps", "etot")}, flush=True)
    json.dump(out, open(stem + f"_nve{'' if a.nve_ps > 0 else '_double'}.json", "w"), indent=1)

elif a.cmd == "gas":
    model, P = tpl.model, jax.tree_util.tree_map(jnp.asarray, tpl.P)
    m = jnp.asarray(tpl.pgm.masses, jnp.float64)[:, None]
    kT = KB * T
    gamma, nrep = 5.0, 256
    ef = jax.value_and_grad(lambda R: model.energy(tpl.index, R, P)[0])
    dipf = lambda R: jnp.linalg.norm(model.energy(tpl.index, R, P)[1])
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1.0 - c1 * c1) * kT)

    def step(c, key):
        R, p, g = c
        p = p - 0.5 * dt * g
        R = R + 0.5 * dt * p / m
        p = c1 * p + c2 * jnp.sqrt(m) * jax.random.normal(key, p.shape)
        R = R + 0.5 * dt * p / m
        e, g = ef(R)
        p = p - 0.5 * dt * g
        return (R, p, g), e

    def block(c, key, n):
        c, e = jax.lax.scan(step, c, jax.random.split(key, n))
        return c, (e[-1], dipf(c[0]))

    run = jax.jit(jax.vmap(lambda c, key, n=100: block(c, key, n)))
    key = jax.random.PRNGKey(0)
    R0 = jnp.broadcast_to(jnp.asarray(tpl.spec.ref_xyz, jnp.float64), (nrep, tpl.n, 3))
    key, k1 = jax.random.split(key)
    p0 = jnp.sqrt(m * kT) * jax.random.normal(k1, R0.shape)
    g0 = jax.vmap(lambda R: ef(R)[1])(R0)
    c = (R0, p0, g0)
    t0 = time.time()
    U, D = [], []
    for b in range(int(round((20.0 + a.prod_ps) / (100 * dt)))):
        key, k = jax.random.split(key)
        c, (e, d) = run(c, jax.random.split(k, nrep))
        if b * 100 * dt >= 20.0:
            U.append(np.asarray(e)); D.append(np.asarray(d))
    U, D = np.array(U), np.array(D)                         # (samples, replicas)
    se = lambda v: float(np.std(v.mean(0), ddof=1) / np.sqrt(v.shape[1]))
    out = {"template": a.template, "flux": tpl.settings.get("flux", 0), "T": T, "replicas": nrep, "prod_ps": a.prod_ps,
           "U_gas": [float(U.mean()), se(U)], "mol_dipole_D": [float(D.mean()) / DEBYE_E_NM, se(D) / DEBYE_E_NM],
           "wall_s": time.time() - t0}
    liq = stem + "_liquid.json"
    if os.path.exists(liq):
        L = json.load(open(liq))
        out["dHvap_kJ_mol"] = [out["U_gas"][0] - L["epot_per_mol"][0] + KB * T,
                               float(np.hypot(out["U_gas"][1], L["epot_per_mol"][1]))]
        out["dHvap_kcal_mol"] = [v / 4.184 for v in out["dHvap_kJ_mol"]]
    print(out, flush=True)
    json.dump(out, open(stem + "_gas.json", "w"), indent=1)

else:
    z = np.load(stem + "_liquid.npz")
    n0 = len(z["pos"]) // tpl.n
    k = a.replicate
    shifts = np.array([(i, j, l) for i in range(k) for j in range(k) for l in range(k)], float) @ z["box"]
    pos = np.concatenate([z["pos"] + s for s in shifts])
    vel = np.concatenate([z["vel"]] * len(shifts))
    H = z["box"] * k
    N = n0 * k ** 3
    sys_ = System([tpl.pgm] * N)
    res = {}
    sims = {}
    for label, t in (("flux", tpl), ("no flux", no_flux(tpl))):
        st = MDSettings() if not a.fixed_iter else MDSettings(dipole_tol=0.0, max_iter=a.fixed_iter)
        sims[label] = FlexibleSimulation(sys_, [t] * N, pos, H, st, dt=dt, ensemble="nvt", temperature=T,
                                         gamma=1.0, thermostat=a.thermostat, tau_t=1.0, vel_nm_ps=vel, log=None)
        sims[label]._advance(500)                           # compile
    for rnd in range(3):
        for label, sim in sims.items():
            jax.block_until_ready(sim.state.dyn.position)
            s0, c0 = int(sim.state.step), float(sim.state.cg_total)
            t0 = time.time()
            sim._advance(a.steps)
            jax.block_until_ready(sim.state.dyn.position)
            w = time.time() - t0
            st_ = int(sim.state.step) - s0
            res.setdefault(label, []).append({"ms_per_step": 1000.0 * w / st_, "cg_per_step": (float(sim.state.cg_total) - c0) / st_})
            print(rnd, label, res[label][-1], flush=True)
    summ = {lab: {"ms_per_step": float(np.min([r["ms_per_step"] for r in v])),
                  "cg_per_step": float(np.mean([r["cg_per_step"] for r in v]))} for lab, v in res.items()}
    for lab in summ:
        summ[lab]["ns_per_day"] = dt * 86400.0 / (summ[lab]["ms_per_step"] * 1e-3) / 1000.0
    out = {"template": a.template, "n_mol": N, "n_atoms": N * tpl.n, "thermostat": a.thermostat,
           "fixed_iter": a.fixed_iter, "rounds": res, "summary": summ,
           "overhead": summ["flux"]["ms_per_step"] / summ["no flux"]["ms_per_step"] - 1.0, "device": str(jax.devices()[0])}
    print(json.dumps({k: v for k, v in out.items() if k != "rounds"}), flush=True)
    json.dump(out, open(stem + f"_speed{k}_{a.thermostat}{'_it%d' % a.fixed_iter if a.fixed_iter else ''}.json", "w"), indent=1)
