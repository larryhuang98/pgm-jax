"""Flexible methanol for the paper: single-molecule forces vs the gas-phase model, NPT density,
temperature partition, NVE energy conservation at two time steps and two precisions, speed.
Needs runs/flex/methanol.flex (examples/flex_methanol_check.py or examples/fit_bonded_template.py).
    python paper/scripts/flex_methanol.py            # -> paper/data/flex_methanol.json"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box
from pgm_jax.units import KB
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.system import System

OUT = os.path.join(ROOT, "paper/data/flex_methanol.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tpl = FlexibleTemplate.load(os.path.join(ROOT, "runs/flex/methanol.flex"))
N, T = 216, 298.0
out = {"n_mol": N, "n_atoms": N * tpl.n, "T": T, "exp_density": 0.7866, "families": list(tpl.settings["families"]),
       "device": str(jax.devices()[0])}

# 1. one molecule in a 4 nm box vs the gas-phase model the bonded terms were fitted with
rng = np.random.default_rng(0)
x = np.asarray(tpl.spec.ref_xyz) + 0.003 * rng.normal(size=(tpl.n, 3))
s1 = FlexibleSimulation(System([tpl.pgm]), [tpl], x + 2.0, np.eye(3) * 4.0,
                        MDSettings(precision="double", dipole_tol=1e-9, cutoff=1.8, skin=0.05, lj_lrc=False),
                        ensemble="nve", log=None)
P = jax.tree_util.tree_map(jnp.asarray, tpl.P)
g = np.asarray(jax.grad(lambda R: tpl.model.energy(tpl.index, R, P)[0])(jnp.asarray(x)))
F = np.asarray(s1.state.dyn.force)
out["single_molecule"] = {"max_abs_diff": float(np.abs(F + g).max()), "rms_force": float(np.sqrt(np.mean(g ** 2))),
                          "units": "kJ/mol/nm"}
print(out["single_molecule"], flush=True)

# 2. NVT 2 ps -> NPT 100 ps
st = MDSettings()                                            # 0.9 nm, PME, tol 1e-5, mixed, LJ tail
pos, H = liquid_box(tpl, N, 0.55, seed=1, min_dist=0.18)
sys_ = System([tpl.pgm] * N)
dt = 0.0005
nvt = FlexibleSimulation(sys_, [tpl] * N, pos, H, st, dt=dt, ensemble="nvt", temperature=T, gamma=5.0, log=None)
nvt._advance(4000)
npt = FlexibleSimulation(sys_, [tpl] * N, nvt.positions_nm(), np.asarray(nvt.state.box), st, dt=dt, ensemble="npt",
                         temperature=T, gamma=1.0, barostat_interval=25, vel_nm_ps=nvt.velocities_nm_ps(), log=None)
rec = {k: [] for k in ("time_ps", "density", "temp_com", "temp_internal", "epot")}
npt._advance(1000)                                            # compile outside the timing
t0, s0 = time.time(), int(npt.state.step)
for k in range(200):                                          # 200 x 0.5 ps
    npt._advance(1000)
    o = npt.observables()
    rec["time_ps"].append(round(o["time_ps"], 3)); rec["density"].append(o["density_g_cm3"])
    rec["temp_com"].append(o["temp_com"]); rec["temp_internal"].append(o["temp_internal"]); rec["epot"].append(o["epot"])
wall = time.time() - t0
steps = int(npt.state.step) - s0
out["npt"] = rec
half = np.array(rec["density"][100:])
blocks = np.array([b.mean() for b in np.array_split(half, 5)])
out["npt_summary"] = {"density_mean_last50ps": float(half.mean()), "density_se": float(blocks.std(ddof=1) / np.sqrt(5)),
                      "temp_com_mean": float(np.mean(rec["temp_com"][100:])),
                      "temp_internal_mean": float(np.mean(rec["temp_internal"][100:])),
                      "ns_per_day": steps * dt / 1000.0 / (wall / 86400.0), "ms_per_step": 1000.0 * wall / steps,
                      "barostat_interval": 25, "dt_fs": 0.5}
print(out["npt_summary"], flush=True)
x_eq, v_eq, H_eq = npt.positions_nm(), npt.velocities_nm_ps(), np.asarray(npt.state.box)

# 3. NVE from the equilibrated state
out["nve"] = []
for label, dt_n, prec, tol, ps in (("0.5 fs, mixed, tol 1e-5", 0.0005, "mixed", 1e-5, 20.0),
                                   ("0.25 fs, mixed, tol 1e-5", 0.00025, "mixed", 1e-5, 10.0),
                                   ("0.5 fs, double, tol 1e-8", 0.0005, "double", 1e-8, 10.0)):
    s = FlexibleSimulation(sys_, [tpl] * N, x_eq, H_eq, MDSettings(precision=prec, dipole_tol=tol), dt=dt_n,
                           ensemble="nve", vel_nm_ps=v_eq, log=None)
    every = int(round(0.1 / dt_n))
    t, E = [], []
    for _ in range(int(round(ps / 0.1))):
        s._advance(every)
        o = s.observables()
        t.append(round(o["time_ps"], 4)); E.append(o["etot"])
    t, E = np.array(t), np.array(E)
    dof = s.integ.dof
    slope = np.polyfit(t, E, 1)[0]                           # kJ/mol/ps
    r = {"label": label, "time_ps": t.tolist(), "etot": E.tolist(), "dof": dof,
         "drift_kT_per_ns_per_dof": float(slope * 1000.0 / (KB * T) / dof),
         "rms_fluct_kT": float(np.std(E - np.polyval(np.polyfit(t, E, 1), t)) / (KB * T))}
    out["nve"].append(r)
    print(label, {k: v for k, v in r.items() if k not in ("time_ps", "etot")}, flush=True)
out["kT_total"] = KB * T
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
