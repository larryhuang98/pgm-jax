"""Validation of holonomic bond constraints in the flexible engine (docs/shake.md).

Liquid methanol with the fitted pGM model (runs/flex/methanol.flex; 216 molecules, 1,296 atoms):
    python scripts/validate_shake.py equil                  # NVT 2 ps + NPT 150 ps (X-H constraints, 2 fs) -> runs/shake/eq.npz
    python scripts/validate_shake.py nve --prec mixed       # NVE drift of every configuration -> runs/shake/nve_mixed.json
    python scripts/validate_shake.py sample hb-2 --ns 2     # NPT, Langevin 1/ps: density, U, T, RDF, angles -> runs/shake/sample_hb-2.npz
    python scripts/validate_shake.py analyze                # table of the samples with block errors
Configurations (CONFIGS): none-0.5 (no constraints, 0.5 fs), hb-* (X-H bonds), hmr-* (X-H bonds and 3.024 amu
hydrogens), ab-* (every bond); the number is the time step in fs."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.system import System  # noqa: E402

OUT = os.path.join(ROOT, "runs/shake")
T0, N = 298.0, 216
CONFIGS = {
    "none-0.5": ("none", 0.5, None), "none-1": ("none", 1.0, None),
    "hb-0.5": ("h-bonds", 0.5, None), "hb-1": ("h-bonds", 1.0, None), "hb-2": ("h-bonds", 2.0, None),
    "hb-2.5": ("h-bonds", 2.5, None),
    "hmr-2": ("h-bonds", 2.0, 3.024), "hmr-3": ("h-bonds", 3.0, 3.024), "hmr-4": ("h-bonds", 4.0, 3.024),
    "hmr-5": ("h-bonds", 5.0, 3.024),
    "ab-2": ("all-bonds", 2.0, None), "ab-3": ("all-bonds", 3.0, None), "ab-hmr-4": ("all-bonds", 4.0, 3.024),
    "ab-hmr-5": ("all-bonds", 5.0, 3.024),
}


def template():
    return FlexibleTemplate.load(os.path.join(ROOT, "runs/flex/methanol.flex"))


def atoms_of(tpl):
    """Indices of C, O, the hydroxyl H and the methyl H's of the template."""
    el = list(tpl.spec.elements)
    bonds = [tuple(b) for b in tpl.spec.bonds]
    C, O = el.index("C"), el.index("O")
    nb = lambda a: [j if i == a else i for i, j in bonds if a in (i, j)]   # noqa: E731
    HO = [h for h in nb(O) if el[h] == "H"][0]
    HC = [h for h in nb(C) if el[h] == "H"]
    return C, O, HO, HC


def sim_for(name, x, H, prec="mixed", tol=1e-5, ensemble="npt", thermostat="langevin", vel=None, seed=0, log=None):
    cons, dt_fs, hmr = CONFIGS[name]
    tpl = template()
    st = MDSettings(precision=prec, dipole_tol=tol)             # 0.9 nm, PME, LJ tail (the model's settings)
    return FlexibleSimulation(System([tpl.pgm] * N), [tpl] * N, x, H, st, dt=dt_fs * 1e-3, ensemble=ensemble,
                              temperature=T0, gamma=1.0, thermostat=thermostat, tau_t=0.5, constraints=cons, hmr=hmr,
                              barostat_interval=max(1, int(round(0.1 / (dt_fs * 1e-3)))), vel_nm_ps=vel, seed=seed,
                              log=log)


def equil():
    tpl = template()
    pos, H = liquid_box(tpl, N, 0.55, seed=1, min_dist=0.18)
    st = MDSettings()
    s = FlexibleSimulation(System([tpl.pgm] * N), [tpl] * N, pos, H, st, dt=0.0005, ensemble="nvt", temperature=T0,
                           gamma=5.0, log=sys.stdout)
    s.run(4000, report=1000)
    s = sim_for("hb-2", s.positions_nm(), np.asarray(s.state.box), vel=s.velocities_nm_ps(), log=sys.stdout)
    s.run(75000, report=5000)                                   # 150 ps NPT at 2 fs
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "eq.npz"), x=s.positions_nm(), v=s.velocities_nm_ps(), H=np.asarray(s.state.box))


def load_eq():
    d = np.load(os.path.join(OUT, "eq.npz"))
    return d["x"], d["v"], d["H"]


def nve(prec, names, ps):
    """From the equilibrated state: 2 ps NVT (Bussi) with the configuration's own constraints,
    masses and time step, then NVE: drift (linear fit) and fluctuation of E_tot, constraint errors
    every step for the first 200 steps, CG iterations and speed."""
    x, v, H = load_eq()
    tol = 1e-5 if prec == "mixed" else 1e-9
    for name in names:
        path = os.path.join(OUT, f"nve_{prec}_{name}.json")
        res = {}
        cons, dt_fs, hmr = CONFIGS[name]
        dt = dt_fs * 1e-3
        pre = sim_for(name, x, H, prec, tol, ensemble="nvt", thermostat="bussi")
        pre._advance(int(round(2.0 / dt)))
        s = sim_for(name, pre.positions_nm(), np.asarray(pre.state.box), prec, tol, ensemble="nve",
                    vel=pre.velocities_nm_ps())
        err_x, err_v = 0.0, 0.0
        for _ in range(200):
            s._advance(1)
            o = s.observables()
            err_x, err_v = max(err_x, o.get("shake_err", 0.0)), max(err_v, o.get("rattle_err", 0.0))
        every = max(1, int(round(0.1 / dt)))
        s._advance(every)                                        # compiled for this block length
        t, E, T, cg = [], [], [], []
        c0, n0, w0 = float(s.state.cg_total), int(s.state.step), time.time()
        for _ in range(int(round(ps / 0.1))):
            s._advance(every)
            o = s.observables()
            t.append(o["time_ps"]); E.append(o["etot"]); T.append(o["temp_K"])
            err_x = max(err_x, o.get("shake_err", 0.0)); err_v = max(err_v, o.get("rattle_err", 0.0))
        wall = time.time() - w0
        steps = int(s.state.step) - n0
        t, E = np.array(t) - t[0], np.array(E)
        a, b = np.polyfit(t, E, 1)
        dof = s.integ.dof
        kT = KB * np.mean(T)
        res[name] = {"constraints": cons, "dt_fs": dt_fs, "hmr": hmr, "precision": prec, "dipole_tol": tol, "ps": ps,
                     "dof": dof, "T_mean": float(np.mean(T)),
                     "drift_kT_per_ns_per_dof": float(a * 1000.0 / (kT * dof)),
                     "fluct_rms_kJmol": float(np.std(E - (a * t + b))),
                     "fluct_over_kT_sqrt_dof": float(np.std(E - (a * t + b)) / (kT * np.sqrt(dof))),
                     "max_shake_err": err_x, "max_rattle_err": err_v,
                     "cg_per_step": (float(s.state.cg_total) - c0) / steps, "ms_per_step": 1e3 * wall / steps,
                     "ns_per_day": steps * dt * 1e-3 / (wall / 86400.0), "device": str(jax.devices()[0])}
        print(name, json.dumps(res[name]), flush=True)
        json.dump(res, open(path, "w"), indent=1)


def new_acc():
    return {"r_edges": np.linspace(0, 1.0, 201), "dih_edges": np.linspace(-180, 180, 73),
            "ang_edges": np.linspace(90, 130, 81), "co_edges": np.linspace(0.13, 0.155, 101),
            "oo": np.zeros(200), "oh": np.zeros(200), "dih": np.zeros(72), "coh": np.zeros(80),
            "co": np.zeros(100), "vol": 0.0, "frames": 0}


def hist_frame(X, L, idx, nmol, acc):
    """Accumulate one frame (positions nm, molecules contiguous; orthorhombic box lengths L nm):
    O-O and O-HO pair distances, H-C-O-H dihedrals, C-O-H angles and C-O lengths."""
    C, O, HO, HC = idx
    Xm = X.reshape(nmol, -1, 3)

    def pairs(A, B, same):
        cnt = np.zeros(len(acc["r_edges"]) - 1)
        for s in range(0, len(A), 250):                           # chunks: memory
            d = A[s:s + 250, None, :] - B[None, :, :]
            d -= np.round(d / L) * L
            r = np.linalg.norm(d, axis=-1)
            ii = np.arange(s, min(s + 250, len(A)))[:, None]
            jj = np.arange(len(B))[None, :]
            keep = (jj > ii) if same else (jj != ii)
            cnt += np.histogram(r[keep], bins=acc["r_edges"])[0]
        return cnt

    acc["oo"] += pairs(Xm[:, O], Xm[:, O], True)
    acc["oh"] += pairs(Xm[:, O], Xm[:, HO], False)
    acc["vol"] += float(np.prod(L))
    b2, b3 = Xm[:, O] - Xm[:, C], Xm[:, HO] - Xm[:, O]
    dih = []
    for h in HC:
        b1 = Xm[:, C] - Xm[:, h]
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        m1 = np.cross(n1, b2 / np.linalg.norm(b2, axis=1)[:, None])
        dih.append(np.degrees(np.arctan2(np.sum(m1 * n2, 1), np.sum(n1 * n2, 1))))
    acc["dih"] += np.histogram(np.concatenate(dih), bins=acc["dih_edges"])[0]
    u, w = Xm[:, C] - Xm[:, O], Xm[:, HO] - Xm[:, O]
    ang = np.degrees(np.arccos(np.clip(np.sum(u * w, 1) / np.linalg.norm(u, axis=1) / np.linalg.norm(w, axis=1), -1, 1)))
    acc["coh"] += np.histogram(ang, bins=acc["ang_edges"])[0]
    acc["co"] += np.histogram(np.linalg.norm(u, axis=1), bins=acc["co_edges"])[0]
    acc["frames"] += 1


def save_blocks(path, blocks, extra):
    out = {"blk_" + k: np.array([a[k] for a in blocks]) for k in ("oo", "oh", "dih", "coh", "co", "vol", "frames")}
    for k in ("r_edges", "dih_edges", "ang_edges", "co_edges"):
        out[k] = blocks[0][k]
    out.update(extra)
    np.savez(path, **out)


def sample(name, ns, seed, frame_ps=0.5, blocks=10):
    """NPT production (Langevin 1/ps, MC barostat every 0.1 ps) after 100 ps of equilibration with
    the configuration's own settings: per frame density, U, temperatures; per block the histograms."""
    x, v, H = load_eq()
    s = sim_for(name, x, H, seed=seed)
    dt = s.dt
    every = int(round(frame_ps / dt))                                   # dt in ps
    s._advance(int(round(100.0 / dt)) // every * every)                 # 100 ps equilibration
    idx = atoms_of(template())
    per = int(round(ns * 1000.0 / frame_ps)) // blocks
    rec = {k: [] for k in ("density", "epot", "temp", "temp_half", "temp_com", "temp_internal", "shake_err", "rattle_err",
                           "press")}
    B = []
    w0, n0, c0 = time.time(), int(s.state.step), float(s.state.cg_total)
    for b in range(blocks):
        acc = new_acc()
        for _ in range(per):
            s._advance(every)
            o = s.observables()
            o["press"] = s.pressure()                        # molecular virial (constraint forces are internal)
            for k, kk in (("density", "density_g_cm3"), ("epot", "epot"), ("temp", "temp_K"), ("temp_half", "temp_half"),
                          ("temp_com", "temp_com"),
                          ("temp_internal", "temp_internal"), ("shake_err", "shake_err"), ("rattle_err", "rattle_err"),
                          ("press", "press")):
                rec[k].append(o.get(kk, 0.0))
            hist_frame(s.positions_nm(), np.diag(np.asarray(s.state.box)), idx, N, acc)
        B.append(acc)
        print(name, b, np.mean(rec["density"][-per:]), np.mean(rec["epot"][-per:]) / N, np.mean(rec["temp"][-per:]),
              flush=True)
    wall = time.time() - w0
    steps = int(s.state.step) - n0
    meta = {"name": name, "config": CONFIGS[name], "ns": ns, "seed": seed, "dof": s.integ.dof,
            "ms_per_step": 1e3 * wall / steps, "ns_per_day": steps * dt * 1e-3 / (wall / 86400.0),
            "cg_per_step": (float(s.state.cg_total) - c0) / steps, "device": str(jax.devices()[0])}
    save_blocks(os.path.join(OUT, f"sample_{name}.npz"), B,
                {**{k: np.array(v) for k, v in rec.items()}, "meta": json.dumps(meta)})
    print(meta)


def block_stats(x, blocks=10):
    b = np.array([c.mean(0) for c in np.array_split(np.asarray(x, float), blocks)])
    return float(b.mean(0)), float(b.std(0, ddof=1) / np.sqrt(len(b)))


def compare_rows(paths, names, nmol, out):
    """Means with block standard errors; distributions as block-averaged histograms and their
    largest deviation from the first run's in units of the combined error.  Prints a table,
    writes `out` (json) and the distributions (npz next to it)."""
    rows = {}
    for path, name in zip(paths, names):
        if not os.path.exists(path):
            print("missing", path)
            continue
        d = np.load(path)
        r = {"meta": json.loads(str(d["meta"]))}
        for k in ("density", "temp", "temp_half", "temp_com", "temp_internal", "press"):
            if k in d.files:
                r[k] = block_stats(d[k])
        r["u_per_mol_kJ"] = block_stats(d["epot"] / nmol)
        if "shake_err" in d.files:
            r["max_shake_err"], r["max_rattle_err"] = float(d["shake_err"].max()), float(d["rattle_err"].max())
        edges = d["r_edges"]
        rr = 0.5 * (edges[1:] + edges[:-1])
        shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        fr, vol = d["blk_frames"][:, None], (d["blk_vol"] / d["blk_frames"])[:, None]
        dist = {"g_oo": d["blk_oo"] / (fr * shell[None] * (nmol * (nmol - 1) / 2) / vol),
                "g_oh": d["blk_oh"] / (fr * shell[None] * (nmol * (nmol - 1)) / vol)}
        for key in ("dih", "coh", "co"):
            dist[key] = d["blk_" + key] / d["blk_" + key].sum(1, keepdims=True)
        r["dist"] = {k: (v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))) for k, v in dist.items()}
        g = r["dist"]["g_oo"][0]
        k = int(np.argmax(g))
        r["g_oo_peak"] = [float(rr[k]), float(g[k])]
        mids = {"coh": 0.5 * (d["ang_edges"][1:] + d["ang_edges"][:-1]), "co": 0.5 * (d["co_edges"][1:] + d["co_edges"][:-1]),
                "dih": 0.5 * (d["dih_edges"][1:] + d["dih_edges"][:-1])}
        r["coh_mean"] = block_stats(dist["coh"] @ mids["coh"], len(dist["coh"]))
        r["co_mean"] = block_stats(dist["co"] @ mids["co"], len(dist["co"]))
        ecl = (np.abs(mids["dih"]) > 120).astype(float)                    # H-C-O-H in the trans well
        r["dih_trans_frac"] = block_stats(dist["dih"] @ ecl, len(dist["dih"]))
        rows[name] = r
    ref = names[0]
    f = lambda t, p=4: f"{t[0]:.{p}f}+-{t[1]:.{p}f}" if isinstance(t, tuple) else "-"   # noqa: E731
    print(f"{'run':10s} {'ns/day':>7s} {'density':>16s} {'U kJ/mol/mol':>17s} {'T':>13s} {'T_half':>13s} {'T_com':>13s} "
          f"{'T_int':>13s} {'C-O-H deg':>13s} {'C-O nm':>18s} {'trans':>14s} {'gOO peak':>10s} {'max dev (sigma) gOO/gOH/dih/COH/CO':>36s}")
    for name, r in rows.items():
        dev = []
        for key in ("g_oo", "g_oh", "dih", "coh", "co"):
            a, ea = r["dist"][key]
            b, eb = rows[ref]["dist"][key]
            e = np.sqrt(ea ** 2 + eb ** 2)
            ok = e > 0
            dev.append(float(np.max(np.abs(a - b)[ok] / e[ok])) if name != ref else 0.0)
        r["max_dev_sigma"] = dict(zip(("g_oo", "g_oh", "dih", "coh", "co"), dev))
        m = r["meta"]
        print(f"{name:10s} {m.get('ns_per_day') or 0:7.1f} {f(r.get('density'))} {f(r['u_per_mol_kJ'], 3)} "
              f"{f(r.get('temp'), 2)} {f(r.get('temp_half'), 2)} {f(r.get('temp_com'), 2)} {f(r.get('temp_internal'), 2)} "
              f"{f(r['coh_mean'], 2)} "
              f"{f(r['co_mean'], 5)} {f(r['dih_trans_frac'], 4)} {r['g_oo_peak'][0]:.3f}/{r['g_oo_peak'][1]:.2f} "
              f"P {f(r.get('press'), 0)} "
              + " ".join(f"{x:5.1f}" for x in dev)
              + (f"  shake {r['max_shake_err']:.1e} rattle {r['max_rattle_err']:.1e}" if "max_shake_err" in r else ""))
    json.dump({n: {k: v for k, v in r.items() if k != "dist"} for n, r in rows.items()}, open(out, "w"), indent=1)
    np.savez(out.replace(".json", "_dist.npz"),
             **{f"{n}__{k}": np.array(v) for n, r in rows.items() for k, v in r["dist"].items()})
    return rows


def peptide(ps=10.0):
    """The solvated peptide of the tests (ACE-ALA-SER-NME, TIP3P, NaCl; placeholder pGM, ff19SB-form
    bonded terms): every protein bond constrained (one cluster: the iterative solver) with 3.024 amu
    hydrogens at 4 fs, against X-H bonds at 2 fs; 5 ps Bussi, then NVE with the constraint errors
    of every step for 200 steps, drift and CG iterations."""
    from pgm_jax.protein import amber_template, load_amber
    prm, crd = os.path.join(ROOT, "tests/data/pep_wat.prmtop"), os.path.join(ROOT, "tests/data/pep_wat.inpcrd")
    asys = load_amber(prm, crd)
    tpl = {k: amber_template(m, prm) for k, m in enumerate(asys.molecules) if m.kind == "protein"}
    templates = asys.templates(tpl)
    st = MDSettings(cutoff=0.8, dipole_tol=1e-6, precision="double")
    path = os.path.join(OUT, "peptide.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    x0 = None
    for cons, dt_fs, hmr, opts in (("h-bonds", 2.0, None, None), ("all-bonds", 2.0, None, None),
                                   ("all-bonds", 4.0, 3.024, None), ("h-bonds", 4.0, 3.024, None),
                                   ("all-bonds", 2.0, None, {"dense_max": 40}), ("all-bonds", 4.0, 3.024, {"dense_max": 40})):
        key = f"{cons} {dt_fs:g} fs" + (f" H {hmr}" if hmr else "") + (" dense" if opts else "")
        if key in out:
            continue
        dt = dt_fs * 1e-3
        s = FlexibleSimulation(asys.system(), templates, asys.system_positions() if x0 is None else x0, asys.box, st,
                               dt=dt, ensemble="nvt", thermostat="bussi", tau_t=0.2, temperature=T0,
                               constraints=cons, hmr=hmr, constraint_options=opts, log=sys.stdout)
        if x0 is None:
            s.minimize(300)
            s._advance(int(round(5.0 / dt)))
            x0 = s.positions_nm()
        else:
            s._advance(int(round(5.0 / dt)))
        e = FlexibleSimulation(asys.system(), templates, s.positions_nm(), np.asarray(s.state.box), st, dt=dt,
                               ensemble="nve", constraints=cons, hmr=hmr, vel_nm_ps=s.velocities_nm_ps(),
                               constraint_options=opts, log=None)
        ex = ev = 0.0
        for _ in range(200):
            e._advance(1)
            o = e.observables()
            ex, ev = max(ex, o["shake_err"]), max(ev, o["rattle_err"])
        t, E, T = [], [], []
        every = int(round(0.1 / dt))
        w0, n0, c0 = time.time(), int(e.state.step), float(e.state.cg_total)
        for _ in range(int(round(ps / 0.1))):
            e._advance(every)
            o = e.observables()
            t.append(o["time_ps"]); E.append(o["etot"]); T.append(o["temp_K"])
            ex, ev = max(ex, o["shake_err"]), max(ev, o["rattle_err"])
        t = np.array(t) - t[0]
        a, b = np.polyfit(t, E, 1)
        kT = KB * np.mean(T)
        out[key] = {"blocks": e.constraints.describe(), "dof": e.integ.dof, "T": float(np.mean(T)),
                    "drift_kT_per_ns_per_dof": float(a * 1000.0 / (kT * e.integ.dof)),
                    "fluct_kJmol": float(np.std(np.array(E) - a * t - b)), "max_shake_err": ex, "max_rattle_err": ev,
                    "cg_per_step": (float(e.state.cg_total) - c0) / (int(e.state.step) - n0),
                    "ms_per_step": 1e3 * (time.time() - w0) / (int(e.state.step) - n0), "device": str(jax.devices()[0])}
        print(key, json.dumps(out[key]), flush=True)
        json.dump(out, open(path, "w"), indent=1)


def nve_table(prec):
    """Markdown table of the NVE runs (nve_<prec>_<config>.json)."""
    rows = []
    for name in CONFIGS:
        p = os.path.join(OUT, f"nve_{prec}_{name}.json")
        if os.path.exists(p):
            rows.append(json.load(open(p))[name] | {"name": name})
    print(f"| config | constraints | dt (fs) | H mass | drift (kT/ns/dof) | E fluct. (kJ/mol) | fluct / (kT sqrt(N_f)) | "
          f"max shake_err | max rattle_err | CG/step | ms/step ({rows[0]['device'] if rows else ''}) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['constraints']} | {r['dt_fs']:g} | {r['hmr'] or 1.008} | {r['drift_kT_per_ns_per_dof']:+.4f} | "
              f"{r['fluct_rms_kJmol']:.2f} | {r['fluct_over_kT_sqrt_dof']:.4f} | {r['max_shake_err']:.1e} | "
              f"{r['max_rattle_err']:.1e} | {r['cg_per_step']:.2f} | {r['ms_per_step']:.2f} |")


def analyze(names):
    compare_rows([os.path.join(OUT, f"sample_{n}.npz") for n in names], names, N, os.path.join(OUT, "analysis.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("equil", "nve", "nve-table", "sample", "analyze", "peptide"))
    ap.add_argument("names", nargs="*")
    ap.add_argument("--prec", default="mixed")
    ap.add_argument("--ps", type=float, default=20.0)
    ap.add_argument("--ns", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == "equil":
        equil()
    elif a.mode == "nve":
        nve(a.prec, a.names or list(CONFIGS), a.ps)
    elif a.mode == "peptide":
        peptide(a.ps)
    elif a.mode == "nve-table":
        nve_table(a.prec)
    elif a.mode == "sample":
        for n in a.names:
            sample(n, a.ns, a.seed)
    else:
        analyze(a.names or ["hb-0.5", "none-0.5", "hb-1", "hb-2", "hmr-4", "ab-2"])
