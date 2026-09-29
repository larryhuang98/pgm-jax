"""Constrained MD of the engine against pmemd.pgm with SHAKE, on the same pGM model.

System: Amber's methanol box (MEOHBOX of solvents.lib, frcmod.meoh, parm10 bonded terms; tleap
inputs in runs/meoh): `meoh125` (125 methanols, 750 atoms, 2.009 nm; pmemd.pgm on the CPU) or
`meoh1000` (2 x 2 x 2 copies, 6,000 atoms; pmemd.pgm.cuda), with placeholder pGM electrostatics
(Amber charges as Gaussian charges, pGM-pol polarizabilities and radii, no covalent dipoles) and
the prmtop's Lennard-Jones and Amber-form bonded terms (protein.amber_template), written for
pmemd-pgm by protein.write_pgm_prmtop.  Both codes: NVT 298 K, Langevin 1/ps, dt 2 fs, X-H bonds
constrained (pmemd: SHAKE, ntc = ntf = 2, tol 1e-7; the engine: constraints="h-bonds"), 9 A
cutoff, PME ~0.8 A order 6, LJ long-range correction, dipole tolerance 1e-5.  Independent runs
from the minimised structure (pmemd: heated 10 ps at 0.5 fs from 0 K; the engine: velocities at
298 K), 50 ps equilibration, then production with a frame every 0.5 ps; each run is one block
for the error bars.  The engine's charges and covalent dipoles are scaled by
sqrt(KE_AMBER_PGM / KE) (pmemd-pgm's Coulomb constant), so both run the same Hamiltonian.

    JAX_PLATFORMS=cpu python scripts/shake_vs_pmemd.py prep --system meoh125     # prmtop, mdin, minimisation, single point
    python scripts/shake_vs_pmemd.py pmemd --system meoh125 --run 0 --kind cpu   # one pmemd run (runs 0..n-1 in parallel)
    python scripts/shake_vs_pmemd.py engine --system meoh125 --run 0             # one engine run
    python scripts/shake_vs_pmemd.py analyze --system meoh125 --runs 8           # runs/meoh/<system>/compare.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts/protein"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from validate_shake import compare_rows, hist_frame, new_acc, save_blocks  # noqa: E402

from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.protein import amber_template, load_amber, pmemd_mdin, write_pgm_prmtop  # noqa: E402
from pgm_jax.protein.pmemd import pmemd_grid  # noqa: E402
from pgm_jax.units import KE, KE_AMBER_PGM  # noqa: E402

KCAL = 4.184
T0 = 298.0


class Paths:
    def __init__(self, system):
        self.system = system
        self.wd = os.path.join(ROOT, "runs/meoh", system)
        self.prm0 = os.path.join(ROOT, "runs/meoh", f"{system}.prmtop")
        self.crd0 = os.path.join(ROOT, "runs/meoh", f"{system}.inpcrd")
        self.prm = os.path.join(self.wd, f"{system}_pgm.prmtop")
        self.minrst = os.path.join(self.wd, "min/restrt")


def model(p):
    asys = load_amber(p.prm0, p.crd0, electrostatics="placeholder")
    tpl = amber_template(asys.molecules[0], p.prm0)
    return asys, tpl, asys.templates({k: tpl for k in range(len(asys.molecules))})


def settings(box):
    return MDSettings(cutoff=0.9, dipole_tol=1e-5, pme_grid=pmemd_grid(box))


def atoms_of(names):
    """Atom indices within a methanol (MEOHBOX names): C, O, hydroxyl H, methyl H's."""
    return names.index("C1"), names.index("O1"), names.index("HO1"), [names.index(h) for h in ("HC1", "HC2", "HC3")]


# ----------------------------------------------------------------------------- pmemd
def prep(p, kind):
    from check_pgm_prmtop import engine, read_energies, run_pmemd, settings_for, sp_mdin

    os.makedirs(p.wd, exist_ok=True)
    asys, tpl, templates = model(p)
    info = write_pgm_prmtop(asys, p.prm, templates)
    print("written", p.prm, {k: v for k, v in info.items() if k != "exported"})
    st = settings(asys.box)
    run_pmemd(kind, os.path.join(p.wd, "min"), p.prm, p.crd0, pmemd_mdin(st, asys.box, maxcyc=500, ntpr=100))
    # single point, float64 on both sides: the written file gives pmemd-pgm the engine's energies
    sp = settings_for(asys.box)
    res = engine(asys, templates, sp)
    e_eng = res[0] if isinstance(res, tuple) else res
    run_pmemd("cpu", os.path.join(p.wd, "sp"), p.prm, p.crd0, sp_mdin(sp, asys.box))
    e_pm = read_energies(os.path.join(p.wd, "sp/mdout"))
    out = {"engine": {k: float(v) for k, v in e_eng.items() if np.isscalar(v)}, "pmemd": e_pm}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(p.wd, "single_point.json"), "w"), indent=1)


def pmemd(p, run, kind, ns, equil_ps=50.0, dt_fs=2.0, tag="pm"):
    from check_pgm_prmtop import run_pmemd

    asys = load_amber(p.prm0, p.crd0, electrostatics="placeholder")
    st = settings(asys.box)
    wd = os.path.join(p.wd, f"{tag}{run}")
    dt = dt_fs * 1e-3
    seed = 1000 + 17 * run
    heat = pmemd_mdin(st, asys.box, nstlim=20000, dt=0.0005, temperature=T0, tempi=0.0, ntpr=2000, ntwr=20000, ig=seed)
    run_pmemd(kind, os.path.join(wd, "heat"), p.prm, p.minrst, heat)
    n_eq = int(round(equil_ps / dt))
    eq = pmemd_mdin(st, asys.box, nstlim=n_eq, dt=dt, temperature=T0, irest=1, ntpr=1000, ntwr=n_eq, ig=seed + 1)
    run_pmemd(kind, os.path.join(wd, "equil"), p.prm, os.path.join(wd, "heat/restrt"), eq)
    n = int(round(ns * 1000.0 / dt))  # ns -> steps (dt in ps)
    every = int(round(0.5 / dt))  # a frame every 0.5 ps
    prod = pmemd_mdin(
        st, asys.box, nstlim=n, dt=dt, temperature=T0, irest=1, ntpr=every, ntwx=every, ntwr=n, ig=seed + 2
    )
    secs = run_pmemd(kind, os.path.join(wd, "prod"), p.prm, os.path.join(wd, "equil/restrt"), prod)
    json.dump({"prod_seconds": secs, "steps": n, "dt": dt}, open(os.path.join(wd, "prod/wall.json"), "w"))


def pmemd_hist(p, tag="pm"):
    """Histograms and energies of every finished pmemd run (one block per run)."""
    from pgm_jax.md.io import read_trajectory

    asys = load_amber(p.prm0, p.crd0, electrostatics="placeholder")
    idx = atoms_of(asys.molecules[0].atom_names)
    nmol = len(asys.molecules)
    B, ep, T, nsday = [], [], [], []
    runs = [d for d in glob.glob(os.path.join(p.wd, f"{tag}*")) if re.fullmatch(tag + r"\d+", os.path.basename(d))]
    for d in sorted(os.path.join(r, "prod") for r in runs):
        if not os.path.exists(os.path.join(d, "wall.json")):
            continue
        X, box, _ = read_trajectory(os.path.join(d, "mdcrd"))
        acc = new_acc()
        for f in range(len(X)):
            hist_frame(X[f] * 0.1, box[f] * 0.1, idx, nmol, acc)
        B.append(acc)
        txt = open(os.path.join(d, "mdout")).read()
        steps = re.findall(r"NSTEP =\s+(\d+)\s+TIME\(PS\) =\s+\S+\s+TEMP\(K\) =\s+(\S+).*?EPtot\s+=\s+(\S+)", txt, re.S)
        steps = steps[1:-2]  # without step 0 and the averages
        ep.append(np.array([float(e) for _, _, e in steps]) * KCAL)
        T.append(np.array([float(t) for _, t, _ in steps]))
        w = json.load(open(os.path.join(d, "wall.json")))
        nsday.append(w["steps"] * w.get("dt", 0.002) * 1e-3 / (w["prod_seconds"] / 86400.0))
    n = min(len(e) for e in ep)
    meta = {
        "runs": len(B),
        "frames": int(sum(a["frames"] for a in B)),
        "ns_per_day": float(np.mean(nsday)),
        "note": "pmemd.pgm, SHAKE ntc=ntf=2",
    }
    save_blocks(
        os.path.join(p.wd, "pmemd.npz" if tag == "pm" else f"{tag}.npz"),
        B,
        {
            "epot": np.concatenate([e[:n] for e in ep]),
            "temp": np.concatenate([t[:n] for t in T]),
            "meta": json.dumps(meta),
        },
    )


# ----------------------------------------------------------------------------- engine
def engine_md(p, run, ns, equil_ps=50.0, frame_ps=0.5, hmr=None, dt_fs=2.0, tag="engine", cons="h-bonds"):
    from pgm_jax.md.flexible import FlexibleSimulation
    from pgm_jax.md.io import read_coordinates

    asys, tpl, templates = model(p)
    sys_ = asys.system()
    s = math.sqrt(KE_AMBER_PGM / KE)
    P = {k: jnp.asarray(v) for k, v in sys_.params0.items()}
    P["q"], P["cov"] = P["q"] * s, P["cov"] * s
    x = read_coordinates(p.minrst)[0] * 0.1  # pmemd's minimised structure
    dt = dt_fs * 1e-3
    sim = FlexibleSimulation(
        sys_,
        templates,
        x[np.asarray(asys.order)],
        asys.box,
        settings(asys.box),
        dt=dt,
        ensemble="nvt",
        temperature=T0,
        gamma=1.0,
        constraints=cons,
        hmr=hmr,
        params=P,
        seed=2000 + 17 * run,
        log=sys.stdout,
    )
    every = int(round(frame_ps / dt))  # dt in ps
    sim._advance(int(round(equil_ps / dt)) // every * every)
    idx = atoms_of(asys.molecules[0].atom_names)
    nmol = len(asys.molecules)
    n_frames = int(round(ns * 1000.0 / frame_ps))
    rec = {k: [] for k in ("epot", "temp", "temp_half", "temp_com", "temp_internal", "shake_err", "rattle_err")}
    L = np.diag(np.asarray(asys.box))
    acc = new_acc()
    w0, n0, c0 = time.time(), int(sim.state.step), float(sim.state.cg_total)
    for f in range(n_frames):
        sim._advance(every)
        o = sim.observables()
        rec["epot"].append(o["epot"])
        rec["temp"].append(o["temp_K"])
        rec["temp_com"].append(o["temp_com"])
        rec["temp_half"].append(o["temp_half"])
        rec["temp_internal"].append(o["temp_internal"])
        rec["shake_err"].append(o["shake_err"])
        rec["rattle_err"].append(o["rattle_err"])
        hist_frame(sim.positions_nm(), L, idx, nmol, acc)
        if f % 200 == 199:
            print(tag, run, f + 1, np.mean(rec["epot"]) / nmol, np.mean(rec["temp"]), flush=True)
    wall = time.time() - w0
    steps = int(sim.state.step) - n0
    meta = {
        "ns": ns,
        "run": run,
        "dof": sim.integ.dof,
        "ms_per_step": 1e3 * wall / steps,
        "hmr": hmr,
        "dt_fs": dt_fs,
        "constraints": cons,
        "ns_per_day": steps * dt * 1e-3 / (wall / 86400.0),
        "cg_per_step": (float(sim.state.cg_total) - c0) / steps,
        "device": str(jax.devices()[0]),
    }
    save_blocks(
        os.path.join(p.wd, f"{tag}_r{run}.npz"),
        [acc],
        {**{k: np.array(v) for k, v in rec.items()}, "meta": json.dumps(meta)},
    )
    print(meta)


def merge(p, tag):
    """One file of the engine runs `tag`_r*.npz (one block per run)."""
    files = sorted(glob.glob(os.path.join(p.wd, f"{tag}_r*.npz")))
    ds = [np.load(f) for f in files]
    n = min(len(d["epot"]) for d in ds)
    out = {
        k: np.concatenate([d[k] for d in ds])
        for k in ("blk_oo", "blk_oh", "blk_dih", "blk_coh", "blk_co", "blk_vol", "blk_frames")
    }
    for k in ("r_edges", "dih_edges", "ang_edges", "co_edges"):
        out[k] = ds[0][k]
    for k in ("epot", "temp", "temp_half", "temp_com", "temp_internal", "shake_err", "rattle_err"):
        out[k] = np.concatenate([d[k][:n] for d in ds])
    metas = [json.loads(str(d["meta"])) for d in ds]
    out["meta"] = json.dumps(
        {
            "runs": len(ds),
            "ns_per_day": float(np.mean([m["ns_per_day"] for m in metas])),
            "cg_per_step": float(np.mean([m["cg_per_step"] for m in metas])),
            "dof": metas[0]["dof"],
            "device": metas[0]["device"],
        }
    )
    np.savez(os.path.join(p.wd, f"{tag}.npz"), **out)


def analyze(p, tags):
    asys = load_amber(p.prm0, p.crd0, electrostatics="placeholder")
    for t in tags:
        if t == "pmemd":
            pmemd_hist(p)
        elif t.startswith("pm"):  # e.g. pm1fs: pmemd runs pm1fs0, pm1fs1, ...
            pmemd_hist(p, t)
        else:
            merge(p, t)
    compare_rows(
        [os.path.join(p.wd, f"{t}.npz") for t in tags],
        tags,
        nmol=len(asys.molecules),
        out=os.path.join(p.wd, "compare.json"),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prep", "pmemd", "engine", "analyze"))
    ap.add_argument("--system", default="meoh125")
    ap.add_argument("--kind", default="cpu", help="pmemd build: cpu | gpu_spfp | gpu_dpfp")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--ns", type=float, default=0.25)
    ap.add_argument("--hmr", type=float, default=None)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--constraints", default="h-bonds")
    ap.add_argument("--tag", default="engine")
    ap.add_argument("--tags", default="pmemd,engine")
    a = ap.parse_args()
    p = Paths(a.system)
    if a.mode == "prep":
        prep(p, a.kind)
    elif a.mode == "pmemd":
        pmemd(p, a.run, a.kind, a.ns, dt_fs=a.dt, tag=a.tag if a.tag != "engine" else "pm")
    elif a.mode == "engine":
        engine_md(p, a.run, a.ns, hmr=a.hmr, dt_fs=a.dt, tag=a.tag, cons=a.constraints)
    else:
        analyze(p, a.tags.split(","))
