"""Compare constrained MD of the engine with pmemd.pgm (SHAKE) on the same pGM model (docs/shake.md).

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

The pmemd helpers (run_pmemd, sp_mdin, ...) come from scripts/protein/check_pgm_prmtop.py, the
histograms and the comparison table from scripts/validation/validate_shake.py.

Usage:

    # prmtop, mdin, minimisation, single point
    JAX_PLATFORMS=cpu python scripts/validation/shake_vs_pmemd.py prep --system meoh125
    # one pmemd run (runs 0..n-1 in parallel)
    python scripts/validation/shake_vs_pmemd.py pmemd --system meoh125 --run 0 --kind cpu
    python scripts/validation/shake_vs_pmemd.py engine --system meoh125 --run 0      # one engine run
    python scripts/validation/shake_vs_pmemd.py analyze --system meoh125             # -> compare.json
    python scripts/validation/shake_vs_pmemd.py --help

Inputs: runs/meoh/<system>.{prmtop,inpcrd} (tleap); pmemd.pgm(.cuda) of PGM_PMEMD_BIN.
Outputs: runs/meoh/<system>/ (prmtop, single_point.json, pmemd and engine runs, <tag>.npz,
compare.json); printed tables.
Units: --time-ns ns, --dt-fs fs, --hmr-amu amu; kcal/mol (single point), nm, K, g/cm^3.
Runtime: pmemd runs on a CPU or GPU node, engine runs on a GPU; minutes to hours per run.
Sets jax_enable_x64.
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

import jax
import jax.numpy as jnp
import numpy as np
from validate_shake import compare_rows, hist_frame, new_acc, save_blocks

from pgm_jax.cli.args import setup_logging
from pgm_jax.cli.main import load_script, scripts_dir
from pgm_jax.md.flexible import FlexibleSimulation
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.thermostats import Langevin
from pgm_jax.paths import repo_path
from pgm_jax.protein import amber_template, load_amber, pmemd_mdin, write_pgm_prmtop
from pgm_jax.protein.pmemd import pmemd_grid
from pgm_jax.units import KCAL, KE, KE_AMBER_PGM

jax.config.update("jax_enable_x64", True)
T0 = 298.0  # K


def check_pgm_prmtop() -> object:
    """Return scripts/protein/check_pgm_prmtop.py as a module (its pmemd helpers)."""
    return load_script(os.path.join(scripts_dir(), "protein", "check_pgm_prmtop.py"))


class Paths:
    """File locations of one methanol system (runs/meoh/<system>...)."""

    def __init__(self, system: str) -> None:
        """Set up the paths of `system` ("meoh125" or "meoh1000")."""
        self.system = system
        self.wd = repo_path("runs", "meoh", system)
        self.prm0 = repo_path("runs", "meoh", f"{system}.prmtop")
        self.crd0 = repo_path("runs", "meoh", f"{system}.inpcrd")
        self.prm = os.path.join(self.wd, f"{system}_pgm.prmtop")
        self.minrst = os.path.join(self.wd, "min/restrt")


def model(p: Paths) -> tuple[object, object, list]:
    """Return the Amber system (placeholder electrostatics), the methanol template and all templates."""
    asys = load_amber(p.prm0, p.crd0, electrostatics="placeholder")
    tpl = amber_template(asys.molecules[0], p.prm0)
    return asys, tpl, asys.templates({k: tpl for k in range(len(asys.molecules))})


def settings(box: np.ndarray) -> MDSettings:
    """Return the MD settings (0.9 nm, dipole tolerance 1e-5, pmemd_grid of the box [nm])."""
    return MDSettings().replace(cutoff=0.9, dipole_tol=1e-5, pme_grid=pmemd_grid(box))


def atoms_of(names):
    """Return the atom indices within a methanol (MEOHBOX names): C, O, hydroxyl H, methyl H's."""
    return names.index("C1"), names.index("O1"), names.index("HO1"), [names.index(h) for h in ("HC1", "HC2", "HC3")]


# ----------------------------------------------------------------------------- pmemd
def prep(p: Paths, kind: str) -> None:
    """Write the pmemd-pgm prmtop, minimise with pmemd and compare a single point of both codes."""
    ck = check_pgm_prmtop()
    engine, read_energies, run_pmemd, settings_for, sp_mdin = (
        ck.engine,
        ck.read_energies,
        ck.run_pmemd,
        ck.settings_for,
        ck.sp_mdin,
    )
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
    with open(os.path.join(p.wd, "single_point.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def pmemd(
    p: Paths, run: int, kind: str, ns: float, equil_ps: float = 50.0, dt_fs: float = 2.0, tag: str = "pm"
) -> None:
    """Run one pmemd run: 10 ps heating at 0.5 fs from 0 K, equil_ps [ps] and ns [ns] of production at dt_fs [fs].

    kind: "cpu", "gpu_spfp" or "gpu_dpfp"; the run directory is <wd>/<tag><run> (seed 1000 + 17 run).
    """
    run_pmemd = check_pgm_prmtop().run_pmemd
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
def engine_md(
    p: Paths,
    run: int,
    ns: float,
    equil_ps: float = 50.0,
    frame_ps: float = 0.5,
    hmr: float | None = None,
    dt_fs: float = 2.0,
    tag: str = "engine",
    cons: str = "h-bonds",
) -> None:
    """Run one engine run from pmemd's minimised structure and save its samples (<wd>/<tag><run>.npz).

    Velocities at 298 K, equil_ps [ps] of equilibration, ns [ns] of production with a frame every
    frame_ps [ps]; hmr: hydrogen mass [amu] (None: unchanged); cons: the engine's constraints.
    """
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
        thermostat=Langevin(1.0),
        temperature=T0,
        constraints=cons,
        hmr=hmr,
        params=P,
        seed=2000 + 17 * run,
        log=sys.stdout,
    )
    every = int(round(frame_ps / dt))  # dt in ps
    sim.advance(int(round(equil_ps / dt)) // every * every)
    idx = atoms_of(asys.molecules[0].atom_names)
    nmol = len(asys.molecules)
    n_frames = int(round(ns * 1000.0 / frame_ps))
    rec = {k: [] for k in ("epot", "temp", "temp_half", "temp_com", "temp_internal", "shake_err", "rattle_err")}
    L = np.diag(np.asarray(asys.box))
    acc = new_acc()
    w0, n0, c0 = time.time(), int(sim.state.step), float(sim.state.cg_total)
    for f in range(n_frames):
        sim.advance(every)
        o = sim.observables()
        rec["epot"].append(o["epot"])
        rec["temp"].append(o["temp_K"])
        rec["temp_com"].append(o["temp_com"])
        rec["temp_half"].append(o["temp_half"])
        rec["temp_internal"].append(o["temp_internal"])
        rec["shake_err"].append(o["shake_err"])
        rec["rattle_err"].append(o["rattle_err"])
        hist_frame(sim.positions(), L, idx, nmol, acc)
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


def analyze(p: Paths, tags: list[str]) -> None:
    """Merge the runs of each tag and print / write the comparison table (compare.json)."""
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


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the mode (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("prep", "pmemd", "engine", "analyze"), help="what to do")
    ap.add_argument("--system", default="meoh125", choices=["meoh125", "meoh1000"], help="methanol box")
    ap.add_argument("--kind", default="cpu", help="pmemd build: cpu | gpu_spfp | gpu_dpfp")
    ap.add_argument("--run", type=int, default=0, help="run index (seed and directory)")
    ap.add_argument("--time-ns", type=float, default=0.25, help="production [ns]")
    ap.add_argument("--hmr-amu", type=float, default=None, help="engine: hydrogen mass [amu] (default: unchanged)")
    ap.add_argument("--dt-fs", type=float, default=2.0, help="time step [fs]")
    ap.add_argument("--constraints", default="h-bonds", help="engine: constraints (h-bonds, all-bonds, none)")
    ap.add_argument("--tag", default="engine", help="name of the run set (pmemd: default pm)")
    ap.add_argument("--tags", default="pmemd,engine", help="analyze: run sets to compare")
    a = ap.parse_args(argv)
    setup_logging()
    p = Paths(a.system)
    if a.mode == "prep":
        prep(p, a.kind)
    elif a.mode == "pmemd":
        pmemd(p, a.run, a.kind, a.time_ns, dt_fs=a.dt_fs, tag=a.tag if a.tag != "engine" else "pm")
    elif a.mode == "engine":
        engine_md(p, a.run, a.time_ns, hmr=a.hmr_amu, dt_fs=a.dt_fs, tag=a.tag, cons=a.constraints)
    else:
        analyze(p, a.tags.split(","))


if __name__ == "__main__":
    main()
