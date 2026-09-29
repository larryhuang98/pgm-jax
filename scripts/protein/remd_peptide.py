"""Temperature replica exchange of a solvated peptide, and plain MD for comparison (`pgm-jax remd`).

A validation of pgm_jax.md.remd: pGM electrostatics (placeholder unless a residue library is given)
+ ff19SB-form bonded terms with CMAP, rigid water, X-H constraints + hydrogen mass repartitioning
(3.024 amu), Bussi thermostat, dt 2 fs, NVT, 0.9 nm cutoff (docs/protein_ff.md).

The first run minimises and equilibrates at --tmin-K (--equil-ps) and saves <out>_equil.chk; the
other mode starts from the same state.  remd writes <out>_Tkk.{log,nc,rst7} (temperature k),
<out>_remd.log (exchanges), <out>_remd.json (summary) and <out>.remd.chk (continue with --resume);
plain writes <out>_plain.{log,nc,chk}.  analyze prints acceptance, round trips, speed and the
backbone phi/psi populations of every residue at each REMD temperature and in the plain run
(after --skip-ps), with block errors, and writes <out>_analysis.json.  bench times the replica
engines (batched for each count in --bench, and sequential for the largest) without exchanges.
Populations:
    alpha_L: phi > 0;  alpha_R: phi < 0 and -120 <= psi < 50;  otherwise beta (phi < -90) or PPII.

Usage:

    python scripts/protein/build_amber.py --sequence "ACE ALA ALA ALA NME" runs/remd/ala3 --buffer-A 8
    python scripts/protein/remd_peptide.py remd runs/remd/ala3.prmtop runs/remd/ala3.inpcrd --out runs/remd/ala3
        --replicas 8 --tmin-K 300 --tmax-K 400 --time-ns 2
    python scripts/protein/remd_peptide.py plain runs/remd/ala3.prmtop runs/remd/ala3.inpcrd --out runs/remd/ala3
        --time-ns 2
    python scripts/protein/remd_peptide.py analyze runs/remd/ala3.prmtop runs/remd/ala3.inpcrd --out runs/remd/ala3
    python scripts/protein/remd_peptide.py bench runs/remd/ala3.prmtop runs/remd/ala3.inpcrd --out runs/remd/ala3
        --bench 1 2 4 8 16
    python scripts/protein/remd_peptide.py --help

Inputs: the tleap prmtop and coordinates (scripts/protein/build_amber.py), optionally --library.
Outputs: as above (prefix --out).
Units: --tmin-K/--tmax-K K, --time-ns ns per replica, --dt-fs fs, durations in ps (--traj-ps,
--report-ps, --equil-ps, --skip-ps), --tau-ps ps, --exchange-every steps.
Runtime: GPU (all replicas batched with jax.vmap).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import numpy as np

from pgm_jax.cli.args import add_dipole_tol_arg, add_dt_arg, setup_logging
from pgm_jax.fit.reweighting import backbone_torsions
from pgm_jax.md.flexible import FlexibleSimulation
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_trajectory
from pgm_jax.md.remd import ReplicaExchange, geometric_ladder
from pgm_jax.md.thermostats import Bussi
from pgm_jax.protein import ResidueLibrary, amber_template, load_amber

jax.config.update("jax_enable_x64", True)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("remd", "plain", "analyze", "bench"), help="what to do")
    ap.add_argument("prmtop", help="tleap prmtop")
    ap.add_argument("inpcrd", help="tleap coordinates")
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    ap.add_argument("--library", default=None, help="pGM residue library (JSON); default: placeholder")
    ap.add_argument("--replicas", type=int, default=8, help="number of replicas (geometric ladder)")
    ap.add_argument("--tmin-K", type=float, default=300.0, help="lowest temperature [K] (and of the plain run)")
    ap.add_argument("--tmax-K", type=float, default=400.0, help="highest temperature [K]")
    ap.add_argument("--time-ns", type=float, default=2.0, help="length per replica [ns]")
    ap.add_argument("--exchange-every", type=int, default=250, help="steps between exchange attempts")
    ap.add_argument("--traj-ps", type=float, default=1.0, help="time between frames [ps]")
    ap.add_argument("--report-ps", type=float, default=10.0, help="time between log lines [ps]")
    ap.add_argument("--equil-ps", type=float, default=100.0, help="equilibration at tmin before both runs [ps]")
    ap.add_argument("--skip-ps", type=float, default=200.0, help="time discarded before the analysis [ps]")
    ap.add_argument("--sequential", action="store_true", help="replicas one after the other instead of batched (vmap)")
    ap.add_argument("--resume", action="store_true", help="continue the REMD run from <out>.remd.chk")
    add_dt_arg(ap, 2.0)
    add_dipole_tol_arg(ap)
    ap.add_argument("--tau-ps", type=float, default=1.0, help="Bussi time constant [ps]")
    ap.add_argument("--seed", type=int, default=1, help="random seed")
    ap.add_argument("--bench", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="replica counts timed by bench")
    ap.add_argument("--bench-steps", type=int, default=2000, help="timed steps per count")
    return ap


def populations(X: np.ndarray, top: object) -> np.ndarray:
    """Return the indicators (F, residues, 4) of alpha_R, beta, PPII, alpha_L for frames X (F, n_prot, 3) [nm].

    top is the peptide's topology (backbone torsions from fit.reweighting.backbone_torsions); the
    classes are those of the module docstring.
    """
    phi, psi = (np.degrees(np.asarray(t)) for t in backbone_torsions(X, top))
    aL = phi > 0
    aR = ~aL & (psi >= -120) & (psi < 50)
    ext = ~aL & ~aR
    return np.stack([aR, ext & (phi < -90), ext & (phi >= -90), aL], -1).astype(float)  # (F, res, 4)


def summarize(label: str, X: np.ndarray, top: object, nblocks: int = 5) -> dict:
    """Print and return the populations per residue with block errors (frames X (F, n_prot, 3) [nm]).

    Returns
    -------
    dict
        frames, mean (residues, 4) and err (residues, 4) as lists.
    """
    P = populations(X, top)
    nres = P.shape[1]
    F = len(P)
    B = np.array([b.mean(0) for b in np.array_split(P, nblocks)])
    m, e = P.mean(0), B.std(0, ddof=1) / np.sqrt(nblocks)
    print(f"{label}: {F} frames")
    for r in range(nres):
        print(
            f"   residue {r + 1}: "
            + "  ".join(f"{name} {m[r, i]:.3f}+-{e[r, i]:.3f}" for i, name in enumerate(("aR", "beta", "PPII", "aL")))
        )
    return {"frames": F, "mean": m.tolist(), "err": e.tolist()}


def read_log(path: str) -> dict[str, np.ndarray]:
    """Return the columns of a Simulation / REMD log (header "# step ...") as float arrays."""
    rows, cols = [], None
    with open(path) as fh:
        lines = fh.readlines()
    for ln in lines:
        if ln.startswith("# ") and ln.split()[1] == "step":
            cols = ln.split()[1:]
        elif ln.strip() and not ln.startswith("#"):
            rows.append([float(x) for x in ln.split()])
    return dict(zip(cols, np.array(rows).T))


def energies(label: str, log: dict, skip_ps: float, nblocks: int = 5) -> dict:
    """Print and return the mean temperature [K] and potential energy [kJ/mol] after skip_ps, with block errors."""
    keep = log["time_ps"] >= log["time_ps"][0] + skip_ps - 1e-6
    out = {}
    for c in ("temp_K", "epot"):
        x = log[c][keep]
        B = np.array([b.mean() for b in np.array_split(x, nblocks)])
        out[c] = (float(x.mean()), float(B.std(ddof=1) / np.sqrt(nblocks)))
    print(
        f"{label}: T {out['temp_K'][0]:.2f} +- {out['temp_K'][1]:.2f} K, U {out['epot'][0]:.1f} +- "
        f"{out['epot'][1]:.1f} "
        f"kJ/mol ({int(keep.sum())} log lines)"
    )
    return out


def analyze(a: argparse.Namespace, sl: slice, top: object) -> None:
    """Print and write (<out>_analysis.json) the REMD summary and the populations of every run (`analyze`).

    Parameters
    ----------
    a : argparse.Namespace
        Options.
    sl : slice
        Atoms of the peptide in the system.
    top : object
        The peptide's topology (backbone torsions).
    """
    res = {}
    js = a.out + "_remd.json"
    if os.path.exists(js):
        with open(js) as fh:
            d = json.load(fh)
        print(
            f"REMD: {len(d['temperatures_K'])} replicas, {d['time_ps']:.0f} ps per replica, {d['exchanges']} exchange "
            "attempts"
        )
        print("   T (K):               " + " ".join(f"{t:7.2f}" for t in d["temperatures_K"]))
        print("   neighbour acceptance: " + " ".join(f"{x:.3f}" for x in d["neighbour_acceptance"]))
        print(f"   round trips {d['round_trips_total']} (per replica {d['round_trips']}), transits {d['transits']}")
        print(f"   {d['ns_per_day_per_replica']:.1f} ns/day per replica, {d['ns_per_day_aggregate']:.1f} aggregate")
        res["remd"] = d
        for k, T in enumerate(d["temperatures_K"]):
            e = energies(f"REMD T = {T:.1f} K", read_log(f"{a.out}_T{k:02d}.log"), a.skip_ps)
            X, _, t = read_trajectory(f"{a.out}_T{k:02d}.nc", atoms=np.arange(sl.start, sl.stop))
            keep = t >= t[0] + a.skip_ps - 1e-6
            res[f"T{k:02d}"] = dict(summarize(f"REMD T = {T:.1f} K", X[keep] * 0.1, top), **e)
    if os.path.exists(a.out + "_plain.nc"):
        e = energies(f"plain MD T = {a.tmin_K:.1f} K", read_log(a.out + "_plain.log"), a.skip_ps)
        X, _, t = read_trajectory(a.out + "_plain.nc", atoms=np.arange(sl.start, sl.stop))
        keep = t >= t[0] + a.skip_ps - 1e-6
        res["plain"] = dict(summarize(f"plain MD T = {a.tmin_K:.1f} K", X[keep] * 0.1, top), **e)
    with open(a.out + "_analysis.json", "w") as fh:
        json.dump(res, fh, indent=1)


def equilibrated(a: argparse.Namespace, sim: FlexibleSimulation) -> None:
    """Load <out>_equil.chk, or minimise and equilibrate at tmin for --equil-ps and save it."""
    equil = a.out + "_equil"
    if os.path.exists(equil + ".chk"):
        sim.load_checkpoint(equil + ".chk")
        print(f"# equilibrated state from {equil}.chk", flush=True)
    else:
        t0 = time.time()
        print("# minimise:", sim.minimize(300), flush=True)
        n = int(round(a.equil_ps / sim.dt))
        sim.run(n, report_every=max(n // 10, 1), prefix=equil)
        sim.save_checkpoint(equil + ".chk")
        sim.write_restart(equil + ".rst7")
        print(f"# equilibration {a.equil_ps:g} ps: {time.time() - t0:.0f} s", flush=True)


def bench(a: argparse.Namespace, sim: FlexibleSimulation) -> None:
    """Time plain MD and the replica engines for the replica counts of --bench (the `bench` mode)."""
    start = sim.state
    n = a.bench_steps
    dt = sim.dt

    def timed(label, R, advance, states):
        """Time n steps of `advance` after 500 compile steps and print ms/step, ns/day and CG iterations."""
        advance(500)  # compile
        t0 = time.time()
        advance(n)
        el = time.time() - t0
        cg = np.mean([float(st.cg_total) / int(st.step) for st in states()])
        print(
            f"{label} R={R:2d}: {el / n * 1e3:.3f} ms/step, {el / n / R * 1e3:.3f} ms per replica-step, "
            f"{n * dt / 1000 / el * 86400:.1f} ns/day per replica, {R * n * dt / 1000 / el * 86400:.1f} "
            f"aggregate; CG {cg:.2f}",
            flush=True,
        )

    for R in a.bench:
        sim.state = start
        if R == 1:  # plain MD of the same system
            timed("plain MD  ", 1, sim.advance, lambda: [sim.state])
            continue
        for batched in (True, False) if R == max(a.bench) else (True,):
            rep = ReplicaExchange(
                sim, geometric_ladder(a.tmin_K, a.tmax_K, R), batched=batched, seed=a.seed, log=None
            ).replicas
            timed("batched   " if batched else "sequential", R, rep.advance, lambda: [rep.state(k) for k in range(R)])


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the mode (see the module docstring)."""
    a = build_parser().parse_args(argv)
    setup_logging()
    elec = ResidueLibrary.load(a.library) if a.library else "placeholder"
    asys = load_amber(a.prmtop, a.inpcrd, electrostatics=elec)
    kp = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
    if len(kp) != 1:
        raise SystemExit("expected one peptide chain")
    kp = kp[0]
    prot = asys.molecules[kp]
    sl = asys.system().atom_slice(kp)
    if a.mode == "analyze":
        analyze(a, sl, prot.spec.top)
        return
    dt = a.dt_fs / 1000
    tpl = amber_template(prot, a.prmtop)
    st = MDSettings().replace(cutoff=0.9, skin=0.1, dipole_tol=a.dipole_tol)
    sim = FlexibleSimulation(
        asys.system(),
        asys.templates({kp: tpl}),
        asys.system_positions(),
        asys.box,
        st,
        dt=dt,
        thermostat=Bussi(a.tau_ps),
        temperature=a.tmin_K,
        constraints="h-bonds",
        hmr=3.024,
        log=sys.stdout,
        seed=a.seed,
    )
    equilibrated(a, sim)
    nsteps = int(round(a.time_ns * 1000.0 / dt))
    traj, report = int(round(a.traj_ps / dt)), int(round(a.report_ps / dt))
    if a.mode == "bench":
        bench(a, sim)
    elif a.mode == "plain":
        sim.run(nsteps, report_every=report, traj_every=traj, checkpoint_every=report * 10, prefix=a.out + "_plain")
    else:
        T = geometric_ladder(a.tmin_K, a.tmax_K, a.replicas)
        rex = ReplicaExchange(
            sim, T, exchange_every=a.exchange_every, batched=not a.sequential, seed=a.seed, log=sys.stdout
        )
        if a.resume:
            rex.load_checkpoint(a.out + ".remd.chk")
        left = nsteps - rex.step
        s = rex.run(
            left, report_every=report, traj_every=traj, checkpoint_every=report * 10, prefix=a.out, append=a.resume
        )
        keys = ("neighbour_acceptance", "round_trips_total", "ns_per_day_per_replica", "ns_per_day_aggregate")
        print(json.dumps({k: s[k] for k in keys}), flush=True)


if __name__ == "__main__":
    main()
