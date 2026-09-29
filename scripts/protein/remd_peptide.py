"""Temperature replica exchange of a solvated peptide (validation of pgm_jax.md.remd), and plain MD
of the same length for comparison: pGM electrostatics (placeholder unless a residue library is
given) + ff19SB-form bonded terms with CMAP, rigid water, X-H constraints + hydrogen mass
repartitioning, Bussi thermostat, dt 2 fs, NVT.

    python scripts/protein/build_amber.py --sequence "ACE ALA ALA ALA NME" runs/remd/ala3 --buffer 8
    python scripts/protein/remd_peptide.py remd runs/remd/ala3.prmtop runs/remd/ala3.inpcrd --out runs/remd/ala3 \\
        --replicas 8 --tmin 300 --tmax 400 --ns 2
    python scripts/protein/remd_peptide.py plain  runs/remd/ala3.prmtop runs/remd/ala3.inpcrd --out \\
        runs/remd/ala3 --ns 2
    python scripts/protein/remd_peptide.py analyze runs/remd/ala3.prmtop runs/remd/ala3.inpcrd --out runs/remd/ala3
    python scripts/protein/remd_peptide.py bench runs/remd/ala3.prmtop runs/remd/ala3.inpcrd --out runs/remd/ala3 \
        --bench 1 2 4 8 16

The first run minimises and equilibrates at tmin (--equil ps) and saves out_equil.chk; the other
mode starts from the same state.  remd writes out_Tkk.{log,nc,rst7} (temperature k), out_remd.log
(exchanges), out_remd.json (summary) and out.remd.chk (continue with --resume); plain writes
out_plain.{log,nc,chk}.  analyze
prints acceptance, round trips, speed and the backbone phi/psi populations of every residue at
each REMD temperature and in the plain run (after --skip ps), with block errors.  bench times the
replica engines (batched for each count in --bench, and sequential for the largest) without
exchanges.  Populations:
    alpha_L: phi > 0;  alpha_R: phi < 0 and -120 <= psi < 50;  otherwise beta (phi < -90) or PPII."""

import argparse
import json
import os
import sys
import time

import jax
import numpy as np

from pgm_jax.fit.reweighting import backbone_torsions
from pgm_jax.md.flexible import FlexibleSimulation
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_trajectory
from pgm_jax.md.remd import ReplicaExchange, geometric_ladder
from pgm_jax.protein import ResidueLibrary, amber_template, load_amber

jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("remd", "plain", "analyze", "bench"))
ap.add_argument("prmtop")
ap.add_argument("inpcrd")
ap.add_argument("--out", required=True, help="output prefix")
ap.add_argument("--library", default=None, help="pGM residue library (JSON); default: placeholder")
ap.add_argument("--replicas", type=int, default=8)
ap.add_argument("--tmin", type=float, default=300.0)
ap.add_argument("--tmax", type=float, default=400.0)
ap.add_argument("--ns", type=float, default=2.0, help="length per replica (ns)")
ap.add_argument("--exchange", type=int, default=250, help="steps between exchange attempts")
ap.add_argument("--traj", type=float, default=1.0, help="ps between frames")
ap.add_argument("--report", type=float, default=10.0, help="ps between log lines")
ap.add_argument("--equil", type=float, default=100.0, help="equilibration at tmin before both runs (ps)")
ap.add_argument("--skip", type=float, default=200.0, help="ps discarded before the analysis")
ap.add_argument("--sequential", action="store_true", help="replicas one after the other instead of batched (vmap)")
ap.add_argument("--resume", action="store_true", help="continue the REMD run from out.remd.chk")
ap.add_argument("--dt", type=float, default=0.002)
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--tau", type=float, default=1.0, help="Bussi time constant (ps)")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--bench", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="replica counts timed by bench")
ap.add_argument("--bench-steps", type=int, default=2000)
a = ap.parse_args()
asys = load_amber(a.prmtop, a.inpcrd, electrostatics=ResidueLibrary.load(a.library) if a.library else "placeholder")
kp = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
if len(kp) != 1:
    raise SystemExit("expected one peptide chain")
kp = kp[0]
prot = asys.molecules[kp]
sl = asys.system().atom_slice(kp)
top = prot.spec.top
nres = len(top.cmaps)


def populations(X):
    """Fractions (alpha_R, beta, PPII, alpha_L) per residue for frames X (F, n_prot, 3) nm."""
    phi, psi = (np.degrees(np.asarray(t)) for t in backbone_torsions(X, top))
    aL = phi > 0
    aR = ~aL & (psi >= -120) & (psi < 50)
    ext = ~aL & ~aR
    return np.stack([aR, ext & (phi < -90), ext & (phi >= -90), aL], -1).astype(float)  # (F, res, 4)


def summarize(label, X, nblocks=5):
    P = populations(X)
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


def read_log(path):
    """Columns of a Simulation / REMD log (header "# step ..."), as float arrays."""
    rows, cols = [], None
    for ln in open(path):
        if ln.startswith("# ") and ln.split()[1] == "step":
            cols = ln.split()[1:]
        elif ln.strip() and not ln.startswith("#"):
            rows.append([float(x) for x in ln.split()])
    return dict(zip(cols, np.array(rows).T))


def energies(label, log, nblocks=5):
    """Mean kinetic temperature and potential energy after --skip ps, with block errors."""
    keep = log["time_ps"] >= log["time_ps"][0] + a.skip - 1e-6
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


if a.mode == "analyze":
    res = {}
    js = a.out + "_remd.json"
    if os.path.exists(js):
        d = json.load(open(js))
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
            e = energies(f"REMD T = {T:.1f} K", read_log(f"{a.out}_T{k:02d}.log"))
            X, _, t = read_trajectory(f"{a.out}_T{k:02d}.nc", atoms=np.arange(sl.start, sl.stop))
            keep = t >= t[0] + a.skip - 1e-6
            res[f"T{k:02d}"] = dict(summarize(f"REMD T = {T:.1f} K", X[keep] * 0.1), **e)
    if os.path.exists(a.out + "_plain.nc"):
        e = energies(f"plain MD T = {a.tmin:.1f} K", read_log(a.out + "_plain.log"))
        X, _, t = read_trajectory(a.out + "_plain.nc", atoms=np.arange(sl.start, sl.stop))
        keep = t >= t[0] + a.skip - 1e-6
        res["plain"] = dict(summarize(f"plain MD T = {a.tmin:.1f} K", X[keep] * 0.1), **e)
    json.dump(res, open(a.out + "_analysis.json", "w"), indent=1)
    raise SystemExit(0)

tpl = amber_template(prot, a.prmtop)
st = MDSettings(cutoff=0.9, skin=0.1, dipole_tol=a.tol)
sim = FlexibleSimulation(
    asys.system(),
    asys.templates({kp: tpl}),
    asys.system_positions(),
    asys.box,
    st,
    dt=a.dt,
    ensemble="nvt",
    temperature=a.tmin,
    constraints="h-bonds",
    hmr=3.024,
    log=sys.stdout,
    thermostat="bussi",
    tau_t=a.tau,
    seed=a.seed,
)
equil = a.out + "_equil"
if os.path.exists(equil + ".chk"):
    sim.load(equil + ".chk")
    print(f"# equilibrated state from {equil}.chk", flush=True)
else:
    t0 = time.time()
    print("# minimise:", sim.minimize(300), flush=True)
    n = int(round(a.equil / a.dt))
    sim.run(n, report=max(n // 10, 1), prefix=equil)
    sim.save(equil)
    print(f"# equilibration {a.equil:g} ps: {time.time() - t0:.0f} s", flush=True)
nsteps = int(round(a.ns * 1000.0 / a.dt))
traj, report = int(round(a.traj / a.dt)), int(round(a.report / a.dt))
if a.mode == "bench":
    start = sim.state
    n = a.bench_steps

    def timed(label, R, advance, states):
        advance(500)  # compile
        t0 = time.time()
        advance(n)
        el = time.time() - t0
        cg = np.mean([float(st.cg_total) / int(st.step) for st in states()])
        print(
            f"{label} R={R:2d}: {el / n * 1e3:.3f} ms/step, {el / n / R * 1e3:.3f} ms per replica-step, "
            f"{n * a.dt / 1000 / el * 86400:.1f} ns/day per replica, {R * n * a.dt / 1000 / el * 86400:.1f} "
            f"aggregate; CG {cg:.2f}",
            flush=True,
        )

    for R in a.bench:
        sim.state = start
        if R == 1:  # plain MD of the same system
            timed("plain MD  ", 1, sim._advance, lambda: [sim.state])
            continue
        for batched in (True, False) if R == max(a.bench) else (True,):
            rep = ReplicaExchange(
                sim, geometric_ladder(a.tmin, a.tmax, R), batched=batched, seed=a.seed, log=None
            ).replicas
            timed("batched   " if batched else "sequential", R, rep.advance, lambda: [rep.state(k) for k in range(R)])
    raise SystemExit(0)
if a.mode == "plain":
    sim.run(nsteps, report=report, traj=traj, restart=report * 10, prefix=a.out + "_plain")
else:
    T = geometric_ladder(a.tmin, a.tmax, a.replicas)
    rex = ReplicaExchange(sim, T, exchange_every=a.exchange, batched=not a.sequential, seed=a.seed)
    if a.resume:
        rex.load(a.out + ".remd.chk")
    left = nsteps - rex.step
    s = rex.run(left, report=report, traj=traj, restart=report * 10, prefix=a.out, append=a.resume)
    print(
        json.dumps(
            {
                k: s[k]
                for k in ("neighbour_acceptance", "round_trips_total", "ns_per_day_per_replica", "ns_per_day_aggregate")
            }
        ),
        flush=True,
    )
