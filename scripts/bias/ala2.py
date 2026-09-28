"""Validation 2 of pgm_jax.bias: alanine dipeptide (ACE-ALA-NME) phi/psi free energy with pGM
electrostatics (placeholder residue parameters, as scripts/protein/remd_peptide.py) and ff19SB-form
bonded terms + CMAP, flexible engine, X-H constraints, Langevin 300 K, 2 fs.

    python scripts/protein/build_amber.py --sequence "ACE ALA NME" runs/ala2/ala2 --buffer 9
    python scripts/bias/ala2.py metad    --out runs/ala2/vac_metad_s1 --ns 20 --seed 1     # 2D WT-metaD (grid)
    python scripts/bias/ala2.py opes     --out runs/ala2/vac_opes_s1 --ns 20 --seed 1      # 2D OPES
    python scripts/bias/ala2.py umbrella --out runs/ala2/vac_us_w07 --window 7 --ns 2      # phi umbrella window
    python scripts/bias/ala2.py plain    --out runs/ala2/vac_plain --ns 20                 # unbiased
    python scripts/bias/ala2.py analyze  --out runs/ala2/vac                               # FES, WHAM, comparison
    python scripts/bias/ala2.py bench    --out runs/ala2/bench --solvated                  # bias overhead

--solvated keeps the pGM water box of the tleap system (default: the peptide alone in a 3.2 nm box,
"vacuum" up to its periodic images beyond the 1.2 nm cutoff).  Umbrella windows: harmonic in phi
(kappa --kappa kJ/mol/rad^2, V = kappa/2 dphi^2), --nwin centres over the circle.  Outputs: the
drivers' prefix.colvar / .hills / .log / .chk; analyze writes OUT_fes.json."""
import argparse
import glob
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pgm_jax import System  # noqa: E402
from pgm_jax.bias import OPES, BiasSet, Harmonic, MetaD, StaticBias, cv  # noqa: E402
from pgm_jax.bias import analysis as A  # noqa: E402
from pgm_jax.bias.core import KB  # noqa: E402
from pgm_jax.bias.io import read_table  # noqa: E402
from pgm_jax.md.flexible import FlexibleSimulation  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.protein import amber_template, load_amber  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("metad", "opes", "umbrella", "plain", "analyze", "bench"))
ap.add_argument("--prmtop", default="runs/ala2/ala2.prmtop")
ap.add_argument("--inpcrd", default="runs/ala2/ala2.inpcrd")
ap.add_argument("--out", required=True)
ap.add_argument("--solvated", action="store_true")
ap.add_argument("--box", type=float, default=3.2, help="vacuum box edge (nm)")
ap.add_argument("--ns", type=float, default=10.0)
ap.add_argument("--dt", type=float, default=0.002)
ap.add_argument("--T", type=float, default=300.0)
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--precision", default="mixed")
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--colvar", type=int, default=100)
ap.add_argument("--pace", type=int, default=250)
ap.add_argument("--height", type=float, default=1.2)
ap.add_argument("--sigma", type=float, default=0.35)
ap.add_argument("--biasfactor", type=float, default=6.0)
ap.add_argument("--barrier", type=float, default=50.0)
ap.add_argument("--window", type=int, default=0)
ap.add_argument("--nwin", type=int, default=24)
ap.add_argument("--first", type=int, default=0, help="umbrella: first window of this run")
ap.add_argument("--count", type=int, default=None, help="umbrella: windows in this run (default all)")
ap.add_argument("--kappa", type=float, default=150.0)
ap.add_argument("--equil", type=float, default=20.0, help="ps before a window's production")
ap.add_argument("--report", type=float, default=10.0, help="ps between log lines")
ap.add_argument("--chunks", type=int, default=1, help="restart segments (continue with --resume)")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--walkers", type=int, default=1, help="walkers in one vmapped program (bias/walkers.py)")
ap.add_argument("--shared", action="store_true", help="walkers share one bias (multiple-walker metaD / OPES)")
ap.add_argument("--skip", type=float, default=0.2, help="fraction discarded before reweighting")
ap.add_argument("--bench-steps", type=int, default=4000)
ap.add_argument("--labels", nargs="+", default=["none", "metad_grid", "metad_hills", "opes"])
ap.add_argument("--pme-grid", type=int, default=None, help="PME grid points per edge (vacuum default 20)")
ap.add_argument("--beta", type=float, default=None, help="Ewald coefficient (nm^-1; vacuum default 2.5)")
a = ap.parse_args()

kT = KB * a.T


def build(bias):
    asys = load_amber(a.prmtop, a.inpcrd, electrostatics="placeholder")
    kp = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"][0]
    prot = asys.molecules[kp]
    tpl = amber_template(prot, a.prmtop)
    q = np.asarray(prot.spec.top.cmaps)[0]
    sl = asys.system().atom_slice(kp)
    q = q + sl.start
    phi, psi = cv.Dihedral(*q[:4], name="phi"), cv.Dihedral(*q[1:], name="psi")
    if a.solvated:
        s = MDSettings(precision=a.precision, dipole_tol=a.tol, cutoff=1.0, skin=0.1)
    else:           # one molecule: a small Ewald coefficient makes a coarse PME grid accurate
        g = a.pme_grid or 20
        s = MDSettings(precision=a.precision, dipole_tol=a.tol, cutoff=1.2, skin=0.1, ewald_beta=a.beta or 2.5,
                       pme_grid=(g, g, g), lj_lrc=False)
    if a.solvated:
        sys_, tpls, pos, H = asys.system(), asys.templates({kp: tpl}), asys.system_positions(), asys.box
    else:
        x = np.asarray(asys.system_positions())[sl]
        pos = x - x.mean(0) + 0.5 * a.box
        sys_, tpls, H = System([asys.molecules[kp].spec.pgm]), [tpl], np.eye(3) * a.box
        phi, psi = cv.Dihedral(*(q[:4] - sl.start), name="phi"), cv.Dihedral(*(q[1:] - sl.start), name="psi")
    b = bias(phi, psi) if bias is not None else None
    sim = FlexibleSimulation(sys_, tpls, pos, H, s, dt=a.dt, temperature=a.T, gamma=a.gamma, thermostat="langevin",
                             constraints="h-bonds", seed=a.seed, bias=b, log=sys.stdout)
    return sim, phi, psi


def run(sim, prefix, ns):
    from pgm_jax.bias.walkers import Walkers
    n = int(round(ns * 1000 / a.dt))
    rep = int(round(a.report / a.dt))
    W = a.walkers
    if W > 1:
        sim.minimize(200, seed=a.seed)
        wk = Walkers(sim, W, shared=a.shared, seed=a.seed)
        if a.resume and os.path.exists(prefix + ".walkers.chk"):
            wk.load(prefix + ".walkers.chk")
            n -= int(np.asarray(wk.S.step)[0])
            print(f"# resumed at step {int(np.asarray(wk.S.step)[0])}; {n} steps to go")
        t0 = time.time()
        wk.run(n, report=rep, restart=rep * 50, prefix=prefix, append=a.resume)
        print(f"# {W} walkers x {n} steps in {time.time() - t0:.0f} s "
              f"({W * n * a.dt / 1000 / max(time.time() - t0, 1e-9) * 86400:.1f} ns/day aggregate)")
        return wk
    if a.resume and os.path.exists(prefix + ".chk"):
        sim.load(prefix + ".chk")
        n -= int(sim.state.step)
        print(f"# resumed at step {int(sim.state.step)}; {n} steps to go")
    else:
        sim.minimize(200, seed=a.seed)
    t0 = time.time()
    sim.run(n, report=rep, restart=rep * 50, prefix=prefix, append=a.resume)
    print(f"# {n} steps in {time.time() - t0:.0f} s ({n * a.dt / 1000 / max(time.time() - t0, 1e-9) * 86400:.1f} ns/day)")


def record_only(phi, psi):
    return BiasSet([StaticBias([phi, psi], lambda s: 0.0 * s[0], name="none")], colvar=a.colvar)


if a.mode == "metad":
    sim, _, _ = build(lambda phi, psi: BiasSet([MetaD([phi, psi], sigma=a.sigma, height=a.height, pace=a.pace,
                                                     biasfactor=a.biasfactor, grid=(-np.pi, np.pi, 128))], colvar=a.colvar))
    run(sim, a.out, a.ns)
elif a.mode == "opes":
    sim, _, _ = build(lambda phi, psi: BiasSet([OPES([phi, psi], sigma=a.sigma, pace=a.pace, barrier=a.barrier)],
                                               colvar=a.colvar))
    run(sim, a.out, a.ns)
elif a.mode == "plain":
    sim, _, _ = build(record_only)
    run(sim, a.out, a.ns)
elif a.mode == "umbrella":
    # all windows as walkers of one program; each first steered from the start to its centre
    from pgm_jax.bias.walkers import Walkers
    cen_all = -np.pi + (np.arange(a.nwin) + 0.5) * 2 * np.pi / a.nwin
    cen = cen_all[a.first:a.first + (a.count or a.nwin)]
    K = len(cen)
    holder = {}

    def mk(phi, psi):
        h = Harmonic([phi], at=0.0, kappa=a.kappa)
        holder["h"] = h
        return BiasSet([h, StaticBias([psi], lambda s: 0.0 * s[0], name="none")], colvar=a.colvar)

    sim, phi, psi = build(mk)
    h = holder["h"]
    rep = int(round(a.report / a.dt))
    if a.resume and os.path.exists(a.out + ".walkers.chk"):
        wk = Walkers(sim, K, bias_states=[sim.state.bias] * K, seed=a.seed)
        wk.load(a.out + ".walkers.chk")
    else:
        sim.minimize(200, seed=a.seed)
        phi0 = float(sim.cv_values()[0][0])
        d = np.mod(cen - phi0 + np.pi, 2 * np.pi) - np.pi
        states = [sim.state.bias._replace(parts=(h.state(at=phi0), ())) for _ in range(K)]
        wk = Walkers(sim, K, bias_states=states, seed=a.seed)
        nst = 40
        per = max(1, int(round(a.equil / a.dt / nst)))
        for j in range(1, nst + 1):
            at = jnp.asarray(np.mod(phi0 + d * j / nst + np.pi, 2 * np.pi) - np.pi)[:, None]
            wk.S = wk.S.set(bias=wk.S.bias._replace(parts=(wk.S.bias.parts[0]._replace(at=at), wk.S.bias.parts[1])))
            wk.S = wk._forces(wk.S).set(induction=wk.S.induction)
            wk.advance(per)
            if j % 10 == 0:
                print(f"# steering {j}/{nst}: phi - centre (deg) " + " ".join(
                    f"{np.degrees(x):.0f}" for x in np.mod(np.asarray(wk.rows(0))[-1:, 1] - at[0, 0] + np.pi, 2 * np.pi) - np.pi), flush=True)
        for w in range(K):
            wk.rows(w)
        wk.S = wk.S.set(step=jnp.zeros_like(wk.S.step))
    n = int(round(a.ns * 1000 / a.dt)) - int(np.asarray(wk.S.step)[0])
    t0 = time.time()
    wk.run(n, report=rep, restart=rep * 50, prefix=a.out, append=a.resume)
    json.dump({"centers": cen.tolist(), "kappa": a.kappa, "nwin": a.nwin, "first": a.first}, open(a.out + ".json", "w"))
    print(f"# {K} windows x {n} steps in {time.time() - t0:.0f} s")
elif a.mode == "bench":
    # the same system and settings with and without a 2D metaD bias (grid; hills every 250 steps)
    out = {}
    for label in a.labels:
        if label == "none":
            sim, _, _ = build(None)
        elif label == "metad_grid":
            sim, _, _ = build(lambda phi, psi: BiasSet([MetaD([phi, psi], sigma=a.sigma, height=a.height, pace=a.pace,
                                                             biasfactor=a.biasfactor, grid=(-np.pi, np.pi, 128))], colvar=a.colvar))
        elif label == "metad_hills":
            sim, _, _ = build(lambda phi, psi: BiasSet([MetaD([phi, psi], sigma=a.sigma, height=a.height, pace=a.pace,
                                                             biasfactor=a.biasfactor, capacity=4096)], colvar=a.colvar))
        else:
            sim, _, _ = build(lambda phi, psi: BiasSet([OPES([phi, psi], sigma=a.sigma, pace=a.pace, barrier=a.barrier)],
                                                       colvar=a.colvar))
        sim._advance(500)
        ts = []
        for rep in range(3):
            t0 = time.time()
            sim._advance(a.bench_steps)
            jax.block_until_ready(sim.state.epot)
            ts.append((time.time() - t0) / a.bench_steps * 1e3)
        o = sim.observables()
        out[label] = {"ms_per_step": min(ts), "all": ts, "cg_mean": o["cg_mean"], "epot": o["epot"]}
        print(label, out[label], flush=True)
    base = out[a.labels[0]]["ms_per_step"]
    for k, v in out.items():
        v["overhead_pct"] = 100.0 * (v["ms_per_step"] / base - 1.0)
    out["device"] = str(jax.devices()[0])
    out["atoms"] = int(sim.sys.n)
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out + ".json", "w"), indent=1)
else:
    raise SystemExit("analyze: see scripts/bias/ala2_analyze.py")
