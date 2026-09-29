"""Validation 2 of pgm_jax.bias: phi/psi free energy of the alanine dipeptide (ACE-ALA-NME) with pGM.

pGM electrostatics (placeholder residue parameters, as scripts/protein/remd_peptide.py) and
ff19SB-form bonded terms + CMAP, flexible engine, X-H constraints, Langevin 300 K, 2 fs.  Modes:
metad (2D well-tempered metadynamics on a grid), opes (2D OPES), umbrella (phi windows as walkers of
one program, each first steered to its centre), plain (unbiased, CVs recorded), remd (temperature
replica exchange, the 300 K replica as an independent reference), bench (cost of the bias); the
analysis is scripts/bias/ala2_analyze.py (docs/enhanced_sampling.md).

--solvated keeps the pGM water box of the tleap system (default: the peptide alone in a 3.2 nm box,
"vacuum" up to its periodic images beyond the 1.2 nm cutoff).  Umbrella windows: harmonic in phi
(--kappa-kJ-rad2, V = kappa/2 dphi^2), --nwin centres over the circle.

Usage:

    python scripts/protein/build_amber.py --sequence "ACE ALA NME" runs/ala2/ala2 --buffer-A 9
    python scripts/bias/ala2.py metad    --out runs/ala2/vac_metad_s1 --time-ns 20 --seed 1   # 2D WT-metaD (grid)
    python scripts/bias/ala2.py opes     --out runs/ala2/vac_opes_s1 --time-ns 20 --seed 1    # 2D OPES
    python scripts/bias/ala2.py umbrella --out runs/ala2/vac_us --time-ns 2                   # phi umbrella windows
    python scripts/bias/ala2.py plain    --out runs/ala2/vac_plain --time-ns 20               # unbiased
    python scripts/bias/ala2.py bench    --out runs/ala2/bench --solvated                     # bias overhead
    python scripts/bias/ala2.py --help

Inputs: the tleap system (--prmtop, --inpcrd; scripts/protein/build_amber.py).
Outputs: the drivers' <out>.colvar / .hills / .log / .chk (walkers: <out>_wNN.*); umbrella and
remd: <out>.json (centres / CV atoms and temperatures); bench: <out>.json.
Units: --time-ns ns, --dt-fs fs, --temperature-K and --tmax-K K, --friction-per-ps 1/ps,
--equil-ps and --report-ps ps, --box-nm nm, --ewald-beta-per-nm 1/nm, --height-kJ and --barrier-kJ
kJ/mol, --sigma-rad rad, --kappa-kJ-rad2 kJ/mol/rad^2; --pace and --colvar-every in steps.
Runtime: GPU; sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax import System
from pgm_jax.bias import OPES, BiasSet, Harmonic, MetaD, StaticBias, cv
from pgm_jax.bias.walkers import Walkers
from pgm_jax.cli.args import (
    add_dipole_tol_arg,
    add_dt_arg,
    add_precision_arg,
    add_seed_arg,
    add_temperature_arg,
    setup_logging,
)
from pgm_jax.md.flexible import FlexibleSimulation
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.remd import ReplicaExchange, geometric_ladder
from pgm_jax.md.thermostats import Langevin
from pgm_jax.protein import amber_template, load_amber

jax.config.update("jax_enable_x64", True)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("metad", "opes", "umbrella", "plain", "remd", "bench"), help="what to run")
    ap.add_argument("--replicas", type=int, default=8, help="remd: replicas (geometric ladder)")
    ap.add_argument("--tmax-K", type=float, default=700.0, help="remd: highest temperature [K]")
    ap.add_argument("--prmtop", default="runs/ala2/ala2.prmtop", help="tleap prmtop")
    ap.add_argument("--inpcrd", default="runs/ala2/ala2.inpcrd", help="tleap coordinates")
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    ap.add_argument("--solvated", action="store_true", help="keep the water box of the tleap system")
    ap.add_argument("--box-nm", type=float, default=3.2, help="vacuum box edge [nm]")
    ap.add_argument("--time-ns", type=float, default=10.0, help="run length [ns] (per walker / window / replica)")
    add_dt_arg(ap, 2.0)
    add_temperature_arg(ap, 300.0)
    ap.add_argument("--friction-per-ps", type=float, default=2.0, help="Langevin friction [1/ps]")
    add_seed_arg(ap, 1)
    add_precision_arg(ap)
    add_dipole_tol_arg(ap)
    ap.add_argument("--colvar-every", type=int, default=100, help="steps between COLVAR rows")
    ap.add_argument("--pace", type=int, default=250, help="steps between bias depositions")
    ap.add_argument("--height-kJ", type=float, default=1.2, help="metaD: hill height [kJ/mol]")
    ap.add_argument("--sigma-rad", type=float, default=0.35, help="hill / kernel width [rad]")
    ap.add_argument("--biasfactor", type=float, default=6.0, help="metaD: bias factor")
    ap.add_argument("--barrier-kJ", type=float, default=50.0, help="OPES: barrier [kJ/mol]")
    ap.add_argument("--nwin", type=int, default=24, help="umbrella: windows over the circle")
    ap.add_argument("--first", type=int, default=0, help="umbrella: first window of this run")
    ap.add_argument("--count", type=int, default=None, help="umbrella: windows in this run (default all)")
    ap.add_argument("--kappa-kJ-rad2", type=float, default=150.0, help="umbrella: force constant [kJ/mol/rad^2]")
    ap.add_argument("--equil-ps", type=float, default=20.0, help="umbrella: steering; remd: equilibration [ps]")
    ap.add_argument("--report-ps", type=float, default=10.0, help="time between log lines [ps]")
    ap.add_argument("--resume", action="store_true", help="continue from the run's checkpoint")
    ap.add_argument("--walkers", type=int, default=1, help="walkers in one vmapped program (bias/walkers.py)")
    ap.add_argument("--shared", action="store_true", help="walkers share one bias (multiple-walker metaD / OPES)")
    ap.add_argument("--bench-steps", type=int, default=4000, help="bench: timed steps")
    ap.add_argument(
        "--labels", nargs="+", default=["none", "metad_grid", "metad_hills", "opes"], help="bench: variants"
    )
    ap.add_argument("--pme-grid", type=int, default=None, help="PME grid points per edge (vacuum default 20)")
    ap.add_argument(
        "--ewald-beta-per-nm", type=float, default=None, help="Ewald coefficient [1/nm] (vacuum default 2.5)"
    )
    return ap


def build(a: argparse.Namespace, bias: Callable | None) -> tuple[FlexibleSimulation, object, object]:
    """Return the simulation of the dipeptide (vacuum box or solvated) with bias(phi, psi) and the phi, psi CVs."""
    asys = load_amber(a.prmtop, a.inpcrd, electrostatics="placeholder")
    kp = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"][0]
    prot = asys.molecules[kp]
    tpl = amber_template(prot, a.prmtop)
    q = np.asarray(prot.spec.top.cmaps)[0]
    sl = asys.system().atom_slice(kp)
    q = q + sl.start
    phi, psi = cv.Dihedral(*q[:4], name="phi"), cv.Dihedral(*q[1:], name="psi")
    if a.solvated:
        s = MDSettings().replace(precision=a.precision, dipole_tol=a.dipole_tol, cutoff=1.0, skin=0.1)
    else:  # one molecule: a small Ewald coefficient makes a coarse PME grid accurate
        g = a.pme_grid or 20
        s = MDSettings().replace(
            precision=a.precision,
            dipole_tol=a.dipole_tol,
            cutoff=1.2,
            skin=0.1,
            ewald_beta=a.ewald_beta_per_nm or 2.5,
            pme_grid=(g, g, g),
            lj_lrc=False,
        )
    if a.solvated:
        sys_, tpls, pos, H = asys.system(), asys.templates({kp: tpl}), asys.system_positions(), asys.box
    else:
        x = np.asarray(asys.system_positions())[sl]
        pos = x - x.mean(0) + 0.5 * a.box_nm
        sys_, tpls, H = System([asys.molecules[kp].spec.pgm]), [tpl], np.eye(3) * a.box_nm
        phi, psi = cv.Dihedral(*(q[:4] - sl.start), name="phi"), cv.Dihedral(*(q[1:] - sl.start), name="psi")
    b = bias(phi, psi) if bias is not None else None
    sim = FlexibleSimulation(
        sys_,
        tpls,
        pos,
        H,
        s,
        dt=a.dt_fs / 1000,
        temperature=a.temperature_K,
        thermostat=Langevin(a.friction_per_ps),
        constraints="h-bonds",
        seed=a.seed,
        bias=b,
        log=sys.stdout,
    )
    return sim, phi, psi


def run(a: argparse.Namespace, sim: FlexibleSimulation, prefix: str, ns: float) -> Walkers | None:
    """Minimise and run ns [ns] (or continue with --resume), as one simulation or as --walkers walkers."""
    dt = a.dt_fs / 1000
    n = int(round(ns * 1000 / dt))
    rep = int(round(a.report_ps / dt))
    W = a.walkers
    if W > 1:
        sim.minimize(200, seed=a.seed)
        wk = Walkers(sim, W, shared=a.shared, seed=a.seed, log=sys.stdout)
        if a.resume and os.path.exists(prefix + ".walkers.chk"):
            wk.load_checkpoint(prefix + ".walkers.chk")
            n -= int(np.asarray(wk.S.step)[0])
            print(f"# resumed at step {int(np.asarray(wk.S.step)[0])}; {n} steps to go")
        t0 = time.time()
        wk.run(n, report_every=rep, checkpoint_every=rep * 50, prefix=prefix, append=a.resume)
        print(
            f"# {W} walkers x {n} steps in {time.time() - t0:.0f} s "
            f"({W * n * dt / 1000 / max(time.time() - t0, 1e-9) * 86400:.1f} ns/day aggregate)"
        )
        return wk
    if a.resume and os.path.exists(prefix + ".chk"):
        sim.load_checkpoint(prefix + ".chk")
        n -= int(sim.state.step)
        print(f"# resumed at step {int(sim.state.step)}; {n} steps to go")
    else:
        sim.minimize(200, seed=a.seed)
    t0 = time.time()
    sim.run(n, report_every=rep, checkpoint_every=rep * 50, prefix=prefix, append=a.resume)
    print(f"# {n} steps in {time.time() - t0:.0f} s ({n * dt / 1000 / max(time.time() - t0, 1e-9) * 86400:.1f} ns/day)")


def record_only(a: argparse.Namespace, phi: object, psi: object) -> BiasSet:
    """Return a zero bias that records phi and psi every --colvar-every steps."""
    return BiasSet([StaticBias([phi, psi], lambda s: 0.0 * s[0], name="none")], colvar=a.colvar_every_every)


def cmd_metad(a: argparse.Namespace) -> None:
    """Run 2D well-tempered metadynamics on (phi, psi) with a grid (the `metad` mode)."""
    sim, _, _ = build(
        a,
        lambda phi, psi: BiasSet(
            [
                MetaD(
                    [phi, psi],
                    sigma=a.sigma_rad,
                    height=a.height_kJ,
                    pace=a.pace,
                    biasfactor=a.biasfactor,
                    grid=(-np.pi, np.pi, 128),
                )
            ],
            colvar=a.colvar_every,
        ),
    )
    run(a, sim, a.out, a.time_ns)


def cmd_opes(a: argparse.Namespace) -> None:
    """Run 2D OPES on (phi, psi) (the `opes` mode)."""
    sim, _, _ = build(
        a,
        lambda phi, psi: BiasSet(
            [OPES([phi, psi], sigma=a.sigma_rad, pace=a.pace, barrier=a.barrier_kJ)], colvar=a.colvar_every
        ),
    )
    run(a, sim, a.out, a.time_ns)


def cmd_plain(a: argparse.Namespace) -> None:
    """Run unbiased MD recording phi and psi (the `plain` mode)."""
    sim, _, _ = build(a, lambda phi, psi: record_only(a, phi, psi))
    run(a, sim, a.out, a.time_ns)


def cmd_umbrella(a: argparse.Namespace) -> None:
    """Run the phi umbrella windows as walkers of one program, each steered to its centre first (`umbrella`)."""
    # all windows as walkers of one program; each first steered from the start to its centre
    cen_all = -np.pi + (np.arange(a.nwin) + 0.5) * 2 * np.pi / a.nwin
    cen = cen_all[a.first : a.first + (a.count or a.nwin)]
    K = len(cen)
    holder = {}

    def mk(phi, psi):
        h = Harmonic([phi], at=0.0, kappa=a.kappa_kJ_rad2)
        holder["h"] = h
        return BiasSet([h, StaticBias([psi], lambda s: 0.0 * s[0], name="none")], colvar=a.colvar_every)

    sim, phi, psi = build(a, mk)
    h = holder["h"]
    rep = int(round(a.report_ps / (a.dt_fs / 1000)))
    if a.resume and os.path.exists(a.out + ".walkers.chk"):
        wk = Walkers(sim, K, bias_states=[sim.state.bias] * K, seed=a.seed, log=sys.stdout)
        wk.load_checkpoint(a.out + ".walkers.chk")
    else:
        sim.minimize(200, seed=a.seed)
        phi0 = float(sim.cv_values()[0][0])
        d = np.mod(cen - phi0 + np.pi, 2 * np.pi) - np.pi
        states = [sim.state.bias._replace(parts=(h.state(at=phi0), ())) for _ in range(K)]
        wk = Walkers(sim, K, bias_states=states, seed=a.seed, log=sys.stdout)
        nst = 40
        per = max(1, int(round(a.equil_ps / (a.dt_fs / 1000) / nst)))
        for j in range(1, nst + 1):
            at = jnp.asarray(np.mod(phi0 + d * j / nst + np.pi, 2 * np.pi) - np.pi)[:, None]
            wk.S = wk.S.set(bias=wk.S.bias._replace(parts=(wk.S.bias.parts[0]._replace(at=at), wk.S.bias.parts[1])))
            wk.S = wk._forces(wk.S).set(induction=wk.S.induction)
            wk.advance(per)
            if j % 10 == 0:
                print(
                    f"# steering {j}/{nst}: phi - centre (deg) "
                    + " ".join(
                        f"{np.degrees(x):.0f}"
                        for x in np.mod(np.asarray(wk.rows(0))[-1:, 1] - at[0, 0] + np.pi, 2 * np.pi) - np.pi
                    ),
                    flush=True,
                )
        for w in range(K):
            wk.rows(w)
        wk.S = wk.S.set(step=jnp.zeros_like(wk.S.step))
    n = int(round(a.time_ns * 1000 / (a.dt_fs / 1000))) - int(np.asarray(wk.S.step)[0])
    t0 = time.time()
    wk.run(n, report_every=rep, checkpoint_every=rep * 50, prefix=a.out, append=a.resume)
    with open(a.out + ".json", "w") as fh:
        json.dump({"centers": cen.tolist(), "kappa": a.kappa_kJ_rad2, "nwin": a.nwin, "first": a.first}, fh)
    print(f"# {K} windows x {n} steps in {time.time() - t0:.0f} s")


def cmd_remd(a: argparse.Namespace) -> None:
    """Run temperature replica exchange; the 300 K replica is an independent reference (the `remd` mode)."""
    # temperature replica exchange (md/remd.py), the replicas batched: the 300 K slot is an
    # independent reference for the phi/psi distribution
    sim, phi, psi = build(a, None)
    rex_T = geometric_ladder(a.temperature_K, a.tmax_K, a.replicas)
    n = int(round(a.time_ns * 1000 / (a.dt_fs / 1000)))
    rep = int(round(a.report_ps / (a.dt_fs / 1000)))
    if a.resume and os.path.exists(a.out + ".remd.chk"):
        rex = ReplicaExchange(sim, rex_T, exchange_every=250, seed=a.seed, log=sys.stdout)
        rex.load_checkpoint(a.out + ".remd.chk")
        n -= int(rex.step)
        print(f"# resumed at step {int(rex.step)}; {n} steps to go", flush=True)
    else:
        sim.minimize(200, seed=a.seed)
        sim.run(
            int(round(a.equil_ps / (a.dt_fs / 1000))),
            report_every=int(round(a.equil_ps / (a.dt_fs / 1000))),
            prefix=a.out + "_eq",
        )
        rex = ReplicaExchange(sim, rex_T, exchange_every=250, seed=a.seed, log=sys.stdout)
    json.dump(
        {"phi": [int(i) for i in phi.idx], "psi": [int(i) for i in psi.idx], "T": list(map(float, rex_T))},
        open(a.out + ".json", "w"),
    )
    t0 = time.time()
    rex.run(n, report_every=rep, traj_every=100, checkpoint_every=rep * 50, prefix=a.out, append=a.resume)
    print(f"# REMD {a.replicas} replicas x {n} steps in {time.time() - t0:.0f} s", flush=True)


def cmd_bench(a: argparse.Namespace) -> None:
    """Time the system with and without the bias variants and print the overhead (the `bench` mode)."""
    # the same system and settings with and without a 2D metaD bias (grid; hills every 250 steps)
    out = {}
    for label in a.labels:
        if label == "none":
            sim, _, _ = build(a, None)
        elif label == "metad_grid":
            sim, _, _ = build(
                a,
                lambda phi, psi: BiasSet(
                    [
                        MetaD(
                            [phi, psi],
                            sigma=a.sigma_rad,
                            height=a.height_kJ,
                            pace=a.pace,
                            biasfactor=a.biasfactor,
                            grid=(-np.pi, np.pi, 128),
                        )
                    ],
                    colvar=a.colvar_every,
                ),
            )
        elif label == "metad_hills":
            sim, _, _ = build(
                a,
                lambda phi, psi: BiasSet(
                    [
                        MetaD(
                            [phi, psi],
                            sigma=a.sigma_rad,
                            height=a.height_kJ,
                            pace=a.pace,
                            biasfactor=a.biasfactor,
                            capacity=4096,
                        )
                    ],
                    colvar=a.colvar_every,
                ),
            )
        else:
            sim, _, _ = build(
                a,
                lambda phi, psi: BiasSet(
                    [OPES([phi, psi], sigma=a.sigma_rad, pace=a.pace, barrier=a.barrier_kJ)], colvar=a.colvar_every
                ),
            )
        sim.advance(500)
        ts = []
        for _rep in range(3):
            t0 = time.time()
            sim.advance(a.bench_steps)
            jax.block_until_ready(sim.state.epot)
            ts.append((time.time() - t0) / a.bench_steps * 1e3)
        o = sim.observables()
        out[label] = {"ms_per_step": min(ts), "all": ts, "cg_mean": o["cg_mean"], "epot": o["epot"]}
        print(label, out[label], flush=True)
    base = out[a.labels[0]]["ms_per_step"]
    for _k, v in out.items():
        v["overhead_pct"] = 100.0 * (v["ms_per_step"] / base - 1.0)
    out["device"] = str(jax.devices()[0])
    out["atoms"] = int(sim.sys.n)
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out + ".json", "w"), indent=1)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the mode (see the module docstring)."""
    a = build_parser().parse_args(argv)
    setup_logging()
    {
        "metad": cmd_metad,
        "opes": cmd_opes,
        "plain": cmd_plain,
        "umbrella": cmd_umbrella,
        "remd": cmd_remd,
        "bench": cmd_bench,
    }[a.mode](a)


if __name__ == "__main__":
    main()
