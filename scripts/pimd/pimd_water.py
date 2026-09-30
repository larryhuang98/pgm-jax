"""Quantum (path-integral) flexible pGM water: template, PIMD / TRPMD runs, RDFs, speed (`pgm-jax pimd`).

The water is the pGM water of the 512-water pGM3P-25 box of the README (PGM_GVDW_DATA/topology/
rayl_512_v2.prmtop: charges, covalent dipoles, radii, polarizabilities, Lennard-Jones on O), made
flexible with bonded terms fitted so that its gas-phase monomer potential (bonded + all-pair
intramolecular pGM) is the q-TIP4P/F intramolecular potential (models.water.flexible_water).  The
runs are NVT at the density of the box's restart (from the rigid model's NPT), or NPT with
--barostat mc.  Subcommands: template (fit and save the flexible template), run (classical
equilibration, then PIMD / TRPMD with PILE; observables, diffusion, radial distribution
functions), bench (ms/step and ns/day for bead numbers and contraction settings), batch (several
runs in one process).  See docs/pimd.md.

Usage:

    python scripts/pimd/pimd_water.py template             # fit the flexible pGM water (q-TIP4P/F monomer PES)
    python scripts/pimd/pimd_water.py run --beads 32 --time-ps 20 --out runs/pimd/w32   # PIMD, 512 waters, 298 K
    python scripts/pimd/pimd_water.py run --beads 1 --time-ps 20 --out runs/pimd/w1     # classical flexible water
    python scripts/pimd/pimd_water.py run --beads 32 --contract 1 ...                   # contracted to the centroid
    python scripts/pimd/pimd_water.py run --beads 32 --mode trpmd --continue-from runs/pimd/w32.pimd.chk
        --time-ps 20                                                                    # dynamics (TRPMD)
    python scripts/pimd/pimd_water.py bench --beads 8 32 --contract 0 1                 # ns/day
    python scripts/pimd/pimd_water.py run --help

Inputs: PGM_GVDW_DATA (pgm_jax.paths): the prmtop and restart of the 512-water box; the flexible
template (--template, default data/validation/pimd/pgm_water_flex.flex, written by `template`).
Outputs: template: the .flex template and <template>.json (fit report).  run: <out>.log
(observables every --report-ps), <out>.json (means and block errors, speed, drift of the conserved
energy, diffusion coefficient), <out>.msd, with --rdf <out>.rdf (g_OO, g_OH, g_HH), with --save
<out>.pimd.chk and <out>.rst7; <out>_classical.* and <out>_equil.* of the equilibration stages.
bench: one JSON line per setting (and --out).
Units: --dt-fs fs, durations in ps (--time-ps, --equil-ps, --classical-ps, --report-ps),
--temperature-K K, --pressure-bar bar, --cutoff-nm and --bead-margin-nm nm, --tau-centroid-ps ps.
Runtime: GPU (512 waters, 32 beads: about 8 ms/step with bead chunks of 8, docs/pimd.md); the
script sets jax_enable_x64.  With the environment variable PIMD_WAIT_GPU set, the script waits for
a shared GPU to become free before it starts (exit status 75 when no GPU can be used).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from typing import TextIO

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.cli.args import (
    add_barostat_args,
    add_cutoff_arg,
    add_dipole_tol_arg,
    add_dt_arg,
    add_precision_arg,
    add_seed_arg,
    add_temperature_arg,
    setup_logging,
)
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.pimd import PILE, PIMDSimulation
from pgm_jax.md.thermostats import Bussi
from pgm_jax.models.water import flexible_water
from pgm_jax.param import read_prmtop_pgm
from pgm_jax.paths import pgm3p25_files, repo_path
from pgm_jax.system import System

jax.config.update("jax_enable_x64", True)

TOP, RST = pgm3p25_files()
TPL = repo_path("data", "validation", "pimd", "pgm_water_flex.flex")


def wait_for_gpu() -> None:
    """Wait until the GPU is free, then keep its context (for a GPU shared with retrying jobs).

    Retains the CUDA primary context of device 0 as soon as it can be created (polling every 50 ms);
    JAX then uses that context.  Exits with status 75 when CUDA cannot be initialized or JAX finds
    no device.  Called by main when the environment variable PIMD_WAIT_GPU is set.
    """
    import ctypes

    cu = ctypes.CDLL("libcuda.so.1")
    if cu.cuInit(0) != 0:
        sys.exit(75)
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    cu.cuDeviceGet(ctypes.byref(dev), 0)
    while cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) != 0:
        time.sleep(0.05)
    try:
        jax.devices()
    except RuntimeError:
        sys.exit(75)


def template(a: argparse.Namespace) -> None:
    """Fit the flexible pGM water template and save it (the `template` subcommand).

    Parameters
    ----------
    a : argparse.Namespace
        Options: out (template file; the fit report goes to <out>.json), samples (monomer
        geometries of the fit).
    """
    w = read_prmtop_pgm(TOP)[0]
    t0 = time.time()
    tpl, rep = flexible_water(w, n_samples=a.samples)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tpl.save(a.out)
    rep["time_s"] = time.time() - t0
    with open(a.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "params"}, indent=1))
    print("saved", a.out)


def build(a: argparse.Namespace, log: TextIO | None = sys.stdout) -> FlexibleSimulation:
    """Return the classical flexible simulation of the 512-water box (Bussi, tau 0.1 ps).

    Parameters
    ----------
    a : argparse.Namespace
        Options: template, cutoff_nm [nm], dipole_tol, precision, dt_fs [fs], temperature_K [K], seed.
    log : text stream, optional
        Stream for the log-table rows (None: no table).

    Returns
    -------
    FlexibleSimulation
        The box of PGM_GVDW_DATA's restart (positions and box converted from Angstrom to nm).
    """
    tpl = FlexibleTemplate.load(a.template)
    xyz, _, box = read_coordinates(RST)
    H = box_from_cell(*box) * 0.1
    pos = xyz * 0.1
    n = len(pos) // 3
    s = MDSettings().replace(cutoff=a.cutoff_nm, dipole_tol=a.dipole_tol, precision=a.precision)
    sim = FlexibleSimulation(
        System([tpl.pgm] * n),
        [tpl] * n,
        pos,
        H,
        s,
        dt=a.dt_fs * 1e-3,
        thermostat=Bussi(0.1),
        temperature=a.temperature_K,
        seed=a.seed,
        log=log,
    )
    return sim


def rdf_fn(sim: FlexibleSimulation, edges: np.ndarray) -> tuple[Callable, int, int]:
    """Return a jitted histogram function of the O-O, O-H and H-H distances of all beads.

    Parameters
    ----------
    sim : FlexibleSimulation
        The simulation (elements and molecule index of the atoms).
    edges : np.ndarray (B + 1,)
        Bin edges [nm].

    Returns
    -------
    hist : callable
        hist(Q, H) -> jax.Array (3, B): the O-O, intermolecular O-H and intermolecular H-H pair
        counts summed over the beads of Q (P, N, 3) [nm] in box H (3, 3) [nm] (minimum image in a
        reduced, lower-triangular box; jax.lax.map over the beads).
    n_O, n_H : int
        Numbers of O and H atoms.
    """
    el = np.array(sim.sys.elements)
    O, Hy = np.nonzero(el == "O")[0], np.nonzero(el == "H")[0]
    mol = np.asarray(sim.sys.mol)
    same_oh = jnp.asarray(mol[O][:, None] == mol[Hy][None, :])
    same_hh = jnp.asarray(mol[Hy][:, None] == mol[Hy][None, :])
    e = jnp.asarray(edges)

    def dist(x, y, H):
        """Minimum-image distances (len(x), len(y)) [nm] in the reduced box H."""
        d = x[:, None, :] - y[None, :, :]
        for c in (2, 1, 0):
            d = d - jnp.round(d[..., c] / H[c, c])[..., None] * H[c]
        return jnp.sqrt(jnp.sum(d * d, -1))

    def one(q, H):
        """Histograms (3, B) of one bead's configuration q (N, 3) [nm]."""
        rOO = dist(q[O], q[O], H)
        iu = jnp.triu_indices(len(O), 1)
        hOO = jnp.histogram(rOO[iu], bins=e)[0]
        rOH = dist(q[O], q[Hy], H)
        hOH = jnp.histogram(jnp.where(same_oh, -1.0, rOH), bins=e)[0]  # intermolecular O-H
        rHH = dist(q[Hy], q[Hy], H)
        hHH = jnp.histogram(jnp.where(same_hh, -1.0, rHH), bins=e)[0] // 2  # intermolecular H-H pairs
        return jnp.stack([hOO, hOH, hHH])

    @jax.jit
    def hist(Q, H):
        """Histograms (3, B) summed over the beads of Q (P, N, 3)."""
        return jnp.sum(jax.lax.map(lambda q: one(q, H), Q), 0)

    return hist, len(O), len(Hy)


def diffusion(C: np.ndarray, dt_ps: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the self-diffusion coefficient from unwrapped molecular centres.

    Parameters
    ----------
    C : np.ndarray (F, M, 3)
        Unwrapped molecular centres of F frames [nm].
    dt_ps : float
        Time between frames [ps].

    Returns
    -------
    D : float
        Self-diffusion coefficient [1e-5 cm^2/s].
    t : np.ndarray (F // 2,)
        Lag times [ps].
    msd : np.ndarray (F // 2,)
        Mean squared displacement over all time origins [nm^2].

    Notes
    -----
    Einstein relation MSD = 6 D t, D from the slope of a line through the lags between 20 % and
    50 % of the run; no finite-size correction.  1 nm^2/ps = 1e-2 cm^2/s = 1000 x 1e-5 cm^2/s.
    """
    F = len(C)
    lags = np.arange(1, F // 2 + 1)
    msd = np.array([np.mean(np.sum((C[k:] - C[:-k]) ** 2, -1)) for k in lags])
    t = lags * dt_ps
    sel = (lags >= max(1, int(0.2 * F))) & (lags <= F // 2)
    slope = np.polyfit(t[sel], msd[sel], 1)[0]
    return slope / 6.0 * 1000.0, t, msd


def run(a: argparse.Namespace) -> None:
    """Run PIMD / TRPMD and write the log, summary, MSD and RDFs (the `run` subcommand).

    Without --continue-from: minimize (200 steps), classical flexible equilibration
    (--classical-ps, Bussi), then --equil-ps with the ring polymer, then --time-ps of production
    with samples every --report-ps.  With --continue-from: the checkpoint's state, with the heat
    and CG counters and the step counter reset, and no classical equilibration.

    Parameters
    ----------
    a : argparse.Namespace
        Options of the `run` subcommand (see --help).
    """
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sim = build(a, log=sys.stdout)
    if a.continue_from is None:
        sim.minimize(200)
        if a.classical_ps > 0:  # classical flexible equilibration (Bussi)
            sim.run(int(round(a.classical_ps / (a.dt_fs * 1e-3))), report_every=1000, prefix=a.out + "_classical")
    pi = PIMDSimulation(
        sim,
        beads=a.beads,
        mode=a.mode,
        thermostat=PILE(a.thermostat, tau_centroid=a.tau_centroid_ps, lam=a.lam),
        propagator=a.propagator,
        contract=a.contract or None,
        bead_margin=a.bead_margin_nm,
        seed=a.seed,
        barostat=MonteCarloBarostat(a.pressure_bar, a.barostat_every) if a.barostat == "mc" else None,
        bead_chunk=a.bead_chunk if a.bead_chunk == "auto" else (int(a.bead_chunk) or None),
        log=sys.stdout,
    )
    if a.continue_from:
        pi.load_checkpoint(a.continue_from)
        pi.state = pi.state.set(
            heat=jnp.zeros(()), step=jnp.zeros((), jnp.int32), eng=pi.state.eng.set(cg_total=jnp.zeros(()))
        )
    nstep = int(round(a.time_ps / (a.dt_fs * 1e-3)))
    neq = int(round(a.equil_ps / (a.dt_fs * 1e-3)))
    rep = max(1, int(round(a.report_ps / (a.dt_fs * 1e-3))))
    if neq:
        pi.run(neq, report_every=rep, prefix=a.out + "_equil")
    edges = np.linspace(0.0, 0.8, 321)
    hist, nO, nH = rdf_fn(sim, edges)
    H = np.zeros((3, len(edges) - 1))
    nfr, vol = 0, []
    samples = []
    coms, last = [], None
    t0 = time.time()
    done = 0
    logf = open(a.out + ".log", "w")
    cols = None
    while done < nstep:
        m = min(rep, nstep - done)
        pi.advance(m)
        done += m
        o = pi.observables()
        if a.report_pressure:
            o["press_bar"] = pi.pressure()
        o["ns_per_day"] = done * a.dt_fs * 1e-6 / max(time.time() - t0, 1e-9) * 86400.0
        if cols is None:
            cols = list(o)
            logf.write("# " + " ".join(f"{c:>14s}" for c in cols) + "\n")
        logf.write("  " + " ".join(f"{o[c]:14.6f}" if isinstance(o[c], float) else f"{o[c]:14d}" for c in cols) + "\n")
        logf.flush()
        samples.append([o[c] for c in cols])
        com = np.asarray(sim.flex.centers(jnp.mean(pi.state.q, 0)))  # centroid molecular centres
        if coms:  # unwrap (minimum image step)
            Hb = np.asarray(pi.state.box)
            d = com - last
            d -= np.round(d @ np.linalg.inv(Hb)) @ Hb
            coms.append(coms[-1] + d)
        else:
            coms.append(com)
        last = com
        if a.rdf:
            H += np.asarray(hist(pi.state.q, pi.state.box))
            nfr += pi.P
            vol.append(float(np.abs(np.linalg.det(np.asarray(pi.state.box)))))
    logf.close()
    el = time.time() - t0
    X = np.array(samples, float)
    nb = 10
    mb = len(X) // nb * nb
    blocks = X[len(X) - mb :].reshape(nb, -1, X.shape[1]).mean(1)
    mean, err = X.mean(0), blocks.std(0, ddof=1) / np.sqrt(nb)
    summ = {
        "beads": a.beads,
        "contract": a.contract,
        "mode": a.mode,
        "thermostat": a.thermostat,
        "dt_fs": a.dt_fs,
        "ps": a.time_ps,
        "ns_per_day": nstep * a.dt_fs * 1e-6 / el * 86400.0,
        "ms_per_step": el / nstep * 1e3,
    }
    drift = np.polyfit(X[:, cols.index("time_ps")], X[:, cols.index("econs")], 1)[0]
    summ["econs_drift_kJmol_ps"] = float(drift)
    for c in cols:
        if c in ("step", "time_ps", "ns_per_day", "cg_iter", "cg_iter_max", "cg_resid_max"):
            continue
        k = cols.index(c)
        summ[c] = [float(mean[k]), float(err[k])]
    D, tl, msd = diffusion(np.array(coms), rep * a.dt_fs * 1e-3)
    summ["D_1e-5cm2_s"] = float(D)
    np.savetxt(a.out + ".msd", np.c_[tl, msd], header="t (ps)  MSD of the centroid molecular centres (nm^2)")
    print(json.dumps(summ, indent=1))
    with open(a.out + ".json", "w") as fh:
        json.dump(summ, fh, indent=1)
    if a.rdf:
        rc = 0.5 * (edges[1:] + edges[:-1])
        shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        V = np.mean(vol)
        g = np.stack(
            [
                H[0] / (nfr * 0.5 * nO * (nO - 1) / V * shell),
                H[1] / (nfr * nO * (nH - 2) / V * shell),
                H[2] / (nfr * 0.5 * nH * (nH - 2) / V * shell),
            ]
        )
        np.savetxt(a.out + ".rdf", np.c_[rc, g.T], header="r (nm)  g_OO  g_OH  g_HH (intermolecular; bead-averaged)")
        for name, gi in zip(("OO", "OH", "HH"), g):
            k = int(np.argmax(gi * (rc > 0.12)))
            print(f"g_{name}: first peak {rc[k]:.4f} nm, height {gi[k]:.3f}")
    if a.save:
        pi.save_checkpoint(a.out + ".pimd.chk")
        pi.write_restart(a.out + ".rst7")


def bench(a: argparse.Namespace) -> None:
    """Time PIMD steps for each bead number, contraction and bead chunk (the `bench` subcommand).

    Parameters
    ----------
    a : argparse.Namespace
        Options of the `bench` subcommand: beads, contract, chunk (lists), warm and steps (steps
        before and during the timing), out (JSON file of all results, optional).
    """
    out = []
    sim = build(a, log=None)
    sim.minimize(100)
    for P in a.beads:
        for c, ch in itertools.product(a.contract, a.chunk):
            pi = PIMDSimulation(
                sim,
                beads=P,
                contract=c or None,
                thermostat=PILE(a.thermostat, tau_centroid=a.tau_centroid_ps),
                bead_margin=a.bead_margin_nm,
                bead_chunk=ch or None,
            )
            pi.advance(a.warm)
            jax.block_until_ready(pi.state.q)
            cg0 = float(pi.state.eng.cg_total)
            t0 = time.time()
            pi.advance(a.steps)
            jax.block_until_ready(pi.state.q)
            el = time.time() - t0
            o = pi.observables()
            r = {
                "beads": P,
                "contract": c,
                "chunk": ch,
                "dt_fs": a.dt_fs,
                "ms_per_step": el / a.steps * 1e3,
                "ns_per_day": a.steps * a.dt_fs * 1e-6 / el * 86400.0,
                "cg_per_step": (float(pi.state.eng.cg_total) - cg0) / a.steps,
                "temp_K": o["temp_K"],
                "ke_H_cv_meV": o["ke_H_cv_meV"],
                "device": str(jax.devices()[0]),
            }
            print(json.dumps(r), flush=True)
            out.append(r)
            del pi
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)


def batch(a: argparse.Namespace) -> None:
    """Run several command lines of this script in this process (one GPU context for all).

    Each line of the file `a.file` is `OUTFILE ARGS...` (the arguments of this script, e.g.
    `run --beads 8 ...`); stdout and stderr of that run go to OUTFILE.  Empty lines and lines
    starting with # are skipped; an exception is printed to OUTFILE and the next line runs.
    """
    for line in open(a.file):
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        with open(f[0], "w") as out, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                main(f[1:])
            except Exception:
                traceback.print_exc()
        gc.collect()
        print("done", f[0], time.strftime("%H:%M:%S"), flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser with the subcommands template, run, bench and batch."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="fit and save the flexible pGM water template")
    t.add_argument("--out", default=TPL, help="template file (the fit report goes to <out>.json)")
    t.add_argument("--samples", type=int, default=2000, help="monomer geometries of the fit")
    for name, text in (("run", "PIMD / TRPMD run of the 512-water box"), ("bench", "speed per setting")):
        r = sub.add_parser(name, help=text)
        r.add_argument("--template", default=TPL, help="flexible water template (.flex)")
        add_dt_arg(r, 0.25)
        add_temperature_arg(r)
        add_cutoff_arg(r, 0.9)
        add_dipole_tol_arg(r)
        add_precision_arg(r)
        r.add_argument(
            "--thermostat",
            default="pile-g",
            choices=["pile-g", "pile-l"],
            help="PILE centroid thermostat: pile-g (Bussi) or pile-l (Langevin with friction 1/tau_centroid)",
        )
        r.add_argument("--tau-centroid-ps", type=float, default=0.1, help="PILE centroid time constant [ps]")
        r.add_argument(
            "--bead-margin-nm", type=float, default=0.08, help="enlargement of the centroid neighbour list [nm]"
        )
        add_seed_arg(r)
    r = sub.choices["run"]
    r.add_argument("--bead-chunk", default="auto", help="beads per vmapped chunk: auto | 0 (all) | n")
    r.add_argument("--beads", type=int, default=32, help="number of beads P")
    r.add_argument("--contract", type=int, default=0, help="beads of the contracted intermolecular forces (0: none)")
    r.add_argument(
        "--mode", default="pimd", choices=["pimd", "trpmd", "rpmd"], help="pimd (PILE), trpmd or rpmd (dynamics)"
    )
    r.add_argument("--lam", type=float, default=None, help="PILE lambda of the internal modes (default: PILE's)")
    r.add_argument("--propagator", default="cayley", help="free ring-polymer propagator (PIMDSimulation)")
    r.add_argument("--classical-ps", type=float, default=2.0, help="classical flexible equilibration [ps]")
    r.add_argument("--equil-ps", type=float, default=2.0, help="ring-polymer equilibration [ps]")
    r.add_argument("--time-ps", type=float, default=10.0, help="production [ps]")
    r.add_argument("--report-ps", type=float, default=0.05, help="time between samples and log lines [ps]")
    add_barostat_args(r, default="none")
    r.add_argument("--rdf", action="store_true", help="radial distribution functions (<out>.rdf)")
    r.add_argument("--report-pressure", action="store_true", help="also report the virial pressure")
    r.add_argument("--continue-from", default=None, help="PIMD checkpoint to continue from (.pimd.chk)")
    r.add_argument("--save", action="store_true", help="write <out>.pimd.chk and <out>.rst7 at the end")
    r.add_argument("-o", "--out", default="runs/pimd/w", help="output prefix")
    b = sub.choices["bench"]
    b.add_argument("--beads", type=int, nargs="+", default=[8, 32], help="bead numbers")
    b.add_argument("--contract", type=int, nargs="+", default=[0], help="contraction settings (0: none)")
    b.add_argument("--chunk", type=int, nargs="+", default=[0], help="beads per vmapped chunk (0: all)")
    b.add_argument("--steps", type=int, default=400, help="timed steps")
    b.add_argument("--warm", type=int, default=100, help="steps before the timing (compilation)")
    b.add_argument("-o", "--out", default=None, help="JSON file of all results")
    bt = sub.add_parser("batch", help="several command lines of this script in one process")
    bt.add_argument("file", help="lines `OUTFILE ARGS...`")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the subcommand (see the module docstring)."""
    a = build_parser().parse_args(argv)
    if os.environ.get("PIMD_WAIT_GPU"):
        wait_for_gpu()
    setup_logging()
    {"template": template, "run": run, "bench": bench, "batch": batch}[a.cmd](a)


if __name__ == "__main__":
    main()
