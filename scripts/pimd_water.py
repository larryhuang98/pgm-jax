"""Quantum (path-integral) flexible pGM water: template, PIMD / TRPMD runs, radial distribution
functions, speed.

    python scripts/pimd_water.py template                       # fit the flexible pGM water (q-TIP4P/F monomer PES)
    python scripts/pimd_water.py run --beads 32 --ps 20 --prefix runs/pimd/w32          # PIMD, 512 waters, 298 K
    python scripts/pimd_water.py run --beads 1 --ps 20 --prefix runs/pimd/w1            # classical flexible water
    python scripts/pimd_water.py run --beads 32 --contract 1 ...                        # contracted to the centroid
    python scripts/pimd_water.py run --beads 32 --mode trpmd --load runs/pimd/w32.pimd.chk --ps 20   # dynamics
    python scripts/pimd_water.py bench --beads 8 32 --contract 0 1                      # ns/day

The water is the pGM water of ~/pgm-gvdw-data/topology/rayl_512_v2.prmtop (the 512-water box of the
README: charges, covalent dipoles, radii, polarizabilities, Lennard-Jones on O), made flexible with
bonded terms fitted so that its gas-phase monomer potential (bonded + all-pair intramolecular pGM)
is the q-TIP4P/F intramolecular potential (pimd.flexible_water).  NVT at the density of the
box's restart (from the rigid model's NPT)."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

if os.environ.get("PIMD_WAIT_GPU"):       # shared GPU: wait until it is idle, take it at once (before the
    import subprocess                      # imports below touch the device); 75: someone else was faster
    while subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip():
        time.sleep(0.1)
    try:
        jax.devices()
    except RuntimeError:
        sys.exit(75)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
jax.config.update("jax_enable_x64", True)

from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.md.io import box_from_cell, read_coordinates  # noqa: E402
from pgm_jax.md.pimd import KJMOL_TO_MEV, PIMDSimulation, flexible_water  # noqa: E402
from pgm_jax.param import read_prmtop_pgm  # noqa: E402
from pgm_jax.system import System  # noqa: E402

TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
RST = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")
TPL = os.path.join(ROOT, "validation/pimd/pgm_water_flex.flex")


def template(a):
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


def build(a, log=sys.stdout):
    tpl = FlexibleTemplate.load(a.template)
    xyz, _, box = read_coordinates(RST)
    H = box_from_cell(*box) * 0.1
    pos = xyz * 0.1
    n = len(pos) // 3
    s = MDSettings(cutoff=a.cut, dipole_tol=a.tol, precision=a.precision)
    sim = FlexibleSimulation(System([tpl.pgm] * n), [tpl] * n, pos, H, s, dt=a.dt * 1e-3, ensemble="nvt",
                             temperature=a.temp, thermostat="bussi", tau_t=0.1, seed=a.seed, log=log)
    return sim


def rdf_fn(sim, edges):
    """Jitted per-bead histograms of O-O, O-H and H-H distances (minimum image, reduced box)."""
    el = np.array(sim.sys.elements)
    O, Hy = np.nonzero(el == "O")[0], np.nonzero(el == "H")[0]
    mol = np.asarray(sim.sys.mol)
    same_oh = jnp.asarray(mol[O][:, None] == mol[Hy][None, :])
    same_hh = jnp.asarray(mol[Hy][:, None] == mol[Hy][None, :])
    e = jnp.asarray(edges)

    def dist(x, y, H):
        d = x[:, None, :] - y[None, :, :]
        for c in (2, 1, 0):
            d = d - jnp.round(d[..., c] / H[c, c])[..., None] * H[c]
        return jnp.sqrt(jnp.sum(d * d, -1))

    def one(q, H):
        rOO = dist(q[O], q[O], H)
        iu = jnp.triu_indices(len(O), 1)
        hOO = jnp.histogram(rOO[iu], bins=e)[0]
        rOH = dist(q[O], q[Hy], H)
        hOH = jnp.histogram(jnp.where(same_oh, -1.0, rOH), bins=e)[0]           # intermolecular O-H
        rHH = dist(q[Hy], q[Hy], H)
        hHH = jnp.histogram(jnp.where(same_hh, -1.0, rHH), bins=e)[0] // 2      # intermolecular H-H pairs
        return jnp.stack([hOO, hOH, hHH])

    @jax.jit
    def hist(Q, H):
        return jnp.sum(jax.lax.map(lambda q: one(q, H), Q), 0)
    return hist, len(O), len(Hy)


def diffusion(C, dt_ps):
    """Self-diffusion coefficient (1e-5 cm^2/s) from unwrapped molecular centres C (F, nmol, 3) nm
    sampled every dt_ps: MSD over all time origins, slope of a line through lags between 20 % and
    50 % of the run (Einstein relation, MSD = 6 D t); no finite-size correction."""
    F = len(C)
    lags = np.arange(1, F // 2 + 1)
    msd = np.array([np.mean(np.sum((C[k:] - C[:-k]) ** 2, -1)) for k in lags])
    t = lags * dt_ps
    sel = (lags >= max(1, int(0.2 * F))) & (lags <= F // 2)
    slope = np.polyfit(t[sel], msd[sel], 1)[0]
    return slope / 6.0 * 1000.0, t, msd


def run(a):
    os.makedirs(os.path.dirname(os.path.abspath(a.prefix)), exist_ok=True)
    sim = build(a)
    if a.load is None:
        sim.minimize(200)
        if a.classical_ps > 0:                    # classical flexible equilibration (Bussi)
            sim.run(int(round(a.classical_ps / (a.dt * 1e-3))), report=1000, prefix=a.prefix + "_classical")
    pi = PIMDSimulation(sim, beads=a.beads, mode=a.mode, thermostat=a.thermostat, tau0=a.tau0, lam=a.lam,
                        propagator=a.propagator, contract=a.contract or None, bead_margin=a.bead_margin,
                        seed=a.seed, ensemble=a.ensemble, pressure=a.press, barostat_interval=a.barostat_interval,
                        bead_chunk=a.bead_chunk or None)
    if a.load:
        pi.load(a.load)
        pi.state = pi.state.set(heat=jnp.zeros(()), step=jnp.zeros((), jnp.int32),
                                eng=pi.state.eng.set(cg_total=jnp.zeros(())))
    nstep = int(round(a.ps / (a.dt * 1e-3)))
    neq = int(round(a.equil_ps / (a.dt * 1e-3)))
    rep = max(1, int(round(a.report_ps / (a.dt * 1e-3))))
    if neq:
        pi.run(neq, report=rep, prefix=a.prefix + "_equil")
    edges = np.linspace(0.0, 0.8, 321)
    hist, nO, nH = rdf_fn(sim, edges)
    H = np.zeros((3, len(edges) - 1))
    nfr, vol = 0, []
    samples = []
    coms, last = [], None
    t0 = time.time()
    done = 0
    logf = open(a.prefix + ".log", "w")
    cols = None
    while done < nstep:
        m = min(rep, nstep - done)
        pi._advance(m)
        done += m
        o = pi.observables()
        if a.pressure:
            o["press_bar"] = pi.pressure()
        o["ns_per_day"] = done * a.dt * 1e-6 / max(time.time() - t0, 1e-9) * 86400.0
        if cols is None:
            cols = list(o)
            logf.write("# " + " ".join(f"{c:>14s}" for c in cols) + "\n")
        logf.write("  " + " ".join(f"{o[c]:14.6f}" if isinstance(o[c], float) else f"{o[c]:14d}" for c in cols) + "\n")
        logf.flush()
        samples.append([o[c] for c in cols])
        com = np.asarray(sim.flex.centers(jnp.mean(pi.state.q, 0)))           # centroid molecular centres
        if coms:                                                                 # unwrap (minimum image step)
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
    blocks = X[len(X) - mb:].reshape(nb, -1, X.shape[1]).mean(1)
    mean, err = X.mean(0), blocks.std(0, ddof=1) / np.sqrt(nb)
    summ = {"beads": a.beads, "contract": a.contract, "mode": a.mode, "thermostat": a.thermostat, "dt_fs": a.dt,
            "ps": a.ps, "ns_per_day": nstep * a.dt * 1e-6 / el * 86400.0, "ms_per_step": el / nstep * 1e3}
    drift = np.polyfit(X[:, cols.index("time_ps")], X[:, cols.index("econs")], 1)[0]
    summ["econs_drift_kJmol_ps"] = float(drift)
    for c in cols:
        if c in ("step", "time_ps", "ns_per_day", "cg_iter", "cg_iter_max", "cg_resid_max"):
            continue
        k = cols.index(c)
        summ[c] = [float(mean[k]), float(err[k])]
    D, tl, msd = diffusion(np.array(coms), rep * a.dt * 1e-3)
    summ["D_1e-5cm2_s"] = float(D)
    np.savetxt(a.prefix + ".msd", np.c_[tl, msd], header="t (ps)  MSD of the centroid molecular centres (nm^2)")
    print(json.dumps(summ, indent=1))
    with open(a.prefix + ".json", "w") as fh:
        json.dump(summ, fh, indent=1)
    if a.rdf:
        rc = 0.5 * (edges[1:] + edges[:-1])
        shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        V = np.mean(vol)
        g = np.stack([H[0] / (nfr * 0.5 * nO * (nO - 1) / V * shell),
                      H[1] / (nfr * nO * (nH - 2) / V * shell),
                      H[2] / (nfr * 0.5 * nH * (nH - 2) / V * shell)])
        np.savetxt(a.prefix + ".rdf", np.c_[rc, g.T], header="r (nm)  g_OO  g_OH  g_HH (intermolecular; bead-averaged)")
        for name, gi in zip(("OO", "OH", "HH"), g):
            k = int(np.argmax(gi * (rc > 0.12)))
            print(f"g_{name}: first peak {rc[k]:.4f} nm, height {gi[k]:.3f}")
    if a.save:
        pi.save(a.prefix)


def bench(a):
    out = []
    sim = build(a, log=None)
    sim.minimize(100)
    for P in a.beads:
        for c, ch in itertools.product(a.contract, a.chunk):
            pi = PIMDSimulation(sim, beads=P, contract=c or None, thermostat=a.thermostat, tau0=a.tau0, log=None,
                                bead_margin=a.bead_margin, bead_chunk=ch or None)
            pi._advance(a.warm)
            jax.block_until_ready(pi.state.q)
            cg0 = float(pi.state.eng.cg_total)
            t0 = time.time()
            pi._advance(a.steps)
            jax.block_until_ready(pi.state.q)
            el = time.time() - t0
            o = pi.observables()
            r = {"beads": P, "contract": c, "chunk": ch, "dt_fs": a.dt, "ms_per_step": el / a.steps * 1e3,
                 "ns_per_day": a.steps * a.dt * 1e-6 / el * 86400.0,
                 "cg_per_step": (float(pi.state.eng.cg_total) - cg0) / a.steps,
                 "temp_K": o["temp_K"], "ke_H_cv_meV": o["ke_H_cv_meV"], "device": str(jax.devices()[0])}
            print(json.dumps(r), flush=True)
            out.append(r)
            del pi
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--out", default=TPL)
    t.add_argument("--samples", type=int, default=2000)
    for name in ("run", "bench"):
        r = sub.add_parser(name)
        r.add_argument("--template", default=TPL)
        r.add_argument("--dt", type=float, default=0.25, help="fs")
        r.add_argument("--temp", type=float, default=298.0)
        r.add_argument("--cut", type=float, default=0.9)
        r.add_argument("--tol", type=float, default=1e-5)
        r.add_argument("--precision", default="mixed")
        r.add_argument("--thermostat", default="pile-g")
        r.add_argument("--tau0", type=float, default=0.1)
        r.add_argument("--bead-margin", type=float, default=0.06)
        r.add_argument("--seed", type=int, default=0)
        r.add_argument("--bead-chunk", type=int, default=0)
    r = sub.choices["run"]
    r.add_argument("--beads", type=int, default=32)
    r.add_argument("--contract", type=int, default=0)
    r.add_argument("--mode", default="pimd")
    r.add_argument("--lam", type=float, default=None)
    r.add_argument("--propagator", default="cayley")
    r.add_argument("--classical-ps", type=float, default=2.0)
    r.add_argument("--equil-ps", type=float, default=2.0)
    r.add_argument("--ps", type=float, default=10.0)
    r.add_argument("--report-ps", type=float, default=0.05)
    r.add_argument("--ensemble", default="nvt")
    r.add_argument("--press", type=float, default=1.0, help="bar")
    r.add_argument("--barostat-interval", type=int, default=100)
    r.add_argument("--rdf", action="store_true")
    r.add_argument("--pressure", action="store_true")
    r.add_argument("--load", default=None)
    r.add_argument("--save", action="store_true")
    r.add_argument("--prefix", default="runs/pimd/w")
    b = sub.choices["bench"]
    b.add_argument("--beads", type=int, nargs="+", default=[8, 32])
    b.add_argument("--contract", type=int, nargs="+", default=[0])
    b.add_argument("--chunk", type=int, nargs="+", default=[0], help="beads per vmapped chunk (0: all)")
    b.add_argument("--steps", type=int, default=400)
    b.add_argument("--warm", type=int, default=100)
    b.add_argument("--out", default=None)
    a = ap.parse_args()
    {"template": template, "run": run, "bench": bench}[a.cmd](a)


if __name__ == "__main__":
    main()
