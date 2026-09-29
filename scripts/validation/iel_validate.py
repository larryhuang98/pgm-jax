"""Validate the extended-Lagrangian induced dipoles (docs/iel.md) on the pGM water box of the README.

512 waters, or replicated; --model pgm3p25: the same box with the paper's geometry and
Lennard-Jones, as scripts/dielectric/water_dielectric.py --model pgm3p25.  From an equilibrated
checkpoint (--start-from), --seeds segments of Bussi-NVT equilibration (fresh Maxwell velocities)
followed by NVE production, sampling

  * the energy drift (econs = E_tot, NVE) per ns per degree of freedom and its fluctuation;
  * the dipole error: RMS |mu - mu*| / RMS |mu*| and the energy error U - U* against fully converged
    float64 dipoles (tol 1e-9) at the same positions, every --err-ps;
  * the translational diffusion coefficient (centre-of-mass MSD, fit from 2 to 20 ps) and the
    rotational correlation times of the dipole axis (P1, P2; integral of the fitted exponential);
  * the O-O radial distribution function, <U>, <T>, the mean molecular dipole, CG iterations.

Other modes: --npt-ns (independent NPT replicas for eps, density, <U>, <mu_mol>), --eps (pooled
static dielectric constant of such replicas), --combine (summary of several runs' JSON files).
Uses engine internals (Simulation.ff / .nb / .rigid / .integ, PGMForceField._atoms) to share one
force field between the NVT and NVE engines and to recompute converged dipoles.

Usage:

    python scripts/validation/iel_validate.py --start-from prod.chk --dt-fs 2 --iel 0scf -o runs/iel/dyn_0scf
    python scripts/validation/iel_validate.py --start-from prod.chk --dt-fs 2 --dipole-tol 1e-5 -o runs/iel/dyn_scf
    python scripts/validation/iel_validate.py --model pgm3p25 --start-from p25.chk --npt-ns 8 --seeds 1 --iel 0scf
        -o runs/iel/p25eps
    python scripts/validation/iel_validate.py --help

Inputs: PGM_GVDW_DATA (the box; pgm_jax.paths), the checkpoint; .dip / .log files (--eps), JSON
files (--combine).
Outputs: <out>.json (every number) and <out>_rdf.dat; --npt-ns: <out>_s<seed>.{log,dip,chk,out};
--eps: <out>_eps.json.
Units: --dt-fs fs, durations in ps (--equil-ps, --time-ps, --sample-ps, --rdf-ps, --err-ps,
--skip-ps), --npt-ns ns, --density-g-cm3 g/cm^3; D in 1e-9 m^2/s, tau in ps, drift in kT/ns/dof.
Runtime: GPU; each segment compiles two engines and a float64 reference force field.  Sets
jax_enable_x64.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.analysis import dielectric as D
from pgm_jax.analysis.stats import block_mean
from pgm_jax.cli.args import add_dipole_tol_arg, add_dt_arg, add_iel_args, iel_settings, setup_logging
from pgm_jax.cli.main import load_script, scripts_dir
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.box import box_from_cell, volume
from pgm_jax.md.dipoles import read_dipoles
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.rigid import RigidBody
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import Bussi
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import pgm3p25_files
from pgm_jax.system import System
from pgm_jax.units import AMU_NM3_TO_G_CM3, DEBYE_E_NM, KB, KCAL

jax.config.update("jax_enable_x64", True)
TOP, RST = pgm3p25_files()


def min_image(d: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Return the minimum image of displacements d (..., 3) [nm] in the lower-triangular (reduced) box H."""
    for c in (2, 1, 0):
        d = d - np.round(d[..., c] / H[c, c])[..., None] * H[c]
    return d


def block_err(x, nb=5):
    """Return the standard error of the mean of x from nb contiguous blocks (nan with fewer than nb samples)."""
    return block_mean(x, nb)[1] if len(x) >= nb else float("nan")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    ap.add_argument("--start-from", help="equilibrated .chk of the same box (e.g. an NPT run)")
    ap.add_argument(
        "--model",
        default="pgm",
        choices=["pgm", "pgm3p25"],
        help="pgm: the README box (pGM3P-25 "
        "electrostatics on TIP3P's geometry and Lennard-Jones); pgm3p25: the paper's geometry and Lennard-Jones",
    )
    add_dt_arg(ap, 2.0)
    add_dipole_tol_arg(ap)
    ap.add_argument("--seeds", type=int, default=5, help="number of segments / replicas")
    ap.add_argument("--seed0", type=int, default=0, help="first seed (independent jobs: one seed each, then --combine)")
    ap.add_argument(
        "--npt-ns",
        type=float,
        default=0.0,
        help="instead: NPT production of this many ns per seed (Bussi 1 ps, "
        "Monte Carlo barostat every 100 steps, cell dipole every 25 steps) from the checkpoint with fresh "
        "velocities, to prefix_s<seed>.{log,dip,chk}: independent replicas for eps, density, <U>, <mu_mol>",
    )
    ap.add_argument(
        "--eps",
        nargs="+",
        help="instead: pooled eps, density, <U>, <mu_mol> of these .dip files (independent "
        "replicas, each with its .log), the first --skip-ps of each dropped; jackknife over the replicas",
    )
    ap.add_argument("--skip-ps", type=float, default=50.0, help="time dropped at the start of each --eps replica [ps]")
    ap.add_argument("--combine", nargs="+", help="summarise these prefix.json files (with their _rdf.dat) into -o")
    ap.add_argument("--equil-ps", type=float, default=10.0, help="Bussi NVT (tau 1 ps) before each NVE segment [ps]")
    ap.add_argument("--time-ps", type=float, default=100.0, help="NVE per segment [ps]")
    ap.add_argument("--sample-ps", type=float, default=0.02, help="time between frames (MSD, rotations) [ps]")
    ap.add_argument("--rdf-ps", type=float, default=0.2, help="time between RDF frames [ps]")
    ap.add_argument("--err-ps", type=float, default=1.0, help="time between converged-dipole comparisons [ps]")
    ap.add_argument("--replicate", type=int, default=1, help="n: n x n x n copies of the box")
    ap.add_argument(
        "--density-g-cm3",
        type=float,
        default=None,
        help="the checkpoint box is scaled to this density [g/cm^3] "
        "(0: kept); default: <density> of the SCF NPT run of the model (1.0178 pgm, 1.0099 pgm3p25)",
    )
    add_iel_args(ap)
    return ap


def build(a: argparse.Namespace) -> tuple[System, np.ndarray, np.ndarray, MDSettings, int]:
    """Return the system, positions [nm], one-copy box [nm], MD settings and the replication factor n."""
    mols = read_prmtop_molecules(TOP)
    xyz, vel, box = read_coordinates(RST)
    if a.model == "pgm3p25":  # as water_dielectric.py --model pgm3p25
        wd = load_script(os.path.join(scripts_dir(), "dielectric", "water_dielectric.py"))
        xyz = wd.paper_geometry(xyz, 0.9745, 103.64, [list(m.elements) for m in mols])
        sig, eps = 3.18156, 0.14473
        rh = np.array([2 ** (1 / 6) * sig / 2 * 0.1, 0.0, 0.0])
        se = np.array([np.sqrt(eps * KCAL), 0.0, 0.0])
        new = {id(m): dataclasses.replace(m, lj_rmin_half=rh, lj_sqrt_eps=se) for m in mols}
        mols = [new[id(m)] for m in mols]
    H = box_from_cell(*box) * 0.1
    n = a.replicate
    shifts = [i * H[0] + j * H[1] + k * H[2] for i in range(n) for j in range(n) for k in range(n)]
    pos = np.concatenate([xyz * 0.1 + s for s in shifts])
    sys_ = System(mols * len(shifts))
    st = MDSettings().replace(
        cutoff=0.9,
        skin=0.1,
        ewald_beta=4.0,
        pme_grid=(48 * n,) * 3,
        pme_order=6,
        lj_lrc=True,
        dipole_tol=a.dipole_tol,
        precision="mixed",
        **iel_settings(a),
    )
    return sys_, pos, H, st, n


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the mode (see the module docstring)."""
    a = build_parser().parse_args(argv)
    setup_logging()
    if a.combine:
        return combine(a.combine, a.out)
    if a.eps:
        return pooled_eps(a.eps, a.skip_ps, a.out)
    if a.density_g_cm3 is None:
        a.density_g_cm3 = {"pgm": 1.0178, "pgm3p25": 1.0099}[a.model]
    sys_, pos, H, st, n = build(a)
    dt = a.dt_fs / 1000
    kw = dict(settings=st, temperature=298.0, dt=dt, log=None, thermostat=Bussi(1.0))
    if a.npt_ns > 0:
        return npt_replicas(a, sys_, pos, H * n, kw)
    nve_segments(a, sys_, pos, H, st, n, kw)


def nve_segments(
    a: argparse.Namespace, sys_: System, pos: np.ndarray, H: np.ndarray, st: MDSettings, n: int, kw: dict
) -> None:
    """Run the NVT + NVE segments, sample them and write <out>.json and <out>_rdf.dat (module docstring)."""
    nvt = Simulation(sys_, pos, H * n, **kw)
    nve = Simulation(sys_, pos, H * n, thermostat=None, **kw)
    nve.ff = nve.integ.ff = nvt.ff  # one force field and neighbour list, two steps
    nve.nb = nve.integ.nb = nvt.nb
    nve.integ.compile()
    nvt.load_checkpoint(a.start_from)
    start = nvt.state
    if a.density_g_cm3:  # molecular scaling to the target density
        mass = float(np.sum(sys_.masses)) * AMU_NM3_TO_G_CM3
        f = (mass / a.density_g_cm3 / float(volume(start.box))) ** (1.0 / 3.0)
        body = start.dyn.position
        start = start.set(dyn=start.dyn.set(position=RigidBody(body.center * f, body.orientation)), box=start.box * f)
        nvt.state = start
    print(
        f"# {sys_.nmol} waters, dt {a.dt_fs:g} fs, {st.describe_induction()}; start {a.start_from} "
        f"(density {nvt.observables()['density_g_cm3']:.4f})",
        flush=True,
    )
    # fully converged reference: float64, tol 1e-9, from scratch
    ref_s = MDSettings().replace(
        cutoff=0.9,
        skin=0.1,
        ewald_beta=4.0,
        pme_grid=(48 * n,) * 3,
        pme_order=6,
        lj_lrc=True,
        dipole_tol=1e-9,
        max_iter=300,
        precision="double",
    )
    ref = PGMForceField(sys_, np.asarray(start.box), ref_s)

    @jax.jit
    def converged(pos, H, idx):
        """Return the converged float64 dipoles, total energy and CG iterations at pos, H."""
        r = ref.compute(pos, H, idx, ref.init_induction())
        return r.induction.mu, r.energy["total"], r.iterations

    nmol = sys_.nmol
    k_every = max(1, int(round(a.sample_ps / a.dt_fs * 1000)))
    k_rdf = max(1, int(round(a.rdf_ps / a.dt_fs * 1000)))
    k_err = max(1, int(round(a.err_ps / a.dt_fs * 1000)))
    n_prod = int(round(a.time_ps / a.dt_fs * 1000)) // k_every * k_every
    edges = np.linspace(0.0, 0.8, 321)
    hist = np.zeros(len(edges) - 1)
    rdf_frames, rdf_vol = 0, []
    out = {"args": vars(a), "segments": []}
    t_run = 0.0
    for seed in range(a.seed0, a.seed0 + a.seeds):
        nvt.state = nvt.integ.init(start.dyn.position, start.box, jax.random.PRNGKey(1000 + seed))
        nvt.advance(int(round(a.equil_ps / a.dt_fs * 1000)))
        nve.integ.compile()  # row capacities may have grown
        nve.state = nvt.state
        nve.advance(k_every)
        ref.mc, ref.mc_e = nve.ff.mc, nve.ff.mc_e
        E, U, T, ts, mud, it0 = [], [], [], [], [], float(nve.state.cg_total)
        s0 = int(nve.state.step)
        com, axis = [], []
        errs = []
        prev = None
        t0 = time.time()
        for _k in range(0, n_prod, k_every):
            nve.advance(k_every)
            s = nve.state
            o = nve.observables()
            step = int(s.step) - s0
            E.append(o["etot"])
            U.append(o["epot"])
            T.append(o["temp_K"])
            ts.append(step * a.dt_fs / 1000)
            x = nve.positions().reshape(nmol, 3, 3)
            Hh = np.asarray(s.box)
            c = np.asarray(s.dyn.position.center)
            if prev is not None:  # unwrap the centres of mass
                c = prev + min_image(c - prev, Hh)
            prev = c
            com.append(c)
            b = x[:, 1] + x[:, 2] - 2.0 * x[:, 0]
            axis.append(b / np.linalg.norm(b, axis=1, keepdims=True))
            if step % k_rdf == 0:
                xo = x[:, 0]
                d = min_image(xo[:, None, :] - xo[None, :, :], Hh)
                r = np.sqrt(np.sum(d * d, -1))[np.triu_indices(nmol, 1)]
                hist += np.histogram(r, edges)[0]
                rdf_frames += 1
                rdf_vol.append(o["volume_nm3"])
            if step % k_err == 0:
                pos = nve.rigid.positions(s.dyn.position)
                idx = nve.nb.candidates(s.nbr, s.dyn.position.center, s.box, pos)[0]
                mu_ref, e_ref, it = converged(pos, s.box, idx)
                mu = s.induction.mu
                errs.append(
                    [
                        float(jnp.sqrt(jnp.mean((mu - mu_ref) ** 2) / jnp.mean(mu_ref**2))),
                        float(jnp.max(jnp.linalg.norm(mu - mu_ref, axis=1))),
                        float(s.epot) - float(e_ref),
                        int(it),
                    ]
                )
                # mean molecular dipole (charges + permanent + induced), D
                mud.append(float(molecular_dipole(nve.ff, pos, s.box, mu)) / DEBYE_E_NM)
        t_run += time.time() - t0
        ts, E = np.array(ts), np.array(E)
        dof = nve.integ.dof
        slope = np.polyfit(ts, E, 1)[0]  # kJ/mol/ps (ts in ps)
        drift = slope * 1000.0 / dof / (KB * 298.0)  # kT / ns / dof
        resid = E - np.polyval(np.polyfit(ts, E, 1), ts)
        com, axis = np.array(com), np.array(axis)
        D, tau1, tau2 = dynamics(com, axis, a.sample_ps)
        errs = np.array(errs)
        seg = {
            "seed": seed,
            "drift_units_ok": 1,
            "drift_kT_ns_dof": drift,
            "econs_rms_kT_per_dof": float(np.std(resid) / (KB * 298.0) / np.sqrt(dof)),
            "econs_rms_kJ": float(np.std(resid)),
            "T": float(np.mean(T)),
            "U": float(np.mean(U)),
            "D_1e-9_m2_s": D,
            "tau1_ps": tau1,
            "tau2_ps": tau2,
            "mu_rel_rms": float(np.sqrt(np.mean(errs[:, 0] ** 2))),
            "mu_max_err_e_nm": float(errs[:, 1].max()),
            "dU_mean": float(errs[:, 2].mean()),
            "dU_rms": float(np.sqrt(np.mean(errs[:, 2] ** 2))),
            "mol_dipole_D": float(np.mean(mud)),
            "cg_mean": (float(nve.state.cg_total) - it0) / (int(nve.state.step) - s0 + k_every),
        }
        out["segments"].append(seg)
        print(json.dumps(seg), flush=True)
    rc = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist / (rdf_frames * 0.5 * nmol * (nmol - 1) / np.mean(rdf_vol) * shell)
    np.savetxt(a.out + "_rdf.dat", np.c_[rc, g], header="r (nm)  g_OO(r)")
    out["rdf_frames"] = rdf_frames
    out["ns_per_day_incl_sampling"] = a.seeds * a.time_ps / 1000 / t_run * 86400
    summarise(out, [(g, rdf_frames)], rc)
    with open(a.out + ".json", "w") as fh:
        json.dump(out, fh, indent=1)


def npt_replicas(a: argparse.Namespace, sys_: System, pos: np.ndarray, H: np.ndarray, kw: dict) -> None:
    """Run independent NPT replicas from one checkpoint (fresh Maxwell velocities and random streams).

    Each replica (seed) continues from its own <out>_s<seed>.chk when it exists.
    """
    for seed in range(a.seed0, a.seed0 + a.seeds):
        prefix = f"{a.out}_s{seed}"
        kw = dict(kw, log=open(prefix + ".out", "a"))
        sim = Simulation(sys_, pos, H, barostat=MonteCarloBarostat(1.0, 100), **kw)
        if os.path.exists(prefix + ".chk"):  # continue this replica
            sim.load_checkpoint(prefix + ".chk")
            done = int(round(sim.time_ps * 1000 / a.dt_fs))
            append = True
        else:
            sim.load_checkpoint(a.start_from)
            st0 = sim.state
            sim.state = sim.integ.init(st0.dyn.position, st0.box, jax.random.PRNGKey(2000 + seed))
            sim.time_ps, done, append = 0.0, 0, False
        total = int(round(a.npt_ns * 1e6 / a.dt_fs))
        total -= total % 5000
        if total > done:
            sim.run(
                total - done, report_every=5000, checkpoint_every=25000, prefix=prefix, dipoles_every=25, append=append
            )


def pooled_eps(files: list[str], skip: float, prefix: str) -> None:
    """Write the pooled static dielectric constant of independent replicas to <prefix>_eps.json.

    Tin-foil, eps_inf from alpha_cell: <M.M> - <M>.<M> over all samples, jackknife with one block
    per replica (or 10 contiguous blocks for a single run); <V>, density, <U>, <T> from the logs,
    <mu_mol> from the .dip files; skip [ps] dropped at the start of each replica.
    """
    Ms, Vs, As, mus, U, rho, T = [], [], [], [], [], [], []
    temp = None
    for f in files:
        meta, d = read_dipoles([f])
        temp = float(meta["temperature_K"])
        t = d["time_ps"]
        sel = t >= t[0] + skip
        Ms.append(d["M"][sel])
        Vs.append(d["volume_nm3"][sel])
        As.append(d["alpha_nm3"][sel])
        mus.append(d["mol_dipole"][sel])
        dt = float(np.median(np.diff(t)))
        log = f[:-4] + ".log"
        if os.path.exists(log):
            names = open(log).readline().lstrip("#").split()
            x = np.loadtxt(log, ndmin=2)
            keep = x[:, names.index("time_ps")] >= x[0, names.index("time_ps")] - x[0, names.index("time_ps")] + skip
            U.append(x[keep, names.index("epot")])
            rho.append(x[keep, names.index("density_g_cm3")])
            T.append(x[keep, names.index("temp_K")])
    lens = [len(m) for m in Ms]
    # one jackknife block per replica when they have equal lengths, else 10 contiguous blocks of the
    # concatenation (replicas of unequal length)
    nblocks = len(Ms) if (len(Ms) >= 4 and max(lens) == min(lens)) else 10
    M, V, alpha = np.concatenate(Ms), np.concatenate(Vs), np.concatenate(As)
    r = D.static_dielectric(M, V, temp, alpha=alpha, nblocks=nblocks)
    out = {
        "files": files,
        "replicas": len(Ms),
        "samples": int(len(M)),
        "ns": float(len(M) * dt / 1000.0),
        "eps": [r["eps"], r["err"]],
        "eps_inf": [r["eps_inf"], r["eps_inf_err"]],
        "fluct": [r["fluct"], r["fluct_err"]],
        "mol_dipole_D": per_block(np.concatenate(mus) / DEBYE_E_NM, 10),
        "replica_ns": [float(n * dt / 1000.0) for n in lens],
    }
    if U:
        out["density"] = per_block(np.concatenate(rho), 10)
        out["U_kJ_mol"] = per_block(np.concatenate(U), 10)
        out["T"] = per_block(np.concatenate(T), 10)
    print(json.dumps(out), flush=True)
    with open(prefix + "_eps.json", "w") as fh:
        json.dump(out, fh, indent=1)


def per_block(x: np.ndarray, nb: int) -> list[float]:
    """Return [mean, standard error from nb contiguous blocks] of x."""
    x = np.asarray(x, float)
    m = len(x) // nb * nb
    b = x[:m].reshape(nb, -1).mean(1)
    return [float(x.mean()), float(b.std(ddof=1) / np.sqrt(nb))]


def summarise(out, rdfs, rc):
    """Mean +- standard error over the segments; RDF peak and minimum of the frame-weighted RDF."""
    segs = out["segments"]
    keys = [k for k in segs[0] if k not in ("seed", "drift_units_ok")]
    summ = {}
    for k in keys:
        v = np.array([sg[k] for sg in segs], float)
        summ[k] = [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")]
    g = sum(gi * n for gi, n in rdfs) / sum(n for _, n in rdfs)
    kmax = int(np.argmax(g))
    kmin = kmax + int(np.argmin(g[kmax : kmax + 60]))
    summ["gOO_peak"] = [float(rc[kmax]), float(g[kmax])]
    summ["gOO_min"] = [float(rc[kmin]), float(g[kmin])]
    summ["n_segments"] = len(segs)
    out["summary"] = summ
    print("SUMMARY " + json.dumps(summ), flush=True)
    return g


def combine(files: list[str], prefix: str) -> None:
    """Summarise several runs' JSON (and _rdf.dat) files into <prefix>.json and <prefix>_rdf.dat."""
    out, rdfs, rc = {"parts": files, "segments": []}, [], None
    for f in files:
        d = json.load(open(f))
        for sg in d["segments"]:  # early runs: drift printed in kT / ps / dof
            if not sg.get("drift_units_ok"):
                sg["drift_kT_ns_dof"] *= 1000.0
                sg["drift_units_ok"] = 1
        out["segments"] += d["segments"]
        r = np.loadtxt(f[:-5] + "_rdf.dat")
        rc = r[:, 0]
        rdfs.append((r[:, 1], d.get("rdf_frames", 1)))
    g = summarise(out, rdfs, rc)
    np.savetxt(prefix + "_rdf.dat", np.c_[rc, g], header="r (nm)  g_OO(r)")
    with open(prefix + ".json", "w") as fh:
        json.dump(out, fh, indent=1)


def molecular_dipole(ff: PGMForceField, pos: jax.Array, H: jax.Array, mu: jax.Array) -> jax.Array:
    """Return the mean |molecular dipole| [e nm]: charges about the centre of mass + permanent + induced."""
    P = ff._atoms(None)
    p = ff.perm_dipoles(pos, H, P["cov"])
    m = ff.masses
    nm = ff.sys.nmol
    mol = ff.mol
    com = jax.ops.segment_sum(m[:, None] * pos, mol, nm) / jax.ops.segment_sum(m, mol, nm)[:, None]
    dx = pos - com[mol]
    dip = jax.ops.segment_sum(P["q"][:, None] * dx + p + mu, mol, nm)
    return jnp.mean(jnp.linalg.norm(dip, axis=1))


def dynamics(com: np.ndarray, axis: np.ndarray, dt_frame: float) -> tuple[float, float, float]:
    """Return D [1e-9 m^2/s] and the rotational correlation times tau1, tau2 [ps] of the dipole axis.

    D from the MSD of the unwrapped centres of mass com (F, M, 3) [nm] (linear fit 2-20 ps, all
    time origins); tau_l = integral of <P_l(u(0).u(t))> of the unit axes (F, M, 3), the integral
    taken to where the correlation falls below 0.05 and completed with the exponential fitted
    between 0.3 and 0.05 (tau2 of pGM water is ~1-2 ps); frames every dt_frame [ps].
    """
    nf = len(com)
    lags = np.unique(np.round(np.geomspace(1, nf // 2, 80)).astype(int))
    t = lags * dt_frame
    msd = np.array([np.mean(np.sum((com[l:] - com[:-l]) ** 2, -1)) for l in lags])
    T = nf * dt_frame
    sel = (t >= min(2.0, 0.1 * T)) & (t <= min(20.0, 0.4 * T))
    D = np.polyfit(t[sel], msd[sel], 1)[0] / 6.0 * 1e3  # nm^2/ps -> 1e-9 m^2/s
    lin = np.unique(np.round(np.geomspace(1, min(nf // 2, int(round(20.0 / dt_frame))), 150)).astype(int))
    tl = np.concatenate([[0.0], lin * dt_frame])
    taus = []
    for l in (1, 2):
        c = [1.0]
        for k in lin:
            x = np.sum(axis[k:] * axis[:-k], -1)
            c.append(np.mean(x if l == 1 else 1.5 * x * x - 0.5))
        c = np.array(c)
        below = np.nonzero(c < 0.05)[0]
        end = below[0] if len(below) else len(c) - 1
        fit = (c > 0.05) & (c < 0.3)
        tail = 0.0
        if fit.sum() >= 3:
            s, b = np.polyfit(tl[fit], np.log(c[fit]), 1)
            tail = c[end] * (-1.0 / s) if s < 0 else 0.0
        taus.append(float(np.trapezoid(c[: end + 1], tl[: end + 1]) + tail))
    return float(D), taus[0], taus[1]


if __name__ == "__main__":
    main()
