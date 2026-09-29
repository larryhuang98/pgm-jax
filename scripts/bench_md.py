"""MD speed benchmark: pGM3P-25 water, optionally replicated n x n x n (the truncated-octahedron
lattice tiles by lattice vectors).  Defaults are the settings of the pmemd.pgm.cuda comparison in
the README (NVT, gamma 1/ps, 9 A cutoff, PME 48^3 per replica, order 6, dipole_scf_tol 1e-5).

    python scripts/bench_md.py --replicate 2 --steps 10000 --precision mixed
    python scripts/bench_md.py --replicate 2 --engine constraints --dt 0.002     # atoms + SHAKE/RATTLE
    python scripts/bench_md.py --replicate 2 --elec-cut 0.7 --grid 48           # electrostatics cut at 0.7 nm
    # r-RESPA
    python scripts/bench_md.py --replicate 2 --engine constraints --hmr 4.0 --thermostat bussi --dt 0.008 --mts 2
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.cli.args import add_iel_arguments, add_mts_arguments, iel_settings, mts_from_args
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.forcefield import DSUM_TOL, MDSettings, ewald_beta_for
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.mts import mts_stats
from pgm_jax.md.simulation import Simulation
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import resource
from pgm_jax.system import System

jax.config.update("jax_enable_x64", True)
TOP = resource("gvdw_data", "topology/rayl_512_v2.prmtop")
RST = resource("gvdw_data", "inputs/lj/inpcrd.restrt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--ensemble", default="nvt")
    ap.add_argument("--cut", type=float, default=0.9, help="van der Waals (and default electrostatics) cutoff (nm)")
    ap.add_argument(
        "--elec-cut",
        type=float,
        default=None,
        help="real-space electrostatics cutoff (nm); --beta then "
        "defaults to ewald_beta_for(elec_cut, dsum_tol) and --grid to 48 (beta / 4)^1.6 (a multiple of 4; "
        "the grid rule of elec_cutoff_settings)",
    )
    ap.add_argument(
        "--dsum-tol", type=float, default=DSUM_TOL, help="direct-sum tolerance for --elec-cut (Amber convention)"
    )
    ap.add_argument("--beta", type=float, default=None, help="Ewald coefficient (nm^-1); default 4.0")
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--lrc", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--thermostat", default="langevin", help="langevin | bussi | gle")
    ap.add_argument("--tau", type=float, default=1.0, help="Bussi time constant (ps)")
    ap.add_argument("--hmr", type=float, default=None, help="hydrogen mass (amu), constraints engine only")
    ap.add_argument("--barostat-interval", type=int, default=100, help="steps between Monte Carlo volume moves (npt)")
    ap.add_argument(
        "--grid", type=int, default=None, help="PME points per replica along each lattice vector (default 48)"
    )
    ap.add_argument("--skin", type=float, default=0.1)
    ap.add_argument(
        "--engine",
        default="rigid",
        help="rigid (rigid bodies) | constraints (atoms + SHAKE/RATTLE, "
        "the engine of flexible and macromolecular systems)",
    )
    ap.add_argument(
        "--ps",
        type=float,
        default=0.0,
        help="after the speed test: this many ps sampled every 0.5 ps (econs drift, <U>, group temperatures, density)",
    )
    ap.add_argument(
        "--rdf",
        default=None,
        help="with --ps: O-O radial distribution function of the samples to this file "
        "(r in nm, g) and its first peak in the log",
    )
    add_mts_arguments(ap)
    add_iel_arguments(ap)
    a = ap.parse_args()
    mts = mts_from_args(a)
    mols = read_prmtop_molecules(TOP)
    xyz, vel, box = read_coordinates(RST)
    H = box_from_cell(*box) * 0.1
    n = a.replicate
    shifts = [i * H[0] + j * H[1] + k * H[2] for i in range(n) for j in range(n) for k in range(n)]
    pos = np.concatenate([xyz * 0.1 + s for s in shifts])
    v = np.concatenate([vel * 0.1] * len(shifts))
    sys_ = System(mols * len(shifts))
    beta = a.beta if a.beta is not None else (4.0 if a.elec_cut is None else ewald_beta_for(a.elec_cut, a.dsum_tol))
    per = a.grid if a.grid is not None else int(np.ceil(48 * (beta / 4.0) ** 1.6 / 4.0 - 1e-9)) * 4
    grid = tuple(per * n for _ in range(3))
    st = MDSettings(
        cutoff=a.cut,
        skin=a.skin,
        ewald_beta=beta,
        pme_grid=grid,
        pme_order=a.order,
        lj_lrc=bool(a.lrc),
        dipole_tol=a.tol,
        precision=a.precision,
        elec_cutoff=a.elec_cut,
        **iel_settings(a),
    )
    if a.engine == "rigid":
        sim = Simulation(
            sys_,
            pos,
            H * n,
            settings=st,
            ensemble=a.ensemble,
            temperature=298.0,
            gamma=a.gamma,
            barostat_interval=a.barostat_interval,
            dt=a.dt,
            vel_nm_ps=v,
            log=sys.stdout,
            thermostat=a.thermostat,
            tau_t=a.tau,
            mts=mts,
        )
    else:
        from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate

        tpl = {id(m): RigidTemplate(m, pos[sys_.atom_slice(k)]) for k, m in enumerate(sys_.molecules)}
        sim = FlexibleSimulation(
            sys_,
            [tpl[id(m)] for m in sys_.molecules],
            pos,
            H * n,
            settings=st,
            ensemble=a.ensemble,
            temperature=298.0,
            gamma=a.gamma,
            barostat_interval=a.barostat_interval,
            dt=a.dt,
            log=sys.stdout,
            thermostat=a.thermostat,
            tau_t=a.tau,
            hmr=a.hmr,
            mts=mts,
        )
        print("# masses of the first molecule:", np.asarray(sim.flex.masses)[:3], flush=True)
    blk = max(1, int(round(1.0 / a.dt)))  # 1 ps blocks
    sim._advance(blk)  # compile + warm up
    cg0, s0 = float(sim.state.cg_total), int(sim.state.step)
    t0 = time.time()
    done = 0
    while done < a.steps:
        sim._advance(blk)
        done += blk
    el = time.time() - t0
    o = sim.observables()
    print(
        f"{a.engine}: {sys_.nmol} waters ({sys_.n} atoms), dt {a.dt * 1000:g} fs, {a.precision}, {a.ensemble}, "
        f"{st.describe_cutoffs()}, beta {beta:.4f}, PME {grid} order {a.order}, "
        f"{st.describe_induction()}, skin {a.skin}, rows {sim.ff.mc} (electrostatic {sim.ff.mc_e or sim.ff.mc}): "
        f"{el / done * 1e3:.3f} ms/step, "
        f"{done * a.dt / 1000 / el * 86400:.1f} ns/day; T {o['temp_K']:.1f} K, density {o['density_g_cm3']:.4f}, "
        f"CG iters {(float(sim.state.cg_total) - cg0) / (int(sim.state.step) - s0):.2f} mean per step, max "
        f"{o['cg_iter_max']}{'; ' + str(mts_stats(sim)) if mts else ''}",
        flush=True,
    )
    if a.ps > 0:
        from pgm_jax.units import KB

        n = max(1, int(round(0.5 / a.dt)))
        keys = (
            "time_ps",
            "econs",
            "epot",
            "temp_K",
            "temp_trans",
            "temp_rot",
            "temp_com",
            "temp_internal",
            "density_g_cm3",
        )
        X = []
        edges = np.linspace(0.0, 0.8, 321)
        hist = np.zeros(len(edges) - 1)
        oxy = np.nonzero(np.array(sys_.elements) == "O")[0]

        @jax.jit
        def oo_hist(x, H):  # O-O distances (minimum image, reduced box)
            d = x[:, None, :] - x[None, :, :]
            for c in (2, 1, 0):
                d = d - jnp.round(d[..., c] / H[c, c])[..., None] * H[c]
            r = jnp.sqrt(jnp.sum(d * d, -1))
            return jnp.histogram(r[jnp.triu_indices(len(x), 1)], bins=jnp.asarray(edges))[0]

        vol = []
        t1 = time.time()
        for _ in range(int(round(a.ps / 0.5))):
            sim._advance(n)
            o = sim.observables()
            X.append([o.get(k, np.nan) for k in keys])
            if a.rdf:
                box = jnp.asarray(sim.state.box)
                hist += np.asarray(oo_hist(jnp.asarray(sim.positions_nm()[oxy]), box))
                vol.append(float(o["volume_nm3"]))
        X = np.array(X)
        nb = 10
        m = len(X) // nb * nb
        blocks = X[len(X) - m :].reshape(nb, -1, X.shape[1]).mean(1)
        err = blocks.std(0, ddof=1) / np.sqrt(nb)
        drift = np.polyfit(X[:, 0], X[:, 1], 1)[0] * 1000.0 / sim.integ.dof / (KB * 298.0)
        mean = X.mean(0)
        print(
            f"sampled {a.ps:g} ps ({time.time() - t1:.0f} s): econs drift {drift:+.4f} kT/ns/dof, <U> {mean[2]:.1f} +- "
            f"{err[2]:.1f} kJ/mol, <T> {mean[3]:.2f} +- {err[3]:.2f}, group T "
            + " ".join(
                f"{k} {mean[i]:.2f} +- {err[i]:.2f}" for i, k in enumerate(keys) if 4 <= i <= 7 and np.isfinite(mean[i])
            )
            + f", density {mean[8]:.4f} +- {err[8]:.4f} g/cm3{'; ' + str(mts_stats(sim)) if mts else ''}",
            flush=True,
        )
        if a.rdf:
            rc = 0.5 * (edges[1:] + edges[:-1])
            shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
            no = len(oxy)
            g = hist / (len(vol) * 0.5 * no * (no - 1) / np.mean(vol) * shell)
            np.savetxt(a.rdf, np.c_[rc, g], header="r (nm)  g_OO(r)")
            k = int(np.argmax(g))
            print(f"g_OO: first peak {rc[k]:.4f} nm, height {g[k]:.3f}", flush=True)


if __name__ == "__main__":
    main()
