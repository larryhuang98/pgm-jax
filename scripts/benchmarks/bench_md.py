"""MD speed benchmark: pGM3P-25 water, optionally replicated n x n x n (the `pgm-jax bench` command).

The truncated-octahedron box tiles by its lattice vectors.  Defaults are the settings of the
pmemd.pgm.cuda comparison in the README (NVT at 298 K, Langevin 1/ps, 0.9 nm cutoff, PME 48^3 per
replica, order 6, dipole tolerance 1e-5); the rigid engine or the constrained flexible engine.
After the speed test, --time-ps samples every 0.5 ps (drift of the conserved energy, <U>, group
temperatures, density; with --rdf the O-O radial distribution function).  scripts/dev/gpu_bench.sh
runs a fixed set of cases for the speed checks of the clean-up.

Usage:

    python scripts/benchmarks/bench_md.py --replicate 2 --steps 10000 --precision mixed
    python scripts/benchmarks/bench_md.py --replicate 2 --engine constraints --dt-fs 2       # atoms + SHAKE/RATTLE
    python scripts/benchmarks/bench_md.py --replicate 2 --elec-cutoff-nm 0.7 --grid 48      # elec. cut at 0.7 nm
    # r-RESPA
    python scripts/benchmarks/bench_md.py --replicate 2 --engine constraints --hmr-amu 4.0 --thermostat bussi
        --dt-fs 8 --mts 2
    python scripts/benchmarks/bench_md.py --help

Inputs: PGM_GVDW_DATA (the 512-water box, pgm_jax.paths).
Outputs: printed timing line (ms/step, ns/day, CG iterations) and statistics; --rdf file.
Units: --dt-fs fs, --cutoff-nm, --elec-cutoff-nm and --skin-nm nm, --ewald-beta-per-nm 1/nm,
--temperature-K K, --time-ps ps, --hmr-amu amu.
Runtime: GPU (a few ms per step for 4,096 waters); sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.cli.args import (
    add_barostat_args,
    add_dipole_tol_arg,
    add_dt_arg,
    add_iel_args,
    add_mts_args,
    add_precision_arg,
    add_temperature_arg,
    add_thermostat_args,
    coupling_from_args,
    iel_settings,
    mts_from_args,
    setup_logging,
)
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
from pgm_jax.md.forcefield import DSUM_TOL, MDSettings, ewald_beta_for
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.mts import mts_stats
from pgm_jax.md.simulation import Simulation
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import pgm3p25_files
from pgm_jax.system import System
from pgm_jax.units import KB

jax.config.update("jax_enable_x64", True)
TOP, RST = pgm3p25_files()
KEYS = ("time_ps", "econs", "epot", "temp_K", "temp_trans", "temp_rot", "temp_com", "temp_internal", "density_g_cm3")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicate", type=int, default=1, help="n: n x n x n copies of the 512-water box")
    ap.add_argument("--steps", type=int, default=5000, help="timed steps (in 1 ps blocks)")
    add_precision_arg(ap)
    ap.add_argument(
        "--cutoff-nm", type=float, default=0.9, help="van der Waals (and default electrostatics) cutoff [nm]"
    )
    ap.add_argument(
        "--elec-cutoff-nm",
        type=float,
        default=None,
        help="real-space electrostatics cutoff [nm]; --ewald-beta-per-nm then "
        "defaults to ewald_beta_for(elec_cutoff, dsum_tol) and --grid to 48 (beta / 4)^1.6 (a multiple of 4; "
        "the grid rule of elec_cutoff_settings)",
    )
    ap.add_argument(
        "--dsum-tol", type=float, default=DSUM_TOL, help="direct-sum tolerance for --elec-cutoff-nm (Amber convention)"
    )
    ap.add_argument("--ewald-beta-per-nm", type=float, default=None, help="Ewald coefficient [1/nm]; default 4.0")
    add_dt_arg(ap, 1.0)
    ap.add_argument("--order", type=int, default=6, help="PME order")
    add_dipole_tol_arg(ap)
    ap.add_argument("--lrc", type=int, default=1, help="LJ long-range correction (1: on)")
    add_temperature_arg(ap, 298.0)
    add_thermostat_args(ap, default="langevin", choices=("langevin", "bussi", "gle", "gle-lowpass"))
    add_barostat_args(ap, default="none")
    ap.add_argument(
        "--hmr-amu", type=float, default=None, help="hydrogen mass [amu], constraints engine only (default: unchanged)"
    )
    ap.add_argument(
        "--grid", type=int, default=None, help="PME points per replica along each lattice vector (default 48)"
    )
    ap.add_argument("--skin-nm", type=float, default=0.1, help="neighbour-list skin [nm]")
    ap.add_argument(
        "--engine",
        default="rigid",
        choices=["rigid", "constraints"],
        help="rigid (rigid bodies) | constraints (atoms + SHAKE/RATTLE, "
        "the engine of flexible and macromolecular systems)",
    )
    ap.add_argument(
        "--time-ps",
        type=float,
        default=0.0,
        help="after the speed test: this many ps sampled every 0.5 ps (econs drift, <U>, group temperatures, density)",
    )
    ap.add_argument(
        "--rdf",
        default=None,
        help="with --time-ps: O-O radial distribution function of the samples to this file "
        "(r in nm, g) and its first peak in the log",
    )
    add_mts_args(ap)
    add_iel_args(ap)
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, build the box, time the MD and sample (see the module docstring)."""
    a = build_parser().parse_args(argv)
    setup_logging()
    dt = a.dt_fs / 1000
    mts = mts_from_args(a)
    mols = read_prmtop_molecules(TOP)
    xyz, vel, box = read_coordinates(RST)
    H = box_from_cell(*box) * 0.1
    n = a.replicate
    shifts = [i * H[0] + j * H[1] + k * H[2] for i in range(n) for j in range(n) for k in range(n)]
    pos = np.concatenate([xyz * 0.1 + s for s in shifts])
    v = np.concatenate([vel * 0.1] * len(shifts))
    sys_ = System(mols * len(shifts))
    beta = (
        a.ewald_beta_per_nm
        if a.ewald_beta_per_nm is not None
        else (4.0 if a.elec_cutoff_nm is None else ewald_beta_for(a.elec_cutoff_nm, a.dsum_tol))
    )
    per = a.grid if a.grid is not None else int(np.ceil(48 * (beta / 4.0) ** 1.6 / 4.0 - 1e-9)) * 4
    grid = tuple(per * n for _ in range(3))
    st = MDSettings().replace(
        cutoff=a.cutoff_nm,
        skin=a.skin_nm,
        ewald_beta=beta,
        pme_grid=grid,
        pme_order=a.order,
        lj_lrc=bool(a.lrc),
        dipole_tol=a.dipole_tol,
        precision=a.precision,
        elec_cutoff=a.elec_cutoff_nm,
        **iel_settings(a),
    )
    th, baro = coupling_from_args(a)
    if a.engine == "rigid":
        sim = Simulation(
            sys_,
            pos,
            H * n,
            settings=st,
            temperature=a.temperature_K,
            dt=dt,
            velocities=v,
            log=sys.stdout,
            thermostat=th,
            barostat=baro,
            mts=mts,
        )
    else:
        tpl = {id(m): RigidTemplate(m, pos[sys_.atom_slice(k)]) for k, m in enumerate(sys_.molecules)}
        sim = FlexibleSimulation(
            sys_,
            [tpl[id(m)] for m in sys_.molecules],
            pos,
            H * n,
            settings=st,
            temperature=a.temperature_K,
            dt=dt,
            log=sys.stdout,
            thermostat=th,
            barostat=baro,
            hmr=a.hmr_amu,
            mts=mts,
        )
        print("# masses of the first molecule:", np.asarray(sim.flex.masses)[:3], flush=True)
    blk = max(1, int(round(1.0 / dt)))  # 1 ps blocks
    sim.advance(blk)  # compile + warm up
    cg0, s0 = float(sim.state.cg_total), int(sim.state.step)
    t0 = time.time()
    done = 0
    while done < a.steps:
        sim.advance(blk)
        done += blk
    el = time.time() - t0
    o = sim.observables()
    print(
        f"{a.engine}: {sys_.nmol} waters ({sys_.n} atoms), dt {dt * 1000:g} fs, {a.precision}, {sim.ensemble}, "
        f"{st.describe_cutoffs()}, beta {beta:.4f}, PME {grid} order {a.order}, "
        f"{st.describe_induction()}, skin {a.skin}, rows {sim.ff.mc} (electrostatic {sim.ff.mc_e or sim.ff.mc}): "
        f"{el / done * 1e3:.3f} ms/step, "
        f"{done * dt / 1000 / el * 86400:.1f} ns/day; T {o['temp_K']:.1f} K, density {o['density_g_cm3']:.4f}, "
        f"CG iters {(float(sim.state.cg_total) - cg0) / (int(sim.state.step) - s0):.2f} mean per step, max "
        f"{o['cg_iter_max']}{'; ' + str(mts_stats(sim)) if mts else ''}",
        flush=True,
    )
    if a.time_ps > 0:
        sample(a, sim, sys_, mts)


def sample(a: argparse.Namespace, sim: Simulation | FlexibleSimulation, sys_: System, mts: object) -> None:
    """Sample --time-ps every 0.5 ps and print drift, <U>, temperatures, density (and write the O-O RDF)."""
    dt = sim.dt
    keys = KEYS
    n = max(1, int(round(0.5 / dt)))
    X = []
    edges = np.linspace(0.0, 0.8, 321)
    hist = np.zeros(len(edges) - 1)
    oxy = np.nonzero(np.array(sys_.elements) == "O")[0]

    @jax.jit
    def oo_hist(x, H):
        """Histogram of the O-O distances of x (n_O, 3) [nm] (minimum image in the reduced box H)."""
        d = x[:, None, :] - x[None, :, :]
        for c in (2, 1, 0):
            d = d - jnp.round(d[..., c] / H[c, c])[..., None] * H[c]
        r = jnp.sqrt(jnp.sum(d * d, -1))
        return jnp.histogram(r[jnp.triu_indices(len(x), 1)], bins=jnp.asarray(edges))[0]

    vol = []
    t1 = time.time()
    for _ in range(int(round(a.time_ps / 0.5))):
        sim.advance(n)
        o = sim.observables()
        X.append([o.get(k, np.nan) for k in keys])
        if a.rdf:
            box = jnp.asarray(sim.state.box)
            hist += np.asarray(oo_hist(jnp.asarray(sim.positions()[oxy]), box))
            vol.append(float(o["volume_nm3"]))
    X = np.array(X)
    nb = 10
    m = len(X) // nb * nb
    blocks = X[len(X) - m :].reshape(nb, -1, X.shape[1]).mean(1)
    err = blocks.std(0, ddof=1) / np.sqrt(nb)
    drift = np.polyfit(X[:, 0], X[:, 1], 1)[0] * 1000.0 / sim.integ.dof / (KB * a.temperature_K)
    mean = X.mean(0)
    print(
        f"sampled {a.time_ps:g} ps ({time.time() - t1:.0f} s): econs drift {drift:+.4f} kT/ns/dof, "
        f"<U> {mean[2]:.1f} +- "
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
