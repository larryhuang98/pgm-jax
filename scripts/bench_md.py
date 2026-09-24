"""MD speed benchmark: pGM3P-25 water, optionally replicated n x n x n (the truncated-octahedron
lattice tiles by lattice vectors), NPT with production-like settings.

    python scripts/bench_md.py --replicate 2 --steps 10000 --precision mixed
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.io import box_from_cell, read_coordinates  # noqa: E402
from pgm_jax.md.simulation import Simulation, _dedupe  # noqa: E402
from pgm_jax.param import read_prmtop_pgm  # noqa: E402
from pgm_jax.system import System  # noqa: E402

TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
RST = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--ensemble", default="npt")
    ap.add_argument("--cut", type=float, default=0.8)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--lrc", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--grid", type=int, default=48, help="PME points per replica along each lattice vector")
    ap.add_argument("--skin", type=float, default=0.1)
    a = ap.parse_args()
    mols = _dedupe(read_prmtop_pgm(TOP, first_residue_only=False))
    xyz, vel, box = read_coordinates(RST)
    H = box_from_cell(*box) * 0.1
    n = a.replicate
    shifts = [i * H[0] + j * H[1] + k * H[2] for i in range(n) for j in range(n) for k in range(n)]
    pos = np.concatenate([xyz * 0.1 + s for s in shifts])
    v = np.concatenate([vel * 0.1] * len(shifts))
    sys_ = System(mols * len(shifts))
    grid = tuple(a.grid * n for _ in range(3))
    st = MDSettings(cutoff=a.cut, skin=a.skin, ewald_beta=4.0, pme_grid=grid, pme_order=a.order, lj_lrc=bool(a.lrc),
                    dipole_tol=a.tol, precision=a.precision)
    sim = Simulation(sys_, pos, H * n, settings=st, ensemble=a.ensemble, temperature=298.0, gamma=a.gamma,
                     barostat_interval=100, dt=a.dt, vel_nm_ps=v, log=sys.stdout)
    sim._advance(1000)                                          # compile + warm up
    t0 = time.time()
    done = 0
    while done < a.steps:
        sim._advance(1000)
        done += 1000
    el = time.time() - t0
    o = sim.observables()
    print(f"{sys_.nmol} waters ({sys_.n} atoms), {a.precision}, {a.ensemble}, cut {a.cut} nm, PME {grid} order {a.order}, "
          f"tol {a.tol:g}, skin {a.skin}, rows {sim.ff.mc}: {el / done * 1e3:.3f} ms/step, "
          f"{done * a.dt / 1000 / el * 86400:.1f} ns/day; T {o['temp_K']:.1f} K, density {o['density_g_cm3']:.4f}, "
          f"CG iters {o['cg_iter']}, max {o['cg_iter_max']}", flush=True)


if __name__ == "__main__":
    main()
