"""Cost of one force evaluation (neighbour list excluded) for the induced-dipole schemes on the pGM
water box (512 waters x replicate^3): SCF with a fixed number of CG iterations (iEL/SCF-k, k = 0..6)
and iEL/0-SCF, per call, on the current device.

    python scripts/iel_cost.py --replicate 2"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.md.box import box_from_cell
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.neighbors import AtomNeighbors
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import resource
from pgm_jax.system import System

jax.config.update("jax_enable_x64", True)
TOP = resource("gvdw_data", "topology/rayl_512_v2.prmtop")
RST = resource("gvdw_data", "inputs/lj/inpcrd.restrt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    mols = read_prmtop_molecules(TOP)
    xyz, vel, box = read_coordinates(RST)
    H = box_from_cell(*box) * 0.1
    n = a.replicate
    shifts = [i * H[0] + j * H[1] + k * H[2] for i in range(n) for j in range(n) for k in range(n)]
    pos = jnp.asarray(np.concatenate([xyz * 0.1 + s for s in shifts]))
    H = jnp.asarray(H * n)
    sys_ = System(mols * len(shifts))
    base = MDSettings().replace(
        cutoff=0.9,
        skin=0.1,
        ewald_beta=4.0,
        pme_grid=(48 * n,) * 3,
        pme_order=6,
        lj_lrc=True,
        dipole_tol=1e-5,
        precision="mixed",
    )
    idx = AtomNeighbors(sys_.n, H, 1.0, 0.0).allocate(pos, None, H).idx
    ff0 = PGMForceField(sys_, H, base)
    ff0.size_rows(pos, H, idx)
    print(f"# {sys_.nmol} waters, rows {ff0.mc}, device {jax.devices()[0]}", flush=True)
    cases = [(f"scf-{k}", base.replace(iel="scf", iel_iter=k)) for k in (1, 2, 3, 4, 6)]
    cases += [("0scf", base.replace(iel="0scf")), ("mu4 tol 1e-5", base)]
    for name, s in cases:
        ff = PGMForceField(sys_, H, s)
        ff.mc = ff0.mc
        f = jax.jit(ff.compute)
        ind = ff.init_induction()
        for _ in range(8):  # past the warm-up / predictor start
            r = f(pos, H, idx, ind)
            ind = r.induction
        jax.block_until_ready(r.forces)
        t = time.perf_counter()
        for _ in range(a.reps):
            r = f(pos, H, idx, ind)
        jax.block_until_ready(r.forces)
        dt = (time.perf_counter() - t) / a.reps
        print(f"{name:14s} {dt * 1e3:8.3f} ms/call  iterations {int(r.iterations)}", flush=True)


if __name__ == "__main__":
    main()
