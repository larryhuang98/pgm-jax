"""Time the core evaluations (reference Ewald model, PeriodicModel, gas-phase n-body) on the current device.

The 512-water pGM3P-25 box: energy and forces of PeriodicPGM, forces, parameter and strain
derivatives and a second-order derivative of PeriodicModel (Ewald beta 3.8 / nm, 1.0 nm cutoff);
the interaction energies of 4,000 random water dimers and the 3-body energies of 45,316 random
trimers with the gas-phase Model.  Each line: time of the first call (compilation) and the best
of the following calls.

Usage:

    python scripts/benchmarks/bench.py
    python scripts/benchmarks/bench.py --help

Inputs: PGM_GVDW_DATA (pgm_jax.paths).
Outputs: printed timings [s].
Units: s.
Runtime: minutes (CPU or GPU).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.channels import ElecChannel
from pgm_jax.ewald import PeriodicPGM
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.io import read_coordinates
from pgm_jax.model import Model
from pgm_jax.param import read_prmtop_pgm
from pgm_jax.paths import pgm3p25_files
from pgm_jax.periodic import PeriodicModel
from pgm_jax.system import System

jax.config.update("jax_enable_x64", True)
TOP, RST = pgm3p25_files()


def random_clusters(mono: np.ndarray, nmol: int, nconf: int, seed: int = 0) -> np.ndarray:
    """Return nconf rigid clusters (nconf, 3 nmol, 3) [nm] of nmol copies of the water `mono` (3, 3) [nm].

    Random orientations, centres 0.28-0.40 nm from the previous molecule (same sizes as the
    dimer / trimer sets in evoff).
    """
    rng = np.random.default_rng(seed)
    mono = mono - mono.mean(0)
    out = np.empty((nconf, 3 * nmol, 3))
    for c in range(nconf):
        centre = np.zeros(3)
        for m in range(nmol):
            Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
            out[c, 3 * m : 3 * m + 3] = mono @ Q.T + centre
            v = rng.normal(size=3)
            centre = centre + v / np.linalg.norm(v) * rng.uniform(0.28, 0.40)
    return out


def timed(f: Callable, n: int = 3) -> tuple[float, float]:
    """Return the time of the first call of f and the best of the next n calls [s] (results awaited)."""
    t0 = time.time()
    jax.block_until_ready(f())
    t_first = time.time() - t0
    ts = []
    for _ in range(n):
        t0 = time.time()
        jax.block_until_ready(f())
        ts.append(time.time() - t0)
    return t_first, min(ts)


def main(argv: list[str] | None = None) -> None:
    """Parse the (empty) command line, time the evaluations and print them (see the module docstring)."""
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    print("device:", jax.devices())
    w = read_prmtop_pgm(TOP)[0]
    xyz, _, (L, ang) = read_coordinates(RST)
    H = box_from_cell(L, ang) * 0.1
    pos = xyz * 0.1
    sys512 = System([w] * 512)
    per = PeriodicPGM(sys512, H, pos, ewald_beta=3.8, cutoff=1.0)
    e_fn = jax.jit(lambda x: per.energy(x)[0]["total"])
    f_fn = jax.jit(per.forces)
    print("periodic 512 waters, energy:        first {:.1f}s, then {:.3f}s".format(*timed(lambda: e_fn(pos))))
    print("periodic 512 waters, forces:        first {:.1f}s, then {:.3f}s".format(*timed(lambda: f_fn(pos))))
    pm = PeriodicModel(sys512, H, pos, cutoff=1.0, ewald_beta=3.8)
    P = sys512.params0
    fm = jax.jit(pm.forces)
    gp = jax.jit(jax.grad(lambda p: pm.energy(pos, p)["total"]))
    sd = jax.jit(pm.strain_derivative)
    wts = np.random.default_rng(0).normal(size=pos.shape)
    g2 = jax.jit(jax.grad(lambda p: jnp.sum(pm.forces(pos, p) * wts)))
    print("  + LJ, forces:                      first {:.1f}s, then {:.3f}s".format(*timed(lambda: fm(pos, P))))
    print("  dE/dparams:                        first {:.1f}s, then {:.3f}s".format(*timed(lambda: gp(P))))
    print("  dE/dstrain (virial):               first {:.1f}s, then {:.3f}s".format(*timed(lambda: sd(pos, P))))
    print("  d(forces.w)/dparams (2nd order):   first {:.1f}s, then {:.3f}s".format(*timed(lambda: g2(P))))
    model = Model([lambda s: ElecChannel()])
    mono = pos[:3]
    c2, c3 = random_clusters(mono, 2, 4000), random_clusters(mono, 3, 45316, seed=1)
    s2, s3 = System([w] * 2), System([w] * 3)
    print(
        "4000 water dimers, interaction E:    first {:.1f}s, then {:.3f}s".format(
            *timed(lambda: model.nbody(s2, c2)["int"]["total"], 1)
        )
    )
    print(
        "45316 water trimers, 3-body E:       first {:.1f}s, then {:.3f}s".format(
            *timed(lambda: model.nbody(s3, c3, order=3)["nb3"]["total"], 1)
        )
    )


if __name__ == "__main__":
    main()
