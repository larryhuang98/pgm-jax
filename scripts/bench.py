"""Timing of the core evaluations on the current device (run on CPU or GPU nodes).

python scripts/bench.py            # prints device, compile time and steady-state time per call
"""

from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate_amber import RST, TOP, read_restart  # noqa: E402

from pgm_jax.channels import ElecChannel  # noqa: E402
from pgm_jax.ewald import PeriodicPGM  # noqa: E402
from pgm_jax.md.box import box_from_cell  # noqa: E402
from pgm_jax.model import Model  # noqa: E402
from pgm_jax.param import read_prmtop_pgm  # noqa: E402
from pgm_jax.periodic import PeriodicModel  # noqa: E402
from pgm_jax.system import System  # noqa: E402


def random_clusters(mono, nmol, nconf, seed=0):
    """nconf rigid clusters of nmol copies of `mono` (3, 3) nm: random orientations, centres
    0.28-0.40 nm from the previous molecule (same sizes as the dimer/trimer sets in evoff)."""
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


def timed(f, n=3):
    t0 = time.time()
    jax.block_until_ready(f())
    t_first = time.time() - t0
    ts = []
    for _ in range(n):
        t0 = time.time()
        jax.block_until_ready(f())
        ts.append(time.time() - t0)
    return t_first, min(ts)


def main():
    print("device:", jax.devices())
    w = read_prmtop_pgm(TOP)[0]
    xyz, (L, ang) = read_restart(RST)
    H = box_from_cell(L, ang) * 0.1
    pos = xyz * 0.1
    sys512 = System([w] * 512)
    per = PeriodicPGM(sys512, H, pos, b0=3.8, rc=1.0)
    e_fn = jax.jit(lambda x: per.energy(x)[0]["total"])
    f_fn = jax.jit(per.forces)
    print("periodic 512 waters, energy:        first {:.1f}s, then {:.3f}s".format(*timed(lambda: e_fn(pos))))
    print("periodic 512 waters, forces:        first {:.1f}s, then {:.3f}s".format(*timed(lambda: f_fn(pos))))
    pm = PeriodicModel(sys512, H, pos, rc=1.0, b0=3.8)
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
