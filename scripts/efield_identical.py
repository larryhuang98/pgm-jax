"""Zero-field identity with the code in another directory (e.g. master): the same short runs (rigid and
flexible engines, NVT Bussi, mixed precision, 300 steps) of a small pGM box, run by the code in
--code (a checkout; default: this repository), positions and energies saved to --out (.npz).

    git archive master | tar -x -C /tmp/master
    python scripts/efield_identical.py --code /tmp/master --out m.npz
    python scripts/efield_identical.py --out b.npz
    python scripts/efield_identical.py --compare m.npz b.npz"""

import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--code", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out")
ap.add_argument("--compare", nargs=2)
a = ap.parse_args()
if a.compare:
    x, y = np.load(a.compare[0]), np.load(a.compare[1])
    for k in x.files:
        d = np.abs(x[k] - y[k]).max()
        print(f"{k:14s} max |difference| {d:.3e}  {'bitwise identical' if np.array_equal(x[k], y[k]) else ''}")
    sys.exit(0)
sys.path.insert(0, a.code)
sys.path.insert(0, os.path.join(a.code, "tests"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from test_md import small_box  # noqa: E402

from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402

sys_, pos, H = small_box(0, nm=0)
s = MDSettings(cutoff=0.6, skin=0.05, pme_grid=(32, 32, 32), dipole_tol=1e-5, precision="mixed")
out = {}
sim = Simulation(sys_, pos, H, s, dt=0.002, ensemble="nvt", thermostat="bussi", log=None, seed=7)
sim._advance(300)
out["rigid_pos"], out["rigid_epot"] = sim.positions_nm(), np.array(float(sim.state.epot))
tpl = [RigidTemplate(m, pos[sys_.atom_slice(k)]) for k, m in enumerate(sys_.molecules)]
sim = FlexibleSimulation(sys_, tpl, pos, H, s, dt=0.002, ensemble="nvt", thermostat="bussi", log=None, seed=7)
sim._advance(300)
out["flex_pos"], out["flex_epot"] = sim.positions_nm(), np.array(float(sim.state.epot))
np.savez(a.out, **out)
print("saved", a.out, {k: float(np.sum(v)) for k, v in out.items()})
