"""NVE checks of the bias hook (both engines): eight pGM waters in a 3 nm box (no pair crosses the
1.2 nm cutoff, so the unbiased NVE energy is conserved to the integration error), float64.
  static   a metadynamics bias with 40 pre-deposited hills on (O-O distance, O...H coordination number),
           never updated, plus an upper wall: E_tot fluctuation vs the bias energy moving in and out;
  none     the same without bias (the integrator's own error);
  growing  hills (1 kJ/mol) deposited every 200 steps inside the compiled loop: econs = E_tot - work of the
           updates (booked as heat) against the work done.
Prints RMS / max deviations for dt = 1.0 and 0.5 fs (a Verlet error falls 4x at half the step).

    python scripts/bias/nve_check.py --ps 2 --out runs/bias/nve.json"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax import System
from pgm_jax.bias import BiasSet, MetaD, UpperWall, cv
from pgm_jax.cli.args import setup_logging
from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.simulation import Simulation
from pgm_jax.models.toy import water, water_cluster_box

# a singular gradient when H-O...O is collinear
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--ps", type=float, default=2.0)
ap.add_argument("--out", default="runs/bias/nve.json")
a = ap.parse_args()
setup_logging()
pos, H, w = water_cluster_box()
wat = water()
sys_ = System([wat] * (len(pos) // 3))
s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
d = cv.Distance(0, 9)
hyd = [i for i in range(len(pos)) if i % 3 and i // 3 != 0]
phi = cv.Coordination([0], hyd, r0=0.35, name="n_OH")  # smooth; a dihedral through two molecules has
out = {}
for engine in ("rigid", "atoms"):
    for mode in ("none", "static", "growing"):
        for dt in (0.001, 0.0005):
            m = MetaD([d, phi], sigma=[0.03, 0.1], height=1.0, pace=200, biasfactor=5.0, temperature=300.0)
            bs = BiasSet([m, UpperWall(d, 0.6, 1000.0)], colvar=0)
            kw = dict(dt=dt, thermostat=None, bias=None if mode == "none" else bs, log=None, temperature=300.0, seed=3)
            sim = (
                Simulation(sys_, pos, H, s, **kw)
                if engine == "rigid"
                else FlexibleSimulation(sys_, [RigidTemplate(wat, w)] * sys_.nmol, pos, H, s, **kw)
            )
            if mode == "static":
                st = m.init()
                rng = np.random.default_rng(0)
                s0 = np.array([float(d(pos, H)), float(phi(pos, H))])
                for k in range(40):
                    st = m.update(st, jnp.asarray(s0 + [0.04 * rng.normal(), 0.2 * rng.normal()]), k)
                m.pace = 0
                sim.set_bias_state(bs.init()._replace(parts=(st, ())))
            n = int(round(a.ps / dt))
            blk = int(round(0.02 / dt))
            E, C, B, Wk = [], [], [], []
            for _ in range(n // blk):
                sim.advance(blk)
                o = sim.observables()
                E.append(o["etot"])
                C.append(o["econs"])
                B.append(o.get("ebias", 0.0))
                Wk.append(o.get("bias_work", 0.0))
            E, C, B = np.array(E), np.array(C), np.array(B)
            key = f"{engine}_{mode}_dt{dt * 1000:g}fs"
            out[key] = {
                "etot_rms": float(E.std()),
                "etot_maxdev": float(np.abs(E - E[0]).max()),
                "econs_rms": float(C.std()),
                "econs_maxdev": float(np.abs(C - C[0]).max()),
                "ebias_range": float(B.max() - B.min()),
                "work": float(Wk[-1]),
                "hills": int(o.get("hills", 0)),
            }
            print(key, out[key], flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
