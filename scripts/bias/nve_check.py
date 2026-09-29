"""NVE checks of the bias hook of both engines (docs/enhanced_sampling.md).

Eight pGM waters in a 3 nm box (no pair crosses the 1.2 nm cutoff, so the unbiased NVE energy is
conserved to the integration error), float64.  Modes:
  static   a metadynamics bias with 40 pre-deposited hills on (O-O distance, O...H coordination number),
           never updated, plus an upper wall: E_tot fluctuation vs the bias energy moving in and out;
  none     the same without bias (the integrator's own error);
  growing  hills (1 kJ/mol) deposited every 200 steps inside the compiled loop: econs = E_tot - work of the
           updates (booked as heat) against the work done.
Prints RMS / max deviations for dt = 1.0 and 0.5 fs (a Verlet error falls 4x at half the step),
for the rigid-body engine and the atomic (constrained flexible) engine.

Usage:

    python scripts/bias/nve_check.py --time-ps 2 --out runs/bias/nve.json
    python scripts/bias/nve_check.py --help

Inputs: none (pgm_jax.models.toy water cluster).
Outputs: the JSON (--out) and a printed line per case.
Units: --time-ps ps; energies kJ/mol.
Runtime: CPU, minutes.  Sets jax_enable_x64.
"""

from __future__ import annotations

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

jax.config.update("jax_enable_x64", True)


def run_case(engine: str, mode: str, dt: float, ps: float) -> dict:
    """Run one NVE case and return the fluctuations of E_tot and econs, the bias energy range and the work.

    Parameters
    ----------
    engine : {"rigid", "atoms"}
        Rigid-body engine or the flexible engine with rigid templates.
    mode : {"none", "static", "growing"}
        See the module docstring.
    dt : float
        Time step [ps].
    ps : float
        Run length [ps] (samples every 0.02 ps).

    Returns
    -------
    dict
        etot_rms, etot_maxdev, econs_rms, econs_maxdev, ebias_range, work [kJ/mol], hills.
    """
    pos, H, w = water_cluster_box()
    wat = water()
    sys_ = System([wat] * (len(pos) // 3))
    s = MDSettings().replace(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
    d = cv.Distance(0, 9)
    hyd = [i for i in range(len(pos)) if i % 3 and i // 3 != 0]
    # smooth; a dihedral through two molecules has a singular gradient when H-O...O is collinear
    phi = cv.Coordination([0], hyd, r0=0.35, name="n_OH")
    m = MetaD([d, phi], sigma=[0.03, 0.1], height=1.0, pace=200, biasfactor=5.0, temperature=300.0)
    bs = BiasSet([m, UpperWall(d, 0.6, 1000.0)], colvar=0)
    kw = dict(dt=dt, thermostat=None, bias=None if mode == "none" else bs, log=None, temperature=300.0, seed=3)
    sim = (
        Simulation(sys_, pos, H, s, **kw)
        if engine == "rigid"
        else FlexibleSimulation(sys_, [RigidTemplate(wat, w)] * sys_.nmol, pos, H, s, **kw)
    )
    if mode == "static":  # 40 hills around the start, never updated
        st = m.init()
        rng = np.random.default_rng(0)
        s0 = np.array([float(d(pos, H)), float(phi(pos, H))])
        for k in range(40):
            st = m.update(st, jnp.asarray(s0 + [0.04 * rng.normal(), 0.2 * rng.normal()]), k)
        m.pace = 0
        sim.set_bias_state(bs.init()._replace(parts=(st, ())))
    n = int(round(ps / dt))
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
    return {
        "etot_rms": float(E.std()),
        "etot_maxdev": float(np.abs(E - E[0]).max()),
        "econs_rms": float(C.std()),
        "econs_maxdev": float(np.abs(C - C[0]).max()),
        "ebias_range": float(B.max() - B.min()),
        "work": float(Wk[-1]),
        "hills": int(o.get("hills", 0)),
    }


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run every case and write the JSON (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-ps", type=float, default=2.0, help="run length per case [ps]")
    ap.add_argument("-o", "--out", default="runs/bias/nve.json", help="JSON output")
    a = ap.parse_args(argv)
    setup_logging()
    out = {}
    for engine in ("rigid", "atoms"):
        for mode in ("none", "static", "growing"):
            for dt in (0.001, 0.0005):
                key = f"{engine}_{mode}_dt{dt * 1000:g}fs"
                out[key] = run_case(engine, mode, dt, a.time_ps)
                print(key, out[key], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
