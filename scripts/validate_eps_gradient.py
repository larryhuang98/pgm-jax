"""Independent-simulation check of the ensemble gradients (pgm_jax/fit): batched NVT replicas of a
small pGM water box at theta, frames analysed by FrameAnalyzer, saved in the format of
scripts/fit_multi.py (prefix_frames*.npz, with a replica index), so that
`scripts/liquid_fit_tools.py fd` compares finite differences between runs at theta -/+ delta with
the fluctuation-formula gradients.  On CPUs the MD step of a small box is overhead bound, so R
replicas advanced together by jax.vmap (md/remd.MDReplicas, all at T) give ~R times the sampling.

    python scripts/validate_eps_gradient.py -o runs/fit/v_m --coords runs/fit/base64.rst7 --cutoff 0.45 \
        --skin 0.08 --params q --start=-0.1 --nrep 8 --segments 20 --seg-ps 100
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from fit_multi import add_arguments, setup  # noqa: E402

from pgm_jax.fit import FrameAnalyzer  # noqa: E402
from pgm_jax.md.forcefield import MDSettings, elec_cutoff_settings  # noqa: E402
from pgm_jax.md.remd import MDReplicas  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    ap.add_argument("--nrep", type=int, default=8)
    ap.add_argument("--segments", type=int, default=10)
    ap.add_argument("--seg-ps", type=float, default=100.0)
    a = ap.parse_args()
    S = setup(a)
    ew = {k: v for k, v in elec_cutoff_settings(a.cutoff).items() if k != "elec_cutoff"}
    if a.ewald_beta:
        ew["ewald_beta"] = a.ewald_beta
    st = MDSettings(
        cutoff=a.cutoff,
        skin=a.skin,
        pme_order=6,
        lj_lrc=True,
        dipole_tol=a.md_tol,
        precision=a.precision,
        pme_grid=(a.nfft,) * 3 if a.nfft else None,
        **ew,
    )
    th = S["theta0"]
    sim = Simulation(
        S["sys"],
        S["pos"],
        S["H"],
        st,
        dt=a.dt / 1000.0,
        ensemble="nvt",
        temperature=a.T,
        thermostat="bussi",
        tau_t=1.0,
        seed=a.seed,
        params=S["space"](jnp.asarray(th)),
        log=None,
    )
    st = dataclasses.replace(st, pme_grid=tuple(int(k) for k in sim.ff.pme.K))
    an = FrameAnalyzer(S["sys"], np.asarray(sim.state.box), st, S["space"], rdf=S["rdf"], tol=a.tol, chunk=a.nrep)
    every = max(1, int(round(a.every / sim.dt)))
    done = sorted(glob.glob(a.out + "_frames*.npz"))
    t0 = time.time()
    sim._advance(int(round(a.equil / sim.dt)))
    rep = MDReplicas(sim, a.T + 1e-6 * np.arange(a.nrep), batched=True, seed=a.seed + 7)
    rep.advance(int(round(a.equil_rep / sim.dt)))
    print(
        f"# {S['sys'].nmol} molecules, NVT, {a.nrep} replicas, theta {th.tolist()}, V "
        f"{float(np.abs(np.linalg.det(np.asarray(sim.state.box)))):.4f} nm^3; "
        f"equilibrated in {time.time() - t0:.0f} s",
        flush=True,
    )
    nper = int(round(a.seg_ps / a.every))
    for seg in range(len(done), a.segments):
        t1 = time.time()
        out, ta = [], 0.0
        for _i in range(nper):
            rep.advance(every)
            P = rep._positions(rep.S.dyn.position)
            frames = [(P[k], rep.S.box[k], rep.S.induction.mu[k]) for k in range(a.nrep)]
            ta0 = time.time()
            out.append(an.analyze(jnp.asarray(th), frames))
            ta += time.time() - ta0
        fr = {k: np.stack([o[k] for o in out], axis=1) for k in out[0]}  # (rep, time, ...)
        fr = {k: v.reshape((-1,) + v.shape[2:]) for k, v in fr.items()}
        fr["rep"] = np.repeat(np.arange(a.nrep), nper)
        np.savez(f"{a.out}_frames{seg:03d}.npz", theta=th, **fr)
        print(
            f"segment {seg}: {nper} x {a.nrep} frames, U/N {fr['U'].mean() / S['sys'].nmol:.3f} kJ/mol, "
            f"D {fr['D'].mean() / 0.020819434:.4f} D, MD+analysis {time.time() - t1:.0f} s (analysis {ta:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
