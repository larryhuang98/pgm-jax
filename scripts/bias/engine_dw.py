"""Validation 1b: an analytic free energy through the MD engine.  Two non-interacting atoms (no
charge, no polarizability, no van der Waals) in a 3 nm box, an external double-well potential U(r)
on their distance r (a StaticBias), walls at 0.3 and 1.4 nm (below half the box, so the minimum
image sphere is complete): the exact FES of r is F(r) = U(r) + walls - 2 kT ln r.  Well-tempered
metadynamics (or OPES) on r with W independent walkers in one program (bias/walkers.py) through
the rigid-body engine (Simulation, Bussi or Langevin), FES from the bias and by reweighting,
compared with the exact F over F < --fmax.

    python scripts/bias/engine_dw.py --method metad --ns 5 --walkers 16 --out runs/bias/engine_dw"""
import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pgm_jax import Molecule, System  # noqa: E402
from pgm_jax.bias import OPES, BiasSet, LowerWall, MetaD, StaticBias, UpperWall, cv  # noqa: E402
from pgm_jax.bias import analysis as A  # noqa: E402
from pgm_jax.bias.core import KB  # noqa: E402
from pgm_jax.bias.io import read_table  # noqa: E402
from pgm_jax.bias.walkers import Walkers  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="metad", choices=("metad", "opes"))
ap.add_argument("--ns", type=float, default=5.0)
ap.add_argument("--walkers", type=int, default=16)
ap.add_argument("--dt", type=float, default=0.002)
ap.add_argument("--T", type=float, default=300.0)
ap.add_argument("--thermostat", default="langevin")
ap.add_argument("--fmax", type=float, default=20.0)
ap.add_argument("--skip", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--out", default="runs/bias/engine_dw")
a = ap.parse_args()
kT = KB * a.T
h, r0, w = 15.0, 0.8, 0.25


def U(s):
    return h * (((s[0] - r0) / w) ** 2 - 1.0) ** 2


atom = Molecule("AR", ["Ar"], ["Ar"], np.zeros(1), np.full(1, 0.1), np.zeros(1), masses=[40.0])
sys_ = System([atom, atom])
L = 3.0
pos = np.array([[1.0, 1.5, 1.5], [1.55, 1.5, 1.5]])
H = np.eye(3) * L
r = cv.Distance(0, 1, name="r")
if a.method == "metad":
    b = MetaD(r, sigma=0.03, height=1.0, pace=500, biasfactor=8.0, grid=(0.0, 1.6, 320))
else:
    b = OPES(r, sigma=0.03, pace=500, barrier=25.0)
bs = BiasSet([b, StaticBias(r, U, name="model"), LowerWall(r, 0.3, 2000.0), UpperWall(r, 1.4, 2000.0)], colvar=250)
s = MDSettings(precision="double", elec="q", vdw="none", cutoff=1.2, skin=0.1, pme_grid=(8, 8, 8), lj_lrc=False)
sim = Simulation(sys_, pos, H, s, dt=a.dt, temperature=a.T, thermostat=a.thermostat, gamma=1.0, tau_t=0.5, bias=bs,
                 seed=a.seed, log=sys.stdout)
wk = Walkers(sim, a.walkers, seed=a.seed)
n = int(round(a.ns * 1000 / a.dt))
t0 = time.time()
wk.run(n, report=n // 20, prefix=a.out)
wall = time.time() - t0
ax = np.linspace(0.36, 1.34, 50)
wall_e = lambda x: 2000.0 * (np.maximum(0.3 - x, 0) ** 2 + np.maximum(x - 1.4, 0) ** 2)     # noqa: E731
Fex = np.array([U([x]) for x in ax]) + wall_e(ax) - 2 * kT * np.log(ax)
Fex -= Fex.min()
m = Fex < a.fmax
res = {"args": vars(a), "wall_s": wall, "ns_per_day_aggregate": a.walkers * a.ns / wall * 86400, "runs": []}
FB, FR = [], []
for k in range(a.walkers):
    _, c = read_table(f"{a.out}_w{k:02d}.colvar")
    st = wk.bias_state(k).parts[0]
    Fb = A.fes_from_bias(b, st, ax[:, None])
    keep = c["step"] > a.skip * n
    if a.method == "metad":
        hs, ct = A.metad_ct(b.hills(st), 8.0, kT, [0.0], np.linspace(0.2, 1.6, 700)[:, None])
        lw = A.ct_weights(c["step"], c["bias0_metad"], hs, ct, kT)
    else:
        lw = A.opes_weights(c["bias0_opes"], kT)
    Fh = A.histogram_fes(c["r"][keep], lw[keep], [ax], kT)
    rb, rh = A.align_rmsd(Fb, Fex, m)[0], A.align_rmsd(Fh, Fex, m)[0]
    res["runs"].append({"rmsd_bias": rb, "rmsd_reweight": rh})
    FB.append(A.align_rmsd(Fb, Fex, m)[2])
    FR.append(A.align_rmsd(Fh, Fex, m)[2])
    print(f"walker {k}: RMSD from the bias {rb:.3f}, reweighted {rh:.3f} kJ/mol")
for name, Fs in (("bias", FB), ("reweight", FR)):
    Fs = np.array(Fs)
    R = np.array([A.align_rmsd(f, Fex, m)[0] for f in Fs])
    Fm, e = Fs.mean(0), Fs.std(0, ddof=1) / np.sqrt(len(Fs))
    rm, mm, _ = A.align_rmsd(Fm, Fex, m)
    res[name] = {"rmsd_mean": float(R.mean()), "rmsd_sem": float(R.std(ddof=1) / np.sqrt(len(R))),
                 "rmsd_of_average": rm, "max_of_average": mm, "mean_err": float(e[m].mean()),
                 "chi2_per_point": float(np.mean(((Fm - Fex)[m] / e[m]) ** 2))}
    print(f"{name}: RMSD per walker {R.mean():.3f} +- {res[name]['rmsd_sem']:.3f}; average of {len(Fs)}: RMSD {rm:.3f}, "
          f"max {mm:.2f}, mean error bar {e[m].mean():.3f}, chi2/point {res[name]['chi2_per_point']:.2f}")
print(f"{a.walkers} walkers x {a.ns} ns in {wall:.0f} s ({res['ns_per_day_aggregate']:.0f} ns/day aggregate)")
json.dump(res, open(a.out + ".json", "w"), indent=1)
