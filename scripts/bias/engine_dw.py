"""Validation 1b of pgm_jax.bias: an analytic free energy through the MD engine (docs/enhanced_sampling.md).

Two non-interacting atoms (no charge, no polarizability, no van der Waals) in a 3 nm box, an
external double-well potential U(r) on their distance r (a StaticBias), walls at 0.3 and 1.4 nm
(below half the box, so the minimum image sphere is complete): the exact FES of r is
F(r) = U(r) + walls - 2 kT ln r.  Well-tempered metadynamics (or OPES) on r with W independent
walkers in one program (bias/walkers.py) through the rigid-body engine (Simulation, Bussi or
Langevin), FES from the bias and by reweighting, compared with the exact F over F < --fmax-kJ.

Usage:

    python scripts/bias/engine_dw.py --method metad --time-ns 5 --walkers 16 --out runs/bias/engine_dw
    python scripts/bias/engine_dw.py --help

Inputs: none.
Outputs: <out>_wNN.colvar / .hills / logs of the walkers, <out>.json (RMSDs per walker and of the
average); printed summary.
Units: --time-ns ns, --dt-fs fs, --temperature-K K, --fmax-kJ kJ/mol; r in nm, F in kJ/mol.
Runtime: GPU or CPU, minutes.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import jax
import numpy as np

from pgm_jax import Molecule, System
from pgm_jax.bias import OPES, BiasSet, LowerWall, MetaD, StaticBias, UpperWall, cv
from pgm_jax.bias import analysis as A
from pgm_jax.bias.io import read_table
from pgm_jax.bias.walkers import Walkers
from pgm_jax.cli.args import add_dt_arg, add_seed_arg, add_temperature_arg, make_coupling, setup_logging
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.simulation import Simulation
from pgm_jax.units import KB

jax.config.update("jax_enable_x64", True)
h, r0, w = 15.0, 0.8, 0.25  # double well: barrier [kJ/mol], centre [nm], half distance of the minima [nm]
WALL_K = 2000.0  # wall force constant [kJ/mol/nm^2]


def U(s: object) -> object:
    """Return the double-well potential h ((r - r0)^2 / w^2 - 1)^2 [kJ/mol] of s = (r,) [nm]."""
    return h * (((s[0] - r0) / w) ** 2 - 1.0) ** 2


def wall_e(x: np.ndarray) -> np.ndarray:
    """Return the energy of the lower (0.3 nm) and upper (1.4 nm) walls at r = x [kJ/mol]."""
    return WALL_K * (np.maximum(0.3 - x, 0) ** 2 + np.maximum(x - 1.4, 0) ** 2)


def build(a: argparse.Namespace) -> tuple[Simulation, object]:
    """Return the two-atom simulation with the bias set, and the adaptive bias (MetaD or OPES) on r."""
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
    bs = BiasSet([b, StaticBias(r, U, name="model"), LowerWall(r, 0.3, WALL_K), UpperWall(r, 1.4, WALL_K)], colvar=250)
    s = MDSettings().replace(
        precision="double", elec="q", vdw="none", cutoff=1.2, skin=0.1, pme_grid=(8, 8, 8), lj_lrc=False
    )
    sim = Simulation(
        sys_,
        pos,
        H,
        s,
        dt=a.dt_fs / 1000,
        temperature=a.temperature_K,
        thermostat=make_coupling(a.thermostat, friction=1.0, tau=0.5)[0],
        bias=bs,
        seed=a.seed,
        log=sys.stdout,
    )
    return sim, b


def analyse(a: argparse.Namespace, wk: Walkers, b: object, n: int, wall: float) -> dict:
    """Compare the FES of every walker (from the bias and reweighted) with the exact F(r); return the results."""
    kT = KB * a.temperature_K
    ax = np.linspace(0.36, 1.34, 50)
    Fex = np.array([U([x]) for x in ax]) + wall_e(ax) - 2 * kT * np.log(ax)
    Fex -= Fex.min()
    m = Fex < a.fmax_kJ
    res = {"args": vars(a), "wall_s": wall, "ns_per_day_aggregate": a.walkers * a.time_ns / wall * 86400, "runs": []}
    FB, FR = [], []
    for k in range(a.walkers):
        _, c = read_table(f"{a.out}_w{k:02d}.colvar")
        st = wk.bias_state(k).parts[0]
        Fb = A.fes_from_bias(b, st, ax[:, None])
        keep = c["step"] > a.skip_fraction * n
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
        res[name] = {
            "rmsd_mean": float(R.mean()),
            "rmsd_sem": float(R.std(ddof=1) / np.sqrt(len(R))),
            "rmsd_of_average": rm,
            "max_of_average": mm,
            "mean_err": float(e[m].mean()),
            "chi2_per_point": float(np.mean(((Fm - Fex)[m] / e[m]) ** 2)),
        }
        print(
            f"{name}: RMSD per walker {R.mean():.3f} +- {res[name]['rmsd_sem']:.3f}; average of {len(Fs)}: "
            f"RMSD {rm:.3f}, max {mm:.2f}, mean error bar {e[m].mean():.3f}, "
            f"chi2/point {res[name]['chi2_per_point']:.2f}"
        )
    return res


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run the walkers and compare with the exact FES (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="metad", choices=("metad", "opes"), help="adaptive bias")
    ap.add_argument("--time-ns", type=float, default=5.0, help="length per walker [ns]")
    ap.add_argument("--walkers", type=int, default=16, help="independent walkers in one program")
    add_dt_arg(ap, 2.0)
    add_temperature_arg(ap, 300.0)
    ap.add_argument(
        "--thermostat", default="langevin", choices=("langevin", "bussi"), help="langevin (1/ps) or bussi (0.5 ps)"
    )
    ap.add_argument("--fmax-kJ", type=float, default=20.0, help="region of the comparison: F < fmax [kJ/mol]")
    ap.add_argument("--skip-fraction", type=float, default=0.2, help="fraction of each run discarded (reweighting)")
    add_seed_arg(ap, 5)
    ap.add_argument("-o", "--out", default="runs/bias/engine_dw", help="output prefix")
    a = ap.parse_args(argv)
    setup_logging()
    sim, b = build(a)
    wk = Walkers(sim, a.walkers, seed=a.seed, log=sys.stdout)
    n = int(round(a.time_ns * 1000 / (a.dt_fs / 1000)))
    t0 = time.time()
    wk.run(n, report_every=n // 20, prefix=a.out)
    wall = time.time() - t0
    res = analyse(a, wk, b, n, wall)
    print(f"{a.walkers} walkers x {a.time_ns} ns in {wall:.0f} s ({res['ns_per_day_aggregate']:.0f} ns/day aggregate)")
    with open(a.out + ".json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
