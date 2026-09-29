"""Fit pGM water (or any box of one rigid molecule type from a pGM prmtop) to several liquid and
gas-phase targets with ensemble gradients: density, heat of vaporization, static dielectric
constant, gas-phase dipole and polarizability, O-O g(r); parameter uncertainties (pgm_jax/fit,
docs/liquid_fit.md).

    # the owner's goal: the base pGM water toward experiment (eps included)
    T=density=0.997:0.002,hvap=10.52:0.05,eps=78.4:1.5,gas_dipole=1.855:0.01
    T=$T,gas_polarizability=1.47:0.01,liquid_dipole
    python scripts/fit_multi.py -o runs/fit/demo --params q,cov,alpha,radius,lj_r,lj_eps --targets $T \\
        --equil 50 --prod 2000 --iters 6
    # measure only (all targets without values): observables, Jacobians, errors at --start
    python scripts/fit_multi.py -o runs/fit/s0 --params q --targets density,hvap,eps,liquid_dipole --iters 1

Targets: name=value:sigma[:weight], or a bare name (evaluated, propagated, not fitted); rdf=FILE:sigma:weight
with FILE a .npy g(r) on the analysis grid or a fit JSON whose last record has the rdf components.
A run stops after --max-minutes (GPU jobs of < 45 min) and continues with the same -o."""

from __future__ import annotations

import argparse
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from pgm_jax.fit import GasPhase, Objective, Param, ParameterSpace, RDFSpec, Target  # noqa: E402
from pgm_jax.fit.liquid import LiquidFit  # noqa: E402
from pgm_jax.md.forcefield import MDSettings, elec_cutoff_settings  # noqa: E402
from pgm_jax.md.io import box_from_cell, read_coordinates  # noqa: E402
from pgm_jax.md.simulation import _dedupe  # noqa: E402
from pgm_jax.param import read_prmtop_pgm  # noqa: E402
from pgm_jax.system import System  # noqa: E402

MODELS = {
    "base": ("~/project/epsp/base/base_512.prmtop", "~/project/epsp/base/base_512.rst7"),
    "p25": ("~/project/epsp/p25_512.prmtop", "~/project/epsp/p25_512.rst7"),
}


def parse_targets(text, rdf_r):
    out = []
    for item in [s for s in text.split(",") if s]:
        if "=" not in item:
            out.append(Target(item, None, 1.0, fit=False))
            continue
        name, spec = item.split("=", 1)
        parts = spec.split(":")
        if name == "rdf":
            src = parts[0]
            if src.endswith(".npy"):
                g = np.load(src)
            else:
                d = json.load(open(src))
                est = d["records"][-1]["estimate"]
                g = np.array([y for n, y in zip(est["names"], est["y"]) if n.startswith("rdf(")])
            sig = float(parts[1]) if len(parts) > 1 else 0.05
            w = float(parts[2]) if len(parts) > 2 else 1.0 / len(rdf_r)
            rr = (float(parts[3]), float(parts[4])) if len(parts) > 4 else (0.24, 0.8)
            full = np.full(len(rdf_r), np.nan)
            full[: len(g)] = g[: len(rdf_r)]
            out.append(Target("rdf", full, sig, weight=w, r_range=rr))
        else:
            out.append(
                Target(
                    name,
                    float(parts[0]),
                    float(parts[1]) if len(parts) > 1 else 1.0,
                    weight=float(parts[2]) if len(parts) > 2 else 1.0,
                )
            )
    return out


def add_arguments(ap):
    ap.add_argument("-o", "--out", default=None, help="output prefix (prefix.json, prefix_state.npz, prefix.log)")
    ap.add_argument("--model", default="base", choices=list(MODELS))
    ap.add_argument("--prmtop", help="pGM prmtop (overrides --model)")
    ap.add_argument("--coords", help="restart matching --prmtop")
    ap.add_argument(
        "--params",
        default="q,cov,alpha,radius,lj_r,lj_eps",
        help="scale factors (ln s): a quantity (every entry) or quantity@key1+key2 (those tying keys, "
        "e.g. alpha@OW,alpha@HW for per-type polarizabilities)",
    )
    ap.add_argument("--start", default="", help="initial ln scales, one per parameter")
    ap.add_argument("--prior", type=float, default=0.1, help="prior width of every ln scale (regularisation)")
    ap.add_argument("--prior-center", default="", help="prior centre (default 0: the prmtop's values)")
    ap.add_argument(
        "--targets",
        default="density=0.997:0.002,hvap=10.52:0.05,eps=78.4:1.5,gas_dipole=1.855:0.01,"
        "gas_polarizability=1.47:0.01,liquid_dipole",
    )
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--T", type=float, default=298.0)
    ap.add_argument("--dt", type=float, default=2.0, help="fs")
    ap.add_argument("--equil", type=float, default=50.0, help="ps of equilibration per iteration")
    ap.add_argument("--prod", type=float, default=1000.0, help="ps of production per iteration")
    ap.add_argument("--every", type=float, default=0.5, help="ps between analysed frames")
    ap.add_argument("--nblocks", type=int, default=10)
    ap.add_argument("--radius", type=float, default=1.0, help="initial trust radius (units of the prior widths)")
    ap.add_argument("--radius-max", type=float, default=2.0)
    ap.add_argument("--exact", type=int, default=0, help="exact reweighting prediction on every k-th frame (0: off)")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-6, help="CG tolerance of the frame analysis")
    ap.add_argument("--md-tol", type=float, default=1e-5, help="dipole_scf_tol of the MD")
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--nfft", type=int, default=48, help="PME grid per side (0: from the Ewald spacing rule)")
    ap.add_argument(
        "--ewald-beta", type=float, default=None, help="nm^-1 (default: 4.0 at 0.9 nm, else the Amber rule)"
    )
    ap.add_argument(
        "--cutoff",
        type=float,
        default=0.9,
        help="nm; other than 0.9: Ewald coefficient and PME spacing "
        "from md.forcefield.elec_cutoff_settings (same direct-sum tolerance)",
    )
    ap.add_argument("--rdf-rmax", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--ensemble", default="npt", choices=["npt", "nvt"])
    ap.add_argument("--replicas", type=int, default=1, help="NVT: replicas advanced together (vmap)")
    ap.add_argument("--equil-rep", type=float, default=20.0, help="ps of the replicas before sampling")
    ap.add_argument("--skin", type=float, default=0.1)
    ap.add_argument(
        "--fixed",
        action="store_true",
        help="measure only: every iteration is a further segment at --start "
        "(no step, no re-equilibration); combine the segments with scripts/liquid_fit_tools.py combine",
    )


def setup(a):
    """System, coordinates, parameter space, RDF spec, gas phase, objective and start from the arguments."""
    top, crd = MODELS[a.model]
    top, crd = a.prmtop or top, a.coords or crd
    top, crd = os.path.expanduser(top), os.path.expanduser(crd)
    mols = _dedupe(read_prmtop_pgm(top, first_residue_only=False))
    xyz, vel, box = read_coordinates(crd)
    if len(xyz) != sum(m.n for m in mols):  # another box of the same (single) molecule
        if any(list(m.elements) != list(mols[0].elements) for m in mols) or len(xyz) % mols[0].n:
            raise SystemExit(f"{crd} has {len(xyz)} atoms; {top} {sum(m.n for m in mols)}")
        mols = [mols[0]] * (len(xyz) // mols[0].n)
    sys_ = System(mols)
    pos, H = xyz * 0.1, box_from_cell(*box) * 0.1
    plist = []
    for item in [s for s in a.params.split(",") if s]:  # quantity or quantity@key1+key2 (per tying key)
        q, _, keys = item.partition("@")
        plist.append(Param(q, "scale", keys=keys.split("+") if keys else None))
    space = ParameterSpace(sys_.table, plist, prior_sigma=a.prior)
    theta0 = np.array([float(x) for x in a.start.split(",")]) if a.start else np.zeros(space.n)
    center = np.array([float(x) for x in a.prior_center.split(",")]) if a.prior_center else None
    types = set(sys_.types)
    rdf = RDFSpec.by_type(
        sys_, "OW" if "OW" in types else sorted(types)[0], rmax=a.rdf_rmax, nbins=int(round(a.rdf_rmax / 0.01))
    )
    gas = GasPhase(sys_.molecules[0], pos[sys_.atom_slice(0)], sys_.table, space)
    targets = parse_targets(a.targets, rdf.r)
    obj = Objective(targets, space, gas=gas, prior_center=center, rdf_r=rdf.r)
    return dict(
        mols=mols, sys=sys_, pos=pos, H=H, space=space, theta0=theta0, rdf=rdf, gas=gas, obj=obj, targets=targets
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    a = ap.parse_args()
    if not a.out:
        ap.error("-o/--out is required")
    S = setup(a)
    mols, sys_, pos, H, space, theta0, rdf, gas, obj = (
        S[k] for k in ("mols", "sys", "pos", "H", "space", "theta0", "rdf", "gas", "obj")
    )
    ew = {"ewald_beta": 4.0, "pme_spacing": 0.08}
    if abs(a.cutoff - 0.9) > 1e-9:
        ew = {k: v for k, v in elec_cutoff_settings(a.cutoff).items() if k != "elec_cutoff"}
    if a.ewald_beta:
        ew["ewald_beta"] = a.ewald_beta
    st = MDSettings(
        cutoff=a.cutoff,
        skin=a.skin,
        pme_grid=(a.nfft,) * 3 if a.nfft else None,
        pme_order=6,
        lj_lrc=True,
        dipole_tol=a.md_tol,
        precision=a.precision,
        **ew,
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log = open(a.out + ".log", "a")

    class Tee:
        def write(self, s):
            sys.stdout.write(s)
            log.write(s)

        def flush(self):
            sys.stdout.flush()
            log.flush()

    fit = LiquidFit(
        sys_,
        pos,
        H,
        space,
        obj,
        T=a.T,
        settings=st,
        dt=a.dt / 1000.0,
        equil_ps=a.equil,
        prod_ps=a.prod,
        every_ps=a.every,
        rdf=rdf,
        chunk=a.chunk,
        tol=a.tol,
        nblocks=a.nblocks,
        radius=a.radius,
        radius_max=a.radius_max,
        prefix=a.out,
        exact_every=a.exact,
        bootstrap=a.bootstrap,
        log=Tee(),
        seed=a.seed,
        fixed=a.fixed,
        ensemble=a.ensemble,
        replicas=a.replicas,
        equil_rep_ps=a.equil_rep,
    )
    print(
        f"# {len(mols)} molecules, {sys_.n} atoms; parameters {space.names}; start {space.describe(theta0)}; "
        f"gas phase at start {gas.values(theta0)}; device {jax.devices()[0]}",
        file=Tee(),
        flush=True,
    )
    fit.run(theta0, a.iters, resume=not a.no_resume, max_seconds=None if a.max_minutes is None else 60 * a.max_minutes)
    if len(fit.records) >= a.iters:
        open(a.out + ".done", "w").write("done\n")


if __name__ == "__main__":
    main()
