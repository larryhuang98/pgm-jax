"""Fit a rigid pGM liquid to several liquid and gas-phase targets (the `pgm-jax fit-multi` command).

pGM water (or any box of one rigid molecule type from a pGM prmtop) is fitted with ensemble
gradients (pgm_jax.fit.LiquidFit; docs/liquid_fit.md) to: density, heat of vaporization, static
dielectric constant, gas-phase dipole and polarizability, O-O g(r); with parameter uncertainties.
The parameters are ln scale factors of parameter quantities (--params).  add_arguments and setup
are shared with scripts/fitting/liquid_fit_tools.py and validate_eps_gradient.py.

Targets (--targets): name=value:sigma[:weight], or a bare name (evaluated, propagated, not
fitted); rdf=FILE:sigma:weight[:rmin:rmax] with FILE a .npy g(r) on the analysis grid or a fit JSON
whose last record has the rdf components.  A run stops after --max-minutes (e.g. GPU jobs of less
than 45 min) and continues with the same --out.

Usage:

    # the base pGM water toward experiment (eps included)
    T=density=0.997:0.002,hvap=10.52:0.05,eps=78.4:1.5,gas_dipole=1.855:0.01
    T=$T,gas_polarizability=1.47:0.01,liquid_dipole
    python scripts/fitting/fit_multi.py -o runs/fit/demo --params q,cov,alpha,radius,lj_r,lj_eps --targets $T
        --equil-ps 50 --prod-ps 2000 --iters 6
    # measure only (all targets without values): observables, Jacobians, errors at --start
    python scripts/fitting/fit_multi.py -o runs/fit/s0 --params q --targets density,hvap,eps,liquid_dipole --iters 1
    python scripts/fitting/fit_multi.py --help

Inputs: --model (512-water boxes in PGM_EPSP, pgm_jax.paths) or --prmtop/--coords.
Outputs: <out>.json (records of every iteration), <out>_state.npz, <out>_frames*.npz (analysed
frames), <out>.log (the printed log), <out>.done when all iterations are done.
Units: --temperature-K K, --dt-fs fs, durations in ps (--equil-ps, --prod-ps, --sample-ps,
--equil-rep-ps), --cutoff-nm, --skin-nm and --rdf-rmax-nm nm, --ewald-beta-per-nm 1/nm; target
values in the units of pgm_jax.fit (density g/cm^3, hvap kcal/mol, dipoles D, polarizability A^3).
Runtime: GPU; one NPT (or batched NVT) simulation per iteration.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TextIO

import jax
import numpy as np

from pgm_jax.cli.args import (
    add_barostat_args,
    add_dipole_tol_arg,
    add_dt_arg,
    add_precision_arg,
    add_seed_arg,
    add_temperature_arg,
    barostat_from_args,
    setup_logging,
)
from pgm_jax.fit import GasPhase, Objective, Param, ParameterSpace, RDFSpec, Target
from pgm_jax.fit.liquid import LiquidFit
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.forcefield import MDSettings, elec_cutoff_settings
from pgm_jax.md.io import read_coordinates
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import resource
from pgm_jax.system import System

jax.config.update("jax_enable_x64", True)
MODELS = {  # name -> (prmtop, coordinates) of the 512-water boxes of the dielectric study
    "base": (resource("epsp", "base/base_512.prmtop"), resource("epsp", "base/base_512.rst7")),
    "p25": (resource("epsp", "p25_512.prmtop"), resource("epsp", "p25_512.rst7")),
}


def parse_targets(text: str, rdf_r: np.ndarray) -> list[Target]:
    """Return the fit targets of --targets (see the module docstring for the syntax).

    Parameters
    ----------
    text : str
        Comma-separated target specifications.
    rdf_r : np.ndarray (B,)
        Radii of the RDF analysis grid [nm] (an rdf target is padded / cut to it; default weight
        1 / B, default range 0.24-0.8 nm).

    Returns
    -------
    list of Target
    """
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


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """Add the options that define the system, the objective and the sampling (shared with the sibling scripts)."""
    ap.add_argument("-o", "--out", default=None, help="output prefix (prefix.json, prefix_state.npz, prefix.log)")
    ap.add_argument("--model", default="base", choices=list(MODELS), help="512-water box of PGM_EPSP")
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
        help="targets (module docstring)",
    )
    ap.add_argument("--iters", type=int, default=5, help="iterations")
    add_temperature_arg(ap, 298.0)
    add_dt_arg(ap, 2.0)
    ap.add_argument("--equil-ps", type=float, default=50.0, help="equilibration per iteration [ps]")
    ap.add_argument("--prod-ps", type=float, default=1000.0, help="production per iteration [ps]")
    ap.add_argument("--sample-ps", type=float, default=0.5, help="time between analysed frames [ps]")
    ap.add_argument("--nblocks", type=int, default=10, help="blocks of the jackknife errors")
    ap.add_argument("--radius", type=float, default=1.0, help="initial trust radius (units of the prior widths)")
    ap.add_argument("--radius-max", type=float, default=2.0, help="largest trust radius")
    ap.add_argument("--exact", type=int, default=0, help="exact reweighting prediction on every k-th frame (0: off)")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap samples of the parameter errors")
    ap.add_argument("--chunk", type=int, default=8, help="frames analysed together")
    ap.add_argument("--analysis-tol", type=float, default=1e-6, help="CG tolerance of the frame analysis")
    add_dipole_tol_arg(ap, 1e-5, help="induced-dipole tolerance of the MD (dipole_scf_tol)")
    add_precision_arg(ap)
    ap.add_argument("--nfft", type=int, default=48, help="PME grid per side (0: from the Ewald spacing rule)")
    ap.add_argument(
        "--ewald-beta-per-nm",
        type=float,
        default=None,
        help="Ewald coefficient [1/nm] (default: 4.0 at 0.9 nm, else the Amber rule)",
    )
    ap.add_argument(
        "--cutoff-nm",
        type=float,
        default=0.9,
        help="cutoff [nm]; other than 0.9: Ewald coefficient and PME spacing "
        "from md.forcefield.elec_cutoff_settings (same direct-sum tolerance)",
    )
    ap.add_argument("--rdf-rmax-nm", type=float, default=0.8, help="range of the RDF analysis [nm] (0.01 nm bins)")
    add_seed_arg(ap)
    ap.add_argument("--max-minutes", type=float, default=None, help="stop after this wall time [min]")
    ap.add_argument("--no-resume", action="store_true", help="start again instead of continuing --out")
    add_barostat_args(ap, default="mc")
    ap.add_argument("--replicas", type=int, default=1, help="--barostat none: replicas advanced together (vmap)")
    ap.add_argument("--equil-rep-ps", type=float, default=20.0, help="time of the replicas before sampling [ps]")
    ap.add_argument("--skin-nm", type=float, default=0.1, help="neighbour-list skin [nm]")
    ap.add_argument(
        "--fixed",
        action="store_true",
        help="measure only: every iteration is a further segment at --start "
        "(no step, no re-equilibration); combine the segments with scripts/fitting/liquid_fit_tools.py combine",
    )


def setup(a: argparse.Namespace) -> dict:
    """Return the system, coordinates, parameter space, RDF spec, gas phase, objective and start.

    Parameters
    ----------
    a : argparse.Namespace
        Options of add_arguments.

    Returns
    -------
    dict
        Keys mols (list of Molecule), sys (System), pos (N, 3) [nm], H (3, 3) [nm], space
        (ParameterSpace), theta0 (n,), rdf (RDFSpec), gas (GasPhase), obj (Objective), targets.

    Raises
    ------
    SystemExit
        Coordinates whose atom count does not fit the prmtop's molecule.
    """
    top, crd = MODELS[a.model]
    top, crd = a.prmtop or top, a.coords or crd
    mols = read_prmtop_molecules(top)
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
        sys_, "OW" if "OW" in types else sorted(types)[0], rmax=a.rdf_rmax_nm, nbins=int(round(a.rdf_rmax_nm / 0.01))
    )
    gas = GasPhase(sys_.molecules[0], pos[sys_.atom_slice(0)], sys_.table, space)
    targets = parse_targets(a.targets, rdf.r)
    obj = Objective(targets, space, gas=gas, prior_center=center, rdf_r=rdf.r)
    return dict(
        mols=mols, sys=sys_, pos=pos, H=H, space=space, theta0=theta0, rdf=rdf, gas=gas, obj=obj, targets=targets
    )


class Tee:
    """Text stream that writes to stdout and to a log file."""

    def __init__(self, log: TextIO) -> None:
        """Set up the stream; `log` is the open log file."""
        self.log = log

    def write(self, s: str) -> None:
        """Write s to stdout and the log."""
        sys.stdout.write(s)
        self.log.write(s)

    def flush(self) -> None:
        """Flush both."""
        sys.stdout.flush()
        self.log.flush()


def md_settings(a: argparse.Namespace) -> MDSettings:
    """Return the MD settings of the options (Ewald coefficient and PME spacing from the cutoff rule)."""
    ew = {"ewald_beta": 4.0, "pme_spacing": 0.08}
    if abs(a.cutoff_nm - 0.9) > 1e-9:
        ew = {k: v for k, v in elec_cutoff_settings(a.cutoff_nm).items() if k != "elec_cutoff"}
    if a.ewald_beta_per_nm:
        ew["ewald_beta"] = a.ewald_beta_per_nm
    return MDSettings().replace(
        cutoff=a.cutoff_nm,
        skin=a.skin_nm,
        pme_grid=(a.nfft,) * 3 if a.nfft else None,
        pme_order=6,
        lj_lrc=True,
        dipole_tol=a.dipole_tol,
        precision=a.precision,
        **ew,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, set up the fit (setup()) and run or resume it (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    a = ap.parse_args(argv)
    setup_logging()
    if not a.out:
        ap.error("-o/--out is required")
    S = setup(a)
    mols, sys_, pos, H, space, theta0, rdf, gas, obj = (
        S[k] for k in ("mols", "sys", "pos", "H", "space", "theta0", "rdf", "gas", "obj")
    )
    st = md_settings(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tee = Tee(open(a.out + ".log", "a"))
    fit = LiquidFit(
        sys_,
        pos,
        H,
        space,
        obj,
        temperature=a.temperature_K,
        settings=st,
        dt=a.dt_fs / 1000.0,
        equil_ps=a.equil_ps,
        prod_ps=a.prod_ps,
        every_ps=a.sample_ps,
        rdf=rdf,
        chunk=a.chunk,
        dipole_tol=a.analysis_tol,
        nblocks=a.nblocks,
        radius=a.radius,
        radius_max=a.radius_max,
        prefix=a.out,
        exact_every=a.exact,
        bootstrap=a.bootstrap,
        log=tee,
        seed=a.seed,
        fixed=a.fixed,
        barostat=barostat_from_args(a),
        replicas=a.replicas,
        equil_rep_ps=a.equil_rep_ps,
    )
    print(
        f"# {len(mols)} molecules, {sys_.n} atoms; parameters {space.names}; start {space.describe(theta0)}; "
        f"gas phase at start {gas.values(theta0)}; device {jax.devices()[0]}",
        file=tee,
        flush=True,
    )
    fit.run(theta0, a.iters, resume=not a.no_resume, max_seconds=None if a.max_minutes is None else 60 * a.max_minutes)
    if len(fit.records) >= a.iters:
        with open(a.out + ".done", "w") as fh:
            fh.write("done\n")


if __name__ == "__main__":
    main()
