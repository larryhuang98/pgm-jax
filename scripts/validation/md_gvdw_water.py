"""NPT pGM3P + GVDW water with the MD engine, at the settings of the GVDW manuscript.

8 A cutoff, PME 48^3 order 8, ew_coeff 0.4 A^-1, no dispersion tail, dipole tolerance 1e-4, 298 K,
1 bar, Langevin 2/ps, Monte Carlo barostat every 100 steps, dt 1 fs, from the model's restart in
PGM_GVDW_DATA/inputs/<model>/; the density is compared with the pmemd-pgm runs (Table 1 of the
manuscript: 0.997 Gaussian, 0.999 Slater g/cm^3); lj is the pGM3P-25 Lennard-Jones model
(docs/howto_vdw.md, docs/model_options.md).

Usage:

    python scripts/validation/md_gvdw_water.py slater|gauss|lj [--time-ps 100] [--equil-ps 20] [--seed 0]
    python scripts/validation/md_gvdw_water.py --help

Inputs: PGM_GVDW_DATA (prmtop and the model's restart; pgm_jax.paths).
Outputs: <out>/<model>_<seed>.json (density with a 5-block error, ns/day; default --out
runs/gvdw_md), also printed.
Units: --time-ps and --equil-ps ps (samples every ps); density g/cm^3.
Runtime: GPU, minutes.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import numpy as np

from pgm_jax.cli.args import add_seed_arg, setup_logging
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import Langevin
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import pgm3p25_files, repo_path, resource
from pgm_jax.system import System
from pgm_jax.vdw import PGM3P_GVDW, set_gvdw

jax.config.update("jax_enable_x64", True)
TOP = pgm3p25_files()[0]


def build(model: str, seed: int) -> Simulation:
    """Return the NPT simulation of the 512-water box with the model's van der Waals term (module docstring)."""
    rst = resource("gvdw_data", f"inputs/{model if model != 'gauss' else 'gaussian'}/inpcrd.restrt")
    mols = read_prmtop_molecules(TOP)
    vdw, rep = "lj", "gauss"
    if model != "lj":
        par = PGM3P_GVDW[model]
        gm = {id(m): set_gvdw(m, {"OW": par["OW"]}) for m in mols}
        mols = [gm[id(m)] for m in mols]
        vdw, rep = "gvdw", par["rep"]
    xyz, _, box = read_coordinates(rst)
    s = MDSettings().replace(
        cutoff=0.8,
        skin=0.1,
        ewald_beta=4.0,
        pme_grid=(48, 48, 48),
        pme_order=8,
        lj_lrc=False,
        dipole_tol=1e-4,
        vdw=vdw,
        gvdw_rep=rep,
    )
    return Simulation(
        System(mols),
        xyz * 0.1,
        box_from_cell(*box) * 0.1,
        s,
        dt=0.001,
        thermostat=Langevin(2.0),
        barostat=MonteCarloBarostat(1.0, 100),
        temperature=298.0,
        seed=seed,
        log=None,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run the NPT simulation and write the density (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["slater", "gauss", "lj"], help="van der Waals model")
    ap.add_argument("--time-ps", type=float, default=100.0, help="production [ps] (a density sample every ps)")
    ap.add_argument("--equil-ps", type=float, default=20.0, help="equilibration [ps]")
    add_seed_arg(ap)
    ap.add_argument("-o", "--out", default=repo_path("runs", "gvdw_md"), help="output directory")
    a = ap.parse_args(argv)
    setup_logging()
    sim = build(a.model, a.seed)
    sim.advance(int(a.equil_ps * 1000))  # 1 fs steps
    rho, t0 = [], time.time()
    for _ in range(int(a.time_ps)):
        sim.advance(1000)
        rho.append(sim.observables()["density_g_cm3"])
    rho = np.array(rho)
    blocks = np.array([b.mean() for b in np.array_split(rho, 5)])
    out = {
        "model": a.model,
        "seed": a.seed,
        "ps": a.time_ps,
        "density": float(rho.mean()),
        "density_se": float(blocks.std(ddof=1) / np.sqrt(5)),
        "density_sd": float(rho.std()),
        "ns_per_day": a.time_ps / 1000 / ((time.time() - t0) / 86400),
    }
    print(json.dumps(out), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.model}_{a.seed}.json"), "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
