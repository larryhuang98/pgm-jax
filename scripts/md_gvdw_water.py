"""NPT pGM3P + GVDW water with the MD engine, at the settings of the GVDW manuscript (8 A cutoff,
PME 48^3 order 8, ew_coeff 0.4 A^-1, no dispersion tail, dipole tol 1e-4, 298 K, 1 bar, Langevin
gamma 2/ps, MC barostat every 100 steps, dt 1 fs); density to compare with the pmemd-pgm runs
(Table 1 of the manuscript: 0.997 Gaussian, 0.999 Slater g/cm^3).
    python scripts/md_gvdw_water.py slater|gauss|lj [--ps 100] [--equil 20] [--seed 0]"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from pgm_jax.md.box import box_from_cell
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.simulation import Simulation
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.system import System
from pgm_jax.vdw import PGM3P_GVDW, set_gvdw

TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["slater", "gauss", "lj"])
ap.add_argument("--ps", type=float, default=100.0)
ap.add_argument("--equil", type=float, default=20.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rst = os.path.expanduser(f"~/pgm-gvdw-data/inputs/{a.model if a.model != 'gauss' else 'gaussian'}/inpcrd.restrt")
mols = read_prmtop_molecules(TOP)
vdw, rep = "lj", "gauss"
if a.model != "lj":
    par = PGM3P_GVDW[a.model]
    gm = {id(m): set_gvdw(m, {"OW": par["OW"]}) for m in mols}
    mols = [gm[id(m)] for m in mols]
    vdw, rep = "gvdw", par["rep"]
xyz, _, box = read_coordinates(rst)
s = MDSettings(
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
sim = Simulation(
    System(mols),
    xyz * 0.1,
    box_from_cell(*box) * 0.1,
    s,
    dt=0.001,
    ensemble="npt",
    temperature=298.0,
    gamma=2.0,
    pressure=1.0,
    barostat_interval=100,
    seed=a.seed,
    log=None,
)
sim._advance(int(a.equil * 1000))
rho, t0 = [], time.time()
for _ in range(int(a.ps)):
    sim._advance(1000)
    rho.append(sim.observables()["density_g_cm3"])
rho = np.array(rho)
blocks = np.array([b.mean() for b in np.array_split(rho, 5)])
out = {
    "model": a.model,
    "seed": a.seed,
    "ps": a.ps,
    "density": float(rho.mean()),
    "density_se": float(blocks.std(ddof=1) / np.sqrt(5)),
    "density_sd": float(rho.std()),
    "ns_per_day": a.ps / 1000 / ((time.time() - t0) / 86400),
}
print(json.dumps(out), flush=True)
os.makedirs(os.path.join(ROOT, "runs/gvdw_md"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, f"runs/gvdw_md/{a.model}_{a.seed}.json"), "w"))
