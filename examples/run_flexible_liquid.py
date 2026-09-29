"""Liquid NPT simulation of flexible molecules from a FlexibleTemplate (fit_bonded_template.py).

    python examples/run_flexible_liquid.py runs/flex/methanol.flex --n 216 --ps 100

Builds a dilute box (random orientations on a lattice), relaxes it with Langevin NVT, then runs
Monte Carlo NPT; writes <prefix>.log (energies, temperatures, density), <prefix>.nc (Amber NetCDF
trajectory) and <prefix>.chk, and prints the mean density of the second half of the run."""

import argparse
import os

import jax
import numpy as np

from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.system import System

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser()
ap.add_argument("template")
ap.add_argument("--n", type=int, default=216, help="molecules")
ap.add_argument("--density0", type=float, default=0.55, help="starting density, g/cm^3 (below the liquid)")
ap.add_argument("--T", type=float, default=298.0)
ap.add_argument("--P", type=float, default=1.0, help="bar")
ap.add_argument("--ps", type=float, default=100.0, help="NPT length")
ap.add_argument("--dt", type=float, default=0.5, help="fs")
ap.add_argument("--prefix", default="")
a = ap.parse_args()
tpl = FlexibleTemplate.load(a.template)
prefix = a.prefix or os.path.splitext(a.template)[0] + "_npt"
pos, H = liquid_box(tpl, a.n, a.density0, seed=1, min_dist=0.18)
sys_ = System([tpl.pgm] * a.n)
dt = a.dt / 1000.0
settings = MDSettings()  # 0.9 nm cutoff, PME, dipole tol 1e-5, mixed precision
nvt = FlexibleSimulation(sys_, [tpl] * a.n, pos, H, settings, dt=dt, ensemble="nvt", temperature=a.T, gamma=5.0)
nvt.run(int(round(2.0 / dt)), report=int(round(1.0 / dt)), prefix=prefix + "_nvt")  # 2 ps relaxation
sim = FlexibleSimulation(
    sys_,
    [tpl] * a.n,
    nvt.positions_nm(),
    np.asarray(nvt.state.box),
    settings,
    dt=dt,
    ensemble="npt",
    temperature=a.T,
    pressure=a.P,
    gamma=1.0,
    barostat_interval=25,
    vel_nm_ps=nvt.velocities_nm_ps(),
)
every = int(round(1.0 / dt))  # report every ps
sim.run(int(round(a.ps / dt)), report=every, traj=every, restart=10 * every, prefix=prefix)
log = [ln.split() for ln in open(prefix + ".log") if ln.strip() and not ln.startswith("#")]
head = [ln for ln in open(prefix + ".log") if ln.startswith("#") and "density_g_cm3" in ln][-1].split()[1:]
rho = np.array([float(r[head.index("density_g_cm3")]) for r in log])
half = rho[len(rho) // 2 :]
print(f"density, second half ({len(half)} ps): {half.mean():.4f} g/cm^3 (sd {half.std():.4f})")
