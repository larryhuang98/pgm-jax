"""Liquid NPT simulation of flexible molecules from a FlexibleTemplate (examples/fit_bonded_template.py).

Builds a dilute box (random orientations on a lattice), relaxes it with 2 ps of Langevin NVT
(friction 5/ps), then runs Monte Carlo NPT (Langevin 1/ps, a volume move every 25 steps) with the
default MDSettings (0.9 nm cutoff, PME, dipole tolerance 1e-5, mixed precision), and prints the
mean density of the second half of the run.

Usage:

    python examples/run_flexible_liquid.py runs/flex/methanol.flex --molecules 216 --time-ps 100
    python examples/run_flexible_liquid.py --help

Inputs: the template file.
Outputs: <out>_nvt.log, <out>.log (energies, temperatures, density; a line per ps), <out>.nc (Amber
NetCDF trajectory, a frame per ps), <out>.chk (every 10 ps); default out: <template>_npt.
Units: --temperature-K K, --pressure-bar bar, --time-ps ps, --dt-fs fs, --density0-g-cm3 g/cm^3.
Runtime: GPU recommended (216 molecules, 100 ps: minutes).
"""

from __future__ import annotations

import argparse
import os

import jax
import numpy as np

from pgm_jax.cli.args import add_dt_arg, add_temperature_arg
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.thermostats import Langevin
from pgm_jax.system import System

jax.config.update("jax_enable_x64", True)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("template", help="FlexibleTemplate file (.flex)")
    ap.add_argument("--molecules", type=int, default=216, help="number of molecules")
    ap.add_argument("--density0-g-cm3", type=float, default=0.55, help="starting density [g/cm^3] (below the liquid)")
    add_temperature_arg(ap)
    ap.add_argument("--pressure-bar", type=float, default=1.0, help="pressure [bar]")
    ap.add_argument("--time-ps", type=float, default=100.0, help="NPT length [ps]")
    add_dt_arg(ap, 0.5)
    ap.add_argument("-o", "--out", default="", help="output prefix (default: <template>_npt)")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run NVT and NPT and print the density (see the module docstring)."""
    a = build_parser().parse_args(argv)
    tpl = FlexibleTemplate.load(a.template)
    prefix = a.out or os.path.splitext(a.template)[0] + "_npt"
    pos, H = liquid_box(tpl, a.molecules, a.density0_g_cm3, seed=1, min_dist=0.18)
    sys_ = System([tpl.pgm] * a.molecules)
    dt = a.dt_fs / 1000.0  # ps
    temp = a.temperature_K
    settings = MDSettings()  # 0.9 nm cutoff, PME, dipole tol 1e-5, mixed precision
    nvt = FlexibleSimulation(
        sys_, [tpl] * a.molecules, pos, H, settings, dt=dt, thermostat=Langevin(5.0), temperature=temp
    )
    nvt.run(int(round(2.0 / dt)), report_every=int(round(1.0 / dt)), prefix=prefix + "_nvt")  # 2 ps relaxation
    sim = FlexibleSimulation(
        sys_,
        [tpl] * a.molecules,
        nvt.positions(),
        np.asarray(nvt.state.box),
        settings,
        dt=dt,
        thermostat=Langevin(1.0),
        barostat=MonteCarloBarostat(a.pressure_bar, 25),
        temperature=temp,
        velocities=nvt.velocities(),
    )
    every = int(round(1.0 / dt))  # report every ps
    sim.run(
        int(round(a.time_ps / dt)), report_every=every, traj_every=every, checkpoint_every=10 * every, prefix=prefix
    )
    with open(prefix + ".log") as fh:
        lines = fh.readlines()
    log = [ln.split() for ln in lines if ln.strip() and not ln.startswith("#")]
    head = [ln for ln in lines if ln.startswith("#") and "density_g_cm3" in ln][-1].split()[1:]
    rho = np.array([float(r[head.index("density_g_cm3")]) for r in log])
    half = rho[len(rho) // 2 :]
    print(f"density, second half ({len(half)} ps): {half.mean():.4f} g/cm^3 (sd {half.std():.4f})")


if __name__ == "__main__":
    main()
