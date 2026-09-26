"""Run pGM + LJ molecular dynamics with JAX-MD from Amber inputs.

Options use Amber's units and names where they exist (Angstrom, fs, ew_coeff in 1/A, ...).

    python scripts/run_md.py -p water.prmtop -c water.rst7 -o md --ensemble npt --temp 298 \
        --nsteps 250000 --dt 1.0 --cut 8.0 --nfft 48 48 48 --order 8 --ew-coeff 0.4 --vdwmeth 0 \
        --dipole-tol 1e-5 --gamma 2.0 --barostat-interval 100 --report 2000 --traj 1000

Outputs: <out>.log (energies, temperature, density, solver iterations, speed), <out>.nc (Amber
NetCDF trajectory), <out>.rst7 (Amber NetCDF restart), <out>.chk (complete checkpoint; continue with
--checkpoint <out>.chk); with --dipoles N, <out>.dip (cell dipole every N steps, for
scripts/dielectric.py); with --induced N, <out>.mu.nc (per-atom induced dipoles).
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--prmtop", required=True)
    ap.add_argument("-c", "--coords", required=True, help="inpcrd / rst7 (ASCII or NetCDF) with a box")
    ap.add_argument("-o", "--out", default="md", help="output prefix")
    ap.add_argument("--checkpoint", help="continue from a .chk file (same system and settings)")
    ap.add_argument("--ensemble", default="npt", choices=["nve", "nvt", "npt"])
    ap.add_argument("--nsteps", type=int, default=10000)
    ap.add_argument("--dt", type=float, default=1.0, help="fs")
    ap.add_argument("--temp", type=float, default=298.0, help="K (temp0)")
    ap.add_argument("--gamma", type=float, default=2.0, help="Langevin friction, 1/ps (gamma_ln)")
    ap.add_argument("--thermostat", default="langevin", choices=["langevin", "bussi", "gle"],
                    help="langevin (ntt=3), bussi (ntt=11, fastest with pGM), gle (smooth slow-band)")
    ap.add_argument("--tautp", type=float, default=1.0, help="Bussi time constant, ps (tautp)")
    ap.add_argument("--press", type=float, default=1.0, help="bar (pres0)")
    ap.add_argument("--barostat-interval", type=int, default=100, help="steps between MC volume moves (mcbarint)")
    ap.add_argument("--cut", type=float, default=9.0, help="A: direct space and LJ cutoff")
    ap.add_argument("--skin", type=float, default=1.0, help="A (skinnb)")
    ap.add_argument("--ew-coeff", type=float, default=0.4, help="1/A")
    ap.add_argument("--nfft", type=int, nargs=3, help="PME grid; default from --pme-spacing")
    ap.add_argument("--pme-spacing", type=float, default=0.5, help="A")
    ap.add_argument("--order", type=int, default=8, help="PME B-spline order")
    ap.add_argument("--vdwmeth", type=int, default=1, choices=[0, 1], help="1: LJ long-range correction")
    ap.add_argument("--dipole-tol", type=float, default=1e-5, help="dipole_scf_tol (pmemd-pgm criterion)")
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--local-niter", type=int, default=0, help="inner iterations of the short-range preconditioner")
    ap.add_argument("--local-cut", type=float, default=3.0, help="A")
    ap.add_argument("--peek", type=float, default=0.65, help="scf_sor_coefficient; 0 disables the peek step")
    ap.add_argument("--predictor", default="mu4", choices=["mu4", "mu3", "ls", "none"],
                    help="initial dipole guess: mu4/mu3 polynomial extrapolation (fused residual), "
                         "ls: pmemd-pgm CPU least squares (dipole_scf_init=3)")
    ap.add_argument("--extrap-order", type=int, default=3, help="dipole_scf_init_order (--predictor ls)")
    ap.add_argument("--extrap-steps", type=int, default=2, help="dipole_scf_init_step (--predictor ls)")
    ap.add_argument("--precision", default="mixed", choices=["mixed", "double"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-velocities", action="store_true", help="ignore velocities in the coordinates (irest=0)")
    ap.add_argument("--report", type=int, default=1000)
    ap.add_argument("--traj", type=int, default=0, help="steps between trajectory frames (0: none)")
    ap.add_argument("--restart", type=int, default=0, help="steps between restart/checkpoint files")
    ap.add_argument("--pressure", action="store_true", help="also report the virial pressure")
    ap.add_argument("--dipoles", type=int, default=0, help="steps between cell-dipole samples (<out>.dip; 0: none)")
    ap.add_argument("--induced", type=int, default=0, help="steps between per-atom induced dipole frames (<out>.mu.nc)")
    a = ap.parse_args(argv)
    st = MDSettings(cutoff=a.cut / 10, skin=a.skin / 10, ewald_beta=a.ew_coeff * 10,
                    pme_grid=tuple(a.nfft) if a.nfft else None, pme_spacing=a.pme_spacing / 10, pme_order=a.order,
                    lj_lrc=bool(a.vdwmeth), dipole_tol=a.dipole_tol, max_iter=a.max_iter, local_cut=a.local_cut / 10,
                    local_niter=a.local_niter, peek=a.peek, predictor=a.predictor,
                    extrap_order=a.extrap_order, extrap_steps=a.extrap_steps,
                    precision=a.precision)
    sim = Simulation.from_amber(a.prmtop, a.coords, use_velocities=not a.no_velocities, settings=st,
                                dt=a.dt / 1000, ensemble=a.ensemble, temperature=a.temp, gamma=a.gamma,
                                thermostat=a.thermostat, tau_t=a.tautp,
                                pressure=a.press, barostat_interval=a.barostat_interval, seed=a.seed)
    if a.checkpoint:
        sim.load(a.checkpoint)
    sim.run(a.nsteps, report=a.report, traj=a.traj, restart=a.restart, prefix=a.out,
            pressure_every_report=a.pressure, append=bool(a.checkpoint), dipoles=a.dipoles, induced=a.induced)


if __name__ == "__main__":
    main()
