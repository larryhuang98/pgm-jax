"""Run pGM + LJ molecular dynamics of rigid molecules from Amber inputs (the `pgm-jax md` command).

The system of an Amber prmtop (a pGM prmtop, or the point charges of a classical one) at the
coordinates of a restart file is run with the rigid-molecule engine (pgm_jax.md.simulation):
NVE, NVT or NPT, optionally with multiple time stepping (docs/mts.md), extended-Lagrangian induced
dipoles (docs/iel.md) or an external electric field (docs/efield.md).

This script keeps Amber's units and option names on purpose, for side-by-side comparisons with
pmemd (docs/api_design.md, D2): lengths in Angstrom (--cut, --es-cut, --skin, --pme-spacing,
--local-cut), --ew-coeff in 1/A, and Amber's names for the coupling constants: --temp (temp0, K),
--gamma (gamma_ln, 1/ps), --tautp (ps), --press (pres0, bar), --barostat-interval (mcbarint).
The time step is --dt-fs (fs; Amber's dt is in ps) and the output options are those of the other
scripts (--report-every, --traj-every, --checkpoint-every, --continue-from).

Usage:

    python scripts/md/run_md.py -p water.prmtop -c water.rst7 -o runs/md --thermostat langevin
        --barostat mc --temp 298 --nsteps 250000 --dt-fs 1.0 --cut 8.0 --nfft 48 48 48 --order 8
        --ew-coeff 0.4 --vdwmeth 0 --dipole-tol 1e-5 --gamma 2.0 --report-every 2000 --traj-every 1000
    pgm-jax md -p water.prmtop -c water.rst7 -o runs/md --continue-from runs/md.chk --nsteps 1000
    python scripts/md/run_md.py --help

Inputs: the prmtop (-p) and coordinates with a periodic box (-c; ASCII or NetCDF restart, inpcrd).
Outputs: <out>.log (energies, temperature, density, solver iterations, speed), <out>.nc (Amber
NetCDF trajectory, --traj-every), <out>.rst7 (Amber NetCDF restart), <out>.chk (checkpoint;
continue with --continue-from <out>.chk); with --dipoles-every N, <out>.dip (cell dipole every N
steps, for scripts/dielectric/dielectric.py); with --multipole-every N, <out>.mpole.nc (per-atom charges,
permanent and induced dipoles).  With --mts N, --dt-fs is the outer step and --nsteps, --report-every, ... count outer
steps.  With --efield-V-nm or --displacement-V-nm the log has the columns efield, field_energy and
the cell dipole Mx My Mz (e nm).
Units: Angstrom and Amber's units as listed above; --dt-fs fs; fields V/nm; frequencies cm^-1.
Runtime: GPU for production (under 1 ms/step for 512 waters, mixed precision, docs/api_design.md
7.3); a few steps of the 512-water box run on a CPU in about a minute.  The script sets
jax_enable_x64 (the engine accumulates in float64 also with --precision mixed).
"""

from __future__ import annotations

import argparse
import math
import sys

import jax

from pgm_jax.cli.args import (
    THERMOSTAT_CHOICES,
    add_iel_args,
    add_mts_args,
    add_output_args,
    add_precision_arg,
    add_seed_arg,
    iel_settings,
    make_coupling,
    mts_from_args,
    setup_logging,
)
from pgm_jax.md.efield import ExternalField, displacement
from pgm_jax.md.forcefield import DSUM_TOL, MDSettings, ewald_beta_for
from pgm_jax.md.simulation import Simulation

jax.config.update("jax_enable_x64", True)


def field_from_args(a: argparse.Namespace) -> ExternalField | None:
    """Return the external field of --efield-V-nm / --displacement-V-nm (None without either).

    Parameters
    ----------
    a : argparse.Namespace
        Parsed options: efield_V_nm (3 floats or None) [V/nm], displacement_V_nm (3 floats or
        None) [V/nm], efield_freq_per_cm [cm^-1].

    Returns
    -------
    ExternalField or None
        E(t) = E0 cos(2 pi c nu t) for --efield-V-nm; constant electric displacement D/eps0 for
        --displacement-V-nm (angular frequency 2 pi c nu [rad/ps], c = 0.0299792458 cm/ps).

    Raises
    ------
    SystemExit
        Both options given.
    """
    if a.efield_V_nm is not None and a.displacement_V_nm is not None:
        raise SystemExit("--efield-V-nm and --displacement-V-nm are exclusive")
    if a.efield_V_nm is not None:
        return ExternalField.from_wavenumber(a.efield_V_nm, a.efield_freq_per_cm)
    if a.displacement_V_nm is not None:
        # angular frequency [rad/ps]: 2 pi c nu with c = 0.0299792458 cm/ps
        return displacement(a.displacement_V_nm, 2 * math.pi * 0.0299792458 * a.efield_freq_per_cm)
    return None


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (Amber-style options; see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--prmtop", required=True, help="Amber prmtop (pGM, or classical with --charges amber)")
    ap.add_argument("-c", "--coords", required=True, help="inpcrd / rst7 (ASCII or NetCDF) with a box")
    ap.add_argument("--nsteps", type=int, default=10000, help="MD steps (nstlim; outer steps with --mts)")
    ap.add_argument("--dt-fs", type=float, default=1.0, help="time step [fs] (outer step with --mts)")
    g = ap.add_argument_group("coupling (Amber names: temp0, gamma_ln, tautp, pres0, mcbarint)")
    g.add_argument(
        "--thermostat",
        default="langevin",
        choices=list(THERMOSTAT_CHOICES),
        help="none (NVE, ntt=0), langevin (ntt=3, --gamma), bussi (ntt=11, --tautp; fastest with pGM), "
        "gle (smooth slow band), gle-lowpass (low-pass GLE, --gamma at zero frequency)",
    )
    g.add_argument("--barostat", default="mc", choices=["none", "mc"], help="mc: Monte Carlo barostat (ntp=1)")
    g.add_argument("--temp", type=float, default=298.0, help="temperature [K] (temp0)")
    g.add_argument("--gamma", type=float, default=2.0, help="Langevin friction [1/ps] (gamma_ln)")
    g.add_argument("--tautp", type=float, default=1.0, help="Bussi time constant [ps] (tautp)")
    g.add_argument("--press", type=float, default=1.0, help="pressure [bar] (pres0)")
    g.add_argument("--barostat-interval", type=int, default=100, help="steps between MC volume moves (mcbarint)")
    g = ap.add_argument_group("nonbonded (Amber units: Angstrom)")
    g.add_argument(
        "--cut", type=float, default=9.0, help="LJ cutoff [A] (vdw_cutoff), and direct space unless --es-cut"
    )
    g.add_argument(
        "--es-cut",
        type=float,
        default=None,
        help="real-space electrostatics cutoff [A] (pmemd es_cutoff); --ew-coeff then defaults to Amber's "
        "dsum_tol rule with --dsum-tol and --pme-spacing to 0.5 (0.4 / ew_coeff)^1.6 A (the grid rule "
        "of elec_cutoff_settings, which keeps the PME force error)",
    )
    g.add_argument(
        "--dsum-tol", type=float, default=DSUM_TOL, help="direct-sum tolerance for --es-cut (Amber dsum_tol)"
    )
    g.add_argument("--skin", type=float, default=1.0, help="neighbour-list skin [A] (skinnb)")
    g.add_argument(
        "--ew-coeff", type=float, default=None, help="Ewald coefficient [1/A] (default 0.4 or from --es-cut)"
    )
    g.add_argument("--nfft", type=int, nargs=3, help="PME grid (nfft1 nfft2 nfft3); default from --pme-spacing")
    g.add_argument(
        "--pme-spacing", type=float, default=None, help="PME grid spacing [A] (default 0.5 or from --es-cut)"
    )
    g.add_argument("--order", type=int, default=8, help="PME B-spline order")
    g.add_argument("--vdwmeth", type=int, default=1, choices=[0, 1], help="1: LJ long-range correction")
    g.add_argument("--long-range", default="pme", choices=["pme", "ips"], help="long-range method (ips: isotropic periodic sum)")
    g.add_argument("--ips-order", type=int, default=4, help="terms of the electrostatic IPS polynomial (2..12; 4 is sander's)")
    g.add_argument("--ips-boundary", action="store_true", help="IPS with DE vdW: add pmemd's constant boundary energy")
    g = ap.add_argument_group("induced dipoles")
    g.add_argument("--dipole-tol", type=float, default=1e-5, help="dipole_scf_tol (pmemd-pgm criterion)")
    g.add_argument("--max-iter", type=int, default=50, help="maximum CG iterations")
    g.add_argument("--local-niter", type=int, default=0, help="inner iterations of the short-range preconditioner")
    g.add_argument("--local-cut", type=float, default=3.0, help="short-range preconditioner cutoff [A]")
    g.add_argument("--peek", type=float, default=0.65, help="scf_sor_coefficient; 0 disables the peek step")
    g.add_argument(
        "--predictor",
        default="mu4",
        choices=["mu4", "mu3", "ls", "none"],
        help="initial dipole guess: mu4/mu3 polynomial extrapolation (fused residual), "
        "ls: pmemd-pgm CPU least squares (dipole_scf_init=3)",
    )
    g.add_argument("--extrap-order", type=int, default=3, help="dipole_scf_init_order (--predictor ls)")
    g.add_argument("--extrap-steps", type=int, default=2, help="dipole_scf_init_step (--predictor ls)")
    add_precision_arg(ap)
    ap.add_argument(
        "--charges",
        default="pgm",
        choices=["pgm", "amber"],
        help="pgm: a pGM prmtop; amber: the point charges of a classical prmtop (e.g. TIP4P-Ew, extra "
        "points as virtual sites), electrostatics level q",
    )
    add_seed_arg(ap)
    ap.add_argument("--no-velocities", action="store_true", help="ignore velocities in the coordinates (irest=0)")
    ap.add_argument(
        "--leapfrog-velocities",
        action="store_true",
        help="the restart velocities are Amber leapfrog velocities v(-dt/2) (sander/pmemd): advance them a half step "
        "to v(0), so the trajectory equals sander's from the same restart",
    )
    add_output_args(ap, out="md", report_every=1000, traj_every=0, checkpoint_every=0, continue_from=True)
    g = ap.add_argument_group("more output")
    g.add_argument("--report-pressure", action="store_true", help="also report the virial pressure")
    g.add_argument(
        "--dipoles-every", type=int, default=0, help="steps between cell-dipole samples (<out>.dip; 0: none)"
    )
    g.add_argument(
        "--multipole-every",
        type=int,
        default=0,
        help="steps between per-atom multipole frames: charges, permanent and induced dipoles (<out>.mpole.nc)",
    )
    g = ap.add_argument_group("external electric field (docs/efield.md)")
    g.add_argument("--efield-V-nm", type=float, nargs=3, help="uniform external field E0 [V/nm]")
    g.add_argument(
        "--efield-freq-per-cm",
        type=float,
        default=0.0,
        help="frequency nu [cm^-1] of E(t) = E0 cos(2 pi c nu t) (0: static)",
    )
    g.add_argument("--displacement-V-nm", type=float, nargs=3, help="constant electric displacement D/eps0 [V/nm]")
    add_mts_args(ap, "--dt-fs")
    add_iel_args(ap)
    return ap


def settings_from_args(a: argparse.Namespace) -> MDSettings:
    """Return the MDSettings of the Amber-style options (Angstrom converted to nm).

    Parameters
    ----------
    a : argparse.Namespace
        Parsed options; `a.ew_coeff` [1/A] and `a.pme_spacing` [A] are filled in when they are None
        (from --es-cut and --dsum-tol, see the --es-cut help).

    Returns
    -------
    MDSettings
        Force-field settings in library units (nm, 1/nm).
    """
    if a.ew_coeff is None:
        a.ew_coeff = 0.4 if a.es_cut is None else ewald_beta_for(a.es_cut / 10, a.dsum_tol) / 10
    if a.pme_spacing is None:
        a.pme_spacing = 0.5 if a.es_cut is None else 0.5 * (0.4 / a.ew_coeff) ** 1.6
    return MDSettings().replace(
        cutoff=a.cut / 10,
        skin=a.skin / 10,
        ewald_beta=a.ew_coeff * 10,
        elec_cutoff=None if a.es_cut is None else a.es_cut / 10,
        pme_grid=tuple(a.nfft) if a.nfft else None,
        pme_spacing=a.pme_spacing / 10,
        pme_order=a.order,
        lj_lrc=bool(a.vdwmeth),
        long_range=a.long_range,
        ips_order=a.ips_order,
        ips_boundary=a.ips_boundary,
        dipole_tol=a.dipole_tol,
        max_iter=a.max_iter,
        local_cut=a.local_cut / 10,
        local_niter=a.local_niter,
        peek=a.peek,
        predictor=a.predictor,
        extrap_order=a.extrap_order,
        extrap_steps=a.extrap_steps,
        precision=a.precision,
        elec="q" if a.charges == "amber" else "qpi",
        **iel_settings(a),
    )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the simulation (see the module docstring)."""
    a = build_parser().parse_args(argv)
    setup_logging()
    st = settings_from_args(a)
    thermostat, barostat = make_coupling(a.thermostat, a.gamma, a.tautp, a.barostat, a.press, a.barostat_interval)
    sim = Simulation.from_amber(
        a.prmtop,
        a.coords,
        use_velocities=not a.no_velocities,
        leapfrog_velocities=a.leapfrog_velocities,
        settings=st,
        charges=a.charges,
        dt=a.dt_fs / 1000,
        temperature=a.temp,
        thermostat=thermostat,
        barostat=barostat,
        seed=a.seed,
        log=sys.stdout,
        mts=mts_from_args(a),
        efield=field_from_args(a),
    )
    if a.continue_from:
        sim.load_checkpoint(a.continue_from)
    sim.run(
        a.nsteps,
        report_every=a.report_every,
        traj_every=a.traj_every,
        checkpoint_every=a.checkpoint_every,
        prefix=a.out,
        report_pressure=a.report_pressure,
        append=bool(a.continue_from),
        dipoles_every=a.dipoles_every,
        multipole_every=a.multipole_every,
    )


if __name__ == "__main__":
    main()
