"""Command-line option groups shared by the scripts and the `pgm-jax` entry point.

One option has one name, one unit and one help text everywhere (docs/api_design.md, decision D2):
options in the library units have the unit in their name (`--dt-fs`, `--cutoff-nm`,
`--temperature-K`, `--pressure-bar`, `--friction-per-ps`, `--tau-ps`, `--time-ns`), intervals
counted in steps end in `-every` (`--report-every`, `--checkpoint-every`), and the output prefix
is `--out`.  scripts/md/run_md.py is the one exception: it keeps Amber's names and units
(Angstrom, `--temp`, `--gamma`, `--tautp`, `--press`) for comparisons with pmemd.

Contents:

- `setup_logging`: print the library's diagnostics (loggers under "pgm_jax") as "# ..." lines.
- Option groups (each adds options to an `argparse.ArgumentParser` and has a matching
  `*_from_args` reader where the options build an object):
  `add_temperature_arg`, `add_thermostat_args` / `thermostat_from_args`, `add_barostat_args` /
  `barostat_from_args`, `coupling_from_args` (both), `add_dt_arg`, `add_seed_arg`,
  `add_precision_arg`, `add_dipole_tol_arg`, `add_cutoff_arg`, `add_output_args`,
  `add_iel_args` / `iel_settings`, `add_mts_args` / `mts_from_args`.
- `make_coupling`: the thermostat and barostat objects from plain values (used by the readers and
  by scripts that build several couplings).

Units: K, bar, 1/ps, ps, fs (time steps on the command line), nm; converted to the library units
(ps for time steps) by the scripts, `dt = dt_fs / 1000` (a correctly rounded division, so that
`--dt-fs 0.5` gives exactly the double 0.0005).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TextIO

from ..md.barostats import MonteCarloBarostat
from ..md.mts import MTS
from ..md.thermostats import GLE, Bussi, Langevin, Thermostat, make_thermostat

#: Values of --thermostat: "none" (NVE), Langevin, Bussi (CSVR), GLE (smooth slow band), GLE low-pass.
THERMOSTAT_CHOICES = ("none", "langevin", "bussi", "gle", "gle-lowpass")

#: Values of --barostat: "none" (constant volume) or "mc" (isotropic Monte Carlo barostat).
BAROSTAT_CHOICES = ("none", "mc")


def setup_logging(level: int = logging.INFO, stream: TextIO = sys.stdout) -> logging.Logger:
    """Show pgm_jax's diagnostics (engine setup, list rebuilds, resizes) on a stream.

    The library logs through loggers under "pgm_jax" and configures no handler; scripts call
    this once.  Messages are printed as "# <message>" so that they read as comments next to the
    log tables.

    Parameters
    ----------
    level : int
        Logging level of the "pgm_jax" logger (default INFO).
    stream : text stream
        Where the messages go (default sys.stdout).

    Returns
    -------
    logging.Logger
        The "pgm_jax" logger.
    """
    logger = logging.getLogger("pgm_jax")
    logger.setLevel(level)
    if not any(getattr(h, "_pgm_jax", False) for h in logger.handlers):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("# %(message)s"))
        handler._pgm_jax = True
        logger.addHandler(handler)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------------------------------
# thermostat and barostat


def make_coupling(
    thermostat: str = "langevin",
    friction: float = 1.0,
    tau: float = 1.0,
    barostat: str = "none",
    pressure: float = 1.0,
    barostat_every: int = 100,
) -> tuple[Thermostat | None, MonteCarloBarostat | None]:
    """Build the thermostat and barostat objects from their names and parameters.

    Parameters
    ----------
    thermostat : {"none", "langevin", "bussi", "gle", "gle-lowpass"}
        "none": no thermostat (NVE; md.barostats.ensemble_name refuses a barostat without one);
        "bussi" also accepts "csvr" and "v-rescale", "gle-lowpass" also "lowpass".
    friction : float
        Langevin friction [1/ps] (also the zero-frequency friction of "gle-lowpass").
    tau : float
        Bussi time constant [ps].
    barostat : {"none", "mc"}
        "mc": isotropic Monte Carlo barostat.
    pressure : float
        Barostat pressure [bar].
    barostat_every : int
        Steps between Monte Carlo volume moves [steps].

    Returns
    -------
    thermostat : Thermostat or None
        For the engines' `thermostat=` keyword (`Langevin(friction)`, `Bussi(tau)`, `GLE.band()`,
        `GLE.lowpass(friction)`, or None).
    barostat : MonteCarloBarostat or None
        For the engines' `barostat=` keyword.

    Raises
    ------
    ValueError
        An unknown thermostat or barostat name.
    """
    name = str(thermostat).lower()
    if name in ("none", "nve"):
        th = None
    elif name == "langevin":
        th = Langevin(friction)
    elif name in ("bussi", "csvr", "v-rescale"):
        th = Bussi(tau)
    elif name in ("gle-lowpass", "lowpass"):
        th = GLE.lowpass(friction)
    else:
        th = make_thermostat(name)  # "gle": GLE.band(); raises ValueError for unknown names
    baro = str(barostat).lower()
    if baro not in BAROSTAT_CHOICES:
        raise ValueError(f"barostat: 'none' or 'mc', not {barostat!r}")
    return th, (MonteCarloBarostat(pressure, barostat_every) if baro == "mc" else None)


def add_temperature_arg(ap: argparse.ArgumentParser, default: float | None = 298.0, help: str = "") -> None:
    """Add --temperature-K (thermostat / ensemble temperature [K]).

    Parameters
    ----------
    ap : argparse.ArgumentParser
        The parser (or argument group).
    default : float, optional
        Default temperature [K]; None makes the option required.
    help : str
        Help text (default: "temperature [K]").
    """
    ap.add_argument(
        "--temperature-K",
        dest="temperature_K",
        type=float,
        default=default,
        required=default is None,
        help=help or "temperature [K]",
    )


def add_thermostat_args(
    ap: argparse.ArgumentParser,
    default: str = "langevin",
    friction: float = 1.0,
    tau: float = 1.0,
    choices: tuple[str, ...] = THERMOSTAT_CHOICES,
) -> None:
    """Add --thermostat, --friction-per-ps and --tau-ps.

    Parameters
    ----------
    ap : argparse.ArgumentParser
        The parser.
    default : str
        Default thermostat (one of `choices`).
    friction : float
        Default Langevin friction [1/ps].
    tau : float
        Default Bussi time constant [ps].
    choices : tuple of str
        Allowed values of --thermostat (default THERMOSTAT_CHOICES; drop "none" where the script
        needs a thermostat).
    """
    g = ap.add_argument_group("thermostat")
    g.add_argument(
        "--thermostat",
        default=default,
        choices=list(choices),
        help="none (NVE), langevin (friction --friction-per-ps), bussi (CSVR, time constant --tau-ps), "
        "gle (smooth slow band, GLE.band()), gle-lowpass (low-pass GLE with --friction-per-ps at zero "
        "frequency)",
    )
    g.add_argument("--friction-per-ps", type=float, default=friction, help="Langevin friction [1/ps]")
    g.add_argument("--tau-ps", type=float, default=tau, help="Bussi time constant [ps]")


def thermostat_from_args(a: argparse.Namespace) -> Thermostat | None:
    """Return the thermostat of add_thermostat_args' options (None for --thermostat none)."""
    return make_coupling(a.thermostat, a.friction_per_ps, a.tau_ps)[0]


def add_barostat_args(
    ap: argparse.ArgumentParser, default: str = "none", pressure: float = 1.0, every: int = 100
) -> None:
    """Add --barostat, --pressure-bar and --barostat-every.

    Parameters
    ----------
    ap : argparse.ArgumentParser
        The parser.
    default : {"none", "mc"}
        Default barostat.
    pressure : float
        Default pressure [bar].
    every : int
        Default number of steps between Monte Carlo volume moves [steps].
    """
    g = ap.add_argument_group("barostat")
    g.add_argument(
        "--barostat", default=default, choices=list(BAROSTAT_CHOICES), help="none (constant volume) or mc (NPT)"
    )
    g.add_argument("--pressure-bar", type=float, default=pressure, help="barostat pressure [bar]")
    g.add_argument("--barostat-every", type=int, default=every, help="steps between Monte Carlo volume moves")


def barostat_from_args(a: argparse.Namespace) -> MonteCarloBarostat | None:
    """Return the barostat of add_barostat_args' options (None for --barostat none)."""
    return make_coupling("none", barostat=a.barostat, pressure=a.pressure_bar, barostat_every=a.barostat_every)[1]


def coupling_from_args(a: argparse.Namespace) -> tuple[Thermostat | None, MonteCarloBarostat | None]:
    """Return (thermostat, barostat) of add_thermostat_args' and add_barostat_args' options."""
    return make_coupling(a.thermostat, a.friction_per_ps, a.tau_ps, a.barostat, a.pressure_bar, a.barostat_every)


# --------------------------------------------------------------------------------------------------
# single options


def add_dt_arg(ap: argparse.ArgumentParser, default: float = 1.0, help: str = "") -> None:
    """Add --dt-fs (time step [fs]; the scripts pass `a.dt_fs / 1000` [ps] to the engines).

    Parameters
    ----------
    ap : argparse.ArgumentParser
        The parser.
    default : float
        Default time step [fs].
    help : str
        Help text (default: "time step [fs]").
    """
    ap.add_argument("--dt-fs", type=float, default=default, help=help or "time step [fs]")


def add_seed_arg(ap: argparse.ArgumentParser, default: int = 0) -> None:
    """Add --seed (random seed of velocities, thermostat noise and Monte Carlo moves)."""
    ap.add_argument("--seed", type=int, default=default, help="random seed")


def add_precision_arg(ap: argparse.ArgumentParser, default: str = "mixed") -> None:
    """Add --precision {mixed, double} (MDSettings.precision)."""
    ap.add_argument(
        "--precision",
        default=default,
        choices=["mixed", "double"],
        help="MDSettings.precision: mixed (float32 kernels, float64 accumulation where it matters) or double",
    )


def add_dipole_tol_arg(ap: argparse.ArgumentParser, default: float = 1e-5, help: str = "") -> None:
    """Add --dipole-tol (convergence tolerance of the induced dipoles, MDSettings.dipole_tol)."""
    ap.add_argument(
        "--dipole-tol",
        type=float,
        default=default,
        help=help or "induced-dipole tolerance (pmemd-pgm's RMS criterion, dimensionless)",
    )


def add_cutoff_arg(ap: argparse.ArgumentParser, default: float = 0.9, help: str = "") -> None:
    """Add --cutoff-nm (real-space and van der Waals cutoff [nm])."""
    ap.add_argument("--cutoff-nm", type=float, default=default, help=help or "real-space and van der Waals cutoff [nm]")


def add_output_args(
    ap: argparse.ArgumentParser,
    out: str | None = "md",
    report_every: int | None = None,
    traj_every: int | None = None,
    checkpoint_every: int | None = None,
    continue_from: bool = False,
) -> None:
    """Add the output options: --out and, where given a default, the step intervals.

    Parameters
    ----------
    ap : argparse.ArgumentParser
        The parser.
    out : str, optional
        Default output prefix (files are <out>.log, <out>.nc, ...); None makes --out required.
    report_every : int, optional
        Default of --report-every [steps] (None: no such option).
    traj_every : int, optional
        Default of --traj-every [steps] (None: no such option).
    checkpoint_every : int, optional
        Default of --checkpoint-every [steps] (None: no such option).
    continue_from : bool
        Add --continue-from (a checkpoint file to continue from).
    """
    g = ap.add_argument_group("output")
    g.add_argument("-o", "--out", default=out, required=out is None, help="output prefix (directories are created)")
    if report_every is not None:
        g.add_argument("--report-every", type=int, default=report_every, help="steps between log lines")
    if traj_every is not None:
        g.add_argument("--traj-every", type=int, default=traj_every, help="steps between trajectory frames (0: none)")
    if checkpoint_every is not None:
        g.add_argument(
            "--checkpoint-every",
            type=int,
            default=checkpoint_every,
            help="steps between checkpoint (and restart) files (0: only at the end)",
        )
    if continue_from:
        g.add_argument("--continue-from", default=None, help="continue from this checkpoint file")


# --------------------------------------------------------------------------------------------------
# extended-Lagrangian dipoles and multiple time stepping


def add_iel_args(ap: argparse.ArgumentParser) -> None:
    """Add the extended-Lagrangian induced-dipole options (docs/iel.md).

    --iel, --iel-iter, --iel-order, --iel-kappa, --iel-alpha, --iel-omega, --iel-precond,
    --iel-no-shadow; read them with iel_settings.
    """
    g = ap.add_argument_group("extended-Lagrangian induced dipoles (docs/iel.md)")
    g.add_argument(
        "--iel",
        default="none",
        choices=["none", "0scf", "scf"],
        help="extended-Lagrangian induced dipoles: 0scf (iEL/0-SCF, no CG, shadow forces) or scf "
        "(--iel-iter CG iterations from the auxiliary dipoles); none: predictor + CG to tolerance",
    )
    g.add_argument("--iel-iter", type=int, default=1, help="--iel scf: CG iterations per step (0: to tolerance)")
    g.add_argument("--iel-order", type=int, default=7, help="Niklasson dissipation order K (0, 3..9)")
    g.add_argument(
        "--iel-kappa", type=float, default=None, help="kappa = (omega dt)^2, dimensionless (default: Niklasson's for K)"
    )
    g.add_argument("--iel-alpha", type=float, default=None, help="dissipation strength (default: Niklasson's for K)")
    g.add_argument("--iel-omega", type=float, default=1.0, help="--iel 0scf: mu = x + omega alpha r(x)")
    g.add_argument(
        "--iel-precond",
        default="block",
        choices=["jacobi", "block"],
        help="--iel 0scf: delta = alpha r (jacobi) or M^-1 r with the intramolecular blocks (block)",
    )
    g.add_argument(
        "--iel-no-shadow",
        action="store_true",
        help="--iel 0scf: fixed-dipole forces at mu = x + alpha r instead of the exact forces of the shadow energy",
    )


def iel_settings(a: argparse.Namespace) -> dict:
    """Return the flat MDSettings keywords (for `MDSettings().replace(**...)`) of add_iel_args' options."""
    return dict(
        iel=a.iel,
        iel_iter=a.iel_iter,
        iel_order=a.iel_order,
        iel_kappa=a.iel_kappa,
        iel_alpha=a.iel_alpha,
        iel_shadow=not a.iel_no_shadow,
        iel_omega=a.iel_omega,
        iel_precond=a.iel_precond,
    )


def add_mts_args(ap: argparse.ArgumentParser, dt_option: str = "--dt-fs") -> None:
    """Add --mts and its options (r-RESPA; docs/mts.md); read them with mts_from_args.

    Parameters
    ----------
    ap : argparse.ArgumentParser
        The parser.
    dt_option : str
        The script's time-step option, named in the help (it is the outer step with --mts).
    """
    g = ap.add_argument_group("multiple time stepping (r-RESPA; docs/mts.md)")
    g.add_argument(
        "--mts", type=int, default=0, help=f"fast steps per outer step (0: off; {dt_option} is then the outer step)"
    )
    g.add_argument(
        "--mts-split",
        default="short",
        choices=["short", "special", "bonded"],
        help="fast group: short (short-range pGM model + bonded), special (the special pairs + bonded; "
        "flexible engine), bonded (bonded terms; flexible engine)",
    )
    g.add_argument(
        "--mts-r-short-nm", type=float, default=0.5, help="fast nonbonded pairs are switched off at this distance [nm]"
    )
    g.add_argument("--mts-width-nm", type=float, default=0.1, help="switch width [nm]")
    g.add_argument("--mts-buffer-nm", type=float, default=0.1, help="buffer of the short-range pair list [nm]")
    g.add_argument(
        "--mts-beta-per-nm",
        type=float,
        default=None,
        help="screening of the fast electrostatics [1/nm] (default: erfc(beta r_short) = 1e-3)",
    )
    g.add_argument(
        "--mts-pol", default="auto", choices=["auto", "mutual", "direct", "none"], help="fast-level induced dipoles"
    )
    g.add_argument("--mts-bonded", type=int, default=1, help="bonded steps per fast step (> 1: three levels)")
    g.add_argument(
        "--mts-o", default="outer", choices=["outer", "inner"], help="thermostat step at the outer or innermost level"
    )
    g.add_argument("--mts-anchor", type=int, default=-1, help="predictor anchored on the fast dipoles: 1, 0, -1 (auto)")


def mts_from_args(a: argparse.Namespace) -> MTS | None:
    """Return the MTS settings of add_mts_args' options (None without --mts)."""
    if not a.mts:
        return None
    return MTS(
        inner=a.mts,
        split=a.mts_split,
        r_short=a.mts_r_short_nm,
        switch_width=a.mts_width_nm,
        buffer=a.mts_buffer_nm,
        beta_short=a.mts_beta_per_nm,
        polarization=a.mts_pol,
        bonded=a.mts_bonded,
        o_step=a.mts_o,
        anchor=None if a.mts_anchor < 0 else bool(a.mts_anchor),
    )
