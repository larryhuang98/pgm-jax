"""argparse option groups shared by the scripts, so that one option has one name, one unit and one
help text everywhere (docs/api_design.md, decision D2)."""

from __future__ import annotations

import logging
import sys

from ..md.barostats import MonteCarloBarostat
from ..md.mts import MTS
from ..md.thermostats import GLE, Bussi, Langevin, Thermostat, make_thermostat


def setup_logging(level: int = logging.INFO, stream=sys.stdout) -> logging.Logger:
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


def coupling_from_options(
    ensemble: str,
    thermostat: str = "langevin",
    friction: float = 1.0,
    tau: float = 1.0,
    pressure: float = 1.0,
    barostat_every: int = 100,
) -> tuple[Thermostat | None, MonteCarloBarostat | None]:
    """Thermostat and barostat objects for the ensemble options of a command line.

    Parameters
    ----------
    ensemble : str
        "nve", "nvt" or "npt".
    thermostat : str
        "langevin", "bussi" (or "csvr", "v-rescale"), "gle" or "gle-lowpass".
    friction : float
        Langevin friction [1/ps] (also the zero-frequency friction of "gle-lowpass").
    tau : float
        Bussi time constant [ps].
    pressure : float
        Barostat pressure [bar] (npt).
    barostat_every : int
        Steps between Monte Carlo volume moves (npt).

    Returns
    -------
    (Thermostat or None, MonteCarloBarostat or None)
        For the engines' thermostat= and barostat= keywords.

    Raises
    ------
    ValueError
        An unknown ensemble or thermostat.
    """
    ens = str(ensemble).lower()
    if ens not in ("nve", "nvt", "npt"):
        raise ValueError(f"ensemble: 'nve', 'nvt' or 'npt', not {ensemble!r}")
    if ens == "nve":
        return None, None
    name = str(thermostat).lower()
    if name == "langevin":
        th = Langevin(friction)
    elif name in ("bussi", "csvr", "v-rescale"):
        th = Bussi(tau)
    elif name in ("gle-lowpass", "lowpass"):
        th = GLE.lowpass(friction)
    else:
        th = make_thermostat(name)
    return th, (MonteCarloBarostat(pressure, barostat_every) if ens == "npt" else None)


def add_iel_arguments(ap):
    """--iel, --iel-iter, --iel-order, --iel-kappa, --iel-alpha."""
    ap.add_argument(
        "--iel",
        default="none",
        choices=["none", "0scf", "scf"],
        help="extended-Lagrangian induced dipoles: 0scf (iEL/0-SCF, no CG, shadow forces) or scf "
        "(--iel-iter CG iterations from the auxiliary dipoles); none: predictor + CG to tolerance",
    )
    ap.add_argument("--iel-iter", type=int, default=1, help="--iel scf: CG iterations per step (0: to tolerance)")
    ap.add_argument("--iel-order", type=int, default=7, help="Niklasson dissipation order K (0, 3..9)")
    ap.add_argument("--iel-kappa", type=float, default=None, help="kappa = (omega dt)^2 (default: Niklasson's for K)")
    ap.add_argument("--iel-alpha", type=float, default=None, help="dissipation strength (default: Niklasson's for K)")
    ap.add_argument("--iel-omega", type=float, default=1.0, help="--iel 0scf: mu = x + omega alpha r(x)")
    ap.add_argument(
        "--iel-precond",
        default="block",
        choices=["jacobi", "block"],
        help="--iel 0scf: delta = alpha r (jacobi) or M^-1 r with the intramolecular blocks (block)",
    )
    ap.add_argument(
        "--iel-no-shadow",
        action="store_true",
        help="--iel 0scf: fixed-dipole forces at mu = x + alpha r instead of the exact forces of the shadow energy",
    )


def iel_settings(a) -> dict:
    """MDSettings keyword arguments from add_iel_arguments' options."""
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


def add_mts_arguments(ap, dt_help: str = "--dt") -> None:
    """--mts and friends for the MD scripts (the time step given by `dt_help` is the outer step)."""
    g = ap.add_argument_group("multiple time stepping (r-RESPA; docs/mts.md)")
    g.add_argument(
        "--mts", type=int, default=0, help=f"fast steps per outer step (0: off; {dt_help} is then the outer step)"
    )
    g.add_argument(
        "--mts-split",
        default="short",
        choices=["short", "special", "bonded"],
        help="fast group: short (short-range pGM model + bonded), special (the special pairs + bonded; "
        "flexible engine), bonded (bonded terms; flexible engine)",
    )
    g.add_argument("--mts-rs", type=float, default=0.5, help="nm: fast nonbonded pairs switched off at this distance")
    g.add_argument("--mts-width", type=float, default=0.1, help="nm: switch width")
    g.add_argument("--mts-buffer", type=float, default=0.1, help="nm: buffer of the short-range pair list")
    g.add_argument(
        "--mts-beta",
        type=float,
        default=None,
        help="1/nm: screening of the fast electrostatics (default: erfc(beta r_short) = 1e-3)",
    )
    g.add_argument(
        "--mts-pol", default="auto", choices=["auto", "mutual", "direct", "none"], help="fast-level induced dipoles"
    )
    g.add_argument("--mts-bonded", type=int, default=1, help="bonded steps per fast step (> 1: three levels)")
    g.add_argument(
        "--mts-o", default="outer", choices=["outer", "inner"], help="thermostat step at the outer or innermost level"
    )
    g.add_argument("--mts-anchor", type=int, default=-1, help="predictor anchored on the fast dipoles: 1, 0, -1 (auto)")


def mts_from_args(a) -> MTS | None:
    """MTS settings from add_mts_arguments' options (None without --mts)."""
    if not a.mts:
        return None
    return MTS(
        inner=a.mts,
        split=a.mts_split,
        r_short=a.mts_rs,
        switch_width=a.mts_width,
        buffer=a.mts_buffer,
        beta_short=a.mts_beta,
        polarization=a.mts_pol,
        bonded=a.mts_bonded,
        o_step=a.mts_o,
        anchor=None if a.mts_anchor < 0 else bool(a.mts_anchor),
    )
