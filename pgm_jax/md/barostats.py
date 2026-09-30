"""Barostat configuration for the MD engines.

`MonteCarloBarostat` is the isotropic Monte Carlo barostat of the integrators (Amber barostat = 2,
OpenMM MonteCarloBarostat): every `every` steps a random volume change is tried, the centres of
mass of the molecules and the box are scaled (orientations, internal geometries and momenta are
kept), and the move is accepted with probability min(1, exp(-(dU + P dV - N kT ln(V'/V)) / kT)).
The maximum volume change adapts to an acceptance of 25-75 %.  The move itself is part of the
compiled step of each integrator (md/integrate.py, md/flexible.py, md/mts.py, md/pimd.py); this
object only carries its settings.

    sim = Simulation(system, positions, box, barostat=MonteCarloBarostat(pressure=1.0, every=100))

Units: bar, steps.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .thermostats import Thermostat


@dataclasses.dataclass(frozen=True)
class MonteCarloBarostat:
    """Isotropic Monte Carlo barostat (see the module docstring).

    Immutable (frozen dataclass); not a pytree: the integrators read its fields when they build
    their compiled step, so a change recompiles.

    Parameters
    ----------
    pressure : float
        Target pressure [bar].
    every : int
        Steps between volume moves (Amber's barostat interval, OpenMM's frequency).

    Raises
    ------
    ValueError
        every < 1 or a non-finite pressure.
    """

    pressure: float = 1.0
    every: int = 100

    def __post_init__(self) -> None:
        """Check the settings (frozen dataclass: values are normalized with object.__setattr__)."""
        object.__setattr__(self, "pressure", float(self.pressure))
        object.__setattr__(self, "every", int(self.every))
        if self.every < 1:
            raise ValueError(f"MonteCarloBarostat: every must be >= 1 step ({self.every!r})")
        if not self.pressure == self.pressure or abs(self.pressure) == float("inf"):  # NaN or +-inf
            raise ValueError(f"MonteCarloBarostat: pressure must be finite ({self.pressure!r} bar)")

    def describe(self) -> str:
        """Return one line for the log header, e.g. "Monte Carlo barostat 1 bar every 100 steps"."""
        return f"Monte Carlo barostat {self.pressure:g} bar every {self.every} steps"


def ensemble_name(thermostat: Thermostat | None, barostat: MonteCarloBarostat | None) -> str:
    """Return the ensemble a thermostat / barostat pair samples.

    Parameters
    ----------
    thermostat : Thermostat or None
        The thermostat (md/thermostats.py; None: none).
    barostat : MonteCarloBarostat or None
        The barostat (None: constant volume).

    Returns
    -------
    str
        "nve", "nvt" or "npt" (the integrators' internal switch).

    Raises
    ------
    ValueError
        A barostat without a thermostat (the Monte Carlo barostat needs a temperature bath).
    """
    if barostat is not None and thermostat is None:
        raise ValueError("barostat without thermostat: constant pressure needs a thermostat (NPT)")
    if thermostat is None:
        return "nve"
    return "npt" if barostat is not None else "nvt"
