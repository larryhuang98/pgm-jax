"""Simulation driver: Amber inputs in, Amber-readable outputs out.

Contents: `Simulation`, the rigid-body MD engine (an engine.MDEngine), with
`Simulation.from_amber`.

    sim = Simulation.from_amber("water.prmtop", "water.rst7", settings=MDSettings(...), temperature=298.0,
                                dt=0.001, thermostat=Bussi(tau=1.0), barostat=MonteCarloBarostat(pressure=1.0))
    sim.run(100000, prefix="md", report_every=1000, traj_every=1000, checkpoint_every=10000)

Steps run in jit-compiled blocks on the device; between blocks the host checks the neighbour
list (reallocates and repeats the block on overflow), re-wraps molecules into the box, reports
and writes files.  Molecules are the prmtop residues; identical residues share one template.
Virtual sites (Amber extra points, Molecule.vsites; md/vsites.py) are massless points of the rigid
templates, placed from their parents at the start.
Options: run(dipoles_every=n) also samples the cell dipole every n steps (on the device, inside the blocks)
into prefix.dip, and run(multipole_every=n) writes per-atom charges and dipoles to prefix.mpole.nc
(md/dipoles.py).
mts=MTS(...) integrates force groups with their own time steps (md/mts.py; dt is the outer step).
bias=... adds biases on collective variables (pgm_jax.bias: metadynamics, OPES, static biases);
run() then writes prefix.colvar, prefix.hills and, with the checkpoints, prefix.bias (bias/io.py).

Units: nm, ps, K, bar, kJ/mol; Amber files in Angstrom.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TextIO

import jax
import jax.numpy as jnp
import numpy as np

from ..system import System
from .barostats import MonteCarloBarostat
from .box import check_box, reduce_box
from .engine import MDEngine
from .forcefield import MDSettings, PGMForceField
from .integrate import Integrator
from .geometry import conform_rigid_geometry
from .io import read_coordinates_nm
from .rigid import RigidMolecules
from .thermostats import Thermostat
from .vsites import VirtualSites

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from .alchemy import Alchemy
    from .efield import ExternalField
    from .mts import MTS
    from .restraints import Restraint, Restraints


class Simulation(MDEngine):
    """Molecular dynamics of rigid molecules (JAX-MD rigid bodies).

    Every molecule of `sys` is rigid at its template geometry; virtual sites are points of the
    templates.  The shared machinery (blocks, observables, run loop, checkpoints) is MDEngine's
    (md/engine.py); the integrator is integrate.Integrator (mts.MTSIntegrator with `mts`).

    Attributes
    ----------
    vsites : VirtualSites or None
        The virtual sites.
    rigid : RigidMolecules
        Body frames and the atom-position map.

    Other attributes as MDEngine.
    """

    checkpoint_kind = "md-rigid"

    def __init__(
        self,
        system: System,
        positions: ArrayLike,
        box: ArrayLike,
        settings: MDSettings = MDSettings(),
        *,
        dt: float = 0.001,
        temperature: float = 298.0,
        thermostat: Thermostat | str | None = "langevin",
        barostat: MonteCarloBarostat | None = None,
        velocities: ArrayLike | None = None,
        leapfrog_velocities: bool = False,
        seed: int = 0,
        params: dict | None = None,
        restraints: Restraints | Restraint | list | None = None,
        alchemy: Alchemy | None = None,
        mts: MTS | None = None,
        bias: Any = None,
        efield: ExternalField | ArrayLike | None = None,
        log: TextIO | None = None,
    ) -> None:
        """Set up rigid-body MD of `system` at `positions` in `box`.

        Parameters
        ----------
        system : System
            Molecules (every one rigid at its template geometry).
        positions : ArrayLike (N, 3)
            Atom positions [nm] (virtual sites are placed from their parents).
        box : ArrayLike (3, 3)
            Box [nm], lattice vectors as rows, lower triangular (reduced to the canonical form of
            box.reduce_box).
        settings : MDSettings
            Force-field, cutoff, PME and solver settings.
        dt : float
            Time step [ps] (the outer step with mts).
        temperature : float
            Temperature [K] of the thermostat, the barostat and drawn momenta.
        thermostat : Thermostat, str or None
            Langevin(friction), Bussi(tau), GLE..., a name for the default settings of a kind
            ("langevin" = Langevin(1/ps), the default; "bussi"; "gle"; "gle-lowpass"), or None for
            NVE (thermostats.make_thermostat).
        barostat : MonteCarloBarostat or None
            Isotropic Monte Carlo barostat (None: constant volume); needs a thermostat.
        velocities : ArrayLike (N, 3), optional
            Atom velocities [nm/ps] (their rigid-body part is kept); default: drawn at the
            temperature.
        leapfrog_velocities : bool
            Treat `velocities` as Amber's leapfrog velocities v(-dt/2) and advance them by a half kick
            (with the forces at step 0) to v(0), which is what this velocity-Verlet integrator starts from.
        seed : int
            Seed of the random stream (momenta, thermostat, barostat).
        params : dict, optional
            Force-field parameters (default: those of the system).
        restraints, alchemy, mts, bias, efield : optional
            Restraints (md/restraints.py), an alchemical region (md/alchemy.py), multiple time
            stepping (md/mts.py), biases on collective variables (pgm_jax.bias), an external field
            (md/efield.py: ExternalField or three numbers in V/nm).
        log : text stream or None
            Receives the rows of the log table of `run` too (diagnostics go to the logger
            "pgm_jax.md.simulation").

        Raises
        ------
        ValueError
            A box too small for the cutoff, invalid options or combinations.
        """
        H = reduce_box(box)
        check_box(H, settings.pair_cutoff + settings.neighbors.skin)
        self.sys, self.settings, self.log = system, settings, log
        self.vsites = VirtualSites.of(system)
        pos_nm = positions
        if self.vsites is not None:  # sites of the rigid templates from their parents
            pos_nm = np.asarray(self.vsites.place(np.asarray(pos_nm, float), H))
            self.vsites.check(pos_nm, H, (system.cov_i, system.cov_j))
        self.rigid = RigidMolecules(system, pos_nm, H)
        self.ff = PGMForceField(system, H, settings)
        self._r_list = float(jnp.max(jnp.linalg.norm(self.rigid.local, axis=1)))
        self._make_neighbors(H)
        self._size_lists(self.rigid.body0, H)
        integ, extra = Integrator, {}
        if mts is not None:  # multiple time stepping: dt is the outer step
            from .mts import MTSIntegrator

            integ, extra = MTSIntegrator, {"mts": mts}
        self.integ = integ(
            self.ff,
            self.rigid,
            self.nb,
            dt,
            temperature,
            thermostat,
            barostat,
            params,
            restraints=restraints,
            alchemy=alchemy,
            bias=bias,
            efield=efield,
            **extra,
        )
        self.dt, self.ensemble, self.T0 = dt, self.integ.ensemble, temperature
        body = self.rigid.body0
        mom = None
        if velocities is not None:
            mom = self.rigid.momenta_from_velocities(body, self.rigid.positions(body), jnp.asarray(velocities))
        self.state = self.integ.init(body, H, jax.random.PRNGKey(seed), mom)
        if leapfrog_velocities and mom is not None:  # Amber's velocities are v(-dt/2): v(0) = v(-dt/2) + a dt/2
            from jax_md import simulate

            st = self.state
            self.state = st.set(dyn=simulate.momentum_step(st.dyn, dt / 2))
        self.time_ps = 0.0
        self._log.info(
            f"pgm_jax MD: {system.nmol} rigid molecules, {system.n} atoms"
            f"{'' if self.vsites is None else f' ({self.vsites.n_sites} virtual sites)'}, "
            f"{self._describe_coupling()}, dt {dt * 1000:g} fs, "
            f"{settings.precision} precision, PME grid {self.ff.pme.K} order {settings.pme.order}, "
            f"{settings.describe_cutoffs()}, {self.nb.kind} neighbour list, {settings.describe_induction()}, "
            f"template fit RMSD {self.rigid.fit_rmsd:.2e} nm, device {jax.devices()[0]}"
        )
        self._describe_options(alchemy, mts)

    @classmethod
    def from_amber(
        cls,
        prmtop: str,
        coords: str,
        use_velocities: bool = True,
        charges: str = "pgm",
        conform_geometry: bool = True,
        **kw: Any,
    ) -> Simulation:
        """Return a simulation of the system of an Amber prmtop at the coordinates of a restart / inpcrd.

        Parameters
        ----------
        prmtop : str
            The prmtop (residues become rigid molecules; identical residues share one template;
            extra points become virtual sites).
        coords : str
            Coordinates with a periodic box (ASCII or NetCDF restart, inpcrd).
        use_velocities : bool
            Start from the file's velocities when it has them.
        charges : str
            "pgm" (a pGM prmtop) or "amber" (the point charges of a classical prmtop, e.g.
            TIP4P-Ew; with MDSettings().replace(elec="q")).
        conform_geometry : bool
            Rebuild three-atom, three-bond molecules (water) at the prmtop's bond lengths when the
            coordinates disagree (pmemd's SHAKE does this at the first step; the rigid-body engine
            would keep the coordinates' geometry and simulate another model).  A warning names how
            many molecules changed.
        **kw
            Keywords of the constructor (settings, dt, temperature, thermostat, ...).

        Returns
        -------
        Simulation

        Raises
        ------
        ValueError
            Coordinates without a periodic box.
        """
        system = System.from_prmtop(prmtop, charges=charges)
        pos, vel, H = read_coordinates_nm(coords)
        if H is None:
            raise ValueError(f"{coords}: the coordinates have no periodic box")
        if conform_geometry:
            pos, changed = conform_rigid_geometry(prmtop, pos)
            if changed:
                warnings.warn(
                    f"{coords}: {changed} rigid molecules rebuilt at the prmtop's bond lengths (conform_geometry)",
                    stacklevel=2,
                )
        return cls(system, pos, H, velocities=vel if use_velocities else None, **kw)

    # ----------------------------------------------------------------- MDEngine hooks
    def _list_groups(self) -> tuple[np.ndarray, int]:
        """Return (molecule of every atom, number of molecules): molecules are the list groups."""
        return self.sys.mol, self.sys.nmol

    def _list_centers(self, dynpos: Any) -> jax.Array:
        """Return the centres of mass (M, 3) [nm] of the rigid bodies (the list groups)."""
        return dynpos.center

    # ----------------------------------------------------------------- coordinates
    def positions(self) -> np.ndarray:
        """Return the atom positions (N, 3) [nm] of the current state (virtual sites included)."""
        return np.asarray(self.rigid.positions(self.state.dyn.position))

    def velocities(self) -> np.ndarray:
        """Return the atom velocities (N, 3) [nm/ps] of the rigid-body motion of the current state."""
        st = self.state
        return np.asarray(self.rigid.atom_velocities(st.dyn.position, st.dyn.momentum))
