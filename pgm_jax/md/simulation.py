"""Simulation driver: Amber inputs in, Amber-readable outputs out.

    sim = Simulation.from_amber("water.prmtop", "water.rst7", settings=MDSettings(...),
                                ensemble="npt", temperature=298.0, dt=0.001)
    sim.run(nsteps=100000, report=1000, traj=1000, restart=10000, prefix="md")

Steps run in jit-compiled blocks on the device; between blocks the host checks the neighbour
list (reallocates and repeats the block on overflow), re-wraps molecules into the box, reports
and writes files.  Molecules are the prmtop residues; identical residues share one template.
Virtual sites (Amber extra points, Molecule.vsites; md/vsites.py) are massless points of the rigid
templates, placed from their parents at the start.
run(dipoles=n) also samples the cell dipole every n steps (on the device, inside the blocks) into
prefix.dip, and run(induced=n) writes per-atom induced dipoles to prefix.mu.nc (md/dipoles.py).
mts=MTS(...) integrates force groups with their own time steps (md/mts.py; dt is the outer step).
bias=... adds biases on collective variables (pgm_jax.bias: metadynamics, OPES, static biases);
run() then writes prefix.colvar, prefix.hills and, with the restarts, prefix.bias (bias/io.py)."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from ..system import System
from .box import check_box, reduce_box
from .engine import MDEngine
from .forcefield import MDSettings, PGMForceField
from .integrate import Integrator
from .io import read_coordinates_nm
from .rigid import RigidMolecules
from .vsites import VirtualSites


class Simulation(MDEngine):
    """Molecular dynamics of rigid molecules (JAX-MD rigid bodies; every molecule of `sys` is rigid
    at its template geometry, virtual sites are points of the templates).  The shared machinery
    (blocks, observables, run loop, checkpoints) is MDEngine's (md/engine.py)."""

    checkpoint_kind = "md-rigid"

    def __init__(
        self,
        sys: System,
        pos_nm,
        H_nm,
        settings: MDSettings = MDSettings(),
        dt: float = 0.001,
        ensemble: str = "nvt",
        temperature: float = 298.0,
        gamma: float = 1.0,
        pressure: float = 1.0,
        barostat_interval: int = 100,
        seed: int = 0,
        vel_nm_ps=None,
        params=None,
        log=sys.stdout,
        neighbor_list: str = "auto",
        thermostat="langevin",
        tau_t: float = 1.0,
        restraints=None,
        alchemy=None,
        mts=None,
        bias=None,
        efield=None,
    ):
        H = reduce_box(H_nm)
        check_box(H, settings.pair_cutoff + settings.skin)
        self.sys, self.settings, self.log = sys, settings, log
        self.vsites = VirtualSites.of(sys)
        if self.vsites is not None:  # sites of the rigid templates from their parents
            pos_nm = np.asarray(self.vsites.place(np.asarray(pos_nm, float), H))
            self.vsites.check(pos_nm, H, (sys.cov_i, sys.cov_j))
        self.rigid = RigidMolecules(sys, pos_nm, H)
        self.ff = PGMForceField(sys, H, settings)
        self._r_list = float(jnp.max(jnp.linalg.norm(self.rigid.local, axis=1)))
        self._nb_mode = neighbor_list
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
            ensemble,
            temperature,
            gamma,
            pressure,
            barostat_interval,
            params,
            thermostat=thermostat,
            tau_t=tau_t,
            restraints=restraints,
            alchemy=alchemy,
            bias=bias,
            efield=efield,
            **extra,
        )
        self.dt, self.ensemble, self.T0 = dt, ensemble, temperature
        body = self.rigid.body0
        mom = None
        if vel_nm_ps is not None:
            mom = self.rigid.momenta_from_velocities(body, self.rigid.positions(body), jnp.asarray(vel_nm_ps))
        self.state = self.integ.init(body, H, jax.random.PRNGKey(seed), mom)
        self.time_ps = 0.0
        thermo = "" if self.integ.thermostat is None else f" ({self.integ.thermostat.describe()})"
        self._print(
            f"# pgm_jax MD: {sys.nmol} rigid molecules, {sys.n} atoms"
            f"{'' if self.vsites is None else f' ({self.vsites.n_sites} virtual sites)'}, {ensemble.upper()}{thermo}, "
            f"dt {dt * 1000:g} fs, "
            f"{settings.precision} precision, PME grid {self.ff.pme.K} order {settings.pme_order}, "
            f"{settings.describe_cutoffs()}, {self.nb.kind} neighbour list, {settings.describe_induction()}, "
            f"template fit RMSD {self.rigid.fit_rmsd:.2e} nm, device {jax.devices()[0]}"
        )
        self._describe_options(alchemy, mts)

    @classmethod
    def from_amber(
        cls, prmtop: str, coords: str, use_velocities: bool = True, charges: str = "pgm", **kw
    ) -> Simulation:
        """charges: "pgm" (a pGM prmtop) or "amber" (the point charges of a classical prmtop, e.g.
        TIP4P-Ew; with MDSettings(elec="q")); extra points become virtual sites (read_prmtop_pgm)."""
        sys = System.from_prmtop(prmtop, charges=charges)
        pos, vel, H = read_coordinates_nm(coords)
        if H is None:
            raise ValueError(f"{coords}: the coordinates have no periodic box")
        return cls(sys, pos, H, vel_nm_ps=vel if use_velocities else None, **kw)

    # ----------------------------------------------------------------- MDEngine hooks
    def _list_groups(self) -> tuple[np.ndarray, int]:
        """Molecules are the groups of the molecular neighbour list."""
        return self.sys.mol, self.sys.nmol

    def _list_centers(self, dynpos):
        """Centres of mass of the rigid bodies (the list groups)."""
        return dynpos.center

    # ----------------------------------------------------------------- coordinates
    def positions_nm(self) -> np.ndarray:
        """Atom positions (N, 3) [nm] of the current state (virtual sites included)."""
        return np.asarray(self.rigid.positions(self.state.dyn.position))

    def velocities_nm_ps(self) -> np.ndarray:
        """Atom velocities (N, 3) [nm/ps] of the rigid-body motion of the current state."""
        st = self.state
        return np.asarray(self.rigid.atom_velocities(st.dyn.position, st.dyn.momentum))
