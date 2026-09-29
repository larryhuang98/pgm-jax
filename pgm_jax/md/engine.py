"""The part of the MD drivers shared by the rigid-body engine (simulation.Simulation) and the
atomistic engine (flexible.FlexibleSimulation): neighbour lists and static sizes, blocks of steps
with overflow handling, observables, pressure, restraints, biases, external fields, the run loop
with its output files, and checkpoints.

An engine subclass builds, in its constructor, the system (`sys`), settings, force field (`ff`),
integrator (`integ`), the molecule representation (`rigid`: rigid bodies, or flexible molecules
with the same `positions` / `wrap` interface), the neighbour list (`_make_neighbors`) and the
initial state, and provides four hooks:

    _list_groups()          (group index of every atom, number of groups) of the molecular list
    _list_centers(dynpos)   centres of those groups at the integrator's positions
    _atom_positions(dynpos) atom positions (N, 3) nm at the integrator's positions
    _after_block()          checks after every block (the flexible engine: molecule extent)

Everything here runs on the host between compiled blocks; the compiled steps are the
integrators' own.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..units import AMU_NM3_TO_G_CM3, BAR_PER_KJMOL_NM3, KB
from .box import volume
from .dipoles import DipoleRecorder, InducedDipoleFile
from .driver import (
    LogTable,
    Stopwatch,
    advance_with_rebuilds,
    block_length,
    device_tree,
    finite_or_raise,
    is_legacy_checkpoint,
    read_checkpoint,
    retry_block,
    write_checkpoint,
)
from .integrate import field_state, upgrade_state
from .io import NetCDFTrajectory, write_restart
from .neighbors import AtomNeighbors, MoleculeNeighbors

# parts of MDState that depend on the options of a run (see tree_from_arrays)
OPTIONAL_STATE = (".bias", ".efield", ".fshift", ".fdip")


class MDEngine:
    """Base class of the MD engines (see the module docstring for what subclasses provide)."""

    _recorder = None  # DipoleRecorder while run(dipoles=...) is running
    checkpoint_kind = "md"

    # ----------------------------------------------------------------- hooks
    def _list_groups(self) -> tuple[np.ndarray, int]:
        """(group index of every atom, number of groups) for the molecular neighbour list."""
        raise NotImplementedError

    def _list_centers(self, dynpos):
        """Centres (G, 3) nm of the neighbour-list groups at the integrator's positions `dynpos`."""
        raise NotImplementedError

    def _atom_positions(self, dynpos):
        """Atom positions (N, 3) nm at the integrator's positions `dynpos`."""
        return self.rigid.positions(dynpos)

    def _after_block(self) -> None:
        """Checks after a block of steps (none by default)."""

    # ----------------------------------------------------------------- log header
    def _print(self, s: str) -> None:
        """Write a diagnostic line to the log stream (if any)."""
        if self.log is not None:
            print(s, file=self.log, flush=True)

    def _describe_options(self, alchemy=None, mts=None) -> None:
        """Log lines for the optional parts of the model: restraints, charge flux, alchemical region,
        multiple time stepping, external field, biases."""
        if self.integ.restraints is not None:
            self._print(f"# restraints: {self.integ.restraints.describe()}")
        if getattr(self.ff, "flux", None) is not None:
            self._print(f"# {self.ff.flux.describe()}")
        if alchemy is not None:
            self._print(f"# alchemical region: {alchemy.describe()}")
        if mts is not None:
            self._print(f"# {self.integ.describe_mts()}")
        if self.integ.efield is not None:
            self._print(f"# {self.integ.efield.describe()}")
        self._describe_bias()

    def _describe_bias(self) -> None:
        """Log line describing the biases (if any)."""
        if self.integ.bias is not None:
            self._print(f"# biases: {self.integ.bias.describe()}")

    # ----------------------------------------------------------------- observables
    def observables(self) -> dict:
        """Observables of the current state (one row of the log table).

        Returns
        -------
        dict
            step, time_ps, temperatures [K] (total and per group of degrees of freedom), energies
            [kJ/mol] (etot, ekin, epot and its parts, econs = conserved quantity including the
            thermostat heat), volume [nm^3], density [g/cm^3], CG statistics of the induced-dipole
            solver; with restraints, biases, the barostat or a field also their energies,
            acceptance and field / cell dipole [V/nm, e nm].
        """
        st = self.state
        ke, _ = (float(x) for x in self.integ.kinetic(st))
        V = float(volume(st.box))
        mass = float(np.sum(self.sys.masses))
        t_tr, t_rot = (float(x) for x in self.integ.temperatures(st))
        out = {
            "step": int(st.step),
            "time_ps": self.time_ps,
            "temp_K": 2 * ke / (self.integ.dof * KB),
            "temp_trans": t_tr,
            "temp_rot": t_rot,
            "etot": ke + float(st.epot),
            "ekin": ke,
            "epot": float(st.epot),
            "elec": float(st.elec),
            "econs": ke + float(st.epot) + 0.5 * float(jnp.sum(st.aux * st.aux)) - float(st.heat),
            "vdw": float(st.vdw),
            "volume_nm3": V,
            "density_g_cm3": mass / V * AMU_NM3_TO_G_CM3,
            "cg_iter": int(st.iters),
            "cg_iter_max": int(st.max_iters),
            "cg_resid_max": float(st.resid),
            "cg_mean": float(st.cg_total) / max(int(st.step), 1),
        }
        if self.integ.restraints is not None:  # part of epot
            out["erestraint"] = float(sum(self.restraint_energies().values()))
        if self.integ.bias is not None and st.bias is not None:  # part of epot
            out["ebias"] = float(np.sum(self.bias_energies()))
            out["bias_work"] = float(st.bias.work)
            out.update(self.integ.bias.info(st.bias))
        if self.ensemble == "npt":
            tries, acc = int(st.mc[0]), int(st.mc[1])
            out["mc_accept"] = acc / max(tries, 1)
        if self.integ.efield is not None:  # the field (V/nm) and the dipole it acts on (e nm)
            fld = self.integ.efield
            E = self.integ.field_at(st, st.step)[0]
            Emac = np.asarray(fld.macroscopic(E, st.fdip, V))
            out.update(
                efield=float(np.linalg.norm(np.asarray(E))),
                field_energy=float(fld.energy(E, st.fdip, V)),
                Mx=float(st.fdip[0]),
                My=float(st.fdip[1]),
                Mz=float(st.fdip[2]),
            )
            if fld.kind == "D":
                out.update(Emac_x=float(Emac[0]), Emac_y=float(Emac[1]), Emac_z=float(Emac[2]))
        return out

    def _pressure(self, st):
        """Instantaneous pressure [bar] of state `st` (traced; see `pressure`)."""
        pos = self._atom_positions(st.dyn.position)
        idx = self.nb.candidates(st.nbr, self._list_centers(st.dyn.position), st.box, pos)[0]
        if self.integ.alchemy is None:
            W = self.ff.strain_derivative(
                pos, st.box, idx, st.induction.mu, self.integ.params, efield=self.integ.field_at(st, st.step)
            )
        else:
            W = self.integ.alchemy.strain_derivative(
                self.ff, pos, st.box, idx, st.induction.mu, self.integ.params, st.lam
            )
        W = W + self.integ.restraint_strain(pos, st.box, st.bias)
        ke_t = self.integ.kinetic(st)[1]
        return (2.0 * ke_t - jnp.trace(W)) / (3.0 * volume(st.box)) * BAR_PER_KJMOL_NM3

    def pressure(self) -> float:
        """Instantaneous pressure [bar]: molecular virial (at the converged dipoles; with the
        restraints and biases) and the centre-of-mass kinetic energy."""
        if not hasattr(self, "_pressure_jit"):
            self._pressure_jit = jax.jit(self._pressure)
        return float(self._pressure_jit(self.state))

    # ----------------------------------------------------------------- restraints, fields, biases
    def set_field(self, E0) -> None:
        """Set the amplitude of the external field of a simulation created with `efield=`.

        Parameters
        ----------
        E0 : sequence of 3 floats
            Field amplitude [V/nm].  No recompilation; the forces are recomputed and epot / econs
            jump by the change of the field energy (the work of the switch).

        Raises
        ------
        ValueError
            The simulation has no external field.
        """
        if self.integ.efield is None:
            raise ValueError("the simulation has no external field: create it with efield=ExternalField(...)")
        E0 = jnp.asarray(np.asarray(E0, float).reshape(3), jnp.float64)
        new = self.integ.forces(self.state.set(efield=E0), False)
        # the dipoles jump with the field: restart the predictor from the new solution
        self.state = new.set(induction=new.induction.set(count=jnp.zeros_like(new.induction.count)))

    def restraint_energies(self) -> dict:
        """Restraint energy by kind [kJ/mol] at the current state ({} without restraints)."""
        if self.integ.restraints is None:
            return {}
        if getattr(self, "_restraint_jit", None) is None:
            self._restraint_jit = jax.jit(self.integ.restraints.energies)
        st = self.state
        return {k: float(v) for k, v in self._restraint_jit(self._atom_positions(st.dyn.position), st.box).items()}

    def set_restraints(self, restraints) -> None:
        """Replace the restraints (md/restraints.py; None removes them), e.g. to release positional
        restraints in stages: recompiles the step and recomputes the forces of the current state;
        epot and econs jump by the change of the restraint energy (the work of the switch)."""
        from .restraints import as_restraints

        r = as_restraints(restraints)
        if r is not None:
            r.check(self.sys.n)
        self.integ.restraints = r
        self.integ.compile()
        self._restraint_jit = None
        self.__dict__.pop("_pressure_jit", None)
        st = self.state
        self.state = self.integ.forces(st, False).set(induction=st.induction)

    def bias_energies(self) -> np.ndarray:
        """Energy of each bias [kJ/mol] at the current state (empty without biases)."""
        if self.integ.bias is None:
            return np.zeros(0)
        if getattr(self, "_bias_jit", None) is None or self._bias_jit[0] is not self.integ.bias:
            b = self.integ.bias

            def energies(st):
                """Bias energies of state st."""
                return b.energies(st.bias, self._atom_positions(st.dyn.position), st.box)

            self._bias_jit = (b, jax.jit(energies))
        return np.asarray(self._bias_jit[1](self.state))

    def cv_values(self) -> list:
        """The collective-variable vectors of each bias at the current state."""
        st = self.state
        return [np.asarray(v) for v in self.integ.bias.cv_values(self._atom_positions(st.dyn.position), st.box)]

    def set_bias_state(self, bias_state) -> None:
        """Replace the bias state (e.g. BiasSet.load of a converged bias for a static run) and
        recompute the forces; epot and econs jump by the change of the bias energy."""
        st = self.state
        self.state = self.integ.forces(st.set(bias=bias_state), False).set(induction=st.induction)

    def load_bias(self, path: str) -> None:
        """Continue with the bias state saved in `path` (prefix.bias, BiasSet.save)."""
        self.set_bias_state(self.integ.bias.load(path))

    def bias_rows(self) -> np.ndarray:
        """COLVAR rows collected since the last call (step, CVs, bias energies), when not writing files."""
        rows = getattr(self, "_bias_rows", [])
        self._bias_rows = []
        return np.concatenate(rows) if rows else np.zeros((0, self.integ.bias.ncol))

    # ----------------------------------------------------------------- neighbour lists and sizes
    def _make_neighbors(self, H) -> None:
        """Neighbour-list object for box H [nm]: molecular-centre list when the box is large enough
        for the groups' radius (`neighbor_list` "auto" / "molecule"), else an atom list.  JAX-MD's
        cell list is laid out for one box shape, so it is rebuilt when the volume drifts by 10 %
        or a block keeps overflowing (driver.advance_with_rebuilds)."""
        s = self.settings
        mode = self._nb_mode
        if mode == "auto":
            mode = "molecule" if MoleculeNeighbors.fits(H, s.pair_cutoff, s.skin, self._r_list) else "atom"
        if mode == "molecule":
            group, n_group = self._list_groups()
            self.nb = MoleculeNeighbors(group, n_group, self._r_list, H, s.pair_cutoff, s.skin)
        else:
            self.nb = AtomNeighbors(self.sys.n, H, s.pair_cutoff, s.skin)
        self._nb_volume = float(volume(jnp.asarray(H)))

    def _size_lists(self, dynpos, H, factor: float = 1.2, nbr=None):
        """Allocate (or reuse) a neighbour list at `dynpos` and size the static capacities.

        Parameters
        ----------
        dynpos
            The integrator's positions (rigid bodies or atom positions).
        H : (3, 3) array
            Box [nm].
        factor : float
            Head-room above the current maxima of the molecules per list row and the pairs per
            force-field row (PGMForceField.size_rows).
        nbr : optional
            A neighbour list to reuse instead of allocating one.

        Returns
        -------
        The neighbour list.
        """
        pos = self._atom_positions(dynpos)
        c = self._list_centers(dynpos)
        nbr = self.nb.allocate(pos, c, H) if nbr is None else nbr
        if self.nb.kind == "molecule":
            self.nb.size(nbr, c, H, pos, factor)
        idx = self.nb.candidates(nbr, c, H, pos)[0]
        self.ff.size_rows(pos, H, idx, factor)
        return nbr

    def _rebuild_neighbors(self) -> None:
        """New neighbour-list layout for the current box; recompile and recompute the forces."""
        self.n_rebuilds = getattr(self, "n_rebuilds", 0) + 1
        st = self.state
        self._print(f"# step {int(st.step)}: neighbour lists rebuilt for volume {float(volume(st.box)):.3f} nm^3")
        H = np.asarray(st.box)
        self._make_neighbors(H)
        nbr = self._size_lists(st.dyn.position, H)
        self.integ.nb = self.nb
        self.integ.compile()
        self.state = self.integ.forces(st.set(nbr=nbr), False).set(induction=st.induction)

    # ----------------------------------------------------------------- blocks of steps
    def _advance(self, n: int) -> None:
        """Advance n steps without writing files (lists rebuilt for a changed box, blocks split when
        they keep overflowing)."""
        advance_with_rebuilds(
            n,
            self._advance_block,
            self._rebuild_neighbors,
            lambda: float(volume(self.state.box)) / self._nb_volume,
        )
        self._after_block()

    def _run_block(self, start, n: int):
        """One compiled block of n steps (through the dipole recorder while one is active)."""
        new = self.integ.run(start, n) if self._recorder is None else self._recorder.run(start, n)
        jax.block_until_ready(new.epot)
        self.integ.check_block(new)
        return new

    def _resize(self, start, n: int, list_bad: bool, rows_bad: bool):
        """Enlarge the list and row capacities after an overflow in the block of n steps from `start`,
        recompile, and return `start` with its forces evaluated at the new sizes."""
        old = (self.ff.capacity, getattr(self.nb, "cap", None))
        nbr = self._size_lists(start.dyn.position, start.box, 1.3, None if list_bad else start.nbr)
        if rows_bad:  # never shrink below what overflowed
            self.ff.grow_rows(old[0])
            if getattr(self.nb, "cap", None) is not None and old[1] is not None:
                self.nb.cap = max(self.nb.cap, old[1] + 4)
        self.integ.compile()
        self._print(
            f"# {'neighbour list' if list_bad else 'row capacity'} overflow in steps {int(start.step)}-"
            f"{int(start.step) + n}: resized (rows {self.ff.mc or self.nb.cap}, list {nbr.idx.shape[1]}), repeating"
        )
        return self.integ.forces(start.set(nbr=nbr), False).set(induction=start.induction)

    def _advance_block(self, n: int) -> None:
        """n steps as one compiled block: overflow handling, then re-wrapping of the molecules into
        the box, the itinerant dipole of re-wrapped charged molecules, the bias output rows."""
        start = self.state
        bias = self.integ.bias
        if bias is not None and start.bias is not None:  # room for the block's hills and COLVAR rows
            start = start.set(bias=bias.reserve(start.bias, n))
        new = retry_block(
            lambda s: self._run_block(s, n),
            start,
            lambda s: (self.nb.failed(s.nbr), bool(s.overflow)),
            lambda s, lb, rb: self._resize(s, n, lb, rb),
        )
        if self._recorder is not None:
            self._recorder.keep()
        body = self.rigid.wrap(new.dyn.position, new.box)
        if self.integ.efield is not None and self.integ.field_charged:  # itinerant dipole of re-wrapped ions
            q = self.ff._atoms(self.integ.params)["q"]
            shift = jnp.sum(q[:, None] * (self._atom_positions(new.dyn.position) - self._atom_positions(body)), axis=0)
            new = new.set(fshift=new.fshift + shift)
        self.state = new.set(dyn=new.dyn.set(position=body))
        if bias is not None and new.bias is not None:
            rows, bs = bias.drain(new.bias)
            self.state = self.state.set(bias=bs)
            if len(rows):
                self._bias_rows = getattr(self, "_bias_rows", []) + [rows]
        self.time_ps += n * self.dt
        finite_or_raise(new.epot, new.step)

    # ----------------------------------------------------------------- running with output files
    def run(
        self,
        nsteps: int,
        report: int = 1000,
        traj: int = 0,
        restart: int = 0,
        prefix: str = "md",
        pressure_every_report: bool = False,
        append: bool = False,
        dipoles: int = 0,
        induced: int = 0,
    ) -> None:
        """Advance nsteps with output files.

        Parameters
        ----------
        nsteps : int
            Steps.
        report : int
            Steps between rows of the log table prefix.log (0: none).
        traj : int
            Steps between frames of the NetCDF trajectory prefix.nc (0: none).
        restart : int
            Steps between an Amber restart prefix.rst7 plus a checkpoint prefix.chk (0: none; with
            restarts also at the end).
        prefix : str
            Path prefix of the files.
        pressure_every_report : bool
            Add the instantaneous pressure [bar] to every log row.
        append : bool
            Append to existing files (a continuation).
        dipoles : int
            Steps between samples of the cell dipole, written to prefix.dip (sampled inside the
            blocks; does not shorten them).
        induced : int
            Steps between frames of the per-atom induced dipoles prefix.mu.nc.

        Raises
        ------
        NotImplementedError
            Cell dipoles with an alchemical region (its charges are not scaled).
        """
        block = block_length(nsteps, report, traj, restart, induced)
        if dipoles and self.integ.alchemy is not None:
            raise NotImplementedError("the cell dipole (dipoles=) does not scale an alchemical region's charges")
        tfile = NetCDFTrajectory(prefix + ".nc", self.sys.n, append=append) if traj else None
        self._recorder = DipoleRecorder(self, prefix + ".dip", dipoles, append=append) if dipoles else None
        mufile = InducedDipoleFile(prefix + ".mu.nc", self.sys.n, append=append) if induced else None
        bout = None
        if self.integ.bias is not None:
            from ..bias.io import BiasOutput

            self.bias_rows()  # rows of earlier _advance calls
            bout = BiasOutput(self.integ.bias, prefix, self.dt, self.T0, append=append, state=self.state.bias)
        table = LogTable(prefix + ".log", append=append, echo=self.log)
        clock = Stopwatch(int(self.state.step), self.dt)
        done = 0
        while done < nsteps:
            n = min(block, nsteps - done)
            self._advance(n)
            done += n
            step = int(self.state.step)
            if self._recorder is not None:
                self._recorder.flush()
            if bout is not None:
                bout.write(self.bias_rows(), self.state.bias)
            if mufile is not None and step % induced == 0:
                mufile.write(step, self.time_ps, self.state.induction.mu)
            if report and step % report == 0:
                obs = self.observables()
                if pressure_every_report:
                    obs["press_bar"] = self.pressure()
                obs["ns_per_day"] = clock.ns_per_day(step)
                table.write(obs)
            if tfile is not None and step % traj == 0:
                tfile.write(self.time_ps, self.positions_nm() * 10.0, np.asarray(self.state.box) * 10.0)
            if restart and step % restart == 0:
                self.save(prefix)
        table.close()
        self._recorder = None
        if restart:
            self.save(prefix)

    # ----------------------------------------------------------------- checkpoints
    def save(self, prefix: str) -> None:
        """Amber NetCDF restart (prefix.rst7) and a checkpoint of the complete state (prefix.chk).

        The checkpoint (driver.write_checkpoint: npz + JSON header) holds coordinates and momenta,
        forces, box, induced dipoles and predictor history, random state, barostat and thermostat
        state, bias state and field amplitude; continuing from it reproduces the run up to
        floating-point summation order (bitwise on the CPU).  The bias state is also written to
        prefix.bias (BiasSet.save).
        """
        write_restart(
            prefix + ".rst7",
            self.positions_nm() * 10.0,
            self.velocities_nm_ps() * 10.0,
            np.asarray(self.state.box) * 10.0,
            self.time_ps,
        )
        write_checkpoint(
            prefix + ".chk",
            self.checkpoint_kind,
            {"time_ps": self.time_ps, "engine": type(self).__name__, "state": self.state.set(nbr=None)},
        )
        if self.integ.bias is not None and self.state.bias is not None:
            self.integ.bias.save(self.state.bias, prefix + ".bias")

    def _bias_of_checkpoint(self, st):
        """A legacy checkpoint's bias state if this simulation has biases (a fresh one if it had none)."""
        if self.integ.bias is None:
            return st.set(bias=None) if getattr(st, "bias", None) is not None else st
        if getattr(st, "bias", None) is None:
            return st.set(bias=self.integ.bias.init())
        return st

    def load(self, path: str) -> None:
        """Continue from a checkpoint written by `save` (same system and settings), or from a legacy
        pickle checkpoint of pgm_jax up to commit e72c57c (save again to convert it).

        Parts of the state that depend on options (a bias, the external field) may differ: a bias
        missing from the checkpoint starts fresh, one in it that this run does not have is dropped,
        a field amplitude in it is taken over.

        Raises
        ------
        ValueError
            A checkpoint of another kind, system or setup.
        """
        template = self.state.set(nbr=None)
        if self.integ.bias is not None and template.bias is None:
            template = template.set(bias=self.integ.bias.init())
        d = read_checkpoint(path, self.checkpoint_kind, template, OPTIONAL_STATE)
        st = d["state"]
        if is_legacy_checkpoint(path):
            st = device_tree(st)
            st = upgrade_state(st, self.state.aux)  # checkpoints from before the thermostat fields
            st = self._bias_of_checkpoint(st)
        time_ps = float(d["time_ps"])
        st = field_state(st, self.integ.efield)  # the checkpoint's field amplitude, or the integrator's
        dynpos = st.dyn.position
        nbr = self.nb.allocate(self._atom_positions(dynpos), self._list_centers(dynpos), st.box)
        self.state = st.set(nbr=nbr)  # forces, dipoles and history are part of the state
        self.time_ps = time_ps
