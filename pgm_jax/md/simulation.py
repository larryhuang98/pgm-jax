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

import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..system import System
from ..units import AMU_NM3_TO_G_CM3, BAR_PER_KJMOL_NM3, KB
from .box import check_box, reduce_box, volume
from .dipoles import DipoleRecorder, InducedDipoleFile
from .forcefield import MDSettings, PGMForceField
from .integrate import Integrator, field_state, upgrade_state
from .io import NetCDFTrajectory, read_coordinates_nm, write_restart
from .neighbors import AtomNeighbors, MoleculeNeighbors
from .rigid import RigidMolecules
from .vsites import VirtualSites


class Simulation:
    _recorder = None  # DipoleRecorder while run(dipoles=...) is running

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
        if self.integ.restraints is not None:
            self._print(f"# restraints: {self.integ.restraints.describe()}")
        if alchemy is not None:
            self._print(f"# alchemical region: {alchemy.describe()}")
        if mts is not None:
            self._print(f"# {self.integ.describe_mts()}")
        if self.integ.efield is not None:
            self._print(f"# {self.integ.efield.describe()}")
        self._describe_bias()

    def _describe_bias(self):
        if self.integ.bias is not None:
            self._print(f"# biases: {self.integ.bias.describe()}")

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

    # ----------------------------------------------------------------- observables
    def observables(self) -> dict:
        st = self.state
        ke, ke_trans = (float(x) for x in self.integ.kinetic(st))
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

    def set_field(self, E0):
        """Set the amplitude of the external field (three numbers, V/nm) of a simulation created with
        `efield=`; no recompilation.  Forces are recomputed; epot and econs jump by the change of the
        field energy (the work of the switch)."""
        if self.integ.efield is None:
            raise ValueError("the simulation has no external field: create it with efield=ExternalField(...)")
        E0 = jnp.asarray(np.asarray(E0, float).reshape(3), jnp.float64)
        new = self.integ.forces(self.state.set(efield=E0), False)
        # the dipoles jump with the field: restart the predictor from the new solution
        self.state = new.set(induction=new.induction.set(count=jnp.zeros_like(new.induction.count)))

    def restraint_energies(self) -> dict:
        """Restraint energy by kind (kJ/mol) at the current state ({} without restraints)."""
        if self.integ.restraints is None:
            return {}
        if getattr(self, "_restraint_jit", None) is None:
            self._restraint_jit = jax.jit(self.integ.restraints.energies)
        st = self.state
        return {k: float(v) for k, v in self._restraint_jit(self.rigid.positions(st.dyn.position), st.box).items()}

    # ----------------------------------------------------------------- biases (pgm_jax.bias)
    def bias_energies(self) -> np.ndarray:
        """Energy of each bias (kJ/mol) at the current state."""
        if self.integ.bias is None:
            return np.zeros(0)
        if getattr(self, "_bias_jit", None) is None or self._bias_jit[0] is not self.integ.bias:
            b = self.integ.bias
            self._bias_jit = (b, jax.jit(lambda st: b.energies(st.bias, self.rigid.positions(st.dyn.position), st.box)))
        return np.asarray(self._bias_jit[1](self.state))

    def cv_values(self) -> list:
        """The CV vectors of each bias at the current state."""
        st = self.state
        return [np.asarray(v) for v in self.integ.bias.cv_values(self.rigid.positions(st.dyn.position), st.box)]

    def set_bias_state(self, bias_state):
        """Replace the bias state (e.g. BiasSet.load of a converged bias for a static run) and
        recompute the forces; epot and econs jump by the change of the bias energy."""
        st = self.state
        self.state = self.integ.forces(st.set(bias=bias_state), False).set(induction=st.induction)

    def load_bias(self, path: str):
        """Continue with the bias state saved in `path` (prefix.bias, BiasSet.save)."""
        self.set_bias_state(self.integ.bias.load(path))

    def bias_rows(self) -> np.ndarray:
        """COLVAR rows collected since the last call (step, CVs, bias energies), when not writing files."""
        rows = getattr(self, "_bias_rows", [])
        self._bias_rows = []
        return np.concatenate(rows) if rows else np.zeros((0, self.integ.bias.ncol))

    def set_restraints(self, restraints):
        """Replace the restraints (md/restraints.py; None removes them), e.g. to release positional
        restraints in stages: recompiles the step and recomputes the forces of the current state.
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

    def _pressure(self, st):
        pos = self.rigid.positions(st.dyn.position)
        idx = self.nb.candidates(st.nbr, st.dyn.position.center, st.box, pos)[0]
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
        """Instantaneous pressure (bar) from the molecular virial (at the converged dipoles; with the
        restraints) and the centre-of-mass kinetic energy."""
        if not hasattr(self, "_pressure_jit"):
            self._pressure_jit = jax.jit(self._pressure)
        return float(self._pressure_jit(self.state))

    def positions_nm(self):
        return np.asarray(self.rigid.positions(self.state.dyn.position))

    def velocities_nm_ps(self):
        st = self.state
        return np.asarray(self.rigid.atom_velocities(st.dyn.position, st.dyn.momentum))

    # ----------------------------------------------------------------- running
    def _size_lists(self, body, H, factor: float = 1.2, nbr=None):
        """Static sizes: molecules per atom row of the molecule list (molecule mode) and pairs per
        compacted force-field row (intramolecular + intermolecular inside the cutoffs; each part of
        split rows), with head-room above the current maxima (PGMForceField.size_rows)."""
        pos = self.rigid.positions(body)
        nbr = self.nb.allocate(pos, body.center, H) if nbr is None else nbr
        if self.nb.kind == "molecule":
            self.nb.size(nbr, body.center, H, pos, factor)
        idx = self.nb.candidates(nbr, body.center, H, pos)[0]
        self.ff.size_rows(pos, H, idx, factor)
        return nbr

    def _make_neighbors(self, H):
        """Neighbour-list object for box H.  JAX-MD's cell list is laid out for one box shape, so
        it is rebuilt when the volume has drifted by more than 10 % (e.g. NPT from a loose start) or
        when a block keeps overflowing (a shrinking box makes the cells smaller than the cutoff)."""
        s = self.settings
        mode = self._nb_mode
        if mode == "auto":
            mode = "molecule" if MoleculeNeighbors.fits(H, s.pair_cutoff, s.skin, self._r_list) else "atom"
        if mode == "molecule":
            self.nb = MoleculeNeighbors(self.sys.mol, self.sys.nmol, self._r_list, H, s.pair_cutoff, s.skin)
        else:
            self.nb = AtomNeighbors(self.sys.n, H, s.pair_cutoff, s.skin)
        self._nb_volume = float(volume(jnp.asarray(H)))

    def _rebuild_neighbors(self):
        self.n_rebuilds = getattr(self, "n_rebuilds", 0) + 1
        st = self.state
        self._print(f"# step {int(st.step)}: neighbour lists rebuilt for volume {float(volume(st.box)):.3f} nm^3")
        H = np.asarray(st.box)
        self._make_neighbors(H)
        nbr = self._size_lists(st.dyn.position, H)
        self.integ.nb = self.nb
        self.integ.compile()
        self.state = self.integ.forces(st.set(nbr=nbr), False).set(induction=st.induction)

    def _print(self, s):
        if self.log is not None:
            print(s, file=self.log, flush=True)

    def _advance(self, n: int):
        if abs(float(volume(self.state.box)) / self._nb_volume - 1.0) > 0.10:
            self._rebuild_neighbors()
        try:
            self._advance_block(n)
        except RuntimeError as err:
            # the box changed too much within the block (NPT far from equilibrium): rebuild the
            # lists and advance in halves, rebuilding between them as the volume drifts
            if "overflowing" not in str(err) or n < 2:
                raise
            self._rebuild_neighbors()
            self._advance(n // 2)
            self._advance(n - n // 2)

    def _advance_block(self, n: int):
        start = self.state
        bias = self.integ.bias
        if bias is not None and start.bias is not None:  # room for the block's hills and COLVAR rows
            start = start.set(bias=bias.reserve(start.bias, n))
        for _attempt in range(6):
            new = self.integ.run(start, n) if self._recorder is None else self._recorder.run(start, n)
            jax.block_until_ready(new.epot)
            self.integ.check_block(new)
            nb_bad, row_bad = self.nb.failed(new.nbr), bool(new.overflow)
            if not (nb_bad or row_bad):
                break
            body = start.dyn.position
            old = (self.ff.capacity, getattr(self.nb, "cap", None))
            nbr = self._size_lists(body, start.box, 1.3, None if nb_bad else start.nbr)
            if row_bad:  # never shrink below what overflowed
                self.ff.grow_rows(old[0])
                if getattr(self.nb, "cap", None) is not None and old[1] is not None:
                    self.nb.cap = max(self.nb.cap, old[1] + 4)
            self.integ.compile()
            self._print(
                f"# {'neighbour list' if nb_bad else 'row capacity'} overflow in steps {int(start.step)}-"
                f"{int(start.step) + n}: resized (rows {self.ff.mc or self.nb.cap}, list {nbr.idx.shape[1]}), repeating"
            )
            start = self.integ.forces(start.set(nbr=nbr), False).set(induction=start.induction)
        else:
            raise RuntimeError("neighbour list keeps overflowing")
        if self._recorder is not None:
            self._recorder.keep()
        body = self.rigid.wrap(new.dyn.position, new.box)
        if self.integ.efield is not None and self.integ.field_charged:  # itinerant dipole of re-wrapped ions
            q = self.ff._atoms(self.integ.params)["q"]
            shift = jnp.sum(q[:, None] * (self.rigid.positions(new.dyn.position) - self.rigid.positions(body)), axis=0)
            new = new.set(fshift=new.fshift + shift)
        self.state = new.set(dyn=new.dyn.set(position=body))
        if bias is not None and new.bias is not None:
            rows, bs = bias.drain(new.bias)
            self.state = self.state.set(bias=bs)
            if len(rows):
                self._bias_rows = getattr(self, "_bias_rows", []) + [rows]
        self.time_ps += n * self.dt
        if not np.isfinite(float(new.epot)):
            raise FloatingPointError(f"energy is not finite at step {int(new.step)}")

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
    ):
        """Every `report` steps a log line, `traj` a trajectory frame (prefix.nc), `restart` a
        restart + checkpoint; `dipoles`: cell dipole sampled every `dipoles` steps into prefix.dip
        (does not shorten the blocks); `induced`: per-atom induced dipoles into prefix.mu.nc."""
        block = int(np.gcd.reduce([x for x in (report, traj, restart, nsteps, induced) if x > 0]))
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
        logf = open(prefix + ".log", "a" if append else "w")
        cols = None
        t0, s0 = time.time(), int(self.state.step)
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
                el = time.time() - t0
                obs["ns_per_day"] = (step - s0) * self.dt / 1000.0 / max(el, 1e-9) * 86400.0
                if cols is None:
                    cols = list(obs)
                    header = "# " + " ".join(f"{c:>14s}" for c in cols)
                    logf.write(header + "\n")
                    self._print(header)
                line = "  " + " ".join(
                    f"{obs[c]:14.6f}" if isinstance(obs[c], float) else f"{obs[c]:14d}" for c in cols
                )
                logf.write(line + "\n")
                logf.flush()
                self._print(line)
            if tfile is not None and step % traj == 0:
                tfile.write(self.time_ps, self.positions_nm() * 10.0, np.asarray(self.state.box) * 10.0)
            if restart and step % restart == 0:
                self.save(prefix)
        logf.close()
        self._recorder = None
        if restart:
            self.save(prefix)

    # ----------------------------------------------------------------- checkpoints
    def save(self, prefix: str):
        """Amber NetCDF restart (prefix.rst7) and a checkpoint of the complete state (prefix.chk):
        rigid-body coordinates and momenta, forces, box, dipoles and extrapolation history, random
        state, barostat state.  Continuing from it reproduces the run up to floating-point summation
        order (GPU atomics in PME spreading make runs non-bitwise-reproducible anyway)."""
        write_restart(
            prefix + ".rst7",
            self.positions_nm() * 10.0,
            self.velocities_nm_ps() * 10.0,
            np.asarray(self.state.box) * 10.0,
            self.time_ps,
        )
        host = jax.tree_util.tree_map(np.asarray, self.state.set(nbr=None))
        with open(prefix + ".chk", "wb") as fh:
            pickle.dump({"state": host, "time_ps": self.time_ps}, fh)
        if self.integ.bias is not None and self.state.bias is not None:
            self.integ.bias.save(self.state.bias, prefix + ".bias")

    def _bias_of_checkpoint(self, st):
        """A checkpoint's bias state if this simulation has biases (a fresh one if it had none)."""
        if self.integ.bias is None:
            return st.set(bias=None) if getattr(st, "bias", None) is not None else st
        if getattr(st, "bias", None) is None:
            return st.set(bias=self.integ.bias.init())
        return st

    def load(self, path: str):
        """Continue from a checkpoint written by `save` (same system and settings)."""
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        st = jax.tree_util.tree_map(jnp.asarray, d["state"])
        st = upgrade_state(st, self.state.aux)  # checkpoints from before the thermostat fields
        st = self._bias_of_checkpoint(st)
        st = field_state(st, self.integ.efield)  # the checkpoint's field amplitude, or the integrator's
        nbr = self.nb.allocate(self.rigid.positions(st.dyn.position), st.dyn.position.center, st.box)
        self.state = st.set(nbr=nbr)  # forces, dipoles and history are part of the state
        self.time_ps = d["time_ps"]
