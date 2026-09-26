"""Simulation driver: Amber inputs in, Amber-readable outputs out.

    sim = Simulation.from_amber("water.prmtop", "water.rst7", settings=MDSettings(...),
                                ensemble="npt", temperature=298.0, dt=0.001)
    sim.run(nsteps=100000, report=1000, traj=1000, restart=10000, prefix="md")

Steps run in jit-compiled blocks on the device; between blocks the host checks the neighbour
list (reallocates and repeats the block on overflow), re-wraps molecules into the box, reports
and writes files.  Molecules are the prmtop residues; identical residues share one template."""
from __future__ import annotations

import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..param import read_prmtop_pgm
from ..system import Molecule, System
from .box import check_box, reduce_box, volume
from .forcefield import MDSettings, PGMForceField
from .integrate import KB, Integrator, upgrade_state
from .io import NetCDFTrajectory, box_from_cell, read_coordinates, write_restart
from .neighbors import AtomNeighbors, MoleculeNeighbors
from .rigid import RigidMolecules

AMU_NM3_TO_G_CM3 = 1.66053906660e-3


def _dedupe(mols: list[Molecule]) -> list[Molecule]:
    seen, out = {}, []
    for m in mols:
        key = (m.name, tuple(m.elements), tuple(m.types), m.q.tobytes(), m.radius.tobytes(), m.alpha.tobytes(),
               tuple(m.cov), m.lj_rmin_half.tobytes(), m.lj_sqrt_eps.tobytes(), tuple(m.bonds))
        out.append(seen.setdefault(key, m))
    return out


class Simulation:
    def __init__(self, sys: System, pos_nm, H_nm, settings: MDSettings = MDSettings(), dt: float = 0.001,
                 ensemble: str = "nvt", temperature: float = 298.0, gamma: float = 1.0, pressure: float = 1.0,
                 barostat_interval: int = 100, seed: int = 0, vel_nm_ps=None, params=None, log=sys.stdout,
                 neighbor_list: str = "auto", thermostat="langevin", tau_t: float = 1.0):
        H = reduce_box(H_nm)
        check_box(H, settings.pair_cutoff + settings.skin)
        self.sys, self.settings, self.log = sys, settings, log
        self.rigid = RigidMolecules(sys, pos_nm, H)
        self.ff = PGMForceField(sys, H, settings)
        self._r_list = float(jnp.max(jnp.linalg.norm(self.rigid.local, axis=1)))
        self._nb_mode = neighbor_list
        self._make_neighbors(H)
        self._size_lists(self.rigid.body0, H)
        self.integ = Integrator(self.ff, self.rigid, self.nb, dt, ensemble, temperature, gamma, pressure,
                                barostat_interval, params, thermostat=thermostat, tau_t=tau_t)
        self.dt, self.ensemble, self.T0 = dt, ensemble, temperature
        body = self.rigid.body0
        mom = None
        if vel_nm_ps is not None:
            mom = self.rigid.momenta_from_velocities(body, self.rigid.positions(body), jnp.asarray(vel_nm_ps))
        self.state = self.integ.init(body, H, jax.random.PRNGKey(seed), mom)
        self.time_ps = 0.0
        thermo = '' if self.integ.thermostat is None else f" ({self.integ.thermostat.describe()})"
        self._print(f"# pgm_jax MD: {sys.nmol} rigid molecules, {sys.n} atoms, {ensemble.upper()}{thermo}, "
                    f"dt {dt * 1000:g} fs, "
                    f"{settings.precision} precision, PME grid {self.ff.pme.K} order {settings.pme_order}, "
                    f"{settings.describe_cutoffs()}, {self.nb.kind} neighbour list, predictor {settings.predictor}"
                    f"{' (fused)' if settings.fused else ''}, dipole tol {settings.dipole_tol:g}, "
                    f"template fit RMSD {self.rigid.fit_rmsd:.2e} nm, device {jax.devices()[0]}")

    @classmethod
    def from_amber(cls, prmtop: str, coords: str, use_velocities: bool = True, **kw) -> "Simulation":
        mols = _dedupe(read_prmtop_pgm(prmtop, first_residue_only=False))
        sys = System(mols)
        xyz, vel, box = read_coordinates(coords)
        if box is None:
            raise ValueError("coordinates have no periodic box")
        H = box_from_cell(*box) * 0.1
        return cls(sys, xyz * 0.1, H, vel_nm_ps=(vel * 0.1 if (use_velocities and vel is not None) else None), **kw)

    # ----------------------------------------------------------------- observables
    def observables(self) -> dict:
        st = self.state
        ke, ke_trans = (float(x) for x in self.integ.kinetic(st))
        V = float(volume(st.box))
        mass = float(np.sum(self.sys.masses))
        t_tr, t_rot = (float(x) for x in self.integ.temperatures(st))
        out = {"step": int(st.step), "time_ps": self.time_ps, "temp_K": 2 * ke / (self.integ.dof * KB),
               "temp_trans": t_tr, "temp_rot": t_rot,
               "etot": ke + float(st.epot), "ekin": ke, "epot": float(st.epot), "elec": float(st.elec),
               "econs": ke + float(st.epot) + 0.5 * float(jnp.sum(st.aux * st.aux)) - float(st.heat),
               "vdw": float(st.vdw), "volume_nm3": V, "density_g_cm3": mass / V * AMU_NM3_TO_G_CM3,
               "cg_iter": int(st.iters), "cg_iter_max": int(st.max_iters), "cg_resid_max": float(st.resid),
               "cg_mean": float(st.cg_total) / max(int(st.step), 1)}
        if self.ensemble == "npt":
            tries, acc = int(st.mc[0]), int(st.mc[1])
            out["mc_accept"] = acc / max(tries, 1)
        return out

    def _pressure(self, st):
        pos = self.rigid.positions(st.dyn.position)
        W = self.ff.strain_derivative(pos, st.box, self.nb.candidates(st.nbr, st.dyn.position.center, st.box, pos)[0],
                                      st.induction.mu, self.integ.params)
        ke_t = self.integ.kinetic(st)[1]
        return (2.0 * ke_t - jnp.trace(W)) / (3.0 * volume(st.box)) * 16.605390671738466

    def pressure(self) -> float:
        """Instantaneous pressure (bar) from the molecular virial (at the converged dipoles) and the
        centre-of-mass kinetic energy."""
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
        for attempt in range(6):
            new = self.integ.run(start, n)
            jax.block_until_ready(new.epot)
            nb_bad, row_bad = self.nb.failed(new.nbr), bool(new.overflow)
            if not (nb_bad or row_bad):
                break
            body = start.dyn.position
            old = (self.ff.capacity, getattr(self.nb, "cap", None))
            nbr = self._size_lists(body, start.box, 1.3, None if nb_bad else start.nbr)
            if row_bad:                                          # never shrink below what overflowed
                self.ff.grow_rows(old[0])
                if getattr(self.nb, "cap", None) is not None and old[1] is not None:
                    self.nb.cap = max(self.nb.cap, old[1] + 4)
            self.integ.compile()
            self._print(f"# {'neighbour list' if nb_bad else 'row capacity'} overflow in steps {int(start.step)}-"
                        f"{int(start.step) + n}: resized (rows {self.ff.mc or self.nb.cap}, list {nbr.idx.shape[1]}), repeating")
            start = self.integ.forces(start.set(nbr=nbr), False).set(induction=start.induction)
        else:
            raise RuntimeError("neighbour list keeps overflowing")
        body = self.rigid.wrap(new.dyn.position, new.box)
        self.state = new.set(dyn=new.dyn.set(position=body))
        self.time_ps += n * self.dt
        if not np.isfinite(float(new.epot)):
            raise FloatingPointError(f"energy is not finite at step {int(new.step)}")

    def run(self, nsteps: int, report: int = 1000, traj: int = 0, restart: int = 0, prefix: str = "md",
            pressure_every_report: bool = False, append: bool = False):
        block = int(np.gcd.reduce([x for x in (report, traj, restart, nsteps) if x > 0]))
        tfile = NetCDFTrajectory(prefix + ".nc", self.sys.n, append=append) if traj else None
        logf = open(prefix + ".log", "a" if append else "w")
        cols = None
        t0, s0 = time.time(), int(self.state.step)
        done = 0
        while done < nsteps:
            n = min(block, nsteps - done)
            self._advance(n)
            done += n
            step = int(self.state.step)
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
                line = "  " + " ".join(f"{obs[c]:14.6f}" if isinstance(obs[c], float) else f"{obs[c]:14d}" for c in cols)
                logf.write(line + "\n")
                logf.flush()
                self._print(line)
            if tfile is not None and step % traj == 0:
                tfile.write(self.time_ps, self.positions_nm() * 10.0, np.asarray(self.state.box) * 10.0)
            if restart and step % restart == 0:
                self.save(prefix)
        logf.close()
        if restart:
            self.save(prefix)

    # ----------------------------------------------------------------- checkpoints
    def save(self, prefix: str):
        """Amber NetCDF restart (prefix.rst7) and a checkpoint of the complete state (prefix.chk):
        rigid-body coordinates and momenta, forces, box, dipoles and extrapolation history, random
        state, barostat state.  Continuing from it reproduces the run up to floating-point summation
        order (GPU atomics in PME spreading make runs non-bitwise-reproducible anyway)."""
        write_restart(prefix + ".rst7", self.positions_nm() * 10.0, self.velocities_nm_ps() * 10.0,
                      np.asarray(self.state.box) * 10.0, self.time_ps)
        host = jax.tree_util.tree_map(np.asarray, self.state.set(nbr=None))
        with open(prefix + ".chk", "wb") as fh:
            pickle.dump({"state": host, "time_ps": self.time_ps}, fh)

    def load(self, path: str):
        """Continue from a checkpoint written by `save` (same system and settings)."""
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        st = jax.tree_util.tree_map(jnp.asarray, d["state"])
        st = upgrade_state(st, self.state.aux)          # checkpoints from before the thermostat fields
        nbr = self.nb.allocate(self.rigid.positions(st.dyn.position), st.dyn.position.center, st.box)
        self.state = st.set(nbr=nbr)                   # forces, dipoles and history are part of the state
        self.time_ps = d["time_ps"]
