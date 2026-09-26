"""Dipole moment of the periodic cell, its electronic polarizability, and their output during MD.

Cell dipole (e nm; 1 D = units.DEBYE_E_NM = 0.020819434 e nm):

    M = M_q + M_perm + M_ind
    M_q    = sum_k sum_{i in k} q_i (r_i - R_k)    Gaussian charges; molecule k whole, R_k its centre of mass
    M_perm = sum_i p_i                            covalent permanent dipoles (PGMForceField.perm_dipoles)
    M_ind  = sum_i mu_i                           induced dipoles, converged at the same positions

A Gaussian charge or dipole density has the dipole moment of the point multipole at its centre, so M
is the dipole moment of the model's charge density.  The terms follow the electrostatics level
(MDSettings.elec): no permanent dipoles for "q" / "qi", no induced dipoles for "q" / "qp".

Conventions
  * Molecules are whole.  Both engines keep them whole (rigid bodies; the atoms of a flexible
    molecule are never wrapped one by one) and only shift whole molecules by lattice vectors.
    CellDipole takes the engine's positions as they are; it applies no minimum image, which would
    break molecules longer than half the box.
  * A neutral molecule's dipole does not depend on the reference point, so for a cell of neutral
    molecules M_q = sum_i q_i r_i.  It is independent of the origin and of the lattice image in
    which each molecule sits, hence continuous in time across the driver's re-wrapping.
  * A charged molecule (ion, charged residue) contributes its dipole about its centre of mass
    (physical masses, sys.masses, also under hydrogen mass repartitioning), as GROMACS's
    `gmx dipoles`.  The translational part sum_k Q_k R_k is left out: it jumps by Q_k L whenever
    molecule k is re-wrapped, it is the time integral of the ionic current (conduction, not
    polarization), and for a net-charged cell it depends on the origin.  With this convention M is
    defined for every cell, charged or not; for an electrolyte it is the "molecular" dipole M_D of
    the literature (its fluctuations leave out the M_D-current cross correlation).  The net charge
    and the number of charged molecules are written to the .dip header, and
    scripts/dielectric.py refuses such series unless asked for the M_D part explicitly.

Electronic polarizability of the cell and the static dielectric constant.  In tin-foil
(conducting) Ewald boundary conditions (smooth PME omits the k = 0 term) the uniform field F acting
on the charges is the Maxwell field, so eps - 1 = 4 pi (1/3) tr d<M>/dF / V (F in e/nm^2, i.e. the
energy is -KE M.F).  The induced dipoles are adiabatic: they minimise the energy at every nuclear
configuration and carry no thermal fluctuation of their own.  With U(R, F) = min_mu E(R, mu) - KE M.F,
    d<M>/dF = beta KE (<M M> - <M><M>) + <dM/dF>_R,
and dM/dF at fixed nuclei is the cell polarizability alpha_cell (nm^3): the response of the
induced dipoles to a uniform field with every dipole-dipole coupling in the same Ewald sums.  Hence
    eps = eps_inf + (<M.M> - <M>.<M>) / (3 eps0 V kB T),     eps_inf = 1 + 4 pi <alpha_cell / V>,
with alpha_cell = (1/3) tr alpha_cell and M the total dipole (induced dipoles included).  The
fluctuation term alone misses the electronic response: a frozen polarizable crystal has M = 0 at
every step but eps = eps_inf > 1.  For water eps_inf - 1 is about 0.7, one per cent of eps.
`CellDipole.polarizability` solves A m_a = e_a for a = x, y, z with the force field's induction
operator A = alpha^-1 - T and CG, and returns alpha_ab = sum_i (m_b)_{i,a}.

Recording during Simulation.run (arguments dipoles=, induced=; the Amber trajectory and restart
files are unchanged):
  * DipoleRecorder, prefix.dip: every `dipoles` steps M_q, M_perm, M_ind, the volume, the kinetic
    temperature and the mean molecular dipole, and every `alpha_every`-th sample the cell
    polarizability; a text table with a commented header (read_dipoles).  Samples are taken on the
    device inside the driver's blocks (lax.scan over sub-blocks of the integrator), so sampling
    every step needs no host round trip; the samples of a block are kept only once the driver has
    accepted the block (not when it is repeated after a list overflow).
  * InducedDipoleFile, prefix.mu.nc: per-atom induced dipoles (e nm, float32) every `induced`
    steps, NetCDF-3 (dimensions frame, atom, spatial; variables time, step, induced_dipoles).

Units: nm, ps, e, e nm, nm^3 (polarizability volume), K."""
from __future__ import annotations

import functools
import os
import struct

import jax
import jax.numpy as jnp
import numpy as np

from ..units import DEBYE_E_NM
from .box import volume

DIP_COLUMNS = ("step", "time_ps", "temp_K", "volume_nm3", "Mq_x", "Mq_y", "Mq_z", "Mp_x", "Mp_y", "Mp_z",
               "Mi_x", "Mi_y", "Mi_z", "mol_dipole", "alpha_nm3")


class CellDipole:
    """Cell dipole, molecular dipoles and cell polarizability for the system of a PGMForceField.
    Positions must hold whole molecules (as the MD engines keep them); params as for
    PGMForceField.compute (None: the system's initial values)."""

    def __init__(self, ff):
        self.ff = ff
        sys = ff.sys
        self.nmol = sys.nmol
        self.mol = jnp.asarray(sys.mol)
        m = np.asarray(sys.masses, float)
        mmol = np.bincount(np.asarray(sys.mol), weights=m, minlength=sys.nmol)
        self.w = jnp.asarray(m / mmol[np.asarray(sys.mol)])             # centre-of-mass weights within a molecule

    def molecular_charges(self, params=None) -> np.ndarray:
        """Net charge of every molecule (e)."""
        q = np.asarray(self.ff._atoms(params)["q"])
        return np.bincount(np.asarray(self.ff.sys.mol), weights=q, minlength=self.nmol)

    def _parts(self, pos, H, mu, params):
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        P = self.ff._atoms(params)
        com = jax.ops.segment_sum(self.w[:, None] * pos, self.mol, self.nmol)
        qr = P["q"][:, None] * (pos - com[self.mol])
        p = self.ff.perm_dipoles(pos, H, P["cov"])
        return qr, p, jnp.asarray(mu, jnp.float64)

    def components(self, pos, H, mu, params=None):
        """(3, 3) float64, e nm: rows M_q, M_perm, M_ind (their sum is M)."""
        return jnp.stack([jnp.sum(x, axis=0) for x in self._parts(pos, H, mu, params)])

    def molecular(self, pos, H, mu, params=None):
        """(nmol, 3) e nm: the total dipole of every molecule (about its centre of mass)."""
        qr, p, mu = self._parts(pos, H, mu, params)
        return jax.ops.segment_sum(qr + p + mu, self.mol, self.nmol)

    def polarizability(self, pos, H, idx, params=None, tol=None):
        """(3, 3) float64, nm^3: alpha_ab = dM_a/dF_b at fixed nuclei, the response of the induced
        dipoles to a uniform field F (e/nm^2) with Ewald (tin-foil) dipole couplings.  idx: candidate
        rows as for PGMForceField.compute; tol: CG tolerance (default settings.dipole_tol).  Zero
        without induced dipoles; nan if a solve does not converge within settings.max_iter."""
        ff = self.ff
        if not ff.ind:
            return jnp.zeros((3, 3))
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        P = ff._atoms(params)
        g = ff.geometry(pos, H, idx, P)
        S, Gk = ff.pme.setup(pos, H), ff.pme.influence(H)
        alpha = P["alpha"]
        A = ff._operator(g, S, Gk, alpha)
        norm = jnp.mean(alpha) / 3.0                                   # mean|alpha b| for a unit field
        tol = ff.s.dipole_tol if tol is None else tol
        cols = []
        for a in range(3):
            b = jnp.zeros((ff.n, 3), ff.cd).at[:, a].set(1.0)
            m, _, err = ff._cg(g, A, alpha, jnp.zeros_like(b), b, norm, tol=tol, peek=0.0)
            cols.append(jnp.where(err <= tol, jnp.sum(m, axis=0), jnp.nan))
        return jnp.stack(cols, axis=1)


def cell_dipole(sim, params=None) -> dict:
    """M of the current state of a Simulation or FlexibleSimulation: {"charge", "perm", "ind",
    "total"} (e nm, (3,) numpy), "debye" (the same four in D), "molecular" (nmol, 3) e nm."""
    cd = CellDipole(sim.ff)
    params = sim.integ.params if params is None else params
    pos, H, mu = sim.positions_nm(), np.asarray(sim.state.box), sim.state.induction.mu
    c = np.asarray(cd.components(pos, H, mu, params))
    out = {"charge": c[0], "perm": c[1], "ind": c[2], "total": c.sum(0)}
    out["debye"] = {k: out[k] / DEBYE_E_NM for k in ("charge", "perm", "ind", "total")}
    out["molecular"] = np.asarray(cd.molecular(pos, H, mu, params))
    return out


# ----------------------------------------------------------------------------- recording
class DipoleRecorder:
    """Cell-dipole time series of a running Simulation / FlexibleSimulation (driver argument
    run(dipoles=n)), written to a .dip text file.  The driver calls run(state, n) instead of
    Integrator.run for each block, keep() once the block is accepted, flush() after each block."""

    alpha_every = 100               # samples between evaluations of the cell polarizability (3 CG solves)

    def __init__(self, sim, path: str, interval: int, append: bool = False):
        if int(interval) <= 0:
            raise ValueError("the dipole sampling interval must be a positive number of steps")
        self.sim, self.path, self.interval = sim, path, int(interval)
        self.cell = CellDipole(sim.ff)
        self._fns, self._key = {}, None
        self._last, self._kept = None, []
        if not (append and os.path.exists(path)):
            with open(path, "w") as fh:
                fh.write(self.header())

    def header(self) -> str:
        sim = self.sim
        Qk = self.cell.molecular_charges(sim.integ.params)
        th = sim.integ.thermostat
        meta = {"temperature_K": sim.T0, "ensemble": sim.ensemble,
                "thermostat": "none" if th is None else th.describe().replace(" ", "_"),
                "dt_ps": sim.dt, "interval": self.interval, "n_atoms": sim.sys.n, "n_molecules": sim.sys.nmol,
                "net_charge": round(float(Qk.sum()), 6), "charged_molecules": int(np.sum(np.abs(Qk) > 1e-6)),
                "elec": sim.settings.elec, "alpha_every": self.alpha_every}
        lines = ["pgm_jax cell dipole series (pgm_jax.md.dipoles; scripts/dielectric.py)",
                 "cell dipole M: M_q + M_perm + M_ind in e nm (1 D is 0.020819434 e nm); molecules whole,",
                 "charged molecules about their centre of mass; mol_dipole: mean |dipole| of the molecules (e nm);",
                 "alpha_nm3: cell electronic polarizability, 1/3 trace of dM_ind/dF at fixed nuclei, tin-foil",
                 "Ewald (every alpha_every-th sample; nan otherwise or if not converged); temp_K: kinetic temperature"]
        lines += [f"{k} = {v}" for k, v in meta.items()]
        lines.append("columns = " + " ".join(DIP_COLUMNS))
        return "".join(f"# {s}\n" for s in lines)

    # -- device side
    def _alpha(self, st):
        sim, integ = self.sim, self.sim.integ
        pos = sim.rigid.positions(st.dyn.position)
        flex = getattr(sim, "flex", None)                  # neighbour-list centres as in each engine's _forces
        centers = st.dyn.position.center if flex is None else flex.list_centers(pos)
        idx = integ.nb.candidates(st.nbr, centers, st.box, pos)[0]
        return jnp.trace(self.cell.polarizability(pos, st.box, idx, integ.params)) / 3.0

    def _scan(self, st, nchunk: int, chunk: int):
        sim, integ = self.sim, self.sim.integ
        every = self.interval * self.alpha_every

        def body(c, _):
            st, ovf, mi, rs = c
            st = integ._run(st, chunk)                     # resets the per-block maxima: carried here
            c = (st, ovf | st.overflow, jnp.maximum(mi, st.max_iters), jnp.maximum(rs, st.resid))
            pos = sim.rigid.positions(st.dyn.position)
            parts = self.cell._parts(pos, st.box, st.induction.mu, integ.params)
            M = jnp.stack([jnp.sum(x, axis=0) for x in parts])
            mmol = jnp.mean(jnp.linalg.norm(jax.ops.segment_sum(sum(parts), self.cell.mol, self.cell.nmol), axis=1))
            a = jax.lax.cond(st.step % every == 0, self._alpha, lambda s: jnp.full((), jnp.nan), st)
            return c, (st.step, M, volume(st.box), integ.temperature(st), mmol, a)

        c0 = (st, jnp.zeros((), bool), jnp.zeros((), jnp.int32), jnp.zeros((), jnp.float64))
        (st, ovf, mi, rs), out = jax.lax.scan(body, c0, None, length=nchunk)
        return st.set(overflow=ovf, max_iters=mi, resid=rs), out

    # -- host side
    def run(self, st, n: int):
        """Advance n steps like Integrator.run, sampling on the way; the samples wait for keep()."""
        integ = self.sim.integ
        if self._key is not integ.run:                     # integ.compile() changed static sizes: re-trace
            self._key, self._fns = integ.run, {}
        s0 = int(st.step)
        chunk = int(np.gcd.reduce([int(n), self.interval, s0 % self.interval]))
        key = (int(n) // chunk, chunk)
        if key not in self._fns:
            self._fns[key] = jax.jit(functools.partial(self._scan, nchunk=key[0], chunk=key[1]))
        new, samples = self._fns[key](st)
        self._last = (self.sim.time_ps, s0, samples)
        return new

    def keep(self):
        """Accept the samples of the last run() (the driver accepted that block)."""
        if self._last is not None:
            self._kept.append(self._last)
            self._last = None

    def flush(self):
        """Append the accepted samples to the file."""
        lines = []
        for t0, s0, out in self._kept:
            step, M, V, T, mmol, a = (np.asarray(x) for x in out)
            for j in np.nonzero(step % self.interval == 0)[0]:
                m = M[j].reshape(-1)
                lines.append(f"{int(step[j]):10d} {t0 + (int(step[j]) - s0) * self.sim.dt:14.6f} {float(T[j]):9.3f} "
                             f"{float(V[j]):14.8f} " + " ".join(f"{x:17.10e}" for x in m)
                             + f" {float(mmol[j]):14.8e} {float(a[j]):14.8e}\n")
        self._kept = []
        if lines:
            with open(self.path, "a") as fh:
                fh.writelines(lines)


def _value(s: str):
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    return s


def read_dipoles(paths) -> tuple[dict, dict]:
    """Read one or more .dip files (continuation segments, in order).  Returns (header of the first
    file, data): step, time_ps, temp_K, volume_nm3, mol_dipole, alpha_nm3 (F,) and M_charge, M_perm,
    M_ind, M (F, 3) in e nm.  Records superseded by a continuation from an earlier checkpoint (the
    step goes back) are dropped, keeping the later ones."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    meta, rows = None, []
    for p in paths:
        m = {}
        with open(p) as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                k, eq, v = line[1:].partition(" = ")
                if eq and k.strip().isidentifier():
                    m[k.strip()] = _value(v.strip())
        if m.get("columns", "").split() != list(DIP_COLUMNS):
            raise ValueError(f"{p}: not a pgm_jax dipole series (columns {m.get('columns')!r})")
        if meta is None:
            meta = m
        else:
            for k in ("n_atoms", "n_molecules", "temperature_K", "elec", "net_charge"):
                if m.get(k) != meta.get(k):
                    raise ValueError(f"{p}: {k} = {m.get(k)} differs from {paths[0]} ({meta.get(k)})")
        x = np.loadtxt(p, comments="#", ndmin=2)
        rows.append(x.reshape(-1, len(DIP_COLUMNS)))
    x = np.concatenate(rows)
    step = x[:, 0].astype(np.int64)
    later_min = np.minimum.accumulate(step[::-1])[::-1]              # min over rows i..end
    keep = np.append(step[:-1] < later_min[1:], True) if len(step) else np.zeros(0, bool)
    x = x[keep]
    d = {c: x[:, k] for k, c in enumerate(DIP_COLUMNS) if c in ("time_ps", "temp_K", "volume_nm3", "mol_dipole", "alpha_nm3")}
    d["step"] = x[:, 0].astype(np.int64)
    d["M_charge"], d["M_perm"], d["M_ind"] = x[:, 4:7], x[:, 7:10], x[:, 10:13]
    d["M"] = x[:, 4:7] + x[:, 7:10] + x[:, 10:13]
    return meta, d


# ----------------------------------------------------------------------------- per-atom induced dipoles
class InducedDipoleFile:
    """Per-atom induced dipoles in an appendable NetCDF-3 file (64-bit offsets): dimensions frame
    (unlimited), atom, spatial; variables time (ps, float64), step (int32) and induced_dipoles
    (frame, atom, spatial; e nm, float32).  The header is written once; each frame appends one
    record and bumps the record count, so the file is valid after every frame (scipy.io.netcdf_file,
    netCDF4, xarray read it)."""

    NC_DIM, NC_VAR, NC_ATT = 10, 11, 12
    CHAR, INT, FLOAT, DOUBLE = 2, 4, 5, 6

    def __init__(self, path: str, n_atoms: int, append: bool = False):
        self.path, self.n = path, int(n_atoms)
        self.recsize = 8 + 4 + 12 * self.n
        if append and os.path.exists(path):
            with open(path, "rb") as fh:
                fh.seek(4)
                self.nframes = struct.unpack(">i", fh.read(4))[0]
            return
        self.nframes = 0
        self._write_header()

    @staticmethod
    def _name(s: str) -> bytes:
        b = s.encode()
        return struct.pack(">i", len(b)) + b + b"\0" * (-len(b) % 4)

    def _atts(self, atts: dict) -> bytes:
        if not atts:
            return b"\0" * 8
        out = struct.pack(">ii", self.NC_ATT, len(atts))
        for k, v in atts.items():
            b = v.encode()
            out += self._name(k) + struct.pack(">ii", self.CHAR, len(b)) + b + b"\0" * (-len(b) % 4)
        return out

    def _write_header(self):
        dims = [("frame", 0), ("atom", self.n), ("spatial", 3)]
        vars_ = [("time", [0], self.DOUBLE, {"units": "picosecond"}, 8),
                 ("step", [0], self.INT, {}, 4),
                 ("induced_dipoles", [0, 1, 2], self.FLOAT, {"units": "e nm"}, 12 * self.n)]
        gatts = {"title": "pgm_jax induced dipoles", "program": "pgm_jax", "Conventions": "pgm_jax induced dipoles 1.0"}

        def header(begins):
            h = b"CDF\x02" + struct.pack(">i", self.nframes)
            h += struct.pack(">ii", self.NC_DIM, len(dims)) + b"".join(self._name(k) + struct.pack(">i", v) for k, v in dims)
            h += self._atts(gatts) + struct.pack(">ii", self.NC_VAR, len(vars_))
            for (name, vd, t, att, size), beg in zip(vars_, begins):
                h += self._name(name) + struct.pack(">i", len(vd)) + b"".join(struct.pack(">i", d) for d in vd)
                h += self._atts(att) + struct.pack(">ii", t, size) + struct.pack(">q", beg)
            return h

        off = len(header([0] * len(vars_)))
        begins = [off, off + 8, off + 12]
        with open(self.path, "wb") as fh:
            fh.write(header(begins))

    def write(self, step: int, time_ps: float, mu):
        mu = np.asarray(mu, ">f4").reshape(-1)
        if mu.size != 3 * self.n:
            raise ValueError(f"expected {self.n} induced dipoles, got {mu.size // 3}")
        rec = struct.pack(">d", float(time_ps)) + struct.pack(">i", int(step)) + mu.tobytes()
        with open(self.path, "r+b") as fh:
            fh.seek(0, 2)
            fh.write(rec)
            self.nframes += 1
            fh.seek(4)
            fh.write(struct.pack(">i", self.nframes))
