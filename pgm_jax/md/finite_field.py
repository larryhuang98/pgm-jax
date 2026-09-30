"""Finite-field static dielectric constant: replicas of one system in several uniform fields.

Contents: `FieldReplicas` (copies of one system in several uniform fields, advanced together as
one vmapped program; the cell dipole of every replica written to prefix.ffd) and `read_series`
(reads those files).  The analysis of the series (`analyse`, `predicted_errors`) is in
pgm_jax/analysis/finite_field.py.

With tin-foil (conducting) boundary conditions the applied field E is the Maxwell field in the
sample (md/efield.py), so in the linear regime

    eps = 1 + <M . e>_E / (eps0 V |E|),      e = E / |E|,

and with a pair of runs at +E and -E

    eps = 1 + (<M . e>_{+E} - <M . e>_{-E}) / (2 eps0 V |E|),

which removes the zero-field bias of a finite run (<M>_0 is zero only on average) and the even
(quadratic) part of the response.  The cubic part (dielectric saturation) is checked by several
|E|: eps(|E|) = eps(0) - c |E|^2 in the non-linear regime.

The statistical error.  The mean of M . e over a run of length T with integrated correlation time
tau_M has the variance 2 tau_M <dM_e^2> / T, and <dM_e^2> = (eps - eps_inf) eps0 V kB T (the
fluctuation formula), so the error of eps from a +-E pair (each replica T long) is

    sigma_pair = sqrt((eps - eps_inf) kB T / (eps0 V)) / |E| x sqrt(tau_M / T),

against sigma_fluct = (eps - eps_inf) sqrt(2 tau_M / (3 T')) from the fluctuations of a zero-field run
of length T'.  At equal cost (T' = 2T) the ratio of the variances, i.e. the cost ratio at equal
error, is (<M.e> / sd(M_e))^2 / 3: the induced mean dipole must exceed the thermal fluctuation of M,
which favours large fields (up to dielectric saturation) and large boxes.  `analyse`
(analysis/finite_field.py) reports the measured errors, and `predicted_errors` these estimates.

    sim = Simulation(sys, pos, H, settings, thermostat="bussi", efield=(0, 0, 0))
    rep = FieldReplicas(sim, [(0, 0, 0.1), (0, 0, -0.1), (0, 0, 0.2), (0, 0, -0.2), (0, 0, 0)])
    rep.run(nsteps, sample_every=25, prefix="ff")   # prefix.ffd: M of every replica every 25 steps
    table = analyse(*read_series("ff.ffd"), skip_ps=50)   # pgm_jax.analysis.finite_field

Units: V/nm, e nm, nm^3, K.

See also docs/dielectric.md and md/efield.py.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TextIO

import jax
import jax.numpy as jnp
import numpy as np

from .driver import LogTable, Stopwatch, read_checkpoint, write_checkpoint
from .engine import OPTIONAL_STATE
from .remd import MDReplicas, _stack

if TYPE_CHECKING:
    from jax.typing import ArrayLike


class FieldReplicas(MDReplicas):
    """Copies of one simulation's state in several uniform fields, advanced together by jax.vmap.

    The replicas of `sim` (a Simulation or FlexibleSimulation created with efield=..., NVE or NVT)
    in the uniform fields `fields`, with independent momenta and thermostat streams (`seed`),
    advanced together by MDReplicas' batched engine (shared static sizes, overflow handling,
    re-wrapping).  Each replica's field amplitude is its MDState.efield.  No exchanges: the replicas
    are independent trajectories at the same temperature.

    Attributes
    ----------
    fields : np.ndarray (R, 3)
        Field (or D/eps0 for a constant-displacement simulation) of every replica [V/nm].

    Other attributes as MDReplicas (batched mode; `temperatures` all equal to the simulation's).
    """

    def __init__(self, sim: Any, fields: ArrayLike, seed: int = 0, log: TextIO | None = None) -> None:
        """Replicas of `sim`'s current state in the given fields.

        Parameters
        ----------
        sim : Simulation or FlexibleSimulation
            Created with efield=... (any amplitude; each replica sets its own), NVE or NVT.
        fields : ArrayLike (R, 3)
            Field of every replica [V/nm].
        seed : int
            Seed of the replicas' momenta and thermostat streams.
        log : text stream or None
            Receives the rows of the log table of `run` too.

        Raises
        ------
        ValueError
            A simulation without a field, NPT, or multiple time stepping.
        NotImplementedError
            Charged molecules (the itinerant dipole of re-wrapped ions is not booked).
        """
        self.log = log
        integ = sim.integ
        if integ.efield is None:
            raise ValueError("create the simulation with efield=... (any amplitude; each replica sets its own)")
        if sim.ensemble == "npt":
            raise ValueError(
                "batched field replicas run NVE or NVT (under vmap the barostat's trial energy would be "
                "evaluated every step); equilibrate the density first"
            )
        if getattr(integ, "mts", None) is not None:
            raise ValueError("field replicas with multiple time stepping are not supported")
        if integ.field_charged:
            raise NotImplementedError(
                "field replicas of systems with charged molecules: the batched driver does not "
                "book the itinerant dipole of re-wrapped ions (use Simulation with efield=)"
            )
        F = np.asarray(fields, float).reshape(-1, 3)
        self.sim, self.integ, self.batched = sim, integ, True
        self.fields = F
        self.n, self.dt = len(F), float(sim.dt)
        self.temperatures = np.full(self.n, float(sim.T0))
        self.pressure = None
        self.time_ps = 0.0
        base = sim.state
        self._template = base.nbr
        states = []
        for E, key in zip(F, jax.random.split(jax.random.PRNGKey(int(seed)), self.n)):
            st = integ.init(base.dyn.position, base.box, key).set(nbr=base.nbr)
            states.append(integ.forces(st.set(efield=jnp.asarray(E, jnp.float64), induction=base.induction), False))
        self._exchange_seq = jax.jit(self._exchange_one)
        self.S = _stack(states)
        self._build()

    def dipoles(self) -> np.ndarray:
        """Return the cell dipole M = sum q r + sum p + sum mu (R, 3) [e nm] of every replica.

        The dipole of the last force evaluation (MDState.fdip).
        """
        return np.asarray(self.S.fdip)

    def header(self, extra: dict[str, Any] | None = None) -> str:
        """Return the "# key = value" header of prefix.ffd.

        The lines give the temperature, ensemble, time step, sizes, volume, field kind, the field of
        every replica (field_k = Ex Ey Ez [V/nm]) and the column names (step, time_ps, then M_x, M_y,
        M_z [e nm] of every replica).

        Parameters
        ----------
        extra : dict, optional
            Further header entries.

        Returns
        -------
        str
            The header lines, each starting with "# ".
        """
        sim = self.sim
        V = float(np.abs(np.linalg.det(np.asarray(self.S.box)[0])))
        meta = {
            "temperature_K": float(sim.T0),
            "ensemble": sim.ensemble,
            "dt_ps": self.dt,
            "n_atoms": sim.sys.n,
            "n_molecules": sim.sys.nmol,
            "volume_nm3": V,
            "replicas": self.n,
            "field_kind": self.integ.efield.kind,
        }
        meta.update(extra or {})
        lines = [
            "pgm_jax finite-field series (pgm_jax.md.finite_field): cell dipole M (e nm) of every replica",
            "M = sum q r + sum p + sum mu (whole molecules); fields in V/nm",
        ]
        lines += [f"{k} = {v}" for k, v in meta.items()]
        lines += [f"field_{k} = {E[0]:.10g} {E[1]:.10g} {E[2]:.10g}" for k, E in enumerate(self.fields)]
        lines.append("columns = step time_ps " + " ".join(f"M{k}_{c}" for k in range(self.n) for c in "xyz"))
        return "".join(f"# {s}\n" for s in lines)

    def run(
        self,
        nsteps: int,
        *,
        sample_every: int = 25,
        prefix: str = "ff",
        report_every: int = 5000,
        checkpoint_every: int = 0,
        append: bool = False,
        extra: dict | None = None,
    ) -> None:
        """Advance every replica nsteps with output files.

        Parameters
        ----------
        nsteps : int
            Steps (a multiple of `sample_every`).
        sample_every : int
            Steps between samples of the cell dipole M [e nm] of every replica, written to
            prefix.ffd (header: `header`).
        prefix : str
            Path prefix of the files.
        report_every : int
            Steps between rows of the log table prefix.log (mean, minimum and maximum temperature
            [K], mean CG iterations, ns/day per replica and aggregate) and flushes of prefix.ffd;
            a multiple of `sample_every` (0: none).
        checkpoint_every : int
            Steps between checkpoints prefix.ffchk (0: none; always one at the end).
        append : bool
            Continue existing files.
        extra : dict, optional
            Further header entries of prefix.ffd.

        Raises
        ------
        ValueError
            nsteps or report_every not a multiple of sample_every.
        """
        every, report, restart = sample_every, report_every, checkpoint_every
        if nsteps % every or (report and report % every):
            raise ValueError("nsteps and report_every must be multiples of sample_every")
        path = prefix + ".ffd"
        if not (append and os.path.exists(path)):
            with open(path, "w") as fh:
                fh.write(self.header(extra))
        table = LogTable(prefix + ".log", append=append, echo=self.log)
        clock = Stopwatch(0, self.dt)
        done, rows = 0, []
        while done < nsteps:
            self.advance(every)
            done += every
            step = int(np.asarray(self.S.step)[0])
            rows.append(
                f"{step:10d} {self.time_ps:12.4f} " + " ".join(f"{x:.9e}" for x in self.dipoles().reshape(-1)) + "\n"
            )
            if report and done % report == 0:
                with open(path, "a") as fh:
                    fh.writelines(rows)
                rows = []
                T = [float(self.integ.temperature(self.state(k))) for k in range(self.n)]
                nsd = clock.ns_per_day(done)
                table.write(
                    {
                        "step": step,
                        "time_ps": self.time_ps,
                        "temp_mean": float(np.mean(T)),
                        "temp_min": float(np.min(T)),
                        "temp_max": float(np.max(T)),
                        "cg_mean": float(np.mean(np.asarray(self.S.cg_total))) / max(step, 1),
                        "ns_per_day": nsd,
                        "ns_per_day_agg": nsd * self.n,
                    }
                )
            if restart and done % restart == 0:
                self.save_checkpoint(prefix + ".ffchk")
        if rows:
            with open(path, "a") as fh:
                fh.writelines(rows)
        table.close()
        self.save_checkpoint(prefix + ".ffchk")

    def save_checkpoint(self, path: str) -> None:
        """Write a checkpoint of every replica and the fields.

        driver.write_checkpoint with kind "field-replicas".

        Parameters
        ----------
        path : str
            The file (prefix.ffchk in `run`).
        """
        d = self.state_dict()
        d["fields"] = self.fields
        write_checkpoint(path, "field-replicas", d)

    def load_checkpoint(self, path: str) -> None:
        """Continue from a checkpoint written by `save_checkpoint` (or a legacy pickle .ffchk).

        Legacy pickle checkpoints are those of pgm_jax up to commit e72c57c; the system and the
        fields must be the same.

        Parameters
        ----------
        path : str
            The checkpoint file.

        Raises
        ------
        ValueError
            Another kind of checkpoint or other fields.
        """
        d = read_checkpoint(path, "field-replicas", self.state_template(), OPTIONAL_STATE)
        if "fields" not in d or not np.allclose(d["fields"], self.fields):
            raise ValueError("checkpoint fields differ")
        self.load_state_dict(d)


def read_series(paths: str | os.PathLike | Sequence[str | os.PathLike]) -> tuple[dict, dict]:
    """Read the cell-dipole series of one or more .ffd files.

    Continuations are given in order; records whose step goes back (a restart from an earlier
    checkpoint) are superseded by the later ones.

    Parameters
    ----------
    paths : str, os.PathLike or Sequence of them
        The .ffd files of FieldReplicas.run.

    Returns
    -------
    meta : dict
        Header entries (numbers as float, others as str) of the first file, and "fields" (R, 3)
        [V/nm].
    data : dict
        "step" (F,) int64, "time_ps" (F,) [ps], "M" (F, R, 3) cell dipoles [e nm].

    Raises
    ------
    ValueError
        If the files have different fields.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    meta, rows = None, []
    for p in paths:
        m, fields = {}, {}
        with open(p) as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                k, eq, v = line[1:].partition(" = ")
                k = k.strip()
                if not eq:
                    continue
                if k.startswith("field_") and k[6:].isdigit():
                    fields[int(k[6:])] = [float(x) for x in v.split()]
                else:
                    try:
                        m[k] = float(v)
                    except ValueError:
                        m[k] = v.strip()
        m["fields"] = np.array([fields[k] for k in sorted(fields)])
        if meta is None:
            meta = m
        elif not np.allclose(m["fields"], meta["fields"]):
            raise ValueError(f"{p}: fields differ from {paths[0]}")
        rows.append(np.loadtxt(p, comments="#", ndmin=2))
    x = np.concatenate(rows)
    step = x[:, 0].astype(np.int64)
    later_min = np.minimum.accumulate(step[::-1])[::-1]
    keep = np.append(step[:-1] < later_min[1:], True)
    x = x[keep]
    R = len(meta["fields"])
    return meta, {"step": x[:, 0].astype(np.int64), "time_ps": x[:, 1], "M": x[:, 2:].reshape(len(x), R, 3)}
