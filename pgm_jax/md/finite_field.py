"""Finite-field static dielectric constant: copies of one system in several uniform fields,
advanced together (one vmapped program), and the analysis of their cell dipoles.

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
which favours large fields (up to dielectric saturation) and large boxes.  `analyse` reports the
measured errors, and `predicted_errors` these estimates.

    sim = Simulation(sys, pos, H, settings, ensemble="nvt", thermostat="bussi", efield=(0, 0, 0))
    rep = FieldReplicas(sim, [(0, 0, 0.1), (0, 0, -0.1), (0, 0, 0.2), (0, 0, -0.2), (0, 0, 0)])
    rep.run(nsteps, every=25, prefix="ff")          # prefix.ffd: M of every replica every 25 steps
    table = analyse(*read_series("ff.ffd"), skip_ps=50)

Units: V/nm, e nm, nm^3, K."""

from __future__ import annotations

import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from .remd import MDReplicas, _stack


class FieldReplicas(MDReplicas):
    """Copies of the state of `sim` (a Simulation or FlexibleSimulation created with efield=...,
    NVE or NVT) in the uniform fields `fields` ((R, 3) V/nm), with independent momenta and
    thermostat streams (`seed`), advanced together by jax.vmap (MDReplicas' batched engine: shared
    static sizes, overflow handling, re-wrapping).  No exchanges: the replicas are independent
    trajectories at the same temperature."""

    def __init__(self, sim, fields, seed: int = 0):
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
        """(R, 3) e nm: the cell dipole M = sum q r + sum p + sum mu of every replica (last force evaluation)."""
        return np.asarray(self.S.fdip)

    def header(self, extra: dict | None = None) -> str:
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
        every: int = 25,
        prefix: str = "ff",
        report: int = 5000,
        append: bool = False,
        restart: int = 0,
        extra: dict | None = None,
        log=None,
    ):
        """Advance every replica nsteps, writing M every `every` steps to prefix.ffd, a log line every
        `report` steps to prefix.log (temperatures, energies, CG iterations, ns/day per replica and
        aggregate) and, every `restart` steps and at the end, a checkpoint prefix.ffchk."""
        if nsteps % every or (report and report % every):
            raise ValueError("nsteps and report must be multiples of every")
        path = prefix + ".ffd"
        if not (append and os.path.exists(path)):
            with open(path, "w") as fh:
                fh.write(self.header(extra))
        logf = open(prefix + ".log", "a" if append else "w")
        t0, done, rows = time.time(), 0, []
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
                T = [self.integ.temperature(self.state(k)) for k in range(self.n)]
                el = time.time() - t0
                nsd = done * self.dt / 1000.0 / max(el, 1e-9) * 86400.0
                line = (
                    f"step {step} t {self.time_ps:.2f} ps  T {np.mean(T):.1f} (min {np.min(T):.1f} max "
                    f"{np.max(T):.1f})  "
                    f"cg {float(np.mean(np.asarray(self.S.cg_total))) / max(step, 1):.2f}  "
                    f"{nsd:.2f} ns/day per replica, {nsd * self.n:.1f} aggregate"
                )
                logf.write(line + "\n")
                logf.flush()
                if log is not None:
                    print(line, file=log, flush=True)
            if restart and done % restart == 0:
                self.save(prefix + ".ffchk")
        if rows:
            with open(path, "a") as fh:
                fh.writelines(rows)
        logf.close()
        self.save(prefix + ".ffchk")

    def save(self, path: str):
        d = self.state_dict()
        d["fields"] = self.fields
        with open(path, "wb") as fh:
            pickle.dump(d, fh)

    def load(self, path: str):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if not np.allclose(d["fields"], self.fields):
            raise ValueError("checkpoint fields differ")
        self.load_state_dict(d)


def read_series(paths) -> tuple[dict, dict]:
    """(meta, data) from one or more .ffd files (continuations in order; records whose step goes
    back are superseded): data["step"], data["time_ps"], data["M"] (F, R, 3), meta["fields"] (R, 3)."""
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
