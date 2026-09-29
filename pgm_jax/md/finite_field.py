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

from ..units import E_CHARGE_C, EPS0_SI, KB_SI
from .efield import EPS_FACTOR, finite_d_eps
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


def block_mean(x, nblocks: int = 10):
    """Mean and its standard error from `nblocks` contiguous block means."""
    x = np.asarray(x, float)
    n = len(x) // nblocks * nblocks
    b = x[len(x) - n :].reshape(nblocks, -1).mean(1)
    return float(x.mean()), float(b.std(ddof=1) / np.sqrt(nblocks))


def correlation_time(x, dt: float) -> float:
    """Integrated autocorrelation time (ps) of a series sampled every dt ps (sum of the normalised
    autocorrelation up to its first zero crossing)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    c = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
    c = c / c[0]
    stop = np.argmax(c <= 0) if np.any(c <= 0) else n
    return float(dt * (0.5 + np.sum(c[1:stop])))


def fluctuation_eps(M, V, T, eps_inf: float = 1.0, nblocks: int = 10):
    """eps = eps_inf + (<M.M> - <M>.<M>) / (3 eps0 V kB T) (tin-foil) and its jackknife error over
    contiguous blocks; M (F, 3) e nm, V nm^3."""
    M = np.asarray(M, float)
    c = (E_CHARGE_C * 1e-9) ** 2 / (3 * EPS0_SI * V * 1e-27 * KB_SI * T)

    def est(m):
        return eps_inf + c * (np.mean(np.sum(m * m, 1)) - np.sum(np.mean(m, 0) ** 2))

    n = len(M) // nblocks * nblocks
    Mb = M[len(M) - n :].reshape(nblocks, -1, 3)
    jk = np.array([est(np.concatenate([Mb[j] for j in range(nblocks) if j != i])) for i in range(nblocks)])
    return float(est(M)), float(np.sqrt((nblocks - 1) / nblocks * np.sum((jk - jk.mean()) ** 2)))


def analyse(meta: dict, data: dict, skip_ps: float = 50.0, nblocks: int = 10, eps_inf: float | None = None) -> dict:
    """Finite-field eps of every replica with a nonzero field (single-sided, with the block error),
    of every +-E pair (antisymmetric combination) and, for zero-field replicas, the fluctuation
    estimate (with eps_inf, default meta['eps_inf'] or 1) and the correlation time of M.  Series at
    constant displacement (meta field_kind = D; fields are D/eps0) give eps = D / (D - <M.e>/(eps0 V))."""
    disp = meta.get("field_kind", "E") == "D"

    def eps_of(m, err, mag):
        if not disp:
            return 1 + EPS_FACTOR * m / (V * mag), EPS_FACTOR * err / (V * mag)
        e = float(finite_d_eps(m, V, mag))
        return e, e * e * EPS_FACTOR * err / (V * mag)

    V, T = float(meta["volume_nm3"]), float(meta["temperature_K"])
    if eps_inf is None:
        eps_inf = float(meta.get("eps_inf", 1.0))
    sel = data["time_ps"] >= data["time_ps"][0] + skip_ps
    M = data["M"][sel]
    t = data["time_ps"][sel]
    dt = float(np.median(np.diff(t)))
    F = np.asarray(meta["fields"])
    out = {
        "volume_nm3": V,
        "temperature_K": T,
        "run_ps": float(t[-1] - t[0] + dt),
        "single": [],
        "pairs": [],
        "zero": [],
    }
    for k, E in enumerate(F):
        mag = float(np.linalg.norm(E))
        if mag == 0.0 and disp:
            continue
        if mag == 0.0:
            eps, err = fluctuation_eps(M[:, k], V, T, eps_inf, nblocks)
            out["zero"].append(
                {
                    "replica": k,
                    "eps": eps,
                    "err": err,
                    "tau_ps": correlation_time(M[:, k, 2], dt),
                    "M_mean": M[:, k].mean(0).tolist(),
                }
            )
            continue
        e = E / mag
        m, err = block_mean(M[:, k] @ e, nblocks)
        tau = correlation_time(M[:, k] @ e, dt)
        eps, eerr = eps_of(m, err, mag)
        out["single"].append(
            {
                "replica": k,
                "E": E.tolist(),
                "E_mag": mag,
                "M_par": m,
                "M_par_err": err,
                "eps": eps,
                "err": eerr,
                "tau_ps": tau,
                "M_perp_rms": float(np.sqrt(np.mean(np.sum((M[:, k] - np.outer(M[:, k] @ e, e)) ** 2, 1)))),
            }
        )
    used = set()
    for i, Ei in enumerate(F):
        for j, Ej in enumerate(F):
            if j <= i or i in used or j in used or np.linalg.norm(Ei) == 0 or not np.allclose(Ei, -Ej):
                continue
            used |= {i, j}
            mag = float(np.linalg.norm(Ei))
            e = Ei / mag
            d = 0.5 * (M[:, i] @ e - M[:, j] @ e)
            m, err = block_mean(d, nblocks)
            eps, eerr = eps_of(m, err, mag)
            out["pairs"].append(
                {"replicas": (i, j), "E_mag": mag, "eps": eps, "err": eerr, "tau_ps": correlation_time(d, dt)}
            )
    out["fits"] = []
    mags = sorted({p["E_mag"] for p in out["pairs"]})
    for emax in mags[1:]:
        f = saturation_fit([p for p in out["pairs"] if p["E_mag"] <= emax + 1e-12])
        if f is not None:
            out["fits"].append(dict(f, E_max=emax))
    return out


def saturation_fit(pairs):
    """Weighted least squares eps(E) = eps0 - c E^2 over +-E pairs (the leading non-linear term of
    the response, dielectric saturation); returns eps0, c and their errors, or None for < 2 fields."""
    E = np.array([p["E_mag"] for p in pairs])
    if len(np.unique(E)) < 2:
        return None
    y = np.array([p["eps"] for p in pairs])
    w = 1.0 / np.array([p["err"] for p in pairs]) ** 2
    A = np.stack([np.ones_like(E), -E * E], 1)
    C = np.linalg.inv(A.T @ (w[:, None] * A))
    b = C @ A.T @ (w * y)
    chi2 = float(np.sum(w * (y - A @ b) ** 2))
    return {
        "eps0": float(b[0]),
        "eps0_err": float(np.sqrt(C[0, 0])),
        "c": float(b[1]),
        "c_err": float(np.sqrt(C[1, 1])),
        "chi2": chi2,
        "n": int(len(E)),
    }


def predicted_errors(eps: float, eps_inf: float, V: float, T: float, E: float, tau_ps: float, run_ps: float) -> dict:
    """Statistical errors expected for eps from a +-E pair (each replica run_ps long) and from the
    fluctuations of a zero-field run of the same total length (2 run_ps), for a Gaussian M with
    integrated correlation time tau_ps (module docstring), and their cost ratio at equal error."""
    var_Me = (eps - eps_inf) * EPS0_SI * V * 1e-27 * KB_SI * T / (E_CHARGE_C * 1e-9) ** 2  # (e nm)^2, one component
    s_mean = np.sqrt(var_Me * 2 * tau_ps / run_ps)  # error of <M_e> of one run
    s_ff = EPS_FACTOR / (V * E) * s_mean / np.sqrt(2)  # the +-E combination
    s_fl = (eps - eps_inf) * np.sqrt(2 * tau_ps / (3 * 2 * run_ps))
    return {"sigma_ff": float(s_ff), "sigma_fluct_same_cost": float(s_fl), "cost_ratio": float((s_fl / s_ff) ** 2)}
