"""Model-system validation of the path-integral integrator (md/pimd.py; docs/pimd.md).

Three model systems with known answers, run with PIMDIntegrator and PotentialEngine directly:

- harmonic: 256 independent 3D harmonic oscillators (mass 1.008 amu) for each angular frequency
  --omega-per-ps and bead number --beads: <V> and <K> (primitive and centroid-virial estimators)
  per degree of freedom vs the exact P-bead value kT sum_k w^2 / (w_k^2 + w^2) / 2 and the quantum
  value (hbar w / 4) coth(beta hbar w / 2);
- free: 512 free particles, 16 beads: centroid and normal-mode temperatures and the spreads of the
  internal modes vs kT_P / (m w_k^2) with PILE-L, PILE-G and TRPMD;
- nve: RPMD (no thermostat) of 64 particles in an anharmonic (quartic Morse) O-H-like well
  (w0 = 700 rad/ps), 32 beads: fluctuation and drift of the conserved ring-polymer energy H_P for
  the time steps --dts-fs, Cayley and exact free-ring propagators.

Usage:

    python scripts/pimd/pimd_validate.py harmonic      # P = 1..64, omega 100 and 700 rad/ps
    python scripts/pimd/pimd_validate.py free          # PILE-L, PILE-G, TRPMD
    python scripts/pimd/pimd_validate.py nve --dts-fs 0.4 0.2 0.1
    python scripts/pimd/pimd_validate.py --help

Inputs: none.
Outputs: <out>/<what><tag>.json (the options and the results; default --out data/validation/pimd)
and a printed line per case.
Units: --temperature-K K, --omega-per-ps rad/ps, --dts-fs fs, --time-ps ps; energies in kJ/mol
(per degree of freedom where named so), temperatures in K.
Runtime: CPU; minutes per case (harmonic with 64 beads is the longest).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.analysis.stats import block_mean
from pgm_jax.cli.args import add_temperature_arg
from pgm_jax.md.pimd import PILE, PIMDIntegrator, PotentialEngine, RingPolymer
from pgm_jax.paths import repo_path
from pgm_jax.units import HBAR_KJMOL_PS, KB

jax.config.update("jax_enable_x64", True)
OUT = repo_path("data", "validation", "pimd")


def block_err(x: np.ndarray, nb: int = 20) -> float:
    """Return the standard error of the mean of x from nb contiguous blocks."""
    return block_mean(x, nb)[1]


def sample(integ: PIMDIntegrator, st: object, nsamp: int, every: int) -> tuple[object, np.ndarray]:
    """Advance the ring polymer and sample the estimators every `every` steps (one jitted scan).

    Parameters
    ----------
    integ : PIMDIntegrator
        The integrator.
    st : state of `integ`
        Start state.
    nsamp : int
        Number of samples.
    every : int
        Steps between samples [steps].

    Returns
    -------
    state
        The final state.
    samples : np.ndarray (nsamp, 5)
        Per sample: potential energy, primitive and centroid-virial kinetic energies (summed over
        atoms), centroid temperature [K], conserved energy; energies in kJ/mol.
    """

    def body(s, _):
        """One scan step: `every` MD steps, then the estimators."""
        s = integ._run(s, every)
        e = integ.estimators(s)
        return s, jnp.stack([e["epot"], e["prim"].sum(), e["cv"].sum(), e["t_centroid"], e["econs"]])

    st, X = jax.jit(lambda s: jax.lax.scan(body, s, None, length=nsamp))(st)
    return st, np.asarray(X)


def harmonic(a: argparse.Namespace) -> list[dict]:
    """Run the harmonic-oscillator validation; return one result row per (omega, P).

    Each run: equilibration for 20 / omega, then --samples samples every 1 / omega (time step
    --dt-factor / omega), thermostat PILE(--thermostat) with tau_centroid = 1 / omega.
    """
    T = a.temperature_K
    beta = 1.0 / (KB * T)
    n, m = 256, 1.008
    rows = []
    for w in a.omega_per_ps:
        # quantum <K> = <V> per degree of freedom
        q = 0.25 * HBAR_KJMOL_PS * w / math.tanh(0.5 * beta * HBAR_KJMOL_PS * w)
        dt = a.dt_factor / w
        for P in a.beads:
            r = RingPolymer(P, T)
            ex = 0.5 / beta * float(np.sum(w**2 / (r.omega**2 + w**2)))
            eng = PotentialEngine(lambda x, box, w=w: 0.5 * m * w**2 * jnp.sum(x * x))
            integ = PIMDIntegrator(eng, np.full(n, m), P, T, dt, "pimd", PILE(a.thermostat, tau_centroid=1.0 / w))
            t0 = time.time()
            st = integ.run(integ.init(jnp.zeros((n, 3)), jnp.eye(3), jax.random.PRNGKey(P)), int(20 / (w * dt)))
            st, X = sample(integ, st, a.samples, max(1, int(round(1.0 / (w * dt)))))
            X = X[:, :3] / (3 * n)
            row = {
                "omega_rad_ps": w,
                "beta_hbar_omega": beta * HBAR_KJMOL_PS * w,
                "P": P,
                "dt_fs": dt * 1e3,
                "exact_P": ex,
                "quantum": q,
                "classical": 0.5 / beta,
                "V": float(X[:, 0].mean()),
                "V_err": block_err(X[:, 0]),
                "K_prim": float(X[:, 1].mean()),
                "K_prim_err": block_err(X[:, 1]),
                "K_cv": float(X[:, 2].mean()),
                "K_cv_err": block_err(X[:, 2]),
                "K_prim_sd": float(X[:, 1].std()),
                "K_cv_sd": float(X[:, 2].std()),
                "rel_exact_P_minus_quantum": ex / q - 1.0,
                "time_s": time.time() - t0,
            }
            rows.append(row)
            print(
                f"w {w:6.0f} bhw {row['beta_hbar_omega']:5.2f} P {P:3d}: exact_P {ex:.5f} (quantum {q:.5f}, "
                f"{100 * (ex / q - 1):+.3f} %)  V {row['V']:.5f}+-{row['V_err']:.5f}  Kprim {row['K_prim']:.5f}+-"
                f"{row['K_prim_err']:.5f}  Kcv {row['K_cv']:.5f}+-{row['K_cv_err']:.5f} kJ/mol/dof  "
                f"({row['time_s']:.0f} s)",
                flush=True,
            )
    return rows


def free(a: argparse.Namespace) -> dict:
    """Run the free-particle validation (16 beads, dt 0.5 fs); return the results per thermostat.

    Keys "pimd/pile-l", "pimd/pile-g", "trpmd": mode temperatures [K] with block errors, ratio of
    the sampled internal-mode spreads to kT_P / (m w_k^2), and the kinetic-energy estimators per
    atom [kJ/mol].
    """
    T, P, n, m = a.temperature_K, 16, 512, 1.008
    out = {}
    for mode, th in (("pimd", "pile-l"), ("pimd", "pile-g"), ("trpmd", "pile-l")):
        eng = PotentialEngine(lambda x, box: 0.0 * jnp.sum(x))
        integ = PIMDIntegrator(eng, np.full(n, m), P, T, 0.0005, mode, PILE(th, tau_centroid=0.05))
        r = integ.ring
        st = integ.run(integ.init(jnp.zeros((n, 3)), jnp.eye(3), jax.random.PRNGKey(0)), 1000)

        def body(s, _):
            """One scan step: 10 MD steps, then mode temperatures, spreads and estimators."""
            s = integ._run(s, 10)
            e = integ.estimators(s)
            qn = r.to_nm(s.q)
            return s, (e["t_modes"], jnp.mean(qn * qn, axis=(1, 2)), e["cv"].mean(), e["prim"].mean())

        st, (tm, q2, kcv, kp) = jax.jit(lambda s: jax.lax.scan(body, s, None, length=a.samples))(st)
        tm, q2 = np.asarray(tm), np.asarray(q2)
        spread = q2.mean(0)[1:] / (r.kT_P / (m * r.omega[1:] ** 2))
        key = f"{mode}/{th}" if mode == "pimd" else mode
        out[key] = {
            "T_modes_K": tm.mean(0).tolist(),
            "T_modes_err": [block_err(tm[:, k]) for k in range(P)],
            "spread_ratio_internal": spread.tolist(),
            "K_cv_per_atom_kJmol": float(np.mean(kcv)),
            "K_prim_per_atom_kJmol": float(np.mean(kp)),
            "classical_3kT/2": 1.5 * KB * T,
        }
        print(
            key,
            "T centroid",
            round(float(tm.mean(0)[0]), 2),
            "internal modes",
            np.round(tm.mean(0)[1:], 1),
            "spread ratio",
            np.round(spread, 3),
            "Kcv",
            float(np.mean(kcv)),
            "Kprim",
            float(np.mean(kp)),
            flush=True,
        )
    return out


def nve(a: argparse.Namespace) -> list[dict]:
    """Run the RPMD energy-conservation test; return one row per propagator and time step.

    Each run: 2 ps of PILE-L equilibration, then RPMD (no thermostat) for --time-ps with 500
    samples of H_P; reported as the standard deviation and drift of H_P per degree of freedom in
    units of kT.
    """
    T, P, n, m = a.temperature_K, 32, 64, 1.008
    w0 = 700.0

    def V(x, box):
        """O-H-stretch-like anharmonic well (Morse to fourth order in y = a x, a = 22.87/nm) [kJ/mol]."""
        al = 22.87
        y = al * x
        D = 0.5 * m * w0**2 / al**2
        return jnp.sum(D * (y * y - y**3 + 7.0 / 12.0 * y**4))

    out = []
    for prop in ("cayley", "exact"):
        for dt in a.dts_fs:
            integ = PIMDIntegrator(
                PotentialEngine(V),
                np.full(n, m),
                P,
                T,
                dt * 1e-3,
                "pimd",
                PILE("l", tau_centroid=0.05),
                propagator=prop,
            )
            st = integ.run(integ.init(jnp.zeros((n, 3)), jnp.eye(3), jax.random.PRNGKey(1)), int(2.0 / (dt * 1e-3)))
            integ.set_thermostat("rpmd")
            nsteps = int(round(a.time_ps / (dt * 1e-3)))
            every = max(1, nsteps // 500)
            st, X = sample(integ, st, nsteps // every, every)
            E = X[:, 4]
            t = np.arange(len(E)) * every * dt * 1e-3
            drift = np.polyfit(t, E, 1)[0]
            kT = KB * T
            r = {
                "propagator": prop,
                "dt_fs": dt,
                "ps": a.time_ps,
                "sd_HP_per_dof_kT": float(E.std() / (3 * n * P) / kT),
                "drift_per_dof_kT_per_ns": float(drift * 1000 / (3 * n * P) / kT),
                "sd_HP_kJmol": float(E.std()),
                "mean_K_cv_per_dof": float(X[:, 2].mean() / (3 * n)),
            }
            out.append(r)
            print(json.dumps(r), flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run one validation and write its JSON (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["harmonic", "free", "nve"], help="model system")
    add_temperature_arg(ap, 300.0)
    ap.add_argument("--beads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64], help="harmonic: bead numbers")
    ap.add_argument(
        "--omega-per-ps", type=float, nargs="+", default=[100.0, 700.0], help="harmonic: angular frequencies [rad/ps]"
    )
    ap.add_argument("--thermostat", default="pile-l", choices=["pile-l", "pile-g"], help="harmonic: PILE kind")
    ap.add_argument("--samples", type=int, default=4000, help="harmonic, free: number of samples")
    ap.add_argument("--dt-factor", type=float, default=0.1, help="harmonic: dt = factor / omega")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    ap.add_argument("--dts-fs", type=float, nargs="+", default=[0.4, 0.2, 0.1], help="nve: time steps [fs]")
    ap.add_argument("--time-ps", type=float, default=10.0, help="nve: RPMD run length [ps]")
    ap.add_argument("-o", "--out", default=OUT, help="output directory")
    a = ap.parse_args(argv)
    res = {"harmonic": harmonic, "free": free, "nve": nve}[a.what](a)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.what}{a.tag}.json"), "w") as fh:
        json.dump({"args": vars(a), "results": res}, fh, indent=1)


if __name__ == "__main__":
    main()
