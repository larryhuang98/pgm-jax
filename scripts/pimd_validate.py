"""Model-system validation of the path-integral integrator (md/pimd.py), CPU is enough.

    python scripts/pimd_validate.py harmonic      # 3D harmonic oscillators: <V>, <K> (primitive, centroid
                                                  # virial) vs the exact P-bead and quantum values, P = 1..64
    python scripts/pimd_validate.py free          # free particles: centroid and normal-mode temperatures,
                                                  # mode spreads (PILE-L, PILE-G, TRPMD)
    python scripts/pimd_validate.py nve           # RPMD (no thermostat): conservation of H_P vs dt
Writes JSON to validation/pimd/."""

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
from pgm_jax.md.pimd import PIMDIntegrator, PotentialEngine, RingPolymer
from pgm_jax.units import HBAR_KJMOL_PS, KB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jax.config.update("jax_enable_x64", True)
OUT = os.path.join(ROOT, "validation/pimd")


def block_err(x, nb=20):
    """Standard error of the mean of x from nb contiguous blocks."""
    return block_mean(x, nb)[1]


def sample(integ, st, nsamp, every):
    def body(s, _):
        s = integ._run(s, every)
        e = integ.estimators(s)
        return s, jnp.stack([e["epot"], e["prim"].sum(), e["cv"].sum(), e["t_centroid"], e["econs"]])

    st, X = jax.jit(lambda s: jax.lax.scan(body, s, None, length=nsamp))(st)
    return st, np.asarray(X)


def harmonic(a):
    T = a.temp
    beta = 1.0 / (KB * T)
    n, m = 256, 1.008
    rows = []
    for w in a.omega:
        # quantum <K> = <V> per degree of freedom
        q = 0.25 * HBAR_KJMOL_PS * w / math.tanh(0.5 * beta * HBAR_KJMOL_PS * w)
        dt = a.dt_factor / w
        for P in a.beads:
            r = RingPolymer(P, T)
            ex = 0.5 / beta * float(np.sum(w**2 / (r.omega**2 + w**2)))
            eng = PotentialEngine(lambda x, box, w=w: 0.5 * m * w**2 * jnp.sum(x * x))
            integ = PIMDIntegrator(eng, np.full(n, m), P, T, dt, "pimd", a.thermostat, tau0=1.0 / w)
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


def free(a):
    T, P, n, m = a.temp, 16, 512, 1.008
    out = {}
    for mode, th in (("pimd", "pile-l"), ("pimd", "pile-g"), ("trpmd", "pile-l")):
        eng = PotentialEngine(lambda x, box: 0.0 * jnp.sum(x))
        integ = PIMDIntegrator(eng, np.full(n, m), P, T, 0.0005, mode, th, tau0=0.05)
        r = integ.ring
        st = integ.run(integ.init(jnp.zeros((n, 3)), jnp.eye(3), jax.random.PRNGKey(0)), 1000)

        def body(s, _):
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


def nve(a):
    T, P, n, m = a.temp, 32, 64, 1.008
    w0 = 700.0

    def V(x, box):  # O-H-stretch-like anharmonic well (Morse to fourth order)
        al = 22.87
        y = al * x
        D = 0.5 * m * w0**2 / al**2
        return jnp.sum(D * (y * y - y**3 + 7.0 / 12.0 * y**4))

    out = []
    for prop in ("cayley", "exact"):
        for dt in a.dts:
            integ = PIMDIntegrator(
                PotentialEngine(V), np.full(n, m), P, T, dt * 1e-3, "pimd", "pile-l", tau0=0.05, propagator=prop
            )
            st = integ.run(integ.init(jnp.zeros((n, 3)), jnp.eye(3), jax.random.PRNGKey(1)), int(2.0 / (dt * 1e-3)))
            integ.set_thermostat("rpmd")
            nsteps = int(round(a.ps / (dt * 1e-3)))
            every = max(1, nsteps // 500)
            st, X = sample(integ, st, nsteps // every, every)
            E = X[:, 4]
            t = np.arange(len(E)) * every * dt * 1e-3
            drift = np.polyfit(t, E, 1)[0]
            kT = KB * T
            r = {
                "propagator": prop,
                "dt_fs": dt,
                "ps": a.ps,
                "sd_HP_per_dof_kT": float(E.std() / (3 * n * P) / kT),
                "drift_per_dof_kT_per_ns": float(drift * 1000 / (3 * n * P) / kT),
                "sd_HP_kJmol": float(E.std()),
                "mean_K_cv_per_dof": float(X[:, 2].mean() / (3 * n)),
            }
            out.append(r)
            print(json.dumps(r), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["harmonic", "free", "nve"])
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--beads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--omega", type=float, nargs="+", default=[100.0, 700.0])
    ap.add_argument("--thermostat", default="pile-l")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--dt-factor", type=float, default=0.1, help="harmonic: dt = factor / omega")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dts", type=float, nargs="+", default=[0.4, 0.2, 0.1])
    ap.add_argument("--ps", type=float, default=10.0)
    a = ap.parse_args()
    res = {"harmonic": harmonic, "free": free, "nve": nve}[a.what](a)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{a.what}{a.tag}.json"), "w") as fh:
        json.dump({"args": vars(a), "results": res}, fh, indent=1)


if __name__ == "__main__":
    main()
