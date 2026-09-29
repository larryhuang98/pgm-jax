"""Independent-code check of the ring-polymer integrator and of ring-polymer contraction with
OpenMM's RPMDIntegrator (PILE-L, exact free ring polymer, Fourier contraction of force groups).

Model: 64 particles (1.008 amu) at 300 K, each in a stiff anharmonic well along x (the q-TIP4P/F
O-H Morse expansion, omega ~ 700 rad/ps, beta hbar omega ~ 18) plus a soft anharmonic potential
(omega_s = 60 rad/ps with cubic and quartic terms along y) in force group 1, which may be contracted to P' beads.
Observables from the bead positions only (same definition in both codes): <V_stiff>, <V_soft>
(bead averages), <x_c^2> (centroid) and the ring-polymer spread <|q_k - q_c|^2>.

    ~/miniconda3/envs/colabfold/bin/python scripts/pimd_openmm.py openmm      # OpenMM side -> JSON
    python scripts/pimd_openmm.py pgmjax                                       # pgm_jax side, comparison
"""

import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "validation/pimd")
T, N, M = 300.0, 64, 1.008
A, W0, WS, B3, B4 = 22.87, 700.0, 60.0, 5.0, 100.0
D = 0.5 * M * W0**2 / A**2  # kJ/mol: harmonic force constant M W0^2
KS = M * WS**2
CASES = [(8, None), (8, 1), (8, 3), (32, None), (32, 1), (32, 5)]
DT, NEQ, NSAMP, EVERY = 0.00005, 20000, 3000, 50
if os.environ.get("PIMD_DT_FS"):  # time-step check: only the P = 32 full case
    f = 0.05 / float(os.environ["PIMD_DT_FS"])
    DT, NEQ, EVERY, CASES = DT / f, int(NEQ * f), int(round(EVERY * f)), [(32, None)]
TAG = os.environ.get("PIMD_TAG", "")


def v_stiff_np(x):
    y = A * x[..., 0]
    return D * (y * y - y**3 + 7.0 / 12.0 * y**4)


def v_soft_np(x):
    r2 = np.sum(x * x, -1)
    return KS * (0.5 * r2 + B3 * x[..., 1] ** 3 + B4 * x[..., 1] ** 4)


def observables(Q):
    """Q (S, P, N, 3) bead positions -> dict of means and block errors."""
    qc = Q.mean(1)
    obs = {
        "V_stiff": v_stiff_np(Q).sum(-1).mean(1) / N,
        "V_soft": v_soft_np(Q).sum(-1).mean(1) / N,
        "xc2": np.sum(qc * qc, -1).mean(-1),
        "spread": np.sum((Q - qc[:, None]) ** 2, -1).mean((1, 2)),
    }
    out = {}
    for k, v in obs.items():
        nb = 20
        m = len(v) // nb * nb
        b = v[len(v) - m :].reshape(nb, -1).mean(1)
        out[k] = [float(v.mean()), float(b.std(ddof=1) / math.sqrt(nb))]
    return out


def run_openmm():
    import openmm as mm
    import openmm.unit as u

    res = {}
    for P, Pc in CASES:
        system = mm.System()
        for _ in range(N):
            system.addParticle(M)
        f1 = mm.CustomExternalForce(f"{D}*((({A})*x)^2 - (({A})*x)^3 + 7/12*(({A})*x)^4)")
        f2 = mm.CustomExternalForce(f"{KS}*(0.5*(x^2+y^2+z^2) + {B3}*y^3 + {B4}*y^4)")
        for i in range(N):
            f1.addParticle(i, [])
            f2.addParticle(i, [])
        f1.setForceGroup(0)
        f2.setForceGroup(1)
        system.addForce(f1)
        system.addForce(f2)
        integ = mm.RPMDIntegrator(
            P, T * u.kelvin, 20.0 / u.picosecond, DT * u.picoseconds, {} if Pc is None else {1: Pc}
        )
        integ.setRandomNumberSeed(7)
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CPU"))
        ctx.setPositions(np.zeros((N, 3)))
        for k in range(P):
            integ.setPositions(k, np.zeros((N, 3)) * u.nanometer)
        ctx.setVelocitiesToTemperature(T * u.kelvin)
        t0 = time.time()
        integ.step(NEQ)
        Q = np.zeros((NSAMP, P, N, 3))
        for s in range(NSAMP):
            integ.step(EVERY)
            for k in range(P):
                Q[s, k] = integ.getState(k, getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
        key = f"P{P}_c{Pc or P}"
        res[key] = observables(Q)
        res[key]["time_s"] = time.time() - t0
        print(key, json.dumps(res[key]), flush=True)
    return res


def run_pgmjax():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, ROOT)
    from pgm_jax.md.pimd import PIMDIntegrator, PotentialEngine

    def vs(x, box):
        y = A * x[:, 0]
        return jnp.sum(D * (y * y - y**3 + 7.0 / 12.0 * y**4))

    def vw(x, box):
        return jnp.sum(KS * (0.5 * jnp.sum(x * x, -1) + B3 * x[:, 1] ** 3 + B4 * x[:, 1] ** 4))

    res = {}
    for P, Pc in CASES:
        eng = PotentialEngine(vs, soft=vw, contract=Pc)
        integ = PIMDIntegrator(eng, np.full(N, M), P, T, DT, "pimd", "pile-l", tau0=1.0 / 20.0, propagator="exact")
        t0 = time.time()
        st = integ.run(integ.init(jnp.zeros((N, 3)), jnp.eye(3), jax.random.PRNGKey(P + (Pc or 0))), NEQ)

        def body(s, _):
            s = integ._run(s, EVERY)
            return s, s.q

        st, Q = jax.jit(lambda s: jax.lax.scan(body, s, None, length=NSAMP))(st)
        key = f"P{P}_c{Pc or P}"
        res[key] = observables(np.asarray(Q))
        res[key]["time_s"] = time.time() - t0
        print(key, json.dumps(res[key]), flush=True)
    return res


if __name__ == "__main__":
    which = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    res = run_openmm() if which == "openmm" else run_pgmjax()
    with open(os.path.join(OUT, f"anharmonic_{which}{TAG}.json"), "w") as fh:
        json.dump({"T": T, "N": N, "mass": M, "dt_ps": DT, "cases": res}, fh, indent=1)
    other = os.path.join(OUT, f"anharmonic_{'openmm' if which != 'openmm' else 'pgmjax'}{TAG}.json")
    if os.path.exists(other):
        o = json.load(open(other))["cases"]
        print("comparison (value +- error, pgm_jax - OpenMM in units of the combined error):")
        for key in res:
            if key in o:
                a, b = (res, o) if which != "openmm" else (o, res)
                line = " ".join(
                    f"{k} {a[key][k][0]:.5g}/{b[key][k][0]:.5g} "
                    f"({(a[key][k][0] - b[key][k][0]) / math.hypot(a[key][k][1], b[key][k][1]):+.1f} sd)"
                    for k in ("V_stiff", "V_soft", "xc2", "spread")
                )
                print(key, line)
