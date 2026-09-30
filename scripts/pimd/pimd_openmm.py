"""Independent-code check of the ring-polymer integrator and of contraction with OpenMM's RPMDIntegrator.

The same model is run with OpenMM's RPMDIntegrator (PILE-L, exact free ring polymer, Fourier
contraction of force groups) and with pgm_jax's PIMDIntegrator (PILE-L with tau_centroid = 1/20
ps, exact propagator, PotentialEngine with a contracted soft part); each side writes a JSON file,
and the side run second prints the comparison in units of the combined statistical error.

Model: 64 particles (1.008 amu) at 300 K, each in a stiff anharmonic well along x (the q-TIP4P/F
O-H Morse expansion, omega ~ 700 rad/ps, beta hbar omega ~ 18) plus a soft anharmonic potential
(omega_s = 60 rad/ps with cubic and quartic terms along y) in force group 1, which may be contracted to P' beads.
Observables from the bead positions only (same definition in both codes): <V_stiff>, <V_soft>
(bead averages), <x_c^2> (centroid) and the ring-polymer spread <|q_k - q_c|^2>.  Cases (P, P'):
(8, full), (8, 1), (8, 3), (32, full), (32, 1), (32, 5); time step 0.05 fs, 20,000 equilibration
steps, 3,000 samples every 50 steps.  --dt-fs runs only the full P = 32 case at another time step
(the equilibration and sampling times are kept).

Usage:

    <python with OpenMM> scripts/pimd/pimd_openmm.py openmm        # OpenMM side -> JSON
    python scripts/pimd/pimd_openmm.py pgmjax                      # pgm_jax side, and the comparison
    python scripts/pimd/pimd_openmm.py pgmjax --dt-fs 0.1 --tag _dt0.1
    python scripts/pimd/pimd_openmm.py --help

Inputs: none (the other side's JSON for the comparison).
Outputs: <out>/anharmonic_{openmm,pgmjax}<tag>.json (default --out data/validation/pimd), printed
per-case results and the comparison.
Units: nm, ps, amu, K, kJ/mol (both codes); --dt-fs fs.
Runtime: CPU, minutes per case (the OpenMM side is the slower one).  The pgm_jax side sets
jax_enable_x64.  The OpenMM side needs an environment with OpenMM (and pgm_jax importable).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from pgm_jax.paths import repo_path

OUT = repo_path("data", "validation", "pimd")
T, N, M = 300.0, 64, 1.008
A, W0, WS, B3, B4 = 22.87, 700.0, 60.0, 5.0, 100.0
D = 0.5 * M * W0**2 / A**2  # kJ/mol: harmonic force constant M W0^2
KS = M * WS**2
CASES = [(8, None), (8, 1), (8, 3), (32, None), (32, 1), (32, 5)]  # (beads P, contracted P'; None: full)
DT, NEQ, NSAMP, EVERY = 0.00005, 20000, 3000, 50  # ps, steps, samples, steps


def run_settings(dt_fs: float | None) -> tuple[float, int, int, list]:
    """Return (dt [ps], equilibration steps, steps between samples, cases) for an optional --dt-fs.

    Without --dt-fs: the default settings and all cases.  With it: only the full P = 32 case, with
    the step counts scaled so that the equilibration and sampling times stay the same.
    """
    if dt_fs is None:
        return DT, NEQ, EVERY, CASES
    f = 0.05 / float(dt_fs)
    return DT / f, int(NEQ * f), int(round(EVERY * f)), [(32, None)]


def v_stiff_np(x: np.ndarray) -> np.ndarray:
    """Stiff well along x, D (y^2 - y^3 + 7/12 y^4) with y = A x [kJ/mol] (per particle, last axis xyz)."""
    y = A * x[..., 0]
    return D * (y * y - y**3 + 7.0 / 12.0 * y**4)


def v_soft_np(x: np.ndarray) -> np.ndarray:
    """Soft anharmonic potential KS (r^2 / 2 + B3 y^3 + B4 y^4) [kJ/mol] (per particle)."""
    r2 = np.sum(x * x, -1)
    return KS * (0.5 * r2 + B3 * x[..., 1] ** 3 + B4 * x[..., 1] ** 4)


def observables(Q: np.ndarray) -> dict[str, list[float]]:
    """Return the observables' means and block errors (20 blocks) of bead positions Q (S, P, N, 3) [nm].

    Keys: V_stiff and V_soft (bead-averaged, per particle) [kJ/mol], xc2 (centroid <x_c^2>) and
    spread (<|q_k - q_c|^2>) [nm^2]; each value is [mean, standard error].
    """
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


def run_openmm(dt: float, neq: int, every: int, cases: list) -> dict:
    """Run the cases with OpenMM's RPMDIntegrator (CPU platform, friction 20/ps, seed 7).

    Parameters
    ----------
    dt : float
        Time step [ps].
    neq : int
        Equilibration steps.
    every : int
        Steps between the NSAMP samples.
    cases : list of (int, int or None)
        (beads, contracted beads of force group 1 or None).

    Returns
    -------
    dict
        Per case "P<P>_c<P'>": the observables and time_s.
    """
    import openmm as mm
    import openmm.unit as u

    res = {}
    for P, Pc in cases:
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
            P, T * u.kelvin, 20.0 / u.picosecond, dt * u.picoseconds, {} if Pc is None else {1: Pc}
        )
        integ.setRandomNumberSeed(7)
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CPU"))
        ctx.setPositions(np.zeros((N, 3)))
        for k in range(P):
            integ.setPositions(k, np.zeros((N, 3)) * u.nanometer)
        ctx.setVelocitiesToTemperature(T * u.kelvin)
        t0 = time.time()
        integ.step(neq)
        Q = np.zeros((NSAMP, P, N, 3))
        for s in range(NSAMP):
            integ.step(every)
            for k in range(P):
                Q[s, k] = integ.getState(k, getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
        key = f"P{P}_c{Pc or P}"
        res[key] = observables(Q)
        res[key]["time_s"] = time.time() - t0
        print(key, json.dumps(res[key]), flush=True)
    return res


def run_pgmjax(dt: float, neq: int, every: int, cases: list) -> dict:
    """Run the cases with pgm_jax's PIMDIntegrator (PILE-L, exact propagator); see run_openmm."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from pgm_jax.md.pimd import PILE, PIMDIntegrator, PotentialEngine

    def vs(x, box):
        """Stiff well [kJ/mol] summed over the particles (force group 0)."""
        y = A * x[:, 0]
        return jnp.sum(D * (y * y - y**3 + 7.0 / 12.0 * y**4))

    def vw(x, box):
        """Soft potential [kJ/mol] summed over the particles (contracted part)."""
        return jnp.sum(KS * (0.5 * jnp.sum(x * x, -1) + B3 * x[:, 1] ** 3 + B4 * x[:, 1] ** 4))

    res = {}
    for P, Pc in cases:
        eng = PotentialEngine(vs, soft=vw, contract=Pc)
        integ = PIMDIntegrator(
            eng, np.full(N, M), P, T, dt, "pimd", PILE("l", tau_centroid=1.0 / 20.0), propagator="exact"
        )
        t0 = time.time()
        st = integ.run(integ.init(jnp.zeros((N, 3)), jnp.eye(3), jax.random.PRNGKey(P + (Pc or 0))), neq)

        def body(s, _):
            """One scan step: `every` MD steps; the bead positions are the sample."""
            s = integ._run(s, every)
            return s, s.q

        st, Q = jax.jit(lambda s: jax.lax.scan(body, s, None, length=NSAMP))(st)
        key = f"P{P}_c{Pc or P}"
        res[key] = observables(np.asarray(Q))
        res[key]["time_s"] = time.time() - t0
        print(key, json.dumps(res[key]), flush=True)
    return res


def compare(res: dict, which: str, other_path: str) -> None:
    """Print this side's results against the other side's JSON (in units of the combined error)."""
    with open(other_path) as fh:
        o = json.load(fh)["cases"]
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


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run one side, write its JSON and compare (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["openmm", "pgmjax"], help="which code to run")
    ap.add_argument("--dt-fs", type=float, default=None, help="time-step check: only P = 32 (full) at this step [fs]")
    ap.add_argument("--tag", default="", help="suffix of the output file names")
    ap.add_argument("-o", "--out", default=OUT, help="output directory")
    a = ap.parse_args(argv)
    dt, neq, every, cases = run_settings(a.dt_fs)
    os.makedirs(a.out, exist_ok=True)
    res = (run_openmm if a.which == "openmm" else run_pgmjax)(dt, neq, every, cases)
    with open(os.path.join(a.out, f"anharmonic_{a.which}{a.tag}.json"), "w") as fh:
        json.dump({"T": T, "N": N, "mass": M, "dt_ps": dt, "cases": res}, fh, indent=1)
    other = os.path.join(a.out, f"anharmonic_{'openmm' if a.which != 'openmm' else 'pgmjax'}{a.tag}.json")
    if os.path.exists(other):
        compare(res, a.which, other)


if __name__ == "__main__":
    main()
