"""Flexible methanol for the paper: forces, NPT density, temperature partition, NVE conservation, speed.

1. One molecule in a 4 nm box against the gas-phase model the bonded terms were fitted with.
2. 216 molecules: 2 ps Langevin NVT (friction 5/ps), then 100 ps Monte Carlo NPT (Langevin 1/ps,
   a volume move every 25 steps), sampled every 0.5 ps; density and centre-of-mass / internal
   temperatures of the last 50 ps; speed.
3. NVE from the equilibrated state at two time steps (0.5 and 0.25 fs) and two precisions:
   total-energy drift [kT/ns per degree of freedom] and fluctuation [kT].

Usage:

    python paper/scripts/flex_methanol.py            # -> paper/data/flex_methanol.json
    python paper/scripts/flex_methanol.py --help

Inputs: runs/flex/methanol.flex (examples/flex_methanol_check.py or examples/fit_bonded_template.py).
Outputs: paper/data/flex_methanol.json (or --out); progress on stdout.
Units: kJ/mol/nm (forces), kJ/mol (energies), g/cm^3, K, ps.
Runtime: GPU, minutes.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.thermostats import Langevin
from pgm_jax.paths import repo_path
from pgm_jax.system import System
from pgm_jax.units import KB

jax.config.update("jax_enable_x64", True)
N, T = 216, 298.0  # molecules, temperature [K]


def single_molecule(tpl: FlexibleTemplate) -> dict:
    """Return the largest MD - gas-phase force difference and the RMS force of one perturbed molecule [kJ/mol/nm]."""
    rng = np.random.default_rng(0)
    x = np.asarray(tpl.spec.ref_xyz) + 0.003 * rng.normal(size=(tpl.n, 3))
    s1 = FlexibleSimulation(
        System([tpl.pgm]),
        [tpl],
        x + 2.0,
        np.eye(3) * 4.0,
        MDSettings().replace(precision="double", dipole_tol=1e-9, cutoff=1.8, skin=0.05, lj_lrc=False),
        thermostat=None,
        log=None,
    )
    P = jax.tree_util.tree_map(jnp.asarray, tpl.P)
    g = np.asarray(jax.grad(lambda R: tpl.model.energy(tpl.index, R, P)[0])(jnp.asarray(x)))
    F = np.asarray(s1.state.dyn.force)
    return {
        "max_abs_diff": float(np.abs(F + g).max()),
        "rms_force": float(np.sqrt(np.mean(g**2))),
        "units": "kJ/mol/nm",
    }


def npt_run(tpl: FlexibleTemplate, out: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run NVT and NPT, store the series and summary in out; return positions, velocities and box at the end."""
    st = MDSettings()  # 0.9 nm, PME, tol 1e-5, mixed, LJ tail
    pos, H = liquid_box(tpl, N, 0.55, seed=1, min_dist=0.18)
    sys_ = System([tpl.pgm] * N)
    dt = 0.0005
    nvt = FlexibleSimulation(sys_, [tpl] * N, pos, H, st, dt=dt, thermostat=Langevin(5.0), temperature=T, log=None)
    nvt.advance(4000)
    npt = FlexibleSimulation(
        sys_,
        [tpl] * N,
        nvt.positions(),
        np.asarray(nvt.state.box),
        st,
        dt=dt,
        thermostat=Langevin(1.0),
        barostat=MonteCarloBarostat(every=25),
        temperature=T,
        velocities=nvt.velocities(),
        log=None,
    )
    rec = {k: [] for k in ("time_ps", "density", "temp_com", "temp_internal", "epot")}
    npt.advance(1000)  # compile outside the timing
    t0, s0 = time.time(), int(npt.state.step)
    for _ in range(200):  # 200 x 0.5 ps
        npt.advance(1000)
        o = npt.observables()
        rec["time_ps"].append(round(o["time_ps"], 3))
        rec["density"].append(o["density_g_cm3"])
        rec["temp_com"].append(o["temp_com"])
        rec["temp_internal"].append(o["temp_internal"])
        rec["epot"].append(o["epot"])
    wall = time.time() - t0
    steps = int(npt.state.step) - s0
    out["npt"] = rec
    half = np.array(rec["density"][100:])
    blocks = np.array([b.mean() for b in np.array_split(half, 5)])
    out["npt_summary"] = {
        "density_mean_last50ps": float(half.mean()),
        "density_se": float(blocks.std(ddof=1) / np.sqrt(5)),
        "temp_com_mean": float(np.mean(rec["temp_com"][100:])),
        "temp_internal_mean": float(np.mean(rec["temp_internal"][100:])),
        "ns_per_day": steps * dt / 1000.0 / (wall / 86400.0),
        "ms_per_step": 1000.0 * wall / steps,
        "barostat_interval": 25,
        "dt_fs": 0.5,
    }
    print(out["npt_summary"], flush=True)
    return npt.positions(), npt.velocities(), np.asarray(npt.state.box)


def nve_runs(tpl: FlexibleTemplate, x_eq: np.ndarray, v_eq: np.ndarray, H_eq: np.ndarray) -> list[dict]:
    """Return the NVE records (energy series, drift, fluctuation) of the three time step / precision variants."""
    sys_ = System([tpl.pgm] * N)
    res = []
    for label, dt_n, prec, tol, ps in (
        ("0.5 fs, mixed, tol 1e-5", 0.0005, "mixed", 1e-5, 20.0),
        ("0.25 fs, mixed, tol 1e-5", 0.00025, "mixed", 1e-5, 10.0),
        ("0.5 fs, double, tol 1e-8", 0.0005, "double", 1e-8, 10.0),
    ):
        s = FlexibleSimulation(
            sys_,
            [tpl] * N,
            x_eq,
            H_eq,
            MDSettings().replace(precision=prec, dipole_tol=tol),
            dt=dt_n,
            thermostat=None,
            velocities=v_eq,
            log=None,
        )
        every = int(round(0.1 / dt_n))
        t, E = [], []
        for _ in range(int(round(ps / 0.1))):
            s.advance(every)
            o = s.observables()
            t.append(round(o["time_ps"], 4))
            E.append(o["etot"])
        t, E = np.array(t), np.array(E)
        dof = s.integ.dof
        slope = np.polyfit(t, E, 1)[0]  # kJ/mol/ps
        r = {
            "label": label,
            "time_ps": t.tolist(),
            "etot": E.tolist(),
            "dof": dof,
            "drift_kT_per_ns_per_dof": float(slope * 1000.0 / (KB * T) / dof),
            "rms_fluct_kT": float(np.std(E - np.polyval(np.polyfit(t, E, 1), t)) / (KB * T)),
        }
        res.append(r)
        print(label, {k: v for k, v in r.items() if k not in ("time_ps", "etot")}, flush=True)
    return res


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run the checks and write the JSON (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=repo_path("runs", "flex", "methanol.flex"), help="methanol template")
    ap.add_argument("-o", "--out", default=repo_path("paper", "data", "flex_methanol.json"), help="output JSON")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tpl = FlexibleTemplate.load(a.template)
    out = {
        "n_mol": N,
        "n_atoms": N * tpl.n,
        "T": T,
        "exp_density": 0.7866,
        "families": list(tpl.settings["families"]),
        "device": str(jax.devices()[0]),
    }
    out["single_molecule"] = single_molecule(tpl)
    print(out["single_molecule"], flush=True)
    x_eq, v_eq, H_eq = npt_run(tpl, out)
    out["nve"] = nve_runs(tpl, x_eq, v_eq, H_eq)
    out["kT_total"] = KB * T
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
