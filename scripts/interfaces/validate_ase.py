"""Validate ASE + pgm_jax (pgm_jax.interfaces.ase) on the 512-water pGM box of the README (docs/interfaces.md).

1. single points against the native engine (Simulation.from_amber): energy, atomic forces, centre-of-
   mass forces of the rigid bodies, pressure (molecular virial), double and mixed precision;
2. NVE: ASE VelocityVerlet + FixRigidMolecules (SHAKE / RATTLE) vs the native rigid-body NVE, same
   start (after a native NVT equilibration), same dt: drift and fluctuation of E_tot;
3. NVT: ASE Langevin vs native Langevin (gamma 1/ps): temperature and <U>;
4. cost per step: native, ASE (engine call, constraints, ASE bookkeeping).

Usage:

    python scripts/interfaces/validate_ase.py --out data/validation/interfaces/ase.json [--nve-ps 10 --nvt-ps 20]
    python scripts/interfaces/validate_ase.py --help

Inputs: PGM_GVDW_DATA (pgm_jax.paths); ASE installed.
Outputs: the JSON (--out, default data/validation/interfaces/ase.json); printed results.
Units: --dt-fs fs, durations in ps, --temperature-K K; energies kJ/mol, forces kJ/mol/nm, pressure bar.
Runtime: GPU or CPU, minutes (ASE's Python loop dominates the cost per step).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import numpy as np

try:  # ASE is optional: without it only --help works
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.verlet import VelocityVerlet

    from pgm_jax.interfaces.ase import PGMCalculator, atoms_from_system, rigid_constraints
except ImportError:  # pragma: no cover
    units = Langevin = VelocityVerlet = None

from pgm_jax.cli.args import add_dt_arg, add_precision_arg, add_temperature_arg, setup_logging
from pgm_jax.interfaces import PGMEngine
from pgm_jax.md.box import volume
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import Langevin as LangevinThermostat
from pgm_jax.paths import pgm3p25_files, repo_path
from pgm_jax.units import BAR_PER_KJMOL_NM3, KB

jax.config.update("jax_enable_x64", True)
TOP, RST = pgm3p25_files()
KJ = units.kJ / units.mol if units is not None else None  # ASE energy unit of 1 kJ/mol


def native_atomic(sim: Simulation) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return positions [nm], forces [kJ/mol/nm], energy [kJ/mol] and induced dipoles [e nm] of the native state."""
    st = sim.state
    pos = sim.rigid.positions(st.dyn.position)
    idx = sim.nb.candidates(st.nbr, st.dyn.position.center, st.box, pos)[0]
    res = jax.jit(sim.ff.compute)(pos, st.box, idx, sim.ff.init_induction())
    return np.asarray(pos), np.asarray(res.forces), float(res.energy["total"]), np.asarray(res.induction.mu)


def single_point(prec: str, tol: float) -> dict:
    """Compare energy, forces, dipoles and pressure of ASE's calculator with the native engine at one precision."""
    s = MDSettings().replace(precision=prec, dipole_tol=tol)
    sim = Simulation.from_amber(TOP, RST, settings=s, thermostat=None, log=None)
    pos, F_nat, E_nat, mu_nat = native_atomic(sim)
    H = np.asarray(sim.state.box)
    out = {"precision": prec, "dipole_tol": tol, "E_native": float(sim.state.epot)}
    for mode in ("molecular", "atomic"):
        eng = PGMEngine.from_simulation(sim, stress=mode)
        atoms = atoms_from_system(sim.sys, pos, H)
        atoms.calc = PGMCalculator(eng)
        E = atoms.get_potential_energy() / KJ
        F = atoms.get_forces() / (KJ / 10.0)
        stress = atoms.get_stress(voigt=False) / (KJ / 1000.0)  # kJ/mol/nm^3
        mu = atoms.calc.get_induced_dipoles(atoms) / 10.0
        if mode == "molecular":
            ke_t = float(sim.integ.kinetic(sim.state)[1])
            V = float(volume(sim.state.box))
            P_ase = (2.0 * ke_t - V * np.trace(stress)) / (3.0 * V) * BAR_PER_KJMOL_NM3
            P_nat = sim.pressure()
            Fcom = np.zeros((sim.sys.nmol, 3))
            np.add.at(Fcom, sim.sys.mol, F)
            Fc_nat = np.asarray(sim.state.dyn.force.center)
            out.update(
                E_ase=E,
                dE=E - out["E_native"],
                dE_rel=abs(E - out["E_native"]) / abs(out["E_native"]),
                F_maxdiff=float(np.abs(F - F_nat).max()),
                F_rms=float(np.sqrt(np.mean(F_nat**2))),
                Fcom_maxdiff=float(np.abs(Fcom - Fc_nat).max()),
                mu_maxdiff=float(np.abs(mu - mu_nat).max()),
                P_native_bar=P_nat,
                P_ase_bar=float(P_ase),
                dP_bar=float(P_ase - P_nat),
            )
        else:
            out["stress_atomic_kjmol_nm3"] = stress.tolist()
            out["P_atomic_virial_bar"] = float(-np.trace(stress) / 3.0 * BAR_PER_KJMOL_NM3)
    return out


def drift(t_ps: list, e: list, dof: int, T: float) -> dict:
    """Return the drift [kT/ns/dof] and fluctuation of an energy series e [kJ/mol] at times t_ps [ps]."""
    slope = np.polyfit(np.asarray(t_ps) / 1000.0, np.asarray(e), 1)[0]  # kJ/mol/ns
    return {
        "drift_kT_per_ns_per_dof": float(slope / (KB * T) / dof),
        "std_kJmol": float(np.std(e)),
        "std_per_dof_kT": float(np.std(e) / (KB * T) / np.sqrt(dof)),
    }


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run the four checks and write the JSON (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "-o", "--out", default=repo_path("data", "validation", "interfaces", "ase.json"), help="JSON output"
    )
    ap.add_argument("--equil-ps", type=float, default=5.0, help="native NVT equilibration [ps]")
    ap.add_argument("--nve-ps", type=float, default=10.0, help="NVE comparison [ps]")
    ap.add_argument("--nvt-ps", type=float, default=20.0, help="NVT comparison [ps]")
    add_dt_arg(ap, 1.0)
    add_precision_arg(ap)
    add_temperature_arg(ap, 298.0)
    args = ap.parse_args(argv)
    if units is None:
        raise SystemExit("validate_ase.py needs ASE (pip install ase)")
    setup_logging()
    res = {"device": str(jax.devices()[0])}
    res["single_point_double"] = single_point("double", 1e-10)
    print(json.dumps(res, indent=1), flush=True)
    res["single_point_mixed"] = single_point("mixed", 1e-5)
    print(json.dumps(res["single_point_mixed"], indent=1), flush=True)

    s = MDSettings(precision=args.precision)
    dt, T = args.dt_fs / 1000, args.temperature_K
    n_eq, n_nve, n_nvt = (int(round(x / dt)) for x in (args.equil_ps, args.nve_ps, args.nvt_ps))
    rep = 100
    sim = Simulation.from_amber(
        TOP, RST, settings=s, thermostat=LangevinThermostat(1.0), temperature=T, dt=dt, log=None
    )
    sim.advance(n_eq)
    pos0, vel0, H = sim.positions(), sim.velocities(), np.asarray(sim.state.box)
    sysm = sim.sys

    # ---- native NVE
    nat = Simulation(sysm, pos0, H, s, dt=dt, thermostat=None, velocities=vel0, log=None)
    dof = nat.integ.dof
    etot_start = float(nat.observables()["etot"])
    nat.advance(rep)  # compile
    t, E, U = [], [], []
    t0 = time.perf_counter()
    for _k in range(n_nve // rep):
        nat.advance(rep)
        o = nat.observables()
        t.append(o["time_ps"])
        E.append(o["etot"])
        U.append(o["epot"])
    t_nat = (time.perf_counter() - t0) / (n_nve - n_nve % rep)
    res["nve_native"] = dict(drift(t, E, dof, T), ms_per_step=1e3 * t_nat, dof=dof)

    # ---- ASE NVE
    eng = PGMEngine(sysm, pos0, H, s)
    atoms = atoms_from_system(sysm, pos0, H)
    atoms.set_constraint(rigid_constraints(sysm))
    atoms.calc = PGMCalculator(eng)
    atoms.set_velocities(vel0 * 0.01 / units.fs)
    e0 = (atoms.get_potential_energy() + atoms.get_kinetic_energy()) / KJ
    dyn = VelocityVerlet(atoms, dt * 1000.0 * units.fs)
    dyn.run(rep)  # compile / warm up
    rec = {"t": [], "E": [], "T": []}
    ncons = sum(len(b) for b in rigid_constraints(sysm).blocks)
    dof_ase = 3 * len(atoms) - ncons - 3

    def obs():
        ke = atoms.get_kinetic_energy() / KJ
        rec["t"].append(dyn.get_time() / (1000.0 * units.fs))
        rec["E"].append(ke + atoms.get_potential_energy() / KJ)
        rec["T"].append(2.0 * ke / (dof_ase * KB))

    dyn.attach(obs, interval=rep)
    eng.stats.update(calls=0, time=0.0, cg=0)
    t0 = time.perf_counter()
    dyn.run(n_nve)
    t_ase = (time.perf_counter() - t0) / n_nve
    st = eng.stats
    res["nve_ase"] = dict(
        drift(rec["t"], rec["E"], dof_ase, T),
        ms_per_step=1e3 * t_ase,
        engine_ms_per_call=1e3 * st["time"] / max(st["calls"], 1),
        cg_per_call=st["cg"] / max(st["calls"], 1),
        dof=dof_ase,
        T_mean=float(np.mean(rec["T"])),
        E0=e0,
        E0_native=etot_start,
        repeats=st["repeats"],
        rebuilds=st["rebuilds"],
    )
    print(json.dumps({k: res[k] for k in ("nve_native", "nve_ase")}, indent=1), flush=True)

    # ---- NVT: native Langevin vs ASE Langevin
    natv = Simulation(
        sysm, pos0, H, s, dt=dt, thermostat=LangevinThermostat(1.0), temperature=T, velocities=vel0, log=None, seed=5
    )
    Tn, Un, Ttr, Trot = [], [], [], []
    t0 = time.perf_counter()
    for _k in range(n_nvt // rep):
        natv.advance(rep)
        o = natv.observables()
        Tn.append(o["temp_K"])
        Un.append(o["epot"] / sysm.nmol)
        Ttr.append(o["temp_trans"])
        Trot.append(o["temp_rot"])
    t_natv = (time.perf_counter() - t0) / n_nvt
    atoms2 = atoms_from_system(sysm, pos0, H)
    atoms2.set_constraint(rigid_constraints(sysm))
    atoms2.calc = PGMCalculator(PGMEngine(sysm, pos0, H, s))
    atoms2.set_velocities(vel0 * 0.01 / units.fs)
    lang = Langevin(
        atoms2,
        dt * 1000.0 * units.fs,
        temperature_K=T,
        friction=1.0 / (1000.0 * units.fs),
        fixcm=True,
        rng=np.random.default_rng(7),
    )
    Ta, Ua = [], []

    def obs2():
        ke = atoms2.get_kinetic_energy() / KJ
        Ta.append(2.0 * ke / (dof_ase * KB))
        Ua.append(atoms2.get_potential_energy() / KJ / sysm.nmol)

    lang.attach(obs2, interval=rep)
    t0 = time.perf_counter()
    lang.run(n_nvt)
    t_lang = (time.perf_counter() - t0) / n_nvt
    skip = len(Ta) // 10

    def blockerr(x, nb=5):
        x = np.asarray(x)[skip:]
        b = np.array_split(x, nb)
        return float(np.std([np.mean(y) for y in b], ddof=1) / np.sqrt(nb))

    res["nvt"] = {
        "T_native": float(np.mean(Tn[skip:])),
        "T_native_err": blockerr(Tn),
        "T_ase": float(np.mean(Ta[skip:])),
        "T_ase_err": blockerr(Ta),
        "U_native_per_mol": float(np.mean(Un[skip:])),
        "U_native_err": blockerr(Un),
        "U_ase_per_mol": float(np.mean(Ua[skip:])),
        "U_ase_err": blockerr(Ua),
        "T_native_trans": float(np.mean(Ttr[skip:])),
        "T_native_rot": float(np.mean(Trot[skip:])),
        "ms_per_step_ase_langevin": 1e3 * t_lang,
        "ms_per_step_native_langevin": 1e3 * t_natv,
        "ps": args.nvt_ps,
    }
    print(json.dumps(res["nvt"], indent=1), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
