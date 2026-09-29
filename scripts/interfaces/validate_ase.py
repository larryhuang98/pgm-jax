"""ASE + pgm_jax (pgm_jax.interfaces.ase) on the 512-water pGM box of the README.

  1. single points against the native engine (Simulation.from_amber): energy, atomic forces, centre-of-
     mass forces of the rigid bodies, pressure (molecular virial), double and mixed precision;
  2. NVE: ASE VelocityVerlet + FixRigidMolecules (SHAKE / RATTLE) vs the native rigid-body NVE, same
     start (after a native NVT equilibration), same dt: drift and fluctuation of E_tot;
  3. NVT: ASE Langevin vs native Langevin (gamma 1/ps): temperature and <U>;
  4. cost per step: native, ASE (engine call, constraints, ASE bookkeeping).

    python scripts/interfaces/validate_ase.py --out validation/interfaces/ase.json [--ps-nve 10 --ps-nvt 20]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
from ase import units  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402

from pgm_jax.interfaces import PGMEngine  # noqa: E402
from pgm_jax.interfaces.ase import PGMCalculator, atoms_from_system, rigid_constraints  # noqa: E402
from pgm_jax.md.box import volume  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.md.simulation import Simulation  # noqa: E402

TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
RST = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")
KJ = units.kJ / units.mol
BAR = 16.605390671738466           # bar per kJ/mol/nm^3


def native_atomic(sim):
    """Atomic forces and dipoles of the native force field at the native state (its own lists)."""
    st = sim.state
    pos = sim.rigid.positions(st.dyn.position)
    idx = sim.nb.candidates(st.nbr, st.dyn.position.center, st.box, pos)[0]
    res = jax.jit(sim.ff.compute)(pos, st.box, idx, sim.ff.init_induction())
    return np.asarray(pos), np.asarray(res.forces), float(res.energy["total"]), np.asarray(res.induction.mu)


def single_point(prec, tol):
    s = MDSettings(precision=prec, dipole_tol=tol)
    sim = Simulation.from_amber(TOP, RST, settings=s, ensemble="nve", log=None)
    pos, F_nat, E_nat, mu_nat = native_atomic(sim)
    H = np.asarray(sim.state.box)
    out = {"precision": prec, "dipole_tol": tol, "E_native": float(sim.state.epot)}
    for mode in ("molecular", "atomic"):
        eng = PGMEngine.from_simulation(sim, stress=mode)
        atoms = atoms_from_system(sim.sys, pos, H)
        atoms.calc = PGMCalculator(eng)
        E = atoms.get_potential_energy() / KJ
        F = atoms.get_forces() / (KJ / 10.0)
        stress = atoms.get_stress(voigt=False) / (KJ / 1000.0)                       # kJ/mol/nm^3
        mu = atoms.calc.get_induced_dipoles(atoms) / 10.0
        if mode == "molecular":
            ke_t = float(sim.integ.kinetic(sim.state)[1])
            V = float(volume(sim.state.box))
            P_ase = (2.0 * ke_t - V * np.trace(stress)) / (3.0 * V) * BAR
            P_nat = sim.pressure()
            Fcom = np.zeros((sim.sys.nmol, 3))
            np.add.at(Fcom, sim.sys.mol, F)
            Fc_nat = np.asarray(sim.state.dyn.force.center)
            out.update(E_ase=E, dE=E - out["E_native"], dE_rel=abs(E - out["E_native"]) / abs(out["E_native"]),
                       F_maxdiff=float(np.abs(F - F_nat).max()), F_rms=float(np.sqrt(np.mean(F_nat ** 2))),
                       Fcom_maxdiff=float(np.abs(Fcom - Fc_nat).max()), mu_maxdiff=float(np.abs(mu - mu_nat).max()),
                       P_native_bar=P_nat, P_ase_bar=float(P_ase), dP_bar=float(P_ase - P_nat))
        else:
            out["stress_atomic_kjmol_nm3"] = stress.tolist()
            out["P_atomic_virial_bar"] = float(-np.trace(stress) / 3.0 * BAR)
    return out


def drift(t_ps, e, dof, T):
    slope = np.polyfit(np.asarray(t_ps) / 1000.0, np.asarray(e), 1)[0]          # kJ/mol/ns
    return {"drift_kT_per_ns_per_dof": float(slope / (KB * T) / dof), "std_kJmol": float(np.std(e)),
            "std_per_dof_kT": float(np.std(e) / (KB * T) / np.sqrt(dof))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation/interfaces/ase.json")
    ap.add_argument("--ps-eq", type=float, default=5.0)
    ap.add_argument("--ps-nve", type=float, default=10.0)
    ap.add_argument("--ps-nvt", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--T", type=float, default=298.0)
    args = ap.parse_args()
    res = {"device": str(jax.devices()[0])}
    res["single_point_double"] = single_point("double", 1e-10)
    print(json.dumps(res, indent=1), flush=True)
    res["single_point_mixed"] = single_point("mixed", 1e-5)
    print(json.dumps(res["single_point_mixed"], indent=1), flush=True)

    s = MDSettings(precision=args.precision)
    dt, T = args.dt, args.T
    n_eq, n_nve, n_nvt = (int(round(x / dt)) for x in (args.ps_eq, args.ps_nve, args.ps_nvt))
    rep = 100
    sim = Simulation.from_amber(TOP, RST, settings=s, ensemble="nvt", temperature=T, gamma=1.0, dt=dt, log=None)
    sim._advance(n_eq)
    pos0, vel0, H = sim.positions_nm(), sim.velocities_nm_ps(), np.asarray(sim.state.box)
    sysm = sim.sys

    # ---- native NVE
    nat = Simulation(sysm, pos0, H, s, dt=dt, ensemble="nve", vel_nm_ps=vel0, log=None)
    dof = nat.integ.dof
    etot_start = float(nat.observables()["etot"])
    nat._advance(rep)                              # compile
    t, E, U = [], [], []
    t0 = time.perf_counter()
    for k in range(n_nve // rep):
        nat._advance(rep)
        o = nat.observables()
        t.append(o["time_ps"]); E.append(o["etot"]); U.append(o["epot"])
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
    dyn.run(rep)                                   # compile / warm up
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
    res["nve_ase"] = dict(drift(rec["t"], rec["E"], dof_ase, T), ms_per_step=1e3 * t_ase,
                          engine_ms_per_call=1e3 * st["time"] / max(st["calls"], 1), cg_per_call=st["cg"] / max(st["calls"], 1),
                          dof=dof_ase, T_mean=float(np.mean(rec["T"])), E0=e0, E0_native=etot_start, repeats=st["repeats"], rebuilds=st["rebuilds"])
    print(json.dumps({k: res[k] for k in ("nve_native", "nve_ase")}, indent=1), flush=True)

    # ---- NVT: native Langevin vs ASE Langevin
    natv = Simulation(sysm, pos0, H, s, dt=dt, ensemble="nvt", temperature=T, gamma=1.0, vel_nm_ps=vel0, log=None, seed=5)
    Tn, Un, Ttr, Trot = [], [], [], []
    t0 = time.perf_counter()
    for k in range(n_nvt // rep):
        natv._advance(rep)
        o = natv.observables()
        Tn.append(o["temp_K"]); Un.append(o["epot"] / sysm.nmol); Ttr.append(o["temp_trans"]); Trot.append(o["temp_rot"])
    t_natv = (time.perf_counter() - t0) / n_nvt
    atoms2 = atoms_from_system(sysm, pos0, H)
    atoms2.set_constraint(rigid_constraints(sysm))
    atoms2.calc = PGMCalculator(PGMEngine(sysm, pos0, H, s))
    atoms2.set_velocities(vel0 * 0.01 / units.fs)
    lang = Langevin(atoms2, dt * 1000.0 * units.fs, temperature_K=T, friction=1.0 / (1000.0 * units.fs), fixcm=True,
                    rng=np.random.default_rng(7))
    Ta, Ua = [], []

    def obs2():
        ke = atoms2.get_kinetic_energy() / KJ
        Ta.append(2.0 * ke / (dof_ase * KB)); Ua.append(atoms2.get_potential_energy() / KJ / sysm.nmol)

    lang.attach(obs2, interval=rep)
    t0 = time.perf_counter()
    lang.run(n_nvt)
    t_lang = (time.perf_counter() - t0) / n_nvt
    skip = len(Ta) // 10

    def blockerr(x, nb=5):
        x = np.asarray(x)[skip:]
        b = np.array_split(x, nb)
        return float(np.std([np.mean(y) for y in b], ddof=1) / np.sqrt(nb))

    res["nvt"] = {"T_native": float(np.mean(Tn[skip:])), "T_native_err": blockerr(Tn),
                  "T_ase": float(np.mean(Ta[skip:])), "T_ase_err": blockerr(Ta),
                  "U_native_per_mol": float(np.mean(Un[skip:])), "U_native_err": blockerr(Un),
                  "U_ase_per_mol": float(np.mean(Ua[skip:])), "U_ase_err": blockerr(Ua),
                  "T_native_trans": float(np.mean(Ttr[skip:])), "T_native_rot": float(np.mean(Trot[skip:])),
                  "ms_per_step_ase_langevin": 1e3 * t_lang, "ms_per_step_native_langevin": 1e3 * t_natv, "ps": args.ps_nvt}
    print(json.dumps(res["nvt"], indent=1), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
