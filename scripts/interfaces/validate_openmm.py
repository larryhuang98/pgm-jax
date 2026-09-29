"""OpenMM (>= 8.4, PythonForce) + pgm_jax (pgm_jax.interfaces.openmm) on the 512-water pGM box.

1. single point: OpenMM's energy and forces (State) vs the engine and the native force field;
2. NVE: OpenMM VerletIntegrator + SETTLE vs the native rigid-body NVE (same start, same dt);
3. NVT: LangevinMiddleIntegrator vs native Langevin (gamma 1/ps): temperature, <U>;
4. NPT: MonteCarloBarostat vs the native Monte Carlo barostat: density;
5. cost per step on OpenMM's CPU and (if present) CUDA platforms vs native.

  PYTHONPATH=runs/ommlib python scripts/interfaces/validate_openmm.py --out validation/interfaces/openmm.json
"""

import argparse
import json
import os
import time

import jax
import numpy as np
import openmm
from openmm import unit

from pgm_jax.cli.args import setup_logging
from pgm_jax.interfaces import PGMEngine
from pgm_jax.interfaces.openmm import PGMOpenMM
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import Langevin
from pgm_jax.paths import resource
from pgm_jax.units import AMU_NM3_TO_G_CM3, KB

jax.config.update("jax_enable_x64", True)
TOP = resource("gvdw_data", "topology/rayl_512_v2.prmtop")
RST = resource("gvdw_data", "inputs/lj/inpcrd.restrt")
PS = unit.picosecond
KJ = unit.kilojoule_per_mole


def platforms():
    """Usable platforms: a context must open (OpenMM's CUDA platform cannot open a context on a GPU in
    exclusive-process mode that JAX already uses)."""
    names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    ok = []
    for p in ("CUDA", "CPU"):
        if p not in names:
            continue
        try:
            s = openmm.System()
            s.addParticle(1.0)
            openmm.Context(s, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(p))
            ok.append(p)
        except Exception as err:  # noqa: BLE001
            print(f"# platform {p} unusable: {err}", flush=True)
    return ok


def context(om, integrator, platform, pos, vel=None, barostat=None):
    system = om.system(rigid=True)
    if barostat is not None:
        system.addForce(barostat)
    plat = openmm.Platform.getPlatformByName(platform)
    props = {"Precision": "mixed"} if platform == "CUDA" else {}
    ctx = openmm.Context(system, integrator, plat, props)
    ctx.setPeriodicBoxVectors(*[openmm.Vec3(*r) for r in om.box()])
    ctx.setPositions(pos)
    if vel is not None:
        ctx.setVelocities(vel)
    ctx.applyConstraints(1e-10)
    ctx.applyVelocityConstraints(1e-10)
    return ctx, system


def blockerr(x, nb=5):
    x = np.asarray(x)
    return float(np.std([np.mean(y) for y in np.array_split(x, nb)], ddof=1) / np.sqrt(nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation/interfaces/openmm.json")
    ap.add_argument("--ps-eq", type=float, default=5.0)
    ap.add_argument("--ps-nve", type=float, default=10.0)
    ap.add_argument("--ps-nvt", type=float, default=20.0)
    ap.add_argument("--ps-npt", type=float, default=50.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--T", type=float, default=298.0)
    ap.add_argument("--platform", default=None)
    args = ap.parse_args()
    setup_logging()
    plats = platforms() if args.platform is None else [args.platform]
    res = {"device": str(jax.devices()[0]), "openmm": openmm.__version__, "platforms": plats}
    dt, T = args.dt, args.T
    rep = 100

    # ---- single point, double precision
    s2 = MDSettings(precision="double", dipole_tol=1e-10)
    sim = Simulation.from_amber(TOP, RST, settings=s2, thermostat=None, log=None)
    st = sim.state
    pos = sim.positions()
    idx = sim.nb.candidates(st.nbr, st.dyn.position.center, st.box, sim.rigid.positions(st.dyn.position))[0]
    nat = jax.jit(sim.ff.compute)(pos, st.box, idx, sim.ff.init_induction())
    om = PGMOpenMM(PGMEngine.from_simulation(sim))
    sp = {}
    for p in plats:
        ctx, _ = context(om, openmm.VerletIntegrator(dt * PS), p, pos)
        ctx.setPositions(pos)  # before constraints: compare the same geometry
        s_ = ctx.getState(getEnergy=True, getForces=True)
        E = s_.getPotentialEnergy().value_in_unit(KJ)
        F = s_.getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer)
        sp[p] = {
            "E_openmm": E,
            "E_native": float(st.epot),
            "dE": E - float(st.epot),
            "F_maxdiff": float(np.abs(F - np.asarray(nat.forces)).max()),
            "F_rms": float(np.sqrt(np.mean(np.asarray(nat.forces) ** 2))),
        }
        del ctx
    res["single_point_double"] = sp
    print(json.dumps(res, indent=1), flush=True)

    # ---- equilibrate natively, then NVE / NVT / NPT
    s = MDSettings()
    sim = Simulation.from_amber(TOP, RST, settings=s, thermostat=Langevin(1.0), temperature=T, dt=dt, log=None)
    sim.advance(int(round(args.ps_eq / dt)))
    pos0, vel0, H = sim.positions(), sim.velocities(), np.asarray(sim.state.box)
    sysm = sim.sys
    n_nve, n_nvt, n_npt = (int(round(x / dt)) for x in (args.ps_nve, args.ps_nvt, args.ps_npt))

    nat = Simulation(sysm, pos0, H, s, dt=dt, thermostat=None, velocities=vel0, log=None)
    dof = nat.integ.dof
    nat.advance(rep)
    t, E = [], []
    t0 = time.perf_counter()
    for _k in range(n_nve // rep):
        nat.advance(rep)
        o = nat.observables()
        t.append(o["time_ps"])
        E.append(o["etot"])
    ms_nat = 1e3 * (time.perf_counter() - t0) / (n_nve // rep * rep)
    slope = np.polyfit(np.asarray(t) / 1000.0, E, 1)[0]
    res["nve_native"] = {
        "drift_kT_per_ns_per_dof": float(slope / (KB * T) / dof),
        "std_kJmol": float(np.std(E)),
        "ms_per_step": ms_nat,
    }

    for p in plats:
        eng = PGMEngine(sysm, pos0, H, s)
        om = PGMOpenMM(eng)
        integ = openmm.VerletIntegrator(dt * PS)
        integ.setConstraintTolerance(1e-10)
        ctx, system = context(om, integ, p, pos0, vel0)
        integ.step(rep)
        eng.stats.update(calls=0, time=0.0, cg=0)
        om.stats.update(calls=0, t_call=0.0, t_convert=0.0)
        t, E = [], []
        t0 = time.perf_counter()
        for _k in range(n_nve // rep):
            integ.step(rep)
            s_ = ctx.getState(getEnergy=True)
            t.append(s_.getTime().value_in_unit(PS))
            E.append(s_.getPotentialEnergy().value_in_unit(KJ) + s_.getKineticEnergy().value_in_unit(KJ))
        wall = time.perf_counter() - t0
        nstep = n_nve // rep * rep
        ndof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
        slope = np.polyfit(np.asarray(t) / 1000.0, E, 1)[0]
        res[f"nve_openmm_{p}"] = {
            "drift_kT_per_ns_per_dof": float(slope / (KB * T) / ndof),
            "std_kJmol": float(np.std(E)),
            "ms_per_step": 1e3 * wall / nstep,
            "engine_ms_per_call": 1e3 * eng.stats["time"] / max(eng.stats["calls"], 1),
            "pythonforce_ms_per_call": 1e3 * om.stats["t_call"] / max(om.stats["calls"], 1),
            "state_to_numpy_ms": 1e3 * om.stats["t_convert"] / max(om.stats["calls"], 1),
            "calls_per_step": eng.stats["calls"] / nstep,
            "cg_per_call": eng.stats["cg"] / max(eng.stats["calls"], 1),
            "dof": ndof,
        }
        print(json.dumps(res[f"nve_openmm_{p}"], indent=1), flush=True)
        del ctx

    # ---- NVT
    p = plats[0]
    natv = Simulation(
        sysm, pos0, H, s, dt=dt, thermostat=Langevin(1.0), temperature=T, velocities=vel0, log=None, seed=5
    )
    Tn, Un = [], []
    for _k in range(n_nvt // rep):
        natv.advance(rep)
        o = natv.observables()
        Tn.append(o["temp_K"])
        Un.append(o["epot"] / sysm.nmol)
    eng = PGMEngine(sysm, pos0, H, s)
    om = PGMOpenMM(eng)
    integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / PS, dt * PS)
    integ.setRandomNumberSeed(11)
    ctx, system = context(om, integ, p, pos0, vel0)
    ndof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
    To, Uo = [], []
    t0 = time.perf_counter()
    for _k in range(n_nvt // rep):
        integ.step(rep)
        s_ = ctx.getState(getEnergy=True)
        To.append(2 * s_.getKineticEnergy().value_in_unit(KJ) / (ndof * KB))
        Uo.append(s_.getPotentialEnergy().value_in_unit(KJ) / sysm.nmol)
    ms = 1e3 * (time.perf_counter() - t0) / (n_nvt // rep * rep)
    sk = len(To) // 10
    res["nvt"] = {
        "platform": p,
        "T_native": float(np.mean(Tn[sk:])),
        "T_native_err": blockerr(Tn[sk:]),
        "T_openmm": float(np.mean(To[sk:])),
        "T_openmm_err": blockerr(To[sk:]),
        "U_native_per_mol": float(np.mean(Un[sk:])),
        "U_native_err": blockerr(Un[sk:]),
        "U_openmm_per_mol": float(np.mean(Uo[sk:])),
        "U_openmm_err": blockerr(Uo[sk:]),
        "ms_per_step": ms,
    }
    print(json.dumps(res["nvt"], indent=1), flush=True)
    del ctx

    # ---- NPT (Monte Carlo barostats), density
    if n_npt > 0:
        mass = float(np.sum(sysm.masses))
        natp = Simulation(
            sysm,
            pos0,
            H,
            s,
            dt=dt,
            thermostat=Langevin(1.0),
            barostat=MonteCarloBarostat(1.0, 25),
            temperature=T,
            velocities=vel0,
            log=None,
            seed=6,
        )
        rn = []
        for _k in range(n_npt // rep):
            natp.advance(rep)
            rn.append(natp.observables()["density_g_cm3"])
        eng = PGMEngine(sysm, pos0, H, s)
        om = PGMOpenMM(eng)
        integ = openmm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / PS, dt * PS)
        integ.setRandomNumberSeed(12)
        ctx, system = context(om, integ, p, pos0, vel0, openmm.MonteCarloBarostat(1.0 * unit.bar, T * unit.kelvin, 25))
        ro = []
        t0 = time.perf_counter()
        for _k in range(n_npt // rep):
            integ.step(rep)
            V = ctx.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
            ro.append(mass / V * AMU_NM3_TO_G_CM3)
        ms = 1e3 * (time.perf_counter() - t0) / (n_npt // rep * rep)
        sk = len(ro) // 5
        res["npt"] = {
            "platform": p,
            "rho_native": float(np.mean(rn[sk:])),
            "rho_native_err": blockerr(rn[sk:]),
            "rho_openmm": float(np.mean(ro[sk:])),
            "rho_openmm_err": blockerr(ro[sk:]),
            "ms_per_step": ms,
            "ps": args.ps_npt,
            "engine_rebuilds": eng.stats["rebuilds"],
            "engine_repeats": eng.stats["repeats"],
            "calls_per_step": eng.stats["calls"] / (n_npt // rep * rep),
        }
        print(json.dumps(res["npt"], indent=1), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
