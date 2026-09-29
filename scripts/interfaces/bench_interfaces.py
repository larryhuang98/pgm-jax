"""Cost per MD step of pGM driven by external codes vs the native engine (same device, same model).

pGM3P-25 water (the 512-water box of the README, replicated n x n x n), rigid molecules, mixed
precision, 9 A cutoff, PME 48^3 per 512 waters, order 6, dipole tol 1e-5, NVE, dt 1 fs:

  native   Simulation (rigid bodies, blocks of steps on the device)
  engine   PGMEngine.compute on successive MD frames (the per-step cost of every interface)
  ase      ASE VelocityVerlet + PGMCalculator + FixRigidMolecules (SHAKE / RATTLE)
  openmm   OpenMM VerletIntegrator + SETTLE + PythonForce (platforms CPU and CUDA if available)

    python scripts/interfaces/bench_interfaces.py --replicate 1 2 --steps 2000 --out validation/interfaces/bench.json
"""

import argparse
import json
import os
import time

import jax
import numpy as np

from pgm_jax.interfaces import PGMEngine
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.simulation import Simulation
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import resource
from pgm_jax.system import System

jax.config.update("jax_enable_x64", True)
TOP = resource("gvdw_data", "topology/rayl_512_v2.prmtop")
RST = resource("gvdw_data", "inputs/lj/inpcrd.restrt")


def system(n):
    mols = read_prmtop_molecules(TOP)
    xyz, vel, box = read_coordinates(RST)
    H = box_from_cell(*box) * 0.1
    shifts = [i * H[0] + j * H[1] + k * H[2] for i in range(n) for j in range(n) for k in range(n)]
    pos = np.concatenate([xyz * 0.1 + s for s in shifts])
    v = np.concatenate([vel * 0.1] * len(shifts))
    s = MDSettings(cutoff=0.9, pme_grid=(48 * n,) * 3, pme_order=6, dipole_tol=1e-5, precision="mixed")
    return System(mols * len(shifts)), pos, v, H * n, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", type=int, nargs="+", default=[1])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--profile", action="store_true", help="cProfile of 100 ASE steps")
    ap.add_argument("--out", default="validation/interfaces/bench.json")
    a = ap.parse_args()
    res = {"device": str(jax.devices()[0]), "steps": a.steps}
    for n in a.replicate:
        sysm, pos, vel, H, s = system(n)
        r = {"atoms": sysm.n}
        sim = Simulation(sysm, pos, H, s, dt=a.dt, ensemble="nve", vel_nm_ps=vel, log=None)
        sim._advance(100)
        t0 = time.perf_counter()
        frames = []
        for _k in range(a.steps // 100):
            sim._advance(100)
            if len(frames) < 200:
                frames.append(sim.positions_nm())
        r["native_ms"] = 1e3 * (time.perf_counter() - t0) / (a.steps // 100 * 100)
        # the engine alone on consecutive MD frames: run a short native trajectory with frames every step
        sim1 = Simulation(sysm, frames[-1], H, s, dt=a.dt, ensemble="nve", vel_nm_ps=sim.velocities_nm_ps(), log=None)
        fr = []
        for _k in range(300):
            sim1._advance(1)
            fr.append(sim1.positions_nm())
        eng = PGMEngine(sysm, fr[0], H, s)
        for x in fr[:50]:
            eng.compute(x, H)
        eng.stats.update(calls=0, time=0.0, cg=0)
        t0 = time.perf_counter()
        for x in fr[50:]:
            eng.compute(x, H)
        r["engine_ms"] = 1e3 * (time.perf_counter() - t0) / 250
        r["engine_cg_per_call"] = eng.stats["cg"] / 250
        for x in fr[:10]:
            eng.compute(x, H, virial=True)  # compile the virial variant
        t0 = time.perf_counter()
        for x in fr[50:]:
            eng.compute(x, H, virial=True)
        r["engine_virial_ms"] = 1e3 * (time.perf_counter() - t0) / 250
        pos0, vel0 = sim.positions_nm(), sim.velocities_nm_ps()
        if "ase" not in a.skip:
            from ase import units
            from ase.md.verlet import VelocityVerlet

            from pgm_jax.interfaces.ase import PGMCalculator, atoms_from_system, rigid_constraints

            e2 = PGMEngine(sysm, pos0, H, s)
            atoms = atoms_from_system(sysm, pos0, H)
            atoms.set_constraint(rigid_constraints(sysm))
            atoms.calc = PGMCalculator(e2)
            atoms.set_velocities(vel0 * 0.01 / units.fs)
            dyn = VelocityVerlet(atoms, a.dt * 1000 * units.fs)
            dyn.run(50)
            e2.stats.update(calls=0, time=0.0)
            t0 = time.perf_counter()
            m = min(a.steps, 1000)
            dyn.run(m)
            r["ase_ms"] = 1e3 * (time.perf_counter() - t0) / m
            r["ase_engine_ms"] = 1e3 * e2.stats["time"] / e2.stats["calls"]
            cons = atoms.constraints[0]
            t0 = time.perf_counter()
            for _ in range(20):
                p = atoms.get_positions()
                cons.adjust_positions(atoms, p + 1e-4)
                q = atoms.get_momenta()
                cons.adjust_momenta(atoms, q)
            r["ase_constraints_ms"] = 1e3 * (time.perf_counter() - t0) / 20
            if a.profile:
                import cProfile
                import io
                import pstats

                pr = cProfile.Profile()
                pr.enable()
                dyn.run(100)
                pr.disable()
                sio = io.StringIO()
                pstats.Stats(pr, stream=sio).sort_stats("tottime").print_stats(12)
                print(sio.getvalue(), flush=True)
        if "openmm" not in a.skip:
            try:
                import openmm
                from openmm import unit

                from pgm_jax.interfaces.openmm import PGMOpenMM

                names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
                for p in [x for x in ("CPU", "CUDA") if x in names]:
                    e3 = PGMEngine(sysm, pos0, H, s)
                    om = PGMOpenMM(e3)
                    integ = openmm.VerletIntegrator(a.dt * unit.picoseconds)
                    try:
                        ctx = openmm.Context(om.system(rigid=True), integ, openmm.Platform.getPlatformByName(p))
                    except Exception as err:  # noqa: BLE001  (CUDA next to JAX on an exclusive GPU)
                        r[f"openmm_{p}"] = str(err)
                        continue
                    ctx.setPeriodicBoxVectors(*[openmm.Vec3(*v) for v in om.box()])
                    ctx.setPositions(pos0)
                    ctx.setVelocities(vel0)
                    integ.step(50)
                    e3.stats.update(calls=0, time=0.0)
                    om.stats.update(calls=0, t_call=0.0, t_convert=0.0)
                    m = min(a.steps, 1000)
                    t0 = time.perf_counter()
                    integ.step(m)
                    ctx.getState(getEnergy=True)
                    r[f"openmm_{p}_ms"] = 1e3 * (time.perf_counter() - t0) / m
                    r[f"openmm_{p}_engine_ms"] = 1e3 * e3.stats["time"] / e3.stats["calls"]
                    r[f"openmm_{p}_state_ms"] = 1e3 * om.stats["t_convert"] / om.stats["calls"]
                    del ctx
            except ImportError as err:
                r["openmm"] = str(err)
        res[f"x{n}"] = r
        print(json.dumps({f"x{n}": r}), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
