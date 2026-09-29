"""Independent check of the finite-field response of TIP3P with OpenMM (CPU platform).

The same 512-water box and topology as scripts/dielectric/finite_field.py --model tip3p
(PGM_EPSP/tip3p/tip3p_512.prmtop), scaled to the same density (molecular centres), rigid water, PME
with a 0.9 nm cutoff (Ewald error tolerance 5e-5) and the dispersion correction, 298 K, NVT
(Langevin middle integrator, 1/ps), 2 fs, a uniform field E along z as a CustomExternalForce
(energy -q E z per atom, force q E); M_z = sum q z of whole molecules sampled every 25 steps
(50 fs).  eps = 1 + <M_z> / (eps0 V E) (tin-foil: OpenMM's PME omits k = 0 too), from the samples
after 50 ps, with a 10-block error.  Checked with OpenMM 8.2.

Usage:

    <python with OpenMM> scripts/dielectric/openmm_tip3p_field.py --field-V-nm 0.1 --time-ns 1 -o runs/ff/omm_0.1
    python scripts/dielectric/openmm_tip3p_field.py --help

Inputs: PGM_EPSP/tip3p/tip3p_512.{prmtop,rst7} (pgm_jax.paths).
Outputs: <out>.txt (time [ps], M_z [e nm]), <out>.json (eps with a block error, speed), printed result.
Units: --field-V-nm V/nm (sign included), --time-ns ns, --density-g-cm3 g/cm^3.
Runtime: CPU (--threads); one field per process.  Needs OpenMM (and pgm_jax importable for the
paths).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pgm_jax.paths import resource
from pgm_jax.units import AMU_NM3_TO_G_CM3

EPSP = resource("epsp", "tip3p")
EPS_FACTOR = 18.0951  # e / (eps0 nm): eps - 1 = EPS_FACTOR <M.e> / (V E), M e nm, V nm^3, E V/nm
FARADAY = 96.48533212  # kJ/mol per (e V)
DT_PS, SAMPLE_EVERY, SKIP_PS, BLOCKS = 0.002, 25, 50.0, 10


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-V-nm", type=float, required=True, help="field along z [V/nm] (sign included)")
    ap.add_argument("--time-ns", type=float, default=1.0, help="run length [ns]")
    ap.add_argument("--density-g-cm3", type=float, default=0.986, help="density the box is scaled to [g/cm^3]")
    ap.add_argument("--threads", type=int, default=16, help="CPU threads of OpenMM")
    ap.add_argument("--seed", type=int, default=1, help="random seed (Langevin noise, velocities)")
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    return ap


def scale_to_density(
    pos: np.ndarray, H: np.ndarray, masses: np.ndarray, density: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a water box (molecular centres of mass and box) isotropically to a density.

    Parameters
    ----------
    pos : np.ndarray (N, 3)
        Positions of the waters (O, H, H per molecule) [nm].
    H : np.ndarray (3, 3)
        Box vectors as rows [nm].
    masses : np.ndarray (N,)
        Atomic masses [amu].
    density : float
        Target density [g/cm^3].

    Returns
    -------
    positions, box : np.ndarray
        Scaled positions and box [nm].
    """
    mol = np.repeat(np.arange(len(pos) // 3), 3)
    V = abs(np.linalg.det(H))
    V_t = masses.sum() * AMU_NM3_TO_G_CM3 / density
    s = (V_t / V) ** (1 / 3)
    com = np.zeros((mol.max() + 1, 3))
    np.add.at(com, mol, masses[:, None] * pos)
    com /= np.bincount(mol, weights=masses)[:, None]
    return pos + ((s - 1) * com)[mol], H * s


def run(a: argparse.Namespace) -> dict:
    """Run the OpenMM simulation in the field and return the result (also written to <out>.txt / .json).

    Returns
    -------
    dict
        field [V/nm], V [nm^3], Mz and Mz_err [e nm], eps and eps_err, ns_per_day.
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    prm = app.AmberPrmtopFile(f"{EPSP}/tip3p_512.prmtop")
    crd = app.AmberInpcrdFile(f"{EPSP}/tip3p_512.rst7")
    system = prm.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * u.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=5e-5,
    )
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setUseDispersionCorrection(True)
            nb = f
    n = system.getNumParticles()
    q = np.array([nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(n)])
    m = np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(n)])
    pos = np.array(crd.positions.value_in_unit(u.nanometer))
    H = np.array([v.value_in_unit(u.nanometer) for v in crd.boxVectors])
    pos, H = scale_to_density(pos, H, m, a.density_g_cm3)
    V = abs(np.linalg.det(H))
    ext = mm.CustomExternalForce("-qE*z")  # kJ/mol with qE = q E F (per atom)
    ext.addPerParticleParameter("qE")
    for i in range(n):
        ext.addParticle(i, [q[i] * a.field_V_nm * FARADAY])
    system.addForce(ext)
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*v) * u.nanometer for v in H])
    integ = mm.LangevinMiddleIntegrator(298 * u.kelvin, 1.0 / u.picosecond, DT_PS * u.picoseconds)
    integ.setRandomNumberSeed(a.seed)
    plat = mm.Platform.getPlatformByName("CPU")
    sim = app.Simulation(prm.topology, system, integ, plat, {"Threads": str(a.threads)})
    sim.context.setPositions(pos)
    sim.context.setPeriodicBoxVectors(*[mm.Vec3(*v) * u.nanometer for v in H])
    sim.minimizeEnergy(maxIterations=200)
    sim.context.setVelocitiesToTemperature(298 * u.kelvin, a.seed)
    nsteps = int(round(a.time_ns * 1e6 / 2.0))
    out = []
    t0 = time.time()
    for k in range(nsteps // SAMPLE_EVERY):
        integ.step(SAMPLE_EVERY)
        x = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
        out.append(((k + 1) * SAMPLE_EVERY * DT_PS, float(np.sum(q * x[:, 2]))))
    el = time.time() - t0
    d = np.array(out)
    np.savetxt(a.out + ".txt", d, header=f"time_ps M_z(e nm); field {a.field_V_nm} V/nm, V {V} nm^3")
    Mz = d[d[:, 0] > SKIP_PS, 1]
    b = Mz[: len(Mz) // BLOCKS * BLOCKS].reshape(BLOCKS, -1).mean(1)
    res = {
        "field": a.field_V_nm,
        "V": V,
        "Mz": float(Mz.mean()),
        "Mz_err": float(b.std(ddof=1) / np.sqrt(BLOCKS)),
        "eps": 1 + EPS_FACTOR * float(Mz.mean()) / (V * a.field_V_nm),
        "ns_per_day": a.time_ns / el * 86400,
    }
    res["eps_err"] = EPS_FACTOR * res["Mz_err"] / (V * abs(a.field_V_nm))
    with open(a.out + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run the simulation and print the result (see the module docstring)."""
    print(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
