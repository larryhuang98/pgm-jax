"""Independent check of the finite-field response of TIP3P with OpenMM (8.2, CPU platform): the same
512-water box and topology (~/project/epsp/tip3p/tip3p_512.prmtop), scaled to the same density,
rigid water, PME with a 0.9 nm cutoff and the dispersion correction, 298 K, NVT, 2 fs, a uniform field
E along z as a CustomExternalForce (-q E z; forces q E), M_z = sum q z of whole molecules sampled
every 50 fs.  eps = 1 + <M_z>/(eps0 V E) (tin-foil: OpenMM's PME omits k = 0 too).

    ~/miniconda3/envs/colabfold/bin/python scripts/openmm_tip3p_field.py --field 0.1 --ns 1 -o runs/ff/omm_0.1
Output: prefix.txt (time_ps, M_z in e nm), prefix.json (eps with a block error, speed)."""
import argparse
import json
import os
import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

EPSP = os.path.expanduser("~/project/epsp/tip3p")
EPS_FACTOR = 18.0951                      # e / (eps0 nm): eps - 1 = EPS_FACTOR <M.e> / (V E), M e nm, V nm^3, E V/nm
FARADAY = 96.48533212                     # kJ/mol per (e V)

ap = argparse.ArgumentParser()
ap.add_argument("--field", type=float, required=True, help="V/nm along z (sign included)")
ap.add_argument("--ns", type=float, default=1.0)
ap.add_argument("--density", type=float, default=0.986)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()

prm = app.AmberPrmtopFile(f"{EPSP}/tip3p_512.prmtop")
crd = app.AmberInpcrdFile(f"{EPSP}/tip3p_512.rst7")
system = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer, constraints=app.HBonds,
                          rigidWater=True, ewaldErrorTolerance=5e-5)
for f in system.getForces():
    if isinstance(f, mm.NonbondedForce):
        f.setUseDispersionCorrection(True)
        nb = f
q = np.array([nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(system.getNumParticles())])
# box and positions scaled to the density (molecular centres)
pos = np.array(crd.positions.value_in_unit(u.nanometer))
H = np.array([v.value_in_unit(u.nanometer) for v in crd.boxVectors])
m = np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())])
mol = np.repeat(np.arange(len(pos) // 3), 3)
V = abs(np.linalg.det(H))
V_t = m.sum() * 1.66053906660e-3 / a.density
s = (V_t / V) ** (1 / 3)
com = np.zeros((mol.max() + 1, 3))
np.add.at(com, mol, m[:, None] * pos)
com /= np.bincount(mol, weights=m)[:, None]
pos = pos + ((s - 1) * com)[mol]
H = H * s
V = abs(np.linalg.det(H))
ext = mm.CustomExternalForce("-qE*z")                      # kJ/mol with qE = q E F (per atom)
ext.addPerParticleParameter("qE")
for i in range(system.getNumParticles()):
    ext.addParticle(i, [q[i] * a.field * FARADAY])
system.addForce(ext)
system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*v) * u.nanometer for v in H])
integ = mm.LangevinMiddleIntegrator(298 * u.kelvin, 1.0 / u.picosecond, 0.002 * u.picoseconds)
integ.setRandomNumberSeed(a.seed)
plat = mm.Platform.getPlatformByName("CPU")
sim = app.Simulation(prm.topology, system, integ, plat, {"Threads": str(a.threads)})
sim.context.setPositions(pos)
sim.context.setPeriodicBoxVectors(*[mm.Vec3(*v) * u.nanometer for v in H])
sim.minimizeEnergy(maxIterations=200)
sim.context.setVelocitiesToTemperature(298 * u.kelvin, a.seed)
nsteps = int(round(a.ns * 1e6 / 2.0))
every = 25
out = []
t0 = time.time()
for k in range(nsteps // every):
    integ.step(every)
    x = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    out.append(((k + 1) * every * 0.002, float(np.sum(q * x[:, 2]))))
el = time.time() - t0
d = np.array(out)
np.savetxt(a.out + ".txt", d, header=f"time_ps M_z(e nm); field {a.field} V/nm, V {V} nm^3")
sel = d[:, 0] > 50
Mz = d[sel, 1]
nb_ = 10
b = Mz[: len(Mz) // nb_ * nb_].reshape(nb_, -1).mean(1)
res = {"field": a.field, "V": V, "Mz": float(Mz.mean()), "Mz_err": float(b.std(ddof=1) / np.sqrt(nb_)),
       "eps": 1 + EPS_FACTOR * float(Mz.mean()) / (V * a.field), "ns_per_day": a.ns / el * 86400}
res["eps_err"] = EPS_FACTOR * res["Mz_err"] / (V * abs(a.field))
json.dump(res, open(a.out + ".json", "w"), indent=1)
print(res)
