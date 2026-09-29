"""Checks of flexible-molecule MD with methanol: gas-phase consistency of forces, NVE energy
conservation, and a short NPT run.  Run on a GPU node:  python examples/flex_methanol_check.py"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.bonded import terms as T
from pgm_jax.bonded.fit import Fitter
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.bonded.study.data import load
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.system import System

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jax.config.update("jax_enable_x64", True)
out = os.path.join(ROOT, "runs/flex")
os.makedirs(out, exist_ok=True)
tpath = os.path.join(out, "methanol.flex")
if not os.path.exists(tpath):
    specs, data = load(["methanol"])
    model = BondedModel(specs, BondedSettings(families=T.PAPER))
    P = Fitter(model, {0: {"train": data[0]["train"]}}).fit(model.init_params(), maxiter=20000, verbose=True)
    FlexibleTemplate.from_fit(model, P).save(tpath)
tpl = FlexibleTemplate.load(tpath)
print("template", tpl.name, tpl.n, "atoms; intramolecular LJ pairs", len(tpl.lj_pairs()[0]))

# 1. one molecule in a large box vs the gas-phase model (images and PME error aside)
x = np.asarray(tpl.spec.ref_xyz) + 0.003 * np.random.default_rng(0).normal(size=(tpl.n, 3))
H = np.eye(3) * 4.0
sim = FlexibleSimulation(
    System([tpl.pgm]),
    [tpl],
    x + 2.0,
    H,
    MDSettings(precision="double", dipole_tol=1e-8, cutoff=1.8, skin=0.05),
    ensemble="nve",
    log=None,
)
F_md = np.asarray(sim.state.dyn.force)
E_md = float(sim.state.epot)
m = tpl.model
e_gas, g_gas = jax.value_and_grad(lambda R: m.energy(0, R, jax.tree_util.tree_map(jnp.asarray, tpl.P))[0])(
    jnp.asarray(x)
)
print(
    f"single molecule: |F_md - F_gas| max {np.abs(F_md + np.asarray(g_gas)).max():.3g}, rms F "
    f"{np.sqrt(np.mean(np.asarray(g_gas) ** 2)):.3g} kJ/mol/nm"
)

# 2. NVE energy conservation, 216 molecules
pos, H = liquid_box(tpl, 216, 0.55, seed=1, min_dist=0.18)
st = MDSettings(precision="mixed", dipole_tol=1e-5)
sim = FlexibleSimulation(
    System([tpl.pgm] * 216), [tpl] * 216, pos, H, st, dt=0.0005, ensemble="nvt", temperature=298.0, gamma=5.0, log=None
)
sim.run(4000, report=4000, prefix=os.path.join(out, "equil"))
sim2 = FlexibleSimulation(
    System([tpl.pgm] * 216),
    [tpl] * 216,
    sim.positions_nm(),
    np.asarray(sim.state.box),
    st,
    dt=0.0005,
    ensemble="nve",
    vel_nm_ps=sim.velocities_nm_ps(),
    log=None,
)
E = []
for _ in range(10):
    sim2.run(400, report=400, prefix=os.path.join(out, "nve"))
    o = sim2.observables()
    E.append(o["etot"])
E = np.array(E)
print(
    "NVE 2 ps: total energy",
    np.round(E, 2),
    "drift per ps per dof (kT)",
    (E[-1] - E[0]) / 1.8 / (3 * 216 * 6) / (0.0083145 * 298),
)

# 3. NPT
t0 = time.time()
sim3 = FlexibleSimulation(
    System([tpl.pgm] * 216),
    [tpl] * 216,
    sim.positions_nm(),
    np.asarray(sim.state.box),
    st,
    dt=0.0005,
    ensemble="npt",
    temperature=298.0,
    gamma=1.0,
    vel_nm_ps=sim.velocities_nm_ps(),
    barostat_interval=50,
    log=None,
)
sim3.run(100000, report=2000, prefix=os.path.join(out, "npt"))
print(
    "NPT 50 ps:",
    {k: round(v, 3) for k, v in sim3.observables().items() if isinstance(v, float)},
    "wall %.0f s" % (time.time() - t0),
)
