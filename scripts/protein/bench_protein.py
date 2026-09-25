"""Speed of a solvated protein in the pgm_jax MD engine: pGM (placeholder electrostatics unless a
residue library is given) + Amber-form bonded terms and CMAP from the prmtop, rigid water by
constraints, X-H constraints, hydrogen mass repartitioning.

    python scripts/protein/bench_protein.py runs/protein/ubq.prmtop runs/protein/ubq.inpcrd --dt 0.002 --steps 2000
"""
import argparse
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pgm_jax.md.flexible import FlexibleSimulation  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.protein import ResidueLibrary, amber_template, load_amber  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("inpcrd")
ap.add_argument("--library", default=None)
ap.add_argument("--dt", type=float, default=0.002)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--cut", type=float, default=0.9)
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--precision", default="mixed")
ap.add_argument("--hmr", type=float, default=3.024)
a = ap.parse_args()
lib = ResidueLibrary.load(a.library) if a.library else "placeholder"
t0 = time.time()
asys = load_amber(a.prmtop, a.inpcrd, electrostatics=lib)
prot = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
tpl = {k: amber_template(asys.molecules[k], a.prmtop) for k in prot}
st = MDSettings(cutoff=a.cut, skin=0.1, dipole_tol=a.tol, precision=a.precision)
sim = FlexibleSimulation(asys.system(), asys.templates(tpl), asys.system_positions(), asys.box, st, dt=a.dt,
                         ensemble="nvt", constraints="h-bonds", hmr=a.hmr, log=sys.stdout)
print(f"setup {time.time() - t0:.1f} s: {sim.sys.n} atoms, {len(prot)} protein chain(s) "
      f"({sum(asys.molecules[k].n for k in prot)} atoms), {sim.topology.n_group} groups, "
      f"special width {sim.topology.special.shape[1]}, rows {sim.ff.mc}", flush=True)
t1 = time.time()
print("minimise:", sim.minimize(300), flush=True)
sim._advance(500)
print(f"minimise + compile + 500 steps {time.time() - t1:.1f} s", flush=True)
t0 = time.time()
done = 0
while done < a.steps:
    sim._advance(500)
    done += 500
el = time.time() - t0
o = sim.observables()
print(f"{sim.sys.n} atoms, dt {a.dt * 1000:g} fs, {a.precision}: {el / done * 1e3:.3f} ms/step, "
      f"{done * a.dt / 1000 / el * 86400:.1f} ns/day; T {o['temp_K']:.1f} K, CG iters {o['cg_iter']} (max {o['cg_iter_max']}), "
      f"shake {o['shake_err']:.1e}", flush=True)
mem = jax.devices()[0].memory_stats() or {}
if "peak_bytes_in_use" in mem:
    print(f"GPU memory peak {mem['peak_bytes_in_use'] / 2**30:.2f} GiB", flush=True)
