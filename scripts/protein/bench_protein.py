"""Speed (and, with --prod-ps, stability and configurational accuracy) of a solvated protein in the
pgm_jax MD engine: pGM (placeholder electrostatics unless a residue library is given) +
Amber-form bonded terms and CMAP from the prmtop, rigid water by constraints, X-H constraints,
hydrogen mass repartitioning (--hmr for the protein, --hmr-water for water).

    python scripts/protein/bench_protein.py runs/protein/ubq.prmtop runs/protein/ubq.inpcrd --dt 0.002 --steps 2000
    # water H 4.0 amu, protein H 3.024, 4 fs, Bussi; 20 ps equilibration, 200 ps sampled every 0.5 ps
    python scripts/protein/bench_protein.py ubq.prmtop ubq.inpcrd --dt 0.004 --hmr-water 4.0 --thermostat bussi \\
        --equil-ps 20 --prod-ps 200 --coords equilibrated.rst7 --minimize 0 --save runs/ubq_4fs

--prod-ps prints the mean potential energy with the error of 10 block averages (masses do not
change the configurational distribution, so <U> measures the time-step error), the mean kinetic
temperature, the largest constraint error and the drift of the effective energy econs.
"""
import argparse
import os
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pgm_jax.md.flexible import FlexibleSimulation  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.md.io import box_from_cell, read_coordinates  # noqa: E402
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
ap.add_argument("--hmr", type=float, default=3.024, help="hydrogen mass (amu) of the protein (and of water "
                "unless --hmr-water is given)")
ap.add_argument("--hmr-water", type=float, default=None, help="hydrogen mass (amu) of water (e.g. 4.0)")
ap.add_argument("--local-niter", type=int, default=0, help="inner CG steps of the short-range preconditioner (0: Jacobi)")
ap.add_argument("--local-cut", type=float, default=0.3, help="preconditioner range (nm)")
ap.add_argument("--predictor", default="mu4")
ap.add_argument("--beta", type=float, default=4.0, help="Ewald coefficient (nm^-1)")
ap.add_argument("--spacing", type=float, default=0.08, help="PME grid spacing (nm)")
ap.add_argument("--thermostat", default="langevin", help="langevin | bussi | gle")
ap.add_argument("--tau", type=float, default=1.0, help="Bussi time constant (ps)")
ap.add_argument("--gamma", type=float, default=1.0, help="Langevin friction (1/ps)")
ap.add_argument("--temp", type=float, default=298.0, help="temperature (K)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--coords", default=None, help="start from these coordinates and box (Amber restart in the "
                "system's atom order, e.g. the .rst7 of --save) instead of the inpcrd; velocities are redrawn")
ap.add_argument("--minimize", type=int, default=300, help="steepest-descent steps before dynamics (0: none)")
ap.add_argument("--equil-ps", type=float, default=0.0, help="equilibration after the speed test (ps)")
ap.add_argument("--prod-ps", type=float, default=0.0, help="sampled production after equilibration (ps)")
ap.add_argument("--sample-ps", type=float, default=0.5, help="sampling interval of the production (ps)")
ap.add_argument("--save", default=None, help="prefix: restart + checkpoint at the end, samples as .npz")
a = ap.parse_args()
lib = ResidueLibrary.load(a.library) if a.library else "placeholder"
t0 = time.time()
asys = load_amber(a.prmtop, a.inpcrd, electrostatics=lib)
prot = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
tpl = {k: amber_template(asys.molecules[k], a.prmtop) for k in prot}
st = MDSettings(cutoff=a.cut, skin=0.1, dipole_tol=a.tol, precision=a.precision, local_niter=a.local_niter,
                local_cut=a.local_cut, predictor=a.predictor, ewald_beta=a.beta, pme_spacing=a.spacing)
hmr = asys.hmr({"protein": a.hmr, "other": a.hmr, "water": a.hmr if a.hmr_water is None else a.hmr_water, "ion": None})
pos, H = asys.system_positions(), asys.box
if a.coords:
    xyz, _, cell = read_coordinates(a.coords)
    pos, H = xyz * 0.1, box_from_cell(*cell) * 0.1
sim = FlexibleSimulation(asys.system(), asys.templates(tpl), pos, H, st, dt=a.dt, temperature=a.temp,
                         ensemble="nvt", constraints="h-bonds", hmr=hmr, log=sys.stdout,
                         thermostat=a.thermostat, tau_t=a.tau, gamma=a.gamma, seed=a.seed)
m = np.asarray(sim.flex.masses)
el = np.array(sim.sys.elements)
wat = [sim.sys.atom_slice(k) for k, mm in enumerate(asys.molecules) if mm.kind == "water"]
print(f"setup {time.time() - t0:.1f} s: {sim.sys.n} atoms, {len(prot)} protein chain(s) "
      f"({sum(asys.molecules[k].n for k in prot)} atoms), {sim.topology.n_group} groups, "
      f"special width {sim.topology.special.shape[1]}, rows {sim.ff.mc}; hydrogen masses: protein "
      f"{sorted({float(x) for x in np.round(m[sim.sys.atom_slice(prot[0])][el[sim.sys.atom_slice(prot[0])] == 'H'], 4)})}, "
      f"water {np.round(m[wat[0]], 4).tolist() if wat else '-'}, lightest heavy atom {m[el != 'H'].min():.3f} amu", flush=True)
t1 = time.time()
if a.minimize > 0:
    print("minimise:", sim.minimize(a.minimize), flush=True)
sim._advance(500)
print(f"minimise + compile + 500 steps {time.time() - t1:.1f} s", flush=True)
t0 = time.time()
done = 0
while done < a.steps:
    sim._advance(500)
    done += 500
el_t = time.time() - t0
o = sim.observables()
print(f"{sim.sys.n} atoms, dt {a.dt * 1000:g} fs, {a.precision}: {el_t / done * 1e3:.3f} ms/step, "
      f"{done * a.dt / 1000 / el_t * 86400:.1f} ns/day; T {o['temp_K']:.1f} K, CG iters {o['cg_mean']:.2f} mean (max {o['cg_iter_max']}), "
      f"shake {o['shake_err']:.1e}", flush=True)
mem = jax.devices()[0].memory_stats() or {}
if "peak_bytes_in_use" in mem:
    print(f"GPU memory peak {mem['peak_bytes_in_use'] / 2**30:.2f} GiB", flush=True)

nblk = max(1, int(round(a.sample_ps / a.dt)))
if a.equil_ps > 0:
    t0 = time.time()
    for _ in range(int(round(a.equil_ps / a.sample_ps))):
        sim._advance(nblk)
    o = sim.observables()
    print(f"equilibrated {a.equil_ps:g} ps ({time.time() - t0:.0f} s): T {o['temp_K']:.1f} K, epot {o['epot']:.1f}", flush=True)
if a.prod_ps > 0:
    cols = ("time_ps", "epot", "temp_K", "econs", "shake_err", "cg_iter_max")
    rows = []
    cg0, s0 = float(sim.state.cg_total), int(sim.state.step)
    t0 = time.time()
    for _ in range(int(round(a.prod_ps / a.sample_ps))):
        sim._advance(nblk)
        o = sim.observables()
        rows.append([o[c] for c in cols])
    wall = time.time() - t0
    X = np.array(rows, float)
    t, U, T, E = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    nb = 10
    n = len(U) // nb * nb
    Ub = U[len(U) - n:].reshape(nb, -1).mean(1)
    kT = KB * a.temp
    drift = np.polyfit(t, E, 1)[0] * 1000.0 / sim.integ.dof / kT
    steps = int(sim.state.step) - s0
    print(f"production {a.prod_ps:g} ps, dt {a.dt * 1000:g} fs, H masses protein {a.hmr} / water "
          f"{a.hmr if a.hmr_water is None else a.hmr_water}: <U> {U.mean():.1f} +- {Ub.std(ddof=1) / np.sqrt(nb):.1f} kJ/mol "
          f"(std {U.std():.1f}), <T> {T.mean():.2f} K (min {T.min():.1f}, max {T.max():.1f}), shake max {X[:, 4].max():.1e}, "
          f"econs drift {drift:+.4f} kT/ns/dof, CG {(float(sim.state.cg_total) - cg0) / steps:.2f} mean "
          f"(max {int(X[:, 5].max())}), {steps * a.dt / 1000 / wall * 86400:.1f} ns/day", flush=True)
    if a.save:
        np.savez(a.save + "_samples.npz", **{c: X[:, i] for i, c in enumerate(cols)}, dof=sim.integ.dof)
if a.save:
    sim.save(a.save)
