"""Speed of a solvated protein in the pgm_jax MD engine: pGM (placeholder electrostatics unless a
residue library is given) + Amber-form bonded terms and CMAP from the prmtop, rigid water by
constraints, X-H constraints, hydrogen mass repartitioning.

    python scripts/protein/bench_protein.py runs/protein/ubq.prmtop runs/protein/ubq.inpcrd --dt 0.002 --steps 2000
    python scripts/protein/bench_protein.py ... --elec-cut 0.7 --thermostat bussi   # LJ 0.9 nm, electrostatics 0.7
"""
import argparse
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pgm_jax.md.flexible import FlexibleSimulation  # noqa: E402
from pgm_jax.md.forcefield import DSUM_TOL, MDSettings, elec_cutoff_settings  # noqa: E402
from pgm_jax.protein import ResidueLibrary, amber_template, load_amber  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("inpcrd")
ap.add_argument("--library", default=None)
ap.add_argument("--dt", type=float, default=0.002)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--cut", type=float, default=0.9, help="van der Waals (and default electrostatics) cutoff (nm)")
ap.add_argument("--elec-cut", type=float, default=None, help="real-space electrostatics cutoff (nm); --beta and "
                "--spacing then default to elec_cutoff_settings(elec_cut, dsum_tol)")
ap.add_argument("--dsum-tol", type=float, default=DSUM_TOL, help="direct-sum tolerance for --elec-cut (Amber convention)")
ap.add_argument("--pme-exponent", type=float, default=1.6, help="grid rule for --elec-cut: spacing 0.08 (4 / beta)^exponent "
                "(1.6: the default's accuracy; 1: beta x spacing fixed, faster, less accurate)")
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--precision", default="mixed")
ap.add_argument("--hmr", type=float, default=3.024)
ap.add_argument("--local-niter", type=int, default=0, help="inner CG steps of the short-range preconditioner (0: Jacobi)")
ap.add_argument("--local-cut", type=float, default=0.3, help="preconditioner range (nm)")
ap.add_argument("--predictor", default="mu4")
ap.add_argument("--beta", type=float, default=None, help="Ewald coefficient (nm^-1); default 4.0 (or from --elec-cut)")
ap.add_argument("--spacing", type=float, default=None, help="PME grid spacing (nm); default 0.08 (or from --elec-cut)")
ap.add_argument("--thermostat", default="langevin", help="langevin | bussi | gle")
ap.add_argument("--tau", type=float, default=1.0, help="Bussi time constant (ps)")
ap.add_argument("--gamma", type=float, default=1.0, help="Langevin friction (1/ps)")
a = ap.parse_args()
lib = ResidueLibrary.load(a.library) if a.library else "placeholder"
ew = {"ewald_beta": 4.0, "pme_spacing": 0.08} if a.elec_cut is None else elec_cutoff_settings(a.elec_cut, a.dsum_tol, a.pme_exponent)
beta = ew["ewald_beta"] if a.beta is None else a.beta
spacing = ew["pme_spacing"] if a.spacing is None else a.spacing
t0 = time.time()
asys = load_amber(a.prmtop, a.inpcrd, electrostatics=lib)
prot = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
tpl = {k: amber_template(asys.molecules[k], a.prmtop) for k in prot}
st = MDSettings(cutoff=a.cut, skin=0.1, dipole_tol=a.tol, precision=a.precision, local_niter=a.local_niter,
                local_cut=a.local_cut, predictor=a.predictor, ewald_beta=beta, pme_spacing=spacing,
                elec_cutoff=a.elec_cut)
sim = FlexibleSimulation(asys.system(), asys.templates(tpl), asys.system_positions(), asys.box, st, dt=a.dt,
                         ensemble="nvt", constraints="h-bonds", hmr=a.hmr, log=sys.stdout,
                         thermostat=a.thermostat, tau_t=a.tau, gamma=a.gamma)
print(f"setup {time.time() - t0:.1f} s: {sim.sys.n} atoms, {len(prot)} protein chain(s) "
      f"({sum(asys.molecules[k].n for k in prot)} atoms), {sim.topology.n_group} groups, "
      f"special width {sim.topology.special.shape[1]}, rows {sim.ff.mc} (electrostatic {sim.ff.mc_e or sim.ff.mc}); "
      f"{st.describe_cutoffs()}, beta {beta:.4f} /nm, spacing {spacing:.4f} nm", flush=True)
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
      f"{done * a.dt / 1000 / el * 86400:.1f} ns/day; T {o['temp_K']:.1f} K, CG iters {o['cg_mean']:.2f} mean (max {o['cg_iter_max']}), "
      f"rows {sim.ff.mc} (electrostatic {sim.ff.mc_e or sim.ff.mc}), "
      f"shake {o['shake_err']:.1e}", flush=True)
mem = jax.devices()[0].memory_stats() or {}
if "peak_bytes_in_use" in mem:
    print(f"GPU memory peak {mem['peak_bytes_in_use'] / 2**30:.2f} GiB", flush=True)
