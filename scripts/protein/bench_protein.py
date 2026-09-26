"""Speed (and, with --prod-ps, stability and configurational accuracy) of a solvated protein in the
pgm_jax MD engine: pGM (placeholder electrostatics unless a residue library is given) +
Amber-form bonded terms and CMAP from the prmtop, rigid water by constraints, X-H constraints,
hydrogen mass repartitioning (--hmr for the protein, --hmr-water for water).

    python scripts/protein/bench_protein.py runs/protein/ubq.prmtop runs/protein/ubq.inpcrd --dt 0.002 --steps 2000
    python scripts/protein/bench_protein.py ... --elec-cut 0.7 --thermostat bussi   # LJ 0.9 nm, electrostatics 0.7
    # water H 4.0 amu, protein H 3.024, 4 fs, Bussi; 20 ps equilibration, 200 ps sampled every 0.5 ps
    python scripts/protein/bench_protein.py ubq.prmtop ubq.inpcrd --dt 0.004 --hmr-water 4.0 --thermostat bussi \\
        --equil-ps 20 --prod-ps 200 --coords equilibrated.rst7 --minimize 0 --save runs/ubq_4fs
    # multiple time stepping: 8 fs outer step, short-range nonbonded + bonded forces every 4 fs (docs/mts.md)
    python scripts/protein/bench_protein.py ... --dt 0.008 --mts 2 --elec-cut 0.7 --thermostat bussi

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
from pgm_jax.md.forcefield import DSUM_TOL, MDSettings, elec_cutoff_settings  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.md.io import box_from_cell, read_coordinates  # noqa: E402
from pgm_jax.md.mts import add_mts_arguments, mts_from_args, mts_stats  # noqa: E402
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
ap.add_argument("--hmr", type=float, default=3.024, help="hydrogen mass (amu) of the protein (and of water "
                "unless --hmr-water is given)")
ap.add_argument("--hmr-water", type=float, default=None, help="hydrogen mass (amu) of water (e.g. 4.0)")
ap.add_argument("--local-niter", type=int, default=0, help="inner CG steps of the short-range preconditioner (0: Jacobi)")
ap.add_argument("--local-cut", type=float, default=0.3, help="preconditioner range (nm)")
ap.add_argument("--predictor", default="mu4")
ap.add_argument("--beta", type=float, default=None, help="Ewald coefficient (nm^-1); default 4.0 (or from --elec-cut)")
ap.add_argument("--spacing", type=float, default=None, help="PME grid spacing (nm); default 0.08 (or from --elec-cut)")
ap.add_argument("--thermostat", default="langevin", help="langevin | bussi | gle")
ap.add_argument("--tau", type=float, default=1.0, help="Bussi time constant (ps)")
ap.add_argument("--gamma", type=float, default=1.0, help="Langevin friction (1/ps)")
ap.add_argument("--temp", type=float, default=298.0, help="temperature (K)")
ap.add_argument("--ensemble", default="nvt", choices=["nvt", "npt"], help="npt: Monte Carlo barostat at 1 bar")
ap.add_argument("--barostat-interval", type=int, default=100, help="steps between Monte Carlo volume moves (npt)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--coords", default=None, help="start from these coordinates and box (Amber restart in the "
                "system's atom order, e.g. the .rst7 of --save) instead of the inpcrd; velocities are redrawn")
ap.add_argument("--minimize", type=int, default=300, help="steepest-descent steps before dynamics (0: none)")
ap.add_argument("--equil-ps", type=float, default=0.0, help="equilibration after the speed test (ps)")
ap.add_argument("--prod-ps", type=float, default=0.0, help="sampled production after equilibration (ps)")
ap.add_argument("--sample-ps", type=float, default=0.5, help="sampling interval of the production (ps)")
ap.add_argument("--save", default=None, help="prefix: restart + checkpoint at the end, samples as .npz")
ap.add_argument("--traj-ps", type=float, default=0.0, help="production frames every this many ps: CA RMSD to the first "
                "frame and radius of gyration of the heavy atoms (and, with --save, the trajectory prefix.nc); 0: none")
add_mts_arguments(ap)
a = ap.parse_args()
ew = {"ewald_beta": 4.0, "pme_spacing": 0.08} if a.elec_cut is None else elec_cutoff_settings(a.elec_cut, a.dsum_tol, a.pme_exponent)
beta = ew["ewald_beta"] if a.beta is None else a.beta
spacing = ew["pme_spacing"] if a.spacing is None else a.spacing
lib = ResidueLibrary.load(a.library) if a.library else "placeholder"
t0 = time.time()
asys = load_amber(a.prmtop, a.inpcrd, electrostatics=lib)
prot = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
tpl = {k: amber_template(asys.molecules[k], a.prmtop) for k in prot}
st = MDSettings(cutoff=a.cut, skin=0.1, dipole_tol=a.tol, precision=a.precision, local_niter=a.local_niter,
                local_cut=a.local_cut, predictor=a.predictor, ewald_beta=beta, pme_spacing=spacing,
                elec_cutoff=a.elec_cut)
hmr = asys.hmr({"protein": a.hmr, "other": a.hmr, "water": a.hmr if a.hmr_water is None else a.hmr_water, "ion": None})
pos, H = asys.system_positions(), asys.box
if a.coords:
    xyz, _, cell = read_coordinates(a.coords)
    pos, H = xyz * 0.1, box_from_cell(*cell) * 0.1
sim = FlexibleSimulation(asys.system(), asys.templates(tpl), pos, H, st, dt=a.dt, temperature=a.temp,
                         ensemble=a.ensemble, barostat_interval=a.barostat_interval, constraints="h-bonds", hmr=hmr,
                         log=sys.stdout,
                         thermostat=a.thermostat, tau_t=a.tau, gamma=a.gamma, seed=a.seed, mts=mts_from_args(a))
m = np.asarray(sim.flex.masses)
el = np.array(sim.sys.elements)
wat = [sim.sys.atom_slice(k) for k, mm in enumerate(asys.molecules) if mm.kind == "water"]
print(f"setup {time.time() - t0:.1f} s: {sim.sys.n} atoms, {len(prot)} protein chain(s) "
      f"({sum(asys.molecules[k].n for k in prot)} atoms), {sim.topology.n_group} groups, "
      f"special width {sim.topology.special.shape[1]}, rows {sim.ff.mc} (electrostatic {sim.ff.mc_e or sim.ff.mc}); "
      f"{st.describe_cutoffs()}, beta {beta:.4f} /nm, spacing {spacing:.4f} nm; hydrogen masses: protein "
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
      f"rows {sim.ff.mc} (electrostatic {sim.ff.mc_e or sim.ff.mc}), "
      f"shake {o['shake_err']:.1e}{'; ' + str(mts_stats(sim)) if a.mts else ''}", flush=True)
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
    cols = ("time_ps", "epot", "temp_K", "econs", "shake_err", "cg_iter_max", "temp_com", "temp_internal", "density_g_cm3")
    rows = []
    cg0, s0 = float(sim.state.cg_total), int(sim.state.step)
    tfile, frames = None, []
    if a.traj_ps > 0:
        tevery = max(1, int(round(a.traj_ps / a.sample_ps)))
        ca, heavy = asys.select("ca"), asys.select("heavy")
        if a.save:
            from pgm_jax.md.io import NetCDFTrajectory
            tfile = NetCDFTrajectory(a.save + ".nc", sim.sys.n)
    t0 = time.time()
    for i in range(int(round(a.prod_ps / a.sample_ps))):
        sim._advance(nblk)
        o = sim.observables()
        rows.append([o[c] for c in cols])
        if a.traj_ps > 0 and (i + 1) % tevery == 0:
            x = sim.positions_nm()
            frames.append((x[ca] * 10.0, x[heavy] * 10.0))
            if tfile is not None:
                tfile.write(sim.time_ps, x * 10.0, np.asarray(sim.state.box) * 10.0)
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
          f"(std {U.std():.1f}), <T> {T.mean():.2f} K (min {T.min():.1f}, max {T.max():.1f}; centre of mass "
          f"{X[:, 6].mean():.2f}, internal {X[:, 7].mean():.2f}), density {X[:, 8].mean():.4f} +- "
          f"{X[len(X) - n:, 8].reshape(nb, -1).mean(1).std(ddof=1) / np.sqrt(nb):.4f} g/cm3, shake max {X[:, 4].max():.1e}, "
          f"econs drift {drift:+.4f} kT/ns/dof, CG {(float(sim.state.cg_total) - cg0) / steps:.2f} mean "
          f"(max {int(X[:, 5].max())}), {steps * a.dt / 1000 / wall * 86400:.1f} ns/day"
          f"{'; ' + str(mts_stats(sim)) if a.mts else ''}", flush=True)
    if frames:                          # molecules are whole, so the protein needs no unwrapping
        ref = frames[0][0] - frames[0][0].mean(0)
        rmsd, rg = [], []
        for xc, xh in frames:
            y = xc - xc.mean(0)
            U, _, Vt = np.linalg.svd(y.T @ ref)
            R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            rmsd.append(np.sqrt(np.mean(np.sum((y @ R - ref) ** 2, 1))))
            h = xh - xh.mean(0)
            rg.append(np.sqrt(np.mean(np.sum(h * h, 1))))
        rmsd, rg, half = np.array(rmsd), np.array(rg), len(frames) // 2
        print(f"{len(frames)} frames: CA RMSD to the first (A) mean {rmsd.mean():.2f}, second half {rmsd[half:].mean():.2f}, "
              f"max {rmsd.max():.2f}; radius of gyration of the heavy atoms (A) {rg.mean():.2f} +- {rg.std():.2f} "
              f"(halves {rg[:half].mean():.2f} / {rg[half:].mean():.2f})", flush=True)
    if a.save:
        np.savez(a.save + "_samples.npz", **{c: X[:, i] for i, c in enumerate(cols)}, dof=sim.integ.dof)
if a.save:
    sim.save(a.save)
