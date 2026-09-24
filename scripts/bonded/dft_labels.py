"""DFT labels (energy, forces, dipole) for sampled frames: wB97M-D3(BJ)/def2-TZVPPD, the level
of MACE-OFF's training data (SPICE).  One task = one line of runs/bonded/dft_tasks.txt:
    <name> <npz key> <start> <end>
    python scripts/bonded/dft_labels.py TASK_INDEX --threads 8
Writes data/bonded/dft/<name>__<key>__<start>.npz; skips frames already done."""
import argparse, json, os, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("task", type=int)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--memory", type=float, default=12)
ap.add_argument("--tasks", default=os.path.join(ROOT, "runs/bonded/dft_tasks.txt"))
a = ap.parse_args()
name, key, lo, hi = open(a.tasks).read().split("\n")[a.task].split()
lo, hi = int(lo), int(hi)
out_dir = os.path.join(ROOT, "data/bonded/dft")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, f"{name}__{key}__{lo}.npz")
if os.path.exists(out):
    raise SystemExit(0)
src = "scan2d" if key.startswith("scan2d") else ("scan" if key.startswith("scan") else "md")
if src == "scan2d":
    fn = os.path.join(ROOT, "data/bonded/frames", f"{name}_{key}.npz"); X = np.load(fn)["X"]
else:
    X = np.load(os.path.join(ROOT, "data/bonded/frames", f"{name}_{src}.npz"))[key]
mol_info = json.load(open(os.path.join(ROOT, "data/bonded/molecules", f"{name}.json")))
el, charge = mol_info["elements"], mol_info["charge"]

os.chdir(os.environ.get("PSI_SCRATCH", "/tmp"))            # psi4 leaves psi.<pid>.clean files in the cwd
import psi4  # noqa: E402
psi4.set_memory(f"{a.memory} GB"); psi4.set_num_threads(a.threads)
psi4.core.set_output_file(f"/tmp/larry_psi4_{os.getpid()}.out", False)
psi4.set_options({"basis": "def2-tzvppd", "scf_type": "df", "d_convergence": 1e-8, "dft_spherical_points": 590,
                  "dft_radial_points": 99})
E, G, D, idx, T = [], [], [], [], []
for i in range(lo, min(hi, len(X))):
    t0 = time.time()
    psi4.core.clean()
    lines = [f"{charge} 1"] + [f"{e} {x:.10f} {y:.10f} {z:.10f}" for e, (x, y, z) in zip(el, X[i])]
    mol = psi4.geometry("\n".join(lines + ["symmetry c1", "no_reorient", "no_com"]))
    try:
        g, wfn = psi4.gradient("wb97m-d3bj", molecule=mol, return_wfn=True)
    except Exception as exc:                        # SCF failure: record and move on
        print("frame", i, "failed", repr(exc)[:120], flush=True)
        continue
    psi4.oeprop(wfn, "DIPOLE")
    dip = wfn.variable("SCF DIPOLE") if wfn.has_variable("SCF DIPOLE") else wfn.variable("CURRENT DIPOLE")
    E.append(wfn.energy()); G.append(np.array(g)); D.append(np.array(dip).ravel()); idx.append(i); T.append(time.time() - t0)
np.savez(out, name=name, key=key, index=np.array(idx), X=X[np.array(idx, int)], energy_Eh=np.array(E),
         gradient_Eh_bohr=np.array(G), dipole_au=np.array(D), time_s=np.array(T),
         level="wB97M-D3(BJ)/def2-TZVPPD")
os.system(f"rm -f /tmp/larry_psi4_{os.getpid()}.out")
print(name, key, lo, hi, "done", len(idx), "frames, mean", np.mean(T) if T else 0, "s", flush=True)
