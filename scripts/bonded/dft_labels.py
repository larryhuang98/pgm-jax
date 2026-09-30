"""Compute DFT labels (energy, forces, dipole) for sampled frames of the bonded study.

wB97M-D3(BJ)/def2-TZVPPD (DF-SCF), the level of MACE-OFF's training data (SPICE), with psi4.  One
task = one line of the task file (scripts/bonded/make_dft_tasks.py): <name> <npz key> <start> <end>;
frames whose SCF fails are skipped (printed).  The output file of a task that is already done is
not recomputed.

Usage:

    <python with psi4> scripts/bonded/dft_labels.py TASK_INDEX --threads 8 --memory-GB 5
    python scripts/bonded/dft_labels.py --help

Inputs: the task file (--tasks), data/bonded/frames/<name>_{md,scan}.npz or <name>_<scan2d key>.npz,
data/bonded/molecules/<name>.json; PSI_SCRATCH (psi4's scratch and working directory, default /tmp).
Outputs: data/bonded/dft/<name>__<key>__<start>.npz (index, X [A], energy_Eh, gradient_Eh_bohr,
dipole_au, time_s, level); a printed summary.
Units: Eh, Eh/bohr, atomic units of dipole (psi4's), A.
Runtime: CPU, minutes per frame.  This script does not import pgm_jax (it runs in a psi4
environment); the repository root is found from the script's location.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # without pgm_jax


def load_task(tasks: str, task: int) -> tuple[str, str, int, int, np.ndarray, list[str], int]:
    """Return name, key, start, end, the frames (F, N, 3) [A], elements and charge of one task line."""
    with open(tasks) as fh:
        name, key, lo, hi = fh.read().split("\n")[task].split()
    lo, hi = int(lo), int(hi)
    src = "scan2d" if key.startswith("scan2d") else ("scan" if key.startswith("scan") else "md")
    if src == "scan2d":
        X = np.load(os.path.join(REPO, "data/bonded/frames", f"{name}_{key}.npz"))["X"]
    else:
        X = np.load(os.path.join(REPO, "data/bonded/frames", f"{name}_{src}.npz"))[key]
    with open(os.path.join(REPO, "data/bonded/molecules", f"{name}.json")) as fh:
        mol_info = json.load(fh)
    return name, key, lo, hi, X, mol_info["elements"], mol_info["charge"]


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and label the frames of one task (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", type=int, help="line of the task file (the Slurm array index)")
    ap.add_argument("--threads", type=int, default=8, help="psi4 threads")
    ap.add_argument("--memory-GB", type=float, default=12, help="psi4 memory [GB]")
    ap.add_argument("--tasks", default=os.path.join(REPO, "runs/bonded/dft_tasks.txt"), help="task file")
    a = ap.parse_args(argv)
    name, key, lo, hi, X, el, charge = load_task(a.tasks, a.task)
    out_dir = os.path.join(REPO, "data/bonded/dft")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}__{key}__{lo}.npz")
    if os.path.exists(out):
        return
    os.chdir(os.environ.get("PSI_SCRATCH", "/tmp"))  # psi4 leaves psi.<pid>.clean files in the cwd
    import psi4  # imported after the chdir: psi4 writes psi.<pid>.clean files into the cwd

    psi4.set_memory(f"{a.memory_GB} GB")
    psi4.set_num_threads(a.threads)
    psi4.core.set_output_file(os.path.join(os.environ.get("PSI_SCRATCH", "/tmp"), f"psi4_{os.getpid()}.out"), False)
    psi4.set_options(
        {
            "basis": "def2-tzvppd",
            "scf_type": "df",
            "d_convergence": 1e-8,
            "dft_spherical_points": 590,
            "dft_radial_points": 99,
        }
    )
    E, G, D, idx, T = [], [], [], [], []
    for i in range(lo, min(hi, len(X))):
        t0 = time.time()
        psi4.core.clean()
        lines = [f"{charge} 1"] + [f"{e} {x:.10f} {y:.10f} {z:.10f}" for e, (x, y, z) in zip(el, X[i])]
        mol = psi4.geometry("\n".join(lines + ["symmetry c1", "no_reorient", "no_com"]))
        try:
            g, wfn = psi4.gradient("wb97m-d3bj", molecule=mol, return_wfn=True)
        except Exception as exc:  # SCF failure: record and move on
            print("frame", i, "failed", repr(exc)[:120], flush=True)
            continue
        psi4.oeprop(wfn, "DIPOLE")
        dip = wfn.variable("SCF DIPOLE") if wfn.has_variable("SCF DIPOLE") else wfn.variable("CURRENT DIPOLE")
        E.append(wfn.energy())
        G.append(np.array(g))
        D.append(np.array(dip).ravel())
        idx.append(i)
        T.append(time.time() - t0)
    np.savez(
        out,
        name=name,
        key=key,
        index=np.array(idx),
        X=X[np.array(idx, int)],
        energy_Eh=np.array(E),
        gradient_Eh_bohr=np.array(G),
        dipole_au=np.array(D),
        time_s=np.array(T),
        level="wB97M-D3(BJ)/def2-TZVPPD",
    )
    os.remove(os.path.join(os.environ.get("PSI_SCRATCH", "/tmp"), f"psi4_{os.getpid()}.out"))
    print(name, key, lo, hi, "done", len(idx), "frames, mean", np.mean(T) if T else 0, "s", flush=True)


if __name__ == "__main__":
    main()
