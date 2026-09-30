"""Write the DFT task list of the bonded study and, with a tag, the Slurm array script.

runs/bonded/dft_tasks.txt gets one line per Slurm task (name key start end) for every frame set in
data/bonded/frames that has no DFT file (data/bonded/dft/<name>__<key>__<start>.npz) yet and was not
submitted before (runs/bonded/dft_tasks_*.txt); chunks of 25 frames (6 for the alanine dipeptide).
With a TAG, the list is renamed runs/bonded/dft_tasks_<TAG>.txt and runs/bonded/dft.sh is written:
a Slurm array (8 cores per task) running scripts/bonded/dft_labels.py on each line.

Usage:

    python scripts/bonded/make_dft_tasks.py                     # count the tasks
    python scripts/bonded/make_dft_tasks.py r3 --partition cpu-long && sbatch runs/bonded/dft.sh
    python scripts/bonded/make_dft_tasks.py r4 --retry --exclude running.txt   # resubmit missing chunks
    python scripts/bonded/make_dft_tasks.py --help

Inputs: data/bonded/frames/*.npz, data/bonded/dft/, runs/bonded/dft_tasks_*.txt.
Outputs: runs/bonded/dft_tasks[_<TAG>].txt, runs/bonded/dft.sh; the number of tasks is printed.
Units: --mem-GB and --psi4-mem-GB GB.
Runtime: seconds.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from pgm_jax.paths import REPO, repo_path

FR = repo_path("data", "bonded", "frames")
DFT = repo_path("data", "bonded", "dft")


def frame_sets(base: str, z: np.lib.npyio.NpzFile) -> tuple[str | None, list[tuple[str, int]]]:
    """Return the molecule name and the (key, number of frames) of the sets of one frame file (None: not a set)."""
    if base.endswith("_md"):
        return base[:-3], [(k, len(z[k])) for k in ("train500", "test298")]
    if base.endswith("_scan"):
        return base[:-5], [(f"scan{t}", len(z[f"scan{t}"])) for t in range(len(z["torsions"]))]
    if "_scan2d_" in base:
        return base.split("_scan2d_")[0], [("scan2d_" + base.split("_scan2d_")[1], len(z["X"]))]
    return None, []


def tasks_to_do(a: argparse.Namespace) -> list[str]:
    """Return the task lines (name key start end) without a DFT file, not submitted, not running or skipped."""
    submitted = set()
    for p in glob.glob(repo_path("runs", "bonded", "dft_tasks_*.txt")):
        with open(p) as fh:
            submitted |= {ln.strip() for ln in fh if ln.strip()}
    running = set()
    if a.exclude:
        with open(a.exclude) as fh:
            running = {ln.strip() for ln in fh if ln.strip()}
    tasks = []
    for f in sorted(glob.glob(os.path.join(FR, "*.npz"))):
        base = os.path.basename(f)[:-4]
        if a.only and not base.startswith(a.only):
            continue
        z = np.load(f)
        name, sets = frame_sets(base, z)
        if name is None:
            continue
        chunk = 6 if base.startswith("alanine_dipeptide") else 25
        for key, n in sets:
            for lo in range(0, n, chunk):
                line = f"{name} {key} {lo} {min(lo + chunk, n)}"
                fresh = (key.startswith("scan") and a.rescan) or a.retry  # redone scans / retries: ignore history
                if key in a.skip_keys.split(",") or line in running:
                    continue
                if not os.path.exists(os.path.join(DFT, f"{name}__{key}__{lo}.npz")) and (
                    fresh or line not in submitted
                ):
                    tasks.append(line)
    return tasks


def write_slurm(a: argparse.Namespace, ntasks: int) -> None:
    """Write runs/bonded/dft.sh, the Slurm array of dft_labels.py over runs/bonded/dft_tasks_<tag>.txt."""
    with open(repo_path("runs", "bonded", "dft.sh"), "w") as fh:
        fh.write(f"""#!/bin/bash
#SBATCH --job-name=pgmjax-dft
#SBATCH --partition={a.partition}
#SBATCH --cpus-per-task=8
#SBATCH --mem={a.mem_GB:g}G
#SBATCH --time=08:00:00
#SBATCH --array=0-{ntasks - 1}
#SBATCH --output={REPO}/runs/bonded/slurm/dft_%A_%a.out
cd {REPO}
export PSI_SCRATCH=/tmp/$USER/pgmjax_dft_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}; mkdir -p $PSI_SCRATCH
export OMP_NUM_THREADS=8
{a.psi4_python} scripts/bonded/dft_labels.py $SLURM_ARRAY_TASK_ID --threads 8 \\
    --memory-GB {a.psi4_mem_GB:g} --tasks {REPO}/runs/bonded/dft_tasks_{a.tag}.txt
rm -rf $PSI_SCRATCH
""")


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and write the task list and the Slurm script (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", nargs="?", default=None, help="name of this submission (writes dft.sh)")
    ap.add_argument("--retry", action="store_true", help="resubmit every missing chunk (e.g. OOM-killed tasks)")
    ap.add_argument("--exclude", default=None, help="file of task lines still running (not resubmitted)")
    ap.add_argument("--only", default=None, help="only frame files starting with this molecule name")
    ap.add_argument("--skip-keys", default="", help="comma-separated frame sets to drop")
    ap.add_argument("--rescan", action="store_true", help="redo the scans (ignore the submission history)")
    ap.add_argument("--partition", default="cpu-long", help="Slurm partition")
    ap.add_argument("--mem-GB", type=float, default=7, help="Slurm memory per task [GB]")
    ap.add_argument("--psi4-mem-GB", type=float, default=5, help="psi4 memory [GB]")
    ap.add_argument(
        "--psi4-python",
        default=os.environ.get("PGM_PSI4_PYTHON", "python"),
        help="python with psi4 (default: $PGM_PSI4_PYTHON or python)",
    )
    a = ap.parse_args(argv)
    tasks = tasks_to_do(a)
    path = repo_path("runs", "bonded", "dft_tasks.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(tasks) + "\n")
    print(len(tasks), "tasks")
    if tasks and a.tag:
        write_slurm(a, len(tasks))
        os.rename(path, repo_path("runs", "bonded", f"dft_tasks_{a.tag}.txt"))


if __name__ == "__main__":
    main()
