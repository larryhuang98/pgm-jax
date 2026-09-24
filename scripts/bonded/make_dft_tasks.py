"""Write runs/bonded/dft_tasks.txt (one line per Slurm task: name key start end) for every frame
set that exists and has no DFT file yet, and the Slurm array script runs/bonded/dft.sh."""
import glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FR = os.path.join(ROOT, "data/bonded/frames")
DFT = os.path.join(ROOT, "data/bonded/dft")
submitted = set()
for p in glob.glob(os.path.join(ROOT, "runs/bonded/dft_tasks_*.txt")):
    submitted |= {ln.strip() for ln in open(p) if ln.strip()}
# RETRY=1: resubmit every missing chunk (e.g. OOM-killed tasks) except the lines in the file $EXCLUDE
# (tasks still running); ONLY=<molecule> restricts to one molecule; SKIPKEYS=a,b drops frame sets
retry = os.environ.get("RETRY")
running = {ln.strip() for ln in open(os.environ["EXCLUDE"]) if ln.strip()} if os.environ.get("EXCLUDE") else set()
tasks = []
for f in sorted(glob.glob(os.path.join(FR, "*.npz"))):
    base = os.path.basename(f)[:-4]
    if os.environ.get("ONLY") and not base.startswith(os.environ["ONLY"]):
        continue
    z = np.load(f)
    big = base.startswith("alanine_dipeptide")
    if base.endswith("_md"):
        name = base[:-3]
        sets = [(k, len(z[k])) for k in ("train500", "test298")]
    elif base.endswith("_scan"):
        name = base[:-5]
        sets = [(f"scan{t}", len(z[f"scan{t}"])) for t in range(len(z["torsions"]))]
    elif "_scan2d_" in base:
        name = base.split("_scan2d_")[0]
        sets = [("scan2d_" + base.split("_scan2d_")[1], len(z["X"]))]
    else:
        continue
    chunk = 6 if big else 25
    for key, n in sets:
        for lo in range(0, n, chunk):
            line = f"{name} {key} {lo} {min(lo + chunk, n)}"
            fresh = (key.startswith("scan") and os.environ.get("RESCAN")) or retry  # redone scans / retries: ignore history
            if key in os.environ.get("SKIPKEYS", "").split(",") or line in running:
                continue
            if not os.path.exists(os.path.join(DFT, f"{name}__{key}__{lo}.npz")) and (fresh or line not in submitted):
                tasks.append(line)
open(os.path.join(ROOT, "runs/bonded/dft_tasks.txt"), "w").write("\n".join(tasks) + "\n")
print(len(tasks), "tasks")
if tasks and len(sys.argv) > 1:
    open(os.path.join(ROOT, "runs/bonded/dft.sh"), "w").write(f"""#!/bin/bash
#SBATCH --job-name=pgmjax-dft
#SBATCH --partition={os.environ.get("PART", "cpu-long")}
#SBATCH --cpus-per-task=8
#SBATCH --mem={os.environ.get("MEM", "7")}G
#SBATCH --time=08:00:00
#SBATCH --array=0-{len(tasks) - 1}
#SBATCH --output={ROOT}/runs/bonded/slurm/dft_%A_%a.out
cd {ROOT}
export PSI_SCRATCH=/tmp/larry/pgmjax_dft_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}; mkdir -p $PSI_SCRATCH
export OMP_NUM_THREADS=8
~/miniconda3/envs/psi4/bin/python scripts/bonded/dft_labels.py $SLURM_ARRAY_TASK_ID --threads 8 --memory {os.environ.get("PMEM", "5")} --tasks {ROOT}/runs/bonded/dft_tasks_{sys.argv[1]}.txt
rm -rf $PSI_SCRATCH
""")
    os.rename(os.path.join(ROOT, "runs/bonded/dft_tasks.txt"), os.path.join(ROOT, f"runs/bonded/dft_tasks_{sys.argv[1]}.txt"))
