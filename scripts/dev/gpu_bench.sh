#!/bin/bash
# Hot-path GPU speed check for the clean-up (docs/api_design.md, phase verification).  Runs a fixed
# set of scripts/benchmarks/bench_md.py cases twice (alternating rounds) on one GPU and prints ms/step, ns/day
# and CG iterations.  Run it on a GPU node, e.g.
#   nohup bash scripts/dev/gpu_bench.sh > bench.log 2>&1 &
# Compare with the baseline in docs/api_design.md (recorded on master e72c57c).  When a phase renames
# bench_md.py options (units in the option names since P9), update the calls below in the same commit.
C=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
# python environment: conda.sh from $PGM_CONDA_SH (default: ~/miniconda3's), environment $PGM_CONDA_ENV
source "${PGM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" && conda activate "${PGM_CONDA_ENV:-pgmjax}"
cd "$C" || exit 1
export PYTHONPATH="$C"  # the package of this tree (docs: README, Install)
echo "# $(git rev-parse --short HEAD) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
b() { echo "### $*"; python scripts/benchmarks/bench_md.py "$@" 2>&1 | grep -E "ms/step" | sed 's/.*): //'; }
for r in 1 2; do
  echo "## round $r"
  b --steps 5000                                                       # rigid 512, mixed, Langevin 1/ps
  b --replicate 2 --steps 3000                                         # rigid 4096
  b --steps 5000 --engine constraints --dt-fs 2                         # atoms + SHAKE/RATTLE 512
  b --replicate 2 --steps 3000 --engine constraints --dt-fs 2 --thermostat bussi
  b --steps 5000 --iel 0scf --dt-fs 2 --thermostat bussi                # iEL/0-SCF
  b --steps 3000 --engine constraints --hmr-amu 4.0 --thermostat bussi --dt-fs 6 --mts 2   # r-RESPA
  b --steps 3000 --precision double                                    # float64
done
