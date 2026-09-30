#!/bin/bash
# Smoke runs of the main scripts on tiny inputs (a few MD steps each), on the CPU.  Checks that the
# command lines work end to end (options, file outputs, restarts, analysis of the outputs), not the
# physics.  Run from anywhere; outputs go to <repo>/runs/smoke/.  Prints "OK <name>" or
# "FAIL <name>" (with the tail of the log) for each run and the number of failures at the end.
#   bash scripts/dev/smoke_scripts.sh [REPO]        
# Needs the external data of pgm_jax/paths.py (pGM3P-25 box, the gvdw data set).
C=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$C" || exit 1
export PYTHONPATH="$C" JAX_PLATFORMS=cpu
O=runs/smoke
mkdir -p $O
read -r PRM RST < <(python -c "from pgm_jax.paths import pgm3p25_files as f; print(*f())")
fails=0
run() {  # run NAME COMMAND...: log to $O/NAME.log
  local name=$1; shift
  local t0=$SECONDS
  if "$@" > "$O/$name.log" 2>&1; then echo "OK   $name ($((SECONDS - t0)) s)"; else
    echo "FAIL $name ($((SECONDS - t0)) s)"; tail -5 "$O/$name.log" | sed 's/^/     /'; fails=$((fails + 1)); fi
}
run md          python scripts/md/run_md.py -p "$PRM" -c "$RST" -o $O/md --nsteps 20 --report-every 10 \
                  --traj-every 10 --checkpoint-every 20 --dipoles-every 2 --thermostat bussi --barostat mc \
                  --barostat-interval 10
run md_continue python scripts/md/run_md.py -p "$PRM" -c "$RST" -o $O/md2 --continue-from $O/md.chk --nsteps 10 \
                  --report-every 10
run md_nve      python -m pgm_jax.cli md -p "$PRM" -c "$RST" -o $O/md_nve --nsteps 10 --report-every 5 \
                  --thermostat none --barostat none --dt-fs 0.5
run dielectric  python -m pgm_jax.cli dielectric $O/md.dip --skip-ps 0 --blocks 2 --eps-inf 1.0
run traj_dip    python scripts/dielectric/trajectory_dipoles.py -o $O/td "$PRM" $O/md.nc
run water_diel  python scripts/dielectric/water_dielectric.py --model pgm -o $O/wd --time-ns 0.00002 \
                  --dipoles-every 5 --report-every 10
run ff_run      python scripts/dielectric/finite_field.py run --model p25 -o $O/ff --fields-V-nm 0.1 --zero 0 \
                  --npt-ps 0 --time-ns 0.00002 --sample-every 5 --report-every 10
run ff_analyse  python scripts/dielectric/finite_field.py analyse $O/ff.ffd --skip-ps 0
run bench_md    python scripts/benchmarks/bench_md.py --steps 20
run pimd_valid  python scripts/pimd/pimd_validate.py harmonic --beads 4 --time-ps 1 --samples 1 -o $O/pv
run pimd_water  python scripts/pimd/pimd_water.py run --beads 2 --time-ps 0.005 --classical-ps 0 --equil-ps 0 \
                  --report-ps 0.0025 -o $O/pimd
run solvation   python scripts/free_energy/solvation_free_energy.py run --model pgm -o $O/fe --lattice 4 \
                  --cutoff-nm 0.4 --nfft 16 16 16 --dt-fs 1 --npt-ps 0.004 --time-ns 0.000024 --discard-ps 0 --n-elec 2 --vdw 0.5,0 \
                  --sample-ps 0.002 --exchange-ps 0.004 --report-ps 0.004 --checkpoint-ps 0.008
run solv_anal   python scripts/free_energy/solvation_free_energy.py analyze $O/fe_fe.npz --discard-ps 0
run toy_bias    python scripts/bias/validate_toy.py dw metad --time-ns 0.05 -o $O/toy
echo "failures: $fails"
exit $fails
