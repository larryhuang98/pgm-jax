# NOTES: fitting pGM parameters to QM cluster data (item 9, branch qmfit)

Running log so the work can be resumed.

## Layout
- scripts/qmfit/smith_opt.py            symmetry-constrained MP2/aDZ dimer stationary structures -> ~/project/qmdata/smith/*.json
- scripts/qmfit/build_water_clusters.py geometries (rigid pGM3P-25 monomers) -> data/qm/water_geoms.json
- scripts/qmfit/psi4_clusters.py        psi4 worker (dynamic queue, mkdir claims) -> ~/project/qmdata/water/results_w*.jsonl
- ~/project/qmdata/qm_submit.sh OUT WORKER THREADS MIN PARTITION [args]   (sbatch one worker)
- data/qm/water27_raw.json              WATER27 (GMTKN55) water cluster geometries, as downloaded

## Log
- 2026-09-28 start. Model geometries: p25 r_OH 0.9745 A, HOH 103.64; base 0.9572 / 104.49.
- Smith-type structures: 7 started, C2_cyclic collapsed onto C2h_cyclic (same energy) -> 6 unique.
  Cs_open needed OPT_COORDS=cartesian (redundant internals broke the Cs symmetry).
  MP2/aDZ relative energies (kcal/mol): Cs_open 0, Cs_planar 0.66, Ci_cyclic 0.88, C2h_cyclic 1.04,
  C2v_bifurcated 1.97, C2v_planar_bifurcated 2.93.
- Timings (16 threads, cpu-2-x): SAPT0/jun-DZ 1.2 s, CP MP2/aTZ 2.4 s, CP MP2/aQZ 9.8 s,
  CP DF-CCSD(T)/aTZ 55 s, CP MP2/aTZ gradient 6.4 s per dimer; MBE(3) MP2/aTZ trimer 12 s,
  pentamer 129 s, hexamer 332 s.
- psi4 options persist between jobs: psi4.core.clean_options() at the start of each job (qc_module
  fnocc from the CCSD(T) job broke the following MP2 jobs).
- Production QM: 844 records (757 dimers incl. 458 pairs of clusters), 10 workers x 16 threads,
  cpu-short, out ~/project/qmdata/water.
- QM done (~1.5 h wall, 12 workers x 16 threads, cpu-long after vacating cpu-short: cpu_run.sh jobs of
  the other agents ask for the whole node memory, so any job of mine there blocked them).
  3350 tasks, 274 core-hours (CCSD(T)/aTZ 202, MP2 43, MBE 18, grad 7, SAPT0 4).
  Monomer props job crashed in production (psi4 array variables); rerun separately
  (~/project/qmdata/mono_test) and copied as results_m0.jsonl. CCSD/aTZ: 1.868 D, 1.439 A^3.
- Water dimer minimum (rigid p25 monomers at the MP2/aDZ Smith Cs geometry): E.ref -5.085 kcal/mol.
- Baselines: p25 test E_int RMSE 1.50 (MAE 0.63), base 2.61 (0.92); p25 3-body/MP2 ratio 0.6-0.76,
  base 0.26-0.45 (large radii damp induction); p25 gas dipole 1.46 D (QM 1.87).
- final fits: runs/qmfit/final.sh -> runs/qmfit/*.json, data/qm/fits/*.json; report shows test E_int,
  components, 3-body, dimer, Smith relative energies, hexamer order, forces, monomer.
- Fits (runs/qmfit/final.sh, combo.sh, probe.sh; reports copied to validation/qmfit/): see
  docs/qmfit.md table. Recommended: all_total (LJ, pmemd compatible; test RMSE 0.75, hexamer order
  right) and rec_gvdw (GVDW O+H; test 0.57, 3-body 0.18, forces 0.49; cage 0.27 below prism).
  LJ fits with SAPT weights push the O LJ to R* 0.3 nm / eps ~1e-3 (not recommended).
- tests/test_qmfit.py: 8 passed (2.7 min, 8 cores). Full suite (24 cores, cpu-long): 215 passed in 29 min
  (run before the 8th qmfit test was added; that one passed separately) -> 216 tests.
