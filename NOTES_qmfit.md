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
