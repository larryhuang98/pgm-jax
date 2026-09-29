"""QM reference data for water clusters with psi4: one worker of a dynamic work queue.

Tasks are (record, job) pairs from the geometry file (data/qm/water_geoms.json, written by
scripts/qm/build_water_clusters.py); workers claim tasks with an atomic mkdir on the shared file
system, so any number of workers (slurm array tasks) balance the load and a killed run is resumed
by starting workers again (a claimed task without a result line is retried after --stale-hours).
Results are appended as JSON lines to <out>/results_<worker>.jsonl (docs/qmfit.md).

Jobs (all counterpoise corrected in the basis of the whole record, frozen core, DF-SCF):
  sapt0                 SAPT0/jun-cc-pVDZ components (all psi4 SAPT* variables, Eh), dimers only
  mp2:<basis>           DF-MP2 energies of the dimer and of each monomer in the dimer basis
  ccsdt:<basis>         DF-CCSD(T) (fnocc) likewise (also the DF-MP2 energies of the same runs)
  mp2grad:<basis>       DF-MP2 gradients of the dimer and of each monomer in the dimer basis
  props:<method>:<basis>  dipole and static polarizability of a monomer (psi4.properties, a.u.)
  mbe:<method>:<basis>:<k>   energies of every subset of <= k molecules and of the whole cluster, all
                        in the cluster basis (the CP many-body expansion; interaction energy from the
                        full cluster and the monomers)

Usage:

    <python with psi4> scripts/qm/psi4_clusters.py GEOMS OUTDIR WORKER --threads 16 --memory-GB 30 [--only ids]
    python scripts/qm/psi4_clusters.py --help

Inputs: the geometry file (records with id, n, xyz_A, jobs, optional elements, atoms_per_mol).
Outputs: <out>/results_<worker>.jsonl (id, job, res or error, sec, threads), <out>/psi4_<worker>.out,
<out>/claims/; one printed line per task.
Units: energies in Eh, gradients in Eh/bohr, properties in atomic units (psi4's); geometries in A.
Runtime: CPU (psi4); from seconds (SAPT0 dimers) to hours (CCSD(T) of large clusters) per task.
Needs psi4 (imported in main; this script does not import pgm_jax).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import time

import numpy as np

SAPT_BASIS = "jun-cc-pvdz"


def mol_string(frags: list, active: set[int]) -> str:
    """Return a psi4 molecule string of the fragments, with the inactive ones as ghost atoms.

    Parameters
    ----------
    frags : list of (list of str, array (A, 3))
        Elements and coordinates [A] of each fragment.
    active : set of int
        Indices of the real fragments (the others are ghosts: their basis functions only).

    Returns
    -------
    str
        Fragments separated by "--", charge 0 multiplicity 1 each, c1 symmetry, fixed frame.
    """
    out = []
    for k, (el, xyz) in enumerate(frags):
        if k:
            out.append("--")
        out.append("0 1")
        for e, (x, y, z) in zip(el, xyz):
            name = e if k in active else f"Gh({e})"
            out.append(f"{name} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(out + ["units angstrom", "symmetry c1", "no_com", "no_reorient"])


def frags_of(rec: dict) -> list:
    """Return the fragments (elements, xyz [A]) of a record (default: water, O H H per molecule)."""
    X = rec["xyz_A"]
    el = rec.get("elements") or ["O", "H", "H"] * rec["n"]
    na = rec.get("atoms_per_mol", 3)
    return [(el[na * k : na * k + na], X[na * k : na * k + na]) for k in range(rec["n"])]


def run_job(psi4: object, rec: dict, job: str) -> dict:
    """Run one job (see the module docstring) on one record and return its results.

    Parameters
    ----------
    psi4 : module
        The imported psi4 module.
    rec : dict
        The geometry record.
    job : str
        Job specification, e.g. "mp2:aug-cc-pvtz".

    Returns
    -------
    dict
        sapt0 and props: psi4 variables by name; mp2, ccsdt, mp2grad, mbe: per subset of molecules
        ("0,1", "0", ...) the energies [Eh] (and gradients [Eh/bohr]).

    Raises
    ------
    ValueError
        An unknown job kind.
    """
    frags = frags_of(rec)
    n = len(frags)
    psi4.core.clean_options()  # options persist between jobs
    base = {"scf_type": "df", "freeze_core": True, "e_convergence": 1e-9, "d_convergence": 1e-9}
    parts = job.split(":")
    kind = parts[0]
    res = {}
    if kind == "sapt0":
        psi4.set_options(dict(base, basis=SAPT_BASIS))
        mol = psi4.geometry(mol_string(frags, set(range(n))))
        psi4.energy("sapt0", molecule=mol)
        res = {
            k: float(v)
            for k, v in psi4.core.variables().items()
            if "SAPT" in k and not hasattr(v, "np") and getattr(v, "ndim", 0) == 0
        }
    elif kind in ("mp2", "ccsdt"):
        basis = parts[1]
        opts = dict(base, basis=basis, mp2_type="df")
        if kind == "ccsdt":
            opts.update(cc_type="df", qc_module="fnocc")
        psi4.set_options(opts)
        method = "mp2" if kind == "mp2" else "ccsd(t)"
        for S in [tuple(range(n))] + [(k,) for k in range(n)]:
            mol = psi4.geometry(mol_string(frags, set(S)))
            psi4.energy(method, molecule=mol)
            r = {"scf": float(psi4.variable("SCF TOTAL ENERGY")), "mp2": float(psi4.variable("MP2 TOTAL ENERGY"))}
            if kind == "ccsdt":
                r["ccsd"] = float(psi4.variable("CCSD TOTAL ENERGY"))
                r["ccsdt"] = float(psi4.variable("CCSD(T) TOTAL ENERGY"))
            res[",".join(map(str, S))] = r
            psi4.core.clean()
    elif kind == "mp2grad":
        psi4.set_options(dict(base, basis=parts[1], mp2_type="df"))
        for S in [tuple(range(n))] + [(k,) for k in range(n)]:
            mol = psi4.geometry(mol_string(frags, set(S)))
            g = psi4.gradient("mp2", molecule=mol)
            res[",".join(map(str, S))] = {"mp2": float(psi4.variable("MP2 TOTAL ENERGY")), "grad": g.np.tolist()}
            psi4.core.clean()
    elif kind == "mbe":
        method, basis, kmax = parts[1], parts[2], int(parts[3])
        psi4.set_options(dict(base, basis=basis, mp2_type="df"))
        subsets = [S for k in range(1, min(kmax, n) + 1) for S in itertools.combinations(range(n), k)]
        if kmax < n:
            subsets.append(tuple(range(n)))
        for S in subsets:
            mol = psi4.geometry(mol_string(frags, set(S)))
            psi4.energy(method, molecule=mol)
            res[",".join(map(str, S))] = {
                "scf": float(psi4.variable("SCF TOTAL ENERGY")),
                "mp2": float(psi4.variable("MP2 TOTAL ENERGY")),
            }
            psi4.core.clean()
    elif kind == "props":  # monomer dipole and static polarizability
        method, basis = parts[1], parts[2]
        psi4.set_options({"basis": basis, "freeze_core": True, "e_convergence": 1e-10, "d_convergence": 1e-10})
        mol = psi4.geometry(mol_string(frags, set(range(n))))
        psi4.properties(method, properties=["dipole", "polarizability"], molecule=mol)
        res = {
            k: np.asarray(v.np if hasattr(v, "np") else v, float).tolist()
            for k, v in psi4.core.variables().items()
            if "DIPOLE" in k or "POLARIZABILITY" in k or "TOTAL ENERGY" in k
        }
    else:
        raise ValueError(job)
    return res


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and work through the tasks (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("geoms", help="geometry file (JSON with records)")
    ap.add_argument("out", help="output directory (shared by all workers)")
    ap.add_argument("worker", help="worker name (e.g. the slurm array index)")
    ap.add_argument("--threads", type=int, default=16, help="psi4 threads")
    ap.add_argument("--memory-GB", type=float, default=30.0, help="psi4 memory [GB]")
    ap.add_argument("--stale-hours", type=float, default=12.0, help="claims older than this are retried [h]")
    ap.add_argument("--only", default="", help="comma-separated record ids (default: all)")
    ap.add_argument("--quiet", action="store_true", help="psi4.core.be_quiet()")
    a = ap.parse_args(argv)
    import psi4

    recs = json.load(open(a.geoms))["records"]
    tasks = [(r, j) for r in recs for j in r["jobs"] if not a.only or r["id"] in a.only.split(",")]
    tasks.sort(key=lambda t: -t[0]["n"])  # large clusters first
    os.makedirs(os.path.join(a.out, "claims"), exist_ok=True)
    done = set()
    for fn in os.listdir(a.out):
        if fn.startswith("results_") and fn.endswith(".jsonl"):
            for ln in open(os.path.join(a.out, fn)):
                if ln.strip():
                    d = json.loads(ln)
                    if "error" not in d:
                        done.add((d["id"], d["job"]))
    out = os.path.join(a.out, f"results_{a.worker}.jsonl")
    psi4.set_memory(f"{a.memory_GB} GB")
    psi4.set_num_threads(a.threads)
    psi4.core.set_output_file(os.path.join(a.out, f"psi4_{a.worker}.out"), False)
    if a.quiet:
        psi4.core.be_quiet()
    for rec, job in tasks:
        if (rec["id"], job) in done:
            continue
        claim = os.path.join(a.out, "claims", (rec["id"] + "__" + job).replace("/", "_").replace(":", "_"))
        try:
            os.mkdir(claim)
        except FileExistsError:
            if time.time() - os.path.getmtime(claim) < a.stale_hours * 3600:
                continue
            os.utime(claim)
        with open(os.path.join(claim, "owner"), "w") as fh:
            fh.write(f"{a.worker} {socket.gethostname()} {time.ctime()}\n")
        t0 = time.time()
        line = {"id": rec["id"], "job": job}
        try:
            line["res"] = run_job(psi4, rec, job)
        except Exception as ex:  # noqa: BLE001
            line["error"] = repr(ex)[:500]
        line["sec"] = round(time.time() - t0, 1)
        line["threads"] = a.threads
        with open(out, "a") as fh:
            fh.write(json.dumps(line) + "\n")
        print(rec["id"], job, line["sec"], line.get("error", ""), flush=True)
        psi4.core.clean()
        psi4.core.clean_variables()


if __name__ == "__main__":
    main()
