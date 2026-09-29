"""QM reference data for water clusters with psi4: one worker of a dynamic work queue.

Tasks are (record, job) pairs from data/qm/water_geoms.json; workers claim tasks with an atomic
mkdir on the shared file system, so any number of workers (slurm array tasks) balance the load and
a killed run is resumed by starting workers again (a claimed task without a result line is retried
after --stale hours).  Results are appended as JSON lines to <out>/results_<worker>.jsonl.

Jobs (all counterpoise corrected in the basis of the whole record, frozen core, DF-SCF):
  sapt0                 SAPT0/jun-cc-pVDZ components (all psi4 SAPT* variables, Eh), dimers only
  mp2:<basis>           DF-MP2 energies of the dimer and of each monomer in the dimer basis
  ccsdt:<basis>         DF-CCSD(T) (fnocc) likewise (also the DF-MP2 energies of the same runs)
  mp2grad:<basis>       DF-MP2 gradients of the dimer and of each monomer in the dimer basis
  props:<method>:<basis>  dipole and static polarizability of a monomer (psi4.properties, a.u.)
  mbe:<method>:<basis>:<k>   energies of every subset of <= k molecules and of the whole cluster, all
                        in the cluster basis (the CP many-body expansion; interaction energy from the
                        full cluster and the monomers)

    python scripts/qmfit/psi4_clusters.py GEOMS OUTDIR WORKER --threads 16 --memory 30 [--only ids]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import time

SAPT_BASIS = "jun-cc-pvdz"


def mol_string(frags, active):
    """frags: list of (elements, xyz_A); active: fragment indices that are real (others ghosts)."""
    out = []
    for k, (el, xyz) in enumerate(frags):
        if k:
            out.append("--")
        out.append("0 1")
        for e, (x, y, z) in zip(el, xyz):
            name = e if k in active else f"Gh({e})"
            out.append(f"{name} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(out + ["units angstrom", "symmetry c1", "no_com", "no_reorient"])


def frags_of(rec):
    X = rec["xyz_A"]
    el = rec.get("elements") or ["O", "H", "H"] * rec["n"]
    na = rec.get("atoms_per_mol", 3)
    return [(el[na * k : na * k + na], X[na * k : na * k + na]) for k in range(rec["n"])]


def run_job(psi4, rec, job):
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
        import numpy as np

        res = {
            k: np.asarray(v.np if hasattr(v, "np") else v, float).tolist()
            for k, v in psi4.core.variables().items()
            if "DIPOLE" in k or "POLARIZABILITY" in k or "TOTAL ENERGY" in k
        }
    else:
        raise ValueError(job)
    return res


def main(a):
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
    psi4.set_memory(f"{a.memory} GB")
    psi4.set_num_threads(a.threads)
    psi4.core.set_output_file(os.path.join(a.out, f"psi4_{a.worker}.out"), False)
    psi4.core.be_quiet() if a.quiet else None
    for rec, job in tasks:
        if (rec["id"], job) in done:
            continue
        claim = os.path.join(a.out, "claims", (rec["id"] + "__" + job).replace("/", "_").replace(":", "_"))
        try:
            os.mkdir(claim)
        except FileExistsError:
            if time.time() - os.path.getmtime(claim) < a.stale * 3600:
                continue
            os.utime(claim)
        open(os.path.join(claim, "owner"), "w").write(f"{a.worker} {socket.gethostname()} {time.ctime()}\n")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("geoms")
    ap.add_argument("out")
    ap.add_argument("worker")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--memory", type=float, default=30.0)
    ap.add_argument("--stale", type=float, default=12.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--quiet", action="store_true")
    main(ap.parse_args())
