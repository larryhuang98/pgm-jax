"""Collect the psi4 results (scripts/qmfit/psi4_clusters.py) into the compact dataset
data/qm/water_qm.json (pgm_jax.qmfit.QMSet): coordinates in Angstrom, energies in kcal/mol.

Per dimer (counterpoise corrected in the dimer basis, frozen core, DF):
  E.mp2_atz, E.mp2_aqz      MP2/aug-cc-pVTZ, aug-cc-pVQZ
  E.mp2_cbs                 HF/aQZ + MP2 correlation extrapolated X^-3 from aTZ and aQZ (Helgaker)
  E.ccsdt_atz               DF-CCSD(T)/aug-cc-pVTZ
  E.ref                     CCSD(T)/CBS estimate = E.mp2_cbs + [CCSD(T) - MP2]/aTZ
  sapt.*                    SAPT0/jun-cc-pVDZ: elst, exch, ind (incl. dHF), disp, total; ssapt0_* (scaled)
  grad_int                  gradient of the CP MP2/aTZ interaction energy (kcal/mol/A, (6, 3))
Per cluster (n >= 3; CP in the cluster basis, MP2/aTZ):
  nb.int_mp2_atz, nb.nb2, nb.nb3, nb.nb4plus  many-body expansion (nb4plus = int - nb2 - nb3)
  E.mp2_atz                 = nb.int_mp2_atz
  E.ref                     = E.mp2_atz + sum over pairs [E.ref(pair) - E.mp2_atz(pair)] (2-body correction
                              to CCSD(T)/CBS; pairs in the dimer basis)
  E.lit_De                  WATER27 reference binding energy (CCSD(T)/CBS, relaxed monomers; Bryantsev et al.
                              JCTC 2009, 5, 1016 / Anacker & Friedrich JCC 2014), for comparison only
monomer: CCSD/aug-cc-pVTZ dipole (D) and static isotropic polarizability (A^3) at the rigid geometry.

    python scripts/qmfit/collect_water_qm.py [~/project/qmdata/water] [--out data/qm/water_qm.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EH = 627.5094740631  # kcal/mol
BOHR_A = 0.529177210903
DEBYE_AU = 2.541746473
AU_A3 = BOHR_A**3
WATER27_LIT = {
    "H2O2": 4.974,
    "H2O3": 15.708,
    "H2O4": 27.353,
    "H2O5": 35.879,
    "H2O6": 45.988,
    "H2O6c": 45.733,
    "H2O6b": 45.292,
    "H2O6c2": 44.296,
    "H2O8d2d": 72.490,
    "H2O8s4": 72.454,
}


def e_int(res, key, n=2):
    full = res[",".join(map(str, range(n)))][key]
    return (full - sum(res[str(k)][key] for k in range(n))) * EH


def main(a):
    geoms = json.load(open(os.path.join(ROOT, "data/qm/water_geoms.json")))
    R = {}
    for fn in sorted(os.listdir(a.qmdir)):
        if fn.startswith("results_") and fn.endswith(".jsonl"):
            for ln in open(os.path.join(a.qmdir, fn)):
                if ln.strip():
                    d = json.loads(ln)
                    if "res" in d:
                        R[(d["id"], d["job"])] = d
    recs, monomer, missing = [], {}, []
    cost = {}
    for g in geoms["records"]:
        for j in g["jobs"]:
            if (g["id"], j) in R:
                cost[j.split(":")[0]] = (
                    cost.get(j.split(":")[0], 0.0) + R[(g["id"], j)]["sec"] * R[(g["id"], j)]["threads"]
                )
            else:
                missing.append((g["id"], j))
    byid = {}
    for g in geoms["records"]:
        r = {k: g[k] for k in ("id", "set", "n", "xyz_A", "meta")}
        get = lambda j: R.get((g["id"], j), {}).get("res")  # noqa: E731
        if g["set"] == "monomer":
            p = get("props:ccsd:aug-cc-pvtz")
            if p:
                dip = next((v for k, v in p.items() if "CCSD DIPOLE" == k), None)
                pol = next((v for k, v in p.items() if "POLARIZABILITY" in k and "CCSD" in k), None)
                monomer = {
                    "level": "CCSD/aug-cc-pVTZ (frozen core), rigid pGM3P-25 geometry",
                    "dipole_D": float(np.linalg.norm(dip)) * DEBYE_AU if dip is not None else None,
                    "polarizability_A3": (
                        float(np.trace(np.asarray(pol).reshape(3, 3)) / 3) if np.size(pol) == 9 else float(pol)
                    )
                    * AU_A3
                    if pol is not None
                    else None,
                    "xyz_A": g["xyz_A"],
                    "raw_keys": sorted(p),
                }
            continue
        E, sapt, nb = {}, {}, {}
        if g["n"] == 2:
            t, q, c, s = get("mp2:aug-cc-pvtz"), get("mp2:aug-cc-pvqz"), get("ccsdt:aug-cc-pvtz"), get("sapt0")
            if t:
                E["mp2_atz"] = e_int(t, "mp2")
            if q:
                E["mp2_aqz"] = e_int(q, "mp2")
            if t and q:
                ct, cq = e_int(t, "mp2") - e_int(t, "scf"), e_int(q, "mp2") - e_int(q, "scf")
                E["hf_aqz"] = e_int(q, "scf")
                E["mp2_cbs"] = E["hf_aqz"] + (64 * cq - 27 * ct) / 37
            if c:
                E["ccsdt_atz"] = e_int(c, "ccsdt")
                E["dccsdt_atz"] = e_int(c, "ccsdt") - e_int(c, "mp2")
            if "mp2_cbs" in E and "dccsdt_atz" in E:
                E["ref"] = E["mp2_cbs"] + E["dccsdt_atz"]
            if s:
                for k, name in (
                    ("SAPT ELST ENERGY", "elst"),
                    ("SAPT EXCH ENERGY", "exch"),
                    ("SAPT IND ENERGY", "ind"),
                    ("SAPT DISP ENERGY", "disp"),
                    ("SAPT0 TOTAL ENERGY", "total"),
                    ("SSAPT0 EXCH ENERGY", "ssapt0_exch"),
                    ("SSAPT0 IND ENERGY", "ssapt0_ind"),
                    ("SSAPT0 DISP ENERGY", "ssapt0_disp"),
                    ("SSAPT0 TOTAL ENERGY", "ssapt0_total"),
                    ("SAPT CT ENERGY", "ct"),
                ):
                    if k in s:
                        sapt[name] = s[k] * EH
            gr = get("mp2grad:aug-cc-pvtz")
            if gr:
                G = np.asarray(gr["0,1"]["grad"]) - np.asarray(gr["0"]["grad"]) - np.asarray(gr["1"]["grad"])
                r["grad_int"] = np.round(G * EH / BOHR_A, 8).tolist()
                E["mp2_atz_grad_run"] = e_int(gr, "mp2")
        else:
            m = get("mbe:mp2:aug-cc-pvtz:3") or get("mbe:mp2:aug-cc-pvtz:1")
            n = g["n"]
            if m:
                Ek = lambda S: m[",".join(map(str, S))]["mp2"]  # noqa: E731
                nb["int_mp2_atz"] = (Ek(range(n)) - sum(Ek([i]) for i in range(n))) * EH
                if "0,1" in m:
                    e2 = {(i, j): (Ek([i, j]) - Ek([i]) - Ek([j])) * EH for i, j in itertools.combinations(range(n), 2)}
                    nb["nb2"] = sum(e2.values())
                    nb3 = 0.0
                    for i, j, k in itertools.combinations(range(n), 3):
                        nb3 += (Ek([i, j, k]) - Ek([i]) - Ek([j]) - Ek([k])) * EH - e2[(i, j)] - e2[(i, k)] - e2[(j, k)]
                    nb["nb3"] = nb3
                    nb["nb4plus"] = nb["int_mp2_atz"] - nb["nb2"] - nb3
                E["mp2_atz"] = nb["int_mp2_atz"]
        if g["set"] == "water27":
            key = g["id"].split("/")[1]
            E["lit_De"] = -WATER27_LIT[key]
        r.update(E=E, sapt=sapt, nb=nb)
        recs.append(r)
        byid[r["id"]] = r
    # 2-body corrections of the clusters
    for r in recs:
        if r["n"] >= 3 and "mp2_atz" in r["E"]:
            corr, ok = 0.0, True
            for i, j in itertools.combinations(range(r["n"]), 2):
                p = byid.get(f"{r['id']}/p{i}-{j}")
                if p is None or "ref" not in p["E"] or "mp2_atz" not in p["E"]:
                    ok = False
                    break
                corr += p["E"]["ref"] - p["E"]["mp2_atz"]
            if ok:
                r["E"]["d2b"] = corr
                r["E"]["ref"] = r["E"]["mp2_atz"] + corr
    out = {
        "about": "water clusters of rigid pGM3P-25 monomers (r_OH 0.9745 A, HOH 103.64 deg); coordinates Angstrom "
        "(O,H,H per molecule); energies kcal/mol, gradients kcal/mol/A; psi4 1.11; "
        "scripts/qmfit/{build_water_clusters,psi4_clusters,collect_water_qm}.py",
        "levels": {
            "E.ref": "CCSD(T)/CBS estimate: CP MP2/CBS(aTZ,aQZ; HF aQZ) + CP [CCSD(T)-MP2]/aug-cc-pVTZ (dimers); "
            "clusters: CP MP2/aTZ (cluster basis) + sum of pair corrections [E.ref - MP2/aTZ]",
            "sapt": "SAPT0/jun-cc-pVDZ (psi4, DF); ind includes dHF",
            "nb": "CP MP2/aug-cc-pVTZ many-body expansion in the cluster basis",
            "grad_int": "gradient of the CP MP2/aug-cc-pVTZ interaction energy",
            "monomer": "CCSD/aug-cc-pVTZ dipole and static polarizability",
        },
        "cost_core_hours": {k: round(v / 3600, 2) for k, v in cost.items()},
        "missing": len(missing),
        "monomer": monomer,
        "records": recs,
    }
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    from collections import Counter

    print("records", len(recs), Counter(r["set"] for r in recs))
    print("with E.ref", sum("ref" in r["E"] for r in recs), "missing tasks", len(missing), missing[:5])
    print("core hours", out["cost_core_hours"], "total", round(sum(cost.values()) / 3600, 1))
    print("monomer", {k: v for k, v in monomer.items() if k != "xyz_A"})
    print("size MB", os.path.getsize(a.out) / 1e6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("qmdir", nargs="?", default=os.path.expanduser("~/project/qmdata/water"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/qm/water_qm.json"))
    main(ap.parse_args())
