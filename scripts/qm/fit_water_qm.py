"""Fit pGM water parameters to the QM cluster set and report the errors (the `pgm-jax fit-qm` command).

The dataset is data/qm/water_qm.json (scripts/qm/collect_water_qm.py); the model and the fit are
pgm_jax.fit.qm (ClusterModel, QMFit; docs/qmfit.md).  Modes: baseline (report of the pGM3P-25 and
base parameter sets), fit NAME (least-squares fit of the free parameters, then its report), summary
NAMES (one line per report).  Each report: error tables per set (interaction energies against
CCSD(T)/CBS*, SAPT0 components), the water dimer (at the QM minimum and the model's rigid-body
minimum), the Smith stationary structures, the WATER27 hexamers and their order, 3-body energies,
rigid-body forces and torques, the monomer dipole and polarizability.

Split: test = everything from the base_4096 snapshot (pairs, trimers, ..., and the pairs of those
clusters), the WATER27 clusters and the Smith-type stationary structures; train = the rest (scans
and p25 liquid snapshots).

Usage:

    python scripts/qm/fit_water_qm.py baseline                       # pGM3P-25 and base parameters
    python scripts/qm/fit_water_qm.py fit NAME [--start p25] [--vdw lj|gvdw]
            [--free "q=all;cov=all;radius=all;alpha=all;lj_rmin_half=OW;lj_sqrt_eps=OW"]
            [--weights "total=1,elst=0.3,ind=0.3,exch_disp=0.3,nb3=1,dipole=1,polarizability=1,prior=0.01"]
    python scripts/qm/fit_water_qm.py summary NAME1,NAME2           # with p25 and base from baseline.json
    python scripts/qm/fit_water_qm.py --help

Inputs: --data (default data/qm/water_qm.json); the start models' prmtops in PGM_EPSP (pgm_jax.paths).
Outputs: <out>/<name>.json (report; <out>/baseline.json for baseline; default --out runs/qmfit) and,
for fits, data/qm/fits/<name>.json (the fitted molecule, pgm_jax.param.load_molecule); printed tables.
Units: kcal/mol (energies), kcal/mol/A (forces), A, D, A^3.
Runtime: CPU, minutes (a fit: up to --nfev evaluations of the residuals).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.fit.qm import (
    KCAL,
    ClusterModel,
    FitWeights,
    Prepared,
    QMFit,
    QMSet,
    error_table,
    evaluate,
    format_table,
    label,
    parameter_space,
    rigid_minimize,
    rigid_water,
    superpose_monomers,
)
from pgm_jax.param import read_prmtop_pgm, save_molecule
from pgm_jax.paths import repo_path, resource
from pgm_jax.units import ANG_NM, DEBYE_E_NM
from pgm_jax.vdw import PGM3P_GVDW, set_gvdw

jax.config.update("jax_enable_x64", True)
MODELS = {  # start model -> (prmtop, rigid geometry (r_OH [A], HOH [deg]))
    "p25": (resource("epsp", "p25_512.prmtop"), (0.9745, 103.64)),
    "base": (resource("epsp", "base/base_512.prmtop"), (0.9572, 104.49)),
}
LIT_DIMER = -5.0  # kcal/mol, CCSD(T)/CBS water dimer minimum (Tschumper et al. JCP 2002: -5.02 +- 0.05; HBB2 -4.98)
HEXAMERS = [
    ("water27/H2O6", "prism"),
    ("water27/H2O6c", "cage"),
    ("water27/H2O6b", "book"),
    ("water27/H2O6c2", "cyclic"),
]


def is_test(r: dict) -> bool:
    """Return True for a test record (base_4096 snapshot, WATER27, Smith structures)."""
    return "base_4096" in r["id"] or r["id"].startswith("water27") or r["set"] == "smith"


def model(name: str, vdw: str = "lj", mol: object = None, init: str = "") -> tuple[ClusterModel, np.ndarray]:
    """Return the ClusterModel of a starting parameter set and its rigid water geometry.

    Parameters
    ----------
    name : {"p25", "base"}
        Start model (MODELS).
    vdw : {"lj", "gvdw"}
        Van der Waals model (gvdw: the pGM3P GVDW "gauss" parameters on OW).
    mol : Molecule, optional
        Molecule to use instead of the prmtop's.
    init : str
        "quantity:type=value;..." overrides initial values per atom type (e.g.
        "lj_rmin_half:HW=0.1;lj_sqrt_eps:HW=0.1").

    Returns
    -------
    ClusterModel
    np.ndarray (3, 3)
        The rigid water (O, H, H) [A].
    """
    prm, geom = MODELS[name]
    mol = mol or read_prmtop_pgm(prm)[0]
    if vdw == "gvdw":
        g = PGM3P_GVDW["gauss"]
        mol = set_gvdw(mol, {"OW": g["OW"]})
    for item in init.split(";"):
        if item.strip():
            qk, v = item.split("=")
            qn, t = qk.split(":")
            arr = getattr(mol, qn).copy()
            arr[[k for k, tt in enumerate(mol.types) if tt == t]] = float(v)
            setattr(mol, qn, arr)
    W = rigid_water(*geom)
    return ClusterModel(mol, vdw=vdw, monomer_xyz_nm=W * ANG_NM), W


def on_geometry(data: QMSet, W: np.ndarray, masses: np.ndarray) -> QMSet:
    """Return data with every monomer replaced by the model's rigid water W [A] (data itself if it already fits)."""
    X0 = np.asarray(data.records[0]["xyz_A"])[:3]
    if abs(np.linalg.norm(X0[1] - X0[0]) - np.linalg.norm(W[1] - W[0])) < 1e-6:
        return data
    recs = []
    for r in data.records:
        r2 = dict(r)
        r2["xyz_A"] = superpose_monomers(r["xyz_A"], W, masses).tolist()
        r2.pop("grad_int", None)
        recs.append(r2)
    return QMSet(recs, data.monomer, data.about)


def groups(data: QMSet) -> tuple[list[str], list[str]]:
    """Return the sorted set names of the training and of the test records."""
    ids = [r for r in data.records]
    tr = sorted({r["set"] for r in ids if not is_test(r)})
    te = sorted({r["set"] for r in ids if is_test(r)})
    return tr, te


def report(cm: ClusterModel, W: np.ndarray, P: dict, data: QMSet, name: str, extra: dict | None = None) -> dict:
    """Print and return the report of a parameter set (see the module docstring).

    Parameters
    ----------
    cm : ClusterModel
        The model.
    W : np.ndarray (3, 3)
        Its rigid water [A].
    P : dict
        Parameters.
    data : QMSet
        The dataset.
    name : str
        Name of the report.
    extra : dict, optional
        Added to the result.

    Returns
    -------
    dict
        name, table (error rows), dimer, smith, hexamers (+ orders), nb3_ratio, forces, monomer, extra.
    """
    d = on_geometry(data, W, cm.mol.masses)
    ev = evaluate(cm, d, P)
    test = np.array([is_test(r) for r in d.records])
    sets = np.array(ev["set"])
    tagged = dict(ev)
    tagged["set"] = [("test:" if t else "train:") + s for s, t in zip(sets, test)]
    grp = {
        "train (all)": [s for s in set(tagged["set"]) if s.startswith("train:")],
        "test (all)": [s for s in set(tagged["set"]) if s.startswith("test:")],
    }
    grp.update({s: [s] for s in sorted(set(tagged["set"]))})
    rows = error_table(tagged, grp)
    out = {"name": name, "table": rows}
    print(f"\n=== {name}: errors vs CCSD(T)/CBS* (E_int) and SAPT0 components, kcal/mol")
    print(format_table(rows))
    report_dimers(cm, P, d, ev, out)
    report_hexamers(cm, P, d, ev, out)
    report_nb3(ev, out)
    report_forces(cm, P, d, test, out)
    report_monomer(cm, P, data, out)
    out.update(extra or {})
    return out


def report_dimers(cm: ClusterModel, P: dict, d: QMSet, ev: dict, out: dict) -> None:
    """Add (and print) the water dimer and the Smith structures relative to it to `out` [kcal/mol, A]."""
    k = ev["ids"].index("smith/Cs_open")
    Emin, Xmin = rigid_minimize(cm, d.records[k]["xyz_A"], P)
    roo = float(np.linalg.norm(Xmin[3] - Xmin[0]))
    out["dimer"] = {
        "E_at_ref_min": float(ev["total"][k]),
        "ref": float(ev["ref"][k]),
        "lit": LIT_DIMER,
        "model_min": Emin,
        "model_min_ROO": roo,
    }
    print(
        f"water dimer: E(model) at the QM minimum {ev['total'][k]:.3f}, QM {ev['ref'][k]:.3f}, literature {LIT_DIMER}; "
        f"model minimum {Emin:.3f} at R_OO {roo:.3f} A"
    )
    smith = [(i, ev["ids"][i]) for i in range(len(ev["ids"])) if ev["set"][i] == "smith"]
    e0m, e0q = ev["total"][k], ev["ref"][k]
    out["smith"] = {}
    for i, sid in smith:
        out["smith"][sid] = {
            "model": float(ev["total"][i]),
            "ref": float(ev["ref"][i]),
            "model_rel": float(ev["total"][i] - e0m),
            "ref_rel": float(ev["ref"][i] - e0q),
        }
        print(
            f"  {sid:32s} E {ev['total'][i]:7.3f} ref {ev['ref'][i]:7.3f}   rel {ev['total'][i] - e0m:6.3f} ref "
            f"{ev['ref'][i] - e0q:6.3f}"
        )


def report_hexamers(cm: ClusterModel, P: dict, d: QMSet, ev: dict, out: dict) -> None:
    """Add (and print) the WATER27 hexamers and their energy order by model, model minimum, QM and literature."""
    out["hexamers"] = {}
    print(
        "hexamers (kcal/mol): model at the rigidified WATER27 geometries / model rigid-body minimum / CCSD(T)/CBS* "
        "(same geometries) / WATER27 literature (relaxed monomers)"
    )
    for hid, lab in HEXAMERS:
        if hid not in ev["ids"]:
            continue
        i = ev["ids"].index(hid)
        Em, _ = rigid_minimize(cm, d.records[i]["xyz_A"], P)
        out["hexamers"][lab] = {
            "model": float(ev["total"][i]),
            "model_min": Em,
            "ref": float(ev["ref"][i]),
            "lit": label(d.records[i], "E.lit_De"),
        }
        print(f"  {lab:8s} {ev['total'][i]:8.2f} {Em:8.2f} {ev['ref'][i]:8.2f} {label(d.records[i], 'E.lit_De'):8.2f}")
    for key in ("model", "model_min", "ref", "lit"):
        order = sorted(out["hexamers"], key=lambda h: out["hexamers"][h][key])
        out["hexamers_order_" + key] = order
        print(f"  order by {key:9s}: {' < '.join(order)}")


def report_nb3(ev: dict, out: dict) -> None:
    """Add (and print) the ratio model / MP2 of the 3-body energies summed per set."""
    if "nb3" in ev:
        idx = {i: j for j, i in enumerate(ev["ids"])}
        rat = {}
        for cid, m3, q3 in zip(ev["nb3_ids"], ev["nb3"], ev["nb3_ref"]):
            s = ev["set"][idx[cid]]
            rat.setdefault(s, []).append((m3, q3))
        out["nb3_ratio"] = {s: float(np.sum([a for a, _ in v]) / np.sum([b for _, b in v])) for s, v in rat.items()}
        print(
            "3-body energy, model / MP2 (sums over the clusters of each set):",
            {s: round(v, 3) for s, v in out["nb3_ratio"].items()},
        )


def report_forces(cm: ClusterModel, P: dict, d: QMSet, test: np.ndarray, out: dict) -> None:
    """Add (and print) the rigid-body force [kcal/mol/A] and torque [kcal/mol] errors of the liquid pairs vs MP2/aTZ."""
    prep = Prepared(d, cm)
    if prep.force_recs:
        out["forces"] = {}
        for n, (ks, F, T) in prep.forces(P).items():
            _, _, Fq, Tq = prep.force_recs[n]
            dF = np.asarray(F - Fq) / (KCAL / ANG_NM)
            dT = np.asarray(T - Tq) / KCAL
            fq = np.asarray(Fq) / (KCAL / ANG_NM)
            tk = np.repeat(test[ks], n)  # rows are (record, molecule)
            for tag, mk in (("train", ~tk), ("test", tk)):
                if mk.any():
                    out["forces"][tag] = {
                        "N": int(mk.sum()) // n,
                        "F_RMSE": float(np.sqrt(np.mean(dF[mk] ** 2))),
                        "F_rms_ref": float(np.sqrt(np.mean(fq[mk] ** 2))),
                        "T_RMSE": float(np.sqrt(np.mean(dT[mk] ** 2))),
                    }
        print(
            "rigid-body forces vs CP MP2/aTZ (kcal/mol/A; torques kcal/mol):",
            {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out["forces"].items()},
        )


def report_monomer(cm: ClusterModel, P: dict, data: QMSet, out: dict) -> None:
    """Add (and print) the monomer dipole [D] and polarizability [A^3] with the QM values."""
    mu = float(np.linalg.norm(cm.monomer_dipole(P)) / DEBYE_E_NM)
    al = float(cm.monomer_polarizability(P) / ANG_NM**3)
    out["monomer"] = {
        "dipole_D": mu,
        "polarizability_A3": al,
        "qm": {k: data.monomer.get(k) for k in ("dipole_D", "polarizability_A3")},
    }
    print(
        f"monomer: dipole {mu:.4f} D (QM {data.monomer.get('dipole_D')}), polarizability {al:.4f} A^3 (QM "
        f"{data.monomer.get('polarizability_A3')})"
    )


def summary(names: list[str], outdir: str) -> list[str]:
    """Print and return a compact comparison of the reports <outdir>/<name>.json (baseline.json holds p25 and base)."""
    reps = {}
    base = os.path.join(outdir, "baseline.json")
    if os.path.exists(base):
        with open(base) as fh:
            for r in json.load(fh):
                reps[r["name"]] = r
    for n in names:
        p = os.path.join(outdir, f"{n}.json")
        if os.path.exists(p):
            with open(p) as fh:
                reps[n] = json.load(fh)

    def get(r, grp, q, key="RMSE"):
        """Return the error `key` of quantity q in group grp of report r (nan if absent)."""
        for row in r["table"]:
            if row["group"] == grp and row["quantity"] == q:
                return row[key]
        return float("nan")

    hdr = (
        f"{'model':16s} {'Eint tr':>8s} {'Eint te':>8s} {'MAE te':>7s} {'elst te':>8s} {'ind te':>7s} {'ex+di te':>8s} "
        f"{'3b te':>6s} {'3b W27':>8s} {'dimer':>7s} {'dim min':>7s} {'hex order':>10s} {'mu D':>6s} {'a A3':>6s} "
        f"{'F te':>6s}"
    )
    lines = [hdr]
    for n, r in reps.items():
        hx = r.get("hexamers_order_model", [])
        ok = "ok" if hx == ["prism", "cage", "book", "cyclic"] else "/".join(h[:2] for h in hx)
        rat = r.get("nb3_ratio", {})
        lines.append(
            f"{n:16s} {get(r, 'train (all)', 'E_int'):8.3f} {get(r, 'test (all)', 'E_int'):8.3f} "
            f"{get(r, 'test (all)', 'E_int', 'MAE'):7.3f} {get(r, 'test (all)', 'elst'):8.3f} "
            f"{get(r, 'test (all)', 'ind'):7.3f} "
            f"{get(r, 'test (all)', 'exch+disp'):8.3f} {get(r, 'test (all)', '3-body'):6.3f} "
            f"{rat.get('water27', float('nan')):8.3f} {r['dimer']['E_at_ref_min']:7.3f} "
            f"{r['dimer']['model_min']:7.3f} {ok:>10s} {r['monomer']['dipole_D']:6.3f} "
            f"{r['monomer']['polarizability_A3']:6.3f} "
            f"{r.get('forces', {}).get('test', {}).get('F_RMSE', float('nan')):6.3f}"
        )
    print("\n".join(lines))
    return lines


def parse_free(s: str) -> dict:
    """Return the free parameters of --free: "quantity=all;quantity=T1,T2" -> {quantity: "all" or [types]}."""
    out = {}
    for item in s.split(";"):
        if item.strip():
            k, v = item.split("=")
            out[k.strip()] = "all" if v.strip() == "all" else [x.strip() for x in v.split(",")]
    return out


def fitted_molecule(cm: ClusterModel, P: dict, vdw: str) -> object:
    """Return the model's molecule with the fitted parameters P (no Lennard-Jones for vdw == "gvdw")."""
    mol = cm.mol
    Pa = cm.system(1).expand(P)
    mol.q = np.asarray(Pa["q"])
    mol.radius, mol.alpha = np.asarray(Pa["radius"]), np.asarray(Pa["alpha"])
    mol.lj_rmin_half, mol.lj_sqrt_eps = np.asarray(Pa["lj_rmin_half"]), np.asarray(Pa["lj_sqrt_eps"])
    mol.gvdw_sqrt_a, mol.gvdw_sqrt_c6, mol.gvdw_b = (
        np.asarray(Pa[k]) for k in ("gvdw_sqrt_a", "gvdw_sqrt_c6", "gvdw_b")
    )
    mol.cov = [(i, j, float(c)) for (i, j, _), c in zip(mol.cov, np.asarray(Pa["cov"]))]
    if vdw == "gvdw":  # the fitted model has no Lennard-Jones
        mol.lj_rmin_half, mol.lj_sqrt_eps = np.zeros(mol.n), np.zeros(mol.n)
    return mol


def run_fit(a: argparse.Namespace, data: QMSet) -> None:
    """Fit the free parameters, save the fitted molecule and write the report (the `fit` mode)."""
    cm, W = model(a.start, a.vdw, init=a.init)
    free = parse_free(a.free)
    space = parameter_space(cm.table, [cm.mol], free)
    w = FitWeights(**{k: float(v) for k, v in (x.split("=") for x in a.weights.split(",") if x)})
    train, test = data.split(is_test)
    print(
        f"fit {a.name}: {len(space)} parameters {space.names}; train {len(train)} records, "
        f"test {len(test)}; weights {w}"
    )
    fit = QMFit(cm, space, on_geometry(train, W, cm.mol.masses), w)
    L0 = float(fit.loss(jnp.asarray(space.theta0)))
    t0 = time.time()
    res = fit.fit(max_nfev=a.nfev, verbose=1)
    P = space(jnp.asarray(res.x))
    print(f"loss {L0:.4g} -> {res.cost:.4g} in {time.time() - t0:.0f} s, {res.nfev} evaluations, status {res.status}")
    named = {n: (float(x0), float(x)) for n, x0, x in zip(space.names, space.theta0, res.x)}
    for n, (x0, x) in named.items():
        print(f"  {n:24s} {x0:12.6g} -> {x:12.6g}")
    mol = fitted_molecule(cm, P, a.vdw)
    os.makedirs(repo_path("data", "qm", "fits"), exist_ok=True)
    save_molecule(mol, repo_path("data", "qm", "fits", f"{a.name}.json"))
    out = report(
        cm,
        W,
        P,
        data,
        a.name,
        {
            "params": named,
            "free": free,
            "weights": w.__dict__,
            "vdw": a.vdw,
            "start": a.start,
            "init": a.init,
            "cost": [L0, float(res.cost)],
        },
    )
    with open(os.path.join(a.out, f"{a.name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


DEFAULT_FREE = "q=all;cov=all;radius=all;alpha=all;lj_rmin_half=OW;lj_sqrt_eps=OW"
DEFAULT_WEIGHTS = "total=1,elst=0.3,ind=0.3,exch_disp=0.3,nb3=1,dipole=1,polarizability=1,prior=0.01"


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run one mode: baseline, fit or summary (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["baseline", "fit", "summary"], help="what to do")
    ap.add_argument("name", nargs="?", default="fit", help="fit: its name; summary: comma-separated report names")
    ap.add_argument("--data", default=repo_path("data", "qm", "water_qm.json"), help="QM dataset")
    ap.add_argument("-o", "--out", default=repo_path("runs", "qmfit"), help="directory of the reports")
    ap.add_argument("--start", default="p25", choices=sorted(MODELS), help="start parameters")
    ap.add_argument("--vdw", default="lj", choices=["lj", "gvdw"], help="van der Waals model")
    ap.add_argument("--free", default=DEFAULT_FREE, help="free parameters, quantity=all|type,type;...")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="FitWeights, name=value,...")
    ap.add_argument("--nfev", type=int, default=400, help="maximum residual evaluations of the fit")
    ap.add_argument("--init", default="", help='initial values per type, "quantity:type=value;..."')
    a = ap.parse_args(argv)
    data = QMSet.load(a.data)
    data = data.select(lambda r: np.isfinite(label(r, "E.ref")))
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "summary":
        summary(a.name.split(","), a.out)
    elif a.mode == "baseline":
        res = []
        for name in ("p25", "base"):
            cm, W = model(name)
            res.append(report(cm, W, cm.table.initial(), data, name))
        with open(os.path.join(a.out, "baseline.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    else:
        run_fit(a, data)


if __name__ == "__main__":
    main()
