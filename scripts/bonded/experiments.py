"""Bonded-term experiments of the bonded study (plan X1-X6).

run: build the model for a set of molecules and settings, fit on 500 K frames + relaxed torsion
scans (the paper's training data), report the paper's metrics on 298 K frames and
force-field-relaxed scans.  l1path: error against the number of active linear parameters along an
L1 path (per-molecule fits, warm-started).  table: one summary line per result.

Usage:

    python scripts/bonded/experiments.py run NAME --mols A1 --families paper [--elec 3] ...
    python scripts/bonded/experiments.py l1path NAME --mols A1 --lams 0,1e-3,1e-2
    python scripts/bonded/experiments.py table NAME [NAME ...]
    python scripts/bonded/experiments.py --help

Inputs: the bonded-study data (pgm_jax.bonded.study.data: data/bonded/...).
Outputs: runs/bonded/results/<name>.json (and <name>_<molecules>.params.pkl with --save-params).
Units: energy MAE [kcal/mol], force MAE [kcal/mol/A], scan errors [kcal/mol].
Runtime: CPU, minutes to hours depending on the molecule set.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import numpy as np

from pgm_jax.bonded import terms as T
from pgm_jax.bonded.fit import SCALES, Fitter
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.bonded.study.bench import scan_metrics
from pgm_jax.bonded.study.data import esp_data, load, mol_list
from pgm_jax.bonded.study.families import families_of
from pgm_jax.paths import repo_path

jax.config.update("jax_enable_x64", True)
RES = repo_path("runs", "bonded", "results")


def run(a: argparse.Namespace) -> None:
    """Fit the model of one experiment, evaluate it and write runs/bonded/results/<name>.json.

    Molecule typing without --joint fits each molecule alone; otherwise one model is fitted to all
    molecules (the --holdout molecules only evaluated).
    """
    names = mol_list(a.mols)
    specs, data = load(names)
    st = BondedSettings(
        families=families_of(a.families),
        typing=a.typing,
        depth=a.depth,
        elec_exclude=a.elec,
        lj_min_sep=a.lj_sep,
        lj14_scale=a.lj14,
        flux=a.flux,
        elec14_scale=a.elec14,
        escale=tuple(int(x) for x in a.escale.split(",")) if a.escale else (),
        qfit=a.qfit,
        qbci=a.qbci,
    )
    out = {"name": a.name, "args": vars(a), "settings": st.__dict__, "molecules": {}}
    groups = [[i] for i in range(len(specs))] if (a.typing == "molecule" and not a.joint) else [list(range(len(specs)))]
    t0 = time.time()
    for g in groups:
        model = BondedModel([specs[i] for i in g], st)
        sub = {k: {"train": data[i]["train"], "test": data[i]["test"]} for k, i in enumerate(g)}
        if a.holdout:
            hold = [k for k, i in enumerate(g) if specs[i].name in a.holdout.split(",")]
            sub_fit = {k: v for k, v in sub.items() if k not in hold}
        else:
            hold, sub_fit = [], sub
        esp = {}
        if a.qfit >= 0 or a.qbci >= 0 or a.wesp > 0:
            for k, i in enumerate(g):
                e = esp_data(specs[i].name)
                if e is not None:
                    esp[k] = e
        fitter = Fitter(
            model,
            sub_fit,
            w_E=a.wE,
            w_F=a.wF,
            w_mu=a.wmu,
            l2=a.l2,
            l2_elec=a.l2_elec,
            esp={k: v for k, v in esp.items() if k in sub_fit},
            w_esp=a.wesp,
        )
        P0 = model.init_params(hold=tuple(hold))
        frozen = tuple(a.frozen.split(",")) if a.frozen else ()
        P = fitter.fit(P0, maxiter=a.maxiter, frozen=frozen, l1=a.l1, verbose=True)
        if a.save_params:  # parameters with their tying keys (pickle of numpy arrays)
            import pickle

            keys = {f: model.keys[f] for f in model.fams}
            keys["escale"] = list(model.es_pos)
            if a.qfit >= 0:
                keys["elec"] = {"q": model.q_keys, "c": model.c_keys}
            if a.qbci >= 0:
                keys["bci"] = {"t": list(model.t_pos), "dc": list(model.dc_pos)}
            keys["ref"] = model.ref_keys
            os.makedirs(RES, exist_ok=True)
            with open(os.path.join(RES, f"{a.name}_{'-'.join(specs[i].name for i in g)[:60]}.params.pkl"), "wb") as fh:
                pickle.dump({"P": jax.tree_util.tree_map(np.asarray, P), "keys": keys}, fh)
        ev = Fitter(model, sub, w_E=a.wE, w_F=a.wF, w_mu=a.wmu, esp=esp)
        mt, mtr = ev.metrics(P, "test"), ev.metrics(P, "train")
        npar = model.n_params(P)
        nz = int(
            sum(
                np.sum(np.abs(np.asarray(v)) > 1e-3 * SCALES.get(p, 1.0))
                for f in model.fams
                for p, v in P[f].items()
                if p in T.REGISTRY[f].linear
            )
        )
        for k, i in enumerate(g):
            spec = specs[i]
            r = {
                "test": mt[k],
                "train": mtr[k],
                "n_params_group": npar,
                "n_linear_nonzero_group": nz,
                "held_out": k in hold,
                "scans": {},
            }
            if k in hold:  # fraction of the held-out molecule's terms typed by training molecules
                seen = {f: {kk for j in sub_fit for kk in model.I[j][f]["k"]} for f in model.fams}
                tot = sum(len(model.I[k][f]["k"]) for f in model.fams)
                cov_ = sum(sum(1 for kk in model.I[k][f]["k"] if kk in seen[f]) for f in model.fams)
                r["coverage"] = cov_ / max(tot, 1)
            if not a.no_scans:
                for sk, fs in data[i]["scans"].items():
                    try:
                        r["scans"][sk] = scan_metrics(model, k, P, fs, relax=not a.no_relax)
                    except Exception as exc:  # a failed relaxation should not lose the run
                        r["scans"][sk] = {"error": repr(exc)[:200]}
            out["molecules"][spec.name] = r
            sc = [
                v.get("relaxed_max", np.inf if v.get("relax_ok") is False else v.get("sp_max", np.nan))
                for v in r["scans"].values()
                if "error" not in v
            ]
            print(
                f"  {spec.name:20s} E_MAE {mt[k]['E_MAE']:.3f}  F_MAE {mt[k]['F_MAE']:.2f}  "
                f"(train {mtr[k]['E_MAE']:.3f}/{mtr[k]['F_MAE']:.2f})  scan max "
                f"{np.max(sc) if sc else float('nan'):.2f}"
                f"  params {npar}{'  HELD OUT coverage {:.2f}'.format(r['coverage']) if k in hold else ''}",
                flush=True,
            )
    out["time_s"] = time.time() - t0
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"{a.name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    summarize([a.name])


def l1path(a: argparse.Namespace) -> None:
    """Write the error against the number of active linear parameters along an L1 path (per-molecule fits)."""
    names = mol_list(a.mols)
    specs, data = load(names, with_scans=True)
    st = BondedSettings(
        families=families_of(a.families),
        typing=a.typing,
        depth=a.depth,
        elec_exclude=a.elec,
        lj_min_sep=a.lj_sep,
        lj14_scale=a.lj14,
        flux=a.flux,
        elec14_scale=a.elec14,
    )
    lams = [float(x) for x in a.lams.split(",")]
    out = {"name": a.name, "args": vars(a), "lams": lams, "molecules": {}}
    for i, spec in enumerate(specs):
        model = BondedModel([spec], st)
        fitter = Fitter(model, {0: {"train": data[i]["train"], "test": data[i]["test"]}}, l2=a.l2)
        P = model.init_params()
        rows = []
        for lam in lams:  # warm start along the path
            P = fitter.fit(P, maxiter=a.maxiter, l1=lam, verbose=False)
            mt = fitter.metrics(P, "test")[0]
            nz = int(
                sum(
                    np.sum(np.abs(np.asarray(v)) > 1e-3 * SCALES.get(p, 1.0))
                    for f in model.fams
                    for p, v in P[f].items()
                    if p in T.REGISTRY[f].linear
                )
            )
            nlin = int(sum(np.size(v) for f in model.fams for p, v in P[f].items() if p in T.REGISTRY[f].linear))
            rows.append({"lam": lam, "E_MAE": mt["E_MAE"], "F_MAE": mt["F_MAE"], "n_active": nz, "n_linear": nlin})
            print(
                f"  {spec.name:20s} l1 {lam:8.1e}  active {nz:4d}/{nlin:4d}  E_MAE {mt['E_MAE']:.3f}  F_MAE "
                f"{mt['F_MAE']:.2f}",
                flush=True,
            )
        out["molecules"][spec.name] = rows
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"{a.name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def scan_max(v: dict) -> float:
    """Return the worst relaxed-scan error of one molecule's record [kcal/mol].

    The single-point profile error is used for a scan that was not relaxed; inf if a relaxation
    left the basin, nan without scans.
    """
    xs = []
    for x in v["scans"].values():
        if "error" in x:
            continue
        xs.append(x["relaxed_max"] if "relaxed_max" in x else (np.inf if x.get("relax_ok") is False else x["sp_max"]))
    return max(xs) if xs else np.nan


def summarize(names: list[str], exclude: tuple[str, ...] = ("methanethiol",)) -> None:
    """Print one line per result: mean test energy and force MAE and mean worst scan error (exclude left out)."""
    for n in names:
        with open(os.path.join(RES, f"{n}.json")) as fh:
            d = json.load(fh)
        ms = {k: v for k, v in d["molecules"].items() if k not in exclude}
        e = [v["test"]["E_MAE"] for v in ms.values()]
        f = [v["test"]["F_MAE"] for v in ms.values()]
        s = np.array([scan_max(v) for v in ms.values()])
        fin = s[np.isfinite(s)]
        print(
            f"{n:28s} mols {len(ms):2d}  E_MAE {np.mean(e):.3f}  F_MAE {np.mean(f):.2f}  scan max "
            f"{np.mean(fin) if len(fin) else np.nan:.2f} kcal/mol"
            f"  (relax failures {int(np.sum(np.isinf(s)))})"
        )


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "l1path", "table"), help="run, l1path or table")
    ap.add_argument("name", nargs="+", help="result name (several for table)")
    ap.add_argument("--mols", default="A1", help="molecule sets or names (bonded.study.data.mol_list)")
    ap.add_argument("--families", default="paper", help="term families (bonded.study.families)")
    ap.add_argument("--typing", default="molecule", help="parameter typing (BondedSettings.typing)")
    ap.add_argument("--depth", type=int, default=2, help="typing depth")
    ap.add_argument("--elec", type=int, default=0, help="electrostatics excluded up to this bond separation")
    ap.add_argument("--lj-sep", type=int, default=4, help="smallest bond separation with Lennard-Jones")
    ap.add_argument("--lj14", type=float, default=0.0, help="1-4 Lennard-Jones scale")
    ap.add_argument("--elec14", type=float, default=1.0, help="1-4 electrostatic scale")
    ap.add_argument("--flux", type=int, nargs="?", const=1, default=0, help="charge flux (BondedSettings.flux)")
    ap.add_argument("--escale", default="", help="learned permanent-pair scales for these separations, e.g. 1,2,3")
    ap.add_argument("--wE", type=float, default=1.0, help="energy weight")
    ap.add_argument("--wF", type=float, default=1.0, help="force weight")
    ap.add_argument("--wmu", type=float, default=0.0, help="dipole weight")
    ap.add_argument("--l2", type=float, default=1e-4, help="L2 weight")
    ap.add_argument("--l1", type=float, default=0.0, help="L1 weight of the linear parameters")
    ap.add_argument("--maxiter", type=int, default=20000, help="fit iterations")
    ap.add_argument("--frozen", default="", help="comma-separated families kept at their start values")
    ap.add_argument("--joint", action="store_true", help="one model for all molecules with molecule typing")
    ap.add_argument("--holdout", default="", help="comma-separated molecules left out of the fit")
    ap.add_argument("--no-scans", action="store_true", help="skip the scan metrics")
    ap.add_argument("--no-relax", action="store_true", help="single-point scan metrics only")
    ap.add_argument("--save-params", action="store_true", help="also write the parameters (pickle)")
    ap.add_argument("--qfit", type=int, default=-1, help="fit typed pGM charges/covalent dipoles (typing depth)")
    ap.add_argument(
        "--l2-elec", type=float, default=None, help="L2 weight of the electrostatic parameters (Fitter l2_elec)"
    )
    ap.add_argument(
        "--qbci", type=int, default=-1, help="fit typed bond-charge increments on the ESP charges (typing depth)"
    )
    ap.add_argument("--wesp", type=float, default=0.0, help="weight of the ESP restraint (relative RMSE / 0.1)^2")
    ap.add_argument("--lams", default="0,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1", help="l1path: L1 weights")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the experiment or print the table (see the module docstring)."""
    a = build_parser().parse_args(argv)
    if a.cmd in ("run", "l1path"):
        a.name = a.name[0]
        (run if a.cmd == "run" else l1path)(a)
    else:
        summarize(a.name)


if __name__ == "__main__":
    main()
