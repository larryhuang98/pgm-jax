"""Bonded-term experiments (plan X1-X6).  Each run: build the model for a set of molecules and
settings, fit on 500 K frames + relaxed torsion scans (the paper's training data), report the
paper's metrics on 298 K frames and force-field-relaxed scans.  Results: runs/bonded/results/.

    python scripts/bonded/experiments.py run NAME --mols A1 --families paper [--elec 3] ...
    python scripts/bonded/experiments.py table NAME [NAME ...]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from pgm_jax.bonded import terms as T  # noqa: E402
from pgm_jax.bonded.bench import scan_metrics  # noqa: E402
from pgm_jax.bonded.data import esp_data, frames, mol_spec, scan_keys  # noqa: E402
from pgm_jax.bonded.fit import SCALES, Fitter, FrameSet  # noqa: E402
from pgm_jax.bonded.model import BondedModel, BondedSettings  # noqa: E402
from pgm_jax.bonded.molecules import MOLECULES  # noqa: E402

RES = os.path.join(ROOT, "runs/bonded/results")
FAMILY_SETS = {
    "protein": T.PROTEIN,
    "paper": T.PAPER,
    "explore": T.PAPER,
    "amber": T.AMBER,
    "nn": ("nnb",),
    "diag": ("bond_morse", "angle_cos", "torsion", "improper"),
    "diag+p14": ("bond_morse", "angle_cos", "torsion", "improper", "pair14_exp"),
    "diag+ub": ("bond_morse", "angle_cos", "torsion", "improper", "pair13_harm", "pair14_exp"),
    "pair": ("bond_morse", "pair13_harm", "pair14_exp", "improper"),
    "pair+tors": ("bond_morse", "pair13_harm", "pair14_exp", "torsion", "improper"),
    "pair+ang": ("bond_morse", "angle_cos", "pair13_harm", "pair14_exp", "torsion", "improper"),
    "paper+pair14": T.PAPER + ("pair14_exp",),
    "tmod": ("bond_morse", "angle_cos", "bond_bond", "bond_angle", "angle_angle", "torsion_mod", "aat", "improper"),
    "paper+x": T.PAPER + ("bond_angle_x", "angle_angle_x", "angle_cubic"),
    "paper+oop": T.PAPER + ("torsion_oop",),
    "paper+tw": T.PAPER + ("twist",),
    "diag+tw": ("bond_morse", "angle_cos", "torsion", "improper", "twist"),
    "diag+oop": ("bond_morse", "angle_cos", "torsion", "improper", "torsion_oop"),
    "paper-pyr": tuple(f for f in T.PAPER if f != "improper") + ("pyramid",),
    "all": T.PAPER + ("bond_angle_x", "angle_angle_x", "angle_cubic", "pair13_harm", "pair14_exp"),
    "all+tw": T.PAPER + ("bond_angle_x", "angle_angle_x", "angle_cubic", "pair13_harm", "pair14_exp", "twist"),
    "diag+ub+tw": ("bond_morse", "angle_cos", "torsion", "improper", "pair13_harm", "pair14_exp", "twist"),
    # F12: electronic-structure-inspired families
    "diag+conj": ("bond_morse", "angle_cos", "torsion", "improper", "conj"),
    "paper+conj": T.PAPER + ("conj",),
    "diag+hc": ("bond_morse", "angle_cos", "torsion", "improper", "hc_sigma", "hc_lone"),
    "diag+vol": ("bond_morse", "angle_cos", "torsion", "volume"),
    "diag+new": ("bond_morse", "angle_cos", "torsion", "volume", "conj", "hc_sigma", "hc_lone"),
    "hyb": ("bond_morse", "angle_hyb", "torsion", "improper"),
    "hybsc": ("bond_morse", "angle_hybsc", "torsion", "improper"),
    "diag+ovl": ("bond_morse", "angle_cos", "torsion", "improper", "pair13_ovl", "pair14_ovl"),
    "chem": ("bond_morse", "angle_cos", "volume", "conj", "hc_sigma", "hc_lone", "pair14_exp"),
    "chem+hyb": ("bond_morse", "angle_hybsc", "volume", "conj", "hc_sigma", "hc_lone", "pair14_exp"),
    "dist": ("bond_morse", "pair13_tanh", "pair14_tanh", "volume"),
    "dist+chem": ("bond_morse", "pair13_tanh", "pair14_tanh", "volume", "conj", "hc_sigma", "hc_lone"),
}


def families_of(spec: str) -> tuple:
    """A FAMILY_SETS name, or families and set names joined by '+' ("amber+cmap", "paper+twist")."""
    if spec in FAMILY_SETS:
        return tuple(FAMILY_SETS[spec])
    out = []
    for tok in spec.split("+"):
        out += list(FAMILY_SETS[tok]) if tok in FAMILY_SETS else [tok]
    return tuple(dict.fromkeys(out))


def concat(sets):
    sets = [s for s in sets if s is not None and len(s)]
    return FrameSet(
        np.concatenate([s.X for s in sets]),
        np.concatenate([s.E for s in sets]),
        np.concatenate([s.F for s in sets]),
        np.concatenate([s.mu for s in sets]),
    )


def mol_list(spec):
    out = []
    for tok in spec.split(","):
        out += [n for n, v in MOLECULES.items() if v[2] == tok] if tok in ("A1", "A2", "A3", "A4", "B") else [tok]
    return out


def load(names, with_scans=True):
    data, specs = {}, []
    for n in names:
        tr = frames(n, "train500")
        te = frames(n, "test298")
        if tr is None or te is None:
            print(f"  {n}: no DFT frames yet, skipped", flush=True)
            continue
        scans = {k: frames(n, k) for k in scan_keys(n)} if with_scans else {}
        scans = {k: v for k, v in scans.items() if v is not None and len(v) >= 20}
        specs.append(mol_spec(n))
        data[len(specs) - 1] = {"train": concat([tr] + list(scans.values())), "test": te, "scans": scans}
    return specs, data


def run(a):
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
            pickle.dump(
                {"P": jax.tree_util.tree_map(np.asarray, P), "keys": keys},
                open(os.path.join(RES, f"{a.name}_{'-'.join(specs[i].name for i in g)[:60]}.params.pkl"), "wb"),
            )
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
                f"(train {mtr[k]['E_MAE']:.3f}/{mtr[k]['F_MAE']:.2f})  scan max {np.max(sc) if sc else float('nan'):.2f}"
                f"  params {npar}{'  HELD OUT coverage %.2f' % r['coverage'] if k in hold else ''}",
                flush=True,
            )
    out["time_s"] = time.time() - t0
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, f"{a.name}.json"), "w"), indent=1)
    summarize([a.name])


def l1path(a):
    """Error vs number of active linear parameters along an L1 path (per-molecule fits)."""
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
                f"  {spec.name:20s} l1 {lam:8.1e}  active {nz:4d}/{nlin:4d}  E_MAE {mt['E_MAE']:.3f}  F_MAE {mt['F_MAE']:.2f}",
                flush=True,
            )
        out["molecules"][spec.name] = rows
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, f"{a.name}.json"), "w"), indent=1)


def scan_max(v):
    """Worst relaxed-scan error of one molecule (single-point profile if not relaxed); inf if a
    relaxation left the basin."""
    xs = []
    for x in v["scans"].values():
        if "error" in x:
            continue
        xs.append(x["relaxed_max"] if "relaxed_max" in x else (np.inf if x.get("relax_ok") is False else x["sp_max"]))
    return max(xs) if xs else np.nan


def summarize(names, exclude=("methanethiol",)):
    for n in names:
        d = json.load(open(os.path.join(RES, f"{n}.json")))
        ms = {k: v for k, v in d["molecules"].items() if k not in exclude}
        e = [v["test"]["E_MAE"] for v in ms.values()]
        f = [v["test"]["F_MAE"] for v in ms.values()]
        s = np.array([scan_max(v) for v in ms.values()])
        fin = s[np.isfinite(s)]
        print(
            f"{n:28s} mols {len(ms):2d}  E_MAE {np.mean(e):.3f}  F_MAE {np.mean(f):.2f}  scan max {np.mean(fin) if len(fin) else np.nan:.2f} kcal/mol"
            f"  (relax failures {int(np.sum(np.isinf(s)))})"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("name", nargs="+")
    ap.add_argument("--mols", default="A1")
    ap.add_argument("--families", default="paper")
    ap.add_argument("--typing", default="molecule")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--elec", type=int, default=0)
    ap.add_argument("--lj_sep", type=int, default=4)
    ap.add_argument("--lj14", type=float, default=0.0)
    ap.add_argument("--elec14", type=float, default=1.0)
    ap.add_argument("--flux", type=int, nargs="?", const=1, default=0)
    ap.add_argument("--escale", default="", help="learned permanent-pair scales for these separations, e.g. 1,2,3")
    ap.add_argument("--wE", type=float, default=1.0)
    ap.add_argument("--wF", type=float, default=1.0)
    ap.add_argument("--wmu", type=float, default=0.0)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--l1", type=float, default=0.0)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--frozen", default="")
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--holdout", default="")
    ap.add_argument("--no_scans", action="store_true")
    ap.add_argument("--no_relax", action="store_true")
    ap.add_argument("--save_params", action="store_true")
    ap.add_argument("--qfit", type=int, default=-1, help="fit typed pGM charges/covalent dipoles (typing depth)")
    ap.add_argument("--l2_elec", type=float, default=None)
    ap.add_argument(
        "--qbci", type=int, default=-1, help="fit typed bond-charge increments on the ESP charges (typing depth)"
    )
    ap.add_argument("--wesp", type=float, default=0.0, help="weight of the ESP restraint (relative RMSE / 0.1)^2")
    ap.add_argument("--lams", default="0,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1")
    a = ap.parse_args()
    if a.cmd in ("run", "l1path"):
        a.name = a.name[0]
        (run if a.cmd == "run" else l1path)(a)
    else:
        summarize(a.name)
