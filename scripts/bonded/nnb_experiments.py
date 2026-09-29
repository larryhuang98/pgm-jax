"""Neural bonded terms (NNB, pgm_jax/bonded/nn.py): accuracy per molecule and transfer to held-out
molecules, against the class II set, on the leave-one-out molecule set of the bonded study.
Frames: MACE-OFF 500 K training frames (no torsion scans) and 298 K test frames, DFT labels.

    python scripts/bonded/nnb_experiments.py permol  [--ref geometry]
    python scripts/bonded/nnb_experiments.py loo     [--ref geometry|predicted]
Results: runs/bonded/nnb/<mode>_<ref>.json"""

import argparse
import json
import os
import time

import jax
import numpy as np

from pgm_jax.bonded.fit import Fitter
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.bonded.study.data import load
from pgm_jax.bonded.study.families import families_of

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jax.config.update("jax_enable_x64", True)
MOLS = [
    "ethane",
    "methanol",
    "methylamine",
    "acetaldehyde",
    "formic_acid",
    "formamide",
    "fluorochloroethane",
    "chloroformic_acid",
    "chloromethanol",
    "acetate",
    "methylammonium",
    "hydrogen_phosphate",
]
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["permol", "loo"])
ap.add_argument("--ref", default="geometry")
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--maxiter", type=int, default=4000)
ap.add_argument("--adam", type=int, default=3000, help="Adam steps before L-BFGS (neural set)")
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--mols", default=",".join(MOLS))
ap.add_argument("--th-span", type=float, default=0.35)
ap.add_argument("--out-scale", type=float, default=2.0)
ap.add_argument("--tag", default="")
ap.add_argument("--basis", default="paper", help="families the network parameterises: a FAMILY_SETS name or f1+f2+...")
ap.add_argument("--l2", type=float, default=1e-4)
ap.add_argument("--no-pgm-features", action="store_true")
ap.add_argument("--table-depth", type=int, default=None, help="typed table + network residual (0 = element-typed)")
ap.add_argument("--resid-l2", type=float, default=0.0)
ap.add_argument("--holds", default="", help="loo: comma-separated indices of the held-out molecules to run")
a = ap.parse_args()
names = a.mols.split(",")
specs, data = load(names, with_scans=False)
out_path = os.path.join(ROOT, f"runs/bonded/nnb/{a.mode}_{a.ref}{a.tag}.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
res = {"args": vars(a), "molecules": {}}
nn = BondedSettings(
    families=("nnb",),
    nn_width=a.width,
    nn_layers=a.layers,
    nn_ref=a.ref,
    nn_th_span=a.th_span,
    nn_out_scale=a.out_scale,
    nn_pgm_features=not a.no_pgm_features,
    nn_basis=families_of(a.basis),
    nn_table_depth=a.table_depth,
    nn_resid_l2=a.resid_l2,
)


def run(model, train_idx, test_idx):
    fit = Fitter(
        model,
        l2=a.l2,
        data={
            k: ({"train": data[i]["train"]} if k in train_idx else {}) | {"test": data[i]["test"]}
            for k, i in enumerate(model_mols)
        },
    )
    t0 = time.time()
    P = fit.fit(
        model.init_params(), maxiter=a.maxiter, verbose=True, adam_steps=a.adam if model.nnb is not None else 0, lr=a.lr
    )
    return fit.metrics(P, "test"), time.time() - t0, model.n_params(P)


if a.mode == "permol":
    for i, s in enumerate(specs):
        row = {}
        for tag, st in (("nnb", nn), ("paper", BondedSettings(families=nn.nn_basis))):
            model_mols = [i]
            model = BondedModel([s], st)
            m, dt, npar = run(model, [0], [0])
            row[tag] = {**m[0], "seconds": dt, "n_params": npar}
        res["molecules"][s.name] = row
        print(s.name, {k: (round(v["E_MAE"], 3), round(v["F_MAE"], 2)) for k, v in row.items()}, flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
else:
    holds = [int(x) for x in a.holds.split(",")] if a.holds else range(len(specs))
    for h in holds:
        s = specs[h]
        model_mols = list(range(len(specs)))
        model = BondedModel(specs, nn)
        train = [k for k in range(len(specs)) if k != h]
        m, dt, npar = run(model, train, list(range(len(specs))))
        res["molecules"][s.name] = {
            "test": m[h],
            "seconds": dt,
            "n_params": npar,
            "train_mean_E": float(np.mean([m[k]["E_MAE"] for k in train])),
            "train_mean_F": float(np.mean([m[k]["F_MAE"] for k in train])),
        }
        print(s.name, round(m[h]["E_MAE"], 3), round(m[h]["F_MAE"], 2), f"{dt:.0f} s", flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
