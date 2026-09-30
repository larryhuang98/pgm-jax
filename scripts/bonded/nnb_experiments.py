"""Neural bonded terms (NNB, pgm_jax/bonded/nn.py) against the class II set on the leave-one-out molecules.

Accuracy per molecule (permol: NNB and the class II set of the network's basis fitted to one
molecule) and transfer to held-out molecules (loo: NNB trained on all but one molecule), on the
leave-one-out molecule set of the bonded study.  Frames: MACE-OFF 500 K training frames (no torsion
scans) and 298 K test frames, DFT labels.

Usage:

    python scripts/bonded/nnb_experiments.py permol  [--ref geometry]
    python scripts/bonded/nnb_experiments.py loo     [--ref geometry|predicted] [--holds 0,3]
    python scripts/bonded/nnb_experiments.py --help

Inputs: the bonded-study data (pgm_jax.bonded.study.data).
Outputs: runs/bonded/nnb/<mode>_<ref><tag>.json (rewritten after each molecule); printed MAEs.
Units: energy MAE [kcal/mol], force MAE [kcal/mol/A] (Fitter.metrics).
Runtime: CPU, minutes per fit.  Sets jax_enable_x64.
"""

from __future__ import annotations

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
from pgm_jax.paths import repo_path

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


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["permol", "loo"], help="permol: one molecule per fit; loo: leave one molecule out")
    ap.add_argument(
        "--ref", default="geometry", help="reference geometry of the network features (geometry or predicted)"
    )
    ap.add_argument("--width", type=int, default=32, help="hidden width of the network")
    ap.add_argument("--layers", type=int, default=3, help="hidden layers")
    ap.add_argument("--maxiter", type=int, default=4000, help="L-BFGS iterations")
    ap.add_argument("--adam", type=int, default=3000, help="Adam steps before L-BFGS (neural set)")
    ap.add_argument("--lr", type=float, default=3e-3, help="Adam learning rate")
    ap.add_argument("--mols", default=",".join(MOLS), help="comma-separated molecules")
    ap.add_argument("--th-span", type=float, default=0.35, help="BondedSettings.nn_th_span")
    ap.add_argument("--out-scale", type=float, default=2.0, help="BondedSettings.nn_out_scale")
    ap.add_argument("--tag", default="", help="suffix of the output name")
    ap.add_argument(
        "--basis", default="paper", help="families the network parameterises: a FAMILY_SETS name or f1+f2+..."
    )
    ap.add_argument("--l2", type=float, default=1e-4, help="L2 weight of the fit")
    ap.add_argument("--no-pgm-features", action="store_true", help="no pGM features in the network input")
    ap.add_argument("--table-depth", type=int, default=None, help="typed table + network residual (0 = element-typed)")
    ap.add_argument(
        "--resid-l2", type=float, default=0.0, help="L2 shrinkage of the network residual (BondedSettings.nn_resid_l2)"
    )
    ap.add_argument("--holds", default="", help="loo: comma-separated indices of the held-out molecules to run")
    return ap


def run(model: BondedModel, model_mols: list[int], train_idx: list[int], data: list, a: argparse.Namespace):
    """Fit a model and return (test metrics per model molecule, seconds, number of parameters).

    Parameters
    ----------
    model : BondedModel
        The model; its molecule k is data[model_mols[k]].
    model_mols : list of int
        Data index of each model molecule.
    train_idx : list of int
        Model molecules whose training frames enter the fit (all molecules are tested).
    data : list
        Frames of every molecule (bonded.study.data.load).
    a : argparse.Namespace
        Options (l2, maxiter, adam, lr).
    """
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


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the permol or loo experiment (see the module docstring)."""
    a = build_parser().parse_args(argv)
    names = a.mols.split(",")
    specs, data = load(names, with_scans=False)
    out_path = repo_path("runs", "bonded", "nnb", f"{a.mode}_{a.ref}{a.tag}.json")
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

    if a.mode == "permol":
        for i, s in enumerate(specs):
            row = {}
            for tag, st in (("nnb", nn), ("paper", BondedSettings(families=nn.nn_basis))):
                model = BondedModel([s], st)
                m, dt, npar = run(model, [i], [0], data, a)
                row[tag] = {**m[0], "seconds": dt, "n_params": npar}
            res["molecules"][s.name] = row
            print(s.name, {k: (round(v["E_MAE"], 3), round(v["F_MAE"], 2)) for k, v in row.items()}, flush=True)
            with open(out_path, "w") as fh:
                json.dump(res, fh, indent=1)
    else:
        holds = [int(x) for x in a.holds.split(",")] if a.holds else range(len(specs))
        for h in holds:
            s = specs[h]
            model = BondedModel(specs, nn)
            train = [k for k in range(len(specs)) if k != h]
            m, dt, npar = run(model, list(range(len(specs))), train, data, a)
            res["molecules"][s.name] = {
                "test": m[h],
                "seconds": dt,
                "n_params": npar,
                "train_mean_E": float(np.mean([m[k]["E_MAE"] for k in train])),
                "train_mean_F": float(np.mean([m[k]["F_MAE"] for k in train])),
            }
            print(s.name, round(m[h]["E_MAE"], 3), round(m[h]["F_MAE"], 2), f"{dt:.0f} s", flush=True)
            with open(out_path, "w") as fh:
                json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
