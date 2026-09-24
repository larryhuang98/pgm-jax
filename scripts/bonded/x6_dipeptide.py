"""X6: alanine dipeptide phi/psi surface.  Bonded terms (1-2, 1-3, 1-4 and couplings) are fitted on
500 K MD frames + every other point of the MACE-relaxed 15-degree phi/psi grid (DFT energies and
forces); the surface is tested on the other half (single points at the reference geometries).
Nonbonded: pGM all pairs (or a control) + GAFF LJ from 1-5.

    python scripts/bonded/x6_dipeptide.py NAME --families paper [--elec 3] ...
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts/bonded"))
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from experiments import FAMILY_SETS, concat  # noqa: E402
from pgm_jax.bonded.data import frames, mol_spec  # noqa: E402
from pgm_jax.bonded.fit import KCAL, Fitter  # noqa: E402
from pgm_jax.bonded.model import BondedModel, BondedSettings  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--families", default="paper")
ap.add_argument("--elec", type=int, default=0)
ap.add_argument("--elec14", type=float, default=1.0)
ap.add_argument("--lj14", type=float, default=0.0)
ap.add_argument("--maxiter", type=int, default=20000)
ap.add_argument("--no_md", action="store_true")
ap.add_argument("--no_grid", action="store_true", help="train on the 500 K MD frames only (the grid is all test)")
ap.add_argument("--ind", type=int, default=-1, help="induction exclusion (default: as --elec)")
ap.add_argument("--l1", type=float, default=3e-3)
a = ap.parse_args()
name = "alanine_dipeptide"
spec = mol_spec(name)
grid = frames(name, "scan2d")
ang = grid.extra["angle"]
k = np.round((ang + 180.0) / 15.0).astype(int)
train_mask = (k[:, 0] + k[:, 1]) % 2 == 0
g_tr, g_te = grid.subset(np.nonzero(train_mask)[0]), grid.subset(np.nonzero(~train_mask)[0])
md = frames(name, "train500")
te_md = frames(name, "test298")
train = concat(([] if a.no_grid else [g_tr]) + ([md] if (md is not None and not a.no_md) else []))
st = BondedSettings(families=FAMILY_SETS.get(a.families, tuple(a.families.split("+"))), elec_exclude=a.elec, ind_exclude=a.ind,
                    elec14_scale=a.elec14, lj14_scale=a.lj14)
model = BondedModel([spec], st)
data = {0: {"train": train, "test": g_te, "grid": grid}}
if te_md is not None:
    data[0]["md"] = te_md
fit = Fitter(model, {0: {"train": train, "test": g_te}})
P = fit.fit(model.init_params(), maxiter=a.maxiter, l1=a.l1)
E = np.asarray(jax.jit(jax.vmap(lambda X: model.energy(0, X, P)[0]))(jnp.asarray(grid.X)))
ref = grid.E
m_ref = np.argmin(ref)
d_ref, d_ff = (ref - ref[m_ref]) / KCAL, (E - E[m_ref]) / KCAL
win = d_ref < 7.0                                                        # the paper contours within 7 kcal/mol
err = d_ff - d_ref
out = {"name": a.name, "args": vars(a), "n_params": model.n_params(P),
       "test_half": {"MAE": float(np.mean(np.abs(err[~train_mask]))), "RMSE": float(np.sqrt(np.mean(err[~train_mask] ** 2))),
                     "max": float(np.max(np.abs(err[~train_mask]))), "MAE_below7": float(np.mean(np.abs(err[~train_mask & win])))},
       "all": {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))), "max": float(np.max(np.abs(err)))},
       "angles": ang.tolist(), "ref": d_ref.tolist(), "ff": d_ff.tolist(), "train_mask": train_mask.tolist()}
ev = Fitter(model, {0: {"test": te_md}}) if te_md is not None else None
if ev is not None:
    out["md_test"] = ev.metrics(P, "test")[0]
os.makedirs(os.path.join(ROOT, "runs/bonded/results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "runs/bonded/results", f"{a.name}.json"), "w"), indent=1)
print(a.name, "surface test half:", {k2: round(v, 3) for k2, v in out["test_half"].items()}, "all:",
      {k2: round(v, 3) for k2, v in out["all"].items()}, "md test:", out.get("md_test"), "params", out["n_params"], flush=True)
