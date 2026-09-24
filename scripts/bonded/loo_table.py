"""Leave-one-molecule-out transfer (typed parameters): held-out test errors, pGM vs classical."""
import glob, json, os
import numpy as np
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs/bonded/results")
rows = {}
for f in sorted(glob.glob(os.path.join(RES, "loo_*.json"))):
    d = json.load(open(f))
    tag = os.path.basename(f)[4:-5]
    mol, el = tag.rsplit("_", 1)
    r = d["molecules"][mol]
    rows.setdefault(mol, {})[el] = (r["test"]["E_MAE"], r["test"]["F_MAE"], r.get("coverage", np.nan))
print(f"{'held out':20s} {'coverage':>8s} {'E pGM':>7s} {'E cls':>7s} {'F pGM':>7s} {'F cls':>7s}")
E = {"pgm": [], "cls": []}; F = {"pgm": [], "cls": []}
for mol, v in rows.items():
    if "pgm" not in v or "cls" not in v:
        continue
    print(f"{mol:20s} {v['pgm'][2]:8.2f} {v['pgm'][0]:7.3f} {v['cls'][0]:7.3f} {v['pgm'][1]:7.2f} {v['cls'][1]:7.2f}")
    for k in ("pgm", "cls"):
        E[k].append(v[k][0]); F[k].append(v[k][1])
print(f"{'mean':20s} {'':8s} {np.mean(E['pgm']):7.3f} {np.mean(E['cls']):7.3f} {np.mean(F['pgm']):7.2f} {np.mean(F['cls']):7.2f}")
