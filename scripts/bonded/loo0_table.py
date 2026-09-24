"""Leave-one-molecule-out with element-level typing (depth 0): held-out test errors."""
import glob, json, os
import numpy as np
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs/bonded/results")
rows = {}
for f in sorted(glob.glob(os.path.join(RES, "loo0_*.json"))):
    tag = os.path.basename(f)[5:-5]
    if tag == "summary":
        continue
    head, fam, el = tag.rsplit("_", 2)
    d = json.load(open(f))
    r = d["molecules"][head]
    rows.setdefault(fam, {}).setdefault(head, {})[el] = (r["test"]["E_MAE"], r["test"]["F_MAE"], r.get("coverage", np.nan))
out = {}
for fam, mols in rows.items():
    print(f"== family {fam}: held-out molecule, element-typed parameters fitted on the other 11")
    els = [e for e in ("pgm", "cls", "x13") if any(e in v for v in mols.values())]
    print(f"{'held out':20s} {'coverage':>8s} " + " ".join(f"{'E ' + e:>8s}" for e in els) + " " + " ".join(f"{'F ' + e:>8s}" for e in els))
    E = {e: [] for e in els}; F = {e: [] for e in els}
    for mol, v in mols.items():
        if not all(e in v for e in els):
            continue
        cov = next(v[e][2] for e in els)
        print(f"{mol:20s} {cov:8.2f} " + " ".join(f"{v[e][0]:8.3f}" for e in els) + " " + " ".join(f"{v[e][1]:8.2f}" for e in els))
        for e in els:
            E[e].append(v[e][0]); F[e].append(v[e][1])
    if E[els[0]]:
        print(f"{'mean':20s} {'':8s} " + " ".join(f"{np.mean(E[e]):8.3f}" for e in els) + " " + " ".join(f"{np.mean(F[e]):8.2f}" for e in els) + f"  (n={len(E[els[0]])})")
    out[fam] = {"molecules": mols}
json.dump(out, open(os.path.join(RES, "loo0_summary.json"), "w"), indent=1)
