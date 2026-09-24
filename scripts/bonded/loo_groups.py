"""Leave-one-out transfer by chemical group (energy MAE, kcal/mol)."""
import glob, json, os
import numpy as np
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs/bonded/results")
G = {"carbonyl/carboxyl": ["acetaldehyde", "acetate", "formic_acid", "chloroformic_acid", "formamide"],
     "amine/ammonium/phosphate": ["methylamine", "methylammonium", "hydrogen_phosphate"],
     "other (alkane, alcohol, halides)": ["ethane", "methanol", "chloromethanol", "fluorochloroethane"]}
rows = {}
for f in glob.glob(os.path.join(RES, "loo0_*.json")):
    tag = os.path.basename(f)[5:-5]
    if tag == "summary":
        continue
    head, fam, el = tag.rsplit("_", 2)
    r = json.load(open(f))["molecules"][head]
    rows.setdefault(fam, {}).setdefault(el, {})[head] = (r["test"]["E_MAE"], r["test"]["F_MAE"])
out = {}
for fam in ("diag", "diag+b1", "diag+b1e10", "diag+q1", "diag+q1e", "diag+q1e10", "diag+es", "diag+es14", "diag+ub", "diag+p14", "paper",
            "diag+conj", "diag+hc", "diag+new", "hyb", "hybsc", "chem", "chem+hyb", "dist", "dist+chem", "diag+ovl"):
    if fam not in rows:
        continue
    print(f"== {fam}")
    for g, mols in list(G.items()) + [("all 12", sum(G.values(), []))]:
        line = f"  {g:34s}"
        for el in ("pgm", "cls", "x13"):
            v = [rows[fam].get(el, {}).get(m) for m in mols]
            if all(x is not None for x in v):
                line += f"  {el} E {np.mean([x[0] for x in v]):5.2f} F {np.mean([x[1] for x in v]):5.1f}"
                out.setdefault(fam, {}).setdefault(g, {})[el] = [float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))]
        print(line)
json.dump(out, open(os.path.join(RES, "loo_groups.json"), "w"), indent=1)
