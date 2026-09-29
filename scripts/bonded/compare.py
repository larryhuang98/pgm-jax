"""Per-molecule and mean test errors of several result files (molecules common to all)."""

import json
import os
import sys

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs/bonded/results")
names = sys.argv[1:]
excl = {"methanethiol"}
D = {n: json.load(open(os.path.join(RES, f"{n}.json")))["molecules"] for n in names}
common = [m for m in D[names[0]] if all(m in D[n] for n in names) and m not in excl]
w = max(len(n) for n in names)
print(f"{'':{w}s} " + " ".join(f"{m[:10]:>10s}" for m in common) + "      mean")
for key in ("E_MAE", "F_MAE"):
    print(f"-- test {key} ({'kcal/mol' if key == 'E_MAE' else 'kcal/mol/A'})")
    for n in names:
        v = [D[n][m]["test"][key] for m in common]
        print(f"{n:{w}s} " + " ".join(f"{x:10.3f}" for x in v) + f"  {np.mean(v):8.3f}")


def smax(v):
    xs = []
    for x in v["scans"].values():
        if "error" in x:
            continue
        xs.append(
            x["relaxed_max"]
            if "relaxed_max" in x
            else (np.inf if x.get("relax_ok") is False else x.get("sp_max", np.nan))
        )
    return max(xs) if xs else np.nan


if any(D[n][m]["scans"] for n in names for m in common):
    print("-- relaxed torsion scan, max |error| (kcal/mol; inf = the relaxation left the basin)")
    for n in names:
        v = [smax(D[n][m]) for m in common]
        fin = [x for x in v if np.isfinite(x)]
        print(
            f"{n:{w}s} "
            + " ".join(f"{x:10.2f}" for x in v)
            + f"  {np.mean(fin) if fin else np.nan:8.2f}  fails {sum(1 for x in v if np.isinf(x))}"
        )
print("-- parameters (per group)")
for n in names:
    print(f"{n:{w}s} " + " ".join(f"{D[n][m]['n_params_group']:10d}" for m in common))
if any("mu_RMSE_D" in D[n][common[0]]["test"] for n in names):
    print("-- dipole RMSE (D)")
    for n in names:
        v = [D[n][m]["test"].get("mu_RMSE_D", np.nan) for m in common]
        print(f"{n:{w}s} " + " ".join(f"{x:10.3f}" for x in v) + f"  {np.nanmean(v):8.3f}")
