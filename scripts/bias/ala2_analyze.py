"""Analysis of the alanine dipeptide runs of scripts/bias/ala2.py: F(phi) from umbrella windows
(WHAM, the reference), from metadynamics (c(t) reweighting and the final bias) and OPES
(reweighting and the final bias) with errors from independent runs; F(phi, psi) of the methods;
Delta G(alpha_L/C7ax: phi > 0 vs phi < 0).

    python scripts/bias/ala2_analyze.py --us runs/ala2/us --metad runs/ala2/md --opes runs/ala2/op \
        --walkers 8 --out runs/ala2/compare.json

Each PREFIX is a walker set: PREFIX_wNN.colvar (+ PREFIX_wNN.hills for independent metaD walkers,
PREFIX.hills for shared ones) and PREFIX.json for the umbrella centres."""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pgm_jax.bias import analysis as A  # noqa: E402
from pgm_jax.bias.core import KB  # noqa: E402
from pgm_jax.bias.io import read_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--us", nargs="*", default=[], help="umbrella walker sets (PREFIX.json: centres)")
ap.add_argument("--metad", nargs="*", default=[])
ap.add_argument("--opes", nargs="*", default=[])
ap.add_argument("--plain", nargs="*", default=[])
ap.add_argument("--remd", nargs="*", default=[], help="REMD prefixes (PREFIX_T00.nc, PREFIX.json with the phi/psi atoms)")
ap.add_argument("--T", type=float, default=300.0)
ap.add_argument("--biasfactor", type=float, default=6.0)
ap.add_argument("--skip", type=float, default=0.2, help="fraction of each biased run discarded")
ap.add_argument("--us-skip", type=float, default=0.1)
ap.add_argument("--bins", type=int, default=36)
ap.add_argument("--bins2", type=int, default=24)
ap.add_argument("--fmax", type=float, default=25.0)
ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--out", required=True)
a = ap.parse_args()
kT = KB * a.T
ax = A.periodic_axis(a.bins)
ax2 = A.periodic_axis(a.bins2)
P = 2 * np.pi
res = {"T": a.T, "bins_phi_deg": 360 / a.bins}


def walkers(prefix):
    files = sorted(glob.glob(prefix + "_w[0-9][0-9].colvar"))
    return files if files else [prefix + ".colvar"]


def dG(phi, lw):
    w = np.exp(lw - lw.max())
    return -kT * np.log(w[phi > 0].sum() / w[phi <= 0].sum())


def marg(F2):
    """F(phi) from F(phi, psi) on the 2D grid."""
    p = np.exp(-(F2 - np.nanmin(F2[np.isfinite(F2)])) / kT)
    p[~np.isfinite(F2)] = 0.0
    with np.errstate(divide="ignore"):
        F = -kT * np.log(p.sum(1))
    return F - F[np.isfinite(F)].min()


# ---- umbrella / WHAM reference (errors from blocks of every window)
if a.us:
    samples, centers, kappa = [], [], None
    for prefix in a.us:
        meta = json.load(open(prefix + ".json"))
        centers += list(meta["centers"])
        kappa = meta["kappa"]
        for f in walkers(prefix):
            _, c = read_table(f)
            x = c["phi"]
            samples.append(x[int(a.us_skip * len(x)):])
    meta = {"centers": centers, "kappa": kappa}
    F, fk = A.wham(samples, meta["centers"], np.full(len(samples), meta["kappa"]), ax, kT, period=P)
    Fb = []
    for b in range(a.blocks):
        sb = [np.array_split(x, a.blocks)[b] for x in samples]
        Fb.append(A.wham(sb, meta["centers"], np.full(len(samples), meta["kappa"]), ax, kT, period=P)[0])
    Fb = np.array(Fb)
    Fb = np.array([A.align_rmsd(f, F, F < a.fmax)[2] for f in Fb])
    err = Fb.std(0, ddof=1) / np.sqrt(a.blocks)
    # Delta G from the WHAM F(phi) (bins are uniform: sum of exp(-F/kT))
    pphi = np.exp(-F / kT)
    dg = -kT * np.log(pphi[ax > 0].sum() / pphi[ax <= 0].sum())
    dgb = [-kT * np.log(np.exp(-f / kT)[ax > 0].sum() / np.exp(-f / kT)[ax <= 0].sum()) for f in Fb]
    res["wham"] = {"F": F.tolist(), "err": err.tolist(), "dG_aL": dg, "dG_aL_err": float(np.std(dgb, ddof=1) / np.sqrt(a.blocks)),
                   "samples_per_window": int(np.mean([len(x) for x in samples])), "windows": len(samples)}
    print(f"WHAM ({len(samples)} windows, {res['wham']['samples_per_window']} samples each): dG(phi>0) = {dg:.2f} +- "
          f"{res['wham']['dG_aL_err']:.2f} kJ/mol; mean error bar of F(phi) (F < {a.fmax}) {err[F < a.fmax].mean():.3f}")
    Fref, eref = F, err
else:
    Fref = eref = None
refs = {}
if Fref is not None:
    refs["wham"] = (Fref, eref)


def dihedral_np(X, idx):
    b0, b1, b2 = X[:, idx[1]] - X[:, idx[0]], X[:, idx[2]] - X[:, idx[1]], X[:, idx[3]] - X[:, idx[2]]
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m1 = np.cross(n1, b1 / np.linalg.norm(b1, axis=1)[:, None])
    return np.arctan2(-np.sum(m1 * n2, 1), np.sum(n1 * n2, 1))


if a.remd:
    from pgm_jax.md.io import read_trajectory
    Sphi, Spsi = [], []
    for prefix in a.remd:
        meta = json.load(open(prefix + ".json"))
        files = sorted(glob.glob(prefix + "_T00*.nc"))
        X = np.concatenate([read_trajectory(f)[0] for f in files])
        X = X[int(0.1 * len(X)):]
        Sphi.append(dihedral_np(X, meta["phi"]))
        Spsi.append(dihedral_np(X, meta["psi"]))
    phi_r, psi_r = np.concatenate(Sphi), np.concatenate(Spsi)
    Fr = A.histogram_fes(phi_r, None, [ax], kT, [P])
    blocks = [A.histogram_fes(b, None, [ax], kT, [P]) for b in np.array_split(phi_r, 5)]
    blocks = np.array([A.align_rmsd(f, Fr, Fr < a.fmax)[2] for f in blocks])
    er = blocks.std(0, ddof=1) / np.sqrt(5)
    dgb = [dG(b, np.zeros(len(b))) for b in np.array_split(phi_r, 5)]
    res["remd"] = {"F": Fr.tolist(), "err": er.tolist(), "frames": int(len(phi_r)), "dG_aL": dG(phi_r, np.zeros(len(phi_r))),
                   "dG_aL_err": float(np.std(dgb, ddof=1) / np.sqrt(5)),
                   "F2": A.histogram_fes(np.stack([phi_r, psi_r], 1), None, [ax2, ax2], kT, [P, P]).tolist()}
    refs["remd"] = (Fr, er)
    print(f"REMD 300 K ({len(phi_r)} frames): dG(phi>0) = {res['remd']['dG_aL']:.2f} +- {res['remd']['dG_aL_err']:.2f} kJ/mol")
    if Fref is not None:
        m = (Fref < a.fmax) & np.isfinite(Fr)
        r, mx, _ = A.align_rmsd(Fr, Fref, m)
        print(f"REMD vs WHAM F(phi): RMSD {r:.3f}, max {mx:.2f}")
        res["remd"]["rmsd_vs_wham"] = r


def compare(name, Fs1, dgs, Fs2=None, Fbias=None):
    Fs1 = np.array(Fs1)
    out = {"runs": len(Fs1), "dG_aL_runs": [float(x) for x in dgs], "dG_aL": float(np.mean(dgs)),
           "dG_aL_err": float(np.std(dgs, ddof=1) / np.sqrt(len(dgs))) if len(dgs) > 1 else None}
    for rname, (Fr, er) in refs.items():
        m = (Fr < a.fmax) & np.isfinite(Fr)
        al = np.array([A.align_rmsd(f, Fr, m)[2] for f in Fs1])
        Fm = al.mean(0)
        em = al.std(0, ddof=1) / np.sqrt(len(al)) if len(al) > 1 else np.zeros_like(Fm)
        r, mx, _ = A.align_rmsd(Fm, Fr, m)
        tot = np.sqrt(em ** 2 + er ** 2)
        o = {"F_mean": Fm.tolist(), "F_err": em.tolist(), "rmsd": r, "max": mx,
             "rmsd_runs": [A.align_rmsd(f, Fr, m)[0] for f in Fs1],
             "chi2_per_bin": float(np.mean(((Fm - Fr)[m] / np.maximum(tot[m], 1e-6)) ** 2)), "mean_err": float(tot[m].mean())}
        if Fbias is not None:
            fb = np.array(Fbias)
            o["rmsd_bias"] = A.align_rmsd(np.mean([A.align_rmsd(f, Fr, m)[2] for f in fb], 0), Fr, m)[0]
        out["vs_" + rname] = o
        print(f"{name} vs {rname}: F(phi) RMSD {r:.3f} kJ/mol (max {mx:.2f}; per run {np.mean(o['rmsd_runs']):.3f}), "
              f"chi2/bin {o['chi2_per_bin']:.2f}, mean error {o['mean_err']:.3f}"
              + (f"; from the final bias RMSD {o['rmsd_bias']:.3f}" if Fbias is not None else ""))
    print(f"{name}: dG(phi>0) {out['dG_aL']:.2f} +- {out['dG_aL_err'] if out['dG_aL_err'] is not None else float('nan'):.2f} "
          f"({len(Fs1)} runs)")
    if Fs2 is not None:
        out["F2_mean"] = np.array(Fs2).mean(0).tolist()
    res[name] = out
    return out


def fes2_from_hills(hills, factor):
    """-factor V(phi, psi) on the 2D analysis grid from a hills table (read_table)."""
    C = np.stack([hills["c_phi"], hills["c_psi"]], 1)
    pts, shape = A.mesh(ax, ax)
    V = np.zeros(len(pts))
    for i in range(0, len(C), 512):
        V += A._hill_values(C[i:i + 512], hills["height"][i:i + 512], [hills["sigma_phi"][0], hills["sigma_psi"][0]],
                            [P, P], pts).sum(0)
    F = -factor * V.reshape(shape)
    return F - F.min()


fine, _ = A.mesh(A.periodic_axis(72), A.periodic_axis(72))
for kind, prefixes in (("metad", a.metad), ("opes", a.opes), ("plain", a.plain)):
    if not prefixes:
        continue
    Fs1, Fs2, dgs, Fb = [], [], [], []
    for prefix in prefixes:
        files = walkers(prefix)
        shared = os.path.exists(prefix + ".hills") and len(files) > 1
        runs = [files] if shared else [[f] for f in files]
        for group in runs:
            S, LW = [], []
            for f in group:
                _, c = read_table(f)
                keep = c["step"] > a.skip * c["step"].max()
                phi, psi = c["phi"], c["psi"]
                if kind == "metad":
                    hf = prefix + ".hills" if shared else f.replace(".colvar", ".hills")
                    _, h = read_table(hf)
                    hills = {"step": h["step"], "center": np.stack([h["c_phi"], h["c_psi"]], 1), "height": h["height"],
                             "sigma": np.array([h["sigma_phi"][0], h["sigma_psi"][0]])}
                    hs, ct = A.metad_ct(hills, a.biasfactor, kT, [P, P], fine)
                    last = np.append(hs[1:] != hs[:-1], True)
                    lw = A.ct_weights(c["step"], c["bias0_metad"], hs[last], ct[last], kT)
                    if f == group[-1]:
                        Fb.append(marg(fes2_from_hills(h, a.biasfactor / (a.biasfactor - 1.0))))
                elif kind == "opes":
                    lw = A.opes_weights(c["bias0_opes"], kT)
                else:
                    lw = np.zeros(len(phi))
                S.append(np.stack([phi, psi], 1)[keep])
                LW.append(lw[keep])
            S, LW = np.concatenate(S), np.concatenate(LW)
            F1 = A.histogram_fes(S[:, 0], LW, [ax], kT, [P])
            F2 = A.histogram_fes(S, LW, [ax2, ax2], kT, [P, P])
            Fs1.append(F1)
            Fs2.append(F2)
            dgs.append(dG(S[:, 0], LW))
    compare(kind, Fs1, dgs, Fs2, Fb if Fb else None)

if "metad" in res and "opes" in res:
    F1, F2 = np.array(res["metad"]["F2_mean"]), np.array(res["opes"]["F2_mean"])
    m = (F1 < 15.0) & (F2 < 15.0) & np.isfinite(F1) & np.isfinite(F2)
    r, mx, _ = A.align_rmsd(F1, F2, m)
    res["metad_vs_opes_2d"] = {"rmsd": r, "max": mx, "bins": int(m.sum())}
    print(f"F(phi, psi) metaD vs OPES over {int(m.sum())} bins with F < 15: RMSD {r:.3f}, max {mx:.2f} kJ/mol")
if "remd" in res:
    F3 = np.array(res["remd"]["F2"])
    for kind in ("metad", "opes"):
        if kind in res:
            F1 = np.array(res[kind]["F2_mean"])
            m = (F1 < 15.0) & (F3 < 15.0) & np.isfinite(F1) & np.isfinite(F3)
            r, mx, _ = A.align_rmsd(F1, F3, m)
            res[f"{kind}_vs_remd_2d"] = {"rmsd": r, "max": mx, "bins": int(m.sum())}
            print(f"F(phi, psi) {kind} vs REMD over {int(m.sum())} bins with F < 15: RMSD {r:.3f}, max {mx:.2f} kJ/mol")
json.dump(res, open(a.out, "w"), indent=1)
