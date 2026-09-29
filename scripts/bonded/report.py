"""Figures and tables for reports/bonded/ from runs/bonded/results/*.json."""

import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "runs/bonded/results")
OUT = os.path.join(ROOT, "reports/bonded")
os.makedirs(OUT, exist_ok=True)
EXCL = {"methanethiol"}
C = {"pgm": "#2f6db3", "cls": "#d4843a", "amber": "#8c8c8c", "flux": "#3a9b6b"}


def load(n):
    p = os.path.join(RES, f"{n}.json")
    return json.load(open(p))["molecules"] if os.path.exists(p) else None


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


def means(n, mols=None):
    d = load(n)
    if d is None:
        return None
    ms = [m for m in d if m not in EXCL and (mols is None or m in mols)]
    s = np.array([smax(d[m]) for m in ms])
    return {
        "E": np.mean([d[m]["test"]["E_MAE"] for m in ms]),
        "F": np.mean([d[m]["test"]["F_MAE"] for m in ms]),
        "S": np.nanmean(s[np.isfinite(s)]) if np.isfinite(s).any() else np.nan,
        "mu": np.mean([d[m]["test"].get("mu_RMSE_D", np.nan) for m in ms]),
        "P": np.mean([d[m]["n_params_group"] for m in ms]),
        "n": len(ms),
    }


def table_main():
    rows = [
        ("class I (diag)", "b2_diag_pgm", "b2_diag_cls", "b2_diag_amber"),
        ("class II (paper)", "b2_paper_pgm", "b2_paper_cls", "b2_paper_amber"),
    ]
    rows += [
        ("class I + Urey-Bradley + 1-4 pair (F2)", "b2_diagub_pgm", "b2_diagub_cls", None),
        ("class II + extended couplings + pairs", "b2_all_pgm", "b2_all_cls", None),
        ("class I + twist (F9)", "b2_diagtw_pgm", "b2_diagtw_cls", None),
        ("class II + twist (F9)", "b2_papertw_pgm", "b2_papertw_cls", None),
    ]
    fam = [
        ("F3 torsion_mod (factorised)", "b2_tmod_pgm"),
        ("class II + 1-4 pair", "b2_pair14_pgm"),
        ("class II + charge/CBV flux (F6)", "b2_flux_pgm"),
        ("class I + charge/CBV flux (F6)", "b2_diagflux_pgm"),
        ("all + charge/CBV flux", "b2_allflux_pgm"),
        ("class I + learned 1-2/1-3/1-4 pair scales (F10)", "b2_diag_es_pgm"),
        ("class II + learned 1-2/1-3/1-4 pair scales (F10)", "b2_paper_es_pgm"),
    ]
    lines = [
        "| Bonded form | Electrostatics | Energy MAE (kcal/mol) | Force MAE (kcal/mol/A) | Relaxed-scan max error (kcal/mol) | Dipole RMSE (D) | Parameters per molecule |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, *names in rows:
        for n, el in zip(names, ("pGM, all pairs", "classical, 1-2/1-3/1-4 excluded", "Amber-like, 1-4 scaled")):
            m = means(n) if n else None
            if m:
                lines.append(
                    f"| {label} | {el} | {m['E']:.2f} | {m['F']:.1f} | {m['S']:.2f} | {m['mu']:.2f} | {m['P']:.0f} |"
                )
    for label, n in fam:
        m = means(n)
        if m:
            lines.append(
                f"| {label} | pGM, all pairs | {m['E']:.2f} | {m['F']:.1f} | {m['S']:.2f} | {m['mu']:.2f} | {m['P']:.0f} |"
            )
    return "\n".join(lines)


def fig_scans():
    runs = [
        ("b2_diag_amber", "class I, Amber-like", C["amber"], "--"),
        ("b2_paper_cls", "class II, classical", C["cls"], "-"),
        ("b2_paper_pgm", "class II, pGM", C["pgm"], "-"),
    ]
    D = {n: load(n) for n, *_ in runs}
    if any(v is None for v in D.values()):
        return
    mols = [
        "methanol",
        "formic_acid",
        "chloroformic_acid",
        "fluorochloroethane",
        "chloromethanol",
        "hydrogen_phosphate",
        "acetaldehyde",
        "methylamine",
        "formamide",
    ]
    fig, axs = plt.subplots(3, 3, figsize=(10, 8.5))
    for ax, m in zip(axs.ravel(), mols):
        sc = D["b2_paper_pgm"][m]["scans"].get("scan0")
        if not sc:
            ax.set_visible(False)
            continue
        ang = np.array(sc["angle"])
        o = np.argsort(ang)
        ax.plot(ang[o], np.array(sc["profile_ref"])[o], "k.", ms=6, label="DFT")
        for n, lab, col, ls in runs:
            s = D[n][m]["scans"].get("scan0", {})
            if "profile_ff" in s:
                ax.plot(ang[o], np.array(s["profile_ff"])[o], ls, color=col, lw=1.6, label=lab)
        ax.set_title(m.replace("_", " "), fontsize=10)
        ax.set_xlabel("dihedral (deg)", fontsize=8)
        ax.set_ylabel("kcal/mol", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scans.png"), dpi=130)
    plt.close(fig)


def fig_l1():
    fs = [
        ("l1_paper_pgm", "class II, pGM", C["pgm"]),
        ("l1_paper_cls", "class II, classical", C["cls"]),
        ("l1_all_pgm", "all families, pGM", C["flux"]),
    ]
    D = {
        n: (
            json.load(open(os.path.join(RES, f"{n}.json")))["molecules"]
            if os.path.exists(os.path.join(RES, f"{n}.json"))
            else None
        )
        for n, *_ in fs
    }
    D = {k: v for k, v in D.items() if v}
    if not D:
        return
    mols = list(next(iter(D.values())).keys())
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    for ax, m in zip(axs.ravel(), mols):
        for n, lab, col in fs:
            if n in D and m in D[n]:
                r = D[n][m]
                ax.plot([x["n_active"] for x in r], [x["F_MAE"] for x in r], "o-", color=col, ms=3, label=lab)
        ax.set_title(m.replace("_", " "), fontsize=10)
        ax.set_xlabel("active linear terms", fontsize=8)
        ax.set_ylabel("test force MAE (kcal/mol/A)", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "l1_path.png"), dpi=130)
    plt.close(fig)


def loo():
    rows = {}
    for f in sorted(glob.glob(os.path.join(RES, "loo0_*.json"))):
        tag = os.path.basename(f)[5:-5]
        if tag == "summary":
            continue
        head, fam, el = tag.rsplit("_", 2)
        r = json.load(open(f))["molecules"][head]
        rows.setdefault(fam, {}).setdefault(head, {})[el] = (r["test"]["E_MAE"], r["test"]["F_MAE"])
    lines = [
        "| Bonded form (element-typed) | Held-out molecules | Energy MAE pGM | Energy MAE classical | Force MAE pGM | Force MAE classical | pGM better (energy) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for fam, mols in rows.items():
        pairs = [(v["pgm"], v["cls"]) for v in mols.values() if "pgm" in v and "cls" in v]
        if not pairs:
            continue
        a = np.array(pairs)
        lines.append(
            f"| {fam} | {len(pairs)} | {a[:, 0, 0].mean():.2f} | {a[:, 1, 0].mean():.2f} | {a[:, 0, 1].mean():.1f} | {a[:, 1, 1].mean():.1f} | {int(np.sum(a[:, 0, 0] < a[:, 1, 0]))}/{len(pairs)} |"
        )
    return "\n".join(lines), rows


def fig_loo(rows):
    fams = [f for f in ("diag", "diag+ub", "paper") if f in rows]
    if not fams:
        return
    fig, axs = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 4.2), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, fam in zip(axs, fams):
        mols = [m for m, v in rows[fam].items() if "pgm" in v and "cls" in v]
        y = np.arange(len(mols))
        ax.barh(
            y - 0.2,
            [rows[fam][m]["cls"][0] for m in mols],
            0.4,
            color=C["cls"],
            label="classical (1-2/1-3/1-4 excluded)",
        )
        ax.barh(y + 0.2, [rows[fam][m]["pgm"][0] for m in mols], 0.4, color=C["pgm"], label="pGM (all pairs)")
        ax.set_yticks(y)
        ax.set_yticklabels([m.replace("_", " ") for m in mols], fontsize=8)
        ax.set_title(
            {"diag": "class I", "diag+ub": "class I + Urey-Bradley + 1-4 pair", "paper": "class II"}[fam], fontsize=10
        )
        ax.set_xlabel("held-out energy MAE (kcal/mol)", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Element-typed bonded terms, leave one molecule out", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transfer_loo.png"), dpi=130)
    plt.close(fig)


def fig_forms():
    rows = [
        ("class I", "b2_diag"),
        ("class I + UB + 1-4 pair", "b2_diagub"),
        ("class II", "b2_paper"),
        ("class II + ext. + pairs", "b2_all"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, lab in ((axs[0], "E", "test energy MAE (kcal/mol)"), (axs[1], "F", "test force MAE (kcal/mol/A)")):
        for k, (name, base) in enumerate(rows):
            for j, (el, col) in enumerate((("pgm", C["pgm"]), ("cls", C["cls"]), ("amber", C["amber"]))):
                m = means(f"{base}_{el}")
                if m:
                    ax.bar(k + (j - 1) * 0.27, m[key], 0.27, color=col, label=el if k == 0 else None)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], fontsize=7, rotation=15)
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].legend(["pGM, all pairs", "classical, excluded", "Amber-like, 1-4 scaled"], fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "forms.png"), dpi=130)
    plt.close(fig)


X6_LABELS = {
    "x6_diag_pgm": "class I, pGM",
    "x6_paper_pgm": "class II, pGM",
    "x6_all_pgm": "class II + ext. + pairs, pGM",
    "x6_paper_cls": "class II, classical (excluded)",
    "x6_paper_amber": "class II, Amber-like",
    "x6_all_cls": "class II + ext. + pairs, classical",
}


def fig_dipeptide(names=("x6_paper_pgm", "x6_all_pgm", "x6_paper_amber", "x6_paper_cls")):
    ds = [
        (n, json.load(open(os.path.join(RES, f"{n}.json"))))
        for n in names
        if os.path.exists(os.path.join(RES, f"{n}.json"))
    ]
    if not ds:
        return
    ang = np.array(ds[0][1]["angles"])
    ref = np.array(ds[0][1]["ref"])
    g = np.arange(-180, 180, 15)

    def grid(v):
        Z = np.full((len(g), len(g)), np.nan)
        for (p, q), z in zip(ang, v):
            Z[np.searchsorted(g, p), np.searchsorted(g, q)] = z
        return Z

    fig, axs = plt.subplots(1, 1 + len(ds), figsize=(3.6 * (1 + len(ds)), 3.4))
    for ax, (title, v) in zip(axs, [("DFT // MACE-OFF geometries", ref)] + [(n, d["ff"]) for n, d in ds]):
        im = ax.contourf(g, g, np.clip(grid(v).T, 0, 7), levels=np.arange(0, 7.5, 0.5), cmap="viridis")
        ax.set_title(X6_LABELS.get(title, title), fontsize=9)
        ax.set_xlabel("phi", fontsize=8)
        ax.set_ylabel("psi", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axs, shrink=0.8, label="kcal/mol")
    fig.savefig(os.path.join(OUT, "dipeptide.png"), dpi=130)
    plt.close(fig)


LOO_NAMES = {
    "diag": "class I",
    "diag+ub": "class I + Urey-Bradley + 1-4 pair",
    "diag+p14": "class I + 1-4 pair",
    "paper": "class II",
}
ELEC_NAMES = [
    ("diag", "fixed ESP charges (pGM, all pairs)"),
    ("diag+es", "learned 1-2/1-3/1-4 pair scales (F10)"),
    ("diag+es14", "learned 1-4 scale (F10)"),
    ("diag+q1", "typed charges fitted to E/F/dipoles (F11)"),
    ("diag+q1e", "typed charges, + ESP w = 1"),
    ("diag+q1e10", "typed charges, + ESP w = 10"),
    ("diag+b1", "ESP charges + typed bond-charge increments (F11)"),
    ("diag+b1e10", "ESP charges + bond-charge increments, + ESP w = 10"),
]


def _loo_json():
    p = os.path.join(RES, "loo_groups.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def table_loo_groups():
    d = _loo_json()
    lines = [
        "| Bonded form (element-typed) | Held-out group | pGM, all pairs | classical, excluded | pGM without 1-2/1-3 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for fam, lab in LOO_NAMES.items():
        for g, v in d.get(fam, {}).items():
            cell = lambda el: f"{v[el][0]:.2f} / {v[el][1]:.1f}" if el in v else ""
            lines.append(f"| {lab} | {g} | {cell('pgm')} | {cell('cls')} | {cell('x13')} |")
    return (
        "\n".join(lines)
        + "\n\nEnergy MAE (kcal/mol) / force MAE (kcal/mol/A), means over the held-out molecules of each group."
    )


def table_loo_elec():
    d = _loo_json()
    groups = ["carbonyl/carboxyl", "amine/ammonium/phosphate", "other (alkane, alcohol, halides)", "all 12"]
    lines = [
        "| Electrostatics (class I bonded terms, element-typed) | " + " | ".join(groups) + " |",
        "| --- | " + " | ".join("---" for _ in groups) + " |",
    ]
    for fam, lab in ELEC_NAMES:
        v = d.get(fam, {})
        if not v:
            continue
        lines.append(
            f"| {lab} | "
            + " | ".join(f"{v[g]['pgm'][0]:.2f}" if g in v and "pgm" in v[g] else "" for g in groups)
            + " |"
        )
    return "\n".join(lines) + "\n\nHeld-out energy MAE (kcal/mol), leave one molecule out; the pGM model in every row."


def table_md():
    runs = [
        ("class II, pGM", "md_paper_pgm_298"),
        ("class II, pGM", "md_paper_pgm_500"),
        ("class II, classical", "md_paper_cls_500"),
        ("class I, pGM", "md_diag_pgm_500"),
        ("class II + ext. + pairs, pGM", "md_all_pgm_500"),
    ]
    lines = [
        "| Force field | T (K) | Stable replicas | Bond fluctuation / MACE | Angle fluctuation / MACE | Torsion histogram L1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for lab, n in runs:
        p = os.path.join(RES, f"{n}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        ms = {k: v for k, v in d["molecules"].items() if k not in EXCL}
        st = sum(sum(v["stable"]) for v in ms.values())
        tot = sum(len(v["stable"]) for v in ms.values())
        g = lambda k: np.median([v[k] for v in ms.values() if k in v])
        lines.append(
            f"| {lab} | {d['args']['T']:.0f} | {st}/{tot} | {g('bond_std_ratio'):.2f} | {g('angle_std_ratio'):.2f} | {g('torsion_hist_L1'):.2f} |"
        )
    return "\n".join(lines) + (
        "\n\nMedian over molecules; fluctuations are standard deviations over the MD relative to the "
        "DFT-labelled MACE-OFF frames at the same temperature (298 K test, 500 K training)."
    )


def table_dipeptide():
    names = [
        ("class I", "diag"),
        ("class I + twist", "diagtw"),
        ("class I + UB + 1-4 pair", "diagub"),
        ("class II", "paper"),
        ("class II + twist", "papertw"),
        ("class II + ext. + pairs", "all"),
        ("class II + ext. + pairs + twist", "alltw"),
    ]
    lines = [
        "| Bonded form | Electrostatics | phi/psi test half MAE | RMSE | max | MAE below 7 kcal/mol | 298 K MD energy / force MAE | Parameters |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for lab, tag in names:
        for el, eln in (("pgm", "pGM, all pairs"), ("cls", "classical, excluded"), ("amber", "Amber-like")):
            p = os.path.join(RES, f"x6_{tag}_{el}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            t = d["test_half"]
            md = d.get("md_test") or {}
            mdc = f"{md['E_MAE']:.2f} / {md['F_MAE']:.1f}" if md else ""
            lines.append(
                f"| {lab} | {eln} | {t['MAE']:.2f} | {t['RMSE']:.2f} | {t['max']:.2f} | {t['MAE_below7']:.2f} | {mdc} | {d['n_params']} |"
            )
    return (
        "\n".join(lines)
        + "\n\nkcal/mol (forces kcal/mol/A); surface errors after removing the energy of the global minimum."
    )


def table_rigid():
    cols = [("diag", "pgm"), ("diag", "cls"), ("diag", "amber"), ("paper", "pgm"), ("paper", "cls"), ("paper", "amber")]
    D = {c: load(f"a4_{c[0]}_{c[1]}") for c in cols}
    if all(v is None for v in D.values()):
        return ""
    mols = sorted({m for v in D.values() if v for m in v})
    lines = [
        "| Molecule | class I pGM | class I classical | class I Amber-like | class II pGM | class II classical | class II Amber-like | Dipole RMSE pGM / classical (D) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in mols:
        cell = lambda c: (
            f"{D[c][m]['test']['E_MAE']:.2f} / {D[c][m]['test']['F_MAE']:.1f}" if D[c] and m in D[c] else ""
        )
        dip = (
            f"{D[('paper', 'pgm')][m]['test']['mu_RMSE_D']:.2f} / {D[('paper', 'cls')][m]['test']['mu_RMSE_D']:.2f}"
            if D[("paper", "pgm")] and D[("paper", "cls")] and m in D[("paper", "cls")]
            else ""
        )
        lines.append(f"| {m.replace('_', ' ')} | " + " | ".join(cell(c) for c in cols) + f" | {dip} |")
    return (
        "\n".join(lines) + "\n\nEnergy MAE (kcal/mol) / force MAE (kcal/mol/A) on the 298 K frames, per-molecule fits."
    )


def fig_twist():
    runs = [
        ("b2_paper_pgm", "class II", C["amber"], "--"),
        ("b2_papertw_pgm", "class II + twist", C["pgm"], "-"),
        ("b2_diagtw_pgm", "class I + twist", C["flux"], "-"),
    ]
    D = {n: load(n) for n, *_ in runs}
    if any(v is None for v in D.values()):
        return
    sc = D["b2_paper_pgm"]["formamide"]["scans"]["scan0"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(sc["angle"], sc["profile_ref"], "k.", label="DFT")
    for n, lab, col, ls in runs:
        s = D[n]["formamide"]["scans"]["scan0"]
        ax.plot(s["angle"], s["profile_ff"], ls, color=col, label=f"{lab} (max err {s['relaxed_max']:.1f})")
    ax.set_xlabel("H-N-C=O dihedral (deg)", fontsize=8)
    ax.set_ylabel("kcal/mol", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Formamide NH2 rotation, force-field-relaxed", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "twist_formamide.png"), dpi=130)
    plt.close(fig)


def table_qfit():
    runs = [
        ("ESP charges per molecule (fixed)", "joint_diag_pgm"),
        ("typed charges, E/F/dipole fit", "joint_diag_q1_pgm"),
        ("typed charges, E/F/dipole + ESP (w = 1)", "joint_diag_q1e_pgm"),
        ("typed charges, E/F/dipole + ESP (w = 10)", "joint_diag_q1e10_pgm"),
        ("typed charges, E/F/dipole + ESP (w = 1000)", "joint_diag_q1e1000_pgm"),
        ("depth-2 typed charges, E/F/dipole + ESP (w = 1)", "joint_diag_q2e_pgm"),
        ("ESP charges + typed bond-charge increments, E/F/dipole", "joint_diag_b1_pgm"),
        ("ESP charges + typed bond-charge increments, E/F/dipole + ESP (w = 1)", "joint_diag_b1e_pgm"),
        ("ESP charges + typed bond-charge increments, E/F/dipole + ESP (w = 10)", "joint_diag_b1e10_pgm"),
    ]
    lines = [
        "| Electrostatics (class I bonded terms, element-typed, fitted on all 12) | Energy MAE | Force MAE | Dipole RMSE (D) | ESP RMSE (mhartree/e) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for lab, n in runs:
        d = load(n)
        if d is None:
            continue
        ms = [m for m in d if m not in EXCL]
        g = lambda k: np.mean([d[m]["test"].get(k, np.nan) for m in ms])
        esp = g("esp_RMSE_mEh")
        fx = os.path.join(RES, "esp_fixed.json")
        if n == "joint_diag_pgm" and os.path.exists(fx):  # fixed ESP charges (per-molecule py_resp fits)
            esp = np.mean([v[0] for k, v in json.load(open(fx)).items() if k in ms])
        lines.append(
            f"| {lab} | {g('E_MAE'):.2f} | {g('F_MAE'):.1f} | {g('mu_RMSE_D'):.2f} | {'' if np.isnan(esp) else f'{esp:.2f}'} |"
        )
    return "\n".join(lines) + (
        "\n\nkcal/mol, kcal/mol/A; 298 K frames. ESP: B3LYP/aug-cc-pVTZ potential at the MACE-OFF minimum "
        "(800 points per molecule; the per-molecule py_resp fits range from 0.65 to 3.2 mhartree/e)."
    )


def table_x6_ablation():
    rows = [
        ("all (pGM)", "all", "x6_paper_pgm"),
        ("1-2/1-3 excluded", "all", "x6_paper_p13"),
        ("1-2/1-3/1-4 excluded", "all", "x6_paper_p14"),
        ("all", "1-2/1-3 excluded", "x6_paper_i13"),
        ("1-2/1-3 excluded", "1-2/1-3 excluded", "x6_paper_x13"),
        ("1-2/1-3/1-4 excluded (classical)", "1-2/1-3/1-4 excluded", "x6_paper_cls"),
        ("all", "1-2/1-3/1-4 excluded", "x6_paper_i14"),
    ]
    lines = [
        "| Permanent pair energies | Induction (fields, dipole-dipole couplings) | phi/psi test half MAE | max |",
        "| --- | --- | --- | --- |",
    ]
    for a, b, n in rows:
        p = os.path.join(RES, f"{n}.json")
        if os.path.exists(p):
            t = json.load(open(p))["test_half"]
            lines.append(f"| {a} | {b} | {t['MAE']:.2f} | {t['max']:.1f} |")
    return "\n".join(lines) + "\n\nClass II bonded terms fitted for each variant; kcal/mol."


def table_x6_nogrid():
    rows = [("class I", "diag"), ("class II", "paper")]
    lines = ["| Bonded form | pGM, all pairs | classical, excluded | Amber-like |", "| --- | --- | --- | --- |"]
    for lab, tag in rows:
        cells = []
        for el in ("pgm", "cls", "amber"):
            p = os.path.join(RES, f"x6ng_{tag}_{el}.json")
            cells.append(f"{json.load(open(p))['all']['MAE']:.2f}" if os.path.exists(p) else "")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    return (
        "\n".join(lines)
        + "\n\nphi/psi MAE over the whole grid (kcal/mol), bonded terms trained on the 500 K MD frames only."
    )


NEW_FORMS = [
    ("class I (reference)", "diag", "b2_diag_pgm"),
    ("class I + Urey-Bradley + 1-4 exp (reference)", "diag+ub", "b2_diagub_pgm"),
    ("class I + pi-axis conjugation", "diag+conj", "b4_diag+conj_pgm"),
    ("class I + hyperconjugation (sigma->sigma*, n->sigma*)", "diag+hc", "b4_diag+hc_pgm"),
    ("class I, signed-volume instead of improper", "diag+vol", "b4_diag+vol_pgm"),
    ("class I + conjugation + hyperconjugation + volume", "diag+new", "b4_diag+new_pgm"),
    ("hybrid-orbital angles, fixed (Bent)", "hyb", "b4_hyb_pgm"),
    ("hybrid-orbital angles, self-consistent", "hybsc", "b4_hybsc_pgm"),
    ("class I + Gaussian-overlap 1-3/1-4 repulsion", "diag+ovl", "b4_diag+ovl_pgm"),
    ("no torsions: conjugation + hyperconjugation + volume + 1-4 exp", "chem", "b4_chem_pgm"),
    ("no torsions, self-consistent hybrid angles", "chem+hyb", "b4_chem+hyb_pgm"),
    ("distance only: 1-2 Morse, 1-3/1-4 tanh series, volume", "dist", "b4_dist_pgm"),
    ("distance only + conjugation + hyperconjugation", "dist+chem", "b4_dist+chem_pgm"),
]


def table_new_forms():
    d = _loo_json()
    lines = [
        "| Bonded form | Parameters per molecule | Energy MAE | Force MAE | Relaxed-scan max | Transfer (leave one out), all 12 | Transfer, N/P group | Dipeptide phi/psi, grid-trained | Dipeptide phi/psi, MD-only |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    x6 = lambda n: (lambda p: f"{json.load(open(p))['test_half']['MAE']:.2f}" if os.path.exists(p) else "")(
        os.path.join(RES, f"{n}.json")
    )
    for lab, tag, run in NEW_FORMS:
        m = means(run)
        if m is None:
            continue
        v = d.get(tag, {})
        t_all = f"{v['all 12']['pgm'][0]:.2f}" if "all 12" in v and "pgm" in v["all 12"] else ""
        t_np = (
            f"{v['amine/ammonium/phosphate']['pgm'][0]:.2f}"
            if "amine/ammonium/phosphate" in v and "pgm" in v["amine/ammonium/phosphate"]
            else ""
        )
        xt = {"diag+ub": "diagub"}.get(tag, tag)
        lines.append(
            f"| {lab} | {m['P']:.0f} | {m['E']:.2f} | {m['F']:.1f} | {m['S']:.2f} | {t_all} | {t_np} | {x6('x6_' + xt + '_pgm')} | {x6('x6ng_' + xt + '_pgm')} |"
        )
    return "\n".join(lines) + (
        "\n\npGM electrostatics throughout; kcal/mol and kcal/mol/A. Per-molecule columns: 12 molecules, 298 K "
        "frames. Transfer: element-typed parameters fitted on 11 molecules, held-out energy MAE. Dipeptide: "
        "held-out half of the phi/psi grid, trained with half the grid + MD, or on MD only."
    )


def table_hyb_elec():
    d = _loo_json()
    groups = ["carbonyl/carboxyl", "amine/ammonium/phosphate", "other (alkane, alcohol, halides)", "all 12"]
    lines = [
        "| Angle term (element-typed) | Electrostatics | " + " | ".join(groups) + " |",
        "| --- | --- | " + " | ".join("---" for _ in groups) + " |",
    ]
    for tag, lab in (
        ("diag", "cosine angles (class I)"),
        ("hyb", "hybrid orbitals, fixed"),
        ("hybsc", "hybrid orbitals, self-consistent"),
    ):
        for el, eln in (("pgm", "pGM, all pairs"), ("cls", "classical, excluded")):
            v = d.get(tag, {})
            if not all(g in v and el in v[g] for g in groups):
                continue
            lines.append(f"| {lab} | {eln} | " + " | ".join(f"{v[g][el][0]:.2f}" for g in groups) + " |")
    return "\n".join(lines) + "\n\nHeld-out energy MAE (kcal/mol), leave one molecule out."


BLOCKS = {
    "TABLE_MAIN": table_main,
    "TABLE_LOO": table_loo_groups,
    "TABLE_LOO_ELEC": table_loo_elec,
    "TABLE_MD": table_md,
    "TABLE_DIPEPTIDE": table_dipeptide,
    "TABLE_RIGID": table_rigid,
    "TABLE_QFIT": table_qfit,
    "TABLE_X6ABL": table_x6_ablation,
    "TABLE_X6NG": table_x6_nogrid,
    "TABLE_NEW": table_new_forms,
    "TABLE_HYB": table_hyb_elec,
}


def fill_readme():
    """Replace the content between <!-- NAME --> and <!-- /NAME --> markers of the README."""
    import re

    p = os.path.join(OUT, "README.md")
    if not os.path.exists(p):
        return
    s = open(p).read()
    for k, f in BLOCKS.items():
        s = re.sub(rf"<!-- {k} -->.*?<!-- /{k} -->", lambda _: f"<!-- {k} -->\n{f()}\n<!-- /{k} -->", s, flags=re.S)
    open(p, "w").write(s)


if __name__ == "__main__":
    print(table_main())
    fig_scans()
    fig_l1()
    t, rows = loo()
    print(t)
    fig_loo(rows)
    fig_forms()
    fig_dipeptide()
    fig_twist()
    if "--fill" in sys.argv:
        fill_readme()
