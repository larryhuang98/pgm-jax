"""Figures of the pGM-JAX paper from the JSON files in paper/data/.
    python paper/figures/make_figures.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})
C = {"mixed": "#1f5fa8", "double": "#7fa7d6", "cuda": "#c8553d", "cpu": "#999999", "water": "#1f5fa8",
     "methanol": "#c8553d"}


def speed():
    d = json.load(open(os.path.join(DATA, "speed.json")))
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    x = np.arange(len(d["systems"]))
    series = [("pgm_jax_mixed", "pGM-JAX, mixed", C["mixed"]), ("pgm_jax_double", "pGM-JAX, double", C["double"]),
              ("pmemd_cuda_spfp", "pmemd.pgm.cuda (SPFP)", C["cuda"]), ("pmemd_cpu_1core", "pmemd-pgm CPU, 1 core", C["cpu"])]
    w = 0.2
    for k, (key, lab, col) in enumerate(series):
        v = [np.nan if y is None else y for y in d[key]]
        b = ax.bar(x + (k - 1.5) * w, v, w, label=lab, color=col)
        for xi, yi in zip(x + (k - 1.5) * w, v):
            if np.isfinite(yi):
                ax.text(xi, yi + 2, f"{yi:g}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, d["systems"])
    ax.set_ylabel("ns/day")
    ax.set_ylim(0, 150)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    fig.savefig(os.path.join(HERE, "speed.pdf"))


def liquid():
    runs = [("water_recovery.json", "water (recovery)", C["water"]), ("methanol.json", "methanol (experiment)", C["methanol"])]
    runs = [(f, l, c) for f, l, c in runs if os.path.exists(os.path.join(DATA, f))]
    if not runs:
        return
    fig, axes = plt.subplots(len(runs), 3, figsize=(7.0, 2.2 * len(runs)), squeeze=False)
    for k_row, (row, (f, lab, col)) in enumerate(zip(axes, runs)):
        d = json.load(open(os.path.join(DATA, f)))
        it = d["iters"]
        n = np.arange(len(it))
        tgt = [d["exp"]["rho"], d["exp"]["dhvap"]]
        for ax, key, ylab, t in ((row[0], "rho", r"$\rho$ (g cm$^{-3}$)", tgt[0]), (row[1], "dhvap", r"$\Delta H_\mathrm{vap}$ (kcal/mol)", tgt[1])):
            y = np.array([r[key] for r in it]); e = np.array([r[key + "_se"] for r in it])
            ax.errorbar(n, y, 2 * e, marker="o", ms=4, color=col, lw=1.2, capsize=2, label="simulated")
            pred = [(k, r["predicted_from_previous"][0 if key == "rho" else 1]) for k, r in enumerate(it) if r["predicted_from_previous"]]
            if pred:
                ax.plot([p[0] for p in pred], [p[1] for p in pred], "x", color="k", ms=5, label="predicted")
            ax.axhline(t, color="gray", ls="--", lw=0.8, label="target")
            ax.set_xlabel("iteration"); ax.set_ylabel(ylab)
            ax.set_xticks(n)
        row[0].set_title(lab, loc="left", fontsize=9)
        if k_row == 0:
            row[0].legend(frameon=False, fontsize=7)
        ax = row[2]
        th = np.array([r["theta"] for r in it])
        ax.plot(np.exp(th[:, 0]), np.exp(th[:, 1]), "-o", color=col, ms=4)
        for k in (0, len(th) - 1):
            ax.annotate(str(k), (np.exp(th[k, 0]), np.exp(th[k, 1])), textcoords="offset points", xytext=(4, 3), fontsize=7)
        if "recovery" in f:
            ax.plot([1.0], [1.0], "*", color="k", ms=8, label="original")
            ax.legend(frameon=False, fontsize=7)
        ax.set_xlabel(r"$s_R$ ($R^*$ scale)"); ax.set_ylabel(r"$s_\varepsilon$ ($\varepsilon$ scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "liquid_fit.pdf"))


def flexible():
    p = os.path.join(DATA, "flex_methanol.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    ax = axes[0]
    t, rho = np.array(d["npt"]["time_ps"]), np.array(d["npt"]["density"])
    ax.plot(t, rho, color=C["methanol"], lw=1)
    ax.axhline(d["exp_density"], color="gray", ls="--", lw=0.8, label="experiment")
    ax.set_xlabel("time (ps)"); ax.set_ylabel(r"$\rho$ (g cm$^{-3}$)"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("NPT, 298 K, 1 bar", loc="left", fontsize=9)
    ax = axes[1]
    cols = {"0.5 fs, mixed, tol 1e-5": C["mixed"], "0.25 fs, mixed, tol 1e-5": "#e39b2d", "0.5 fs, double, tol 1e-8": "#7fb27f"}
    for run in sorted(d["nve"], key=lambda r: -np.std(r["etot"])):
        t, e = np.array(run["time_ps"]), np.array(run["etot"])
        ax.plot(t, (e - e.mean()) / d["kT_total"], lw=0.9, label=run["label"], color=cols.get(run["label"]))
    ax.set_xlabel("time (ps)"); ax.set_ylabel(r"$E_\mathrm{tot}-\langle E_\mathrm{tot}\rangle$ ($kT$)")
    ax.set_ylim(-2.2, 2.2)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2)
    ax.set_title("NVE", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "flexible.pdf"))


if __name__ == "__main__":
    speed(); liquid(); flexible()
