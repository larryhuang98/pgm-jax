"""Static dielectric constant (and optionally the infrared spectrum) from cell-dipole series
written by Simulation.run(dipoles=n) / run_md.py --dipoles n (prefix.dip; pgm_jax/md/dipoles.py).

    python scripts/dielectric.py runs/water.dip [more.dip ...] --skip 500 --blocks 10
    python scripts/dielectric.py runs/ir.dip --ir runs/ir_spectrum.dat          # M sampled every 1-2 steps

Tin-foil Ewald boundary conditions, adiabatic induced dipoles (pgm_jax/analysis/dielectric.py):
    eps = eps_inf + (<M.M> - <M>.<M>) / (3 eps0 <V> kB T),   eps_inf = 1 + 4 pi <alpha_cell / V>,
M the total cell dipole (charges + permanent + induced dipoles), alpha_cell recorded in the series.
Several files are read as consecutive segments of one run (continuations with --checkpoint).
Error bars: jackknife over contiguous blocks; the table of errors against the number of blocks and
the running estimate against the run length show whether they have converged.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pgm_jax.analysis import dielectric as D
from pgm_jax.cli.args import setup_logging
from pgm_jax.md.dipoles import read_dipoles
from pgm_jax.units import C_LIGHT_M_S, DEBYE_E_NM


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help=".dip files, in order")
    ap.add_argument("--skip", type=float, default=0.0, help="ps discarded at the start (equilibration)")
    ap.add_argument("--blocks", type=int, default=10, help="blocks for the jackknife error")
    ap.add_argument("--temp", type=float, default=None, help="K; default: the thermostat target in the header")
    ap.add_argument("--eps-inf", type=float, default=None, help="use this eps_inf instead of the recorded alpha_cell")
    ap.add_argument(
        "--molecular",
        action="store_true",
        help="accept charged molecules: eps from the molecular dipole M_D only (no ionic current)",
    )
    ap.add_argument("--ir", help="write alpha(w) n(w) (cm^-1) vs wavenumber (cm^-1) to this file")
    ap.add_argument("--ir-segment", type=float, default=10.0, help="ps per Welch segment (resolution)")
    ap.add_argument("--plot", help="PNG with the running estimate and the error against the block count")
    a = ap.parse_args(argv)
    setup_logging()

    meta, d = read_dipoles(a.files)
    if meta["charged_molecules"] and not a.molecular:
        sys.exit(
            f"{meta['charged_molecules']} charged molecules (net charge {meta['net_charge']} e): M excludes the "
            "ionic current, so its fluctuation is not the full permittivity; rerun with --molecular to accept M_D"
        )
    T = a.temp if a.temp is not None else float(meta["temperature_K"])
    if meta["ensemble"] == "nve" and a.temp is None:
        T = float(np.mean(d["temp_K"]))
        print(f"# NVE: T from the mean kinetic temperature, {T:.2f} K")
    t = d["time_ps"]
    sel = t >= t[0] + a.skip if a.skip > 0 else np.ones(len(t), bool)
    M, V, alpha = d["M"][sel], d["volume_nm3"][sel], d["alpha_nm3"][sel]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    span = t[sel][-1] - t[sel][0] + dt
    print(
        f"# {', '.join(a.files)}: {meta['n_molecules']} molecules, {meta['n_atoms']} atoms, {meta['ensemble'].upper()} "
        f"({meta['thermostat']}), elec {meta['elec']}"
    )
    print(
        f"# samples {len(M)} every {dt:g} ps, {span / 1000:.3f} ns after skipping {a.skip:g} ps; T = {T:g} K "
        f"(mean kinetic {np.mean(d['temp_K'][sel]):.2f} K); <V> = {np.mean(V):.4f} nm^3"
    )
    print(
        f"# mean molecular dipole {np.mean(d['mol_dipole'][sel]) / DEBYE_E_NM:.4f} D; "
        f"<|M|^2>^1/2 = {np.sqrt(np.mean(np.sum(M * M, 1))) / DEBYE_E_NM:.2f} D, |<M>| = "
        f"{np.linalg.norm(M.mean(0)) / DEBYE_E_NM:.2f} D"
    )

    if a.eps_inf is None:
        due = int(np.sum(d["step"][sel] % (int(meta["interval"]) * int(meta["alpha_every"])) == 0))
        if np.isfinite(alpha).sum() < due:
            print(
                f"# warning: {due - int(np.isfinite(alpha).sum())} of {due} cell-polarizability solves did not converge"
            )
    r = D.static_dielectric(M, V, T, alpha=alpha, eps_inf_value=a.eps_inf, nblocks=a.blocks)
    src = "given" if a.eps_inf is not None else f"{int(np.isfinite(alpha).sum())} alpha_cell samples"
    print(f"eps_inf  = {r['eps_inf']:.4f} +- {r['eps_inf_err']:.4f}   ({src})")
    print(
        f"fluct    = {r['fluct']:.3f} +- {r['fluct_err']:.3f}   (<M.M> - <M>.<M>) / (3 eps0 V kB T), {a.blocks} blocks"
    )
    print(f"eps      = {r['eps']:.3f} +- {r['err']:.3f}")
    dec = D.decomposition({"perm": d["M_charge"][sel] + d["M_perm"][sel], "ind": d["M_ind"][sel]}, V, T)
    print(
        "# fluctuation split, permanent (charges + covalent dipoles) / induced: "
        + ", ".join(f"{k} {v:.3f}" for k, v in dec.items())
    )
    tau = D.correlation_time(M, dt)
    if np.isfinite(tau):
        print(
            f"# tau_M = {tau:.2f} ps (exponential fit of the dipole autocorrelation); expected relative error "
            f"sqrt(2 tau / 3 T_run) = {np.sqrt(2 * tau / (3 * span)) * 100:.2f} % -> +- "
            f"{np.sqrt(2 * tau / (3 * span)) * r['fluct']:.3f}"
        )
    print("# jackknife error of the fluctuation term against the number of blocks (block length ps)")
    be = D.block_errors(M, V, T)
    for b, e in be:
        print(f"  {b:4d} {span / b:10.1f} {e:8.3f}")
    print("# running estimate: first fraction of the series -> length (ns), eps, error")
    run = D.running(M, V, T, nblocks=a.blocks)
    for f, v, e in run:
        print(f"  {f:6.3f} {f * span / 1000:9.3f} {r['eps_inf'] + v:9.3f} {e:8.3f}")
    if a.ir:
        wn, an = D.ir_spectrum(M, dt, V, T, a.ir_segment)
        np.savetxt(a.ir, np.c_[wn, an], fmt="%12.4f %14.6e", header="wavenumber_cm-1 alpha_n_cm-1 (alpha(w) n(w))")
        band = (wn > 20) & (wn < 1500)
        pk = wn[band][np.argmax(an[band])] if band.any() else float("nan")
        print(
            f"# IR spectrum -> {a.ir} (resolution {1e12 / (a.ir_segment * C_LIGHT_M_S * 100):.1f} cm^-1, Nyquist "
            f"{0.5e12 / (dt * C_LIGHT_M_S * 100):.0f} cm^-1); strongest band below 1500 cm^-1 at {pk:.0f} cm^-1"
        )
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        x = np.array([f * span / 1000 for f, _, _ in run])
        y = np.array([r["eps_inf"] + v for _, v, _ in run])
        e = np.array([e for _, _, e in run])
        ax[0].errorbar(x, y, e, marker="o", capsize=3)
        ax[0].set_xlabel("run length (ns)")
        ax[0].set_ylabel("eps")
        ax[1].plot([span / b for b, _ in be], [e for _, e in be], marker="o")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("block length (ps)")
        ax[1].set_ylabel("jackknife error")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)


if __name__ == "__main__":
    main()
