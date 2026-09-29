"""Analyse the alanine dipeptide runs of scripts/bias/ala2.py (docs/enhanced_sampling.md).

F(phi) from umbrella windows (WHAM, the reference), from metadynamics (c(t) reweighting and the
final bias) and OPES (reweighting and the final bias) with errors from independent runs; F(phi,
psi) of the methods; Delta G(alpha_L/C7ax: phi > 0 vs phi < 0); the 300 K replica of REMD as a
second reference.

Each PREFIX is a walker set: PREFIX_wNN.colvar (+ PREFIX_wNN.hills for independent metaD walkers,
PREFIX.hills for shared ones) and PREFIX.json for the umbrella centres.

Usage:

    python scripts/bias/ala2_analyze.py --us runs/ala2/us --metad runs/ala2/md --opes runs/ala2/op
        --out runs/ala2/compare.json
    python scripts/bias/ala2_analyze.py --help

Inputs: the runs' .colvar / .hills / .json files (REMD: PREFIX_T00*.nc).
Outputs: the JSON (--out) with all free energies and comparisons; printed summaries.
Units: kJ/mol (free energies), rad (angles; bins in degrees in the output), --temperature-K K,
--fmax-kJ and --fmax-remd-kJ kJ/mol, --skip-fraction and --us-skip-fraction dimensionless.
Runtime: seconds to minutes (CPU).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from pgm_jax.bias import analysis as A
from pgm_jax.bias.io import read_table
from pgm_jax.cli.args import add_temperature_arg
from pgm_jax.md.io import read_trajectory
from pgm_jax.units import KB

P = 2 * np.pi  # period of the dihedrals [rad]


def walkers(prefix: str) -> list[str]:
    """Return the COLVAR files of a walker set (PREFIX_wNN.colvar, or PREFIX.colvar for a single run)."""
    files = sorted(glob.glob(prefix + "_w[0-9][0-9].colvar"))
    return files if files else [prefix + ".colvar"]


def dG(phi: np.ndarray, lw: np.ndarray, kT: float) -> float:
    """Return Delta G(phi > 0 vs phi <= 0) [kJ/mol] of samples phi with log-weights lw at kT [kJ/mol]."""
    w = np.exp(lw - lw.max())
    return -kT * np.log(w[phi > 0].sum() / w[phi <= 0].sum())


def marg(F2: np.ndarray, kT: float) -> np.ndarray:
    """Return F(phi) from F(phi, psi) on the 2D grid [kJ/mol], minimum 0."""
    p = np.exp(-(F2 - np.nanmin(F2[np.isfinite(F2)])) / kT)
    p[~np.isfinite(F2)] = 0.0
    with np.errstate(divide="ignore"):
        F = -kT * np.log(p.sum(1))
    return F - F[np.isfinite(F)].min()


def dihedral_np(X: np.ndarray, idx: list[int]) -> np.ndarray:
    """Return the dihedral angles [rad] of atoms idx (4) in frames X (F, N, 3)."""
    b0, b1, b2 = X[:, idx[1]] - X[:, idx[0]], X[:, idx[2]] - X[:, idx[1]], X[:, idx[3]] - X[:, idx[2]]
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m1 = np.cross(n1, b1 / np.linalg.norm(b1, axis=1)[:, None])
    return np.arctan2(-np.sum(m1 * n2, 1), np.sum(n1 * n2, 1))


class Ala2Analysis:
    """The analysis state: options, kT, the 1D and 2D grids, the results and the references.

    Attributes
    ----------
    a : argparse.Namespace
        Options.
    kT : float
        Thermal energy [kJ/mol].
    ax, ax2 : np.ndarray
        Periodic grids of F(phi) and F(phi, psi) [rad].
    res : dict
        Results (written to --out).
    refs : dict
        Reference F(phi) by name: (F, error, region F < fmax) [kJ/mol].
    """

    def __init__(self, a: argparse.Namespace) -> None:
        """Set up the grids and the empty results from the options."""
        self.a = a
        self.kT = KB * a.temperature_K
        self.ax = A.periodic_axis(a.bins)
        self.ax2 = A.periodic_axis(a.bins2)
        self.res = {"T": a.temperature_K, "bins_phi_deg": 360 / a.bins}
        self.refs = {}

    def wham_reference(self) -> None:
        """Compute the umbrella / WHAM reference F(phi) with errors from blocks of every window."""
        a = self.a
        samples, centers, kappa = [], [], None
        for prefix in a.us:
            meta = json.load(open(prefix + ".json"))
            centers += list(meta["centers"])
            kappa = meta["kappa"]
            for f in walkers(prefix):
                _, c = read_table(f)
                x = c["phi"]
                samples.append(x[int(a.us_skip_fraction * len(x)) :])
        meta = {"centers": centers, "kappa": kappa}
        F, fk = A.wham(samples, meta["centers"], np.full(len(samples), meta["kappa"]), self.ax, self.kT, period=P)
        Fb = []
        for b in range(a.blocks):
            sb = [np.array_split(x, a.blocks)[b] for x in samples]
            Fb.append(A.wham(sb, meta["centers"], np.full(len(samples), meta["kappa"]), self.ax, self.kT, period=P)[0])
        Fb = np.array(Fb)
        Fb = np.array([A.align_rmsd(f, F, F < a.fmax_kJ)[2] for f in Fb])
        err = Fb.std(0, ddof=1) / np.sqrt(a.blocks)
        # Delta G from the WHAM F(phi) (bins are uniform: sum of exp(-F/self.kT))
        pphi = np.exp(-F / self.kT)
        dg = -self.kT * np.log(pphi[self.ax > 0].sum() / pphi[self.ax <= 0].sum())
        dgb = [
            -self.kT * np.log(np.exp(-f / self.kT)[self.ax > 0].sum() / np.exp(-f / self.kT)[self.ax <= 0].sum())
            for f in Fb
        ]
        self.res["wham"] = {
            "F": F.tolist(),
            "err": err.tolist(),
            "dG_aL": dg,
            "dG_aL_err": float(np.std(dgb, ddof=1) / np.sqrt(a.blocks)),
            "samples_per_window": int(np.mean([len(x) for x in samples])),
            "windows": len(samples),
        }
        print(
            f"WHAM ({len(samples)} windows, {self.res['wham']['samples_per_window']} samples each): "
            f"dG(phi>0) = {dg:.2f} +- {self.res['wham']['dG_aL_err']:.2f} kJ/mol; "
            f"mean error bar of F(phi) (F < {a.fmax_kJ}) {err[F < a.fmax_kJ].mean():.3f}"
        )
        self.refs["wham"] = (F, err, a.fmax_kJ)

    def remd_reference(self) -> None:
        """Compute F(phi) and F(phi, psi) of the 300 K REMD replica, with block errors."""
        a = self.a
        Sphi, Spsi = [], []
        for prefix in a.remd:
            meta = json.load(open(prefix + ".json"))
            files = sorted(glob.glob(prefix + "_T00*.nc"))
            X = np.concatenate([read_trajectory(f)[0] for f in files])
            X = X[int(0.1 * len(X)) :]
            Sphi.append(dihedral_np(X, meta["phi"]))
            Spsi.append(dihedral_np(X, meta["psi"]))
        phi_r, psi_r = np.concatenate(Sphi), np.concatenate(Spsi)
        Fr = A.histogram_fes(phi_r, None, [self.ax], self.kT, [P])
        blocks = [A.histogram_fes(b, None, [self.ax], self.kT, [P]) for b in np.array_split(phi_r, 5)]
        blocks = np.array([A.align_rmsd(f, Fr, Fr < a.fmax_kJ)[2] for f in blocks])
        with np.errstate(invalid="ignore"):
            er = np.nanstd(np.where(np.isfinite(blocks), blocks, np.nan), 0, ddof=1) / np.sqrt(5)
        dgb = [dG(b, np.zeros(len(b)), self.kT) for b in np.array_split(phi_r, 5)]
        self.res["remd"] = {
            "F": Fr.tolist(),
            "err": er.tolist(),
            "frames": int(len(phi_r)),
            "dG_aL": dG(phi_r, np.zeros(len(phi_r)), self.kT),
            "dG_aL_err": float(np.std(dgb, ddof=1) / np.sqrt(5)),
            "F2": A.histogram_fes(np.stack([phi_r, psi_r], 1), None, [self.ax2, self.ax2], self.kT, [P, P]).tolist(),
        }
        self.refs["remd"] = (Fr, er, a.fmax_remd_kJ)
        print(
            f"REMD 300 K ({len(phi_r)} frames): dG(phi>0) = {self.res['remd']['dG_aL']:.2f} +- "
            f"{self.res['remd']['dG_aL_err']:.2f} kJ/mol"
        )
        if "wham" in self.refs:
            Fref = self.refs["wham"][0]
            m = (Fref < a.fmax_kJ) & np.isfinite(Fr)
            r, mx, _ = A.align_rmsd(Fr, Fref, m)
            print(f"REMD vs WHAM F(phi): RMSD {r:.3f}, max {mx:.2f}")
            self.res["remd"]["rmsd_vs_wham"] = r

    def compare(self, name: str, Fs1: list, dgs: list, Fs2: list | None = None, Fbias: list | None = None) -> dict:
        """Compare the F(phi) of a method's runs with the references; store and return the result."""
        Fs1 = np.array(Fs1)
        out = {
            "runs": len(Fs1),
            "dG_aL_runs": [float(x) for x in dgs],
            "dG_aL": float(np.mean(dgs)),
            "dG_aL_err": float(np.std(dgs, ddof=1) / np.sqrt(len(dgs))) if len(dgs) > 1 else None,
        }
        for rname, (Fr, er, fmax) in self.refs.items():
            m = (Fr < fmax) & np.isfinite(Fr) & np.isfinite(er)
            al = np.array([A.align_rmsd(f, Fr, m)[2] for f in Fs1])
            Fm = al.mean(0)
            em = al.std(0, ddof=1) / np.sqrt(len(al)) if len(al) > 1 else np.zeros_like(Fm)
            r, mx, _ = A.align_rmsd(Fm, Fr, m)
            tot = np.sqrt(em**2 + er**2)
            o = {
                "region_F_below": fmax,
                "bins": int(m.sum()),
                "F_mean": Fm.tolist(),
                "F_err": em.tolist(),
                "rmsd": r,
                "max": mx,
                "rmsd_runs": [A.align_rmsd(f, Fr, m)[0] for f in Fs1],
                "chi2_per_bin": float(np.mean(((Fm - Fr)[m] / np.maximum(tot[m], 1e-6)) ** 2)),
                "mean_err": float(tot[m].mean()),
            }
            if Fbias is not None:
                fb = np.array(Fbias)
                o["rmsd_bias"] = A.align_rmsd(np.mean([A.align_rmsd(f, Fr, m)[2] for f in fb], 0), Fr, m)[0]
            out["vs_" + rname] = o
            print(
                f"{name} vs {rname} ({int(m.sum())} bins, F < {fmax:g}): F(phi) RMSD {r:.3f} kJ/mol (max {mx:.2f}; per "
                f"run {np.mean(o['rmsd_runs']):.3f}), "
                f"chi2/bin {o['chi2_per_bin']:.2f}, mean error {o['mean_err']:.3f}"
                + (f"; from the final bias RMSD {o['rmsd_bias']:.3f}" if Fbias is not None else "")
            )
        print(
            f"{name}: dG(phi>0) {out['dG_aL']:.2f} +- "
            f"{out['dG_aL_err'] if out['dG_aL_err'] is not None else float('nan'):.2f} "
            f"({len(Fs1)} runs)"
        )
        if Fs2 is not None:
            out["F2_mean"] = np.array(Fs2).mean(0).tolist()
        self.res[name] = out
        return out

    def fes2_from_hills(self, hills: dict, factor: float) -> np.ndarray:
        """-factor V(phi, psi) on the 2D analysis grid from a hills table (read_table)."""
        C = np.stack([hills["c_phi"], hills["c_psi"]], 1)
        pts, shape = A.mesh(self.ax, self.ax)
        V = np.zeros(len(pts))
        for i in range(0, len(C), 512):
            V += A._hill_values(
                C[i : i + 512],
                hills["height"][i : i + 512],
                [hills["sigma_phi"][0], hills["sigma_psi"][0]],
                [P, P],
                pts,
            ).sum(0)
        F = -factor * V.reshape(shape)
        return F - F.min()

    def biased(self) -> None:
        """Analyse the metaD, OPES and plain runs (reweighted F(phi), F(phi, psi), Delta G) and compare them."""
        a = self.a
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
                        keep = c["step"] > a.skip_fraction * c["step"].max()
                        phi, psi = c["phi"], c["psi"]
                        if kind == "metad":
                            hf = prefix + ".hills" if shared else f.replace(".colvar", ".hills")
                            _, h = read_table(hf)
                            hills = {
                                "step": h["step"],
                                "center": np.stack([h["c_phi"], h["c_psi"]], 1),
                                "height": h["height"],
                                "sigma": np.array([h["sigma_phi"][0], h["sigma_psi"][0]]),
                            }
                            hs, ct = A.metad_ct(hills, a.biasfactor, self.kT, [P, P], fine)
                            last = np.append(hs[1:] != hs[:-1], True)
                            lw = A.ct_weights(c["step"], c["bias0_metad"], hs[last], ct[last], self.kT)
                            if f == group[-1]:
                                Fb.append(marg(self.fes2_from_hills(h, a.biasfactor / (a.biasfactor - 1.0)), self.kT))
                        elif kind == "opes":
                            lw = A.opes_weights(c["bias0_opes"], self.kT)
                        else:
                            lw = np.zeros(len(phi))
                        S.append(np.stack([phi, psi], 1)[keep])
                        LW.append(lw[keep])
                    S, LW = np.concatenate(S), np.concatenate(LW)
                    F1 = A.histogram_fes(S[:, 0], LW, [self.ax], self.kT, [P])
                    F2 = A.histogram_fes(S, LW, [self.ax2, self.ax2], self.kT, [P, P])
                    Fs1.append(F1)
                    Fs2.append(F2)
                    dgs.append(dG(S[:, 0], LW, self.kT))
            self.compare(kind, Fs1, dgs, Fs2, Fb if Fb else None)

    def cross_checks(self) -> None:
        """Compare F(phi, psi) of metaD with OPES and of both with REMD."""
        if "metad" in self.res and "opes" in self.res:
            F1, F2 = np.array(self.res["metad"]["F2_mean"]), np.array(self.res["opes"]["F2_mean"])
            m = (F1 < 15.0) & (F2 < 15.0) & np.isfinite(F1) & np.isfinite(F2)
            r, mx, _ = A.align_rmsd(F1, F2, m)
            self.res["metad_vs_opes_2d"] = {"rmsd": r, "max": mx, "bins": int(m.sum())}
            print(f"F(phi, psi) metaD vs OPES over {int(m.sum())} bins with F < 15: RMSD {r:.3f}, max {mx:.2f} kJ/mol")
        if "remd" in self.res:
            F3 = np.array(self.res["remd"]["F2"])
            for kind in ("metad", "opes"):
                if kind in self.res:
                    F1 = np.array(self.res[kind]["F2_mean"])
                    m = (F1 < 15.0) & (F3 < 15.0) & np.isfinite(F1) & np.isfinite(F3)
                    r, mx, _ = A.align_rmsd(F1, F3, m)
                    self.res[f"{kind}_vs_remd_2d"] = {"rmsd": r, "max": mx, "bins": int(m.sum())}
                    print(
                        f"F(phi, psi) {kind} vs REMD over {int(m.sum())} bins with F < 15: "
                        f"RMSD {r:.3f}, max {mx:.2f} kJ/mol"
                    )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run the analyses and write the JSON (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--us", nargs="*", default=[], help="umbrella walker sets (PREFIX.json: centres)")
    ap.add_argument("--metad", nargs="*", default=[], help="metadynamics walker sets")
    ap.add_argument("--opes", nargs="*", default=[], help="OPES walker sets")
    ap.add_argument("--plain", nargs="*", default=[], help="unbiased runs")
    ap.add_argument(
        "--remd", nargs="*", default=[], help="REMD prefixes (PREFIX_T00.nc, PREFIX.json with the phi/psi atoms)"
    )
    add_temperature_arg(ap, 300.0)
    ap.add_argument("--biasfactor", type=float, default=6.0, help="metaD bias factor of the runs")
    ap.add_argument("--skip-fraction", type=float, default=0.2, help="fraction of each biased run discarded")
    ap.add_argument("--us-skip-fraction", type=float, default=0.1, help="fraction of each window discarded")
    ap.add_argument("--bins", type=int, default=36, help="bins of F(phi)")
    ap.add_argument("--bins2", type=int, default=24, help="bins per angle of F(phi, psi)")
    ap.add_argument("--fmax-kJ", type=float, default=25.0, help="region of the comparisons: F < fmax [kJ/mol]")
    ap.add_argument(
        "--fmax-remd-kJ", type=float, default=12.0, help="region of the REMD comparison (300 K samples) [kJ/mol]"
    )
    ap.add_argument("--blocks", type=int, default=4, help="blocks of the WHAM errors")
    ap.add_argument("-o", "--out", required=True, help="JSON output")
    a = ap.parse_args(argv)
    an = Ala2Analysis(a)
    if a.us:
        an.wham_reference()
    if a.remd:
        an.remd_reference()
    an.biased()
    an.cross_checks()
    with open(a.out, "w") as fh:
        json.dump(an.res, fh, indent=1)


if __name__ == "__main__":
    main()
