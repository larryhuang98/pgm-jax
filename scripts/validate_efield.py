"""Validation of the external electric field (pgm_jax/md/efield.py; docs/efield.md).

    python scripts/validate_efield.py gas       # gas phase: induced-dipole response = molecular polarizability,
                                                # energy = E0 - E.M0 - E.alpha.E/2, forces vs finite differences
    python scripts/validate_efield.py box1      # one molecule in growing periodic boxes -> the gas-phase response
    python scripts/validate_efield.py nve       # 512 pGM3P-25 waters, NVE 1 fs, 20 ps: drift at 0, 0.1, 0.5 V/nm,
                                                # E(t) = 0.2 cos(w t) V/nm (econs with the work booked), constant D
    python scripts/validate_efield.py fluct a.dip [b.dip ...] --seg-ns 1   # zero-field references: fluctuation eps
                                                # of each file and the spread of the estimate over segments

Results: validation/validate_efield_<part>.json and the printed tables."""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax import ElecChannel, Model, System
from pgm_jax.channels import molecular_polarizability
from pgm_jax.cli.args import setup_logging
from pgm_jax.md import efield as EF
from pgm_jax.param import read_prmtop_molecules, read_prmtop_pgm
from pgm_jax.paths import resource
from pgm_jax.units import AMU_NM3_TO_G_CM3, DEBYE_E_NM, KB, KE

jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P25 = resource("epsp", "p25_512.prmtop")
P25_RST = resource("epsp", "p25_512.rst7")


def _save(part, d):
    os.makedirs(os.path.join(ROOT, "validation"), exist_ok=True)
    with open(os.path.join(ROOT, "validation", f"validate_efield_{part}.json"), "w") as fh:
        json.dump(d, fh, indent=1)


def _water_and_methanol():
    from pgm_jax.md.io import read_coordinates

    w = read_prmtop_pgm(P25)[0]
    xyz = read_coordinates(P25_RST)[0][:3] * 0.1
    from pgm_jax.models.toy import cluster, methanol

    m, xm = methanol()
    sc, xc = cluster(0)
    return {
        "pGM3P-25 water": (System([w]), xyz),
        "methanol (test model)": (System([m]), xm),
        "water-methanol-water cluster": (sc, xc),
    }


def part_gas():
    E = np.array([0.02, -0.05, 0.1])  # V/nm
    out = {}
    print("# gas phase, E = (0.02, -0.05, 0.1) V/nm")
    print(
        f"{'system':32s} {'|dmu - alpha E| / |alpha E|':>28s} {'energy error (kJ/mol)':>22s} "
        f"{'max |F - F_fd| (kJ/mol/nm)':>28s}"
    )
    for name, (sys_, x) in _water_and_methanol().items():
        x = jnp.asarray(x)
        P = sys_.expand(None)
        e0, a0 = ElecChannel().energy(x, sys_, None)
        e1, a1 = ElecChannel(efield=tuple(E)).energy(x, sys_, None)
        A = np.asarray(molecular_polarizability(x, sys_))
        Ei = E * EF.VNM_TO_INTERNAL
        dmu = np.asarray(a1["mu"]).sum(0) - np.asarray(a0["mu"]).sum(0)
        rel = float(np.linalg.norm(dmu - A @ Ei) / np.linalg.norm(A @ Ei))
        M0 = (
            (np.asarray(P["q"])[:, None] * np.asarray(x)).sum(0)
            + np.asarray(a0["p"]).sum(0)
            + np.asarray(a0["mu"]).sum(0)
        )

        def tot(e):
            return float(sum(e.values()))

        expect = tot(e0) - KE * (Ei @ M0 + 0.5 * Ei @ A @ Ei)
        f = Model([ElecChannel(efield=tuple(E))]).energy_fn(sys_)
        F = np.asarray(-jax.grad(lambda y: f(y)["total"])(x))
        h, err = 1e-5, 0.0
        for i in range(sys_.n):
            for c in range(3):
                d = jnp.zeros_like(x).at[i, c].set(h)
                fd = -(float(f(x + d)["total"]) - float(f(x - d)["total"])) / (2 * h)
                err = max(err, abs(fd - F[i, c]))
        out[name] = {
            "alpha_nm3": A.tolist(),
            "dmu_rel_err": rel,
            "energy_err": tot(e1) - expect,
            "force_fd_err": err,
            "M0_debye": float(np.linalg.norm(M0) / DEBYE_E_NM),
            "alpha_iso_A3": float(np.trace(A) / 3 * 1000),
        }
        print(
            f"{name:32s} {rel:28.2e} {tot(e1) - expect:22.2e} {err:28.2e}   (alpha_iso {np.trace(A) / 3 * 1000:.4f} "
            "A^3)"
        )
    _save("gas", out)


def part_box1():
    """One pGM3P-25 water in cubic boxes of growing edge, rigid engine force field (float64, tight
    tolerance): d(sum mu)/dE vs the gas-phase polarizability; the difference is the Ewald field of the
    images, ~ 1/V."""
    from pgm_jax.md.dipoles import CellDipole
    from pgm_jax.md.forcefield import MDSettings, PGMForceField

    sys_, x = _water_and_methanol()["pGM3P-25 water"]
    x = np.asarray(x) - np.asarray(x).mean(0)
    A_gas = np.asarray(molecular_polarizability(jnp.asarray(x), sys_))
    E = np.array([0.0, 0.0, 0.1])
    rows = []
    print("# one pGM3P-25 water in a cubic box, E = 0.1 V/nm along z: engine response vs gas phase")
    for L in (2.0, 3.0, 4.0, 6.0):
        H = np.eye(3) * L
        s = MDSettings(
            cutoff=0.9,
            skin=0.0,
            ewald_beta=4.0,
            pme_spacing=0.05,
            pme_order=8,
            precision="double",
            dipole_tol=1e-12,
            max_iter=500,
            peek=0.0,
            lj_lrc=False,
        )
        ff = PGMForceField(sys_, H, s)
        pos = jnp.asarray(x + L / 2)
        idx = ff.rows_for(pos, H)
        r0 = ff.compute(pos, H, idx, ff.init_induction())
        r1 = ff.compute(pos, H, idx, ff.init_induction(), efield=(jnp.asarray(E), None))
        dM = np.asarray(r1.dipole) - np.asarray(CellDipole(ff).components(pos, H, r0.induction.mu)).sum(0)
        a_eff = dM / (E * EF.VNM_TO_INTERNAL)[2]
        rel = float(abs(a_eff[2] - A_gas[2, 2]) / A_gas[2, 2])
        rows.append({"L_nm": L, "alpha_zz_box": float(a_eff[2]), "alpha_zz_gas": float(A_gas[2, 2]), "rel_diff": rel})
        print(
            f"  L {L:4.1f} nm: alpha_zz {a_eff[2] * 1000:.5f} A^3 (gas {A_gas[2, 2] * 1000:.5f}), rel. diff {rel:.2e}, "
            f"x V = {rel * L**3:.3f}"
        )
    _save("box1", rows)


def part_nve(ps: float = 20.0, only=None):
    from finite_field import scale_to

    from pgm_jax.md.box import box_from_cell
    from pgm_jax.md.forcefield import MDSettings
    from pgm_jax.md.io import read_coordinates
    from pgm_jax.md.simulation import Simulation

    mols = read_prmtop_molecules(P25)
    sys_ = System(mols)
    xyz, vel, box = read_coordinates(P25_RST)
    pos, H = scale_to(sys_, xyz * 0.1, box_from_cell(*box) * 0.1, float(np.sum(sys_.masses)) / 1.010 * AMU_NM3_TO_G_CM3)
    s = MDSettings(
        cutoff=0.9,
        skin=0.1,
        ewald_beta=4.0,
        pme_grid=(48, 48, 48),
        pme_order=6,
        lj_lrc=True,
        dipole_tol=1e-5,
        precision="mixed",
    )
    # equilibrate at 298 K without a field (NVT, 10 ps), then NVE from the same state for each field
    sim = Simulation(sys_, pos, H, settings=s, thermostat="bussi", dt=0.001, log=None, seed=1)
    sim.run(10000, report_every=10000, prefix=os.path.join(ROOT, "runs", "ff", "nve_equil"))
    pos, vel = sim.positions(), sim.velocities()
    cases = [
        ("no field", None),
        ("E = 0.1 V/nm", (0.0, 0.0, 0.1)),
        ("E = 0.5 V/nm", (0.0, 0.0, 0.5)),
        ("E = 0.2 cos(w t) V/nm, 200 cm^-1", EF.ExternalField.from_wavenumber((0.0, 0.0, 0.2), 200.0)),
        ("D/eps0 = 3 V/nm", EF.displacement((0.0, 0.0, 3.0))),
    ]
    if only:
        cases = [cases[i] for i in only]
    out = {}
    nblk = 40
    n = int(round(ps / 0.001)) // nblk
    for name, fld in cases:
        sim = Simulation(sys_, pos, H, settings=s, thermostat=None, dt=0.001, log=None, velocities=vel, efield=fld)
        t0 = time.time()
        t, ec, et, ef, hh = [], [], [], [], []
        for _ in range(nblk):
            sim.advance(n)
            o = sim.observables()
            t.append(o["time_ps"])
            ec.append(o["econs"])
            et.append(o["etot"])
            ef.append(o.get("field_energy", 0.0))
            hh.append(float(sim.state.heat))
        el = time.time() - t0
        t, ec = np.asarray(t), np.asarray(ec)
        slope = np.polyfit(t, ec, 1)[0] * 1000.0  # kJ/mol/ns
        kT = KB * 298.0
        drift = slope / (kT * sim.integ.dof)
        out[name] = {
            "drift_kT_per_ns_per_dof": float(drift),
            "econs_std": float(np.std(ec - np.polyval(np.polyfit(t, ec, 1), t))),
            "etot_range": float(np.ptp(et)),
            "field_energy_mean": float(np.mean(ef)),
            "T_mean": float(o["temp_K"]),
            "cg_mean": float(o["cg_mean"]),
            "ns_per_day": ps / 1000 / el * 86400,
            "work_booked": hh[-1],
            "Emac_z_mean": float(np.mean([0.0])) if "Emac_z" not in o else o["Emac_z"],
        }
        print(
            f"{name:34s} drift {drift:+.5f} kT/ns/dof, econs rms {out[name]['econs_std']:.3f} kJ/mol, "
            f"E_tot range {np.ptp(et):.2f}, <field energy> {np.mean(ef):.2f} kJ/mol, T {o['temp_K']:.1f}, "
            f"CG {o['cg_mean']:.2f}, work of the field {hh[-1]:.1f} kJ/mol",
            flush=True,
        )
    _save("nve" if not only else "nve_" + "_".join(map(str, only)), out)


def part_speed(nsteps: int = 5000):
    """ms/step of 512 pGM3P-25 waters (rigid, 2 fs, NVT Bussi, mixed) without and with fields, and of
    batched field replicas."""
    from finite_field import scale_to

    from pgm_jax.md.box import box_from_cell
    from pgm_jax.md.finite_field import FieldReplicas
    from pgm_jax.md.forcefield import MDSettings
    from pgm_jax.md.io import read_coordinates
    from pgm_jax.md.simulation import Simulation

    mols = read_prmtop_molecules(P25)
    sys_ = System(mols)
    xyz, vel, box = read_coordinates(P25_RST)
    pos, H = scale_to(sys_, xyz * 0.1, box_from_cell(*box) * 0.1, float(np.sum(sys_.masses)) / 1.010 * AMU_NM3_TO_G_CM3)
    s = MDSettings(
        cutoff=0.9,
        skin=0.1,
        ewald_beta=4.0,
        pme_grid=(48, 48, 48),
        pme_order=6,
        lj_lrc=True,
        dipole_tol=1e-5,
        precision="mixed",
    )
    out = {}
    for name, fld in [
        ("no field", None),
        ("static E 0.1 V/nm", (0.0, 0.0, 0.1)),
        ("E(t) 0.1 V/nm, 200 cm^-1", EF.ExternalField.from_wavenumber((0, 0, 0.1), 200.0)),
        ("constant D/eps0 3 V/nm", EF.displacement((0, 0, 3.0))),
    ]:
        sim = Simulation(sys_, pos, H, settings=s, thermostat="bussi", dt=0.002, log=None, efield=fld)
        sim.advance(1000)
        jax.block_until_ready(sim.state.epot)
        t0 = time.time()
        sim.advance(nsteps)
        jax.block_until_ready(sim.state.epot)
        ms = (time.time() - t0) / nsteps * 1000
        out[name] = {"ms_per_step": ms, "ns_per_day": 0.002 * 86400 / ms, "cg_mean": sim.observables()["cg_mean"]}
        print(
            f"{name:30s} {ms:.3f} ms/step, {0.002 * 86400 / ms:.1f} ns/day, CG {out[name]['cg_mean']:.2f}", flush=True
        )
    sim = Simulation(sys_, pos, H, settings=s, thermostat="bussi", dt=0.002, log=None, efield=(0, 0, 0))
    for R in (1, 2, 4, 10):
        rep = FieldReplicas(sim, [(0.0, 0.0, 0.1 * (-1) ** k) for k in range(R)])
        rep.advance(500)
        jax.block_until_ready(rep.S.epot)
        t0 = time.time()
        for _ in range(nsteps // 25):
            rep.advance(25)
        jax.block_until_ready(rep.S.epot)
        ms = (time.time() - t0) / nsteps * 1000
        out[f"replicas {R}"] = {
            "ms_per_step": ms,
            "ns_per_day_per_replica": 0.002 * 86400 / ms,
            "aggregate_ns_per_day": R * 0.002 * 86400 / ms,
        }
        print(
            f"FieldReplicas x{R:<3d} (M sampled every 25 steps) {ms:.3f} ms/step: {0.002 * 86400 / ms:.1f} ns/day per "
            f"replica, {R * 0.002 * 86400 / ms:.1f} aggregate",
            flush=True,
        )
    _save("speed", out)


def part_fluct(files, seg_ns: float, skip_ps: float):
    """Fluctuation eps of zero-field .dip series (each file an independent run), and the scatter of
    the estimate over segments of seg_ns: the measured statistical error of a run of that length."""
    from pgm_jax.analysis.finite_field import fluctuation_eps
    from pgm_jax.analysis.stats import integrated_correlation_time
    from pgm_jax.md.dipoles import read_dipoles

    out, segs = {}, []
    for f in files:
        meta, d = read_dipoles(f)
        t = d["time_ps"]
        sel = t >= t[0] + skip_ps
        M, V, T = d["M"][sel], float(np.mean(d["volume_nm3"][sel])), float(meta["temperature_K"])
        a = d["alpha_nm3"][sel]
        a = a[np.isfinite(a)]
        eps_inf = 1.0 + 4 * np.pi * float(np.mean(a)) / V if len(a) else 1.0
        eps, err = fluctuation_eps(M, V, T, eps_inf, 10)
        dt = float(np.median(np.diff(t)))
        tau = integrated_correlation_time(M[:, 2], dt)
        n = int(round(seg_ns * 1000 / dt))
        e_seg = [fluctuation_eps(M[i : i + n], V, T, eps_inf, 5)[0] for i in range(0, len(M) - n + 1, n)]
        segs += e_seg
        out[f] = {
            "eps": eps,
            "err": err,
            "eps_inf": eps_inf,
            "run_ns": float((t[sel][-1] - t[sel][0]) / 1000),
            "tau_ps": tau,
            "seg_eps": e_seg,
        }
        print(
            f"{f}: eps {eps:.2f} +- {err:.2f} (eps_inf {eps_inf:.3f}), {out[f]['run_ns']:.2f} ns, tau_M {tau:.1f} ps; "
            f"{len(e_seg)} segments of {seg_ns:g} ns: mean {np.mean(e_seg):.2f}, std {np.std(e_seg, ddof=1):.2f}"
        )
    if len(files) > 1:
        e = np.array([out[f]["eps"] for f in files])
        print(
            f"# {len(files)} runs: eps {e.mean():.2f} +- {e.std(ddof=1) / np.sqrt(len(e)):.2f}; all {len(segs)} "
            "segments "
            f"of {seg_ns:g} ns: std {np.std(segs, ddof=1):.2f}"
        )
    out["segments_std"] = float(np.std(segs, ddof=1))
    _save("fluct_" + os.path.basename(files[0]).split(".")[0], out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["gas", "box1", "nve", "fluct", "speed"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--seg-ns", type=float, default=1.0)
    ap.add_argument("--skip-ps", type=float, default=200.0)
    ap.add_argument("--ps", type=float, default=20.0)
    ap.add_argument("--only", type=int, nargs="+", help="nve: case indices (0 none, 1 0.1, 2 0.5, 3 E(t), 4 D)")
    a = ap.parse_args()
    setup_logging()
    os.makedirs(os.path.join(ROOT, "runs", "ff"), exist_ok=True)
    {
        "gas": part_gas,
        "box1": part_box1,
        "nve": lambda: part_nve(a.ps, a.only),
        "fluct": lambda: part_fluct(a.files, a.seg_ns, a.skip_ps),
        "speed": part_speed,
    }[a.part]()


if __name__ == "__main__":
    main()
