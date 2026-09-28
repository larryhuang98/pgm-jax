"""Validation of the external electric field (pgm_jax/md/efield.py; docs/efield.md).

    python scripts/validate_efield.py gas       # gas phase: induced-dipole response = molecular polarizability,
                                                # energy = E0 - E.M0 - E.alpha.E/2, forces vs finite differences
    python scripts/validate_efield.py box1      # one molecule in growing periodic boxes -> the gas-phase response
    python scripts/validate_efield.py nve       # 512 pGM3P-25 waters, NVE 1 fs, 20 ps: drift at 0, 0.1, 0.5 V/nm
                                                # and with E(t) = 0.5 cos(w t) V/nm (econs with the work booked)

Results: validation/validate_efield_<part>.json and the printed tables."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pgm_jax import ElecChannel, Model, System  # noqa: E402
from pgm_jax.channels import molecular_polarizability  # noqa: E402
from pgm_jax.md import efield as EF  # noqa: E402
from pgm_jax.param import read_prmtop_pgm  # noqa: E402
from pgm_jax.units import KE  # noqa: E402

P25 = os.path.expanduser("~/project/epsp/p25_512.prmtop")
P25_RST = os.path.expanduser("~/project/epsp/p25_512.rst7")


def _save(part, d):
    os.makedirs(os.path.join(ROOT, "validation"), exist_ok=True)
    with open(os.path.join(ROOT, "validation", f"validate_efield_{part}.json"), "w") as fh:
        json.dump(d, fh, indent=1)


def _water_and_methanol():
    from pgm_jax.md.io import read_coordinates
    w = read_prmtop_pgm(P25)[0]
    xyz = read_coordinates(P25_RST)[0][:3] * 0.1
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_grad import cluster, methanol
    m, xm = methanol()
    sc, xc = cluster(np.random.default_rng(0))
    return {"pGM3P-25 water": (System([w]), xyz), "methanol (test model)": (System([m]), xm),
            "water-methanol-water cluster": (sc, xc)}


def part_gas():
    E = np.array([0.02, -0.05, 0.1])            # V/nm
    out = {}
    print("# gas phase, E = (0.02, -0.05, 0.1) V/nm")
    print(f"{'system':32s} {'|dmu - alpha E| / |alpha E|':>28s} {'energy error (kJ/mol)':>22s} {'max |F - F_fd| (kJ/mol/nm)':>28s}")
    for name, (sys_, x) in _water_and_methanol().items():
        x = jnp.asarray(x)
        P = sys_.expand(None)
        e0, a0 = ElecChannel().energy(x, sys_, None)
        e1, a1 = ElecChannel(efield=tuple(E)).energy(x, sys_, None)
        A = np.asarray(molecular_polarizability(x, sys_))
        Ei = E * EF.VNM_TO_INTERNAL
        dmu = np.asarray(a1["mu"]).sum(0) - np.asarray(a0["mu"]).sum(0)
        rel = float(np.linalg.norm(dmu - A @ Ei) / np.linalg.norm(A @ Ei))
        M0 = (np.asarray(P["q"])[:, None] * np.asarray(x)).sum(0) + np.asarray(a0["p"]).sum(0) + np.asarray(a0["mu"]).sum(0)
        tot = lambda e: float(sum(e.values()))                            # noqa: E731
        expect = tot(e0) - KE * (Ei @ M0 + 0.5 * Ei @ A @ Ei)
        f = Model([ElecChannel(efield=tuple(E))]).energy_fn(sys_)
        F = np.asarray(-jax.grad(lambda y: f(y)["total"])(x))
        h, err = 1e-5, 0.0
        for i in range(sys_.n):
            for c in range(3):
                d = jnp.zeros_like(x).at[i, c].set(h)
                fd = -(float(f(x + d)["total"]) - float(f(x - d)["total"])) / (2 * h)
                err = max(err, abs(fd - F[i, c]))
        out[name] = {"alpha_nm3": A.tolist(), "dmu_rel_err": rel, "energy_err": tot(e1) - expect, "force_fd_err": err,
                     "M0_debye": float(np.linalg.norm(M0) / 0.020819434), "alpha_iso_A3": float(np.trace(A) / 3 * 1000)}
        print(f"{name:32s} {rel:28.2e} {tot(e1) - expect:22.2e} {err:28.2e}   (alpha_iso {np.trace(A) / 3 * 1000:.4f} A^3)")
    _save("gas", out)


def part_box1():
    """One pGM3P-25 water in cubic boxes of growing edge, rigid engine force field (float64, tight
    tolerance): d(sum mu)/dE vs the gas-phase polarizability; the difference is the Ewald field of the
    images, ~ 1/V."""
    from pgm_jax.md.forcefield import MDSettings, PGMForceField
    from pgm_jax.md.dipoles import CellDipole
    sys_, x = _water_and_methanol()["pGM3P-25 water"]
    x = np.asarray(x) - np.asarray(x).mean(0)
    A_gas = np.asarray(molecular_polarizability(jnp.asarray(x), sys_))
    E = np.array([0.0, 0.0, 0.1])
    rows = []
    print("# one pGM3P-25 water in a cubic box, E = 0.1 V/nm along z: engine response vs gas phase")
    for L in (2.0, 3.0, 4.0, 6.0):
        H = np.eye(3) * L
        s = MDSettings(cutoff=0.9, skin=0.0, ewald_beta=4.0, pme_spacing=0.05, pme_order=8, precision="double",
                       dipole_tol=1e-12, max_iter=500, peek=0.0, lj_lrc=False)
        ff = PGMForceField(sys_, H, s)
        pos = jnp.asarray(x + L / 2)
        idx = ff.rows_for(pos, H)
        r0 = ff.compute(pos, H, idx, ff.init_induction())
        r1 = ff.compute(pos, H, idx, ff.init_induction(), efield=(jnp.asarray(E), None))
        dM = (np.asarray(r1.dipole) - np.asarray(CellDipole(ff).components(pos, H, r0.induction.mu)).sum(0))
        a_eff = dM / (E * EF.VNM_TO_INTERNAL)[2]
        rel = float(abs(a_eff[2] - A_gas[2, 2]) / A_gas[2, 2])
        rows.append({"L_nm": L, "alpha_zz_box": float(a_eff[2]), "alpha_zz_gas": float(A_gas[2, 2]), "rel_diff": rel})
        print(f"  L {L:4.1f} nm: alpha_zz {a_eff[2] * 1000:.5f} A^3 (gas {A_gas[2, 2] * 1000:.5f}), rel. diff {rel:.2e}, "
              f"x V = {rel * L ** 3:.3f}")
    _save("box1", rows)


def part_nve(ps: float = 20.0):
    from pgm_jax.md.forcefield import MDSettings
    from pgm_jax.md.integrate import KB
    from pgm_jax.md.io import box_from_cell, read_coordinates
    from pgm_jax.md.simulation import Simulation, _dedupe
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from finite_field import scale_to
    mols = _dedupe(read_prmtop_pgm(P25, first_residue_only=False))
    sys_ = System(mols)
    xyz, vel, box = read_coordinates(P25_RST)
    pos, H = scale_to(sys_, xyz * 0.1, box_from_cell(*box) * 0.1, float(np.sum(sys_.masses)) / 1.010 * 1.66053906660e-3)
    s = MDSettings(cutoff=0.9, skin=0.1, ewald_beta=4.0, pme_grid=(48, 48, 48), pme_order=6, lj_lrc=True,
                   dipole_tol=1e-5, precision="mixed")
    # equilibrate at 298 K without a field (NVT, 10 ps), then NVE from the same state for each field
    sim = Simulation(sys_, pos, H, settings=s, ensemble="nvt", thermostat="bussi", dt=0.001, log=None, seed=1)
    sim.run(10000, report=10000, prefix=os.path.join(ROOT, "runs", "ff", "nve_equil"))
    pos, vel = sim.positions_nm(), sim.velocities_nm_ps()
    cases = [("no field", None), ("E = 0.1 V/nm", (0.0, 0.0, 0.1)), ("E = 0.5 V/nm", (0.0, 0.0, 0.5)),
             ("E = 0.5 cos(w t) V/nm, 200 cm^-1", EF.ExternalField.from_wavenumber((0.0, 0.0, 0.5), 200.0))]
    out = {}
    nblk = 40
    n = int(round(ps / 0.001)) // nblk
    for name, fld in cases:
        sim = Simulation(sys_, pos, H, settings=s, ensemble="nve", dt=0.001, log=None, vel_nm_ps=vel, efield=fld)
        t0 = time.time()
        t, ec, et, ef = [], [], [], []
        for _ in range(nblk):
            sim._advance(n)
            o = sim.observables()
            t.append(o["time_ps"]); ec.append(o["econs"]); et.append(o["etot"]); ef.append(o.get("field_energy", 0.0))
        el = time.time() - t0
        t, ec = np.asarray(t), np.asarray(ec)
        slope = np.polyfit(t, ec, 1)[0] * 1000.0                       # kJ/mol/ns
        kT = KB * 298.0
        drift = slope / (kT * sim.integ.dof)
        out[name] = {"drift_kT_per_ns_per_dof": float(drift), "econs_std": float(np.std(ec - np.polyval(np.polyfit(t, ec, 1), t))),
                     "etot_range": float(np.ptp(et)), "field_energy_mean": float(np.mean(ef)), "T_mean": float(o["temp_K"]),
                     "cg_mean": float(o["cg_mean"]), "ns_per_day": ps / 1000 / el * 86400}
        print(f"{name:34s} drift {drift:+.5f} kT/ns/dof, econs rms {out[name]['econs_std']:.3f} kJ/mol, "
              f"E_tot range {np.ptp(et):.2f}, <field energy> {np.mean(ef):.2f} kJ/mol, T {o['temp_K']:.1f}, "
              f"CG {o['cg_mean']:.2f}", flush=True)
    _save("nve", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["gas", "box1", "nve"])
    ap.add_argument("--ps", type=float, default=20.0)
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "runs", "ff"), exist_ok=True)
    {"gas": part_gas, "box1": part_box1, "nve": lambda: part_nve(a.ps)}[a.part]()


if __name__ == "__main__":
    main()
