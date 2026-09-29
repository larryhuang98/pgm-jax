"""Finite-field static dielectric constant of a water box (or any box of rigid molecules): copies of
the box in uniform fields +-E (several |E|) and at zero field, advanced together on one device
(pgm_jax.md.finite_field.FieldReplicas), then eps = 1 + <M.e>/(eps0 V |E|) per replica and per +-E
pair (tin-foil boundary conditions: E is the Maxwell field; docs/efield.md).

    # 512 waters of a model, NVT at its density, fields 0.02..0.2 V/nm along z, two zero-field copies
    python scripts/finite_field.py run --model p25 --density 1.010 --fields 0.02 0.05 0.1 0.2 --zero 2 \
        --ns 1.0 -o runs/ff/p25                       # continue: the same command again (appends)
    python scripts/finite_field.py analyse runs/ff/p25.ffd --skip-ps 50

Models (512 waters in a truncated octahedron; the rigid geometry is taken from the coordinates):
  p25   pGM3P-25 with its published geometry and Lennard-Jones (~/project/epsp/p25_512.prmtop)
  base  the Amber test pGM water, q_O = -1.73 (~/project/epsp/base/base_512.prmtop)
  tip3p TIP3P point charges (classical prmtop, MDSettings(elec="q"))
or --prmtop/--coords [--amber-charges].  The box is scaled (molecular centres) to --density
(g/cm^3), or to the mean volume of a zero-field NPT run of --npt-ps ps first.  Settings as
scripts/water_dielectric.py: 0.9 nm cutoff with the LJ tail, PME 48^3 order 6, beta 4 nm^-1, dipole
tol 1e-5, mixed precision, Bussi 1 ps, rigid bodies, 2 fs, 298 K."""

from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgm_jax.md.finite_field import FieldReplicas, analyse, predicted_errors, read_series  # noqa: E402

EPSP = os.path.expanduser("~/project/epsp")
MODELS = {
    "p25": (f"{EPSP}/p25_512.prmtop", f"{EPSP}/p25_512.rst7", "pgm"),
    "base": (f"{EPSP}/base/base_512.prmtop", f"{EPSP}/base/base_512.rst7", "pgm"),
    "tip3p": (f"{EPSP}/tip3p/tip3p_512.prmtop", f"{EPSP}/tip3p/tip3p_512.rst7", "amber"),
}
AMU_NM3_TO_G_CM3 = 1.66053906660e-3


def build(a, ensemble="nvt"):
    from pgm_jax.md.forcefield import MDSettings
    from pgm_jax.md.io import box_from_cell, read_coordinates
    from pgm_jax.md.simulation import _dedupe
    from pgm_jax.param import read_prmtop_pgm
    from pgm_jax.system import System

    top, crd, charges = MODELS[a.model] if a.model else (a.prmtop, a.coords, "amber" if a.amber_charges else "pgm")
    mols = _dedupe(read_prmtop_pgm(top, first_residue_only=False, charges=charges))
    sys_ = System(mols)
    xyz, vel, box = read_coordinates(crd)
    pos, H = xyz * 0.1, box_from_cell(*box) * 0.1
    elec = "q" if charges == "amber" else a.elec
    st = MDSettings(
        cutoff=0.9,
        skin=0.1,
        ewald_beta=4.0,
        pme_grid=(48, 48, 48),
        pme_order=6,
        lj_lrc=True,
        dipole_tol=a.tol,
        precision=a.precision,
        elec=elec,
    )
    kw = dict(
        settings=st,
        ensemble=ensemble,
        temperature=a.temp,
        pressure=1.0,
        barostat_interval=100,
        dt=a.dt / 1000,
        log=sys.stdout,
        thermostat="bussi",
        tau_t=1.0,
        seed=a.seed,
    )
    return sys_, pos, H, kw


def scale_to(sys_, pos, H, V_target):
    """Molecular centres (and the box) scaled isotropically to the volume V_target (nm^3)."""
    m = np.asarray(sys_.masses)
    mol = np.asarray(sys_.mol)
    s = (V_target / abs(np.linalg.det(H))) ** (1.0 / 3.0)
    com = np.zeros((sys_.nmol, 3))
    np.add.at(com, mol, m[:, None] * pos)
    com /= np.bincount(mol, weights=m)[:, None]
    return pos + ((s - 1.0) * com)[mol], H * s


def cmd_run(a):
    from pgm_jax.md.dipoles import CellDipole
    from pgm_jax.md.simulation import Simulation

    sys_, pos, H, kw = build(a)
    mass = float(np.sum(sys_.masses))
    chk = a.out + ".ffchk"
    resume = os.path.exists(chk)
    if a.density and not resume:
        pos, H = scale_to(sys_, pos, H, mass / a.density * AMU_NM3_TO_G_CM3)
    elif a.npt_ps and not resume:
        sim = Simulation(sys_, pos, H, **dict(kw, ensemble="npt"))
        n = int(round(a.npt_ps / (a.dt / 1000))) // 20
        vols = []
        for _ in range(20):
            sim.run(n, report=n, prefix=a.out + ".npt", append=bool(vols))
            vols.append(float(np.abs(np.linalg.det(np.asarray(sim.state.box)))))
        V = float(np.mean(vols[10:]))
        pos, H = scale_to(sys_, sim.positions_nm(), np.asarray(sim.state.box), V)
        print(f"# NPT {a.npt_ps} ps: mean volume {V:.4f} nm^3 (density {mass / V * AMU_NM3_TO_G_CM3:.4f})", flush=True)
    from pgm_jax.md.efield import ExternalField

    sim = Simulation(sys_, pos, H, efield=ExternalField((0.0, 0.0, 0.0), kind=a.kind), **kw)
    fields = []
    for e in a.fields:
        fields += [(0.0, 0.0, e), (0.0, 0.0, -e)]
    fields += [(0.0, 0.0, 0.0)] * a.zero
    rep = FieldReplicas(sim, fields, seed=a.seed)
    extra = {"model": a.model or a.prmtop, "elec": sim.settings.elec, "dt_fs": a.dt, "dipole_tol": a.tol}
    if resume:
        rep.load(chk)
        print(f"# continuing from {chk} at {rep.time_ps:.2f} ps", flush=True)
    else:
        extra["density_g_cm3"] = mass / abs(np.linalg.det(H)) * AMU_NM3_TO_G_CM3
        if sim.ff.ind:  # eps_inf of the start configuration (for the zero-field fluctuations)
            st = sim.state
            p = sim.rigid.positions(st.dyn.position)
            idx = sim.nb.candidates(st.nbr, st.dyn.position.center, st.box, p)[0]
            a_cell = float(np.trace(np.asarray(CellDipole(sim.ff).polarizability(p, st.box, idx))) / 3.0)
            extra["eps_inf"] = 1.0 + 4.0 * np.pi * a_cell / abs(np.linalg.det(H))
    nsteps = int(round(a.ns * 1e6 / a.dt))
    nsteps -= nsteps % a.report
    rep.run(
        nsteps,
        every=a.every,
        prefix=a.out,
        report=a.report,
        append=resume,
        restart=a.report * 10,
        extra=extra,
        log=sys.stdout,
    )


def cmd_analyse(a):
    meta, data = read_series(a.files)
    res = analyse(meta, data, skip_ps=a.skip_ps, nblocks=a.blocks, eps_inf=a.eps_inf)
    print(
        f"# {a.files[0]}: V {res['volume_nm3']:.4f} nm^3, T {res['temperature_K']:g} K, {res['run_ps']:.1f} ps per "
        f"replica after {a.skip_ps:g} ps, eps_inf {meta.get('eps_inf', 1.0)}"
    )
    print("# single replicas: E_z (V/nm)  <M.e> (e nm)  eps  tau_M (ps)")
    for r in res["single"]:
        print(
            f"  {r['E'][2]:+8.4f}  {r['M_par']:9.4f} +- {r['M_par_err']:.4f}   {r['eps']:8.2f} +- {r['err']:.2f}   {r['tau_ps']:.1f}"
        )
    print("# +-E pairs: |E|  eps")
    for r in res["pairs"]:
        print(f"  {r['E_mag']:8.4f}  {r['eps']:8.2f} +- {r['err']:.2f}")
    for f in res.get("fits", []):
        print(
            f"# fit eps(E) = eps0 - c E^2 over |E| <= {f['E_max']:g}: eps0 {f['eps0']:.2f} +- {f['eps0_err']:.2f}, "
            f"c {f['c']:.0f} +- {f['c_err']:.0f} (V/nm)^-2, chi2 {f['chi2']:.2f} for {f['n'] - 2} dof"
        )
    for r in res["zero"]:
        print(
            f"# zero field replica {r['replica']}: fluctuation eps {r['eps']:.2f} +- {r['err']:.2f}, tau_M(z) {r['tau_ps']:.1f} ps"
        )
    if res["pairs"] and res["zero"]:
        tau = np.mean([r["tau_ps"] for r in res["zero"]])
        for r in res["pairs"]:
            p = predicted_errors(
                r["eps"],
                float(meta.get("eps_inf", 1.0)),
                res["volume_nm3"],
                res["temperature_K"],
                r["E_mag"],
                tau,
                res["run_ps"],
            )
            print(
                f"# |E| {r['E_mag']:g}: predicted error of the pair {p['sigma_ff']:.2f}, of the fluctuations at the "
                f"same cost {p['sigma_fluct_same_cost']:.2f}: finite field {p['cost_ratio']:.1f}x cheaper"
            )
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("-o", "--out", required=True)
    r.add_argument("--model", choices=sorted(MODELS))
    r.add_argument("--prmtop")
    r.add_argument("--coords")
    r.add_argument("--amber-charges", action="store_true")
    r.add_argument("--elec", default="qpi")
    r.add_argument(
        "--fields", type=float, nargs="+", default=[0.05, 0.1, 0.2], help="|E| (V/nm), each run as +-E along z"
    )
    r.add_argument("--zero", type=int, default=1, help="zero-field replicas")
    r.add_argument(
        "--kind",
        default="E",
        choices=["E", "D"],
        help="constant field E, or constant displacement (--fields are then D/eps0 in V/nm)",
    )
    r.add_argument("--density", type=float, help="g/cm^3: scale the box to it")
    r.add_argument(
        "--npt-ps", type=float, default=0.0, help="zero-field NPT first; the box is scaled to its mean volume"
    )
    r.add_argument("--ns", type=float, default=1.0, help="ns per replica in this invocation")
    r.add_argument("--dt", type=float, default=2.0, help="fs")
    r.add_argument("--temp", type=float, default=298.0)
    r.add_argument("--tol", type=float, default=1e-5)
    r.add_argument("--precision", default="mixed")
    r.add_argument("--every", type=int, default=25)
    r.add_argument("--report", type=int, default=5000)
    r.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("analyse")
    s.add_argument("files", nargs="+")
    s.add_argument("--skip-ps", type=float, default=50.0)
    s.add_argument("--blocks", type=int, default=10)
    s.add_argument("--eps-inf", type=float, default=None)
    s.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "run":
        if not (a.model or (a.prmtop and a.coords)):
            ap.error("--model or --prmtop and --coords")
        cmd_run(a)
    else:
        cmd_analyse(a)


if __name__ == "__main__":
    main()
