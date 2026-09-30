"""Finite-field static dielectric constant of a box of rigid molecules (the `pgm-jax finite-field` command).

Copies of the box in uniform fields +-E (several |E|) and at zero field are advanced together on
one device (pgm_jax.md.finite_field.FieldReplicas); then eps = 1 + <M.e> / (eps0 V |E|) per
replica and per +-E pair (tin-foil boundary conditions: E is the Maxwell field; docs/efield.md).
Subcommands: run (the replicas; the same command again continues from <out>.ffchk and appends)
and analyse (eps per replica and pair, fits eps(E) = eps0 - c E^2, zero-field fluctuations and the
predicted errors of both methods).

Models (512 waters in a truncated octahedron; the rigid geometry is taken from the coordinates;
files in PGM_EPSP, pgm_jax.paths):
  p25   pGM3P-25 with its published geometry and Lennard-Jones (p25_512.prmtop,
        scripts/dielectric/pgm3p25_prmtop.py)
  base  the Amber test pGM water, q_O = -1.73 (base/base_512.prmtop)
  tip3p TIP3P point charges (classical prmtop, MDSettings(elec="q"))
or --prmtop/--coords [--amber-charges].  The box is scaled (molecular centres) to
--density-g-cm3, or to the mean volume of a zero-field NPT run of --npt-ps ps first.  Settings as
scripts/dielectric/water_dielectric.py: 0.9 nm cutoff with the LJ tail, PME 48^3 order 6, beta
4 nm^-1, mixed precision, Bussi 1 ps, rigid bodies; defaults 2 fs, 298 K, dipole tolerance 1e-5.

Usage:

    # 512 waters of a model, NVT at its density, fields 0.02..0.2 V/nm along z, two zero-field copies
    python scripts/dielectric/finite_field.py run --model p25 --density-g-cm3 1.010
        --fields-V-nm 0.02 0.05 0.1 0.2 --zero 2 --time-ns 1.0 -o runs/ff/p25
    python scripts/dielectric/finite_field.py analyse runs/ff/p25.ffd --skip-ps 50
    pgm-jax finite-field run --help

Inputs: the model's prmtop and coordinates (PGM_EPSP) or --prmtop/--coords.
Outputs: run: <out>.ffd (per-replica dipole series), <out>.ffchk (checkpoint), <out> log tables,
<out>.npt.* of the NPT stage; analyse: printed tables, --json the full result.
Units: --fields-V-nm V/nm (D/eps0 in V/nm with --kind D), --density-g-cm3 g/cm^3, --time-ns ns per
replica, --dt-fs fs, --temperature-K K, --npt-ps and --skip-ps ps; dipoles e nm.
Runtime: GPU (all replicas in one program); sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import numpy as np

from pgm_jax.analysis.finite_field import analyse, predicted_errors
from pgm_jax.cli.args import (
    add_dipole_tol_arg,
    add_dt_arg,
    add_precision_arg,
    add_seed_arg,
    add_temperature_arg,
    setup_logging,
)
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.dipoles import CellDipole
from pgm_jax.md.efield import ExternalField
from pgm_jax.md.finite_field import FieldReplicas, read_series
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import Bussi
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import resource
from pgm_jax.system import System
from pgm_jax.units import AMU_NM3_TO_G_CM3

jax.config.update("jax_enable_x64", True)
EPSP = resource("epsp")
MODELS = {  # name -> (prmtop, coordinates, charges)
    "p25": (f"{EPSP}/p25_512.prmtop", f"{EPSP}/p25_512.rst7", "pgm"),
    "base": (f"{EPSP}/base/base_512.prmtop", f"{EPSP}/base/base_512.rst7", "pgm"),
    "tip3p": (f"{EPSP}/tip3p/tip3p_512.prmtop", f"{EPSP}/tip3p/tip3p_512.rst7", "amber"),
}


def build(a: argparse.Namespace) -> tuple[System, np.ndarray, np.ndarray, dict]:
    """Return the system, positions, box and Simulation keywords (NVT, Bussi 1 ps) of the chosen model.

    Parameters
    ----------
    a : argparse.Namespace
        Options of `run`: model or prmtop / coords / amber_charges, elec, dipole_tol, precision,
        temperature_K, dt_fs, seed.

    Returns
    -------
    system : System
    positions : np.ndarray (N, 3)
        Positions [nm].
    box : np.ndarray (3, 3)
        Box, lattice vectors as rows [nm].
    kw : dict
        Keywords of Simulation (settings, temperature, dt [ps], log, thermostat, seed).
    """
    top, crd, charges = MODELS[a.model] if a.model else (a.prmtop, a.coords, "amber" if a.amber_charges else "pgm")
    mols = read_prmtop_molecules(top, charges=charges)
    sys_ = System(mols)
    xyz, vel, box = read_coordinates(crd)
    pos, H = xyz * 0.1, box_from_cell(*box) * 0.1
    elec = "q" if charges == "amber" else a.elec
    st = MDSettings().replace(
        cutoff=0.9,
        skin=0.1,
        ewald_beta=4.0,
        pme_grid=(48, 48, 48),
        pme_order=6,
        lj_lrc=True,
        dipole_tol=a.dipole_tol,
        precision=a.precision,
        elec=elec,
    )
    kw = dict(
        settings=st, temperature=a.temperature_K, dt=a.dt_fs / 1000, log=sys.stdout, thermostat=Bussi(1.0), seed=a.seed
    )
    return sys_, pos, H, kw


def scale_to(sys_: System, pos: np.ndarray, H: np.ndarray, V_target: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale the molecular centres and the box isotropically to a volume.

    Parameters
    ----------
    sys_ : System
        The system (masses, molecule index of the atoms).
    pos : np.ndarray (N, 3)
        Positions [nm].
    H : np.ndarray (3, 3)
        Box [nm].
    V_target : float
        Target volume [nm^3].

    Returns
    -------
    positions, box : np.ndarray
        Molecules translated rigidly with their centres of mass; the scaled box [nm].
    """
    m = np.asarray(sys_.masses)
    mol = np.asarray(sys_.mol)
    s = (V_target / abs(np.linalg.det(H))) ** (1.0 / 3.0)
    com = np.zeros((sys_.nmol, 3))
    np.add.at(com, mol, m[:, None] * pos)
    com /= np.bincount(mol, weights=m)[:, None]
    return pos + ((s - 1.0) * com)[mol], H * s


def cmd_run(a: argparse.Namespace) -> None:
    """Run (or continue) the field replicas (the `run` subcommand; see the module docstring)."""
    sys_, pos, H, kw = build(a)
    mass = float(np.sum(sys_.masses))
    chk = a.out + ".ffchk"
    resume = os.path.exists(chk)
    if a.density_g_cm3 and not resume:
        pos, H = scale_to(sys_, pos, H, mass / a.density_g_cm3 * AMU_NM3_TO_G_CM3)
    elif a.npt_ps and not resume:
        sim = Simulation(sys_, pos, H, **dict(kw, barostat=MonteCarloBarostat()))
        n = int(round(a.npt_ps / (a.dt_fs / 1000))) // 20
        vols = []
        for _ in range(20):
            sim.run(n, report_every=n, prefix=a.out + ".npt", append=bool(vols))
            vols.append(float(np.abs(np.linalg.det(np.asarray(sim.state.box)))))
        V = float(np.mean(vols[10:]))  # second half of the 20 blocks
        pos, H = scale_to(sys_, sim.positions(), np.asarray(sim.state.box), V)
        print(f"# NPT {a.npt_ps} ps: mean volume {V:.4f} nm^3 (density {mass / V * AMU_NM3_TO_G_CM3:.4f})", flush=True)
    sim = Simulation(sys_, pos, H, efield=ExternalField((0.0, 0.0, 0.0), kind=a.kind), **kw)
    fields = []
    for e in a.fields_V_nm:
        fields += [(0.0, 0.0, e), (0.0, 0.0, -e)]
    fields += [(0.0, 0.0, 0.0)] * a.zero
    rep = FieldReplicas(sim, fields, seed=a.seed, log=sys.stdout)
    extra = {
        "model": a.model or a.prmtop,
        "elec": sim.settings.terms.elec,
        "dt_fs": a.dt_fs,
        "dipole_tol": a.dipole_tol,
    }
    if resume:
        rep.load_checkpoint(chk)
        print(f"# continuing from {chk} at {rep.time_ps:.2f} ps", flush=True)
    else:
        extra["density_g_cm3"] = mass / abs(np.linalg.det(H)) * AMU_NM3_TO_G_CM3
        if sim.ff.ind:  # eps_inf of the start configuration (for the zero-field fluctuations)
            st = sim.state
            p = sim.rigid.positions(st.dyn.position)
            idx = sim.nb.candidates(st.nbr, st.dyn.position.center, st.box, p)[0]
            a_cell = float(np.trace(np.asarray(CellDipole(sim.ff).polarizability(p, st.box, idx))) / 3.0)
            extra["eps_inf"] = 1.0 + 4.0 * np.pi * a_cell / abs(np.linalg.det(H))
    nsteps = int(round(a.time_ns * 1e6 / a.dt_fs))
    nsteps -= nsteps % a.report_every
    rep.run(
        nsteps,
        sample_every=a.sample_every,
        prefix=a.out,
        report_every=a.report_every,
        append=resume,
        checkpoint_every=a.report_every * 10,
        extra=extra,
    )


def cmd_analyse(a: argparse.Namespace) -> None:
    """Analyse .ffd series and print eps per replica, per pair, the fits and error estimates (`analyse`)."""
    meta, data = read_series(a.files)
    res = analyse(meta, data, skip_ps=a.skip_ps, nblocks=a.blocks, eps_inf=a.eps_inf)
    print(
        f"# {a.files[0]}: V {res['volume_nm3']:.4f} nm^3, T {res['temperature_K']:g} K, {res['run_ps']:.1f} ps per "
        f"replica after {a.skip_ps:g} ps, eps_inf {meta.get('eps_inf', 1.0)}"
    )
    print("# single replicas: E_z (V/nm)  <M.e> (e nm)  eps  tau_M (ps)")
    for r in res["single"]:
        print(
            f"  {r['E'][2]:+8.4f}  {r['M_par']:9.4f} +- {r['M_par_err']:.4f}   {r['eps']:8.2f} +- {r['err']:.2f}   "
            f"{r['tau_ps']:.1f}"
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
            f"# zero field replica {r['replica']}: fluctuation eps {r['eps']:.2f} +- {r['err']:.2f}, tau_M(z) "
            f"{r['tau_ps']:.1f} ps"
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


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser with the subcommands run and analyse."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run (or continue) the field replicas")
    r.add_argument("-o", "--out", required=True, help="output prefix")
    r.add_argument("--model", choices=sorted(MODELS), help="one of the 512-water models (module docstring)")
    r.add_argument("--prmtop", help="prmtop (instead of --model)")
    r.add_argument("--coords", help="coordinates with a box matching --prmtop")
    r.add_argument("--amber-charges", action="store_true", help="--prmtop is classical: point charges, elec q")
    r.add_argument("--elec", default="qpi", help="electrostatics level of a pGM model (MDSettings elec)")
    r.add_argument(
        "--fields-V-nm", type=float, nargs="+", default=[0.05, 0.1, 0.2], help="|E| [V/nm], each run as +-E along z"
    )
    r.add_argument("--zero", type=int, default=1, help="zero-field replicas")
    r.add_argument(
        "--kind",
        default="E",
        choices=["E", "D"],
        help="constant field E, or constant displacement (--fields are then D/eps0 in V/nm)",
    )
    r.add_argument("--density-g-cm3", type=float, help="scale the box to this density [g/cm^3]")
    r.add_argument(
        "--npt-ps", type=float, default=0.0, help="zero-field NPT first; the box is scaled to its mean volume"
    )
    r.add_argument("--time-ns", type=float, default=1.0, help="time per replica in this invocation [ns]")
    add_dt_arg(r, 2.0)
    add_temperature_arg(r, 298.0)
    add_dipole_tol_arg(r)
    add_precision_arg(r)
    r.add_argument("--sample-every", type=int, default=25, help="steps between dipole samples")
    r.add_argument(
        "--report-every", type=int, default=5000, help="steps between log lines (checkpoints every 10 reports)"
    )
    add_seed_arg(r)
    s = sub.add_parser("analyse", help="analyse .ffd series")
    s.add_argument("files", nargs="+", help=".ffd files, in order")
    s.add_argument("--skip-ps", type=float, default=50.0, help="time discarded at the start [ps]")
    s.add_argument("--blocks", type=int, default=10, help="blocks for the error estimates")
    s.add_argument("--eps-inf", type=float, default=None, help="eps_inf (default: the recorded one)")
    s.add_argument("--json", help="write the full result to this JSON file")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the subcommand (see the module docstring)."""
    ap = build_parser()
    a = ap.parse_args(argv)
    setup_logging()
    if a.cmd == "run":
        if not (a.model or (a.prmtop and a.coords)):
            ap.error("--model or --prmtop and --coords")
        cmd_run(a)
    else:
        cmd_analyse(a)


if __name__ == "__main__":
    main()
