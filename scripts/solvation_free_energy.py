"""Hydration (solvation) free energy of one rigid molecule by alchemical lambda windows
(pgm_jax/md/alchemy.py, estimators in pgm_jax/md/free_energy.py; docs/free_energy.md).

    # water in water: the 512-water box of the README (pGM3P-25 electrostatics on TIP3P geometry and LJ)
    python scripts/solvation_free_energy.py run --model pgm -o runs/fe/pgm --ns 2
    python scripts/solvation_free_energy.py run --model tip3p -o runs/fe/tip3p --ns 2       # TIP3P control
    python scripts/solvation_free_energy.py run --prmtop sys.prmtop --coords sys.rst7 --solute 0 -o runs/fe/x
    python scripts/solvation_free_energy.py analyze runs/fe/pgm_fe.npz --discard-ps 200
    python scripts/solvation_free_energy.py run ... --checkpoint runs/fe/pgm.fe.chk        # continue a run

Protocol (`run`): NPT at full coupling (--npt-ps; Monte Carlo barostat, the alchemical Hamiltonian
at lambda = (1, 1)), the box then scaled to the mean volume of the second half; all windows of
`standard_schedule` (--n-elec electrostatics windows at lambda_vdw = 1, then the van der Waals
windows at lambda_elec = 0) batched on the GPU (NVT, Bussi thermostat), samples of u_k(x_n) and
dU/dlambda every --sample-ps, Hamiltonian replica exchange between neighbours every --exchange-ps
(0: none).  The gas-phase leg of the rigid solute (exact from its geometry) is stored with the
samples; `analyze` prints TI, BAR and MBAR (Delta G of switching off in solution, the two stages,
the hydration free energy) with the statistical inefficiencies and the smallest overlap.

Models (--model, the box of ~/pgm-gvdw-data): "pgm" as in the prmtop (pGM3P-25 charges, covalent
dipoles, radii and polarizabilities of Wu et al., JCTC 21, 3563 (2025), on TIP3P's geometry
0.9572 A / 104.52 deg and TIP3P's Lennard-Jones); "pgm3p25" with the paper's geometry (0.9745 A,
103.64 deg) and Lennard-Jones (sigma 3.18156 A, epsilon 0.14473 kcal/mol); "tip3p" the TIP3P
point charges (q_O = -0.834 e; Gaussian radii 1e-4 nm, elec "q"), geometry and LJ of the box.
The solute is the molecule --solute (0: the first water) with its own copy of the parameters."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgm_jax.md import free_energy as fe  # noqa: E402
from pgm_jax.md.alchemy import (KCAL, Alchemy, FreeEnergyRun, GasPhaseLeg, LambdaWindows,  # noqa: E402
                                alchemical_system, standard_schedule)
from pgm_jax.md.box import volume  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.io import box_from_cell, read_coordinates  # noqa: E402
from pgm_jax.md.rigid import RigidBody  # noqa: E402
from pgm_jax.md.simulation import Simulation, _dedupe  # noqa: E402
from pgm_jax.param import read_prmtop_pgm  # noqa: E402
from pgm_jax.system import System  # noqa: E402

TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
RST = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")


def water_model(model: str):
    """(molecules, xyz (A), velocities (A/ps) or None, box (a, b, c, alpha, beta, gamma), elec level)
    of the 512-water box with the chosen water model."""
    mols = _dedupe(read_prmtop_pgm(TOP, first_residue_only=False))
    xyz, vel, box = read_coordinates(RST)
    elec = "qpi"
    if model == "tip3p":
        tip = {id(m): dataclasses.replace(m, name="TIP3", q=np.array([-0.834, 0.417, 0.417]), radius=np.full(3, 1e-4),
                                          cov=[]) for m in mols}
        mols = [tip[id(m)] for m in mols]
        elec = "q"
    elif model == "pgm3p25":
        from water_dielectric import paper_geometry
        xyz = paper_geometry(xyz, 0.9745, 103.64, [list(m.elements) for m in mols])
        sig, eps = 3.18156, 0.14473
        rh = np.array([2 ** (1 / 6) * sig / 2 * 0.1, 0.0, 0.0])
        se = np.array([np.sqrt(eps * 4.184), 0.0, 0.0])
        new = {id(m): dataclasses.replace(m, lj_rmin_half=rh, lj_sqrt_eps=se) for m in mols}
        mols = [new[id(m)] for m in mols]
        vel = None
    elif model != "pgm":
        raise ValueError(model)
    return mols, xyz, vel, box, elec


def build(a):
    if a.prmtop:
        mols = _dedupe(read_prmtop_pgm(a.prmtop, first_residue_only=False))
        xyz, vel, box = read_coordinates(a.coords)
        elec = a.elec
    else:
        mols, xyz, vel, box, elec = water_model(a.model)
    if box is None:
        raise ValueError("coordinates have no periodic box")
    sys0 = System(mols)
    sysA, P = alchemical_system(sys0, a.solute)
    settings = MDSettings(cutoff=a.cut, skin=0.1, ewald_beta=a.ew_coeff, pme_grid=tuple(a.nfft), pme_order=a.order,
                          lj_lrc=True, dipole_tol=a.tol, precision=a.precision, elec=elec)
    return sysA, P, xyz * 0.1, None if vel is None else vel * 0.1, box_from_cell(*box) * 0.1, settings, elec


def scale_to_volume(sim, V):
    """Positions (nm) of sim's current configuration with molecular centres scaled to volume V, and the box."""
    st = sim.state
    s = (V / float(volume(st.box))) ** (1.0 / 3.0)
    body = st.dyn.position
    body = RigidBody(body.center * s, body.orientation)
    return np.asarray(sim.rigid.positions(body)), np.asarray(st.box) * s


def cmd_run(a):
    sysA, P, pos, vel, H, settings, elec = build(a)
    alch = Alchemy(sysA, a.solute, sc_alpha=a.sc_alpha)
    kw = dict(settings=settings, dt=a.dt / 1000.0, temperature=a.temp, thermostat="bussi", tau_t=1.0, params=P,
              alchemy=alch, log=sys.stdout, seed=a.seed)
    lam = standard_schedule(a.n_elec, None if a.vdw is None else [float(x) for x in a.vdw.split(",")])
    if a.checkpoint is None and a.npt_ps > 0:
        npt = Simulation(sysA, pos, H, ensemble="npt", pressure=1.0, barostat_interval=100, vel_nm_ps=vel, **kw)
        n = int(round(a.npt_ps / (a.dt / 1000.0)))
        rep = max(n // 20, 1)
        vols = []
        for _ in range(20):
            npt.run(rep, report=rep, prefix=a.out + "_npt", append=bool(vols))
            vols.append(float(volume(npt.state.box)))
        Vm = float(np.mean(vols[10:]))
        print(f"# NPT {a.npt_ps} ps: volume {vols[-1]:.4f} nm^3, mean of the second half {Vm:.4f} nm^3 "
              f"(density {float(np.sum(sysA.masses)) / Vm * 1.66053906660e-3:.4f} g/cm^3)", flush=True)
        pos, H = scale_to_volume(npt, Vm)
        vel = npt.velocities_nm_ps()
        del npt
    sim = Simulation(sysA, pos, H, ensemble="nvt", vel_nm_ps=vel, **kw)
    X = sim.positions_nm()[sysA.atom_slice(a.solute)]
    gas = GasPhaseLeg(alch, X, elec)
    meta = {"model": a.model if not a.prmtop else a.prmtop, "solute": a.solute, "elec": elec,
            "gas_delta_g": gas.delta_g(P), "gas_e1": gas.energy(1.0, P),
            "gas_dudl": [gas.dudl(le, P) for le in lam[:, 0]], "settings": dataclasses.asdict(settings),
            "dt_fs": a.dt, "temperature": a.temp, "volume_nm3": float(volume(sim.state.box)),
            "sc_alpha": a.sc_alpha, "alpha_floor": alch.alpha_floor}
    print(f"# gas-phase leg: E_gas(1) = {meta['gas_e1']:.4f} kJ/mol, Delta G_gas(1 -> 0) = {meta['gas_delta_g']:.4f} kJ/mol",
          flush=True)
    t0 = time.time()
    win = LambdaWindows(sim, lam, batched=not a.sequential, seed=a.seed + 1)
    to_steps = lambda ps: int(round(ps / (a.dt / 1000.0)))                      # noqa: E731
    run = FreeEnergyRun(win, sample_every=to_steps(a.sample_ps), exchange_every=to_steps(a.exchange_ps),
                        seed=a.seed, meta=meta)
    if a.checkpoint:
        run.load(a.checkpoint)
    total = to_steps(a.ns * 1000.0)
    summary = run.run(total - run.step, prefix=a.out, report=to_steps(a.report_ps), restart=to_steps(a.restart_ps))
    summary["wall_s"] = time.time() - t0
    print(json.dumps(summary, indent=1))
    report(fe.load(a.out + "_fe.npz"), a.discard_ps)


def report(d, discard_ps):
    meta = d["meta"]
    gas = {"delta_g": meta["gas_delta_g"], "dudl": meta["gas_dudl"]} if "gas_delta_g" in meta else None
    r = fe.estimate(d, discard_ps=discard_ps, gas=gas)
    k = lambda x: x / KCAL                                                    # noqa: E731
    print(f"# {r['windows']} windows, {r['samples_per_window']} samples each after {discard_ps} ps; kT = {r['kT']:.4f} kJ/mol")
    print("# statistical inefficiency (dE):", " ".join(f"{g:.1f}" for g in r["g_dE"]))
    print("# statistical inefficiency (dU/dl):", " ".join(f"{g:.1f}" for g in r["g_dudl"]))
    print(f"# smallest neighbour overlap (MBAR): {r['overlap_min']:.3f}")
    print("#  window  lambda_e lambda_v   <dU/dl_e>      <dU/dl_v>     (kJ/mol)   MBAR step   BAR step")
    L = d["lambdas"]
    for i in range(r["windows"]):
        st = "" if i == r["windows"] - 1 else f"{r['mbar_steps'][i]:10.3f} {r['bar_steps'][i]:10.3f} +- {r['bar_steps_err'][i]:.3f}"
        print(f"  {i:6d} {L[i, 0]:9.3f} {L[i, 1]:8.3f} {r['dudl_mean'][i][0]:12.3f} {r['dudl_mean'][i][1]:12.3f}   {st}")
    print("# Delta G of switching the solute off in solution (kJ/mol | kcal/mol):")
    for m in ("ti", "bar", "mbar"):
        print(f"  {m:5s} {r[m]:10.3f} +- {r[m + '_err']:.3f} | {k(r[m]):9.3f} +- {k(r[m + '_err']):.3f}")
    if "mbar_elec" in r:
        print(f"  stages (MBAR): electrostatics {r['mbar_elec']:.3f} +- {r['mbar_elec_err']:.3f}, van der Waals "
              f"{r['mbar_vdw']:.3f} +- {r['mbar_vdw_err']:.3f} kJ/mol; TI {r['ti_elec']:.3f}, {r['ti_vdw']:.3f}")
    if gas is not None:
        print(f"# gas-phase leg Delta G_gas(1 -> 0) = {r['gas']:.4f} kJ/mol; hydration free energy (kJ/mol | kcal/mol):")
        for m in ("ti", "ti_sub", "bar", "mbar"):
            v, e = r[f"dG_hyd_{m}"], r[f"dG_hyd_{m}_err"]
            print(f"  {m:6s} {v:10.3f} +- {e:.3f} | {k(v):9.3f} +- {k(e):.3f}")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("-o", "--out", required=True, help="output prefix")
    r.add_argument("--model", default="pgm", choices=["pgm", "pgm3p25", "tip3p"])
    r.add_argument("--prmtop", help="Amber pGM prmtop instead of --model (with --coords)")
    r.add_argument("--coords")
    r.add_argument("--elec", default="qpi", help="electrostatics level for --prmtop")
    r.add_argument("--solute", type=int, default=0, help="molecule (residue) index of the solute")
    r.add_argument("--ns", type=float, default=2.0, help="length of every window (ns)")
    r.add_argument("--npt-ps", type=float, default=100.0, help="NPT equilibration at full coupling (ps)")
    r.add_argument("--n-elec", type=int, default=8)
    r.add_argument("--vdw", help="lambda_vdw values of the second stage (comma separated, decreasing to 0)")
    r.add_argument("--sample-ps", type=float, default=1.0)
    r.add_argument("--exchange-ps", type=float, default=1.0, help="0: no Hamiltonian exchange")
    r.add_argument("--report-ps", type=float, default=20.0)
    r.add_argument("--restart-ps", type=float, default=200.0)
    r.add_argument("--discard-ps", type=float, default=200.0)
    r.add_argument("--dt", type=float, default=2.0, help="fs")
    r.add_argument("--temp", type=float, default=298.0)
    r.add_argument("--tol", type=float, default=1e-5)
    r.add_argument("--cut", type=float, default=0.9)
    r.add_argument("--ew-coeff", type=float, default=4.0)
    r.add_argument("--nfft", type=int, nargs=3, default=[48, 48, 48])
    r.add_argument("--order", type=int, default=6)
    r.add_argument("--precision", default="mixed")
    r.add_argument("--sc-alpha", type=float, default=0.5)
    r.add_argument("--sequential", action="store_true", help="windows one after the other (not batched)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--checkpoint", help="continue from prefix.fe.chk")
    z = sub.add_parser("analyze")
    z.add_argument("npz")
    z.add_argument("--discard-ps", type=float, default=200.0)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        report(fe.load(a.npz), a.discard_ps)


if __name__ == "__main__":
    main()
