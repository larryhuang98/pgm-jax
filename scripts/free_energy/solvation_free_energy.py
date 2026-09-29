"""Hydration (solvation) free energy of one molecule by alchemical lambda windows (`pgm-jax solvation`).

The alchemical Hamiltonian is pgm_jax/md/alchemy.py, the estimators pgm_jax/analysis/free_energy.py
(docs/free_energy.md).  Subcommands: run (sample the windows), analyze (TI, BAR, MBAR), bench
(cost per window), finite-size (periodic self-image and PME error of the solute's intramolecular
electrostatics).

Protocol (`run`): NPT at full coupling (--npt-ps; Monte Carlo barostat, the alchemical Hamiltonian
at lambda = (1, 1)), the box then scaled to the mean volume of the second half; all windows of
`standard_schedule` (--n-elec electrostatics windows at lambda_vdw = 1, then the van der Waals
windows at lambda_elec = 0) batched on the GPU (NVT, Bussi thermostat), samples of u_k(x_n) and
dU/dlambda every --sample-ps, Hamiltonian replica exchange between neighbours every --exchange-ps
(0: none).  The gas-phase leg of the rigid solute (exact from its geometry) is stored with the
samples; `analyze` prints TI, BAR and MBAR (Delta G of switching off in solution, the two stages,
the hydration free energy) with the statistical inefficiencies and the smallest overlap.

Models (--model, the 512-water box of PGM_GVDW_DATA): "pgm" as in the prmtop (pGM3P-25 charges, covalent
dipoles, radii and polarizabilities of Wu et al., JCTC 21, 3563 (2025), on TIP3P's geometry
0.9572 A / 104.52 deg and TIP3P's Lennard-Jones); "pgm3p25" with the paper's geometry (0.9745 A,
103.64 deg) and Lennard-Jones (sigma 3.18156 A, epsilon 0.14473 kcal/mol); "tip3p" the TIP3P
point charges (q_O = -0.834 e; Gaussian radii 1e-4 nm, elec "q"), geometry and LJ of the box.
The solute is the molecule --solute (0: the first water) with its own copy of the parameters.

Flexible solutes (--solute-template, a FlexibleTemplate from pgm_jax.bonded, e.g. methanol): the
molecule is put at the centre of the water box (waters within --clear nm of it removed; with
--rigid-solute it is held rigid at its reference geometry in the rigid engine instead) and the
flexible engine runs everything (rigid waters by constraints, X-H bonds of the solute constrained,
2 fs).  Its intramolecular electrostatics is kept at every lambda by the gas-phase correction
(--intramolecular keep, the default for them): the decoupled state is the gas-phase molecule, so
Delta G_hyd = -Delta G(1 -> 0) with no separate gas leg.  Rigid solutes default to annihilation with
the exact gas-phase leg (--intramolecular annihilate); both modes give the same free energy.

Parameter gradients (pgm_jax/md/fe_grad.py, pgm_jax/fit/free_energy.py, docs/fe_gradients.md): `run --grad` also samples
dU/dP of the two end-state Hamiltonians at every window's configuration, and `analyze` prints
d DeltaG_hyd / dP (MBAR-weighted and end-state estimators, block jackknife errors), with the
derivatives along scale directions of the solute's and the environment's parameters (charge =
charges and covalent dipoles, eps, rmin, alpha, radius).  --solute-scale charge=1.05,eps=0.9 runs
at scaled solute parameters (finite-difference checks: scripts/free_energy/fe_gradient_check.py);
--start-from prefix.fe.chk starts the windows from another run's configurations (no NPT);
--elec-only runs only the electrostatics stage (its free energy is the whole dependence on the
solute's electrostatic parameters: the van der Waals stage runs at lambda_elec = 0).

Usage:

    # water in water: the 512-water box of the README (pGM3P-25 electrostatics on TIP3P geometry and LJ)
    python scripts/free_energy/solvation_free_energy.py run --model pgm -o runs/fe/pgm --time-ns 2
    python scripts/free_energy/solvation_free_energy.py run --model tip3p -o runs/fe/tip3p --time-ns 2   # control
    python scripts/free_energy/solvation_free_energy.py run --prmtop s.prmtop --coords s.rst7 --solute 0 -o runs/fe/x
    python scripts/free_energy/solvation_free_energy.py run --solute-template methanol.flex -o runs/fe/meoh  # flexible
    python scripts/free_energy/solvation_free_energy.py analyze runs/fe/pgm_fe.npz --discard-ps 200
    python scripts/free_energy/solvation_free_energy.py run ... --continue-from runs/fe/pgm.fe.chk   # continue a run
    python scripts/free_energy/solvation_free_energy.py bench --model pgm --windows 1,4,19        # cost per window
    python scripts/free_energy/solvation_free_energy.py finite-size --model pgm                   # PME / image check
    pgm-jax solvation run --help

Inputs: the water box of PGM_GVDW_DATA (pgm_jax.paths) or --prmtop/--coords; --solute-template.
Outputs: run: <out>_fe.npz (samples, meta), <out>.fe.chk (checkpoint), <out>_npt.* (NPT stage), the
log tables and the printed report; analyze, bench, finite-size: printed.
Units: --time-ns ns (per window), durations in ps (--npt-ps, --sample-ps, --exchange-ps,
--report-ps, --checkpoint-ps, --discard-ps), --dt-fs fs, --temperature-K K, --cutoff-nm and
--clear-nm nm, --ewald-beta-per-nm 1/nm; free energies printed in kJ/mol and kcal/mol.
Runtime: GPU (all windows batched); sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import jax
import numpy as np

from pgm_jax.analysis import free_energy as fe
from pgm_jax.cli.args import (
    add_dipole_tol_arg,
    add_dt_arg,
    add_precision_arg,
    add_seed_arg,
    add_temperature_arg,
    setup_logging,
)
from pgm_jax.cli.main import load_script, scripts_dir
from pgm_jax.fit.free_energy import gradient_estimate
from pgm_jax.fit.params import SCALE_GROUPS, ParameterSpace
from pgm_jax.md import fe_grad as fg
from pgm_jax.md.alchemy import (
    Alchemy,
    FreeEnergyRun,
    GasPhaseLeg,
    LambdaWindows,
    alchemical_system,
    standard_schedule,
)
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.box import (
    box_from_cell,
    reduce_box,
    volume,
)
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, RigidTemplate
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.rigid import RigidBody
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import Bussi
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import pgm3p25_files
from pgm_jax.system import System
from pgm_jax.units import AMU_NM3_TO_G_CM3, KCAL

jax.config.update("jax_enable_x64", True)
TOP, RST = pgm3p25_files()
MODELS = ["pgm", "pgm3p25", "tip3p"]


def water_model(model: str) -> tuple[list, np.ndarray, np.ndarray | None, tuple, str]:
    """Return the 512-water box with the chosen water model.

    Parameters
    ----------
    model : {"pgm", "pgm3p25", "tip3p"}
        See the module docstring.

    Returns
    -------
    mols : list of Molecule
    xyz : np.ndarray (N, 3)
        Coordinates [A].
    vel : np.ndarray (N, 3) or None
        Velocities of the restart [A/ps] (None for pgm3p25, whose geometry is rebuilt).
    box : tuple
        (lengths [A], angles [deg]) of the restart.
    elec : str
        Electrostatics level ("qpi", or "q" for tip3p).

    Raises
    ------
    ValueError
        An unknown model.
    """
    mols = read_prmtop_molecules(TOP)
    xyz, vel, box = read_coordinates(RST)
    elec = "qpi"
    if model == "tip3p":
        tip = {
            id(m): dataclasses.replace(
                m, name="TIP3", q=np.array([-0.834, 0.417, 0.417]), radius=np.full(3, 1e-4), cov=[]
            )
            for m in mols
        }
        mols = [tip[id(m)] for m in mols]
        elec = "q"
    elif model == "pgm3p25":
        # the hydrogens rebuilt at the paper's geometry (scripts/dielectric/water_dielectric.py)
        paper_geometry = load_script(os.path.join(scripts_dir(), "dielectric", "water_dielectric.py")).paper_geometry
        xyz = paper_geometry(xyz, 0.9745, 103.64, [list(m.elements) for m in mols])
        sig, eps = 3.18156, 0.14473  # A, kcal/mol
        rh = np.array([2 ** (1 / 6) * sig / 2 * 0.1, 0.0, 0.0])
        se = np.array([np.sqrt(eps * KCAL), 0.0, 0.0])
        new = {id(m): dataclasses.replace(m, lj_rmin_half=rh, lj_sqrt_eps=se) for m in mols}
        mols = [new[id(m)] for m in mols]
        vel = None
    elif model != "pgm":
        raise ValueError(model)
    return mols, xyz, vel, box, elec


def lattice_box(mols: list, xyz: np.ndarray, n: int, seed: int = 0) -> tuple[list, np.ndarray, tuple]:
    """Return n^3 randomly rotated copies of the first molecule on a cubic lattice at 1 g/cm^3.

    A small box for cheap checks (equilibrate it with --npt-ps).

    Parameters
    ----------
    mols : list of Molecule
        Molecules of the box (the first one is copied).
    xyz : np.ndarray (N, 3)
        Coordinates [A] (the first molecule's geometry is used).
    n : int
        Copies per box edge.
    seed : int
        Seed of the random rotations.

    Returns
    -------
    mols : list of Molecule
    xyz : np.ndarray (n^3 A, 3)
        Coordinates [A].
    box : tuple
        (lengths [A], angles [deg]) of the cubic box.
    """
    m = mols[0]
    x = np.asarray(xyz[: m.n], float)
    x = x - x.mean(axis=0)
    rng = np.random.default_rng(seed)
    a = (float(np.sum(m.masses)) * 1.66053906660 / 1.0) ** (1.0 / 3.0)  # A per molecule at 1 g/cm^3
    pos = [
        x @ np.linalg.qr(rng.normal(size=(3, 3)))[0].T + (np.array([i, j, k]) + 0.5) * a
        for i in range(n)
        for j in range(n)
        for k in range(n)
    ]
    L = n * a
    print(f"# lattice box: {n**3} x {m.name}, {L:.3f} A", flush=True)
    return [m] * n**3, np.concatenate(pos), (np.full(3, L), np.full(3, 90.0))


def insert_solute(
    tpl: FlexibleTemplate, mols: list, xyz_nm: np.ndarray, H: np.ndarray, clear: float
) -> tuple[list, np.ndarray, list]:
    """Put the template's molecule at the centre of the box and remove the waters that overlap it.

    Parameters
    ----------
    tpl : FlexibleTemplate
        The solute (inserted at its reference geometry).
    mols : list of Molecule
        Molecules of the box.
    xyz_nm : np.ndarray (N, 3)
        Coordinates [nm].
    H : np.ndarray (3, 3)
        Box [nm].
    clear : float
        Molecules with an atom within this distance of a solute atom are removed [nm].

    Returns
    -------
    mols : list of Molecule
        The solute first, then the kept molecules.
    positions : np.ndarray (N', 3)
        Positions [nm].
    templates : list
        The solute's template, then a RigidTemplate per kept molecule.
    """
    x = np.asarray(tpl.spec.ref_xyz, float)
    x = x - x.mean(axis=0) + 0.5 * H.sum(axis=0)
    Hinv = np.linalg.inv(H)
    keep_m, keep_x, templates, rigid = [tpl.pgm], [x], [tpl], {}
    off = 0
    for m in mols:
        y = xyz_nm[off : off + m.n]
        off += m.n
        d = y[:, None, :] - x[None, :, :]
        d = d - np.round(d @ Hinv) @ H
        if np.min(np.linalg.norm(d, axis=-1)) < clear:
            continue
        keep_m.append(m)
        keep_x.append(y)
        templates.append(rigid.setdefault(id(m), RigidTemplate(m, y)))
    print(f"# solute {tpl.name} ({tpl.n} atoms) inserted, {len(mols) - len(keep_m) + 1} waters removed", flush=True)
    return keep_m, np.concatenate(keep_x), templates


def build(a: argparse.Namespace) -> tuple:
    """Return the alchemical system and its settings from the options.

    Parameters
    ----------
    a : argparse.Namespace
        Options: model or prmtop / coords / elec, lattice, solute_template, clear_nm,
        rigid_solute, solute, cutoff_nm, ewald_beta_per_nm, nfft, order, dipole_tol, precision, seed.

    Returns
    -------
    tuple
        (alchemical System, parameters, positions (N, 3) [nm], velocities [nm/ps] or None, box (3, 3)
        [nm], MDSettings, electrostatics level, templates (None: rigid engine)).

    Raises
    ------
    ValueError
        Coordinates without a box, or --solute other than 0 with --solute-template.
    """
    templates = None
    if a.prmtop:
        mols = read_prmtop_molecules(a.prmtop)
        xyz, vel, box = read_coordinates(a.coords)
        elec = a.elec
    else:
        mols, xyz, vel, box, elec = water_model(a.model)
    if getattr(a, "lattice", 0):
        mols, xyz, box = lattice_box(mols, xyz, a.lattice, a.seed)
        vel = None
    if box is None:
        raise ValueError("coordinates have no periodic box")
    H = box_from_cell(*box) * 0.1
    xyz = xyz * 0.1
    if getattr(a, "solute_template", None):
        mols, xyz, templates = insert_solute(FlexibleTemplate.load(a.solute_template), mols, xyz, H, a.clear_nm)
        vel = None
        if getattr(a, "rigid_solute", False):  # the template's reference geometry, rigid
            templates = None
        if a.solute != 0:
            raise ValueError("with --solute-template the solute is molecule 0")
    sys0 = System(mols)
    sysA, P = alchemical_system(sys0, a.solute)
    settings = MDSettings().replace(
        cutoff=a.cutoff_nm,
        skin=0.1,
        ewald_beta=a.ewald_beta_per_nm,
        pme_grid=tuple(a.nfft),
        pme_order=a.order,
        lj_lrc=True,
        dipole_tol=a.dipole_tol,
        precision=a.precision,
        elec=elec,
    )
    return sysA, P, xyz, None if vel is None else vel * 0.1, H, settings, elec, templates


def engine(
    sysA: System, templates: list | None, pos: np.ndarray, H: np.ndarray, **kw
) -> Simulation | FlexibleSimulation:
    """Return a Simulation (rigid molecules) or FlexibleSimulation (a flexible solute: X-H bonds constrained).

    `**kw` goes to the engine's constructor.
    """
    if templates is None:
        return Simulation(sysA, pos, H, **kw)
    return FlexibleSimulation(sysA, templates, pos, H, constraints="h-bonds", **kw)


def scale_to_volume(sim: Simulation | FlexibleSimulation, V: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the current positions with the molecular centres scaled to a volume, and the scaled box.

    The barostat's molecular scaling; V in nm^3, positions and box in nm.
    """
    st = sim.state
    s = (V / float(volume(st.box))) ** (1.0 / 3.0)
    if hasattr(sim, "flex"):
        pos = st.dyn.position
        return np.asarray(pos + ((s - 1.0) * sim.flex.centers(pos))[sim.flex.mol]), np.asarray(st.box) * s
    body = st.dyn.position
    body = RigidBody(body.center * s, body.orientation)
    return np.asarray(sim.rigid.positions(body)), np.asarray(st.box) * s


def cmd_run(a: argparse.Namespace) -> None:
    """Build the solvated system and the lambda windows, and sample them (the `run` subcommand).

    Parameters
    ----------
    a : argparse.Namespace
        Parsed options.
    """
    sysA, P, pos, vel, H, settings, elec, templates = build(a)
    mode = a.intramolecular or ("keep" if templates is not None else "annihilate")
    if templates is not None and mode != "keep":
        raise ValueError(
            "a flexible pGM solute needs --intramolecular keep: its bonded terms were fitted with its "
            "intramolecular electrostatics"
        )
    alch = Alchemy(sysA, a.solute, sc_alpha=a.sc_alpha, intramolecular=mode)
    quant = None if not a.grad_quantities else a.grad_quantities.split(",")
    space = ParameterSpace.values(sysA.table, quant)
    scales = parse_scales(a.solute_scale)
    if scales:
        P = fg.scaled_params(space, P, scales, solute=True)
        print(f"# solute parameters scaled: {scales}", flush=True)
    kw = dict(
        settings=settings,
        dt=a.dt_fs / 1000.0,
        temperature=a.temperature_K,
        thermostat=Bussi(1.0),
        params=P,
        alchemy=alch,
        log=sys.stdout,
        seed=a.seed,
    )
    lam = standard_schedule(a.n_elec, None if a.vdw is None else [float(x) for x in a.vdw.split(",")])
    if a.elec_only:
        lam = lam[: a.n_elec]
    if a.continue_from is None and a.start_from is None and a.npt_ps > 0:
        npt = engine(sysA, templates, pos, H, barostat=MonteCarloBarostat(1.0, 100), velocities=vel, **kw)
        n = int(round(a.npt_ps / (a.dt_fs / 1000.0)))
        rep = max(n // 20, 1)
        vols = []
        for _ in range(20):
            npt.run(rep, report_every=rep, prefix=a.out + "_npt", append=bool(vols))
            vols.append(float(volume(npt.state.box)))
        Vm = float(np.mean(vols[10:]))
        print(
            f"# NPT {a.npt_ps} ps: volume {vols[-1]:.4f} nm^3, mean of the second half {Vm:.4f} nm^3 "
            f"(density {float(np.sum(sysA.masses)) / Vm * AMU_NM3_TO_G_CM3:.4f} g/cm^3)",
            flush=True,
        )
        pos, H = scale_to_volume(npt, Vm)
        vel = npt.velocities()
        del npt
    sim = engine(sysA, templates, pos, H, velocities=vel, **kw)
    meta = {
        "model": a.model if not a.prmtop else a.prmtop,
        "solute": a.solute,
        "elec": elec,
        "settings": dataclasses.asdict(settings),
        "dt_fs": a.dt_fs,
        "temperature": a.temperature_K,
        "volume_nm3": float(volume(sim.state.box)),
        "sc_alpha": a.sc_alpha,
        "alpha_floor": alch.alpha_floor,
        "intramolecular": mode,
        "solute_scale": scales,
        "elec_only": bool(a.elec_only),
    }
    if templates is not None:
        meta["solute_template"] = os.path.abspath(a.solute_template)
    if mode == "annihilate":  # rigid solute: exact gas-phase leg
        gas = GasPhaseLeg(alch, sim.positions()[sysA.atom_slice(a.solute)], elec)
        meta.update(
            gas_delta_g=gas.delta_g(P), gas_e1=gas.energy(1.0, P), gas_dudl=[gas.dudl(le, P) for le in lam[:, 0]]
        )
        if a.grad:
            meta["gas_grad"] = fg.gas_leg_gradient(gas, P, space)[1].tolist()
        print(
            f"# gas-phase leg: E_gas(1) = {meta['gas_e1']:.4f} kJ/mol, Delta G_gas(1 -> 0) = "
            f"{meta['gas_delta_g']:.4f} kJ/mol",
            flush=True,
        )
    else:  # the gas-phase leg is in the Hamiltonian
        meta.update(gas_delta_g=0.0, gas_dudl=[0.0] * len(lam))
        if a.grad:
            meta["gas_grad"] = [0.0] * space.n
    t0 = time.time()
    win = LambdaWindows(sim, lam, batched=not a.sequential, seed=a.seed + 1)

    def to_steps(ps):
        """Return the number of steps in `ps` picoseconds."""
        return int(round(ps / (a.dt_fs / 1000.0)))

    pg = fg.ParameterGradients(win, quantities=quant) if a.grad else None
    run = FreeEnergyRun(
        win,
        sample_every=to_steps(a.sample_ps),
        exchange_every=to_steps(a.exchange_ps),
        seed=a.seed,
        log=sys.stdout,
        meta=meta,
        param_grad=pg,
    )
    if a.continue_from:
        run.load_checkpoint(a.continue_from)
    elif a.start_from:
        run.load_windows(a.start_from)
        print(f"# windows start from {a.start_from}", flush=True)
    total = to_steps(a.time_ns * 1000.0)
    summary = run.run(
        total - run.step, prefix=a.out, report_every=to_steps(a.report_ps), checkpoint_every=to_steps(a.checkpoint_ps)
    )
    summary["wall_s"] = time.time() - t0
    print(json.dumps(summary, indent=1))
    report(fe.load(a.out + "_fe.npz"), a.discard_ps)


def report(d: dict, discard_ps: float) -> dict:
    """Print the free-energy analysis of a run (the `analyze` subcommand) and return the estimates.

    Statistical inefficiencies, smallest MBAR overlap, per-window <dU/dlambda> and BAR / MBAR
    steps, Delta G of switching the solute off in solution (TI, BAR, MBAR, per stage), the
    hydration free energy with the gas-phase leg, the two halves of the run, the parameter
    gradients (when sampled) and the detected equilibration times.

    Parameters
    ----------
    d : dict
        The run's samples (analysis.free_energy.load of <out>_fe.npz).
    discard_ps : float
        Time discarded at the start of every window [ps].

    Returns
    -------
    dict
        analysis.free_energy.estimate's result [kJ/mol].
    """
    meta = d["meta"]
    gas = {"delta_g": meta["gas_delta_g"], "dudl": meta["gas_dudl"]} if "gas_delta_g" in meta else None
    r = fe.estimate(d, discard_ps=discard_ps, gas=gas)

    def k(x):
        """Convert kJ/mol to kcal/mol."""
        return x / KCAL

    print(
        f"# {r['windows']} windows, {r['samples_per_window']} samples each after {discard_ps} ps; "
        f"kT = {r['kT']:.4f} kJ/mol"
    )
    print("# statistical inefficiency (dE):", " ".join(f"{g:.1f}" for g in r["g_dE"]))
    print("# statistical inefficiency (dU/dl):", " ".join(f"{g:.1f}" for g in r["g_dudl"]))
    print(f"# smallest neighbour overlap (MBAR): {r['overlap_min']:.3f}")
    print("#  window  lambda_e lambda_v   <dU/dl_e>      <dU/dl_v>     (kJ/mol)   MBAR step   BAR step")
    L = d["lambdas"]
    for i in range(r["windows"]):
        st = (
            ""
            if i == r["windows"] - 1
            else f"{r['mbar_steps'][i]:10.3f} {r['bar_steps'][i]:10.3f} +- {r['bar_steps_err'][i]:.3f}"
        )
        print(
            f"  {i:6d} {L[i, 0]:9.3f} {L[i, 1]:8.3f} {r['dudl_mean'][i][0]:12.3f} {r['dudl_mean'][i][1]:12.3f}   {st}"
        )
    print("# Delta G of switching the solute off in solution (kJ/mol | kcal/mol):")
    for m in ("ti", "bar", "mbar"):
        print(f"  {m:5s} {r[m]:10.3f} +- {r[m + '_err']:.3f} | {k(r[m]):9.3f} +- {k(r[m + '_err']):.3f}")
    if "mbar_elec" in r:
        print(
            f"  stages (MBAR): electrostatics {r['mbar_elec']:.3f} +- {r['mbar_elec_err']:.3f}, van der Waals "
            f"{r['mbar_vdw']:.3f} +- {r['mbar_vdw_err']:.3f} kJ/mol; TI {r['ti_elec']:.3f}, {r['ti_vdw']:.3f}"
        )
    if gas is not None:
        print(
            f"# gas-phase leg Delta G_gas(1 -> 0) = {r['gas']:.4f} kJ/mol; hydration free energy (kJ/mol | kcal/mol):"
        )
        for m in ("ti", "ti_sub", "bar", "mbar"):
            v, e = r[f"dG_hyd_{m}"], r[f"dG_hyd_{m}_err"]
            print(f"  {m:6s} {v:10.3f} +- {e:.3f} | {k(v):9.3f} +- {k(e):.3f}")
        t = np.asarray(d["time_ps"])
        mid = 0.5 * (discard_ps + t[-1])
        if np.sum((t > discard_ps) & (t <= mid)) >= 20:
            h1 = fe.estimate(d, discard_ps=discard_ps, gas=gas, end_ps=mid)
            h2 = fe.estimate(d, discard_ps=mid, gas=gas)
            print(
                f"# halves ({discard_ps:g}-{mid:g} ps | {mid:g}-{t[-1]:g} ps), MBAR: "
                f"{k(h1['dG_hyd_mbar']):.3f} +- {k(h1['dG_hyd_mbar_err']):.3f} | "
                f"{k(h2['dG_hyd_mbar']):.3f} +- {k(h2['dG_hyd_mbar_err']):.3f} kcal/mol"
            )
    if "dudp" in d:
        grad_report(d, discard_ps)
    teq = fe.equilibration_times(d)
    print(
        f"# equilibration detected (ps after the windows start): max {teq.max():.0f}, per window "
        + " ".join(f"{x:.0f}" for x in teq)
    )
    return r


def parse_scales(text: str | None) -> dict[str, float]:
    """Return the scale factors of --solute-scale: 'charge=1.05,eps=0.9' -> {'charge': 1.05, 'eps': 0.9}."""
    out = {}
    for item in (text or "").split(","):
        if item.strip():
            k, v = item.split("=")
            out[k.strip()] = float(v)
    return out


def grad_report(d: dict, discard_ps: float, n_blocks: int = 10) -> tuple[dict, dict]:
    """Print the parameter gradients of the hydration free energy (or of the solution leg without a gas leg).

    MBAR-weighted and end-state estimators with block jackknife errors; derivatives along the scale
    directions of the solute's and of the environment's parameters [kcal/mol per ln s], and per
    solute entry [kJ/mol per unit of the parameter].

    Parameters
    ----------
    d : dict
        The run's samples (with "dudp").
    discard_ps : float
        Time discarded at the start of every window [ps].
    n_blocks : int
        Blocks of the jackknife.

    Returns
    -------
    (dict, dict)
        fit.free_energy.gradient_estimate's result, and the printed scale derivatives [kcal/mol].
    """
    meta = d["meta"]
    gas = {"delta_g": meta["gas_delta_g"], "grad": meta.get("gas_grad")} if "gas_delta_g" in meta else None
    r = gradient_estimate(d, discard_ps=discard_ps, gas=gas, n_blocks=n_blocks)
    leg = "hyd" if "hyd" in r else "solv"
    space = ParameterSpace.from_names(r["names"])
    p = np.asarray(meta["params_flat"], float)

    def k(x):
        """Convert kJ/mol to kcal/mol."""
        return x / KCAL

    m, e = r[leg]["mbar"], r[leg]["end"]  # k() converts kJ/mol to kcal/mol
    print(
        f"# parameter gradients of {'the hydration free energy' if leg == 'hyd' else 'Delta G(first -> last)'}, "
        f"block jackknife over {n_blocks} blocks: value {k(m.value):.3f} +- {k(m.value_err):.3f} kcal/mol"
    )
    print("#  d/d ln s (kcal/mol)           MBAR               end states")
    out = {"value_kcal": k(m.value), "value_err_kcal": k(m.value_err), "scale": {}}
    for who, sol in (("solute", True), ("environment", False)):
        for g in SCALE_GROUPS:
            v = space.scale_direction(p, g, sol)
            if not np.any(v):
                continue
            (a1, b1), (a2, b2) = m.project(v), e.project(v)
            out["scale"][f"{who}:{g}"] = {"mbar": [k(a1), k(b1)], "end": [k(a2), k(b2)]}
            print(f"  {who:12s} {g:7s} {k(a1):9.3f} +- {k(b1):.3f}   {k(a2):9.3f} +- {k(b2):.3f}")
    print("#  solute entries: dG/dp (kJ/mol per unit; MBAR | end states)")
    for i in space.select(solute=True):
        if p[i] != 0.0 or m.grad[i] != 0.0:
            print(
                f"  {r['names'][i]:28s} p = {p[i]:11.6f}  {m.grad[i]:12.4f} +- {m.grad_err[i]:.4f} | "
                f"{e.grad[i]:12.4f} +- {e.grad_err[i]:.4f}"
            )
    return r, out


def cmd_bench(a: argparse.Namespace) -> None:
    """Print the cost of plain MD, of one alchemical window and of K batched windows (the `bench` subcommand).

    ms per step of plain MD, of one alchemical window, and per window of K batched windows (K
    windows spread over the schedule), the cost of one sample of all windows (and of the
    parameter-gradient samples with --grad), and the corresponding ns/day.
    """
    sysA, P, pos, vel, H, settings, elec, templates = build(a)
    kw = dict(
        settings=settings,
        dt=a.dt_fs / 1000.0,
        temperature=a.temperature_K,
        thermostat=Bussi(1.0),
        params=P,
        log=None,
        seed=a.seed,
        velocities=vel,
    )
    lam = standard_schedule(a.n_elec)
    n = a.steps

    def timed(f, reps=3):
        """Return the best wall time [s] of `reps` calls of f (after one warm-up call)."""
        f()
        best = np.inf
        for _ in range(reps):
            t = time.time()
            f()
            best = min(best, time.time() - t)
        return best

    def md(sim):
        """Return ms per MD step of sim."""

        def f():
            """Advance n steps and wait for the result."""
            sim.advance(n)
            jax.block_until_ready(sim.state.epot)

        return timed(f) / n * 1e3

    plain = engine(sysA, templates, pos, H, **kw)  # the same system, no alchemical region
    out = {"atoms": sysA.n, "steps": n, "plain_ms_per_step": md(plain)}
    del plain
    mode = "keep" if templates else "annihilate"
    sim = engine(sysA, templates, pos, H, alchemy=Alchemy(sysA, a.solute, intramolecular=mode), **kw)
    out["alchemy_1_window_ms_per_step"] = md(sim)
    for K in [int(x) for x in a.windows.split(",")]:
        if K < 2:
            continue
        idx = np.unique(np.round(np.linspace(0, len(lam) - 1, K)).astype(int))
        win = LambdaWindows(sim, lam[idx], batched=True, seed=1)

        def f():
            """Advance the batched windows n steps and wait for the result."""
            win.advance(n)
            jax.block_until_ready(win.S.epot)

        t = timed(f)
        ts = timed(lambda: win.sample(), reps=2)
        out[f"batched_{len(idx)}_ms_per_window_step"] = t / n / len(idx) * 1e3
        out[f"batched_{len(idx)}_sample_ms"] = ts * 1e3
        if a.grad:  # parameter gradients of the end states at every window
            pg = fg.ParameterGradients(win)
            out[f"batched_{len(idx)}_grad_sample_ms"] = timed(lambda: pg.sample(), reps=2) * 1e3
    for k, v in out.items():
        print(f"{k:40s} {v:.4f}" if isinstance(v, float) else f"{k:40s} {v}")

    def per_day(ms):
        """Return ns/day at `ms` milliseconds per step."""
        return a.dt_fs * 1e-6 / (ms * 1e-3) * 86400.0

    print(
        f"# ns/day: plain MD {per_day(out['plain_ms_per_step']):.1f}, one alchemical window "
        f"{per_day(out['alchemy_1_window_ms_per_step']):.1f}"
        + "".join(
            f"; {k.split('_')[1]} batched windows {per_day(v * int(k.split('_')[1])):.1f} per window "
            f"({per_day(v):.1f} aggregate)"
            for k, v in out.items()
            if k.endswith("window_step")
        )
    )


def cmd_finite_size(a: argparse.Namespace) -> None:
    """Print the periodic self-image energy of switching the solute's electrostatics off (`finite-size`).

    The solute alone in the production box (its PME settings) at --placements random positions
    and orientations: [E_pbc(1) - E_pbc(0)] - [E_gas(1) - E_gas(0)] of switching its
    electrostatics off, i.e. the periodic self-image energy plus the PME error of its
    intramolecular terms, which the solution leg carries and the gas-phase leg does not (a rigid
    solute at its geometry in the box).
    """
    sysA, P, pos, vel, H, settings, elec, templates = build(a)
    sub, idx = sysA.sub((a.solute,))
    x0 = pos[idx] - pos[idx].mean(axis=0)
    H = reduce_box(H)
    alch = Alchemy(sub, 0)
    gas = GasPhaseLeg(alch, x0, elec)
    dg = gas.energy(1.0, P) - gas.energy(0.0, P)
    ff = PGMForceField(sub, H, settings.replace(dipole_tol=min(settings.induction.tol, 1e-7), max_iter=200))
    alch.check(ff)
    E = jax.jit(lambda x, i, le: alch.energy(ff, x, H, i, ff.init_induction(), P, (le, 1.0))[0])
    rng = np.random.default_rng(a.seed)
    d = []
    for _ in range(a.placements):
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        x = x0 @ R.T + rng.uniform(size=3) @ H
        i = ff.rows_for(x, H)
        d.append(float(E(x, i, 1.0)) - float(E(x, i, 0.0)) - dg)
    d = np.array(d)
    print(
        f"# E_gas(1) - E_gas(0) = {dg:.4f} kJ/mol; periodic minus gas-phase: {d.mean():+.5f} kJ/mol "
        f"(spread {d.std():.5f} over {len(d)} placements; {d.mean() / KCAL:+.5f} kcal/mol)"
    )


def add_system_args(p: argparse.ArgumentParser, md: bool = True) -> None:
    """Add the options that define the system and the force-field settings to a subcommand.

    Parameters
    ----------
    p : argparse.ArgumentParser
        The subcommand's parser.
    md : bool
        Also the MD options (time step, temperature, solute insertion, lattice box).
    """
    p.add_argument("--model", default="pgm", choices=MODELS, help="water model of the 512-water box")
    p.add_argument("--prmtop", help="Amber pGM prmtop instead of --model (with --coords)")
    p.add_argument("--coords", help="coordinates with a box matching --prmtop")
    p.add_argument("--elec", default="qpi", help="electrostatics level for --prmtop")
    p.add_argument("--solute", type=int, default=0, help="molecule (residue) index of the solute")
    add_dipole_tol_arg(p)
    p.add_argument("--cutoff-nm", type=float, default=0.9, help="cutoff [nm]")
    p.add_argument("--ewald-beta-per-nm", type=float, default=4.0, help="Ewald coefficient [1/nm]")
    p.add_argument("--nfft", type=int, nargs=3, default=[48, 48, 48], help="PME grid")
    p.add_argument("--order", type=int, default=6, help="PME order")
    add_precision_arg(p)
    add_seed_arg(p)
    if not md:
        return
    add_dt_arg(p, 2.0)
    add_temperature_arg(p, 298.0)
    p.add_argument("--solute-template", help="flexible solute (FlexibleTemplate file) inserted into the water box")
    p.add_argument(
        "--clear-nm", type=float, default=0.25, help="waters this close to the inserted solute are removed [nm]"
    )
    p.add_argument(
        "--lattice",
        type=int,
        default=0,
        help="n: a small box of n^3 copies of the model's first "
        "molecule on a lattice (cheap checks; with --npt-ps, --cutoff-nm, --nfft for the small box)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser with the subcommands run, bench, finite-size and analyze."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="sample the lambda windows")
    r.add_argument("-o", "--out", required=True, help="output prefix")
    add_system_args(r)
    r.add_argument(
        "--rigid-solute",
        action="store_true",
        help="hold the --solute-template molecule rigid at its reference geometry (rigid engine)",
    )
    r.add_argument(
        "--intramolecular",
        choices=["annihilate", "keep"],
        help="solute's intramolecular electrostatics (default: annihilate for rigid, keep for flexible)",
    )
    r.add_argument("--time-ns", type=float, default=2.0, help="length of every window [ns]")
    r.add_argument("--npt-ps", type=float, default=100.0, help="NPT equilibration at full coupling [ps]")
    r.add_argument("--n-elec", type=int, default=8, help="electrostatics windows")
    r.add_argument("--vdw", help="lambda_vdw values of the second stage (comma separated, decreasing to 0)")
    r.add_argument("--sample-ps", type=float, default=1.0, help="time between samples [ps]")
    r.add_argument("--exchange-ps", type=float, default=1.0, help="time between exchanges [ps] (0: none)")
    r.add_argument("--report-ps", type=float, default=20.0, help="time between log lines [ps]")
    r.add_argument("--checkpoint-ps", type=float, default=200.0, help="time between checkpoints [ps]")
    r.add_argument("--discard-ps", type=float, default=200.0, help="time discarded by the final report [ps]")
    r.add_argument("--sc-alpha", type=float, default=0.5, help="soft-core alpha of the van der Waals stage")
    r.add_argument("--sequential", action="store_true", help="windows one after the other (not batched)")
    r.add_argument("--continue-from", help="continue from <prefix>.fe.chk")
    r.add_argument(
        "--start-from",
        help="start the windows from the configurations of another run's prefix.fe.chk "
        "(no NPT, no samples taken over; windows matched by lambda)",
    )
    r.add_argument("--grad", action="store_true", help="sample parameter gradients of the end states (fe_grad.py)")
    r.add_argument("--grad-quantities", help="comma-separated parameter quantities for --grad (default: all)")
    r.add_argument(
        "--solute-scale",
        help="scale solute parameters, e.g. charge=1.05,eps=0.9 (charge: q and "
        "covalent dipoles; eps, rmin, alpha, radius)",
    )
    r.add_argument("--elec-only", action="store_true", help="only the electrostatics windows (lambda_vdw = 1)")
    b = sub.add_parser("bench", help="cost per window")
    add_system_args(b)
    b.add_argument("--windows", default="1,4,8,19", help="numbers of batched windows")
    b.add_argument("--steps", type=int, default=1000, help="timed steps")
    b.add_argument("--n-elec", type=int, default=8, help="electrostatics windows of the schedule")
    b.add_argument("--grad", action="store_true", help="also time the parameter-gradient samples")
    fs = sub.add_parser("finite-size", help="periodic self-image energy of the solute")
    add_system_args(fs, md=False)
    fs.add_argument("--placements", type=int, default=40, help="random placements of the solute")
    z = sub.add_parser("analyze", help="TI, BAR, MBAR of a run")
    z.add_argument("npz", help="<out>_fe.npz of a run")
    z.add_argument("--discard-ps", type=float, default=200.0, help="time discarded at the start of every window [ps]")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the subcommand (see the module docstring)."""
    a = build_parser().parse_args(argv)
    setup_logging()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "bench":
        cmd_bench(a)
    elif a.cmd == "finite-size":
        cmd_finite_size(a)
    else:
        report(fe.load(a.npz), a.discard_ps)


if __name__ == "__main__":
    main()
