"""Speed (and, with --prod-ps, stability and configurational accuracy) of a solvated protein in the MD engine.

The system: pGM (placeholder electrostatics unless a residue library is given) + Amber-form
bonded terms and CMAP from the prmtop, rigid water by constraints, X-H constraints, hydrogen mass
repartitioning (--hmr-amu for the protein, --hmr-water-amu for water), in the flexible engine
(docs/protein_ff.md, docs/shake.md, docs/mts.md).

--prod-ps prints the mean potential energy with the error of 10 block averages (masses do not
change the configurational distribution, so <U> measures the time-step error), the mean kinetic
temperature, the largest constraint error and the drift of the effective energy econs; with
--traj-ps also the CA RMSD to the first frame and the radius of gyration of the heavy atoms.

Usage:

    python scripts/protein/bench_protein.py runs/protein/ubq.prmtop runs/protein/ubq.inpcrd --dt-fs 2 --steps 2000
    python scripts/protein/bench_protein.py ... --elec-cutoff-nm 0.7 --thermostat bussi   # LJ 0.9 nm, elec 0.7
    # water H 4.0 amu, protein H 3.024, 4 fs, Bussi; 20 ps equilibration, 200 ps sampled every 0.5 ps
    python scripts/protein/bench_protein.py ubq.prmtop ubq.inpcrd --dt-fs 4 --hmr-water-amu 4.0 --thermostat bussi
        --equil-ps 20 --prod-ps 200 --coords equilibrated.rst7 --minimize 0 --save runs/ubq_4fs
    # multiple time stepping: 8 fs outer step, short-range nonbonded + bonded forces every 4 fs
    python scripts/protein/bench_protein.py ... --dt-fs 8 --mts 2 --elec-cutoff-nm 0.7 --thermostat bussi
    python scripts/protein/bench_protein.py --help

Inputs: the tleap prmtop and coordinates (scripts/protein/build_amber.py); --library, --coords.
Outputs: printed timings and statistics; with --save <save>.chk, <save>.rst7, <save>_samples.npz
and (with --traj-ps) <save>.nc.
Units: --dt-fs fs, --cutoff-nm, --elec-cutoff-nm, --local-cut-nm and --pme-spacing-nm nm,
--ewald-beta-per-nm 1/nm, --temperature-K K, --hmr-amu amu, durations in ps; RMSD and radius of
gyration printed in A.
Runtime: GPU (the speed test: --steps steps after minimisation and compilation).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import numpy as np

from pgm_jax.cli.args import (
    add_barostat_args,
    add_dipole_tol_arg,
    add_dt_arg,
    add_mts_args,
    add_precision_arg,
    add_seed_arg,
    add_temperature_arg,
    add_thermostat_args,
    coupling_from_args,
    mts_from_args,
    setup_logging,
)
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.flexible import FlexibleSimulation
from pgm_jax.md.forcefield import DSUM_TOL, MDSettings, elec_cutoff_settings
from pgm_jax.md.io import NetCDFTrajectory, read_coordinates
from pgm_jax.md.mts import mts_stats
from pgm_jax.protein import ResidueLibrary, amber_template, load_amber
from pgm_jax.units import KB

jax.config.update("jax_enable_x64", True)
COLUMNS = (
    "time_ps",
    "epot",
    "temp_K",
    "econs",
    "shake_err",
    "cg_iter_max",
    "temp_com",
    "temp_internal",
    "density_g_cm3",
)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prmtop", help="tleap prmtop")
    ap.add_argument("inpcrd", help="tleap coordinates")
    ap.add_argument("--library", default=None, help="pGM residue library (JSON); default: placeholder")
    add_dt_arg(ap, 2.0)
    ap.add_argument("--steps", type=int, default=2000, help="timed steps (multiples of 500)")
    ap.add_argument(
        "--cutoff-nm", type=float, default=0.9, help="van der Waals (and default electrostatics) cutoff [nm]"
    )
    ap.add_argument(
        "--elec-cutoff-nm",
        type=float,
        default=None,
        help="real-space electrostatics cutoff [nm]; --ewald-beta-per-nm and "
        "--pme-spacing-nm then default to elec_cutoff_settings(elec_cutoff, dsum_tol)",
    )
    ap.add_argument(
        "--dsum-tol", type=float, default=DSUM_TOL, help="direct-sum tolerance for --elec-cutoff-nm (Amber convention)"
    )
    ap.add_argument(
        "--pme-exponent",
        type=float,
        default=1.6,
        help="grid rule for --elec-cutoff-nm: spacing 0.08 (4 / beta)^exponent "
        "(1.6: the default's accuracy; 1: beta x spacing fixed, faster, less accurate)",
    )
    add_dipole_tol_arg(ap)
    add_precision_arg(ap)
    ap.add_argument(
        "--hmr-amu",
        type=float,
        default=3.024,
        help="hydrogen mass [amu] of the protein (and of water unless --hmr-water-amu is given)",
    )
    ap.add_argument("--hmr-water-amu", type=float, default=None, help="hydrogen mass [amu] of water (e.g. 4.0)")
    ap.add_argument(
        "--local-niter", type=int, default=0, help="inner CG steps of the short-range preconditioner (0: Jacobi)"
    )
    ap.add_argument("--local-cut-nm", type=float, default=0.3, help="preconditioner range [nm]")
    ap.add_argument("--predictor", default="mu4", help="initial induced-dipole guess (MDSettings predictor)")
    ap.add_argument(
        "--ewald-beta-per-nm",
        type=float,
        default=None,
        help="Ewald coefficient [1/nm]; default 4.0 (or from --elec-cutoff-nm)",
    )
    ap.add_argument(
        "--pme-spacing-nm",
        type=float,
        default=None,
        help="PME grid spacing [nm]; default 0.08 (or from --elec-cutoff-nm)",
    )
    add_temperature_arg(ap, 298.0)
    add_thermostat_args(ap, default="langevin", choices=("langevin", "bussi", "gle", "gle-lowpass"))
    add_barostat_args(ap, default="none")
    add_seed_arg(ap)
    ap.add_argument(
        "--coords",
        default=None,
        help="start from these coordinates and box (Amber restart in the "
        "system's atom order, e.g. the .rst7 of --save) instead of the inpcrd; velocities are redrawn",
    )
    ap.add_argument("--minimize", type=int, default=300, help="steepest-descent steps before dynamics (0: none)")
    ap.add_argument("--equil-ps", type=float, default=0.0, help="equilibration after the speed test [ps]")
    ap.add_argument("--prod-ps", type=float, default=0.0, help="sampled production after equilibration [ps]")
    ap.add_argument("--sample-ps", type=float, default=0.5, help="sampling interval of the production [ps]")
    ap.add_argument("--save", default=None, help="prefix: restart + checkpoint at the end, samples as .npz")
    ap.add_argument(
        "--traj-ps",
        type=float,
        default=0.0,
        help="production frames every this many ps: CA RMSD to the first "
        "frame and radius of gyration of the heavy atoms (and, with --save, the trajectory prefix.nc); 0: none",
    )
    add_mts_args(ap)
    return ap


def build(a: argparse.Namespace) -> tuple[FlexibleSimulation, object, list[int], MDSettings, float, float]:
    """Return the simulation, the Amber system, the protein molecule indices, settings, beta [1/nm], spacing [nm]."""
    ew = (
        {"ewald_beta": 4.0, "pme_spacing": 0.08}
        if a.elec_cutoff_nm is None
        else elec_cutoff_settings(a.elec_cutoff_nm, a.dsum_tol, a.pme_exponent)
    )
    beta = ew["ewald_beta"] if a.ewald_beta_per_nm is None else a.ewald_beta_per_nm
    spacing = ew["pme_spacing"] if a.pme_spacing_nm is None else a.pme_spacing_nm
    lib = ResidueLibrary.load(a.library) if a.library else "placeholder"
    asys = load_amber(a.prmtop, a.inpcrd, electrostatics=lib)
    prot = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
    tpl = {k: amber_template(asys.molecules[k], a.prmtop) for k in prot}
    st = MDSettings().replace(
        cutoff=a.cutoff_nm,
        skin=0.1,
        dipole_tol=a.dipole_tol,
        precision=a.precision,
        local_niter=a.local_niter,
        local_cut=a.local_cut_nm,
        predictor=a.predictor,
        ewald_beta=beta,
        pme_spacing=spacing,
        elec_cutoff=a.elec_cutoff_nm,
    )
    h_water = a.hmr_amu if a.hmr_water_amu is None else a.hmr_water_amu
    hmr = asys.hmr({"protein": a.hmr_amu, "other": a.hmr_amu, "water": h_water, "ion": None})
    pos, H = asys.system_positions(), asys.box
    if a.coords:
        xyz, _, cell = read_coordinates(a.coords)
        pos, H = xyz * 0.1, box_from_cell(*cell) * 0.1
    thermostat, barostat = coupling_from_args(a)
    sim = FlexibleSimulation(
        asys.system(),
        asys.templates(tpl),
        pos,
        H,
        st,
        dt=a.dt_fs / 1000,
        temperature=a.temperature_K,
        thermostat=thermostat,
        barostat=barostat,
        constraints="h-bonds",
        hmr=hmr,
        log=sys.stdout,
        seed=a.seed,
        mts=mts_from_args(a),
    )
    return sim, asys, prot, st, beta, spacing


def speed_test(a: argparse.Namespace, sim: FlexibleSimulation) -> None:
    """Minimise, compile, then time --steps steps (in blocks of 500) and print ms/step, ns/day, CG, memory."""
    dt = sim.dt
    t1 = time.time()
    if a.minimize > 0:
        print("minimise:", sim.minimize(a.minimize), flush=True)
    sim.advance(500)
    print(f"minimise + compile + 500 steps {time.time() - t1:.1f} s", flush=True)
    t0 = time.time()
    done = 0
    while done < a.steps:
        sim.advance(500)
        done += 500
    el_t = time.time() - t0
    o = sim.observables()
    print(
        f"{sim.sys.n} atoms, dt {dt * 1000:g} fs, {a.precision}: {el_t / done * 1e3:.3f} ms/step, "
        f"{done * dt / 1000 / el_t * 86400:.1f} ns/day; T {o['temp_K']:.1f} K, CG iters {o['cg_mean']:.2f} mean (max "
        f"{o['cg_iter_max']}), "
        f"rows {sim.ff.mc} (electrostatic {sim.ff.mc_e or sim.ff.mc}), "
        f"shake {o['shake_err']:.1e}{'; ' + str(mts_stats(sim)) if a.mts else ''}",
        flush=True,
    )
    mem = jax.devices()[0].memory_stats() or {}
    if "peak_bytes_in_use" in mem:
        print(f"GPU memory peak {mem['peak_bytes_in_use'] / 2**30:.2f} GiB", flush=True)


def structure_report(frames: list) -> None:
    """Print the CA RMSD to the first frame and the heavy-atom radius of gyration [A] of (CA, heavy) frames.

    Molecules are whole in the engine, so the protein needs no unwrapping; the RMSD is after the
    optimal superposition (Kabsch).
    """
    ref = frames[0][0] - frames[0][0].mean(0)
    rmsd, rg = [], []
    for xc, xh in frames:
        y = xc - xc.mean(0)
        U, _, Vt = np.linalg.svd(y.T @ ref)
        R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        rmsd.append(np.sqrt(np.mean(np.sum((y @ R - ref) ** 2, 1))))
        h = xh - xh.mean(0)
        rg.append(np.sqrt(np.mean(np.sum(h * h, 1))))
    rmsd, rg, half = np.array(rmsd), np.array(rg), len(frames) // 2
    print(
        f"{len(frames)} frames: CA RMSD to the first (A) mean {rmsd.mean():.2f}, second half "
        f"{rmsd[half:].mean():.2f}, "
        f"max {rmsd.max():.2f}; radius of gyration of the heavy atoms (A) {rg.mean():.2f} +- {rg.std():.2f} "
        f"(halves {rg[:half].mean():.2f} / {rg[half:].mean():.2f})",
        flush=True,
    )


def production(a: argparse.Namespace, sim: FlexibleSimulation, asys: object, nblk: int) -> None:
    """Sample --prod-ps every --sample-ps (blocks of nblk steps) and print the statistics (module docstring)."""
    dt = sim.dt
    rows = []
    cg0, s0 = float(sim.state.cg_total), int(sim.state.step)
    tfile, frames = None, []
    if a.traj_ps > 0:
        tevery = max(1, int(round(a.traj_ps / a.sample_ps)))
        ca, heavy = asys.select("ca"), asys.select("heavy")
        if a.save:
            tfile = NetCDFTrajectory(a.save + ".nc", sim.sys.n)
    t0 = time.time()
    for i in range(int(round(a.prod_ps / a.sample_ps))):
        sim.advance(nblk)
        o = sim.observables()
        rows.append([o[c] for c in COLUMNS])
        if a.traj_ps > 0 and (i + 1) % tevery == 0:
            x = sim.positions()
            frames.append((x[ca] * 10.0, x[heavy] * 10.0))
            if tfile is not None:
                tfile.write(sim.time_ps, x * 10.0, np.asarray(sim.state.box) * 10.0)
    wall = time.time() - t0
    X = np.array(rows, float)
    t, U, T, E = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    nb = 10
    n = len(U) // nb * nb
    Ub = U[len(U) - n :].reshape(nb, -1).mean(1)
    kT = KB * a.temperature_K
    drift = np.polyfit(t, E, 1)[0] * 1000.0 / sim.integ.dof / kT  # kT per ns per degree of freedom
    steps = int(sim.state.step) - s0
    h_water = a.hmr_amu if a.hmr_water_amu is None else a.hmr_water_amu
    print(
        f"production {a.prod_ps:g} ps, dt {dt * 1000:g} fs, H masses protein {a.hmr_amu} / water "
        f"{h_water}: <U> {U.mean():.1f} +- {Ub.std(ddof=1) / np.sqrt(nb):.1f} "
        "kJ/mol "
        f"(std {U.std():.1f}), <T> {T.mean():.2f} K (min {T.min():.1f}, max {T.max():.1f}; centre of mass "
        f"{X[:, 6].mean():.2f}, internal {X[:, 7].mean():.2f}), density {X[:, 8].mean():.4f} +- "
        f"{X[len(X) - n :, 8].reshape(nb, -1).mean(1).std(ddof=1) / np.sqrt(nb):.4f} g/cm3, shake max "
        f"{X[:, 4].max():.1e}, "
        f"econs drift {drift:+.4f} kT/ns/dof, CG {(float(sim.state.cg_total) - cg0) / steps:.2f} mean "
        f"(max {int(X[:, 5].max())}), {steps * dt / 1000 / wall * 86400:.1f} ns/day"
        f"{'; ' + str(mts_stats(sim)) if a.mts else ''}",
        flush=True,
    )
    if frames:
        structure_report(frames)
    if a.save:
        np.savez(a.save + "_samples.npz", **{c: X[:, i] for i, c in enumerate(COLUMNS)}, dof=sim.integ.dof)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, build the system and run the tests (see the module docstring)."""
    a = build_parser().parse_args(argv)
    setup_logging()
    t0 = time.time()
    sim, asys, prot, st, beta, spacing = build(a)
    m = np.asarray(sim.flex.masses)
    el = np.array(sim.sys.elements)
    prot_sl = sim.sys.atom_slice(prot[0])
    h_prot = sorted({float(x) for x in np.round(m[prot_sl][el[prot_sl] == "H"], 4)})
    wat = [sim.sys.atom_slice(k) for k, mm in enumerate(asys.molecules) if mm.kind == "water"]
    print(
        f"setup {time.time() - t0:.1f} s: {sim.sys.n} atoms, {len(prot)} protein chain(s) "
        f"({sum(asys.molecules[k].n for k in prot)} atoms), {sim.topology.n_group} groups, "
        f"special width {sim.topology.special.shape[1]}, rows {sim.ff.mc} (electrostatic {sim.ff.mc_e or sim.ff.mc}); "
        f"{st.describe_cutoffs()}, beta {beta:.4f} /nm, spacing {spacing:.4f} nm; hydrogen masses: protein "
        f"{h_prot}, "
        f"water {np.round(m[wat[0]], 4).tolist() if wat else '-'}, lightest heavy atom {m[el != 'H'].min():.3f} amu",
        flush=True,
    )
    speed_test(a, sim)
    nblk = max(1, int(round(a.sample_ps / sim.dt)))
    if a.equil_ps > 0:
        t0 = time.time()
        for _ in range(int(round(a.equil_ps / a.sample_ps))):
            sim.advance(nblk)
        o = sim.observables()
        print(
            f"equilibrated {a.equil_ps:g} ps ({time.time() - t0:.0f} s): T {o['temp_K']:.1f} K, epot {o['epot']:.1f}",
            flush=True,
        )
    if a.prod_ps > 0:
        production(a, sim, asys, nblk)
    if a.save:
        sim.save_checkpoint(a.save + ".chk")
        sim.write_restart(a.save + ".rst7")


if __name__ == "__main__":
    main()
