"""Hydration (solvation) free energy of one rigid molecule by alchemical lambda windows
(pgm_jax/md/alchemy.py, estimators in pgm_jax/md/free_energy.py; docs/free_energy.md).

    # water in water: the 512-water box of the README (pGM3P-25 electrostatics on TIP3P geometry and LJ)
    python scripts/solvation_free_energy.py run --model pgm -o runs/fe/pgm --ns 2
    python scripts/solvation_free_energy.py run --model tip3p -o runs/fe/tip3p --ns 2       # TIP3P control
    python scripts/solvation_free_energy.py run --prmtop sys.prmtop --coords sys.rst7 --solute 0 -o runs/fe/x
    python scripts/solvation_free_energy.py run --solute-template methanol.flex -o runs/fe/meoh    # flexible solute
    python scripts/solvation_free_energy.py analyze runs/fe/pgm_fe.npz --discard-ps 200
    python scripts/solvation_free_energy.py run ... --checkpoint runs/fe/pgm.fe.chk        # continue a run
    python scripts/solvation_free_energy.py bench --model pgm --windows 1,4,19              # cost per window
    python scripts/solvation_free_energy.py finite-size --model pgm                          # PME / image check

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
The solute is the molecule --solute (0: the first water) with its own copy of the parameters.

Flexible solutes (--solute-template, a FlexibleTemplate from pgm_jax.bonded, e.g. methanol): the
molecule is put at the centre of the water box (waters within --clear nm of it removed; with
--rigid-solute it is held rigid at its reference geometry in the rigid engine instead) and the
flexible engine runs everything (rigid waters by constraints, X-H bonds of the solute constrained,
2 fs).  Its intramolecular electrostatics is kept at every lambda by the gas-phase correction
(--intramolecular keep, the default for them): the decoupled state is the gas-phase molecule, so
Delta G_hyd = -Delta G(1 -> 0) with no separate gas leg.  Rigid solutes default to annihilation with
the exact gas-phase leg (--intramolecular annihilate); both modes give the same free energy.

Parameter gradients (pgm_jax/md/fe_grad.py, docs/fe_gradients.md): `run --grad` also samples
dU/dP of the two end-state Hamiltonians at every window's configuration, and `analyze` prints
d DeltaG_hyd / dP (MBAR-weighted and end-state estimators, block jackknife errors), with the
derivatives along scale directions of the solute's and the environment's parameters (charge =
charges and covalent dipoles, eps, rmin, alpha, radius).  --solute-scale charge=1.05,eps=0.9 runs
at scaled solute parameters (finite-difference checks: scripts/fe_gradient_check.py);
--start-from prefix.fe.chk starts the windows from another run's configurations (no NPT);
--elec-only runs only the electrostatics stage (its free energy is the whole dependence on the
solute's electrostatic parameters: the van der Waals stage runs at lambda_elec = 0)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgm_jax.md import fe_grad as fg  # noqa: E402
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


def lattice_box(mols, xyz, n: int, seed: int = 0):
    """n^3 copies of the first molecule (its geometry in xyz, A) on a cubic lattice at 1 g/cm^3,
    randomly rotated: a small box for cheap checks (equilibrate it with --npt-ps)."""
    m = mols[0]
    x = np.asarray(xyz[:m.n], float)
    x = x - x.mean(axis=0)
    rng = np.random.default_rng(seed)
    a = (float(np.sum(m.masses)) * 1.66053906660 / 1.0) ** (1.0 / 3.0)          # A per molecule at 1 g/cm^3
    pos = [x @ np.linalg.qr(rng.normal(size=(3, 3)))[0].T + (np.array([i, j, k]) + 0.5) * a
           for i in range(n) for j in range(n) for k in range(n)]
    L = n * a
    print(f"# lattice box: {n ** 3} x {m.name}, {L:.3f} A", flush=True)
    return [m] * n ** 3, np.concatenate(pos), (np.full(3, L), np.full(3, 90.0))


def insert_solute(tpl, mols, xyz_nm, H, clear: float):
    """The template's molecule at the centre of the box (its reference geometry), waters with an atom
    within `clear` nm of it removed: (molecules, positions nm, templates) with the solute first."""
    from pgm_jax.md.flexible import RigidTemplate
    x = np.asarray(tpl.spec.ref_xyz, float)
    x = x - x.mean(axis=0) + 0.5 * H.sum(axis=0)
    Hinv = np.linalg.inv(H)
    keep_m, keep_x, templates, rigid = [tpl.pgm], [x], [tpl], {}
    off = 0
    for m in mols:
        y = xyz_nm[off:off + m.n]
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


def build(a):
    templates = None
    if a.prmtop:
        mols = _dedupe(read_prmtop_pgm(a.prmtop, first_residue_only=False))
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
        from pgm_jax.md.flexible import FlexibleTemplate
        mols, xyz, templates = insert_solute(FlexibleTemplate.load(a.solute_template), mols, xyz, H, a.clear)
        vel = None
        if getattr(a, "rigid_solute", False):                 # the template's reference geometry, rigid
            templates = None
        if a.solute != 0:
            raise ValueError("with --solute-template the solute is molecule 0")
    sys0 = System(mols)
    sysA, P = alchemical_system(sys0, a.solute)
    settings = MDSettings(cutoff=a.cut, skin=0.1, ewald_beta=a.ew_coeff, pme_grid=tuple(a.nfft), pme_order=a.order,
                          lj_lrc=True, dipole_tol=a.tol, precision=a.precision, elec=elec)
    return sysA, P, xyz, None if vel is None else vel * 0.1, H, settings, elec, templates


def engine(sysA, templates, pos, H, **kw):
    """Simulation (rigid molecules) or FlexibleSimulation (a flexible solute: X-H bonds constrained)."""
    if templates is None:
        return Simulation(sysA, pos, H, **kw)
    from pgm_jax.md.flexible import FlexibleSimulation
    return FlexibleSimulation(sysA, templates, pos, H, constraints="h-bonds", **kw)


def scale_to_volume(sim, V):
    """Positions (nm) of sim's current configuration with molecular centres scaled to volume V (the
    barostat's molecular scaling), and the box."""
    st = sim.state
    s = (V / float(volume(st.box))) ** (1.0 / 3.0)
    if hasattr(sim, "flex"):
        pos = st.dyn.position
        return np.asarray(pos + ((s - 1.0) * sim.flex.centers(pos))[sim.flex.mol]), np.asarray(st.box) * s
    body = st.dyn.position
    body = RigidBody(body.center * s, body.orientation)
    return np.asarray(sim.rigid.positions(body)), np.asarray(st.box) * s


def cmd_run(a):
    sysA, P, pos, vel, H, settings, elec, templates = build(a)
    mode = a.intramolecular or ("keep" if templates is not None else "annihilate")
    if templates is not None and mode != "keep":
        raise ValueError("a flexible pGM solute needs --intramolecular keep: its bonded terms were fitted with its "
                         "intramolecular electrostatics")
    alch = Alchemy(sysA, a.solute, sc_alpha=a.sc_alpha, intramolecular=mode)
    quant = None if not a.grad_quantities else a.grad_quantities.split(",")
    space = fg.ParamSpace(sysA.table, quant)
    scales = parse_scales(a.solute_scale)
    if scales:
        P = fg.scaled_params(space, P, scales, solute=True)
        print(f"# solute parameters scaled: {scales}", flush=True)
    kw = dict(settings=settings, dt=a.dt / 1000.0, temperature=a.temp, thermostat="bussi", tau_t=1.0, params=P,
              alchemy=alch, log=sys.stdout, seed=a.seed)
    lam = standard_schedule(a.n_elec, None if a.vdw is None else [float(x) for x in a.vdw.split(",")])
    if a.elec_only:
        lam = lam[:a.n_elec]
    if a.checkpoint is None and a.start_from is None and a.npt_ps > 0:
        npt = engine(sysA, templates, pos, H, ensemble="npt", pressure=1.0, barostat_interval=100, vel_nm_ps=vel, **kw)
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
    sim = engine(sysA, templates, pos, H, ensemble="nvt", vel_nm_ps=vel, **kw)
    meta = {"model": a.model if not a.prmtop else a.prmtop, "solute": a.solute, "elec": elec,
            "settings": dataclasses.asdict(settings), "dt_fs": a.dt, "temperature": a.temp,
            "volume_nm3": float(volume(sim.state.box)), "sc_alpha": a.sc_alpha, "alpha_floor": alch.alpha_floor,
            "intramolecular": mode, "solute_scale": scales, "elec_only": bool(a.elec_only)}
    if templates is not None:
        meta["solute_template"] = os.path.abspath(a.solute_template)
    if mode == "annihilate":                                  # rigid solute: exact gas-phase leg
        gas = GasPhaseLeg(alch, sim.positions_nm()[sysA.atom_slice(a.solute)], elec)
        meta.update(gas_delta_g=gas.delta_g(P), gas_e1=gas.energy(1.0, P), gas_dudl=[gas.dudl(le, P) for le in lam[:, 0]])
        if a.grad:
            meta["gas_grad"] = fg.gas_leg_gradient(gas, P, space)[1].tolist()
        print(f"# gas-phase leg: E_gas(1) = {meta['gas_e1']:.4f} kJ/mol, Delta G_gas(1 -> 0) = "
              f"{meta['gas_delta_g']:.4f} kJ/mol", flush=True)
    else:                                                     # the gas-phase leg is in the Hamiltonian
        meta.update(gas_delta_g=0.0, gas_dudl=[0.0] * len(lam))
        if a.grad:
            meta["gas_grad"] = [0.0] * space.n
    t0 = time.time()
    win = LambdaWindows(sim, lam, batched=not a.sequential, seed=a.seed + 1)
    to_steps = lambda ps: int(round(ps / (a.dt / 1000.0)))                      # noqa: E731
    pg = fg.ParamGradients(win, quantities=quant) if a.grad else None
    run = FreeEnergyRun(win, sample_every=to_steps(a.sample_ps), exchange_every=to_steps(a.exchange_ps),
                        seed=a.seed, meta=meta, param_grad=pg)
    if a.checkpoint:
        run.load(a.checkpoint)
    elif a.start_from:
        run.load_windows(a.start_from)
        print(f"# windows start from {a.start_from}", flush=True)
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
    print(f"# {r['windows']} windows, {r['samples_per_window']} samples each after {discard_ps} ps; "
          f"kT = {r['kT']:.4f} kJ/mol")
    print("# statistical inefficiency (dE):", " ".join(f"{g:.1f}" for g in r["g_dE"]))
    print("# statistical inefficiency (dU/dl):", " ".join(f"{g:.1f}" for g in r["g_dudl"]))
    print(f"# smallest neighbour overlap (MBAR): {r['overlap_min']:.3f}")
    print("#  window  lambda_e lambda_v   <dU/dl_e>      <dU/dl_v>     (kJ/mol)   MBAR step   BAR step")
    L = d["lambdas"]
    for i in range(r["windows"]):
        st = "" if i == r["windows"] - 1 else \
            f"{r['mbar_steps'][i]:10.3f} {r['bar_steps'][i]:10.3f} +- {r['bar_steps_err'][i]:.3f}"
        print(f"  {i:6d} {L[i, 0]:9.3f} {L[i, 1]:8.3f} {r['dudl_mean'][i][0]:12.3f} {r['dudl_mean'][i][1]:12.3f}   {st}")
    print("# Delta G of switching the solute off in solution (kJ/mol | kcal/mol):")
    for m in ("ti", "bar", "mbar"):
        print(f"  {m:5s} {r[m]:10.3f} +- {r[m + '_err']:.3f} | {k(r[m]):9.3f} +- {k(r[m + '_err']):.3f}")
    if "mbar_elec" in r:
        print(f"  stages (MBAR): electrostatics {r['mbar_elec']:.3f} +- {r['mbar_elec_err']:.3f}, van der Waals "
              f"{r['mbar_vdw']:.3f} +- {r['mbar_vdw_err']:.3f} kJ/mol; TI {r['ti_elec']:.3f}, {r['ti_vdw']:.3f}")
    if gas is not None:
        print(f"# gas-phase leg Delta G_gas(1 -> 0) = {r['gas']:.4f} kJ/mol; hydration free energy "
              f"(kJ/mol | kcal/mol):")
        for m in ("ti", "ti_sub", "bar", "mbar"):
            v, e = r[f"dG_hyd_{m}"], r[f"dG_hyd_{m}_err"]
            print(f"  {m:6s} {v:10.3f} +- {e:.3f} | {k(v):9.3f} +- {k(e):.3f}")
        t = np.asarray(d["time_ps"])
        mid = 0.5 * (discard_ps + t[-1])
        if np.sum((t > discard_ps) & (t <= mid)) >= 20:
            h1 = fe.estimate(d, discard_ps=discard_ps, gas=gas, end_ps=mid)
            h2 = fe.estimate(d, discard_ps=mid, gas=gas)
            print(f"# halves ({discard_ps:g}-{mid:g} ps | {mid:g}-{t[-1]:g} ps), MBAR: "
                  f"{k(h1['dG_hyd_mbar']):.3f} +- {k(h1['dG_hyd_mbar_err']):.3f} | "
                  f"{k(h2['dG_hyd_mbar']):.3f} +- {k(h2['dG_hyd_mbar_err']):.3f} kcal/mol")
    if "dudp" in d:
        grad_report(d, discard_ps)
    teq = fe.equilibration_times(d)
    print(f"# equilibration detected (ps after the windows start): max {teq.max():.0f}, per window "
          + " ".join(f"{x:.0f}" for x in teq))
    return r


def parse_scales(text):
    """'charge=1.05,eps=0.9' -> {'charge': 1.05, 'eps': 0.9}."""
    out = {}
    for item in (text or "").split(","):
        if item.strip():
            k, v = item.split("=")
            out[k.strip()] = float(v)
    return out


def grad_report(d, discard_ps, n_blocks=10):
    """Parameter gradients of the hydration free energy (or of the solution leg without a gas leg):
    MBAR-weighted and end-state estimators, block jackknife errors; derivatives along the scale
    directions of the solute's and of the environment's parameters, and per solute entry."""
    meta = d["meta"]
    gas = {"delta_g": meta["gas_delta_g"], "grad": meta.get("gas_grad")} if "gas_delta_g" in meta else None
    r = fg.gradient_estimate(d, discard_ps=discard_ps, gas=gas, n_blocks=n_blocks)
    leg = "hyd" if "hyd" in r else "solv"
    space = fg.ParamSpace.from_names(r["names"])
    p = np.asarray(meta["params_flat"], float)
    k = lambda x: x / KCAL                                                    # noqa: E731
    m, e = r[leg]["mbar"], r[leg]["end"]
    print(f"# parameter gradients of {'the hydration free energy' if leg == 'hyd' else 'Delta G(first -> last)'}, "
          f"block jackknife over {n_blocks} blocks: value {k(m.value):.3f} +- {k(m.value_err):.3f} kcal/mol")
    print("#  d/d ln s (kcal/mol)           MBAR               end states")
    out = {"value_kcal": k(m.value), "value_err_kcal": k(m.value_err), "scale": {}}
    for who, sol in (("solute", True), ("environment", False)):
        for g in fg.SCALE_GROUPS:
            v = space.scale_direction(p, g, sol)
            if not np.any(v):
                continue
            (a1, b1), (a2, b2) = m.project(v), e.project(v)
            out["scale"][f"{who}:{g}"] = {"mbar": [k(a1), k(b1)], "end": [k(a2), k(b2)]}
            print(f"  {who:12s} {g:7s} {k(a1):9.3f} +- {k(b1):.3f}   {k(a2):9.3f} +- {k(b2):.3f}")
    print("#  solute entries: dG/dp (kJ/mol per unit; MBAR | end states)")
    for i in space.select(solute=True):
        if p[i] != 0.0 or m.grad[i] != 0.0:
            print(f"  {r['names'][i]:28s} p = {p[i]:11.6f}  {m.grad[i]:12.4f} +- {m.grad_err[i]:.4f} | "
                  f"{e.grad[i]:12.4f} +- {e.grad_err[i]:.4f}")
    return r, out


def cmd_bench(a):
    """ms per step of plain MD, of one alchemical window, and per window of K batched windows (the
    first K windows of the schedule, spread over it), and the cost of one sample of all windows."""
    sysA, P, pos, vel, H, settings, elec, templates = build(a)
    kw = dict(settings=settings, dt=a.dt / 1000.0, temperature=a.temp, thermostat="bussi", tau_t=1.0, params=P,
              log=None, seed=a.seed, ensemble="nvt", vel_nm_ps=vel)
    lam = standard_schedule(a.n_elec)
    n = a.steps

    def timed(f, reps=3):
        f()
        best = np.inf
        for _ in range(reps):
            t = time.time()
            f()
            best = min(best, time.time() - t)
        return best

    def md(sim):
        def f():
            sim._advance(n)
            jax.block_until_ready(sim.state.epot)
        return timed(f) / n * 1e3

    plain = engine(sysA, templates, pos, H, **kw)                           # the same system, no alchemical region
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
            win.advance(n)
            jax.block_until_ready(win.S.epot)
        t = timed(f)
        ts = timed(lambda: win.sample(), reps=2)
        out[f"batched_{len(idx)}_ms_per_window_step"] = t / n / len(idx) * 1e3
        out[f"batched_{len(idx)}_sample_ms"] = ts * 1e3
        if a.grad:                                    # parameter gradients of the end states at every window
            pg = fg.ParamGradients(win)
            out[f"batched_{len(idx)}_grad_sample_ms"] = timed(lambda: pg.sample(), reps=2) * 1e3
        del win
    for k, v in out.items():
        print(f"{k:40s} {v:.4f}" if isinstance(v, float) else f"{k:40s} {v}")
    per_day = lambda ms: a.dt * 1e-6 / (ms * 1e-3) * 86400.0                  # noqa: E731
    print(f"# ns/day: plain MD {per_day(out['plain_ms_per_step']):.1f}, one alchemical window "
          f"{per_day(out['alchemy_1_window_ms_per_step']):.1f}" + "".join(
              f"; {k.split('_')[1]} batched windows {per_day(v * int(k.split('_')[1])):.1f} per window "
              f"({per_day(v):.1f} aggregate)" for k, v in out.items() if k.endswith("window_step")))


def cmd_finite_size(a):
    """The solute alone in the production box (its PME settings) at random positions and
    orientations: [E_pbc(1) - E_pbc(0)] - [E_gas(1) - E_gas(0)] of switching its electrostatics off,
    i.e. the periodic self-image energy plus the PME error of its intramolecular terms, which the
    solution leg carries and the gas-phase leg does not (a rigid solute; its template geometry)."""
    sysA, P, pos, vel, H, settings, elec, templates = build(a)
    from pgm_jax.md.box import reduce_box
    from pgm_jax.md.forcefield import PGMForceField
    sub, idx = sysA.sub((a.solute,))
    x0 = pos[idx] - pos[idx].mean(axis=0)
    H = reduce_box(H)
    alch = Alchemy(sub, 0)
    gas = GasPhaseLeg(alch, x0, elec)
    dg = gas.energy(1.0, P) - gas.energy(0.0, P)
    ff = PGMForceField(sub, H, dataclasses.replace(settings, dipole_tol=min(settings.dipole_tol, 1e-7), max_iter=200))
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
    print(f"# E_gas(1) - E_gas(0) = {dg:.4f} kJ/mol; periodic minus gas-phase: {d.mean():+.5f} kJ/mol "
          f"(spread {d.std():.5f} over {len(d)} placements; {d.mean() / KCAL:+.5f} kcal/mol)")


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
    r.add_argument("--lattice", type=int, default=0, help="n: a small box of n^3 copies of the model's first "
                   "molecule on a lattice (cheap checks; with --npt-ps, --cut, --nfft for the small box)")
    r.add_argument("--solute-template", help="flexible solute (FlexibleTemplate file) inserted into the water box")
    r.add_argument("--clear", type=float, default=0.25, help="nm: waters this close to the inserted solute are removed")
    r.add_argument("--rigid-solute", action="store_true",
                   help="hold the --solute-template molecule rigid at its reference geometry (rigid engine)")
    r.add_argument("--intramolecular", choices=["annihilate", "keep"],
                   help="solute's intramolecular electrostatics (default: annihilate for rigid, keep for flexible)")
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
    r.add_argument("--start-from", help="start the windows from the configurations of another run's prefix.fe.chk "
                   "(no NPT, no samples taken over; windows matched by lambda)")
    r.add_argument("--grad", action="store_true", help="sample parameter gradients of the end states (fe_grad.py)")
    r.add_argument("--grad-quantities", help="comma-separated parameter quantities for --grad (default: all)")
    r.add_argument("--solute-scale", help="scale solute parameters, e.g. charge=1.05,eps=0.9 (charge: q and "
                   "covalent dipoles; eps, rmin, alpha, radius)")
    r.add_argument("--elec-only", action="store_true", help="only the electrostatics windows (lambda_vdw = 1)")
    b = sub.add_parser("bench")
    b.add_argument("--model", default="pgm", choices=["pgm", "pgm3p25", "tip3p"])
    b.add_argument("--windows", default="1,4,8,19", help="numbers of batched windows")
    b.add_argument("--steps", type=int, default=1000)
    b.add_argument("--n-elec", type=int, default=8)
    b.add_argument("--grad", action="store_true", help="also time the parameter-gradient samples")
    for x in r._actions:
        if x.dest in ("prmtop", "coords", "elec", "solute", "dt", "temp", "tol", "cut", "ew_coeff", "nfft", "order",
                      "precision", "seed", "solute_template", "clear", "lattice"):
            b._add_action(x)
    fs = sub.add_parser("finite-size")
    fs.add_argument("--model", default="pgm", choices=["pgm", "pgm3p25", "tip3p"])
    fs.add_argument("--placements", type=int, default=40)
    for x in r._actions:
        if x.dest in ("prmtop", "coords", "elec", "solute", "tol", "cut", "ew_coeff", "nfft", "order", "precision", "seed"):
            fs._add_action(x)
    z = sub.add_parser("analyze")
    z.add_argument("npz")
    z.add_argument("--discard-ps", type=float, default=200.0)
    a = ap.parse_args()
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
