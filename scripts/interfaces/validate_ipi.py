"""Validate i-PI + pgm_jax (pgm_jax.interfaces.ipi) on flexible pGM water (docs/interfaces.md).

512 molecules of the flexible template of the PIMD work (data/validation/interfaces/
pgm_water_flex.flex), 298 K, dt 0.25 fs.  Commands: start (native minimisation + classical NVT,
the state shared by every run), nve (i-PI NVE vs native NVE from the same state: trajectories,
drift), nvt (classical i-PI with the SVR thermostat vs native Bussi: T, <U>), pimd (i-PI PIMD with
PILE-G vs native PIMD: KE_H, KE_O by the centroid virial).  The native PIMD numbers for the
comparison come from data/validation/pimd/water/w<P>.json (scripts/pimd/pimd_water.py, same
template and settings).

Usage:

    python scripts/interfaces/validate_ipi.py start   # -> runs/ipi_val/start.npz
    python scripts/interfaces/validate_ipi.py nve
    python scripts/interfaces/validate_ipi.py nvt
    python scripts/interfaces/validate_ipi.py pimd --beads 8
    python scripts/interfaces/validate_ipi.py --help

Inputs: the template (--template), PGM_GVDW_DATA (start); i-PI, found through IPI_ROOT (default
runs/pylib; see pgm_jax/interfaces/ipi_tools.py).
Outputs: results added to --out (default data/validation/interfaces/ipi.json); i-PI run
directories under --work (default runs/ipi_val); printed results.
Units: --time-ps and --equil-ps ps; energies kJ/mol, kinetic energies per atom meV.
Runtime: GPU (the engine) with i-PI on the CPU; minutes to hours (pimd).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable

import jax
import numpy as np

from pgm_jax.cli.args import add_dipole_tol_arg, add_precision_arg, setup_logging
from pgm_jax.interfaces import PGMEngine
from pgm_jax.interfaces import ipi_tools as T
from pgm_jax.interfaces.ipi import IPIClient
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.thermostats import Bussi
from pgm_jax.paths import repo_path, resource
from pgm_jax.system import System
from pgm_jax.units import BOHR_NM_CODATA2022, KB, KJMOL_TO_MEV

os.environ.setdefault("IPI_ROOT", repo_path("runs", "pylib"))
jax.config.update("jax_enable_x64", True)
TOP = resource("gvdw_data", "topology/rayl_512_v2.prmtop")
RST = resource("gvdw_data", "inputs/lj/inpcrd.restrt")
TPL = repo_path("data", "validation", "interfaces", "pgm_water_flex.flex")
WORK = repo_path("runs", "ipi_val")
OUT = repo_path("data", "validation", "interfaces", "ipi.json")
ATU_PS = 2.4188843265864e-5  # atomic unit of time, ps


def ensure_quartic_bond() -> None:
    """Register the bond_quartic family of the PIMD work when the bonded registry lacks it.

    The flexible water template uses it; the definition is identical to the PIMD work's.
    """
    from pgm_jax.bonded import terms as Tm

    if "bond_quartic" in Tm.REGISTRY:
        return
    import jax.numpy as jnp

    from pgm_jax.bonded.terms.core import Family, register

    @register
    class BondQuartic(Family):
        """Quartic bond: K2/2 db^2 + K3 db^3 + K4 db^4 (db: bond deviation [nm], K in kJ/mol/nm^k)."""

        name = "bond_quartic"
        params = {"K2": ((), 2.5e5), "K3": ((), 0.0), "K4": ((), 0.0)}
        linear = ("K2", "K3", "K4")

        def index(self, top, keyf):
            """Return the bond indices and parameter keys."""
            return {"i": np.arange(len(top.bonds))}, [keyf(b, "bond") for b in top.bonds]

        def energy(self, G, dev, I, p):
            """Return the energy [kJ/mol] of the bonds."""
            db = dev["db"][I["i"]]
            return jnp.sum(0.5 * p["K2"] * db**2 + p["K3"] * db**3 + p["K4"] * db**4)


def setup(args: argparse.Namespace) -> tuple[FlexibleTemplate, MDSettings]:
    """Return the flexible water template and the force-field settings (0.9 nm, --dipole-tol, --precision)."""
    ensure_quartic_bond()
    tpl = FlexibleTemplate.load(args.template)
    s = MDSettings().replace(cutoff=0.9, dipole_tol=args.dipole_tol, precision=args.precision)
    return tpl, s


def native_sim(
    tpl: FlexibleTemplate,
    s: MDSettings,
    pos: np.ndarray,
    H: np.ndarray,
    nvt: bool = True,
    vel: np.ndarray | None = None,
    seed: int = 0,
    dt: float = 0.00025,
    T: float = 298.0,
) -> FlexibleSimulation:
    """Return the same water in pgm_jax's own engine: Bussi 0.1 ps (nvt) or NVE; dt [ps], T [K]."""
    n = len(pos) // 3
    return FlexibleSimulation(
        System([tpl.pgm] * n),
        [tpl] * n,
        pos,
        H,
        s,
        dt=dt,
        temperature=T,
        thermostat=Bussi(0.1) if nvt else None,
        seed=seed,
        velocities=vel,
    )


def load_start() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positions [nm], velocities [nm/ps] and box [nm] of the shared start state (<work>/start.npz)."""
    d = np.load(os.path.join(WORK, "start.npz"))
    return d["pos"], d["vel"], d["H"]


def save(key: str, value: object) -> None:
    """Store value under key in the results JSON (--out) and print it."""
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            res = json.load(fh)
    res[key] = value
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({key: value}, indent=1), flush=True)


def cmd_start(args: argparse.Namespace) -> None:
    """Minimise and equilibrate (classical NVT, --time-ps) the flexible water; save <work>/start.npz."""
    tpl, s = setup(args)
    xyz, _, box = read_coordinates(RST)
    H = box_from_cell(*box) * 0.1
    sim = native_sim(tpl, s, xyz * 0.1, H)
    sim.minimize(200)
    sim.advance(int(round(args.time_ps / 0.00025)))
    os.makedirs(WORK, exist_ok=True)
    np.savez(
        os.path.join(WORK, "start.npz"),
        pos=sim.positions(),
        vel=sim.velocities(),
        H=np.asarray(sim.state.box),
    )
    print(sim.observables())


def client_factory(
    sysm: System,
    tpls: list,
    s: MDSettings,
    slots: int,
    address: str,
    stress: str = "atomic",
    vmap: bool = True,
    virial: bool = True,
) -> Callable[[], IPIClient]:
    """Return a factory of the i-PI client (unix socket `address`) with a PGMEngine of `slots` structures."""
    return lambda: IPIClient(
        lambda p, c: PGMEngine(sysm, p, c, s, templates=tpls, slots=slots, stress=stress),
        address,
        unix=True,
        log=None,
        vmap_beads=vmap,
        virial=virial,
    )


def cmd_nve(args: argparse.Namespace) -> None:
    """Compare i-PI NVE with native NVE: same start (positions, velocities, masses), velocity Verlet in both."""
    tpl, s = setup(args)
    pos, vel, H = load_start()
    n = len(pos) // 3
    sysm, tpls = System([tpl.pgm] * n), [tpl] * n
    steps, rep = args.steps, 20
    nat = native_sim(tpl, s, pos, H, False, vel)
    t, E, U = [0.0], [nat.observables()["etot"]], [nat.observables()["epot"]]
    t0 = time.perf_counter()
    for _k in range(steps // rep):
        nat.advance(rep)
        o = nat.observables()
        t.append(o["time_ps"])
        E.append(o["etot"])
        U.append(o["epot"])
    ms_nat = 1e3 * (time.perf_counter() - t0) / steps
    wd = os.path.join(WORK, "nve")
    symbols = [e for m in sysm.molecules for e in m.elements]
    T.write_input(
        wd,
        symbols,
        pos,
        H,
        sysm.masses,
        nbeads=1,
        steps=steps,
        dt_fs=0.25,
        thermostat=None,
        stride=rep,
        address="pgmval_nve",
        velocities=vel / (BOHR_NM_CODATA2022 / ATU_PS),
        velocity_units="atomic_unit",
        pressure_output=not args.no_virial,
    )
    client, st, props, wall = T.run(
        wd, "pgmval_nve", client_factory(sysm, tpls, s, 1, "pgmval_nve", virial=not args.no_virial)
    )
    Ui, Ei = props["potential"], props["conserved"]
    m = min(len(Ui), len(U))
    dof = 3 * sysm.n - 3

    def slope(tt, ee):
        return float(np.polyfit(np.asarray(tt) / 1000.0, ee, 1)[0] / (KB * 298.0) / dof)

    ti = props["time"]
    save(
        "nve" + ("_novirial" if args.no_virial else ""),
        {
            "steps": steps,
            "virial": not args.no_virial,
            "dt_fs": 0.25,
            "precision": args.precision,
            "dipole_tol": args.dipole_tol,
            "U_diff_first": [float(Ui[k] - U[k]) for k in range(0, min(m, 6))],
            "U_maxdiff_first_100_steps": float(np.max(np.abs(Ui[:6] - np.asarray(U[:6])))),
            "U_rel_maxdiff_all": float(np.max(np.abs(Ui[:m] - np.asarray(U[:m]))) / abs(U[0])),
            "E0_native": float(E[0]),
            "E0_ipi": float(Ei[0]),
            "drift_native_kT_ns_dof": slope(t, E),
            "drift_ipi_kT_ns_dof": slope(ti, Ei),
            "std_native_kJmol": float(np.std(E)),
            "std_ipi_kJmol": float(np.std(Ei)),
            "ms_per_step_native": ms_nat,
            "ms_per_step_ipi": 1e3 * st["t_total"] / steps,
            "engine_ms_per_call": 1e3 * client.engine.stats["time"] / client.engine.stats["calls"],
            "client_ms_per_call": 1e3 * st["t_engine"] / max(st["structures"], 1),
        },
    )


def cmd_nvt(args: argparse.Namespace) -> None:
    """Compare classical i-PI NVT (SVR thermostat) with native Bussi NVT: T and <U>."""
    tpl, s = setup(args)
    pos, vel, H = load_start()
    n = len(pos) // 3
    sysm, tpls = System([tpl.pgm] * n), [tpl] * n
    steps, rep = args.steps, 40
    nat = native_sim(tpl, s, pos, H, True, vel, seed=3)
    Tn, Un = [], []
    t0 = time.perf_counter()
    for _k in range(steps // rep):
        nat.advance(rep)
        o = nat.observables()
        Tn.append(o["temp_K"])
        Un.append(o["epot"] / n)
    ms_nat = 1e3 * (time.perf_counter() - t0) / steps
    wd = os.path.join(WORK, "nvt")
    symbols = [e for m in sysm.molecules for e in m.elements]
    T.write_input(
        wd,
        symbols,
        pos,
        H,
        sysm.masses,
        nbeads=1,
        steps=steps,
        dt_fs=0.25,
        thermostat="svr",
        tau_fs=100.0,
        stride=rep,
        address="pgmval_nvt",
        velocities=vel / (BOHR_NM_CODATA2022 / ATU_PS),
        velocity_units="atomic_unit",
        seed=7,
    )
    client, st, props, wall = T.run(wd, "pgmval_nvt", client_factory(sysm, tpls, s, 1, "pgmval_nvt"))
    sk = len(Tn) // 10
    Ti, Ui = props["temperature"], props["potential"] / n
    ski = len(Ti) // 10

    def be(x):
        """Return the standard error of the mean of x from 5 blocks."""
        return float(np.std([np.mean(y) for y in np.array_split(np.asarray(x), 5)], ddof=1) / np.sqrt(5))

    # i-PI's temperature counts 3N degrees of freedom (no centre-of-mass correction): rescale to 3N - 3
    corr = 3 * sysm.n / (3 * sysm.n - 3)
    save(
        "nvt",
        {
            "steps": steps,
            "T_native": float(np.mean(Tn[sk:])),
            "T_native_err": be(Tn[sk:]),
            "T_ipi": float(np.mean(Ti[ski:])),
            "T_ipi_err": be(Ti[ski:]),
            "T_ipi_3N-3": float(np.mean(Ti[ski:]) * corr),
            "U_native_per_mol": float(np.mean(Un[sk:])),
            "U_native_err": be(Un[sk:]),
            "U_ipi_per_mol": float(np.mean(Ui[ski:])),
            "U_ipi_err": be(Ui[ski:]),
            "ms_per_step_native": ms_nat,
            "ms_per_step_ipi": 1e3 * st["t_total"] / steps,
            "engine_ms_per_call": 1e3 * client.engine.stats["time"] / client.engine.stats["calls"],
        },
    )


def cmd_pimd(args: argparse.Namespace) -> None:
    """Run i-PI PIMD (PILE-G) and compare KE_H, KE_O and <U> with the native PIMD results."""
    tpl, s = setup(args)
    pos, vel, H = load_start()
    n = len(pos) // 3
    sysm, tpls = System([tpl.pgm] * n), [tpl] * n
    P, steps, rep = args.beads, args.steps, 20
    wd = os.path.join(WORK, f"pimd{P}{'b' if args.batch else 's'}_{steps}_{args.splitting}_{args.propagator}")
    symbols = [e for m in sysm.molecules for e in m.elements]
    T.write_input(
        wd,
        symbols,
        pos,
        H,
        sysm.masses,
        nbeads=P,
        steps=steps,
        dt_fs=0.25,
        thermostat="pile_g",
        tau_fs=100.0,
        stride=rep,
        address=f"pgmval_p{P}",
        batch_size=P if args.batch else 1,
        seed=11,
        splitting=args.splitting,
        nm_propagator=args.propagator,
        extra_props=("kinetic_cv(H)", "kinetic_cv(O)", "kinetic_td(H)"),
        pressure_output=not args.no_virial,
    )
    client, st, props, wall = T.run(
        wd,
        f"pgmval_p{P}",
        client_factory(sysm, tpls, s, P, f"pgmval_p{P}", vmap=bool(args.vmap), virial=not args.no_virial),
    )
    nH, nO = 2 * n, n
    eq = int(round(args.equil_ps / (0.00025 * rep)))
    keH = props["kinetic_cv(H)"][eq:] / nH * KJMOL_TO_MEV
    keO = props["kinetic_cv(O)"][eq:] / nO * KJMOL_TO_MEV
    U = props["potential"][eq:]

    def be(x):
        """Return the standard error of the mean of x from 5 blocks."""
        return float(np.std([np.mean(y) for y in np.array_split(np.asarray(x), 5)], ddof=1) / np.sqrt(5))

    eng = client.engine
    blocks = np.array_split(np.asarray(U), 5)
    out = {
        "beads": P,
        "batch": bool(args.batch),
        "vmap": bool(args.vmap),
        "steps": steps,
        "equil_ps": args.equil_ps,
        "splitting": args.splitting or "obabo",
        "propagator": args.propagator or "exact",
        "virial": not args.no_virial,
        "epot_blocks": [float(np.mean(b)) for b in blocks],
        "ke_H_cv_meV": [float(np.mean(keH)), be(keH)],
        "ke_O_cv_meV": [float(np.mean(keO)), be(keO)],
        "epot_bead_mean": [float(np.mean(U)), be(U)],
        "T": float(np.mean(props["temperature"][eq:])),
        "ms_per_step_ipi": 1e3 * st["t_total"] / steps,
        "engine_ms_per_call": 1e3 * eng.stats["time"] / eng.stats["calls"],
        "cg_per_call": eng.stats["cg"] / eng.stats["calls"],
        "engine_resets": eng.stats["resets"],
        "engine_repeats": eng.stats["repeats"],
        "engine_stats": dict(eng.stats),
    }
    ref = repo_path("data", "validation", "pimd", "water", f"w{P}.json")
    if os.path.exists(ref):
        with open(ref) as fh:
            r = json.load(fh)
        out["native"] = {
            k: r[k] for k in ("ke_H_cv_meV", "ke_O_cv_meV", "epot", "ms_per_step", "ps", "cg_mean") if k in r
        }
    save(
        f"pimd{P}{'_batch' if args.batch else '_serial'}{'_vmap' if args.vmap and args.batch else ''}_{steps}"
        f"{'_' + args.splitting if args.splitting else ''}{'_' + args.propagator if args.propagator else ''}"
        f"{'_novirial' if args.no_virial else ''}",
        out,
    )
    print(json.dumps(client.engine.stats), flush=True)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the command (see the module docstring)."""
    global WORK, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("start", "nve", "nvt", "pimd"), help="what to run")
    ap.add_argument("--template", default=TPL, help="flexible water template (.flex)")
    add_dipole_tol_arg(ap)
    add_precision_arg(ap)
    ap.add_argument("--time-ps", type=float, default=1.0, help="start: classical NVT [ps]")
    ap.add_argument("--steps", type=int, default=2000, help="nve, nvt, pimd: MD steps")
    ap.add_argument("--beads", type=int, default=8, help="pimd: beads")
    ap.add_argument("--batch", type=int, default=1, help="pimd: 1: all beads in one request (batch), 0: one by one")
    ap.add_argument("--vmap", type=int, default=1, help="batched beads in one vmapped engine call")
    ap.add_argument("--no-virial", action="store_true", help="no virial (and no pressure output): timing runs")
    ap.add_argument(
        "--splitting", default=None, help="i-PI splitting: obabo (i-PI default) | baoab (as the native PIMD)"
    )
    ap.add_argument(
        "--propagator", default=None, help="i-PI free ring-polymer propagator: exact | cayley (native default)"
    )
    ap.add_argument("--equil-ps", type=float, default=0.5, help="pimd: time discarded from the averages [ps]")
    ap.add_argument("--work", default=WORK, help="i-PI run directories")
    ap.add_argument("-o", "--out", default=OUT, help="results JSON")
    args = ap.parse_args(argv)
    setup_logging()
    WORK, OUT = os.path.abspath(args.work), os.path.abspath(args.out)
    {"start": cmd_start, "nve": cmd_nve, "nvt": cmd_nvt, "pimd": cmd_pimd}[args.cmd](args)


if __name__ == "__main__":
    main()
