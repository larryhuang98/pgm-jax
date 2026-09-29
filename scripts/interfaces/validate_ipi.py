"""i-PI + pgm_jax (pgm_jax.interfaces.ipi): flexible pGM water (512 molecules; the flexible template
of the PIMD work, validation/interfaces/pgm_water_flex.flex), 298 K, dt 0.25 fs.

    python scripts/interfaces/validate_ipi.py start   # native minimisation + classical NVT -> runs/ipi_val/start.npz
                                                      # (shared by every run; --work / --out: run directories, results)
    python scripts/interfaces/validate_ipi.py nve     # i-PI NVE vs native NVE from the same state (trajectories, drift)
    python scripts/interfaces/validate_ipi.py nvt     # classical i-PI (SVR thermostat) vs native Bussi: T, <U>
    python scripts/interfaces/validate_ipi.py pimd --beads 8   # i-PI PIMD (PILE-G) vs native PIMD: KE_H, KE_O (centroid virial)

Results are appended to validation/interfaces/ipi.json.  i-PI is found through IPI_ROOT (see
ipi_tools.py).  Native PIMD numbers for the comparison come from the PIMD branch
(validation/pimd/water/w<P>.json there), run with the same template and settings."""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("IPI_ROOT", os.path.join(ROOT, "runs", "pylib"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

import ipi_tools as T  # noqa: E402
from pgm_jax.interfaces import PGMEngine  # noqa: E402
from pgm_jax.interfaces.ipi import BOHR_NM, IPIClient  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.system import System  # noqa: E402

TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
RST = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")
TPL = os.path.join(ROOT, "validation/interfaces/pgm_water_flex.flex")
WORK = os.path.join(ROOT, "runs/ipi_val")
OUT = os.path.join(ROOT, "validation/interfaces/ipi.json")
ATU_PS = 2.4188843265864e-5          # atomic unit of time, ps


def ensure_quartic_bond():
    """The flexible water template uses the bond_quartic family of the PIMD work; register it when
    this branch does not have it (identical definition)."""
    from pgm_jax.bonded import terms as Tm
    if "bond_quartic" in Tm.REGISTRY:
        return
    import jax.numpy as jnp
    from pgm_jax.bonded.terms.core import Family, register

    @register
    class BondQuartic(Family):
        name = "bond_quartic"
        params = {"K2": ((), 2.5e5), "K3": ((), 0.0), "K4": ((), 0.0)}
        linear = ("K2", "K3", "K4")

        def index(self, top, keyf):
            return {"i": np.arange(len(top.bonds))}, [keyf(b, "bond") for b in top.bonds]

        def energy(self, G, dev, I, p):
            db = dev["db"][I["i"]]
            return jnp.sum(0.5 * p["K2"] * db ** 2 + p["K3"] * db ** 3 + p["K4"] * db ** 4)


def setup(args):
    ensure_quartic_bond()
    from pgm_jax.md.flexible import FlexibleTemplate
    tpl = FlexibleTemplate.load(args.template)
    s = MDSettings(cutoff=0.9, dipole_tol=args.tol, precision=args.precision)
    return tpl, s


def native_sim(tpl, s, pos, H, ensemble="nvt", vel=None, seed=0, dt=0.00025, T=298.0):
    from pgm_jax.md.flexible import FlexibleSimulation
    n = len(pos) // 3
    return FlexibleSimulation(System([tpl.pgm] * n), [tpl] * n, pos, H, s, dt=dt, ensemble=ensemble, temperature=T,
                              thermostat="bussi", tau_t=0.1, seed=seed, vel_nm_ps=vel, log=None)


def load_start():
    d = np.load(os.path.join(ROOT, "runs/ipi_val", "start.npz"))
    return d["pos"], d["vel"], d["H"]


def save(key, value):
    res = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            res = json.load(fh)
    res[key] = value
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({key: value}, indent=1), flush=True)


def cmd_start(args):
    from pgm_jax.md.io import box_from_cell, read_coordinates
    tpl, s = setup(args)
    xyz, _, box = read_coordinates(RST)
    H = box_from_cell(*box) * 0.1
    sim = native_sim(tpl, s, xyz * 0.1, H)
    sim.minimize(200)
    sim._advance(int(round(args.ps / 0.00025)))
    os.makedirs(os.path.join(ROOT, "runs/ipi_val"), exist_ok=True)
    np.savez(os.path.join(ROOT, "runs/ipi_val", "start.npz"), pos=sim.positions_nm(), vel=sim.velocities_nm_ps(), H=np.asarray(sim.state.box))
    print(sim.observables())


def client_factory(sysm, tpls, s, slots, address, stress="atomic", vmap=True, virial=True):
    return lambda: IPIClient(lambda p, c: PGMEngine(sysm, p, c, s, templates=tpls, slots=slots, stress=stress),
                             address, unix=True, log=None, vmap_beads=vmap, virial=virial)


def cmd_nve(args):
    """Same start (positions, velocities, masses), velocity Verlet in both codes."""
    tpl, s = setup(args)
    pos, vel, H = load_start()
    n = len(pos) // 3
    sysm, tpls = System([tpl.pgm] * n), [tpl] * n
    steps, rep = args.steps, 20
    nat = native_sim(tpl, s, pos, H, "nve", vel)
    t, E, U = [0.0], [nat.observables()["etot"]], [nat.observables()["epot"]]
    t0 = time.perf_counter()
    for k in range(steps // rep):
        nat._advance(rep)
        o = nat.observables()
        t.append(o["time_ps"]); E.append(o["etot"]); U.append(o["epot"])
    ms_nat = 1e3 * (time.perf_counter() - t0) / steps
    wd = os.path.join(WORK, "nve")
    symbols = [e for m in sysm.molecules for e in m.elements]
    T.write_input(wd, symbols, pos, H, sysm.masses, nbeads=1, steps=steps, dt_fs=0.25, ensemble="nve", stride=rep,
                  address="pgmval_nve", velocities=vel / (BOHR_NM / ATU_PS), velocity_units="atomic_unit",
                  pressure_output=not args.no_virial)
    client, st, props, wall = T.run(wd, "pgmval_nve", client_factory(sysm, tpls, s, 1, "pgmval_nve",
                                                                     virial=not args.no_virial))
    Ui, Ei = props["potential"], props["conserved"]
    m = min(len(Ui), len(U))
    dof = 3 * sysm.n - 3
    slope = lambda tt, ee: float(np.polyfit(np.asarray(tt) / 1000.0, ee, 1)[0] / (KB * 298.0) / dof)
    ti = props["time"]
    save("nve" + ("_novirial" if args.no_virial else ""), {"steps": steps, "virial": not args.no_virial, "dt_fs": 0.25, "precision": args.precision, "dipole_tol": args.tol,
                 "U_diff_first": [float(Ui[k] - U[k]) for k in range(0, min(m, 6))],
                 "U_maxdiff_first_100_steps": float(np.max(np.abs(Ui[:6] - np.asarray(U[:6])))),
                 "U_rel_maxdiff_all": float(np.max(np.abs(Ui[:m] - np.asarray(U[:m]))) / abs(U[0])),
                 "E0_native": float(E[0]), "E0_ipi": float(Ei[0]),
                 "drift_native_kT_ns_dof": slope(t, E), "drift_ipi_kT_ns_dof": slope(ti, Ei),
                 "std_native_kJmol": float(np.std(E)), "std_ipi_kJmol": float(np.std(Ei)),
                 "ms_per_step_native": ms_nat, "ms_per_step_ipi": 1e3 * st["t_total"] / steps,
                 "engine_ms_per_call": 1e3 * client.engine.stats["time"] / client.engine.stats["calls"],
                 "client_ms_per_call": 1e3 * st["t_engine"] / max(st["structures"], 1)})


def cmd_nvt(args):
    tpl, s = setup(args)
    pos, vel, H = load_start()
    n = len(pos) // 3
    sysm, tpls = System([tpl.pgm] * n), [tpl] * n
    steps, rep = args.steps, 40
    nat = native_sim(tpl, s, pos, H, "nvt", vel, seed=3)
    Tn, Un = [], []
    t0 = time.perf_counter()
    for k in range(steps // rep):
        nat._advance(rep)
        o = nat.observables()
        Tn.append(o["temp_K"]); Un.append(o["epot"] / n)
    ms_nat = 1e3 * (time.perf_counter() - t0) / steps
    wd = os.path.join(WORK, "nvt")
    symbols = [e for m in sysm.molecules for e in m.elements]
    T.write_input(wd, symbols, pos, H, sysm.masses, nbeads=1, steps=steps, dt_fs=0.25, ensemble="nvt",
                  thermostat="svr", tau_fs=100.0, stride=rep, address="pgmval_nvt",
                  velocities=vel / (BOHR_NM / ATU_PS), velocity_units="atomic_unit", seed=7)
    client, st, props, wall = T.run(wd, "pgmval_nvt", client_factory(sysm, tpls, s, 1, "pgmval_nvt"))
    sk = len(Tn) // 10
    Ti, Ui = props["temperature"], props["potential"] / n
    ski = len(Ti) // 10
    be = lambda x: float(np.std([np.mean(y) for y in np.array_split(np.asarray(x), 5)], ddof=1) / np.sqrt(5))
    # i-PI's temperature counts 3N degrees of freedom (no centre-of-mass correction): rescale to 3N - 3
    corr = 3 * sysm.n / (3 * sysm.n - 3)
    save("nvt", {"steps": steps, "T_native": float(np.mean(Tn[sk:])), "T_native_err": be(Tn[sk:]),
                 "T_ipi": float(np.mean(Ti[ski:])), "T_ipi_err": be(Ti[ski:]), "T_ipi_3N-3": float(np.mean(Ti[ski:]) * corr),
                 "U_native_per_mol": float(np.mean(Un[sk:])), "U_native_err": be(Un[sk:]),
                 "U_ipi_per_mol": float(np.mean(Ui[ski:])), "U_ipi_err": be(Ui[ski:]),
                 "ms_per_step_native": ms_nat, "ms_per_step_ipi": 1e3 * st["t_total"] / steps,
                 "engine_ms_per_call": 1e3 * client.engine.stats["time"] / client.engine.stats["calls"]})


def cmd_pimd(args):
    tpl, s = setup(args)
    pos, vel, H = load_start()
    n = len(pos) // 3
    sysm, tpls = System([tpl.pgm] * n), [tpl] * n
    P, steps, rep = args.beads, args.steps, 20
    wd = os.path.join(WORK, f"pimd{P}{'b' if args.batch else 's'}_{steps}_{args.splitting}_{args.propagator}")
    symbols = [e for m in sysm.molecules for e in m.elements]
    T.write_input(wd, symbols, pos, H, sysm.masses, nbeads=P, steps=steps, dt_fs=0.25, ensemble="nvt",
                  thermostat="pile_g", tau_fs=100.0, stride=rep, address=f"pgmval_p{P}",
                  batch_size=P if args.batch else 1, seed=11, splitting=args.splitting, nm_propagator=args.propagator,
                  extra_props=("kinetic_cv(H)", "kinetic_cv(O)", "kinetic_td(H)"), pressure_output=not args.no_virial)
    client, st, props, wall = T.run(wd, f"pgmval_p{P}", client_factory(sysm, tpls, s, P, f"pgmval_p{P}",
                                                                        vmap=bool(args.vmap), virial=not args.no_virial))
    nH, nO = 2 * n, n
    eq = int(round(args.equil_ps / (0.00025 * rep)))
    keH = props["kinetic_cv(H)"][eq:] / nH * T.KJMOL_MEV
    keO = props["kinetic_cv(O)"][eq:] / nO * T.KJMOL_MEV
    U = props["potential"][eq:]
    be = lambda x: float(np.std([np.mean(y) for y in np.array_split(np.asarray(x), 5)], ddof=1) / np.sqrt(5))
    eng = client.engine
    blocks = np.array_split(np.asarray(U), 5)
    out = {"beads": P, "batch": bool(args.batch), "vmap": bool(args.vmap), "steps": steps, "equil_ps": args.equil_ps,
           "splitting": args.splitting or "obabo", "propagator": args.propagator or "exact", "virial": not args.no_virial,
           "epot_blocks": [float(np.mean(b)) for b in blocks],
           "ke_H_cv_meV": [float(np.mean(keH)), be(keH)], "ke_O_cv_meV": [float(np.mean(keO)), be(keO)],
           "epot_bead_mean": [float(np.mean(U)), be(U)], "T": float(np.mean(props["temperature"][eq:])),
           "ms_per_step_ipi": 1e3 * st["t_total"] / steps, "engine_ms_per_call": 1e3 * eng.stats["time"] / eng.stats["calls"],
           "cg_per_call": eng.stats["cg"] / eng.stats["calls"], "engine_resets": eng.stats["resets"],
           "engine_repeats": eng.stats["repeats"], "engine_stats": dict(eng.stats)}
    ref = os.path.expanduser(f"~/project/pGM-JAX-pimd/validation/pimd/water/w{P}.json")
    if os.path.exists(ref):
        with open(ref) as fh:
            r = json.load(fh)
        out["native"] = {k: r[k] for k in ("ke_H_cv_meV", "ke_O_cv_meV", "epot", "ms_per_step", "ps", "cg_mean") if k in r}
    save(f"pimd{P}{'_batch' if args.batch else '_serial'}{'_vmap' if args.vmap and args.batch else ''}_{steps}"
         f"{'_' + args.splitting if args.splitting else ''}{'_' + args.propagator if args.propagator else ''}"
         f"{'_novirial' if args.no_virial else ''}", out)
    print(json.dumps(client.engine.stats), flush=True)


def main():
    global WORK, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("start", "nve", "nvt", "pimd"))
    ap.add_argument("--template", default=TPL)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--ps", type=float, default=1.0, help="start: classical NVT (ps)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--beads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--vmap", type=int, default=1, help="batched beads in one vmapped engine call")
    ap.add_argument("--no-virial", action="store_true", help="no virial (and no pressure output): timing runs")
    ap.add_argument("--splitting", default=None, help="i-PI splitting: obabo (i-PI default) | baoab (as the native PIMD)")
    ap.add_argument("--propagator", default=None, help="i-PI free ring-polymer propagator: exact | cayley (native default)")
    ap.add_argument("--equil-ps", type=float, default=0.5)
    ap.add_argument("--work", default=WORK, help="i-PI run directories")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    WORK, OUT = os.path.abspath(args.work), os.path.abspath(args.out)
    {"start": cmd_start, "nve": cmd_nve, "nvt": cmd_nvt, "pimd": cmd_pimd}[args.cmd](args)


if __name__ == "__main__":
    main()
