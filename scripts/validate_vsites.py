"""Virtual sites on TIP4P-Ew (Horn et al., J. Chem. Phys. 120, 9665 (2004)): Amber extra points
against sander, energy conservation and the liquid at 298 K / 1 atm in both MD engines, and the
GPU cost of sites.  Point charges (elec "q", Gaussian radii 1e-4 nm), Amber's EP frame.

    python scripts/validate_vsites.py build       # 512 waters (8^3 lattice, 24.87 A cube) through tleap
    python scripts/validate_vsites.py equil       # NPT 298 K, rigid engine, 200 ps -> runs/vsites/equil.rst7
    python scripts/validate_vsites.py sander      # sander single point (PME 64^3, order 8) at that frame
    python scripts/validate_vsites.py compare     # energies, forces and EP positions vs sander
    python scripts/validate_vsites.py nve         # NVE drift, rigid and constrained engines, mixed precision
    python scripts/validate_vsites.py npt --engine rigid --ns 4      # production (log: runs/vsites/npt_rigid.log)
    python scripts/validate_vsites.py analyse     # density and <U> per molecule, block errors, vs Horn et al.
    python scripts/validate_vsites.py bench       # ms/step: TIP4P-Ew vs the same water without its site
    python scripts/validate_vsites.py identical --base DIR   # no sites: bitwise the same as the code in DIR

Energies: the engine's potential energy includes the intramolecular Coulomb energy of every
molecule (pGM has no exclusions; a constant for rigid water), which Amber excludes: it is
subtracted (and its forces, which a rigid molecule does not feel) for comparisons.  sander's
Coulomb constant is 18.2223^2 kcal A/mol e^-2, the engine's CODATA's (3.5e-5 larger): the engine's
electrostatics are rescaled for the comparison.  Results go to runs/vsites/ and
validation/validate_vsites.json."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pgm_jax.md.forcefield import MDSettings, PGMForceField  # noqa: E402
from pgm_jax.md.integrate import KB  # noqa: E402
from pgm_jax.md.io import box_from_cell, read_coordinates  # noqa: E402
from pgm_jax.md.simulation import Simulation, _dedupe  # noqa: E402
from pgm_jax.md.vsites import VirtualSites  # noqa: E402
from pgm_jax.param import read_prmtop_pgm  # noqa: E402
from pgm_jax.system import System  # noqa: E402
from pgm_jax.units import KE  # noqa: E402

OUT = os.path.join(ROOT, "runs", "vsites")
JSON = os.path.join(ROOT, "validation", "validate_vsites.json")
AMBERHOME = os.path.expanduser("~/amber25")
SANDER = os.path.join(AMBERHOME, "bin", "sander")
TOP = os.path.join(OUT, "tip4pew512.prmtop")
CRD = os.path.join(OUT, "tip4pew512.inpcrd")
KCAL = 4.184
KE_AMBER = 18.2223 ** 2 * KCAL / 10.0          # kJ/mol nm e^-2: sander's Coulomb constant
R_OH, R_HH, D_OM = 0.09572, 0.15136, 0.0125    # nm: Amber's SHAKE lengths and the EP bond
NW = 512
HORN = {"T_K": 297.7, "density": 0.9954, "density_err": 0.0003, "U_kcal_512": -5687.4, "U_err_kcal_512": 1.9,
        "dHvap_kcal": 10.583, "source": "Horn et al., J. Chem. Phys. 120, 9665 (2004), Table V, T_bath = 298.0 K"}


def settings(**kw):
    """Production settings: 0.9 nm cutoff with the LJ tail, PME 0.08 nm order 6, Ewald 4.0 nm^-1."""
    base = dict(elec="q", cutoff=0.9, skin=0.1, lj_lrc=True, pme_order=6, precision="mixed")
    base.update(kw)
    return MDSettings(**base)


def update_json(key, value):
    d = json.load(open(JSON)) if os.path.exists(JSON) else {}
    d[key] = value
    os.makedirs(os.path.dirname(JSON), exist_ok=True)
    json.dump(d, open(JSON, "w"), indent=1)


def ideal_water():
    """O, H1, H2 at the model geometry (nm), O at the origin, bisector along +y."""
    t = np.arcsin(R_HH / 2 / R_OH)
    return np.array([[0, 0, 0], [R_OH * np.sin(t), R_OH * np.cos(t), 0], [-R_OH * np.sin(t), R_OH * np.cos(t), 0]])


def load(prmtop=TOP, coords=CRD):
    mols = _dedupe(read_prmtop_pgm(prmtop, first_residue_only=False, charges="amber"))
    xyz, vel, box = read_coordinates(coords)
    sys_ = System(mols)
    return sys_, xyz * 0.1, None if vel is None else vel * 0.1, box_from_cell(*box) * 0.1


def intramolecular(sys_, pos):
    """Energy (kJ/mol) and forces of the intramolecular Coulomb pairs (erf(a r) / r kernel of the
    model, a = 1/sqrt(2 (R_i^2 + R_j^2))) that the engine includes and Amber excludes."""
    from scipy.special import erf
    P = {k: np.asarray(v) for k, v in sys_.expand().items()}
    q, R = P["q"], P["radius"]
    E, F = 0.0, np.zeros_like(pos)
    for k in range(sys_.nmol):
        idx = np.arange(sys_.offsets[k], sys_.offsets[k + 1])
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if q[i] == 0 or q[j] == 0:
                    continue
                x = pos[i] - pos[j]
                r = np.linalg.norm(x)
                al = 1.0 / np.sqrt(2 * (R[i] ** 2 + R[j] ** 2))
                E += KE * q[i] * q[j] * erf(al * r) / r
                dB = KE * q[i] * q[j] * (erf(al * r) / r ** 2 - 2 * al / np.sqrt(np.pi) * np.exp(-(al * r) ** 2) / r)  # -dE/dr
                F[i] += dB * x / r
                F[j] -= dB * x / r
    return E, F


def write_inpcrd(path, xyz_A, box_A, title="pgm_jax"):
    with open(path, "w") as fh:
        fh.write(title + "\n%6d\n" % len(xyz_A))
        flat = np.asarray(xyz_A).ravel()
        for s in range(0, len(flat), 6):
            fh.write("".join("%12.7f" % v for v in flat[s:s + 6]) + "\n")
        fh.write("".join("%12.7f" % v for v in list(box_A) + [90.0, 90.0, 90.0]) + "\n")


def tleap(script, cwd):
    subprocess.run(["bash", "-c", f"source {AMBERHOME}/amber.sh && tleap -f {script} > {script}.log 2>&1"],
                   cwd=cwd, check=True)


# ----------------------------------------------------------------------------- steps
def build(a):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    L = (NW * 18.01528 * 1.66053906660 / 0.995) ** (1 / 3)              # Angstrom
    w = ideal_water() * 10
    lines = []
    k = 0
    for i in range(8):
        for j in range(8):
            for l in range(8):
                R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                c = (np.array([i, j, l]) + 0.5) * L / 8
                y = (w - w.mean(0)) @ R.T + c
                ep = y[0] + D_OM * 10 * (y[1] + y[2] - 2 * y[0]) / np.linalg.norm(y[1] + y[2] - 2 * y[0])
                k += 1
                for name, x in zip(("O", "H1", "H2", "EPW"), (y[0], y[1], y[2], ep)):
                    lines.append("ATOM  %5d %-4s WAT %5d    %8.3f%8.3f%8.3f  1.00  0.00" % (len(lines) + 1, name, k, *x))
                lines.append("TER")
    open(os.path.join(OUT, "w512.pdb"), "w").write("\n".join(lines) + "\nEND\n")
    open(os.path.join(OUT, "leap512.in"), "w").write(f"""source leaprc.water.tip4pew
x = loadpdb w512.pdb
set x box {{ {L:.6f} {L:.6f} {L:.6f} }}
saveamberparm x tip4pew512.prmtop tip4pew512_leap.inpcrd
quit
""")
    tleap("leap512.in", OUT)
    sys_, pos, _, H = load(TOP, os.path.join(OUT, "tip4pew512_leap.inpcrd"))
    X = pos.reshape(NW, 4, 3)
    ideal = ideal_water()
    for m in range(NW):                                  # PDB precision -> exact model geometry (Kabsch)
        y = X[m, :3] - X[m, :3].mean(0)
        c = ideal - ideal.mean(0)
        U, _, Vt = np.linalg.svd(c.T @ y)
        Rm = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
        X[m, :3] = c @ Rm.T + X[m, :3].mean(0)
    pos = np.asarray(VirtualSites.of(sys_).place(X.reshape(-1, 3), H))
    write_inpcrd(CRD, pos * 10, np.diag(H) * 10, "TIP4P-Ew 512 waters, model geometry")
    print(f"{NW} waters, box {L:.4f} A, {sys_.n} atoms -> {TOP}, {CRD}")


def equil(a):
    sys_, pos, _, H = load()
    sim = Simulation(sys_, pos, H, settings(), dt=0.002, ensemble="npt", temperature=298.0, thermostat="bussi",
                     tau_t=0.5, seed=3, log=open(os.path.join(OUT, "equil.out"), "w"))
    sim.run(int(a.ps / 0.002), report=500, restart=int(a.ps / 0.002), prefix=os.path.join(OUT, "equil"))
    o = sim.observables()
    print(f"equilibrated {a.ps} ps: density {o['density_g_cm3']:.4f} g/cm^3, T {o['temp_K']:.1f} K")


def sander(a):
    sys_, pos, _, H = load(TOP, os.path.join(OUT, "equil.rst7"))
    wd = os.path.join(OUT, "sander")
    os.makedirs(wd, exist_ok=True)
    write_inpcrd(os.path.join(wd, "inpcrd"), pos * 10, np.diag(H) * 10, "equilibrated TIP4P-Ew")
    open(os.path.join(wd, "mdin"), "w").write(f"""TIP4P-Ew single point: PME 64^3 order 8, ew_coeff 0.4, 9 A, LJ tail
 &cntrl
   imin=0, nstlim=1, irest=0, ntx=1, tempi=0.0, ntb=1, ntt=0, ntc=1, ntf=1, cut=9.0,
   ntpr=1, ntwx=1, ntwf=1, ioutfm=1, ntwr=1000, dt=0.00001, ig=1,
 /
 &ewald
   nfft1={a.grid}, nfft2={a.grid}, nfft3={a.grid}, order=8, ew_coeff=0.4, vdwmeth=1,
 /
""")
    t0 = time.time()
    subprocess.run(["bash", "-c", f"source {AMBERHOME}/amber.sh && {SANDER} -O -i mdin -c inpcrd -p {TOP} -o mdout "
                    "-x mdcrd -frc mdfrc"], cwd=wd, check=True)
    print(f"sander done in {time.time() - t0:.0f} s")


def _step0(path):
    txt = open(path).read()
    blk = txt[txt.index("NSTEP =        0"):]
    get = lambda k: float(re.search(rf"{k}\s*=\s*(-?\d+\.\d+)", blk).group(1))           # noqa: E731
    return {k: get(k) for k in ("EELEC", "VDWAALS", "BOND", "EPtot")}


def compare(a):
    from scipy.io import netcdf_file
    wd = os.path.join(OUT, "sander")
    sys_, pos, _, H = load(TOP, os.path.join(wd, "inpcrd"))
    vs = VirtualSites.of(sys_)
    amb = _step0(os.path.join(wd, "mdout"))
    f = netcdf_file(os.path.join(wd, "mdfrc"), "r", mmap=False)
    F_amb = np.array(f.variables["forces"][0], float)                    # kcal/mol/A
    f.close()
    f = netcdf_file(os.path.join(wd, "mdcrd"), "r", mmap=False)
    X_amb = np.array(f.variables["coordinates"][0], float)              # after one step of 1e-5 ps
    f.close()
    s = settings(precision="double", pme_grid=(a.grid,) * 3, pme_order=8, ewald_beta=4.0, dipole_tol=1e-12)
    ff = PGMForceField(sys_, H, s)
    ffe = PGMForceField(sys_, H, settings(precision="double", pme_grid=(a.grid,) * 3, pme_order=8, ewald_beta=4.0,
                                          vdw="none"))
    idx = ff.rows_for(pos, H)
    res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    rese = jax.jit(ffe.compute)(pos, H, idx, ffe.init_induction())
    E_in, F_in = intramolecular(sys_, pos)
    sc = KE_AMBER / KE
    eelec = (float(res.energy["elec"]) - E_in) * sc / KCAL
    evdw = float(res.energy["vdw"]) / KCAL
    Fe, Ft = np.asarray(rese.forces), np.asarray(res.forces)
    F = np.asarray(vs.spread(pos, H, sc * (Fe - F_in) + (Ft - Fe))) / KCAL / 10.0
    real = vs.real
    dF = F[real] - F_amb[real]
    ep = vs.is_site
    out = {"EELEC_sander": amb["EELEC"], "EELEC_pgm_jax": eelec, "EELEC_diff": eelec - amb["EELEC"],
           "VDWAALS_sander": amb["VDWAALS"], "VDWAALS_pgm_jax": evdw, "VDWAALS_diff": evdw - amb["VDWAALS"],
           "BOND_sander": amb["BOND"], "intramolecular_coulomb_kcal": E_in / KCAL,
           "force_rms_diff_kcal_A": float(np.sqrt(np.mean(dF ** 2))), "force_max_diff_kcal_A": float(np.abs(dF).max()),
           "force_rms_kcal_A": float(np.sqrt(np.mean(F_amb[real] ** 2))),
           "sander_ep_force_max": float(np.abs(F_amb[ep]).max()),
           "ep_position_max_diff_A": float(np.abs(X_amb[ep] - pos[ep] * 10).max()),
           "real_position_max_diff_A": float(np.abs(X_amb[real] - pos[real] * 10).max()),
           "settings": f"PME {a.grid}^3 order 8, ew_coeff 0.4 A^-1, cut 9 A, vdwmeth 1, float64"}
    print(json.dumps(out, indent=1))
    update_json("sander_single_point", out)


def _drift(times, E, dof, T):
    """Linear drift of E (kJ/mol) in kT per ns per degree of freedom."""
    slope = np.polyfit(np.asarray(times) / 1000.0, np.asarray(E), 1)[0]      # kJ/mol/ns
    return slope / (dof * KB * T)


def nve(a):
    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    sys_, pos, vel, H = load(TOP, os.path.join(OUT, "equil.rst7"))
    out = {}
    for engine in ("rigid", "constraints"):
        for dt in (0.001, 0.002):
            s = settings()
            if engine == "rigid":
                sim = Simulation(sys_, pos, H, s, dt=dt, ensemble="nve", vel_nm_ps=vel, log=None)
            else:
                tpl = RigidTemplate(sys_.molecules[0], pos[:4])
                sim = FlexibleSimulation(sys_, [tpl] * sys_.nmol, pos, H, s, dt=dt, ensemble="nve", vel_nm_ps=vel,
                                         log=None)
            nrep = int(round(1.0 / dt)) // 10                          # 0.1 ps
            t, E, T = [], [], []
            t0 = time.time()
            for k in range(int(a.ps * 10)):
                sim._advance(nrep)
                o = sim.observables()
                t.append(o["time_ps"]); E.append(o["etot"]); T.append(o["temp_K"])
            el = time.time() - t0
            d = _drift(t, E, sim.integ.dof, float(np.mean(T)))
            key = f"{engine}_dt{dt * 1000:g}fs"
            out[key] = {"drift_kT_per_ns_per_dof": float(d), "etot_std_kJ": float(np.std(E)), "T_mean": float(np.mean(T)),
                        "ps": a.ps, "wall_s": el}
            print(key, out[key], flush=True)
    update_json("nve_mixed", out)


def npt(a):
    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    sys_, pos, vel, H = load(TOP, os.path.join(OUT, "equil.rst7"))
    prefix = os.path.join(OUT, f"npt_{a.engine}")
    kw = dict(dt=0.002, ensemble="npt", temperature=298.0, pressure=1.01325, thermostat="bussi", tau_t=1.0,
              vel_nm_ps=vel, seed=11, log=open(prefix + ".out", "w"))
    if a.engine == "rigid":
        sim = Simulation(sys_, pos, H, settings(), **kw)
    else:
        tpl = RigidTemplate(sys_.molecules[0], pos[:4])
        sim = FlexibleSimulation(sys_, [tpl] * sys_.nmol, pos, H, settings(), **kw)
    n = int(round(a.ns * 1000 / 0.002))
    sim.run(n, report=500, restart=50000, prefix=prefix)


def _log(path):
    rows = [l.split() for l in open(path) if not l.startswith("#") and l.strip()]
    cols = next(l[1:].split() for l in open(path) if l.startswith("#") and "step" in l)
    return {c: np.array([float(r[k]) for r in rows]) for k, c in enumerate(cols)}


def _block(x, nb=10):
    x = np.asarray(x)
    b = np.array_split(x, nb)
    m = np.array([bb.mean() for bb in b])
    return float(x.mean()), float(m.std(ddof=1) / np.sqrt(nb))


def analyse(a):
    sys_, pos, _, H = load()
    E_in, _ = intramolecular(sys_, pos)                                  # rigid: the same for every frame
    out = {"horn2004": HORN, "intramolecular_coulomb_kJ_per_molecule": E_in / NW}
    for engine in ("rigid", "constraints"):
        path = os.path.join(OUT, f"npt_{engine}.log")
        if not os.path.exists(path):
            continue
        d = _log(path)
        keep = d["time_ps"] > a.skip
        rho, rho_e = _block(d["density_g_cm3"][keep])
        U = (d["epot"][keep] - E_in) / NW
        u, u_e = _block(U)
        T, _ = _block(d["temp_K"][keep])
        out[engine] = {"ns": float(d["time_ps"][keep][-1] - d["time_ps"][keep][0]) / 1000, "density": rho, "density_err": rho_e,
                       "U_kJ_per_molecule": u, "U_err": u_e, "U_kcal_per_molecule": u / KCAL, "U_err_kcal": u_e / KCAL,
                       "T_mean": T, "ns_per_day_last": float(d["ns_per_day"][-1]),
                       "dHvap_uncorrected_kcal": -u / KCAL + KB * 298.0 / KCAL}
        print(engine, json.dumps(out[engine], indent=1))
    h = HORN["U_kcal_512"] / NW
    print(f"Horn et al.: density {HORN['density']} +- {HORN['density_err']}, U {h:.4f} kcal/mol ({h * KCAL:.3f} kJ/mol)")
    update_json("npt_298K", out)


def _timed(sim, steps):
    sim._advance(200)                                                     # compile + warm up
    jax.block_until_ready(sim.state.epot)
    t0 = time.time()
    sim._advance(steps)
    jax.block_until_ready(sim.state.epot)
    return (time.time() - t0) / steps * 1000.0


def bench(a):
    import dataclasses
    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    sys4, pos4, vel, H = load(TOP, os.path.join(OUT, "equil.rst7"))
    n = a.replicate
    if n > 1:                                                            # n^3 copies (cubic box)
        shifts = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)]) @ H
        pos4 = np.concatenate([pos4 + s for s in shifts])
        H = H * n
        sys4 = System(sys4.molecules * n ** 3)
    m = sys4.molecules[0]
    m3 = dataclasses.replace(m, elements=m.elements[:3], types=m.types[:3], q=np.array([m.q[3], m.q[1], m.q[2]]),
                             radius=m.radius[:3], alpha=m.alpha[:3], lj_rmin_half=m.lj_rmin_half[:3],
                             lj_sqrt_eps=m.lj_sqrt_eps[:3], gvdw_sqrt_a=m.gvdw_sqrt_a[:3], gvdw_sqrt_c6=m.gvdw_sqrt_c6[:3],
                             gvdw_b=m.gvdw_b[:3], masses=m.masses[:3], bonds=[b for b in m.bonds if 3 not in b],
                             vsites=[], keys={})
    sys3 = System([m3] * sys4.nmol)
    pos3 = pos4.reshape(-1, 4, 3)[:, :3].reshape(-1, 3)
    out = {}
    for label, S, P in (("tip4pew (4 sites, 1 virtual)", sys4, pos4), ("3-site control (M charge on O)", sys3, pos3)):
        for engine in ("rigid", "constraints"):
            kw = dict(dt=0.002, ensemble="nvt", temperature=298.0, thermostat="bussi", tau_t=1.0, log=None)
            if engine == "rigid":
                sim = Simulation(S, P, H, settings(), **kw)
            else:
                tpl = RigidTemplate(S.molecules[0], P[:S.molecules[0].n])
                sim = FlexibleSimulation(S, [tpl] * S.nmol, P, H, settings(), **kw)
            ms = _timed(sim, a.steps)
            key = f"{label} | {engine}"
            out[key] = {"ms_per_step": ms, "ns_per_day": 0.002 * 86400.0 / ms, "atoms": S.n}
            print(key, out[key], flush=True)
    update_json(f"bench_{sys4.nmol}_waters", out)


def identical(a):
    """No virtual sites: run the same short MD with this code and with the code in a.base (e.g. the
    commit before the feature, `git archive`), each in its own process; positions must be bitwise equal."""
    script = r'''
import sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, sys.argv[1])
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import box_from_cell, read_coordinates
from pgm_jax.md.simulation import Simulation, _dedupe
from pgm_jax.param import read_prmtop_pgm
from pgm_jax.system import System
import os
top = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop"); rst = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")
xyz, vel, box = read_coordinates(rst); H = box_from_cell(*box) * 0.1
out = {}
for prec in ("mixed", "double"):
    sim = Simulation(System(_dedupe(read_prmtop_pgm(top, first_residue_only=False))), xyz * 0.1, H, MDSettings(precision=prec),
                     dt=0.001, ensemble="npt", seed=5, log=None)
    sim.run(400, report=0, prefix=sys.argv[2] + prec)
    out[prec + "_pos"] = sim.positions_nm(); out[prec + "_mu"] = np.asarray(sim.state.induction.mu)
    out[prec + "_epot"] = np.array(sim.state.epot)
from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
mols = _dedupe(read_prmtop_pgm(top, first_residue_only=False)); sysf = System(mols)
tpl = RigidTemplate(mols[0], xyz[:3] * 0.1)
fs = FlexibleSimulation(sysf, [tpl] * sysf.nmol, xyz * 0.1, H, MDSettings(), dt=0.002, ensemble="nvt", constraints="none",
                        thermostat="langevin", seed=5, log=None)
fs._advance(200); out["flex_pos"] = fs.positions_nm()
np.savez(sys.argv[2] + ".npz", **out)
'''
    path = os.path.join(OUT, "identical.py")
    open(path, "w").write(script)
    res = {}
    for tag, root in (("base", os.path.abspath(a.base)), ("branch", ROOT)):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, path, root, os.path.join(OUT, "ident_" + tag)], check=True, env=env)
    A = np.load(os.path.join(OUT, "ident_base.npz"))
    B = np.load(os.path.join(OUT, "ident_branch.npz"))
    for k in A.files:
        res[k] = bool(np.array_equal(A[k], B[k]))
        print(k, "bitwise equal" if res[k] else f"DIFFERENT (max {np.abs(A[k] - B[k]).max():.3g})")
    update_json("no_sites_bitwise", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "equil", "sander", "compare", "nve", "npt", "analyse", "bench", "identical"])
    ap.add_argument("--ps", type=float, default=200.0, help="equil: ps; nve: ps per run")
    ap.add_argument("--ns", type=float, default=4.0)
    ap.add_argument("--engine", default="rigid", choices=["rigid", "constraints"])
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--skip", type=float, default=0.0, help="analyse: ps discarded at the start")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    globals()[a.step](a)


if __name__ == "__main__":
    main()
