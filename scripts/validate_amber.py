"""Validate the JAX pGM implementation against Amber (sander / pmemd-pgm) on the 512-water box.

The Amber reference runs (inputs and outputs) are kept in validation/amber_ref/, so `compare`
works without re-running Amber.  All numbers go to validation/validate_amber.json.

Steps:
  prep      wrap molecules into the cell -> dense 512-water cluster inpcrd for gas-phase runs
  amber     re-run sander (gas-phase cluster, no cutoff); pmemd-pgm (periodic, tight PME) is
            run by hand from validation/amber_ref/pmemd_pbc/mdin (command printed)
  compare   our energies, forces, induced dipoles, molecular dipoles vs Amber
  pyresp    induced dipoles of a pGM water monomer vs PyRESP (independent Python code)
  amber_virial  sander single points with ntp=1 (vdwmeth=0 and 1): VIRIAL, PRESS, VDWAALS
  virial    our molecular virial (strain derivative) and LJ long-range correction vs those runs

    python scripts/validate_amber.py prep
    python scripts/validate_amber.py amber      # ~2 min
    python scripts/validate_amber.py compare    # ~5-10 min on CPU
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time

import jax
import numpy as np
from scipy.io import netcdf_file

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgm_jax.channels import ElecChannel
from pgm_jax.ewald import PeriodicPGM, box_matrix
from pgm_jax.lj import LJChannel, PeriodicLJ
from pgm_jax.model import Model
from pgm_jax.param import read_prmtop_pgm
from pgm_jax.periodic import strain_derivative
from pgm_jax.system import System
from pgm_jax.units import KE, KE_AMBER_PGM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "validation", "amber_ref")     # Amber runs: inputs + outputs (in git)
OUT = os.path.join(ROOT, "runs", "validate")             # our own outputs (not in git)
RESULT = os.path.join(ROOT, "validation", "validate_amber.json")
TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
RST = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")
SANDER = os.path.expanduser("~/ambers/pgm-larry/build/AmberTools/src/sander/sander")
PMEMD = os.path.expanduser("~/ambers/pgm-vdw/build/src/pmemd-pgm/src/pmemd-pgm")
KCAL = 4.184
SCALE = KE_AMBER_PGM / KE          # Amber pGM uses Tinker's Coulomb constant


def read_restart(path):
    f = netcdf_file(path, "r", mmap=False)
    xyz = np.array(f.variables["coordinates"][:], float)
    cell = np.array(f.variables["cell_lengths"][:], float), np.array(f.variables["cell_angles"][:], float)
    f.close()
    return xyz, cell


def read_nc_frames(path, var):
    f = netcdf_file(path, "r", mmap=False)
    v = np.array(f.variables[var][:], float)
    f.close()
    return v


def mdout_step0(path):
    txt = open(path).read()
    blk = txt[txt.index("NSTEP =        0"):]
    get = lambda k: float(re.search(rf"{k}\s*=\s*(-?\d+\.\d+)", blk).group(1))
    out = {"EELEC": get("EELEC"), "VDWAALS": get("VDWAALS"), "BOND": get("BOND"), "ANGLE": get("ANGLE")}
    for k in ("VIRIAL", "PRESS", "VOLUME"):
        if re.search(rf"{k}\s*=", blk):
            out[k] = get(k)
    return out


def write_inpcrd(path, xyz, title="wrapped"):
    with open(path, "w") as fh:
        fh.write(title + "\n%6d\n" % len(xyz))
        flat = xyz.ravel()
        for s in range(0, len(flat), 6):
            fh.write("".join("%12.7f" % v for v in flat[s:s + 6]) + "\n")


def wrap_molecules(xyz, H):
    """Shift each water (O,H,H) by lattice vectors so that its O lies in the unit cell."""
    Hinv = np.linalg.inv(H)
    out = xyz.copy()
    for m in range(len(xyz) // 3):
        sl = slice(3 * m, 3 * m + 3)
        shift = -np.floor(xyz[3 * m] @ Hinv) @ H
        out[sl] = xyz[sl] + shift
    return out


# ------------------------------------------------------------------------ steps --

def prep():
    xyz, (L, ang) = read_restart(RST)
    H = box_matrix(*L, *ang)
    w = wrap_molecules(xyz, H)
    os.makedirs(os.path.join(REF, "sander_gas512w"), exist_ok=True)
    write_inpcrd(os.path.join(REF, "sander_gas512w", "inpcrd"), w)
    ext = w.max(0) - w.min(0)
    print(f"box {L} {ang}; wrapped cluster extent {np.round(ext, 1)} A")


def amber():
    gas = os.path.join(REF, "sander_gas512w")
    open(os.path.join(gas, "mdin"), "w").write("""gas-phase 512-water cluster (wrapped), pGM3P-25, no cutoff, tight induction
 &cntrl
   imin=0, nstlim=1, irest=0, ntx=1, tempi=0.0,
   ntb=0, ntt=0, ntc=1, ntf=1,
   cut=999., ntpr=1, ntwx=1, ntwf=1, ioutfm=1, ntwr=1000,
   dt=0.00001, ipgm=1,
 /
 &ewald
  eedmeth=4,
 /
 &pol_gauss
   pol_gauss_verbose=0, dipole_scf_tol=1e-9, ee_dsum_cut=999.,
   scf_cg_niter=5000, scf_solv_opt=1, dipole_scf_init=1
 /
""")
    t0 = time.time()
    subprocess.run([SANDER, "-O", "-i", "mdin", "-c", "inpcrd", "-p", TOP, "-o", "mdout", "-x", "mdcrd", "-frc", "mdfrc"],
                   cwd=gas, check=True)
    print(f"sander gas-phase done in {time.time() - t0:.0f}s: {mdout_step0(os.path.join(gas, 'mdout'))}")
    print("pmemd-pgm periodic run (by hand):\n"
          f"  cd {os.path.join(REF, 'pmemd_pbc')} && {PMEMD} -O -i mdin -c {RST} -p {TOP} -o mdout -x mdcrd -frc mdfrc")


def compare():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    xyz, (L, ang) = read_restart(RST)
    H = box_matrix(*L, *ang) * 0.1
    w = read_prmtop_pgm(TOP)[0]
    sys = System([w] * 512)
    q = np.asarray(sys.expand()["q"])
    to_kcal_A = lambda F: np.asarray(F) / 41.84

    # ================= gas phase: dense wrapped cluster vs sander (no cutoff) ==========
    gas = os.path.join(REF, "sander_gas512w")
    lines = open(os.path.join(gas, "inpcrd")).read().split("\n")
    nat = int(lines[1])
    vals = [float(l[k:k + 12]) for l in lines[2:] for k in range(0, len(l), 12) if l[k:k + 12].strip()]
    crd = np.array(vals[:3 * nat]).reshape(nat, 3)
    amb = mdout_step0(os.path.join(gas, "mdout"))
    f_amb = read_nc_frames(os.path.join(gas, "mdfrc"), "forces")[0]
    elec, vdw = Model([ElecChannel()]), Model([LJChannel()])
    t0 = time.time()
    e = elec.energy_fn(sys)(crd * 0.1)
    F = to_kcal_A(elec.forces_fn(sys)(crd * 0.1, None))
    e_lj = float(vdw.energy_fn(sys)(crd * 0.1)["total"]) / KCAL
    f_lj = to_kcal_A(vdw.forces_fn(sys)(crd * 0.1, None))
    ours = float(e["total"]) / KCAL
    df = (F + f_lj) - f_amb
    df_s = (F * SCALE + f_lj) - f_amb
    res["gas"] = {"EELEC_amber": amb["EELEC"], "EELEC_ours": ours, "dE": ours - amb["EELEC"], "rel": (ours - amb["EELEC"]) / abs(amb["EELEC"]),
                  "dE_sameconst": ours * SCALE - amb["EELEC"], "force_rmsd_sameconst": float(np.sqrt(np.mean(df_s ** 2))),
                  "force_maxdev_sameconst": float(np.abs(df_s).max()),
                  "VDW_amber": amb["VDWAALS"], "VDW_ours": e_lj,
                  "force_rms_amber": float(np.sqrt(np.mean(f_amb ** 2))), "force_rmsd": float(np.sqrt(np.mean(df ** 2))),
                  "force_maxdev": float(np.abs(df).max()), "seconds": time.time() - t0}
    print("GAS", json.dumps(res["gas"], indent=1))

    # ================= periodic: vs pmemd-pgm (PME, tight) ==============================
    pb = os.path.join(REF, "pmemd_pbc")
    amb = mdout_step0(os.path.join(pb, "mdout"))
    f_amb = read_nc_frames(os.path.join(pb, "mdfrc"), "forces")[0]
    last = sorted(glob.glob(os.path.join(pb, "cpu_cg_iter_*.dat")))[-1]
    mu_amb = np.loadtxt(last, comments="#")[:, 1:4]
    mol_amb = []
    for line in open(os.path.join(pb, "fort.100")):
        if "Step #" in line and mol_amb:
            break
        if "total moment" in line:
            mol_amb.append([float(x) for x in line.split()[-4:-1]])
    mol_amb = np.array(mol_amb)
    pos = xyz * 0.1
    res["pbc"] = {"EELEC_amber": amb["EELEC"], "VDW_amber": amb["VDWAALS"]}
    ljp = PeriodicLJ(sys, H, pos, rc=1.0)                                 # Amber cut = 10 A
    e_lj = float(ljp.energy(pos)[0]["vdw"]) / KCAL
    f_lj = to_kcal_A(-jax.grad(lambda x: ljp.energy(x)[0]["vdw"])(pos))
    for tag, b0, rc in (("b0=3.8,rc=1.0", 3.8, 1.0), ("b0=3.5,rc=1.1", 3.5, 1.1)):
        t0 = time.time()
        per = PeriodicPGM(sys, H, pos, b0=b0, rc=rc)
        e, aux = per.energy(pos)
        F = to_kcal_A(per.forces(pos))
        mu = np.asarray(per.induced_dipoles(pos)) * 10                   # e A
        p = np.asarray(aux["p"]) * 10
        # molecular total moment with whole molecules: sum_i (q_i r_i + p_i + mu_i), r in A
        mol = np.array([np.sum(q[3 * m:3 * m + 3, None] * xyz[3 * m:3 * m + 3], 0) + (p + mu)[3 * m:3 * m + 3].sum(0) for m in range(512)])
        ours = float(e["total"]) / KCAL
        df = (F + f_lj) - f_amb
        df_s = (F * SCALE + f_lj) - f_amb
        dmol = mol - mol_amb
        match = np.linalg.norm(dmol, axis=1) < 1e-5
        res["pbc"][tag] = {"n_pairs": int(len(per.pi)), "n_k": int(len(per.m)), "EELEC_ours": ours, "dE": ours - amb["EELEC"],
                           "rel": (ours - amb["EELEC"]) / abs(amb["EELEC"]), "VDW_ours": e_lj,
                           "dE_sameconst": ours * SCALE - amb["EELEC"], "force_rmsd_sameconst": float(np.sqrt(np.mean(df_s ** 2))),
                           "force_maxdev_sameconst": float(np.abs(df_s).max()),
                           "moldip_n_match_1e-5": int(match.sum()), "moldip_maxdev_matched": float(np.abs(dmol[match]).max()) if match.any() else None,
                           "force_rms_amber": float(np.sqrt(np.mean(f_amb ** 2))), "force_rmsd": float(np.sqrt(np.mean(df ** 2))),
                           "force_maxdev": float(np.abs(df).max()),
                           "mu_rms_amber": float(np.sqrt(np.mean(mu_amb ** 2))), "mu_rmsd": float(np.sqrt(np.mean((mu - mu_amb) ** 2))),
                           "mu_maxdev": float(np.abs(mu - mu_amb).max()),
                           "moldip_mean_amber": float(np.mean(np.linalg.norm(mol_amb, axis=1))),
                           "moldip_mean_ours": float(np.mean(np.linalg.norm(mol, axis=1))),
                           "moldip_maxdev": float(np.abs(dmol).max()), "seconds": time.time() - t0}
        print("PBC", tag, json.dumps(res["pbc"][tag], indent=1))
        np.save(os.path.join(OUT, f"ours_mu_{tag}.npy"), mu)

    d = json.load(open(RESULT)) if os.path.exists(RESULT) else {}
    d.update(res)
    json.dump(d, open(RESULT, "w"), indent=1)




def pyresp():
    """Independent implementation check: PyRESP (AmberTools, Python) water example, resp-perm with pGM
    polarizabilities (ipol=5, igdm=1, 1-2/1-3 included).  Rebuild the molecule from its output and
    compare our induced dipoles with its 'IND DIP GLOBAL' block (atomic units)."""
    from pgm_jax.system import Molecule
    B = 0.052917721067
    ex = os.path.expanduser("~/amber25/AmberTools/examples/PyRESP")
    txt = open(os.path.join(ex, "test/water/resp-perm/wat.chg")).read()

    def block(flag, ncol):
        seg = txt.split(f"%FLAG {flag}")[1].split("%FLAG")[0].strip().split("\n")[2:]
        return np.array([[float(x) for x in l.split()[-ncol:]] for l in seg if l.strip()])

    crd = block("ATOM CRD", 3)
    q = block("ATOM CHRG", 1)[:, 0]
    loc = [l.split() for l in txt.split("%FLAG PERM DIP LOCAL")[1].split("%FLAG")[0].strip().split("\n")[2:] if l.strip()]
    cov = [(int(r[1]) - 1, int(r[2]) - 1, float(r[4]) * B) for r in loc]
    ind_ref = block("IND DIP GLOBAL", 3)
    perm_ref = block("PERM DIP GLOBAL", 3)
    tab = {}
    for l in open(os.path.join(ex, "polarizability/pGM-pol-2016-09-01")):
        t = l.split()
        if len(t) >= 3 and not l.startswith("!") and t[0] != "EQ":
            try:
                tab[t[0]] = (float(t[1]), float(t[2]))
            except ValueError:
                pass
    alpha = np.array([tab["ow"][0], tab["hw"][0], tab["hw"][0]]) * B ** 3
    rad = np.array([tab["ow"][1], tab["hw"][1], tab["hw"][1]]) * B
    m = Molecule("WAT", ["O", "H", "H"], ["ow", "hw", "hw"], q, rad, alpha, cov)
    s = System([m])
    _, aux = ElecChannel().energy(crd * B, s)
    mu = np.asarray(aux["mu"]) / B
    p = np.asarray(aux["p"]) / B
    out = {"perm_dip_maxdev_au": float(np.abs(p - perm_ref).max()), "ind_dip_maxdev_au": float(np.abs(mu - ind_ref).max()),
           "ind_dip_ref_au": ind_ref.tolist(), "ind_dip_ours_au": mu.tolist()}
    print(json.dumps(out, indent=1))
    d = json.load(open(RESULT)) if os.path.exists(RESULT) else {}
    d["pyresp_monomer"] = out
    json.dump(d, open(RESULT, "w"), indent=1)


NPT_MDIN = """single point pGM3P-25 512 water, virial (ntp=1), vdwmeth={vdw}
 &cntrl
  imin=0, nstlim=1, irest=0, ntx=1, tempi=0.0,
  ntb=2, ntp=1, taup=1.0, pres0=1.0, ntt=0, ntc=1, ntf=1,
  cut=10.0, ntpr=1, ntwr=1000, dt=0.00001, ipgm=1,
 /
 &ewald
  skinnb=1.0, ew_coeff=0.38, nfft1=72, nfft2=72, nfft3=72, order=8, vdwmeth={vdw},
 /
 &pol_gauss
  pol_gauss_verbose=0, ee_dsum_cut=10.0, dipole_scf_tol=1e-9, dipole_scf_init=1,
  scf_solv_opt=1, scf_cg_niter=500,
 /
"""


def amber_virial():
    for vdw in (0, 1):
        wd = os.path.join(REF, f"sander_npt_vdw{vdw}")
        os.makedirs(wd, exist_ok=True)
        open(os.path.join(wd, "mdin"), "w").write(NPT_MDIN.format(vdw=vdw))
        t0 = time.time()
        subprocess.run([SANDER, "-O", "-i", "mdin", "-c", RST, "-p", TOP, "-o", "mdout", "-r", "restrt", "-inf", "mdinfo"],
                       cwd=wd, check=True)
        print(f"vdwmeth={vdw}: {time.time() - t0:.0f}s {mdout_step0(os.path.join(wd, 'mdout'))}")


def virial():
    """Molecular virial tr(dE/d eps)/2 (Amber's VIRIAL) from JAX strain derivatives, electrostatics
    rescaled to Amber's Coulomb constant; LJ long-range correction energy and virial."""
    xyz, (L, ang) = read_restart(RST)
    H = box_matrix(*L, *ang) * 0.1
    pos = xyz * 0.1
    sys = System([read_prmtop_pgm(TOP)[0]] * 512)
    per = PeriodicPGM(sys, H, pos, b0=3.8, rc=1.0)
    out = {}
    t0 = time.time()
    W_el = np.asarray(strain_derivative(lambda x, h: per.energy(x, None, h)[0]["total"], pos, H, sys)) / KCAL
    out["seconds_elec"] = time.time() - t0
    for vdw in (0, 1):
        amb = mdout_step0(os.path.join(REF, f"sander_npt_vdw{vdw}", "mdout"))
        ljp = PeriodicLJ(sys, H, pos, rc=1.0, lrc=bool(vdw))
        W_lj = (np.asarray(strain_derivative(lambda x, h: ljp.energy(x, None, h)[0]["vdw"], pos, H, sys))
                + np.asarray(ljp.tail_virial())) / KCAL
        e_lj = float(ljp.energy(pos)[0]["vdw"]) / KCAL
        vir = 0.5 * (np.trace(W_el) * SCALE + np.trace(W_lj))
        out[f"vdwmeth={vdw}"] = {"VIRIAL_amber": amb.get("VIRIAL"), "VIRIAL_ours": float(vir),
                                 "dVIRIAL": float(vir - amb["VIRIAL"]) if "VIRIAL" in amb else None,
                                 "VIRIAL_elec_ours": float(0.5 * np.trace(W_el) * SCALE), "VIRIAL_vdw_ours": float(0.5 * np.trace(W_lj)),
                                 "VDW_amber": amb["VDWAALS"], "VDW_ours": e_lj, "EELEC_amber": amb["EELEC"],
                                 "PRESS_amber": amb.get("PRESS"), "VOLUME_amber": amb.get("VOLUME"),
                                 "VOLUME_ours": float(abs(np.linalg.det(H)) * 1000)}
    out["dE_deps_elec_kcal"] = (W_el * SCALE).tolist()
    print(json.dumps(out, indent=1))
    d = json.load(open(RESULT)) if os.path.exists(RESULT) else {}
    d["virial"] = out
    json.dump(d, open(RESULT, "w"), indent=1)


if __name__ == "__main__":
    {"prep": prep, "amber": amber, "compare": compare, "pyresp": pyresp,
     "amber_virial": amber_virial, "virial": virial}[sys.argv[1]]()
