"""Check write_pgm_prmtop against pmemd-pgm: the written model must give pmemd-pgm the engine's energies and forces.

The MD engine's model written as a pmemd-pgm prmtop (pgm_jax.protein.write_pgm_prmtop) is run by
pmemd-pgm and compared with the engine (docs/protein_ff.md).

`sp` (single points): for each system the engine's model (load_amber + templates, as
FlexibleSimulation runs it) is written with write_pgm_prmtop, and the energy components and
forces of pmemd-pgm on the written file are compared with the engine at the same coordinates
(PGMForceField with the templates' pair rules, float64, dipole tolerance 1e-9):
  water512   512 pGM3P-25 waters (rayl_512_v2.prmtop, truncated octahedron) and
  water4096  the 4,096 pGM waters of the pmemd.pgm.cuda benchmark (PGM_GPUBENCH/w4096.prmtop,
             an earlier pGM water parameter set): the round trip of pGM prmtops (electrostatics
             read from them, rewritten); pmemd-pgm also runs the original files.  The 512-water
             box is too small for pmemd.pgm.cuda's neighbour list at a 9 A cutoff (CPU only).
  pep      the solvated peptide fixture (ACE-ALA-SER-NME, TIP3P, NaCl; tests/data)
  trpcage  Trp-cage (1L2Y) in a TIP3P box (runs/protein/trpcage.*, scripts/protein/build_amber.py)
Proteins: placeholder electrostatics, ff19SB-form bonded terms with Fourier CMAP (amber_template,
lj14_scale 1/2); water and ions rigid.  pmemd.pgm (CPU, float64), pmemd.pgm.cuda_DPFP and
_SPFP (GPU, --gpu) run with the engine's cutoff, Ewald coefficient, PME grid and order (vdwmeth
0, dipole_scf_tol 1e-9, netfrc 0: no removal of the net PME force, which the CPU code does by
default and the engine does not) and ntc = ntf = 1 (no SHAKE at the single point), so pmemd's BOND
includes the bonds of the rigid molecules: that energy and its forces (constraints in the engine,
zero on the constraint surface) are computed from the prmtop and removed.  The engine's charges
and covalent dipoles are scaled by sqrt(KE_AMBER_PGM / KE), which gives exactly its energies and
forces with pmemd-pgm's Coulomb constant; DIHED gets the constant of the torsion sign convention
(bonded/amber.py); 1-4 NB (lj14_scale x LJ of the pairs 3 bonds apart) is split from the engine's
van der Waals.

`md`: pmemd.pgm.cuda_SPFP NVT MD of a written system (Langevin 1/ps, SHAKE on X-H bonds, rigid
water, dt 2 fs, hydrogen masses 3.024 amu optional; 9 A cutoff, PME 0.8 A order 6, dipole_scf_tol
1e-5: the engine's protein benchmark settings, docs/protein_ff.md), after 500 minimisation steps
and 2 ps at 0.5 fs heating from 0 K from the tleap structure: temperatures, energies, CA RMSD
and ns/day (also for ubq, dhfr, mbp: speed).  `md-engine`: the same model and protocol in the
engine (after its own minimisation), for the CA RMSD and the speed.

Usage:

    JAX_PLATFORMS=cpu python scripts/protein/check_pgm_prmtop.py sp --gpu [--amber-lambda | --tight]
    python scripts/protein/check_pgm_prmtop.py md --system trpcage --time-ps 200 [--seed 11]
    python scripts/protein/check_pgm_prmtop.py md-engine --system trpcage --time-ps 200 --seed 3   # the engine
    python scripts/protein/check_pgm_prmtop.py --help

Inputs: the systems' prmtops and coordinates (PGM_GVDW_DATA, PGM_GPUBENCH, tests/data, runs/protein);
pmemd.pgm and pmemd.pgm.cuda_* in PGM_PMEMD_BIN (pgm_jax.paths).
Outputs: results added to data/validation/check_pgm_prmtop.json (one key per system / run); run
directories under runs/check_pgm_prmtop/; printed differences.
Units: kcal/mol and kcal/mol/A (pmemd's units, also for the engine's values), A; --time-ps ps,
--hmr-amu amu.
Runtime: run on a GPU node for --gpu / md (the engine stays on the CPU unless JAX_PLATFORMS is
set: the GPU is exclusive-process and belongs to pmemd.pgm.cuda; md-engine runs the engine on the
GPU).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import netcdf_file

import pgm_jax.md.pme as pme_module
from pgm_jax.cli.args import setup_logging
from pgm_jax.md.flexible import FlexibleSimulation
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.io import read_trajectory
from pgm_jax.md.thermostats import Langevin
from pgm_jax.md.topology import MDTopology
from pgm_jax.paths import repo_path, resource
from pgm_jax.prmtop import Prmtop
from pgm_jax.protein import amber_template, load_amber, pmemd_mdin, write_pgm_prmtop
from pgm_jax.protein.pmemd import pair_classes, pmemd_grid
from pgm_jax.units import KCAL, KE, KE_AMBER_PGM

jax.config.update("jax_enable_x64", True)

AMBER = resource("pmemd_pgm_bin")
EXE = {"cpu": "pmemd.pgm", "gpu_dpfp": "pmemd.pgm.cuda_DPFP", "gpu_spfp": "pmemd.pgm.cuda_SPFP"}
OUT = repo_path("runs", "check_pgm_prmtop")
RESULT = repo_path("data", "validation", "check_pgm_prmtop.json")
SYSTEMS = {
    "water512": (
        resource("gvdw_data", "topology/rayl_512_v2.prmtop"),
        resource("gvdw_data", "inputs/lj/inpcrd.restrt"),
        "prmtop",
    ),
    "water4096": (
        resource("gpubench", "w4096.prmtop"),
        resource("gpubench", "w4096.rst7"),
        "prmtop",
    ),
    "pep": (repo_path("tests", "data", "pep_wat.prmtop"), repo_path("tests", "data", "pep_wat.inpcrd"), "placeholder"),
    "trpcage": (
        repo_path("runs", "protein", "trpcage.prmtop"),
        repo_path("runs", "protein", "trpcage.inpcrd"),
        "placeholder",
    ),
}  # name -> (prmtop, coordinates, electrostatics of load_amber)
for _p in ("ubq", "dhfr", "mbp"):  # MD speed (scripts/protein/build_amber.py, 10 A TIP3P buffer)
    SYSTEMS[_p] = (
        repo_path("runs", "protein", f"{_p}.prmtop"),
        repo_path("runs", "protein", f"{_p}.inpcrd"),
        "placeholder",
    )
TERMS = ("BOND", "ANGLE", "DIHED", "CMAP", "1-4 NB", "1-4 EEL", "VDWAALS", "EELEC")


def model(name: str) -> tuple[object, list]:
    """Return (AmberSystem, templates) of a system: the engine's model (amber_template for flexible molecules)."""
    prm, crd, elec = SYSTEMS[name]
    asys = load_amber(prm, crd, electrostatics=elec)
    flex = {k: amber_template(m, prm) for k, m in enumerate(asys.molecules) if m.kind not in ("water", "ion")}
    return asys, asys.templates(flex)


def settings_for(
    H: np.ndarray, cutoff: float = 0.9, spacing: float = 0.08, order: int = 6, beta: float = 4.0
) -> MDSettings:
    """Return the single-point settings: cutoff [nm], beta [1/nm], pmemd_grid(H [nm], spacing [nm]), order; float64.

    No LJ tail, dipole tolerance 1e-9, no predictor.
    """
    return MDSettings().replace(
        cutoff=cutoff,
        skin=0.1,
        ewald_beta=beta,
        pme_grid=pmemd_grid(H, spacing),
        pme_order=order,
        lj_lrc=False,
        dipole_tol=1e-9,
        max_iter=500,
        precision="double",
        predictor="none",
    )


def amber_lambda(K: int, order: int, kcut: int = 50) -> np.ndarray:
    """Return pmemd's factor on the PME influence function per dimension (K,) (pme_recip_dat.F90, factor_lambda).

    lambda(m)^2 with lambda = S_order(m) / S_2order(m), S_p(m) = sum_k (x / (x + pi k))^p over
    k = -kcut..kcut, x = pi m / K; lambda(0) = 1.  The engine uses the plain Euler-spline moduli
    (lambda = 1): the two influence functions differ by O(1e-7) of the energy at 0.8 A spacing.
    """
    out = np.ones(K)
    for i in range(K):
        m = i if i < K // 2 else i - K
        if m == 0:
            continue
        x = math.pi * m / K
        k = np.arange(1, kcut + 1) * math.pi

        def g(p):
            """Return S_p(m) (the sum over k of the module docstring)."""
            return 1.0 + np.sum((x / (x + k)) ** p) + np.sum((x / (x - k)) ** p)

        out[i] = (g(order) / g(2 * order)) ** 2
    return out


def use_amber_lambda() -> None:
    """Give the engine's PME pmemd's influence function (diagnosis only; see amber_lambda; patches md.pme)."""
    plain = pme_module.bspline_moduli
    pme_module.bspline_moduli = lambda K, order: plain(K, order) / amber_lambda(K, order)


# ----------------------------------------------------------------------------- pmemd
def sp_mdin(st: MDSettings, H: np.ndarray) -> str:
    """Return a single-point mdin: pmemd_mdin's nonbonded model, no constraints, induction solved tightly.

    dipole_scf_init=1, scf_solv_opt=1 in &pol_gauss, and netfrc=0 in &ewald (the CPU code removes
    the net PME force by default; the engine does not).
    """
    txt = pmemd_mdin(
        st,
        H,
        nstlim=1,
        dt=0.00001,
        ensemble="nve",
        constraints="none",
        ntpr=1,
        ntwf=1,
        ntwr=1000,
        title="single point",
    )
    txt = txt.replace(" &pol_gauss\n", " &pol_gauss\n   dipole_scf_init=1, scf_solv_opt=1,\n")
    return txt.replace(" &ewald\n", " &ewald\n   netfrc=0,\n")  # the CPU code removes the net PME force by default


def run_pmemd(kind: str, wd: str, prmtop: str, crd: str, mdin: str, env: dict | None = None) -> float:
    """Run a pmemd-pgm executable in a directory and return its wall time [s].

    Parameters
    ----------
    kind : {"cpu", "gpu_dpfp", "gpu_spfp"}
        Executable (EXE); the GPU ones run with CUDA_VISIBLE_DEVICES=0.
    wd : str
        Run directory (mdin, mdout, mdfrc, restrt, mdcrd, mdinfo are written there).
    prmtop, crd : str
        Topology and coordinates.
    mdin : str
        Input text.
    env : dict, optional
        Extra environment variables.

    Raises
    ------
    SmallBox
        pmemd.pgm.cuda refused the box ("Small box detected").
    RuntimeError
        Any other failure (the end of the output is in the message).
    """
    os.makedirs(wd, exist_ok=True)
    with open(os.path.join(wd, "mdin"), "w") as fh:
        fh.write(mdin)
    e = dict(os.environ)
    if kind.startswith("gpu"):
        e["CUDA_VISIBLE_DEVICES"] = "0"
    e.update(env or {})
    t0 = time.time()
    r = subprocess.run(
        [
            os.path.join(AMBER, EXE[kind]),
            "-O",
            "-i",
            "mdin",
            "-p",
            prmtop,
            "-c",
            crd,
            "-o",
            "mdout",
            "-frc",
            "mdfrc",
            "-r",
            "restrt",
            "-x",
            "mdcrd",
            "-inf",
            "mdinfo",
        ],
        cwd=wd,
        env=e,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        if "Small box detected" in r.stdout + r.stderr:
            raise SmallBox(f"{EXE[kind]}: box too small for the GPU neighbour list at this cutoff")
        raise RuntimeError(
            f"{EXE[kind]} failed in {wd}:\n{r.stdout[-1000:]}{r.stderr[-2000:]}\n"
            f"{open(os.path.join(wd, 'mdout')).read()[-3000:]}"
        )
    return time.time() - t0


class SmallBox(RuntimeError):
    """pmemd.pgm.cuda's neighbour list does not fit the box at this cutoff."""


def read_energies(mdout: str) -> dict[str, float]:
    """Return the energy terms [kcal/mol] of the first NSTEP block of an mdout.

    That block is of the input coordinates (step 0 on the CPU; the GPU code prints step 1, whose
    energies are those of the input coordinates too).
    """
    txt = open(mdout).read()
    m = re.search(r"NSTEP =\s+\d+\s+TIME.*?\n(.*?)-{20}", txt, re.S)
    return {k.strip(): float(v) for k, v in re.findall(r"([A-Za-z0-9\- ]+?)\s*=\s*(-?\d+\.\d+)", m.group(1))}


def read_forces(path: str) -> np.ndarray:
    """Return the forces (N, 3) [kcal/mol/A] of the first frame of a NetCDF mdfrc file."""
    f = netcdf_file(path, "r", mmap=False)
    F = np.array(f.variables["forces"][0], float)
    f.close()
    return F


def rigid_bond_terms(prmtop: str, asys: object, templates: list, xyz: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the energy [kcal/mol] and forces (N, 3) [kcal/mol/A] of the prmtop bonds inside rigid molecules.

    xyz: coordinates (N, 3) [A] in prmtop order.  E = sum K (r - r0)^2 (Amber's harmonic bond).
    """
    pt = Prmtop.read(prmtop)
    B = np.concatenate([pt.get("BONDS_INC_HYDROGEN"), pt.get("BONDS_WITHOUT_HYDROGEN")]).reshape(-1, 3)
    K, r0 = pt.get("BOND_FORCE_CONSTANT"), pt.get("BOND_EQUIL_VALUE")
    rigid = np.zeros(len(xyz), bool)
    for m, t in zip(asys.molecules, templates):
        if not t.has_bonded:
            rigid[m.atoms] = True
    a, b, t = B[:, 0] // 3, B[:, 1] // 3, B[:, 2] - 1
    keep = rigid[a] & rigid[b]
    a, b, t = a[keep], b[keep], t[keep]
    d = xyz[a] - xyz[b]
    r = np.linalg.norm(d, axis=1)
    E = float(np.sum(K[t] * (r - r0[t]) ** 2))
    g = (2.0 * K[t] * (r - r0[t]) / r)[:, None] * d
    F = np.zeros_like(xyz)
    np.add.at(F, a, -g)
    np.add.at(F, b, g)
    return E, F


# ----------------------------------------------------------------------------- engine
def family_energy(tpl: object, R: jax.Array, fams: tuple[str, ...]) -> float:
    """Return the bonded energy [kcal/mol] of the template's term families `fams` alone at R [nm]."""
    P = jax.tree_util.tree_map(jnp.asarray, tpl.P)
    Q = jax.tree_util.tree_map(jnp.zeros_like, P)
    Q["ref"] = P["ref"]
    for f in fams:
        if f in P:
            Q[f] = P[f]
    return float(tpl.bonded_energy(R, Q)) / KCAL


def engine(asys: object, templates: list, st: MDSettings) -> tuple[dict, np.ndarray]:
    """Return the engine's energy components [kcal/mol] (pmemd's names) and forces (N, 3) [kcal/mol/A], prmtop order.

    With pmemd-pgm's Coulomb constant (charges and covalent dipoles scaled by
    sqrt(KE_AMBER_PGM / KE)); the 1-4 van der Waals is split from VDWAALS, and DIHED gets the
    constant of the torsion sign convention (see the module docstring).
    """
    sys_ = asys.system()
    topo = MDTopology.build(sys_, [t.md_rule("none") for t in templates])
    H = jnp.asarray(asys.box)
    ff = PGMForceField(sys_, H, st, topology=topo)
    pos = jnp.asarray(asys.system_positions())
    s = math.sqrt(KE_AMBER_PGM / KE)
    P = {k: jnp.asarray(v) for k, v in sys_.params0.items()}
    P["q"], P["cov"] = P["q"] * s, P["cov"] * s
    t0 = time.time()
    res = ff.compute(pos, H, ff.rows_for(pos, H), ff.init_induction(), P)
    out = {
        "EELEC": float(res.energy["elec"]) / KCAL,
        "1-4 EEL": 0.0,
        "cg_iterations": int(res.iterations),
        "cg_residual": float(res.residual),
    }
    F = np.asarray(res.forces) / (10.0 * KCAL)
    # 1-4 van der Waals of the flexible molecules
    X = np.asarray(pos)
    Hn = np.asarray(asys.box)
    Pa = {k: np.asarray(v) for k, v in sys_.expand(P).items()}
    e14 = 0.0
    bond = {"BOND": 0.0, "ANGLE": 0.0, "DIHED": 0.0, "CMAP": 0.0}
    for k, (m, tpl) in enumerate(zip(asys.molecules, templates)):
        if not tpl.has_bonded:
            continue
        o = int(sys_.offsets[k])
        rule = tpl.md_rule("none")
        p14 = np.array(pair_classes(m.n, rule)[1], int).reshape(-1, 2) + o
        d = X[p14[:, 0]] - X[p14[:, 1]]
        d = d - np.round(d @ np.linalg.inv(Hn)) @ Hn
        r = np.linalg.norm(d, axis=1)
        rmin = Pa["lj_rmin_half"][p14[:, 0]] + Pa["lj_rmin_half"][p14[:, 1]]
        eps = Pa["lj_sqrt_eps"][p14[:, 0]] * Pa["lj_sqrt_eps"][p14[:, 1]]
        s6 = (rmin / r) ** 6
        e14 += float(rule.lj14_scale * np.sum(eps * (s6 * s6 - 2.0 * s6))) / KCAL
        R = pos[o : o + m.n]
        bond["BOND"] += family_energy(tpl, R, ("bond_harm",))
        bond["ANGLE"] += family_energy(tpl, R, ("angle_harm",))
        bond["CMAP"] += family_energy(tpl, R, ("cmap", "cmap6"))
        Im, Pt = tpl.terms.I[tpl.index], tpl.P
        const = 2 * np.sum(
            np.abs(np.minimum(np.asarray(Pt["torsion_amber"]["K"])[Im["torsion_amber"]["k"]], 0))
        ) + 2 * np.sum(np.abs(np.minimum(np.asarray(Pt["improper_amber"]["K"])[Im["improper_amber"]["k"]], 0)))
        bond["DIHED"] += family_energy(tpl, R, ("torsion_amber", "improper_amber")) + const / KCAL
        g = np.asarray(jax.grad(lambda y: tpl.bonded_energy(y))(R))
        F[o : o + m.n] -= g / (10.0 * KCAL)
        P = jax.tree_util.tree_map(jnp.asarray, tpl.P)
        Q = jax.tree_util.tree_map(jnp.zeros_like, P)
        Q["ref"] = P["ref"]
        Q.update({f: P[f] for f in ("cmap", "cmap6") if f in P})
        gc = np.linalg.norm(np.asarray(jax.grad(lambda y: tpl.bonded_energy(y, Q))(R)), axis=1) / (10.0 * KCAL)
        out["cmap_force_max"] = max(out.get("cmap_force_max", 0.0), float(gc.max()))
    out.update(bond)
    out["1-4 NB"] = e14
    out["VDWAALS"] = float(res.energy["vdw"]) / KCAL - e14
    out["seconds"] = time.time() - t0
    Fp = np.empty_like(F)
    Fp[np.asarray(asys.order)] = F
    return out, Fp


def cmap_atoms(asys: object, templates: list) -> np.ndarray:
    """Return the prmtop indices of the atoms of the backbone maps.

    pmemd interpolates its 24 x 24 grid bicubically, the engine evaluates the Fourier map: their
    forces differ by the interpolation.
    """
    out = set()
    for m, tpl in zip(asys.molecules, templates):
        if tpl.has_bonded:
            for f in ("cmap", "cmap6"):
                if f in tpl.terms.I[tpl.index]:
                    out.update(int(m.atoms[a]) for a in np.asarray(tpl.terms.I[tpl.index][f]["q"]).ravel())
    return np.array(sorted(out), int)


# ----------------------------------------------------------------------------- single points
def single_points(names: list[str], gpu: bool, pme: dict | None = None, tag: str = "") -> None:
    """Compare pmemd-pgm's single points with the engine for the systems; add the results to RESULT.

    Parameters
    ----------
    names : list of str
        Systems (SYSTEMS).
    gpu : bool
        Also pmemd.pgm.cuda_DPFP and _SPFP.
    pme : dict, optional
        settings_for keywords (e.g. spacing, order).
    tag : str
        Suffix of the result keys and run directories.
    """
    res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}
    for name in names:
        t0 = time.time()
        asys, templates = model(name)
        wd = os.path.join(OUT, name + tag)
        os.makedirs(wd, exist_ok=True)
        prm = os.path.join(wd, f"{name}_pgm.prmtop")
        info = write_pgm_prmtop(asys, prm, templates)
        crd = SYSTEMS[name][1]
        st = settings_for(asys.box, **(pme or {}))
        mdin = sp_mdin(st, asys.box)
        xyz = asys.positions * 10.0
        eng, F_eng = engine(asys, templates, st)
        other = np.setdiff1d(np.arange(len(F_eng)), cmap_atoms(asys, templates))
        r = {
            "atoms": int(len(xyz)),
            "write": {k: v for k, v in info.items() if k != "exported"},
            "settings": {
                "cut_A": 10 * st.cutoffs.cutoff,
                "ew_coeff": st.pme.ewald_beta / 10,
                "nfft": list(st.pme.grid),
                "order": st.pme.order,
                "dipole_scf_tol": st.induction.tol,
            },
            "engine": eng,
            "rms_force": float(np.sqrt(np.mean(F_eng**2))),
        }
        runs = ["cpu"] + (["gpu_dpfp", "gpu_spfp"] if gpu else [])
        if st.pme.order not in (4, 5, 6):
            runs = ["cpu"]  # pmemd.pgm.cuda: PME orders 4, 5, 6 only
        if name.startswith("water"):
            runs.append("cpu_original")
        for kind in runs:
            exe, top = (kind, prm) if kind != "cpu_original" else ("cpu", SYSTEMS[name][0])
            try:
                secs = run_pmemd(exe, os.path.join(wd, kind), top, crd, mdin)
            except SmallBox as e:
                r[kind] = {"skipped": str(e)}
                print(name, kind, "skipped:", e, flush=True)
                continue
            E = read_energies(os.path.join(wd, kind, "mdout"))
            F = read_forces(os.path.join(wd, kind, "mdfrc"))
            e_rb, f_rb = rigid_bond_terms(top, asys, templates, xyz)
            E["BOND"] -= e_rb
            F = F - f_rb
            dF = F - F_eng
            r[kind] = {
                "energies": {k: E.get(k, 0.0) for k in TERMS},
                "rigid_bond_energy_removed": e_rb,
                "diff": {k: E.get(k, 0.0) - eng[k] for k in TERMS},
                "force_rms_diff": float(np.sqrt(np.mean(dF**2))),
                "force_max_diff": float(np.abs(dF).max()),
                "force_max_diff_without_cmap_atoms": float(np.abs(dF[other]).max()),
                "force_rms_diff_without_cmap_atoms": float(np.sqrt(np.mean(dF[other] ** 2))),
                "seconds": secs,
            }
            print(
                name,
                kind,
                json.dumps(r[kind]["diff"]),
                "force rms/max diff",
                r[kind]["force_rms_diff"],
                r[kind]["force_max_diff"],
                "without CMAP atoms",
                r[kind]["force_rms_diff_without_cmap_atoms"],
                r[kind]["force_max_diff_without_cmap_atoms"],
                flush=True,
            )
        r["seconds"] = time.time() - t0
        res[name + tag] = r
        json.dump(res, open(RESULT, "w"), indent=1)


# ----------------------------------------------------------------------------- MD
def ca_rmsd(asys: object, frames_prm: np.ndarray) -> list[float]:
    """Return the CA RMSD [A] (after optimal superposition) from the input structure along frames (F, N, 3) [A].

    The frames are in prmtop order; the CA atoms of all protein molecules are used.
    """
    ca = np.array(
        [a for m in asys.molecules if m.kind == "protein" for a, nm in zip(m.atoms, m.atom_names) if nm == "CA"]
    )
    ref = asys.positions[ca] * 10.0
    ref = ref - ref.mean(0)
    out = []
    for X in frames_prm:
        y = X[ca] - X[ca].mean(0)
        u, _, vt = np.linalg.svd(y.T @ ref)
        d = np.sign(np.linalg.det(u @ vt))
        Rm = u @ np.diag([1.0, 1.0, d]) @ vt
        out.append(float(np.sqrt(np.mean(np.sum((y @ Rm - ref) ** 2, axis=1)))))
    return out


def _rmsd_summary(r: list[float]) -> dict:
    """Return the RMSD series [A], the mean over its second half and its maximum."""
    half = r[len(r) // 2 :]
    return {
        "ca_rmsd_A": [round(x, 3) for x in r],
        "ca_rmsd_second_half_mean": float(np.mean(half)),
        "ca_rmsd_max": float(np.max(r)),
    }


def md_engine(name: str, ps: float, seed: int) -> None:
    """Run the same model and protocol in the engine and add CA RMSD and ns/day to RESULT.

    Mixed precision, GPU: minimisation, then NVT at 298 K, Langevin 1/ps, dt 2 fs, X-H
    constraints, rigid water, no HMR; ps [ps] of production.
    """
    asys, templates = model(name)
    wd = os.path.join(OUT, f"engine_md_{name}_s{seed}")
    os.makedirs(wd, exist_ok=True)
    st = MDSettings().replace(cutoff=0.9, dipole_tol=1e-5, pme_grid=pmemd_grid(asys.box))
    sim = FlexibleSimulation(
        asys.system(),
        templates,
        asys.system_positions(),
        asys.box,
        st,
        dt=0.002,
        thermostat=Langevin(1.0),
        constraints="h-bonds",
        temperature=298.0,
        seed=seed,
    )
    sim.minimize(300)
    n = int(round(ps / 0.002))
    t0 = time.time()
    sim.run(n, report_every=5000, traj_every=5000, prefix=os.path.join(wd, "md"))
    secs = time.time() - t0
    X, _, _ = read_trajectory(os.path.join(wd, "md.nc"))
    Xp = np.empty_like(X)
    Xp[:, np.asarray(asys.order)] = X
    res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}
    res[f"engine_md_{name}_s{seed}"] = {
        "atoms": int(len(asys.positions)),
        "ps": ps,
        "ns_per_day": ps * 1e-3 / (secs / 86400.0),
        **_rmsd_summary(ca_rmsd(asys, Xp)),
    }
    print(json.dumps(res[f"engine_md_{name}_s{seed}"], indent=1))
    json.dump(res, open(RESULT, "w"), indent=1)


def md(name: str, ps: float, hmr: float | None, seed: int = 11) -> None:
    """Run pmemd.pgm.cuda_SPFP NVT MD of a written system and add the statistics to RESULT.

    500 minimisation steps and 2 ps at 0.5 fs heating from 0 K (tleap structures have close
    contacts), then ps [ps] at dt 2 fs with SHAKE; hmr: hydrogen mass [amu] (None: unchanged).
    """
    asys, templates = model(name)
    tag = f"md_{name}" + ("" if seed == 11 else f"_s{seed}")
    wd = os.path.join(OUT, tag)
    os.makedirs(wd, exist_ok=True)
    prm = os.path.join(wd, f"{name}_pgm.prmtop")
    write_pgm_prmtop(asys, prm, templates, hmr=hmr)
    st = MDSettings().replace(cutoff=0.9, dipole_tol=1e-5, pme_grid=pmemd_grid(asys.box))  # docs/protein_ff.md settings
    run_pmemd(
        "gpu_spfp", os.path.join(wd, "min"), prm, SYSTEMS[name][1], pmemd_mdin(st, asys.box, maxcyc=500, ntpr=100)
    )
    warm = pmemd_mdin(st, asys.box, nstlim=4000, dt=0.0005, temperature=298.0, tempi=0.0, ntpr=500, ntwr=4000, ig=seed)
    run_pmemd("gpu_spfp", os.path.join(wd, "warm"), prm, os.path.join(wd, "min/restrt"), warm)
    n = int(round(ps / 0.002))
    prod = pmemd_mdin(
        st, asys.box, nstlim=n, dt=0.002, temperature=298.0, irest=1, ntpr=500, ntwx=5000, ntwr=n, ig=seed + 1
    )
    secs = run_pmemd("gpu_spfp", os.path.join(wd, "prod"), prm, os.path.join(wd, "warm/restrt"), prod)
    txt = open(os.path.join(wd, "prod/mdout")).read()
    steps = [
        (int(a), float(b), float(c), float(d))
        for a, b, c, d in re.findall(
            r"NSTEP =\s+(\d+)\s+TIME\(PS\) =\s+\S+\s+TEMP\(K\) =\s+(\S+).*?Etot\s+=\s+(\S+)\s+EKtot\s+=\s+\S+\s+"
            r"EPtot\s+=\s+(\S+)",
            txt,
            re.S,
        )
    ][:-2]  # without the averages
    nsday = re.findall(r"ns/day =\s+([\d.]+)", txt)
    T = np.array([s[1] for s in steps])
    rmsd = (
        ca_rmsd(asys, read_trajectory(os.path.join(wd, "prod/mdcrd"))[0])
        if any(m.kind == "protein" for m in asys.molecules) and ps >= 20
        else None
    )
    res = json.load(open(RESULT)) if os.path.exists(RESULT) else {}
    ms = re.findall(r"Per Step\(ms\) =\s+([\d.]+)", txt)
    res[tag] = {
        "atoms": int(len(asys.positions)),
        "ps": ps,
        "dt_fs": 2.0,
        "hmr": hmr,
        "thermostat": "Langevin 1/ps",
        "ms_per_step": float(ms[-1]) if ms else None,
        "pme_grid": list(st.pme.grid),
        "shake": "X-H + rigid water",
        "temperature_mean": float(T.mean()),
        "temperature_std": float(T.std()),
        "eptot_first_last": [steps[0][3], steps[-1][3]],
        "ns_per_day": float(nsday[-1]) if nsday else None,
        "wall_seconds": secs,
        **(_rmsd_summary(rmsd) if rmsd else {}),
    }
    print(json.dumps(res[tag], indent=1))
    json.dump(res, open(RESULT, "w"), indent=1)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the mode (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("sp", "md", "md-engine"), help="single points, pmemd MD or engine MD")
    ap.add_argument("--systems", default="water512,water4096,pep,trpcage", help="sp: comma-separated systems")
    ap.add_argument("--system", default="trpcage", choices=sorted(SYSTEMS), help="md, md-engine: the system")
    ap.add_argument("--gpu", action="store_true", help="sp: also pmemd.pgm.cuda_DPFP and _SPFP")
    ap.add_argument("--time-ps", type=float, default=200.0, help="md, md-engine: production [ps]")
    ap.add_argument("--hmr-amu", type=float, default=None, help="md: hydrogen mass [amu] (default: unchanged)")
    ap.add_argument("--seed", type=int, default=11, help="md: pmemd ig of the warm-up (production: seed + 1)")
    ap.add_argument("--tight", action="store_true", help="PME spacing 0.04 nm, order 8 (result key <system>_tight)")
    ap.add_argument(
        "--amber-lambda",
        action="store_true",
        help="engine PME with pmemd's influence-function factor (result key <system>_lambda)",
    )
    a = ap.parse_args(argv)
    if a.mode != "md-engine" and "JAX_PLATFORMS" not in os.environ:
        jax.config.update("jax_platforms", "cpu")  # the GPU is pmemd.pgm.cuda's (exclusive-process)
    setup_logging()
    if a.mode == "sp":
        pme, tag = ({"spacing": 0.04, "order": 8}, "_tight") if a.tight else (None, "")
        if a.amber_lambda:
            use_amber_lambda()
            tag += "_lambda"
        single_points(a.systems.split(","), a.gpu, pme, tag)
    elif a.mode == "md":
        md(a.system, a.time_ps, a.hmr_amu, a.seed)
    else:
        md_engine(a.system, a.time_ps, a.seed)


if __name__ == "__main__":
    main()
