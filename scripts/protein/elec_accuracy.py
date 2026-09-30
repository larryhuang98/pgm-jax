"""Electrostatic accuracy of real-space cutoffs (MDSettings.elec_cutoff) against a tight reference.

The van der Waals cutoff is kept at --cutoff-nm; the reference is 0.9 nm, beta 4.0 nm^-1, PME
spacing 0.04 nm, order 6, dipole tolerance 1e-9, float64.  Every case has the same van der Waals
term, so force differences are electrostatic; they are reported as rms (and max) force difference
/ rms electrostatic force, with the electrostatic energy difference and the rms induced-dipole
difference.  Cases: the default (0.9 nm / 4.0 / 0.08 nm) and, for every --elec-cutoff-nm,
elec_cutoff_settings with the grid exponents of --exponents; also the reference itself against a
longer real-space cutoff (1.2 nm or the largest the box allows, order 8, 0.03 nm).  See
docs/protein_ff.md.

Usage:

    python scripts/protein/elec_accuracy.py runs/protein/ubq.prmtop runs/protein/ubq.inpcrd --frame eq.npz
    python scripts/protein/elec_accuracy.py --water                   # 512 pGM3P-25 waters
    python scripts/protein/elec_accuracy.py --help

Inputs: a tleap prmtop and coordinates (or --water: PGM_GVDW_DATA's box); --frame: an .npz with
`pos` and `box` [nm], e.g. an equilibrated frame (the tleap coordinates have clashes that dominate
the forces).
Outputs: a printed table.
Units: nm (--cutoff-nm, --elec-cutoff-nm), kJ/mol, kJ/mol/nm.
Runtime: GPU or CPU; one force evaluation per case (compilation dominates).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.cli.args import setup_logging
from pgm_jax.md.box import box_from_cell, max_cutoff
from pgm_jax.md.forcefield import MDSettings, PGMForceField, elec_cutoff_settings
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.topology import MDTopology
from pgm_jax.paths import pgm3p25_files
from pgm_jax.protein import ResidueLibrary, amber_template, load_amber
from pgm_jax.system import System

jax.config.update("jax_enable_x64", True)


def load_system(a: argparse.Namespace) -> tuple[System, jax.Array, jax.Array, MDTopology | None]:
    """Return the system, positions [nm], box [nm] and topology (None for the water box) of the options."""
    top = None
    if a.water:
        prmtop, rst = pgm3p25_files()
        system = System.from_prmtop(prmtop)
        xyz, _, cell = read_coordinates(rst)
        pos, H = xyz * 0.1, box_from_cell(*cell) * 0.1
    else:
        lib = ResidueLibrary.load(a.library) if a.library else "placeholder"
        asys = load_amber(a.prmtop, a.inpcrd, electrostatics=lib)
        prot = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
        temps = asys.templates({k: amber_template(asys.molecules[k], a.prmtop) for k in prot})
        system = asys.system()
        rules = {id(t): t.md_rule("h-bonds") for t in {id(t): t for t in temps}.values()}
        top = MDTopology.build(system, [rules[id(t)] for t in temps])  # special pairs as FlexibleSimulation
        pos, H = asys.system_positions(), asys.box
    if a.frame:
        f = np.load(a.frame)
        pos, H = f["pos"], f["box"]
    return system, jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64), top


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, evaluate every case and print the table (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prmtop", nargs="?", help="tleap prmtop (not with --water)")
    ap.add_argument("inpcrd", nargs="?", help="tleap coordinates")
    ap.add_argument("--water", action="store_true", help="the 512-water pGM3P-25 box of the tests instead of a protein")
    ap.add_argument("--frame", help=".npz with pos and box [nm]")
    ap.add_argument("--library", default=None, help="pGM residue library (default: placeholder electrostatics)")
    ap.add_argument("--cutoff-nm", type=float, default=0.9, help="van der Waals cutoff [nm]")
    ap.add_argument(
        "--elec-cutoff-nm", type=float, nargs="+", default=[0.8, 0.7, 0.6], help="electrostatic cutoffs [nm]"
    )
    ap.add_argument("--exponents", type=float, nargs="+", default=[1.6, 1.0], help="grid rules of elec_cutoff_settings")
    ap.add_argument("--precision", nargs="+", default=["mixed"], help="precisions of the cases (mixed, double)")
    a = ap.parse_args(argv)
    setup_logging()
    system, pos, H, top = load_system(a)
    far = min(1.2, np.floor((max_cutoff(np.asarray(H)) - 0.002) * 100) / 100)
    print(
        f"{system.n} atoms, box heights up to {max_cutoff(np.asarray(H)) * 2:.3f} nm, device {jax.devices()[0]}",
        flush=True,
    )

    def run(label, **kw):
        """Return the force-field result of one case (the reference settings updated with kw)."""
        base = dict(
            cutoff=a.cutoff_nm,
            skin=0.1,
            ewald_beta=4.0,
            pme_spacing=0.08,
            pme_order=6,
            dipole_tol=1e-9,
            max_iter=200,
            precision="double",
            lj_lrc=True,
        )
        s = MDSettings().replace(**{**base, **kw})
        t = time.time()
        ff = PGMForceField(system, H, s, topology=top)
        idx = ff.rows_for(pos, H)
        ff.size_rows(pos, H, idx)
        r = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
        jax.block_until_ready(r.forces)
        if bool(r.overflow):
            raise RuntimeError("row capacity overflow")
        print(
            f"  [{label}: grid {ff.pme.K}, rows {ff.capacity}, {int(r.iterations)} CG iterations, "
            f"{time.time() - t:.1f} s]",
            flush=True,
        )
        return r

    ref = run("reference", pme_spacing=0.04)
    rmsF = float(jnp.sqrt(jnp.mean(run("reference, no van der Waals", pme_spacing=0.04, vdw="none").forces ** 2)))
    Eel = float(ref.energy["elec"])
    rmsmu = float(jnp.sqrt(jnp.mean(ref.induction.mu**2)))
    print(f"rms electrostatic force {rmsF:.2f} kJ/mol/nm, E_elec {Eel:.2f} kJ/mol", flush=True)
    print(f"{'case':58s} {'F rms':>9s} {'F max':>9s} {'dE kJ/mol':>10s} {'dE rel':>8s} {'mu rms':>8s}", flush=True)

    def report(label, r):
        """Print the force, energy and induced-dipole differences of a case to the reference."""
        dF = r.forces - ref.forces
        print(
            f"{label:58s} {float(jnp.sqrt(jnp.mean(dF**2))) / rmsF:9.2e} {float(jnp.max(jnp.abs(dF))) / rmsF:9.2e} "
            f"{float(r.energy['elec']) - Eel:10.4f} {abs(float(r.energy['elec']) - Eel) / abs(Eel):8.1e} "
            f"{float(jnp.sqrt(jnp.mean((r.induction.mu - ref.induction.mu) ** 2))) / rmsmu:8.1e}",
            flush=True,
        )

    report(
        f"reference vs {far} nm / 4.0 / 0.03, order 8",
        run("longer cutoff", elec_cutoff=far, pme_spacing=0.03, pme_order=8),
    )
    for prec in a.precision:
        tol = 1e-9 if prec == "double" else 1e-6
        report(f"0.9 / 4.00 / 0.0800 (default), {prec}", run("default", precision=prec, dipole_tol=tol))
        for rc in a.elec_cutoff_nm:
            for ex in a.exponents:
                kw = elec_cutoff_settings(rc, exponent=ex)
                report(
                    f"{rc} / {kw['ewald_beta']:.2f} / {kw['pme_spacing']:.4f} (exponent {ex:g}), {prec}",
                    run(f"{rc}, exponent {ex:g}", precision=prec, dipole_tol=tol, **kw),
                )


if __name__ == "__main__":
    main()
