"""Check flexible-molecule MD with methanol: gas-phase forces, NVE energy conservation and a short NPT run.

1. One molecule in a 4 nm box against the gas-phase model the bonded terms were fitted with
   (images and PME error aside).
2. 216 molecules: 2 ps Langevin NVT (0.5 fs, friction 5/ps), then 2 ps NVE; total energy and its
   drift.
3. 50 ps Monte Carlo NPT (Langevin 1/ps, a volume move every 50 steps) from the NVT state.
The template runs/flex/methanol.flex is fitted first (class II set) when it does not exist.

Usage:

    python examples/flex_methanol_check.py            # on a GPU node
    python examples/flex_methanol_check.py --help

Inputs: runs/flex/methanol.flex, or the bonded data set to fit it (examples/fit_bonded_template.py).
Outputs: <out-dir>/{equil,nve,npt}.log (and trajectory/checkpoint files); results on stdout.
Units: kJ/mol/nm (forces), kJ/mol (energies), ps, K.
Runtime: GPU, minutes (plus the fit on first use).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.bonded import terms as T
from pgm_jax.bonded.fit import Fitter
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.bonded.study.data import load
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.thermostats import Langevin
from pgm_jax.paths import repo_path
from pgm_jax.system import System

jax.config.update("jax_enable_x64", True)


def methanol_template(out: str) -> FlexibleTemplate:
    """Return the methanol template <out>/methanol.flex, fitting it (class II set) if it does not exist."""
    tpath = os.path.join(out, "methanol.flex")
    if not os.path.exists(tpath):
        specs, data = load(["methanol"])
        model = BondedModel(specs, BondedSettings(families=T.PAPER))
        P = Fitter(model, {0: {"train": data[0]["train"]}}).fit(model.init_params(), maxiter=20000, verbose=True)
        FlexibleTemplate.from_fit(model, P).save(tpath)
    return FlexibleTemplate.load(tpath)


def check_single_molecule(tpl: FlexibleTemplate) -> None:
    """Print the largest difference between MD and gas-phase forces of one perturbed molecule [kJ/mol/nm]."""
    x = np.asarray(tpl.spec.ref_xyz) + 0.003 * np.random.default_rng(0).normal(size=(tpl.n, 3))
    H = np.eye(3) * 4.0
    sim = FlexibleSimulation(
        System([tpl.pgm]),
        [tpl],
        x + 2.0,
        H,
        MDSettings().replace(precision="double", dipole_tol=1e-8, cutoff=1.8, skin=0.05),
        thermostat=None,
        log=None,
    )
    F_md = np.asarray(sim.state.dyn.force)
    m = tpl.model
    _e_gas, g_gas = jax.value_and_grad(lambda R: m.energy(0, R, jax.tree_util.tree_map(jnp.asarray, tpl.P))[0])(
        jnp.asarray(x)
    )
    print(
        f"single molecule: |F_md - F_gas| max {np.abs(F_md + np.asarray(g_gas)).max():.3g}, rms F "
        f"{np.sqrt(np.mean(np.asarray(g_gas) ** 2)):.3g} kJ/mol/nm"
    )


def check_nve(tpl: FlexibleTemplate, out: str, st: MDSettings) -> FlexibleSimulation:
    """Equilibrate 216 molecules (2 ps NVT), run 2 ps NVE, print the total energies; return the NVT simulation."""
    pos, H = liquid_box(tpl, 216, 0.55, seed=1, min_dist=0.18)
    sim = FlexibleSimulation(
        System([tpl.pgm] * 216),
        [tpl] * 216,
        pos,
        H,
        st,
        dt=0.0005,
        thermostat=Langevin(5.0),
        temperature=298.0,
        log=None,
    )
    sim.run(4000, report_every=4000, prefix=os.path.join(out, "equil"))
    sim2 = FlexibleSimulation(
        System([tpl.pgm] * 216),
        [tpl] * 216,
        sim.positions(),
        np.asarray(sim.state.box),
        st,
        dt=0.0005,
        thermostat=None,
        velocities=sim.velocities(),
        log=None,
    )
    E = []
    for _ in range(10):
        sim2.run(400, report_every=400, prefix=os.path.join(out, "nve"))
        o = sim2.observables()
        E.append(o["etot"])
    E = np.array(E)
    print(
        "NVE 2 ps: total energy",
        np.round(E, 2),
        "drift per ps per dof (kT)",
        (E[-1] - E[0]) / 1.8 / (3 * 216 * 6) / (0.0083145 * 298),
    )
    return sim


def check_npt(tpl: FlexibleTemplate, sim: FlexibleSimulation, out: str, st: MDSettings) -> None:
    """Run 50 ps of NPT from the state of sim and print the mean observables and the wall time."""
    t0 = time.time()
    sim3 = FlexibleSimulation(
        System([tpl.pgm] * 216),
        [tpl] * 216,
        sim.positions(),
        np.asarray(sim.state.box),
        st,
        dt=0.0005,
        thermostat=Langevin(1.0),
        barostat=MonteCarloBarostat(every=50),
        temperature=298.0,
        velocities=sim.velocities(),
        log=None,
    )
    sim3.run(100000, report_every=2000, prefix=os.path.join(out, "npt"))
    print(
        "NPT 50 ps:",
        {k: round(v, 3) for k, v in sim3.observables().items() if isinstance(v, float)},
        "wall %.0f s" % (time.time() - t0),
    )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the three checks (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=repo_path("runs", "flex"), help="directory of the template and the logs")
    a = ap.parse_args(argv)
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    tpl = methanol_template(out)
    print("template", tpl.name, tpl.n, "atoms; intramolecular LJ pairs", len(tpl.lj_pairs()[0]))
    check_single_molecule(tpl)
    st = MDSettings().replace(precision="mixed", dipole_tol=1e-5)
    sim = check_nve(tpl, out, st)
    check_npt(tpl, sim, out, st)


if __name__ == "__main__":
    main()
