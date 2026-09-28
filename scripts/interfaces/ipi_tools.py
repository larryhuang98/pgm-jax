"""Helpers to run i-PI with the pgm_jax client: input files, server process, output parsing.

i-PI is found through the environment variable IPI_ROOT (a directory holding the `ipi` package and
`ipi-*.data/scripts/i-pi`, e.g. an unpacked wheel) or an installed `i-pi` on PATH."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np

KJMOL_MEV = 10.364269656262175        # meV per kJ/mol


def ipi_command():
    root = os.environ.get("IPI_ROOT")
    if root:
        scr = sorted(glob.glob(os.path.join(root, "ipi-*.data", "scripts", "i-pi")))
        if scr:
            return [sys.executable, scr[-1]], root
    exe = shutil.which("i-pi")
    if exe:
        return [exe], None
    raise FileNotFoundError("i-PI not found: set IPI_ROOT or install i-pi")


def cell_abc(H_nm):
    H = np.asarray(H_nm, float) * 10.0
    a, b, c = (np.linalg.norm(v) for v in H)
    ang = lambda u, v: np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
    return a, b, c, ang(H[1], H[2]), ang(H[0], H[2]), ang(H[0], H[1])


def write_xyz(path, symbols, pos_nm, H_nm):
    a, b, c, al, be, ga = cell_abc(H_nm)
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\n# CELL(abcABC): {a:.10f} {b:.10f} {c:.10f} {al:.8f} {be:.8f} {ga:.8f} "
                 f"cell{{angstrom}} positions{{angstrom}}\n")
        for s, x in zip(symbols, np.asarray(pos_nm) * 10.0):
            fh.write(f"{s} {x[0]:.10f} {x[1]:.10f} {x[2]:.10f}\n")


def write_input(workdir, symbols, pos_nm, H_nm, masses, *, nbeads=1, steps=1000, dt_fs=0.5, T=298.0,
                ensemble="nvt", thermostat="pile_g", tau_fs=100.0, address="pgmjax", stride=10,
                batch_size=1, seed=31415, extra_props=(), traj_stride=0, velocities=None, pile_lambda=None,
                pressure=None, barostat_tau_fs=200.0, velocity_units="atomic_unit", splitting=None,
                nm_propagator=None):
    """init.xyz + input.xml for i-PI (unix socket `address`).  velocities: (N, 3) in velocity_units
    (i-PI's units; atomic units by default) or None (thermal at T)."""
    os.makedirs(workdir, exist_ok=True)
    write_xyz(os.path.join(workdir, "init.xyz"), symbols, pos_nm, H_nm)
    props = ["step", "time{picosecond}", "conserved", "temperature{kelvin}", "potential", "kinetic_md",
             "kinetic_cv", "pressure_cv{bar}", "volume"] + list(extra_props)          # energies: Hartree
    mass = "[ " + ", ".join(f"{m:.6f}" for m in masses) + " ]"
    vel = f'<velocities mode="thermal" units="kelvin"> {T} </velocities>'
    if velocities is not None:
        v = np.asarray(velocities, float)
        vel = (f'<velocities mode="manual" units="{velocity_units}"> [ ' +
               ", ".join(f"{x:.12e}" for x in v.reshape(-1)) + " ] </velocities>")
    thermo = ""
    if ensemble in ("nvt", "npt"):
        lam = "" if pile_lambda is None else f"<pile_lambda> {pile_lambda} </pile_lambda>"
        thermo = f'<thermostat mode="{thermostat}"><tau units="femtosecond"> {tau_fs} </tau>{lam}</thermostat>'
    baro = ""
    if ensemble == "npt":
        baro = (f'<barostat mode="isotropic"><tau units="femtosecond"> {barostat_tau_fs} </tau>'
                f'<thermostat mode="langevin"><tau units="femtosecond"> {tau_fs} </tau></thermostat></barostat>')
    ens = f'<temperature units="kelvin"> {T} </temperature>'
    if pressure is not None:
        ens += f'<pressure units="bar"> {pressure} </pressure>'
    traj = (f'<trajectory filename="pos" stride="{traj_stride}" format="xyz" cell_units="angstrom"> x_centroid{{angstrom}} </trajectory>'
            if traj_stride else "")
    batch = f"<batch_size> {batch_size} </batch_size>" if batch_size > 1 else ""
    split = f' splitting="{splitting}"' if splitting else ""
    nm = f'<normal_modes propagator="{nm_propagator}"/>' if nm_propagator else ""
    xml = f"""<simulation verbosity="low" safe_stride="100000">
  <output prefix="sim">
    <properties stride="{stride}" filename="out"> [ {", ".join(props)} ] </properties>
    {traj}
  </output>
  <total_steps> {steps} </total_steps>
  <prng><seed> {seed} </seed></prng>
  <ffsocket name="pgm" mode="unix" pbc="false">
    <address> {address} </address>
    <latency> 1e-4 </latency>
    {batch}
  </ffsocket>
  <system>
    <initialize nbeads="{nbeads}">
      <file mode="xyz" units="angstrom"> init.xyz </file>
      {vel}
      <masses mode="manual" units="dalton"> {mass} </masses>
    </initialize>
    <forces><force forcefield="pgm"/></forces>
    {nm}
    <motion mode="dynamics">
      <dynamics mode="{ensemble}"{split}>
        <timestep units="femtosecond"> {dt_fs} </timestep>
        {thermo}
        {baro}
      </dynamics>
    </motion>
    <ensemble> {ens} </ensemble>
  </system>
</simulation>
"""
    with open(os.path.join(workdir, "input.xml"), "w") as fh:
        fh.write(xml)
    return os.path.join(workdir, "input.xml")


def start_server(workdir, address):
    """Start the i-PI server (background process) in workdir; removes a stale socket first."""
    try:
        os.remove("/tmp/ipi_" + address)
    except FileNotFoundError:
        pass
    cmd, root = ipi_command()
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # leave the cores to the engine
        env[k] = os.environ.get("IPI_THREADS", "1")
    if root:
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    log = open(os.path.join(workdir, "ipi.log"), "w")
    return subprocess.Popen(cmd + ["input.xml"], cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT)


HARTREE_KJMOL = 2625.4996394799


def read_properties(path, hartree_to_kjmol=("conserved", "potential", "kinetic_md", "kinetic_cv")):
    """i-PI properties file -> dict of column name -> array; energies written in atomic units
    (names starting with those in hartree_to_kjmol) are converted to kJ/mol."""
    names = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if "-->" in line:
                    names.append(line.split("-->")[1].split(":")[0].strip())
            else:
                break
    data = np.loadtxt(path, ndmin=2)
    out = {}
    for k, n in enumerate(names[:data.shape[1]]):
        base = n.split("{")[0].split("(")[0]
        v = data[:, k] * (HARTREE_KJMOL if base in hartree_to_kjmol and "{" not in n else 1.0)
        out[n] = v
        out.setdefault(n.split("{")[0], v)                  # also without the unit
    return out


def run(workdir, address, client_factory, timeout_s=36000):
    """Start i-PI, serve it with client_factory() (an IPIClient), wait for the server; returns
    (client stats, properties dict, wall time)."""
    t0 = time.perf_counter()
    srv = start_server(workdir, address)
    client = client_factory()
    try:
        stats = client.run()
        srv.wait(timeout=timeout_s)
    finally:
        if srv.poll() is None:
            srv.terminate()
    wall = time.perf_counter() - t0
    props = read_properties(os.path.join(workdir, "sim.out"))
    return client, stats, props, wall
