"""Run i-PI with the pgm_jax client: input files, server process, output parsing.

Contents: `ipi_command`, `write_xyz`, `write_input` (init.xyz + input.xml), `start_server`,
`read_properties`, `run` (server + client, used by the tests and validation scripts).

i-PI is found through the environment variable IPI_ROOT (a directory holding the `ipi` package and
`ipi-*.data/scripts/i-pi`, e.g. an unpacked wheel) or an installed `i-pi` on PATH.

Units: pgm_jax's nm at the Python side; i-PI files in Angstrom, fs, K, bar and atomic units as
named in each function.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..units import HARTREE_KJMOL

if TYPE_CHECKING:
    from .ipi import IPIClient


def ipi_command() -> tuple[list[str], str | None]:
    """Return the command that starts i-PI and the root to put on PYTHONPATH (None for an installed i-pi).

    Raises
    ------
    FileNotFoundError
        If neither IPI_ROOT nor an i-pi on PATH is found.
    """
    root = os.environ.get("IPI_ROOT")
    if root:
        scr = sorted(glob.glob(os.path.join(root, "ipi-*.data", "scripts", "i-pi")))
        if scr:
            return [sys.executable, scr[-1]], root
    exe = shutil.which("i-pi")
    if exe:
        return [exe], None
    raise FileNotFoundError("i-PI not found: set IPI_ROOT or install i-pi")


def cell_abc(H_nm: ArrayLike) -> tuple[float, float, float, float, float, float]:
    """Return the cell parameters a, b, c [Angstrom] and alpha, beta, gamma [deg] of a box.

    `H_nm` (3, 3) holds the lattice vectors as rows [nm].
    """
    H = np.asarray(H_nm, float) * 10.0
    a, b, c = (np.linalg.norm(v) for v in H)

    def ang(u: np.ndarray, v: np.ndarray) -> float:
        """Return the angle between u and v [deg]."""
        return np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))

    return a, b, c, ang(H[1], H[2]), ang(H[0], H[2]), ang(H[0], H[1])


def write_xyz(path: str, symbols: Sequence[str], pos_nm: ArrayLike, H_nm: ArrayLike) -> None:
    """Write an extended xyz file with i-PI's CELL(abcABC) comment line (Angstrom).

    Parameters
    ----------
    path : str
        Output file.
    symbols : sequence of str
        Element symbols.
    pos_nm : ArrayLike (N, 3)
        Positions [nm].
    H_nm : ArrayLike (3, 3)
        Box, lattice vectors as rows [nm].
    """
    a, b, c, al, be, ga = cell_abc(H_nm)
    with open(path, "w") as fh:
        fh.write(
            f"{len(symbols)}\n# CELL(abcABC): {a:.10f} {b:.10f} {c:.10f} {al:.8f} {be:.8f} {ga:.8f} "
            f"cell{{angstrom}} positions{{angstrom}}\n"
        )
        for s, x in zip(symbols, np.asarray(pos_nm) * 10.0):
            fh.write(f"{s} {x[0]:.10f} {x[1]:.10f} {x[2]:.10f}\n")


def write_input(
    workdir: str,
    symbols: Sequence[str],
    pos_nm: ArrayLike,
    H_nm: ArrayLike,
    masses: Sequence[float],
    *,
    nbeads: int = 1,
    steps: int = 1000,
    dt_fs: float = 0.5,
    T: float = 298.0,
    ensemble: str = "nvt",
    thermostat: str = "pile_g",
    tau_fs: float = 100.0,
    address: str = "pgmjax",
    stride: int = 10,
    batch_size: int = 1,
    seed: int = 31415,
    extra_props: Sequence[str] = (),
    traj_stride: int = 0,
    velocities: ArrayLike | None = None,
    pile_lambda: float | None = None,
    pressure: float | None = None,
    barostat_tau_fs: float = 200.0,
    velocity_units: str = "atomic_unit",
    splitting: str | None = None,
    nm_propagator: str | None = None,
    pressure_output: bool = True,
) -> str:
    """Write init.xyz and input.xml for an i-PI run served over the unix socket `address`.

    Parameters
    ----------
    workdir : str
        Directory (created if needed).
    symbols : sequence of str
        Element symbols.
    pos_nm : ArrayLike (N, 3)
        Positions [nm].
    H_nm : ArrayLike (3, 3)
        Box [nm].
    masses : sequence of float
        Masses [amu].
    nbeads : int
        Ring-polymer beads.
    steps : int
        Total steps.
    dt_fs : float
        Time step [fs].
    T : float
        Temperature [K].
    ensemble : str
        i-PI dynamics mode ("nve", "nvt", "npt", ...); a thermostat is written for nvt / npt, an
        isotropic barostat (with a Langevin thermostat) for npt.
    thermostat : str
        i-PI thermostat mode.
    tau_fs : float
        Thermostat time constant [fs].
    address : str
        Unix socket name.
    stride : int
        Steps between rows of the properties file.
    batch_size : int
        i-PI's batch size (> 1: batched requests).
    seed : int
        PRNG seed.
    extra_props : sequence of str
        Further i-PI properties.
    traj_stride : int
        Steps between centroid trajectory frames (0: none).
    velocities : ArrayLike (N, 3), optional
        Initial velocities in `velocity_units`; None: thermal at T.
    pile_lambda : float, optional
        PILE lambda; None: i-PI's default.
    pressure : float, optional
        Pressure [bar]; None: none written.
    barostat_tau_fs : float
        Barostat time constant [fs].
    velocity_units : str
        i-PI units of `velocities` (atomic units by default).
    splitting : str, optional
        Integrator splitting ("obabo", "baoab"); None: i-PI's default.
    nm_propagator : str, optional
        Normal-mode propagator; None: i-PI's default.
    pressure_output : bool
        Include pressure_cv{bar} in the properties.

    Returns
    -------
    str
        Path of input.xml.
    """
    os.makedirs(workdir, exist_ok=True)
    write_xyz(os.path.join(workdir, "init.xyz"), symbols, pos_nm, H_nm)
    props = (
        ["step", "time{picosecond}", "conserved", "temperature{kelvin}", "potential", "kinetic_md", "kinetic_cv"]
        + (["pressure_cv{bar}"] if pressure_output else [])
        + ["volume"]
        + list(extra_props)
    )
    mass = "[ " + ", ".join(f"{m:.6f}" for m in masses) + " ]"
    vel = f'<velocities mode="thermal" units="kelvin"> {T} </velocities>'
    if velocities is not None:
        v = np.asarray(velocities, float)
        vel = (
            f'<velocities mode="manual" units="{velocity_units}"> [ '
            + ", ".join(f"{x:.12e}" for x in v.reshape(-1))
            + " ] </velocities>"
        )
    thermo = ""
    if ensemble in ("nvt", "npt"):
        lam = "" if pile_lambda is None else f"<pile_lambda> {pile_lambda} </pile_lambda>"
        thermo = f'<thermostat mode="{thermostat}"><tau units="femtosecond"> {tau_fs} </tau>{lam}</thermostat>'
    baro = ""
    if ensemble == "npt":
        baro = (
            f'<barostat mode="isotropic"><tau units="femtosecond"> {barostat_tau_fs} </tau>'
            f'<thermostat mode="langevin"><tau units="femtosecond"> {tau_fs} </tau></thermostat></barostat>'
        )
    ens = f'<temperature units="kelvin"> {T} </temperature>'
    if pressure is not None:
        ens += f'<pressure units="bar"> {pressure} </pressure>'
    traj = (
        f'<trajectory filename="pos" stride="{traj_stride}" format="xyz" cell_units="angstrom"> '
        "x_centroid{angstrom} </trajectory>"
        if traj_stride
        else ""
    )
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


def start_server(workdir: str, address: str) -> subprocess.Popen:
    """Start the i-PI server (background process) in workdir; removes a stale socket first.

    The server's threads are limited to IPI_THREADS (default 1) so that the engine keeps the cores;
    its output goes to workdir/ipi.log.  Returns the process.
    """
    try:
        os.remove("/tmp/ipi_" + address)
    except FileNotFoundError:
        pass
    cmd, root = ipi_command()
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # leave the cores to the engine
        env[k] = os.environ.get("IPI_THREADS", "1")
    if root:
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    log = open(os.path.join(workdir, "ipi.log"), "w")
    return subprocess.Popen(cmd + ["input.xml"], cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT)


def read_properties(
    path: str, hartree_to_kjmol: Sequence[str] = ("conserved", "potential", "kinetic_md", "kinetic_cv")
) -> dict[str, np.ndarray]:
    """Return an i-PI properties file as {column name: array}.

    Energies written in atomic units (names starting with those in `hartree_to_kjmol`, without an
    explicit unit) are converted to kJ/mol; every column is also available under its name without
    the unit.
    """
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
    for k, n in enumerate(names[: data.shape[1]]):
        base = n.split("{")[0].split("(")[0]
        v = data[:, k] * (HARTREE_KJMOL if base in hartree_to_kjmol and "{" not in n else 1.0)
        out[n] = v
        out.setdefault(n.split("{")[0], v)  # also without the unit
    return out


def run(
    workdir: str, address: str, client_factory: Callable[[], IPIClient], timeout_s: float = 36000
) -> tuple[IPIClient, dict, dict[str, np.ndarray], float]:
    """Start i-PI, serve it with client_factory() (an IPIClient) and wait for the server.

    Parameters
    ----------
    workdir : str
        Directory with input.xml (`write_input`).
    address : str
        Unix socket name (a stale socket is removed).
    client_factory : callable
        Returns the IPIClient that serves the run.
    timeout_s : float
        Longest wait for the server after the client finished [s]; the server is terminated if it
        is still running.

    Returns
    -------
    client : IPIClient
        The client.
    stats : dict
        The client's stats.
    props : dict
        Properties of workdir/sim.out (`read_properties`).
    wall : float
        Wall time [s].
    """
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
