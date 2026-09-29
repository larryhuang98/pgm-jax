"""Amber file I/O: coordinates, velocities and box in; NetCDF trajectories and restarts out.

Contents: `read_coordinates` (ASCII inpcrd / restart or NetCDF restart, Amber units),
`read_coordinates_nm` (the same in library units), `read_trajectory` (NetCDF trajectory frames),
`write_restart` (NetCDF restart) and `NetCDFTrajectory` (an appendable NetCDF trajectory written
without a NetCDF library).  The files follow the AMBER NetCDF convention 1.0, readable by
cpptraj, VMD and MDTraj.

Units: Amber's (Angstrom, Angstrom/ps, ps, degrees) in every name with a suffix (`xyz_A`,
`vel_A_ps`, `H_A`, `time_ps`); `read_coordinates_nm` returns nm and nm/ps.  Velocities in ASCII
files and NetCDF restarts are stored in Amber's unit A / (1/20.455 ps) (`AMBER_VEL`).
"""

from __future__ import annotations

import os
import struct
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.io import netcdf_file

from ..units import ANG_NM
from .box import box_from_cell, cell_parameters

AMBER_VEL = 20.455  # Amber velocity unit: A / (1/20.455 ps); v [A/ps] = v_file * AMBER_VEL


def read_coordinates(
    path: str,
) -> tuple[np.ndarray, np.ndarray | None, tuple[np.ndarray, np.ndarray] | None]:
    """Read coordinates, velocities and box of an Amber coordinate file (Amber units).

    The format is detected from the magic number: NetCDF ("CDF") or ASCII inpcrd / restart (fixed
    12-character fields after the title and atom-count lines; velocities if there are at least
    3 N more numbers, a box if 6 numbers remain).

    Parameters
    ----------
    path : str
        ASCII inpcrd / restart or NetCDF restart.

    Returns
    -------
    xyz : np.ndarray (N, 3)
        Coordinates [Angstrom].
    vel : np.ndarray (N, 3) or None
        Velocities [Angstrom/ps] (converted with the file's scale factor, default 20.455), None if
        the file has none.
    box : tuple of np.ndarray or None
        (cell lengths [Angstrom] (3,), cell angles [deg] (3,)), None if the file has no box.
    """
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:3] == b"CDF":
        f = netcdf_file(path, "r", mmap=False)
        v = f.variables
        xyz = np.array(v["coordinates"][:], float).reshape(-1, 3)
        vel = None
        if "velocities" in v:
            sc = float(getattr(v["velocities"], "scale_factor", AMBER_VEL))
            vel = np.array(v["velocities"][:], float).reshape(-1, 3) * sc
        box = (
            (
                np.array(v["cell_lengths"][:], float).reshape(-1)[:3],
                np.array(v["cell_angles"][:], float).reshape(-1)[:3],
            )
            if "cell_lengths" in v
            else None
        )
        f.close()
        return xyz, vel, box
    lines = open(path).read().splitlines()
    n = int(lines[1].split()[0])
    vals = [float(l[k : k + 12]) for l in lines[2:] for k in range(0, len(l.rstrip()), 12) if l[k : k + 12].strip()]
    xyz = np.array(vals[: 3 * n]).reshape(n, 3)
    rest = vals[3 * n :]
    vel = None
    if len(rest) >= 3 * n:
        vel = np.array(rest[: 3 * n]).reshape(n, 3) * AMBER_VEL
        rest = rest[3 * n :]
    box = (np.array(rest[:3]), np.array(rest[3:6])) if len(rest) >= 6 else None
    return xyz, vel, box


def read_coordinates_nm(path: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read the coordinates of an Amber file in library units.

    Parameters
    ----------
    path : str
        ASCII inpcrd / restart or NetCDF restart (read_coordinates).

    Returns
    -------
    positions : np.ndarray (N, 3)
        Positions [nm].
    velocities : np.ndarray (N, 3) or None
        Velocities [nm/ps].
    box : np.ndarray (3, 3) or None
        Lattice vectors as rows, lower triangular (box.box_from_cell) [nm].
    """
    xyz, vel, box = read_coordinates(path)
    H = None if box is None else box_from_cell(*box) * ANG_NM
    return xyz * ANG_NM, None if vel is None else vel * ANG_NM, H


def read_trajectory(
    path: str, atoms: Sequence[int] | np.ndarray | None = None, stride: int = 1
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Read the frames of an Amber NetCDF trajectory.

    Parameters
    ----------
    path : str
        NetCDF trajectory.
    atoms : Sequence[int] or np.ndarray, optional
        Atoms to keep (e.g. the protein; None: all).
    stride : int
        Keep every `stride`-th frame.

    Returns
    -------
    xyz : np.ndarray (F, N, 3) float64
        Coordinates [Angstrom].
    lengths : np.ndarray (F, 3) or None
        Cell lengths [Angstrom] (None without a box).
    time : np.ndarray (F,)
        Time [ps] (the frame index if the file has no time variable).
    """
    f = netcdf_file(path, "r", mmap=False)
    v = f.variables
    X = np.array(v["coordinates"][::stride], float)
    if atoms is not None:
        X = X[:, np.asarray(atoms)]
    box = np.array(v["cell_lengths"][::stride], float) if "cell_lengths" in v else None
    t = np.array(v["time"][::stride], float) if "time" in v else np.arange(len(X), dtype=float)
    f.close()
    return X, box, t


def write_restart(
    path: str,
    xyz_A: ArrayLike,
    vel_A_ps: ArrayLike | None,
    H_A: ArrayLike,
    time_ps: float,
    title: str = "pgm_jax restart",
) -> None:
    """Write an Amber NetCDF restart (AMBERRESTART convention 1.0, float64).

    Parameters
    ----------
    path : str
        Output file (overwritten).
    xyz_A : ArrayLike (N, 3)
        Coordinates [Angstrom].
    vel_A_ps : ArrayLike (N, 3), optional
        Velocities [Angstrom/ps] (stored divided by AMBER_VEL with scale_factor AMBER_VEL; None:
        no velocities).
    H_A : ArrayLike (3, 3)
        Box, lattice vectors as rows [Angstrom] (stored as cell lengths and angles).
    time_ps : float
        Simulation time [ps].
    title : str
        Title attribute.
    """
    f = netcdf_file(path, "w", version=2)
    f.Conventions, f.ConventionVersion, f.program, f.programVersion, f.title = (
        "AMBERRESTART",
        "1.0",
        "pgm_jax",
        "0.2",
        title,
    )
    n = len(xyz_A)
    f.createDimension("spatial", 3)
    f.createDimension("atom", n)
    f.createDimension("cell_spatial", 3)
    f.createDimension("cell_angular", 3)
    f.createDimension("label", 5)
    sp = f.createVariable("spatial", "c", ("spatial",))
    sp[:] = np.array(list("xyz"), "S1")
    cs = f.createVariable("cell_spatial", "c", ("cell_spatial",))
    cs[:] = np.array(list("abc"), "S1")
    ca = f.createVariable("cell_angular", "c", ("cell_angular", "label"))
    ca[:] = np.array([list("alpha"), list("beta "), list("gamma")], "S1")
    t = f.createVariable("time", "d", ())
    t.units = "picosecond"
    t[...] = time_ps
    c = f.createVariable("coordinates", "d", ("atom", "spatial"))
    c.units = "angstrom"
    c[:] = np.asarray(xyz_A, float)
    if vel_A_ps is not None:
        v = f.createVariable("velocities", "d", ("atom", "spatial"))
        v.units = "angstrom/picosecond"
        v.scale_factor = AMBER_VEL
        v[:] = np.asarray(vel_A_ps, float) / AMBER_VEL
    L, A = cell_parameters(H_A)
    cl = f.createVariable("cell_lengths", "d", ("cell_spatial",))
    cl.units = "angstrom"
    cl[:] = L
    cg = f.createVariable("cell_angles", "d", ("cell_angular",))
    cg.units = "degree"
    cg[:] = A
    f.close()


class NetCDFTrajectory:
    """Appendable Amber NetCDF trajectory, written byte by byte (NetCDF-3, 64-bit offsets).

    Variables: time (ps, float32), coordinates (Angstrom, float32), cell_lengths (Angstrom) and
    cell_angles (deg).  The header is written once; each frame appends one record and bumps the
    record count, so the file is valid after every frame (a crashed run keeps its frames).

        traj = NetCDFTrajectory("run.nc", n_atoms)
        traj.write(time_ps, xyz_A, H_A)

    Attributes
    ----------
    path : str
        File name.
    n : int
        Number of atoms.
    recsize : int
        Bytes per frame record (4 + 12 n + 24 + 24).
    nframes : int
        Frames in the file.
    NC_DIM, NC_VAR, NC_ATT : int
        NetCDF header tags of the dimension, variable and attribute lists.
    CHAR, FLOAT, DOUBLE : int
        NetCDF type codes.
    """

    NC_DIM, NC_VAR, NC_ATT = 10, 11, 12
    CHAR, FLOAT, DOUBLE = 2, 5, 6

    def __init__(self, path: str, n_atoms: int, append: bool = False) -> None:
        """Create the file and write its header, or open an existing file for appending.

        Parameters
        ----------
        path : str
            File name.
        n_atoms : int
            Number of atoms.
        append : bool
            If the file exists, append to it (the frame count is read from its header; the atom count
            is not checked); otherwise the file is created (overwritten).
        """
        self.path, self.n = path, int(n_atoms)
        self.recsize = 4 + 12 * self.n + 24 + 24  # time, coordinates, cell lengths, cell angles
        if append and os.path.exists(path):
            with open(path, "rb") as fh:
                fh.seek(4)  # after the magic "CDF\x02": numrecs, big-endian int32
                self.nframes = struct.unpack(">i", fh.read(4))[0]
            return
        self.nframes = 0
        self._write_header()

    @staticmethod
    def _name(s: str) -> bytes:
        """Return a NetCDF name: big-endian length, the bytes, zero padding to a multiple of 4."""
        b = s.encode()
        return struct.pack(">i", len(b)) + b + b"\0" * (-len(b) % 4)

    def _att(self, name: str, value: str | float) -> bytes:
        """Return one NetCDF attribute: a string (CHAR) or a number (one DOUBLE)."""
        if isinstance(value, str):
            b = value.encode()
            return self._name(name) + struct.pack(">ii", self.CHAR, len(b)) + b + b"\0" * (-len(b) % 4)
        return self._name(name) + struct.pack(">iid", self.DOUBLE, 1, float(value))

    def _atts(self, atts: dict) -> bytes:
        """Return a NetCDF attribute list (ABSENT, eight zero bytes, if empty)."""
        if not atts:
            return b"\0" * 8
        return struct.pack(">ii", self.NC_ATT, len(atts)) + b"".join(self._att(k, v) for k, v in atts.items())

    def _write_header(self) -> None:
        """Write the header and the fixed-size variables (spatial, cell_spatial, cell_angular).

        The header is built twice: once with zero offsets to get its length, then with the begin
        offsets of the fixed variables (after the header) and of the record variables (after the fixed
        ones; one record = time, coordinates, cell_lengths, cell_angles).
        """
        n = self.n
        dims = [("frame", 0), ("spatial", 3), ("atom", n), ("cell_spatial", 3), ("cell_angular", 3), ("label", 5)]
        D = {k: i for i, (k, _) in enumerate(dims)}
        # (name, dims, type, attributes, bytes per record or total, is a record variable)
        vars_ = [
            ("spatial", ["spatial"], self.CHAR, {}, 4, False),
            ("cell_spatial", ["cell_spatial"], self.CHAR, {}, 4, False),
            ("cell_angular", ["cell_angular", "label"], self.CHAR, {}, 16, False),
            ("time", ["frame"], self.FLOAT, {"units": "picosecond"}, 4, True),
            ("coordinates", ["frame", "atom", "spatial"], self.FLOAT, {"units": "angstrom"}, 12 * n, True),
            ("cell_lengths", ["frame", "cell_spatial"], self.DOUBLE, {"units": "angstrom"}, 24, True),
            ("cell_angles", ["frame", "cell_angular"], self.DOUBLE, {"units": "degree"}, 24, True),
        ]
        gatts = {"Conventions": "AMBER", "ConventionVersion": "1.0", "program": "pgm_jax", "programVersion": "0.2"}

        def header(begins: Sequence[int]) -> bytes:
            """Return the header bytes for the given begin offsets of the variables."""
            h = b"CDF\x02" + struct.pack(">i", self.nframes)
            h += struct.pack(">ii", self.NC_DIM, len(dims)) + b"".join(
                self._name(k) + struct.pack(">i", v) for k, v in dims
            )
            h += self._atts(gatts)
            h += struct.pack(">ii", self.NC_VAR, len(vars_))
            for (name, vd, t, att, size, _), beg in zip(vars_, begins):
                h += self._name(name) + struct.pack(">i", len(vd)) + b"".join(struct.pack(">i", D[d]) for d in vd)
                h += self._atts(att) + struct.pack(">ii", t, size) + struct.pack(">q", beg)
            return h

        hlen = len(header([0] * len(vars_)))  # the header length does not depend on the offsets
        begins, off = [], hlen
        for v in vars_:
            if not v[5]:
                begins.append(off)
                off += v[4]
        rec = off
        for v in vars_:
            if v[5]:
                begins.append(rec)
                rec += v[4]
        assert rec - off == self.recsize
        with open(self.path, "wb") as fh:
            fh.write(header(begins))
            fh.write(b"xyz\0" + b"abc\0" + b"alphabeta gamma\0")  # fixed-size variables, padded to 4 bytes

    def write(self, time_ps: float, xyz_A: ArrayLike, H_A: ArrayLike) -> None:
        """Append one frame and update the record count.

        Parameters
        ----------
        time_ps : float
            Time [ps].
        xyz_A : ArrayLike (N, 3)
            Coordinates [Angstrom] (stored as float32).
        H_A : ArrayLike (3, 3)
            Box, lattice vectors as rows [Angstrom] (stored as cell lengths and angles).
        """
        L, A = cell_parameters(H_A)
        rec = (
            struct.pack(">f", float(time_ps))
            + np.asarray(xyz_A, ">f4").reshape(-1).tobytes()
            + np.asarray(L, ">f8").tobytes()
            + np.asarray(A, ">f8").tobytes()
        )
        assert len(rec) == self.recsize
        with open(self.path, "r+b") as fh:
            fh.seek(0, 2)
            fh.write(rec)
            self.nframes += 1
            fh.seek(4)
            fh.write(struct.pack(">i", self.nframes))
