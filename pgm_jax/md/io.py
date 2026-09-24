"""Amber file I/O: coordinates/velocities/box in (ASCII inpcrd or NetCDF restart), NetCDF
trajectories and restarts out (AMBER convention 1.0, readable by cpptraj, VMD, MDTraj)."""
from __future__ import annotations

import os
import struct

import numpy as np
from scipy.io import netcdf_file

from ..ewald import box_matrix

AMBER_VEL = 20.455                 # Amber velocity unit: A / (1/20.455 ps)


def read_coordinates(path: str):
    """-> (xyz A (N, 3), vel A/ps (N, 3) or None, box lengths A, box angles deg or None)."""
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
        box = (np.array(v["cell_lengths"][:], float).reshape(-1)[:3], np.array(v["cell_angles"][:], float).reshape(-1)[:3]) if "cell_lengths" in v else None
        f.close()
        return xyz, vel, box
    lines = open(path).read().splitlines()
    n = int(lines[1].split()[0])
    vals = [float(l[k:k + 12]) for l in lines[2:] for k in range(0, len(l.rstrip()), 12) if l[k:k + 12].strip()]
    xyz = np.array(vals[:3 * n]).reshape(n, 3)
    rest = vals[3 * n:]
    vel = None
    if len(rest) >= 3 * n:
        vel = np.array(rest[:3 * n]).reshape(n, 3) * AMBER_VEL
        rest = rest[3 * n:]
    box = (np.array(rest[:3]), np.array(rest[3:6])) if len(rest) >= 6 else None
    return xyz, vel, box


def cell_parameters(H):
    """Box matrix (rows, A or nm) -> lengths, angles (deg)."""
    H = np.asarray(H, float)
    a, b, c = np.linalg.norm(H, axis=1)
    alpha = np.degrees(np.arccos(np.dot(H[1], H[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(H[0], H[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(H[0], H[1]) / (a * b)))
    return np.array([a, b, c]), np.array([alpha, beta, gamma])


def box_from_cell(lengths, angles):
    return box_matrix(*lengths, *angles)


def write_restart(path: str, xyz_A, vel_A_ps, H_A, time_ps: float, title: str = "pgm_jax restart"):
    f = netcdf_file(path, "w", version=2)
    f.Conventions, f.ConventionVersion, f.program, f.programVersion, f.title = "AMBERRESTART", "1.0", "pgm_jax", "0.2", title
    n = len(xyz_A)
    f.createDimension("spatial", 3); f.createDimension("atom", n); f.createDimension("cell_spatial", 3)
    f.createDimension("cell_angular", 3); f.createDimension("label", 5)
    sp = f.createVariable("spatial", "c", ("spatial",)); sp[:] = np.array(list("xyz"), "S1")
    cs = f.createVariable("cell_spatial", "c", ("cell_spatial",)); cs[:] = np.array(list("abc"), "S1")
    ca = f.createVariable("cell_angular", "c", ("cell_angular", "label"))
    ca[:] = np.array([list("alpha"), list("beta "), list("gamma")], "S1")
    t = f.createVariable("time", "d", ()); t.units = "picosecond"; t[...] = time_ps
    c = f.createVariable("coordinates", "d", ("atom", "spatial")); c.units = "angstrom"; c[:] = np.asarray(xyz_A, float)
    if vel_A_ps is not None:
        v = f.createVariable("velocities", "d", ("atom", "spatial")); v.units = "angstrom/picosecond"
        v.scale_factor = AMBER_VEL
        v[:] = np.asarray(vel_A_ps, float) / AMBER_VEL
    L, A = cell_parameters(H_A)
    cl = f.createVariable("cell_lengths", "d", ("cell_spatial",)); cl.units = "angstrom"; cl[:] = L
    cg = f.createVariable("cell_angles", "d", ("cell_angular",)); cg.units = "degree"; cg[:] = A
    f.close()


class NetCDFTrajectory:
    """Appendable Amber NetCDF trajectory (NetCDF-3, 64-bit offsets): time, coordinates (A,
    float32), cell_lengths, cell_angles.  The header is written once; each frame appends one
    record and bumps the record count, so the file is valid after every frame."""

    NC_DIM, NC_VAR, NC_ATT = 10, 11, 12
    CHAR, FLOAT, DOUBLE = 2, 5, 6

    def __init__(self, path: str, n_atoms: int, append: bool = False):
        self.path, self.n = path, int(n_atoms)
        self.recsize = 4 + 12 * self.n + 24 + 24                 # time, coordinates, cell lengths, cell angles
        if append and os.path.exists(path):
            with open(path, "rb") as fh:
                fh.seek(4)
                self.nframes = struct.unpack(">i", fh.read(4))[0]
            return
        self.nframes = 0
        self._write_header()

    @staticmethod
    def _name(s: str) -> bytes:
        b = s.encode()
        return struct.pack(">i", len(b)) + b + b"\0" * (-len(b) % 4)

    def _att(self, name, value) -> bytes:
        if isinstance(value, str):
            b = value.encode()
            return self._name(name) + struct.pack(">ii", self.CHAR, len(b)) + b + b"\0" * (-len(b) % 4)
        return self._name(name) + struct.pack(">iid", self.DOUBLE, 1, float(value))

    def _atts(self, atts: dict) -> bytes:
        if not atts:
            return b"\0" * 8
        return struct.pack(">ii", self.NC_ATT, len(atts)) + b"".join(self._att(k, v) for k, v in atts.items())

    def _write_header(self):
        n = self.n
        dims = [("frame", 0), ("spatial", 3), ("atom", n), ("cell_spatial", 3), ("cell_angular", 3), ("label", 5)]
        D = {k: i for i, (k, _) in enumerate(dims)}
        # (name, dims, type, attributes, bytes per record or total)
        vars_ = [("spatial", ["spatial"], self.CHAR, {}, 4, False),
                 ("cell_spatial", ["cell_spatial"], self.CHAR, {}, 4, False),
                 ("cell_angular", ["cell_angular", "label"], self.CHAR, {}, 16, False),
                 ("time", ["frame"], self.FLOAT, {"units": "picosecond"}, 4, True),
                 ("coordinates", ["frame", "atom", "spatial"], self.FLOAT, {"units": "angstrom"}, 12 * n, True),
                 ("cell_lengths", ["frame", "cell_spatial"], self.DOUBLE, {"units": "angstrom"}, 24, True),
                 ("cell_angles", ["frame", "cell_angular"], self.DOUBLE, {"units": "degree"}, 24, True)]
        gatts = {"Conventions": "AMBER", "ConventionVersion": "1.0", "program": "pgm_jax", "programVersion": "0.2"}

        def header(begins):
            h = b"CDF\x02" + struct.pack(">i", self.nframes)
            h += struct.pack(">ii", self.NC_DIM, len(dims)) + b"".join(self._name(k) + struct.pack(">i", v) for k, v in dims)
            h += self._atts(gatts)
            h += struct.pack(">ii", self.NC_VAR, len(vars_))
            for (name, vd, t, att, size, _), beg in zip(vars_, begins):
                h += self._name(name) + struct.pack(">i", len(vd)) + b"".join(struct.pack(">i", D[d]) for d in vd)
                h += self._atts(att) + struct.pack(">ii", t, size) + struct.pack(">q", beg)
            return h

        hlen = len(header([0] * len(vars_)))
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
            fh.write(b"xyz\0" + b"abc\0" + b"alphabeta gamma\0")

    def write(self, time_ps: float, xyz_A, H_A):
        L, A = cell_parameters(H_A)
        rec = (struct.pack(">f", float(time_ps)) + np.asarray(xyz_A, ">f4").reshape(-1).tobytes()
               + np.asarray(L, ">f8").tobytes() + np.asarray(A, ">f8").tobytes())
        assert len(rec) == self.recsize
        with open(self.path, "r+b") as fh:
            fh.seek(0, 2)
            fh.write(rec)
            self.nframes += 1
            fh.seek(4)
            fh.write(struct.pack(">i", self.nframes))
