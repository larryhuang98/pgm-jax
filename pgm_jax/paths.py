"""Locations of files outside the package: the repository's data and output directories, and
external data sets and programs (Amber builds, the pGM3P-25 files of the GVDW paper, QM data).

External locations are read from environment variables, with defaults for the development
cluster; set the variable to use another location:

    PGM_GVDW_DATA   ~/pgm-gvdw-data                  topology/rayl_512_v2.prmtop, inputs/lj/inpcrd.restrt
    AMBERHOME       ~/amber25                        cpptraj, PyRESP examples, Amber test inputs
    PGM_PMEMD_BIN   ~/ambers/pgm-larry-install/bin   pmemd.pgm, pmemd.pgm.cuda_*, cpptraj of the pGM build
    PGM_SANDER      ~/ambers/pgm-larry/build/AmberTools/src/sander/sander
    PGM_PMEMD_CPU   ~/ambers/pgm-vdw/build/src/pmemd-pgm/src/pmemd-pgm
    PGM_EPSP        ~/project/epsp                   water boxes of the dielectric study
    PGM_QMDATA      ~/project/qmdata                 raw psi4 outputs of the QM water set
    PGM_GPUBENCH    ~/pgm-exp/gpubench               4,096-water benchmark inputs
    PGM_UBQ_RUNS    ~/project/pGM-JAX/runs/protein/ubq   ubiquitin runs used by bench_shake.py

Repository paths (data/, runs/, validation/) come from repo_path()."""

from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESOURCES = {
    "gvdw_data": ("PGM_GVDW_DATA", "~/pgm-gvdw-data"),
    "amberhome": ("AMBERHOME", "~/amber25"),
    "pmemd_pgm_bin": ("PGM_PMEMD_BIN", "~/ambers/pgm-larry-install/bin"),
    "sander_pgm": ("PGM_SANDER", "~/ambers/pgm-larry/build/AmberTools/src/sander/sander"),
    "pmemd_pgm_cpu": ("PGM_PMEMD_CPU", "~/ambers/pgm-vdw/build/src/pmemd-pgm/src/pmemd-pgm"),
    "epsp": ("PGM_EPSP", "~/project/epsp"),
    "qmdata": ("PGM_QMDATA", "~/project/qmdata"),
    "gpubench": ("PGM_GPUBENCH", "~/pgm-exp/gpubench"),
    "ubq_runs": ("PGM_UBQ_RUNS", "~/project/pGM-JAX/runs/protein/ubq"),
}


def resource(name: str, *parts: str) -> str:
    """Path of an external resource.

    Parameters
    ----------
    name : str
        A key of RESOURCES (e.g. "gvdw_data").
    *parts : str
        Path components below the resource's location.

    Returns
    -------
    str
        The resource location from its environment variable (or the default), with `parts` joined.
    """
    env, default = RESOURCES[name]
    return os.path.join(os.path.expanduser(os.environ.get(env, default)), *parts)


def repo_path(*parts: str) -> str:
    """Path below the repository root (data/, runs/, validation/, ...)."""
    return os.path.join(REPO, *parts)


def pgm3p25_files() -> tuple[str, str]:
    """(prmtop, restart) of the 512-water pGM3P-25 box of the GVDW paper."""
    return resource("gvdw_data", "topology/rayl_512_v2.prmtop"), resource("gvdw_data", "inputs/lj/inpcrd.restrt")
