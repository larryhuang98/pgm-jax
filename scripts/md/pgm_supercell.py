"""Add the pGM sections to a replicated (n x n x n) Amber topology (the `pgm-jax supercell` command).

cpptraj writes the supercell coordinates and the standard prmtop sections of a replicated system,
but drops the pGM sections it does not know.  This script copies the POL_GAUSS_* sections (and
IPOL when cpptraj dropped it) from the original pGM prmtop, replicated NCOPIES times, with the atom
indices of POL_GAUSS_COVALENT_ATOMS_LIST offset by the atom count of the original per copy.  Used
for the benchmark boxes (scripts/benchmarks/bench_md.py).

Usage:

    cpptraj: parm in.prmtop / trajin in.rst7 / replicatecell out sc.rst7 parmout sc_std.prmtop dir 000 dir 100 ...
    python scripts/md/pgm_supercell.py in.prmtop sc_std.prmtop sc.prmtop 8
    python scripts/md/pgm_supercell.py --help

Inputs: the original pGM prmtop and cpptraj's replicated prmtop.
Outputs: the replicated pGM prmtop (cpptraj's file with the pGM sections appended).
Units: none (integer and float fields are copied unchanged).
Runtime: seconds.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable


def sections(path: str) -> dict[str, list[str]]:
    """Return the %FLAG sections of a prmtop: name -> its lines (the %FLAG line first)."""
    out, cur = {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("%FLAG"):
                cur = line.split()[1]
                out[cur] = [line]
            elif cur is not None:
                out[cur].append(line)
    return out


def values(block: list[str]) -> tuple[str, list[str]]:
    """Return the %FORMAT line and the fixed-width fields of a section.

    Parameters
    ----------
    block : list of str
        The section's lines as returned by `sections`.

    Returns
    -------
    fmt : str
        The %FORMAT line (with its newline).
    fields : list of str
        The non-blank fields, cut at the width given in the format (e.g. 10I8 -> 8 characters).
    """
    fmt = [line for line in block if line.startswith("%FORMAT")][0]
    data = [line.rstrip("\n") for line in block if not line.startswith("%")]
    m = re.search(r"\((\d+)([aAiIeEfF])(\d+)", fmt)
    w = int(m.group(3))
    return fmt, [ln[s : s + w] for ln in data for s in range(0, len(ln), w) if ln[s : s + w].strip()]


def write_block(
    name: str, fmt: str, vals: list, per_line: int, width: int, fmtfun: Callable[[object], str]
) -> list[str]:
    """Return the lines of a prmtop section.

    Parameters
    ----------
    name : str
        Section name (after %FLAG).
    fmt : str
        The %FORMAT line (with its newline).
    vals : list
        The values.
    per_line : int
        Values per line.
    width : int
        Field width [characters]; values are right-justified.
    fmtfun : callable
        Formats one value as a string.

    Returns
    -------
    list of str
        The %FLAG line, the %FORMAT line and the data lines (one empty line for no values).
    """
    lines = [f"%FLAG {name:<74s}\n", fmt]
    for s in range(0, len(vals), per_line):
        lines.append("".join(fmtfun(v).rjust(width) for v in vals[s : s + per_line]) + "\n")
    if not vals:
        lines.append("\n")
    return lines


def replicate_pgm_sections(src: str, std: str, dst: str, n: int) -> None:
    """Write `dst`: cpptraj's replicated prmtop `std` plus the pGM sections of `src` replicated n times.

    Parameters
    ----------
    src : str
        The original pGM prmtop.
    std : str
        cpptraj's prmtop of the supercell (without the pGM sections).
    dst : str
        Output prmtop.
    n : int
        Number of copies in the supercell.
    """
    S = sections(src)
    natom = int(values(S["POINTERS"])[1][0])
    with open(std) as fh:
        out = fh.read()
    extra = []
    for name, blk in S.items():
        if not name.startswith("POL_GAUSS") and name != "IPOL":
            continue
        if name == "IPOL" and "%FLAG IPOL" in out:
            continue
        if name == "POL_GAUSS_FORCEFIELD":
            extra += blk
            continue
        fmt, v = values(blk)
        if "I" in fmt.split("(")[1]:
            ints = [int(x) for x in v]
            if name == "POL_GAUSS_COVALENT_ATOMS_LIST":
                rep = [x + c * natom for c in range(n) for x in ints]  # atom indices of copy c
            elif name == "IPOL":
                rep = ints
            else:
                rep = ints * n
            extra += write_block(name, fmt, rep, 10, 8, lambda x: str(x))
        else:
            fl = [float(x) for x in v]
            extra += write_block(name, fmt, fl * n, 5, 16, lambda x: f"{x:.8E}")
    with open(dst, "w") as fh:
        fh.write(out + "".join(extra))


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and write the replicated pGM prmtop (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="original pGM prmtop")
    ap.add_argument("std", help="cpptraj's replicated prmtop (parmout)")
    ap.add_argument("dst", help="output prmtop")
    ap.add_argument("copies", type=int, help="number of copies in the supercell (e.g. 8 for 2 x 2 x 2)")
    a = ap.parse_args(argv)
    replicate_pgm_sections(a.src, a.std, a.dst, a.copies)


if __name__ == "__main__":
    main()
