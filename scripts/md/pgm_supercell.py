"""Replicate a pGM system n x n x n for benchmarks.

cpptraj writes the supercell coordinates and the standard prmtop sections (it drops the pGM
sections it does not know); this script adds the POL_GAUSS_* sections, replicated with atom
indices offset per copy.

    cpptraj: parm in.prmtop / trajin in.rst7 / replicatecell out sc.rst7 parmout sc_std.prmtop dir 000 dir 100 ...
    python scripts/pgm_supercell.py in.prmtop sc_std.prmtop sc.prmtop NCOPIES
"""

from __future__ import annotations

import sys


def sections(path):
    out, cur = {}, None
    for line in open(path):
        if line.startswith("%FLAG"):
            cur = line.split()[1]
            out[cur] = [line]
        elif cur is not None:
            out[cur].append(line)
    return out


def values(block):
    fmt = [l for l in block if l.startswith("%FORMAT")][0]
    data = [l.rstrip("\n") for l in block if not l.startswith("%")]
    import re

    m = re.search(r"\((\d+)([aAiIeEfF])(\d+)", fmt)
    w = int(m.group(3))
    return fmt, [ln[s : s + w] for ln in data for s in range(0, len(ln), w) if ln[s : s + w].strip()]


def write_block(name, fmt, vals, per_line, width, fmtfun):
    lines = [f"%FLAG {name:<74s}\n", fmt]
    for s in range(0, len(vals), per_line):
        lines.append("".join(fmtfun(v).rjust(width) for v in vals[s : s + per_line]) + "\n")
    if not vals:
        lines.append("\n")
    return lines


def main(src, std, dst, n):
    n = int(n)
    S = sections(src)
    natom = int(values(S["POINTERS"])[1][0])
    out = open(std).read()
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
                rep = [x + c * natom for c in range(n) for x in ints]
            elif name == "IPOL":
                rep = ints
            else:
                rep = ints * n
            extra += write_block(name, fmt, rep, 10, 8, lambda x: str(x))
        else:
            fl = [float(x) for x in v]
            extra += write_block(name, fmt, fl * n, 5, 16, lambda x: f"{x:.8E}")
    open(dst, "w").write(out + "".join(extra))


if __name__ == "__main__":
    main(*sys.argv[1:5])
