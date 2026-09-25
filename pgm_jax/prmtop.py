"""Amber prmtop files as ordered raw sections: every %FLAG with its %FORMAT and %COMMENT lines is
read, can be read typed, changed, added or removed, and is written back in Amber's fixed-width
format.  Sections this module does not interpret (pGM's POL_GAUSS_*, CMAP grids, ...) are kept as
they are, so a prmtop can be edited without knowing all of it (bonded/amber.py export).

    top = Prmtop.read("protein.prmtop")
    k = top.get("BOND_FORCE_CONSTANT")               # numpy array
    top.set("BOND_FORCE_CONSTANT", k * 1.1)
    top.write("scaled.prmtop")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

POINTER_NAMES = ("NATOM", "NTYPES", "NBONH", "MBONA", "NTHETH", "MTHETA", "NPHIH", "MPHIA", "NHPARM", "NPARM",
                 "NNB", "NRES", "NBONA", "NTHETA", "NPHIA", "NUMBND", "NUMANG", "NPTRA", "NATYP", "NPHB",
                 "IFPERT", "NBPER", "NGPER", "NDPER", "MBPER", "MGPER", "MDPER", "IFBOX", "NMXRS", "IFCAP",
                 "NUMEXTRA", "NCOPY")


def parse_format(fmt: str):
    """'10I8' -> (10, 'I', 8, None); '5E16.8' -> (5, 'E', 16, 8); '20a4' -> (20, 'a', 4, None)."""
    m = re.fullmatch(r"\s*(\d*)\s*([aAiIeEfF])(\d+)(?:\.(\d+))?\s*", fmt)
    if not m:
        raise ValueError(f"unsupported prmtop format {fmt!r}")
    kind = "a" if m.group(2) in "aA" else m.group(2).upper()
    return int(m.group(1) or 1), kind, int(m.group(3)), (int(m.group(4)) if m.group(4) else None)


@dataclass
class Section:
    name: str
    fmt: str                                   # e.g. "10I8"
    values: list
    comments: list = field(default_factory=list)


class Prmtop:
    def __init__(self, version: str, sections: list[Section]):
        self.version = version
        self.sections = {s.name: s for s in sections}

    # ------------------------------------------------------------------ reading
    @classmethod
    def read(cls, path: str) -> "Prmtop":
        version, secs, cur, raw = "", [], None, []

        def finish():
            if cur is not None:
                cur.values = cls._parse(cur.fmt, raw)
                secs.append(cur)

        with open(path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("%VERSION"):
                    version = line
                elif line.startswith("%FLAG"):
                    finish()
                    cur, raw = Section(line.split()[1], "", []), []
                elif line.startswith("%FORMAT"):
                    cur.fmt = re.search(r"\((.*)\)", line).group(1)
                elif line.startswith("%COMMENT"):
                    if cur is not None:
                        cur.comments.append(line[len("%COMMENT"):])      # verbatim
                elif cur is not None:
                    raw.append(line)
        finish()
        return cls(version, secs)

    @staticmethod
    def _parse(fmt: str, lines: list[str]) -> list:
        count, kind, width, _ = parse_format(fmt)
        out = []
        for ln in lines:
            if kind == "a":
                for s in range(0, len(ln), width):
                    out.append(ln[s:s + width])
                continue
            for s in range(0, len(ln), width):
                t = ln[s:s + width].strip()
                if t:
                    out.append(int(t) if kind == "I" else float(t))
        return out

    # ------------------------------------------------------------------ access
    def __contains__(self, name: str) -> bool:
        return name in self.sections

    def get(self, name: str):
        s = self.sections[name]
        kind = parse_format(s.fmt)[1]
        if kind == "a":
            return [v.strip() for v in s.values]
        return np.asarray(s.values, int if kind == "I" else float)

    def set(self, name: str, values, fmt: str | None = None, comments=None, after: str | None = None):
        """Replace a section's values (keeping its format), or add a new one (fmt required) after
        the section `after` (default: at the end)."""
        vals = list(values.tolist() if isinstance(values, np.ndarray) else values)
        if name in self.sections:
            s = self.sections[name]
            s.values = vals
            if fmt is not None:
                s.fmt = fmt
            if comments is not None:
                s.comments = ["  " + c for c in comments]
            return
        if fmt is None:
            raise ValueError(f"new section {name} needs a format")
        new = Section(name, fmt, vals, ["  " + c for c in (comments or [])])
        items = list(self.sections.items())
        pos = len(items) if after is None or after not in self.sections else [k for k, _ in items].index(after) + 1
        items.insert(pos, (name, new))
        self.sections = dict(items)

    def remove(self, name: str):
        self.sections.pop(name, None)

    @property
    def pointers(self) -> dict:
        v = self.get("POINTERS")
        return {k: int(x) for k, x in zip(POINTER_NAMES, v)}

    def set_pointers(self, **kw):
        v = self.get("POINTERS").copy()
        for k, x in kw.items():
            v[POINTER_NAMES.index(k)] = int(x)
        self.set("POINTERS", v)

    # ------------------------------------------------------------------ writing
    @staticmethod
    def _format(fmt: str, values: list) -> list[str]:
        count, kind, width, prec = parse_format(fmt)
        cells = []
        for v in values:
            if kind == "a":
                cells.append(f"{str(v):<{width}.{width}s}")
            elif kind == "I":
                cells.append(f"{int(v):>{width}d}")
            elif kind == "E":
                cells.append(f"{float(v):>{width}.{prec}E}")
            else:
                cells.append(f"{float(v):>{width}.{prec}f}")
        lines = ["".join(cells[i:i + count]) for i in range(0, len(cells), count)]
        return lines or [""]

    def write(self, path: str):
        with open(path, "w") as fh:
            fh.write((self.version or "%VERSION  VERSION_STAMP = V0001.000") + "\n")
            for s in self.sections.values():
                fh.write(f"%FLAG {s.name:<74s}\n")
                for c in s.comments:
                    fh.write(f"%COMMENT{c}\n")
                fh.write(f"%FORMAT({s.fmt})".ljust(80) + "\n")
                for ln in self._format(s.fmt, s.values):
                    fh.write(ln + "\n")
