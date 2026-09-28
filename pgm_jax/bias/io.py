"""Output files of biased simulations and their readers.

  prefix.colvar   every `colvar` steps: step, time (ps), the CVs of each bias, the energy of each
                  bias (kJ/mol) at that configuration (the bias acting on that step, before any
                  update at the same step)
  prefix.hills    metadynamics hills as deposited: step, time, centre (d), sigma (d), height
                  (prefix.hills1, ... for further metadynamics biases)
  prefix.bias     the complete bias state (hills, OPES kernels and normalisation; BiasSet.save),
                  written with the restarts; Simulation.load_bias reads it."""
from __future__ import annotations

import os

import numpy as np

from .core import MetaD


def _header(meta: dict, columns) -> str:
    lines = [f"{k} = {v}" for k, v in meta.items()] + ["columns = " + " ".join(columns)]
    return "".join(f"# {s}\n" for s in lines)


class BiasOutput:
    """COLVAR and HILLS files of a running simulation (driver side)."""

    def __init__(self, bias_set, prefix: str, dt: float, temperature: float, append: bool = False, state=None,
                 colvar: bool = True, hills: bool = True):
        self.bs, self.dt = bias_set, float(dt)
        self.colvar_path = prefix + ".colvar"
        meta = {"pgm_jax": "COLVAR (pgm_jax.bias)", "dt_ps": self.dt, "temperature_K": temperature,
                "biases": "; ".join(b.describe() for b in bias_set.biases)}
        self.colvar = bool(colvar) and bias_set.colvar > 0
        if self.colvar and not (append and os.path.exists(self.colvar_path)):
            with open(self.colvar_path, "w") as fh:
                cols = ["step", "time_ps"] + bias_set.columns()[1:]
                fh.write(_header(meta, cols))
        self.hills = []
        m = 0
        for k, b in enumerate(bias_set.biases):
            if isinstance(b, MetaD) and hills:
                path = prefix + (".hills" if m == 0 else f".hills{m}")
                m += 1
                done = 0 if state is None else int(state.parts[k].n)
                if not (append and os.path.exists(path)):
                    cols = ["step", "time_ps"] + [f"c_{n}" for n in b.cvs.names] + [f"sigma_{n}" for n in b.cvs.names] + ["height"]
                    with open(path, "w") as fh:
                        fh.write(_header({"pgm_jax": "HILLS (pgm_jax.bias)", "biasfactor": b.biasfactor,
                                          "temperature_K": b.temperature, "periods": " ".join(f"{p:g}" for p in b.cvs.periods),
                                          "bias": b.describe()}, cols))
                    done = 0
                self.hills.append((k, path, done))

    def write(self, rows, state) -> None:
        if len(rows) and self.colvar:
            with open(self.colvar_path, "a") as fh:
                for r in rows:
                    fh.write(f"{int(r[0]):12d} {r[0] * self.dt:14.6f} " + " ".join(f"{x:16.9e}" for x in r[1:]) + "\n")
        new = []
        for k, path, done in self.hills:
            b = self.bs.biases[k]
            h = b.hills(state.parts[k])
            n = len(h["height"])
            if n > done:
                with open(path, "a") as fh:
                    for i in range(done, n):
                        fh.write(f"{int(h['step'][i]):12d} {h['step'][i] * self.dt:14.6f} "
                                 + " ".join(f"{x:16.9e}" for x in h["center"][i]) + " "
                                 + " ".join(f"{x:14.7e}" for x in h["sigma"]) + f" {h['height'][i]:16.9e}\n")
            new.append((k, path, n))
        self.hills = new


def read_table(path) -> tuple[dict, dict]:
    """A COLVAR or HILLS file (or a list of continuation segments): (header, {column: array}).
    Rows superseded by a continuation from an earlier checkpoint (where the step goes back, the rows
    of earlier segments at that step or later) are dropped; equal steps in a row (walkers) are kept."""
    paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
    meta, rows, cols = {}, [], None
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                k, eq, v = line[1:].partition(" = ")
                if eq:
                    meta.setdefault(k.strip(), v.strip())
                    if k.strip() == "columns":
                        cols = v.split()
        x = np.loadtxt(p, comments="#", ndmin=2)
        rows.append(x.reshape(-1, len(cols)))
    x = np.concatenate(rows)
    if len(x):                       # continuation segments start where the step goes back
        step = x[:, 0]
        starts = np.append(False, step[1:] < step[:-1])
        s_at = np.where(starts, step, np.inf)
        later = np.append(np.minimum.accumulate(s_at[::-1])[::-1][1:], np.inf)    # min over starts after i
        x = x[step < later]
    return meta, {c: x[:, k] for k, c in enumerate(cols)}
