"""Write the output files of biased simulations and read them back.

Contents: `BiasOutput` (the COLVAR and HILLS writers of a running simulation) and
`read_table` (reader of both, with continuation segments).

    prefix.colvar   every `colvar` steps: step, time (ps), the CVs of each bias, the energy of each
                    bias (kJ/mol) at that configuration (the bias acting on that step, before any
                    update at the same step)
    prefix.hills    metadynamics hills as deposited: step, time (ps), centre (d), sigma (d), height
                    (kJ/mol) (prefix.hills1, ... for further metadynamics biases)
    prefix.bias     the complete bias state (hills, OPES kernels and normalisation; BiasSet.save),
                    written with the restarts; Simulation.load_bias reads it

The text files start with "# key = value" header lines (the last one "# columns = ...") and have
one whitespace-separated row per record.

    meta, c = read_table("md.colvar")          # c["step"], c["time_ps"], c["phi"], c["bias0_metad"]

Units: time ps, energies kJ/mol, CVs in their own units (nm, rad).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .core import MetaD

if TYPE_CHECKING:
    from .core import BiasSet, BiasState


def _header(meta: dict, columns: Sequence[str]) -> str:
    """Return the "# key = value" header lines of `meta`, then "# columns = ..." (newline-terminated)."""
    lines = [f"{k} = {v}" for k, v in meta.items()] + ["columns = " + " ".join(columns)]
    return "".join(f"# {s}\n" for s in lines)


class BiasOutput:
    """COLVAR and HILLS files of a running simulation (driver side).

    The drivers create one per run (or walker) and call `write` at the end of every block with the
    drained COLVAR rows and the current bias state; only hills not yet written are appended.

    Attributes
    ----------
    bs : BiasSet
        The biases.
    dt : float
        Time step [ps] (time column = step * dt).
    colvar_path : str
        prefix.colvar.
    colvar : bool
        Whether COLVAR rows are written.
    hills : list of tuple
        (bias index, path, hills already written) per metadynamics bias with a HILLS file.
    """

    def __init__(
        self,
        bias_set: BiasSet,
        prefix: str,
        dt: float,
        temperature: float,
        append: bool = False,
        state: BiasState | None = None,
        colvar: bool = True,
        hills: bool = True,
    ) -> None:
        """Open (create or continue) the COLVAR and HILLS files and write their headers.

        Parameters
        ----------
        bias_set : BiasSet
            The biases of the simulation.
        prefix : str
            Path prefix of the files.
        dt : float
            Time step [ps].
        temperature : float
            Simulation temperature [K] (header only).
        append : bool
            Continue existing files (a restart): no new header, and hills up to the count in `state`
            count as written.
        state : BiasState, optional
            Current bias state (with `append`: the hills already in the files); None: none written.
        colvar : bool
            Write prefix.colvar (only if the set has colvar > 0).
        hills : bool
            Write prefix.hills for the metadynamics biases.
        """
        self.bs, self.dt = bias_set, float(dt)
        self.colvar_path = prefix + ".colvar"
        meta = {
            "pgm_jax": "COLVAR (pgm_jax.bias)",
            "dt_ps": self.dt,
            "temperature_K": temperature,
            "biases": "; ".join(b.describe() for b in bias_set.biases),
        }
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
                    cols = (
                        ["step", "time_ps"]
                        + [f"c_{n}" for n in b.cvs.names]
                        + [f"sigma_{n}" for n in b.cvs.names]
                        + ["height"]
                    )
                    with open(path, "w") as fh:
                        fh.write(
                            _header(
                                {
                                    "pgm_jax": "HILLS (pgm_jax.bias)",
                                    "biasfactor": b.biasfactor,
                                    "temperature_K": b.temperature,
                                    "periods": " ".join(f"{p:g}" for p in b.cvs.periods),
                                    "bias": b.describe(),
                                },
                                cols,
                            )
                        )
                    done = 0
                self.hills.append((k, path, done))

    def write(self, rows: np.ndarray, state: BiasState) -> None:
        """Append COLVAR rows and the hills deposited since the last call.

        Parameters
        ----------
        rows : np.ndarray (n, ncol)
            COLVAR rows from `BiasSet.drain` (step, CVs, bias energies [kJ/mol]).
        state : BiasState
            Current bias state (for the hills).
        """
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
                        fh.write(
                            f"{int(h['step'][i]):12d} {h['step'][i] * self.dt:14.6f} "
                            + " ".join(f"{x:16.9e}" for x in h["center"][i])
                            + " "
                            + " ".join(f"{x:14.7e}" for x in h["sigma"])
                            + f" {h['height'][i]:16.9e}\n"
                        )
            new.append((k, path, n))
        self.hills = new


def read_table(path: str | os.PathLike | Sequence[str | os.PathLike]) -> tuple[dict, dict]:
    """Read a COLVAR or HILLS file, or a list of continuation segments, into header and columns.

    Rows superseded by a continuation from an earlier checkpoint are dropped: where the step goes
    back, the rows of earlier segments at that step or later are removed.  Equal steps in a row
    (walkers) are kept.

    Parameters
    ----------
    path : str, os.PathLike, or sequence of them
        File(s), in order.

    Returns
    -------
    meta : dict
        Header entries (str -> str) of the first file that has each key.
    columns : dict
        Column name -> np.ndarray (rows,).
    """
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
    if len(x):  # continuation segments start where the step goes back
        step = x[:, 0]
        starts = np.append(False, step[1:] < step[:-1])
        s_at = np.where(starts, step, np.inf)  # first step of each continuation segment
        later = np.append(np.minimum.accumulate(s_at[::-1])[::-1][1:], np.inf)  # min over starts after i
        x = x[step < later]
    return meta, {c: x[:, k] for k, c in enumerate(cols)}
