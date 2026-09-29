"""Host-side machinery shared by the MD drivers (Simulation, FlexibleSimulation, PIMDSimulation,
the replica drivers of REMD, bias walkers, finite-field replicas and lambda windows).

Nothing here is traced by JAX: the compiled steps of the integrators are untouched.  The drivers
use these pieces for

  * blocks of steps: `block_length` (the largest block that hits every output interval) and
    `advance_with_rebuilds` / `retry_block` (repeat a block after an overflow of the neighbour
    list or of the pair rows, rebuild the lists when the box volume has drifted by 10 %);
  * tables of observables: `LogTable` (one header, fixed-width columns) and `ns_per_day`;
  * checkpoints: `write_checkpoint` / `read_checkpoint` store arrays in an ``.npz`` file whose
    ``__header__`` entry is a JSON header (format, version, kind, metadata); pytrees such as
    ``MDState`` are stored leaf by leaf under their key paths (`tree_arrays`) and rebuilt on a
    template of the same structure (`tree_from_arrays`).  Checkpoints of pgm_jax up to commit
    e72c57c were pickles; `read_legacy_checkpoint` reads them, and loading one and saving again
    converts it to the new format.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "pgm_jax checkpoint"
CHECKPOINT_VERSION = 1
HEADER_KEY = "__header__"
_ZIP_MAGIC = b"PK\x03\x04"


# ----------------------------------------------------------------------------- blocks of steps
def block_length(nsteps: int, *intervals: int) -> int:
    """Number of steps per compiled block.

    Parameters
    ----------
    nsteps : int
        Steps of the whole run.
    *intervals : int
        Output intervals in steps (report, trajectory, checkpoint, ...); zeros are ignored.

    Returns
    -------
    int
        The greatest common divisor of nsteps and the non-zero intervals, so that every output
        falls on the end of a block.
    """
    return int(np.gcd.reduce([int(x) for x in (nsteps, *intervals) if int(x) > 0]))


def ns_per_day(steps: int, dt_ps: float, seconds: float) -> float:
    """Simulated time per wall-clock day [ns/day] of `steps` steps of `dt_ps` [ps] in `seconds` [s]."""
    return steps * dt_ps / 1000.0 / max(seconds, 1e-9) * 86400.0


class Stopwatch:
    """Wall-clock time and step count since the start of a run, for ns/day in the logs."""

    def __init__(self, step0: int, dt_ps: float):
        """Start timing at step `step0` of a run with time step `dt_ps` [ps]."""
        self.t0, self.step0, self.dt = time.time(), int(step0), float(dt_ps)

    def ns_per_day(self, step: int) -> float:
        """Speed [ns/day] from the start to `step`."""
        return ns_per_day(int(step) - self.step0, self.dt, time.time() - self.t0)

    def seconds(self) -> float:
        """Wall-clock seconds since the start."""
        return time.time() - self.t0


def retry_block(run: Callable, start, failed: Callable, resize: Callable, attempts: int = 6):
    """Run a block of steps, re-sizing the static capacities and repeating it after an overflow.

    Parameters
    ----------
    run : callable
        ``run(start) -> new`` advances the state by one block.
    start
        State at the start of the block.
    failed : callable
        ``failed(new) -> (list_failed, rows_failed)``: whether the neighbour list or the pair rows
        overflowed in the block (the results of such a block are invalid).
    resize : callable
        ``resize(start, list_failed, rows_failed) -> start'`` enlarges the capacities (and
        recompiles) and returns the start state with forces evaluated at the new sizes.
    attempts : int
        Number of tries before giving up.

    Returns
    -------
    The state after the block.

    Raises
    ------
    RuntimeError
        "neighbour list keeps overflowing" after `attempts` tries (callers split the block).
    """
    for _ in range(attempts):
        new = run(start)
        list_bad, rows_bad = failed(new)
        if not (list_bad or rows_bad):
            return new
        start = resize(start, list_bad, rows_bad)
    raise RuntimeError("neighbour list keeps overflowing")


def advance_with_rebuilds(n: int, advance_block: Callable, rebuild: Callable, volume_ratio: Callable) -> None:
    """Advance n steps, rebuilding neighbour lists for a changed box and splitting stuck blocks.

    JAX-MD's cell lists are laid out for one box shape: they are rebuilt when the volume differs
    by more than 10 % from the one they were built for (NPT from a loose start).  A block that
    keeps overflowing (the box shrank within it) is split in halves with a rebuild in between.

    Parameters
    ----------
    n : int
        Steps.
    advance_block : callable
        ``advance_block(n)`` runs n steps as one block (raises RuntimeError "... overflowing").
    rebuild : callable
        ``rebuild()`` rebuilds the lists for the current box.
    volume_ratio : callable
        ``volume_ratio()`` = current volume / volume the lists were built for.
    """
    if abs(volume_ratio() - 1.0) > 0.10:
        rebuild()
    try:
        advance_block(n)
    except RuntimeError as err:
        if "overflowing" not in str(err) or n < 2:
            raise
        rebuild()
        advance_with_rebuilds(n // 2, advance_block, rebuild, volume_ratio)
        advance_with_rebuilds(n - n // 2, advance_block, rebuild, volume_ratio)


# ----------------------------------------------------------------------------- tables
def format_row(values: Iterable) -> str:
    """One table row: floats as %14.6f, integers as %14d, anything else right-aligned in 14."""
    out = []
    for v in values:
        if isinstance(v, (bool, np.bool_)):
            out.append(f"{int(v):14d}")
        elif isinstance(v, (float, np.floating)):
            out.append(f"{float(v):14.6f}")
        elif isinstance(v, (int, np.integer)):
            out.append(f"{int(v):14d}")
        else:
            out.append(f"{v!s:>14s}")
    return "  " + " ".join(out)


def format_header(columns: Iterable[str]) -> str:
    """The header line of a table: ``#`` and the column names right-aligned in 14 characters."""
    return "# " + " ".join(f"{c:>14s}" for c in columns)


class LogTable:
    """A table of observables in a text file: a header once, then one fixed-width row per report.

    The columns are the keys of the first row written.  A new file (or an empty one when
    appending) gets the optional title lines and the header; appending to a non-empty file adds
    rows only.  Rows can be echoed to a text stream (e.g. sys.stdout) as well.
    """

    def __init__(self, path: str, append: bool = False, title: Iterable[str] = (), echo=None):
        """Open `path` for writing ('a' with `append`); `title`: lines written above the header
        (without the leading '#'); `echo`: a text stream that receives the header and rows too."""
        self.path, self.echo = path, echo
        self.fh = open(path, "a" if append else "w")
        self._needs_header = self.fh.tell() == 0
        self.title = [str(t) for t in title]
        self.columns: list[str] | None = None

    def write(self, row: dict) -> str:
        """Append one row (a dict: column -> int or float) and return its text."""
        if self.columns is None:
            self.columns = list(row)
            header = format_header(self.columns)
            if self._needs_header:
                self.fh.write("".join(f"# {t}\n" for t in self.title) + header + "\n")
            self._echo(header)
        line = format_row(row[c] for c in self.columns)
        self.fh.write(line + "\n")
        self.fh.flush()
        self._echo(line)
        return line

    def _echo(self, text: str) -> None:
        """Copy a line to the echo stream, if any."""
        if self.echo is not None:
            print(text, file=self.echo, flush=True)

    def close(self) -> None:
        """Close the file."""
        self.fh.close()

    def __enter__(self) -> LogTable:
        """Context-manager entry: the table itself."""
        return self

    def __exit__(self, *exc) -> None:
        """Context-manager exit: close the file."""
        self.close()


# ----------------------------------------------------------------------------- checkpoints
def to_jsonable(x):
    """A JSON-serializable copy of x: dicts, lists and tuples recursively, numpy arrays and scalars
    as Python lists and numbers (NaN and infinities are kept: Python's json writes them)."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return to_jsonable(np.asarray(x).tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def tree_arrays(tree, name: str) -> dict[str, np.ndarray]:
    """The leaves of a pytree as host arrays, keyed ``name`` + the leaf's key path.

    Parameters
    ----------
    tree
        A pytree of arrays (an MDState, a dict of arrays, a list of states, ...); None subtrees
        have no leaves.
    name : str
        Prefix of the keys (the name of the tree in the checkpoint).

    Returns
    -------
    dict
        ``{name + keystr(path): np.ndarray}``.
    """
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {name + jax.tree_util.keystr(p): np.asarray(v) for p, v in leaves}


def tree_from_arrays(template, arrays: dict, name: str, optional: Iterable[str] = ()):
    """Rebuild a pytree from `tree_arrays` output on a template of the same structure.

    Parameters
    ----------
    template
        A pytree with the structure of the stored one (e.g. the driver's current state); its
        leaves are replaced.
    arrays : dict
        Stored arrays (all trees of a checkpoint).
    name : str
        The tree's name in the checkpoint.
    optional : iterable of str
        Key-path prefixes (e.g. ".bias") whose leaves may be missing from the checkpoint (the
        template's leaves are kept) or present in it without a place in the template (ignored):
        parts of the state that depend on options of the run (a bias added or removed).

    Returns
    -------
    The template's structure with the stored leaves as device arrays.

    Raises
    ------
    ValueError
        When leaves that are not optional are missing or extra (a different system or options).
    """
    optional = tuple(optional)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    out, used, missing = [], set(), []
    for p, leaf in leaves:
        path = jax.tree_util.keystr(p)
        key = name + path
        if key in arrays:
            out.append(jnp.asarray(arrays[key]))
            used.add(key)
        elif path.startswith(optional):
            out.append(leaf)
        else:
            missing.append(path)
    extra = [
        k[len(name) :]
        for k in arrays
        if k.startswith(name)
        and k not in used
        and k[len(name) :][:1] in (".", "[")
        and not k[len(name) :].startswith(optional)
    ]
    if missing or extra:
        raise ValueError(
            f"checkpoint tree {name!r} does not match this run (other system or options): "
            f"missing {missing[:5]}, unexpected {extra[:5]}"
        )
    return jax.tree_util.tree_unflatten(treedef, out)


def write_checkpoint(path: str, kind: str, meta: dict, trees: dict, arrays: dict | None = None) -> None:
    """Write a checkpoint: an ``.npz`` file with a JSON header and the arrays of some pytrees.

    Parameters
    ----------
    path : str
        File name (written as given, atomically through a temporary file).
    kind : str
        What wrote it ("simulation", "pimd", "remd", ...); checked when reading.
    meta : dict
        JSON-serializable metadata (step, time, random-number state, settings to check).
    trees : dict
        ``{name: pytree}`` stored leaf by leaf (tree_arrays).
    arrays : dict, optional
        Further named arrays stored as they are.
    """
    header = {"format": CHECKPOINT_FORMAT, "version": CHECKPOINT_VERSION, "kind": kind, "meta": to_jsonable(meta)}
    data = {HEADER_KEY: np.array(json.dumps(header))}
    for name, tree in trees.items():
        data.update(tree_arrays(tree, name))
    for k, v in (arrays or {}).items():
        data[k] = np.asarray(v)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **data)
    os.replace(tmp, path)


def is_legacy_checkpoint(path: str) -> bool:
    """True for a pickle checkpoint of pgm_jax up to commit e72c57c (not an npz archive)."""
    with open(path, "rb") as fh:
        return fh.read(4) != _ZIP_MAGIC


def read_checkpoint(path: str, kind: str) -> tuple[dict, dict]:
    """Read a checkpoint written by `write_checkpoint`.

    Parameters
    ----------
    path : str
        The file.
    kind : str
        The expected kind.

    Returns
    -------
    meta : dict
        The metadata.
    arrays : dict
        All stored arrays (numpy), keyed as written.

    Raises
    ------
    ValueError
        Not a pgm_jax checkpoint, a newer format version, or another kind.
    """
    if is_legacy_checkpoint(path):
        raise ValueError(f"{path}: a legacy (pickle) checkpoint; read it with read_legacy_checkpoint")
    with np.load(path, allow_pickle=False) as z:
        if HEADER_KEY not in z.files:
            raise ValueError(f"{path}: not a pgm_jax checkpoint (no header)")
        header = json.loads(str(z[HEADER_KEY]))
        arrays = {k: z[k] for k in z.files if k != HEADER_KEY}
    if header.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a pgm_jax checkpoint (format {header.get('format')!r})")
    if int(header.get("version", 0)) > CHECKPOINT_VERSION:
        raise ValueError(
            f"{path}: checkpoint version {header['version']} is newer than this code ({CHECKPOINT_VERSION})"
        )
    if header.get("kind") != kind:
        raise ValueError(f"{path}: a {header.get('kind')!r} checkpoint, not a {kind!r} one")
    return header["meta"], arrays


def read_legacy_checkpoint(path: str):
    """The content of a pickle checkpoint of pgm_jax up to commit e72c57c.

    Such files hold the pickled state objects (``MDState`` and friends, numpy leaves), so they can
    only be read while those classes keep their import paths; load them into a driver and save
    again to convert them to the current format.

    Parameters
    ----------
    path : str
        The ``.chk`` / ``.pimd.chk`` / ``.remd.chk`` / ``.fe.chk`` / ``.ffchk`` / ``.walkers.chk`` file.

    Returns
    -------
    The unpickled object (a dict).
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)


def host_tree(tree):
    """A pytree with every leaf copied to the host as a numpy array."""
    return jax.tree_util.tree_map(np.asarray, tree)


def device_tree(tree):
    """A pytree with every leaf as a device array (legacy checkpoints hold numpy leaves)."""
    return jax.tree_util.tree_map(jnp.asarray, tree)


def split_scalars(d: dict) -> tuple[dict, dict]:
    """Split a flat dict into (JSON-able scalars and lists, numpy arrays) for a checkpoint."""
    arrays = {k: np.asarray(v) for k, v in d.items() if isinstance(v, (np.ndarray, jnp.ndarray))}
    rest = {k: v for k, v in d.items() if k not in arrays}
    return rest, arrays


def finite_or_raise(value: float, step: int) -> None:
    """Raise FloatingPointError when an energy is not finite (a crashed trajectory)."""
    if not math.isfinite(float(value)):
        raise FloatingPointError(f"energy is not finite at step {int(step)}")
