# pGM-JAX documentation and code-structure standard

This is the standard for the documentation phases (P8: library, P9: scripts) and for all new code.
It implements decisions D5 (every function documented) and D11 (numpy docstrings, type hints,
ruff D) of `docs/api_design.md`; the naming and unit rules repeat section 3.3 of that file with
the P5-P7 results. When this guide and existing code disagree, follow this guide in new or
touched code, and do not "fix" unrelated code on the way (see section 1).

Contents: 1 rules for the documentation phases, 2 tools, 3 units and names, 4 module docstrings,
5 function and method docstrings, 6 classes, dataclasses and pytrees, 7 JAX code, 8 physics notes
and references, 9 examples, 10 inline comments, 11 type hints, 12 function size and structure,
13 scripts, 14 tests, 15 checklist.

## 1. Rules for the documentation phases (P8, P9)

Several agents work in parallel, each on its own set of files. So:

- **Documentation only.** A P8/P9 commit changes docstrings, comments, type hints and hint-only
  imports, nothing else: no renames, no moved code, no reordered operations, no changed defaults,
  no "small fixes". `check_docstrings.py --same-code <base>` must report 0 files (section 2).
  Everything else you notice (a bug, a misleading name, a function to split, dead code, a wrong
  unit) goes into your report as a list `file:line: finding`, for a later phase.
- **Only your files.** Do not touch files outside your assignment, not even their docstrings.
  If a docstring in your file must mention an object in another file, refer to it by name
  (`md/pme.py`, `PGMForceField.compute`) and do not document it.
- **Describe what the code does, not what it should do.** Read the code before writing. Units,
  shapes, defaults and conventions come from the code (and `pgm_jax/units.py`), never from
  memory or from a similar function. If the code and an existing comment disagree, the code wins
  and the finding goes into your report.
- **Keep the physics prose.** Existing module and class docstrings often hold careful physics and
  algorithm notes. Restructure them into the template, but do not drop content, equations or
  literature references. Shorten only what is repeated.
- **Harness and tests.** Documentation-only changes cannot change results, but import every
  touched module (or `python -m py_compile` it) and run the tests of the touched subpackage
  before committing.
- **Commits:** one per subpackage or script folder, message `P8 docs: pgm_jax/<subpackage>`.

## 2. Tools

| Command | What it does |
|---|---|
| `python scripts/dev/check_docstrings.py PATHS --list` | every module, class, function, method (public and private) and nested function of 6+ lines without a docstring, as `file:line` |
| `python scripts/dev/check_docstrings.py --by-dir 2` | missing / required per directory (library by subpackage, scripts by folder, tests) |
| `python scripts/dev/check_docstrings.py PATHS --fail` | exit status 1 if anything is missing (for a pytest test: `assert not find_missing(["pgm_jax/md"])`) |
| `python scripts/dev/check_docstrings.py PATHS --same-code REV` | exit status 1 if the code (docstrings, annotations, hint-only imports and comments removed) differs from git revision REV |
| `.tools/bin/ruff check --config ruff-docstrings.toml --exit-zero PATHS` | ruff D (pydocstyle, numpy convention): section layout, summary line, blank lines, D1xx presence rules for public objects |
| `.tools/bin/ruff check . && .tools/bin/ruff format --check .` | the enforced lint and format (must stay clean) |

A file is done when `check_docstrings.py FILE --fail` and the ruff D report on FILE are both
empty, `ruff check` and `ruff format --check` pass, and `--same-code` reports no change. After P8
and P9, D moves into `select` in `pyproject.toml` and a test runs `find_missing`.

## 3. Units and names

**Library units** (`pgm_jax/units.py`): nm, ps, amu, K, bar, kJ/mol, e, e nm (dipoles), nm^3
(polarizabilities, volumes), V/nm (fields), rad, g/cm^3 (densities only as results). Arguments,
attributes, state and results in library units carry **no unit suffix**: `positions`,
`velocities`, `box`, `dt`, `temperature`, `pressure`, `cutoff`, `ewald_beta` (1/nm),
`dipole_tol`.

**Any other unit is in the name**: `xyz_A`, `dt_fs`, `energy_kcal`, `dipole_D`,
`polarizability_A3`, `box_A`, `time_ns`. Amber file readers/writers say so in the name or
return a name with the suffix.

**Counts and durations**: intervals counted in steps end in `_every` (`report_every`,
`sample_every`, `MonteCarloBarostat.every`); durations in ps end in `_ps` (`equil_ps`,
`discard_ps`, `tau_ps`, `segment_ps`); a time step is `dt` (ps).

**Standard names** (use exactly these for the concepts):

| Concept | Name |
|---|---|
| positions, velocities, box (lattice vectors as rows, nm) | `positions`, `velocities`, `box`; `pos`, `vel`, `H` only as local variables and in the positional per-configuration methods of low-level kernels (`energy(pos, params, H)`) |
| system, parameters | `system` (a `System`), `params` (parameter pytree, dict by quantity), `table` (`ParamTable`), `space` (`ParameterSpace`), `theta` (fitted vector) |
| temperature(s), pressure | `temperature` [K], `temperatures`, `pressure` [bar]; `kT` [kJ/mol] and `beta` [mol/kJ] only as derived values |
| thermostats, barostat | `Langevin(friction)` [1/ps], `Bussi(tau)` [ps], `GLE`, `PILE(tau_centroid, lam)`, `MonteCarloBarostat(pressure, every)` |
| tolerances | `dipole_tol` (induced dipoles), `adjoint_tol`, `k_tol`; `tol` only inside an object that has one solver |
| copies | `beads`, `replicas`, `walkers`, `windows`; ladders `temperatures`, `lambdas`, `fields` |
| output | `prefix` (path prefix, `None` = no files), `log` (stream for table rows or `None`); diagnostics through `logging.getLogger(__name__)` (module variable `logger`) |
| randomness | `seed` (int) |

Other rules: functions are verbs (`compute_forces`, `read_prmtop`), classes nouns; booleans read
as a statement (`has_induction`, `molecular=True`); no abbreviations in public names beyond the
established physics ones (`eps`, `alpha`, `mu`, `q`, `lj`, `pme`, `cv`, `fe`); private helpers
start with `_`. Physics symbols as local variables (`U`, `F`, `M`, `mu`, `A`, `N`) are fine when
the docstring or a comment says what they are.

## 4. Module docstrings

Every module (also `__init__.py`, scripts, tests) starts with a docstring:

```python
"""One line: what the module provides, ending in a period.

What is in it: the main classes / functions and how they relate (one short paragraph or a list).

The physics or algorithm in a few paragraphs, with the equations the code implements (plain
text, section 8), the approximations made, and the conventions (sign of the virial, box rows,
which atoms are excluded, ...).

    system = System([water] * 512)                 # a short usage sketch (section 9)
    model = PeriodicModel(system, box, positions)

Units: nm, ps, kJ/mol, e (library units; name anything else).

References
----------
.. [1] A. Author, B. Author, Journal 12, 345 (2020). doi:10.xxxx/xxxxx

See also docs/<feature>.md; related modules: md/pme.py (reciprocal space).
"""
```

Keep the order: summary, contents, physics, usage, units, references, see-also. A small module
may have only the summary, contents and units.

## 5. Function and method docstrings

Every function and method, public and private, has a docstring (D5). Nested functions of 6 or
more lines need one; shorter nested functions and lambdas may rely on their name or a comment.

**Summary line**: one line (at most ~100 characters, fits on the `"""` line), imperative mood
for functions and methods ("Return", "Compute", "Build", "Solve", not "Returns" or "The ..."),
ending in a period, followed by a blank line if more follows. Properties and attributes use a
noun phrase ("Box volume [nm^3]."). The closing `"""` of a multi-line docstring is on its own
line.

**Template** (sections in this order, only those that apply):

```python
def virial_pressure(dE_deps: ArrayLike, box: ArrayLike, kinetic: float = 0.0) -> float:
    """Return the instantaneous pressure from the strain derivative of the energy.

    Extended description: what is computed and when to use it, in a few sentences.  Mention
    what the function does not do (e.g. "no kinetic part unless `kinetic` is given").

    Parameters
    ----------
    dE_deps : ArrayLike (3, 3)
        Derivative of the potential energy with respect to the strain [kJ/mol].
    box : ArrayLike (3, 3)
        Box, lattice vectors as rows [nm].
    kinetic : float
        Kinetic energy [kJ/mol] (0: potential part only).

    Returns
    -------
    float
        Pressure [bar].

    Raises
    ------
    ValueError
        If the box is singular.

    Notes
    -----
    P = (2 K - tr(dE/deps)) / (3 V), converted with BAR_PER_KJMOL_NM3 (units.py).  Physics,
    algorithm, numerics, and anything surprising (section 8).

    References
    ----------
    .. [1] ...

    Examples
    --------
    >>> virial_pressure(np.zeros((3, 3)), np.eye(3) * 2.0)
    0.0
    """
```

Rules for the sections:

- **Parameters**: every argument (not `self`/`cls`), in signature order, `name : type (shape)`
  then an indented description. Every quantity with a dimension has its **unit in brackets** at
  the end of the first sentence (`[nm]`, `[kJ/mol]`, `[1/ps]`, `[e nm]`, `[bar]`, `[K]`,
  `[steps]`); dimensionless quantities say so where it is not obvious ("dimensionless",
  "relative"). Shapes use the symbols of the module (`N` atoms, `M` molecules or parameters, `K`
  windows, `F` frames, `B` blocks, `n` fitted parameters) and are stated once in the type line:
  `positions : jax.Array (N, 3)`. Option strings list their values: `elec : {"q", "qp", "qi",
  "qpi"}`. Write `, optional` only for arguments whose default is `None`, and say what `None`
  means; do not repeat defaults visible in the signature unless their meaning needs a word.
  `*args` / `**kwargs` are documented as `**kw` with where they go.
- **Returns**: type (shape) and description with units; several values as several entries with
  names (numpy style); a dict lists its keys with shapes and units.
- **Yields** for generators; **Raises** for every exception the function raises itself (not
  those of callees), with the condition.
- **Notes**: equations, derivations, algorithm, complexity, numerical choices, invariants
  (section 8). **References** for literature. **See Also** for closely related functions.
- **Examples**: see section 9.

**Private helpers** (`_name`) may be shorter: a one-line summary is enough when the name,
signature and summary together say what goes in and comes out, with units. Give them the full
Parameters/Returns sections when an argument's meaning, shape or unit is not obvious, and a
Notes section when they hold the physics.

## 6. Classes, dataclasses and pytrees

- **Class docstring**: what the object represents, the one-paragraph physics/algorithm summary,
  a usage sketch, and an **Attributes** section for the public attributes set in `__init__`
  (name, type (shape), unit, meaning). Say whether instances are immutable and whether they are
  pytrees.
- **`__init__` docstring**: "Set up ..." / "Build ..." summary, then **Parameters** and
  **Raises** for the constructor arguments. (Constructor parameters go into `__init__`, not the
  class docstring, so that the class docstring stays short; ruff's numpy convention does not
  require an `__init__` docstring, `check_docstrings.py` does.)
- **Dataclasses** (no hand-written `__init__`): the fields are the constructor parameters, so the
  class docstring has a **Parameters** section with one entry per field (type, unit, meaning);
  `__post_init__` gets a docstring saying what it checks or normalizes. Settings groups
  (`MDSettings`, `PMESettings`, `Induction`, ...) document every field this way.
- **Magic methods** (`__call__`, `__len__`, ...) get a docstring; `__call__` a full one.
- **Pytrees** (NamedTuple / dataclass states such as `MDState`, parameter dicts): document the
  structure where the type is defined: every leaf with its shape, dtype if not float64, unit and
  meaning, and which fields are static (not traced; changing them recompiles). A function that
  takes a pytree argument refers to that definition instead of repeating it (`state : MDState`);
  a plain dict pytree is described by its keys in the Parameters entry or once in the module
  docstring.

## 7. JAX code

- **Jitted functions** (`@jax.jit`, `jax.jit(f)`, `partial(jax.jit, static_argnames=...)`):
  document the Python function as usual, then in Notes say which arguments are static (a change
  recompiles), which shapes are fixed at trace time, and whether the function is meant to be
  called under `jit` by the caller or is jitted itself. Mention host-side effects that happen only
  at trace time (prints, Python-side caches).
- **Factories of compiled functions** (`make_step(...)` returning a closure that is later
  jitted, scanned or vmapped): the factory's docstring documents the returned callable in
  Returns: its signature, the meaning and shapes of its arguments and results, and which
  settings are baked in. The inner function gets at least a one-line docstring.
- **vmap / pmap / lax.map / scan**: state which axis is batched ("batched over the leading axis
  of `positions` (B, N, 3) with jax.vmap"), what is shared, and for `scan` the carry and the per-
  step output. Name the batched shapes in Parameters (`positions : jax.Array (B, N, 3)`).
- **Differentiability**: say whether a function is differentiable (and in which arguments), and
  describe custom rules: for `jax.custom_vjp` / `custom_jvp`, the Notes give the forward result,
  the residuals saved and the math of the backward rule (e.g. the adjoint solve), with the
  tolerance that makes it approximate.
- **Precision**: note functions that need `jax_enable_x64` or that compute in float32 on purpose
  (`precision="mixed"`), and any dtype conversions.
- **Traced control flow**: comments for `jnp.where` masks that replace branches, for `lax.cond` /
  `while_loop` conditions, and for code written a certain way to keep results bitwise identical
  or to avoid recompilation (section 10).

## 8. Physics notes and references

- Write equations in plain text (ASCII or simple Unicode symbols as they appear in the module),
  one per line, indented, with every symbol defined and its unit given:
  `eps = 1 + 4 pi <alpha/V> + 4 pi KE (<M.M> - <M>.<M>) / (3 kB T <V>)`.
- State the conventions: sign of forces and virial, box vectors as rows, minimum image, which
  pairs are excluded or scaled, Gaussian widths vs radii, what "tin-foil" or "adiabatic" means
  here, and when the implementation differs from the paper or from Amber (and why).
- Say which quantity is exact and which approximate (cutoffs, tolerances, first-order
  reweighting), and what a test checks it against.
- **References**: numpy `References` section with numbered entries, `.. [1] Authors, Journal
  volume, page (year). doi:...`; cite them in the text as `[1]_`. Only cite what you have
  checked (a reference already in the code or docs, or one you can verify). Do not invent
  references, page numbers or DOIs: if you are not sure, describe the method without a citation
  and list "reference needed" in your report.
- Physics that belongs to one algorithm goes into that function's Notes; physics shared by a
  module goes into the module docstring and the functions refer to it.

## 9. Examples

- The module and class docstrings may have a short usage sketch as an indented code block in the
  description (the style used across the library); it shows the current API with library units
  and realistic argument names, and may be schematic (`...`, undefined inputs).
- A numpy **Examples** section with `>>>` lines is for small public functions whose use is not
  obvious and whose output is cheap and deterministic (unit conversions, helpers, estimators on
  tiny arrays). Examples are not executed as tests yet, so they must be simple enough to be
  obviously right; do not add examples that need a GPU, a data file or a long compile.
- Examples in docstrings follow the API changes like any call site (D1): a phase that renames an
  argument updates the examples.

## 10. Inline comments

Docstrings say what and why at the level of the function; comments explain individual lines.
Write a comment where a careful reader would otherwise stop:

- non-obvious physics or algebra in a line (`# d/deps of the reciprocal energy: E_rec(k) term`);
- units conversions and the shapes of intermediate arrays (`# (N, K, 3), K candidate partners`);
- index and ordering conventions (`# rows of the per-frame Jacobian: U, Mx, My, Mz, alpha, D`);
- code kept a certain way on purpose: bitwise reproducibility, avoiding recompilation, memory,
  numerical stability (`# do not reorder: the golden files depend on this summation order`);
- workarounds for JAX or library behaviour, with the reason;
- magic numbers (every literal other than 0, 1, 2 and obvious ones gets a name or a comment).

Do not comment what the code already says, do not leave commented-out code or change-log
comments ("fixed in ..."), and keep comments true (a stale comment is a finding for your
report). `TODO` comments name the issue: `# TODO: ...` (no names, no dates).

## 11. Type hints

- Every module has `from __future__ import annotations`. Every function and method signature is
  annotated (arguments and return; `-> None` included); private code too, where the type is
  clear.
- Arrays: `jax.Array` for JAX arrays returned or required; `np.ndarray` for host numpy arrays;
  `ArrayLike` (`from jax.typing import ArrayLike`) for inputs that accept numpy, JAX, lists or
  scalars. Shapes and units are in the docstring, not in the hints.
- Containers from `collections.abc` (`Sequence[int]`, `Mapping[str, jax.Array]`,
  `Callable[[jax.Array], jax.Array]`), `X | None` instead of `Optional[X]`,
  `Literal["q", "qp", "qi", "qpi"]` for option strings, builtin generics (`list[str]`,
  `dict[str, float]`, `tuple[float, float]`).
- Pytrees: the state class itself (`MDState`), or `dict[str, jax.Array]` for parameter dicts
  (a module may define an alias, e.g. `Params = dict[str, jax.Array]`, and use it throughout).
- Hints must not change behaviour: imports needed only for hints go under `if TYPE_CHECKING:`
  (or are from `typing`, `collections.abc`, `jax.typing`, `numpy.typing`); never add an
  annotation to a dataclass class attribute that has none (it would become a field), and never
  add a default. No runtime type checks; no mypy / pyright gate (D11).

## 12. Function size and structure

For new code and later refactoring phases (not P8/P9, which only document; record candidates in
your report):

- A function does one thing that its summary line can say without "and". Aim for bodies under
  ~50 lines; longer than ~80 lines needs a reason (one straight-line physics formula, a jitted
  kernel that must stay one function). Split setup code of long constructors into `_build_*`
  helpers.
- More than ~7 parameters: group them (a settings dataclass, a coupling object).
- One place for each piece of logic: constants in `units.py`, box helpers in `md/box.py`,
  statistics in `analysis/stats.py`, option groups in `cli/args.py`; do not copy a helper, import
  it.
- Modules have one responsibility, stated in the module docstring's first line; a module that
  needs two unrelated summary lines is a candidate for splitting.
- Nesting deeper than three levels, flags that switch a function between two behaviours, and
  functions that both compute and print are candidates for restructuring.

## 13. Scripts

Every script under `scripts/` (and `examples/`) starts with this header:

```python
"""One line: what the script does, ending in a period.

What it computes and why (a short paragraph): the method, the system, what the output is used
for, and the docs page (docs/<feature>.md).

Usage:

    python scripts/<folder>/<name>.py <subcommand> [options]          # the common case
    python scripts/<folder>/<name>.py --help                          # all options

Inputs: files read (and the environment variables that locate them, e.g. PGM_DATA, AMBERHOME).
Outputs: files written (prefix_*.npz, <name>.json in data/validation/, ...), and what is printed.
Units: of the command-line options (D2: the unit is in the option name, --dt-fs, --cutoff-nm;
scripts/run_md.py keeps Amber's units and names) and of the outputs.
Runtime: typical time and hardware (CPU / GPU, memory), and restart behaviour.
"""

from __future__ import annotations

import argparse
import logging

from pgm_jax.cli.args import setup_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ...
    a = ap.parse_args(argv)
    setup_logging()
    ...


if __name__ == "__main__":
    main()
```

- Options come from the `pgm_jax/cli/args.py` groups where one exists; every option has a help
  text with its unit. The script's own functions follow sections 5 and 12 (a `main` that parses,
  and functions that compute, write and report).
- No `sys.path` edits, no user paths (data locations through `pgm_jax.paths` / environment
  variables), `jax_enable_x64` set explicitly when the script needs it (and said in the header).
- Research and study scripts (`scripts/bonded/`, validation scripts) follow the same standard
  (D4).

## 14. Tests

- Module docstring: what the tests in the file check, against what reference (analytic result,
  finite differences, Amber, the golden files), and the tolerances' rationale if they are shared.
- Every test function: a one-line docstring stating the property checked ("PME energy and induced
  dipoles match the exact Ewald sum."); fixtures and helpers: what they build, with sizes and
  units. Add a sentence when a tolerance is not obvious (where it comes from: solver tolerance,
  float32, statistical error).
- The regression harness (`tests/regression/`) documents each case by its docstring (what is
  run and what is compared); golden files are never edited.

## 15. Checklist per file

1. Module docstring (section 4) and a docstring on every class, function and method (sections 5-7).
2. Units in every Parameters / Returns entry with a dimension; shapes; option values.
3. Physics and algorithm in Notes or the module docstring; references verified (section 8).
4. Comments where a reader would stop; stale comments reported (section 10).
5. Type hints on every signature, hint-only imports safe (section 11).
6. `check_docstrings.py FILE --fail` and `ruff check --config ruff-docstrings.toml --exit-zero
   FILE` empty; `ruff check`, `ruff format --check` clean.
7. `check_docstrings.py FILE --same-code <base>` reports no code change; the module imports;
   the subpackage's tests pass.
8. Findings (bugs, names, structure, missing references) listed in the report, not fixed.
