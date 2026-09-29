"""The `pgm-jax` command: one entry point for the main scripts of the repository.

`pgm-jax <subcommand> [options]` runs the `main()` of the script that implements the
subcommand (SUBCOMMANDS: the script's path below scripts/ and a one-line summary);
`pgm-jax --help` lists the subcommands, `pgm-jax <subcommand> --help` the script's options.
Validation, benchmark and study scripts are not subcommands; run them as
`python scripts/<folder>/<name>.py`.

    pgm-jax md -p water.prmtop -c water.rst7 -o runs/md --nsteps 1000
    pgm-jax dielectric runs/md.dip --temperature-K 298
    python -m pgm_jax.cli --help                     # the same without the console script

The scripts live in the repository (scripts/), not in the installed package, so the command
needs the source tree: an editable install (`pip install -e .`) or PYTHONPATH pointing at the
repository.  A subcommand runs its script as `python scripts/<folder>/<name>.py` would: the
script's folder is first on sys.path while it runs (the scripts of one folder may share helpers by
importing each other).

Units: those of each script's options (docs/api_design.md, D2: the unit is in the option name).
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from collections.abc import Iterator, Sequence
from types import ModuleType

from ..paths import repo_path

#: subcommand -> (script path below scripts/, one-line summary)
SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "md": ("md/run_md.py", "MD of a pGM system from Amber inputs (Amber-style options: Angstrom, --temp, ...)"),
    "pimd": ("pimd/pimd_water.py", "path-integral MD of flexible pGM water (template, run, analysis)"),
    "remd": ("protein/remd_peptide.py", "temperature replica exchange of a solvated peptide"),
    "solvation": ("free_energy/solvation_free_energy.py", "hydration free energy by alchemical lambda windows"),
    "finite-field": ("dielectric/finite_field.py", "finite-field static dielectric constant of a box"),
    "dielectric": ("dielectric/dielectric.py", "static dielectric constant (and IR spectrum) from dipole series"),
    "trajectory-dipoles": ("dielectric/trajectory_dipoles.py", "cell-dipole series of an Amber NetCDF trajectory"),
    "fit-liquid": ("fitting/fit_liquid.py", "fit Lennard-Jones parameters to liquid density and heat of vaporization"),
    "fit-multi": ("fitting/fit_multi.py", "fit a rigid pGM liquid to several targets (density, dHvap, eps, ...)"),
    "analyze-fit": ("fitting/liquid_fit_tools.py", "analysis of liquid-fit runs (segments, checks, summaries)"),
    "fit-qm": ("qm/fit_water_qm.py", "fit pGM water parameters to the QM cluster set"),
    "write-prmtop": ("protein/write_pgm_prmtop.py", "write the MD engine's model as a pmemd-pgm prmtop and mdin"),
    "build-amber": ("protein/build_amber.py", "PDB or sequence -> solvated Amber topology (tleap)"),
    "supercell": ("md/pgm_supercell.py", "replicate a pGM system n x n x n"),
    "bench": ("benchmarks/bench_md.py", "MD speed benchmark (pGM3P-25 water)"),
}


def scripts_dir() -> str:
    """Return the repository's scripts/ directory (absolute path)."""
    return repo_path("scripts")


def script_path(subcommand: str) -> str:
    """Return the absolute path of the script behind a subcommand.

    Raises
    ------
    KeyError
        An unknown subcommand.
    """
    return os.path.join(scripts_dir(), SUBCOMMANDS[subcommand][0])


@contextlib.contextmanager
def script_context(path: str, prog: str | None = None) -> Iterator[None]:
    """Run code as if in `python <path>`: the script's folder first on sys.path, sys.argv[0] = prog.

    Parameters
    ----------
    path : str
        The script file.
    prog : str, optional
        Program name for the script's argparse usage line (None: the script's file name).
    """
    folder = os.path.dirname(os.path.abspath(path))
    argv0 = sys.argv[0] if sys.argv else ""
    sys.path.insert(0, folder)
    if sys.argv:
        sys.argv[0] = prog or path
    else:
        sys.argv.append(prog or path)
    try:
        yield
    finally:
        sys.argv[0] = argv0
        with contextlib.suppress(ValueError):
            sys.path.remove(folder)


def load_script(path: str) -> ModuleType:
    """Import a script file as a module (its `__main__` block does not run).

    Call it inside `script_context(path)` when the script imports helpers from its own folder.

    Parameters
    ----------
    path : str
        The script file.

    Returns
    -------
    module
        The imported module, entered in sys.modules as "pgm_jax_script_<file name>" (dataclasses
        and pickling look a class's module up there); a later load of the same file replaces it.

    Raises
    ------
    FileNotFoundError
        The file does not exist (e.g. a non-editable install without the repository).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path}: script not found; pgm-jax runs the scripts of the repository (install with `pip install -e .`)"
        )
    name = "pgm_jax_script_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def run_script(path: str, argv: Sequence[str], prog: str | None = None) -> object:
    """Import a script and call its `main(argv)`.

    Parameters
    ----------
    path : str
        The script file.
    argv : sequence of str
        Command-line arguments for the script (without the program name).
    prog : str, optional
        Program name shown in the script's usage line.

    Returns
    -------
    object
        What the script's main returns (usually None).
    """
    with script_context(path, prog):
        module = load_script(path)
        return module.main(list(argv))


def _usage() -> str:
    """Return the help text of `pgm-jax` (subcommands and their summaries)."""
    width = max(len(k) for k in SUBCOMMANDS)
    lines = [
        "usage: pgm-jax <subcommand> [options]    (pgm-jax <subcommand> --help: the options)",
        "",
        "pGM-JAX command-line tools.  Each subcommand runs a script of the repository:",
        "",
    ]
    for name, (_, summary) in SUBCOMMANDS.items():
        lines.append(f"  {name:<{width}}  {summary}")
    lines += [
        "",
        "The scripts are listed in pgm_jax/cli/main.py (SUBCOMMANDS); validation, benchmark and study",
        "scripts are run directly: python scripts/<folder>/<name>.py --help",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run `pgm-jax <subcommand> [options]` (see the module docstring).

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name (None: sys.argv[1:]).

    Returns
    -------
    int
        Exit status: 0 after help or a finished subcommand, 2 for an unknown subcommand.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in SUBCOMMANDS:
        print(f"pgm-jax: unknown subcommand {name!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    run_script(script_path(name), rest, prog=f"pgm-jax {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
