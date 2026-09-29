"""Command-line support: shared argparse option groups (args.py) and the `pgm-jax` entry point (main.py).

`pgm-jax` (`[project.scripts]` in pyproject.toml, or `python -m pgm_jax.cli`) dispatches its
subcommands to the `main()` of the repository's scripts; the scripts build their options from the
groups of args.py, so that one option has one name and unit everywhere (docs/api_design.md, D2).
"""
