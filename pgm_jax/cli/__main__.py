"""`python -m pgm_jax.cli <subcommand> [options]`: the `pgm-jax` command without the console script."""

from __future__ import annotations

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
