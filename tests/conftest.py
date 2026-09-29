"""pytest configuration shared by all tests.

pgm_jax is validated in float64: 64-bit JAX types are enabled here, before any test module (and
so before pgm_jax) is imported.

Shared systems, settings and helpers are plain functions in tests/_systems.py (imported by the
test modules; test modules never import each other).  The markers (slow, needs_data,
needs_external, optional_deps, regression) are registered in pyproject.toml and described in
tests/README.md.
"""

import jax

jax.config.update("jax_enable_x64", True)
