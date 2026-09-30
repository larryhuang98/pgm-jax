"""pytest configuration shared by all tests.

pgm_jax is validated in float64: 64-bit JAX types are enabled here, before any test module (and
so before pgm_jax) is imported.

Shared systems, settings and helpers are plain functions in tests/_systems.py (imported by the
test modules; test modules never import each other).  The markers (slow, gpu, needs_data,
needs_external, optional_deps, regression) are registered in pyproject.toml and described in
tests/README.md.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)


def pytest_collection_modifyitems(config, items):
    """Skip the tests marked gpu when JAX has no GPU backend (JAX_PLATFORMS=cpu or a CPU-only jaxlib)."""
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if not gpu_items:
        return
    try:
        has_gpu = bool(jax.devices("gpu"))
    except RuntimeError:
        has_gpu = False
    if not has_gpu:
        skip = pytest.mark.skip(reason="needs a GPU (no JAX GPU backend)")
        for item in gpu_items:
            item.add_marker(skip)
