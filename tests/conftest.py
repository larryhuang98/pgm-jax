"""pytest configuration shared by all tests.

pgm_jax is validated in float64: 64-bit JAX types are enabled here, before any test module (and
so before pgm_jax) is imported."""

import jax

jax.config.update("jax_enable_x64", True)
