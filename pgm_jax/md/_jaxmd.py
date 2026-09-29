"""Import JAX-MD's simulation core (space, partition, simulate, rigid_body, quantity, dataclasses,
util) without executing `jax_md/__init__.py`, which also imports its machine-learning and
force-field modules (flax, e3nn, ...).  None of those are needed here, and flax releases can lag
behind JAX (flax 0.12.9 does not import with JAX 0.11.2).  If `jax_md` was already imported by
the user, that module is used."""

from __future__ import annotations

import importlib.util
import sys
import types

if "jax_md" not in sys.modules:
    _spec = importlib.util.find_spec("jax_md")
    if _spec is None:
        raise ImportError("jax-md is not installed (pip install jax-md)")
    _pkg = types.ModuleType("jax_md")
    _pkg.__path__ = list(_spec.submodule_search_locations)
    _pkg.__spec__ = _spec
    _pkg.__file__ = _spec.origin
    sys.modules["jax_md"] = _pkg

from jax_md import dataclasses, partition, quantity, rigid_body, simulate, space, util  # noqa: E402

__all__ = ["dataclasses", "partition", "quantity", "rigid_body", "simulate", "space", "util"]
