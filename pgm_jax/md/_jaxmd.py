"""JAX-MD's simulation core, imported without executing `jax_md/__init__.py`.

Provides the JAX-MD submodules space, partition, simulate, rigid_body, quantity, dataclasses and
util.  `jax_md/__init__.py` also imports JAX-MD's machine-learning and force-field modules (flax,
e3nn, ...); none of those are needed here, and flax releases can lag behind JAX (flax 0.12.9 does
not import with JAX 0.11.2).  So, unless `jax_md` is already in `sys.modules` (imported by the
user, in which case that module is used), an empty package module named `jax_md` with the real
package's search path is registered first, and only the submodules listed above are imported.

    from ._jaxmd import rigid_body, space

Raises ImportError at import time if jax-md is not installed.
"""

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
