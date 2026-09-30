"""Documentation completeness: docstrings of the library and of the tests.

The check is scripts/dev/check_docstrings.py (find_missing: every module, class, function and method,
public or private, and every nested function of six or more lines needs a docstring;
docs/dev/style_guide.md, section 5).  The library check is an expected failure until the
documentation phase P8 is merged; the tests are complete and must stay so.  Both skip when the
checker script is not in the repository.
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "scripts", "dev", "check_docstrings.py")


def _find_missing(*paths):
    """Return find_missing(paths) of scripts/dev/check_docstrings.py for paths relative to the repository.

    Parameters
    ----------
    *paths : str
        Directories or files, relative to the repository root.

    Returns
    -------
    list
        The objects without a docstring (check_docstrings.Missing, printed as file:line: kind name).
    """
    if not os.path.exists(CHECKER):
        pytest.skip("scripts/dev/check_docstrings.py not found")
    spec = importlib.util.spec_from_file_location("check_docstrings", CHECKER)
    mod = sys.modules.get("check_docstrings")
    if mod is None:
        mod = importlib.util.module_from_spec(spec)
        # Registered before execution: dataclasses resolves the module of its classes via sys.modules.
        sys.modules["check_docstrings"] = mod
        spec.loader.exec_module(mod)
    return mod.find_missing([os.path.join(ROOT, p) for p in paths])


def _report(missing):
    """Format the first 40 missing docstrings for the assertion message."""
    lines = [str(m) for m in missing[:40]]
    if len(missing) > 40:
        lines.append(f"... and {len(missing) - 40} more")
    return f"{len(missing)} missing docstrings:\n" + "\n".join(lines)


def test_library_docstrings_complete():
    """Every module, class, function and method of pgm_jax/ has a docstring."""
    missing = _find_missing("pgm_jax")
    assert not missing, _report(missing)


def test_test_suite_docstrings_complete():
    """Every test module, test, fixture and helper in tests/ has a docstring."""
    missing = _find_missing("tests")
    assert not missing, _report(missing)
