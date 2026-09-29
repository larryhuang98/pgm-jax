"""pytest entry of the regression harness (tests/regression/regress.py): opt-in, because the cases
take ~10-20 min on the CPU.  Run with

    PGM_REGRESSION=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 python -m pytest tests/regression -q

Bitwise by default; PGM_REGRESSION_RTOL / PGM_REGRESSION_ATOL set a tolerance."""

import os

import numpy as np
import pytest

if not os.environ.get("PGM_REGRESSION"):
    pytest.skip("regression harness: set PGM_REGRESSION=1", allow_module_level=True)

import regress  # noqa: E402  (sets jax_enable_x64 and the import path)
import regression_cases as C  # noqa: E402


@pytest.mark.parametrize("name", list(C.CASES))
def test_case_reproduces_golden(name):
    if not C.available(name):
        pytest.skip(f"needs {C.CASES[name]['needs']}")
    path = os.path.join(regress.GOLDEN, f"{name}.npz")
    assert os.path.exists(path), f"no golden file {path}"
    out, _ = regress._run(name)
    with np.load(path) as g:
        old = {k: g[k] for k in g.files}
    rep = regress._compare(
        out, old, float(os.environ.get("PGM_REGRESSION_RTOL", 0.0)), float(os.environ.get("PGM_REGRESSION_ATOL", 0.0))
    )
    assert rep["ok"], {"missing": rep["missing"], "diff": dict(list(rep["diff"].items())[:10])}
