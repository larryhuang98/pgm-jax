"""Command-line interface of every script and the `pgm-jax` entry point (pgm_jax/cli).

Every Python script below scripts/, examples/ and paper/scripts/ must import and answer `--help`
through its `main(argv)` (docs/dev/style_guide.md, script template).  The scripts are imported in
one child process: several set `jax_enable_x64` or other JAX options at import time, which must
not leak into this test session.  The child reports one JSON record per script.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import pytest

from pgm_jax.cli import main as cli
from pgm_jax.paths import repo_path

SCRIPT_DIRS = ("scripts", "examples", os.path.join("paper", "scripts"))

#: the child process: for each script path (argv), run main(["--help"]) with stdout captured
_CHILD = r"""
import contextlib, io, json, sys, traceback
from pgm_jax.cli.main import run_script
out = {}
for path in sys.argv[1:]:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run_script(path, ["--help"])
        out[path] = {"code": None, "help": buf.getvalue(), "error": "main returned without SystemExit"}
    except SystemExit as e:
        out[path] = {"code": e.code, "help": buf.getvalue(), "error": ""}
    except BaseException:
        out[path] = {"code": -1, "help": buf.getvalue(), "error": traceback.format_exc()[-2000:]}
sys.stdout = sys.__stdout__
print("@@JSON@@" + json.dumps(out))
"""


def script_files() -> list[str]:
    """Return the repository's Python scripts (relative paths, sorted)."""
    root = repo_path()
    files = []
    for d in SCRIPT_DIRS:
        files += glob.glob(os.path.join(root, d, "**", "*.py"), recursive=True)
    return sorted(os.path.relpath(f, root) for f in files if "__pycache__" not in f)


SCRIPTS = script_files()


@pytest.fixture(scope="module")
def help_results() -> dict:
    """Run every script's `--help` in one child process and return {relative path: record}."""
    root = repo_path()
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    paths = [os.path.join(root, s) for s in SCRIPTS]
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, *paths], cwd=root, env=env, capture_output=True, text=True, timeout=1800
    )
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")]
    assert line, f"child failed (status {r.returncode}):\n{r.stderr[-3000:]}"
    res = json.loads(line[-1][len("@@JSON@@") :])
    return {os.path.relpath(k, root): v for k, v in res.items()}


def test_scripts_found():
    """The script folders are where the layout says (docs/api_design.md, D13) and not empty."""
    assert len(SCRIPTS) > 50
    for folder in ("md", "pimd", "dielectric", "fitting", "free_energy", "qm", "validation", "benchmarks"):
        assert any(s.startswith(os.path.join("scripts", folder) + os.sep) for s in SCRIPTS), folder


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help(help_results, script):
    """`main(["--help"])` of the script exits with status 0 and prints a usage line."""
    rec = help_results[script]
    assert rec["code"] == 0, rec["error"]
    assert "usage:" in rec["help"]


def test_pgm_jax_help(capsys):
    """`pgm-jax --help` lists every subcommand; an unknown subcommand gives status 2."""
    assert cli.main(["--help"]) == 0
    text = capsys.readouterr().out
    for name in cli.SUBCOMMANDS:
        assert f"  {name} " in text
    assert cli.main(["no-such-command"]) == 2


def test_subcommand_scripts_exist():
    """Every subcommand points at an existing script with a main function."""
    for name in cli.SUBCOMMANDS:
        path = cli.script_path(name)
        assert os.path.isfile(path), (name, path)
        with open(path) as fh:
            assert "def main(argv" in fh.read(), name


def test_subcommand_dispatch():
    """`python -m pgm_jax.cli dielectric --help` runs the dielectric script's parser under the subcommand's name."""
    root = repo_path()
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run(
        [sys.executable, "-m", "pgm_jax.cli", "dielectric", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.startswith("usage: pgm-jax dielectric")
    assert "--skip-ps" in r.stdout
