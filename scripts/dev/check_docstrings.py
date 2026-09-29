"""List the modules, classes and functions without a docstring (docs/dev/style_guide.md).

Every module, every class and every function or method, public or private, needs a docstring;
nested functions need one when they are longer than a few lines (default: more than 5 lines;
shorter ones may be explained by a comment); lambdas are exempt.

    python scripts/dev/check_docstrings.py                         # pgm_jax, scripts, tests, examples
    python scripts/dev/check_docstrings.py pgm_jax/md --list       # every missing docstring, file:line
    python scripts/dev/check_docstrings.py --by-dir 2              # counts per pgm_jax/<subpackage>, ...
    python scripts/dev/check_docstrings.py pgm_jax --fail          # exit status 1 if anything is missing
    python scripts/dev/check_docstrings.py pgm_jax/md --same-code HEAD   # only docs / hints changed?

--same-code REV compares every file with its version in git revision REV after removing
docstrings, type annotations and hint-only imports (__future__, typing, collections.abc,
jax.typing, `if TYPE_CHECKING:` blocks); comments are not part of the syntax tree.  A file that
differs had a code change, which a documentation-only commit must not have.

As a test (once a directory is complete):

    from check_docstrings import find_missing
    assert not find_missing(["pgm_jax/md"])
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass

DEFAULT_PATHS = ("pgm_jax", "scripts", "tests", "examples")
SKIP_DIRS = {"__pycache__", ".git", "runs", ".tools"}


@dataclass(frozen=True)
class Missing:
    """One object without a docstring.

    Attributes
    ----------
    path : str
        File.
    line : int
        Line of the definition (1 for a module).
    kind : str
        "module", "class", "function", "method" or "nested function".
    name : str
        Qualified name (Class.method, outer.<locals>.inner).
    """

    path: str
    line: int
    kind: str
    name: str

    def __str__(self) -> str:
        """Format as file:line: kind name."""
        return f"{self.path}:{self.line}: {self.kind} {self.name}"


def python_files(paths) -> list[str]:
    """Collect the .py files under `paths` (files or directories), sorted, skipping caches and runs.

    Parameters
    ----------
    paths : iterable of str
        Files or directories.

    Returns
    -------
    list of str
    """
    out = []
    for p in paths:
        if os.path.isfile(p) and p.endswith(".py"):
            out.append(p)
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".py")]
    return sorted(set(out))


def _length(node: ast.AST) -> int:
    """Return the number of source lines of a definition (decorators excluded)."""
    return (node.end_lineno or node.lineno) - node.lineno + 1


def scan_source(source: str, path: str = "<string>", nested_min_lines: int = 6) -> tuple[int, list[Missing]]:
    """Count the objects of one module that need a docstring and list those without one.

    Parameters
    ----------
    source : str
        Python source.
    path : str
        Its file name (for the report).
    nested_min_lines : int
        Nested functions (defined inside a function) need a docstring from this many lines on.

    Returns
    -------
    required : int
        Number of objects that need a docstring (the module, classes, functions, methods, long
        nested functions).
    missing : list of Missing
        Those without one.
    """
    tree = ast.parse(source, filename=path)
    out = []
    required = [1]
    if ast.get_docstring(tree) is None:
        out.append(Missing(path, 1, "module", os.path.basename(path)))

    def visit(node: ast.AST, prefix: str, in_function: bool, in_class: bool) -> None:
        """Check the definitions directly inside `node`, then recurse."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                name = prefix + child.name
                required[0] += 1
                if ast.get_docstring(child) is None:
                    out.append(Missing(path, child.lineno, "class", name))
                visit(child, name + ".", in_function, True)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                kind = "nested function" if in_function else ("method" if in_class else "function")
                short_nested = in_function and _length(child) < nested_min_lines
                required[0] += 0 if short_nested else 1
                if ast.get_docstring(child) is None and not short_nested:
                    out.append(Missing(path, child.lineno, kind, name))
                visit(child, name + ".<locals>.", True, False)
            else:
                visit(child, prefix, in_function, in_class)

    visit(tree, "", False, False)
    return required[0], out


def missing_in_source(source: str, path: str = "<string>", nested_min_lines: int = 6) -> list[Missing]:
    """List the objects of one module without a docstring (see scan_source)."""
    return scan_source(source, path, nested_min_lines)[1]


def scan(paths=DEFAULT_PATHS, nested_min_lines: int = 6) -> tuple[dict, list[Missing]]:
    """Scan the Python files under `paths`.

    Parameters
    ----------
    paths : iterable of str
        Files or directories.
    nested_min_lines : int
        See scan_source.

    Returns
    -------
    required : dict
        {file: number of objects that need a docstring}.
    missing : list of Missing
        The objects without one, in file order.
    """
    required, out = {}, []
    for f in python_files(paths):
        with open(f, encoding="utf-8") as fh:
            required[f], miss = scan_source(fh.read(), f, nested_min_lines)
        out += miss
    return required, out


def find_missing(paths=DEFAULT_PATHS, nested_min_lines: int = 6) -> list[Missing]:
    """List the objects without a docstring under `paths` (for tests: assert not find_missing(...))."""
    return scan(paths, nested_min_lines)[1]


def group_key(path: str, depth: int) -> str:
    """Return the first `depth` components of a path (its directory for depth 0)."""
    parts = os.path.normpath(path).split(os.sep)
    if depth <= 0:
        return os.path.dirname(path) or "."
    return os.sep.join(parts[: min(depth, len(parts) - 1)]) or "."


HINT_MODULES = ("__future__", "typing", "collections.abc", "jax.typing", "numpy.typing")


class _StripDocsAndHints(ast.NodeTransformer):
    """Remove docstrings, annotations and hint-only imports from a syntax tree (see code_fingerprint)."""

    def _body(self, body: list) -> list:
        """Drop a leading docstring, hint-only imports and `if TYPE_CHECKING:` blocks of a body."""
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            if isinstance(body[0].value.value, str):
                body = body[1:]
        out = []
        for node in body:
            if isinstance(node, ast.ImportFrom) and node.module in HINT_MODULES:
                continue
            if isinstance(node, ast.Import) and all(a.name in HINT_MODULES for a in node.names):
                continue
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
                continue
            out.append(node)
        return out or [ast.Pass()]

    def generic_visit(self, node: ast.AST) -> ast.AST:
        """Strip the body of modules, classes and functions, then recurse."""
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = self._body(node.body)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
        if isinstance(node, ast.arg):
            node.annotation = None
        if isinstance(node, ast.AnnAssign):
            node.annotation = ast.Name("_", ast.Load())
        return super().generic_visit(node)


def code_fingerprint(source: str) -> str:
    """Return the syntax tree of `source` without docstrings, annotations and hint-only imports.

    Two versions of a file with the same fingerprint differ only in docstrings, comments,
    formatting, type hints and hint-only imports.

    Parameters
    ----------
    source : str
        Python source.

    Returns
    -------
    str
        ast.dump of the stripped tree (no line numbers).
    """
    return ast.dump(_StripDocsAndHints().visit(ast.parse(source)), include_attributes=False)


def changed_code(paths, rev: str) -> list[str]:
    """List the files under `paths` whose code (see code_fingerprint) differs from git revision `rev`.

    Parameters
    ----------
    paths : iterable of str
        Files or directories (inside the git work tree; run from its top directory).
    rev : str
        Git revision, e.g. HEAD or a commit id.

    Returns
    -------
    list of str
        Files that differ, and files new since `rev` (marked "(new)").
    """
    out = []
    for f in python_files(paths):
        old = subprocess.run(["git", "show", f"{rev}:{f}"], capture_output=True, text=True)
        if old.returncode != 0:
            out.append(f + " (new)")
            continue
        with open(f, encoding="utf-8") as fh:
            if code_fingerprint(fh.read()) != code_fingerprint(old.stdout):
                out.append(f)
    return out


def main(argv=None) -> int:
    """Run the command line.

    Prints the counts per file (or per directory), optionally every missing docstring; returns
    exit status 1 with --fail when anything is missing.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=list(DEFAULT_PATHS), help="files or directories")
    ap.add_argument("--list", action="store_true", help="print every missing docstring as file:line")
    ap.add_argument("--by-dir", type=int, default=None, metavar="DEPTH", help="counts per directory prefix")
    ap.add_argument("--nested-min-lines", type=int, default=6, help="nested functions from this length on")
    ap.add_argument("--fail", action="store_true", help="exit status 1 if anything is missing")
    ap.add_argument("--same-code", metavar="REV", help="only check that the code equals git revision REV")
    a = ap.parse_args(argv)
    if a.same_code:
        diff = changed_code(a.paths, a.same_code)
        for f in diff:
            print(f"code differs from {a.same_code}: {f}")
        print(f"{len(diff)} file(s) with code changes")
        return 1 if diff else 0
    required, missing = scan(a.paths, a.nested_min_lines)
    if a.list:
        for m in missing:
            print(m)

    def key(path: str) -> str:
        """Return the row of the table a file belongs to."""
        return path if a.by_dir is None else group_key(path, a.by_dir)

    counts, totals = Counter(key(m.path) for m in missing), Counter()
    for f, n in required.items():
        totals[key(f)] += n
    print(f"{'missing':>7s} {'of':>6s}  {'files' if a.by_dir is None else 'directory'}")
    for k in sorted(totals):
        if counts[k] or a.by_dir is not None:
            print(f"{counts[k]:7d} {totals[k]:6d}  {k}")
    kinds = Counter(m.kind for m in missing)
    detail = ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items()))
    print(f"{len(missing):7d} {sum(totals.values()):6d}  total ({detail})")
    return 1 if (a.fail and missing) else 0


if __name__ == "__main__":
    sys.exit(main())
