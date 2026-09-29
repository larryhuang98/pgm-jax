"""Host-side driver machinery (pgm_jax.md.driver): blocks, overflow retries, log tables and the
checkpoint format."""

import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgm_jax.md import driver
from pgm_jax.md._jaxmd import dataclasses


@dataclasses.dataclass
class Toy:
    """A small state pytree (a dataclass like MDState) with an optional part."""

    x: jnp.ndarray
    n: jnp.ndarray
    extra: object = None


def test_block_length():
    """Blocks end on every output step."""
    assert driver.block_length(1000, 100, 0, 250) == 50
    assert driver.block_length(7) == 7
    assert driver.block_length(12, 0, 0) == 12


def test_retry_block_and_splitting():
    """A block is repeated after a resize; a block that keeps failing is split with a rebuild."""
    calls = []

    def run(s):
        calls.append(s)
        return s + 1

    out = driver.retry_block(run, 0, lambda s: (s < 3, False), lambda s, lb, rb: s + 1)
    assert out == 3 and calls == [0, 1, 2]
    with pytest.raises(RuntimeError, match="overflowing"):
        driver.retry_block(run, 0, lambda s: (True, False), lambda s, lb, rb: s, attempts=3)
    done, rebuilds = [], []

    def block(n):
        if n > 4:
            raise RuntimeError("neighbour list keeps overflowing")
        done.append(n)

    driver.advance_with_rebuilds(10, block, lambda: rebuilds.append(1), lambda: 1.0)
    assert sum(done) == 10 and max(done) <= 4 and len(rebuilds) >= 2
    rebuilds.clear()
    driver.advance_with_rebuilds(4, block, lambda: rebuilds.append(1), lambda: 1.2)  # volume drift: rebuild
    assert rebuilds == [1]


def test_log_table(tmp_path, capsys):
    """One header per file; appending continues without a second header; rows echoed."""
    import sys

    p = str(tmp_path / "a.log")
    with driver.LogTable(p, title=["T = 300 K"], echo=sys.stdout) as t:
        t.write({"step": 10, "epot": -1.5, "flag": True})
        t.write({"step": 20, "epot": -2.0, "flag": False})
    with driver.LogTable(p, append=True) as t:
        t.write({"step": 30, "epot": -2.5, "flag": True})
    lines = open(p).read().splitlines()
    assert lines[0] == "# T = 300 K" and lines[1].split() == ["#", "step", "epot", "flag"]
    assert len(lines) == 5 and lines[4].split() == ["30", "-2.500000", "1"]
    assert "step" in capsys.readouterr().out
    driver.LogTable(None, echo=None).write({"a": 1})  # no file


def test_checkpoint_round_trip(tmp_path):
    """Scalars, arrays, random-generator state and state pytrees survive bitwise; kinds, versions
    and structures are checked."""
    rng = np.random.default_rng(3)
    st = Toy(jnp.asarray(rng.normal(size=(4, 3))), jnp.asarray(7, jnp.int32), extra=jnp.ones(2, jnp.float32))
    content = {
        "step": 12,
        "time_ps": 0.1 + 0.2,
        "flag": True,
        "name": "x",
        "none": None,
        "rng": np.random.default_rng(5).bit_generator.state,
        "stats": {"counts": np.arange(4), "n": np.int64(3)},
        "samples": [np.ones(2), np.zeros(2)],
        "states": [st, st],
    }
    path = str(tmp_path / "c.chk")
    driver.write_checkpoint(path, "toy", content)
    assert not driver.is_legacy_checkpoint(path)
    back = driver.read_checkpoint(path, "toy", Toy(jnp.zeros((4, 3)), jnp.asarray(0, jnp.int32), jnp.zeros(2)))
    assert back["step"] == 12 and back["time_ps"] == 0.1 + 0.2 and back["flag"] is True and back["none"] is None
    assert back["rng"] == content["rng"] and back["stats"]["n"] == 3
    assert np.array_equal(back["stats"]["counts"], np.arange(4)) and back["stats"]["counts"].dtype == np.int64
    for a, b in zip(jax.tree_util.tree_leaves(back["states"]), jax.tree_util.tree_leaves(content["states"])):
        assert np.array_equal(np.asarray(a), np.asarray(b)) and np.asarray(a).dtype == np.asarray(b).dtype
    with pytest.raises(ValueError, match="not a 'other' one"):
        driver.read_checkpoint(path, "other")
    with pytest.raises(ValueError, match="template"):
        driver.read_checkpoint(path, "toy")
    with pytest.raises(ValueError, match="does not match"):  # a template without the extra part
        driver.read_checkpoint(path, "toy", Toy(jnp.zeros((4, 3)), jnp.asarray(0, jnp.int32)))
    back = driver.read_checkpoint(path, "toy", Toy(jnp.zeros((4, 3)), jnp.asarray(0, jnp.int32)), [".extra"])
    assert back["states"][0].extra is None
    # a newer format version is refused
    with np.load(path) as z:
        data = dict(z)
    header = json.loads(str(data["__header__"]))
    header["version"] = driver.CHECKPOINT_VERSION + 1
    data["__header__"] = np.array(json.dumps(header))
    np.savez(str(tmp_path / "v.npz"), **data)
    with pytest.raises(ValueError, match="newer"):
        driver.read_checkpoint(str(tmp_path / "v.npz"), "toy")


def test_legacy_pickle(tmp_path):
    """Pickle files are recognised and returned as written; their format entry is checked."""
    path = str(tmp_path / "old.chk")
    with open(path, "wb") as fh:
        pickle.dump({"format": "pgm_jax toy 1", "x": np.arange(3)}, fh)
    assert driver.is_legacy_checkpoint(path)
    d = driver.read_checkpoint(path, "toy", legacy_format="pgm_jax toy 1")
    assert np.array_equal(d["x"], np.arange(3))
    with pytest.raises(ValueError):
        driver.read_checkpoint(path, "toy", legacy_format="pgm_jax other 1")


def test_finite_or_raise():
    """Non-finite energies stop a run."""
    driver.finite_or_raise(1.0, 3)
    with pytest.raises(FloatingPointError, match="step 3"):
        driver.finite_or_raise(float("nan"), 3)
