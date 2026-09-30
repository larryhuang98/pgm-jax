"""Install check: ten MD steps of 512 pGM3P-25 waters on CPU (double), GPU (double) and GPU (mixed precision).

`examples/water512` holds the 512-water box (Amber prmtop and rst7; truncated octahedron, 1,536
atoms).  Each test runs `pgm-jax md` (`pgm_jax.cli.main.main`, in process, on the default device of
the named platform) for 10 NVE steps of 1 fs and reads the last two rows of `md.log`.  The reference
values were recorded on an NVIDIA RTX PRO 6000 (Blackwell) and an Intel Xeon (double precision agrees
between both to all printed digits).

Checked, and how tightly: the potential energy of step 10 against the recorded value (double
precision 1e-9 relative, mixed precision 1e-6: float32 pair kernels with float64 accumulation differ by
~1e-7); the temperature (1e-6 relative, double) and conservation of the total energy between steps 5
and 10 (0.5 kJ/mol of 2.1e6).  The GPU tests are skipped without a GPU (marker `gpu`).
"""

import jax
import pytest
from _systems import requires

from pgm_jax.cli.main import main as pgm_jax_cli
from pgm_jax.paths import repo_path

TOP = repo_path("examples", "water512", "pgm3p25_512.prmtop")
CRD = repo_path("examples", "water512", "pgm3p25_512.rst7")

# step 10 of the run below (double precision; potential energy and temperature)
EPOT_REF = -2114884.256504  # kJ/mol
TEMP_REF = 291.356783  # K
STEPS, EVERY = 10, 5


def run_md(tmp_path, platform, precision):
    """Run the 10-step NVE simulation of `pgm-jax md` on the first device of a platform.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    platform : {"cpu", "gpu"}
        JAX platform whose first device is the default device of the run.
    precision : {"double", "mixed"}
        Precision of the pair kernels (`--precision`).

    Returns
    -------
    list of dict
        One dict per log row (steps 5 and 10): the columns of md.log (step, temp_K, epot, etot, ...).
    """
    argv = [
        "md", "-p", TOP, "-c", CRD, "-o", str(tmp_path / "md"),
        "--nsteps", str(STEPS), "--report-every", str(EVERY), "--thermostat", "none", "--barostat", "none",
        "--dt-fs", "1.0", "--precision", precision,
    ]  # fmt: skip
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"no {platform} backend (JAX_PLATFORMS excludes it)")
    with jax.default_device(device):
        pgm_jax_cli(argv)
    lines = (tmp_path / "md.log").read_text().splitlines()
    names = [ln for ln in lines if ln.startswith("#") and "epot" in ln][-1].lstrip("#").split()
    rows = [dict(zip(names, map(float, ln.split()))) for ln in lines if ln.strip() and not ln.startswith("#")]
    assert [int(r["step"]) for r in rows] == [EVERY, STEPS], rows
    return rows


def check(rows, rtol_epot, rtol_temp):
    """Assert the energy, temperature and energy conservation of the two log rows."""
    last = rows[-1]
    assert abs(last["epot"] - EPOT_REF) < rtol_epot * abs(EPOT_REF), (last["epot"], EPOT_REF)
    assert abs(last["temp_K"] - TEMP_REF) < rtol_temp * TEMP_REF, (last["temp_K"], TEMP_REF)
    assert abs(rows[1]["etot"] - rows[0]["etot"]) < 0.5, (rows[0]["etot"], rows[1]["etot"])


@requires("jax_md")
def test_water512_cpu_double(tmp_path):
    """512 waters, 10 steps on the CPU in double precision reproduce the recorded energy (1e-9)."""
    check(run_md(tmp_path, "cpu", "double"), 1e-9, 1e-6)


@requires("jax_md")
@pytest.mark.gpu
def test_water512_gpu_double(tmp_path):
    """512 waters, 10 steps on the GPU in double precision reproduce the recorded energy (1e-9)."""
    check(run_md(tmp_path, "gpu", "double"), 1e-9, 1e-6)


@requires("jax_md")
@pytest.mark.gpu
def test_water512_gpu_mixed(tmp_path):
    """512 waters, 10 steps on the GPU in mixed precision agree with double precision (energy 1e-6, T 1e-5)."""
    check(run_md(tmp_path, "gpu", "mixed"), 1e-6, 1e-5)
