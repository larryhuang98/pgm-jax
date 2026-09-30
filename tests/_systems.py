"""Shared test systems, settings and helpers for the pgm_jax test suite.

Every builder that more than one test module uses lives here (test modules do not import each
other).  The small molecules and boxes are those of pgm_jax.models.toy (the example systems of the
library, also used by the regression harness); this module adds the builders that only the tests
need.  Everything is deterministic: random geometries come from numpy generators with fixed seeds,
so the numbers the tests compare against are the same in every run.

Contents:

- molecules and boxes: `water`, `methanol`, `water_geometry`, `water_lattice`, `small_box`,
  `water_cluster_box` (pgm_jax.models.toy), `cluster` (gas-phase cluster from a generator),
  `random_atoms_box`, `ethanal`, `peptide` / `peptide_template` / `peptide_spec` (RDKit),
  `tip4pew_ideal` (tleap TIP4P-Ew box), `flexible_water_template` / `flexible_water_box` (PIMD);
- MD settings: `md_settings` (tight PME against exact Ewald), `alch_settings`, `flux_settings`;
- simulations: `methanol_template` / `methanol_liquid` (class II methanol), `flux_template`,
  `water_sim` (64 rigid waters), `rigid_water_sim` (rigid-body or constraint engine),
  `alch_sim` / `alch_frame` / `flex_solute_box` (alchemical water box, flexible solute);
- numerical checks: `fd_check` (directional central differences of a gradient);
- external data: `PGM3P25_TOP`, `PGM3P25_RST` and the skip mark `requires_pgm3p25`.

Units: nm, ps, kJ/mol, e, e nm, nm^3 (library units).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from pgm_jax import System
from pgm_jax.md.box import reduce_box
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.models.toy import (  # noqa: F401  (re-exported for the test modules)
    METHANOL_BONDS,
    methanol,
    small_box,
    water,
    water_cluster_box,
    water_geometry,
    water_lattice,
)
from pgm_jax.paths import resource
from pgm_jax.system import Molecule

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PGM3P25_TOP = resource("gvdw_data", "topology/rayl_512_v2.prmtop")
PGM3P25_RST = resource("gvdw_data", "inputs/lj/inpcrd.restrt")


def requires_pgm3p25(test):
    """Mark a test that needs the pGM3P-25 water box (needs_data; skipped without the files).

    The 512-water topology and restart come from the gvdw_data resource (pgm_jax.paths.resource,
    environment variable PGM_GVDW_DATA); they are not part of the repository.
    """
    return pytest.mark.needs_data(
        pytest.mark.skipif(
            not (os.path.exists(PGM3P25_TOP) and os.path.exists(PGM3P25_RST)), reason="pGM3P-25 box not available"
        )(test)
    )


def requires(module):
    """Return a decorator marking a test that needs the optional package `module` (optional_deps).

    Parameters
    ----------
    module : str
        Importable name (e.g. "rdkit", "ase"); the test is skipped when its top-level package is not
        installed.

    Returns
    -------
    callable
        Decorator adding the optional_deps marker and the skip condition.
    """
    missing = importlib.util.find_spec(module.split(".")[0]) is None

    def mark(test):
        return pytest.mark.optional_deps(pytest.mark.skipif(missing, reason=f"{module} not installed")(test))

    return mark


# ----------------------------------------------------------------------------- small systems
def _rot(rng):
    """Return a random orthogonal 3 x 3 matrix (QR of a Gaussian matrix; det = +-1) drawn from `rng`."""
    return np.linalg.qr(rng.normal(size=(3, 3)))[0]


def cluster(rng):
    """Build a gas-phase water + methanol + water cluster, molecules ~0.3 nm apart.

    The same geometry as pgm_jax.models.toy.cluster(seed), but drawn from a generator that the caller
    keeps using afterwards (the tests perturb parameters with the same `rng`, so the random stream
    must continue exactly where the geometry left it).

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of the two random orientations (advanced by two 3 x 3 normal draws).

    Returns
    -------
    system : System
        Water, methanol, water (toy parameters of pgm_jax.models.toy).
    positions : np.ndarray (12, 3)
        Positions [nm].
    """
    w = water_geometry()
    m, xm = methanol()
    pos = np.concatenate([w, (xm - xm.mean(0)) @ _rot(rng).T + [0.33, 0.05, 0.0], w @ _rot(rng).T + [0.12, 0.30, 0.08]])
    return System([water(), m, water()]), pos


def random_atoms_box(seed=0, n=14):
    """Return random atoms in a skewed triclinic box (most pairs are minimum images across the boundary).

    Parameters
    ----------
    seed : int
        Seed of the uniform fractional coordinates.
    n : int
        Number of atoms.

    Returns
    -------
    positions : np.ndarray (n, 3)
        Positions [nm].
    box : np.ndarray (3, 3)
        Reduced lower-triangular box, lattice vectors as rows [nm] (about 1.6 x 1.5 x 1.45 nm).
    """
    rng = np.random.default_rng(seed)
    H = reduce_box(np.array([[1.6, 0.0, 0.0], [0.5, 1.5, 0.0], [-0.4, 0.6, 1.45]]))
    return rng.uniform(size=(n, 3)) @ H, H


def ethanal():
    """Return acetaldehyde (planar carbonyl carbon, so the topology has one improper).

    Returns
    -------
    elements : list of str (7)
    bonds : list of tuple (6)
        Atom pairs, zero-based.
    orders : list of int (6)
        Bond orders (the C=O bond is double).
    positions : np.ndarray (7, 3)
        Geometry [nm].
    """
    x = np.array(
        [
            [0.0000, 0.0000, 0.0],
            [0.1500, 0.0000, 0.0],
            [0.2180, 0.1030, 0.0],
            [0.1980, -0.0990, 0.0],
            [-0.0360, -0.1030, 0.0],
            [-0.0380, 0.0520, 0.0890],
            [-0.0380, 0.0520, -0.0890],
        ]
    )
    el = ["C", "C", "O", "H", "H", "H", "H"]
    bonds = [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (0, 6)]
    return el, bonds, [1, 2, 1, 1, 1, 1], x


# ----------------------------------------------------------------------------- settings
def md_settings(**kw):
    """Return tight MD settings for comparisons against exact references.

    PME with a 48^3 grid of order 8 at ewald_beta 6 / nm and a 0.6 nm cutoff reproduces the exact Ewald
    sum (PeriodicPGM) to ~1e-6 relative; induced dipoles to 1e-12, float64 throughout, no predictor
    extrapolation and no van der Waals tail, so that energies and forces are deterministic functions of
    the positions.

    Parameters
    ----------
    **kw
        Flat MDSettings names that override the defaults (MDSettings.replace).

    Returns
    -------
    MDSettings
    """
    base = dict(
        cutoff=0.6,
        skin=0.05,
        ewald_beta=6.0,
        pme_grid=(48, 48, 48),
        pme_order=8,
        lj_lrc=False,
        dipole_tol=1e-12,
        max_iter=500,
        peek=0.0,
        extrap_order=0,
        precision="double",
    )
    base.update(kw)
    return MDSettings().replace(**base)


def alch_settings(**kw):
    """Return the MD settings of the alchemical water box (0.55 nm cutoff, 32^3 PME grid, float64).

    Parameters
    ----------
    **kw
        Flat MDSettings names that override the defaults (MDSettings.replace).

    Returns
    -------
    MDSettings
    """
    base = dict(
        precision="double",
        dipole_tol=1e-11,
        max_iter=400,
        cutoff=0.55,
        skin=0.05,
        ewald_beta=5.0,
        pme_grid=(32, 32, 32),
        peek=0.0,
    )
    base.update(kw)
    return MDSettings().replace(**base)


def flux_settings(**kw):
    """Return tight MD settings of the charge-flux tests (0.5 nm cutoff, 32^3 grid of order 8, float64).

    Parameters
    ----------
    **kw
        Flat MDSettings names that override the defaults (MDSettings.replace).

    Returns
    -------
    MDSettings
    """
    base = dict(
        cutoff=0.5,
        skin=0.05,
        ewald_beta=6.0,
        pme_grid=(32, 32, 32),
        pme_order=8,
        lj_lrc=False,
        dipole_tol=1e-12,
        max_iter=500,
        peek=0.0,
        precision="double",
    )
    base.update(kw)
    return MDSettings().replace(**base)


# ----------------------------------------------------------------------------- numerical checks
def fd_check(E, x, rng, h=1e-6):
    """Assert that jax.grad(E) matches central differences along three random directions.

    The directional derivative (E(x + h d) - E(x - h d)) / (2 h) with h = 1e-6 nm has a truncation
    error ~h^2 |E'''| and a round-off error ~1e-16 |E| / h, both below the tolerance
    1e-6 max(1, |fd|).

    Parameters
    ----------
    E : callable
        Scalar energy E(x) [kJ/mol] of a jax array of positions.
    x : np.ndarray (N, 3)
        Point of the check [nm].
    rng : numpy.random.Generator
        Source of the directions.
    h : float
        Step [nm].
    """
    import jax
    import jax.numpy as jnp

    g = jax.grad(E)(jnp.asarray(x))
    for _ in range(3):
        d = rng.normal(size=x.shape)
        fd = (E(jnp.asarray(x + h * d)) - E(jnp.asarray(x - h * d))) / (2 * h)
        assert abs(float(fd) - float(jnp.sum(g * d))) < 1e-6 * max(1.0, abs(float(fd)))


# ----------------------------------------------------------------------------- flexible molecules
def methanol_template(**kw):
    """Build a flexible methanol template (class II bonded terms, 1-4 LJ scaled by 0.5).

    Methanol with class II bonded terms at their initial values (reference values from the
    geometry) and scaled 1-4 LJ, so that every intramolecular channel is exercised.

    Parameters
    ----------
    **kw
        Further BondedSettings options (e.g. vdw="gvdw").

    Returns
    -------
    template : FlexibleTemplate
    positions : np.ndarray (6, 3)
        Reference geometry [nm].
    """
    from pgm_jax.bonded import terms as T
    from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec
    from pgm_jax.md.flexible import FlexibleTemplate

    m, x = methanol()
    spec = MolSpec("methanol", list(m.elements), METHANOL_BONDS, [1] * len(METHANOL_BONDS), 0, x, m)
    st = BondedSettings(families=T.PAPER, lj14_scale=0.5, **kw)
    model = BondedModel([spec], st)
    return FlexibleTemplate.from_fit(model, model.init_params()), x


def methanol_liquid(n=32, density=0.55):
    """Build a liquid of flexible methanols (methanol_template) at the given density.

    Parameters
    ----------
    n : int
        Number of molecules.
    density : float
        Density [g/cm^3] (liquid_box, seed 0, atoms at least 0.18 nm apart).

    Returns
    -------
    template : FlexibleTemplate
    system : System
    positions : np.ndarray (6 n, 3)
        Positions [nm].
    box : np.ndarray (3, 3)
        Cubic box [nm].
    """
    from pgm_jax.md.flexible import liquid_box

    tpl, _ = methanol_template()
    pos, H = liquid_box(tpl, n, density, seed=0, min_dist=0.18)
    return tpl, System([tpl.pgm] * n), pos, H


def flux_template(order=2, seed=3):
    """Build a flexible methanol template with charge flux.

    Methanol with class II bonded terms (initial values) and made-up flux parameters of the
    size of a fit (jb up to 3 e/nm, jc up to 1 e, jc2 up to 8 e/nm).

    Parameters
    ----------
    order : {1, 2}
        Flux order (2 adds the second-order jc2 terms).
    seed : int
        Seed of the flux parameters.

    Returns
    -------
    template : FlexibleTemplate
    positions : np.ndarray (6, 3)
        Reference geometry [nm].
    """
    import jax.numpy as jnp

    from pgm_jax.bonded import terms as T
    from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec
    from pgm_jax.md.flexible import FlexibleTemplate

    m, x = methanol()
    spec = MolSpec("methanol", list(m.elements), METHANOL_BONDS, [1] * len(METHANOL_BONDS), 0, x, m)
    model = BondedModel([spec], BondedSettings(families=T.PAPER, lj14_scale=0.5, flux=order))
    P = model.init_params()
    rng = np.random.default_rng(seed)
    nk = len(P["flux"]["jb"])
    P["flux"] = {"jb": jnp.asarray(rng.uniform(-3, 3, nk)), "jc": jnp.asarray(rng.uniform(-1, 1, nk))}
    if order >= 2:
        P["flux"]["jc2"] = jnp.asarray(rng.uniform(-8, 8, nk))
    return FlexibleTemplate.from_fit(model, P), x


def flexible_water_template():
    """Build a flexible water template for PIMD (quartic bonds, harmonic and cubic angle, cross terms).

    The bonded model of pgm_jax.models.water.WATER_FAMILIES with a stiff quartic bond (K2 = 4.5e5
    kJ/mol/nm^2) and a harmonic angle (Ka = 350 kJ/mol/rad^2); the pGM part is the toy water.

    Returns
    -------
    template : FlexibleTemplate
    positions : np.ndarray (3, 3)
        Reference geometry (0.0957 nm, 104.5 deg) [nm].
    """
    import math

    import jax.numpy as jnp

    from pgm_jax.bonded.model import BondedModel, BondedSettings, MolSpec
    from pgm_jax.md.flexible import FlexibleTemplate
    from pgm_jax.models.water import WATER_FAMILIES

    m = water()
    t = math.radians(104.5)
    x = np.array([[0, 0, 0], [0.0957, 0, 0], [0.0957 * math.cos(t), 0.0957 * math.sin(t), 0]])
    spec = MolSpec("WAT", ["O", "H", "H"], [(0, 1), (0, 2)], [1, 1], 0, x, m)
    model = BondedModel([spec], BondedSettings(families=WATER_FAMILIES))
    P = model.init_params()
    P["bond_quartic"]["K2"] = jnp.full_like(P["bond_quartic"]["K2"], 4.5e5)
    P["angle_harm"]["Ka"] = jnp.full_like(P["angle_harm"]["Ka"], 350.0)
    return FlexibleTemplate.from_fit(model, P), x


def flexible_water_box(n_side=2, L=1.5, seed=0):
    """Place randomly rotated flexible waters on a jittered cubic lattice (the PIMD test system).

    Parameters
    ----------
    n_side : int
        Molecules per box edge (n_side^3 waters).
    L : float
        Box edge [nm].
    seed : int
        Seed of the orientations and the 0.02 nm jitter.

    Returns
    -------
    template : FlexibleTemplate
    system : System
    positions : np.ndarray (3 n_side^3, 3)
        Positions [nm].
    box : np.ndarray (3, 3)
        Cubic box [nm].
    """
    tpl, x = flexible_water_template()
    rng = np.random.default_rng(seed)
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                c = (np.array([i, j, k]) + 0.5) * L / n_side + rng.normal(scale=0.02, size=3)
                pos.append((x - x.mean(0)) @ R.T + c)
    n = n_side**3
    return tpl, System([tpl.pgm] * n), np.concatenate(pos), np.eye(3) * L


# ----------------------------------------------------------------------------- peptides (RDKit)
ACE_ALA_NME = "CC(=O)N[C@@H](C)C(=O)NC"
ACE_ALA_GLY_NME = "CC(=O)N[C@@H](C)C(=O)NCC(=O)NC"  # 29 atoms
_RAD = {"H": 0.05, "C": 0.07, "N": 0.065, "O": 0.06}
_ALP = {"H": 0.4e-3, "C": 1.2e-3, "N": 1.0e-3, "O": 0.8e-3}
_RMH = {"H": 0.13, "C": 0.19, "N": 0.18, "O": 0.17}
_SEP = {"H": 0.12, "C": 0.33, "N": 0.40, "O": 0.45}


def peptide_spec(smiles, name="peptide", seed=7):
    """Build the MolSpec of a small peptide from SMILES, without pGM parameters (skip without RDKit).

    Parameters
    ----------
    smiles : str
        Heavy-atom SMILES (hydrogens are added).
    name : str
        Name of the MolSpec.
    seed : int
        Seed of RDKit's ETKDG embedding (then MMFF-optimised).

    Returns
    -------
    MolSpec
        Elements, bonds, bond orders, charge 0 and the geometry [nm].
    """
    from pgm_jax.bonded.model import MolSpec

    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    x = m.GetConformer().GetPositions() * 0.1
    el = [a.GetSymbol() for a in m.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    orders = [b.GetBondTypeAsDouble() for b in m.GetBonds()]
    return MolSpec(name, el, bonds, orders, 0, x)


def peptide(smiles=ACE_ALA_GLY_NME, seed=5):
    """Build a peptide with made-up but plausible pGM parameters (skip without RDKit).

    Gasteiger charges (shifted to a neutral molecule), element radii, polarizabilities and LJ
    parameters, and random covalent dipoles (0.004 e nm scale) on every bond, on the MMFF geometry.

    Parameters
    ----------
    smiles : str
        Heavy-atom SMILES (default Ace-Ala-Gly-Nme, 29 atoms).
    seed : int
        Seed of the embedding and of the covalent dipoles.

    Returns
    -------
    spec : MolSpec
        With the pGM Molecule attached.
    positions : np.ndarray (N, 3)
        Geometry [nm].
    """
    from pgm_jax.bonded.model import MolSpec

    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    AllChem.ComputeGasteigerCharges(m)
    x = m.GetConformer().GetPositions() * 0.1
    el = [a.GetSymbol() for a in m.GetAtoms()]
    q = np.array([a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()])
    q -= q.mean()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    orders = [b.GetBondTypeAsDouble() for b in m.GetBonds()]
    rng = np.random.default_rng(seed)
    cov = [c for i, j in bonds for c in ((i, j, 0.004 * rng.normal()), (j, i, 0.004 * rng.normal()))]
    mol = Molecule(
        "pep",
        el,
        el,
        q,
        [_RAD[e] for e in el],
        [_ALP[e] for e in el],
        cov=cov,
        lj_rmin_half=[_RMH[e] for e in el],
        lj_sqrt_eps=[_SEP[e] for e in el],
        bonds=bonds,
    )
    return MolSpec("pep", el, bonds, orders, 0, x, mol), x


def peptide_template():
    """Build the flexible Ace-Ala-Gly-Nme template (protein term set, random CMAP and torsions).

    Returns
    -------
    template : FlexibleTemplate
    model : BondedModel
        The gas-phase model the template is made from.
    params : dict
        Its parameters (random CMAP coefficients and Amber torsion barriers, seed 1).
    positions : np.ndarray (29, 3)
        Geometry [nm].
    """
    import jax.numpy as jnp

    from pgm_jax.bonded import terms as T
    from pgm_jax.bonded.model import BondedModel, BondedSettings
    from pgm_jax.md.flexible import FlexibleTemplate

    s, x = peptide()
    model = BondedModel([s], BondedSettings(families=T.PROTEIN, lj14_scale=0.5))
    P = model.init_params()
    rng = np.random.default_rng(1)
    P["cmap"]["cm"] = jnp.asarray(rng.normal(size=P["cmap"]["cm"].shape))
    P["torsion_amber"]["K"] = jnp.asarray(rng.normal(size=P["torsion_amber"]["K"].shape))
    return FlexibleTemplate.from_fit(model, P), model, P, x


# ----------------------------------------------------------------------------- virtual sites
def tip4pew_ideal():
    """Return the tleap TIP4P-Ew test box with ideal water geometry and placed extra points.

    Every water of tests/data/tip4pew_small is superposed onto the model geometry (Amber's SHAKE
    lengths 0.9572 / 1.5136 A) and the extra points are placed by VirtualSites.

    Returns
    -------
    system : System
        Point-charge TIP4P-Ew waters (Amber charges, zero polarizability) with one virtual site each.
    positions : np.ndarray (4 n, 3)
        Positions [nm].
    box : np.ndarray (3, 3)
        Box [nm].
    """
    from pgm_jax.md.box import box_from_cell
    from pgm_jax.md.io import read_coordinates
    from pgm_jax.md.vsites import VirtualSites
    from pgm_jax.param import read_prmtop_molecules

    prm = os.path.join(DATA, "tip4pew_small.prmtop")
    mols = read_prmtop_molecules(prm, charges="amber")
    xyz, _, box = read_coordinates(os.path.join(DATA, "tip4pew_small.inpcrd"))
    H = box_from_cell(*box) * 0.1
    X = (xyz * 0.1).reshape(-1, 4, 3)
    r, hh = 0.09572, 0.15136
    t = np.arcsin(hh / 2 / r)
    ideal = np.array([[0, 0, 0], [r * np.sin(t), r * np.cos(t), 0], [-r * np.sin(t), r * np.cos(t), 0]])
    for k in range(len(X)):
        y = X[k, :3] - X[k, :3].mean(0)
        c = ideal - ideal.mean(0)
        U, _, Vt = np.linalg.svd(c.T @ y)
        R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
        X[k, :3] = c @ R.T + X[k, :3].mean(0)
    sys = System(mols)
    return sys, np.asarray(VirtualSites.of(sys).place(X.reshape(-1, 3), H)), H


# ----------------------------------------------------------------------------- water simulations
S_WATER = MDSettings().replace(precision="double", dipole_tol=1e-12, max_iter=300, cutoff=0.55, skin=0.05)


def water_sim(engine, mts=None, dt=0.002, thermostat=None, seed=3, settings=S_WATER, **kw):
    """Build the MD simulation of 64 rigid pGM-like waters (the multiple-time-step test system).

    64 pGM-like waters (rigid bodies or constraints), 0.55 nm cutoff, float64; NVE by default.

    Parameters
    ----------
    engine : {"rigid", "constraints"}
        Rigid bodies (Simulation) or rigid waters by constraints (FlexibleSimulation).
    mts : MTS, optional
        Multiple time stepping (None: the ordinary integrator).
    dt : float
        Time step [ps].
    thermostat : object, optional
        Thermostat (None: NVE).
    seed : int
        Random seed of the engine.
    settings : MDSettings
        Force-field settings.
    **kw
        Further keywords of the engine.

    Returns
    -------
    Simulation or FlexibleSimulation
    """
    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    from pgm_jax.md.simulation import Simulation

    pos, H, w = water_lattice(n_side=4, spacing=0.31)
    wat = water()
    sys = System([wat] * (len(pos) // 3))
    if engine == "rigid":
        return Simulation(sys, pos, H, settings, dt=dt, thermostat=thermostat, log=None, seed=seed, mts=mts, **kw)
    return FlexibleSimulation(
        sys,
        [RigidTemplate(wat, w)] * sys.nmol,
        pos,
        H,
        settings,
        dt=dt,
        thermostat=thermostat,
        log=None,
        seed=seed,
        mts=mts,
        **kw,
    )


def rigid_water_sim(engine, pos, H, w, settings, **kw):
    """Build a simulation of rigid toy waters in either engine.

    Parameters
    ----------
    engine : {"rigid", "atoms"}
        Rigid bodies (Simulation) or atoms with rigid-water constraints (FlexibleSimulation).
    positions : np.ndarray (3 n, 3)
        Water positions [nm].
    H : np.ndarray (3, 3)
        Box [nm].
    w : np.ndarray (3, 3)
        Water geometry of the rigid templates [nm].
    settings : MDSettings
        Force-field settings.
    **kw
        Further keywords of the engine (dt, thermostat, bias, restraints, ...).

    Returns
    -------
    Simulation or FlexibleSimulation
    """
    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    from pgm_jax.md.simulation import Simulation

    wat = water()
    sys = System([wat] * (len(pos) // 3))
    if engine == "rigid":
        return Simulation(sys, pos, H, settings, log=None, **kw)
    return FlexibleSimulation(sys, [RigidTemplate(wat, w)] * sys.nmol, pos, H, settings, log=None, **kw)


# ----------------------------------------------------------------------------- alchemical systems
def alch_box(seed=0):
    """Return the 64-water lattice box of the alchemical tests.

    Parameters
    ----------
    seed : int
        Seed of the water orientations.

    Returns
    -------
    positions : np.ndarray (192, 3)
        Positions [nm].
    box : np.ndarray (3, 3)
        Cubic box, 1.24 nm [nm].
    system : System
        64 toy waters.
    """
    pos, H, _ = water_lattice(4, 0.31, seed)
    return pos, H, System([water()] * (len(pos) // 3))


def alch_sim(s=None, lam=(1.0, 1.0), box_seed=0, **kw):
    """Build a rigid-body simulation of the water box with the first water alchemical.

    Parameters
    ----------
    s : MDSettings, optional
        Settings (None: alch_settings()).
    lam : tuple of float
        Initial (lambda_elec, lambda_vdw).
    box_seed : int
        Seed of the box (alch_box).
    **kw
        Further Simulation keywords.

    Returns
    -------
    sim : Simulation
    alchemy : Alchemy
    params : dict
        Parameters of the alchemical system (the solute has its own table entries).
    reference : tuple
        (positions [nm], box [nm], the plain System without the alchemical copy).
    """
    from pgm_jax.md.alchemy import Alchemy, alchemical_system
    from pgm_jax.md.simulation import Simulation

    pos, H, sys0 = alch_box(box_seed)
    sysA, P = alchemical_system(sys0, 0)
    alch = Alchemy(sysA, 0, lam=lam)
    sim = Simulation(sysA, pos, H, s or alch_settings(), dt=0.001, log=None, params=P, alchemy=alch, **kw)
    return sim, alch, P, (pos, H, sys0)


def alch_frame(sim):
    """Return the current frame of a rigid-body simulation as the alchemical energy functions take it.

    Parameters
    ----------
    sim : Simulation

    Returns
    -------
    positions : jax.Array (N, 3)
        Atom positions [nm].
    box : jax.Array (3, 3)
        Box [nm].
    candidates : jax.Array (N, K)
        Neighbour candidate rows of the current list (padded with N).
    """
    st = sim.state
    X = sim.rigid.positions(st.dyn.position)
    cand = sim.integ.nb.candidates(st.nbr, st.dyn.position.center, st.box, X)[0]
    return X, st.box, cand


def flex_solute_box():
    """Build a flexible methanol solute in a box of rigid waters.

    Flexible methanol (the class II template of test_flexible, scaled 1-4 LJ) at the centre of a
    box of rigid waters (constraints), the waters within 0.25 nm of it removed.

    Returns
    -------
    template : FlexibleTemplate
        The methanol template.
    system : System
        Methanol first, then the waters.
    templates : list
        The template of every molecule (RigidTemplate for the waters).
    positions : np.ndarray (N, 3)
        Positions [nm].
    box : np.ndarray (3, 3)
        Cubic box, 1.24 nm [nm].
    """
    from pgm_jax.md.flexible import RigidTemplate

    tpl, xm = methanol_template()
    pos, H, w = water_lattice(4, 0.31)
    L = H[0, 0]
    xm = xm - xm.mean(0) + 0.5 * L
    wat = water()
    keep = []
    for k in range(len(pos) // 3):
        d = pos[3 * k : 3 * k + 3, None, :] - xm[None]
        d -= L * np.round(d / L)
        if np.linalg.norm(d, axis=-1).min() > 0.25:
            keep.append(k)
    X = np.concatenate([xm] + [pos[3 * k : 3 * k + 3] for k in keep])
    rt = RigidTemplate(wat, w)
    return tpl, System([tpl.pgm] + [wat] * len(keep)), [tpl] + [rt] * len(keep), X, H
