"""Fitting pGM parameters to QM cluster data (pgm_jax/qmfit.py): dataset IO, model components vs
the Model / nbody reference, charge-neutral parameter map, loss gradients vs finite differences,
exact recovery of a synthetic target, rigid-body forces."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pgm_jax import ElecChannel, LJChannel, Model, Molecule, System  # noqa: E402
from pgm_jax.qmfit import (  # noqa: E402
    KCAL,
    ClusterModel,
    FitWeights,
    ParamMap,
    Prepared,
    QMFit,
    QMSet,
    error_table,
    evaluate,
    label,
    rigid_body_forces,
    rigid_water,
    superpose_monomers,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def water():
    """pGM3P-25 water (values of p25_512.prmtop)."""
    return Molecule(
        "WAT",
        ["O", "H", "H"],
        ["OW", "HW", "HW"],
        np.array([-2.0405622, 1.0202811, 1.0202811]),
        np.array([0.0605150752, 0.0536231183, 0.0536231183]),
        np.array([1.11765163e-3, 3.2963077e-4, 3.2963077e-4]),
        cov=[(0, 1, -0.019120135), (0, 2, -0.019120135), (1, 0, 0.00858770326), (2, 0, 0.00858770326)],
        lj_rmin_half=np.array([0.178559032523734, 0, 0]),
        lj_sqrt_eps=np.array([0.7781620777141389, 0, 0]),
        bonds=[(0, 1), (0, 2), (1, 2)],
    )


W0 = rigid_water(0.9745, 103.64)  # Angstrom


def random_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
        ]
    )


def cluster(rng, n, d=2.9):
    """n rigid waters, O atoms on a loose ring of radius ~d (Angstrom), random orientations."""
    X = []
    for k in range(n):
        ang = 2 * np.pi * k / n
        c = (d / (2 * np.sin(np.pi / n)) if n > 2 else d / 2) * np.array([np.cos(ang), np.sin(ang), 0.3 * rng.normal()])
        X.append(W0 @ random_rot(rng).T + c)
    return np.concatenate(X)


def synthetic_set(rng, labels_from=None, cm=None, P=None):
    recs = []
    for k in range(6):
        recs.append(
            {
                "id": f"d{k}",
                "set": "dimers",
                "n": 2,
                "xyz_A": cluster(rng, 2, 2.7 + 0.25 * k).tolist(),
                "meta": {},
                "E": {},
                "sapt": {},
                "nb": {},
            }
        )
    for k in range(3):
        recs.append(
            {
                "id": f"t{k}",
                "set": "trimers",
                "n": 3,
                "xyz_A": cluster(rng, 3, 2.8 + 0.2 * k).tolist(),
                "meta": {},
                "E": {},
                "sapt": {},
                "nb": {},
            }
        )
    recs.append(
        {
            "id": "q0",
            "set": "tetramers",
            "n": 4,
            "xyz_A": cluster(rng, 4, 2.9).tolist(),
            "meta": {},
            "E": {},
            "sapt": {},
            "nb": {},
        }
    )
    data = QMSet(recs, {"xyz_A": W0.tolist()})
    if cm is not None:  # labels = the model's own predictions at P
        prep = Prepared(data, cm)
        pr = {k: np.asarray(v) / KCAL for k, v in prep.predict(P).items()}
        for i, r in enumerate(recs):
            r["E"]["ref"] = float(pr["total"][i])
            if r["n"] == 2:
                r["sapt"] = {
                    "elst": float(pr["elst"][i]),
                    "ind": float(pr["ind"][i]),
                    "exch": float(pr["vdw"][i]),
                    "disp": 0.0,
                }
        for c, k in enumerate(prep.clusters):
            recs[k]["nb"] = {"nb3": float(pr["nb3"][c]), "nb2": float(pr["nb2"][c])}
        for n, (ks, F, T) in prep.forces(P).items():
            for b, k in enumerate(ks):
                X = np.asarray(recs[k]["xyz_A"]) * 0.1
                g = np.asarray(cm.batch_grad(n)(jnp.asarray(X[None]), P)[0])
                recs[k]["grad_int"] = (g / (KCAL / 0.1)).tolist()  # kcal/mol/A
        dip = float(np.linalg.norm(cm.monomer_dipole(P)) / 0.020819434)
        pol = float(cm.monomer_polarizability(P) / 1e-3)
        data.monomer.update(dipole_D=dip, polarizability_A3=pol)
    return data


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(3)
    w = water()
    cm = ClusterModel(w, monomer_xyz_nm=W0 * 0.1)
    return rng, w, cm


def test_dataset_io_roundtrip(tmp_path, setup):
    rng, w, cm = setup
    d = synthetic_set(rng, cm=cm, P=cm.table.initial())
    p = tmp_path / "set.json"
    d.save(str(p))
    e = QMSet.load(str(p))
    assert e.ids == d.ids and e.sets() == ["dimers", "tetramers", "trimers"]
    assert label(e.records[0], "E.ref") == pytest.approx(d.records[0]["E"]["ref"])
    assert np.isnan(label(e.records[0], "E.nothing"))
    tr, te = e.split(lambda r: r["set"] == "tetramers")
    assert len(tr) == 9 and len(te) == 1


def test_committed_dataset_loads():
    path = os.path.join(ROOT, "data/qm/water_qm.json")
    if not os.path.exists(path):
        pytest.skip("data/qm/water_qm.json not built")
    d = QMSet.load(path)
    assert len(d) > 100
    r = next(r for r in d.records if r["id"] == "smith/Cs_open")
    assert -5.5 < r["E"]["ref"] < -4.5  # water dimer minimum, CCSD(T)/CBS ~ -5.0 kcal/mol
    X = np.asarray(r["xyz_A"]).reshape(-1, 3, 3)
    assert np.allclose(np.linalg.norm(X[:, 1] - X[:, 0], axis=1), 0.9745, atol=1e-6)


def test_components_match_model_and_nbody(setup):
    rng, w, cm = setup
    P = cm.table.initial()
    X = cluster(rng, 3) * 0.1
    s3 = System([w] * 3)
    model = Model([ElecChannel(), LJChannel()])
    E = model.energy_fn(s3)
    E1 = model.energy_fn(System([w]))
    ref = E(jnp.asarray(X), P)["total"] - sum(E1(jnp.asarray(X[3 * k : 3 * k + 3]), P)["total"] for k in range(3))
    c = cm.components(jnp.asarray(X), P, 3)
    assert float(c["total"]) == pytest.approx(float(ref), rel=1e-10, abs=1e-9)
    assert float(c["ind"]) <= 0
    assert float(c["elst"] + c["ind"] + c["vdw"]) == pytest.approx(float(c["total"]), rel=1e-12)
    # 3-body of the Prepared machinery vs Model.nbody (inclusion-exclusion)
    rec = {"id": "t", "set": "t", "n": 3, "xyz_A": (X * 10).tolist(), "nb": {"nb3": 0.0}}
    prep = Prepared(QMSet([rec]), cm)
    pr = prep.predict(P)
    nb = model.nbody(s3, X[None], P, order=3)
    assert float(pr["nb3"][0]) == pytest.approx(float(nb["nb3"]["total"][0]), rel=1e-8, abs=1e-9)
    assert float(pr["nb2"][0]) == pytest.approx(float(nb["nb2"]["total"][0]), rel=1e-10)
    # pairwise channels have no 3-body part: the model's 3-body energy is pure induction
    assert abs(float(nb["nb3"]["vdw"][0])) < 1e-10 and abs(float(nb["nb3"]["perm"][0])) < 1e-9


def test_param_map_keeps_neutrality(setup):
    rng, w, cm = setup
    pm = ParamMap(cm.table, [w], {"q": "all", "cov": "all", "radius": "all", "alpha": ["OW"], "lj_rmin_half": ["OW"]})
    assert len(pm) == 1 + 2 + 2 + 1 + 1
    P = pm.params(jnp.asarray(pm.theta0))
    for k in P:
        assert np.allclose(np.asarray(P[k]), np.asarray(cm.table.initial()[k]))
    th = pm.theta0 + rng.normal(size=len(pm)) * pm.scale
    q = np.asarray(System([w]).expand(pm.params(jnp.asarray(th)))["q"])
    assert abs(q.sum()) < 1e-12 and abs(q[0] - w.q[0]) > 1e-4


def test_rigid_body_forces_and_superposition(setup):
    rng, w, cm = setup
    # monomer energies exert no net force or torque on a rigid molecule
    X = jnp.asarray(cluster(rng, 2) * 0.1)
    s1 = System([w])
    E1 = Model([ElecChannel(), LJChannel()]).energy_fn(s1)
    g = jax.grad(lambda x: E1(x[:3], None)["total"] + E1(x[3:], None)["total"])(X)
    F, T = rigid_body_forces(g, X, w.masses, 3)
    assert float(jnp.max(jnp.abs(F))) < 1e-8 and float(jnp.max(jnp.abs(T))) < 1e-9
    # net force from the interaction gradient = minus the derivative along a rigid translation
    g = cm.batch_grad(2)(X[None], None)[0]
    F, _ = rigid_body_forces(g, X, w.masses, 3)
    h, u = 1e-5, jnp.array([0.3, -0.5, 0.8])
    e = lambda t: cm.components(X.at[3:].add(t * u), None, 2)["total"]  # noqa: E731
    assert float(F[1] @ u) == pytest.approx(-float((e(h) - e(-h)) / (2 * h)), rel=1e-6)
    # superposition onto another rigid geometry keeps the centres of mass
    Y = superpose_monomers(np.asarray(X) * 10, rigid_water(0.9572, 104.52), w.masses)
    for k in range(2):
        a, b = np.asarray(X[3 * k : 3 * k + 3]) * 10, Y[3 * k : 3 * k + 3]
        assert np.allclose(w.masses @ a / w.masses.sum(), w.masses @ b / w.masses.sum(), atol=1e-10)
        assert np.linalg.norm(b[1] - b[0]) == pytest.approx(0.9572)


def test_loss_gradient_matches_finite_differences(setup):
    rng, w, cm = setup
    P0 = cm.table.initial()
    Pt = dict(P0)
    Pt["alpha"] = P0["alpha"] * 1.1
    Pt["lj_sqrt_eps"] = P0["lj_sqrt_eps"] * 0.9
    data = synthetic_set(np.random.default_rng(5), cm=cm, P=Pt)
    pm = ParamMap(
        cm.table,
        [w],
        {"q": "all", "cov": "all", "radius": "all", "alpha": "all", "lj_rmin_half": ["OW"], "lj_sqrt_eps": ["OW"]},
    )
    fw = FitWeights(total=1, elst=0.5, ind=0.5, exch_disp=0.5, nb3=1, force=0.1, dipole=1, polarizability=1, prior=0.1)
    fit = QMFit(cm, pm, data, fw)
    th = jnp.asarray(pm.theta0 + 0.3 * rng.normal(size=len(pm)) * pm.scale)
    g = np.asarray(jax.grad(fit.loss)(th))
    for i in range(len(pm)):
        h = 1e-3 * pm.scale[i]
        e = np.zeros(len(pm))
        e[i] = h
        fd = (float(fit.loss(th + e)) - float(fit.loss(th - e))) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=2e-5, abs=1e-6 * max(1.0, abs(fd))), pm.names[i]


def test_fit_recovers_synthetic_target_exactly(setup):
    """Labels made by the model at perturbed parameters; the fit from the original parameters
    (no prior) finds them again and the residual vanishes."""
    rng, w, cm = setup
    P0 = cm.table.initial()
    free = {"q": "all", "cov": "all", "alpha": "all", "lj_rmin_half": ["OW"], "lj_sqrt_eps": ["OW"]}
    pm = ParamMap(cm.table, [w], free)
    th_true = pm.theta0 + np.array([0.02, 0.001, -0.0005, 1e-4, -3e-5, 0.004, -0.05])
    Pt = pm.params(jnp.asarray(th_true))
    data = synthetic_set(np.random.default_rng(11), cm=cm, P=Pt)
    fw = FitWeights(total=1, elst=1, ind=1, exch_disp=1, nb3=1, force=0.1, dipole=1, polarizability=1, prior=0.0)
    fit = QMFit(cm, pm, data, fw)
    assert float(fit.loss(jnp.asarray(pm.theta0))) > 1.0
    res = fit.fit(max_nfev=100, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    assert float(fit.loss(jnp.asarray(res.x))) < 1e-12
    assert np.allclose(res.x, th_true, rtol=1e-5, atol=1e-6 * pm.scale.max())
    ev = evaluate(cm, data, pm.params(jnp.asarray(res.x)))
    rows = error_table(ev)
    assert all(r["RMSE"] < 1e-5 for r in rows)


def test_committed_fit_reproduces_its_report():
    """data/qm/fits/all_total.json (LJ water fitted to the set) gives the dimer energy of its report."""
    fitp, datap = os.path.join(ROOT, "data/qm/fits/all_total.json"), os.path.join(ROOT, "data/qm/water_qm.json")
    if not (os.path.exists(fitp) and os.path.exists(datap)):
        pytest.skip("fit or data set not present")
    from pgm_jax.param import load_molecule

    w = load_molecule(fitp)
    cm = ClusterModel(w)
    d = QMSet.load(datap).select(lambda r: r["id"] in ("smith/Cs_open", "water27/H2O6"))
    ev = evaluate(cm, d, cm.table.initial())
    e = dict(zip(ev["ids"], ev["total"]))
    assert e["smith/Cs_open"] == pytest.approx(-4.926, abs=2e-3)
    assert e["water27/H2O6"] == pytest.approx(-42.94, abs=1e-2)
    assert abs(float(np.sum(w.q))) < 1e-10
