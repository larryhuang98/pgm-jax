"""Virtual sites: massless interaction sites placed from parent atoms of the same molecule.

A virtual site is an atom of its `Molecule` (it carries a charge and, like any atom, may carry a
pGM Gaussian radius, a polarizability and van der Waals parameters) whose mass is zero and whose
position is a function of other atoms of the molecule, its parents.  `Molecule.vsites` lists the
sites of a molecule (`VirtualSite`, local atom indices); `VirtualSites.of(system)` flattens those
of a `System` for vectorised construction and force spreading.

Kinds (the conventions of OpenMM, GROMACS and Amber).  The first parent, atoms[0], is the site's
host; every construction uses the minimum-image displacements d_k = mi(r_k - r_host) from the host,
so molecules may straddle the box, and is translation invariant:

  "average2"    two-particle average (OpenMM TwoParticleAverageSite, GROMACS virtual_sites2):
                r = w_a r_a + w_b r_b with w_a + w_b = 1, evaluated as r_a + w_b d_b; w_b outside
                [0, 1] extrapolates along the bond (e.g. a sigma-hole site beyond a halogen).
  "average3"    three-particle average (ThreeParticleAverageSite; TIP4P's M site with weights
                (1 - 2a, a, a), `VirtualSite.tip4p`): r = w_a r_a + w_b r_b + w_c r_c, weights
                summing to 1, evaluated as r_a + w_b d_b + w_c d_c.
  "outofplane"  OpenMM OutOfPlaneSite (GROMACS virtual_sites3 3out; TIP5P's lone pairs):
                r = r_a + w_ab d_b + w_ac d_c + w_x (d_b x d_c), w_x in nm^-1.
  "local"       OpenMM LocalCoordinatesSite: origin o = sum_k wo_k r_k (sum wo = 1), directions
                x = sum_k wx_k r_k and y = sum_k wy_k r_k (sum wx = sum wy = 0); e_x = x / |x|,
                e_z = (x cross y) / |x cross y|, e_y = e_z cross e_x;
                r = o + p_x e_x + p_y e_y + p_z e_z  (p in nm).
  "amber"       Amber extra-point frame (sander / pmemd extra_pts, after Stone & Alderton, Mol.
                Phys. 56, 1047 (1985)): the host B and two points A = sum_k wa_k r_k,
                C = sum_k wc_k r_k (atoms, or bond midpoints for carbonyl oxygens);
                u = unit(A - B), v = unit(C - B), e_z = -unit(u + v), e_x = unit(v - u),
                e_y = e_z cross e_x;  r = r_B + p_x e_x + p_y e_y + p_z e_z.  TIP4P-Ew's EP is
                p = (0, 0, -d_OM): on the HOH bisector at d_OM from the oxygen, for any geometry.
                `amber_extra_points` infers these frames from a prmtop's bond graph by Amber's rules.

Forces.  With the site positions s(R) a function of the real atoms R, the energy is U(R, s(R)) and
the force on the real atoms is F_R + (ds/dR)^T F_s: the vector-Jacobian product of the construction
(`VirtualSites.spread`, by automatic differentiation of `place`), exact for every kind, so energy
is conserved.  Site forces enter the molecular virial unchanged: under the molecular scaling of
the virial and of the Monte Carlo barostat a molecule is translated rigidly and every
construction is translation invariant, so the sites move with their molecule.  (An atomic, affine
virial would need the sites rebuilt from the deformed parents: PGMForceField.strain_derivative
refuses molecular=False with virtual sites.)

Integration.  Rigid molecules (`Simulation`): a site is one more point of the rigid template,
at zero mass; its position is placed once from the parents of the first instance (all
constructions are rigid functions of a rigid molecule), and its force enters the body force and
torque like any other.  Flexible molecules (`FlexibleSimulation`): sites are not integrated (no
momentum; excluded from constraints, thermostat, kinetic energy, degrees of freedom and hydrogen
mass repartitioning); they are rebuilt after every position update (after SHAKE) and their forces
are spread to the parents before every momentum update.

Pair topology (md/topology.py): a site is part of its host.  It shares the host's neighbour-list
group and takes the host's graph distances for the van der Waals weights (so it is excluded from
its host and from every atom its host is excluded from, as Amber's extra points; a site-host
bond in `Molecule.bonds`, as Amber's topologies have, is not a bond of the graph).
Electrostatics has no exclusions (pGM): sites interact with every atom, also within the molecule.

Covalent dipoles (p_i += c unit(r_j - r_i)) may have a site as i or j: their gradient reaches the
site's position and is spread with the other site forces.  The two points must not coincide
(checked at setup, `VirtualSites.check`).

Units nm; the parameters of each kind are dimensionless except w_x (nm^-1) and p (nm)."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .box import min_image

KINDS = ("average2", "average3", "outofplane", "local", "amber")
POINT_RADIUS = 1e-4          # nm: Gaussian radius of a point charge (exponent 1/sqrt(2 (R_i^2 + R_j^2)) >= 5000 nm^-1)
_SUM_TOL = 1e-9
_DEGENERATE = 1e-6           # nm: smallest frame vector / covalent-dipole distance accepted at setup


def _tuple(x):
    """Nested lists / arrays -> nested tuples of floats (hashable, JSON-friendly)."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return tuple(_tuple(v) for v in x)
    return float(x)


@dataclass(frozen=True)
class VirtualSite:
    """One virtual site of a molecule: `site` is the site's atom (local index), `atoms` its parents
    with the host first, `params` the kind's parameters (module docstring):
        average2    (w_a, w_b)
        average3    (w_a, w_b, w_c)
        outofplane  (w_ab, w_ac, w_x)
        local       ((wo_k), (wx_k), (wy_k), (p_x, p_y, p_z))    one weight per parent
        amber       ((wa_k), (wc_k), (p_x, p_y, p_z))            one weight per parent; the host is B
    Use the constructors (average2, average3, out_of_plane, local, amber, tip4p)."""
    site: int
    kind: str
    atoms: tuple
    params: tuple

    def __post_init__(self):
        object.__setattr__(self, "site", int(self.site))
        object.__setattr__(self, "atoms", tuple(int(a) for a in self.atoms))
        object.__setattr__(self, "params", _tuple(self.params))
        k, a, p = self.kind, self.atoms, self.params
        if k not in KINDS:
            raise ValueError(f"unknown virtual-site kind {k!r}; kinds are {KINDS}")
        if self.site in a:
            raise ValueError(f"virtual site {self.site} is one of its own parents {a}")
        if len(set(a)) != len(a):
            raise ValueError(f"virtual site {self.site}: repeated parents {a}")
        n = {"average2": 2, "average3": 3, "outofplane": 3}.get(k)
        if n is not None:
            if len(a) != n or len(p) != n:
                raise ValueError(f"{k} site {self.site}: {n} parents and {n} parameters, got {a}, {p}")
            if k != "outofplane" and abs(sum(p) - 1.0) > _SUM_TOL:
                raise ValueError(f"{k} site {self.site}: weights {p} must sum to 1 (translation invariance)")
        elif k == "local":
            if len(a) < 2 or len(p) != 4 or any(len(w) != len(a) for w in p[:3]) or len(p[3]) != 3:
                raise ValueError(f"local site {self.site}: >= 2 parents, params ((wo), (wx), (wy), (px, py, pz)) "
                                 f"with one weight per parent; got {a}, {p}")
            for name, w, target in (("origin", p[0], 1.0), ("x", p[1], 0.0), ("y", p[2], 0.0)):
                if abs(sum(w) - target) > _SUM_TOL:
                    raise ValueError(f"local site {self.site}: {name} weights {w} must sum to {target:g}")
        else:                                                        # amber
            if len(a) < 2 or len(p) != 3 or any(len(w) != len(a) for w in p[:2]) or len(p[2]) != 3:
                raise ValueError(f"amber site {self.site}: >= 2 parents, params ((wa), (wc), (px, py, pz)); got {a}, {p}")
            for name, w in (("A", p[0]), ("C", p[1])):
                if abs(sum(w) - 1.0) > _SUM_TOL:
                    raise ValueError(f"amber site {self.site}: weights of {name} {w} must sum to 1")

    @property
    def host(self) -> int:
        return self.atoms[0]

    # ------------------------------------------------------------------ constructors
    @classmethod
    def average2(cls, site, a, b, w_a, w_b):
        return cls(site, "average2", (a, b), (w_a, w_b))

    @classmethod
    def average3(cls, site, a, b, c, w_a, w_b, w_c):
        return cls(site, "average3", (a, b, c), (w_a, w_b, w_c))

    @classmethod
    def out_of_plane(cls, site, a, b, c, w_ab, w_ac, w_x):
        return cls(site, "outofplane", (a, b, c), (w_ab, w_ac, w_x))

    @classmethod
    def local(cls, site, atoms, origin_weights, x_weights, y_weights, p):
        return cls(site, "local", tuple(atoms), (tuple(origin_weights), tuple(x_weights), tuple(y_weights), tuple(p)))

    @classmethod
    def amber(cls, site, center, first, third, p, middle=None):
        """Amber frame on `center`: A = first, C = third (frame type 1), or, with `middle` (the
        carbonyl carbon, frame type 2), A = (first + middle) / 2, C = (third + middle) / 2."""
        if middle is None:
            return cls(site, "amber", (center, first, third), ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), tuple(p)))
        return cls(site, "amber", (center, first, middle, third),
                   ((0.0, 0.5, 0.5, 0.0), (0.0, 0.0, 0.5, 0.5), tuple(p)))

    @classmethod
    def tip4p(cls, site, o, h1, h2, d_om, r_oh=0.09572, theta_deg=104.52):
        """TIP4P-type M site on the bisector at d_om (nm) from the oxygen of a rigid water (r_oh nm,
        theta_deg): three-particle average (1 - 2a, a, a), a = d_om / (2 r_oh cos(theta / 2)).
        TIP4P-Ew: d_om = 0.0125 nm (a = 0.10667672)."""
        a = float(d_om) / (2.0 * float(r_oh) * np.cos(np.radians(float(theta_deg)) / 2.0))
        return cls.average3(site, o, h1, h2, 1.0 - 2.0 * a, a, a)

    # ------------------------------------------------------------------ serialisation
    def to_list(self) -> list:
        return [self.site, self.kind, list(self.atoms), _lists(self.params)]

    @classmethod
    def from_list(cls, x) -> "VirtualSite":
        return cls(int(x[0]), str(x[1]), tuple(x[2]), x[3])

    def shifted(self, offset: int) -> "VirtualSite":
        """The same site with atom indices shifted by `offset` (local -> global, and back)."""
        return VirtualSite(self.site + offset, self.kind, tuple(a + offset for a in self.atoms), self.params)


def _lists(x):
    return [_lists(v) for v in x] if isinstance(x, tuple) else x


# ----------------------------------------------------------------------------- construction kernels
def _unit(v):
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)


def _disp(pos, parents, H):
    """(n, P, 3) displacements of the parents from the host (column 0), minimum image if H."""
    d = pos[parents] - pos[parents[:, :1]]
    return d if H is None else min_image(d, H)


def _linear(pos, H, par, w):
    """average2 / average3 / outofplane: r_a + w_b d_b + w_c d_c + w_x (d_b x d_c)."""
    d = _disp(pos, par, H)
    db, dc = d[:, 1], d[:, 2]
    return pos[par[:, 0]] + w[:, 0:1] * db + w[:, 1:2] * dc + w[:, 2:3] * jnp.cross(db, dc)


def _local(pos, H, par, wo, wx, wy, p):
    """OpenMM LocalCoordinatesSite in host-relative form (valid because sum wo = 1, sum wx = sum wy = 0)."""
    d = _disp(pos, par, H)
    o = pos[par[:, 0]] + jnp.einsum("nk,nkc->nc", wo, d)
    x = jnp.einsum("nk,nkc->nc", wx, d)
    y = jnp.einsum("nk,nkc->nc", wy, d)
    ex = _unit(x)
    ez = _unit(jnp.cross(x, y))
    ey = jnp.cross(ez, ex)
    return o + p[:, 0:1] * ex + p[:, 1:2] * ey + p[:, 2:3] * ez


def _amber(pos, H, par, wa, wc, p):
    """Amber extra-point frame (sander extra_pts.F90 do_local_global), host = B."""
    d = _disp(pos, par, H)
    u = _unit(jnp.einsum("nk,nkc->nc", wa, d))
    v = _unit(jnp.einsum("nk,nkc->nc", wc, d))
    ez = -_unit(0.5 * (u + v))
    ex = _unit(0.5 * (v - u))
    ey = jnp.cross(ez, ex)
    return pos[par[:, 0]] + p[:, 0:1] * ex + p[:, 1:2] * ey + p[:, 2:3] * ez


def _frame_norms(pos, H, kind, par, *w):
    """Smallest norm of the vectors a frame normalises (setup check against degenerate frames)."""
    d = _disp(pos, par, H)
    if kind == "local":
        x = jnp.einsum("nk,nkc->nc", w[1], d)
        y = jnp.einsum("nk,nkc->nc", w[2], d)
        vs = [x, jnp.cross(x, y) / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-300)]
    else:
        A, C = jnp.einsum("nk,nkc->nc", w[0], d), jnp.einsum("nk,nkc->nc", w[1], d)
        u, v = _unit(A), _unit(C)
        vs = [A, C, 0.5 * (u + v) * 0.1, 0.5 * (v - u) * 0.1]           # unit-vector sums scaled to 0.1 nm
    return min(float(jnp.min(jnp.linalg.norm(x, axis=-1))) for x in vs)


# ----------------------------------------------------------------------------- the sites of a system
class VirtualSites:
    """The virtual sites of a System (`VirtualSites.of(sys)`, None without sites): placement of the
    site positions from the parents and spreading of site forces to the parents, both vectorised
    per kind and jit-compatible (static index arrays).  Indices are global (system order)."""

    def __init__(self, sys):
        entries = []
        for k, m in enumerate(sys.molecules):
            off = int(sys.offsets[k])
            for vs in getattr(m, "vsites", None) or ():
                if not isinstance(vs, VirtualSite):
                    raise TypeError(f"{m.name}: Molecule.vsites holds VirtualSite objects, got {type(vs).__name__}")
                if not all(0 <= a < m.n for a in (vs.site,) + vs.atoms):
                    raise ValueError(f"{m.name}: virtual site {vs.site} with parents {vs.atoms} outside the molecule ({m.n} atoms)")
                entries.append((m.name, vs.shifted(off)))
        if not entries:
            raise ValueError("the system has no virtual sites (use VirtualSites.of, which returns None)")
        self.n = int(sys.n)
        masses = np.asarray(sys.masses, float)
        site = np.array([vs.site for _, vs in entries], np.int32)
        if len(set(site.tolist())) != len(site):
            raise ValueError("an atom is defined as a virtual site more than once")
        is_site = np.zeros(self.n, bool)
        is_site[site] = True
        for name, vs in entries:
            if masses[vs.site] != 0.0:
                raise ValueError(f"{name}: virtual site {vs.site} (global index) has mass {masses[vs.site]:g}; "
                                 "virtual sites are massless")
            bad = [a for a in vs.atoms if is_site[a] or not masses[a] > 0.0]
            if bad:
                raise ValueError(f"{name}: virtual site {vs.site} has parents {bad} that are virtual sites or massless "
                                 "(sites are built from real atoms only)")
        others = np.nonzero(~is_site & ~(masses > 0.0))[0]
        if len(others):
            raise ValueError(f"atoms {others[:10].tolist()} have no mass but are not virtual sites")
        self.site = site
        self.host = np.array([vs.host for _, vs in entries], np.int32)
        self.kinds = tuple(sorted({vs.kind for _, vs in entries}))
        self.is_site = is_site
        self.real = ~is_site
        self.n_sites = int(len(site))
        self.sites = [vs for _, vs in entries]
        # kernels: (function, site indices, static arrays)
        self._kernels = []
        lin = [vs for vs in self.sites if vs.kind in ("average2", "average3", "outofplane")]
        if lin:
            par, w = [], []
            for vs in lin:
                a, p = vs.atoms, vs.params
                if vs.kind == "average2":
                    par.append((a[0], a[1], a[1])); w.append((p[1], 0.0, 0.0))
                elif vs.kind == "average3":
                    par.append(a); w.append((p[1], p[2], 0.0))
                else:
                    par.append(a); w.append(p)
            self._kernels.append((_linear, np.array([vs.site for vs in lin], np.int32),
                                  (np.array(par, np.int32), np.array(w, float))))
        for kind, fn in (("local", _local), ("amber", _amber)):
            group = [vs for vs in self.sites if vs.kind == kind]
            if not group:
                continue
            P = max(len(vs.atoms) for vs in group)
            nw = 3 if kind == "local" else 2
            par = np.array([vs.atoms + (vs.atoms[0],) * (P - len(vs.atoms)) for vs in group], np.int32)
            ws = [np.array([vs.params[j] + (0.0,) * (P - len(vs.atoms)) for vs in group], float) for j in range(nw)]
            p = np.array([vs.params[nw] for vs in group], float)
            self._kernels.append((fn, np.array([vs.site for vs in group], np.int32), (par, *ws, p)))
        self._order = np.concatenate([idx for _, idx, _ in self._kernels])

    @classmethod
    def of(cls, sys) -> "VirtualSites | None":
        """The system's virtual sites, or None when no molecule has any."""
        if not any(getattr(m, "vsites", None) for m in sys.molecules):
            return None
        return cls(sys)

    def __repr__(self) -> str:
        return f"VirtualSites({self.n_sites} sites: {', '.join(self.kinds)})"

    # ------------------------------------------------------------------ placement and forces
    def positions(self, pos, H=None):
        """(n_sites, 3) site positions in the order of self._order."""
        pos = jnp.asarray(pos)
        Hj = None if H is None else jnp.asarray(H, pos.dtype)
        return jnp.concatenate([fn(pos, Hj, *(jnp.asarray(a) for a in arrs)) for fn, _, arrs in self._kernels])

    def place(self, pos, H=None):
        """Positions with every site rebuilt from its parents (H: box for the minimum image, or
        None for an isolated molecule).  Differentiable; the site rows of `pos` are ignored."""
        pos = jnp.asarray(pos)
        return pos.at[self._order].set(self.positions(pos, H))

    def spread(self, pos, H, forces):
        """Forces on the real atoms: every site's force moved to its parents by the transposed
        Jacobian of the construction (vector-Jacobian product of `place` at pos); the site rows
        of the result are zero.  Total force and torque about any point are conserved for the
        linear kinds; for every kind the work of any virtual displacement of the parents is."""
        _, pull = jax.vjp(lambda x: self.place(x, H), jnp.asarray(pos))
        return pull(jnp.asarray(forces, jnp.asarray(pos).dtype))[0]

    # ------------------------------------------------------------------ checks
    def check(self, pos, H=None, cov_pairs=None) -> None:
        """Setup checks at positions pos (nm): no degenerate frame (the vectors a "local" or "amber"
        frame normalises are longer than 1e-6 nm), and no covalent dipole (cov_pairs: (i, j)
        arrays, global) between two points closer than 1e-6 nm, which has no direction."""
        pos = jnp.asarray(pos, jnp.float64)
        Hj = None if H is None else jnp.asarray(H, jnp.float64)
        placed = self.place(pos, Hj)
        for fn, idx, arrs in self._kernels:
            if fn is _linear:
                continue
            kind = "local" if fn is _local else "amber"
            mn = _frame_norms(placed, Hj, kind, *(jnp.asarray(a) for a in arrs[:-1]))
            if not mn > _DEGENERATE:
                raise ValueError(f"degenerate {kind} virtual-site frame (a frame vector of {mn:.3g} nm)")
        if cov_pairs is not None and len(cov_pairs[0]):
            ci, cj = (np.asarray(c, int) for c in cov_pairs)
            touch = self.is_site[ci] | self.is_site[cj]
            if np.any(touch):
                d = placed[cj[touch]] - placed[ci[touch]]
                d = d if Hj is None else min_image(d, Hj)
                r = np.linalg.norm(np.asarray(d), axis=-1)
                if np.any(r < _DEGENERATE):
                    k = int(np.argmin(r))
                    raise ValueError(f"covalent dipole between atoms {int(ci[touch][k])} and {int(cj[touch][k])} "
                                     f"(a virtual site) of length {r[k]:.3g} nm has no direction")


# ----------------------------------------------------------------------------- Amber extra points
_TET = np.radians(54.735)
AMBER_EP_TYPE = "EP"


def amber_extra_points(atom_types, bonds_h, bonds_heavy, bond_req) -> dict:
    """Frames of Amber extra points from a topology's bond graph, by the rules of sander and pmemd
    (extra_pts.F90 define_frames; frameon = 1): an extra point is an atom of type "EP" bonded to its
    centre atom, at the equilibrium length of that bond; the frame depends on the centre's other
    neighbours:
      * TIP4P (the centre has two neighbours, both through BONDS_INC_HYDROGEN, and one EP): the EP
        on the H-O-H bisector toward the hydrogens, p = (0, 0, -req);
      * two EPs on such a centre (TIP5P), or on a centre with two heavy neighbours, or one heavy and
        one hydrogen neighbour: tetrahedral lone pairs in the local zy plane, p = (0, +-sin(54.735)
        req, cos(54.735) req) (for atom types S / SH: (0, +-req, 0)); a single EP: p = (0, 0, req);
      * a centre with one heavy neighbour and no hydrogen (carbonyl oxygen): the frame of the bond
        midpoints of the carbon's other two bonds, EPs at 60 degrees in the xz plane
        (p = (+-sin 60 req, 0, cos 60 req)) or one EP at (0, 0, req).
    Neighbour order is the order of the bond lists, as in Amber (it fixes which EP is which).
    atom_types: per atom; bonds_h / bonds_heavy: (i, j, bond type) 0-based, as BONDS_INC_HYDROGEN /
    BONDS_WITHOUT_HYDROGEN; bond_req: equilibrium lengths per bond type (Angstrom).
    Returns {ep atom: VirtualSite with global indices, p in nm}; raises where Amber would stop."""
    types = [str(t).strip() for t in atom_types]
    n = len(types)
    is_ep = np.array([t == AMBER_EP_TYPE for t in types])
    if not is_ep.any():
        return {}
    heavy = [[] for _ in range(n)]
    hyd = [[] for _ in range(n)]
    eps = [[] for _ in range(n)]
    for i, j, t in bonds_heavy:
        i, j = int(i), int(j)
        if is_ep[i] or is_ep[j]:
            e, c = (i, j) if is_ep[i] else (j, i)
            if is_ep[c]:
                raise ValueError(f"extra points {i + 1} and {j + 1} are bonded to each other")
            eps[c].append((e, float(bond_req[int(t)])))
        else:
            heavy[i].append(j); heavy[j].append(i)
    for i, j, _ in bonds_h:
        i, j = int(i), int(j)
        if is_ep[i] or is_ep[j]:
            raise ValueError(f"extra point in BONDS_INC_HYDROGEN ({i + 1}-{j + 1}); Amber reads EP bonds from "
                             "BONDS_WITHOUT_HYDROGEN")
        hyd[i].append(j); hyd[j].append(i)
    out = {}
    s, c60 = np.sin(np.radians(60.0)), 0.5
    for c in range(n):
        if not eps[c]:
            continue
        nh, nx, ne = len(heavy[c]), len(hyd[c]), len(eps[c])
        where = f"atom {c + 1} ({types[c]}): {nx} hydrogen, {nh} heavy, {ne} extra-point neighbours"
        if ne > 2:
            raise ValueError(f"more than two extra points on {where}")
        if nh + nx > 2:
            raise ValueError(f"Amber extra points: too many neighbours of {where}")
        req = [r * 0.1 for _, r in eps[c]]                               # nm
        middle = None
        if nh == 0 and nx == 2:                                        # water (TIP4P, TIP5P)
            first, third = hyd[c][0], hyd[c][1]
            tip4p = ne == 1
        elif nh > 1:
            first, third, tip4p = heavy[c][0], heavy[c][1], False
        elif nh == 1 and nx == 1:
            first, third, tip4p = heavy[c][0], hyd[c][0], False
        elif nh == 1 and nx == 0:                                      # carbonyl oxygen: frame type 2
            m = heavy[c][0]
            if len(heavy[m]) != 3 or len(hyd[m]) > 0:
                raise ValueError(f"Amber extra points (carbonyl frame): atom {m + 1} bonded to {where} must have "
                                 f"three heavy neighbours and no hydrogen")
            other = [a for a in heavy[m] if a != c]
            first, third, middle, tip4p = other[0], other[1], m, False
        else:
            raise ValueError(f"Amber extra points: unexpected neighbours of {where}")
        if middle is None:
            if ne == 1:
                ps = [(0.0, 0.0, -req[0] if tip4p else req[0])]
            elif types[c] in ("S", "SH"):
                ps = [(0.0, req[0], 0.0), (0.0, -req[1], 0.0)]
            else:
                ps = [(0.0, np.sin(_TET) * req[0], np.cos(_TET) * req[0]),
                      (0.0, -np.sin(_TET) * req[1], np.cos(_TET) * req[1])]
        else:
            ps = ([(0.0, 0.0, req[0])] if ne == 1 else
                  [(s * req[0], 0.0, c60 * req[0]), (-s * req[1], 0.0, c60 * req[1])])
        for (e, _), p in zip(eps[c], ps):
            out[e] = VirtualSite.amber(e, c, first, third, p, middle=middle)
    return out
