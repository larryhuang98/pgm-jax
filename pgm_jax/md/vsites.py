"""Virtual sites: massless interaction sites placed from parent atoms of the same molecule.

Contents: `VirtualSite` (one site of a Molecule: kind, parents, parameters; constructors),
`VirtualSites` (the sites of a System: vectorised placement, force spreading, setup checks), the
construction kernels `_linear`, `_local`, `_amber`, and `amber_extra_points` (Amber's extra-point
frames from a prmtop's bond graph).

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
  "amber"       Amber extra-point frame (sander / pmemd extra_pts, after [1]_): the host B and
                two points A = sum_k wa_k r_k,
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
mass repartitioning).  With zero momentum the drifts leave them in place; they are rebuilt once
per step, after the last drift (after SHAKE) and before the forces, and their forces are spread to
the parents before every momentum update.

Pair topology (md/topology.py): a site is part of its host.  It shares the host's neighbour-list
group and takes the host's graph distances for the van der Waals weights (so it is excluded from
its host and from every atom its host is excluded from, as Amber's extra points; a site-host
bond in `Molecule.bonds`, as Amber's topologies have, is not a bond of the graph).
Electrostatics has no exclusions (pGM): sites interact with every atom, also within the molecule.

Covalent dipoles (p_i += c unit(r_j - r_i)) may have a site as i or j: their gradient reaches the
site's position and is spread with the other site forces.  The two points must not coincide
(checked at setup, `VirtualSites.check`).

Units: nm; the parameters of each kind are dimensionless except w_x (nm^-1) and p (nm).

References
----------
.. [1] A. J. Stone, M. Alderton, Mol. Phys. 56, 1047 (1985).

See also docs/virtual_sites.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from .box import min_image

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from ..system import System

KINDS = ("average2", "average3", "outofplane", "local", "amber")
POINT_RADIUS = 1e-4  # nm: Gaussian radius of a point charge (exponent 1/sqrt(2 (R_i^2 + R_j^2)) >= 5000 nm^-1)
_SUM_TOL = 1e-9  # tolerance of the weight sums (1 or 0) checked at construction
_DEGENERATE = 1e-6  # nm: smallest frame vector / covalent-dipole distance accepted at setup


def _tuple(x: Any) -> Any:
    """Return nested lists / arrays as nested tuples of floats (hashable, JSON-friendly)."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return tuple(_tuple(v) for v in x)
    return float(x)


@dataclass(frozen=True)
class VirtualSite:
    """One virtual site of a molecule (immutable; indices local to the molecule).

    Use the constructors (average2, average3, out_of_plane, local, amber, tip4p).

    Parameters
    ----------
    site : int
        The site's atom.
    kind : {"average2", "average3", "outofplane", "local", "amber"}
        Construction (module docstring).
    atoms : tuple of int
        Parents, the host first.
    params : tuple
        The kind's parameters (w_x [1/nm], p [nm], weights dimensionless)::

            average2    (w_a, w_b)
            average3    (w_a, w_b, w_c)
            outofplane  (w_ab, w_ac, w_x)
            local       ((wo_k), (wx_k), (wy_k), (p_x, p_y, p_z))    one weight per parent
            amber       ((wa_k), (wc_k), (p_x, p_y, p_z))            one weight per parent; host = B
    """

    site: int
    kind: str
    atoms: tuple
    params: tuple

    def __post_init__(self) -> None:
        """Normalize the fields (ints, tuples of floats) and check them.

        Raises
        ------
        ValueError
            An unknown kind, the site among its parents, repeated parents, wrong numbers of
            parents or parameters, or weights that do not sum to 1 (0 for the "local" directions).
        """
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
                raise ValueError(
                    f"local site {self.site}: >= 2 parents, params ((wo), (wx), (wy), (px, py, pz)) "
                    f"with one weight per parent; got {a}, {p}"
                )
            for name, w, target in (("origin", p[0], 1.0), ("x", p[1], 0.0), ("y", p[2], 0.0)):
                if abs(sum(w) - target) > _SUM_TOL:
                    raise ValueError(f"local site {self.site}: {name} weights {w} must sum to {target:g}")
        else:  # amber
            if len(a) < 2 or len(p) != 3 or any(len(w) != len(a) for w in p[:2]) or len(p[2]) != 3:
                raise ValueError(
                    f"amber site {self.site}: >= 2 parents, params ((wa), (wc), (px, py, pz)); got {a}, {p}"
                )
            for name, w in (("A", p[0]), ("C", p[1])):
                if abs(sum(w) - 1.0) > _SUM_TOL:
                    raise ValueError(f"amber site {self.site}: weights of {name} {w} must sum to 1")

    @property
    def host(self) -> int:
        """The host atom (first parent)."""
        return self.atoms[0]

    # ------------------------------------------------------------------ constructors
    @classmethod
    def average2(cls, site: int, a: int, b: int, w_a: float, w_b: float) -> VirtualSite:
        """Return a two-particle average r = w_a r_a + w_b r_b (w_a + w_b = 1; host a)."""
        return cls(site, "average2", (a, b), (w_a, w_b))

    @classmethod
    def average3(cls, site: int, a: int, b: int, c: int, w_a: float, w_b: float, w_c: float) -> VirtualSite:
        """Return a three-particle average r = w_a r_a + w_b r_b + w_c r_c (weights sum to 1; host a)."""
        return cls(site, "average3", (a, b, c), (w_a, w_b, w_c))

    @classmethod
    def out_of_plane(cls, site: int, a: int, b: int, c: int, w_ab: float, w_ac: float, w_x: float) -> VirtualSite:
        """Return an out-of-plane site r = r_a + w_ab d_b + w_ac d_c + w_x (d_b x d_c) (w_x [1/nm])."""
        return cls(site, "outofplane", (a, b, c), (w_ab, w_ac, w_x))

    @classmethod
    def local(
        cls,
        site: int,
        atoms: Sequence[int],
        origin_weights: Sequence[float],
        x_weights: Sequence[float],
        y_weights: Sequence[float],
        p: Sequence[float],
    ) -> VirtualSite:
        """Return an OpenMM LocalCoordinatesSite (module docstring; p [nm], one weight per parent)."""
        return cls(site, "local", tuple(atoms), (tuple(origin_weights), tuple(x_weights), tuple(y_weights), tuple(p)))

    @classmethod
    def amber(
        cls, site: int, center: int, first: int, third: int, p: Sequence[float], middle: int | None = None
    ) -> VirtualSite:
        """Return an Amber extra-point frame on `center` (the host B), position p [nm] in the frame.

        Frame type 1: A = first, C = third; with `middle` (the carbonyl carbon, frame type 2):
        A = (first + middle) / 2, C = (third + middle) / 2.
        """
        if middle is None:
            return cls(site, "amber", (center, first, third), ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), tuple(p)))
        return cls(
            site, "amber", (center, first, middle, third), ((0.0, 0.5, 0.5, 0.0), (0.0, 0.0, 0.5, 0.5), tuple(p))
        )

    @classmethod
    def tip4p(
        cls, site: int, o: int, h1: int, h2: int, d_om: float, r_oh: float = 0.09572, theta_deg: float = 104.52
    ) -> VirtualSite:
        """Return a TIP4P-type M site on the bisector at d_om from the oxygen of a rigid water.

        A three-particle average (1 - 2a, a, a), a = d_om / (2 r_oh cos(theta / 2)), exact for the
        water geometry (r_oh [nm], theta_deg [deg]).  TIP4P-Ew: d_om = 0.0125 nm (a = 0.10667672).

        Parameters
        ----------
        site, o, h1, h2 : int
            Local indices of the site, the oxygen (host) and the hydrogens.
        d_om : float
            O-M distance [nm].
        r_oh : float
            O-H bond length [nm].
        theta_deg : float
            H-O-H angle [deg].

        Returns
        -------
        VirtualSite
        """
        a = float(d_om) / (2.0 * float(r_oh) * np.cos(np.radians(float(theta_deg)) / 2.0))
        return cls.average3(site, o, h1, h2, 1.0 - 2.0 * a, a, a)

    # ------------------------------------------------------------------ serialisation
    def to_list(self) -> list:
        """Return [site, kind, atoms, params] as nested lists (JSON)."""
        return [self.site, self.kind, list(self.atoms), _lists(self.params)]

    @classmethod
    def from_list(cls, x: Sequence) -> VirtualSite:
        """Return the site of a `to_list` entry."""
        return cls(int(x[0]), str(x[1]), tuple(x[2]), x[3])

    def shifted(self, offset: int) -> VirtualSite:
        """Return the same site with atom indices shifted by `offset` (local -> global, and back)."""
        return VirtualSite(self.site + offset, self.kind, tuple(a + offset for a in self.atoms), self.params)


def _lists(x: Any) -> Any:
    """Return nested tuples as nested lists (the inverse of _tuple, for JSON)."""
    return [_lists(v) for v in x] if isinstance(x, tuple) else x


# ----------------------------------------------------------------------------- construction kernels
def _unit(v: jax.Array) -> jax.Array:
    """Return v / |v| over the last axis."""
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)


def _rel(X: jax.Array, H: jax.Array | None) -> jax.Array:
    """Return the displacements (n, P, 3) [nm] of the gathered parents X from the host (column 0).

    Minimum image if H is given (None: an isolated molecule).
    """
    d = X - X[:, :1]
    return d if H is None else min_image(d, H)


# Every kernel takes the gathered parents X = pos[parents] (n, P, 3), host first, so that `spread`
# can differentiate it with respect to X alone and move all forces with one scatter-add.
def _linear(X: jax.Array, H: jax.Array | None, w: jax.Array) -> jax.Array:
    """Return average2 / average3 / outofplane sites r_a + w_b d_b + w_c d_c + w_x (d_b x d_c).

    Parameters
    ----------
    X : jax.Array (n, 3, 3)
        Gathered parents (host first; average2 repeats b) [nm].
    H : jax.Array (3, 3) or None
        Box [nm] for the minimum image.
    w : jax.Array (n, 3)
        (w_b, w_c, w_x) per site (average2: (w_b, 0, 0); average3: (w_b, w_c, 0)).

    Returns
    -------
    jax.Array (n, 3)
        Site positions [nm].
    """
    d = _rel(X, H)
    db, dc = d[:, 1], d[:, 2]
    return X[:, 0] + w[:, 0:1] * db + w[:, 1:2] * dc + w[:, 2:3] * jnp.cross(db, dc)


def _local(X: jax.Array, H: jax.Array | None, wo: jax.Array, wx: jax.Array, wy: jax.Array, p: jax.Array) -> jax.Array:
    """Return OpenMM LocalCoordinatesSite positions (n, 3) [nm] in host-relative form.

    Valid because sum wo = 1 and sum wx = sum wy = 0.  X (n, P, 3) gathered parents (host first,
    padded with the host), wo, wx, wy (n, P) weights (0 for padding), p (n, 3) [nm].
    """
    d = _rel(X, H)
    o = X[:, 0] + jnp.einsum("nk,nkc->nc", wo, d)
    x = jnp.einsum("nk,nkc->nc", wx, d)
    y = jnp.einsum("nk,nkc->nc", wy, d)
    ex = _unit(x)
    ez = _unit(jnp.cross(x, y))
    ey = jnp.cross(ez, ex)
    return o + p[:, 0:1] * ex + p[:, 1:2] * ey + p[:, 2:3] * ez


def _amber(X: jax.Array, H: jax.Array | None, wa: jax.Array, wc: jax.Array, p: jax.Array) -> jax.Array:
    """Return Amber extra-point positions (n, 3) [nm] (sander extra_pts.F90 do_local_global), host = B.

    X (n, P, 3) gathered parents (host first, padded with the host), wa, wc (n, P) weights of the
    points A and C, p (n, 3) [nm] in the frame of the module docstring.
    """
    d = _rel(X, H)
    u = _unit(jnp.einsum("nk,nkc->nc", wa, d))
    v = _unit(jnp.einsum("nk,nkc->nc", wc, d))
    ez = -_unit(0.5 * (u + v))
    ex = _unit(0.5 * (v - u))
    ey = jnp.cross(ez, ex)
    return X[:, 0] + p[:, 0:1] * ex + p[:, 1:2] * ey + p[:, 2:3] * ez


def _frame_norms(pos: jax.Array, H: jax.Array | None, kind: str, par: jax.Array, *w: jax.Array) -> float:
    """Return the smallest length [nm] among the vectors a frame normalises (host side).

    Each vector is expressed as a length: for "local" |x| and the part of y perpendicular to x;
    for "amber" |A - B|, |C - B| and 0.1 nm times |u + v| / 2 and |v - u| / 2.  A setup check
    against degenerate frames (NaN counts as 0).

    Parameters
    ----------
    pos : jax.Array (N, 3)
        Positions [nm].
    H : jax.Array (3, 3) or None
        Box [nm].
    kind : {"local", "amber"}
        Frame kind.
    par : jax.Array (n, P) int
        Parents of the sites.
    *w : jax.Array (n, P)
        The kernel's weight arrays (local: wo, wx, wy; amber: wa, wc).

    Returns
    -------
    float
        Smallest length [nm].
    """
    d = _rel(pos[par], H)
    if kind == "local":
        x = jnp.einsum("nk,nkc->nc", w[1], d)
        y = jnp.einsum("nk,nkc->nc", w[2], d)
        lengths = [jnp.linalg.norm(x, axis=-1), jnp.linalg.norm(jnp.cross(_unit(x), y), axis=-1)]
    else:
        A, C = jnp.einsum("nk,nkc->nc", w[0], d), jnp.einsum("nk,nkc->nc", w[1], d)
        u, v = _unit(A), _unit(C)
        lengths = [
            jnp.linalg.norm(A, axis=-1),
            jnp.linalg.norm(C, axis=-1),
            0.05 * jnp.linalg.norm(u + v, axis=-1),
            0.05 * jnp.linalg.norm(v - u, axis=-1),
        ]
    return min(float(jnp.min(jnp.nan_to_num(x))) for x in lengths)


# ----------------------------------------------------------------------------- the sites of a system
class VirtualSites:
    """The virtual sites of a System: placement from the parents and spreading of site forces.

    `VirtualSites.of(sys)` returns None without sites.  Placement and spreading are vectorised per
    kind (one kernel for the linear kinds, one each for "local" and "amber") and jit-compatible
    (static numpy index arrays).  Indices are global (system order).  Not a pytree.

    Attributes
    ----------
    n : int
        Atoms of the system.
    site, host : np.ndarray (n_sites,) int32
        Site atoms and their hosts.
    kinds : tuple of str
        Kinds present.
    is_site, real : np.ndarray (N,) bool
        Site mask and its complement.
    n_sites : int
        Number of sites.
    sites : list of VirtualSite
        The sites (global indices).
    """

    def __init__(self, sys: System) -> None:
        """Collect and check the sites of a system and build the kernels.

        Parameters
        ----------
        sys : System
            The system (Molecule.vsites of every molecule).

        Raises
        ------
        TypeError
            A Molecule.vsites entry that is not a VirtualSite.
        ValueError
            No sites, sites or parents outside their molecule, a site defined twice, a site with
            mass, a parent that is a site or massless, or a massless atom that is not a site.
        """
        entries = []
        for k, m in enumerate(sys.molecules):
            off = int(sys.offsets[k])
            for vs in getattr(m, "vsites", None) or ():
                if not isinstance(vs, VirtualSite):
                    raise TypeError(f"{m.name}: Molecule.vsites holds VirtualSite objects, got {type(vs).__name__}")
                if not all(0 <= a < m.n for a in (vs.site,) + vs.atoms):
                    raise ValueError(
                        f"{m.name}: virtual site {vs.site} with parents {vs.atoms} outside the molecule ({m.n} atoms)"
                    )
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
                raise ValueError(
                    f"{name}: virtual site {vs.site} (global index) has mass {masses[vs.site]:g}; "
                    "virtual sites are massless"
                )
            bad = [a for a in vs.atoms if is_site[a] or not masses[a] > 0.0]
            if bad:
                raise ValueError(
                    f"{name}: virtual site {vs.site} has parents {bad} that are virtual sites or massless "
                    "(sites are built from real atoms only)"
                )
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
        # kernels: (function, site indices, static arrays (parents first, then the kernel's weights))
        self._kernels = []
        lin = [vs for vs in self.sites if vs.kind in ("average2", "average3", "outofplane")]
        if lin:
            par, w = [], []
            for vs in lin:
                a, p = vs.atoms, vs.params
                if vs.kind == "average2":
                    par.append((a[0], a[1], a[1]))
                    w.append((p[1], 0.0, 0.0))
                elif vs.kind == "average3":
                    par.append(a)
                    w.append((p[1], p[2], 0.0))
                else:
                    par.append(a)
                    w.append(p)
            self._kernels.append(
                (_linear, np.array([vs.site for vs in lin], np.int32), (np.array(par, np.int32), np.array(w, float)))
            )
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
        self._order = np.concatenate([idx for _, idx, _ in self._kernels])  # site of each row of `positions`

    @classmethod
    def of(cls, sys: System) -> VirtualSites | None:
        """Return the system's virtual sites, or None when no molecule has any."""
        if not any(getattr(m, "vsites", None) for m in sys.molecules):
            return None
        return cls(sys)

    def __repr__(self) -> str:
        """Return e.g. "VirtualSites(512 sites: average3)"."""
        return f"VirtualSites({self.n_sites} sites: {', '.join(self.kinds)})"

    # ------------------------------------------------------------------ placement and forces
    def positions(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the site positions (n_sites, 3) [nm] in the order of self._order.

        pos (N, 3) [nm]; H (3, 3) [nm] for the minimum image, or None for an isolated molecule.
        """
        pos = jnp.asarray(pos)
        Hj = None if H is None else jnp.asarray(H, pos.dtype)
        return jnp.concatenate(
            [fn(pos[arrs[0]], Hj, *(jnp.asarray(a) for a in arrs[1:])) for fn, _, arrs in self._kernels]
        )

    def place(self, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the positions (N, 3) [nm] with every site rebuilt from its parents.

        H: box for the minimum image, or None for an isolated molecule.  Differentiable; the site
        rows of `pos` are ignored.
        """
        pos = jnp.asarray(pos)
        return pos.at[self._order].set(self.positions(pos, H))

    def spread(self, pos: ArrayLike, H: ArrayLike | None, forces: ArrayLike) -> jax.Array:
        """Return the forces on the real atoms, every site's force moved to its parents.

        The transposed Jacobian of the construction (the vector-Jacobian product of each kernel
        with respect to its gathered parents); the site rows of the result are zero.  One
        scatter-add moves everything (the site rows receive minus their force).  Every
        construction is equivariant under rigid motions, so the total force and the total torque
        about any point are conserved, and so is the work of any displacement of the parents.
        Differentiable.

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Positions [nm] (sites placed).
        H : ArrayLike (3, 3) or None
            Box [nm], None for an isolated molecule.
        forces : ArrayLike (N, 3)
            Forces on all atoms, sites included [kJ/mol/nm].

        Returns
        -------
        jax.Array (N, 3)
            Forces [kJ/mol/nm], zero on the sites.
        """
        pos = jnp.asarray(pos)
        F = jnp.asarray(forces, pos.dtype)
        Hj = None if H is None else jnp.asarray(H, pos.dtype)
        idx, vals = [], []
        for fn, site, arrs in self._kernels:
            par, rest = arrs[0], [jnp.asarray(a) for a in arrs[1:]]
            _, pull = jax.vjp(lambda X: fn(X, Hj, *rest), pos[par])
            Fs = F[site]
            idx += [site, par.reshape(-1)]
            vals += [-Fs, pull(Fs)[0].reshape(-1, 3)]
        return F.at[np.concatenate(idx)].add(jnp.concatenate(vals))

    # ------------------------------------------------------------------ checks
    def check(
        self, pos: ArrayLike, H: ArrayLike | None = None, cov_pairs: tuple[ArrayLike, ArrayLike] | None = None
    ) -> None:
        """Check the sites at positions pos [nm] (host side, at setup).

        No degenerate frame (the vectors a "local" or "amber" frame normalises are longer than
        1e-6 nm), and no covalent dipole (cov_pairs: (i, j) arrays, global) involving a site
        between two points closer than 1e-6 nm, which has no direction.

        Raises
        ------
        ValueError
            A degenerate frame or a covalent dipole without direction.
        """
        pos = jnp.asarray(pos, jnp.float64)
        Hj = None if H is None else jnp.asarray(H, jnp.float64)
        placed = self.place(pos, Hj)
        for fn, _idx, arrs in self._kernels:
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
                    raise ValueError(
                        f"covalent dipole between atoms {int(ci[touch][k])} and {int(cj[touch][k])} "
                        f"(a virtual site) of length {r[k]:.3g} nm has no direction"
                    )


# ----------------------------------------------------------------------------- Amber extra points
_TET = np.radians(54.735)  # half the tetrahedral angle (109.47 / 2), Amber's lone-pair frame
AMBER_EP_TYPE = "EP"


def amber_extra_points(
    atom_types: Sequence[str],
    bonds_h: Sequence[Sequence[int]],
    bonds_heavy: Sequence[Sequence[int]],
    bond_req: Sequence[float],
) -> dict[int, VirtualSite]:
    """Return the frames of Amber extra points from a topology's bond graph.

    By the rules of sander and pmemd (extra_pts.F90 define_frames; frameon = 1): an extra point is
    an atom of type "EP" bonded to its centre atom, at the equilibrium length of that bond; the
    frame depends on the centre's other neighbours:

      * TIP4P (the centre has two neighbours, both through BONDS_INC_HYDROGEN, and one EP): the EP
        on the H-O-H bisector toward the hydrogens, p = (0, 0, -req);
      * two EPs on such a centre (TIP5P), or on a centre with two heavy neighbours, or one heavy and
        one hydrogen neighbour: tetrahedral lone pairs in the local zy plane, p = (0, +-sin(54.735)
        req, cos(54.735) req) (for atom types S / SH: (0, +-req, 0)); a single EP: p = (0, 0, req);
      * a centre with one heavy neighbour and no hydrogen (carbonyl oxygen): the frame of the bond
        midpoints of the carbon's other two bonds, EPs at 60 degrees in the xz plane
        (p = (+-sin 60 req, 0, cos 60 req)) or one EP at (0, 0, req).

    Neighbour order is the order of the bond lists, as in Amber (it fixes which EP is which).

    Parameters
    ----------
    atom_types : Sequence[str] (N,)
        Amber atom types.
    bonds_h, bonds_heavy : Sequence of (i, j, bond type)
        BONDS_INC_HYDROGEN and BONDS_WITHOUT_HYDROGEN, 0-based atom indices.
    bond_req : Sequence[float]
        Equilibrium length per bond type [Angstrom].

    Returns
    -------
    dict of int to VirtualSite
        {extra-point atom: VirtualSite with global indices, p in nm}; empty without extra points.

    Raises
    ------
    ValueError
        Where Amber would stop: bonded extra points, an extra point in BONDS_INC_HYDROGEN, more
        than two extra points or too many / unexpected neighbours of a centre, a carbonyl carbon
        without three heavy neighbours, or an extra point bonded to no atom.
    """
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
            heavy[i].append(j)
            heavy[j].append(i)
    for i, j, _ in bonds_h:
        i, j = int(i), int(j)
        if is_ep[i] or is_ep[j]:
            raise ValueError(
                f"extra point in BONDS_INC_HYDROGEN ({i + 1}-{j + 1}); Amber reads EP bonds from BONDS_WITHOUT_HYDROGEN"
            )
        hyd[i].append(j)
        hyd[j].append(i)
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
        req = [r * 0.1 for _, r in eps[c]]  # nm
        middle = None
        if nh == 0 and nx == 2:  # water (TIP4P, TIP5P)
            first, third = hyd[c][0], hyd[c][1]
            tip4p = ne == 1
        elif nh > 1:
            first, third, tip4p = heavy[c][0], heavy[c][1], False
        elif nh == 1 and nx == 1:
            first, third, tip4p = heavy[c][0], hyd[c][0], False
        elif nh == 1 and nx == 0:  # carbonyl oxygen: frame type 2
            m = heavy[c][0]
            if len(heavy[m]) != 3 or len(hyd[m]) > 0:
                raise ValueError(
                    f"Amber extra points (carbonyl frame): atom {m + 1} bonded to {where} must have "
                    f"three heavy neighbours and no hydrogen"
                )
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
                ps = [
                    (0.0, np.sin(_TET) * req[0], np.cos(_TET) * req[0]),
                    (0.0, -np.sin(_TET) * req[1], np.cos(_TET) * req[1]),
                ]
        else:
            ps = (
                [(0.0, 0.0, req[0])] if ne == 1 else [(s * req[0], 0.0, c60 * req[0]), (-s * req[1], 0.0, c60 * req[1])]
            )
        for (e, _), p in zip(eps[c], ps):
            out[e] = VirtualSite.amber(e, c, first, third, p, middle=middle)
    missing = [int(e) + 1 for e in np.nonzero(is_ep)[0] if int(e) not in out]
    if missing:
        raise ValueError(
            f"extra points {missing[:10]} are bonded to no atom (Amber defines the frame through that bond)"
        )
    return out
