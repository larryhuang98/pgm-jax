"""Compute per-frame observables of a pGM liquid and their explicit derivatives in theta.

Contents: FrameAnalyzer (per-frame values and derivatives with respect to the fitting parameters
theta of params.py, batched over frames on the device), RDFSpec (a g(r) to histogram) and
QUANTITIES (the rows of the per-frame Jacobian).

For each saved frame (positions with whole molecules, box, and the MD's induced dipoles as the
initial guess) FrameAnalyzer computes, at theta:

  U       potential energy (kJ/mol) at the converged induced dipoles, and dU/dtheta.  The pGM
          energy is variational in the dipoles, so dU/dtheta is taken at fixed mu (Hellmann-Feynman).
  M       cell dipole M = M_q + M_perm + M_ind (e nm; md/dipoles.py), and dM/dtheta.  The induced
          part depends on theta through the induction solve A(theta) mu = b(theta):
              dM_c/dtheta = d(M_q + M_perm)_c/dtheta + lam_c . d(b - A mu)/dtheta |_(mu fixed),
          with the adjoint A lam_c = e_c (a unit vector on every atom; A is symmetric).  lam_c is
          the response of the induced dipoles to a uniform unit field, so the same three solves
          give the cell polarizability alpha_ab = sum_i (lam_b)_{i,a} (CellDipole.polarizability).
  alpha   isotropic cell polarizability (1/3 trace, nm^3), and d alpha/dtheta = -(1/3) sum_c
          lam_c . (dA/dtheta) lam_c.
  D       mean magnitude of the molecular dipoles (e nm; molecules about their centre of mass), and
          dD/dtheta with one more adjoint solve A lam_D = dD/dmu, where dD/dmu_i is the unit
          vector of the dipole of i's molecule divided by the number of molecules.
  V       volume (nm^3); rdf: pair histogram of two atom selections as g(r) of this frame.

Every derivative is exact for the discretised model (PME, cutoff, precision) up to the CG
tolerances; tests/test_liquid_fit.py checks them against finite differences with the dipoles
re-solved.  One jax.jacrev of the six outputs (U, Mx, My, Mz, alpha, D) per frame, with the
dipoles mu and the adjoints lam held fixed; frames are processed in
vmapped chunks (`chunk`), with candidate pair rows built on the device by a dense cutoff search
(no neighbour list state needed: frames can come from any engine or from a trajectory).

    an = FrameAnalyzer(system, box, settings.replace(pme_grid=sim.ff.pme.K), space, rdf=rdf)
    frames = an.analyze(theta, [(positions, box, dipoles), ...], grad=True)

Units: nm, e, e nm, nm^3, kJ/mol.

See also docs/liquid_fit.md; estimators.LiquidSamples consumes the output.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..md.box import min_image, volume
from ..md.dipoles import CellDipole
from ..md.forcefield import MDSettings, PGMForceField

if TYPE_CHECKING:
    from ..system import System
    from .params import ParameterSpace

QUANTITIES = ("U", "M", "alpha", "D")  # rows of the per-frame Jacobian: U, Mx, My, Mz, alpha, D


@dataclasses.dataclass
class RDFSpec:
    """Pair distribution between two atom selections (indices into the system), r < rmax.

    A mutable dataclass.  The histogram of every frame is normalised to g(r) with the frame's
    volume; identical selections count each pair once.

    Parameters
    ----------
    a, b : np.ndarray of int
        Atom indices of the two selections.
    rmax : float
        Largest distance [nm]; must be below half the box (checked by FrameAnalyzer).
    nbins : int
        Number of bins of width rmax / nbins.
    name : str
        Name of the g(r) (used as the observable name by the fits).
    """

    a: np.ndarray
    b: np.ndarray
    rmax: float = 0.8
    nbins: int = 80
    name: str = "rdf"

    @property
    def same(self) -> bool:
        """Whether the two selections are identical (then pairs i < j are counted once)."""
        return len(self.a) == len(self.b) and bool(np.all(np.asarray(self.a) == np.asarray(self.b)))

    @property
    def r(self) -> np.ndarray:
        """Bin centres (nbins,) [nm]."""
        dr = self.rmax / self.nbins
        return (np.arange(self.nbins) + 0.5) * dr

    @classmethod
    def by_type(cls, sys: System, type_a: str, type_b: str | None = None, **kw: Any) -> RDFSpec:
        """Return the RDFSpec between the atoms of two atom types.

        Parameters
        ----------
        sys : System
            The system.
        type_a : str
            Atom type of the first selection.
        type_b : str, optional
            Atom type of the second selection; None: the same as `type_a`.
        **kw
            rmax, nbins, name (default name "g_<type_a><type_b>").

        Returns
        -------
        RDFSpec

        Raises
        ------
        ValueError
            If a selection is empty.
        """
        t = np.asarray(sys.types)
        a = np.flatnonzero(t == type_a)
        b = a if type_b is None or type_b == type_a else np.flatnonzero(t == type_b)
        if len(a) == 0 or len(b) == 0:
            raise ValueError(f"no atoms of type {type_a!r} / {type_b!r}")
        return cls(a, b, name=kw.pop("name", f"g_{type_a}{type_b or type_a}"), **kw)


class FrameAnalyzer:
    """Per-frame energies, cell dipoles, polarizabilities and molecular dipoles of a liquid, with derivatives.

    The derivatives are exact with respect to the fitting parameters theta (see the module docstring).

        an = FrameAnalyzer(system, box, settings.replace(pme_grid=sim.ff.pme.K), space, rdf=rdf)
        frames = an.analyze(theta, [(positions, box, dipoles), ...], grad=True)

    Frames are analysed in vmapped chunks on the device; the pair rows of every frame come from a
    dense cutoff search, so frames may come from any engine or trajectory.  The row width is sized
    from the first frame and widened (with a recompile) when a later frame overflows it.

    Attributes
    ----------
    ff : PGMForceField
        The MD force field (non-differentiable settings, no predictor), used for its kernels.
    sys : System
    space : ParameterSpace
    rdf : RDFSpec or None
    cell : CellDipole
        Cell and molecular dipoles (md/dipoles.py).
    tol : float
        CG tolerance of the dipole and adjoint solves.
    chunk : int
        Frames per vmapped batch.
    rc : float
        Pair distance of the candidate rows [nm] (the force field's pair cutoff + margin).
    row_block : int
        Rows per block of the dense search.
    width : int or None
        Candidate slots per atom (static; set by `size`).
    mass : float
        Total mass [amu].
    """

    def __init__(
        self,
        system: System,
        box: ArrayLike,
        settings: MDSettings,
        space: ParameterSpace,
        rdf: RDFSpec | None = None,
        dipole_tol: float = 1e-6,
        max_iter: int = 300,
        chunk: int = 8,
        margin: float = 0.02,
        row_block: int = 1024,
    ) -> None:
        """Set up the per-frame analysis.

        Parameters
        ----------
        system : System
            The liquid (as in the MD).
        box : ArrayLike (3, 3)
            A box of the run [nm] (sizes the rows and the PME grid).
        settings : MDSettings
            The MD settings; pass them with the run's PME grid set, e.g.
            settings.replace(pme_grid=sim.ff.pme.K), so that every frame uses the same grid.
        space : ParameterSpace
            theta -> parameters.
        rdf : RDFSpec, optional
            Radial distribution function to histogram; None: none.
        dipole_tol : float
            Tolerance of the dipole and adjoint CG solves (pmemd-pgm's criterion,
            max |alpha r| / mean |alpha b|).
        max_iter : int
            Largest number of CG iterations.
        chunk : int
            Frames per vmapped batch.
        margin : float
            Extra pair distance [nm] of the frame rows.
        row_block : int
            Rows per block of the pair evaluation.

        Raises
        ------
        NotImplementedError
            Charge flux or virtual sites.
        ValueError
            If the RDF's rmax is not below half the smallest diagonal box element.
        """
        s = settings.replace(differentiable=False, dipole_tol=dipole_tol, max_iter=max_iter, peek=0.0, predictor="none")
        self.ff = PGMForceField(system, np.asarray(box), s)
        if self.ff.flux is not None or any(getattr(m, "vsites", None) for m in system.molecules):
            raise NotImplementedError("charge flux / virtual sites are not handled by FrameAnalyzer")
        self.sys, self.space, self.rdf = system, space, rdf
        self.cell = CellDipole(self.ff)
        self.tol, self.chunk = float(dipole_tol), int(chunk)
        self.rc = float(self.ff.rc_pair) + float(margin)
        self.row_block = int(row_block)
        self.width = None
        self.mass = float(np.sum(system.masses))
        self._fns = {}
        if rdf is not None and rdf.rmax > 0.5 * float(np.min(np.diag(np.asarray(box)))):
            raise ValueError("rdf rmax must be below half the box")

    # ------------------------------------------------------------------ candidate rows
    def _candidates(self, pos: jax.Array, H: jax.Array, width: int) -> tuple[jax.Array, jax.Array]:
        """Return the atoms within rc of every atom and the largest neighbour count.

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm].
        H : jax.Array (3, 3)
            Box [nm], rows.
        width : int
            Slots per atom (static).

        Returns
        -------
        idx : jax.Array (N, width) int32
            Neighbour indices (minimum image), padded with N; neighbours beyond `width` are dropped.
        count : jax.Array () int
            The largest neighbour count (overflow if > width).

        Notes
        -----
        Dense search in blocks of row_block rows (lax.map), in float32: O(N^2) distance tests per
        frame, no neighbour-list state.  Each row's matches are compacted into slots by a cumulative sum.
        """
        N = self.ff.n
        B = min(N, self.row_block)
        nb = -(-N // B)
        p = pos.astype(jnp.float32)
        Hc = H.astype(jnp.float32)
        cols = jnp.arange(N, dtype=jnp.int32)

        def block(i0: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Return the candidate slots (B, width) of the rows i0 ... i0 + B - 1 and their largest count."""
            rows = i0 + jnp.arange(B, dtype=jnp.int32)
            d = min_image(p[jnp.minimum(rows, N - 1)][:, None, :] - p[None, :, :], Hc)
            m = (jnp.sum(d * d, -1) < self.rc**2) & (rows[:, None] != cols[None, :]) & (rows[:, None] < N)
            slot = jnp.cumsum(m, axis=1) - 1
            tgt = jnp.where(m & (slot < width), slot, width)
            r = jnp.broadcast_to(jnp.arange(B)[:, None], m.shape)
            out = jnp.full((B, width + 1), N, jnp.int32).at[r, tgt].set(jnp.broadcast_to(cols, m.shape))
            return out[:, :width], jnp.max(slot[:, -1] + 1)

        out, cnt = jax.lax.map(block, jnp.arange(nb, dtype=jnp.int32) * B)
        return out.reshape(-1, width)[:N], jnp.max(cnt)

    def size(self, pos: ArrayLike, H: ArrayLike, factor: float = 1.25) -> int:
        """Set the row width from one frame and return it.

        The width is factor x the largest neighbour count + 16, rounded up to a multiple of 8 and at
        most N.  It is static: a new width clears the compiled functions (re-jit).  `pos` (N, 3) [nm],
        `H` (3, 3) [nm].
        """
        _, c = jax.jit(self._candidates, static_argnums=2)(jnp.asarray(pos), jnp.asarray(H), min(self.ff.n, 2048))
        self.width = int(min(self.ff.n, int(np.ceil((int(c) * factor + 16) / 8.0) * 8)))
        self._fns = {}
        return self.width

    # ------------------------------------------------------------------ one frame
    def _setup(self, theta: jax.Array, pos: jax.Array, H: jax.Array, idx: jax.Array) -> tuple:
        """Return (per-atom parameters, pair geometry, permanent dipoles, PME setup, PME influence function).

        The parameters are space(theta) expanded per atom by the force field; `idx` are the candidate
        rows (_candidates).
        """
        ff = self.ff
        P = ff._atoms(self.space(theta))
        g = ff.geometry(pos, H, idx, P)
        p = ff.perm_dipoles(pos, H, P["cov"])
        S, Gk = ff.pme.setup(pos, H), ff.pme.influence(H)
        return P, g, p, S, Gk

    def _solve_mu(
        self, P: dict, g: dict, p: jax.Array, S: Any, Gk: jax.Array, mu0: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return (mu (N, 3) [e nm], CG iterations, final relative residual) of the induced dipoles.

        Solves A mu = b by the force field's preconditioned CG from the guess `mu0`, with the
        convergence norm mean |alpha b| (pmemd-pgm's criterion); zeros without induction.  Arguments
        are the outputs of _setup.
        """
        ff, cd = self.ff, self.ff.cd
        if not ff.ind:
            return jnp.zeros((ff.n, 3)), jnp.zeros((), jnp.int32), jnp.zeros(())
        alpha = P["alpha"]
        A = ff._operator(g, S, Gk, alpha)
        b = ff._field(g, S, Gk, P["q"].astype(cd), p)
        norm = jnp.mean(jnp.abs(alpha[:, None] * b.astype(jnp.float64))) + 1e-300
        x0 = mu0.astype(cd)
        return ff._cg(g, A, alpha, x0, b - A(x0), norm)

    def _mol_dipoles(
        self, P: dict, pos: jax.Array, H: jax.Array, mu: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the molecular dipoles (nmol, 3) [e nm] about each molecule's centre of mass, and q r, p.

        `P` is the parameter pytree (not expanded); `mu` the induced dipoles [e nm].  The atomic parts
        q (r - R_com) and p are returned as well (N, 3) [e nm].
        """
        qr, p, mu = self.cell._parts(pos, H, mu, P)
        return jax.ops.segment_sum(qr + p + mu, self.cell.mol, self.cell.nmol), qr, p

    def _rdf(self, pos: jax.Array, H: jax.Array) -> jax.Array:
        """Return g(r) (nbins,) of this frame for self.rdf.

        g_k = count_k V / (n_pairs 4/3 pi (r_{k+1}^3 - r_k^3)), minimum-image distances, pairs i < j for
        identical selections and a != b otherwise (n_pairs = n_a n_b then, even if the selections
        overlap).
        """
        s = self.rdf
        a, b = jnp.asarray(s.a), jnp.asarray(s.b)
        d = min_image(pos[a][:, None, :] - pos[b][None, :, :], H)
        r = jnp.sqrt(jnp.sum(d * d, -1))
        ok = r < s.rmax
        if s.same:
            ok = ok & (jnp.arange(len(s.a))[:, None] < jnp.arange(len(s.b))[None, :])
            npair = len(s.a) * (len(s.a) - 1) / 2.0
        else:
            ok = ok & (a[:, None] != b[None, :])
            npair = float(len(s.a) * len(s.b))
        dr = s.rmax / s.nbins
        k = jnp.clip(jnp.floor(r / dr).astype(jnp.int32), 0, s.nbins - 1)
        cnt = jnp.zeros(s.nbins).at[k.reshape(-1)].add(ok.reshape(-1).astype(jnp.float64))
        edges = np.arange(s.nbins + 1) * dr
        shell = jnp.asarray(4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3))
        return cnt * volume(H) / (npair * shell)

    def _frame(
        self, theta: jax.Array, pos: jax.Array, H: jax.Array, mu0: jax.Array, width: int, grad: bool = True
    ) -> dict[str, jax.Array]:
        """Analyse one frame (traced; vmapped and jitted by _fn).

        Parameters
        ----------
        theta : jax.Array (n,)
            Fitted parameters.
        pos : jax.Array (N, 3)
            Positions [nm], molecules whole.
        H : jax.Array (3, 3)
            Box [nm], rows.
        mu0 : jax.Array (N, 3)
            Initial guess of the induced dipoles [e nm].
        width : int
            Candidate slots per atom (static).
        grad : bool
            Also compute the theta-derivatives (static).

        Returns
        -------
        dict of str to jax.Array
            "V" [nm^3], "M" (3,) [e nm], "D" [e nm], "alpha" [nm^3], "U" [kJ/mol], "rdf" (if set), "mu"
            (N, 3) [e nm], solver diagnostics "count", "iters", "resid" (and "adj_iters", "adj_resid"
            with induction); with grad also "dU" (n,), "dM" (3, n), "dalpha" (n,), "dD" (n,).

        Notes
        -----
        Three adjoint solves A lam_c = e_c (unit field on every atom along c) and one A lam_D = dD/dmu,
        vmapped together.  alpha = (1/3) sum_c sum_i (lam_c)_{i,c}.  The Jacobian is one jax.jacrev of
        `rows` (module docstring), with mu and lam fixed.
        """
        ff, cd = self.ff, self.ff.cd
        n = ff.n
        idx, count = self._candidates(pos, H, width)
        P, g, p, S, Gk = self._setup(theta, pos, H, idx)
        mu, it, err = self._solve_mu(P, g, p, S, Gk, mu0)
        molD, _, _ = self._mol_dipoles(self.space(theta), pos, H, mu)
        Dn = jnp.linalg.norm(molD, axis=1)
        D = jnp.mean(Dn)
        M = jnp.sum(molD, axis=0)
        out = {"V": volume(H), "M": M, "D": D, "count": count, "iters": it, "resid": err}
        alpha = P["alpha"]
        if ff.ind:
            A = ff._operator(g, S, Gk, alpha)
            rhs = [jnp.zeros((n, 3), cd).at[:, c].set(1.0) for c in range(3)]
            u = molD / jnp.maximum(Dn, 1e-30)[:, None] / self.cell.nmol
            rhs.append(u[self.cell.mol].astype(cd))
            rhs = jnp.stack(rhs)
            norms = jnp.mean(jnp.abs(alpha[None, :, None] * rhs.astype(jnp.float64)), axis=(1, 2)) + 1e-300
            lam, lit, lerr = jax.vmap(
                lambda r, nr: ff._cg(g, A, alpha, jnp.zeros_like(r), r, nr, tol=self.tol, peek=0.0)
            )(rhs, norms)
            out["alpha"] = sum(jnp.sum(lam[c][:, c]) for c in range(3)) / 3.0
            out["adj_iters"], out["adj_resid"] = jnp.max(lit), jnp.max(lerr)
        else:
            lam = None
            out["alpha"] = jnp.zeros(())
        U = ff.energy_fixed_mu(pos, H, mu, idx, P)[0]
        out["U"] = U
        if self.rdf is not None:
            out["rdf"] = self._rdf(pos, H)
        if not grad:
            out["mu"] = mu
            return out

        def rows(th: jax.Array) -> jax.Array:
            """Return (U, Mx, My, Mz, alpha, D) (6,) as functions of theta at fixed mu and lam.

            Their gradients are the exact total derivatives: Hellmann-Feynman for U, adjoint terms
            lam . (b - A mu) for M and D, and -(1/3) sum_c lam_c . A(theta) lam_c for alpha.
            """
            Pp = self.space(th)
            Pa = ff._atoms(Pp)
            ga = ff.geometry(pos, H, idx, Pa)
            pa = ff.perm_dipoles(pos, H, Pa["cov"])
            molx, _, _ = self._mol_dipoles(Pp, pos, H, jax.lax.stop_gradient(mu))
            Mx = jnp.sum(molx, axis=0)
            Dx = jnp.mean(jnp.linalg.norm(molx, axis=1))
            Ux = ff.energy_fixed_mu(pos, H, mu, idx, Pa)[0]
            if lam is None:
                return jnp.concatenate([Ux[None], Mx, jnp.zeros(1), Dx[None]])
            R = ff._residual(ga, S, Gk, Pa["alpha"], Pa["q"], pa, mu).astype(jnp.float64)
            L = lam.astype(jnp.float64)
            Mx = Mx + jnp.stack([jnp.sum(L[c] * R) for c in range(3)])
            Dx = Dx + jnp.sum(L[3] * R)
            Aop = ff._operator(ga, S, Gk, Pa["alpha"])
            ax = -sum(jnp.sum(L[c] * Aop(lam[c]).astype(jnp.float64)) for c in range(3)) / 3.0
            return jnp.concatenate([Ux[None], Mx, ax[None], Dx[None]])

        Jr = jax.jacrev(rows)(jnp.asarray(theta, jnp.float64))
        out.update(dU=Jr[0], dM=Jr[1:4], dalpha=Jr[4], dD=Jr[5], mu=mu)
        return out

    # ------------------------------------------------------------------ batches
    def _fn(self, grad: bool) -> Any:
        """Return the jitted frame function vmapped over frames, f(theta, pos, H, mu0), cached by (grad, width).

        pos (C, N, 3), H (C, 3, 3) and mu0 (C, N, 3) are batched over their leading axis (C = chunk);
        theta is shared.
        """
        key = (grad, self.width)
        if key not in self._fns:

            def f(th: jax.Array, pos: jax.Array, H: jax.Array, mu: jax.Array) -> dict[str, jax.Array]:
                return self._frame(th, pos, H, mu, self.width, grad)

            self._fns[key] = jax.jit(jax.vmap(f, in_axes=(None, 0, 0, 0)))
        return self._fns[key]

    def frame(
        self, theta: ArrayLike, pos: ArrayLike, H: ArrayLike, mu0: ArrayLike | None = None, grad: bool = True
    ) -> dict:
        """Return the analysis of one frame as a numpy dict (keys as analyze, without the frame axis)."""
        out = self.analyze(theta, [(pos, H, mu0)], grad)
        return {k: v[0] for k, v in out.items()}

    def analyze(self, theta: ArrayLike, frames: list[tuple], grad: bool = True, keep_mu: bool = False) -> dict:
        """Analyse a list of frames in vmapped chunks.

        Parameters
        ----------
        theta : ArrayLike (n,)
            Fitted parameters.
        frames : list of (pos, H, mu0)
            pos (N, 3) [nm], H (3, 3) [nm], mu0 (N, 3) [e nm] or None (zeros).
        grad : bool
            Compute the theta-derivatives.
        keep_mu : bool
            Keep the induced dipoles "mu" (F, N, 3) in the output.

        Returns
        -------
        dict of str to np.ndarray
            Arrays with a leading frame axis F: U, M (F, 3), alpha, D, V, rdf (F, nbins) if set, solver
            diagnostics (count, iters, resid, adj_iters, adj_resid) and "converged" (bool: the dipole
            and adjoint residuals within the tolerance); with grad also dU (F, n), dM (F, 3, n), dalpha,
            dD (F, n).  Units as _frame.

        Notes
        -----
        The last chunk is padded with copies of its last frame (fixed batch shape; the copies are
        dropped).  If a chunk overflows the row width, the width is increased and the chunk repeated.
        """
        theta = jnp.asarray(theta, jnp.float64)
        if self.width is None:
            self.size(frames[0][0], frames[0][1])
        res = []
        k = 0
        while k < len(frames):
            part = frames[k : k + self.chunk]
            m = len(part)
            part = part + [part[-1]] * (self.chunk - m)  # fixed batch shape
            pos = jnp.stack([jnp.asarray(f[0], jnp.float64) for f in part])
            H = jnp.stack([jnp.asarray(f[1], jnp.float64) for f in part])
            mu = jnp.stack(
                [jnp.zeros((self.ff.n, 3)) if f[2] is None else jnp.asarray(f[2], jnp.float64) for f in part]
            )
            out = self._fn(grad)(theta, pos, H, mu)
            cmax = int(jnp.max(out["count"]))
            if cmax > self.width:  # rows too narrow: widen, repeat
                self.width = int(min(self.ff.n, int(np.ceil((cmax * 1.25 + 16) / 8.0) * 8)))
                continue
            if not keep_mu:
                out.pop("mu", None)
            res.append({key: np.asarray(v)[:m] for key, v in out.items()})
            k += m
        out = {key: np.concatenate([r[key] for r in res]) for key in res[0]}
        bad = np.asarray(out["resid"]) > self.tol
        if "adj_resid" in out:
            bad |= np.asarray(out["adj_resid"]) > self.tol
        out["converged"] = ~bad
        return out
