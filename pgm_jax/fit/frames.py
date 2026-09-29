"""Per-frame observables of a pGM liquid and their explicit derivatives with respect to the fitting
parameters theta (params.py), batched over frames on the device.

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
          dD/dtheta with one more adjoint solve A lam_D = dD/dmu.
  V       volume (nm^3); rdf: pair histogram of two atom selections as g(r) of this frame.

Every derivative is exact for the discretised model (PME, cutoff, precision) up to the CG
tolerances; tests/test_liquid_fit.py checks them against finite differences with the dipoles
re-solved.  One jax.jacrev of the six outputs (U, M, alpha, D) per frame; frames are processed in
vmapped chunks (`chunk`), with candidate pair rows built on the device by a dense cutoff search
(no neighbour list state needed: frames can come from any engine or from a trajectory).

Units: nm, e, e nm, nm^3, kJ/mol."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from ..md.box import min_image, volume
from ..md.dipoles import CellDipole
from ..md.forcefield import MDSettings, PGMForceField

QUANTITIES = ("U", "M", "alpha", "D")  # rows of the per-frame Jacobian: U, Mx, My, Mz, alpha, D


@dataclasses.dataclass
class RDFSpec:
    """Pair distribution between two atom selections (indices into the system), r < rmax."""

    a: np.ndarray
    b: np.ndarray
    rmax: float = 0.8
    nbins: int = 80
    name: str = "rdf"

    @property
    def same(self) -> bool:
        return len(self.a) == len(self.b) and bool(np.all(np.asarray(self.a) == np.asarray(self.b)))

    @property
    def r(self) -> np.ndarray:
        dr = self.rmax / self.nbins
        return (np.arange(self.nbins) + 0.5) * dr

    @classmethod
    def by_type(cls, sys, type_a: str, type_b: str | None = None, **kw):
        t = np.asarray(sys.types)
        a = np.flatnonzero(t == type_a)
        b = a if type_b is None or type_b == type_a else np.flatnonzero(t == type_b)
        if len(a) == 0 or len(b) == 0:
            raise ValueError(f"no atoms of type {type_a!r} / {type_b!r}")
        return cls(a, b, name=kw.pop("name", f"g_{type_a}{type_b or type_a}"), **kw)


class FrameAnalyzer:
    def __init__(
        self,
        sys,
        H,
        settings: MDSettings,
        space,
        rdf: RDFSpec | None = None,
        tol: float = 1e-6,
        max_iter: int = 300,
        chunk: int = 8,
        margin: float = 0.02,
        row_block: int = 1024,
    ):
        """sys, settings: those of the MD (same PME grid: pass settings with pme_grid set, e.g.
        settings.replace(pme_grid=sim.ff.pme.K)); space: ParameterSpace; tol: CG
        tolerance of the dipole and adjoint solves (pmemd-pgm's criterion)."""
        s = settings.replace(differentiable=False, dipole_tol=tol, max_iter=max_iter, peek=0.0, predictor="none")
        self.ff = PGMForceField(sys, np.asarray(H), s)
        if self.ff.flux is not None or any(getattr(m, "vsites", None) for m in sys.molecules):
            raise NotImplementedError("charge flux / virtual sites are not handled by FrameAnalyzer")
        self.sys, self.space, self.rdf = sys, space, rdf
        self.cell = CellDipole(self.ff)
        self.tol, self.chunk = float(tol), int(chunk)
        self.rc = float(self.ff.rc_pair) + float(margin)
        self.row_block = int(row_block)
        self.width = None
        self.mass = float(np.sum(sys.masses))
        self._fns = {}
        if rdf is not None and rdf.rmax > 0.5 * float(np.min(np.diag(np.asarray(H)))):
            raise ValueError("rdf rmax must be below half the box")

    # ------------------------------------------------------------------ candidate rows
    def _candidates(self, pos, H, width):
        """Atoms within rc of every atom (N, width), padding N, by a dense search in row blocks; and
        the largest count (overflow if > width)."""
        N = self.ff.n
        B = min(N, self.row_block)
        nb = -(-N // B)
        p = pos.astype(jnp.float32)
        Hc = H.astype(jnp.float32)
        cols = jnp.arange(N, dtype=jnp.int32)

        def block(i0):
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

    def size(self, pos, H, factor: float = 1.25):
        """Row width from one frame (static: re-jit when it changes)."""
        _, c = jax.jit(self._candidates, static_argnums=2)(jnp.asarray(pos), jnp.asarray(H), min(self.ff.n, 2048))
        self.width = int(min(self.ff.n, int(np.ceil((int(c) * factor + 16) / 8.0) * 8)))
        self._fns = {}
        return self.width

    # ------------------------------------------------------------------ one frame
    def _setup(self, theta, pos, H, idx):
        ff = self.ff
        P = ff._atoms(self.space(theta))
        g = ff.geometry(pos, H, idx, P)
        p = ff.perm_dipoles(pos, H, P["cov"])
        S, Gk = ff.pme.setup(pos, H), ff.pme.influence(H)
        return P, g, p, S, Gk

    def _solve_mu(self, P, g, p, S, Gk, mu0):
        ff, cd = self.ff, self.ff.cd
        if not ff.ind:
            return jnp.zeros((ff.n, 3)), jnp.zeros((), jnp.int32), jnp.zeros(())
        alpha = P["alpha"]
        A = ff._operator(g, S, Gk, alpha)
        b = ff._field(g, S, Gk, P["q"].astype(cd), p)
        norm = jnp.mean(jnp.abs(alpha[:, None] * b.astype(jnp.float64))) + 1e-300
        x0 = mu0.astype(cd)
        return ff._cg(g, A, alpha, x0, b - A(x0), norm)

    def _mol_dipoles(self, P, pos, H, mu):
        qr, p, mu = self.cell._parts(pos, H, mu, P)
        return jax.ops.segment_sum(qr + p + mu, self.cell.mol, self.cell.nmol), qr, p

    def _rdf(self, pos, H):
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

    def _frame(self, theta, pos, H, mu0, width, grad: bool = True):
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

        def rows(th):
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
    def _fn(self, grad: bool):
        key = (grad, self.width)
        if key not in self._fns:

            def f(th, pos, H, mu):
                return self._frame(th, pos, H, mu, self.width, grad)

            self._fns[key] = jax.jit(jax.vmap(f, in_axes=(None, 0, 0, 0)))
        return self._fns[key]

    def frame(self, theta, pos, H, mu0=None, grad: bool = True) -> dict:
        """One frame (numpy dict)."""
        out = self.analyze(theta, [(pos, H, mu0)], grad)
        return {k: v[0] for k, v in out.items()}

    def analyze(self, theta, frames, grad: bool = True, keep_mu: bool = False) -> dict:
        """frames: list of (pos (N, 3), H (3, 3), mu0 (N, 3) or None); returns numpy arrays with a
        leading frame axis: U, dU (F, n), M (F, 3), dM (F, 3, n), alpha, dalpha, D, dD, V, rdf,
        solver diagnostics (with grad=False only the values)."""
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
