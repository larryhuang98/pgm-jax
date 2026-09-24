"""Rigid molecules as JAX-MD rigid bodies (centre of mass + unit quaternion).

Every molecule is rigid (the model has no bonded terms; for water this is SHAKE/SETTLE-rigid
water).  Each molecule type gets a body frame from its first instance: principal axes of inertia
about the centre of mass.  Instances are fitted to the template (Kabsch) to get their
orientations; the largest fit RMSD is reported.  Atom positions are
    r_a = R_k + rotate(q_k, s_a),
and body forces/torques are the vector-Jacobian product of this map with the atomic forces.

Monatomic and linear molecules have no (or one degenerate) principal moment: they get a nominal
moment so the integrator stays regular; the rotations it adds do not move any atom and are
removed from the degree-of-freedom count (dof_correction)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..system import System
from ._jaxmd import rigid_body

RigidBody = rigid_body.RigidBody
Quaternion = rigid_body.Quaternion


def matrix_to_quaternion(R) -> np.ndarray:
    """Unit quaternion (w, x, y, z) with rotate(q, v) = R v (Hamilton convention, as JAX-MD)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def _unwrap(x, H):
    """Make a molecule whole: every atom at the minimum image of the first atom."""
    Hinv = np.linalg.inv(H)
    d = x - x[0]
    for _ in range(2):
        f = d @ Hinv
        d = d - np.round(f) @ H
    return x[0] + d


class RigidMolecules:
    def __init__(self, sys: System, pos, H, nominal_moment: float = 1e-4):
        pos, H = np.asarray(pos, float), np.asarray(H, float)
        self.sys = sys
        self.nmol = sys.nmol
        self.mol = jnp.asarray(sys.mol)
        m = np.asarray(sys.masses, float)
        templates, local = {}, np.zeros((sys.n, 3))
        M = np.zeros(sys.nmol)
        moments = np.zeros((sys.nmol, 3))
        centers = np.zeros((sys.nmol, 3))
        quats = np.zeros((sys.nmol, 4))
        self.dof_correction = 0
        rmsd = 0.0
        for k, molk in enumerate(sys.molecules):
            sl = sys.atom_slice(k)
            x = _unwrap(pos[sl], H)
            mk = m[sl]
            com = (mk[:, None] * x).sum(0) / mk.sum()
            key = id(molk)
            if key not in templates:
                y = x - com
                I = (mk[:, None, None] * (np.sum(y * y, 1)[:, None, None] * np.eye(3) - y[:, :, None] * y[:, None, :])).sum(0)
                ev, U = np.linalg.eigh(I)
                if np.linalg.det(U) < 0:
                    U[:, 0] = -U[:, 0]
                body = y @ U                                           # template in its principal frame
                big = max(ev.max(), nominal_moment)
                corr = 0
                if len(mk) == 1:
                    ev = np.full(3, nominal_moment)
                    corr = 3
                elif ev.min() < 1e-6 * big:                            # linear: nominal axial moment
                    ev = ev.copy()
                    ev[np.argmin(ev)] = np.sort(ev)[1]
                    corr = 1
                templates[key] = (body, ev, corr)
            body, ev, corr = templates[key]
            y = x - com
            A = body.T @ y
            U_, _, Vt = np.linalg.svd(A)
            dd = np.sign(np.linalg.det(Vt.T @ U_.T))
            Rk = Vt.T @ np.diag([1, 1, dd]) @ U_.T
            rmsd = max(rmsd, float(np.sqrt(np.mean(np.sum((body @ Rk.T - y) ** 2, 1)))))
            local[sl] = body
            M[k], moments[k], centers[k] = mk.sum(), ev, com
            quats[k] = matrix_to_quaternion(Rk)
            self.dof_correction += corr
        self.fit_rmsd = rmsd
        self.local = jnp.asarray(local)
        self.mass = RigidBody(jnp.asarray(M), jnp.asarray(moments))
        self.body0 = RigidBody(jnp.asarray(centers), Quaternion(jnp.asarray(quats)))

    def positions(self, body):
        q = Quaternion(body.orientation.vec[self.mol])
        return body.center[self.mol] + rigid_body.quaternion_rotate(q, self.local)

    def forces(self, body, atom_forces):
        """Generalised forces on the bodies (centre force and quaternion force) from atomic forces."""
        _, vjp = jax.vjp(self.positions, body)
        return vjp(atom_forces)[0]

    def momenta_from_velocities(self, body, pos, vel):
        """Rigid-body momenta (linear, quaternion conjugate) from atomic velocities (nm/ps)."""
        m = jnp.asarray(self.sys.masses)
        P = jax.ops.segment_sum(m[:, None] * vel, self.mol, self.nmol)
        rel = pos - body.center[self.mol]
        L = jax.ops.segment_sum(m[:, None] * jnp.cross(rel, vel), self.mol, self.nmol)
        Lb = jnp.einsum("kij,kj->ki", rigid_body.space_to_body_rotation(body.orientation), L)
        return RigidBody(P, rigid_body.angular_momentum_to_conjugate_momentum(body.orientation, Lb))

    def atom_velocities(self, body, momentum):
        """Atomic velocities (nm/ps) implied by rigid-body momenta."""
        v_com = momentum.center / self.mass.center[:, None]
        Lb = rigid_body.conjugate_momentum_to_angular_momentum(body.orientation, momentum.orientation)
        w_body = Lb / self.mass.orientation
        Rs2b = rigid_body.space_to_body_rotation(body.orientation)
        w = jnp.einsum("kji,kj->ki", Rs2b, w_body)                        # body -> space
        rel = self.positions(body) - body.center[self.mol]
        return v_com[self.mol] + jnp.cross(w[self.mol], rel)

    def wrap(self, body, H):
        """Centres of mass into the primary cell (molecules stay whole)."""
        H = jnp.asarray(H)
        f = body.center @ jnp.linalg.inv(H)
        return RigidBody((f - jnp.floor(f)) @ H, body.orientation)
