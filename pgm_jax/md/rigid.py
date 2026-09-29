"""Rigid molecules as JAX-MD rigid bodies (centre of mass + unit quaternion).

Contents: `RigidMolecules` (body frames, the atom-position map and its vector-Jacobian product,
conversions between atomic velocities and body momenta, wrapping) and `matrix_to_quaternion`.

Every molecule is rigid (the model has no bonded terms; for water this is SHAKE/SETTLE-rigid
water).  Each molecule type gets a body frame from its first instance: principal axes of inertia
about the centre of mass.  Instances are fitted to the template (Kabsch) to get their
orientations; the largest fit RMSD is reported.  Atom positions are

    r_a = R_k + rotate(q_k, s_a),

with R_k the centre of mass and q_k the orientation of molecule k and s_a the body-frame position
of atom a, and body forces/torques are the vector-Jacobian product of this map with the atomic
forces.

Monatomic and linear molecules have no (or one degenerate) principal moment: they get a nominal
moment so the integrator stays regular; the rotations it adds do not move any atom and are
removed from the degree-of-freedom count (dof_correction).

Units: nm, ps, amu (moments of inertia amu nm^2, momenta amu nm/ps).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..system import System
from ._jaxmd import rigid_body
from .box import inv3

RigidBody = rigid_body.RigidBody
Quaternion = rigid_body.Quaternion


def matrix_to_quaternion(R: ArrayLike) -> np.ndarray:
    """Return the unit quaternion (w, x, y, z) with rotate(q, v) = R v (Hamilton convention, as JAX-MD).

    Parameters
    ----------
    R : ArrayLike (3, 3)
        Rotation matrix.

    Returns
    -------
    np.ndarray (4,)
        Unit quaternion with w >= 0 (Shepperd's branch on the largest diagonal element for
        stability).
    """
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


def _unwrap(x: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Make a molecule whole: every atom at the minimum image of the first atom (host).

    Parameters
    ----------
    x : np.ndarray (n, 3)
        Atom positions of one molecule [nm].
    H : np.ndarray (3, 3)
        Box, lattice vectors as rows [nm].

    Returns
    -------
    np.ndarray (n, 3)
        Positions [nm], the first atom unchanged.
    """
    Hinv = np.linalg.inv(H)
    d = x - x[0]
    for _ in range(2):  # a second pass catches shifts a skewed box leaves after one rounding
        f = d @ Hinv
        d = d - np.round(f) @ H
    return x[0] + d


class RigidMolecules:
    """Rigid-body description of a system whose molecules are all rigid (module docstring).

        rigid = RigidMolecules(system, positions, box)
        x = rigid.positions(rigid.body0)                  # (N, 3) atom positions [nm]

    Built once on the host; the methods are JAX functions of the body state (traceable,
    differentiable).  Not a pytree.

    Attributes
    ----------
    sys : System
        The system.
    nmol : int
        Number of molecules M.
    mol : jax.Array (N,) int
        Molecule of every atom.
    local : jax.Array (N, 3)
        Body-frame position s_a of every atom (principal frame of its template) [nm].
    mass : RigidBody
        center (M,): molecular masses [amu]; orientation (M, 3): principal moments [amu nm^2]
        (nominal ones for monatomic and linear molecules).
    body0 : RigidBody
        Initial state: center (M, 3) centres of mass [nm] (molecules made whole); orientation
        Quaternion (M, 4).
    dof_correction : int
        Rotational degrees of freedom that move no atom (3 per monatomic, 1 per linear molecule),
        to subtract from the count.
    fit_rmsd : float
        Largest RMSD of an instance from its template after the Kabsch fit [nm].
    """

    def __init__(self, sys: System, pos: ArrayLike, H: ArrayLike, nominal_moment: float = 1e-4) -> None:
        """Build the body frames, masses, moments and the initial body state.

        Parameters
        ----------
        sys : System
            The system (pgm_jax/system.py); instances of the same Molecule object share one template.
        pos : ArrayLike (N, 3)
            Positions [nm]; molecules may be broken across the box (they are made whole).
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].
        nominal_moment : float
            Principal moment given to monatomic molecules (all three) [amu nm^2]; a linear molecule's
            vanishing axial moment (below 1e-6 of the largest) is set to its middle moment instead.
        """
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
            key = id(molk)  # instances share their Molecule object: one template per object
            if key not in templates:
                y = x - com
                I = (
                    mk[:, None, None] * (np.sum(y * y, 1)[:, None, None] * np.eye(3) - y[:, :, None] * y[:, None, :])
                ).sum(0)
                ev, U = np.linalg.eigh(I)
                if np.linalg.det(U) < 0:  # proper rotation (right-handed principal frame)
                    U[:, 0] = -U[:, 0]
                body = y @ U  # template in its principal frame
                big = max(ev.max(), nominal_moment)
                corr = 0
                if len(mk) == 1:
                    ev = np.full(3, nominal_moment)
                    corr = 3
                elif ev.min() < 1e-6 * big:  # linear: nominal axial moment
                    ev = ev.copy()
                    ev[np.argmin(ev)] = np.sort(ev)[1]
                    corr = 1
                templates[key] = (body, ev, corr)
            body, ev, corr = templates[key]
            y = x - com
            # Kabsch: rotation Rk with y ~ body @ Rk.T (best fit of the template to this instance)
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

    def positions(self, body: RigidBody) -> jax.Array:
        """Return the atom positions r_a = R_k + rotate(q_k, s_a) of a body state.

        Parameters
        ----------
        body : RigidBody
            center (M, 3) [nm], orientation Quaternion (M, 4).

        Returns
        -------
        jax.Array (N, 3)
            Atom positions [nm] (not wrapped; molecules whole).
        """
        q = Quaternion(body.orientation.vec[self.mol])
        return body.center[self.mol] + rigid_body.quaternion_rotate(q, self.local)

    def forces(self, body: RigidBody, atom_forces: jax.Array) -> RigidBody:
        """Return the generalised forces on the bodies (centre force and quaternion force).

        The vector-Jacobian product of `positions` with the atomic forces.

        Parameters
        ----------
        body : RigidBody
            Body state.
        atom_forces : jax.Array (N, 3)
            Atomic forces [kJ/mol/nm].

        Returns
        -------
        RigidBody
            center (M, 3): total force [kJ/mol/nm]; orientation: quaternion force (M, 4) [kJ/mol].
        """
        _, vjp = jax.vjp(self.positions, body)
        return vjp(atom_forces)[0]

    def momenta_from_velocities(self, body: RigidBody, pos: jax.Array, vel: jax.Array) -> RigidBody:
        """Return the rigid-body momenta that best match atomic velocities.

        Linear momentum P_k = sum_a m_a v_a and angular momentum L_k = sum_a m_a (r_a - R_k) x v_a,
        rotated into the body frame and converted to the quaternion conjugate momentum.

        Parameters
        ----------
        body : RigidBody
            Body state (centres [nm], orientations).
        pos : jax.Array (N, 3)
            Atom positions [nm], consistent with `body` (molecules whole).
        vel : jax.Array (N, 3)
            Atomic velocities [nm/ps].

        Returns
        -------
        RigidBody
            center (M, 3): linear momenta [amu nm/ps]; orientation: quaternion conjugate momenta
            (M, 4).
        """
        m = jnp.asarray(self.sys.masses)
        P = jax.ops.segment_sum(m[:, None] * vel, self.mol, self.nmol)
        rel = pos - body.center[self.mol]
        L = jax.ops.segment_sum(m[:, None] * jnp.cross(rel, vel), self.mol, self.nmol)
        Lb = jnp.einsum(
            "kij,kj->ki", rigid_body.space_to_body_rotation(body.orientation), L, precision=jax.lax.Precision.HIGHEST
        )
        return RigidBody(P, rigid_body.angular_momentum_to_conjugate_momentum(body.orientation, Lb))

    def atom_velocities(self, body: RigidBody, momentum: RigidBody) -> jax.Array:
        """Return the atomic velocities v_a = V_k + w_k x (r_a - R_k) implied by rigid-body momenta.

        Parameters
        ----------
        body : RigidBody
            Body state.
        momentum : RigidBody
            Linear momenta [amu nm/ps] and quaternion conjugate momenta.

        Returns
        -------
        jax.Array (N, 3)
            Atomic velocities [nm/ps].
        """
        v_com = momentum.center / self.mass.center[:, None]
        Lb = rigid_body.conjugate_momentum_to_angular_momentum(body.orientation, momentum.orientation)
        w_body = Lb / self.mass.orientation
        Rs2b = rigid_body.space_to_body_rotation(body.orientation)
        w = jnp.einsum("kji,kj->ki", Rs2b, w_body)  # body -> space
        rel = self.positions(body) - body.center[self.mol]
        return v_com[self.mol] + jnp.cross(w[self.mol], rel)

    def wrap(self, body: RigidBody, H: ArrayLike) -> RigidBody:
        """Return the body state with the centres of mass wrapped into the primary cell.

        Molecules stay whole (atoms follow their centre); orientations are unchanged.

        Parameters
        ----------
        body : RigidBody
            Body state.
        H : ArrayLike (3, 3)
            Box, lattice vectors as rows [nm].

        Returns
        -------
        RigidBody
            Centres at fractional coordinates in [0, 1) [nm].
        """
        H = jnp.asarray(H)
        hi = jax.lax.Precision.HIGHEST
        f = jnp.matmul(body.center, inv3(H), precision=hi)
        return RigidBody(jnp.matmul(f - jnp.floor(f), H, precision=hi), body.orientation)
