"""Conic geometry for Non-Sequential Raytracing.

ConicGeometry (paraboloid, ellipsoid, hyperboloid) and the ParaboloidGeometry
convenience subclass. All operations in LOCAL coordinates.

The conic surface is defined as:
    z = r^2 / (R * (1 + sqrt(1 - (1+K) * r^2 / R^2)))
where r^2 = x^2 + y^2, R is the radius of curvature, and K is the conic constant.

Special cases:
    K = 0  -> sphere
    K = -1 -> paraboloid
    K < -1 -> hyperboloid
    -1 < K < 0 -> prolate ellipsoid
    K > 0  -> oblate ellipsoid

Kramer Harrison, 2026
"""

from __future__ import annotations

import math

import numpy as np

import optiland.backend as be
from optiland.nonsequential._utils import as_float, as_param
from optiland.nonsequential.components.geometry.base import AABB, AnalyticGeometry

# Denominators below this magnitude are treated as degenerate rather than
# divided by, and the affected root is discarded.
_TINY = 1e-30


class ConicGeometry(AnalyticGeometry):
    """Conic section surface (z = f(r)) centred on the local z-axis.

    The surface vertex is at the local origin. The aperture is circular with
    radius aperture_radius.

    Attributes:
        radius: Radius of curvature at the vertex [mm]. Positive = centre of
            curvature on +z side.
        conic: Conic constant K.
        aperture_radius: Semi-aperture radius [mm].
    """

    def __init__(self, radius: float, conic: float, aperture_radius: float) -> None:
        """Initialize ConicGeometry.

        Args:
            radius: Vertex radius of curvature [mm].
            conic: Conic constant K.
            aperture_radius: Aperture semi-diameter [mm].
        """
        self.radius = as_param(radius)
        self.conic = as_param(conic)
        self.aperture_radius = as_param(aperture_radius)

    def _curvature(self):
        """Return the vertex curvature c = 1 / radius.

        A radius of 0 or infinity both denote a flat surface (``r2=0.0`` is
        the plano convention used by :class:`LensConfig`), and both map to
        c = 0.  A degenerate radius is returned as a detached float: there is
        no meaningful derivative at the flat limit, and evaluating ``1 / R``
        there would emit an inf that backpropagates as NaN.

        Returns:
            Curvature [1/mm], as a tensor when ``radius`` is a tensor.
        """
        r_val = as_float(self.radius)
        if r_val == 0.0 or not math.isfinite(r_val):
            return 0.0
        return 1.0 / self.radius

    def _sag(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute conic surface sag z(x, y).

        Args:
            x: x-coordinates, shape (N,).
            y: y-coordinates, shape (N,).

        Returns:
            Sag values z, shape (N,).
        """
        r2 = x**2 + y**2
        c = self._curvature()
        K = self.conic
        # Curvature form. The radius form (dividing by R * R) evaluates to
        # inf * 0 for a flat surface (radius = inf); the forward value is
        # right but the backward pass returns NaN for every scene parameter.
        # With c = 1/R a flat surface is simply c = 0.
        under_root = 1.0 - (1.0 + K) * c**2 * r2
        # Clamp to a small positive epsilon (not 0): sqrt has an infinite
        # derivative at 0, which would poison gradients for rays at the conic
        # edge. The forward value changes by <= 1e-6 (sqrt(1e-12)).
        safe_root = be.maximum(under_root, 1e-12)
        return c * r2 / (1.0 + safe_root**0.5)

    def _normal_local(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute outward surface normals in local frame at (x, y, z(x,y)).

        The gradient of f(x,y,z) = z - sag(x,y) is (df/dx, df/dy, 1).

        Args:
            x: Hit x-coordinates, shape (N,).
            y: Hit y-coordinates, shape (N,).

        Returns:
            Outward normals (not yet normalized), shape (N, 3).
        """
        r2 = x**2 + y**2
        c = self._curvature()
        K = self.conic
        # Epsilon-clamped (not 0) so the sqrt derivative stays finite at the edge.
        under_root = be.maximum(1.0 - (1.0 + K) * c**2 * r2, 1e-12)
        sqrt_term = under_root**0.5

        # For a conic, dz/dr = c * r / sqrt(1 - (1+K) c^2 r^2), so
        # dz/dx = c * x / S and dz/dy = c * y / S. Written in curvature form
        # this stays finite for a flat surface (c = 0), where the radius form
        # produces an inf/inf that returns NaN gradients.
        gx = -c * x / sqrt_term
        gy = -c * y / sqrt_term
        gz = be.ones_like(x)
        norms = be.stack([gx, gy, gz], axis=1)
        return norms

    def _root_valid(
        self,
        t: np.ndarray,
        solvable: np.ndarray,
        origins: np.ndarray,
        directions: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Test one quadratic root for a physical hit on the sag sheet.

        Args:
            t: Candidate distances along the ray, shape (N,).
            solvable: Lanes where this root came from a well-posed division.
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3).
            eps: Minimum accepted distance [mm].

        Returns:
            (valid, px, py): hit flag and the transverse hit coordinates.
        """
        px = origins[:, 0] + t * directions[:, 0]
        py = origins[:, 1] + t * directions[:, 1]
        pz = origins[:, 2] + t * directions[:, 2]

        in_aperture = (px**2 + py**2) <= self.aperture_radius**2

        # On the surface, sqrt(1 - (1+K) c^2 r^2) = 1 - (1+K) c z, so the sheet
        # the sag function describes (positive root) is 1 - (1+K) c z >= 0.
        on_sag_sheet = (1.0 - (1.0 + self.conic) * self._curvature() * pz) >= 0.0

        valid = solvable & be.isfinite(t) & (t > eps) & in_aperture & on_sag_sheet
        return valid, px, py

    def ray_intersect(
        self, origins: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays with the conic surface.

        Solved in closed form: the sag form in the module docstring is
        algebraically the quadric

            c * (x^2 + y^2) + (1 + K) * c * z^2 - 2 * z = 0,

        so substituting p(t) = o + t*d gives a quadratic in t. The same
        equation covers the flat limit (c = 0), where it becomes linear.

        The quadric is the whole conic, including points the sag function does
        not describe (the far side of an ellipsoid, the second branch of a
        hyperboloid), which :meth:`_root_valid` rejects.

        Args:
            origins: Ray origins in local frame, shape (N, 3) [mm].
            directions: Ray directions in local frame, shape (N, 3).

        Returns:
            (t, normals, hit_mask, n_geom). n_geom points toward local +z
            (the ``material_back`` side by contract; see
            :meth:`ComponentGeometry.ray_intersect`).
        """
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

        c = self._curvature()
        kp = 1.0 + self.conic

        # Coefficients of a*t^2 + b*t + c_0 = 0.
        a = c * (dx**2 + dy**2 + kp * dz**2)
        b = 2.0 * (c * (ox * dx + oy * dy + kp * oz * dz) - dz)
        c_0 = c * (ox**2 + oy**2 + kp * oz**2) - 2.0 * oz

        disc = b**2 - 4.0 * a * c_0
        disc_ok = disc >= 0.0
        # Clamp the radicand to a small positive epsilon (not 0): sqrt has an
        # infinite derivative at 0, which combined with be.where yields a
        # 0 * inf = NaN in the backward pass even though the forward is masked.
        sqrt_disc = be.where(
            disc_ok, be.maximum(disc, 1e-12) ** 0.5, be.zeros_like(disc)
        )

        # Numerically stable roots (Numerical Recipes' "citardauque" form): the
        # textbook (-b +/- sqrt(disc)) / (2a) cancels catastrophically near the
        # paraboloid limit and for near-axial rays, where "a" is tiny. This form
        # also reduces continuously to the linear solution as a -> 0.
        sign_b = be.where(b >= 0.0, 1.0, -1.0)
        q = -0.5 * (b + sign_b * sqrt_disc)
        # Guard both denominators in place: masking a division by ~0 after the
        # fact still leaves an inf in the graph, which backpropagates as NaN.
        a_ok = be.abs(a) > _TINY
        q_ok = be.abs(q) > _TINY
        t1 = q / be.where(a_ok, a, be.ones_like(a))
        t2 = c_0 / be.where(q_ok, q, be.ones_like(q))

        eps = 1e-9
        args = (origins, directions, eps)
        valid1, px1, py1 = self._root_valid(t1, disc_ok & a_ok, *args)
        valid2, px2, py2 = self._root_valid(t2, disc_ok & q_ok, *args)

        # Nearest valid root along the ray.
        pick1 = valid1 & (~valid2 | (t1 <= t2))
        pick2 = valid2 & ~pick1
        hit_mask = pick1 | pick2

        inf_arr = be.ones_like(t1) * be.inf
        t = be.where(pick1, t1, be.where(pick2, t2, be.zeros_like(t1)))
        t_out = be.where(hit_mask, t, inf_arr)

        # Normals from the finite hit points, not t_out: its inf lanes would
        # poison every lane's gradient through be.where.
        px = be.where(pick1, px1, be.where(pick2, px2, be.zeros_like(px1)))
        py = be.where(pick1, py1, be.where(pick2, py2, be.zeros_like(py1)))
        n_raw = self._normal_local(px, py)
        n_len = (n_raw * n_raw).sum(axis=1, keepdims=True) ** 0.5
        n_geom = n_raw / (n_len + 1e-30)

        # Flip to face incoming ray
        dot = (directions * n_geom).sum(axis=1, keepdims=True)
        normals = be.where(dot > 0, -n_geom, n_geom)

        return t_out, normals, hit_mask, n_geom

    def bounding_box(self, transform: tuple[np.ndarray, np.ndarray]) -> AABB:
        """Return AABB in global coordinates.

        Args:
            transform: (translation, rotation_matrix).

        Returns:
            AABB in global frame.
        """
        t_vec = np.array(transform[0], dtype=float)
        R = np.array(transform[1], dtype=float)
        r = as_float(self.aperture_radius)
        # Max sag (at edge of aperture).  Computed from detached floats: the
        # AABB is NumPy-only bookkeeping and never carries gradients, and
        # mixing a torch parameter with NumPy arrays is disallowed.
        c = as_float(self._curvature())
        K = as_float(self.conic)
        r2 = r * r
        under_root = max(1.0 - (1.0 + K) * c * c * r2, 1e-12)
        sag_edge = c * r2 / (1.0 + np.sqrt(under_root))
        z_max = max(0.0, sag_edge)
        z_min = min(0.0, sag_edge)

        corners_local = np.array(
            [
                [-r, -r, z_min],
                [-r, r, z_min],
                [r, -r, z_min],
                [r, r, z_min],
                [-r, -r, z_max],
                [-r, r, z_max],
                [r, -r, z_max],
                [r, r, z_max],
            ],
            dtype=float,
        )
        corners_global = corners_local @ R.T + t_vec
        return AABB(corners_global.min(axis=0), corners_global.max(axis=0))


class ParaboloidGeometry(ConicGeometry):
    """Convenience subclass for paraboloid (conic constant K = -1).

    Attributes:
        radius: Radius of curvature [mm].
        aperture_radius: Semi-aperture [mm].
    """

    def __init__(self, radius: float, aperture_radius: float) -> None:
        """Initialize ParaboloidGeometry.

        Args:
            radius: Vertex radius of curvature [mm].
            aperture_radius: Aperture semi-diameter [mm].
        """
        super().__init__(radius=radius, conic=-1.0, aperture_radius=aperture_radius)
