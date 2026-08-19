"""This module provides implementations of NURBS for optical surfaces.

The geometric implementation of the surface is mostly based on the work of
Roberto Agromayor (https://github.com/turbo-sim/nurbspy), the surface
fitting is mostly based on the work of Onur R. Bingol
(https://github.com/orbingol/NURBS-Python), and the ray intersection with the
NURBS and the computation of the sag are based on my understanding of the
algorithms described in the paper 'Practical ray tracing of trimmed nurbs
surfaces'.

Matteo Taccola, 2025
"""

from __future__ import annotations

import numpy as np
from scipy.special import binom

import optiland.backend as be
from optiland.geometries.base import BaseGeometry

from .nurbs_basis_functions import (
    compute_basis_polynomials,
    compute_basis_polynomials_derivatives,
)
from .nurbs_fitting import approximate_surface


class NurbsGeometry(BaseGeometry):
    """Creates a NURBS (Non-Uniform Rational Basis Spline) geometry.

    This class can be used to represent polynomial and rational Bézier,
    B-Spline, and NURBS surfaces. The type of surface depends on the
    initialization arguments.

    Args:
        coordinate_system: The coordinate system of the geometry.
        radius: The radius of curvature of the base conic. (ignored if control
            points are passed)
        conic: The conic constant of the base conic. (ignored if control
            points are passed)
        nurbs_norm_x: Defines, along with nurbs_norm_y, x_center, and y_center,
            the rectangular area for the fit of the surface with the NURBS.
            If None, scales automatically during paraxial updates. Defaults to None.
        nurbs_norm_y: Defines, along with nurbs_norm_x, x_center, and y_center,
            the rectangular area for the fit of the surface with the NURBS.
            If None, scales automatically during paraxial updates. Defaults to None.
        x_center: Defines, along with nurbs_norm_x, nurbs_norm_y, and y_center,
            the rectangular area for the fit of the surface with the NURBS.
        y_center: Defines, along with nurbs_norm_x, nurbs_norm_y, and x_center,
            the rectangular area for the fit of the surface with the NURBS.
        control_points: An array with shape (ndim, n+1, m+1) containing the
            coordinates of the control points. The first dimension of ´P´ spans
            the coordinates of the control points (any number of dimensions),
            the second dimension of ´P´ spans the u-direction control points
            (0, 1, ..., n), and the third dimension of ´P´ spans the v-direction
            control points (0, 1, ..., m).
        weights: An array with shape (n+1, m+1) containing the weight of the
            control points. The first dimension of ´W´ spans the u-direction
            control points weights (0, 1, ..., n), and the second dimension of
            ´W´ spans the v-direction control points weights (0, 1, ..., m).
        u_degree: The degree of the u-basis polynomials.
        v_degree: The degree of the v-basis polynomials.
        u_knots: A knot vector in the u-direction with shape (r+1=n+p+2,). Set
            the multiplicity of the first and last entries equal to ´p+1´ to
            obtain a clamped spline.
        v_knots: A knot vector in the v-direction with shape (s+1=m+q+2,). Set
            the multiplicity of the first and last entries equal to ´q+1´ to
            obtain a clamped spline.
        n_points_u: Defines the grid size of control points (n_points_u x
            n_points_v). The default value is updated in case control points
            are passed.
        n_points_v: Defines the grid size of control points (n_points_u x
            n_points_v). The default value is updated in case control points
            are passed.
        tol: The tolerance for Newton-Raphson iteration.
        max_iter: The maximum number of iterations for Newton-Raphson.

    References:
        - The NURBS Book. See references to equations and algorithms throughout
          the code. L. Piegl and W. Tiller. Springer, second edition.
        - Curves and Surfaces for CADGD. See references to equations in the
          source code. G. Farin. Morgan Kaufmann Publishers, fifth edition.
        - All references correspond to The NURBS book unless it is explicitly
          stated that they come from Farin's book.
    """

    def __init__(
        self,
        coordinate_system,
        radius=be.inf,
        conic=0.0,
        nurbs_norm_x=None,
        nurbs_norm_y=None,
        x_center=0.0,
        y_center=0.0,
        control_points=None,
        weights=None,
        u_degree=None,
        v_degree=None,
        u_knots=None,
        v_knots=None,
        n_points_u=4,
        n_points_v=4,
        tol=1e-10,
        max_iter=100,
    ):
        super().__init__(coordinate_system)
        self.P = be.asarray(control_points) if control_points is not None else None
        self.W = be.asarray(weights) if weights is not None else None
        self.p = u_degree
        self.q = v_degree
        self.U = be.asarray(u_knots) if u_knots is not None else None
        self.V = be.asarray(v_knots) if v_knots is not None else None

        self.normalization_mode = (
            "manual"
            if (nurbs_norm_x is not None or nurbs_norm_y is not None)
            else "auto"
        )
        self.nurbs_norm_x = nurbs_norm_x if nurbs_norm_x is not None else 1.0
        self.nurbs_norm_y = nurbs_norm_y if nurbs_norm_y is not None else 1.0

        self.x_center = x_center
        self.y_center = y_center
        self.radius = radius
        self.k = conic
        self.tol = tol
        self.max_iter = max_iter

        self.is_symmetric = False

        # If control points are not provided, the NURBS is obtained as a fit
        # of a standard surface.
        if control_points is None:
            self.is_fitted = True
            self.ndim = 3
            self.P_size_u = n_points_u + 1
            self.P_size_v = n_points_v + 1

        # Polynomial Bezier surface initialization
        elif (
            weights is None
            and u_degree is None
            and u_knots is None
            and v_degree is None
            and v_knots is None
        ):
            self.is_fitted = False
            self.surface_type = "Bezier"
            self.ndim = be.shape(control_points)[0]

            n = be.shape(control_points)[1] - 1
            m = be.shape(control_points)[2] - 1
            self.P_size_u = n + 1
            self.P_size_v = m + 1
            weights = be.ones((n + 1, m + 1), dtype=control_points.dtype)

            u_degree = n
            v_degree = m

            u_knots = be.concatenate(
                (
                    be.zeros(u_degree),
                    be.linspace(0, 1, n - u_degree + 2),
                    be.ones(u_degree),
                )
            )
            v_knots = be.concatenate(
                (
                    be.zeros(v_degree),
                    be.linspace(0, 1, m - v_degree + 2),
                    be.ones(v_degree),
                )
            )

        # Rational Bezier surface initialization
        elif (
            u_degree is None
            and u_knots is None
            and v_degree is None
            and v_knots is None
        ):
            self.is_fitted = False
            self.surface_type = "R-Bezier"
            self.ndim = be.shape(control_points)[0]

            n = be.shape(control_points)[1] - 1
            m = be.shape(control_points)[2] - 1
            self.P_size_u = n + 1
            self.P_size_v = m + 1

            u_degree = n
            v_degree = m

            u_knots = be.concatenate(
                (
                    be.zeros(u_degree),
                    be.linspace(0, 1, n - u_degree + 2),
                    be.ones(u_degree),
                )
            )
            v_knots = be.concatenate(
                (
                    be.zeros(v_degree),
                    be.linspace(0, 1, m - v_degree + 2),
                    be.ones(v_degree),
                )
            )

        # B-Spline surface initialization
        elif weights is None and u_knots is not None and v_knots is not None:
            self.is_fitted = False
            self.surface_type = "B-Spline"
            self.ndim = be.shape(control_points)[0]

            n = be.shape(control_points)[1] - 1
            m = be.shape(control_points)[2] - 1
            self.P_size_u = n + 1
            self.P_size_v = m + 1

            weights = be.ones((n + 1, m + 1), dtype=control_points.dtype)

        # B-Spline surface initialization
        elif weights is None and u_knots is None and v_knots is None:
            self.is_fitted = False

            self.surface_type = "B-Spline"
            self.ndim = be.shape(control_points)[0]

            n = be.shape(control_points)[1] - 1
            m = be.shape(control_points)[2] - 1
            self.P_size_u = n + 1
            self.P_size_v = m + 1

            u_knots = be.concatenate(
                (
                    be.zeros(u_degree),
                    be.linspace(0, 1, n - u_degree + 2),
                    be.ones(u_degree),
                )
            )
            v_knots = be.concatenate(
                (
                    be.zeros(v_degree),
                    be.linspace(0, 1, m - v_degree + 2),
                    be.ones(v_degree),
                )
            )

            weights = be.ones((n + 1, m + 1), dtype=control_points.dtype)

        # NURBS surface initialization
        else:
            self.is_fitted = False
            self.surface_type = "NURBS"

            if u_knots is None and v_knots is None:
                n = be.shape(control_points)[1] - 1
                m = be.shape(control_points)[2] - 1
                self.P_size_u = n + 1
                self.P_size_v = m + 1

                u_knots = be.concatenate(
                    (
                        be.zeros(u_degree),
                        be.linspace(0, 1, n - u_degree + 2),
                        be.ones(u_degree),
                    )
                )
                v_knots = be.concatenate(
                    (
                        be.zeros(v_degree),
                        be.linspace(0, 1, m - v_degree + 2),
                        be.ones(v_degree),
                    )
                )
            self.ndim = be.shape(control_points)[0]

    def set_radius(self, value: float) -> None:
        """Set the radius of curvature.

        Args:
            value (float): The new radius of curvature.
        """
        self.radius = be.array(value)

    def flip(self):
        """Flip the geometry.

        This method changes the sign of the radius and the z-coordinate of the
        control points.
        """
        self.radius = -self.radius
        if self.P is not None:
            self.P[2, :, :] = -self.P[2, :, :]

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        self.radius = self.radius * scale_factor

        if self.nurbs_norm_x is not None:
            self.nurbs_norm_x = self.nurbs_norm_x * scale_factor

        if self.nurbs_norm_y is not None:
            self.nurbs_norm_y = self.nurbs_norm_y * scale_factor

        self.x_center = self.x_center * scale_factor
        self.y_center = self.y_center * scale_factor

        if self.P is not None:
            self.P = self.P * scale_factor

    def update_normalization(self, semi_aperture: float) -> None:
        if self.normalization_mode == "auto":
            self.nurbs_norm_x = semi_aperture * 1.25
            self.nurbs_norm_y = semi_aperture * 1.25
            if getattr(self, "is_fitted", False):
                self.fit_surface()

    def get_value(self, u, v):
        """Evaluates the coordinates of the surface.

        Args:
            u: The u-parameter used to evaluate the surface.
            v: The v-parameter used to evaluate the surface.

        Returns:
            An array containing the coordinates of the surface.
        """
        if be.isscalar(u) and be.isscalar(v) or be.size(u) == be.size(v):
            pass
        else:
            raise Exception("u and v must have the same size")

        if u.ndim > 1:
            a, b = u.shape
            u = be.ravel(u)
            v = be.ravel(v)
            S = self.compute_nurbs_coordinates(
                self.P, self.W, self.p, self.q, self.U, self.V, u, v
            )
            S = be.reshape(S, (self.ndim, a, b))
        else:
            S = self.compute_nurbs_coordinates(
                self.P, self.W, self.p, self.q, self.U, self.V, u, v
            )
        return S

    @staticmethod
    def compute_nurbs_coordinates(P, W, p, q, U, V, u, v):
        """Evaluates the coordinates of the NURBS surface.

        This function computes the coordinates of the NURBS surface in
        homogeneous space using equation 4.15 and then maps the coordinates
        to ordinary space using the perspective map given by equation 1.16. See
        algorithm A4.3 in The NURBS Book.

        Args:
            P: An array containing the coordinates of the control points.
            W: An array containing the weight of the control points.
            p: The degree of the u-basis polynomials.
            q: The degree of the v-basis polynomials.
            U: The knot vector in the u-direction.
            V: The knot vector in the v-direction.
            u: The u-parameter used to evaluate the surface.
            v: The v-parameter used to evaluate the surface.

        Returns:
            An array containing the NURBS surface coordinates.
        """
        if P.ndim > 3:
            raise Exception("P must be an array of shape (ndim, n+1, m+1)")
        if W.ndim > 2:
            raise Exception("W must be an array of shape (n+1, m+1)")
        if U.ndim > 1:
            raise Exception("U must be an array of shape (r+1=n+p+2,)")
        if V.ndim > 1:
            raise Exception("V must be an array of shape (s+1=m+q+2,)")
        if be.isscalar(u):
            u = be.asarray(u)
        elif u.ndim > 1:
            raise Exception("u must be a scalar or an array of shape (N,)")
        if be.isscalar(v):
            v = be.asarray(v)
        elif u.ndim > 1:
            raise Exception("v must be a scalar or an array of shape (N,)")

        n_dim, nn, mm = be.shape(P)

        n = nn - 1
        m = mm - 1

        if be.get_backend() == "torch":
            np_u = be.to_numpy(u).copy().astype(np.float64)
            np_U = be.to_numpy(U).copy().astype(np.float64)
            np_v = be.to_numpy(v).copy().astype(np.float64)
            np_V = be.to_numpy(V).copy().astype(np.float64)
            N_basis_u_np = compute_basis_polynomials(n, p, np_U, np_u)
            N_basis_v_np = compute_basis_polynomials(m, q, np_V, np_v)
            N_basis_u = be.asarray(N_basis_u_np)
            N_basis_v = be.asarray(N_basis_v_np)
        else:
            N_basis_u = compute_basis_polynomials(n, p, U, u)
            N_basis_v = compute_basis_polynomials(m, q, V, v)

        P_w = be.concatenate((P * W[None, :], W[None, :]), axis=0)

        A = be.matmul(P_w, N_basis_v)
        B = be.stack([N_basis_u] * (n_dim + 1), axis=0)
        S_w = be.sum(A * B, axis=1)

        S = S_w[0:-1, :] / S_w[-1, :]

        return S

    @staticmethod
    def compute_bspline_coordinates(P, p, q, U, V, u, v):
        """Evaluates the coordinates of the B-Spline surface.

        This function computes the coordinates of a B-Spline surface as given
        by equation 3.11. See algorithm A3.5 in The NURBS Book.

        Args:
            P: An array containing the coordinates of the control points.
            p: The degree of the u-basis polynomials.
            q: The degree of the v-basis polynomials.
            U: The knot vector in the u-direction.
            V: The knot vector in the v-direction.
            u: The u-parameter used to evaluate the surface.
            v: The v-parameter used to evaluate the surface.

        Returns:
            An array containing the B-Spline surface coordinates.
        """
        if P.ndim > 3:
            raise Exception("P must be an array of shape (ndim, n+1, m+1)")
        if not be.isscalar(p):
            raise Exception("p must be an scalar")
        if not be.isscalar(q):
            raise Exception("q must be an scalar")
        if U.ndim > 1:
            raise Exception("U must be an array of shape (r+1=n+p+2,)")
        if V.ndim > 1:
            raise Exception("V must be an array of shape (s+1=m+q+2,)")
        if be.isscalar(u):
            u = be.asarray(u)
        elif u.ndim > 1:
            raise Exception("u must be a scalar or an array of shape (N,)")
        if be.isscalar(v):
            v = be.asarray(v)
        elif u.ndim > 1:
            raise Exception("v must be a scalar or an array of shape (N,)")

        n_dim, nn, mm = be.shape(P)

        n = nn - 1
        m = mm - 1

        N_basis_u = compute_basis_polynomials(n, p, U, u)
        N_basis_v = compute_basis_polynomials(m, q, V, v)

        A = be.dot(P, N_basis_v)
        B = be.repeat(N_basis_u[be.newaxis], repeats=n_dim, axis=0)
        S = be.sum(A * B, axis=1)

        return S

    def get_derivative(self, u, v, order_u, order_v):
        """Evaluates the derivative of the surface.

        Args:
            u: The u-parameter used to evaluate the surface.
            v: The v-parameter used to evaluate the surface.
            order_u: The order of the partial derivative in the u-direction.
            order_v: The order of the partial derivative in the v-direction.

        Returns:
            An array containing the derivative of the desired order.
        """
        if u.ndim > 1:
            a, b = u.shape
            u = be.ravel(u)
            v = be.ravel(v)
            dS = self.compute_nurbs_derivatives(
                self.P, self.W, self.p, self.q, self.U, self.V, u, v, order_u, order_v
            )[order_u, order_v, ...]
            dS = be.reshape(dS, (self.ndim, a, b))
        else:
            dS = self.compute_nurbs_derivatives(
                self.P, self.W, self.p, self.q, self.U, self.V, u, v, order_u, order_v
            )[order_u, order_v, ...]

        return dS

    def compute_nurbs_derivatives(
        self, P, W, p, q, U, V, u, v, up_to_order_u, up_to_order_v
    ):
        """Computes the derivatives of a NURBS surface.

        This function computes the analytic derivatives of the NURBS surface
        in ordinary space using equation 4.20 and the derivatives of the NURBS
        surface in homogeneous space obtained from
        compute_bspline_derivatives().

        The derivatives are computed recursively in a fashion similar to
        algorithm A4.4 in The NURBS Book.

        Args:
            P: An array containing the coordinates of the control points.
            W: An array containing the weight of the control points.
            p: The degree of the u-basis polynomials.
            q: The degree of the v-basis polynomials.
            U: The knot vector in the u-direction.
            V: The knot vector in the v-direction.
            u: The u-parameter used to evaluate the surface.
            v: The v-parameter used to evaluate the surface.
            up_to_order_u: The order of the highest derivative in the
                u-direction.
            up_to_order_v: The order of the highest derivative in the
                v-direction.

        Returns:
            An array containing the NURBS surface derivatives.
        """
        u, v = be.asarray(u), be.asarray(v)

        P_w = be.concatenate((P * W[None, :], W[None, :]), axis=0)

        bspline_derivatives = self.compute_bspline_derivatives(
            P_w, p, q, U, V, u, v, up_to_order_u, up_to_order_v
        )
        A_ders = bspline_derivatives[:, :, 0:-1, :]
        w_ders = bspline_derivatives[:, :, [-1], :]

        _n_dim, _N = be.shape(P)[0], be.size(u)

        rows = []
        for k in range(up_to_order_u + 1):
            cols = []
            for L in range(up_to_order_v + 1):
                temp_numerator = A_ders[k, L, ...]

                for i in range(1, k + 1):
                    temp_numerator = temp_numerator - (
                        binom(k, i) * w_ders[i, 0, ...] * rows[k - i][L]
                    )
                for j in range(1, L + 1):
                    temp_numerator = temp_numerator - (
                        binom(L, j) * w_ders[0, j, ...] * cols[L - j]
                    )
                for i in range(1, k + 1):
                    for j in range(1, L + 1):
                        temp_numerator = temp_numerator - (
                            binom(k, i)
                            * binom(L, j)
                            * w_ders[i, j, ...]
                            * rows[k - i][L - j]
                        )
                cols.append(temp_numerator / w_ders[0, 0, ...])
            rows.append(be.stack(cols, axis=0))
        nurbs_derivatives = be.stack(rows, axis=0)
        return nurbs_derivatives

    @staticmethod
    def compute_bspline_derivatives(P, p, q, U, V, u, v, up_to_order_u, up_to_order_v):
        """Computes the derivatives of a B-Spline surface.

        This function computes the analytic derivatives of a B-Spline surface
        using equation 3.17. See algorithm A3.6 in The NURBS Book.

        Args:
            P: An array containing the coordinates of the control points.
            p: The degree of the u-basis polynomials.
            q: The degree of the v-basis polynomials.
            U: The knot vector in the u-direction.
            V: The knot vector in the v-direction.
            u: The u-parameter used to evaluate the surface.
            v: The v-parameter used to evaluate the surface.
            up_to_order_u: The order of the highest derivative in the
                u-direction.
            up_to_order_v: The order of the highest derivative in the
                v-direction.

        Returns:
            An array containing the B-Spline surface derivatives.
        """
        u = be.asarray(u)

        n_dim, _N = be.shape(P)[0], be.size(u)
        rows = []
        for order_u in range(min(p, up_to_order_u) + 1):
            cols = []
            for order_v in range(min(q, up_to_order_v) + 1):
                n = be.shape(P)[1] - 1
                m = be.shape(P)[2] - 1

                if be.get_backend() == "torch":
                    np_u = be.to_numpy(u).copy().astype(np.float64)
                    np_U = be.to_numpy(U).copy().astype(np.float64)
                    np_v = be.to_numpy(v).copy().astype(np.float64)
                    np_V = be.to_numpy(V).copy().astype(np.float64)
                    N_basis_u_np = compute_basis_polynomials_derivatives(
                        n, p, np_U, np_u, order_u
                    )
                    N_basis_v_np = compute_basis_polynomials_derivatives(
                        m, q, np_V, np_v, order_v
                    )
                    N_basis_u = be.asarray(N_basis_u_np)
                    N_basis_v = be.asarray(N_basis_v_np)
                else:
                    N_basis_u = compute_basis_polynomials_derivatives(
                        n, p, U, u, order_u
                    )
                    N_basis_v = compute_basis_polynomials_derivatives(
                        m, q, V, v, order_v
                    )

                A = be.matmul(P, N_basis_v)
                B = be.stack([N_basis_u] * n_dim, axis=0)
                cols.append(be.sum(A * B, axis=1))
            rows.append(be.stack(cols, axis=0))
        bspline_derivatives = be.stack(rows, axis=0)
        return bspline_derivatives

    def get_normals(self, u, v):
        """Evaluates the unitary vectors normal to the surface.

        The definition of the unitary normal vector is given in section 19.2
        of Farin's textbook.

        Args:
            u: The u-parameter used to evaluate the normals.
            v: The v-parameter used to evaluate the normals.

        Returns:
            An array containing the unitary vectors normal to the surface.
        """
        S_u = self.get_derivative(u, v, order_u=1, order_v=0)
        S_v = self.get_derivative(u, v, order_u=0, order_v=1)

        normals = be.cross(S_u, S_v, axisa=0, axisb=0, axisc=0)
        normals = normals / be.sum(normals**2, axis=0) ** (1 / 2)

        return normals

    def _corr_general(self, u, v, d1, d2, N1, N2):
        """Defines the correction step for the update of u,v coordinates.

        See paper Practical ray tracing of trimmed NURBS surface - section 2.3.

        Args:
            u: The u-parameter.
            v: The v-parameter.
            d1: Contains minus dot product of N1 rows with ray (x,y,z).
            d2: Contains minus dot product of N2 rows with ray (x,y,z).
            N1: Each row of N1 is a normal vector of a plane.
            N2: Each row of N2 is a normal vector of a plane. Intersection of
                planes defined by N1[i,:] and N2[i,:] define a specific ray.

        Returns:
            A tuple containing the correction steps for u and v parameters,
            and the maximum distance between all rays and surface intersection.
        """
        S_uv = self.get_value(u, v).T
        r = be.stack(
            [be.sum(N1 * S_uv, axis=1) + d1, be.sum(N2 * S_uv, axis=1) + d2], axis=0
        )

        _, Np = r.shape
        a = be.sum(N1 * self.get_derivative(u, v, order_u=1, order_v=0).T, axis=1)
        b = be.sum(N1 * self.get_derivative(u, v, order_u=0, order_v=1).T, axis=1)
        c = be.sum(N2 * self.get_derivative(u, v, order_u=1, order_v=0).T, axis=1)
        d = be.sum(N2 * self.get_derivative(u, v, order_u=0, order_v=1).T, axis=1)

        J = be.vstack((a, b, c, d)).T.reshape((Np, 2, 2))
        adj = be.stack(
            [
                be.stack([J[:, 1, 1], -J[:, 0, 1]], axis=1),
                be.stack([-J[:, 1, 0], J[:, 0, 0]], axis=1),
            ],
            axis=1,
        )

        detJ = be.linalg.det(J)
        detJ = detJ[:, None, None]
        invj = adj / detJ

        correction = be.einsum("ijk,ki->ji", invj, r)
        residual = be.max(be.abs(r))

        return correction, residual

    def _corr(self, u, v, d1, d2):
        """Computes the intersection between rays and NURBS surface.

        This is a specific version of _corr_general that computes the
        intersection between rays and the NURBS surface when the rays'
        direction is along the Z-axis. This is used to compute the surface
        sag.

        Args:
            u: The u-parameter.
            v: The v-parameter.
            d1: A parameter related to the ray's position.
            d2: A parameter related to the ray's position.

        Returns:
            A tuple containing the correction steps and the residual.
        """
        S_uv = self.get_value(u, v)
        r = be.stack([S_uv[1, :] + d1, S_uv[0, :] + d2], axis=0)
        _, Np = r.shape
        a = self.get_derivative(u, v, order_u=1, order_v=0)[1, :]
        b = self.get_derivative(u, v, order_u=0, order_v=1)[1, :]
        c = self.get_derivative(u, v, order_u=1, order_v=0)[0, :]
        d = self.get_derivative(u, v, order_u=0, order_v=1)[0, :]

        J = be.vstack((a, b, c, d)).T.reshape((Np, 2, 2))
        adj = be.stack(
            [
                be.stack([J[:, 1, 1], -J[:, 0, 1]], axis=1),
                be.stack([-J[:, 1, 0], J[:, 0, 0]], axis=1),
            ],
            axis=1,
        )

        detJ = be.linalg.det(J)
        detJ = detJ[:, None, None]
        invj = adj / detJ

        correction = be.einsum("ijk,ki->ji", invj, r)
        residual = be.max(be.abs(r))

        return correction, residual

    def sag(self, x=0, y=0):
        """Computes the surface sag.

        The sag is computed for a specific x,y from the z-coordinate of the
        intersection point between the ray with direction (0,0,1) and passing
        from (x,y,0) and the NURBS surface.

        Args:
            x: The x-coordinate.
            y: The y-coordinate.

        Returns:
            The surface sag.
        """
        shape = x.shape
        u = be.zeros(be.size(x))
        v = be.zeros(be.size(x))
        for _ in range(self.max_iter):
            correction, residual = self._corr(u, v, -be.ravel(y), -be.ravel(x))
            u = u - correction[0, :]
            v = v - correction[1, :]
            u[be.logical_or(u < 0.0, v < 0.0)] = be.rand()
            v[be.logical_or(u < 0.0, v < 0.0)] = be.rand()
            u[be.logical_or(u > 1.0, v > 1.0)] = be.rand()
            v[be.logical_or(u > 1.0, v > 1.0)] = be.rand()

            if residual < self.tol:
                break
        return self.get_value(u, v)[2, :].reshape(shape)

    def distance(self, rays):
        """Finds the propagation distance to the geometry for the given rays.

        The approach is described in the paper "Practical ray tracing of
        trimmed NURBS surfaces" from William Martin etc.

        Args:
            rays: The rays for which to calculate the distance.

        Returns:
            An array of distances from each ray's current position to its
            intersection point with the geometry.
        """
        N1x = be.zeros_like(rays.x)
        N1y = be.zeros_like(rays.x)
        N1z = be.zeros_like(rays.x)
        mask = be.logical_and(rays.L > rays.M, rays.L > rays.N)

        N1x = be.where(
            mask,
            rays.M / be.sqrt(rays.L**2 + rays.M**2),
            N1x,
        )
        N1y = be.where(
            mask,
            -rays.L / be.sqrt(rays.L**2 + rays.M**2),
            N1y,
        )
        N1y = be.where(
            ~mask,
            rays.N / be.sqrt(rays.N**2 + rays.M**2),
            N1y,
        )
        N1z = be.where(
            ~mask,
            -rays.M / be.sqrt(rays.N**2 + rays.M**2),
            N1z,
        )

        N1 = be.column_stack((N1x, N1y, N1z))

        d = be.column_stack((rays.L, rays.M, rays.N))

        N2 = be.cross(N1, d)

        P0 = be.column_stack((rays.x, rays.y, rays.z))

        d1 = -be.sum(N1 * P0, axis=1)
        d2 = -be.sum(N2 * P0, axis=1)

        u = be.zeros_like(rays.x)
        v = be.zeros_like(u)

        for _ in range(self.max_iter):
            correction, residual = self._corr_general(u, v, d1, d2, N1, N2)
            u = u - correction[0, :]
            v = v - correction[1, :]
            u[be.logical_or(u < 0.0, v < 0.0)] = be.rand()
            v[be.logical_or(u < 0.0, v < 0.0)] = be.rand()
            u[be.logical_or(u > 1.0, v > 1.0)] = be.rand()
            v[be.logical_or(u > 1.0, v > 1.0)] = be.rand()
            if residual < self.tol:
                break

        t = be.sqrt(be.sum((self.get_value(u, v).T - P0) ** 2, axis=1))

        return t

    def surface_normal(self, rays):
        """Computes the surface normal.

        Args:
            rays: The rays for which to calculate the surface normal.

        Returns:
            A tuple containing the x, y, and z components of the surface
            normal.
        """
        x = rays.x
        y = rays.y
        u = be.zeros_like(x)
        v = be.zeros_like(u)
        for _ in range(self.max_iter):
            correction, residual = self._corr(u, v, -y, -x)
            u = u - correction[0, :]
            v = v - correction[1, :]
            u[be.logical_or(u < 0.0, v < 0.0)] = be.rand()
            v[be.logical_or(u < 0.0, v < 0.0)] = be.rand()
            u[be.logical_or(u > 1.0, v > 1.0)] = be.rand()
            v[be.logical_or(u > 1.0, v > 1.0)] = be.rand()
            if residual < self.tol:
                break
        n = self.get_normals(u, v)
        nx = n[0, :]
        ny = n[1, :]
        nz = n[2, :]

        return nx, ny, nz

    def __str__(self) -> str:
        return "NURBS"

    def fit_surface(self):
        """Handles the NURBS surface approximation.

        This function calls specific functions depending on the nature of the
        surface that we want to fit. For the time being, standard surface and
        plane surface are implemented.
        """
        if be.isinf(self.radius):
            self._plane_surface()
        else:
            self._standard_surface()

    def _standard_surface(self):
        """Generates NURBS parameters based on a standard surface."""
        radius = self.radius
        k = self.k
        nurbs_norm_x = self.nurbs_norm_x
        nurbs_norm_y = self.nurbs_norm_y
        xc = self.x_center
        yc = self.y_center
        P_size_u = self.P_size_u
        P_size_v = self.P_size_v
        P_ndim = self.ndim

        x = be.linspace(xc - nurbs_norm_x, xc + nurbs_norm_x, P_size_u)
        y = be.linspace(yc - nurbs_norm_y, yc + nurbs_norm_y, P_size_v)
        x, y = be.meshgrid(x, y)
        r2 = x**2 + y**2
        z = r2 / (radius * (1 + be.sqrt(1 - (1 + k) * r2 / radius**2)))
        points = be.stack((x.T, y.T, z.T), axis=0)

        xp = (points.reshape(P_ndim, -1).T).tolist()

        u_degree = 3
        v_degree = 3

        ctrlpts, u_degree, v_degree, num_cpts_u, num_cpts_v, kv_u, kv_v = (
            approximate_surface(xp, P_size_u, P_size_v, u_degree, v_degree)
        )

        self.P_size_u = num_cpts_u
        self.P_size_v = num_cpts_v

        ctrlpts = be.asarray(ctrlpts)
        ctrlpts = (ctrlpts.T).reshape((P_ndim, num_cpts_u, num_cpts_v))
        weights = be.ones((num_cpts_u, num_cpts_v))

        u_knots = be.asarray(kv_u)
        v_knots = be.asarray(kv_v)

        self.surface_type = "NURBS"

        self.P = ctrlpts
        self.W = weights
        self.p = u_degree
        self.q = v_degree
        self.U = u_knots
        self.V = v_knots

    def _plane_surface(self):
        """Generates a plane surface."""
        x = be.linspace(
            self.x_center - self.nurbs_norm_x,
            self.x_center + self.nurbs_norm_x,
            self.P_size_u,
        )
        y = be.linspace(
            self.y_center - self.nurbs_norm_y,
            self.y_center + self.nurbs_norm_y,
            self.P_size_v,
        )
        x, y = be.meshgrid(x, y)
        z = be.zeros_like(x)
        control_points = be.stack((x.T, y.T, z.T), axis=0)

        weights = be.ones((self.P_size_u, self.P_size_v))

        n = be.shape(control_points)[1] - 1
        m = be.shape(control_points)[2] - 1

        u_degree = 3
        v_degree = 3

        u_knots = be.concatenate(
            (
                be.zeros(u_degree),
                be.linspace(0, 1, n - u_degree + 2),
                be.ones(u_degree),
            )
        )
        v_knots = be.concatenate(
            (
                be.zeros(v_degree),
                be.linspace(0, 1, m - v_degree + 2),
                be.ones(v_degree),
            )
        )

        self.surface_type = "NURBS"
        self.P = control_points
        self.W = weights
        self.p = u_degree
        self.q = v_degree
        self.U = u_knots
        self.V = v_knots
