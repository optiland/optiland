"""Forbes Polynomial Geometries for Optical Surfaces.

This module provides implementations for optical surfaces described by
Forbes polynomials, which are superimposed on a base conic section.
Forbes polynomials offer a modern alternative to standard power-series
aspheres.

The implementation is based on the work of G. W. Forbes. See, for example,
G. W. Forbes, "Manufacturability estimates for optical aspheres," Opt. Express (2011).

This module implements two types of Forbes surfaces:
    1.  **ForbesQNormalSlopeGeometry** (slope-orthogonal Q polynomials):
        This class represents rotationally symmetric aspheric surfaces using
        Forbes Q polynomials (historically called Q^bfs in Forbes 2007). These
        polynomials are orthogonal with respect to the normal-departure
        slope metric, and the implementation supports a general conic
        reference surface (Forbes 2011 generalization). The deprecated alias
        ``ForbesQbfsGeometry`` is retained for backward compatibility.
    2.  **ForbesQ2dGeometry (Q-type, 2D)**:
        This class represents non-rotationally symmetric, or "freeform," surfaces.
        The Q2D polynomials extend the Q-type formalism to two dimensions, allowing
        for the description of complex, freeform optical surfaces that lack
        rotational symmetry.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry

from .qpoly import (
    clenshaw_q2d,
    clenshaw_qbfs,
    compute_z_zprime_q2d,
    compute_z_zprime_qbfs,
    q2d_nm_coeffs_to_ams_bms,
    q2d_sum_from_alphas,
)

_EPSILON = 1e-12


@dataclass
class ForbesSolverConfig:
    """Configuration for the Newton-Raphson numerical solver.

    Attributes:
        tol (float): Tolerance for the iterative solver.
        max_iter (int): Maximum number of iterations for the solver.
    """

    tol: float = 1e-10
    max_iter: int = 100


@dataclass
class ForbesSurfaceConfig:
    """Configuration for a surface's core geometric properties.

    Attributes:
        radius (float): The vertex radius of curvature of the base surface.
        conic (float): The conic constant of the base surface.
        norm_radius (float, optional): The normalization radius for the
            polynomial terms. If None, the radius scales automatically
            during paraxial updates. Defaults to None.
        terms (dict, optional): A dictionary of polynomial coefficients.
        The key format depends on the
            specific geometry type (e.g., `radial_terms` for Qbfs,
            `freeform_coeffs` for Q2d).
    """

    radius: float
    conic: float = 0.0
    norm_radius: float | None = None
    # either radial_terms or freeform_coeffs
    terms: dict[Any, float] | None = None


class ForbesGeometryBase(NewtonRaphsonGeometry):
    """Base class for Forbes geometries to share common mathematical logic."""

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        surface_config: ForbesSurfaceConfig,
        solver_config: ForbesSolverConfig = None,
    ):
        """Initializes the base Forbes geometry.

        Args:
            coordinate_system (CoordinateSystem): The local coordinate system of
                                                    the surface.
            surface_config (ForbesSurfaceConfig): An object containing the
                                                    core geometric parameters.
            solver_config (ForbesSolverConfig, optional): An object containing
                                            parameters for the numerical solver.
                If None, defaults are used.
        """
        if solver_config is None:
            solver_config = ForbesSolverConfig()

        super().__init__(
            coordinate_system,
            surface_config.radius,
            surface_config.conic,
            solver_config.tol,
            solver_config.max_iter,
        )
        self.surface_config = surface_config
        self.solver_config = solver_config

    def _base_sag(self, r2):
        """Calculates the sag of the base conic surface.

        Args:
            r2 (float or array_like): The squared radial coordinate (rho^2).

        Returns:
            float or array_like: The sag of the base conic.
        """
        if be.isinf(self.radius):
            return be.zeros_like(r2)

        sqrt_arg = 1 - (1 + self.k) * r2 / self.radius**2
        safe_sqrt_arg = be.where(sqrt_arg < 0, 0, sqrt_arg)
        return r2 / (self.radius * (1 + be.sqrt(safe_sqrt_arg)))

    def _base_sag_derivative(self, rho, r2):
        """Calculates the derivative of the base conic sag w.r.t. rho.

        Args:
            rho (float or array_like): The radial coordinate (rho).
            r2 (float or array_like): The squared radial coordinate (rho^2).

        Returns:
            float or array_like: The derivative dz_base/drho.
        """
        if be.isinf(self.radius) or self.radius == 0:
            return be.zeros_like(rho)

        c = 1.0 / self.radius
        sqrt_arg_base = 1 - (self.k + 1) * c**2 * r2
        safe_sqrt_base = be.sqrt(be.where(sqrt_arg_base > 0, sqrt_arg_base, 1e-12))
        return c * rho / safe_sqrt_base

    def _conic_correction_factor(self, r2):
        """Calculates the Forbes conic correction factor and its derivative.

        This factor projects the normal departure from the base conic onto the
        sag axis.

        Args:
            r2 (float or array_like): The squared radial coordinate (rho^2).

        Returns:
            tuple[float or array_like, float or array_like]: A tuple containing:
                - factor (float or array_like): The unitless conic correction factor.
                - derivative (float or array_like): The derivative of the
                                                        factor with respect to rho.
        """
        if be.isinf(self.radius):
            return 1.0, 0.0

        c2 = (1.0 / self.radius) ** 2
        rho = be.sqrt(r2 + _EPSILON**2)
        num_arg = 1 - self.k * c2 * r2
        den_arg = 1 - (self.k + 1) * c2 * r2

        safe_num_arg = be.where(num_arg > 0, num_arg, 1e-12)
        safe_den_arg = be.where(den_arg > 0, den_arg, 1e-12)
        N = be.sqrt(safe_num_arg)
        D = be.sqrt(safe_den_arg)

        factor = N / D
        derivative = (c2 * rho) / (N * D**3)
        return factor, derivative


class ForbesQNormalSlopeGeometry(ForbesGeometryBase):
    r"""Represents a Forbes Q surface using slope-orthogonal polynomials.

    This surface uses the Forbes Q polynomials that are orthogonal with respect
    to the normal-departure slope metric. These were historically called
    Q^bfs ("best-fit sphere") in Forbes 2007, but the implementation here
    supports a general conic reference surface following the Forbes 2011
    generalization.

    The surface sag is defined as:
    $z(\rho) = z_{base}(\rho) + \phi(\rho)
    \left[ u^2(1-u^2) \sum_{m=0}^{M} a_m Q_m(u^2) \right]$

    where:
        - $z_{base}(\rho) = \frac{c\rho^2}{1 + \sqrt{1 - (1+k)c^2\rho^2}}$
          is the base conic.
        - $c = 1/R$ is the curvature, $k$ is the conic constant.
        - $u = \rho/\rho_{max}$ is the normalized radial coordinate.
        - $Q_m(u^2)$ are the Forbes slope-orthogonal polynomials.
        - $a_m$ are the polynomial coefficients.
        - $\phi(\rho) = \sqrt{\frac{1 - kc^2\rho^2}{1 - (1+k)c^2\rho^2}}$ is the
          conic correction factor that projects the normal departure onto the
          sag axis.

    Note:
        The internal polynomial basis functions (named ``*_qbfs`` in the code)
        retain the historical "qbfs" identifier for stability. This does not
        imply a spherical reference; the conic constant $k$ can be nonzero.

    Args:
        coordinate_system (CoordinateSystem): The local coordinate system of the
                                                surface.
        surface_config (ForbesSurfaceConfig): An object containing the core
            geometric parameters. The `terms` dictionary should be provided
            as `radial_terms`.
        solver_config (ForbesSolverConfig, optional): An object containing
            parameters for the numerical solver.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        surface_config: ForbesSurfaceConfig,
        solver_config: ForbesSolverConfig = None,
    ):
        super().__init__(coordinate_system, surface_config, solver_config)
        if be.get_backend() == "torch":
            self.radial_terms = {
                k: be.array(v) for k, v in (self.surface_config.terms or {}).items()
            }
        else:
            self.radial_terms = self.surface_config.terms or {}

        if self.surface_config.norm_radius is not None:
            self.normalization_mode = "manual"
            self.norm_radius = be.array(self.surface_config.norm_radius)
        else:
            self.normalization_mode = "auto"
            self.norm_radius = be.array(1.0)

        self.is_symmetric = True

    def _prepare_coeffs(self):
        """Prepares the internal coefficient lists from the radial_terms dictionary."""
        if not self.radial_terms:
            self.coeffs_n, self.coeffs_c = [], be.array([])
            return

        max_n = max(self.radial_terms.keys())
        if max_n >= 0:
            terms_list = [
                self.radial_terms.get(n, be.array(0.0)) for n in range(max_n + 1)
            ]
            self.coeffs_c = be.stack(terms_list)

            self.coeffs_n = [(n, 0) for n in range(max_n + 1)]
        else:
            self.coeffs_n, self.coeffs_c = [], be.array([])

    def sag(self, x=0, y=0):
        """Calculate the sag of the Forbes Q (slope-orthogonal) surface.

        Args:
            x (int, optional): x-coordinate. Defaults to 0.
            y (int, optional): y-coordinate. Defaults to 0.

        Returns:
            The sag of the Forbes Q (slope-orthogonal) surface.
        """
        self._prepare_coeffs()
        x, y = be.array(x), be.array(y)
        r2 = x**2 + y**2
        z_base = self._base_sag(r2)

        usq = r2 / (self.norm_radius**2)

        poly_sum_m0 = clenshaw_qbfs(self.coeffs_c, usq)
        prefactor = usq * (1 - usq)
        conic_correction_factor, _ = self._conic_correction_factor(r2)
        departure = prefactor * conic_correction_factor * poly_sum_m0

        S = be.where(usq > 1, 0.0, departure)
        return z_base + S

    def _surface_normal(self, x, y):
        """Calculates the unit vector normal to the surface.

        Uses analytical method for both torch and numpy backends, with proper
        handling of the vertex case to avoid NaN gradients.

        Args:
            x (float or array_like): X coordinate(s).
            y (float or array_like): Y coordinate(s).

        Returns:
            tuple[float or array_like, float or array_like, float or array_like]:
                Components of the unit normal vector (nx, ny, nz).
        """
        self._prepare_coeffs()
        x_in, y_in = be.array(x), be.array(y)

        df_dx, df_dy = self._surface_normal_analytical(x_in, y_in)

        mag = be.sqrt(df_dx**2 + df_dy**2 + 1)
        safe_mag = be.where(mag < _EPSILON, 1.0, mag)
        return df_dx / safe_mag, df_dy / safe_mag, -1 / safe_mag

    def _surface_normal_analytical(self, x, y):
        """Computes the analytical surface derivatives.

        This method handles the vertex case by ensuring numerically stable
        computation that works with both numpy and torch autodiff.

        Args:
            x (float or array_like): X coordinate(s).
            y (float or array_like): Y coordinate(s).

        Returns:
            tuple[float or array_like, float or array_like]:
                The partial derivatives (df_dx, df_dy).
        """
        r2 = x**2 + y**2

        # For exact vertex in torch autodiff, add a tiny perturbation to ensure
        # gradient stability in full ray tracing context
        is_exact_vertex = (be.abs(x) < _EPSILON) & (be.abs(y) < _EPSILON)
        if be.get_backend() == "torch" and be.any(is_exact_vertex):
            # Add tiny perturbation only at exact vertex for torch
            x_safe = be.where(is_exact_vertex, x + _EPSILON * 1e-3, x)
            y_safe = be.where(is_exact_vertex, y, y)
            r2_safe = x_safe**2 + y_safe**2
            rho_safe = be.sqrt(r2_safe + _EPSILON**2)
        else:
            x_safe, y_safe = x, y
            r2_safe = r2
            rho_safe = be.sqrt(r2 + _EPSILON**2)

        ds_base_d_rho = self._base_sag_derivative(rho_safe, r2_safe)

        if len(self.coeffs_c) == 0 or be.all(self.coeffs_c == 0):
            df_d_rho = ds_base_d_rho
        else:
            u = rho_safe / self.norm_radius

            poly_val, dpoly_d_u = compute_z_zprime_qbfs(self.coeffs_c, u, u**2)
            dprefactor_d_rho = (2 * u - 4 * u**3) / self.norm_radius
            dpoly_d_rho = dpoly_d_u / self.norm_radius
            conic_factor, dconic_factor_d_rho = self._conic_correction_factor(r2_safe)

            usq = u**2
            ds_dep_d_rho = (
                dprefactor_d_rho * conic_factor * poly_val
                + (usq - usq**2) * dconic_factor_d_rho * poly_val
                + (usq - usq**2) * conic_factor * dpoly_d_rho
            )
            df_d_rho = ds_base_d_rho + be.where(u >= 1, 0.0, ds_dep_d_rho)

        # Use the safe coordinates for directional computation
        cos_t = x_safe / rho_safe
        sin_t = y_safe / rho_safe

        # Ensure smooth behavior at vertex for rotationally symmetric surfaces
        df_dx = df_d_rho * cos_t
        df_dy = df_d_rho * sin_t

        return df_dx, df_dy

    def _surface_normal_analytical_vertex(self):
        """Computes the stable analytical derivative exactly at the vertex.

        For rotationally symmetric surfaces, the gradient at the vertex
        (x=0, y=0) should be (0, 0) due to symmetry.

        Returns:
            tuple[float, float]: The partial derivatives (df_dx, df_dy) at the vertex.
        """
        # For rotationally symmetric surfaces, the gradient at the vertex is (0, 0)
        # due to symmetry - there's no preferred direction for the slope
        return 0.0, 0.0

    def to_dict(self):
        """Serializes the geometry to a dictionary.

        Returns:
            dict: A dictionary representation of the geometry.
        """
        return {
            "type": self.__class__.__name__,
            "cs": self.cs.to_dict(),
            "surface_config": asdict(self.surface_config),
            "solver_config": asdict(self.solver_config),
        }

    @classmethod
    def from_dict(cls, data):
        """Creates an instance from a dictionary.

        Accepts both "ForbesQNormalSlopeGeometry" and legacy "ForbesQbfsGeometry"
        type identifiers for backward compatibility.

        Args:
            data (dict): A dictionary representation of the geometry.

        Returns:
            ForbesQNormalSlopeGeometry: An instance of the class.
        """
        cs = CoordinateSystem.from_dict(data["cs"])
        surface_config = ForbesSurfaceConfig(**data["surface_config"])
        solver_config = ForbesSolverConfig(**data.get("solver_config", {}))
        # Always return the canonical class, regardless of whether the
        # serialized type was "ForbesQbfsGeometry" or "ForbesQNormalSlopeGeometry"
        return ForbesQNormalSlopeGeometry(cs, surface_config, solver_config)

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Scales the radius, normalization radius, and radial coefficients.
        The polynomial coefficients scale linearly with the sag when the
        normalization radius is also scaled.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        super().scale(scale_factor)
        self.surface_config.radius = self.radius
        if self.surface_config.norm_radius is not None:
            self.surface_config.norm_radius *= scale_factor
        self.norm_radius = self.norm_radius * scale_factor

        for key in self.radial_terms:
            self.radial_terms[key] = self.radial_terms[key] * scale_factor

    def update_normalization(self, semi_aperture: float) -> None:
        if self.normalization_mode == "auto":
            self.norm_radius = be.array(semi_aperture * 1.25)
            self.surface_config.norm_radius = float(self.norm_radius)

    def __str__(self):
        return "ForbesQNormalSlope"


class ForbesQ2dGeometry(ForbesGeometryBase):
    r"""Forbes Q2D freeform surface.

    The Q2D surface is defined by a departure $\delta(u, \theta)$ from a base conic:
    $z(\rho, \theta) = z_{base}(\rho) + \frac{1}{\sigma(\rho)} \delta(u, \theta)$

    The departure term $\delta(u, \theta)$ is given by:
    $\delta(u, \theta) = u^2(1-u^2)\sum_{n=0}^{N} a_n^0 Q_n^0(u^2) +
    \sum_{m=1}^{M} u^m \sum_{n=0}^{N} [a_n^m \cos(m\theta) +
    b_n^m \sin(m\theta)] Q_n^m(u^2)$

    Args:
        coordinate_system (CoordinateSystem): The local coordinate system of the
                                                surface.
        surface_config (ForbesSurfaceConfig): An object containing the core geometric
            parameters. The `terms` dictionary should be provided as `freeform_coeffs`.
        solver_config (ForbesSolverConfig, optional): An object containing parameters
        for the numerical solver.

    Notes:
        The `freeform_coeffs` dictionary keys follow the Zemax convention to ensure
        intuitive use for optical designers.
        - Cosine term $a_n^m$: `('a', m, n)`
        - Sine term $b_n^m$: `('b', m, n)`
        Note the index order `(m, n)` matches the Zemax UI, where `m` is the
        azimuthal frequency and `n` is the radial order. The code handles the
        translation to the mathematical convention internally.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        surface_config: ForbesSurfaceConfig,
        solver_config: ForbesSolverConfig = None,
    ):
        super().__init__(coordinate_system, surface_config, solver_config)
        self.c = 1 / self.radius if self.radius != 0 else 0
        if be.get_backend() == "torch":
            self.freeform_coeffs = {
                k: be.array(v) for k, v in (self.surface_config.terms or {}).items()
            }
        else:
            self.freeform_coeffs = self.surface_config.terms or {}

        if self.surface_config.norm_radius is not None:
            self.normalization_mode = "manual"
            self.norm_radius = be.array(self.surface_config.norm_radius)
        else:
            self.normalization_mode = "auto"
            self.norm_radius = be.array(1.0)

        self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = [], [], []
        self._prepare_coeffs()

    def _prepare_coeffs(self):
        """
        Translates the user-facing Zemax-style `freeform_coeffs` dictionary
        into the internal coefficient lists required by the `qpoly` backend.
        """
        if not self.freeform_coeffs:
            self.coeffs_n, self.coeffs_c = [], be.array([])
            return

        internal_coeffs = {}
        for key, value in self.freeform_coeffs.items():
            term_type, idx1, idx2 = key
            if term_type.lower() == "a":
                n, m = idx2, idx1
                internal_coeffs[(n, m)] = value
            elif term_type.lower() == "b":
                n, m = idx2, idx1
                internal_coeffs[(n, m, "sin")] = value

        sorted_keys = sorted(
            internal_coeffs.keys(),
            key=lambda k: (k[0], abs(k[1]), 0 if len(k) == 2 else 1),
        )

        coeffs_n, coeffs_c = [], []
        for key in sorted_keys:
            value = internal_coeffs[key]
            m_val = -key[1] if len(key) == 3 and key[2].lower() == "sin" else key[1]
            coeffs_n.append((key[0], m_val))
            coeffs_c.append(value)

        self.coeffs_n, self.coeffs_c = (
            coeffs_n,
            be.stack(coeffs_c) if coeffs_c else be.array([]),
        )
        if self.coeffs_n:
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs = (
                q2d_nm_coeffs_to_ams_bms(self.coeffs_n, self.coeffs_c)
            )

    def sag(self, x, y):
        """Calculate the sag of the Forbes Q2D freeform surface.

        Args:
            x (float or array_like): x-coordinate
            y (float or array_like): y-coordinate

        Returns:
            float or array_like: The sag of the Forbes Q2D freeform surface.
        """
        x, y = be.array(x), be.array(y)
        r2 = x**2 + y**2
        z_base = self._base_sag(r2)

        rho = be.sqrt(r2 + _EPSILON)

        u = rho / self.norm_radius
        safe_x = be.where(r2 < _EPSILON, x + 1e-12, x)
        theta = be.arctan2(y, safe_x)

        poly_sum_m0, _, poly_sum_m_gt0, _, _ = compute_z_zprime_q2d(
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs, u, theta
        )
        conic_correction_factor, _ = self._conic_correction_factor(r2)
        usq = u**2
        prefactor_m0 = usq * (1 - usq)

        departure_m0 = prefactor_m0 * conic_correction_factor * poly_sum_m0
        departure_m_gt0 = conic_correction_factor * poly_sum_m_gt0
        total_departure = departure_m0 + departure_m_gt0

        S = be.where(u > 1, 0.0, total_departure)
        return z_base + S

    def _surface_normal(self, x, y):
        """Calculates the unit vector normal to the surface.

        Dispatches to an autograd-based method for the torch backend and an
        analytical method for the numpy backend. For torch, it patches the `NaN`
        gradient from autograd at the vertex with the correct analytical value.

        Args:
            x (float or array_like): X coordinate(s).
            y (float or array_like): Y coordinate(s).

        Returns:
            tuple[float or array_like, float or array_like, float or array_like]:
                Components of the unit normal vector (nx, ny, nz).
        """
        x_in, y_in = be.array(x), be.array(y)

        df_dx, df_dy = self._surface_normal_analytical(x_in, y_in)

        mag = be.sqrt(df_dx**2 + df_dy**2 + 1)
        safe_mag = be.where(mag < _EPSILON, 1.0, mag)
        return df_dx / safe_mag, df_dy / safe_mag, -1.0 / safe_mag

    def _surface_normal_analytical_vertex(self):
        """Computes the stable analytical derivative exactly at the vertex."""
        df_dx_vertex, df_dy_vertex = 0.0, 0.0
        if self.ams_coeffs and self.ams_coeffs[0]:
            a_coeffs = self.ams_coeffs[0]
            alphas_a = clenshaw_q2d(a_coeffs, m=1, usq=0.0)
            sum_a1 = q2d_sum_from_alphas(alphas_a, m=1, num_coeffs=len(a_coeffs))
            df_dx_vertex = sum_a1 / self.norm_radius
        if self.bms_coeffs and self.bms_coeffs[0]:
            b_coeffs = self.bms_coeffs[0]
            alphas_b = clenshaw_q2d(b_coeffs, m=1, usq=0.0)
            sum_b1 = q2d_sum_from_alphas(alphas_b, m=1, num_coeffs=len(b_coeffs))
            df_dy_vertex = sum_b1 / self.norm_radius
        return df_dx_vertex, df_dy_vertex

    def _surface_normal_analytical(self, x_in, y_in):
        """
        Computes the analytical surface derivatives for the numpy backend.

        This method is fully vectorized and uses a `where` clause to combine
        the stable vertex calculation with the general non-vertex calculation.

        Args:
            x_in (float or array_like): x-coordinate
            y_in (float or array_like): y-coordinate

        Returns:
            tuple[float or array_like, float or array_like]: The analytical surface
                                                derivatives for the numpy backend.
        """
        df_dx_vertex, df_dy_vertex = self._surface_normal_analytical_vertex()

        r2 = x_in**2 + y_in**2
        is_vertex = r2 < _EPSILON**2
        rho = be.sqrt(r2 + _EPSILON**2)

        rho_safe = be.where(is_vertex, _EPSILON, rho)
        u = rho / self.norm_radius
        safe_x = be.where(is_vertex, x_in + _EPSILON, x_in)
        theta = be.arctan2(y_in, safe_x)

        vals = compute_z_zprime_q2d(
            self.cm0_coeffs, self.ams_coeffs, self.bms_coeffs, u, theta
        )
        poly_sum_m0, d_poly_m0_du, poly_sum_m_gt0, dr_poly_m_gt0_du, dt_poly_m_gt0 = (
            vals
        )

        d_poly_m0_drho = d_poly_m0_du / self.norm_radius
        dr_poly_m_gt0_drho = dr_poly_m_gt0_du / self.norm_radius
        conic_factor, dconic_d_rho = self._conic_correction_factor(r2)

        usq = u**2
        dprefactor_d_rho = (2 * u - 4 * u**3) / self.norm_radius
        ds0_d_rho = (
            dprefactor_d_rho * poly_sum_m0 + (usq - usq**2) * d_poly_m0_drho
        ) * conic_factor + (usq - usq**2) * poly_sum_m0 * dconic_d_rho

        ds_gt0_d_rho = (dconic_d_rho * poly_sum_m_gt0) + (
            conic_factor * dr_poly_m_gt0_drho
        )
        ds_d_rho = be.where(u > 1, 0.0, ds0_d_rho + ds_gt0_d_rho)
        ds_d_theta = be.where(u > 1, 0.0, conic_factor * dt_poly_m_gt0)

        cos_t, sin_t = x_in / rho_safe, y_in / rho_safe
        ds_dx = cos_t * ds_d_rho - (sin_t / rho_safe) * ds_d_theta
        ds_dy = sin_t * ds_d_rho + (cos_t / rho_safe) * ds_d_theta

        d_base_dr = self._base_sag_derivative(rho, r2)
        d_base_dx, d_base_dy = d_base_dr * cos_t, d_base_dr * sin_t

        df_dx_non_vertex = d_base_dx + ds_dx
        df_dy_non_vertex = d_base_dy + ds_dy

        df_dx = be.where(is_vertex, df_dx_vertex, df_dx_non_vertex)
        df_dy = be.where(is_vertex, df_dy_vertex, df_dy_non_vertex)

        return df_dx, df_dy

    def to_dict(self):
        """Serializes the geometry to a dictionary.

        Returns:
            dict: A dictionary representation of the geometry.
        """
        return {
            "type": self.__class__.__name__,
            "cs": self.cs.to_dict(),
            "surface_config": asdict(self.surface_config),
            "solver_config": asdict(self.solver_config),
        }

    @classmethod
    def from_dict(cls, data):
        """Creates an instance from a dictionary.

        Args:
            data (dict): A dictionary representation of the geometry.

        Returns:
            ForbesQbfsGeometry: An instance of the class.
        """
        cs = CoordinateSystem.from_dict(data["cs"])
        surface_config = ForbesSurfaceConfig(**data["surface_config"])
        solver_config = ForbesSolverConfig(**data.get("solver_config", {}))
        return cls(cs, surface_config, solver_config)

    def scale(self, scale_factor: float):
        """Scale the geometry parameters.

        Scales the radius, normalization radius, and freeform coefficients.
        The polynomial coefficients scale linearly with the sag when the
        normalization radius is also scaled.

        Args:
            scale_factor (float): The factor by which to scale the geometry.
        """
        super().scale(scale_factor)
        self.surface_config.radius = self.radius
        if self.surface_config.norm_radius is not None:
            self.surface_config.norm_radius *= scale_factor
        self.norm_radius = self.norm_radius * scale_factor

        for key in self.freeform_coeffs:
            self.freeform_coeffs[key] = self.freeform_coeffs[key] * scale_factor

        self.c = 1 / self.radius if self.radius != 0 else 0
        self._prepare_coeffs()

    def update_normalization(self, semi_aperture: float) -> None:
        if self.normalization_mode == "auto":
            self.norm_radius = be.array(semi_aperture * 1.25)
            self.surface_config.norm_radius = float(self.norm_radius)

    def __str__(self):
        return "ForbesQ2d"


class ForbesQbfsGeometry(ForbesQNormalSlopeGeometry):
    """Deprecated alias for ForbesQNormalSlopeGeometry.

    .. deprecated::
        Use :class:`ForbesQNormalSlopeGeometry` instead. The name \"Qbfs\" was
        historically used in the Forbes literature for the slope-orthogonal
        Q basis, but it misleadingly suggests a best-fit sphere reference.
        The implementation supports a general conic reference.
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        surface_config: ForbesSurfaceConfig,
        solver_config: ForbesSolverConfig = None,
    ):
        warnings.warn(
            "ForbesQbfsGeometry is deprecated; use ForbesQNormalSlopeGeometry "
            "(slope-orthogonal Q basis with a general conic reference).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(coordinate_system, surface_config, solver_config)

    def __str__(self):
        # Keep legacy string for backward compatibility in displays
        return "ForbesQbfs"
