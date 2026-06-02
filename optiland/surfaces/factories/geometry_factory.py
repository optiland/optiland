"""Geometry Factory Module

This module contains the GeometryFactory class, which is responsible for generating
an appropriate geometry instance, given an input configuration. The class interfaces
tightly with the surface factory for building surfaces, which are the building
blocks of optical systems in Optiland.

Kramer Harrison, 2025
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.geometries import (
    BiconicGeometry,
    ChebyshevPolynomialGeometry,
    EvenAsphere,
    ForbesQ2dGeometry,
    ForbesQNormalSlopeGeometry,
    ForbesSolverConfig,  # forbes
    ForbesSurfaceConfig,  # forbes
    GridSagGeometry,
    NurbsGeometry,
    OddAsphere,
    Plane,
    PlaneGrating,
    PolynomialGeometry,
    StandardGeometry,
    StandardGratingGeometry,
    ToroidalGeometry,
    ZernikePolynomialGeometry,
)
from optiland.surfaces.factories.geometry_configs import (
    BiconicConfig,
    ChebyshevConfig,
    EvenAsphereConfig,
    ForbesQ2dConfig,
    ForbesQbfsConfig,
    GratingConfig,
    GridSagConfig,
    NurbsConfig,
    OddAsphereConfig,
    PlaneConfig,
    PolynomialConfig,
    StandardConfig,
    ToroidalConfig,
    ZernikeConfig,
    config_registry,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.coordinate_system import CoordinateSystem


def _create_plane(cs: CoordinateSystem, config: PlaneConfig):
    """
    Create a planar geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (PlaneConfig): configuration of the geometry.

    Returns:
        Plane
    """
    return Plane(coordinate_system=cs)


def _create_standard(cs: CoordinateSystem, config: StandardConfig):
    """
    Create a standard geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (StandardConfig): configuration of the geometry.

    Returns:
        StandardGeometry or Plane
    """
    # Use a Plane if the radius is infinity.
    if be.isinf(config.radius):
        return Plane(cs)
    return StandardGeometry(
        coordinate_system=cs, radius=config.radius, conic=config.conic
    )


def _create_even_asphere(cs: CoordinateSystem, config: EvenAsphereConfig):
    """
    Create an even asphere geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (EvenAsphereConfig): configuration of the geometry.

    Returns:
        EvenAsphere
    """
    return EvenAsphere(
        coordinate_system=cs,
        radius=config.radius,
        conic=config.conic,
        tol=config.tol,
        max_iter=config.max_iter,
        coefficients=config.coefficients,
    )


def _create_odd_asphere(cs: CoordinateSystem, config: OddAsphereConfig):
    """
    Create an odd asphere geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (OddAsphereConfig): configuration of the geometry.

    Returns:
        OddAsphere
    """
    return OddAsphere(
        coordinate_system=cs,
        radius=config.radius,
        conic=config.conic,
        tol=config.tol,
        max_iter=config.max_iter,
        coefficients=config.coefficients,
    )


def _create_polynomial(cs: CoordinateSystem, config: PolynomialConfig):
    """
    Create a polynomial geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (PolynomialConfig): configuration of the geometry.

    Returns:
        PolynomialGeometry
    """
    return PolynomialGeometry(
        coordinate_system=cs,
        radius=config.radius,
        conic=config.conic,
        tol=config.tol,
        max_iter=config.max_iter,
        coefficients=config.coefficients,
    )


def _create_grating(cs: CoordinateSystem, config: GratingConfig):
    """
    Create a grating geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (GratingConfig): configuration of the geometry.

    Returns:
        StandardGratingGeometry
    """
    # Use a Plane if the radius is infinity.
    if be.isinf(config.radius):
        return PlaneGrating(
            cs,
            config.grating_order,
            config.grating_period,
            config.groove_orientation_angle,
        )
    return StandardGratingGeometry(
        cs,
        config.radius,
        config.grating_order,
        config.grating_period,
        config.groove_orientation_angle,
        config.conic,
    )


def _create_chebyshev(cs: CoordinateSystem, config: ChebyshevConfig):
    """
    Create a Chebyshev geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (ChebyshevConfig): configuration of the geometry.

    Returns:
        ChebyshevPolynomialGeometry
    """
    return ChebyshevPolynomialGeometry(
        coordinate_system=cs,
        radius=config.radius,
        conic=config.conic,
        tol=config.tol,
        max_iter=config.max_iter,
        coefficients=config.coefficients,
        norm_x=config.norm_x,
        norm_y=config.norm_y,
    )


def _create_zernike(cs: CoordinateSystem, config: ZernikeConfig):
    """
    Create a Zernike geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (ZernikeConfig): configuration of the geometry.

    Returns:
        ZernikePolynomialGeometry
    """
    return ZernikePolynomialGeometry(
        coordinate_system=cs,
        radius=config.radius,
        conic=config.conic,
        tol=config.tol,
        max_iter=config.max_iter,
        zernike_type=config.zernike_type,
        coefficients=config.coefficients,
        norm_radius=config.norm_radius,
    )


def _create_biconic(cs: CoordinateSystem, config: BiconicConfig):
    """
    Create a biconic geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (BiconicConfig): configuration of the geometry.

    Returns:
        BiconicGeometry
    """
    if (
        be.isinf(config.radius_x)
        and be.isinf(config.radius_y)
        and config.conic_x == 0.0
        and config.conic_y == 0.0
    ):
        # If all radii are infinite and conics are zero, it's a plane
        return Plane(cs)
    return BiconicGeometry(
        coordinate_system=cs,
        radius_x=config.radius_x,
        radius_y=config.radius_y,
        conic_x=config.conic_x,
        conic_y=config.conic_y,
        tol=config.tol,
        max_iter=config.max_iter,
    )


def _create_toroidal(cs: CoordinateSystem, config: ToroidalConfig):
    """
    Create a Toroidal geometry

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (ToroidalConfig): configuration of the geometry.

    Returns:
        ToroidalGeometry
    """

    return ToroidalGeometry(
        coordinate_system=cs,
        radius_x=config.radius_x,  # Toroidal uses radius_x for its X radius
        radius_y=config.radius_y,  # Toroidal uses 'radius_y' for its YZ radius
        conic=config.conic,
        coeffs_poly_y=config.toroidal_coeffs_poly_y,
        tol=config.tol,
        max_iter=config.max_iter,
    )


def _create_forbes_qbfs(cs: CoordinateSystem, config: ForbesQbfsConfig):
    """Create a Forbes Q (slope-orthogonal) Geometry.

    Note: The surface type key \"forbes_qbfs\" is retained for backward
    compatibility, but the canonical class is now ForbesQNormalSlopeGeometry.
    """
    surface_config = ForbesSurfaceConfig(
        radius=config.radius,
        conic=config.conic,
        terms=config.radial_terms,
        norm_radius=config.norm_radius,
    )
    solver_config = ForbesSolverConfig(tol=config.tol, max_iter=config.max_iter)

    return ForbesQNormalSlopeGeometry(
        cs,
        surface_config=surface_config,
        solver_config=solver_config,
    )


def _create_forbes_q2d(cs: CoordinateSystem, config: ForbesQ2dConfig):
    """Create a Forbes (Q-2D) geometry."""
    surface_config = ForbesSurfaceConfig(
        radius=config.radius,
        conic=config.conic,
        terms=config.freeform_coeffs,
        norm_radius=config.norm_radius,
    )
    solver_config = ForbesSolverConfig(tol=config.tol, max_iter=config.max_iter)

    return ForbesQ2dGeometry(
        cs,
        surface_config=surface_config,
        solver_config=solver_config,
    )


def _create_nurbs(cs: CoordinateSystem, config: NurbsConfig):
    """Create a NURBS geometry."""

    return NurbsGeometry(
        cs,
        config.radius,
        config.conic,
        config.nurbs_norm_x,
        config.nurbs_norm_y,
        config.nurbs_x_center,
        config.nurbs_y_center,
        config.control_points,
        config.weights,
        config.u_degree,
        config.v_degree,
        config.u_knots,
        config.v_knots,
        config.n_points_u,
        config.n_points_v,
        config.tol,
        config.max_iter,
    )


def _create_grid_sag(cs: CoordinateSystem, config: GridSagConfig):
    """Create a Grid Sag geometry."""
    return GridSagGeometry(
        coordinate_system=cs,
        x_coordinates=config.x_coordinates,
        y_coordinates=config.y_coordinates,
        sag_values=config.sag_values,
        tol=config.tol,
        max_iter=config.max_iter,
    )


def _create_paraxial(cs: CoordinateSystem, config: PlaneConfig):
    """
    Create a paraxial geometry, which is simply a planar surface.

    Args:
        cs (CoordinateSystem): coordinate system of the geometry.
        config (PlaneConfig): configuration of the geometry.

    Returns:
        Plane
    """
    return _create_plane(cs, config)


geometry_mapper = {
    "biconic": _create_biconic,
    "chebyshev": _create_chebyshev,
    "even_asphere": _create_even_asphere,
    "grating": _create_grating,
    "grid_sag": _create_grid_sag,
    "odd_asphere": _create_odd_asphere,
    "paraxial": _create_paraxial,
    "plane": _create_plane,
    "polynomial": _create_polynomial,
    "standard": _create_standard,
    "toroidal": _create_toroidal,
    "zernike": _create_zernike,
    "forbes_qbfs": _create_forbes_qbfs,
    "forbes_q2d": _create_forbes_q2d,
    "nurbs": _create_nurbs,
}


class GeometryFactory:
    """Factory for creating surface geometry objects based on configuration."""

    @classmethod
    def register(
        cls,
        name: str,
        create_fn: Callable,
        config_cls: type,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a new geometry type.

        Args:
            name: The surface_type string key (e.g. 'even_asphere').
            create_fn: A function ``(cs, config) -> geometry`` instance.
            config_cls: A dataclass whose fields define the accepted kwargs.
            overwrite: Allow replacing an existing registration.

        Raises:
            ValueError: If name is already registered and overwrite is False.
        """
        if name in geometry_mapper and not overwrite:
            raise ValueError(
                f"Geometry type '{name}' is already registered. "
                "Pass overwrite=True to replace it."
            )
        geometry_mapper[name] = create_fn
        config_registry[name] = config_cls

    @staticmethod
    def create(surface_type: str, cs: Any, **kwargs: Any) -> Any:
        """
        Create and return a geometry object based on the surface type and configuration.
        Args:
            surface_type (str): The type of surface (e.g., 'standard', 'even_asphere').
            cs: The coordinate system for the geometry.
            **kwargs: Configuration parameters for the geometry.
        Returns:
            The constructed geometry object.
        Raises:
            ValueError: If the surface type is not recognized.
        """
        try:
            config_cls = config_registry[surface_type]
            create_fn = geometry_mapper[surface_type]
        except KeyError as err:
            raise ValueError(f"Surface type '{surface_type}' not recognized.") from err

        # Filter kwargs to only include those relevant to the specific config class
        config_fields = {f.name for f in fields(config_cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

        config = config_cls(**filtered_kwargs)
        return create_fn(cs, config)
