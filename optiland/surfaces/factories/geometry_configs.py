"""Geometry Configuration Module

This module contains the dataclasses for configuring different geometry types.

Kramer Harrison, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import optiland.backend as be

if TYPE_CHECKING:
    from optiland._types import ZernikeType

config_registry: dict[str, type[GeometryConfig]] = {}


class GeometryConfig:
    """Base class for geometry configurations."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "surface_type"):
            config_registry[cls.surface_type] = cls


@dataclass
class StandardConfig(GeometryConfig):
    """Configuration for a standard geometry."""

    surface_type: ClassVar[str] = "standard"
    radius: float = be.inf
    conic: float = 0.0


@dataclass
class PlaneConfig(GeometryConfig):
    """Configuration for a plane geometry."""

    surface_type: ClassVar[str] = "plane"


@dataclass
class EvenAsphereConfig(GeometryConfig):
    """Configuration for a even_asphere geometry."""

    surface_type: ClassVar[str] = "even_asphere"
    radius: float = be.inf
    conic: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    tol: float = 1e-6
    max_iter: int = 100


@dataclass
class OddAsphereConfig(GeometryConfig):
    """Configuration for a odd_asphere geometry."""

    surface_type: ClassVar[str] = "odd_asphere"
    radius: float = be.inf
    conic: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    tol: float = 1e-6
    max_iter: int = 100


@dataclass
class PolynomialConfig(GeometryConfig):
    """Configuration for a polynomial geometry."""

    surface_type: ClassVar[str] = "polynomial"
    radius: float = be.inf
    conic: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    tol: float = 1e-6
    max_iter: int = 100


@dataclass
class GratingConfig(GeometryConfig):
    """Configuration for a grating geometry."""

    surface_type: ClassVar[str] = "grating"
    radius: float = be.inf
    conic: float = 0.0
    grating_order: int = 0
    grating_period: float = be.inf
    groove_orientation_angle: float = 0.0


@dataclass
class ChebyshevConfig(GeometryConfig):
    """Configuration for a chebyshev geometry."""

    surface_type: ClassVar[str] = "chebyshev"
    radius: float = be.inf
    conic: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    tol: float = 1e-6
    max_iter: int = 100
    norm_x: float | None = None
    norm_y: float | None = None


@dataclass
class ZernikeConfig(GeometryConfig):
    """Configuration for a zernike geometry."""

    surface_type: ClassVar[str] = "zernike"
    radius: float = be.inf
    conic: float = 0.0
    coefficients: list[float] = field(default_factory=list)
    tol: float = 1e-6
    max_iter: int = 100
    norm_radius: float | None = None
    zernike_type: ZernikeType = "fringe"


@dataclass
class BiconicConfig(GeometryConfig):
    """Configuration for a biconic geometry."""

    surface_type: ClassVar[str] = "biconic"
    radius_x: float = be.inf
    radius_y: float = be.inf
    conic_x: float = 0.0
    conic_y: float = 0.0
    tol: float = 1e-6
    max_iter: int = 100


@dataclass
class ToroidalConfig(GeometryConfig):
    """Configuration for a toroidal geometry."""

    surface_type: ClassVar[str] = "toroidal"
    radius_x: float = be.inf
    radius_y: float = be.inf
    conic: float = 0.0
    toroidal_coeffs_poly_y: list[float] = field(default_factory=list)
    tol: float = 1e-6
    max_iter: int = 100


@dataclass
class ForbesQbfsConfig(GeometryConfig):
    """Configuration for a forbes_qbfs geometry."""

    surface_type: ClassVar[str] = "forbes_qbfs"
    radius: float = be.inf
    conic: float = 0.0
    radial_terms: dict[int, float] = field(default_factory=dict)
    norm_radius: float | None = None
    tol: float = 1e-6
    max_iter: int = 100


@dataclass
class ForbesQ2dConfig(GeometryConfig):
    """Configuration for a forbes_q2d geometry."""

    surface_type: ClassVar[str] = "forbes_q2d"
    radius: float = be.inf
    conic: float = 0.0
    freeform_coeffs: dict[tuple[str, int, int], float] = field(default_factory=dict)
    norm_radius: float | None = None
    tol: float = 1e-6
    max_iter: int = 100


@dataclass
class NurbsConfig(GeometryConfig):
    """Configuration for a nurbs geometry."""

    surface_type: ClassVar[str] = "nurbs"
    radius: float = be.inf
    conic: float = 0.0
    control_points: list[list[list[float]]] | None = None
    weights: list[float] | None = None
    u_knots: list[float] | None = None
    v_knots: list[float] | None = None
    nurbs_norm_x: float | None = None
    nurbs_norm_y: float | None = None
    nurbs_x_center: float = 0.0
    nurbs_y_center: float = 0.0
    u_degree: int = 3
    v_degree: int = 3
    n_points_u: int = 5
    n_points_v: int = 5
    tol: float = 1e-6
    max_iter: int = 100


@dataclass
class GridSagConfig(GeometryConfig):
    """Configuration for a grid_sag geometry."""

    surface_type: ClassVar[str] = "grid_sag"
    x_coordinates: list[float] = field(default_factory=list)
    y_coordinates: list[float] = field(default_factory=list)
    sag_values: list[list[float]] = field(default_factory=list)
    tol: float = 1e-6
    max_iter: int = 100


@dataclass
class ParaxialConfig(GeometryConfig):
    """Configuration for a paraxial geometry."""

    surface_type: ClassVar[str] = "paraxial"
