from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar, Union, Unpack

from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor  # noqa: F401

    from optiland.coatings import BaseCoating
    from optiland.physical_apertures.base import BaseAperture

    BEArray = Tensor | NDArray
    ScalarOrArray = float | Tensor | NDArray


__all__ = [
    "BEArrayT",
    "BEArray",
    "DistributionType",
    "ApertureType",
    "Fields",
    "FieldType",
    "PlotProjection",
    "ReferenceRay",
    "WavelengthUnit",
    "Wavelengths",
    "ScalarOrArrayT",
    "SurfaceType",
    "SurfaceParameters",
    "Unpack",
    "ZernikeType",
]

BEArrayT = TypeVar("BEArrayT", NDArray, "Tensor", Union[NDArray, "Tensor"])
ScalarOrArrayT = TypeVar(
    "ScalarOrArrayT", float, NDArray, "Tensor", Union[NDArray, "Tensor"]
)

DistributionType = Literal[
    "line_x",
    "line_y",
    "positive_line_x",
    "positive_line_y",
    "random",
    "uniform",
    "hexapolar",
    "cross",
    "ring",
    "sobol",
]
ApertureType = Literal["EPD", "imageFNO", "objectNA", "float_by_stop_size"]
Fields = Literal["all"] | Sequence[tuple[float, float]]
FieldType = Literal["angle", "object_height"]
PlotProjection = Literal["2d", "3d"]
ReferenceRay = Literal["chief", "marginal"]
Wavelengths = Literal["all", "primary"] | Sequence[float]
WavelengthUnit = Literal["nm", "um", "mm", "cm", "m"]
ZernikeType = Literal["standard", "noll", "fringe"]

SurfaceType = Literal[
    "biconic",
    "chebyshev",
    "even_asphere",
    "forbes_q2d",
    "forbes_qbfs",
    "odd_asphere",
    "paraxial",
    "polynomial",
    "standard",
    "toroidal",
    "zernike",
    "grating",
    "nurbs",
]


class SurfaceParameters(TypedDict, total=False):
    # Geometry parameters
    radius: float
    conic: float
    coefficients: list[float]
    tol: float
    max_iter: int
    norm_x: float
    norm_y: float
    norm_radius: float
    radius_x: float
    radius_y: float
    conic_x: float
    conic_y: float
    toroidal_coeffs_poly_y: list[float]
    zernike_type: ZernikeType
    radial_terms: dict[int, float]
    freeform_coeffs: dict[tuple[str, int, int], float]
    grating_order: int
    grating_period: float
    groove_orientation_angle: float
    control_points: list[list[list[float]]]
    weights: list[float]
    u_knots: list[float]
    v_knots: list[float]
    nurbs_norm_x: float
    nurbs_norm_y: float
    nurbs_x_center: float
    nurbs_y_center: float
    u_degree: int
    v_degree: int
    n_points_u: int
    n_points_v: int

    # Coordinate system parameters
    thickness: float
    x: float
    y: float
    z: float
    dx: float
    dy: float
    rx: float
    ry: float
    rz: float

    # Coating parameters
    coating: str | BaseCoating

    # Paraxial parameters
    f: float  # focal length
    aperture: BaseAperture
