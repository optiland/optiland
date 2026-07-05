"""Distortion models: distortion-free reference mappings and residuals.

A :class:`DistortionModel` defines (1) the distortion-free reference image
mapping and (2) how the residual distortion of real landing points about that
reference is reported. Two models are provided:

* :class:`RotationalDistortionModel` — the classical rotationally symmetric
  ``f-tan`` / ``f-theta`` model. The plate scale is fixed by a single near-axis
  reference ray and distortion is the radial departure from it.

* :class:`AffineDistortionModel` — the axis-free, best-fit affine (plate-scale)
  reference. A full ``2x2`` linear
  map from field-angle tangents to image coordinates is fit by least squares
  over a field grid; distortion is the residual about that fit. This requires no
  symmetry, no scalar focal length, and no chief ray, so it is valid for
  off-axis, freeform, and obscured systems.

Both models consume a :class:`ReferencePointStrategy` to obtain real landing
points, keeping reference-point location (how a field maps to the image) cleanly
separated from the reference mapping (what counts as undistorted).

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import optiland.backend as be

from .reference_point import (
    CentroidReferencePoint,
    ChiefRayReferencePoint,
    ReferencePointStrategy,
)

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.optic import Optic


@dataclass
class DistortionResult:
    """Container for the outcome of a distortion evaluation.

    All coordinates are expressed relative to the reference origin (the image
    point of the field center / the affine offset ``b``), so an undistorted
    field center maps to ``(0, 0)``.

    Attributes:
        x_real: Real image x coordinates of the reference points.
        y_real: Real image y coordinates of the reference points.
        x_ideal: Ideal (distortion-free) image x coordinates.
        y_ideal: Ideal (distortion-free) image y coordinates.
        reference_radius: Half-field image radius used to normalize percent
            distortion. ``None`` when distortion is normalized per-point against
            the local ideal radius (rotational model).
    """

    x_real: BEArray
    y_real: BEArray
    x_ideal: BEArray
    y_ideal: BEArray
    reference_radius: float | None = None


class DistortionModel(ABC):
    """Abstract distortion model: reference mapping plus residual reporting.

    The lifecycle is :meth:`fit` (calibrate the reference mapping for a
    wavelength) followed by :meth:`evaluate` (compute real and ideal points at
    query fields). :meth:`compute` is a convenience that performs both.
    """

    @abstractmethod
    def fit(self, optic: Optic, wavelength: float) -> None:
        """Calibrate the distortion-free reference mapping for a wavelength."""

    @abstractmethod
    def evaluate(
        self, optic: Optic, Hx: BEArray, Hy: BEArray, wavelength: float
    ) -> DistortionResult:
        """Evaluate real and ideal image points at the query field points."""

    @abstractmethod
    def percent(self, result: DistortionResult, *, signed: bool = False) -> BEArray:
        """Return per-field distortion as a percentage.

        Args:
            result: A result previously produced by :meth:`evaluate`.
            signed: If True, request a signed scalar distortion where the model
                supports it (the rotationally symmetric radial convention).
                Models without a meaningful sign return the magnitude.
        """

    def compute(
        self, optic: Optic, Hx: BEArray, Hy: BEArray, wavelength: float
    ) -> DistortionResult:
        """Fit then evaluate in a single call."""
        self.fit(optic, wavelength)
        return self.evaluate(optic, Hx, Hy, wavelength)


class RotationalDistortionModel(DistortionModel):
    """Classical rotationally symmetric ``f-tan`` / ``f-theta`` distortion model.

    The plate scale is determined from a single near-axis reference ray, exactly
    reproducing Optiland's historical distortion computation.

    Args:
        reference_point: Strategy used to locate image reference points. Defaults
            to :class:`ChiefRayReferencePoint`.
        distortion_type: Either ``"f-tan"`` or ``"f-theta"``.

    Raises:
        ValueError: If ``distortion_type`` is not ``"f-tan"`` or ``"f-theta"``.
    """

    _NEAR_AXIS = 1e-10

    def __init__(
        self,
        reference_point: ReferencePointStrategy | None = None,
        distortion_type: str = "f-tan",
    ):
        if distortion_type not in ("f-tan", "f-theta"):
            raise ValueError('distortion_type must be "f-tan" or "f-theta"')
        self.reference_point = reference_point or ChiefRayReferencePoint()
        self.distortion_type = distortion_type
        self._origin: tuple[BEArray, BEArray] | None = None
        self._scale: BEArray | None = None
        self._max_field_rad: BEArray | None = None

    def _project(self, H: BEArray) -> BEArray:
        """Project a normalized field coordinate to its reference variable."""
        angle = H * self._max_field_rad
        if self.distortion_type == "f-tan":
            return be.tan(angle)
        return angle

    def fit(self, optic: Optic, wavelength: float) -> None:
        """Fix the origin (field-center intercept) and the scalar plate scale."""
        self._max_field_rad = be.radians(optic.fields.max_field)

        x_c, y_c = self.reference_point.locate(optic, 0.0, 0.0, wavelength)
        x_c, y_c = x_c[0], y_c[0]

        _, y_ref = self.reference_point.locate(optic, 0.0, self._NEAR_AXIS, wavelength)
        if self.distortion_type == "f-tan":
            denom = be.tan(self._NEAR_AXIS * self._max_field_rad)
        else:
            denom = self._NEAR_AXIS * self._max_field_rad

        self._scale = (y_ref[0] - y_c) / denom
        self._origin = (x_c, y_c)

    def evaluate(
        self, optic: Optic, Hx: BEArray, Hy: BEArray, wavelength: float
    ) -> DistortionResult:
        """Trace real points and compute the rotationally symmetric ideal grid."""
        if self._origin is None or self._scale is None:
            raise RuntimeError("fit() must be called before evaluate().")

        x_c, y_c = self._origin
        x_real, y_real = self.reference_point.locate(optic, Hx, Hy, wavelength)
        x_real = x_real - x_c
        y_real = y_real - y_c

        x_ideal = self._scale * self._project(Hx)
        y_ideal = self._scale * self._project(Hy)
        return DistortionResult(x_real, y_real, x_ideal, y_ideal, None)

    def percent(self, result: DistortionResult, *, signed: bool = False) -> BEArray:
        """Radial distortion as a percentage of the local ideal radius."""
        if signed:
            return 100 * (result.y_real - result.y_ideal) / result.y_ideal
        delta = be.sqrt(
            (result.x_real - result.x_ideal) ** 2
            + (result.y_real - result.y_ideal) ** 2
        )
        ideal_radius = be.sqrt(result.x_ideal**2 + result.y_ideal**2)
        return 100 * delta / ideal_radius


class AffineDistortionModel(DistortionModel):
    """Axis-free best-fit affine (plate-scale) distortion model.

    A full ``2x2`` linear map ``u = A @ p + b`` from field-angle tangents
    ``p = (tan(alpha_x), tan(alpha_y))`` to image coordinates is fit by least
    squares over a field grid. The affine map absorbs intended first-order
    behavior (plate scale, anamorphic magnification, image rotation, shear,
    origin offset); distortion is the residual about it. No chief ray or scalar
    focal length is required, so the model is valid for off-axis, freeform, and
    obscured systems.

    Args:
        reference_point: Strategy used to locate image reference points. Defaults
            to :class:`CentroidReferencePoint`, which is obscuration-proof.
        fit_grid_size: Number of field samples per axis used to fit the affine
            map. The grid spans the normalized field square ``[-1, 1]``.
        field_projection: ``"f-tan"`` (default, rectilinear reference using
            tangents) or ``"f-theta"`` (scanning reference using angles).

    Raises:
        ValueError: If ``field_projection`` is not ``"f-tan"`` or ``"f-theta"``.
    """

    def __init__(
        self,
        reference_point: ReferencePointStrategy | None = None,
        fit_grid_size: int = 11,
        field_projection: str = "f-tan",
    ):
        if field_projection not in ("f-tan", "f-theta"):
            raise ValueError('field_projection must be "f-tan" or "f-theta"')
        self.reference_point = reference_point or CentroidReferencePoint()
        self.fit_grid_size = fit_grid_size
        self.field_projection = field_projection
        self._A: BEArray | None = None
        self._b: tuple[BEArray, BEArray] | None = None
        self._reference_radius: BEArray | None = None
        self._max_field_rad: BEArray | None = None

    def _project(self, H: BEArray) -> BEArray:
        """Project a normalized field coordinate to its reference variable."""
        angle = H * self._max_field_rad
        if self.field_projection == "f-tan":
            return be.tan(angle)
        return angle

    def fit(self, optic: Optic, wavelength: float) -> None:
        """Fit the affine reference map by least squares over a field grid."""
        self._max_field_rad = be.radians(optic.fields.max_field)

        axis = be.linspace(-1.0, 1.0, self.fit_grid_size)
        Hx_grid, Hy_grid = be.meshgrid(axis, axis)
        Hx = Hx_grid.flatten()
        Hy = Hy_grid.flatten()

        x_bar, y_bar = self.reference_point.locate(optic, Hx, Hy, wavelength)
        px = self._project(Hx)
        py = self._project(Hy)

        # Exclude fully obscured / failed fields from the fit.
        valid = be.isfinite(x_bar) & be.isfinite(y_bar)
        px_v = px[valid]
        py_v = py[valid]

        design = be.stack([px_v, py_v, be.ones_like(px_v)], axis=1)
        coeff_x = be.lstsq(design, x_bar[valid])  # [A11, A12, b_x]
        coeff_y = be.lstsq(design, y_bar[valid])  # [A21, A22, b_y]

        self._A = be.stack(
            [
                be.stack([coeff_x[0], coeff_x[1]]),
                be.stack([coeff_y[0], coeff_y[1]]),
            ]
        )
        self._b = (coeff_x[2], coeff_y[2])

        # Half-field image radius R_max = max ||A p|| over the (valid) fit grid.
        x_ideal = coeff_x[0] * px_v + coeff_x[1] * py_v
        y_ideal = coeff_y[0] * px_v + coeff_y[1] * py_v
        self._reference_radius = be.max(be.sqrt(x_ideal**2 + y_ideal**2))

    def evaluate(
        self, optic: Optic, Hx: BEArray, Hy: BEArray, wavelength: float
    ) -> DistortionResult:
        """Trace real centroids and evaluate the affine ideal at query fields."""
        if self._A is None or self._b is None:
            raise RuntimeError("fit() must be called before evaluate().")

        x_bar, y_bar = self.reference_point.locate(optic, Hx, Hy, wavelength)
        px = self._project(Hx)
        py = self._project(Hy)

        b_x, b_y = self._b
        # Ideal points expressed relative to the origin b (A @ p).
        x_ideal = self._A[0, 0] * px + self._A[0, 1] * py
        y_ideal = self._A[1, 0] * px + self._A[1, 1] * py
        x_real = x_bar - b_x
        y_real = y_bar - b_y
        return DistortionResult(
            x_real, y_real, x_ideal, y_ideal, self._reference_radius
        )

    def percent(self, result: DistortionResult, *, signed: bool = False) -> BEArray:
        """Residual distortion as a percentage of the half-field image radius."""
        delta = be.sqrt(
            (result.x_real - result.x_ideal) ** 2
            + (result.y_real - result.y_ideal) ** 2
        )
        return 100 * delta / result.reference_radius


_PARAXIAL_ALIASES = frozenset(
    {"paraxial", "chief", "chief_ray", "chief-ray", "rotational"}
)
_NONPARAXIAL_ALIASES = frozenset({"nonparaxial", "non-paraxial", "centroid", "affine"})


def create_distortion_model(
    method: str | DistortionModel = "paraxial",
    *,
    distortion_type: str = "f-tan",
    **kwargs,
) -> DistortionModel:
    """Build a :class:`DistortionModel` from a method name.

    Args:
        method: Either an existing :class:`DistortionModel` instance (returned
            unchanged, enabling fully custom strategies), or one of the strings
            ``"paraxial"`` (chief ray + rotationally symmetric reference) or
            ``"nonparaxial"`` (energy centroid + best-fit affine reference).
            Common aliases are accepted.
        distortion_type: Reference model, ``"f-tan"`` or ``"f-theta"``. Used as
            the rotational model type for the paraxial method and as the field
            projection for the non-paraxial method.
        **kwargs: Forwarded to the selected model constructor (e.g.
            ``reference_point``, ``fit_grid_size``).

    Returns:
        A configured :class:`DistortionModel`.

    Raises:
        ValueError: If ``method`` is an unrecognized string.
    """
    if isinstance(method, DistortionModel):
        return method

    key = method.lower()
    if key in _PARAXIAL_ALIASES:
        return RotationalDistortionModel(distortion_type=distortion_type, **kwargs)
    if key in _NONPARAXIAL_ALIASES:
        return AffineDistortionModel(field_projection=distortion_type, **kwargs)
    raise ValueError(
        f"Unknown distortion method '{method}'. "
        "Expected 'paraxial' or 'nonparaxial' (or a DistortionModel instance)."
    )
