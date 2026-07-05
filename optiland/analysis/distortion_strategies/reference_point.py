"""Per-field image reference point strategies.

The image reference point is the single image-plane coordinate associated with a
field direction. Classically this is the chief-ray intercept, but for off-axis,
freeform, or obscured systems no chief ray exists. This module abstracts the
reference point behind :class:`ReferencePointStrategy` so that the chief ray can
be transparently replaced by the transmitted-energy centroid of a traced pupil
bundle.

All strategies expose the same :meth:`ReferencePointStrategy.locate` interface and
are therefore interchangeable wherever a per-field landing point is required
(standard distortion, grid distortion, and image-simulation warping).

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.distribution import create_distribution

if TYPE_CHECKING:
    from optiland._types import BEArray, DistributionType
    from optiland.distribution import BaseDistribution
    from optiland.optic import Optic


class ReferencePointStrategy(ABC):
    """Abstract strategy for locating per-field image reference points.

    Implementations map one or more normalized field coordinates to the
    corresponding image-plane ``(x, y)`` landing points for a given wavelength.
    """

    @abstractmethod
    def locate(
        self,
        optic: Optic,
        Hx: float | BEArray,
        Hy: float | BEArray,
        wavelength: float,
    ) -> tuple[BEArray, BEArray]:
        """Locate the image reference point(s) for the given field(s).

        Args:
            optic: The optical system to trace through.
            Hx: Normalized x field coordinate(s). Scalar or 1D array.
            Hy: Normalized y field coordinate(s). Scalar or 1D array.
            wavelength: Wavelength of the trace, in microns.

        Returns:
            A tuple ``(x, y)`` of 1D arrays giving the image-plane coordinates
            of the reference point for each input field. Fields for which no
            reference point can be determined are returned as ``NaN``.
        """


class ChiefRayReferencePoint(ReferencePointStrategy):
    """Reference point given by the chief ray (pupil center) intercept.

    This reproduces the classical behavior and is valid for rotationally
    symmetric, well-behaved systems where a chief ray can be traced.
    """

    def locate(
        self,
        optic: Optic,
        Hx: float | BEArray,
        Hy: float | BEArray,
        wavelength: float,
    ) -> tuple[BEArray, BEArray]:
        """Trace the chief ray (``Px = Py = 0``) for each field point."""
        optic.trace_generic(Hx=Hx, Hy=Hy, Px=0.0, Py=0.0, wavelength=wavelength)
        x = optic.surfaces.x[-1, :]
        y = optic.surfaces.y[-1, :]
        return x, y


class CentroidReferencePoint(ReferencePointStrategy):
    """Reference point given by the transmitted-energy centroid of a ray bundle.

    For each field, a dense, approximately equal-area pupil distribution is
    traced. Rays flagged as blocked/vignetted (zero intensity) or that fail to
    trace (non-finite coordinates) are discarded, and the (optionally
    flux-weighted) centroid of the survivors is returned. This is obscuration-
    proof and remains well-defined when no chief ray exists.

    Args:
        num_rays: Sampling density of the pupil distribution. For the default
            ``"uniform"`` distribution this is the number of points per axis of
            the underlying square grid (clipped to the unit disk).
        distribution: Name of the pupil distribution to sample. An
            (approximately) equal-area distribution such as ``"uniform"`` or
            ``"hexapolar"`` is recommended so that an unweighted centroid equals
            the energy centroid under uniform illumination.
        flux_weighted: If True (default), weight each surviving ray by its
            transmitted intensity (handling apodization and partial vignetting).
            If False, all surviving rays are weighted equally.
    """

    def __init__(
        self,
        num_rays: int = 20,
        distribution: DistributionType = "uniform",
        *,
        flux_weighted: bool = True,
    ):
        self.num_rays = num_rays
        self.distribution = distribution
        self.flux_weighted = flux_weighted

    def _sample_pupil(self) -> BaseDistribution:
        """Create and populate the configured pupil distribution."""
        dist = create_distribution(self.distribution)
        dist.generate_points(self.num_rays)
        return dist

    def locate(
        self,
        optic: Optic,
        Hx: float | BEArray,
        Hy: float | BEArray,
        wavelength: float,
    ) -> tuple[BEArray, BEArray]:
        """Trace a pupil bundle per field and return the energy centroid."""
        Hx = be.atleast_1d(be.array(Hx))
        Hy = be.atleast_1d(be.array(Hy))
        num_fields = Hx.shape[0]

        pupil = self._sample_pupil()
        Px = pupil.x
        Py = pupil.y
        num_pupil = Px.shape[0]

        # One ray per (field, pupil-point); fields vary slowest.
        Hx_full = be.repeat(Hx, num_pupil)
        Hy_full = be.repeat(Hy, num_pupil)
        Px_full = be.tile(Px, num_fields)
        Py_full = be.tile(Py, num_fields)

        optic.trace_generic(
            Hx=Hx_full, Hy=Hy_full, Px=Px_full, Py=Py_full, wavelength=wavelength
        )

        shape = (num_fields, num_pupil)
        x = be.reshape(optic.surfaces.x[-1, :], shape)
        y = be.reshape(optic.surfaces.y[-1, :], shape)
        intensity = be.reshape(optic.surfaces.intensity[-1, :], shape)

        # Survivors: transmitted (positive intensity) and successfully traced.
        valid = be.isfinite(x) & be.isfinite(y) & (intensity > 0)

        if self.flux_weighted:
            weights = be.where(valid, intensity, be.zeros_like(intensity))
        else:
            weights = be.where(valid, be.ones_like(intensity), be.zeros_like(intensity))

        x_clean = be.where(valid, x, be.zeros_like(x))
        y_clean = be.where(valid, y, be.zeros_like(y))

        weight_sum = be.sum(weights, axis=1)
        # Fully obscured fields (weight_sum == 0) yield NaN, flagging them as
        # undefined for downstream masking.
        x_bar = be.sum(weights * x_clean, axis=1) / weight_sum
        y_bar = be.sum(weights * y_clean, axis=1) / weight_sum
        return x_bar, y_bar
