"""Free-space scalar-field propagation algorithms."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Literal

import optiland.backend as be
from optiland.physical_optics.field import ScalarField

if TYPE_CHECKING:
    from optiland._types import BEArrayT, ScalarOrArrayT

EvanescentPolicy = Literal["discard", "decay"]


def _frequency_axis(size: int, spacing: float):
    indices = be.cast(be.arange_indices(size))
    positive_limit = (size - 1) // 2
    ordered_indices = be.where(indices <= positive_limit, indices, indices - size)
    return ordered_indices / (size * spacing)


def _validate_distance(distance: float | ScalarOrArrayT) -> None:
    if isinstance(distance, Real):
        if not math.isfinite(float(distance)):
            raise ValueError("distance must be finite.")
        return
    if getattr(distance, "ndim", None) != 0:
        raise TypeError("distance must be a real scalar or scalar backend array.")
    if not bool(be.all(be.isfinite(distance))):
        raise ValueError("distance must be finite.")


def angular_spectrum(
    field: ScalarField[BEArrayT],
    distance: float | ScalarOrArrayT,
    evanescent: EvanescentPolicy = "discard",
) -> ScalarField[BEArrayT]:
    """Propagate a scalar field with the angular spectrum method.

    The input and output use the same rectangular sampling grid. Consequently,
    the usual discrete-Fourier periodic-boundary assumption applies; callers
    should provide enough zero padding to prevent wraparound for expanding
    fields.

    Args:
        field: Input scalar field.
        distance: Signed propagation distance. It must use the same unit as the
            field spacing and wavelength. A backend scalar is accepted so that
            PyTorch can differentiate with respect to distance.
        evanescent: Handling of spatial frequencies above the propagating
            cutoff. ``"discard"`` removes them. ``"decay"`` attenuates them
            exponentially away from the input plane.

    Returns:
        ScalarField: Propagated field on the original sampling grid.

    Raises:
        TypeError: If ``distance`` is not scalar.
        ValueError: If the distance or evanescent policy is invalid.
    """
    if not isinstance(field, ScalarField):
        raise TypeError("field must be a ScalarField.")
    field._ensure_active_backend()
    _validate_distance(distance)
    if evanescent not in ("discard", "decay"):
        raise ValueError("evanescent must be either 'discard' or 'decay'.")

    ny, nx = field.shape
    fx = _frequency_axis(nx, field.dx)
    fy = _frequency_axis(ny, field.dy)
    kx, ky = be.meshgrid(2 * be.pi * fx, 2 * be.pi * fy)

    wavenumber = 2 * be.pi * field.refractive_index / field.wavelength
    kz_squared = wavenumber**2 - kx * kx - ky * ky
    propagating = kz_squared >= 0
    kz = be.sqrt(be.clip(kz_squared, 0.0, be.inf))

    transfer = be.exp(1j * distance * kz)
    if evanescent == "discard":
        transfer = be.where(propagating, transfer, 0.0)
    else:
        decay_rate = be.sqrt(be.clip(-kz_squared, 0.0, be.inf))
        transfer = transfer * be.exp(-abs(distance) * decay_rate)

    spectrum = be.fft.fft2(field.data)
    propagated_data = be.fft.ifft2(spectrum * transfer)
    return ScalarField(
        data=propagated_data,
        dx=field.dx,
        dy=field.dy,
        wavelength=field.wavelength,
        refractive_index=field.refractive_index,
    )
