"""Free-space scalar-field propagation algorithms."""

from __future__ import annotations

import math
from numbers import Complex, Real
from typing import TYPE_CHECKING, Literal

import optiland.backend as be
from optiland.backend.utils import is_torch_tensor
from optiland.physical_optics.field import ScalarField, _cast_real_like

if TYPE_CHECKING:
    from optiland._types import BEArrayT, ScalarOrArrayT

EvanescentPolicy = Literal["discard", "decay"]


def _frequency_axis(size: int, spacing: float, like: BEArrayT):
    indices = _cast_real_like(be.arange_indices(size), like)
    positive_limit = (size - 1) // 2
    ordered_indices = be.where(indices <= positive_limit, indices, indices - size)
    return ordered_indices / (size * spacing)


def _validate_distance(distance: float | ScalarOrArrayT) -> None:
    if isinstance(distance, Real):
        if not math.isfinite(float(distance)):
            raise ValueError("distance must be finite.")
        return
    if isinstance(distance, Complex):
        raise TypeError("distance must be real.")

    if not isinstance(distance, be.ndarray):
        raise TypeError("distance must be a real scalar or scalar backend array.")
    backend = be.get_backend()
    if (backend == "torch") != is_torch_tensor(distance):
        raise TypeError(f"distance must belong to the active {backend!r} backend.")
    if distance.ndim != 0:
        raise TypeError("distance must be a real scalar or scalar backend array.")
    is_complex = (
        distance.is_complex()
        if is_torch_tensor(distance)
        else distance.dtype.kind == "c"
    )
    if is_complex:
        raise TypeError("distance must be real.")
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
            PyTorch can differentiate with respect to distance, including at
            zero for propagating components.
        evanescent: Handling of spatial frequencies above the propagating
            cutoff. ``"discard"`` removes them at every distance, including
            zero, so zero-distance propagation is an identity only for fields
            without evanescent content. ``"decay"`` attenuates them
            exponentially with ``abs(distance)`` and preserves the complete
            field at zero, up to FFT roundoff. With evanescent content, this
            absolute-value decay has no two-sided distance derivative at zero;
            PyTorch uses a zero subgradient for the absolute-value factor there.

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
    if not isinstance(distance, Real):
        distance = _cast_real_like(distance, field.data)

    ny, nx = field.shape
    fx = _frequency_axis(nx, field.dx, field.data)
    fy = _frequency_axis(ny, field.dy, field.data)
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
