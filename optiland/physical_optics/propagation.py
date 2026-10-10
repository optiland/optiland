"""Free-space scalar-field propagation algorithms."""

from __future__ import annotations

import cmath
import math
from numbers import Complex, Real
from typing import TYPE_CHECKING, Literal

import optiland.backend as be
from optiland.backend.utils import is_torch_tensor
from optiland.physical_optics.field import ScalarField

if TYPE_CHECKING:
    from optiland._types import BEArrayT, ScalarOrArrayT

EvanescentPolicy = Literal["discard", "decay"]


def _phase_precision(values: BEArrayT, like: BEArrayT) -> BEArrayT:
    """Evaluate transfer phases precisely without moving tensors to the host."""
    if is_torch_tensor(like):
        if like.device.type == "mps":
            # MPS has no float64 support. The rationalized phase below still
            # avoids cancellation and a separate large carrier at every pixel.
            return values.to(device=like.device, dtype=like.real.dtype)
        return values.to(device=like.device).double()
    return values.astype("float64", copy=False)


def _frequency_axis(size: int, spacing: float, like: BEArrayT):
    indices = _phase_precision(be.arange_indices(size), like)
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

    The longitudinal phase is evaluated as a uniform carrier plus the stable
    difference ``kz - k = -kt**2 / (kz + k)``. Transfer calculations use float64
    on NumPy and non-MPS Torch devices, then return to the field's dtype. This
    avoids spurious float32 diffraction halos without widening the field or its
    FFT. MPS uses the same formula in native precision; large relative phases
    and tensor-distance carrier phases remain limited by that precision. Tensor
    distances stay on the field's device and retain their autograd graph. A
    float32 distance's already-rounded physical value cannot be recovered by
    phase evaluation.

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
        distance = _phase_precision(distance, field.data)

    ny, nx = field.shape
    fx = _frequency_axis(nx, field.dx, field.data)
    fy = _frequency_axis(ny, field.dy, field.data)
    wavenumber = 2 * be.pi * field.refractive_index / field.wavelength
    medium_wavelength = field.wavelength / field.refractive_index
    ux, uy = be.meshgrid(fx * medium_wavelength, fy * medium_wavelength)
    # Dimensionless squared wavevectors also avoid overflow from squaring a
    # dimensional optical wavenumber in very small spatial units.
    transverse_squared = ux * ux + uy * uy
    longitudinal_squared = 1.0 - transverse_squared
    propagating = longitudinal_squared >= 0
    longitudinal = be.sqrt(be.clip(longitudinal_squared, 0.0, be.inf))
    # Only propagating components need an oscillatory correction. Mask before
    # arithmetic so discarded/decaying components cannot produce huge phases.
    relative_kz = -be.where(propagating, transverse_squared, 0.0) / (longitudinal + 1.0)
    carrier_phase = (
        float(distance) * wavenumber
        if isinstance(distance, Real)
        else distance * wavenumber
    )
    carrier = (
        cmath.exp(1j * carrier_phase)
        if isinstance(distance, Real)
        else be.exp(1j * carrier_phase)
    )
    transfer = carrier * be.exp(1j * carrier_phase * relative_kz)
    # At the exact propagating cutoff kz=0, the transfer and its distance
    # derivative are exactly 1 and 0; do not cancel two large rounded phases.
    transfer = be.where(longitudinal == 0, 1.0, transfer)
    if evanescent == "discard":
        transfer = be.where(propagating, transfer, 0.0)
    else:
        decay_rate = be.sqrt(be.clip(-longitudinal_squared, 0.0, be.inf))
        # Evanescent components have no carrier phase in the existing ASM decay
        # policy: their transfer is real exp(-abs(z) * sqrt(kt**2 - k**2)).
        transfer = be.where(
            propagating, transfer, be.exp(-abs(carrier_phase) * decay_rate) + 0j
        )

    if is_torch_tensor(field.data):
        transfer = transfer.to(dtype=field.data.dtype)
    else:
        transfer = transfer.astype(field.data.dtype, copy=False)

    spectrum = be.fft.fft2(field.data)
    propagated_data = be.fft.ifft2(spectrum * transfer)
    return ScalarField(
        data=propagated_data,
        dx=field.dx,
        dy=field.dy,
        wavelength=field.wavelength,
        refractive_index=field.refractive_index,
    )
