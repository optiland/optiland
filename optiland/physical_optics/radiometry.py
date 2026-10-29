"""Explicit radiometric calibration and detector-sampled photon rates.

These helpers do not infer SI electric-field units from scalar amplitudes.
They preserve the active backend, device, and differentiable array operations.
Photon rates can be saved with NumPy's existing ``save`` function after an
explicit host conversion; no detector package or custom file format is needed.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TYPE_CHECKING

from scipy.constants import c, h

import optiland.backend as be
from optiland.backend.utils import is_torch_tensor
from optiland.physical_optics.field import ScalarField, _positive_float

if TYPE_CHECKING:
    from optiland._types import BEArrayT


def _validate_image(image: BEArrayT, name: str) -> None:
    if not isinstance(image, be.ndarray):
        raise TypeError(f"{name} must be a NumPy array or PyTorch tensor.")
    backend = be.get_backend()
    if (backend == "torch") != is_torch_tensor(image):
        raise TypeError(f"{name} must belong to the active {backend!r} backend.")
    is_floating = (
        image.is_floating_point() if is_torch_tensor(image) else image.dtype.kind == "f"
    )
    if not is_floating:
        raise TypeError(f"{name} must contain real floating-point values.")
    if image.ndim != 2 or any(size == 0 for size in image.shape):
        raise ValueError(f"{name} must be a nonempty two-dimensional array.")
    if not bool(be.all(be.isfinite(image))):
        raise ValueError(f"{name} must contain only finite values.")
    if bool(be.any(image < 0)):
        raise ValueError(f"{name} must be nonnegative.")


def field_to_irradiance(
    field: ScalarField[BEArrayT],
    *,
    irradiance_scale_w_per_mm2: float,
) -> BEArrayT:
    """Calibrate squared scalar amplitude as irradiance in W/mm².

    Args:
        field: Scalar field on the active backend.
        irradiance_scale_w_per_mm2: Finite, nonnegative irradiance corresponding
            to one unit of ``field.intensity``. This calibration is required;
            a bare scalar amplitude is not assumed to be an SI electric field.

    Returns:
        Backend array of irradiance samples in W/mm².

    Raises:
        TypeError: If the field or calibration has an invalid type.
        ValueError: If the calibration or resulting irradiance is invalid.
        RuntimeError: If the active backend differs from the field's backend.

    Notes:
        This changes neither the field nor its spatial units. A power-normalized
        amplitude in sqrt(W/mm²) uses a scale of one. For an explicitly declared
        complex peak electric-field phasor in V/m, a normally propagating wave
        in a lossless nonmagnetic medium uses ``n * epsilon_0 * c / (2 * 1e6)``.
        That electromagnetic factor must not be applied to a power-normalized
        amplitude, and the peak convention must not be confused with RMS.
        Keep the calibration fixed to retain optical attenuation and crop loss;
        this helper never renormalizes total power.
    """
    if not isinstance(field, ScalarField):
        raise TypeError("field must be a ScalarField.")
    scale = irradiance_scale_w_per_mm2
    if not isinstance(scale, Real) or isinstance(scale, bool):
        raise TypeError("irradiance_scale_w_per_mm2 must be a real scalar.")
    scale = float(scale)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("irradiance_scale_w_per_mm2 must be finite and nonnegative.")
    intensity = field.intensity
    _validate_image(intensity, "field intensity")
    irradiance = intensity * scale
    _validate_image(irradiance, "irradiance")
    return irradiance


def irradiance_to_photon_rate(
    irradiance: BEArrayT,
    *,
    dx_mm: float,
    dy_mm: float,
    wavelength_nm: float,
    binning: tuple[int, int] = (1, 1),
) -> BEArrayT:
    """Integrate aligned rectangular cells into photon rates per detector pixel.

    Args:
        irradiance: Nonnegative, finite, real floating-point array in W/mm²,
            ordered as (y, x). Values are cell-mean irradiances, or a rectangle
            quadrature approximation to them when using point samples.
        dx_mm: Width of each input sampling cell in mm, not the binned pitch.
        dy_mm: Height of each input sampling cell in mm.
        wavelength_nm: Positive vacuum wavelength in nm. It is supplied
            independently of any ScalarField spatial-unit convention.
        binning: Positive integer factors (by, bx). Each output pixel collects
            a block of by rows and bx columns. Both dimensions must divide
            exactly; there is no interpolation, padding, or automatic crop.

    Returns:
        Backend array of expected photons per detector pixel per second. Its
        shape is (ny // by, nx // bx), and its physical pitch is
        (by * dy_mm, bx * dx_mm) in (y, x) order.

    Raises:
        TypeError: If array, scalar, or binning types are invalid.
        ValueError: If values, shape, or binning factors are invalid.

    Notes:
        Uses the exact SI Planck constant and vacuum speed of light from SciPy:
        ``cell_rate = irradiance * dx_mm * dy_mm * wavelength_nm * 1e-9 / (h*c)``.
        Exposure duration and quantum efficiency are deliberately not applied.
        For Pyxel's monochromatic ``load_image`` model, save this rate map and
        use ``time_scale=1.0`` and ``convert_to_photons=False``. Pyxel multiplies
        by its current readout interval. Binning sums intensities, not coherent
        amplitudes, and does not restore power lost outside the sampled plane.
    """
    _validate_image(irradiance, "irradiance")
    for value, name in (
        (dx_mm, "dx_mm"),
        (dy_mm, "dy_mm"),
        (wavelength_nm, "wavelength_nm"),
    ):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a real scalar.")
    dx_mm = _positive_float(dx_mm, "dx_mm")
    dy_mm = _positive_float(dy_mm, "dy_mm")
    wavelength_nm = _positive_float(wavelength_nm, "wavelength_nm")
    if not isinstance(binning, tuple) or len(binning) != 2:
        raise TypeError("binning must be a (by, bx) tuple of positive integers.")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) for value in binning
    ):
        raise TypeError("binning must contain positive integers.")
    by, bx = (int(value) for value in binning)
    if by <= 0 or bx <= 0:
        raise ValueError("binning factors must be positive.")
    ny, nx = irradiance.shape
    if ny % by or nx % bx:
        raise ValueError("irradiance dimensions must be divisible by binning factors.")
    conversion = dx_mm * dy_mm * (wavelength_nm * 1e-9 / (h * c))
    rate = irradiance * conversion
    if binning != (1, 1):
        blocks = be.reshape(rate, (ny // by, by, nx // bx, bx))
        rate = be.sum(be.sum(blocks, axis=3), axis=1)
    _validate_image(rate, "photon rate")
    return rate


def normalized_psf(detector_sampled_intensity: BEArrayT) -> BEArrayT:
    """Return a unit-sum PSF kernel on an already sampled detector grid.

    Args:
        detector_sampled_intensity: Nonnegative, finite, real floating-point
            intensity array in (y, x) order, with a positive finite sum.

    Returns:
        Backend array with the same shape and a sum of one.

    Raises:
        TypeError: If the input is not a real floating-point backend array.
        ValueError: If the input or its sum is invalid.

    Notes:
        This explicitly normalizes only the captured PSF shape. Absolute power,
        optical throughput, and uncaptured tails cannot be recovered from it.
        No resampling or recentering occurs. Match sampling to detector pitch
        before export to Pyxel's ``load_psf`` model or another convolution tool.
    """
    _validate_image(detector_sampled_intensity, "detector_sampled_intensity")
    total = be.sum(detector_sampled_intensity)
    if not bool(be.isfinite(total)) or not bool(total > 0):
        raise ValueError("detector_sampled_intensity must have a positive finite sum.")
    return detector_sampled_intensity / total
