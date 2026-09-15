"""Shared validation and interpolation of spectral material data."""

from __future__ import annotations

import math
from typing import Any, Literal

import optiland.backend as be


def finite_values(values: Any, label: str) -> tuple[float, ...]:
    """Copy a finite numerical sequence to owned, backend-independent values."""
    try:
        source = tuple(values)
        if any(getattr(value, "requires_grad", False) for value in source):
            raise ValueError("Owned optical data cannot contain trainable parameters")
        result = tuple(float(value) for value in source)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite numerical sequence") from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must be finite")
    return result


def wavelength_limits(values: Any) -> tuple[float, float] | None:
    """Validate optional, positive and increasing wavelength limits."""
    if values is None:
        return None
    limits = finite_values(values, "Wavelength limits")
    if len(limits) != 2 or not 0 < limits[0] < limits[1]:
        raise ValueError("Wavelength limits must be positive and increasing")
    return limits[0], limits[1]


def paired_samples(
    wavelengths: Any, values: Any, *, nonnegative: bool = False
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate and sort at least two paired samples without changing their values."""
    waves = finite_values(wavelengths, "Sample wavelengths")
    samples = finite_values(values, "Sample values")
    if len(waves) < 2 or len(waves) != len(samples):
        raise ValueError("Spectral data requires at least two paired samples")
    if any(wave <= 0 for wave in waves) or len(set(waves)) != len(waves):
        raise ValueError("Sample wavelengths must be positive and distinct")
    if nonnegative and any(value < 0 for value in samples):
        raise ValueError("Extinction samples must be nonnegative")
    pairs = sorted(zip(waves, samples, strict=True))
    return tuple(w for w, _ in pairs), tuple(v for _, v in pairs)


def checked_wavelengths(wavelength: Any, limits: Any = None) -> Any:
    """Check finite positive wavelengths and preserve the query shape."""
    wave = be.atleast_1d(be.asarray(wavelength))
    if not be.all(be.isfinite(wave)) or be.any(wave <= 0):
        raise ValueError("Wavelengths must be finite and positive")
    if limits is not None and (be.any(wave < limits[0]) or be.any(wave > limits[1])):
        raise ValueError("Wavelength outside material range")
    return wave


def interpolate_linear(
    wavelength: Any,
    sample_wavelengths: Any,
    values: Any,
    *,
    bounds: Literal["raise", "clamp"] = "raise",
) -> Any:
    """Interpolate spectral samples with an explicit adapter-selected bounds policy.

    Owned data rejects extrapolation; existing file materials retain endpoint
    clamping. Sorting/physical validation belongs to the adapter. Distinct
    samples must remain ordered at the active numerical precision.
    """
    wave = be.asarray(wavelength)
    samples = be.asarray(sample_wavelengths)
    data = be.asarray(values)
    if bounds not in {"raise", "clamp"}:
        raise ValueError(f"Unknown spectral bounds policy: {bounds!r}")
    if len(samples) == 0 or len(samples) != len(data):
        raise ValueError("Spectral interpolation requires paired samples")
    if be.any(samples[1:] <= samples[:-1]):
        raise ValueError(
            "Sample wavelengths must remain distinct and increasing at the "
            "active backend precision"
        )
    if bounds == "raise" and (be.any(wave < samples[0]) or be.any(wave > samples[-1])):
        raise ValueError("Wavelength outside tabulated range")
    if len(samples) == 1:
        return data[0] + wave * 0
    return be.interp(wave, samples, data)
