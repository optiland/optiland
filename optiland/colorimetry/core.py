"""
Colorimetry calculation engine.
Handles spectral interpolation, XYZ calculation, and sRGB conversions.
Uses the optiland backend for mathematical operations.
"""

from __future__ import annotations

from typing import Any

from scipy.interpolate import interp1d

import optiland.backend as be

from .constants import CIE_1931_2DEG, CIE_1964_10DEG, ILLUMINANT_D65, WAVELENGTHS_STD


def _interpolate_spectrum(
    x_vals: list[float], y_vals: list[float], x_target: list[float], kind: str = "cubic"
) -> list[float]:
    """
    Interpolate spectral data using scipy.

    Args:
        x_vals: Source wavelengths.
        y_vals: Source values.
        x_target: Target wavelengths.
        kind: Type of interpolation ('linear', 'cubic', etc.). Default is 'cubic'.

    Returns:
        Interpolated values at x_target.
    """
    # Create interpolation function
    # bounds_error=False and fill_value parameters handle extrapolation safely
    # by clamping to the nearest value (or 0 if preferred, but clamping is safer
    # for spectra)
    f = interp1d(
        x_vals,
        y_vals,
        kind=kind,
        bounds_error=False,
        fill_value=(y_vals[0], y_vals[-1]),
    )
    return f(x_target)


def _extract_xyz(X: Any, Y: Any = None, Z: Any = None) -> tuple[Any, Any, Any]:
    """
    Helper to extract X, Y, Z components from inputs.
    Supports separate arguments or a single packed array.
    Detects the dimension with size 3 automatically.
    """
    if Y is None and Z is None:
        # Packed input
        arr = be.asarray(X)
        shape = arr.shape

        if len(shape) == 0:
            raise ValueError("Scalar input requires X, Y, Z arguments")

        # Heuristic: check last dim (e.g. image HxWx3) then first dim (3xN)
        if shape[-1] == 3:
            return arr[..., 0], arr[..., 1], arr[..., 2]
        elif shape[0] == 3:
            return arr[0], arr[1], arr[2]

        # Search for any dimension with size 3
        found_axis = None
        for i, dim in enumerate(shape):
            if dim == 3:
                found_axis = i
                break

        if found_axis is not None:
            # Generic slicing
            indexer = [slice(None)] * len(shape)

            indexer[found_axis] = 0
            X_out = arr[tuple(indexer)]

            indexer[found_axis] = 1
            Y_out = arr[tuple(indexer)]

            indexer[found_axis] = 2
            Z_out = arr[tuple(indexer)]
            return X_out, Y_out, Z_out

        raise ValueError(
            "Input array must have a dimension of size 3 representing X, Y, Z"
        )
    else:
        return be.asarray(X), be.asarray(Y), be.asarray(Z)


def spectrum_to_xyz(
    wavelengths: list[float],
    values: list[float],
    illuminant: list[float] | None = None,
    observer: str = "2deg",
) -> tuple[float, float, float]:
    """
    Converts a spectrum (R or T) to CIE XYZ coordinates.

    References:
        - CIE 15:2004 Colorimetry, 3rd Edition.
        - ISO 11664-3:2019 (CIE S 014-3/E:2011) Colorimetry — Part 3: CIE
          Tristimulus Values.

    Args:
        wavelengths: List of wavelengths of the measured spectrum (in nm).
        values: Spectral values (0.0 to 1.0).
        illuminant: Illuminant spectrum aligned with WAVELENGTHS_STD. Default: D65.
        observer: '2deg' (CIE 1931) or '10deg' (CIE 1964).

    Returns:
        (X, Y, Z) normalized (Y=100 for a perfect white).
    """
    # Select constants
    std_wl = WAVELENGTHS_STD

    # Check spectral coverage
    # We ensure the input spectrum covers at least the standard range (380-780nm)
    # to avoid inaccurate color calculations due to missing data.
    min_wl_in = min(wavelengths)
    max_wl_in = max(wavelengths)
    min_wl_std = std_wl[0]
    max_wl_std = std_wl[-1]

    if min_wl_in > min_wl_std or max_wl_in < max_wl_std:
        raise ValueError(
            f"Input spectrum range ({min_wl_in:.1f}-{max_wl_in:.1f} nm) does not cover "
            f"the required visible range ({min_wl_std}-{max_wl_std} nm)."
        )

    cmf = CIE_1964_10DEG if observer == "10deg" else CIE_1931_2DEG

    S = illuminant if illuminant is not None else ILLUMINANT_D65

    # Check illuminant size
    if len(S) != len(std_wl):
        raise ValueError(
            f"Illuminant must have the same size as WAVELENGTHS_STD ({len(std_wl)})"
        )

    # Interpolate user spectrum onto standard grid
    # Using cubic interpolation for better accuracy on spectral curves
    interp_values = _interpolate_spectrum(wavelengths, values, std_wl, kind="cubic")

    # Calculate integrals (Riemann Sum)
    X_sum = 0.0
    Y_sum = 0.0
    Z_sum = 0.0
    k_sum = 0.0  # Normalization factor

    d_lambda = float(WAVELENGTHS_STD[1] - WAVELENGTHS_STD[0])

    for i in range(len(std_wl)):
        r = interp_values[i]
        illuminant_val = S[i]
        x_bar, y_bar, z_bar = cmf[i]

        # Weighting
        X_sum += r * illuminant_val * x_bar * d_lambda
        Y_sum += r * illuminant_val * y_bar * d_lambda
        Z_sum += r * illuminant_val * z_bar * d_lambda

        # Normalization (perfect source)
        k_sum += illuminant_val * y_bar * d_lambda

    k = 100.0 / k_sum if k_sum != 0 else 0

    return (X_sum * k, Y_sum * k, Z_sum * k)


def xyz_to_xyY(
    X: float | Any,
    Y: float | Any | None = None,
    Z: float | Any | None = None,
) -> tuple[Any, Any, Any]:
    """
    Converts XYZ to xyY (chromaticity + luminance).
    Vectorized: supports scalars, separate arrays, or a single packed array with a
    dimension of size 3.
    """
    X, Y, Z = _extract_xyz(X, Y, Z)
    sum_xyz = X + Y + Z

    # Handle division by zero (vectorized)
    mask = be.abs(sum_xyz) == 0
    safe_sum = be.where(mask, 1.0, sum_xyz)

    x = X / safe_sum
    y = Y / safe_sum

    # Default white point (approx D65) if absolute black
    x = be.where(mask, 0.3127, x)
    y = be.where(mask, 0.3290, y)

    return x, y, Y


def xyz_to_srgb(
    X: float | Any,
    Y: float | Any | None = None,
    Z: float | Any | None = None,
) -> tuple[Any, Any, Any]:
    """
    Converts XYZ (Y=100 max) to sRGB (0-255).
    Vectorized: supports scalars, separate arrays, or a single packed array with a
    dimension of size 3.
    Uses the standard sRGB matrix (D65) and sRGB Gamma correction.

    References:
        - IEC 61966-2-1:1999 Multimedia systems and equipment - Colour
          measurement and management - Part 2-1: Colour management - Default
          RGB colour space - sRGB.
        - Anderson, M., Motta, R., Chandrasekar, S., & Stokes, M. (1996).
          Proposal for a Standard Default Color Space for the Internet—sRGB.
    """
    X, Y, Z = _extract_xyz(X, Y, Z)

    # 1. Normalize XYZ to 0-1
    X_n = X / 100.0
    Y_n = Y / 100.0
    Z_n = Z / 100.0

    # 2. Linear Transformation (sRGB D65 Matrix)
    # | R_l |   |  3.2404542 -1.5371385 -0.4985314 |   | X |
    # | G_l | = | -0.9692660  1.8760108  0.0415560 | * | Y |
    # | B_l |   |  0.0556434 -0.2040259  1.0572252 |   | Z |

    r_l = 3.2404542 * X_n - 1.5371385 * Y_n - 0.4985314 * Z_n
    g_l = -0.9692660 * X_n + 1.8760108 * Y_n + 0.0415560 * Z_n
    b_l = 0.0556434 * X_n - 0.2040259 * Y_n + 1.0572252 * Z_n

    # 3. Gamma Correction (sRGB Transfer Function) - Vectorized
    def gamma_correct(v):
        mask = v <= 0.0031308
        linear_part = 12.92 * v

        # Protection against negative numbers before pow
        v_safe = be.where(v < 0.0, 0.0, v)
        power_part = 1.055 * be.power(v_safe, 1.0 / 2.4) - 0.055

        return be.where(mask, linear_part, power_part)

    r = gamma_correct(r_l)
    g = gamma_correct(g_l)
    b = gamma_correct(b_l)

    # 4. Clipping [0, 1] and conversion 0-255
    def clip_and_scale(v):
        # Clip to 0-1
        v_clipped = be.where(v < 0.0, 0.0, be.where(v > 1.0, 1.0, v))
        scaled = v_clipped * 255
        # Try to cast to int if backend supports it (like numpy)
        try:
            return scaled.astype(int)
        except AttributeError:
            if hasattr(scaled, "to"):
                return scaled.to(dtype=be._lib.int64)
            return scaled

    return (clip_and_scale(r), clip_and_scale(g), clip_and_scale(b))
