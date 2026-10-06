"""Optional HCIPy interchange for sampled scalar fields.

The boundary is an explicit host copy, not a zero-copy GPU or autograd bridge.
HCIPy retains its Field/Grid representation; Optiland owns the conversion to
its plane and length conventions. Neither package is a required dependency
of the other's core numerical implementation.
"""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.physical_optics.field import ScalarField, _positive_float


def _host_array(data) -> np.ndarray:
    """Copy boundary metadata to NumPy, downloading CuPy explicitly if needed."""
    if hasattr(data, "get"):
        data = data.get()
    return np.array(data, copy=True)


def from_hcipy(
    wavefront,
    *,
    length_scale: float = 1000.0,
    refractive_index: float = 1.0,
) -> ScalarField:
    """Import an HCIPy scalar Wavefront on a regular Cartesian plane.

    Args:
        wavefront: HCIPy Wavefront with a scalar electric field. Its grid
            weights must be the rectangular sample area. Vector fields and
            nonuniform grids are deliberately not approximated.
        length_scale: Multiply input coordinates and vacuum wavelength by
            this factor. The default converts meters to millimeters, as
            required by the Optiland prescription adapter. HCIPy does not
            enforce meters; choose this factor for the actual input units.
        refractive_index: Incident homogeneous-medium index. HCIPy Wavefront
            does not carry this property; supply it from the source context.

    Returns:
        ScalarField: Independent field on the active Optiland backend. On
        Torch, the configured device and floating-point precision are used.

    Notes:
        Complex samples are divided by ``length_scale`` so the area-integrated
        squared amplitude is preserved. This treats amplitude as a square root
        of power density, consistent with both packages' power integrals; it
        does not interpret a bare array as an SI electric field in V/m.
        Values and metadata are explicitly copied through host memory. No
        upstream gradient graph crosses this boundary. Backend-resident
        NewStyleField export requires HCIPy's working ``Field.to_dict``.
    """
    import hcipy as hp

    if not isinstance(wavefront, hp.Wavefront):
        raise TypeError("wavefront must be an HCIPy Wavefront.")
    if not wavefront.is_scalar:
        raise ValueError("only scalar HCIPy wavefronts are supported.")
    scale = _positive_float(length_scale, "length_scale")
    index = _positive_float(refractive_index, "refractive_index")
    wavelength = _positive_float(wavefront.wavelength, "wavefront wavelength")
    grid = wavefront.grid
    if not isinstance(grid, hp.CartesianGrid) or not grid.is_regular or grid.ndim != 2:
        raise ValueError("wavefront requires a regular two-dimensional Cartesian grid.")
    dims = _host_array(grid.dims)
    delta = _host_array(grid.delta)
    zero = _host_array(grid.zero)
    if np.any(dims < 2) or dims.shape != (2,):
        raise ValueError("each grid axis requires at least two samples.")
    if not np.all(np.isfinite(delta)) or np.any(delta <= 0):
        raise ValueError("grid pitches must be finite and positive.")
    if not np.all(np.isfinite(zero)):
        raise ValueError("grid coordinates must be finite.")
    weights = _host_array(grid.weights)
    if not np.allclose(weights, delta[0] * delta[1], rtol=1e-12, atol=0):
        raise ValueError("custom grid weights do not match rectangular sample areas.")

    # Reuse HCIPy's public field export rather than discarding phase or using
    # implicit __array__ conversion on its backend field wrapper.
    values = np.asarray(wavefront.electric_field.to_dict()["values"])
    nx, ny = (int(value) for value in dims)
    if values.shape != (nx * ny,):
        raise ValueError("electric field sample count does not match the scalar grid.")
    if not np.all(np.isfinite(values)):
        raise ValueError("electric field samples must be finite.")
    values = values.reshape(ny, nx).copy() / scale
    if be.get_backend() == "torch":
        # be.array uses the configured real dtype, so convert both components
        # separately: directly passing complex NumPy data would lose its phase.
        data = be.array(values.real) + 1j * be.array(values.imag)
    else:
        data = values
    center = zero + (dims - 1) * delta / 2
    return ScalarField(
        data,
        dx=float(delta[0]) * scale,
        dy=float(delta[1]) * scale,
        wavelength=wavelength * scale,
        refractive_index=index,
        center=(float(center[0]) * scale, float(center[1]) * scale),
    )


def to_hcipy(field: ScalarField, *, length_scale: float = 0.001):
    """Export a scalar field as an independent NumPy-backed HCIPy Wavefront.

    Args:
        field: Optiland scalar field on the active backend.
        length_scale: Multiply coordinates and vacuum wavelength by this
            factor. The default converts millimeters to meters. Amplitudes
            are divided by the factor to preserve integrated squared amplitude.

    Returns:
        Wavefront: HCIPy scalar wavefront with phase, axis order and grid
        center preserved. HCIPy does not store refractive index: retain
        ``field.refractive_index`` separately for propagation in a medium.

    Notes:
        Export downloads device data, detaches Torch gradients, and copies
        into host-owned arrays. No interpolation, transpose, phase conjugation,
        or FFT shift is performed.
    """
    import hcipy as hp

    if not isinstance(field, ScalarField):
        raise TypeError("field must be a ScalarField.")
    field._ensure_active_backend()
    scale = _positive_float(length_scale, "length_scale")
    values = np.array(be.to_numpy(field.data), copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError("field samples must be finite.")
    ny, nx = field.shape
    delta = np.array([field.dx, field.dy]) * scale
    center = np.array(field.center) * scale
    zero = center - (np.array([nx, ny]) - 1) * delta / 2
    grid = hp.CartesianGrid(hp.RegularCoords(delta, [nx, ny], zero))
    return hp.Wavefront(
        hp.Field((values / scale).ravel(), grid), wavelength=field.wavelength * scale
    )
