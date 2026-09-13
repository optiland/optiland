"""Two-dimensional scalar optical fields."""

from __future__ import annotations

import math
from numbers import Complex, Real
from typing import Generic, Literal

import optiland.backend as be
from optiland._types import BEArrayT, ScalarOrArrayT
from optiland.backend.utils import is_torch_tensor


def _positive_float(value: float, name: str) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return value


def _cast_real_like(values: BEArrayT, like: BEArrayT) -> BEArrayT:
    if is_torch_tensor(like):
        return values.to(device=like.device, dtype=like.real.dtype)
    return values.astype(like.real.dtype, copy=False)


def _centered_axis(size: int, spacing: float, like: BEArrayT | None = None):
    indices = be.arange_indices(size)
    indices = be.cast(indices) if like is None else _cast_real_like(indices, like)
    return (indices - (size - 1) / 2) * spacing


def _validate_amplitude(amplitude: complex | ScalarOrArrayT) -> None:
    if isinstance(amplitude, Complex):
        return
    if not isinstance(amplitude, be.ndarray):
        raise TypeError("amplitude must be a complex scalar or scalar backend array.")

    backend = be.get_backend()
    if (backend == "torch") != is_torch_tensor(amplitude):
        raise TypeError(f"amplitude must belong to the active {backend!r} backend.")
    if amplitude.ndim != 0:
        raise ValueError("amplitude must be scalar.")


class ScalarField(Generic[BEArrayT]):
    """A sampled two-dimensional complex scalar optical field.

    The last array dimension is the x-axis and the first is the y-axis. All
    spatial quantities, including ``dx``, ``dy``, and ``wavelength``, must use
    the same unit.

    Args:
        data: Two-dimensional NumPy array or PyTorch tensor containing the
            sampled complex amplitude. Real arrays are promoted to complex
            without changing their device or floating-point precision.
        dx: Sample spacing along x.
        wavelength: Vacuum wavelength in the same unit as ``dx``.
        dy: Sample spacing along y. Defaults to ``dx``.
        refractive_index: Homogeneous-medium refractive index. Defaults to 1.

    Raises:
        TypeError: If ``data`` does not belong to the active backend.
        ValueError: If the field or physical parameters are invalid.
    """

    def __init__(
        self,
        data: BEArrayT,
        dx: float,
        wavelength: float,
        dy: float | None = None,
        refractive_index: float = 1.0,
    ) -> None:
        if not isinstance(data, be.ndarray):
            raise TypeError("data must be a NumPy array or PyTorch tensor.")

        backend = be.get_backend()
        if (backend == "torch") != is_torch_tensor(data):
            raise TypeError(f"data must belong to the active {backend!r} backend.")
        if data.ndim != 2:
            raise ValueError("data must be a two-dimensional array.")
        if any(size < 2 for size in data.shape):
            raise ValueError("each field dimension must contain at least two samples.")

        self.data = data + 0j
        self.dx = _positive_float(dx, "dx")
        self.dy = self.dx if dy is None else _positive_float(dy, "dy")
        self.wavelength = _positive_float(wavelength, "wavelength")
        self.refractive_index = _positive_float(refractive_index, "refractive_index")
        self._backend = backend

    @property
    def shape(self) -> tuple[int, int]:
        """Return the field shape as ``(ny, nx)``."""
        return int(self.data.shape[0]), int(self.data.shape[1])

    @property
    def intensity(self) -> BEArrayT:
        """Return sampled intensity, ``abs(data) ** 2``."""
        self._ensure_active_backend()
        return be.real(self.data * be.conj(self.data))

    @property
    def power(self) -> ScalarOrArrayT:
        """Return the sampled intensity integral over the field plane."""
        return be.sum(self.intensity) * self.dx * self.dy

    def coordinates(self) -> tuple[BEArrayT, BEArrayT]:
        """Return centered one-dimensional x and y coordinate arrays."""
        self._ensure_active_backend()
        ny, nx = self.shape
        return (
            _centered_axis(nx, self.dx, like=self.data),
            _centered_axis(ny, self.dy, like=self.data),
        )

    def propagate(
        self,
        distance: float | ScalarOrArrayT,
        evanescent: Literal["discard", "decay"] = "discard",
    ) -> ScalarField[BEArrayT]:
        """Propagate the field through a homogeneous medium.

        Args:
            distance: Signed propagation distance in the field's spatial unit.
            evanescent: ``"discard"`` filters evanescent content even at zero
                distance. ``"decay"`` preserves the complete field at zero,
                up to FFT roundoff. See
                :func:`~optiland.physical_optics.propagation.angular_spectrum`
                for distance-gradient behavior.

        Returns:
            ScalarField: Propagated field on the same sampling grid.
        """
        from optiland.physical_optics.propagation import angular_spectrum

        return angular_spectrum(self, distance, evanescent=evanescent)

    def _ensure_active_backend(self) -> None:
        if be.get_backend() != self._backend:
            raise RuntimeError(
                "the active backend changed after this ScalarField was created"
            )

    def __repr__(self) -> str:
        """Return a concise field description."""
        return (
            f"ScalarField(shape={self.shape}, dx={self.dx}, dy={self.dy}, "
            f"wavelength={self.wavelength}, "
            f"refractive_index={self.refractive_index})"
        )


def gaussian_field(
    shape: tuple[int, int],
    dx: float,
    wavelength: float,
    waist_radius: float,
    dy: float | None = None,
    refractive_index: float = 1.0,
    amplitude: complex | ScalarOrArrayT = 1.0,
) -> ScalarField:
    """Create a fundamental Gaussian beam sampled at its waist.

    ``waist_radius`` is the conventional 1/e field-amplitude radius, or
    equivalently the 1/e-squared intensity radius.

    Args:
        shape: Number of samples as ``(ny, nx)``.
        dx: Sample spacing along x.
        wavelength: Vacuum wavelength.
        waist_radius: Gaussian beam waist radius.
        dy: Sample spacing along y. Defaults to ``dx``.
        refractive_index: Homogeneous-medium refractive index. Defaults to 1.
        amplitude: On-axis complex-field amplitude. Defaults to 1.

    Returns:
        ScalarField: Gaussian field at its waist plane.
    """
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(not isinstance(size, int) or size < 2 for size in shape)
    ):
        raise ValueError("shape must be a (ny, nx) tuple with dimensions >= 2.")

    dx = _positive_float(dx, "dx")
    dy = dx if dy is None else _positive_float(dy, "dy")
    waist_radius = _positive_float(waist_radius, "waist_radius")
    _validate_amplitude(amplitude)
    ny, nx = shape
    like = amplitude if isinstance(amplitude, be.ndarray) else None
    x = _centered_axis(nx, dx, like=like)
    y = _centered_axis(ny, dy, like=like)
    x_grid, y_grid = be.meshgrid(x, y)
    data = amplitude * be.exp(-(x_grid * x_grid + y_grid * y_grid) / waist_radius**2)
    return ScalarField(
        data=data,
        dx=dx,
        dy=dy,
        wavelength=wavelength,
        refractive_index=refractive_index,
    )
