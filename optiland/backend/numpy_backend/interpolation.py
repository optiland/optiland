"""
NumPy backend -- interpolation, polynomial, and signal-processing operations.

Provides InterpolationMixin, one of the mixins composed into
NumpyBackend (see optiland/backend/numpy_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve as _fftconvolve

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class InterpolationMixin:
    """Interpolation, polynomial, and signal-processing operations."""

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def nearest_nd_interpolator(
        self,
        points: NDArray,
        values: NDArray,
        x: Any,
        y: Any,
    ) -> NDArray:
        """Nearest-neighbour interpolation on an N-D dataset.

        Args:
            points: Known sample points.
            values: Values at the sample points.
            x: Query x coordinates.
            y: Query y coordinates.

        Returns:
            NDArray: Interpolated values.
        """
        interpolator = NearestNDInterpolator(points, values)
        return interpolator(x, y)

    def interp(self, x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> NDArray:
        """1-D linear interpolation.

        Args:
            x: x-coordinates of the interpolated values.
            xp: x-coordinates of the data points.
            fp: y-coordinates of the data points.

        Returns:
            NDArray: Interpolated values.
        """
        return np.interp(x, xp, fp)

    def grid_sample(
        self,
        input: NDArray,
        grid: NDArray,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> NDArray:
        """Sample input using bilinear/nearest interpolation on a grid.

        NumPy/SciPy implementation of ``torch.nn.functional.grid_sample``.

        Args:
            input: Input array of shape (N, C, H_in, W_in).
            grid: Grid of shape (N, H_out, W_out, 2). Coordinates in [-1, 1].
            mode: Interpolation mode (``'bilinear'`` or ``'nearest'``).
            padding_mode: Padding mode (``'zeros'``, ``'border'``,
                ``'reflection'``).
            align_corners: Whether to align corners.

        Returns:
            NDArray: Output array of shape (N, C, H_out, W_out).
        """
        N, C, H_in, W_in = input.shape
        _N, H_out, W_out, _ = grid.shape
        if N != _N:
            raise ValueError("Input and grid must have same batch size")

        x = grid[..., 0]
        y = grid[..., 1]

        if align_corners:
            x_pix = ((x + 1) / 2) * (W_in - 1)
            y_pix = ((y + 1) / 2) * (H_in - 1)
        else:
            x_pix = ((x + 1) * W_in / 2) - 0.5
            y_pix = ((y + 1) * H_in / 2) - 0.5

        output = np.zeros((N, C, H_out, W_out), dtype=input.dtype)
        order = 0 if mode == "nearest" else 1
        scipy_mode = "constant"
        cval = 0.0
        if padding_mode == "border":
            scipy_mode = "nearest"
        elif padding_mode == "reflection":
            scipy_mode = "reflect"

        for n in range(N):
            for c in range(C):
                coords = np.stack((y_pix[n], x_pix[n]))
                output[n, c] = map_coordinates(
                    input[n, c], coords, order=order, mode=scipy_mode, cval=cval
                )

        return output

    # ------------------------------------------------------------------
    # Polynomial
    # ------------------------------------------------------------------

    def polyfit(self, x: ArrayLike, y: ArrayLike, degree: int) -> NDArray:
        """Least-squares polynomial fit.

        Args:
            x: x-coordinates of the sample points.
            y: y-coordinates of the sample points.
            degree: Degree of the polynomial.

        Returns:
            NDArray: Polynomial coefficients, highest power first.
        """
        return np.polyfit(x, y, degree)

    def polyval(self, coeffs: ArrayLike, x: ArrayLike) -> NDArray:
        """Evaluate a polynomial at specific values.

        Args:
            coeffs: Polynomial coefficients, highest power first.
            x: Values at which to evaluate the polynomial.

        Returns:
            NDArray: Evaluated polynomial.
        """
        return np.polyval(coeffs, x)

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def fftconvolve(
        self,
        in1: ArrayLike,
        in2: ArrayLike,
        mode: Literal["full", "valid", "same"] = "full",
    ) -> NDArray:
        """FFT-based convolution using SciPy.

        Args:
            in1: First input array.
            in2: Second input array.
            mode: Convolution mode (``'full'``, ``'valid'``, ``'same'``).

        Returns:
            NDArray: Convolved array.
        """
        a = self.array(in1)
        b = self.array(in2)

        if a.ndim >= 2 and b.ndim >= 2:
            return _fftconvolve(a, b, mode=mode, axes=(-2, -1))

        return _fftconvolve(a, b, mode=mode)
