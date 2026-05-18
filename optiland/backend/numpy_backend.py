"""
NumPy backend — implements AbstractBackend using NumPy and SciPy.

Kramer Harrison, 2024, 2025
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from matplotlib.path import Path
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve as _fftconvolve
from scipy.spatial.transform import Rotation as R
from scipy.special import gamma

from optiland.backend.base import AbstractBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from numpy.random import Generator as NpGenerator
    from numpy.typing import ArrayLike, NDArray


class NumpyBackend(AbstractBackend):
    """Backend implementation using NumPy and SciPy.

    Attributes:
        _lib: The NumPy module (used by passthrough methods).
        _precision: Current floating-point precision string.
    """

    _lib = np
    _precision: Literal["float32", "float64"] = "float64"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "numpy"

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------

    @property
    def _dtype(self) -> type:
        """Return the NumPy dtype for the current precision."""
        return np.float32 if self._precision == "float32" else np.float64

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        """Set the floating-point precision.

        Args:
            precision: Either ``'float32'`` or ``'float64'``.
        """
        if precision not in ("float32", "float64"):
            raise ValueError("Precision must be 'float32' or 'float64'.")
        self._precision = precision

    def get_precision(self) -> int:
        """Return the current precision as an integer (32 or 64)."""
        return 32 if self._precision == "float32" else 64

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    def array(self, x: ArrayLike) -> NDArray:
        """Create a NumPy array cast to the current precision.

        Args:
            x: Input data.

        Returns:
            NDArray: NumPy array with dtype matching current precision.
        """
        return np.array(x, dtype=self._dtype)

    def zeros(self, shape: Sequence[int], dtype: Any = None) -> NDArray:
        """Return a zero array of given shape with current precision dtype.

        Args:
            shape: Shape of the output array.
            dtype: Optional dtype override.

        Returns:
            NDArray: Zero array.
        """
        return np.zeros(shape, dtype=dtype if dtype is not None else self._dtype)

    def ones(self, shape: Sequence[int], dtype: Any = None) -> NDArray:
        """Return an array of ones with current precision dtype.

        Args:
            shape: Shape of the output array.
            dtype: Optional dtype override.

        Returns:
            NDArray: Ones array.
        """
        return np.ones(shape, dtype=dtype if dtype is not None else self._dtype)

    def full(self, shape: Sequence[int], fill_value: Any, dtype: Any = None) -> NDArray:
        """Return a constant-filled array with current precision dtype.

        Args:
            shape: Shape of the output array.
            fill_value: Fill value.
            dtype: Optional dtype override.

        Returns:
            NDArray: Filled array.
        """
        _dtype = dtype if dtype is not None else self._dtype
        return np.full(shape, fill_value, dtype=_dtype)

    def linspace(self, start: float, stop: float, num: int = 50) -> NDArray:
        """Return evenly spaced numbers over an interval.

        Args:
            start: Start of the interval.
            stop: End of the interval.
            num: Number of samples.

        Returns:
            NDArray: Evenly spaced samples.
        """
        return np.linspace(start, stop, num, dtype=self._dtype)

    def arange(self, *args: Any, **kwargs: Any) -> NDArray:
        """Return evenly spaced values within a given interval.

        Args:
            *args: start, stop, step (same as np.arange).
            **kwargs: Additional keyword arguments passed to np.arange.

        Returns:
            NDArray: Array of evenly spaced values.
        """
        return np.arange(*args, **kwargs)

    def zeros_like(self, x: ArrayLike) -> NDArray:
        """Return a zero array with the same shape as x.

        Args:
            x: Reference array.

        Returns:
            NDArray: Zero array.
        """
        return np.zeros_like(x, dtype=self._dtype)

    def ones_like(self, x: ArrayLike) -> NDArray:
        """Return an array of ones with the same shape as x.

        Args:
            x: Reference array.

        Returns:
            NDArray: Ones array.
        """
        return np.ones_like(x, dtype=self._dtype)

    def full_like(self, x: ArrayLike, fill_value: Any) -> NDArray:
        """Return a full array with the same shape as x.

        Args:
            x: Reference array.
            fill_value: Fill value.

        Returns:
            NDArray: Filled array.
        """
        return np.full_like(x, fill_value, dtype=self._dtype)

    def empty(self, shape: Sequence[int]) -> NDArray:
        """Return an uninitialized array of the given shape.

        Args:
            shape: Shape of the output array.

        Returns:
            NDArray: Uninitialized array.
        """
        return np.empty(shape, dtype=self._dtype)

    def empty_like(self, x: ArrayLike) -> NDArray:
        """Return an uninitialized array with the same shape as x.

        Args:
            x: Reference array.

        Returns:
            NDArray: Uninitialized array.
        """
        return np.empty_like(x, dtype=self._dtype)

    def eye(self, n: int) -> NDArray:
        """Return a 2D identity matrix.

        Args:
            n: Size of the identity matrix.

        Returns:
            NDArray: Identity matrix.
        """
        return np.eye(n, dtype=self._dtype)

    def asarray(self, x: ArrayLike, **kwargs: Any) -> NDArray:
        """Convert x to a NumPy array without copying if possible.

        Args:
            x: Input data.
            **kwargs: Keyword arguments forwarded to ``np.asarray``
                (e.g. ``dtype``).

        Returns:
            NDArray: NumPy array view (or copy if necessary).
        """
        dtype = kwargs.pop("dtype", self._dtype)
        return np.asarray(x, dtype=dtype, **kwargs)

    # ------------------------------------------------------------------
    # Array utilities
    # ------------------------------------------------------------------

    def cast(self, x: ArrayLike) -> NDArray:
        """Cast x to the current floating-point dtype.

        Args:
            x: Input data.

        Returns:
            NDArray: Array cast to current precision.
        """
        return np.array(x, dtype=self._dtype)

    def is_array_like(self, x: Any) -> bool:
        """Return True if x is a list, tuple, or ndarray.

        Args:
            x: Object to check.

        Returns:
            bool: True if x is array-like.
        """
        return isinstance(x, np.ndarray | list | tuple)

    def arange_indices(self, start: Any, stop: Any = None, step: int = 1) -> NDArray:
        """Create an integer array of indices.

        Args:
            start: Start index (or stop if stop is None).
            stop: Stop index.
            step: Step size.

        Returns:
            NDArray: Integer index array.
        """
        return np.arange(start, stop, step, dtype=np.int64)

    def ravel(self, x: ArrayLike) -> NDArray:
        """Return a contiguous flattened array cast to float.

        Args:
            x: Input array.

        Returns:
            NDArray: 1-D float array.
        """
        return np.ravel(x).astype(float)

    # ------------------------------------------------------------------
    # Shape and indexing
    # ------------------------------------------------------------------

    def transpose(self, x: ArrayLike, axes: Sequence[int] | None = None) -> NDArray:
        """Permute the dimensions of x.

        Args:
            x: Input array.
            axes: Permutation of dimensions.

        Returns:
            NDArray: Transposed array.
        """
        return np.transpose(x, axes)

    def reshape(self, x: ArrayLike, shape: Sequence[int]) -> NDArray:
        """Return x with a new shape.

        Args:
            x: Input array.
            shape: New shape.

        Returns:
            NDArray: Reshaped array.
        """
        return np.reshape(x, shape)

    def atleast_1d(self, x: ArrayLike) -> NDArray:
        """Convert x to an array with at least one dimension.

        Args:
            x: Input data.

        Returns:
            NDArray: Array with at least 1 dimension, cast to float.
        """
        return np.atleast_1d(x).astype(float)

    def atleast_2d(self, x: ArrayLike) -> NDArray:
        """Convert x to an array with at least two dimensions.

        Args:
            x: Input data.

        Returns:
            NDArray: Array with at least 2 dimensions.
        """
        return np.atleast_2d(x)

    def as_array_1d(self, data: Any) -> NDArray:
        """Force conversion to a 1-D array.

        Args:
            data: Scalar, list, tuple, or array.

        Returns:
            NDArray: 1-D array.

        Raises:
            ValueError: If data type is not supported.
        """
        if isinstance(data, int | float):
            return self.array([data])
        elif isinstance(data, list | tuple):
            return self.array(data)
        elif self.is_array_like(data):
            return data.reshape(-1)
        else:
            raise ValueError(
                "Unsupported input type: expected scalar, list, tuple, or array-like."
            )

    def stack(self, xs: Sequence[ArrayLike], axis: int = 0) -> NDArray:
        """Join a sequence of arrays along a new axis.

        Args:
            xs: Sequence of arrays.
            axis: Axis along which to stack.

        Returns:
            NDArray: Stacked array.
        """
        return np.stack(xs, axis=axis)

    def concatenate(self, arrays: Sequence[ArrayLike], axis: int = 0) -> NDArray:
        """Join arrays along an existing axis.

        Args:
            arrays: Sequence of arrays to concatenate.
            axis: Axis along which to concatenate.

        Returns:
            NDArray: Concatenated array.
        """
        return np.concatenate(arrays, axis=axis)

    def flip(self, x: ArrayLike) -> NDArray:
        """Reverse the order of elements along axis 0.

        Args:
            x: Input array.

        Returns:
            NDArray: Flipped array.
        """
        return np.flip(x, axis=0)

    def roll(self, x: ArrayLike, shift: Any, axis: Any = ()) -> NDArray:
        """Roll x elements along the given axis.

        Args:
            x: Input array.
            shift: Number of places to shift.
            axis: Axis or axes along which to roll.

        Returns:
            NDArray: Rolled array.
        """
        return np.roll(x, shift, axis=axis if axis != () else None)

    def repeat(self, x: ArrayLike, repeats: int) -> NDArray:
        """Repeat elements of x.

        Args:
            x: Input array.
            repeats: Number of repetitions.

        Returns:
            NDArray: Repeated array.
        """
        return np.repeat(x, repeats)

    def broadcast_to(self, x: ArrayLike, shape: Sequence[int]) -> NDArray:
        """Broadcast x to the given shape.

        Args:
            x: Input array.
            shape: Target shape.

        Returns:
            NDArray: Broadcast view.
        """
        return np.broadcast_to(x, shape)

    def tile(self, x: ArrayLike, dims: Any) -> NDArray:
        """Construct an array by tiling x.

        Args:
            x: Input array.
            dims: Number of repetitions per dimension.

        Returns:
            NDArray: Tiled array.
        """
        return np.tile(x, dims)

    def expand_dims(self, x: ArrayLike, axis: int) -> NDArray:
        """Insert a new axis into x.

        Args:
            x: Input array.
            axis: Position of the new axis.

        Returns:
            NDArray: Expanded array.
        """
        return np.expand_dims(x, axis)

    def meshgrid(self, *arrays: ArrayLike) -> tuple[NDArray, ...]:
        """Return coordinate matrices from coordinate vectors (xy indexing).

        Args:
            *arrays: 1-D arrays representing grid coordinates.

        Returns:
            tuple[NDArray, ...]: Coordinate matrices.
        """
        return np.meshgrid(*arrays, indexing="xy")

    def unsqueeze_last(self, x: ArrayLike) -> NDArray:
        """Add a trailing dimension to x.

        Args:
            x: Input array.

        Returns:
            NDArray: Array with an extra trailing dimension.
        """
        return x[..., np.newaxis]

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, x: ArrayLike, axis: int | None = None) -> NDArray:
        """Sum array elements over a given axis.

        Args:
            x: Input array.
            axis: Axis along which to sum.

        Returns:
            NDArray: Sum of x.
        """
        return np.sum(x, axis=axis)

    def mean(
        self, x: ArrayLike, axis: int | None = None, keepdims: bool = False
    ) -> NDArray:
        """Compute the arithmetic mean along an axis.

        Args:
            x: Input array.
            axis: Axis along which to compute the mean.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            NDArray: Mean of x.
        """
        return np.mean(x, axis=axis, keepdims=keepdims)

    def std(self, x: ArrayLike, axis: int | None = None) -> NDArray:
        """Compute the standard deviation along an axis.

        Args:
            x: Input array.
            axis: Axis along which to compute the std.

        Returns:
            NDArray: Standard deviation.
        """
        return np.std(x, axis=axis)

    def max(self, x: ArrayLike) -> Any:
        """Return the maximum value of x.

        Args:
            x: Input array.

        Returns:
            float or NDArray: Maximum value.
        """
        return np.max(x)

    def min(self, x: ArrayLike) -> Any:
        """Return the minimum value of x.

        Args:
            x: Input array.

        Returns:
            float or NDArray: Minimum value.
        """
        return np.min(x)

    def argmin(self, x: ArrayLike, axis: int | None = None) -> NDArray:
        """Return indices of the minimum values along an axis.

        Args:
            x: Input array.
            axis: Axis along which to find the minimum.

        Returns:
            NDArray: Index array.
        """
        return np.argmin(x, axis=axis)

    def argwhere(self, x: ArrayLike) -> NDArray:
        """Return indices of non-zero elements.

        Args:
            x: Input array.

        Returns:
            NDArray: Index array of shape (N, ndim).
        """
        return np.argwhere(x)

    def clip(self, x: ArrayLike, a_min: Any, a_max: Any) -> NDArray:
        """Clip values in x to [a_min, a_max].

        Args:
            x: Input array.
            a_min: Minimum value.
            a_max: Maximum value.

        Returns:
            NDArray: Clipped array.
        """
        return np.clip(x, a_min, a_max)

    def where(self, condition: Any, x: Any, y: Any) -> NDArray:
        """Return elements from x or y depending on condition.

        Args:
            condition: Boolean array.
            x: Values where condition is True.
            y: Values where condition is False.

        Returns:
            NDArray: Output array.
        """
        return np.where(condition, x, y)

    def maximum(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """Element-wise maximum of a and b.

        Args:
            a: First input array.
            b: Second input array.

        Returns:
            NDArray: Element-wise maximum.
        """
        return np.maximum(a, b)

    def minimum(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """Element-wise minimum of a and b.

        Args:
            a: First input array.
            b: Second input array.

        Returns:
            NDArray: Element-wise minimum.
        """
        return np.minimum(a, b)

    def fmax(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """Element-wise maximum, ignoring NaNs.

        Args:
            a: First input array.
            b: Second input array.

        Returns:
            NDArray: Element-wise maximum ignoring NaN.
        """
        return np.fmax(a, b)

    def power(self, x: ArrayLike, y: ArrayLike) -> NDArray:
        """Return x raised to the power y.

        Args:
            x: Base array.
            y: Exponent array.

        Returns:
            NDArray: x ** y.
        """
        return np.power(x, y)

    def diff(self, x: ArrayLike, n: int = 1, axis: int = -1, **kwargs: Any) -> NDArray:
        """Calculate the n-th discrete difference along the given axis.

        Args:
            x: Input array.
            n: Number of times to apply the difference.
            axis: Axis along which to compute differences.
            **kwargs: Additional keyword arguments forwarded to ``np.diff``
                (e.g. ``prepend``, ``append``).

        Returns:
            NDArray: Differences array.
        """
        return np.diff(x, n=n, axis=axis, **kwargs)

    def all(self, x: Any) -> bool:
        """Return True if all elements of x are True.

        Args:
            x: Input array.

        Returns:
            bool: Whether all elements are True.
        """
        return bool(np.all(x))

    def any(self, x: Any) -> bool:
        """Return True if any element of x is True.

        Args:
            x: Input array.

        Returns:
            bool: Whether any element is True.
        """
        return bool(np.any(x))

    def nanmax(
        self, x: ArrayLike, axis: int | None = None, keepdim: bool = False
    ) -> NDArray:
        """Return the maximum value, ignoring NaNs.

        Args:
            x: Input array.
            axis: Axis along which to compute the maximum.
            keepdim: Whether to keep reduced dimensions.

        Returns:
            NDArray: Maximum value ignoring NaN.
        """
        return np.nanmax(x, axis=axis, keepdims=keepdim)

    def sort(self, x: ArrayLike, axis: int = -1) -> NDArray:
        """Return a sorted copy of x.

        Args:
            x: Input array.
            axis: Axis along which to sort.

        Returns:
            NDArray: Sorted array.
        """
        return np.sort(x, axis=axis)

    def isclose(
        self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8
    ) -> NDArray:
        """Return a boolean array where elements are close.

        Args:
            a: First input.
            b: Second input.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            NDArray: Boolean array.
        """
        return np.isclose(a, b, rtol=rtol, atol=atol)

    def radians(self, x: ArrayLike) -> NDArray:
        """Convert angles from degrees to radians.

        Args:
            x: Angle in degrees.

        Returns:
            NDArray: Angle in radians.
        """
        return np.radians(x)

    def degrees(self, x: ArrayLike) -> NDArray:
        """Convert angles from radians to degrees.

        Args:
            x: Angle in radians.

        Returns:
            NDArray: Angle in degrees.
        """
        return np.degrees(x)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def matmul(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """Matrix product of two arrays.

        Args:
            a: First matrix.
            b: Second matrix.

        Returns:
            NDArray: Matrix product.
        """
        return np.matmul(a, b)

    def cross(
        self,
        a: ArrayLike,
        b: ArrayLike,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int | None = None,
    ) -> NDArray:
        """Return the cross product of two vectors.

        Args:
            a: First vector array.
            b: Second vector array.
            axisa: Axis of a that defines the vector(s).
            axisb: Axis of b that defines the vector(s).
            axisc: Axis of c that contains the cross product vector.
            axis: If defined, the axis of a, b and c that defines the vectors.

        Returns:
            NDArray: Cross product.
        """
        return np.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

    def batched_chain_matmul3(
        self, a: ArrayLike, b: ArrayLike, c: ArrayLike
    ) -> NDArray:
        """Compute a @ b @ c with promoted dtype.

        Args:
            a: First matrix.
            b: Second matrix.
            c: Third matrix.

        Returns:
            NDArray: Result of a @ b @ c.
        """
        dtype = np.result_type(a, b, c)
        return np.matmul(np.matmul(a.astype(dtype), b.astype(dtype)), c.astype(dtype))

    def matrix_vector_multiply_and_squeeze(
        self, p: NDArray, E: NDArray, backend: Literal["numpy"] = "numpy"
    ) -> NDArray:
        """Multiply p @ E[..., newaxis] and squeeze trailing dimension.

        Args:
            p: Matrix array.
            E: Vector array.
            backend: Unused; kept for backward compatibility.

        Returns:
            NDArray: Result with trailing dimension squeezed.
        """
        return np.squeeze(np.matmul(p, E[:, :, np.newaxis]), axis=2)

    def mult_p_E(self, p: NDArray, E: NDArray) -> NDArray:
        """Complex matrix-vector multiply used for polarized fields.

        Args:
            p: Jones matrix array.
            E: Electric field array.

        Returns:
            NDArray: Result of complex matrix-vector multiplication.
        """
        return np.squeeze(np.matmul(p, E[:, :, np.newaxis]), axis=2)

    def lstsq(self, a: ArrayLike, b: ArrayLike) -> NDArray:
        """Compute the least-squares solution to a @ x = b.

        Args:
            a: Left-hand side matrix (M, N).
            b: Right-hand side matrix (M,) or (M, K).

        Returns:
            NDArray: Least-squares solution (N,) or (N, K).
        """
        return np.linalg.lstsq(a, b, rcond=None)[0]

    def to_complex(self, x: NDArray) -> NDArray:
        """Cast x to complex128.

        Args:
            x: Input array.

        Returns:
            NDArray: Complex128 array.
        """
        return x.astype(np.complex128) if np.isrealobj(x) else x

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

    # ------------------------------------------------------------------
    # Random number generation
    # ------------------------------------------------------------------

    def default_rng(self, seed: int | None = None) -> NpGenerator:
        """Return a NumPy random number generator.

        Args:
            seed: Optional seed.

        Returns:
            Generator: NumPy random generator.
        """
        return np.random.default_rng(seed)

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Any = None,
        generator: NpGenerator | None = None,
    ) -> NDArray:
        """Uniform random samples in [low, high).

        Args:
            low: Lower boundary.
            high: Upper boundary.
            size: Output shape.
            generator: Optional NumPy random generator.

        Returns:
            NDArray: Uniform random samples.
        """
        if generator is None:
            generator = np.random.default_rng()
        return generator.uniform(low, high, size)

    def rand(self, *size: int) -> NDArray:
        """Random values from a uniform distribution on [0, 1).

        Args:
            *size: Shape of the output array.

        Returns:
            NDArray: Random values.
        """
        return np.random.rand(*size) if size else np.random.rand()

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Any = None,
        generator: NpGenerator | None = None,
    ) -> NDArray:
        """Random samples from a Gaussian distribution.

        Args:
            loc: Mean of the distribution.
            scale: Standard deviation.
            size: Output shape.
            generator: Optional NumPy random generator.

        Returns:
            NDArray: Normal random samples.
        """
        if generator is None:
            generator = np.random.default_rng()
        return generator.normal(loc, scale, size)

    def sobol_sampler(
        self,
        dim: int,
        num_samples: int,
        scramble: bool = True,
        seed: int | None = None,
    ) -> NDArray:
        """Generate quasi-random samples using Sobol sequences.

        Args:
            dim: Dimension of the samples.
            num_samples: Number of samples to generate.
            scramble: Whether to scramble the sequence.
            seed: Random seed for scrambling.

        Returns:
            NDArray: Samples of shape (num_samples_pow2, dim).
        """
        try:
            from scipy.stats import qmc
        except ImportError as exc:
            raise ImportError(
                "scipy is required for Sobol sampling with numpy backend"
            ) from exc

        if num_samples > 0:
            num_samples_pow2 = 1 << (num_samples - 1).bit_length()
        else:
            num_samples_pow2 = num_samples

        sobol = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
        samples = sobol.random(n=num_samples_pow2)
        return samples[:num_samples].astype(self._dtype)

    def erfinv(self, x: ArrayLike) -> NDArray:
        """Inverse error function.

        Args:
            x: Input array.

        Returns:
            NDArray: Inverse error function of x.
        """
        try:
            from scipy.special import erfinv as scipy_erfinv
        except ImportError as exc:
            raise ImportError(
                "scipy is required for erfinv with numpy backend"
            ) from exc
        return scipy_erfinv(np.asarray(x))

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def factorial(self, n: Any) -> NDArray:
        """Compute the factorial of n using the gamma function.

        Args:
            n: Non-negative integer or array of integers.

        Returns:
            NDArray: Factorial values.
        """
        return gamma(n + 1)

    def path_contains_points(self, vertices: NDArray, points: NDArray) -> NDArray:
        """Return a boolean mask of points inside the polygon.

        Args:
            vertices: Polygon vertices as (N, 2) array.
            points: Query points as (M, 2) array.

        Returns:
            NDArray: Boolean mask of shape (M,).
        """
        path = Path(vertices)
        mask = path.contains_points(points)
        return np.asarray(mask, dtype=bool)

    def pad(
        self,
        tensor: NDArray,
        pad_width: Any,
        mode: str = "constant",
        constant_values: float | None = 0,
    ) -> NDArray:
        """Pad an array.

        Args:
            tensor: Input array.
            pad_width: Number of values padded per axis.
            mode: Padding mode (only ``'constant'`` is supported).
            constant_values: Value used for constant padding.

        Returns:
            NDArray: Padded array.
        """
        if mode == "constant":
            return np.pad(tensor, pad_width, mode=mode, constant_values=constant_values)

        return np.pad(tensor, pad_width, mode=mode)

    def vectorize(self, pyfunc: Callable[..., Any]) -> Callable[..., Any]:
        """Vectorize a scalar Python function.

        Args:
            pyfunc: The scalar function to vectorize.

        Returns:
            Callable: Vectorized function.
        """
        return np.vectorize(pyfunc)

    @contextlib.contextmanager
    def errstate(self, **kwargs: Any) -> Generator[None, None, None]:  # type: ignore[override]
        """Context manager for NumPy floating-point error state.

        Args:
            **kwargs: Keyword arguments forwarded to ``np.errstate``.

        Yields:
            None
        """
        with np.errstate(**kwargs):
            yield

    def histogram(self, x: ArrayLike, bins: Any = 10) -> tuple[NDArray, NDArray]:
        """Compute a histogram of x.

        Args:
            x: Input data.
            bins: Number of bins or bin edges.

        Returns:
            tuple[NDArray, NDArray]: Bin counts and bin edges.
        """
        return np.histogram(x, bins=bins)

    def histogram2d(
        self,
        x: ArrayLike,
        y: ArrayLike,
        bins: Any,
        weights: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Compute a 2-D histogram.

        Args:
            x: x-coordinates of the sample points.
            y: y-coordinates of the sample points.
            bins: Bin specification (list of two edge arrays).
            weights: Optional weights for each sample.

        Returns:
            tuple[NDArray, NDArray, NDArray]: Histogram, x edges, y edges.
        """
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)
        return hist, xedges, yedges

    def copy_to(self, source: NDArray, destination: NDArray) -> None:
        """Copy source array into destination in-place.

        Args:
            source: Source array.
            destination: Destination array (modified in place).
        """
        np.copyto(destination, source)

    # ------------------------------------------------------------------
    # Numpy-specific helpers (not in ABC — kept for backward compatibility)
    # ------------------------------------------------------------------

    def from_matrix(self, matrix: NDArray) -> R:
        """Create a SciPy Rotation from a rotation matrix.

        Args:
            matrix: Rotation matrix.

        Returns:
            Rotation: SciPy Rotation object.
        """
        return R.from_matrix(matrix)

    def from_euler(self, euler: NDArray) -> R:
        """Create a SciPy Rotation from Euler angles.

        Args:
            euler: Euler angles in the 'xyz' convention.

        Returns:
            Rotation: SciPy Rotation object.
        """
        return R.from_euler("xyz", euler)
