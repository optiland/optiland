"""
AbstractBackend ABC, @passthrough decorator, and BackendCapabilityError.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType


class BackendCapabilityError(Exception):
    """Raised when an operation is not supported by the current backend.

    Example:
        >>> be.grad_mode.enable()  # on numpy backend
        BackendCapabilityError: grad_mode requires a backend that supports
        gradients. Current backend: 'numpy'. Try: be.set_backend('torch')
    """


def passthrough(*func_names: str):
    """Inject concrete passthrough methods into the decorated class.

    For each name in func_names, adds a method that calls
    ``self._lib.<name>(*args, **kwargs)``. Only injected if the class does
    not already define the method — explicit overrides always take priority.

    Args:
        *func_names: Names of functions to inject from the backend library.

    Returns:
        A class decorator that injects the passthrough methods.
    """

    def decorator(cls: type) -> type:
        for name in func_names:
            if not hasattr(cls, name):

                def _make(n: str) -> Callable[..., Any]:
                    def method(self: AbstractBackend, *args: Any, **kwargs: Any) -> Any:
                        return getattr(self._lib, n)(*args, **kwargs)

                    method.__name__ = n
                    method.__qualname__ = f"{cls.__name__}.{n}"
                    return method

                setattr(cls, name, _make(name))
        return cls

    return decorator


@passthrough(
    # Trigonometric
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "sinh",
    "cosh",
    "tanh",
    # Math
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "abs",
    "sign",
    "floor",
    "ceil",
    "hypot",
    # Angle conversion (np.deg2rad / torch.deg2rad share same name)
    "deg2rad",
    "rad2deg",
    # Checks
    "isnan",
    "isinf",
    "isfinite",
    # Logic
    "logical_and",
    "logical_or",
    "logical_not",
    # Complex
    "conj",
    "real",
    # Linear algebra (identical API in np and torch)
    "outer",
    "einsum",
    "dot",
    # NaN-safe reductions (np.nansum/torch.nansum, np.nanmean/torch.nanmean)
    "nansum",
    "nanmean",
    # Stack variants with matching names
    "vstack",
    "column_stack",
    # Sorting / searching
    "searchsorted",
    "round",
    # Array info — torch backend overrides these with its own implementations
    "shape",
    "size",
    "copy",
    "isscalar",
    "load",
    # dtype / machine-epsilon info
    "finfo",
    # Closeness / comparison (np.allclose / torch.allclose)
    "allclose",
    # Complex helpers
    "imag",
    # Sign copying
    "copysign",
)
class AbstractBackend(ABC):
    """Abstract base class that defines the full backend contract.

    All backends must subclass this class and implement every abstract method.
    Concrete passthrough methods (injected by @passthrough) delegate to
    ``self._lib.<name>(...)``; subclasses may override them.

    Attributes:
        _lib: The underlying library module (``numpy`` or ``torch``).
    """

    _lib: ModuleType  # Each subclass sets: _lib = np  or  _lib = torch

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g. 'numpy' or 'torch')."""

    # ------------------------------------------------------------------
    # Submodule proxies
    # ------------------------------------------------------------------

    @property
    def linalg(self) -> Any:
        """Expose the linear-algebra submodule of the underlying library."""
        return self._lib.linalg

    @property
    def fft(self) -> Any:
        """Expose the FFT submodule of the underlying library."""
        return self._lib.fft

    @property
    def random(self) -> Any:
        """Expose the random submodule of the underlying library."""
        return self._lib.random

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_gradients(self) -> bool:
        """Return True if this backend supports automatic differentiation."""
        return False

    @property
    def supports_gpu(self) -> bool:
        """Return True if this backend can use GPU acceleration."""
        return False

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------

    @abstractmethod
    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        """Set the floating-point precision used by this backend.

        Args:
            precision: Either ``'float32'`` or ``'float64'``.
        """

    @abstractmethod
    def get_precision(self) -> int:
        """Return the current precision as an integer (32 or 64)."""

    # ------------------------------------------------------------------
    # Capability-gated torch-only features
    # Default implementations raise BackendCapabilityError.
    # TorchBackend overrides these.
    # ------------------------------------------------------------------

    @property
    def grad_mode(self) -> Any:
        """Control object for gradient computation (torch only)."""
        raise BackendCapabilityError(
            f"grad_mode requires a backend that supports gradients. "
            f"Current backend: '{self.name}'. Try: be.set_backend('torch')"
        )

    @property
    def autograd(self) -> Any:
        """The autograd submodule (torch only)."""
        raise BackendCapabilityError(
            f"autograd is not supported by backend '{self.name}'."
        )

    def set_device(self, device: str) -> None:
        """Set the compute device (torch only).

        Args:
            device: Device string (e.g. ``'cpu'`` or ``'cuda'``).

        Raises:
            BackendCapabilityError: Always, on non-torch backends.
        """
        raise BackendCapabilityError(
            f"set_device is not supported by backend '{self.name}'."
        )

    def get_device(self) -> str:
        """Return the current compute device (torch only).

        Raises:
            BackendCapabilityError: Always, on non-torch backends.
        """
        raise BackendCapabilityError(
            f"get_device is not supported by backend '{self.name}'."
        )

    def get_complex_precision(self) -> Any:
        """Return the complex dtype matching the current precision (torch only).

        Raises:
            BackendCapabilityError: Always, on non-torch backends.
        """
        raise BackendCapabilityError(
            f"get_complex_precision is not supported by backend '{self.name}'."
        )

    def to_tensor(self, data: Any, device: Any = None) -> Any:
        """Convert data to a backend tensor with current precision (torch only).

        Raises:
            BackendCapabilityError: Always, on non-torch backends.
        """
        raise BackendCapabilityError(
            f"to_tensor is not supported by backend '{self.name}'."
        )

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    @abstractmethod
    def array(self, x: Any) -> Any:
        """Create a backend array/tensor from x."""

    @abstractmethod
    def zeros(self, shape: Sequence[int], dtype: Any = None) -> Any:
        """Return a new array of the given shape filled with zeros."""

    @abstractmethod
    def ones(self, shape: Sequence[int], dtype: Any = None) -> Any:
        """Return a new array of the given shape filled with ones."""

    @abstractmethod
    def full(self, shape: Sequence[int], fill_value: Any, dtype: Any = None) -> Any:
        """Return a new array of the given shape filled with fill_value."""

    @abstractmethod
    def linspace(self, start: float, stop: float, num: int = 50) -> Any:
        """Return evenly spaced numbers over the specified interval."""

    @abstractmethod
    def arange(self, *args: Any, **kwargs: Any) -> Any:
        """Return evenly spaced values within a given interval."""

    @abstractmethod
    def zeros_like(self, x: Any) -> Any:
        """Return an array of zeros with the same shape and type as x."""

    @abstractmethod
    def ones_like(self, x: Any) -> Any:
        """Return an array of ones with the same shape and type as x."""

    @abstractmethod
    def full_like(self, x: Any, fill_value: Any) -> Any:
        """Return a full array with the same shape as x."""

    @abstractmethod
    def empty(self, shape: Sequence[int]) -> Any:
        """Return a new uninitialized array of the given shape."""

    @abstractmethod
    def empty_like(self, x: Any) -> Any:
        """Return an uninitialized array with the same shape as x."""

    @abstractmethod
    def eye(self, n: int) -> Any:
        """Return a 2D identity matrix of size n."""

    @abstractmethod
    def asarray(self, x: Any, **kwargs: Any) -> Any:
        """Convert x to a backend array without copying if possible."""

    # ------------------------------------------------------------------
    # Array utilities
    # ------------------------------------------------------------------

    @abstractmethod
    def cast(self, x: Any) -> Any:
        """Cast x to the current floating-point precision."""

    @abstractmethod
    def is_array_like(self, x: Any) -> bool:
        """Return True if x is a list, tuple, or backend array."""

    @abstractmethod
    def arange_indices(self, start: Any, stop: Any = None, step: int = 1) -> Any:
        """Return an integer array of indices."""

    @abstractmethod
    def ravel(self, x: Any) -> Any:
        """Return a 1D float array of x."""

    # ------------------------------------------------------------------
    # Shape and indexing
    # ------------------------------------------------------------------

    @abstractmethod
    def transpose(self, x: Any, axes: Sequence[int] | None = None) -> Any:
        """Permute the dimensions of x."""

    @abstractmethod
    def reshape(self, x: Any, shape: Sequence[int]) -> Any:
        """Return x reshaped to the given shape."""

    @abstractmethod
    def atleast_1d(self, x: Any) -> Any:
        """Return x as an array with at least one dimension."""

    @abstractmethod
    def atleast_2d(self, x: Any) -> Any:
        """Return x as an array with at least two dimensions."""

    @abstractmethod
    def as_array_1d(self, data: Any) -> Any:
        """Force conversion to a 1D array."""

    @abstractmethod
    def stack(self, xs: Sequence[Any], axis: int = 0) -> Any:
        """Join a sequence of arrays along a new axis."""

    @abstractmethod
    def concatenate(self, arrays: Sequence[Any], axis: int = 0) -> Any:
        """Join arrays along an existing axis."""

    @abstractmethod
    def flip(self, x: Any) -> Any:
        """Reverse the order of elements in x along axis 0."""

    @abstractmethod
    def roll(self, x: Any, shift: Any, axis: Any = ()) -> Any:
        """Roll x elements along the given axis."""

    @abstractmethod
    def repeat(self, x: Any, repeats: int) -> Any:
        """Repeat elements of x."""

    @abstractmethod
    def broadcast_to(self, x: Any, shape: Sequence[int]) -> Any:
        """Broadcast x to the given shape."""

    @abstractmethod
    def tile(self, x: Any, dims: Any) -> Any:
        """Construct an array by tiling x."""

    @abstractmethod
    def expand_dims(self, x: Any, axis: int) -> Any:
        """Expand the shape of x by inserting a new axis."""

    @abstractmethod
    def meshgrid(self, *arrays: Any) -> tuple[Any, ...]:
        """Return coordinate matrices from coordinate vectors."""

    @abstractmethod
    def unsqueeze_last(self, x: Any) -> Any:
        """Add a trailing dimension to x."""

    # ------------------------------------------------------------------
    # Reductions and math with semantic mismatches
    # ------------------------------------------------------------------

    @abstractmethod
    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum array elements over a given axis."""

    @abstractmethod
    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        """Compute the arithmetic mean, ignoring NaNs."""

    @abstractmethod
    def std(self, x: Any, axis: int | None = None) -> Any:
        """Compute the standard deviation along the given axis."""

    @abstractmethod
    def max(self, x: Any) -> Any:
        """Return the maximum value of x."""

    @abstractmethod
    def min(self, x: Any) -> Any:
        """Return the minimum value of x."""

    @abstractmethod
    def argmin(self, x: Any, axis: int | None = None) -> Any:
        """Return indices of the minimum values along an axis."""

    @abstractmethod
    def argwhere(self, x: Any) -> Any:
        """Return indices of non-zero elements."""

    @abstractmethod
    def clip(self, x: Any, a_min: Any, a_max: Any) -> Any:
        """Clip the values in x to [a_min, a_max]."""

    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Return elements chosen from x or y depending on condition."""

    @abstractmethod
    def maximum(self, a: Any, b: Any) -> Any:
        """Element-wise maximum of a and b."""

    @abstractmethod
    def minimum(self, a: Any, b: Any) -> Any:
        """Element-wise minimum of a and b."""

    @abstractmethod
    def fmax(self, a: Any, b: Any) -> Any:
        """Element-wise maximum, ignoring NaNs."""

    @abstractmethod
    def power(self, x: Any, y: Any) -> Any:
        """Return x raised to the power y."""

    @abstractmethod
    def diff(self, x: Any, n: int = 1, axis: int = -1, **kwargs: Any) -> Any:
        """Calculate the n-th discrete difference along the given axis."""

    @abstractmethod
    def all(self, x: Any) -> bool:
        """Return True if all elements of x are True."""

    @abstractmethod
    def any(self, x: Any) -> bool:
        """Return True if any element of x is True."""

    @abstractmethod
    def nanmax(self, x: Any, axis: int | None = None, keepdim: bool = False) -> Any:
        """Return the maximum, ignoring NaNs."""

    @abstractmethod
    def sort(self, x: Any, axis: int = -1) -> Any:
        """Return a sorted copy of x."""

    @abstractmethod
    def isclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
        """Return a boolean array where elements are close."""

    @abstractmethod
    def radians(self, x: Any) -> Any:
        """Convert angles from degrees to radians."""

    @abstractmethod
    def degrees(self, x: Any) -> Any:
        """Convert angles from radians to degrees."""

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix product of two arrays."""

    @abstractmethod
    def cross(
        self,
        a: Any,
        b: Any,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int | None = None,
    ) -> Any:
        """Return the cross product of two vectors."""

    @abstractmethod
    def batched_chain_matmul3(self, a: Any, b: Any, c: Any) -> Any:
        """Compute a @ b @ c with promoted dtype."""

    @abstractmethod
    def matrix_vector_multiply_and_squeeze(self, p: Any, E: Any) -> Any:
        """Multiply p @ E[..., newaxis] and squeeze the trailing dimension."""

    @abstractmethod
    def mult_p_E(self, p: Any, E: Any) -> Any:
        """Complex matrix-vector multiply used for polarized fields."""

    @abstractmethod
    def lstsq(self, a: Any, b: Any) -> Any:
        """Return the least-squares solution to a @ x = b."""

    @abstractmethod
    def to_complex(self, x: Any) -> Any:
        """Cast x to complex128."""

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    @abstractmethod
    def nearest_nd_interpolator(self, points: Any, values: Any, x: Any, y: Any) -> Any:
        """Nearest-neighbour interpolation on an N-D dataset."""

    @abstractmethod
    def interp(self, x: Any, xp: Any, fp: Any) -> Any:
        """1-D linear interpolation."""

    @abstractmethod
    def grid_sample(
        self,
        input: Any,
        grid: Any,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> Any:
        """Sample input using bilinear/nearest interpolation on a grid."""

    # ------------------------------------------------------------------
    # Polynomial
    # ------------------------------------------------------------------

    @abstractmethod
    def polyfit(self, x: Any, y: Any, degree: int) -> Any:
        """Least-squares polynomial fit."""

    @abstractmethod
    def polyval(self, coeffs: Any, x: Any) -> Any:
        """Evaluate a polynomial at specific values."""

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    @abstractmethod
    def fftconvolve(self, in1: Any, in2: Any, mode: str = "full") -> Any:
        """FFT-based convolution."""

    # ------------------------------------------------------------------
    # Random number generation
    # ------------------------------------------------------------------

    @abstractmethod
    def default_rng(self, seed: int | None = None) -> Any:
        """Return a random number generator seeded with seed."""

    @abstractmethod
    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Any = None,
        generator: Any = None,
    ) -> Any:
        """Uniform random samples in [low, high)."""

    @abstractmethod
    def rand(self, *size: int) -> Any:
        """Random values from a uniform distribution on [0, 1)."""

    @abstractmethod
    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Any = None,
        generator: Any = None,
    ) -> Any:
        """Random samples from a normal (Gaussian) distribution."""

    @abstractmethod
    def sobol_sampler(
        self,
        dim: int,
        num_samples: int,
        scramble: bool = True,
        seed: int | None = None,
    ) -> Any:
        """Generate quasi-random samples using Sobol sequences."""

    @abstractmethod
    def erfinv(self, x: Any) -> Any:
        """Inverse error function."""

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    @abstractmethod
    def factorial(self, n: Any) -> Any:
        """Compute the factorial of n."""

    @abstractmethod
    def path_contains_points(self, vertices: Any, points: Any) -> Any:
        """Return a boolean mask of points inside the polygon."""

    @abstractmethod
    def pad(
        self,
        tensor: Any,
        pad_width: Any,
        mode: str = "constant",
        constant_values: float | None = 0,
    ) -> Any:
        """Pad an array."""

    @abstractmethod
    def vectorize(self, pyfunc: Callable[..., Any]) -> Callable[..., Any]:
        """Vectorize a scalar function over array inputs."""

    @abstractmethod
    def errstate(self, **kwargs: Any) -> Any:
        """Context manager for floating-point error state."""

    @abstractmethod
    def histogram(self, x: Any, bins: Any = 10) -> tuple[Any, Any]:
        """Compute a histogram of x."""

    @abstractmethod
    def histogram2d(
        self,
        x: Any,
        y: Any,
        bins: Any,
        weights: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Compute a 2-D histogram."""
