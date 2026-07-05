"""
PyTorch backend -- explicit passthrough overrides with array() casting.

Provides PassthroughMixin, one of the mixins composed into
TorchBackend (see optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Tensor


class PassthroughMixin:
    """Explicit passthrough overrides with array() casting."""

    # ------------------------------------------------------------------
    # Passthrough overrides — explicit implementations with array() cast
    # These override the passthrough methods inherited from AbstractBackend
    # to ensure Python scalars and lists are accepted (via self.array()).
    # ------------------------------------------------------------------

    def tan(self, x: Any) -> Tensor:
        """Compute the tangent of x (element-wise).

        Args:
            x: Input data.

        Returns:
            Tensor: Tangent values.
        """
        return torch.tan(self.array(x))

    def arcsin(self, x: Any) -> Tensor:
        """Compute the arcsine of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Arcsine values.
        """
        return torch.arcsin(self.array(x))

    def arccos(self, x: Any) -> Tensor:
        """Compute the arccosine of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Arccosine values.
        """
        return torch.arccos(self.array(x))

    def arctan(self, x: Any) -> Tensor:
        """Compute the arctangent of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Arctangent values.
        """
        return torch.arctan(self.array(x))

    def arctan2(self, y: Any, x: Any) -> Tensor:
        """Compute the element-wise arctan2.

        Args:
            y: y-coordinates.
            x: x-coordinates.

        Returns:
            Tensor: Arctangent values (angle in radians).
        """
        return torch.arctan2(self.array(y), self.array(x))

    def sinh(self, x: Any) -> Tensor:
        """Compute the hyperbolic sine of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Hyperbolic sine values.
        """
        return torch.sinh(self.array(x))

    def cosh(self, x: Any) -> Tensor:
        """Compute the hyperbolic cosine of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Hyperbolic cosine values.
        """
        return torch.cosh(self.array(x))

    def tanh(self, x: Any) -> Tensor:
        """Compute the hyperbolic tangent of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Hyperbolic tangent values.
        """
        return torch.tanh(self.array(x))

    def log(self, x: Any) -> Tensor:
        """Compute the natural logarithm of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Natural log values.
        """
        return torch.log(self.array(x))

    def log10(self, x: Any) -> Tensor:
        """Compute the base-10 logarithm of x.

        Args:
            x: Input data.

        Returns:
            Tensor: log10 values.
        """
        return torch.log10(self.array(x))

    def sign(self, x: Any) -> Tensor:
        """Compute the sign of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Sign values (-1, 0, or 1).
        """
        return torch.sign(self.array(x))

    def floor(self, x: Any) -> Tensor:
        """Round down to the nearest integer.

        Args:
            x: Input data.

        Returns:
            Tensor: Floor values.
        """
        return torch.floor(self.array(x))

    def ceil(self, x: Any) -> Tensor:
        """Round up to the nearest integer.

        Args:
            x: Input data.

        Returns:
            Tensor: Ceiling values.
        """
        return torch.ceil(self.array(x))

    def hypot(self, x: Any, y: Any) -> Tensor:
        """Compute the hypotenuse given legs x and y.

        Args:
            x: First leg.
            y: Second leg.

        Returns:
            Tensor: Hypotenuse values.
        """
        return torch.hypot(self.array(x), self.array(y))

    def deg2rad(self, x: Any) -> Tensor:
        """Convert angles from degrees to radians.

        Args:
            x: Angle in degrees.

        Returns:
            Tensor: Angle in radians.
        """
        return torch.deg2rad(self.array(x))

    def rad2deg(self, x: Any) -> Tensor:
        """Convert angles from radians to degrees.

        Args:
            x: Angle in radians.

        Returns:
            Tensor: Angle in degrees.
        """
        return torch.rad2deg(self.array(x))

    def conj(self, x: Any) -> Tensor:
        """Compute the complex conjugate of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Complex conjugate.
        """
        return torch.conj(self.array(x))

    def real(self, x: Any) -> Tensor:
        """Return the real part of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Real part.
        """
        return torch.real(self.array(x))

    def imag(self, x: Any) -> Tensor:
        """Return the imaginary part of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Imaginary part.
        """
        return torch.imag(self.array(x))

    def allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Return True if all elements in a and b are close.

        Args:
            a: First input.
            b: Second input.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            bool: Whether all elements are close.
        """
        return bool(torch.allclose(self.array(a), self.array(b), rtol=rtol, atol=atol))

    def copysign(self, x: Any, y: Any) -> Tensor:
        """Return x with the sign of y (element-wise).

        Args:
            x: Magnitude array.
            y: Sign array.

        Returns:
            Tensor: x with sign from y.
        """
        return torch.copysign(self.array(x), self.array(y))

    def sin(self, x: Any) -> Tensor:
        """Compute the sine of x (element-wise).

        Args:
            x: Input data.

        Returns:
            Tensor: Sine values.
        """
        return torch.sin(self.array(x))

    def cos(self, x: Any) -> Tensor:
        """Compute the cosine of x (element-wise).

        Args:
            x: Input data.

        Returns:
            Tensor: Cosine values.
        """
        return torch.cos(self.array(x))

    def sqrt(self, x: Any) -> Tensor:
        """Compute the square root of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Square root values.
        """
        return torch.sqrt(self.array(x))

    def exp(self, x: Any) -> Tensor:
        """Compute the exponential of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Exponential values.
        """
        return torch.exp(self.array(x))

    def abs(self, x: Any) -> Tensor:
        """Compute the absolute value of x.

        Args:
            x: Input data.

        Returns:
            Tensor: Absolute values.
        """
        return torch.abs(self.array(x))

    def log2(self, x: Any) -> Tensor:
        """Compute the base-2 logarithm of x.

        Args:
            x: Input data.

        Returns:
            Tensor: log2 values.
        """
        return torch.log2(self.array(x))

    def isinf(self, x: Any) -> Any:
        """Check if input is infinity, accepting scalars and tensors.

        Args:
            x: Input (scalar, ndarray, or Tensor).

        Returns:
            bool or Tensor: Whether x is infinite.
        """
        if isinstance(x, torch.Tensor):
            return torch.isinf(x)
        return torch.isinf(torch.tensor(x, dtype=self._dtype()))

    def isnan(self, x: Any) -> Any:
        """Check if input is NaN, accepting scalars and tensors.

        Args:
            x: Input (scalar, ndarray, or Tensor).

        Returns:
            bool or Tensor: Whether x is NaN.
        """
        if isinstance(x, torch.Tensor):
            return torch.isnan(x)
        return torch.isnan(torch.tensor(x, dtype=self._dtype()))

    def isfinite(self, x: Any) -> Any:
        """Check if input is finite, accepting scalars and tensors.

        Args:
            x: Input (scalar, ndarray, or Tensor).

        Returns:
            bool or Tensor: Whether x is finite.
        """
        if isinstance(x, torch.Tensor):
            return torch.isfinite(x)
        return torch.isfinite(torch.tensor(x, dtype=self._dtype()))
