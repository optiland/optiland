"""
PyTorch backend -- reduction operations.

Provides ReductionsMixin, one of the mixins composed into
TorchBackend (see optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor


class ReductionsMixin:
    """Reduction operations."""

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, x: Any, axis: int | None = None) -> Tensor:
        """Sum tensor elements over a given axis.

        Args:
            x: Input tensor.
            axis: Dimension along which to sum.

        Returns:
            Tensor: Sum.
        """
        return torch.sum(x, dim=axis) if axis is not None else torch.sum(x)

    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Compute the arithmetic mean, ignoring NaNs.

        Args:
            x: Input tensor.
            axis: Dimension along which to compute the mean.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            Tensor: Mean.
        """
        x = self.array(x)
        mask = ~torch.isnan(x)
        cnt = mask.sum(dim=axis, keepdim=keepdims).to(x.dtype)
        x0 = torch.where(mask, x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
        s = x0.sum(dim=axis, keepdim=keepdims)
        return torch.where(
            cnt > 0,
            s / cnt,
            torch.tensor(float("nan"), dtype=x.dtype, device=x.device),
        )

    def std(self, x: Any, axis: int | None = None) -> Tensor:
        """Compute the standard deviation along an axis.

        Args:
            x: Input tensor.
            axis: Dimension along which to compute std.

        Returns:
            Tensor: Standard deviation.
        """
        return torch.std(x, dim=axis) if axis is not None else torch.std(x)

    def max(self, x: Any) -> Any:
        """Return the maximum value of x.

        Args:
            x: Input tensor or array.

        Returns:
            float: Maximum value as a Python scalar.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().max().item()
        return np.max(x)

    def min(self, x: Any) -> Any:
        """Return the minimum value of x.

        Args:
            x: Input tensor or array.

        Returns:
            float: Minimum value as a Python scalar.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().min().item()
        return np.min(x)

    def argmin(self, x: Any, axis: int | None = None) -> Tensor:
        """Return indices of the minimum values along a dimension.

        Args:
            x: Input tensor.
            axis: Dimension along which to find the minimum.

        Returns:
            Tensor: Index tensor.
        """
        return torch.argmin(x, dim=axis)

    def argwhere(self, x: Any) -> Tensor:
        """Return indices of non-zero elements.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Index tensor of shape (N, ndim).
        """
        return torch.nonzero(x, as_tuple=False)

    def clip(self, x: Any, a_min: Any, a_max: Any) -> Tensor:
        """Clip values in x to [a_min, a_max].

        Args:
            x: Input tensor.
            a_min: Minimum value.
            a_max: Maximum value.

        Returns:
            Tensor: Clipped tensor.
        """
        return torch.clamp(x, a_min, a_max)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Return elements from x or y depending on condition.

        Args:
            condition: Boolean tensor or bool.
            x: Values where condition is True.
            y: Values where condition is False.

        Returns:
            Tensor: Output tensor.
        """
        if isinstance(condition, bool):
            return x if condition else y

        # Bare Python numbers passed straight to torch.where are "wrapped"
        # scalars: combined with a non-scalar tensor of a different dtype,
        # PyTorch's type promotion keeps the *tensor's* dtype rather than
        # upcasting, so a plain 1.0/-1.0 silently downgrades a float64
        # computation to float32. Materializing a bare scalar against the
        # other operand's own dtype (when it's a tensor) preserves that
        # operand's precision/kind instead -- including int index tensors
        # and complex tensors, which the backend's default float precision
        # would otherwise clobber.
        if not isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            x = torch.tensor(x, device=y.device, dtype=y.dtype)
        elif not isinstance(x, torch.Tensor):
            x = self.array(x)
        if not isinstance(y, torch.Tensor) and isinstance(x, torch.Tensor):
            y = torch.tensor(y, device=x.device, dtype=x.dtype)
        elif not isinstance(y, torch.Tensor):
            y = self.array(y)
        return torch.where(condition, x, y)

    def maximum(self, a: Any, b: Any) -> Tensor:
        """Element-wise maximum of a and b.

        Args:
            a: First input tensor.
            b: Second input tensor.

        Returns:
            Tensor: Element-wise maximum.
        """
        return torch.maximum(self.array(a), self.array(b))

    def minimum(self, a: Any, b: Any) -> Tensor:
        """Element-wise minimum of a and b.

        Args:
            a: First input tensor.
            b: Second input tensor.

        Returns:
            Tensor: Element-wise minimum.
        """
        return torch.minimum(self.array(a), self.array(b))

    def fmax(self, a: Any, b: Any) -> Tensor:
        """Element-wise maximum, ignoring NaNs.

        Args:
            a: First input tensor.
            b: Second input tensor.

        Returns:
            Tensor: Element-wise maximum ignoring NaN.
        """
        return torch.fmax(self.array(a), self.array(b))

    def power(self, x: Any, y: Any) -> Tensor:
        """Return x raised to the power y.

        Args:
            x: Base tensor.
            y: Exponent tensor.

        Returns:
            Tensor: x ** y.
        """
        return torch.pow(self.array(x), self.array(y))

    def diff(self, x: Any, n: int = 1, axis: int = -1, **kwargs: Any) -> Tensor:
        """Calculate the n-th discrete difference along a dimension.

        Args:
            x: Input tensor.
            n: Number of times to apply the difference.
            axis: Dimension along which to compute differences.
            **kwargs: Additional keyword arguments forwarded to ``torch.diff``
                (e.g. ``prepend``, ``append``).

        Returns:
            Tensor: Differences tensor.
        """
        return torch.diff(x, n=n, dim=axis, **kwargs)

    def all(self, x: Any) -> bool:
        """Return True if all elements of x are True.

        Args:
            x: Input tensor or bool.

        Returns:
            bool: Whether all elements are True.
        """
        if isinstance(x, bool):
            return x
        t = torch.as_tensor(x, dtype=self._dtype(), device=self._device())
        return bool(torch.all(t).item())

    def any(self, x: Any) -> bool:
        """Return True if any element of x is True.

        Args:
            x: Input tensor or bool.

        Returns:
            bool: Whether any element is True.
        """
        if isinstance(x, bool):
            return x
        t = torch.as_tensor(x, dtype=self._dtype(), device=self._device())
        return bool(torch.any(t).item())

    def nanmax(
        self, x: Tensor, axis: int | None = None, keepdim: bool = False
    ) -> Tensor:
        """Return the maximum value, ignoring NaNs.

        Args:
            x: Input tensor.
            axis: Dimension along which to compute the maximum.
            keepdim: Whether to keep reduced dimensions.

        Returns:
            Tensor: Maximum value ignoring NaN.
        """
        nan_mask = torch.isnan(x)
        replaced = x.clone()
        replaced[nan_mask] = float("-inf")
        if axis is not None:
            result, _ = torch.max(replaced, dim=axis, keepdim=keepdim)
        else:
            result = torch.max(replaced)
        return result

    def sort(self, x: Any, axis: int = -1) -> Tensor:
        """Return a sorted tensor along the given dimension.

        Args:
            x: Input tensor.
            axis: Dimension along which to sort.

        Returns:
            Tensor: Sorted tensor (values only, not the indices).
        """
        return torch.sort(x, dim=axis).values

    def isclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Tensor:
        """Return a boolean tensor where elements are close.

        Args:
            a: First input.
            b: Second input.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            Tensor: Boolean tensor.
        """
        return torch.isclose(self.array(a), self.array(b), rtol=rtol, atol=atol)

    def radians(self, x: Any) -> Tensor:
        """Convert angles from degrees to radians.

        Args:
            x: Angle in degrees.

        Returns:
            Tensor: Angle in radians.
        """
        return torch.deg2rad(self.array(x))

    def degrees(self, x: Any) -> Tensor:
        """Convert angles from radians to degrees.

        Args:
            x: Angle in radians.

        Returns:
            Tensor: Angle in degrees.
        """
        return torch.rad2deg(self.array(x))
