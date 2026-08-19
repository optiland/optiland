"""
PyTorch backend -- interpolation, polynomial, and signal-processing operations.

Provides InterpolationMixin, one of the mixins composed into
TorchBackend (see optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


class InterpolationMixin:
    """Interpolation, polynomial, and signal-processing operations."""

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interp(self, x: Any, xp: Any, fp: Any) -> Tensor:
        """1-D linear interpolation.

        Args:
            x: x-coordinates of the interpolated values.
            xp: x-coordinates of the data points.
            fp: y-coordinates of the data points.

        Returns:
            Tensor: Interpolated values.
        """
        x = torch.as_tensor(x, dtype=self._dtype(), device=self._device())
        xp = torch.as_tensor(xp, dtype=self._dtype(), device=self._device())
        fp = torch.as_tensor(fp, dtype=self._dtype(), device=self._device())
        sorted_indices = torch.argsort(xp)
        xp = xp[sorted_indices]
        fp = fp[sorted_indices]
        x_clipped = torch.clip(x, xp[0], xp[-1])
        indices = torch.searchsorted(xp, x_clipped, right=True)
        indices = torch.clamp(indices, 1, len(xp) - 1)
        x0 = xp[indices - 1]
        x1 = xp[indices]
        y0 = fp[indices - 1]
        y1 = fp[indices]
        return y0 + (y1 - y0) * (x_clipped - x0) / (x1 - x0)

    def nearest_nd_interpolator(
        self, points: Tensor, values: Tensor, Hx: Tensor, Hy: Tensor
    ) -> Tensor:
        """Nearest-neighbour interpolation on an N-D dataset.

        Args:
            points: Known sample points of shape (N, D).
            values: Values at the sample points.
            Hx: Query x coordinates.
            Hy: Query y coordinates.

        Returns:
            Tensor: Interpolated values.

        Raises:
            ValueError: If Hx or Hy is None.
        """
        if Hx is None or Hy is None:
            raise ValueError("Hx and Hy must be provided")
        Hx, Hy = self.array(Hx), self.array(Hy)
        Hx, Hy = torch.broadcast_tensors(Hx, Hy)
        q_flat = torch.stack([Hx, Hy], dim=-1).reshape(-1, 2)
        d = torch.cdist(q_flat, points.to(dtype=q_flat.dtype, device=q_flat.device))
        idx = d.argmin(dim=1)
        vals = values.view(points.shape[0], -1)
        out = vals[idx].view(*Hx.shape, -1)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def grid_sample(
        self,
        input: Tensor,
        grid: Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> Tensor:
        """Sample input using torch.nn.functional.grid_sample.

        Args:
            input: Input tensor of shape (N, C, H_in, W_in).
            grid: Grid tensor of shape (N, H_out, W_out, 2).
            mode: Interpolation mode.
            padding_mode: Padding mode.
            align_corners: Whether to align corners.

        Returns:
            Tensor: Output tensor of shape (N, C, H_out, W_out).
        """
        return F.grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    # ------------------------------------------------------------------
    # Polynomial
    # ------------------------------------------------------------------

    def polyfit(self, x: Tensor, y: Tensor, degree: int) -> Tensor:
        """Least-squares polynomial fit.

        Args:
            x: x-coordinates of the sample points.
            y: y-coordinates of the sample points.
            degree: Degree of the polynomial.

        Returns:
            Tensor: Polynomial coefficients, highest power first.
        """
        X = torch.stack([x**i for i in range(degree, -1, -1)], dim=1)
        result = torch.linalg.lstsq(X, y.unsqueeze(1))
        coeffs = result.solution
        return coeffs[: degree + 1].squeeze()

    def polyval(self, coeffs: Any, x: Any) -> Any:
        """Evaluate a polynomial at specific values.

        Args:
            coeffs: Polynomial coefficients, highest power first.
            x: Values at which to evaluate.

        Returns:
            Tensor or float: Evaluated polynomial.
        """
        return sum(c * x**i for i, c in enumerate(reversed(coeffs)))

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def fftconvolve(
        self, in1: Tensor, in2: Tensor, mode: Literal["full", "valid", "same"] = "full"
    ) -> Tensor:
        """FFT-based convolution using PyTorch.

        Args:
            in1: First input tensor (N-D).
            in2: Second input tensor (N-D).
            mode: Convolution mode (``'full'``, ``'valid'``, ``'same'``).

        Returns:
            Tensor: Convolved tensor.

        Raises:
            ValueError: If inputs have different dimensionality or mode is
                unknown.
        """
        in1 = self.array(in1)
        in2 = self.array(in2)

        ndim = in1.ndim
        if in2.ndim != ndim:
            raise ValueError("Inputs must have the same dimensionality.")

        axes = tuple(range(ndim)) if ndim < 2 else (-2, -1)

        s1 = in1.shape
        s2 = in2.shape

        fft_shape = list(in1.shape)
        for axis in axes:
            fft_shape[axis] = s1[axis] + s2[axis] - 1

        IN1 = torch.fft.fftn(in1, s=[fft_shape[axis] for axis in axes], dim=axes)
        IN2 = torch.fft.fftn(in2, s=[fft_shape[axis] for axis in axes], dim=axes)
        ret = torch.fft.ifftn(
            IN1 * IN2, s=[fft_shape[axis] for axis in axes], dim=axes
        ).real

        if mode == "full":
            return ret

        crop_slices = [slice(None)] * ndim

        if mode == "same":
            for axis in axes:
                start = (s2[axis] - 1) // 2
                end = start + s1[axis]
                crop_slices[axis] = slice(start, end)
            return ret[tuple(crop_slices)]

        if mode == "valid":
            for axis in axes:
                start = s2[axis] - 1
                end = s1[axis]
                crop_slices[axis] = slice(start, end)
            return ret[tuple(crop_slices)]

        raise ValueError(f"Unknown mode: {mode}")
