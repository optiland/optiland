"""
PyTorch backend -- miscellaneous operations.

Provides MiscMixin, one of the mixins composed into
TorchBackend (see optiland/backend/torch_backend/__init__.py).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from torch import Tensor


class MiscMixin:
    """Miscellaneous operations."""

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def factorial(self, n: Any) -> Tensor:
        """Compute the factorial of n using the log-gamma function.

        Args:
            n: Non-negative integer or tensor.

        Returns:
            Tensor: Factorial values.
        """
        return torch.lgamma(self.array(n + 1)).exp()

    def path_contains_points(self, vertices: Tensor, points: Tensor) -> Tensor:
        """Return a boolean mask of points inside the polygon.

        Uses a vectorized ray-crossing algorithm.

        Args:
            vertices: Polygon vertices as (N, 2) tensor (closed implicitly).
            points: Query points as (M, 2) tensor.

        Returns:
            Tensor: Boolean mask of shape (M,).
        """
        vx, vy = vertices[:, 0], vertices[:, 1]
        px = points[:, 0].unsqueeze(1)
        py = points[:, 1].unsqueeze(1)

        vx_next = torch.roll(vx, -1)
        vy_next = torch.roll(vy, -1)

        cond = (vy > py) != (vy_next > py)
        slope = (vx_next - vx) / (vy_next - vy)
        x_int = vx + slope * (py - vy)
        cross = cond & (px < x_int)

        inside = torch.sum(cross, dim=1) % 2 == 1
        return inside

    def pad(
        self,
        tensor: Tensor,
        pad_width: Any,
        mode: str = "constant",
        constant_values: float | None = 0,
    ) -> Tensor:
        """Pad a tensor.

        Args:
            tensor: Input tensor.
            pad_width: Padding per axis.
            mode: Padding mode.
            constant_values: Value used for constant padding.

        Returns:
            Tensor: Padded tensor.
        """
        if len(pad_width) == 2:
            (pt, pb), (pl, pr) = pad_width
        elif len(pad_width) == 4:
            if pad_width[0] != (0, 0) or pad_width[1] != (0, 0):
                raise NotImplementedError(
                    "Padding batch or channel dimensions is not supported"
                )
            (pt, pb), (pl, pr) = pad_width[2:]
        else:
            raise ValueError("pad_width must have length 2 or 4")

        if mode == "constant":
            return F.pad(
                tensor, (pl, pr, pt, pb), mode="constant", value=constant_values
            )

        if mode in ("reflect", "replicate", "circular"):
            return F.pad(tensor, (pl, pr, pt, pb), mode=mode)

        raise NotImplementedError(f"Unsupported padding mode: {mode}")

    def vectorize(self, pyfunc: Callable[..., Any]) -> Callable[..., Any]:
        """Vectorize a scalar Python function over tensor inputs.

        Args:
            pyfunc: The scalar function to vectorize.

        Returns:
            Callable: Vectorized function.
        """

        def wrapped(x: Tensor) -> Tensor:
            flat = x.reshape(-1)
            mapped = [pyfunc(xi) for xi in flat]
            out = torch.stack(
                [
                    (
                        m
                        if isinstance(m, torch.Tensor)
                        else torch.tensor(m, dtype=self._dtype(), device=self._device())
                    )
                    for m in mapped
                ]
            )
            return out.view(x.shape)

        return wrapped

    @contextlib.contextmanager
    def errstate(self, **kwargs: Any) -> Generator[None, None, None]:  # type: ignore[override]
        """No-op context manager (torch has no equivalent of np.errstate).

        Args:
            **kwargs: Ignored.

        Yields:
            None
        """
        yield

    def histogram(self, x: Any, bins: Any = 10) -> tuple[Tensor, Tensor]:
        """Compute a histogram of x.

        Args:
            x: Input tensor.
            bins: Number of bins or bin edge tensor.

        Returns:
            tuple[Tensor, Tensor]: Bin counts and bin edges.
        """
        if isinstance(bins, int):
            return torch.histogram(x.float(), bins=bins)
        return torch.histogram(x.float(), bins=bins.float())

    def histogram2d(
        self,
        x: Tensor,
        y: Tensor,
        bins: Any,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute a 2-D histogram.

        Args:
            x: x-coordinates of the sample points.
            y: y-coordinates of the sample points.
            bins: List or tuple of two edge tensors.
            weights: Optional weights for each sample.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Histogram, x edges, y edges.

        Raises:
            ValueError: If bins is not a list/tuple of two edge tensors.
        """
        if not isinstance(bins, list | tuple) or len(bins) != 2:
            raise ValueError("`bins` must be a list or tuple of two edge tensors.")

        x_edges, y_edges = bins[0], bins[1]
        nx = x_edges.numel() - 1
        ny = y_edges.numel() - 1

        x_bin_indices = torch.searchsorted(x_edges, x, right=False) - 1
        y_bin_indices = torch.searchsorted(y_edges, y, right=False) - 1
        x_bin_indices = torch.clamp(x_bin_indices, 0, nx - 1)
        y_bin_indices = torch.clamp(y_bin_indices, 0, ny - 1)

        mask = (
            (x >= x_edges[0])
            & (x <= x_edges[-1])
            & (y >= y_edges[0])
            & (y <= y_edges[-1])
        )

        if weights is None:
            weights = torch.ones_like(x)

        valid_x = x_bin_indices[mask]
        valid_y = y_bin_indices[mask]
        valid_w = weights[mask]

        linear_indices = (valid_y * nx + valid_x).long()
        hist_flat = torch.zeros(nx * ny, device=x.device, dtype=valid_w.dtype)
        hist_flat.index_add_(0, linear_indices, valid_w)
        hist = hist_flat.reshape(ny, nx).T

        return hist, x_edges, y_edges
