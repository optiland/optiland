"""
Backend-agnostic 2D grid interpolation with NumPy (spline) and Torch (grid_sample).

Gustavo Vasconcelos, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland import backend as be

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from scipy.interpolate import RectBivariateSpline
except Exception:
    RectBivariateSpline = None


class GridInterpolator:
    """Backend-agnostic interpolator for a 2D grid.

    Provides:
      - height(x, y) -> values sampled from the grid
      - gradient(x, y) -> tuple(dx, dy) of partial derivatives

    x_coords and y_coords are 1D arrays describing the grid axis ordering:
    grid should have shape (len(y_coords), len(x_coords)).
    """

    def __init__(self, x_coords: be.Array, y_coords: be.Array, grid: be.Array):
        self.backend = be.get_backend()
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.grid = grid

        setup = {"numpy": self._setup_numpy, "torch": self._setup_torch}.get(
            self.backend
        )
        if setup is None:
            raise ValueError(f"Unsupported backend: {self.backend}")

        setup()

    def _setup_numpy(self) -> None:
        if RectBivariateSpline is None:
            raise ImportError("scipy is required for numpy interpolator")

        x_np = be.to_numpy(self.x_coords)
        y_np = be.to_numpy(self.y_coords)
        grid_np = be.to_numpy(self.grid)
        self._spline = RectBivariateSpline(y_np, x_np, grid_np)

        def height_fn(x: be.Array, y: be.Array) -> be.Array:
            out = self._spline.ev(be.to_numpy(y), be.to_numpy(x))
            return be.array(out)

        def grad_fn(x: be.Array, y: be.Array) -> tuple[be.Array, be.Array]:
            x_np = be.to_numpy(x)
            y_np = be.to_numpy(y)
            dh_dx = self._spline.ev(y_np, x_np, dy=1)
            dh_dy = self._spline.ev(y_np, x_np, dx=1)
            return be.array(dh_dx), be.array(dh_dy)

        self._height_fn: Callable[[be.Array, be.Array], be.Array] = height_fn
        self._grad_fn: Callable[[be.Array, be.Array], tuple[be.Array, be.Array]] = (
            grad_fn
        )

    def _setup_torch(self) -> None:
        self._xmin = self.x_coords[0]
        self._xmax = self.x_coords[-1]
        self._ymin = self.y_coords[0]
        self._ymax = self.y_coords[-1]

        g = self.grid
        if g.ndim == 2:
            g = g[None, None, ...]
        self._grid_4d = g

        self._dx = self.x_coords[1] - self.x_coords[0]
        self._dy = self.y_coords[1] - self.y_coords[0]

        def _normalize_and_grid(x_flat: be.Array, y_flat: be.Array) -> be.Array:
            x_norm = 2 * (x_flat - self._xmin) / (self._xmax - self._xmin) - 1
            y_norm = 2 * (y_flat - self._ymin) / (self._ymax - self._ymin) - 1
            return be.stack([x_norm, y_norm], axis=-1).reshape(1, -1, 1, 2)

        def height_fn(x: be.Array, y: be.Array) -> be.Array:
            original_shape = x.shape
            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)
            grid = _normalize_and_grid(x_flat, y_flat)
            out = be.grid_sample(
                self._grid_4d, grid, mode="bilinear", align_corners=True
            )
            return out.reshape(-1).reshape(original_shape)

        def grad_fn(x: be.Array, y: be.Array):
            x = be.array(x)
            y = be.array(y)

            x_req = x.detach().clone().requires_grad_(True)
            y_req = y.detach().clone().requires_grad_(True)

            phi = height_fn(x_req, y_req)

            grad_x, grad_y = be.autograd.grad(
                phi,
                (x_req, y_req),
                grad_outputs=be.ones_like(phi),
                create_graph=True,
                retain_graph=True,
            )

            return grad_x, grad_y

        self._height_fn = height_fn
        self._grad_fn = grad_fn

    def height(self, x: be.Array, y: be.Array) -> be.Array:
        return self._height_fn(x, y)

    def gradient(self, x: be.Array, y: be.Array) -> tuple[be.Array, be.Array]:
        return self._grad_fn(x, y)
