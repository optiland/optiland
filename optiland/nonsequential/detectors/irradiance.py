"""Irradiance detector for Non-Sequential Raytracing.

Accumulates a 2D flux map on a planar rectangular surface with
differentiable splatting support.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential._utils import as_float, as_param
from optiland.nonsequential.components.base import _get_transform
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.detectors.base import BaseDetector
from optiland.nonsequential.results.irradiance_map import IrradianceMap

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.ray_bundle import NSQRayBundle


class IrradianceDetector(BaseDetector):
    """2D irradiance map detector on a planar rectangular surface.

    Records flux in a pixel grid with differentiable splatting.

    Attributes:
        cs: Coordinate system.
        width: Detector width [mm].
        height: Detector height [mm].
        num_pixels_x: Number of pixels along x.
        num_pixels_y: Number of pixels along y.
        splat: Splatting mode — 'bilinear', 'gaussian', or 'hard'.
        splat_sigma: Gaussian splat sigma in pixels (used when splat='gaussian').
    """

    def __init__(
        self,
        cs: CoordinateSystem,
        width: float,
        height: float,
        num_pixels_x: int,
        num_pixels_y: int,
        splat: Literal["bilinear", "gaussian", "hard"] = "bilinear",
        splat_sigma: float = 0.5,
        name: str = "",
        absorb: bool = True,
    ) -> None:
        """Initialize IrradianceDetector.

        Args:
            cs: Coordinate system for detector position/orientation.
            width: Detector width [mm].
            height: Detector height [mm].
            num_pixels_x: Number of pixels along x.
            num_pixels_y: Number of pixels along y.
            splat: Splatting mode. 'bilinear' (differentiable, default),
                'hard' (forward-only histogram), or 'gaussian' (a true
                Gaussian kernel of width ``splat_sigma``, truncated and
                renormalised -- see :meth:`_record_gaussian`).
            splat_sigma: Gaussian sigma in pixels. Only used when
                ``splat='gaussian'``.
            name: Optional label.
            absorb: Whether a hit terminates the ray (default True).
        """
        geometry = FinitePlaneGeometry(width=width, height=height)
        super().__init__(cs, geometry, name=name, absorb=absorb)
        self.width = as_param(width)
        self.height = as_param(height)
        self.num_pixels_x = int(num_pixels_x)
        self.num_pixels_y = int(num_pixels_y)
        self.splat = splat
        self.splat_sigma = as_float(splat_sigma)

        # Flat accumulation buffer: shape (ny * nx,)
        self._data = be.zeros(num_pixels_y * num_pixels_x)
        self._num_rays_hit: int = 0

        # Pixel bin edges (NumPy, used for index arithmetic -- always detached)
        w_f = as_float(width)
        h_f = as_float(height)
        self._x_edges = np.linspace(-w_f / 2.0, w_f / 2.0, num_pixels_x + 1)
        self._y_edges = np.linspace(-h_f / 2.0, h_f / 2.0, num_pixels_y + 1)

    def record(self, rays: NSQRayBundle, t: np.ndarray, hit_mask: np.ndarray) -> None:
        """Accumulate flux from hit rays into the pixel grid.

        Computes hit positions in local detector frame and accumulates
        flux using the configured splatting mode.

        Args:
            rays: Current ray bundle (positions not yet advanced to hit
                point).
            t: Hit distances [mm], shape (N,).
            hit_mask: Boolean mask of hitting rays, shape (N,).
        """
        hit_mask_np = to_numpy(hit_mask).astype(bool)
        if not hit_mask_np.any():
            return

        translation, rot = _get_transform(self.cs)

        # Advance hit rays to intersection point in backend-attached arrays
        t_hit_be = be.where(hit_mask, t, be.zeros_like(t))
        hx_g = rays.x + t_hit_be * rays.L
        hy_g = rays.y + t_hit_be * rays.M
        hz_g = rays.z + t_hit_be * rays.N

        t_arr = be.array(translation)
        R_arr = be.array(rot)
        pos_g = be.stack([hx_g, hy_g, hz_g], axis=1)
        pos_l = (pos_g - t_arr) @ R_arr
        hx_l = pos_l[:, 0]
        hy_l = pos_l[:, 1]

        # Zero out non-hit ray contributions while keeping graph attached
        hit_mask_be = be.array(hit_mask_np)
        flux_masked = be.where(hit_mask_be, rays.flux, be.zeros_like(rays.flux))

        nx = self.num_pixels_x
        ny = self.num_pixels_y
        dx = self.width / nx
        dy = self.height / ny

        if self.splat == "hard":
            self._record_hard(hx_l, hy_l, flux_masked, hit_mask_np, nx, ny)
        elif self.splat == "gaussian":
            self._record_gaussian(hx_l, hy_l, flux_masked, nx, ny, dx, dy)
        else:
            self._record_bilinear(hx_l, hy_l, flux_masked, hit_mask_np, nx, ny, dx, dy)

        self._num_rays_hit += int(hit_mask_np.sum())

    def _record_hard(
        self,
        hx_l,
        hy_l,
        flux_masked,
        hit_mask_np: np.ndarray,
        nx: int,
        ny: int,
    ) -> None:
        """Hard-bin accumulation (forward-only, NumPy path).

        Args:
            hx_l: Local x coordinates, be-array shape (N,).
            hy_l: Local y coordinates, be-array shape (N,).
            flux_masked: Per-ray flux (non-hit rays zeroed), be-array.
            hit_mask_np: Boolean NumPy mask, shape (N,).
            nx: Number of pixels along x.
            ny: Number of pixels along y.
        """
        hx_np = to_numpy(hx_l)
        hy_np = to_numpy(hy_l)
        flux_np = to_numpy(flux_masked)
        idx = np.where(hit_mask_np)[0]

        ix = np.clip(
            np.searchsorted(self._x_edges, hx_np[idx], side="right") - 1,
            0,
            nx - 1,
        )
        iy = np.clip(
            np.searchsorted(self._y_edges, hy_np[idx], side="right") - 1,
            0,
            ny - 1,
        )
        flat = (iy * nx + ix).astype(np.int64)
        data_np = to_numpy(self._data).copy()
        np.add.at(data_np, flat, flux_np[idx])
        self._data = be.array(data_np)

    def _record_bilinear(
        self,
        hx_l,
        hy_l,
        flux_masked,
        hit_mask_np: np.ndarray,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
    ) -> None:
        """Bilinear splat — differentiable w.r.t. landing position and flux.

        Distributes each ray's flux to the four surrounding pixel centres
        with bilinear weights. Index arithmetic uses detached NumPy arrays;
        the flux contribution (flux * weight) carries gradients.

        Args:
            hx_l: Local x coordinates, be-array shape (N,).
            hy_l: Local y coordinates, be-array shape (N,).
            flux_masked: Per-ray flux (non-hit rays zeroed), be-array.
            hit_mask_np: Boolean NumPy mask, shape (N,).
            nx: Number of pixels along x.
            ny: Number of pixels along y.
            dx: Pixel width [mm].
            dy: Pixel height [mm].
        """
        # Continuous pixel coordinate — centre of pixel ix is at 0.0 when
        # ix == 0, i.e.  px = (hx_l + width/2) / dx - 0.5
        px = (hx_l + self.width / 2.0) / dx - 0.5
        py = (hy_l + self.height / 2.0) / dy - 0.5

        # Base pixel index (detached — index must not carry gradient)
        px_np = to_numpy(px)
        py_np = to_numpy(py)
        ix0_np = np.floor(px_np).astype(np.int64)
        iy0_np = np.floor(py_np).astype(np.int64)

        # Fractional weights (attached to graph via be-arrays)
        wx1 = px - be.array(ix0_np.astype(np.float64))  # fraction toward ix+1
        wy1 = py - be.array(iy0_np.astype(np.float64))
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        # Distribute flux to all four neighbour pixels.
        # index must be an integer array — pass flat_np directly (not via
        # be.array which casts to float) so that numpy uses int indexing and
        # torch receives a LongTensor.
        for dix, diy, wx, wy in (
            (0, 0, wx0, wy0),
            (1, 0, wx1, wy0),
            (0, 1, wx0, wy1),
            (1, 1, wx1, wy1),
        ):
            ix_np = np.clip(ix0_np + dix, 0, nx - 1)
            iy_np = np.clip(iy0_np + diy, 0, ny - 1)
            flat_np = (iy_np * nx + ix_np).astype(np.int64)

            contrib = flux_masked * wx * wy  # attached
            self._data = be.index_add(self._data, 0, self._flat_index(flat_np), contrib)

    def _record_gaussian(
        self,
        hx_l,
        hy_l,
        flux_masked,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
    ) -> None:
        """Gaussian splat — differentiable, energy-conserving.

        Distributes each ray's flux over a ``(2*radius+1) x (2*radius+1)``
        neighbourhood using a separable Gaussian kernel of width
        ``self.splat_sigma`` pixels, truncated at ``radius = ceil(3 *
        splat_sigma)`` pixels. The truncated kernel is renormalised per ray
        (weights sum to 1 over exactly the pixels actually touched), so
        truncating the tail never loses energy -- an untruncated Gaussian
        would only asymptotically conserve flux, and a non-renormalised
        truncation would be a new flux-truncation bias of exactly the kind
        D-9 exists to avoid.

        Args:
            hx_l: Local x coordinates, be-array shape (N,).
            hy_l: Local y coordinates, be-array shape (N,).
            flux_masked: Per-ray flux (non-hit rays zeroed), be-array.
            nx: Number of pixels along x.
            ny: Number of pixels along y.
            dx: Pixel width [mm].
            dy: Pixel height [mm].
        """
        sigma = self.splat_sigma
        if sigma <= 0.0:
            hit_mask_np = to_numpy(flux_masked) != 0.0
            self._record_hard(hx_l, hy_l, flux_masked, hit_mask_np, nx, ny)
            return

        radius = max(1, int(np.ceil(3.0 * sigma)))

        px = (hx_l + self.width / 2.0) / dx - 0.5
        py = (hy_l + self.height / 2.0) / dy - 0.5
        px_np = to_numpy(px)
        py_np = to_numpy(py)
        ix0_np = np.floor(px_np).astype(np.int64)
        iy0_np = np.floor(py_np).astype(np.int64)

        offsets = range(-radius, radius + 1)
        gx: dict[int, object] = {}
        gy: dict[int, object] = {}
        sx = be.zeros_like(px)
        sy = be.zeros_like(py)
        for d in offsets:
            ddx = be.array((ix0_np + d).astype(np.float64)) - px
            wx = be.exp(-0.5 * (ddx / sigma) ** 2)
            gx[d] = wx
            sx = sx + wx

            ddy = be.array((iy0_np + d).astype(np.float64)) - py
            wy = be.exp(-0.5 * (ddy / sigma) ** 2)
            gy[d] = wy
            sy = sy + wy

        norm = sx * sy  # separable kernel: total weight = Sx * Sy
        for dix in offsets:
            ix_np = np.clip(ix0_np + dix, 0, nx - 1)
            for diy in offsets:
                iy_np = np.clip(iy0_np + diy, 0, ny - 1)
                flat_np = (iy_np * nx + ix_np).astype(np.int64)
                weight = (gx[dix] * gy[diy]) / norm
                contrib = flux_masked * weight
                self._data = be.index_add(
                    self._data, 0, self._flat_index(flat_np), contrib
                )

    def _flat_index(self, flat_np: np.ndarray):
        """Convert a flat NumPy pixel-index array to the active backend's format.

        Args:
            flat_np: Flat pixel indices, shape (N,), int64.

        Returns:
            ``flat_np`` unchanged for NumPy; a ``LongTensor`` on the same
            device as ``self._data`` for Torch.
        """
        try:
            import torch  # noqa: PLC0415

            if isinstance(self._data, torch.Tensor):
                return torch.from_numpy(flat_np).to(
                    device=self._data.device, dtype=torch.long
                )
        except ImportError:
            pass
        return flat_np

    def get_result(self) -> IrradianceMap:
        """Return the accumulated irradiance map.

        Returns:
            IrradianceMap with irradiance [W/mm^2] computed from stored flux.
            The ``data`` attribute of the returned map is the attached flat
            flux buffer that supports gradient computation.
        """
        nx = self.num_pixels_x
        ny = self.num_pixels_y
        pixel_area = (self.width / nx) * (self.height / ny)
        data_2d = self._data.reshape(ny, nx) / pixel_area

        x_centres = 0.5 * (self._x_edges[:-1] + self._x_edges[1:])
        y_centres = 0.5 * (self._y_edges[:-1] + self._y_edges[1:])

        return IrradianceMap(
            data=self._data,
            irradiance=to_numpy(data_2d),
            x_coords=x_centres,
            y_coords=y_centres,
            # Attached: be.sum keeps this on the autograd graph, so
            # `result.detectors["D1"].total_flux.backward()` carries a
            # gradient. Use `.total_flux_float` for printing/formatting.
            total_flux=be.sum(self._data),
            num_rays_hit=self._num_rays_hit,
        )

    def reset(self) -> None:
        """Clear accumulated data.

        Re-initialises the internal buffer to a fresh ``be.zeros`` array,
        disconnecting it from the previous trace's computation graph.
        """
        self._data = be.zeros(self.num_pixels_y * self.num_pixels_x)
        self._num_rays_hit = 0
