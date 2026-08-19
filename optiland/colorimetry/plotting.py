"""
Colorimetry plotting utilities.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from scipy.interpolate import interp1d

import optiland.backend as be

from .constants import CIE_1931_2DEG, WAVELENGTHS_STD


def _to_numpy_safe(x):
    try:
        return be.to_numpy(x)
    except Exception:
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            return x.detach().cpu().numpy()
        return x


def _to_float_safe(x) -> float:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu()
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def plot_cie_1931_chromaticity_diagram(
    ax: plt.Axes | None = None,
    title: str = "CIE 1931 Chromaticity Diagram",
    color: Literal["no", "contour", "fill"] = "contour",
    show_legend: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the CIE 1931 chromaticity diagram (spectral locus).

    Generates the spectral locus by converting the standard observer CMFs
    to xy coordinates for each wavelength.

    Args:
        ax: Optional matplotlib Axes. If None, a new figure is created.
        title: Title of the plot.
        color: Coloring mode.
               - "no": Black and white outline.
               - "contour": Spectral locus colored by wavelength (default).
               - "fill": Gamut filled with approximate sRGB colors.
        show_legend: Whether to display the legend.

    Returns:
        Tuple (Figure, Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # 1. Upsample Data for smoothness
    # Extract raw data
    wls_raw = be.asarray(WAVELENGTHS_STD)
    cmfs_raw = be.asarray(CIE_1931_2DEG)
    x_bar_raw, y_bar_raw, z_bar_raw = cmfs_raw[:, 0], cmfs_raw[:, 1], cmfs_raw[:, 2]

    # Create fine wavelength grid (0.5nm steps) for smooth locus
    wl_fine = be.linspace(390, 700, 621)

    # Interpolate CMFs (Cubic)
    # We use scipy for cubic interpolation as requested
    def cubic_interp(x, y, x_new):
        f = interp1d(
            _to_numpy_safe(x),
            _to_numpy_safe(y),
            kind="cubic",
            fill_value="extrapolate",
        )
        return be.asarray(f(_to_numpy_safe(x_new)))

    x_bar = cubic_interp(wls_raw, x_bar_raw, wl_fine)
    y_bar = cubic_interp(wls_raw, y_bar_raw, wl_fine)
    z_bar = cubic_interp(wls_raw, z_bar_raw, wl_fine)

    # Calculate xy coordinates for the spectral locus
    sum_xyz = x_bar + y_bar + z_bar
    # Avoid division by zero
    mask_valid = sum_xyz > 0

    x_locus = x_bar[mask_valid] / sum_xyz[mask_valid]
    y_locus = y_bar[mask_valid] / sum_xyz[mask_valid]

    # Calculate RGB colors for the locus (normalized 0-1)
    # We use a local float conversion to avoid integer rounding artifacts
    def _get_rgb_float(X, Y, Z):
        # Normalize
        Xn, Yn, Zn = X / 100.0, Y / 100.0, Z / 100.0
        # sRGB Matrix
        r_l = 3.2404542 * Xn - 1.5371385 * Yn - 0.4985314 * Zn
        g_l = -0.9692660 * Xn + 1.8760108 * Yn + 0.0415560 * Zn
        b_l = 0.0556434 * Xn - 0.2040259 * Yn + 1.0572252 * Zn

        # Gamma
        def gamma(v):
            v_safe = be.where(v < 0.0, 0.0, v)
            return be.where(
                v <= 0.0031308,
                12.92 * v,
                1.055 * be.power(v_safe, 1.0 / 2.4) - 0.055,
            )

        # Clip 0-1
        def clip(v):
            return be.where(v < 0.0, 0.0, be.where(v > 1.0, 1.0, v))

        return (
            clip(gamma(r_l)),
            clip(gamma(g_l)),
            clip(gamma(b_l)),
        )

    # Colors for the spectral locus
    # Use fixed luminance Y=50 to ensure visibility across the spectrum
    # (Raw CMFs would result in black ends)
    Y_const = 50.0
    y_locus_safe = be.where(y_locus <= 0, 1e-6, y_locus)
    X_loc = (x_locus / y_locus_safe) * Y_const
    Z_loc = ((1 - x_locus - y_locus) / y_locus_safe) * Y_const

    r_loc, g_loc, b_loc = _get_rgb_float(X_loc, be.ones_like(X_loc) * Y_const, Z_loc)
    rgb_locus = be.transpose(be.stack([r_loc, g_loc, b_loc], axis=0))

    # Polygon for filling/masking (closed loop)
    x_poly = be.concatenate((x_locus, be.atleast_1d(x_locus[0])))
    y_poly = be.concatenate((y_locus, be.atleast_1d(y_locus[0])))

    # 2. Plotting based on color mode
    if color == "fill":
        # Generate grid for filling
        res = 500  # Increased resolution
        x_min, x_max = -0.05, 0.85
        y_min, y_max = -0.05, 0.9
        x_grid = be.linspace(x_min, x_max, res)
        y_grid = be.linspace(y_min, y_max, res)
        X_grid, Y_grid = be.meshgrid(x_grid, y_grid)

        xf = be._lib.reshape(X_grid, (-1,))
        yf = be._lib.reshape(Y_grid, (-1,))

        # Mask points inside locus
        verts = _to_numpy_safe(be.transpose(be.stack([x_poly, y_poly], axis=0)))
        path = Path(verts)
        points = _to_numpy_safe(be.transpose(be.stack([xf, yf], axis=0)))
        mask = path.contains_points(points)

        # Compute colors (Y=50 for reasonable brightness)
        Y_val = 50.0
        yf_safe = be.where(yf <= 0, 1e-6, yf)
        X_C = (xf / yf_safe) * Y_val
        Z_C = ((1 - xf - yf) / yf_safe) * Y_val

        # Use float conversion for smooth gradients (no int rounding)
        R, G, B = _get_rgb_float(X_C, be.ones_like(X_C) * Y_val, Z_C)

        # Create RGBA image
        alpha = be.asarray(mask, dtype=float)
        img_flat = be.transpose(be.stack([R, G, B, alpha], axis=0))
        img = be.reshape(img_flat, (res, res, 4))

        ax.imshow(
            _to_numpy_safe(img),
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            interpolation="bilinear",
        )
        ax.plot(_to_numpy_safe(x_locus), _to_numpy_safe(y_locus), "k-", linewidth=1.5)
        # Purple line (black dashed for fill mode to define boundary)
        ax.plot(
            [_to_float_safe(x_locus[-1]), _to_float_safe(x_locus[0])],
            [_to_float_safe(y_locus[-1]), _to_float_safe(y_locus[0])],
            "k--",
            linewidth=1,
        )

    elif color == "contour":
        # 1. Spectral Locus (Colored)
        points = _to_numpy_safe(
            be.reshape(be.transpose(be.stack([x_locus, y_locus], axis=0)), (-1, 1, 2))
        )
        segments = np.concatenate((points[:-1], points[1:]), axis=1)

        lc = LineCollection(
            segments, colors=_to_numpy_safe(rgb_locus[:-1]), linewidth=2
        )
        ax.add_collection(lc)

        # 2. Purple Line (Colored Gradient)
        # Interpolate between Red end and Blue end
        n_purple = 50
        x_p = be.linspace(x_locus[-1], x_locus[0], n_purple)
        y_p = be.linspace(y_locus[-1], y_locus[0], n_purple)

        # Interpolate colors using xy and fixed luminance
        y_p_safe = be.where(y_p <= 0, 1e-6, y_p)
        X_p = (x_p / y_p_safe) * Y_const
        Z_p = ((1 - x_p - y_p) / y_p_safe) * Y_const

        r_p, g_p, b_p = _get_rgb_float(X_p, be.ones_like(X_p) * Y_const, Z_p)
        rgb_purple = be.transpose(be.stack([r_p, g_p, b_p], axis=0))

        points_p = _to_numpy_safe(
            be.reshape(be.transpose(be.stack([x_p, y_p], axis=0)), (-1, 1, 2))
        )
        segments_p = np.concatenate((points_p[:-1], points_p[1:]), axis=1)

        lc_p = LineCollection(
            segments_p,
            colors=_to_numpy_safe(rgb_purple[:-1]),
            linewidth=2,
            linestyle="--",
        )
        ax.add_collection(lc_p)

        # Dummy line for legend
        if show_legend:
            ax.plot([], [], "-", color="gray", linewidth=2, label="Spectral Locus")
            ax.plot([], [], "--", color="gray", linewidth=2, label="Purple Line")

        # Light fill
        ax.fill(_to_numpy_safe(x_poly), _to_numpy_safe(y_poly), "k", alpha=0.05)

    else:  # "no"
        ax.plot(
            _to_numpy_safe(x_locus),
            _to_numpy_safe(y_locus),
            "k-",
            linewidth=1.5,
            label="Spectral Locus",
        )
        ax.plot(
            [_to_float_safe(x_locus[-1]), _to_float_safe(x_locus[0])],
            [_to_float_safe(y_locus[-1]), _to_float_safe(y_locus[0])],
            "k--",
            linewidth=1,
            label="Purple Line",
        )
        ax.fill(_to_numpy_safe(x_poly), _to_numpy_safe(y_poly), "k", alpha=0.05)

    # 3. Add Wavelength Markers
    target_wls = [450, 480, 520, 560, 600, 620, 700]

    for wl in target_wls:
        # Find closest index in fine grid
        idx = int(be.argmin(be.abs(wl_fine - wl)))

        # Check if we are close enough (within 1nm)
        if abs(wl_fine[idx] - wl) < 1.0:
            x, y = _to_float_safe(x_locus[idx]), _to_float_safe(y_locus[idx])
            ax.plot(x, y, "ko", markersize=3)

            # Offset text slightly towards outside
            dx = 0.02 if x > 0.3 else -0.04
            dy = 0.02 if y > 0.3 else -0.02
            ax.text(x + dx, y + dy, f"{wl}", fontsize=8, ha="center")

    # Setup axes
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-0.05, 0.9)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    if show_legend:
        ax.legend()

    return fig, ax
