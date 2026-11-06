"""Color Scale Module

This module contains the ColorScale class, which maps scalar values to colors and
produces the matching 2D colorbar (matplotlib) or 3D scalar bar (VTK).

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

RGBA = tuple[float, float, float, float]

_VTK_TABLE_SIZE = 256


@dataclass(frozen=True)
class ColorScale:
    """Maps scalar values to colors through a colormap.

    Limits that are left as ``None`` are filled from the data by
    :meth:`resolved`. Limits set by the user are always kept.

    Args:
        cmap (str | Colormap): Matplotlib colormap or its name.
            Defaults to ``"viridis"``.
        vmin (float | None): Value mapped to the low end of the colormap.
        vmax (float | None): Value mapped to the high end of the colormap.
        log (bool): If True, values are mapped on a base-10 logarithmic axis.
        label (str): Colorbar / scalar bar title.
    """

    cmap: str | Colormap = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    log: bool = False
    label: str = ""

    def resolved(self, values: ArrayLike) -> ColorScale:
        """Return a copy whose missing limits are filled from ``values``.

        Args:
            values (ArrayLike): Scalar values the scale will be applied to.

        Returns:
            ColorScale: A scale with both limits set.

        Raises:
            ValueError: If ``values`` has no finite entry, or if ``log`` is
                set and any finite value or user limit is not positive.
        """
        finite = self._finite(values)
        if finite.size == 0:
            raise ValueError("ColorScale: no finite values to scale")
        vmin = finite.min() if self.vmin is None else self.vmin
        vmax = finite.max() if self.vmax is None else self.vmax
        if self.log and (finite.min() <= 0 or vmin <= 0 or vmax <= 0):
            raise ValueError("ColorScale: log scale requires positive values")
        return replace(self, vmin=float(vmin), vmax=float(vmax))

    def to_rgba(self, values: ArrayLike) -> np.ndarray:
        """Map values to RGBA rows.

        Args:
            values (ArrayLike): Scalar values; any shape is flattened.

        Returns:
            np.ndarray: Array of shape ``(N, 4)``. Entries that are not finite
            (or, for a log scale, not positive) give a row of NaN, which
            callers treat as "use the default color".

        Raises:
            ValueError: If the scale is not resolved.
        """
        self._require_limits()
        flat = np.asarray(values, dtype=float).ravel()
        usable = np.isfinite(flat)
        if self.log:
            usable &= flat > 0
        rgba = np.full((flat.size, 4), np.nan)
        rgba[usable] = self._colormap()(self._fraction(flat[usable]))
        return rgba

    def add_colorbar(self, fig: Figure, ax: Axes) -> Colorbar:
        """Add a colorbar for this scale to a matplotlib figure.

        Args:
            fig (Figure): Figure that receives the colorbar.
            ax (Axes): Axes the colorbar steals space from.

        Returns:
            Colorbar: The new colorbar.

        Raises:
            ValueError: If the scale is not resolved.
        """
        self._require_limits()
        mappable = cm.ScalarMappable(norm=self._norm(), cmap=self._colormap())
        return fig.colorbar(mappable, ax=ax, label=self.label)

    def scalar_bar_actor(self) -> vtkScalarBarActor:  # noqa: F821
        """Build a VTK scalar bar for this scale.

        Returns:
            vtkScalarBarActor: Actor titled with :attr:`label`. Position and size
            are left to the caller.

        Raises:
            ValueError: If the scale is not resolved.
        """
        import vtk

        self._require_limits()
        table = vtk.vtkLookupTable()
        table.SetNumberOfTableValues(_VTK_TABLE_SIZE)
        # Table entries are evenly spaced; VTK's log mapping makes that
        # log-spaced in value, matching the 2D log norm.
        if self.log:
            table.SetScaleToLog10()
        table.SetTableRange(self.vmin, self.vmax)
        samples = self._colormap()(np.linspace(0.0, 1.0, _VTK_TABLE_SIZE))
        for index, color in enumerate(samples):
            table.SetTableValue(index, *color)
        table.Build()

        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(table)
        bar.SetTitle(self.label)
        return bar

    def _require_limits(self) -> None:
        if self.vmin is None or self.vmax is None:
            raise ValueError(
                "ColorScale: limits are unset; call resolved(values) first"
            )

    def _colormap(self) -> Colormap:
        return mpl.colormaps[self.cmap] if isinstance(self.cmap, str) else self.cmap

    def _fraction(self, values: np.ndarray) -> np.ndarray:
        """Position of each value in [0, 1] along the colormap."""
        if self.vmin == self.vmax:
            return np.full(values.shape, 0.5)
        if self.log:
            values, low, high = (
                np.log10(values),
                np.log10(self.vmin),
                np.log10(self.vmax),
            )
        else:
            low, high = self.vmin, self.vmax
        return np.clip((values - low) / (high - low), 0.0, 1.0)

    def _norm(self) -> Normalize:
        vmin, vmax = self.vmin, self.vmax
        if vmin == vmax:  # matplotlib cannot draw an empty colorbar range
            vmin, vmax = (vmin / 2, vmax * 2) if self.log else (vmin - 0.5, vmax + 0.5)
        return LogNorm(vmin, vmax) if self.log else Normalize(vmin, vmax)

    @staticmethod
    def _finite(values: ArrayLike) -> np.ndarray:
        flat = np.asarray(values, dtype=float).ravel()
        return flat[np.isfinite(flat)]


class _HasScale(Protocol):
    scale: ColorScale | None


def collect_scales(*colorizers: _HasScale | None) -> list[ColorScale]:
    """Gather the resolved scales to display from colorizers, skipping absent ones.

    Args:
        *colorizers: Component and/or ray colorizers, or ``None``.

    Returns:
        list[ColorScale]: The scales of the colorizers that expose one, in order.
    """
    return [c.scale for c in colorizers if c is not None and c.scale is not None]
