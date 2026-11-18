"""Footprint Diagram Analysis

This module provides a footprint diagram analysis: where real rays land on
any surface of an optical system, in that surface's local coordinates.

Cr, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.utils import resolve_fields
from optiland.visualization.system.utils import transform

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland._types import BEArray, DistributionType
    from optiland.optic import Optic


@dataclass
class FootprintData:
    """Ray intersections with the analyzed surface, for one field and wavelength.

    Attributes:
        x: Local x-coordinates of every traced ray on the surface.
        y: Local y-coordinates of every traced ray on the surface.
        reached: True for the rays that arrive at the surface, i.e. that have
            a nonzero intensity on the previous surface.
        clipped: True for the rays that arrive at the surface but are blocked
            by it (zero intensity on the surface itself). A ray that misses the
            surface geometrically is also clipped, and its coordinates are NaN.
    """

    x: BEArray
    y: BEArray
    reached: BEArray
    clipped: BEArray


class FootprintDiagram(BaseAnalysis):
    """Footprint diagram: real ray intersections with any surface.

    For each field and wavelength, rays are traced in the same way as for
    :class:`~optiland.analysis.SpotDiagram`, and their intersections with the
    chosen surface are converted to that surface's local frame. The footprint
    shows how much of a surface each field uses, which helps to size clear
    apertures and to find where a beam is clipped.

    Args:
        optic (Optic): The optic object to analyze.
        surface_index (int): Position of the surface in ``optic.surfaces``.
            Negative values count from the end, so ``-1`` is the image
            surface. The object surface (index 0) cannot be analyzed.
        fields (str or list, optional): The fields to trace: 'all' or a list
            of (Hx, Hy) tuples. Defaults to 'all'.
        wavelengths (str or list, optional): The wavelengths to trace: 'all',
            'primary', or a list of values in µm. Defaults to 'primary'.
        num_rays (int, optional): The number of rays, or the number of rings
            for the hexapolar distribution. Defaults to 12.
        distribution (str, optional): The pupil distribution of the rays.
            Defaults to 'hexapolar'.

    Attributes:
        optic (Optic): The optic object being analyzed.
        surface_index (int): The non-negative index of the analyzed surface.
        fields (list[FieldPoint]): The fields being analyzed.
        wavelengths (list[WavelengthPoint]): The wavelengths being analyzed.
        num_rays (int): The number of rays or rings per trace.
        distribution (str): The pupil distribution of the rays.
        data (list[list[FootprintData]]): The footprint data, indexed by field
            and then by wavelength.

    Raises:
        ValueError: If ``surface_index`` is the object surface or is out of
            range.
    """

    def __init__(
        self,
        optic: Optic,
        surface_index: int,
        fields: str | list = "all",
        wavelengths: str | list = "primary",
        num_rays: int = 12,
        distribution: DistributionType = "hexapolar",
    ):
        num_surfaces = len(optic.surfaces)
        index = surface_index + num_surfaces if surface_index < 0 else surface_index
        if not 0 <= index < num_surfaces:
            raise ValueError(
                f"surface_index {surface_index} is out of range for an optic "
                f"with {num_surfaces} surfaces."
            )
        if index == 0:
            raise ValueError(
                "The object surface (index 0) has no footprint; choose a "
                "surface index of 1 or more."
            )
        self.surface_index = index
        self.fields = resolve_fields(optic, fields)
        self.num_rays = num_rays
        self.distribution: DistributionType = distribution
        super().__init__(optic, wavelengths)

    def extent(self) -> tuple[float, float, float, float]:
        """Returns the bounding box of the rays that reach the surface.

        Returns:
            tuple[float, float, float, float]: ``(xmin, xmax, ymin, ymax)`` in
            lens units, over all fields and wavelengths.

        Raises:
            ValueError: If no ray reaches the surface.
        """
        x, y = self._reached_points()
        return (
            float(be.min(x)),
            float(be.max(x)),
            float(be.min(y)),
            float(be.max(y)),
        )

    def max_radius(self) -> float:
        """Returns the largest radial distance of the rays that reach the surface.

        Returns:
            float: The largest ``hypot(x, y)`` in lens units, over all fields
            and wavelengths.

        Raises:
            ValueError: If no ray reaches the surface.
        """
        x, y = self._reached_points()
        return float(be.max(be.hypot(x, y)))

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (6, 6),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Displays the footprint diagram.

        Rays that pass the surface are drawn in one color per field. Rays that
        reach the surface but are blocked by it are drawn as grey crosses. The
        boundary of the surface's physical aperture is drawn when it has one,
        otherwise its semi-aperture circle when that is set.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (6, 6).
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            tuple: The current figure and its axes.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        clipped_x, clipped_y = [], []
        for i, (field, field_data) in enumerate(
            zip(self.fields, self.data, strict=True)
        ):
            label = f"Hx: {field.coord[0]:.3f}, Hy: {field.coord[1]:.3f}"
            for points in field_data:
                x = be.to_numpy(points.x)
                y = be.to_numpy(points.y)
                reached = be.to_numpy(points.reached)
                clipped = be.to_numpy(points.clipped)
                passed = reached & ~clipped
                if passed.any():
                    ax.scatter(
                        x[passed],
                        y[passed],
                        s=10,
                        color=f"C{i}",
                        alpha=0.7,
                        label=label,
                    )
                    label = None  # one legend entry per field
                clipped_x.append(x[clipped])
                clipped_y.append(y[clipped])

        if clipped_x and np.concatenate(clipped_x).size > 0:
            ax.scatter(
                np.concatenate(clipped_x),
                np.concatenate(clipped_y),
                s=12,
                marker="x",
                color="grey",
                linewidths=0.8,
                label="Clipped",
            )

        self._plot_aperture_outline(ax)

        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"Footprint, surface {self.surface_index}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.grid(True)
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, ax

    def _generate_data(self) -> list[list[FootprintData]]:
        """Generates footprint data for all fields and wavelengths.

        Returns:
            list[list[FootprintData]]: The footprint data, indexed by field
            and then by wavelength.
        """
        return [
            [self._generate_field_data(fp.coord, wp.value) for wp in self.wavelengths]
            for fp in self.fields
        ]

    def _generate_field_data(
        self, field: tuple[float, float], wavelength: float
    ) -> FootprintData:
        """Generates footprint data for a single field and wavelength.

        Args:
            field (tuple[float, float]): The (Hx, Hy) field coordinates.
            wavelength (float): The wavelength in µm.

        Returns:
            FootprintData: The local ray coordinates on the surface and the
            reached and clipped masks.
        """
        self.optic.trace(*field, wavelength, self.num_rays, self.distribution)
        surface = self.optic.surfaces[self.surface_index]
        previous = self.optic.surfaces[self.surface_index - 1]

        x, y, _ = transform(surface.x, surface.y, surface.z, surface, is_global=True)
        reached = previous.intensity > 0
        clipped = be.logical_and(reached, be.logical_not(surface.intensity > 0))
        return FootprintData(x=x, y=y, reached=reached, clipped=clipped)

    def _reached_points(self) -> tuple[BEArray, BEArray]:
        """Collects the local coordinates of the rays that reach the surface.

        Rays that miss the surface geometrically have no intersection point
        (NaN coordinates) and are left out.

        Returns:
            tuple: The x and y coordinates over all fields and wavelengths.

        Raises:
            ValueError: If no ray reaches the surface.
        """
        points = [p for field_data in self.data for p in field_data]
        masks = [
            be.logical_and(
                p.reached, be.logical_and(be.isfinite(p.x), be.isfinite(p.y))
            )
            for p in points
        ]
        xs = [p.x[mask] for p, mask in zip(points, masks, strict=True)]
        ys = [p.y[mask] for p, mask in zip(points, masks, strict=True)]
        if not xs or sum(be.size(x) for x in xs) == 0:
            raise ValueError(f"No ray reaches surface {self.surface_index}.")
        return be.concatenate(xs), be.concatenate(ys)

    def _plot_aperture_outline(self, ax: Axes, num_points: int = 256) -> None:
        """Draws the boundary of the surface's aperture on the axes.

        The boundary of a physical aperture is found the way
        ``BaseAperture.view`` does it, by evaluating ``contains`` on a grid
        over the aperture's extent. Without a physical aperture, the
        semi-aperture circle is drawn if it is set.

        Args:
            ax (Axes): The axes to draw on.
            num_points (int, optional): The number of grid points per axis
                used to find the boundary of a physical aperture.
                Defaults to 256.
        """
        surface = self.optic.surfaces[self.surface_index]
        if surface.aperture is not None:
            x_min, x_max, y_min, y_max = surface.aperture.extent
            pad_x = 0.05 * (x_max - x_min)
            pad_y = 0.05 * (y_max - y_min)
            X, Y = be.meshgrid(
                be.linspace(x_min - pad_x, x_max + pad_x, num_points),
                be.linspace(y_min - pad_y, y_max + pad_y, num_points),
            )
            inside = be.to_numpy(surface.aperture.contains(X, Y)).astype(float)
            ax.contour(
                be.to_numpy(X),
                be.to_numpy(Y),
                inside,
                levels=[0.5],
                colors="k",
                linewidths=1.0,
            )
        elif surface.semi_aperture is not None:
            radius = float(surface.semi_aperture)
            theta = np.linspace(0, 2 * np.pi, 361)
            ax.plot(radius * np.cos(theta), radius * np.sin(theta), "k-", linewidth=1.0)
