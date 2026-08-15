"""
This module defines the OPD class.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import optiland.backend as be
from optiland.utils import resolve_wavelength

from .wavefront import Wavefront

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.typing import NDArray

    from optiland._types import DistributionType, PlotProjection
    from optiland.optic.optic import Optic
    from optiland.wavefront.strategy import WavefrontStrategyType


class OPDData(TypedDict):
    """Per-field OPD sample grid: pupil coordinates and wavefront error."""

    x: NDArray
    y: NDArray
    z: NDArray


class OPD(Wavefront):
    """Represents an Optical Path Difference (OPD) wavefront.

    Args:
        optic (Optic): The optic object.
        field (tuple): The field at which to calculate the OPD.
        wavelength (str | float): The wavelength of the wavefront. Can be 'primary'
            or a float value.
        num_rings (int, optional): The number of rings for ray tracing.
            Defaults to 15.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray", "centroid", and "best_fit".
            Defaults to "chief_ray".
        afocal (bool): If True, uses a planar reference geometry for afocal systems.
            If False, uses a spherical reference geometry. Defaults to False.
        remove_tilt (bool): If True, removes tilt and piston from the OPD data.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the strategy.

    Attributes:
        optic (Optic): The optic object.
        field (tuple[float, float]): The field coordinates (Hx, Hy).
        wavelength (float): The wavelength of the wavefront in micrometers.
        num_rays (int): The number of rays (or rings for hexapolar distribution) to use
            for pupil sampling.
        distribution (BaseDistribution): The pupil sampling distribution instance.
        data (dict): A dictionary mapping (field, wavelength) tuples to
            `WavefrontData` objects. Inherited from `Wavefront`.

    Methods:
        view(projection='2d', num_points=256, figsize=(7, 5.5)): Visualizes
            the OPD wavefront.
        rms(): Calculates the root mean square (RMS) of the OPD wavefront.

    """

    def __init__(
        self,
        optic: Optic,
        field: tuple[float, float],
        wavelength: str | float,
        num_rays: int = 15,
        distribution: DistributionType = "hexapolar",
        strategy: WavefrontStrategyType = "chief_ray",
        remove_tilt: bool = False,
        **kwargs,
    ) -> None:
        resolved_wavelength = resolve_wavelength(optic, wavelength)
        super().__init__(
            optic,
            fields=[field],
            wavelengths=[resolved_wavelength],
            num_rays=num_rays,
            distribution=distribution,
            strategy=strategy,
            remove_tilt=remove_tilt,
            **kwargs,
        )

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        projection: PlotProjection = "2d",
        num_points: int = 256,
        figsize: tuple[float, float] = (7, 5.5),
    ) -> tuple[Figure, Axes]:
        """Visualizes the OPD wavefront.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure is created.
            projection (str, optional): The projection type. Defaults to '2d'.
            num_points (int, optional): The number of points for interpolation.
                Defaults to 256.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).
        Returns:
            tuple: A tuple containing the figure and axes objects.
        Raises:
            ValueError: If the projection is not '2d' or '3d'.
        """
        is_gui_embedding = fig_to_plot_on is not None
        if is_gui_embedding:
            current_fig = cast("Figure", fig_to_plot_on)
            current_fig.clear()
            ax = (
                current_fig.add_subplot(111)
                if projection == "2d"
                else current_fig.add_subplot(111, projection="3d")
            )
        else:
            current_fig, ax = (
                plt.subplots(figsize=figsize)
                if projection == "2d"
                else plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
            )

        opd_map = self.generate_opd_map(num_points)
        if projection == "2d":
            self._plot_2d(data=opd_map, ax=ax)
        elif projection == "3d":
            self._plot_3d(fig=current_fig, ax=ax, data=opd_map)
        else:
            raise ValueError('OPD projection must be "2d" or "3d".')

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        return current_fig, ax

    def rms(self) -> be.ndarray:
        """Calculates the root mean square (RMS) of the OPD wavefront.

        Returns:
            float: The RMS value.

        """
        data = self.get_data(self.fields[0], self.wavelengths[0])
        mask = data.intensity > 0
        if not be.any(mask):
            raise ValueError(
                "No valid rays with non-zero intensity for RMS calculation."
            )
        opd = data.opd[mask]
        return be.sqrt(be.mean(opd**2))

    def _plot_2d(self, ax: Axes, data: dict[str, NDArray]) -> None:
        """Plots the 2D visualization of the OPD wavefront.

        Args:
            data (dict[str, np.ndarray]): The OPD map data, where keys are 'x', 'y', 'z'
                and values are NumPy arrays suitable for plotting.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).

        """
        im = ax.imshow(
            np.flipud(data["z"]), extent=(-1, 1, -1, 1)
        )  # np.flipud is fine here as data['z'] is already numpy

        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_title(f"OPD Map: RMS={self.rms():.3f} waves")

        cbar = plt.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("OPD (waves)", rotation=270)

    def _plot_3d(self, fig: Figure, ax: Axes3D, data: dict[str, NDArray]) -> None:
        """Plots the 3D visualization of the OPD wavefront.

        Args:
            data (dict[str, np.ndarray]): The OPD map data, where keys are 'x', 'y', 'z'
                and values are NumPy arrays suitable for plotting.
            figsize (tuple, optional): The figure size. Defaults to (7, 5.5).

        """
        surf = ax.plot_surface(
            data["x"],
            data["y"],
            data["z"],
            rstride=1,
            cstride=1,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
        )

        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_zlabel("OPD (waves)")
        ax.set_title(f"OPD Map: RMS={self.rms():.3f} waves")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.15)
        fig.tight_layout()

    def generate_opd_map(self, num_points: int = 256) -> OPDData:
        """Generates the OPD map data.

        Args:
            num_points (int, optional): The number of points for interpolation
                along each axis of the grid. Defaults to 256.

        Returns:
            dict[str, np.ndarray]: A dictionary containing the interpolated OPD map,
            with keys 'x', 'y', and 'z'. The values are NumPy arrays.

        """
        data = self.get_data(self.fields[0], self.wavelengths[0])
        x = be.to_numpy(self.distribution.x)
        y = be.to_numpy(self.distribution.y)
        z = be.to_numpy(data.opd)
        intensity = be.to_numpy(data.intensity)

        # Ignore zero intensity points
        mask = intensity > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        intensity = intensity[mask]

        x_interp, y_interp = np.meshgrid(
            np.linspace(-1, 1, num_points),
            np.linspace(-1, 1, num_points),
        )

        points = np.column_stack((x.flatten(), y.flatten()))
        values = z.flatten() * intensity.flatten()

        z_interp = griddata(points, values, (x_interp, y_interp), method="cubic")

        data = OPDData(x=x_interp, y=y_interp, z=z_interp)
        return data
