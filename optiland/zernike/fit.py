"""Zernike Fit Module

This module contains the ZernikeFit class, which can be used to fit Zernike
polynomial to a set of points. This is commonly used for wavefront calculations,
but the class can be used for any fitting operation requiring Zernike polynomials.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

from optiland import backend as be
from optiland.zernike import ZernikeFringe, ZernikeNoll, ZernikeStandard

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland._types import BEArray, ScalarOrArray

ZERNIKE_CLASSES: dict[str, type[ZernikeFringe | ZernikeStandard | ZernikeNoll]] = {
    "fringe": ZernikeFringe,
    "standard": ZernikeStandard,
    "noll": ZernikeNoll,
}


class ZernikeFit:
    """
    Fit Zernike polynomials to wavefront or arbitrary data points.

    This class constructs a linear design matrix of Zernike basis functions and solves
    for the coefficients via least squares.

    Args:
        x (array-like): X-coordinates of data points.
        y (array-like): Y-coordinates of data points.
        z (array-like): Values at (x, y) to fit.
        zernike_type (str): Type of Zernike basis: 'fringe', 'standard', or 'noll'.
        num_terms (int): Number of Zernike terms to include in the fit.

    Attributes:
        x (array-like): Flattened x-coordinates.
        y (array-like): Flattened y-coordinates.
        z (array-like): Flattened target values.
        radius (array-like): Radial coordinate of each point.
        phi (array-like): Azimuthal coordinate of each point.
        num_pts (int): Number of data points.
        zernike (BaseZernike): Zernike basis instance with fitted coefficients.
    """

    def __init__(
        self,
        x: ScalarOrArray,
        y: ScalarOrArray,
        z: ScalarOrArray,
        zernike_type: Literal["fringe", "standard", "noll"] = "fringe",
        num_terms: int = 36,
    ):
        # Convert inputs to backend tensors and flatten
        self.x = be.asarray(x).reshape(-1)
        self.y = be.asarray(y).reshape(-1)
        self.z = be.asarray(z).reshape(-1)

        if self.x.shape != self.y.shape or self.x.shape != self.z.shape:
            raise ValueError("`x`, `y`, and `z` must have the same number of elements.")

        self.num_terms = num_terms
        self.num_pts = int(be.size(self.x))

        # Compute polar coordinates
        self.radius = be.sqrt(self.x**2 + self.y**2)
        self.phi = be.arctan2(self.y, self.x)

        # Validate Zernike type and instantiate basis
        if zernike_type not in ZERNIKE_CLASSES:
            raise ValueError(
                f"Invalid Zernike type '{zernike_type}'. "
                f"Choose from: {list(ZERNIKE_CLASSES)}"
            )
        self.zernike_type = zernike_type
        self.zernike: ZernikeFringe | ZernikeStandard | ZernikeNoll = ZERNIKE_CLASSES[
            zernike_type
        ](be.ones([num_terms]))

        # Fit coefficients
        self._fit()

    @property
    def coeffs(self) -> BEArray:
        """
        Tensor: The fitted Zernike coefficients.
        """
        return self.zernike.coeffs

    def _fit(self):
        """
        Build design matrix of Zernike basis functions and solve linear least squares.
        """
        # Build design matrix A
        A = be.stack(self.zernike.terms(self.radius, self.phi), axis=1)

        # Solve linear least squares A c = z
        try:
            solution = be.linalg.lstsq(A, self.z, rcond=None)
            coeffs = solution[0]
        except (AttributeError, TypeError):
            # Fallback via pseudoinverse
            pinv = be.linalg.pinv(A)
            coeffs = be.matmul(pinv, self.z)

        # Assign coefficients to the main zernike instance
        self.zernike.coeffs = coeffs

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        projection: str = "2d",
        num_points: int = 128,
        figsize: tuple[float, float] = (7, 5.5),
        z_label: str = "OPD (waves)",
    ) -> tuple[Figure, Axes]:
        """
        Visualize the fitted Zernike surface.

        Args:
            fig_to_plot_on (plt.Figure, optional): Figure to plot on.
                If None, a new figure is created.
            projection (str): '2d' for image plot, '3d' for surface plot.
            num_points (int): Grid resolution for display.
            figsize (tuple): Figure size in inches.
            defaults to (7, 5.5).
            z_label (str): Label for the z-axis or colorbar.
            defaults to 'OPD (waves)'.
        Returns:
            tuple: A tuple containing the figure and axes objects.

        Raises:
            ValueError: If `projection` is not '2d' or '3d'.
        """
        is_gui_embedding = fig_to_plot_on is not None
        if is_gui_embedding:
            current_fig = fig_to_plot_on
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
                else plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
            )

        # Create grid in unit circle
        grid_x, grid_y = be.meshgrid(
            be.linspace(-1.0, 1.0, num_points),
            be.linspace(-1.0, 1.0, num_points),
        )
        grid_r = be.sqrt(grid_x**2 + grid_y**2)
        grid_phi = be.arctan2(grid_y, grid_x)
        grid_z = self.zernike.poly(grid_r, grid_phi)

        # Mask outside unit circle
        grid_z = be.where(grid_r > 1.0, be.nan, grid_z)

        # Convert to NumPy for plotting
        x_np = be.to_numpy(grid_x)
        y_np = be.to_numpy(grid_y)
        z_np = be.to_numpy(grid_z)

        if projection == "2d":
            self._plot_2d(current_fig, ax, z_np, z_label=z_label)
        elif projection == "3d":
            self._plot_3d(current_fig, ax, x_np, y_np, z_np, z_label=z_label)
        else:
            raise ValueError("`projection` must be '2d' or '3d'.")

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        return current_fig, ax

    def view_residual(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (7, 5.5),
        z_label: str = "Residual (waves)",
    ):
        """
        Scatter plot of residuals between fitted surface and original data.

        Args:
            fig_to_plot_on (plt.Figure, optional): Figure to plot on.
                If None, a new figure is created.
            figsize (tuple): Figure size in inches.
                Defaults to (7, 5.5).
            z_label (str): Label for the colorbar.
                Defaults to 'Residual (waves)'.

        Returns:
            tuple: A tuple containing the figure and axes objects.
        """
        # Compute fitted values and residuals
        fitted = self.zernike.poly(self.radius, self.phi)
        residuals = fitted - self.z
        rms = be.sqrt(be.mean(residuals**2))

        is_gui_embedding = fig_to_plot_on is not None
        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)

        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        sc = ax.scatter(
            be.to_numpy(self.x),
            be.to_numpy(self.y),
            c=be.to_numpy(residuals),
            marker="o",
            edgecolors="none",
        )
        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_title(f"Residuals (RMS={rms:.3f})")
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(z_label, rotation=270, labelpad=15)

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        return current_fig, ax

    def _plot_2d(self, fig: Figure, ax: Axes, z: np.ndarray, z_label: str) -> None:
        """Plot a 2D representation of the given data.

        Args:
            z (numpy.ndarray): The data to be plotted.
            figsize (tuple, optional): The size of the figure
                (default is (7, 5.5)).
            z_label (str, optional): The label for the colorbar
                (default is 'OPD (waves)').

        """
        im = ax.imshow(np.flipud(z), extent=[-1, 1, -1, 1])
        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_title(f"Zernike {self.zernike_type.capitalize()} Fit")
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel(z_label, rotation=270, labelpad=15)

    def _plot_3d(
        self,
        fig: Figure,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        z_label: str,
    ) -> None:
        """Plot a 3D surface plot of the given data.

        Args:
            fig (Figure): The figure to plot on.
            ax (Axes): The axes to plot on.
            x (numpy.ndarray): Array of x-coordinates.
            y (numpy.ndarray): Array of y-coordinates.
            z (numpy.ndarray): Array of z-coordinates.
            z_label (str, optional): Label for the z-axis.

        """
        surf = ax.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
        )
        ax.set_xlabel("Pupil X")
        ax.set_ylabel("Pupil Y")
        ax.set_zlabel(z_label)
        ax.set_title(f"Zernike {self.zernike_type.capitalize()} Fit")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.15)
        fig.tight_layout()
