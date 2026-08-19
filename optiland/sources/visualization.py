"""Sources Visualization Module

This module provides visualization tools for extended sources, allowing users
to validate and visualize their source definitions before running full
optical system traces.

The SourceViewer class creates a 3-panel plot showing:
1. XY spatial distribution with intensity color-coding
2. XZ ray propagation paths
3. YZ ray propagation paths

This helps users verify the spatial and angular properties of their sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.visualization.base import BaseViewer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.sources.base import BaseSource


class SourceViewer(BaseViewer):
    """A class used to visualize extended sources.

    This viewer creates a comprehensive 3-panel visualization of an extended
    source showing the spatial distribution and propagation characteristics
    of the generated rays.

    Args:
        source (BaseSource): The extended source to be visualized.

    Attributes:
        source (BaseSource): The extended source being visualized.

    Methods:
        view(num_rays, propagation_distance, figsize): Creates the source visualization.

    """

    def __init__(self, source: BaseSource):
        """Initialize the SourceViewer with a source.

        Args:
            source (BaseSource): The extended source to visualize.
        """
        self.source = source

    def view(
        self,
        num_rays: int = 5000,
        propagation_distance: float = 0.1,
        figsize: tuple[float, float] = (20, 8),
        cross_spatial: tuple[str, str] = ("x", "y"),
        cross_angular: tuple[str, str] = ("L", "M"),
    ) -> tuple[Figure, list[Axes]]:
        """Create a comprehensive visualization of the extended source.

        This method generates a multi-panel plot showing:
        1. Column 1: XY spatial distribution and angular distribution
        2. Column 2: Cross-sections (spatial and angular)
        3. Column 3: XZ and YZ ray propagation paths

        Args:
            num_rays (int, optional): Number of rays to generate for visualization.
                Defaults to 5000.
            propagation_distance (float, optional): Distance in mm to propagate rays
                for path visualization. Defaults to 0.1.
            figsize (tuple[float, float], optional): Figure size (width, height)
                in inches. Defaults to (18, 8).
            cross_spatial (tuple[str, str], optional): Spatial cross-section axes.
                Defaults to ("x", "y").
            cross_angular (tuple[str, str], optional): Angular cross-section axes.
                Defaults to ("L", "M").

        Returns:
            tuple[Figure, list[Axes]]: Matplotlib figure and list of 6 axes objects.

        """
        # Generate rays from the source
        rays = self.source.generate_rays(num_rays)

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        x = be.to_numpy(rays.x)
        y = be.to_numpy(rays.y)
        z = be.to_numpy(rays.z)
        L = be.to_numpy(rays.L)
        M = be.to_numpy(rays.M)
        N = be.to_numpy(rays.N)

        # --- Calculate Theoretical Radiance for Visualization ---
        # This value is for color-coding only and is separate from rays.i (power)
        from optiland.sources.smf import SMFSource

        if isinstance(self.source, SMFSource):
            # SMFSource has both spatial and angular Gaussian profiles
            w_spatial = self.source.sigma_spatial_mm * 2  # 1/e² radius in mm
            w_angular = self.source.sigma_angular_rad * 2  # 1/e² half-angle

            term_x = -2.0 * (rays.x / w_spatial) ** 2
            term_y = -2.0 * (rays.y / w_spatial) ** 2
            term_L = -2.0 * (rays.L / w_angular) ** 2
            term_M = -2.0 * (rays.M / w_angular) ** 2
            radiance = be.exp(term_x + term_y + term_L + term_M)
        else:
            # Default for unknown source types: use ray power
            radiance = rays.i

        # Convert to numpy and normalize for plotting
        radiance_np = be.to_numpy(radiance)
        radiance_norm = (
            radiance_np / np.max(radiance_np)
            if np.max(radiance_np) > 0
            else radiance_np
        )

        # Column 1: Spatial and Angular Distributions
        # Panel (0,0): XY Spatial Distribution
        scatter1 = axes[0, 0].scatter(
            x, y, c=radiance_norm, s=5, alpha=0.6, cmap="viridis"
        )
        axes[0, 0].set_xlabel("X [mm]")
        axes[0, 0].set_ylabel("Y [mm]")
        axes[0, 0].set_title(f"{type(self.source).__name__}\nSpatial Distribution")
        axes[0, 0].set_aspect("equal", adjustable="box")
        axes[0, 0].grid(alpha=0.3)
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        cbar1.set_label("Normalized Radiance")

        # Panel (1,0): Angular Distribution (L vs M)
        scatter2 = axes[1, 0].scatter(
            L, M, c=radiance_norm, s=5, alpha=0.6, cmap="viridis"
        )
        axes[1, 0].set_xlabel("L (Direction Cosine)")
        axes[1, 0].set_ylabel("M (Direction Cosine)")
        axes[1, 0].set_title("Angular Distribution")
        axes[1, 0].set_aspect("equal", adjustable="box")
        axes[1, 0].grid(alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cbar2.set_label("Normalized Radiance")

        # Column 2: Cross-sections
        # Panel (0,1): Spatial Cross-sections
        self._plot_cross_sections(
            axes[0, 1],
            x,
            y,
            radiance_norm,
            cross_spatial,
            ["X [mm]", "Y [mm]"],
            "Spatial Cross-Sections",
            spatial=True,
        )

        # Panel (1,1): Angular Cross-sections
        self._plot_cross_sections(
            axes[1, 1],
            L,
            M,
            radiance_norm,
            cross_angular,
            ["L", "M"],
            "Angular Cross-Sections",
            spatial=False,
        )

        # Column 3: Propagation Views
        # Panel (0,2): XZ Propagation
        self._plot_ray_propagation(
            axes[0, 2],
            x,
            z,
            L,
            N,
            radiance_norm,
            propagation_distance,
            "Z [mm]",
            "X [mm]",
            "XZ Ray Propagation",
        )

        # Panel (1,2): YZ Propagation
        self._plot_ray_propagation(
            axes[1, 2],
            y,
            z,
            M,
            N,
            radiance_norm,
            propagation_distance,
            "Z [mm]",
            "Y [mm]",
            "YZ Ray Propagation",
        )

        plt.tight_layout()
        return fig, axes.flatten().tolist()

    def _plot_cross_sections(
        self,
        ax: Axes,
        coord1: np.ndarray,
        coord2: np.ndarray,
        intensity: np.ndarray,
        axes_labels: tuple[str, str],
        axis_units: list[str],
        title: str,
        spatial: bool = True,
        num_bins: int = 50,
    ) -> None:
        """Plot cross-sections of spatial or angular distributions.

        Args:
            ax (Axes): Matplotlib axes to plot on.
            coord1 (np.ndarray): First coordinate array.
            coord2 (np.ndarray): Second coordinate array.
            intensity (np.ndarray): Intensity values for weighting.
            axes_labels (tuple[str, str]): Labels for the axes being plotted.
            axis_units (list[str]): Units for axis labels.
            title (str): Plot title.
            spatial (bool): Whether this is spatial (True) or angular (False).
            num_bins (int): Number of bins for histograms.
        """
        # Handle spatial vs angular cross-sections differently
        if spatial:
            # For spatial coordinates (x, y): Use density=True to avoid double-weighting
            # Rays are already spatially distributed according to the beam profile
            hist1, bins1 = np.histogram(coord1, bins=num_bins, density=True)
            hist2, bins2 = np.histogram(coord2, bins=num_bins, density=True)
        else:
            # For angular coordinates (L, M): Use intensity weighting
            weights1 = intensity / np.sum(intensity) if np.sum(intensity) > 0 else None
            weights2 = intensity / np.sum(intensity) if np.sum(intensity) > 0 else None

            # Create bins
            bins1 = np.linspace(coord1.min(), coord1.max(), num_bins)
            bins2 = np.linspace(coord2.min(), coord2.max(), num_bins)

            # Calculate weighted histograms
            hist1, _ = np.histogram(coord1, bins=bins1, weights=weights1)
            hist2, _ = np.histogram(coord2, bins=bins2, weights=weights2)

        # Normalize histograms
        hist1 = hist1 / np.max(hist1) if np.max(hist1) > 0 else hist1
        hist2 = hist2 / np.max(hist2) if np.max(hist2) > 0 else hist2

        # Plot cross-sections
        bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
        bin_centers2 = (bins2[:-1] + bins2[1:]) / 2

        ax.plot(bin_centers1, hist1, "b-", linewidth=2, label=f"{axes_labels[0]}")
        ax.plot(bin_centers2, hist2, "r-", linewidth=2, label=f"{axes_labels[1]}")

        ax.set_xlabel(f"{axes_labels[0]} / {axes_labels[1]} {axis_units[0]}")
        ax.set_ylabel("Normalized Intensity")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    def _plot_ray_propagation(
        self,
        ax: Axes,
        coord1: np.ndarray,
        coord2: np.ndarray,
        dir1: np.ndarray,
        dir2: np.ndarray,
        intensity: np.ndarray,
        distance: float,
        xlabel: str,
        ylabel: str,
        title: str,
    ) -> None:
        """Plot ray propagation in a 2D plane with intensity-based coloring.

        Args:
            ax (Axes): Matplotlib axes to plot on.
            coord1 (np.ndarray): Starting coordinates for first dimension.
            coord2 (np.ndarray): Starting coordinates for second dimension.
            dir1 (np.ndarray): Direction cosines for first dimension.
            dir2 (np.ndarray): Direction cosines for second dimension.
            intensity (np.ndarray): Ray intensities for color mapping.
            distance (float): Propagation distance in mm.
            xlabel (str): Label for x-axis.
            ylabel (str): Label for y-axis.
            title (str): Plot title.
        """
        # Calculate end points after propagation
        end_coord1 = coord1 + dir1 * distance
        end_coord2 = coord2 + dir2 * distance

        # Sample subset of rays for clearer visualization (max 1000 rays)
        num_rays = len(coord1)
        if num_rays > 1000:
            indices = np.linspace(0, num_rays - 1, 1000, dtype=int)
            coord1 = coord1[indices]
            coord2 = coord2[indices]
            end_coord1 = end_coord1[indices]
            end_coord2 = end_coord2[indices]
            intensity_subset = intensity[indices]
        else:
            intensity_subset = intensity

        # Create colormap for intensity visualization
        colors = plt.cm.viridis(intensity_subset)

        # Plot ray paths with intensity-based coloring
        for i in range(len(coord1)):
            ax.plot(
                [coord2[i], end_coord2[i]],
                [coord1[i], end_coord1[i]],
                color=colors[i],
                alpha=0.1,
                linewidth=0.5,
            )

        # Plot end points with intensity coloring
        ax.scatter(
            end_coord2,
            end_coord1,
            c=intensity_subset,
            s=3,
            alpha=0.8,
            cmap="viridis",
            label="Ray Origins",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
