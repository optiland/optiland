"""Extended Source Optic Module

This module defines the ExtendedSourceOptic class, a wrapper around the core
Optic class that enables extended source ray tracing. This design keeps the
core Optic class unchanged while providing source-based ray tracing and
visualization functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import optiland.backend as be

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.optic.optic import Optic
    from optiland.rays import RealRays
    from optiland.sources.base import BaseSource


class ExtendedSourceOptic:
    """Wrapper for Optic that enables extended source ray tracing.

    This class wraps a standard Optic instance and provides source-based
    ray tracing and visualization methods without modifying the core
    Optic API. It delegates all standard Optic attributes and methods
    to the underlying optic instance.

    Args:
        optic (Optic): The optical system to wrap.
        source (BaseSource): The extended source for ray generation.

    Attributes:
        optic (Optic): The underlying optical system.
        source (BaseSource): The extended source for generating rays.
    """

    def __init__(self, optic: Optic, source: BaseSource):
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, "optic", optic)
        object.__setattr__(self, "source", source)

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying optic.

        This allows transparent access to all Optic properties and methods
        that are not overridden by ExtendedSourceOptic.

        Args:
            name (str): The attribute name to look up.

        Returns:
            The attribute value from the underlying optic.

        Raises:
            AttributeError: If the attribute is not found on the optic.
        """
        return getattr(self.optic, name)

    def __setattr__(self, name: str, value: Any):
        """Delegate attribute setting to the underlying optic.

        Attributes specifics to ExtendedSourceOptic ('optic', 'source')
        are set locally. All others are set on the underlying optic.
        """
        if name in ("optic", "source"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.optic, name, value)

    def __repr__(self) -> str:
        optic_name = self.optic.name or "Unnamed"
        source_type = type(self.source).__name__
        return f"ExtendedSourceOptic(optic='{optic_name}', source={source_type})"

    def trace(self, num_rays: int = 1000) -> tuple[RealRays, dict]:
        """Trace rays generated from the extended source through the optical system.

        This method generates rays using the attached source and traces them
        through the optical system.

        Args:
            num_rays (int): The number of rays to generate and trace.
                Defaults to 1000.

        Returns:
            tuple: A tuple containing:
                - RealRays: The traced rays (final positions and directions).
                - dict: Ray path data with 'x', 'y', 'z' arrays of shape
                        (num_surfaces, num_rays).
        """
        # Generate rays from the source
        rays = self.source.generate_rays(num_rays)

        # Trace the rays through the optical system
        traced_rays = self.optic.surfaces.trace(rays)

        # Get the full ray path through all surfaces
        ray_path = {
            "x": self.optic.surfaces.x,
            "y": self.optic.surfaces.y,
            "z": self.optic.surfaces.z,
        }

        return traced_rays, ray_path

    def draw(
        self,
        num_rays: int = 1000,
        figsize: tuple[float, float] = (10, 4),
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str | None = None,
        projection: Literal["XY", "XZ", "YZ"] = "YZ",
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw a 2D representation of the optical system with rays from the
        extended source.

        This method traces rays from the attached source and visualizes them
        through the optical system alongside the rendered surfaces.

        Args:
            num_rays (int, optional): The number of rays to generate and trace.
                Defaults to 1000.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (10, 4).
            xlim (tuple[float, float] | None, optional): The x-axis limits of
                the plot. Defaults to None.
            ylim (tuple[float, float] | None, optional): The y-axis limits of
                the plot. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to
                None, which auto-generates a title from the source type.
            projection (Literal["XY", "XZ", "YZ"], optional): The projection
                plane. Defaults to "YZ".
            ax (matplotlib.axes.Axes, optional): The axes to plot on.
                If None, a new figure and axes are created. Defaults to None.

        Returns:
            tuple[Figure, Axes]: A tuple containing the matplotlib Figure and
            Axes objects of the plot.
        """
        import matplotlib.pyplot as plt

        from optiland.visualization.system.system import OpticalSystem
        from optiland.visualization.themes import get_active_theme

        # Generate and trace rays from the source
        traced_rays, ray_path = self.trace(num_rays)

        # Set up the figure
        theme = get_active_theme()
        params = theme.parameters

        if figsize is None:
            figsize = params["figure.figsize"]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_facecolor(params["figure.facecolor"])
        else:
            fig = ax.get_figure()

        ax.set_facecolor(params["axes.facecolor"])

        # We need surface extents for drawing. Trace a few traditional rays
        # if we have fields defined to establish proper surface extents.
        from optiland.visualization.system.rays import Rays2D

        rays_2d = Rays2D(self.optic)
        if len(self.optic.fields.fields) > 0:
            field = self.optic.fields.get_field_coords()[0]
            wavelength = self.optic.wavelengths.primary_wavelength.value
            rays_2d._trace(field, wavelength, 3, "line_y")

        # Draw the optical system surfaces
        system = OpticalSystem(self.optic, rays_2d, projection="2d")
        system.plot(ax, theme=theme, projection=projection)

        # Plot the rays from the source
        self._plot_source_rays(ax, ray_path, traced_rays, projection=projection)

        # Style the plot
        if projection == "YZ":
            ax.set_xlabel("Z [mm]", color=params["axes.labelcolor"])
            ax.set_ylabel("Y [mm]", color=params["axes.labelcolor"])
        elif projection == "XZ":
            ax.set_xlabel("Z [mm]", color=params["axes.labelcolor"])
            ax.set_ylabel("X [mm]", color=params["axes.labelcolor"])
        else:  # XY
            ax.set_xlabel("X [mm]", color=params["axes.labelcolor"])
            ax.set_ylabel("Y [mm]", color=params["axes.labelcolor"])

        ax.tick_params(axis="x", colors=params["xtick.color"])
        ax.tick_params(axis="y", colors=params["ytick.color"])
        for spine in ax.spines.values():
            spine.set_color(params["axes.edgecolor"])

        ax.axis("image")

        if title:
            ax.set_title(title, color=params["text.color"])
        else:
            ax.set_title(
                f"Optical System with {type(self.source).__name__}",
                color=params["text.color"],
            )

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.grid(
            visible=True,
            color=params["grid.color"],
            alpha=params["grid.alpha"],
        )

        plt.tight_layout()
        return fig, ax

    def _plot_source_rays(
        self,
        ax,
        ray_path: dict,
        traced_rays: RealRays,
        projection: str = "YZ",
    ):
        """Plot rays from the extended source on the given axis.

        Args:
            ax: The matplotlib axis to plot on.
            ray_path: Dictionary with 'x', 'y', 'z' arrays of ray paths,
                     each of shape (num_surfaces, num_rays).
            traced_rays: The traced rays object for intensity information.
            projection: The projection plane ('YZ', 'XZ', or 'XY').
        """
        x_coords = be.to_numpy(ray_path["x"])
        y_coords = be.to_numpy(ray_path["y"])
        z_coords = be.to_numpy(ray_path["z"])

        # Get final ray intensities
        intensities = be.to_numpy(traced_rays.i)

        if len(y_coords.shape) != 2 or len(z_coords.shape) != 2:
            return

        num_surfaces, num_rays = y_coords.shape

        # Plot each ray path
        for k in range(num_rays):
            # Skip rays that were completely blocked
            if intensities[k] <= 0:
                continue

            if projection == "YZ":
                ax.plot(
                    z_coords[:, k],
                    y_coords[:, k],
                    "b-",
                    alpha=0.3,
                    linewidth=0.5,
                )
            elif projection == "XZ":
                ax.plot(
                    z_coords[:, k],
                    x_coords[:, k],
                    "b-",
                    alpha=0.3,
                    linewidth=0.5,
                )
            else:  # XY
                ax.plot(
                    x_coords[:, k],
                    y_coords[:, k],
                    "b-",
                    alpha=0.3,
                    linewidth=0.5,
                )

    # --- Methods that don't apply for extended source optics ---

    def trace_generic(self, *args, **kwargs):
        """Not available for ExtendedSourceOptic.

        Raises:
            NotImplementedError: Always raised. Use trace(num_rays) instead.
        """
        raise NotImplementedError(
            "trace_generic() is not available for ExtendedSourceOptic. "
            "Use trace(num_rays) instead."
        )
