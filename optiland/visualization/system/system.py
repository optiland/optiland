"""System Visualization Module

This module contains the OpticalSystem class for visualizing optical systems.

Kramer Harrison, 2024
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.visualization.system.lens import Lens2D, Lens3D
from optiland.visualization.system.mirror import Mirror3D
from optiland.visualization.system.surface import Surface2D, Surface3D
from optiland.visualization.system.utils import transform

if TYPE_CHECKING:
    from optiland.visualization.component_renderer import ComponentRenderer

# Registry for user-defined ComponentRenderer extensions.
# Built-in component types ("lens", "mirror", "surface") are handled via the
# internal component_registry on each OpticalSystem instance.
_CUSTOM_RENDERER_REGISTRY: dict[str, ComponentRenderer] = {}


class _CustomRendererAdapter:
    """Adapts a ComponentRenderer to the plot()-based component interface."""

    def __init__(
        self,
        renderer: ComponentRenderer,
        component_data: dict,
        projection: str,
    ) -> None:
        self._renderer = renderer
        self._component_data = component_data
        self._projection = projection

    def plot(self, ax, **kwargs):
        if self._projection == "2d":
            self._renderer.render_2d(ax, self._component_data)
        else:
            self._renderer.render_3d(ax, self._component_data)
        return {}


class OpticalSystem:
    """A class to represent an optical system for visualization. The optical
    system contains surfaces and lenses.

    Args:
        optic (Optic): The optical system to be used for plotting.
        rays (Rays): The rays interacting with the optical system.
        projection (str): The type of projection for visualization.
            Must be '2d' or '3d'.

    Attributes:
        optic (Optic): The optical system to be used for plotting.
        rays (Rays): The rays interacting with the optical system.
        projection (str): The type of projection for visualization.
            Must be '2d' or '3d'.
        components (list): A list to store the components of the optical
            system.
        component_registry (dict): A registry mapping component names to their
            respective classes for 2D and 3D projections.

    Methods:
        plot(ax):
            Identifies and plots the components of the optical system on the
                given axis (or renderer for 3D plotting).

    """

    def __init__(self, optic, rays, projection="2d"):
        self.optic = optic
        self.rays = rays
        self.projection = projection
        self.components = []  # initialize empty list of components

        if self.projection not in ["2d", "3d"]:
            raise ValueError("Invalid projection type. Must be '2d' or '3d'.")

        self.component_registry = {
            "lens": {"2d": Lens2D, "3d": Lens3D},
            "mirror": {"2d": Surface2D, "3d": Mirror3D},
            "surface": {"2d": Surface2D, "3d": Surface3D},
        }

    @classmethod
    def register_component_renderer(
        cls,
        component_type: str,
        renderer: ComponentRenderer,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a renderer for a custom component type.

        The registered renderer is used when ``_identify_components`` adds a
        component of this type. Custom types are checked before the built-in
        registry, so this can also override existing built-in types when
        ``overwrite=True``.

        Args:
            component_type: String key identifying the component type.
            renderer: A ComponentRenderer instance.
            overwrite: Allow replacing an existing registration.

        Raises:
            ValueError: If component_type is already registered and
                overwrite is False.
        """
        if component_type in _CUSTOM_RENDERER_REGISTRY and not overwrite:
            raise ValueError(
                f"Component type '{component_type}' is already registered. "
                "Pass overwrite=True to replace it."
            )
        _CUSTOM_RENDERER_REGISTRY[component_type] = renderer

    def plot(self, ax, theme=None, projection="YZ", show_apertures=True):
        """Plots the components of the optical system on the given
        axis (or renderer for 3D plotting).
        """
        self._identify_components()
        artists = {}
        for component in self.components:
            component_artists = component.plot(ax, theme=theme, projection=projection)
            if component_artists:
                artists.update(component_artists)
        if show_apertures and self.projection == "2d":
            aperture_artists = self._plot_apertures(ax, projection=projection)
            artists.update(aperture_artists)
        return artists

    def _identify_components(self):
        """Identifies the components of the optical system and adds them to the
        list of components.
        """
        self.components = []
        n = self.optic.surfaces.n(self.optic.primary_wavelength)  # refractive indices
        num_surf = self.optic.surfaces.num_surfaces

        lens_surfaces = []

        for k, surf in enumerate(self.optic.surfaces):
            # Get the surface extent
            extent = self.rays.r_extent[k]

            # Object surface
            if k == 0:
                if not surf.is_infinite:
                    self._add_component("surface", surf, extent)

            # Image surface or paraxial surface
            elif k == num_surf - 1 or surf.surface_type == "paraxial":
                self._add_component("surface", surf, extent)

            # Surface is a mirror
            elif surf.interaction_model.is_reflective:
                if lens_surfaces:  # Second surface mirror (lens + mirror)
                    surface = self._get_lens_surface(surf, extent)
                    lens_surfaces.append(surface)
                    self._add_component("lens", lens_surfaces)
                    lens_surfaces = []
                else:
                    self._add_component("mirror", surf, extent)

            # Front surface of a lens
            elif n[k] > 1:
                surface = self._get_lens_surface(surf, extent)
                lens_surfaces.append(surface)

            # Back surface of a lens
            elif n[k] == 1 and n[k - 1] > 1 and lens_surfaces:
                surface = self._get_lens_surface(surf, extent)
                lens_surfaces.append(surface)
                self._add_component("lens", lens_surfaces)

                lens_surfaces = []

            # Standalone phase surface
            elif surf.interaction_model.interaction_type == "phase":
                self._add_component("surface", surf, extent)

        # add final lens, if any
        if lens_surfaces:
            self._add_component("lens", lens_surfaces)

    def _add_component(self, component_name, *args):
        """Adds a component to the list of components."""
        if component_name in _CUSTOM_RENDERER_REGISTRY:
            renderer = _CUSTOM_RENDERER_REGISTRY[component_name]
            component_data = {"args": args, "projection": self.projection}
            self.components.append(
                _CustomRendererAdapter(renderer, component_data, self.projection)
            )
        elif component_name in self.component_registry:
            component_class = self.component_registry[component_name][self.projection]
            self.components.append(component_class(*args))
        else:
            raise ValueError(f"Component {component_name} not found in registry.")

    def _get_lens_surface(self, surface, *args):
        """Gets the lens surface based on the projection type."""
        surface_class = self.component_registry["surface"][self.projection]
        return surface_class(surface, *args)

    def _plot_apertures(self, ax, projection="YZ"):
        if projection == "XY":
            return {}
        if projection not in ("XZ", "YZ"):
            raise ValueError("Invalid projection type. Must be 'XY', 'XZ', or 'YZ'.")

        stop_color = "black"  # arrow color for stop apertures
        aperture_color = "grey"  # arrow color for other apertures

        artists = {}
        n = self.optic.surfaces.n(self.optic.primary_wavelength)
        for idx, surface in enumerate(self.optic.surfaces):
            if idx > 0:
                is_lens_surface = n[idx] > 1 or (n[idx] == 1 and n[idx - 1] > 1)
            else:
                is_lens_surface = n[idx] > 1
            if is_lens_surface and not surface.is_stop:
                continue
            # Skip surfaces without apertures (unless stop)
            if surface.aperture is None and not surface.is_stop:
                continue

            # Determine aperture extent
            if surface.aperture is not None:
                x_min, x_max, y_min, y_max = surface.aperture.extent
            elif surface.semi_aperture is not None:
                r = surface.semi_aperture
                x_min, x_max, y_min, y_max = -r, r, -r, r
            elif (
                surface.is_stop
                and self.optic.aperture is not None
                and self.optic.aperture.ap_type == "float_by_stop_size"
            ):
                r = 0.5 * self.optic.aperture.value
                x_min, x_max, y_min, y_max = -r, r, -r, r
            elif surface.is_stop and self.rays is not None:
                r = be.to_numpy(self.rays.r_extent[idx]).item()
                if r <= 0:
                    continue
                x_min, x_max, y_min, y_max = -r, r, -r, r
            else:
                continue

            # Define local coordinates based on projection. Only the axis
            # actually shown in this projection is swept between its aperture
            # bounds; the other is held at 0 (its axis-of-symmetry value)
            # rather than paired corner-to-corner, which would otherwise mix
            # in the wrong sag contribution for offset apertures. The sag is
            # evaluated at each point (instead of assuming z=0) so the
            # indicator line follows the true, possibly-tilted surface
            # instead of a flat plane through the vertex.
            if projection == "XZ":
                x_local = be.array([x_min, x_max])
                y_local = be.array([0.0, 0.0])
            else:  # YZ
                x_local = be.array([0.0, 0.0])
                y_local = be.array([y_min, y_max])
            # Apertures with an unbounded extent (e.g. an annular
            # obstruction defined with r_max = inf) have no well-defined
            # sag at their outer edge; fall back to the vertex plane there
            # rather than evaluating sag() out of its domain.
            finite = be.isfinite(x_local) & be.isfinite(y_local)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z_local = be.where(finite, surface.geometry.sag(x_local, y_local), 0.0)
            x_global, y_global, z_global = transform(
                x_local, y_local, z_local, surface, is_global=False
            )
            x_global = be.to_numpy(x_global)
            y_global = be.to_numpy(y_global)
            z_global = be.to_numpy(z_global)

            # Draw line for aperture edge
            axis_vals = x_global if projection == "XZ" else y_global
            (line,) = ax.plot(
                z_global,
                axis_vals,
                color="black",
                linewidth=0.3,
            )
            artists[line] = surface

            # Add arrows to indicate aperture extent
            eps = 1e-6
            facecolor = stop_color if surface.is_stop else aperture_color
            arrowprops = {"arrowstyle": "-|>", "facecolor": facecolor, "linewidth": 0}
            axis_vals = x_global if projection == "XZ" else y_global
            for z_val, axis_val, sign in (
                (z_global[1], axis_vals[1], 1),  # top
                (z_global[0], axis_vals[0], -1),  # bottom
            ):
                ax.annotate(
                    "",
                    xy=(z_val, axis_val),
                    xytext=(z_val, axis_val + sign * eps),
                    arrowprops=arrowprops,
                )

        return artists
