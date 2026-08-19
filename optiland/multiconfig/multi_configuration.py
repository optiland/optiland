"""Multi-Configuration Module

This module provides the MultiConfiguration class for managing optical systems
with multiple configurations, such as zoom lenses.

Kramer Harrison, 2025
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt

from optiland.utils import set_attr_by_path
from optiland.visualization import OpticViewer
from optiland.visualization.themes import get_active_theme

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.materials.base import BaseMaterial
    from optiland.optic import Optic
    from optiland.pickup import Pickup


class MultiConfiguration:
    """Manages multiple configurations of an optical system.

    This class holds a list of independent Optic instances, one for each
    configuration. It provides methods to create new configurations
    derived from a base configuration and link them using Pickups.

    Args:
        base_optic (Optic): The initial optical system (Configuration 0).

    Attributes:
        configurations (list[Optic]): The list of Optic instances.
    """

    # Aliases resolve straight to a dedicated setter; anything else is
    # treated as a generic dotted attribute path.
    _PROPERTY_ALIASES = {
        "radius": "set_radius",
        "thickness": "set_thickness",
        "conic": "set_conic",
        "material": "set_material",
    }

    # attr_type -> Optic.updater method name, for the standard properties.
    _STANDARD_UPDATERS = {
        "radius": "set_radius",
        "conic": "set_conic",
        "thickness": "set_thickness",
        "material": "set_material",
    }

    # Geometry attributes linked automatically when a configuration is
    # derived from a source: (attribute on the geometry, pickup attr_type).
    _LINKED_GEOMETRY_ATTRS = (
        ("radius", "radius"),
        ("k", "conic"),
    )

    def __init__(self, base_optic: Optic):
        self.configurations: list[Optic] = [base_optic]

    def add_configuration(self, source_config_idx: int = 0) -> Optic:
        """Creates a new configuration based on a source configuration.

        The new configuration is a deep copy of the source. By default,
        Pickups are added to the new configuration that link all its
        surface geometries and basic properties back to the source.
        This ensures that, initially, both configurations are identical
        and controlled by the source's variables.

        Args:
            source_config_idx (int): The index of the configuration to copy.
                Defaults to 0.

        Returns:
            Optic: The new configuration instance.
        """
        source_optic = self.configurations[source_config_idx]
        new_optic = copy.deepcopy(source_optic)
        self.configurations.append(new_optic)

        # Link the new optic to the source optic
        self._link_configurations(source_optic, new_optic)

        return new_optic

    def _link_configurations(self, source: Optic, target: Optic) -> None:
        """Internal method to link generic surface properties."""
        num_surfaces = len(source.surfaces)
        for i, surf_s in enumerate(source.surfaces):
            for geometry_attr, attr_type in self._LINKED_GEOMETRY_ATTRS:
                if hasattr(surf_s.geometry, geometry_attr):
                    target.pickups.add(
                        source_surface_idx=i,
                        attr_type=attr_type,
                        target_surface_idx=i,
                        source_optic=source,
                    )

            # Thickness is linked for every surface except the last.
            if i < num_surfaces - 1:
                target.pickups.add(
                    source_surface_idx=i,
                    attr_type="thickness",
                    target_surface_idx=i,
                    source_optic=source,
                )

    def _resolve_configs(self, configurations) -> tuple[list[int], bool]:
        """Expand "all" to explicit indices; report whether it was "all"."""
        if configurations == "all":
            return list(range(len(self.configurations))), True
        return configurations, False

    def _apply_across_configs(
        self,
        configs_to_update: list[int],
        all_selected: bool,
        set_value: Callable[[int], None],
        ensure_pickup: Callable[[int], None],
        remove_pickup: Callable[[int], None],
    ) -> None:
        """Shared update algorithm for both standard and generic properties.

        Configuration 0 always receives the value directly. Other
        configurations either stay linked to configuration 0 (when
        updating "all") or get an independent value with any existing
        link removed (when updating specific configurations).
        """
        for config_idx in configs_to_update:
            if config_idx == 0:
                set_value(0)
            elif all_selected:
                ensure_pickup(config_idx)
            else:
                remove_pickup(config_idx)
                set_value(config_idx)

    def set_property(
        self,
        value: Any,
        configurations: list[int] | Literal["all"] = "all",
        surface_index: int | None = None,
        attribute_path: str = None,
    ):
        """Set a property value across one or more configurations.

        Args:
            value: The value to set.
            configurations: A list of configuration indices to update, or "all"
                to update the base configuration and ensure links (pickups)
                exist (or are created) for other configurations.
            surface_index: The index of the surface to modify. If None, the
                property is assumed to be on the Optic itself.
            attribute_path: The dot-separated path to the attribute, or a
                known alias ('radius', 'thickness', 'conic', 'material').
        """
        alias_method = self._PROPERTY_ALIASES.get(attribute_path)
        if alias_method is not None:
            getattr(self, alias_method)(surface_index, value, configurations)
            return

        configs_to_update, all_selected = self._resolve_configs(configurations)
        self._apply_across_configs(
            configs_to_update,
            all_selected,
            set_value=lambda idx: self._set_generic_value(
                idx, surface_index, attribute_path, value
            ),
            ensure_pickup=lambda idx: self._ensure_generic_pickup(
                idx, 0, surface_index, attribute_path
            ),
            remove_pickup=lambda idx: self._remove_generic_pickup(
                idx, surface_index, attribute_path
            ),
        )

    def set_radius(
        self,
        surface_index: int,
        value: float,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Set the radius of a surface."""
        self._set_standard_property("radius", surface_index, value, configurations)

    def set_thickness(
        self,
        surface_index: int,
        value: float,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Set the thickness of a surface."""
        self._set_standard_property("thickness", surface_index, value, configurations)

    def set_conic(
        self,
        surface_index: int,
        value: float,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Set the conic constant of a surface."""
        self._set_standard_property("conic", surface_index, value, configurations)

    def set_material(
        self,
        surface_index: int,
        value: str | BaseMaterial,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Set the material of a surface."""
        self._set_standard_property("material", surface_index, value, configurations)

    def set_surface_property(
        self,
        surface_index: int,
        attribute_path: str,
        value: Any,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Convenience wrapper for set_property on a surface."""
        self.set_property(value, configurations, surface_index, attribute_path)

    def set_optic_property(
        self,
        attribute_path: str,
        value: Any,
        configurations: list[int] | Literal["all"] = "all",
    ):
        """Convenience wrapper for set_property on the optic."""
        self.set_property(value, configurations, None, attribute_path)

    def _set_standard_property(
        self,
        attr_type: str,
        surface_index: int | None,
        value: Any,
        configurations: list[int] | Literal["all"],
    ) -> None:
        """Internal helper for standard properties (radius, conic, etc)."""
        configs_to_update, all_selected = self._resolve_configs(configurations)

        if attr_type == "material":
            # Materials are pointer-like objects, so they're linked via a
            # generic pickup on 'material_post' rather than a standard one.
            def ensure_pickup(idx: int) -> None:
                self._ensure_generic_pickup(idx, 0, surface_index, "material_post")

            def remove_pickup(idx: int) -> None:
                self._remove_generic_pickup(idx, surface_index, "material_post")
        else:

            def ensure_pickup(idx: int) -> None:
                self._ensure_pickup(idx, surface_index, attr_type)

            def remove_pickup(idx: int) -> None:
                self._remove_pickup(idx, surface_index, attr_type)

        self._apply_across_configs(
            configs_to_update,
            all_selected,
            set_value=lambda idx: self._apply_standard_value(
                idx, surface_index, attr_type, value
            ),
            ensure_pickup=ensure_pickup,
            remove_pickup=remove_pickup,
        )

    def _apply_standard_value(
        self, config_idx: int, surface_index: int, attr_type: str, value: Any
    ) -> None:
        optic = self.configurations[config_idx]
        method = getattr(optic.updater, self._STANDARD_UPDATERS[attr_type])
        method(value, surface_index)

    @staticmethod
    def _full_attribute_path(surface_index: int | None, path: str) -> str:
        if surface_index is None:
            return path
        return f"surfaces.surfaces[{surface_index}].{path}"

    def _set_generic_value(
        self, config_idx: int, surface_index: int | None, path: str, value: Any
    ) -> None:
        optic = self.configurations[config_idx]
        full_path = self._full_attribute_path(surface_index, path)
        set_attr_by_path(optic, full_path, value)

    def _find_pickup(
        self, config_idx: int, matches: Callable[[Pickup], bool]
    ) -> Pickup | None:
        return next(
            (p for p in self.configurations[config_idx].pickups.pickups if matches(p)),
            None,
        )

    def _remove_pickups(
        self, config_idx: int, matches: Callable[[Pickup], bool]
    ) -> None:
        pickups = self.configurations[config_idx].pickups.pickups
        pickups[:] = [p for p in pickups if not matches(p)]

    def _ensure_pickup(
        self, config_idx: int, surface_index: int, attr_type: str
    ) -> None:
        """Ensure a standard pickup exists linking to config 0."""
        source = self.configurations[0]

        def matches(p: Pickup) -> bool:
            return (
                p.target_surface_idx == surface_index
                and p.attr_type == attr_type
                and p.source_optic == source
            )

        if self._find_pickup(config_idx, matches) is not None:
            return

        self.configurations[config_idx].pickups.add(
            source_surface_idx=surface_index,
            attr_type=attr_type,
            target_surface_idx=surface_index,
            source_optic=source,
        )

    def _remove_pickup(
        self, config_idx: int, surface_index: int, attr_type: str
    ) -> None:
        """Remove a standard pickup."""
        self._remove_pickups(
            config_idx,
            lambda p: (
                p.target_surface_idx == surface_index and p.attr_type == attr_type
            ),
        )

    def _ensure_generic_pickup(
        self, config_idx: int, source_idx: int, surface_index: int | None, path: str
    ) -> None:
        """Ensure a generic pickup exists.

        For Generic Pickups:
        - target_surface_idx: surface_index
        - attr_type: path (e.g. 'geometry.coefficients[0]')
        """
        source_optic = self.configurations[source_idx]
        full_path = self._full_attribute_path(surface_index, path)

        def matches(p: Pickup) -> bool:
            return p.attr_type == full_path and p.source_optic == source_optic

        if self._find_pickup(config_idx, matches) is not None:
            return

        self.configurations[config_idx].pickups.add(
            source_surface_idx=0,  # Ignored for generic
            attr_type=full_path,
            target_surface_idx=0,  # Ignored for generic
            source_optic=source_optic,
        )

    def _remove_generic_pickup(
        self, config_idx: int, surface_index: int | None, path: str
    ) -> None:
        full_path = self._full_attribute_path(surface_index, path)
        self._remove_pickups(config_idx, lambda p: p.attr_type == full_path)

    def current_config(self, index: int) -> Optic:
        """Returns the configuration at the given index."""
        return self.configurations[index]

    @staticmethod
    def _config_title(base_title: str | None, index: int) -> str:
        if base_title:
            return f"{base_title} (Config {index})"
        return f"Configuration {index}"

    @staticmethod
    def _as_axes_list(axes: Any, num_configs: int) -> list:
        if num_configs == 1:
            return [axes]
        if hasattr(axes, "flat"):
            return axes.flatten()
        return axes

    def draw(
        self,
        figsize: tuple[float, float] | None = None,
        sharex: bool = True,
        sharey: bool = True,
        **kwargs,
    ):
        """Draw the multi-configuration system.

        Args:
            figsize: The size of the figure for a SINGLE configuration.
                The total figure height will be scaled by the number of configs.
                If None, uses the active theme's default figsize.
            sharex: If True, share the x-axis limits and labels.
            sharey: If True, share the y-axis limits and labels.
            **kwargs: Additional arguments passed to OpticViewer.view().
        """
        theme = get_active_theme()
        params = theme.parameters

        if figsize is None:
            figsize = params["figure.figsize"]

        num_configs = len(self.configurations)
        total_height = figsize[1] * num_configs
        fig, axes = plt.subplots(
            nrows=num_configs,
            figsize=(figsize[0], total_height),
            sharex=sharex,
            sharey=sharey,
        )
        fig.set_facecolor(params["figure.facecolor"])
        axes = self._as_axes_list(axes, num_configs)

        base_title = kwargs.get("title")
        for i, (optic, ax) in enumerate(zip(self.configurations, axes, strict=False)):
            ax.set_facecolor(params["axes.facecolor"])

            plot_kwargs = kwargs.copy()
            plot_kwargs["title"] = self._config_title(base_title, i)

            OpticViewer(optic).view(ax=ax, **plot_kwargs)

        plt.tight_layout()
        return fig, axes
