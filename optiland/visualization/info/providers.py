"""Info Providers Module

This module provides a framework for generating informational text for
different types of objects in the Optiland visualization system. It uses a
strategy pattern to allow for extensible and modular information generation.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.visualization.system.lens import Lens2D
    from optiland.visualization.system.ray_bundle import RayBundle
    from optiland.visualization.system.surface import Surface2D


class BaseInfoProvider(abc.ABC):
    """Base class for information providers."""

    @abc.abstractmethod
    def get_info(self, obj: any) -> str:
        """Returns informational text for the given object."""


class SurfaceInfoProvider(BaseInfoProvider):
    """Provides information for Surface2D objects."""

    def __init__(self, surface_group):
        self.surfaces = surface_group

    def get_info(self, obj: Surface2D) -> str:
        surface = obj.surf
        # Find the index of the surface
        try:
            surface_index = self.surfaces.index(surface)
            info = [f"Surface: {surface_index}"]
        except ValueError:
            info = ["Surface: (Unknown)"]

        # Add STOP flag
        if surface.is_stop:
            info[0] += " (STOP)"

        if hasattr(surface, "comment") and surface.comment:
            info.append(f"Comment: {surface.comment}")
        if hasattr(surface, "geometry") and hasattr(surface.geometry, "radius"):
            info.append(f"Radius: {surface.geometry.radius:.3f}")
        if hasattr(surface, "geometry") and hasattr(surface.geometry, "conic"):
            info.append(f"Conic: {surface.geometry.conic:.3f}")

        # Add Thickness and Material
        if hasattr(surface, "thickness"):
            info.append(f"Thickness: {surface.thickness:.3f}")
        if hasattr(surface, "material_post") and hasattr(surface.material_post, "name"):
            info.append(f"Material: {surface.material_post.name}")

        return "\n".join(info)


class LensInfoProvider(BaseInfoProvider):
    """Provides information for Lens2D objects."""

    def __init__(self, surface_group):
        self.surfaces = surface_group

    def get_info(self, obj: Lens2D) -> str:
        if not obj.surfaces:
            return "Lens (Empty)"

        # Get first and last surface objects
        first_surf = obj.surfaces[0].surf
        second_surf = obj.surfaces[1].surf

        # Find indices
        try:
            first_idx = self.surfaces.index(first_surf)
            second_idx = self.surfaces.index(second_surf)
            info = [f"Lens (Surfaces: {first_idx}-{second_idx})"]
        except ValueError:
            info = ["Lens"]

        # Get Lens Material (from first surface's post_material)
        if hasattr(first_surf, "material_post"):
            material = first_surf.material_post
            nd = material.n(0.5875618).item()
            vd = material.abbe().item()
            info.append(f"Material: n_d={nd:.4f}, Vd={vd:.1f}")

        # Get Center Thickness (from first surface's thickness)
        if hasattr(first_surf, "thickness"):
            info.append(f"Center Thickness: {first_surf.thickness:.3f}")

        return "\n".join(info)


class RayBundleInfoProvider(BaseInfoProvider):
    """Provides information for ray bundles."""

    def get_info(self, obj: RayBundle) -> str:
        info = ["Ray Bundle"]

        # Field is always a tuple of two floats (x, y)
        field = obj.field
        x, y = field
        x_val = x.item() if hasattr(x, "item") else float(x)
        y_val = y.item() if hasattr(y, "item") else float(y)
        info.append(f"Field: ({x_val:.2f}, {y_val:.2f})")

        # TODO: Add wavelength info to RayBundle
        if hasattr(obj, "wavelength"):
            if isinstance(obj.wavelength, int | float):
                info.append(f"Wavelength: {obj.wavelength:.1f} nm")
            else:
                info.append(f"Wavelength: {obj.wavelength}")

        return "\n".join(info)


# A registry to map object types to their info providers
INFO_PROVIDER_REGISTRY = {
    # Note: LensInfoProvider and SurfaceInfoProvider are
    # instantiated directly in interaction.py to pass surface_group
    "RayBundle": RayBundleInfoProvider(),
}
