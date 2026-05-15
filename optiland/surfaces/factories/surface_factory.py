"""Surface Factory

This module contains the SurfaceFactory class, which is used to create surface
objects based on the given parameters. The SurfaceFactory class is used by the
SurfaceGroup class to create surfaces for the optical system. The class
abstracts the creation of surface objects and allows for easy configuration of
the surface parameters.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.surfaces.factories.coating_factory import CoatingFactory
from optiland.surfaces.factories.coordinate_system_factory import (
    CoordinateSystemFactory,
)
from optiland.surfaces.factories.geometry_factory import GeometryFactory
from optiland.surfaces.factories.interaction_model_factory import (
    InteractionModelFactory,
)
from optiland.surfaces.factories.material_factory import MaterialFactory
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.standard_surface import Surface

if TYPE_CHECKING:
    from optiland._types import SurfaceParameters, SurfaceType, Unpack
    from optiland.materials.base import BaseMaterial


# Surface type → forced interaction_type override.
# None value means: use caller-supplied interaction_type or the default.
_SURFACE_INTERACTION_OVERRIDES: dict[str, str] = {
    "paraxial": "thin_lens",
    "grating": "diffractive",
}


class SurfaceFactory:
    """A factory class for creating surface objects by delegating to sub-factories.

    Args:
        surface_group (SurfaceGroup): The surface group to which the surfaces belong.

    Attributes:
        _surface_group (SurfaceGroup): The surface group to which the surfaces belong.
        _coordinate_factory (CoordinateSystemFactory): Factory for coordinate systems.
        _geometry_factory (GeometryFactory): Factory for surface geometries.
        _material_factory (MaterialFactory): Factory for materials.
        _coating_factory (CoatingFactory): Factory for coatings.
    """

    @classmethod
    def register_surface_interaction(
        cls,
        surface_type: str,
        interaction_type: str,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a mandatory interaction type for a given surface type.

        When a surface of this type is created, the factory always uses the
        specified interaction_type regardless of any caller kwarg.

        Args:
            surface_type: The surface type string key (e.g. 'paraxial').
            interaction_type: The interaction model string key to force.
            overwrite: Allow replacing an existing mapping.

        Raises:
            ValueError: If surface_type already has a mapping and overwrite is False.
        """
        if surface_type in _SURFACE_INTERACTION_OVERRIDES and not overwrite:
            raise ValueError(
                f"Surface type '{surface_type}' already has a registered "
                "interaction override. Pass overwrite=True to replace it."
            )
        _SURFACE_INTERACTION_OVERRIDES[surface_type] = interaction_type

    def __init__(self, surface_group):
        self._surface_group = surface_group

        # CoordinateSystemFactory requires access to SurfaceFactory attributes
        self._coordinate_factory = CoordinateSystemFactory(self)
        self._geometry_factory = GeometryFactory()
        self.material_factory = MaterialFactory()
        self._coating_factory = CoatingFactory()
        self._interaction_model_factory = InteractionModelFactory()

        self.use_absolute_cs = False

    def create_surface(
        self,
        surface_type: SurfaceType,
        comment: str,
        index: int | None,
        is_stop: bool,
        material: BaseMaterial | str,
        **kwargs: Unpack[SurfaceParameters],
    ):
        """Creates a surface object based on the given parameters.

        Args:
            surface_type (str): The type of surface to create.
            comment (str): A comment for the surface.
            index (int): The index of the surface.
            is_stop (bool): Indicates whether the surface is a stop surface.
            material (str or tuple or BaseMaterial): The material of the surface.
            **kwargs: Additional keyword arguments for configuring the surface.

        Returns:
            Surface: The created surface object.

        Raises:
            ValueError: If the index is greater than the number of surfaces.
        """
        if index > self._surface_group.num_surfaces:
            raise IndexError("Surface index cannot be greater than number of surfaces.")

        # Build coordinate system
        coordinate_system = self._coordinate_factory.create(
            index, self._surface_group, **kwargs
        )

        # Build pre and post surface materials
        material_pre, material_post = self.material_factory.create(
            index, material, self._surface_group
        )

        # Build coating
        coating = self._coating_factory.create(
            kwargs.get("coating"), material_pre, material_post
        )

        is_reflective = material == "mirror"

        # Build geometry
        geometry = self._geometry_factory.create(
            surface_type, coordinate_system, **kwargs
        )

        # Special case: object surface
        if index == 0:
            if surface_type == "paraxial":
                raise ValueError("Paraxial surface cannot be the object surface.")
            surface_obj = ObjectSurface(geometry, material_post, comment)
            surface_obj.thickness = kwargs.get("thickness", 0.0)
            return surface_obj

        # Determine interaction type
        interaction_type = kwargs.get("interaction_type", "refractive_reflective")

        if surface_type in _SURFACE_INTERACTION_OVERRIDES:
            interaction_type = _SURFACE_INTERACTION_OVERRIDES[surface_type]
        elif kwargs.get("phase_profile") is not None:
            interaction_type = "phase"

        # Build interaction model
        interaction_kwargs = {
            "focal_length": kwargs.get("f"),
            "phase_profile": kwargs.get("phase_profile"),
        }
        interaction_model = self._interaction_model_factory.create(
            parent_surface=None,  # Hooked up in Surface.__init__()
            interaction_type=interaction_type,
            is_reflective=is_reflective,
            coating=coating,
            bsdf=kwargs.get("bsdf"),
            **interaction_kwargs,
        )

        # Standard surface - `surface_type` indicates geometrical shape of surface
        surface_obj = Surface(
            previous_surface=None,  #  To be fixed by surface_group.surfaces.add()
            geometry=geometry,
            material_post=material_post,
            is_stop=is_stop,
            surface_type=surface_type,
            comment=comment,
            aperture=kwargs.get("aperture"),
            interaction_model=interaction_model,
        )

        # Add the thickness as an attribute to the surface
        surface_obj.thickness = kwargs.get("thickness", 0.0)
        return surface_obj
