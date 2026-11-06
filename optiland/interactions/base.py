"""Base Interaction Model

Defines the abstract base class for ray-surface interaction models.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pragma: no cover
    from optiland._types import BEArray
    from optiland.coatings import BaseCoating
    from optiland.rays import ParaxialRays, RealRays
    from optiland.scatter import BaseBSDF
    from optiland.surfaces import Surface


class BaseInteractionModel(ABC):
    """Abstract base class for ray-surface interaction models."""

    _registry = {}

    def __init__(
        self,
        parent_surface: Surface | None,
        is_reflective: bool,
        coating: BaseCoating | None = None,
        bsdf: BaseBSDF | None = None,
    ):
        self.parent_surface = parent_surface
        self.is_reflective = is_reflective
        self.coating = coating
        self.bsdf = bsdf

    @property
    def material_pre(self):
        return (
            self.parent_surface.material_post
            if self.parent_surface.previous_surface is None
            else self.parent_surface.previous_surface.material_post
        )

    @property
    def material_post(self):
        return self.parent_surface.material_post

    @property
    def geometry(self):
        return self.parent_surface.geometry

    def _real_indices(self, rays: RealRays) -> tuple[BEArray, BEArray]:
        """Refractive indices before and after the surface at the ray hits.

        Each medium's propagation model supplies its index, so media whose
        index varies with position are evaluated where the rays land.

        Args:
            rays: The rays, in the local frame of the parent surface.

        Returns:
            The indices ``(n1, n2)`` before and after the surface.
        """
        surface = self.parent_surface
        previous = surface.previous_surface
        entrance = surface if previous is None else previous
        n1 = entrance.material_post.propagation_model.boundary_index(
            rays, surface, entrance
        )
        n2 = surface.material_post.propagation_model.boundary_index(
            rays, surface, surface
        )
        return n1, n2

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseInteractionModel._registry[cls.__name__] = cls

    @abstractmethod
    def interact_real_rays(self, rays: RealRays) -> RealRays:
        """Interact with real rays."""
        pass  # pragma: no cover

    @abstractmethod
    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Interact with paraxial rays."""
        pass  # pragma: no cover

    @abstractmethod
    def flip(self):
        """Flip the interaction model."""
        pass  # pragma: no cover

    def to_dict(self):
        """Returns a dictionary representation of the interaction model."""
        return {
            "type": self.__class__.__name__,
            "is_reflective": self.is_reflective,
            "coating": self.coating.to_dict() if self.coating else None,
            "bsdf": self.bsdf.to_dict() if self.bsdf else None,
        }

    @classmethod
    def _deserialize_init_data(cls, data):
        from optiland.coatings import BaseCoating
        from optiland.scatter import BaseBSDF

        init_data = data.copy()
        init_data.pop("type", None)
        init_data.pop("material_pre", None)

        if init_data.get("coating") is not None:
            init_data["coating"] = BaseCoating.from_dict(init_data["coating"])

        if init_data.get("bsdf") is not None:
            init_data["bsdf"] = BaseBSDF.from_dict(init_data["bsdf"])

        return init_data

    @classmethod
    def from_dict(cls, data, parent_surface):
        """Creates an interaction model from a dictionary representation."""
        interaction_type = data["type"]
        subclass = cls._registry.get(interaction_type)
        if subclass is None:
            raise ValueError(f"Unknown interaction model type: {interaction_type}")

        init_data = subclass._deserialize_init_data(data)

        return subclass(
            parent_surface=parent_surface,
            **init_data,
        )

    def _apply_coating_and_bsdf(
        self, rays: RealRays, nx: float, ny: float, nz: float
    ) -> RealRays:
        """Apply coating and BSDF to the rays."""
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx, ny, nz)

        if self.coating:
            rays = self.coating.interact(
                rays,
                reflect=self.is_reflective,
                nx=nx,
                ny=ny,
                nz=nz,
            )
        else:
            rays.update()
        return rays
