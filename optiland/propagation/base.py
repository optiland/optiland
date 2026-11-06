"""Abstract base class for propagation models.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.materials.base import BaseMaterial
    from optiland.rays.real_rays import RealRays
    from optiland.surfaces.standard_surface import Surface


class BasePropagationModel(abc.ABC):
    """Abstract base class for all propagation models."""

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in the factory registry."""
        super().__init_subclass__(**kwargs)
        BasePropagationModel._registry[cls.__name__] = cls

    @abc.abstractmethod
    def propagate(self, rays: RealRays, t: float) -> None:
        """Propagates rays a distance t through the medium.

        This method modifies the ray state in-place.

        Args:
            rays: The rays object to be propagated.
            t: The distance to propagate along the z-axis.
        """
        raise NotImplementedError

    def propagate_to_surface(self, rays: RealRays, surface: Surface) -> None:
        """Propagates rays through this medium onto ``surface``.

        The default is straight-line travel: the surface geometry gives the
        distance, :meth:`propagate` moves the rays and the optical path
        ``t * n`` is added to ``rays.opd``. Models whose index varies with
        position override this method and own the optical path themselves.

        Args:
            rays: The rays, in the local frame of ``surface``. Modified
                in-place.
            surface: The surface at the end of this medium.
        """
        from optiland.surfaces.standard_surface import _aperture_aware_distance

        t = _aperture_aware_distance(surface, rays)
        self.propagate(rays, t)
        # t is oriented along the ray: virtual propagation subtracts OPL.
        # Real return paths after reflection still have t > 0 and add OPL.
        rays.opd = rays.opd + t * surface.material_pre.n(rays.w)

    def paraxial_transfer(
        self, y: BEArray, u: BEArray, t: BEArray
    ) -> tuple[BEArray, BEArray]:
        """Transfers paraxial rays an axial distance ``t`` through this medium.

        Args:
            y: Ray heights.
            u: Ray slopes.
            t: The axial distance.

        Returns:
            The heights and slopes after the transfer.
        """
        return y + t * u, u

    def boundary_index(
        self, rays: RealRays, surface: Surface, entrance: Surface
    ) -> BEArray:
        """Returns the refractive index of this medium where rays meet a surface.

        Args:
            rays: The rays, in the local frame of ``surface``.
            surface: The surface the rays are on.
            entrance: The surface where this medium starts, so
                ``entrance.material_post`` is the medium. Equal to ``surface``
                on the medium's entrance face.

        Returns:
            The refractive index for each ray.
        """
        return entrance.material_post.n(rays.w)

    def display_path(self, rays: RealRays, surface: Surface) -> BEArray | None:
        """Returns intermediate points of the ray path through this medium.

        Drawings join recorded surface hits with straight lines. A medium in
        which rays curve returns the points to draw between those hits.

        Args:
            rays: The rays leaving the previous surface, in global coordinates.
                Not modified.
            surface: The surface at the end of this medium.

        Returns:
            ``None`` for straight paths (the default), otherwise global points
            of shape ``(num_samples, num_rays, 3)``, NaN-padded per ray.
        """
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serializes the propagation model to a dictionary.

        By default, this only stores the class name, as the model is
        reconstructed and linked by the parent material during its own
        deserialization.
        """
        return {"class": self.__class__.__name__}

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], material: BaseMaterial
    ) -> BasePropagationModel:
        """Deserializes a propagation model from a dictionary using a factory pattern.

        This factory method finds the correct propagation model subclass
        from the registry and delegates the deserialization to it.

        Args:
            d: A dictionary containing the serialized data, including a 'class' key.
            material: The parent material instance, passed to the subclass constructor.

        Returns:
            An instance of a specific propagation model subclass.
        """
        model_class_name = d.get("class")
        if model_class_name not in cls._registry:
            raise ValueError(f"Unknown propagation model class: {model_class_name}")

        # Look up the specific subclass in the registry.
        model_subclass = cls._registry[model_class_name]

        # Delegate the actual object creation to the subclass's from_dict method.
        return model_subclass.from_dict(d, material)
