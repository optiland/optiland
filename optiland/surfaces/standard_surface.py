"""Standard Surface

This module defines the `Surface` class, which represents a surface in an
optical system. Surfaces are characterized by their geometry, materials before
and after the surface, and optional properties such as being an aperture stop,
having a physical aperture, and a coating. The module facilitates the tracing
of rays through these surfaces, accounting for refraction, reflection, and
absorption based on the surface properties and materials involved.

Kramer Harrison, 2023
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.coatings import BaseCoating, FresnelCoating, ThinFilmCoating
from optiland.geometries import BaseGeometry
from optiland.interactions.base import BaseInteractionModel
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.materials import BaseMaterial
from optiland.physical_apertures import BaseAperture
from optiland.physical_apertures.radial import configure_aperture
from optiland.scatter import BaseBSDF
from optiland.surfaces.observer import ObserverMixin

if TYPE_CHECKING:
    from optiland.rays import BaseRays, ParaxialRays, RealRays


class _TracingCoordinator:
    """Owns the localize → intersect → globalize → clip → interact → record pipeline.

    Stateless: all surface components are accessed via the ``surface`` argument
    passed to :meth:`trace`, so geometry/aperture/interaction_model changes on
    the surface are always reflected without recreating the coordinator.
    """

    def trace(self, rays: BaseRays, surface: Surface) -> BaseRays:
        """Execute the full ray-surface interaction pipeline.

        Args:
            rays: The rays to be traced.
            surface: The surface being traced through.

        Returns:
            The traced rays.
        """
        surface.reset()
        surface.geometry.localize(rays)
        rays = rays.trace_on_surface(surface)
        surface.geometry.globalize(rays)
        rays.record_on_surface(surface)
        return rays


class Surface(ObserverMixin):
    """Represents a standard refractice surface in an optical system.

    Args:
        geometry (BaseGeometry): The geometry of the surface.
        previous_surface (Surface): The surface preceding this instance.
        material_post (BaseMaterial): The material after the surface.
        is_stop (bool, optional): Indicates if the surface is the aperture
            stop. Defaults to False.
        aperture (BaseAperture, int, float, optional): The physical aperture of the
            surface. Defaults to None. If a scalar is provided, it specifies the
            diameter of the lens.
        surface_type (str, optional): The type of surface. Defaults to None.
        comment (str, optional): A comment for the surface. Defaults to ''.
        interaction_model (BaseInteractionModel, optional): The interaction
            model for the surface. Defaults to None.

    """

    _registry = {}  # registry for all surfaces

    def __init__(
        self,
        previous_surface: Surface | None,
        material_post: BaseMaterial,
        geometry: BaseGeometry,
        is_stop: bool = False,
        aperture: BaseAperture | None = None,
        surface_type: str | None = None,
        comment: str = "",
        interaction_model: BaseInteractionModel | None = None,
    ):
        self.geometry = geometry
        self._previous_surface = previous_surface
        self._material_post = material_post
        self.is_stop = is_stop
        self.aperture = configure_aperture(aperture)
        self.semi_aperture = None
        self.surface_type = surface_type
        self.comment = comment

        if interaction_model is None:
            self.interaction_model = RefractiveReflectiveModel(
                parent_surface=self,
                is_reflective=False,
                coating=None,
                bsdf=None,
            )
        else:
            self.interaction_model = interaction_model
            self.interaction_model.parent_surface = self

        self.thickness = 0.0  # used for surface positioning
        ObserverMixin.__init__(self)
        self.reset()

    def _on_upstream_material_change(self) -> None:
        """Called by the upstream (previous) surface when its material changes."""
        if isinstance(getattr(self.interaction_model, "coating", None), FresnelCoating):
            self.set_fresnel_coating()
        elif isinstance(
            getattr(self.interaction_model, "coating", None), ThinFilmCoating
        ):
            self.interaction_model.coating.stack.incident_material = self.material_pre
            self.interaction_model.coating.stack.substrate_material = self.material_post

    @property
    def previous_surface(self):
        return self._previous_surface

    @previous_surface.setter
    def previous_surface(self, surface: Surface):
        if self._previous_surface is not None:
            self._previous_surface.unsubscribe(self._on_upstream_material_change)

        self._previous_surface = surface
        if surface is not None:
            surface.subscribe(self._on_upstream_material_change)
        self._on_upstream_material_change()  # Explicit trigger

    @property
    def material_pre(self) -> BaseMaterial | None:
        return (
            self.previous_surface.material_post
            if self.previous_surface is not None
            else self.material_post
        )

    @property
    def material_post(self) -> BaseMaterial | None:
        return self._material_post

    @material_post.setter
    def material_post(self, material: BaseMaterial):
        self._material_post = material

        # Update coating for new material
        _coating = getattr(self.interaction_model, "coating", None)
        if isinstance(_coating, FresnelCoating):
            self.set_fresnel_coating()
        elif isinstance(_coating, ThinFilmCoating):
            _coating.stack.incident_material = self.material_pre
            _coating.stack.substrate_material = self.material_post

        self._notify()

    @property
    def coating(self):
        warnings.warn(
            "surface.coating is deprecated; "
            "use surface.interaction_model.coating instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if (interaction_model := getattr(self, "interaction_model", None)) is not None:
            return getattr(interaction_model, "coating", None)

    @coating.setter
    def coating(self, value):
        warnings.warn(
            "surface.coating setter is deprecated; "
            "use surface.interaction_model.coating instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "interaction_model"):
            self.interaction_model.coating = value

    def flip(self):
        """Flips the surface, swapping materials and reversing geometry."""
        self.material_post = self.previous_surface.material_post
        self.geometry.flip()

        # Re-create the interaction model with flipped properties
        self.interaction_model.flip()

        if self.interaction_model.coating is not None and hasattr(
            self.interaction_model.coating, "flip"
        ):
            self.interaction_model.coating.flip()

        self.reset()

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        Surface._registry[cls.__name__] = cls

    def trace(self, rays: BaseRays) -> BaseRays:
        """Traces the given rays through the surface.

        Args:
            rays (BaseRays): The rays to be traced.

        Returns:
            BaseRays: The traced rays.

        """
        return self._coordinator.trace(rays, self)

    @property
    def _coordinator(self) -> _TracingCoordinator:
        """Lazy coordinator — created once, holds no mutable state."""
        try:
            return self.__coordinator
        except AttributeError:
            self.__coordinator = _TracingCoordinator()
            return self.__coordinator

    def _trace_paraxial(self, rays: ParaxialRays) -> ParaxialRays:
        """Paraxial physics kernel: propagate and interact.

        Args:
            rays (ParaxialRays): The paraxial rays.

        Returns:
            ParaxialRays: The traced paraxial rays.

        """
        t = -rays.z
        rays.propagate(t)
        rays = self.interaction_model.interact_paraxial_rays(rays)
        return rays

    def _trace_real(self, rays: RealRays) -> RealRays:
        """Real ray physics kernel: propagate and interact.

        Args:
            rays (RealRays): The real rays.

        Returns:
            RealRays: The traced real rays.

        """
        t = self.geometry.distance(rays)
        self.material_pre.propagation_model.propagate(rays, t)
        rays.opd = rays.opd + be.abs(t * self.material_pre.n(rays.w))
        if self.aperture:
            self.aperture.clip(rays)
        rays = self.interaction_model.interact_real_rays(rays)
        return rays

    def _record_paraxial(self, rays: ParaxialRays) -> None:
        """Records paraxial ray information after tracing.

        Args:
            rays (ParaxialRays): The paraxial rays.

        """
        self.y = be.copy(be.atleast_1d(rays.y))
        self.u = be.copy(be.atleast_1d(rays.u))

    def _record_real(self, rays: RealRays) -> None:
        """Records real ray information after tracing.

        Args:
            rays (RealRays): The real rays.

        """
        self.x = be.copy(be.atleast_1d(rays.x))
        self.y = be.copy(be.atleast_1d(rays.y))
        self.z = be.copy(be.atleast_1d(rays.z))
        self.L = be.copy(be.atleast_1d(rays.L))
        self.M = be.copy(be.atleast_1d(rays.M))
        self.N = be.copy(be.atleast_1d(rays.N))
        self.intensity = be.copy(be.atleast_1d(rays.i))
        self.opd = be.copy(be.atleast_1d(rays.opd))

    def set_semi_aperture(self, r_max: float):
        """Sets the physical semi-aperture of the surface.

        Args:
            r_max (float): The maximum radius of the semi-aperture.

        """
        self.semi_aperture = r_max

    def reset(self):
        """Resets the recorded information of the surface."""
        self.y = be.empty(0)
        self.u = be.empty(0)
        self.x = be.empty(0)
        self.y = be.empty(0)
        self.z = be.empty(0)

        self.L = be.empty(0)
        self.M = be.empty(0)
        self.N = be.empty(0)

        self.intensity = be.empty(0)
        self.aoi = be.empty(0)
        self.opd = be.empty(0)

    def set_fresnel_coating(self):
        """Sets the coating of the surface to a Fresnel coating."""
        self.interaction_model.coating = FresnelCoating(
            self.material_pre, self.material_post
        )

    def is_rotationally_symmetric(self):
        """Returns True if the surface is rotationally symmetric, False otherwise."""
        if not self.geometry.is_symmetric:
            return False

        cs = self.geometry.cs
        return not (cs.rx != 0 or cs.ry != 0 or cs.x != 0 or cs.y != 0)

    def to_dict(self):
        """Returns a dictionary representation of the surface."""
        # backward compatibility
        if not hasattr(self, "interaction_model"):
            self.interaction_model = RefractiveReflectiveModel(
                parent_surface=self,
                is_reflective=self.is_reflective,
                bsdf=self.bsdf,
            )

        return {
            "type": self.__class__.__name__,
            "thickness": self.thickness,
            "geometry": self.geometry.to_dict(),
            "material_post": self.material_post.to_dict(),
            "is_stop": self.is_stop,
            "aperture": self.aperture.to_dict() if self.aperture else None,
            "comment": self.comment,
            "interaction_model": self.interaction_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        """Creates a surface from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the surface.

        Returns:
            Surface: The surface.

        """
        if "type" not in data:
            raise ValueError("Missing 'type' field.")

        type_name = data["type"]
        subclass = cls._registry.get(type_name, cls)
        return subclass._from_dict(data)

    @classmethod
    def _from_dict(cls, data):
        """Protected deserialization logic for direct initialization.

        Args:
            data (dict): The dictionary representation of the surface.

        Returns:
            Surface: The surface.

        """
        surface_type = data.get("type")
        geometry = BaseGeometry.from_dict(data["geometry"])
        material_post = BaseMaterial.from_dict(data["material_post"])
        aperture = (
            BaseAperture.from_dict(data["aperture"]) if data.get("aperture") else None
        )

        interaction_model_data = data.get("interaction_model")
        if interaction_model_data:
            interaction_model = BaseInteractionModel.from_dict(
                interaction_model_data, None
            )
        else:
            # Backward compatibility
            coating = (
                BaseCoating.from_dict(data["coating"]) if data.get("coating") else None
            )
            bsdf = BaseBSDF.from_dict(data["bsdf"]) if data.get("bsdf") else None
            interaction_model = RefractiveReflectiveModel(
                parent_surface=None,
                is_reflective=data.get("is_reflective", False),
                coating=coating,
                bsdf=bsdf,
            )

        surface_class = cls._registry.get(surface_type, cls)
        surface = surface_class(
            previous_surface=None,
            geometry=geometry,
            material_post=material_post,
            is_stop=data.get("is_stop", False),
            aperture=aperture,
            comment=data.get("comment", ""),
            interaction_model=interaction_model,
        )
        interaction_model.parent_surface = surface
        surface.thickness = data.get("thickness", 0.0)
        return surface
