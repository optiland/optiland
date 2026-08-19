"""SurfaceView: a per-sequence proxy over a shared base ``Surface``.

A ``SurfaceView`` shares geometry, material, aperture and coating with a
base :class:`~optiland.surfaces.standard_surface.Surface` by reference, but
owns its own per-visit record buffers, direction and interaction model. It
implements the same duck-typed interface that ``_TracingCoordinator`` and
the ``BaseRays`` subclasses use (``reset``, ``geometry``, ``_trace_real``,
``_record_real``, ``_trace_paraxial``, ``_record_paraxial``), so it traces
through the existing, unforked tracing pipeline with zero changes to
``Surface`` or the rays classes.

Kramer Harrison, 2026
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial
    from optiland.physical_apertures import BaseAperture
    from optiland.rays import ParaxialRays, RealRays
    from optiland.surfaces.standard_surface import Surface


def resolve_view_materials(
    base_surface: Surface,
    reverse: bool,
    interaction_override: str | None,
) -> tuple[BaseMaterial, BaseMaterial]:
    """Resolve a view's incident/exit materials from its base surface.

    Args:
        base_surface: The shared base surface.
        reverse: Whether this step traverses the surface in the reverse
            physical direction.
        interaction_override: ``"reflect"``, ``"refract"``, or ``None``.

    Returns:
        The ``(material_pre, material_post)`` pair for this view.
    """
    if reverse:
        material_pre = base_surface.material_post
        material_post = base_surface.material_pre
    else:
        material_pre = base_surface.material_pre
        material_post = base_surface.material_post

    if interaction_override == "reflect":
        # A reflection never crosses into the far medium: the ray leaves on
        # the same side it arrived on.
        material_post = material_pre

    return material_pre, material_post


class SurfaceView:
    """A view of a base ``Surface`` within one step of a sequence.

    Geometry, aperture, and coating/BSDF objects are shared with the base
    surface by reference, so editing the base surface (or optimizing a
    variable on it) is immediately visible through every view. Record
    buffers, the interaction model instance, and the traversal direction
    are owned by the view.

    Args:
        base_surface: The shared base surface.
        reverse: Whether this step traverses the surface in the reverse
            physical direction.
        interaction_override: ``"reflect"``, ``"refract"``, or ``None`` to
            use the base surface's nominal interaction.
        previous_view: The preceding view in the sequence, or ``None`` if
            this is the first step.
    """

    def __init__(
        self,
        base_surface: Surface,
        reverse: bool = False,
        interaction_override: str | None = None,
        previous_view: SurfaceView | None = None,
    ):
        self.base_surface = base_surface
        self.reverse = reverse
        self.interaction_override = interaction_override
        self.previous_view = previous_view

        self.interaction_model = copy.copy(base_surface.interaction_model)
        self.interaction_model.parent_surface = self
        if interaction_override == "reflect":
            self.interaction_model.is_reflective = True
        elif interaction_override == "refract":
            self.interaction_model.is_reflective = False

        self._rebind_coating()
        self.reset()

    def _rebind_coating(self) -> None:
        """Rebind polarized coatings to this view's resolved media."""
        coating = getattr(self.interaction_model, "coating", None)
        if coating is None:
            return

        from optiland.coatings import FresnelCoating, ThinFilmCoating

        if isinstance(coating, FresnelCoating):
            self.interaction_model.coating = FresnelCoating(
                self.material_pre, self.material_post
            )
        elif isinstance(coating, ThinFilmCoating):
            layers = [
                (layer.material, layer.thickness_nm, layer.name)
                for layer in coating.stack.layers
            ]
            if self.reverse:
                layers = layers[::-1]
            self.interaction_model.coating = ThinFilmCoating(
                self.material_pre, self.material_post, layers=layers
            )

    # -- Shared-by-reference passthrough properties -----------------------

    @property
    def geometry(self):
        return self.base_surface.geometry

    @property
    def aperture(self) -> BaseAperture | None:
        return self.base_surface.aperture

    @property
    def semi_aperture(self) -> float | None:
        return self.base_surface.semi_aperture

    @property
    def is_stop(self) -> bool:
        return self.base_surface.is_stop

    @property
    def comment(self) -> str:
        return self.base_surface.comment

    @property
    def surface_type(self) -> str | None:
        return self.base_surface.surface_type

    @property
    def thickness(self) -> float:
        return self.base_surface.thickness

    # -- Owned, per-view material resolution -------------------------------

    @property
    def material_pre(self) -> BaseMaterial:
        pre, _ = resolve_view_materials(
            self.base_surface, self.reverse, self.interaction_override
        )
        return pre

    @property
    def material_post(self) -> BaseMaterial:
        _, post = resolve_view_materials(
            self.base_surface, self.reverse, self.interaction_override
        )
        return post

    @property
    def previous_surface(self):
        """Anchor for ``BaseInteractionModel.material_pre``'s chain lookup.

        ``BaseInteractionModel.material_pre`` resolves as
        ``self.parent_surface.previous_surface.material_post``. Views resolve
        their materials directly from the base surface (see
        :func:`resolve_view_materials`) rather than by walking a physical
        chain, so this returns a stand-in object exposing exactly that
        already-resolved value — always taking the non-``None`` branch,
        which keeps the first-step case correct too.
        """
        return SimpleNamespace(material_post=self.material_pre)

    # -- Tracing pipeline ----------------------------------------------------
    #
    # Every kernel below dispatches to the *base surface's class*, bound to
    # ``self`` (the view). This is deliberate, not cosmetic: some Surface
    # subclasses override these kernels — ObjectSurface.trace() skips
    # localize/globalize entirely (the object is often at z=-inf, where that
    # transform is undefined), and ImageSurface/ObjectSurface override the
    # paraxial and/or real-ray kernels to no-ops. Reimplementing the generic
    # Surface physics inline here would silently diverge from those
    # overrides for object/image steps. Dispatching through
    # ``type(self.base_surface)`` reuses whatever kernel actually applies,
    # bound to the view so it reads/writes the view's own state (geometry,
    # material_pre/post, aperture, interaction_model, and record buffers all
    # resolve through the view's own attributes) — the same "ride the
    # existing coordinator" approach the base tracing pipeline itself uses.

    def reset(self) -> None:
        """Resets the recorded information owned by this view."""
        type(self.base_surface).reset(self)

    def trace(self, rays):
        return type(self.base_surface).trace(self, rays)

    @property
    def _coordinator(self):
        from optiland.surfaces.standard_surface import _TracingCoordinator

        try:
            return self.__coordinator
        except AttributeError:
            self.__coordinator = _TracingCoordinator()
            return self.__coordinator

    def _trace_paraxial(self, rays: ParaxialRays) -> ParaxialRays:
        return type(self.base_surface)._trace_paraxial(self, rays)

    def _trace_real(self, rays: RealRays) -> RealRays:
        return type(self.base_surface)._trace_real(self, rays)

    def _record_paraxial(self, rays: ParaxialRays) -> None:
        type(self.base_surface)._record_paraxial(self, rays)

    def _record_real(self, rays: RealRays) -> None:
        type(self.base_surface)._record_real(self, rays)

    def __repr__(self) -> str:
        direction = "reverse" if self.reverse else "forward"
        override = self.interaction_override or "nominal"
        return (
            f"SurfaceView(base={self.base_surface!r}, {direction}, "
            f"interaction={override})"
        )
