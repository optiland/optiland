"""Interaction Model Factory

This module contains the InteractionModelFactory class, which is used to create
interaction model objects based on the given parameters.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.interactions.diffractive_model import DiffractiveInteractionModel
from optiland.interactions.phase_interaction_model import PhaseInteractionModel
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel

if TYPE_CHECKING:
    from collections.abc import Callable  # pragma: no cover

    from optiland.coatings import BaseCoating
    from optiland.interactions.base import BaseInteractionModel
    from optiland.scatter import BaseBSDF
    from optiland.surfaces import Surface


def _build_refractive_reflective(
    parent_surface: Surface | None,
    is_reflective: bool,
    coating: BaseCoating | None,
    bsdf: BaseBSDF | None,
    **_,
) -> RefractiveReflectiveModel:
    return RefractiveReflectiveModel(
        parent_surface=parent_surface,
        is_reflective=is_reflective,
        coating=coating,
        bsdf=bsdf,
    )


def _build_thin_lens(
    parent_surface: Surface | None,
    is_reflective: bool,
    coating: BaseCoating | None,
    bsdf: BaseBSDF | None,
    focal_length: float | None = None,
    **_,
) -> ThinLensInteractionModel:
    if focal_length is None:
        raise ValueError("Focal length is required for thin lens.")
    return ThinLensInteractionModel(
        parent_surface=parent_surface,
        focal_length=focal_length,
        is_reflective=is_reflective,
        coating=coating,
        bsdf=bsdf,
    )


def _build_diffractive(
    parent_surface: Surface | None,
    is_reflective: bool,
    coating: BaseCoating | None,
    bsdf: BaseBSDF | None,
    **_,
) -> DiffractiveInteractionModel:
    return DiffractiveInteractionModel(
        parent_surface=parent_surface,
        is_reflective=is_reflective,
        coating=coating,
        bsdf=bsdf,
    )


def _build_phase(
    parent_surface: Surface | None,
    is_reflective: bool,
    coating: BaseCoating | None,
    bsdf: BaseBSDF | None,
    phase_profile=None,
    **_,
) -> PhaseInteractionModel:
    if phase_profile is None:
        raise ValueError("phase_profile is required for phase interaction.")
    return PhaseInteractionModel(
        parent_surface=parent_surface,
        phase_profile=phase_profile,
        is_reflective=is_reflective,
        coating=coating,
        bsdf=bsdf,
    )


_INTERACTION_REGISTRY: dict[str, Callable] = {
    "refractive_reflective": _build_refractive_reflective,
    "thin_lens": _build_thin_lens,
    "diffractive": _build_diffractive,
    "phase": _build_phase,
}


class InteractionModelFactory:
    """A factory class for creating interaction model objects."""

    @classmethod
    def register(
        cls,
        name: str,
        builder: Callable,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a new interaction model builder.

        Args:
            name: The string key used when specifying interaction_type.
            builder: A callable with signature
                ``(parent_surface, is_reflective, coating, bsdf, **kwargs)``
                that returns a ``BaseInteractionModel`` instance.
            overwrite: Allow replacing an existing registration.

        Raises:
            ValueError: If name is already registered and overwrite is False.
        """
        if name in _INTERACTION_REGISTRY and not overwrite:
            raise ValueError(
                f"Interaction model '{name}' is already registered. "
                "Pass overwrite=True to replace it."
            )
        _INTERACTION_REGISTRY[name] = builder

    def create(
        self,
        parent_surface: Surface | None,
        interaction_type: str,
        is_reflective: bool,
        coating: BaseCoating | None,
        bsdf: BaseBSDF | None,
        **kwargs,
    ) -> BaseInteractionModel:
        """Creates an interaction model object based on the given parameters.

        Args:
            parent_surface: The parent surface (hooked up later in Surface.__init__).
            interaction_type (str): The type of interaction model to create.
            is_reflective (bool): Indicates whether the surface is reflective.
            coating (Optional[BaseCoating]): The coating of the surface.
            bsdf (Optional[BaseBSDF]): The BSDF of the surface.
            **kwargs: Additional keyword arguments forwarded to the builder
                (e.g. ``focal_length`` for thin_lens, ``phase_profile`` for phase).

        Returns:
            BaseInteractionModel: The created interaction model object.

        Raises:
            ValueError: If the interaction_type is unknown.
        """
        if interaction_type not in _INTERACTION_REGISTRY:
            raise ValueError(f"Unknown interaction_type: {interaction_type!r}")
        builder = _INTERACTION_REGISTRY[interaction_type]
        return builder(
            parent_surface=parent_surface,
            is_reflective=is_reflective,
            coating=coating,
            bsdf=bsdf,
            **kwargs,
        )
