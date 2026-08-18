"""Sequence resolution and consistency validation.

Turns a list of base surfaces plus raw steps into a validated list of
:class:`~optiland.sequences.surface_view.SurfaceView` objects. Construction
fails loudly (``SequenceValidationError``) if adjacent steps are not
physically consistent, rather than tracing plausible nonsense.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.sequences.steps import SequenceStep, parse_steps
from optiland.sequences.surface_view import SurfaceView, resolve_view_materials

if TYPE_CHECKING:
    from optiland.sequences.steps import RawStep
    from optiland.surfaces.standard_surface import Surface


class SequenceValidationError(ValueError):
    """Raised when a sequence's steps are not physically consistent."""


def _is_step_reflective(step: SequenceStep, base_surface: Surface) -> bool:
    """Whether a step reflects, considering both overrides and nominal mirrors."""
    if step.interaction_override == "reflect":
        return True
    if step.interaction_override == "refract":
        return False
    return getattr(base_surface.interaction_model, "is_reflective", False)


def _effective_exit_material(step: SequenceStep, base_surfaces: list[Surface]):
    """The material a ray is actually in when it leaves this step.

    Equal to ``material_post`` for refractive/nominal steps, and to
    ``material_pre`` for reflective steps, since a reflection never crosses
    into the far medium.
    """
    base_surface = base_surfaces[step.index]
    pre, post = resolve_view_materials(
        base_surface, step.reverse, step.interaction_override
    )
    is_reflective = _is_step_reflective(step, base_surface)
    return pre if is_reflective else post


def validate_sequence(steps: list[SequenceStep], base_surfaces: list[Surface]) -> None:
    """Validate that adjacent steps share a consistent medium at their join.

    Args:
        steps: The resolved sequence steps.
        base_surfaces: The optic's base surfaces, indexed as in ``steps``.

    Raises:
        SequenceValidationError: If the exit medium of some step does not
            equal the incident medium of the following step, naming the
            offending step index.
    """
    for i in range(len(steps) - 1):
        current, following = steps[i], steps[i + 1]

        exit_material = _effective_exit_material(current, base_surfaces)
        incident_material, _ = resolve_view_materials(
            base_surfaces[following.index],
            following.reverse,
            following.interaction_override,
        )

        if exit_material != incident_material:
            raise SequenceValidationError(
                f"Sequence step {i + 1} (surface {following.index}) expects "
                f"incident medium {incident_material!r}, but step {i} "
                f"(surface {current.index}) exits into {exit_material!r}."
            )


def resolve_sequence(
    base_surfaces: list[Surface], raw_steps: list[RawStep]
) -> list[SurfaceView]:
    """Resolve a raw step list into a validated list of ``SurfaceView``.

    Args:
        base_surfaces: The optic's base surfaces, indexed by the step
            indices used in ``raw_steps``.
        raw_steps: The raw sequence, as bare surface indices and/or
            ``(index, interaction_override)`` pairs. See
            :func:`optiland.sequences.steps.parse_steps`.

    Returns:
        The resolved, validated sequence of views, in traversal order.

    Raises:
        ValueError: If ``raw_steps`` is empty, malformed, or references an
            out-of-range surface index.
        SequenceValidationError: If adjacent steps are not physically
            consistent.
    """
    steps = parse_steps(raw_steps)

    for step in steps:
        if not (0 <= step.index < len(base_surfaces)):
            raise ValueError(
                f"Sequence step references surface index {step.index}, but "
                f"the optic only has {len(base_surfaces)} surfaces."
            )

    # Refine reverse direction inference taking nominal mirrors into account
    refined_steps: list[SequenceStep] = []
    reverse = False
    for step in steps:
        base_surface = base_surfaces[step.index]
        refined_steps.append(
            SequenceStep(
                index=step.index,
                reverse=reverse,
                interaction_override=step.interaction_override,
            )
        )
        if _is_step_reflective(step, base_surface):
            reverse = not reverse

    steps = refined_steps

    validate_sequence(steps, base_surfaces)

    views: list[SurfaceView] = []
    previous_view: SurfaceView | None = None
    for step in steps:
        view = SurfaceView(
            base_surface=base_surfaces[step.index],
            reverse=step.reverse,
            interaction_override=step.interaction_override,
            previous_view=previous_view,
        )
        views.append(view)
        previous_view = view

    return views
