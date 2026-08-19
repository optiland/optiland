"""Sequence step parsing.

A sequence is a list of *raw steps*, each either a bare surface index
(forward traversal, nominal interaction) or a ``(index, interaction_override)``
pair, e.g.::

    steps = [0, 1, 2, (3, "reflect"), (2, "reflect"), 3, 4]

The propagation direction (forward/reverse) is not specified explicitly by
the user. It starts forward and flips every time a step reflects, since a
reflection is what reverses the physical direction of travel. This module
turns the raw, ergonomic step list into fully-resolved :class:`SequenceStep`
objects that carry that inferred direction explicitly.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VALID_OVERRIDES = frozenset({"reflect", "refract"})

RawStep = int | tuple[int, str] | list[Any]


@dataclass(frozen=True)
class SequenceStep:
    """A single, fully-resolved step in a surface sequence.

    Args:
        index: Index of the base surface in the optic's nominal surface
            list.
        reverse: Whether this step is traversed in the reverse physical
            direction (light travelling from the base surface's nominal
            "post" side toward its "pre" side).
        interaction_override: ``"reflect"`` or ``"refract"`` to force the
            interaction type at this step, or ``None`` to use the base
            surface's nominal interaction.
    """

    index: int
    reverse: bool = False
    interaction_override: str | None = None


def _parse_raw_step(raw: RawStep) -> tuple[int, str | None]:
    """Split a raw step into ``(index, interaction_override)``."""
    if isinstance(raw, tuple | list):
        if len(raw) != 2:
            raise ValueError(
                f"Step pair must be (index, interaction_override), got {raw!r}."
            )
        index, override = raw
        if not isinstance(index, int):
            raise ValueError(
                f"Surface index must be an int, got {index!r} of type "
                f"{type(index).__name__}."
            )
        if not isinstance(override, str) or override not in VALID_OVERRIDES:
            raise ValueError(
                f"Unknown interaction override {override!r} at step for surface "
                f"{index}; expected one of {sorted(VALID_OVERRIDES)}."
            )
        return index, override

    if not isinstance(raw, int):
        raise ValueError(
            f"Invalid raw step {raw!r}. Expected an int surface index or a "
            "(index, interaction_override) pair."
        )

    return raw, None


def parse_steps(raw_steps: list[RawStep]) -> list[SequenceStep]:
    """Parse a raw step list into resolved :class:`SequenceStep` objects.

    Direction is inferred: it starts forward (``reverse=False``) and flips
    after every step whose resolved interaction is a reflection, since that
    is the point at which the physical direction of propagation reverses.

    Args:
        raw_steps: The raw sequence, as bare surface indices and/or
            ``(index, interaction_override)`` pairs.

    Returns:
        The resolved sequence steps, in order.

    Raises:
        ValueError: If ``raw_steps`` is empty or contains an invalid step.
    """
    if not raw_steps:
        raise ValueError("A sequence must contain at least one step.")

    steps = []
    reverse = False
    for raw in raw_steps:
        index, override = _parse_raw_step(raw)
        steps.append(
            SequenceStep(index=index, reverse=reverse, interaction_override=override)
        )
        if override == "reflect":
            reverse = not reverse

    return steps
