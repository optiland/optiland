"""Multi-sequence tracing.

Sub-sequences are alternative traversal orders over the same ``Surface``
objects of an ``Optic`` (ghost paths, reverse traces, sub-component views).
See ``SPEC_multi_sequence_20260731.md`` for the full design.

Kramer Harrison, 2026
"""

from __future__ import annotations

from optiland.sequences.resolver import SequenceValidationError, resolve_sequence
from optiland.sequences.steps import SequenceStep, parse_steps
from optiland.sequences.surface_view import SurfaceView

__all__ = [
    "SequenceStep",
    "SequenceValidationError",
    "SurfaceView",
    "parse_steps",
    "resolve_sequence",
]
