"""Base Perturbation Module

This module defines the interface shared by everything a tolerancing analysis
can apply to an optic: the low-level ``Perturbation`` (absolute values of an
optimization variable) and the ``Tolerance`` (signed deviations from nominal
within limits). Analyses depend on this interface only.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum


class Phase(IntEnum):
    """Order in which perturbations are applied within one trial.

    Perturbations are applied phase by phase, and in insertion order within a
    phase. Structural changes (radius, thickness) come first because a
    thickness change rebuilds downstream surface positions; rigid-body moves
    (decenters and tilts) come last so they are not overwritten.
    """

    STRUCTURE = 0
    MATERIAL = 1
    FORM = 2
    RIGID_BODY = 3


class BasePerturbation(ABC):
    """Interface for a change that a tolerancing analysis applies to an optic.

    Attributes:
        phase: The phase in which the perturbation is applied.
        value: The value applied by the last call to ``apply``, or the nominal
            value after ``reset``.
    """

    phase: Phase = Phase.STRUCTURE
    value: float | None = None

    @property
    @abstractmethod
    def label(self) -> str:
        """Return a short, human-readable name used in result tables."""

    @abstractmethod
    def apply(self) -> None:
        """Draw a value and apply it to the optic."""

    @abstractmethod
    def reset(self) -> None:
        """Restore the optic to its nominal state for this perturbation."""
