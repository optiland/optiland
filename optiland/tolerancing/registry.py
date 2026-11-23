"""Tolerance Registry Module

This module contains the base class for tolerance types and the registry that
maps tolerance type names, as used by ``Tolerancing.add_tolerance``, to their
classes. Register a ``ToleranceType`` subclass to add a new kind of tolerance
without changing core code.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from optiland.tolerancing.base import Phase

if TYPE_CHECKING:
    from optiland.optic import Optic


class ToleranceType(ABC):
    """Applies a signed deviation from nominal to one property of an optic.

    Subclasses read the nominal state in ``__init__`` and must restore it
    exactly in ``reset``.

    Args:
        optic: The optic the tolerance applies to.
        **kwargs: Type-specific arguments, such as ``surface_number``.

    Attributes:
        phase: The phase in which the deviation is applied.
        units: Units of the deviation, shown in docs and result tables.
    """

    phase: Phase = Phase.STRUCTURE
    units: str = ""

    def __init__(self, optic: Optic, **kwargs) -> None:
        self.optic = optic

    @property
    @abstractmethod
    def label(self) -> str:
        """Return a short, human-readable name, e.g. ``"Radius S1"``."""

    @abstractmethod
    def apply(self, delta: float) -> None:
        """Apply a deviation from nominal.

        Args:
            delta: The signed deviation, in ``units``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Restore the nominal state."""


class ToleranceRegistry:
    """Registry of tolerance types, keyed by name."""

    def __init__(self) -> None:
        self._types: dict[str, type[ToleranceType]] = {}

    def register(
        self, name: str, tolerance_cls: type[ToleranceType], overwrite: bool = False
    ) -> None:
        """Register a tolerance type.

        Args:
            name: The name used in ``Tolerancing.add_tolerance``.
            tolerance_cls: A ``ToleranceType`` subclass.
            overwrite: Replace an existing registration with the same name.

        Raises:
            TypeError: If ``tolerance_cls`` is not a ``ToleranceType`` subclass.
            ValueError: If ``name`` is already registered and ``overwrite`` is
                False.
        """
        if not (
            isinstance(tolerance_cls, type) and issubclass(tolerance_cls, ToleranceType)
        ):
            raise TypeError("tolerance_cls must be a ToleranceType subclass.")
        if name in self._types and not overwrite:
            raise ValueError(f"Tolerance type {name!r} is already registered.")
        self._types[name] = tolerance_cls

    def get(self, name: str) -> type[ToleranceType]:
        """Return the class registered under ``name``.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        try:
            return self._types[name]
        except KeyError:
            valid = ", ".join(sorted(self._types))
            raise ValueError(
                f"Unknown tolerance type {name!r}. Valid options: {valid}."
            ) from None

    def names(self) -> list[str]:
        """Return the registered names, sorted."""
        return sorted(self._types)

    def __contains__(self, name: str) -> bool:
        return name in self._types


tolerance_registry = ToleranceRegistry()
