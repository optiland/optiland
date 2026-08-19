"""CODE V Surface Handler Registry

Defines BaseSurfaceHandler and concrete per-surface-type handlers used by
both the reader (parse) and writer (format) paths. Adding support for a new
CODE V surface type requires only a new registered subclass.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.surfaces.standard_surface import Surface

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, BaseSurfaceHandler] = {}


def register(handler: type[BaseSurfaceHandler]) -> type[BaseSurfaceHandler]:
    """Class decorator to register a surface handler by its codev_type key.

    Args:
        handler: The handler class to register.

    Returns:
        The handler class unchanged (so it can still be used as a class).
    """
    _REGISTRY[handler.codev_type] = handler()
    return handler


def get_handler(codev_type: str) -> BaseSurfaceHandler:
    """Look up a registered handler by CODE V surface type string.

    Args:
        codev_type: The CODE V surface profile identifier (e.g. ``"SPH"``).

    Returns:
        The registered handler instance.

    Raises:
        NotImplementedError: If ``codev_type`` is not in the registry.
    """
    if codev_type not in _REGISTRY:
        raise NotImplementedError(
            f"CODE V surface type '{codev_type}' is not supported. "
            "Supported types: " + ", ".join(sorted(_REGISTRY))
        )
    return _REGISTRY[codev_type]


def get_handler_for_optiland_type(optiland_type: str) -> BaseSurfaceHandler:
    """Look up a registered handler by Optiland surface type string.

    Args:
        optiland_type: The Optiland surface type string
            (e.g. ``"standard"``).

    Returns:
        The registered handler instance.

    Raises:
        NotImplementedError: If no handler matches ``optiland_type``.
    """
    for handler in _REGISTRY.values():
        if handler.optiland_type == optiland_type:
            return handler
    raise NotImplementedError(
        f"No handler registered for Optiland surface type '{optiland_type}'. "
        "Supported types: " + ", ".join(h.optiland_type for h in _REGISTRY.values())
    )


# ---------------------------------------------------------------------------
# Base handler
# ---------------------------------------------------------------------------


class BaseSurfaceHandler(ABC):
    """Handles serialisation and deserialisation for one CODE V surface type.

    Subclasses must declare class-level ``codev_type`` and
    ``optiland_type`` strings and implement ``parse()`` and ``format()``.
    """

    codev_type: ClassVar[str]
    optiland_type: ClassVar[str]

    @abstractmethod
    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw CodeVDataParser surface dict to optiland surface kwargs.

        Args:
            raw: The raw surface dictionary produced by CodeVDataParser for
                one surface.

        Returns:
            A dict of keyword arguments suitable for ``optic.surfaces.add()``.
        """

    @abstractmethod
    def format(self, surface: Surface) -> dict[str, Any]:
        """Convert an Optiland Surface to a raw operand dict for CodeVFileEncoder.

        Args:
            surface: The Optiland Surface to serialise.

        Returns:
            A dict of raw CODE V operand values keyed by operand name
            (e.g. ``{"radius": 50.0, "conic": -1.0}``).
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _radius_from_raw(val: float) -> float:
    """Return a radius from a raw value, treating 0.0 as infinity (planar).

    Args:
        val: Raw radius value from the file.

    Returns:
        The radius, or ``float(be.inf)`` if ``val`` is 0.0.
    """
    if val == 0.0:
        return float(be.inf)
    return float(val)


def _raw_from_radius(radius: float) -> float:
    """Return the raw file value for a radius, treating infinity as 0.0.

    Args:
        radius: Optiland surface radius.

    Returns:
        The radius as a float, or 0.0 for infinite/planar surfaces.
    """
    if be.isinf(radius) or radius == 0:
        return 0.0
    return float(radius)


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------


@register
class StandardSurfaceHandler(BaseSurfaceHandler):
    """Handler for SPH (spherical/conic/planar) surfaces."""

    codev_type: ClassVar[str] = "SPH"
    optiland_type: ClassVar[str] = "standard"

    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Parse a SPH surface raw dict.

        Args:
            raw: Raw surface dict from CodeVDataParser.

        Returns:
            Kwargs for ``optic.surfaces.add()``.
        """
        return {
            "surface_type": self.optiland_type,
            "radius": raw.get("radius", float(be.inf)),
            "conic": raw.get("conic", 0.0),
        }

    def format(self, surface: Surface) -> dict[str, Any]:
        """Format a standard surface to CODE V operand dict.

        Args:
            surface: The Optiland surface.

        Returns:
            Raw operand dict for CodeVFileEncoder.
        """
        geom = surface.geometry
        return {
            "profile": self.codev_type,
            "radius": _raw_from_radius(float(geom.radius)),
            "conic": float(getattr(geom, "k", 0.0)),
        }


@register
class EvenAsphereSurfaceHandler(BaseSurfaceHandler):
    """Handler for ASP (even asphere) surfaces."""

    codev_type: ClassVar[str] = "ASP"
    optiland_type: ClassVar[str] = "even_asphere"

    # Ordered coefficient keys as used by the encoder/parser (4th–20th order).
    _COEFF_KEYS: ClassVar[tuple[str, ...]] = (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "J",
    )

    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Parse an ASP surface raw dict.

        Args:
            raw: Raw surface dict from CodeVDataParser.

        Returns:
            Kwargs for ``optic.surfaces.add()``.
        """
        coeffs = [raw.get(f"coeff_{k}", 0.0) for k in range(len(self._COEFF_KEYS))]
        return {
            "surface_type": self.optiland_type,
            "radius": raw.get("radius", float(be.inf)),
            "conic": raw.get("conic", 0.0),
            "coefficients": coeffs,
        }

    def format(self, surface: Surface) -> dict[str, Any]:
        """Format an even-asphere surface to CODE V operand dict.

        Args:
            surface: The Optiland surface.

        Returns:
            Raw operand dict for CodeVFileEncoder.
        """
        geom = surface.geometry
        coeffs: list[float] = list(geom.coefficients) if geom.coefficients else []
        # Pad to 9 terms (A–J)
        while len(coeffs) < len(self._COEFF_KEYS):
            coeffs.append(0.0)

        result: dict[str, Any] = {
            "profile": self.codev_type,
            "radius": _raw_from_radius(float(geom.radius)),
            "conic": float(getattr(geom, "k", 0.0)),
        }
        for i, _key in enumerate(self._COEFF_KEYS):
            result[f"coeff_{i}"] = float(coeffs[i])
        return result
