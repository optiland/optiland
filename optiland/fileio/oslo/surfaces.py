"""OSLO Surface Handler Registry

Defines BaseSurfaceHandler and concrete per-surface-type handlers used by
both the reader (parse) and writer (format) paths.

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
    """Class decorator to register a surface handler by its oslo_type key."""
    _REGISTRY[handler.oslo_type] = handler()
    return handler


def get_handler(oslo_type: str) -> BaseSurfaceHandler:
    """Look up a registered handler by OSLO surface type string."""
    if oslo_type not in _REGISTRY:
        raise NotImplementedError(
            f"OSLO surface type '{oslo_type}' is not supported. "
            "Supported types: " + ", ".join(sorted(_REGISTRY))
        )
    return _REGISTRY[oslo_type]


def get_handler_for_optiland_type(optiland_type: str) -> BaseSurfaceHandler:
    """Look up a registered handler by Optiland surface type string."""
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
    """Handles serialisation and deserialisation for one OSLO surface type."""

    oslo_type: ClassVar[str]
    optiland_type: ClassVar[str]

    @abstractmethod
    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw OsloDataParser surface dict to optiland surface kwargs."""

    @abstractmethod
    def format(self, surface: Surface) -> dict[str, Any]:
        """Convert an Optiland Surface to a raw command dict for OsloDataFormatter."""


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------


@register
class StandardSurfaceHandler(BaseSurfaceHandler):
    """Handler for standard (spherical/conic) surfaces."""

    oslo_type: ClassVar[str] = "standard"
    optiland_type: ClassVar[str] = "standard"

    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "surface_type": self.optiland_type,
            "radius": raw.get("RD", float(be.inf)),
            "conic": raw.get("CC", 0.0),
        }

    def format(self, surface: Surface) -> dict[str, Any]:
        geom = surface.geometry
        return {
            "RD": float(geom.radius),
            "CC": float(getattr(geom, "k", 0.0)),
        }


@register
class EvenAsphereSurfaceHandler(BaseSurfaceHandler):
    """Handler for even asphere surfaces (ad, ae, af, ag)."""

    oslo_type: ClassVar[str] = "even_asphere"
    optiland_type: ClassVar[str] = "even_asphere"

    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        # OSLO ad=4th, ae=6th, af=8th, ag=10th
        coeffs = [
            raw.get("AD", 0.0),
            raw.get("AE", 0.0),
            raw.get("AF", 0.0),
            raw.get("AG", 0.0),
        ]
        return {
            "surface_type": self.optiland_type,
            "radius": raw.get("RD", float(be.inf)),
            "conic": raw.get("CC", 0.0),
            "coefficients": coeffs,
        }

    def format(self, surface: Surface) -> dict[str, Any]:
        geom = surface.geometry
        coeffs = list(geom.coefficients) if geom.coefficients else []
        while len(coeffs) < 4:
            coeffs.append(0.0)
        return {
            "RD": float(geom.radius),
            "CC": float(getattr(geom, "k", 0.0)),
            "AD": float(coeffs[0]),
            "AE": float(coeffs[1]),
            "AF": float(coeffs[2]),
            "AG": float(coeffs[3]),
        }
