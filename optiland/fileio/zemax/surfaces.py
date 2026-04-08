"""Zemax Surface Handler Registry

Defines BaseSurfaceHandler and concrete per-surface-type handlers used by
both the reader (parse) and writer (format) paths. Adding support for a new
Zemax surface type requires only a new registered subclass.

Kramer Harrison, 2024
"""

from __future__ import annotations

import math
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
    """Class decorator to register a surface handler by its zemax_type key.

    Args:
        handler: The handler class to register.

    Returns:
        The handler class unchanged (so it can still be used as a class).
    """
    _REGISTRY[handler.zemax_type] = handler()
    return handler


def get_handler(zemax_type: str) -> BaseSurfaceHandler:
    """Look up a registered handler by Zemax surface type string.

    Args:
        zemax_type: The Zemax surface type identifier (e.g. ``"STANDARD"``).

    Returns:
        The registered handler instance.

    Raises:
        NotImplementedError: If ``zemax_type`` is not in the registry.
    """
    if zemax_type not in _REGISTRY:
        raise NotImplementedError(
            f"Zemax surface type '{zemax_type}' is not supported. "
            "Supported types: " + ", ".join(sorted(_REGISTRY))
        )
    return _REGISTRY[zemax_type]


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
    """Handles serialisation and deserialisation for one Zemax surface type.

    Subclasses must declare class-level ``zemax_type`` and
    ``optiland_type`` strings and implement ``parse()`` and ``format()``.
    """

    zemax_type: ClassVar[str]
    optiland_type: ClassVar[str]

    @abstractmethod
    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw ZemaxDataParser surface dict to optiland surface kwargs.

        Args:
            raw: The raw surface dictionary produced by ZemaxDataParser for
                one surface.

        Returns:
            A dict of keyword arguments suitable for ``optic.surfaces.add()``.
        """

    @abstractmethod
    def format(self, surface: Surface) -> dict[str, Any]:
        """Convert an Optiland Surface to a raw operand dict for ZemaxFileEncoder.

        Args:
            surface: The Optiland Surface to serialise.

        Returns:
            A dict of raw Zemax operand values keyed by operand name
            (e.g. ``{"CURV": 0.01, "CONI": -1.0, "DISZ": 5.0, ...}``).
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _curvature(radius: float) -> float:
    """Return Zemax curvature (1/R), handling infinite radius as 0.0."""
    if be.isinf(radius) or radius == 0:
        return 0.0
    return 1.0 / float(radius)


def _radius(curvature: float) -> float:
    """Return radius from Zemax curvature, handling 0 as infinity."""
    if curvature == 0.0:
        return float(be.inf)
    return 1.0 / curvature


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------


@register
class StandardSurfaceHandler(BaseSurfaceHandler):
    """Handler for STANDARD (spherical/conic) surfaces."""

    zemax_type: ClassVar[str] = "STANDARD"
    optiland_type: ClassVar[str] = "standard"

    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Parse a STANDARD surface raw dict.

        Args:
            raw: Raw surface dict from ZemaxDataParser.

        Returns:
            Kwargs for ``optic.surfaces.add()``.
        """
        return {
            "surface_type": self.optiland_type,
            "radius": raw.get("radius", float(be.inf)),
            "conic": raw.get("conic", 0.0),
        }

    def format(self, surface: Surface) -> dict[str, Any]:
        """Format a standard surface to Zemax operand dict.

        Args:
            surface: The Optiland surface.

        Returns:
            Raw operand dict for ZemaxFileEncoder.
        """
        geom = surface.geometry
        return {
            "TYPE": self.zemax_type,
            "CURV": _curvature(float(geom.radius)),
            "CONI": float(getattr(geom, "k", 0.0)),
        }


@register
class EvenAsphereSurfaceHandler(BaseSurfaceHandler):
    """Handler for EVENASPH (even asphere) surfaces."""

    zemax_type: ClassVar[str] = "EVENASPH"
    optiland_type: ClassVar[str] = "even_asphere"

    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Parse an EVENASPH surface raw dict.

        Args:
            raw: Raw surface dict from ZemaxDataParser.

        Returns:
            Kwargs for ``optic.surfaces.add()``.
        """
        coeffs = [raw.get(f"param_{k}", 0.0) for k in range(8)]
        return {
            "surface_type": self.optiland_type,
            "radius": raw.get("radius", float(be.inf)),
            "conic": raw.get("conic", 0.0),
            "coefficients": coeffs,
        }

    def format(self, surface: Surface) -> dict[str, Any]:
        """Format an even-asphere surface to Zemax operand dict.

        Args:
            surface: The Optiland surface.

        Returns:
            Raw operand dict for ZemaxFileEncoder.
        """
        geom = surface.geometry
        coeffs: list[float] = list(geom.coefficients) if geom.coefficients else []
        # Pad to 8 terms
        while len(coeffs) < 8:
            coeffs.append(0.0)
        result: dict[str, Any] = {
            "TYPE": self.zemax_type,
            "CURV": _curvature(float(geom.radius)),
            "CONI": float(getattr(geom, "k", 0.0)),
        }
        for i, c in enumerate(coeffs[:8]):
            result[f"PARM_{i + 1}"] = float(c)
        return result


@register
class OddAsphereSurfaceHandler(BaseSurfaceHandler):
    """Handler for ODDASPHE (odd asphere) surfaces."""

    zemax_type: ClassVar[str] = "ODDASPHE"
    optiland_type: ClassVar[str] = "odd_asphere"

    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Parse an ODDASPHE surface raw dict.

        Args:
            raw: Raw surface dict from ZemaxDataParser.

        Returns:
            Kwargs for ``optic.surfaces.add()``.
        """
        coeffs = [raw.get(f"param_{k}", 0.0) for k in range(8)]
        return {
            "surface_type": self.optiland_type,
            "radius": raw.get("radius", float(be.inf)),
            "conic": raw.get("conic", 0.0),
            "coefficients": coeffs,
        }

    def format(self, surface: Surface) -> dict[str, Any]:
        """Format an odd-asphere surface to Zemax operand dict.

        Args:
            surface: The Optiland surface.

        Returns:
            Raw operand dict for ZemaxFileEncoder.
        """
        geom = surface.geometry
        coeffs: list[float] = list(geom.coefficients) if geom.coefficients else []
        while len(coeffs) < 8:
            coeffs.append(0.0)
        result: dict[str, Any] = {
            "TYPE": self.zemax_type,
            "CURV": _curvature(float(geom.radius)),
            "CONI": float(getattr(geom, "k", 0.0)),
        }
        for i, c in enumerate(coeffs[:8]):
            result[f"PARM_{i + 1}"] = float(c)
        return result


@register
class CoordinateBreakSurfaceHandler(BaseSurfaceHandler):
    """Handler for COORDBRK (coordinate break) surfaces.

    In the read direction, COORDBRK surfaces are consumed by the converter
    to build cumulative coordinate systems; they do not become Optiland
    surfaces directly.

    In the write direction, this handler is called internally by
    OpticToZemaxConverter to synthesise COORDBRK entries for surfaces that
    carry a non-trivial coordinate system.
    """

    zemax_type: ClassVar[str] = "COORDBRK"
    optiland_type: ClassVar[str] = "coordinate_break"

    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Parse a COORDBRK surface raw dict.

        Args:
            raw: Raw surface dict from ZemaxDataParser.

        Returns:
            Kwargs dict (coordinate breaks are handled specially by the
            converter and this result is not used directly).
        """
        return {
            "surface_type": self.optiland_type,
            "dx": raw.get("param_0", 0.0),
            "dy": raw.get("param_1", 0.0),
            "rx": be.deg2rad(be.array(raw.get("param_2", 0.0))),
            "ry": be.deg2rad(be.array(raw.get("param_3", 0.0))),
            "rz": be.deg2rad(be.array(raw.get("param_4", 0.0))),
            "order_flag": raw.get("param_5", 0.0),
        }

    def format(self, surface: Surface) -> dict[str, Any]:  # pragma: no cover
        """Format a coordinate-break entry.

        This method is not called for real Optiland surfaces; coordinate
        breaks are synthesised by OpticToZemaxConverter based on the
        surface's coordinate system.

        Args:
            surface: Unused.

        Returns:
            Empty dict.
        """
        return {}

    def format_cs(
        self,
        dx: float,
        dy: float,
        dz: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
        order: int = 0,
    ) -> dict[str, Any]:
        """Build a COORDBRK raw operand dict from explicit decenter/tilt values.

        Args:
            dx: X decenter in lens units.
            dy: Y decenter in lens units.
            dz: Z thickness of the coordinate break.
            rx_deg: X tilt in degrees.
            ry_deg: Y tilt in degrees.
            rz_deg: Z tilt in degrees.
            order: Order flag (0 = decenter then tilt; 1 = tilt then decenter).

        Returns:
            Raw operand dict for ZemaxFileEncoder.
        """
        return {
            "TYPE": self.zemax_type,
            "CURV": 0.0,
            "DISZ": dz,
            "PARM_1": dx,
            "PARM_2": dy,
            "PARM_3": rx_deg,
            "PARM_4": ry_deg,
            "PARM_5": rz_deg,
            "PARM_6": float(order),
        }


@register
class ToroidalSurfaceHandler(BaseSurfaceHandler):
    """Handler for TOROIDAL surfaces."""

    zemax_type: ClassVar[str] = "TOROIDAL"
    optiland_type: ClassVar[str] = "toroidal"

    def parse(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Parse a TOROIDAL surface raw dict.

        In Zemax, CURV is the Y-radius and PARM 1 is the X-radius (R_rot).
        Polynomial coefficients start at PARM 2.

        Args:
            raw: Raw surface dict from ZemaxDataParser.

        Returns:
            Kwargs for ``optic.surfaces.add()``.
        """
        radius_x = raw.get("param_1", 0.0)
        if radius_x == 0.0:
            radius_x = float(be.inf)
        coeffs = [raw.get(f"param_{k}", 0.0) for k in range(2, 10)]
        return {
            "surface_type": self.optiland_type,
            "radius_y": raw.get("radius", float(be.inf)),
            "radius_x": radius_x,
            "conic": raw.get("conic", 0.0),
            "toroidal_coeffs_poly_y": coeffs,
        }

    def format(self, surface: Surface) -> dict[str, Any]:
        """Format a toroidal surface to Zemax operand dict.

        Args:
            surface: The Optiland surface.

        Returns:
            Raw operand dict for ZemaxFileEncoder.
        """
        geom = surface.geometry
        r_yz = float(geom.R_yz)
        r_rot = float(geom.R_rot)

        # R_rot == inf -> PARM 1 = 0 in Zemax convention
        parm1 = 0.0 if (math.isinf(r_rot) or r_rot == 0.0) else (1.0 / r_rot)

        raw_coeffs = geom.coeffs_poly_y
        coeffs: list[float] = (
            [float(c) for c in raw_coeffs] if raw_coeffs is not None else []
        )
        while len(coeffs) < 8:
            coeffs.append(0.0)

        result: dict[str, Any] = {
            "TYPE": self.zemax_type,
            "CURV": _curvature(r_yz),
            "CONI": float(getattr(geom, "k_yz", 0.0)),
            "PARM_1": parm1,
        }
        for i, c in enumerate(coeffs[:8]):
            result[f"PARM_{i + 2}"] = float(c)
        return result
