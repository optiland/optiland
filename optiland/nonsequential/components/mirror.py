"""Mirror compound component for Non-Sequential Raytracing.

A single reflective surface (paraboloid, spherical, flat, etc.).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.nonsequential.components.compound import CompoundComponent
from optiland.nonsequential.components.configs import InteractionType, MirrorConfig
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.lens import _make_surface, _resolve_interaction
from optiland.nonsequential.materials.nsq_material import VACUUM

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.base import BaseComponent


class Mirror(CompoundComponent):
    """Single reflective mirror surface.

    Wraps a :class:`ConicGeometry` (or any overridden geometry) inside a
    :class:`ReflectiveComponent`.  The user can attach a custom BSDF via
    ``MirrorConfig.surface.bsdf``.

    Unlike ``Lens``/``Doublet``, this is not wrapped in a
    :class:`~optiland.nonsequential.components.volume.Volume`: a mirror is a
    single open reflective surface with vacuum on both sides, not a closed
    solid with an interior medium, so watertightness has nothing to check.

    Attributes:
        _name: Registry name.
        _cs: Surface coordinate system.
        _config: MirrorConfig.
        _surfaces: Single-element list containing the reflective surface.
    """

    def __init__(
        self,
        name: str,
        cs: CoordinateSystem,
        config: MirrorConfig,
    ) -> None:
        """Initialize Mirror from a MirrorConfig.

        Args:
            name: Unique identifier for this mirror in the registry.
            cs: Coordinate system for the mirror surface.
            config: Mirror geometry configuration.
        """
        self._name = name
        self._cs = cs
        self._config = config
        self._surfaces: list[BaseComponent] = self._build()

    @property
    def name(self) -> str:
        """Registry name of this mirror."""
        return self._name

    @property
    def surfaces(self) -> list[BaseComponent]:
        """Single-element list containing the reflective surface."""
        return self._surfaces

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Mirror surface coordinate system."""
        return self._cs

    def _build(self) -> list[BaseComponent]:
        """Construct the reflective surface.

        Returns:
            List containing exactly one ReflectiveComponent.
        """
        cfg = self._config
        geom = ConicGeometry(cfg.radius, cfg.conic, cfg.aperture_radius)
        interaction = _resolve_interaction(cfg.surface, InteractionType.REFLECTIVE)
        return [
            _make_surface(
                self._cs,
                geom,
                VACUUM,
                VACUUM,
                cfg.surface,
                interaction,
                f"{self._name}.surface",
                reflectance=cfg.reflectance,
            )
        ]
