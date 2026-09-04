"""Doublet compound component for Non-Sequential Raytracing.

A cemented achromatic doublet: front face, cemented interface, back face,
edge (cylindrical frustum).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.nonsequential.components.compound import CompoundComponent
from optiland.nonsequential.components.configs import DoubletConfig, InteractionType
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.geometry.analytic.frustum import (
    CylindricalFrustumGeometry,
)
from optiland.nonsequential.components.lens import (
    _make_surface,
    _offset_cs,
    _resolve_material,
    _sag_at_rim,
)
from optiland.nonsequential.components.volume import Volume
from optiland.nonsequential.materials.nsq_material import VACUUM

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.base import BaseComponent


class Doublet(CompoundComponent):
    """Cemented achromatic doublet.

    Assembles five physical surfaces:

    1. **Front face** -- refractive, conic (crown element).
    2. **Cemented interface** -- refractive, conic (crown->flint).
    3. **Back face** -- refractive, conic (flint element).
    4. **Crown edge** -- cylindrical frustum, absorbing.
    5. **Flint edge** -- cylindrical frustum, absorbing.

    Unlike ``Lens``, a doublet is two closed solids (the crown element and
    the flint element) sharing the cemented interface as a common boundary
    surface -- so the edge is split into a crown segment and a flint
    segment at the cemented interface's axial position, and each element
    is validated as its own :class:`Volume`.

    Attributes:
        _name: Registry name.
        _cs: Front-vertex coordinate system.
        _config: DoubletConfig.
        _surfaces: Built list of sub-surfaces.
        _volumes: The two validated ``Volume`` instances (crown, flint).
    """

    def __init__(
        self,
        name: str,
        cs: CoordinateSystem,
        config: DoubletConfig,
    ) -> None:
        """Initialize Doublet from a DoubletConfig.

        Args:
            name: Unique identifier in the registry.
            cs: Front-vertex coordinate system.
            config: Doublet geometry and material configuration.

        Raises:
            NonWatertightVolumeError: If either element's surfaces do not
                form a closed, consistently outward-oriented solid.
        """
        self._name = name
        self._cs = cs
        self._config = config
        mat1 = _resolve_material(config.material1)
        mat2 = _resolve_material(config.material2)
        self._surfaces: list[BaseComponent] = self._build()
        front, cemented, back, edge_crown, edge_flint = self._surfaces
        self._volumes = [
            Volume(
                name=f"{name}.crown",
                boundary=[front, cemented, edge_crown],
                interior=mat1,
            ),
            Volume(
                name=f"{name}.flint",
                boundary=[cemented, back, edge_flint],
                interior=mat2,
            ),
        ]

    @property
    def name(self) -> str:
        """Registry name."""
        return self._name

    @property
    def surfaces(self) -> list[BaseComponent]:
        """Ordered flat list of sub-surfaces."""
        return self._surfaces

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Front-vertex coordinate system."""
        return self._cs

    def _build(self) -> list[BaseComponent]:
        """Construct all four sub-surfaces from the config.

        Returns:
            Ordered list of BaseComponent sub-surfaces.
        """
        cfg = self._config
        mat1 = _resolve_material(cfg.material1)
        mat2 = _resolve_material(cfg.material2)
        r = cfg.aperture_radius

        cs_front = self._cs
        cs_cemented = _offset_cs(cs_front, cfg.thickness1)
        cs_back = _offset_cs(cs_cemented, cfg.thickness2)

        surfaces: list[BaseComponent] = []

        # 1. Front face (vacuum -> crown)
        surfaces.append(
            _make_surface(
                cs_front,
                ConicGeometry(cfg.r1, cfg.conic1, r),
                VACUUM,
                mat1,
                cfg.front,
                InteractionType.REFRACTIVE,
                f"{self._name}.front",
            )
        )

        # 2. Cemented interface (crown -> flint)
        surfaces.append(
            _make_surface(
                cs_cemented,
                ConicGeometry(cfg.r2, cfg.conic2, r),
                mat1,
                mat2,
                cfg.cemented,
                InteractionType.REFRACTIVE,
                f"{self._name}.cemented",
            )
        )

        # 3. Back face (flint -> vacuum)
        surfaces.append(
            _make_surface(
                cs_back,
                ConicGeometry(cfg.r3, cfg.conic3, r),
                mat2,
                VACUUM,
                cfg.back,
                InteractionType.REFRACTIVE,
                f"{self._name}.back",
            )
        )

        # 4. Crown edge (cylindrical frustum, absorbing) -- front face to
        #    the cemented interface, so the crown element's own boundary
        #    closes without the flint element.
        sag_front = _sag_at_rim(cfg.r1, cfg.conic1, r)
        sag_cemented = _sag_at_rim(cfg.r2, cfg.conic2, r)
        sag_back = _sag_at_rim(cfg.r3, cfg.conic3, r)
        crown_edge_geom = CylindricalFrustumGeometry(
            r_front=r,
            r_back=r,
            z_front=sag_front,
            z_back=cfg.thickness1 + sag_cemented,
        )
        surfaces.append(
            _make_surface(
                cs_front,
                crown_edge_geom,
                VACUUM,
                VACUUM,
                cfg.edge,
                InteractionType.ABSORBING,
                f"{self._name}.edge_crown",
            )
        )

        # 5. Flint edge -- cemented interface to back face.
        total_thickness = cfg.thickness1 + cfg.thickness2
        flint_edge_geom = CylindricalFrustumGeometry(
            r_front=r,
            r_back=r,
            z_front=cfg.thickness1 + sag_cemented,
            z_back=total_thickness + sag_back,
        )
        surfaces.append(
            _make_surface(
                cs_front,
                flint_edge_geom,
                VACUUM,
                VACUUM,
                cfg.edge,
                InteractionType.ABSORBING,
                f"{self._name}.edge_flint",
            )
        )

        return surfaces
