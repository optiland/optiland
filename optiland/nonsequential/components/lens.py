"""Lens compound component for Non-Sequential Raytracing.

A single refractive lens element: front face + back face + edge + optional rim.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from optiland.nonsequential._utils import as_float, as_param
from optiland.nonsequential.components.absorbing import AbsorbingComponent
from optiland.nonsequential.components.compound import CompoundComponent
from optiland.nonsequential.components.configs import (
    InteractionType,
    LensConfig,
    SurfaceConfig,
)
from optiland.nonsequential.components.geometry.analytic.annulus import (
    AnnularPlaneGeometry,
)
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.geometry.analytic.frustum import (
    CylindricalFrustumGeometry,
)
from optiland.nonsequential.components.reflective import ReflectiveComponent
from optiland.nonsequential.components.refractive import RefractiveComponent
from optiland.nonsequential.components.volume import Volume
from optiland.nonsequential.materials.nsq_material import VACUUM, NSQMaterial

if TYPE_CHECKING:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential.components.base import BaseComponent


class Lens(CompoundComponent):
    """Single refractive lens element.

    Assembles up to four physical surfaces:

    1. **Front face** -- refractive, conic.
    2. **Back face** -- refractive, conic.
    3. **Edge** -- cylindrical frustum, absorbing by default.
    4. **Rim** -- annular plane, absorbing; only when
       ``front_aperture_radius != back_aperture_radius``.

    The built surfaces are validated as a single closed :class:`Volume`
    (watertight, consistently outward-oriented) at construction time --
    a lens whose faces and edge do not actually close up raises
    :class:`~optiland.nonsequential.components.volume.NonWatertightVolumeError`
    immediately rather than producing silently wrong flux accounting later.

    Attributes:
        _name: Registry name.
        _cs: Front-vertex coordinate system.
        _config: LensConfig describing the lens geometry.
        _surfaces: Built list of sub-surfaces.
        _volume: The validated :class:`Volume` these surfaces form.
    """

    def __init__(
        self,
        name: str,
        cs: CoordinateSystem,
        config: LensConfig,
    ) -> None:
        """Initialize Lens from a LensConfig.

        Args:
            name: Unique identifier for this lens in the registry.
            cs: Coordinate system for the front vertex.
            config: Lens geometry and material configuration.

        Raises:
            NonWatertightVolumeError: If the built surfaces do not form a
                closed, consistently outward-oriented solid.
        """
        self._name = name
        self._cs = cs
        self._config = config
        mat = _resolve_material(config.material)
        self._surfaces: list[BaseComponent] = self._build()
        self._volume = Volume(name=name, boundary=self._surfaces, interior=mat)

    @property
    def name(self) -> str:
        """Registry name of this lens."""
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
        """Construct all sub-surfaces from the config.

        Returns:
            Ordered list of BaseComponent sub-surfaces.
        """
        cfg = self._config
        mat = _resolve_material(cfg.material)

        back_r = cfg.back_aperture_radius
        if back_r is None:
            back_r = cfg.front_aperture_radius
        front_r = cfg.front_aperture_radius

        cs_front = self._cs
        cs_back = _offset_cs(cs_front, cfg.thickness)

        surfaces: list[BaseComponent] = []

        # 1. Front face (refractive by default)
        front_geom = ConicGeometry(cfg.r1, cfg.conic1, front_r)
        surfaces.append(
            _make_surface(
                cs_front,
                front_geom,
                VACUUM,
                mat,
                cfg.front,
                InteractionType.REFRACTIVE,
                f"{self._name}.front",
            )
        )

        # 2. Back face (refractive by default)
        back_geom = ConicGeometry(cfg.r2, cfg.conic2, back_r)
        surfaces.append(
            _make_surface(
                cs_back,
                back_geom,
                mat,
                VACUUM,
                cfg.back,
                InteractionType.REFRACTIVE,
                f"{self._name}.back",
            )
        )

        # 3. Edge (cylindrical frustum, absorbing by default)
        sag_front = _sag_at_rim(cfg.r1, cfg.conic1, front_r)
        sag_back = _sag_at_rim(cfg.r2, cfg.conic2, back_r)

        wider_r = max(as_float(front_r), as_float(back_r))
        narrower_r = min(as_float(front_r), as_float(back_r))

        if as_float(front_r) > as_float(back_r):
            z_front_edge = sag_front
            z_back_edge = as_float(cfg.thickness) + sag_back
            cs_rim = cs_back
            rim_z = sag_back
        elif as_float(back_r) > as_float(front_r):
            z_front_edge = sag_front
            z_back_edge = as_float(cfg.thickness) + sag_back
            cs_rim = cs_front
            rim_z = sag_front
        else:
            z_front_edge = sag_front
            z_back_edge = as_float(cfg.thickness) + sag_back

        edge_geom = CylindricalFrustumGeometry(
            r_front=wider_r,
            r_back=wider_r,
            z_front=z_front_edge,
            z_back=z_back_edge,
        )
        surfaces.append(
            _make_surface(
                cs_front,
                edge_geom,
                VACUUM,
                VACUUM,
                cfg.edge,
                InteractionType.ABSORBING,
                f"{self._name}.edge",
            )
        )

        # 4. Rim annulus (only when aperture radii differ)
        if not _approx_equal(front_r, back_r):
            rim_geom = AnnularPlaneGeometry(
                inner_radius=narrower_r,
                outer_radius=wider_r,
                z_offset=rim_z,
            )
            surfaces.append(
                _make_surface(
                    cs_rim,
                    rim_geom,
                    VACUUM,
                    VACUUM,
                    cfg.rim,
                    InteractionType.ABSORBING,
                    f"{self._name}.rim",
                )
            )

        return surfaces


def _resolve_material(mat: str | NSQMaterial) -> NSQMaterial:
    """Resolve a glass name or NSQMaterial to NSQMaterial.

    Args:
        mat: Glass catalog name or ready-made NSQMaterial.

    Returns:
        NSQMaterial instance.
    """
    if isinstance(mat, str):
        return NSQMaterial.from_glass(mat)
    return mat


def _offset_cs(cs: CoordinateSystem, dz: float) -> CoordinateSystem:
    """Return a new CoordinateSystem translated by ``dz`` along local +z.

    Args:
        cs: Reference coordinate system.
        dz: Offset along the local z-axis [mm].

    Returns:
        New CoordinateSystem at ``cs.origin + R_cs @ [0, 0, dz]``.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415

    return CoordinateSystem(z=dz, reference_cs=cs)


def _sag_at_rim(radius: float, conic: float, aperture_radius: float) -> float:
    """Compute the sag of a conic surface at the aperture rim.

    The edge/rim surfaces this feeds are absorbing bookkeeping geometry, not
    part of the differentiable optical path, so the sag is evaluated from
    detached floats.  This keeps a tensor-valued ``radius`` from leaking a
    partial (and physically incomplete) gradient into the lens edge.

    Args:
        radius: Vertex radius of curvature [mm].  0 -> flat (sag = 0).
        conic: Conic constant K.
        aperture_radius: Semi-aperture radius [mm].

    Returns:
        Sag value z(aperture_radius) [mm].
    """
    radius = as_float(radius)
    conic = as_float(conic)
    aperture_radius = as_float(aperture_radius)
    if radius == 0.0 or aperture_radius == 0.0:
        return 0.0
    r2 = aperture_radius**2
    R = radius
    K = conic
    under_root = 1.0 - (1.0 + K) * r2 / (R * R)
    if under_root < 0.0:
        under_root = 0.0
    return r2 / (R * (1.0 + math.sqrt(under_root)))


def _approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    """Return True if ``|a - b| <= tol``."""
    return abs(as_float(a) - as_float(b)) <= tol


def _resolve_interaction(
    cfg: SurfaceConfig | None,
    default: InteractionType,
) -> InteractionType:
    """Return the interaction type, applying SurfaceConfig override if set.

    Args:
        cfg: Optional surface config; may override the interaction type.
        default: Default interaction type to use when no override is set.

    Returns:
        Resolved InteractionType.
    """
    if cfg is not None and cfg.interaction is not None:
        return cfg.interaction
    return default


def _make_surface(
    cs: CoordinateSystem,
    geom,
    mat_front: NSQMaterial,
    mat_back: NSQMaterial,
    cfg: SurfaceConfig | None,
    interaction: InteractionType,
    name: str = "",
    reflectance: object | None = None,
) -> BaseComponent:
    """Construct the correct BaseComponent from config overrides.

    Args:
        cs: Coordinate system for this surface.
        geom: Geometry object (ConicGeometry, frustum, etc.).
        mat_front: Front-side material.
        mat_back: Back-side material.
        cfg: Optional SurfaceConfig with overrides.
        interaction: Default interaction type.
        name: Label for this sub-surface, e.g. ``"L1.front"``.
        reflectance: Compound-level reflectance (e.g. ``MirrorConfig
            .reflectance``), used when this surface resolves to
            ``InteractionType.REFLECTIVE`` and ``cfg.reflectance`` does not
            override it. Ignored for non-reflective surfaces.

    Returns:
        The constructed BaseComponent subclass.

    Raises:
        ValueError: If the surface resolves to REFLECTIVE and neither
            ``cfg.reflectance`` nor ``reflectance`` supplies a value.
    """
    resolved = _resolve_interaction(cfg, interaction)
    bsdf = cfg.bsdf if cfg is not None else None
    scatter_fraction = cfg.scatter_fraction if cfg is not None else 1.0
    aperture_override = cfg.aperture_radius if cfg is not None else None
    if aperture_override is not None and hasattr(geom, "aperture_radius"):
        geom.aperture_radius = as_param(aperture_override)

    if resolved == InteractionType.REFRACTIVE:
        coating = cfg.coating if cfg is not None else None
        return RefractiveComponent(
            cs,
            geom,
            mat_front,
            mat_back,
            bsdf=bsdf,
            name=name,
            scatter_fraction=scatter_fraction,
            coating=coating,
        )
    if resolved == InteractionType.REFLECTIVE:
        surface_reflectance = cfg.reflectance if cfg is not None else None
        if surface_reflectance is None:
            surface_reflectance = reflectance
        if surface_reflectance is None:
            raise ValueError(
                f"Surface {name!r} resolves to InteractionType.REFLECTIVE but "
                "no reflectance was supplied. Set MirrorConfig.reflectance "
                "(or this surface's SurfaceConfig.reflectance) to a constant, "
                "a callable(wavelength_um) -> reflectance, or an unpolarized "
                "optiland.coatings.BaseCoating."
            )
        return ReflectiveComponent(
            cs,
            geom,
            surface_reflectance,
            bsdf=bsdf,
            material_front=mat_front,
            name=name,
            scatter_fraction=scatter_fraction,
        )
    return AbsorbingComponent(cs, geom, mat_front, name=name)
