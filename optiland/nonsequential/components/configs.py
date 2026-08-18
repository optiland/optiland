"""Config dataclasses for NSQ compound components.

SurfaceConfig, InteractionType, LensConfig, DoubletConfig, MirrorConfig.

Kramer Harrison, 2026
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.bsdf.base import BaseBSDF
    from optiland.nonsequential.materials.nsq_material import NSQMaterial


class InteractionType(enum.Enum):
    """Optical interaction type for a single surface.

    Attributes:
        REFRACTIVE: Surface refracts (and optionally reflects via Fresnel).
        REFLECTIVE: Surface reflects only; no transmission.
        ABSORBING: Surface absorbs all incident rays.
    """

    REFRACTIVE = "refractive"
    REFLECTIVE = "reflective"
    ABSORBING = "absorbing"


@dataclass
class SurfaceConfig:
    """Optional per-surface overrides within a compound component.

    All fields default to ``None``, meaning "use the compound-level default."
    When a field is set, it overrides the compound's default for that surface.

    Attributes:
        bsdf: Custom BSDF for this surface.  Routes rays through the scatter
            model instead of the surface's specular/refractive behaviour.
        scatter_fraction: Probability in [0, 1] that a ray striking this
            surface is routed through ``bsdf`` rather than following the
            specular or refractive path.  The default of 1.0 sends every ray
            to the BSDF, turning the surface into a pure diffuser.  Set it
            below 1 to model a partially scattering surface, e.g. 0.1 for a
            surface that scatters a tenth of the light and transmits the
            rest.  Ignored when ``bsdf`` is None.
        coating: An ``optiland.coatings.BaseCoating`` for a refractive
            surface (e.g. an AR coating). When set, its reflectance/
            transmittance replace the bare Fresnel calculation, so NSQ and
            the sequential engine agree on R. Must be an unpolarized coating
            (``SimpleCoating``); a ``BaseCoatingPolarized`` instance (Jones-
            matrix based -- ``FresnelCoating``, ``ThinFilmCoating``, ...)
            raises ``NotImplementedError`` rather than being silently
            degraded to its scalar average. Ignored on absorbing surfaces.
        aperture_radius: Semi-diameter override [mm].  Overrides the aperture
            computed from the compound config.
        interaction: Force a specific interaction type on this surface.
        reflectance: Required when ``interaction`` selects
            ``InteractionType.REFLECTIVE``: a constant in [0, 1], a
            wavelength-dependent ``callable(wavelength_um) -> reflectance``,
            or an unpolarized ``BaseCoating``. See
            ``ReflectiveComponent`` -- constructing a reflective surface
            without one raises rather than defaulting to a perfect mirror.
    """

    bsdf: BaseBSDF | None = None
    scatter_fraction: float = 1.0
    coating: object | None = None  # optiland.coatings.BaseCoating
    aperture_radius: float | None = None
    interaction: InteractionType | None = None
    reflectance: object | None = None  # float | Callable | BaseCoating


@dataclass
class LensConfig:
    """Configuration for a single-element refractive lens.

    The lens assembles up to four physical surfaces:

    1. **Front face** -- refractive, conic.
    2. **Back face** -- refractive, conic.
    3. **Edge** -- cylindrical frustum, absorbing by default.
    4. **Rim** -- annular plane, absorbing; only created when
       ``front_aperture_radius != back_aperture_radius``.

    Attributes:
        r1: Front vertex radius of curvature [mm].  Positive = centre of
            curvature on +z side.
        r2: Back vertex radius of curvature [mm].
        thickness: Centre thickness of the lens [mm].
        material: Glass name (e.g. ``'N-BK7'``) or a ready-made
            :class:`~optiland.nonsequential.materials.NSQMaterial` instance.
        front_aperture_radius: Semi-diameter of the front face [mm].
        back_aperture_radius: Semi-diameter of the back face [mm].
            Defaults to ``front_aperture_radius`` when ``None``.
        conic1: Conic constant of the front face (0 = sphere).
        conic2: Conic constant of the back face (0 = sphere).
        front: Per-surface overrides for the front face.
        back: Per-surface overrides for the back face.
        edge: Per-surface overrides for the edge (barrel) surface.
        rim: Per-surface overrides for the rim annulus (only used when
            apertures differ).
    """

    r1: float
    r2: float
    thickness: float
    material: str | NSQMaterial
    front_aperture_radius: float
    back_aperture_radius: float | None = None
    conic1: float = 0.0
    conic2: float = 0.0
    front: SurfaceConfig | None = None
    back: SurfaceConfig | None = None
    edge: SurfaceConfig | None = None
    rim: SurfaceConfig | None = None


@dataclass
class DoubletConfig:
    """Configuration for a cemented achromatic doublet.

    Surfaces in order (front -> back): front face, cemented interface, back
    face, edge.

    Attributes:
        r1: Front radius of curvature [mm].
        r2: Cemented interface radius of curvature [mm].
        r3: Back radius of curvature [mm].
        thickness1: Thickness of the crown element [mm].
        thickness2: Thickness of the flint element [mm].
        material1: Crown element glass name or NSQMaterial.
        material2: Flint element glass name or NSQMaterial.
        aperture_radius: Common semi-diameter for all surfaces [mm].
        conic1: Conic constant of the front face.
        conic2: Conic constant of the cemented interface.
        conic3: Conic constant of the back face.
        front: Per-surface overrides for the front face.
        cemented: Per-surface overrides for the cemented interface.
        back: Per-surface overrides for the back face.
        edge: Per-surface overrides for the edge surface.
    """

    r1: float
    r2: float
    r3: float
    thickness1: float
    thickness2: float
    material1: str | NSQMaterial
    material2: str | NSQMaterial
    aperture_radius: float
    conic1: float = 0.0
    conic2: float = 0.0
    conic3: float = 0.0
    front: SurfaceConfig | None = None
    cemented: SurfaceConfig | None = None
    back: SurfaceConfig | None = None
    edge: SurfaceConfig | None = None


@dataclass
class MirrorConfig:
    """Configuration for a single reflective mirror surface.

    Attributes:
        radius: Vertex radius of curvature [mm].  Negative = concave when
            oriented with the normal pointing toward +z.
        reflectance: Mirror reflectance: a constant in [0, 1], a
            wavelength-dependent ``callable(wavelength_um) -> reflectance``,
            or an unpolarized ``optiland.coatings.BaseCoating`` (e.g.
            ``SimpleCoating``). Required -- there is no implicit
            perfect-mirror default: a mirror built without specifying
            how much light it reflects is a modelling bug, not a 100%
            reflector. Overridden per-surface by ``surface.reflectance``.
        conic: Conic constant (0 = sphere, -1 = paraboloid, etc.).
        aperture_radius: Semi-diameter [mm].
        surface: Per-surface overrides (e.g. to attach a scatter BSDF).
    """

    radius: float
    reflectance: object  # float | Callable | BaseCoating
    conic: float = 0.0
    aperture_radius: float = 25.0
    surface: SurfaceConfig | None = None
