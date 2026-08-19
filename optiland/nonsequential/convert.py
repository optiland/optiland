"""Sequential -> Non-Sequential Converter.

Provides :func:`sequential_to_nonsequential` to convert an
:class:`~optiland.optic.Optic` instance into a fully-populated
:class:`~optiland.nonsequential.scene.NSQScene`.

Differentiable-ready output
---------------------------
The returned scene stores all geometric parameters as plain Python
:class:`float` values extracted from the sequential surfaces.  This is
intentional: the converter does *not* know which parameters the user wants to
differentiate.  To optimize a parameter with PyTorch, re-assign it as a
tensor after conversion::

    scene = sequential_to_nonsequential(optic)
    # Access the config on the compound component:
    cfg = scene.component_registry.get("L1")._config
    cfg.r1 = torch.tensor(cfg.r1, requires_grad=True)

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NSQScene
    from optiland.optic.optic import Optic


class ConversionError(Exception):
    """Raised when a sequential surface or element cannot be converted to NSQ."""


@dataclass
class ConversionReport:
    """Structured record of what a conversion dropped or approximated.

    Attached to the returned scene as ``scene.conversion_report`` rather
    than requiring the caller to parse warning text -- the whole point of
    this class is that "what did the converter have to guess or drop" is
    inspectable data, not something a user has to have watched the log for.

    Attributes:
        coated_surfaces: Names of refractive surfaces whose sequential
            unpolarized coating was carried over to ``SurfaceConfig.coating``
            (so NSQ and the sequential engine agree on R).
        uncoated_surfaces: Names of refractive surfaces with no usable
            coating in the sequential system; these get bare Fresnel
            reflection/refraction in NSQ, which the sequential engine never
            applies (transmission is always 100% there).
        mirror_reflectance_defaulted: Names of mirrors with no usable scalar
            reflectance in the sequential system, defaulted to a perfect
            reflector (R=1.0) -- NSQ has no implicit-mirror default,
            so the converter must supply *something*, and a perfect
            reflector is the most visible (least silently-wrong) choice.
        estimated_apertures: Names of surfaces whose aperture radius was not
            explicitly set in the sequential system and had to be estimated
            from paraxial ray heights (or, failing that, a fixed 10 mm
            fallback) rather than read directly.
        polarization_dropped: True if any sequential surface had a
            polarization-sensitive (Jones-matrix) coating -- NSQ rays carry
            no polarization state, so these are dropped entirely, not
            approximated.
    """

    coated_surfaces: list[str] = _dc_field(default_factory=list)
    uncoated_surfaces: list[str] = _dc_field(default_factory=list)
    mirror_reflectance_defaulted: list[str] = _dc_field(default_factory=list)
    estimated_apertures: list[str] = _dc_field(default_factory=list)
    polarization_dropped: bool = False

    def summary(self) -> str:
        """Human-readable multi-line summary of everything dropped/approximated.

        Returns:
            A multi-line string, or a single "nothing dropped" line if the
            conversion was fully faithful.
        """
        lines: list[str] = []
        if self.coated_surfaces:
            lines.append(
                f"Coatings carried over ({len(self.coated_surfaces)}): "
                f"{', '.join(self.coated_surfaces)}"
            )
        if self.uncoated_surfaces:
            lines.append(
                f"Bare Fresnel, no sequential coating found "
                f"({len(self.uncoated_surfaces)}): "
                f"{', '.join(self.uncoated_surfaces)}"
            )
        if self.mirror_reflectance_defaulted:
            lines.append(
                f"Mirror reflectance defaulted to 1.0, no usable coating "
                f"found ({len(self.mirror_reflectance_defaulted)}): "
                f"{', '.join(self.mirror_reflectance_defaulted)}"
            )
        if self.estimated_apertures:
            lines.append(
                f"Apertures estimated (not explicitly set) "
                f"({len(self.estimated_apertures)}): "
                f"{', '.join(self.estimated_apertures)}"
            )
        if self.polarization_dropped:
            lines.append(
                "Polarization-sensitive coatings were dropped entirely "
                "(NSQ rays carry no polarization state)."
            )
        if not lines:
            return "Conversion was fully faithful: nothing was dropped or approximated."
        return "\n".join(lines)


def _has_polarization_surfaces(optic) -> bool:
    """Return True if any surface has polarization-sensitive coatings.

    Uses the canonical ``surface.interaction_model.coating`` accessor (not the
    deprecated ``surface.coating`` property) and checks against the
    :class:`~optiland.coatings.BaseCoatingPolarized` ABC, which is the
    authoritative discriminator for polarization-sensitive coatings.

    Args:
        optic: Sequential Optic.

    Returns:
        True if any surface has a polarization-sensitive coating.
    """
    try:
        from optiland.coatings import BaseCoatingPolarized  # noqa: PLC0415

        for surf in optic.surfaces.surfaces:
            im = getattr(surf, "interaction_model", None)
            if im is None:
                continue
            coating = getattr(im, "coating", None)
            if coating is not None and isinstance(coating, BaseCoatingPolarized):
                return True
        return False
    except Exception:
        return False


def sequential_to_nonsequential(
    optic: Optic,
    *,
    num_rays: int = 1_000,
    detector_width: float | None = None,
    detector_height: float | None = None,
    detector_pixels: tuple[int, int] = (512, 512),
    beam_diameter: float | None = None,
    half_angle_deg: float | None = None,
) -> NSQScene:
    """Convert a sequential Optic to a non-sequential NSQScene.

    The image surface is converted to an IrradianceDetector. Each sequential
    field is mapped to one NSQ source (CollimatedSource for angle fields,
    PointSource for object-height fields). Consecutive refractive surfaces are
    grouped into Lens or Doublet compound components.

    Args:
        optic: Sequential optical system to convert.
        num_rays: Default ray count for visualization sources. Does not affect
            scene.trace() -- pass num_rays there directly.
        detector_width: Semi-width of the image detector [mm]. Defaults to
            2x the paraxial image height.
        detector_height: Semi-height of the image detector [mm]. Defaults to
            detector_width.
        detector_pixels: (num_pixels_x, num_pixels_y) for the irradiance detector.
        beam_diameter: Override entrance pupil diameter for collimated sources
            [mm]. Defaults to paraxial EPD.
        half_angle_deg: Override cone half-angle for point sources [degrees].
            Defaults to paraxial marginal-ray angle at the object plane.

    Returns:
        NSQScene populated with lens/mirror components, sources, and a
        detector. ``scene.conversion_report`` is a :class:`ConversionReport`
        listing everything the converter dropped or had to
        approximate -- coatings, apertures, mirror reflectance, polarization
        -- as structured data rather than only warning text.

    Raises:
        ConversionError: If any surface has an unsupported geometry type
            (coordinate breaks, diffraction gratings, NURBS, Zernike freeforms,
            or lens elements with more than 3 surfaces).

    Warns:
        UserWarning: NSQ surfaces with no carried-over coating apply bare
            Fresnel reflection/refraction; sequential surfaces always
            transmit. See ``scene.conversion_report.uncoated_surfaces`` for
            exactly which surfaces this applies to.
    """
    from optiland.nonsequential.scene import NSQScene  # noqa: PLC0415

    scene = NSQScene()
    report = ConversionReport()

    surfs = optic.surfaces.surfaces  # list of all surfaces including obj and img
    n = len(surfs)

    i = 1
    elem_idx = 0
    while i < n - 1:
        surf = surfs[i]
        _check_geometry(surf, i)

        if _is_reflective(surf):
            _add_mirror(scene, optic, surf, i, report)
            elem_idx += 1
            i += 1
            continue

        if _is_glass(surf):
            # Gather all surfaces that start or continue a glass element.
            # element_surfaces collects the glass-entry surfaces;
            # the loop advances j until we find the glass->air boundary.
            element_surfaces = [surf]
            element_indices = [i]
            j = i + 1
            while j < n - 1 and _is_glass(surfs[j]):
                _check_geometry(surfs[j], j)
                element_surfaces.append(surfs[j])
                element_indices.append(j)
                j += 1
            # surfs[j] is the last surface of the element (exits to air / image)
            if j < n - 1:
                _check_geometry(surfs[j], j)
                element_surfaces.append(surfs[j])
                element_indices.append(j)

            if len(element_surfaces) == 2:
                _add_lens(scene, optic, element_surfaces, element_indices, report)
            elif len(element_surfaces) == 3:
                _add_doublet(scene, optic, element_surfaces, element_indices, report)
            else:
                raise ConversionError(
                    f"Lens element starting at surface index {i} has "
                    f"{len(element_surfaces)} surfaces -- only singlets (2) "
                    f"and cemented doublets (3) are supported."
                )

            elem_idx += 1
            i = j + 1
            continue

        # Standalone air-air surface (e.g. pure aperture stop in air).
        raise ConversionError(
            f"Surface at index {i} is an air-to-air surface "
            "(not reflective, not entering glass). "
            "Standalone aperture stops in air are not supported by the converter. "
            "Move the stop to a glass surface or remove it before converting."
        )

    _add_sources(scene, optic, beam_diameter, half_angle_deg)

    _add_detector(scene, optic, detector_width, detector_height, detector_pixels)

    if _has_polarization_surfaces(optic):
        report.polarization_dropped = True
        warnings.warn(
            "Polarization coatings and Jones matrices on sequential surfaces are not "
            "carried over to the NSQ scene. Polarization tracking in NSQ is deferred.",
            UserWarning,
            stacklevel=2,
        )

    if report.uncoated_surfaces:
        coated_note = (
            f" ({len(report.coated_surfaces)} surface(s) had a coating carried "
            "over and agree with the sequential engine on R.)"
            if report.coated_surfaces
            else ""
        )
        warnings.warn(
            f"{len(report.uncoated_surfaces)} refractive surface(s) had no "
            "usable coating in the sequential system and use bare Fresnel "
            "reflection/refraction in NSQ; the sequential engine always "
            "transmits 100% at an uncoated surface, so ghost reflections may "
            "appear in NSQ ray traces that don't exist in the sequential "
            "trace. See scene.conversion_report.uncoated_surfaces for which "
            "ones, or attach SurfaceConfig.coating after conversion to "
            f"suppress this.{coated_note}",
            UserWarning,
            stacklevel=2,
        )

    scene.conversion_report = report
    return scene


def _check_geometry(surf, index: int) -> None:
    """Raise ConversionError if the surface geometry is unsupported.

    Args:
        surf: Sequential Surface object.
        index: Surface index (for error messages).

    Raises:
        ConversionError: If the geometry type is not supported.
    """
    geo = surf.geometry
    geo_type = type(geo).__name__

    _unsupported = {
        "CoordinateBreak",
        "DiffractionGrating",
        "PlaneGrating",
        "StandardGrating",
        "GridSag",
        "NURBS",
        "ZernikeStandardSag",
        "ZernikeStandardPhase",
        "ZernikeGeometry",
        "ChebyshevGeometry",
        "PolynomialGeometry",
        "BionicGeometry",
        "ToroidalGeometry",
    }

    if geo_type in _unsupported:
        raise ConversionError(
            f"Surface {index} has geometry type '{geo_type}' which is not "
            "supported by the sequential-to-NSQ converter."
        )


def _is_reflective(surf) -> bool:
    """Return True if the surface is a mirror (reflective interaction model).

    Args:
        surf: Sequential Surface object.

    Returns:
        True if the surface interaction model is reflective.
    """
    model = getattr(surf, "interaction_model", None)
    if model is None:
        return False
    return bool(getattr(model, "is_reflective", False))


def _is_glass(surf) -> bool:
    """Return True if the material after the surface is glass (not air/vacuum).

    Args:
        surf: Sequential Surface object.

    Returns:
        True if ``material_post`` is a named catalog glass.
    """
    from optiland.materials.ideal import IdealMaterial  # noqa: PLC0415

    mat = surf.material_post
    if mat is None:
        return False
    return not isinstance(mat, IdealMaterial)


def _material_name(surf) -> str:
    """Extract the glass name from the material after the surface.

    Args:
        surf: Sequential Surface object.

    Returns:
        Material name string (e.g. ``'N-BK7'``).

    Raises:
        ConversionError: If the material type is not recognisable.
    """
    mat = surf.material_post
    if hasattr(mat, "name"):
        return mat.name
    raise ConversionError(
        f"Cannot extract material name from {type(mat).__name__}. "
        "Only catalog materials (Material class) are supported by the converter."
    )


def _surface_z(surf) -> float:
    """Return the global z-position of a surface vertex.

    Args:
        surf: Sequential Surface object.

    Returns:
        Z-coordinate in the global frame [mm].
    """
    x_gcs, y_gcs, z_gcs = surf.geometry.cs.position_in_gcs
    try:
        return float(z_gcs.item())
    except AttributeError:
        return float(z_gcs)


def _surface_radius(surf) -> float:
    """Return the radius of curvature of a surface.

    Args:
        surf: Sequential Surface object.

    Returns:
        Radius of curvature [mm]. inf for a plane.
    """
    r = surf.geometry.radius
    try:
        return float(r.item())
    except AttributeError:
        return float(r)


def _surface_conic(surf) -> float:
    """Return the conic constant of a surface (0 if not applicable).

    Args:
        surf: Sequential Surface object.

    Returns:
        Conic constant.
    """
    try:
        k = surf.geometry.k
        try:
            return float(k.item())
        except AttributeError:
            return float(k)
    except AttributeError:
        return 0.0


def _surface_semi_diameter(
    surf, optic=None, idx: int | None = None
) -> tuple[float, bool]:
    """Return the semi-diameter (aperture radius) of a surface.

    Args:
        surf: Sequential Surface object.
        optic: Sequential Optic (for the paraxial-ray fallback).
        idx: Surface index within ``optic`` (for the paraxial-ray fallback).

    Returns:
        ``(semi_diameter [mm], estimated)`` -- ``estimated`` is True when
        the value could not be read directly from an explicit aperture/
        semi_aperture on the surface and had to be inferred (paraxial ray
        heights, or -- worst case -- a fixed 10 mm fallback), for
        :class:`ConversionReport`.
    """
    ap = surf.aperture
    if ap is not None and hasattr(ap, "radius"):
        try:
            return float(ap.radius.item()), False
        except AttributeError:
            return float(ap.radius), False
    # Fall back to semi_aperture if set
    if surf.semi_aperture is not None:
        try:
            return float(surf.semi_aperture.item()), False
        except AttributeError:
            return float(surf.semi_aperture), False

    # Use paraxial rays from optic if available
    if optic is not None and idx is not None:
        try:
            yb, _ = optic.paraxial.marginal_ray()
            yc, _ = optic.paraxial.chief_ray()

            import optiland.backend as be

            ybi = be.to_numpy(yb[idx]).item()
            yci = be.to_numpy(yc[idx]).item()
            r_ext = abs(ybi) + abs(yci)

            if r_ext > 0.0:
                return float(r_ext), True
        except Exception:
            pass

    return 10.0, True  # Default if not set


def _surface_coating(surf) -> object | None:
    """Extract a usable unpolarized coating from a sequential surface.

    Reads ``surf.interaction_model.coating`` (the non-deprecated accessor).
    A polarized coating is not returned here -- ``_has_polarization_surfaces``
    already surfaces that case globally: polarized coatings must never be
    silently degraded to a scalar average.

    Args:
        surf: Sequential Surface object.

    Returns:
        The coating object (an ``optiland.coatings.BaseCoating``, e.g.
        ``SimpleCoating``) if present and unpolarized, else ``None``.
    """
    from optiland.coatings import BaseCoating, BaseCoatingPolarized  # noqa: PLC0415

    model = getattr(surf, "interaction_model", None)
    coating = getattr(model, "coating", None) if model is not None else None
    if coating is None or isinstance(coating, BaseCoatingPolarized):
        return None
    if isinstance(coating, BaseCoating):
        return coating
    return None


def _add_mirror(scene, optic, surf, elem_idx: int, report: ConversionReport) -> None:
    """Add a Mirror component to the scene.

    Args:
        scene: NSQScene to add to.
        surf: Sequential Surface object for the mirror.
        elem_idx: Element index (used for naming).
        report: ConversionReport to record fidelity notes into.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415
    from optiland.nonsequential.components.configs import MirrorConfig  # noqa: PLC0415

    name = f"M{elem_idx}"
    z = _surface_z(surf)
    radius = _surface_radius(surf)
    conic = _surface_conic(surf)
    ap_r, ap_estimated = _surface_semi_diameter(surf, optic, elem_idx)
    if ap_estimated:
        report.estimated_apertures.append(name)
    reflectance = _mirror_reflectance(surf, elem_idx, report)

    cs = CoordinateSystem(z=z)
    config = MirrorConfig(
        radius=radius, reflectance=reflectance, conic=conic, aperture_radius=ap_r
    )
    scene.add_mirror(name, cs, config)


def _mirror_reflectance(surf, elem_idx: int, report: ConversionReport) -> float:
    """Extract a scalar reflectance for a mirror surface being converted.

    Reads ``surf.interaction_model.coating`` (the non-deprecated accessor).
    An unpolarized ``SimpleCoating``-like coating's ``.reflectance``
    attribute is used directly; anything else (no coating attached, or a
    coating without a plain ``.reflectance`` attribute) falls back to a
    perfect mirror, since NSQ requires an explicit reflectance and the
    sequential engine has no equivalent implicit default to defer to.

    Args:
        surf: Sequential Surface object for the mirror.
        elem_idx: Element index (for the warning message).
        report: ConversionReport to record the default into.

    Returns:
        Scalar reflectance in [0, 1].

    Warns:
        UserWarning: If no usable reflectance could be read from the
            sequential surface, so the mirror is defaulting to R=1.0.
    """
    model = getattr(surf, "interaction_model", None)
    coating = getattr(model, "coating", None) if model is not None else None
    reflectance = getattr(coating, "reflectance", None)
    if reflectance is not None:
        return float(reflectance)

    report.mirror_reflectance_defaulted.append(f"M{elem_idx}")
    warnings.warn(
        f"Mirror M{elem_idx} has no coating with a scalar .reflectance "
        "attached in the sequential system; defaulting to a perfect "
        "reflector (reflectance=1.0). Set MirrorConfig.reflectance "
        "explicitly after conversion if this is not correct.",
        UserWarning,
        stacklevel=2,
    )
    return 1.0


def _add_lens(
    scene, optic, element_surfaces: list, elem_indices: list, report: ConversionReport
) -> None:
    """Add a singlet Lens component to the scene.

    Args:
        scene: NSQScene to add to.
        element_surfaces: [front_surf, back_surf] -- front is glass entry,
            back exits to air.
        elem_indices: Sequential surface indices matching ``element_surfaces``.
        report: ConversionReport to record fidelity notes into.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415
    from optiland.nonsequential.components.configs import LensConfig  # noqa: PLC0415

    s_front, s_back = element_surfaces
    idx_front, idx_back = elem_indices
    z_front = _surface_z(s_front)
    z_back = _surface_z(s_back)
    thickness = z_back - z_front

    front_ap, front_estimated = _surface_semi_diameter(s_front, optic, idx_front)
    back_ap, back_estimated = _surface_semi_diameter(s_back, optic, idx_back)
    front_name = f"L{idx_front}.front"
    back_name = f"L{idx_front}.back"
    if front_estimated:
        report.estimated_apertures.append(front_name)
    if back_estimated:
        report.estimated_apertures.append(back_name)

    cs = CoordinateSystem(z=z_front)
    config = LensConfig(
        r1=_surface_radius(s_front),
        r2=_surface_radius(s_back),
        thickness=thickness,
        material=_material_name(s_front),
        front_aperture_radius=front_ap,
        back_aperture_radius=back_ap,
        conic1=_surface_conic(s_front),
        conic2=_surface_conic(s_back),
        front=_surface_config_with_coating(s_front, front_name, report),
        back=_surface_config_with_coating(s_back, back_name, report),
    )
    scene.add_lens(f"L{idx_front}", cs, config)


def _surface_config_with_coating(
    surf, name: str, report: ConversionReport
) -> object | None:
    """Build a ``SurfaceConfig`` carrying a surface's coating, if any.

    Args:
        surf: Sequential Surface object.
        name: This surface's name, for the report.
        report: ConversionReport to record coated/uncoated into.

    Returns:
        A ``SurfaceConfig(coating=...)`` if the sequential surface had a
        usable unpolarized coating, else ``None`` (bare Fresnel -- NSQ's
        default, and the sequential engine's transmission behaviour, will
        disagree on reflectance in that case; see
        ``ConversionReport.uncoated_surfaces``).
    """
    from optiland.nonsequential.components.configs import SurfaceConfig  # noqa: PLC0415

    coating = _surface_coating(surf)
    if coating is not None:
        report.coated_surfaces.append(name)
        return SurfaceConfig(coating=coating)
    report.uncoated_surfaces.append(name)
    return None


def _add_doublet(
    scene, optic, element_surfaces: list, elem_indices: list, report: ConversionReport
) -> None:
    """Add a cemented Doublet component to the scene.

    Args:
        scene: NSQScene to add to.
        element_surfaces: [front, cement, back] surfaces.
        elem_indices: Sequential surface indices matching ``element_surfaces``.
        report: ConversionReport to record fidelity notes into.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415
    from optiland.nonsequential.components.configs import DoubletConfig  # noqa: PLC0415

    s_front, s_cement, s_back = element_surfaces
    idx_front, idx_cement, idx_back = elem_indices
    z_front = _surface_z(s_front)
    z_cement = _surface_z(s_cement)
    z_back = _surface_z(s_back)

    thickness1 = z_cement - z_front
    thickness2 = z_back - z_cement

    front_ap, front_est = _surface_semi_diameter(s_front, optic, idx_front)
    cement_ap, cement_est = _surface_semi_diameter(s_cement, optic, idx_cement)
    back_ap, back_est = _surface_semi_diameter(s_back, optic, idx_back)
    base_name = f"D{elem_indices[0]}"
    front_name = f"{base_name}.front"
    cement_name = f"{base_name}.cemented"
    back_name = f"{base_name}.back"
    for estimated, name in (
        (front_est, front_name),
        (cement_est, cement_name),
        (back_est, back_name),
    ):
        if estimated:
            report.estimated_apertures.append(name)

    ap_r = max(front_ap, cement_ap, back_ap)

    cs = CoordinateSystem(z=z_front)
    config = DoubletConfig(
        r1=_surface_radius(s_front),
        r2=_surface_radius(s_cement),
        r3=_surface_radius(s_back),
        thickness1=thickness1,
        thickness2=thickness2,
        material1=_material_name(s_front),
        material2=_material_name(s_cement),
        aperture_radius=ap_r,
        conic1=_surface_conic(s_front),
        conic2=_surface_conic(s_cement),
        conic3=_surface_conic(s_back),
        front=_surface_config_with_coating(s_front, front_name, report),
        cemented=_surface_config_with_coating(s_cement, cement_name, report),
        back=_surface_config_with_coating(s_back, back_name, report),
    )
    scene.add_doublet(base_name, cs, config)


def _build_spectrum(optic) -> object:
    """Build an NSQ Spectrum from the optic's wavelength group.

    Args:
        optic: Sequential Optic.

    Returns:
        Spectrum instance.
    """
    import numpy as np  # noqa: PLC0415

    from optiland.nonsequential.sources.base import Spectrum  # noqa: PLC0415

    wls_um = [w.value for w in optic.wavelengths.wavelengths]
    weights = list(optic.wavelengths.weights)
    if not weights or all(w == 0 for w in weights):
        weights = [1.0] * len(wls_um)

    return Spectrum(
        wavelengths=np.array(wls_um, dtype=np.float64),
        weights=np.array(weights, dtype=np.float64),
    )


def _collimated_source_cs(y_angle_deg: float, x_angle_deg: float, epd: float):
    """Build a CoordinateSystem for a CollimatedSource at the given field angles.

    The source is placed upstream of the system so that rays propagate at the
    specified field angles.  The CS is rotated by the field angles so that
    the beam propagates in the correct direction.

    Args:
        y_angle_deg: Field angle in the Y direction [degrees].
        x_angle_deg: Field angle in the X direction [degrees].
        epd: Entrance pupil diameter [mm] -- used to set the upstream offset.

    Returns:
        CoordinateSystem for the source.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415

    upstream_z = -max(epd, 10.0)
    # rx rotates around x-axis (tilts in Y), ry around y-axis (tilts in X)
    rx_rad = math.radians(-y_angle_deg)
    ry_rad = math.radians(x_angle_deg)
    return CoordinateSystem(z=upstream_z, rx=rx_rad, ry=ry_rad)


def _point_source_cs(y_height: float, obj_dist: float):
    """Build a CoordinateSystem for a PointSource at the object plane.

    Args:
        y_height: Object height [mm] (y-field coordinate).
        obj_dist: Object distance from the first surface [mm].

    Returns:
        CoordinateSystem for the source.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415

    return CoordinateSystem(y=y_height, z=obj_dist)


def _marginal_ray_angle_at_object(optic) -> float:
    """Compute the paraxial marginal-ray half-angle at the object plane.

    Args:
        optic: Sequential Optic.

    Returns:
        Half-angle of the marginal ray in degrees.
    """
    ya, ua = optic.paraxial.marginal_ray()
    try:
        u0 = float(ua[0].item())
    except AttributeError:
        u0 = float(ua[0])
    return abs(math.degrees(math.atan(u0))) if u0 != 0 else 5.0


def _add_sources(
    scene,
    optic,
    beam_diameter: float | None,
    half_angle_deg_override: float | None,
) -> None:
    """Add one NSQ source per sequential field to the scene.

    Args:
        scene: NSQScene to add to.
        optic: Sequential Optic.
        beam_diameter: Override EPD [mm] or None.
        half_angle_deg_override: Override half-angle [deg] or None.
    """
    from optiland.fields.field_types import (  # noqa: PLC0415
        AngleField,
        ObjectHeightField,
        ParaxialImageHeightField,
        RealImageHeightField,
    )
    from optiland.nonsequential.sources.configs import (  # noqa: PLC0415
        CollimatedSourceConfig,
        PointSourceConfig,
    )

    field_def = optic.fields.field_definition
    if isinstance(field_def, ParaxialImageHeightField | RealImageHeightField):
        raise ConversionError(
            "Fields of type 'paraxial_image_height' or 'real_image_height' "
            "are not supported by the converter. Switch to 'angle' or "
            "'object_height' fields before converting."
        )

    spectrum = _build_spectrum(optic)

    epd = beam_diameter if beam_diameter is not None else float(optic.paraxial.EPD())

    for i, field in enumerate(optic.fields.fields):
        if isinstance(field_def, AngleField):
            cs = _collimated_source_cs(field.y, field.x, epd)
            config = CollimatedSourceConfig(
                spectrum=spectrum,
                total_flux=1.0,
                aperture_radius=epd / 2.0,
            )
            scene.add_source(f"S{i}", cs, config)

        elif isinstance(field_def, ObjectHeightField):
            obj_surf = optic.object_surface
            obj_z = _surface_z(obj_surf) if obj_surf is not None else -1e9

            ha_deg = (
                half_angle_deg_override
                if half_angle_deg_override is not None
                else _marginal_ray_angle_at_object(optic)
            )

            cs = _point_source_cs(field.y, obj_z)
            config = PointSourceConfig(
                spectrum=spectrum,
                total_flux=1.0,
                half_angle_deg=ha_deg,
            )
            scene.add_source(f"S{i}", cs, config)

        else:
            raise ConversionError(
                f"Field definition type '{type(field_def).__name__}' is not "
                "supported by the converter."
            )


def _add_detector(
    scene,
    optic,
    detector_width: float | None,
    detector_height: float | None,
    detector_pixels: tuple[int, int],
) -> None:
    """Add an IrradianceDetector at the image surface.

    Args:
        scene: NSQScene to add to.
        optic: Sequential Optic.
        detector_width: Semi-width override [mm] or None.
        detector_height: Semi-height override [mm] or None.
        detector_pixels: (num_pixels_x, num_pixels_y).
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415
    from optiland.nonsequential.detectors.configs import (  # noqa: PLC0415
        IrradianceDetectorConfig,
    )

    img_surf = optic.image_surface
    z_img = _surface_z(img_surf)

    if detector_width is not None:
        width = detector_width
    else:
        # Use paraxial chief ray height at image plane x 2 as the full width
        try:
            yb, _ub = optic.paraxial.chief_ray()
            try:
                img_ht = abs(float(yb[-1].item()))
            except AttributeError:
                img_ht = abs(float(yb[-1]))
            width = max(img_ht * 2.0, 1.0)
        except Exception:
            width = 10.0  # sensible fallback

    height = detector_height if detector_height is not None else width
    nx, ny = detector_pixels

    cs = CoordinateSystem(z=z_img)
    config = IrradianceDetectorConfig(
        width=width,
        height=height,
        num_pixels_x=nx,
        num_pixels_y=ny,
    )
    scene.add_detector("D1", cs, config)
