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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NSQScene
    from optiland.optic.optic import Optic


class ConversionError(Exception):
    """Raised when a sequential surface or element cannot be converted to NSQ."""


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
        NSQScene populated with lens/mirror components, sources, and a detector.

    Raises:
        ConversionError: If any surface has an unsupported geometry type
            (coordinate breaks, diffraction gratings, NURBS, Zernike freeforms,
            or lens elements with more than 3 surfaces).

    Warns:
        UserWarning: NSQ uncoated surfaces apply Fresnel reflection/refraction;
            sequential surfaces always transmit. Add coatings after conversion
            if reflection losses are undesirable.
    """
    from optiland.nonsequential.scene import NSQScene  # noqa: PLC0415

    scene = NSQScene()

    surfs = optic.surfaces.surfaces  # list of all surfaces including obj and img
    n = len(surfs)

    i = 1
    elem_idx = 0
    while i < n - 1:
        surf = surfs[i]
        _check_geometry(surf, i)

        if _is_reflective(surf):
            _add_mirror(scene, optic, surf, i)
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
                _add_lens(scene, optic, element_surfaces, element_indices)
            elif len(element_surfaces) == 3:
                _add_doublet(scene, optic, element_surfaces, element_indices)
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
        warnings.warn(
            "Polarization coatings and Jones matrices on sequential surfaces are not "
            "carried over to the NSQ scene. Polarization tracking in NSQ is deferred.",
            UserWarning,
            stacklevel=2,
        )

    warnings.warn(
        "The converted NSQ scene uses Fresnel reflections at all uncoated interfaces. "
        "In sequential mode, uncoated surfaces always transmit. "
        "Ghost reflections may appear in NSQ ray traces. "
        "Add AR coatings via SurfaceConfig.coating after conversion to suppress this.",
        UserWarning,
        stacklevel=2,
    )

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


def _surface_semi_diameter(surf, optic=None, idx: int | None = None) -> float:
    """Return the semi-diameter (aperture radius) of a surface.

    Args:
        surf: Sequential Surface object.

    Returns:
        Semi-diameter [mm].
    """
    ap = surf.aperture
    if ap is not None and hasattr(ap, "radius"):
        try:
            return float(ap.radius.item())
        except AttributeError:
            return float(ap.radius)
    # Fall back to semi_aperture if set
    if surf.semi_aperture is not None:
        try:
            return float(surf.semi_aperture.item())
        except AttributeError:
            return float(surf.semi_aperture)

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
                return float(r_ext)
        except Exception:
            pass

    return 10.0  # Default if not set


def _add_mirror(scene, optic, surf, elem_idx: int) -> None:
    """Add a Mirror component to the scene.

    Args:
        scene: NSQScene to add to.
        surf: Sequential Surface object for the mirror.
        elem_idx: Element index (used for naming).
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415
    from optiland.nonsequential.components.configs import MirrorConfig  # noqa: PLC0415

    z = _surface_z(surf)
    radius = _surface_radius(surf)
    conic = _surface_conic(surf)
    ap_r = _surface_semi_diameter(surf, optic, elem_idx)

    cs = CoordinateSystem(z=z)
    config = MirrorConfig(radius=radius, conic=conic, aperture_radius=ap_r)
    scene.add_mirror(f"M{elem_idx}", cs, config)


def _add_lens(scene, optic, element_surfaces: list, elem_indices: list) -> None:
    """Add a singlet Lens component to the scene.

    Args:
        scene: NSQScene to add to.
        element_surfaces: [front_surf, back_surf] -- front is glass entry,
            back exits to air.
        elem_idx: Element index (used for naming).
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415
    from optiland.nonsequential.components.configs import LensConfig  # noqa: PLC0415

    s_front, s_back = element_surfaces
    idx_front, idx_back = elem_indices
    z_front = _surface_z(s_front)
    z_back = _surface_z(s_back)
    thickness = z_back - z_front

    cs = CoordinateSystem(z=z_front)
    config = LensConfig(
        r1=_surface_radius(s_front),
        r2=_surface_radius(s_back),
        thickness=thickness,
        material=_material_name(s_front),
        front_aperture_radius=_surface_semi_diameter(s_front, optic, idx_front),
        back_aperture_radius=_surface_semi_diameter(s_back, optic, idx_back),
        conic1=_surface_conic(s_front),
        conic2=_surface_conic(s_back),
    )
    scene.add_lens(f"L{idx_front}", cs, config)


def _add_doublet(scene, optic, element_surfaces: list, elem_indices: list) -> None:
    """Add a cemented Doublet component to the scene.

    Args:
        scene: NSQScene to add to.
        element_surfaces: [front, cement, back] surfaces.
        elem_idx: Element index (used for naming).
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

    ap_r = max(
        _surface_semi_diameter(s_front, optic, idx_front),
        _surface_semi_diameter(s_cement, optic, idx_cement),
        _surface_semi_diameter(s_back, optic, idx_back),
    )

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
    )
    scene.add_doublet(f"D{elem_indices[0]}", cs, config)


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
