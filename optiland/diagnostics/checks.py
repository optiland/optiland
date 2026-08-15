"""Individual diagnostic checks.

Each check is a small function that reads an `Optic` and returns zero or
more `Diagnostic` findings. Checks never mutate the `Optic` they inspect and
never call back into it beyond reading its public attributes — the
`optiland.diagnostics` subpackage depends on `optiland.optic`, never the
other way around.

To add a new check: write a function matching `SystemCheck` and append it to
`CHECKS`. Nothing else needs to change (open/closed).

Kramer Harrison, 2026
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.diagnostics.report import Diagnostic, Severity
from optiland.surfaces.object_surface import ObjectSurface

if TYPE_CHECKING:
    from optiland.optic.optic import Optic

SystemCheck = Callable[["Optic"], "list[Diagnostic]"]

_DOC_BASE = "https://optiland.readthedocs.io/en/latest/conventions.html"


def _doc_url(code: str) -> str:
    return f"{_DOC_BASE}#{code.lower()}"


def check_no_wavelengths(lens: Optic) -> list[Diagnostic]:
    """OPT001: no wavelengths are defined."""
    if lens.wavelengths.num_wavelengths > 0:
        return []
    return [
        Diagnostic(
            severity=Severity.ERROR,
            code="OPT001",
            message="No wavelengths are defined on the optical system.",
            fix="lens.wavelengths.add(value=0.55, is_primary=True)",
            doc_url=_doc_url("OPT001"),
        )
    ]


def check_no_primary_wavelength(lens: Optic) -> list[Diagnostic]:
    """OPT002: wavelengths exist, but none is marked primary."""
    if lens.wavelengths.num_wavelengths == 0:
        return []
    if any(w.is_primary for w in lens.wavelengths):
        return []
    return [
        Diagnostic(
            severity=Severity.ERROR,
            code="OPT002",
            message="Wavelengths are defined, but none is marked primary.",
            fix="lens.wavelengths.add(value=0.55, is_primary=True)",
            doc_url=_doc_url("OPT002"),
        )
    ]


def check_no_aperture(lens: Optic) -> list[Diagnostic]:
    """OPT003: no aperture is defined."""
    if lens.aperture is not None:
        return []
    return [
        Diagnostic(
            severity=Severity.ERROR,
            code="OPT003",
            message="No aperture is defined on the optical system.",
            fix='lens.set_aperture(aperture_type="EPD", value=25)',
            doc_url=_doc_url("OPT003"),
        )
    ]


def check_no_stop_surface(lens: Optic) -> list[Diagnostic]:
    """OPT004: no surface is marked as the aperture stop."""
    surfaces = lens.surfaces.surfaces
    if len(surfaces) == 0:
        return []
    if any(surf.is_stop for surf in surfaces):
        return []
    return [
        Diagnostic(
            severity=Severity.ERROR,
            code="OPT004",
            message="No stop surface is defined.",
            fix="lens.surfaces.add(..., is_stop=True) on the aperture surface",
            doc_url=_doc_url("OPT004"),
        )
    ]


def check_too_few_surfaces(lens: Optic) -> list[Diagnostic]:
    """OPT005: fewer than 2 surfaces, so there is no object/image pair."""
    if lens.surfaces.num_surfaces >= 2:
        return []
    return [
        Diagnostic(
            severity=Severity.ERROR,
            code="OPT005",
            message=(
                f"The system has only {lens.surfaces.num_surfaces} surface(s); "
                "at least an object and an image surface are required."
            ),
            fix=(
                "lens.surfaces.add(index=0, thickness=be.inf) for the object "
                "surface, then add an image surface last"
            ),
            doc_url=_doc_url("OPT005"),
        )
    ]


def check_no_fields(lens: Optic) -> list[Diagnostic]:
    """OPT006: no fields are defined."""
    if len(lens.fields.fields) > 0:
        return []
    return [
        Diagnostic(
            severity=Severity.ERROR,
            code="OPT006",
            message="No fields are defined on the optical system.",
            fix="lens.fields.add(y=0)",
            doc_url=_doc_url("OPT006"),
        )
    ]


def check_object_surface_not_first(lens: Optic) -> list[Diagnostic]:
    """OPT007: the object surface is not at index 0."""
    surfaces = lens.surfaces.surfaces
    if len(surfaces) == 0:
        return []
    for index, surf in enumerate(surfaces):
        if isinstance(surf, ObjectSurface) and index != 0:
            return [
                Diagnostic(
                    severity=Severity.WARNING,
                    code="OPT007",
                    message=f"The object surface is at index {index}, not 0.",
                    fix="Reorder surfaces; the object surface must be index 0.",
                    where=index,
                    doc_url=_doc_url("OPT007"),
                )
            ]
    return []


def check_non_finite_interior_thickness(lens: Optic) -> list[Diagnostic]:
    """OPT008: an interior surface has a non-finite thickness or NaN radius."""
    surfaces = lens.surfaces.surfaces
    n = len(surfaces)
    findings: list[Diagnostic] = []

    for index, surf in enumerate(surfaces):
        # The object surface (index 0) may legitimately sit at infinity, and
        # the image surface's thickness is not meaningful, so only interior
        # surfaces are checked for non-finite thickness.
        if 0 < index < n - 1 and not be.isfinite(be.array(surf.thickness)).all():
            findings.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="OPT008",
                    message=(
                        f"Surface {index} has a non-finite thickness "
                        f"({surf.thickness})."
                    ),
                    fix=f"Set a finite thickness on surface {index}.",
                    where=index,
                    doc_url=_doc_url("OPT008"),
                )
            )

        radius = getattr(surf.geometry, "radius", None)
        # An infinite radius is a normal, flat surface; only NaN is a bug.
        if radius is not None and be.isnan(be.array(radius)).any():
            findings.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="OPT008",
                    message=f"Surface {index} has a NaN radius of curvature.",
                    fix=f"Set a valid radius on surface {index}.",
                    where=index,
                    doc_url=_doc_url("OPT008"),
                )
            )

    return findings


def check_wavelength_outside_material_range(lens: Optic) -> list[Diagnostic]:
    """OPT009: a wavelength falls outside a material's dispersion data range."""
    findings: list[Diagnostic] = []
    for index, surf in enumerate(lens.surfaces.surfaces):
        material = getattr(surf, "material_post", None)
        material_data = getattr(material, "material_data", None)
        if not material_data:
            continue

        min_wl = material_data.get("min_wavelength")
        max_wl = material_data.get("max_wavelength")
        if min_wl is None or max_wl is None:
            continue

        material_name = getattr(material, "name", None) or repr(material)
        for wavelength in lens.wavelengths:
            if wavelength.value < min_wl or wavelength.value > max_wl:
                findings.append(
                    Diagnostic(
                        severity=Severity.WARNING,
                        code="OPT009",
                        message=(
                            f"Wavelength {wavelength.value} um is outside the "
                            f"valid dispersion range [{min_wl}, {max_wl}] um of "
                            f"material '{material_name}' on surface {index}."
                        ),
                        fix=(
                            "Use a wavelength within the material's data range, "
                            "or choose a different material."
                        ),
                        where=index,
                        doc_url=_doc_url("OPT009"),
                    )
                )
    return findings


def check_non_positive_thickness(lens: Optic) -> list[Diagnostic]:
    """OPT010: a thickness has an unexpected sign, or is exactly zero.

    A reflective surface (a mirror) flips the sign of every thickness that
    follows it, since propagation now runs the other way along z — see the
    "Sign Conventions" section of the conventions doc. The expected sign is
    therefore tracked by the parity of the number of reflective surfaces
    seen so far (including the current one), not assumed to always be
    positive; a two-mirror system like a Cassegrain returns to positive
    thicknesses after its second mirror.
    """
    surfaces = lens.surfaces.surfaces
    n = len(surfaces)
    findings: list[Diagnostic] = []

    reflections = 0
    for index in range(n - 1):
        interaction_model = getattr(surfaces[index], "interaction_model", None)
        if getattr(interaction_model, "is_reflective", False):
            reflections += 1

        thickness = surfaces[index].thickness
        if not be.isfinite(be.array(thickness)).all():
            continue

        expect_negative = reflections % 2 == 1
        is_bad = thickness >= 0 if expect_negative else thickness <= 0
        if is_bad:
            expected = "negative" if expect_negative else "positive"
            findings.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="OPT010",
                    message=(
                        f"Surface {index} has thickness {thickness}, but "
                        f"{reflections} reflection(s) precede it (inclusive), "
                        f"so a {expected} value was expected."
                    ),
                    fix=f"Set a {expected} thickness on surface {index}.",
                    where=index,
                    doc_url=_doc_url("OPT010"),
                )
            )
    return findings


def check_stop_at_object_or_image(lens: Optic) -> list[Diagnostic]:
    """OPT011: the stop surface is the object or image surface."""
    surfaces = lens.surfaces.surfaces
    n = len(surfaces)
    if n == 0:
        return []
    for index, surf in enumerate(surfaces):
        if surf.is_stop and index in (0, n - 1):
            role = "object" if index == 0 else "image"
            return [
                Diagnostic(
                    severity=Severity.WARNING,
                    code="OPT011",
                    message=(
                        f"The stop surface is the {role} surface (index {index}); "
                        "this is usually a mistake."
                    ),
                    fix="Move is_stop=True to the intended aperture surface.",
                    where=index,
                    doc_url=_doc_url("OPT011"),
                )
            ]
    return []


def check_no_rays_reach_image(lens: Optic) -> list[Diagnostic]:
    """OPT012: a probe trace shows no rays reaching the image surface.

    Only runs once the system is complete enough to trace: it needs
    wavelengths, a primary wavelength, an aperture, a stop, at least two
    surfaces, and at least one field. Any check failure above already
    reports the root cause, so this is skipped rather than duplicated.
    """
    if lens.surfaces.num_surfaces < 2:
        return []
    if lens.wavelengths.num_wavelengths == 0:
        return []
    if not any(w.is_primary for w in lens.wavelengths):
        return []
    if lens.aperture is None:
        return []
    if not any(surf.is_stop for surf in lens.surfaces.surfaces):
        return []
    if len(lens.fields.fields) == 0:
        return []

    hx, hy = lens.fields.get_field_coords()[0]
    try:
        lens.trace(hx, hy, lens.primary_wavelength, num_rays=8)
    except Exception:
        # Ray tracing failures surface through their own exceptions; a probe
        # trace that raises is not this check's concern to report.
        return []

    intensity = lens.surfaces.intensity[-1, :]
    if be.any(intensity != 0):
        return []

    return [
        Diagnostic(
            severity=Severity.WARNING,
            code="OPT012",
            message="No rays from a probe trace reached the image surface.",
            fix=(
                "Check ray aiming and aperture sizing; the pupil or field may "
                "be too large for this system."
            ),
            doc_url=_doc_url("OPT012"),
        )
    ]


CHECKS: list[SystemCheck] = [
    check_no_wavelengths,
    check_no_primary_wavelength,
    check_no_aperture,
    check_no_stop_surface,
    check_too_few_surfaces,
    check_no_fields,
    check_object_surface_not_first,
    check_non_finite_interior_thickness,
    check_wavelength_outside_material_range,
    check_non_positive_thickness,
    check_stop_at_object_or_image,
    check_no_rays_reach_image,
]
