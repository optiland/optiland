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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.diagnostics.report import Diagnostic, Severity
from optiland.surfaces.object_surface import ObjectSurface

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray
    from optiland.optic.optic import Optic
    from optiland.surfaces.standard_surface import Surface

SystemCheck = Callable[["Optic"], "list[Diagnostic]"]

_DOC_BASE = "https://optiland.readthedocs.io/en/latest/conventions.html"


def _doc_url(code: str) -> str:
    return f"{_DOC_BASE}#{code.lower()}"


# --------------------------------------------------------------------------
# OPT001-OPT006: simple, single-condition presence checks.
#
# Each of these fires at most one diagnostic, based on one boolean fact
# about the system. Rather than repeat the same "if broken: build and
# return a Diagnostic" shape six times, each is declared as data and run
# through a single, shared executor.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _PresenceRule:
    """A diagnostic that fires when `is_broken(lens)` is true."""

    code: str
    severity: Severity
    is_broken: Callable[[Optic], bool]
    message: Callable[[Optic], str]
    fix: str


def _run_presence_rule(lens: Optic, rule: _PresenceRule) -> list[Diagnostic]:
    if not rule.is_broken(lens):
        return []
    return [
        Diagnostic(
            severity=rule.severity,
            code=rule.code,
            message=rule.message(lens),
            fix=rule.fix,
            doc_url=_doc_url(rule.code),
        )
    ]


_OPT001_NO_WAVELENGTHS = _PresenceRule(
    code="OPT001",
    severity=Severity.ERROR,
    is_broken=lambda lens: lens.wavelengths.num_wavelengths == 0,
    message=lambda lens: "No wavelengths are defined on the optical system.",
    fix="lens.wavelengths.add(value=0.55, is_primary=True)",
)

_OPT002_NO_PRIMARY_WAVELENGTH = _PresenceRule(
    code="OPT002",
    severity=Severity.ERROR,
    is_broken=lambda lens: (
        lens.wavelengths.num_wavelengths > 0
        and not any(w.is_primary for w in lens.wavelengths)
    ),
    message=lambda lens: "Wavelengths are defined, but none is marked primary.",
    fix="lens.wavelengths.add(value=0.55, is_primary=True)",
)

_OPT003_NO_APERTURE = _PresenceRule(
    code="OPT003",
    severity=Severity.ERROR,
    is_broken=lambda lens: lens.aperture is None,
    message=lambda lens: "No aperture is defined on the optical system.",
    fix='lens.set_aperture(aperture_type="EPD", value=25)',
)

_OPT004_NO_STOP_SURFACE = _PresenceRule(
    code="OPT004",
    severity=Severity.ERROR,
    is_broken=lambda lens: (
        len(lens.surfaces.surfaces) > 0
        and not any(surf.is_stop for surf in lens.surfaces.surfaces)
    ),
    message=lambda lens: "No stop surface is defined.",
    fix="lens.surfaces.add(..., is_stop=True) on the aperture surface",
)

_OPT005_TOO_FEW_SURFACES = _PresenceRule(
    code="OPT005",
    severity=Severity.ERROR,
    is_broken=lambda lens: lens.surfaces.num_surfaces < 2,
    message=lambda lens: (
        f"The system has only {lens.surfaces.num_surfaces} surface(s); "
        "at least an object and an image surface are required."
    ),
    fix=(
        "lens.surfaces.add(index=0, thickness=be.inf) for the object "
        "surface, then add an image surface last"
    ),
)

_OPT006_NO_FIELDS = _PresenceRule(
    code="OPT006",
    severity=Severity.ERROR,
    is_broken=lambda lens: len(lens.fields.fields) == 0,
    message=lambda lens: "No fields are defined on the optical system.",
    fix="lens.fields.add(y=0)",
)


def check_no_wavelengths(lens: Optic) -> list[Diagnostic]:
    """OPT001: no wavelengths are defined."""
    return _run_presence_rule(lens, _OPT001_NO_WAVELENGTHS)


def check_no_primary_wavelength(lens: Optic) -> list[Diagnostic]:
    """OPT002: wavelengths exist, but none is marked primary."""
    return _run_presence_rule(lens, _OPT002_NO_PRIMARY_WAVELENGTH)


def check_no_aperture(lens: Optic) -> list[Diagnostic]:
    """OPT003: no aperture is defined."""
    return _run_presence_rule(lens, _OPT003_NO_APERTURE)


def check_no_stop_surface(lens: Optic) -> list[Diagnostic]:
    """OPT004: no surface is marked as the aperture stop."""
    return _run_presence_rule(lens, _OPT004_NO_STOP_SURFACE)


def check_too_few_surfaces(lens: Optic) -> list[Diagnostic]:
    """OPT005: fewer than 2 surfaces, so there is no object/image pair."""
    return _run_presence_rule(lens, _OPT005_TOO_FEW_SURFACES)


def check_no_fields(lens: Optic) -> list[Diagnostic]:
    """OPT006: no fields are defined."""
    return _run_presence_rule(lens, _OPT006_NO_FIELDS)


# --------------------------------------------------------------------------
# OPT007, OPT011: "find the one offending surface" checks.
# --------------------------------------------------------------------------


def check_object_surface_not_first(lens: Optic) -> list[Diagnostic]:
    """OPT007: the object surface is not at index 0."""
    surfaces = lens.surfaces.surfaces
    misplaced = next(
        (
            index
            for index, surf in enumerate(surfaces)
            if isinstance(surf, ObjectSurface) and index != 0
        ),
        None,
    )
    if misplaced is None:
        return []
    return [
        Diagnostic(
            severity=Severity.WARNING,
            code="OPT007",
            message=f"The object surface is at index {misplaced}, not 0.",
            fix="Reorder surfaces; the object surface must be index 0.",
            where=misplaced,
            doc_url=_doc_url("OPT007"),
        )
    ]


def check_stop_at_object_or_image(lens: Optic) -> list[Diagnostic]:
    """OPT011: the stop surface is the object or image surface."""
    surfaces = lens.surfaces.surfaces
    last = len(surfaces) - 1
    found = next(
        (
            index
            for index, surf in enumerate(surfaces)
            if surf.is_stop and index in (0, last)
        ),
        None,
    )
    if found is None:
        return []
    role = "object" if found == 0 else "image"
    return [
        Diagnostic(
            severity=Severity.WARNING,
            code="OPT011",
            message=(
                f"The stop surface is the {role} surface (index {found}); "
                "this is usually a mistake."
            ),
            fix="Move is_stop=True to the intended aperture surface.",
            where=found,
            doc_url=_doc_url("OPT011"),
        )
    ]


# --------------------------------------------------------------------------
# OPT008: non-finite interior thickness or NaN radius.
# --------------------------------------------------------------------------


def _interior_thickness_diagnostic(
    index: int, surf: Surface, n: int
) -> Diagnostic | None:
    # The object surface (index 0) may legitimately sit at infinity, and
    # the image surface's thickness is not meaningful, so only interior
    # surfaces are checked for non-finite thickness.
    if not (0 < index < n - 1):
        return None
    if be.isfinite(be.array(surf.thickness)).all():
        return None
    return Diagnostic(
        severity=Severity.WARNING,
        code="OPT008",
        message=f"Surface {index} has a non-finite thickness ({surf.thickness}).",
        fix=f"Set a finite thickness on surface {index}.",
        where=index,
        doc_url=_doc_url("OPT008"),
    )


def _nan_radius_diagnostic(index: int, surf: Surface) -> Diagnostic | None:
    radius = getattr(surf.geometry, "radius", None)
    # An infinite radius is a normal, flat surface; only NaN is a bug.
    if radius is None or not be.isnan(be.array(radius)).any():
        return None
    return Diagnostic(
        severity=Severity.WARNING,
        code="OPT008",
        message=f"Surface {index} has a NaN radius of curvature.",
        fix=f"Set a valid radius on surface {index}.",
        where=index,
        doc_url=_doc_url("OPT008"),
    )


def check_non_finite_interior_thickness(lens: Optic) -> list[Diagnostic]:
    """OPT008: an interior surface has a non-finite thickness or NaN radius."""
    surfaces = lens.surfaces.surfaces
    n = len(surfaces)
    candidates = [
        _interior_thickness_diagnostic(index, surf, n)
        for index, surf in enumerate(surfaces)
    ] + [_nan_radius_diagnostic(index, surf) for index, surf in enumerate(surfaces)]
    return [d for d in candidates if d is not None]


# --------------------------------------------------------------------------
# OPT009: a wavelength falls outside a material's dispersion data range.
# --------------------------------------------------------------------------


def _material_dispersion_range(surf: Surface) -> tuple[float, float, str] | None:
    material = getattr(surf, "material_post", None)
    material_data = getattr(material, "material_data", None)
    if not material_data:
        return None

    min_wl = material_data.get("min_wavelength")
    max_wl = material_data.get("max_wavelength")
    if min_wl is None or max_wl is None:
        return None

    name = getattr(material, "name", None) or repr(material)
    return min_wl, max_wl, name


def _wavelength_range_diagnostic(
    index: int, name: str, min_wl: float, max_wl: float, value: float
) -> Diagnostic:
    return Diagnostic(
        severity=Severity.WARNING,
        code="OPT009",
        message=(
            f"Wavelength {value} um is outside the valid dispersion range "
            f"[{min_wl}, {max_wl}] um of material '{name}' on surface {index}."
        ),
        fix=(
            "Use a wavelength within the material's data range, "
            "or choose a different material."
        ),
        where=index,
        doc_url=_doc_url("OPT009"),
    )


def check_wavelength_outside_material_range(lens: Optic) -> list[Diagnostic]:
    """OPT009: a wavelength falls outside a material's dispersion data range."""
    findings: list[Diagnostic] = []
    for index, surf in enumerate(lens.surfaces.surfaces):
        dispersion_range = _material_dispersion_range(surf)
        if dispersion_range is None:
            continue
        min_wl, max_wl, name = dispersion_range
        findings.extend(
            _wavelength_range_diagnostic(index, name, min_wl, max_wl, w.value)
            for w in lens.wavelengths
            if w.value < min_wl or w.value > max_wl
        )
    return findings


# --------------------------------------------------------------------------
# OPT010: a thickness has an unexpected sign, or is exactly zero.
# --------------------------------------------------------------------------


def _thickness_sign_diagnostic(
    index: int, thickness: ScalarOrArray, expect_negative: bool, reflections: int
) -> Diagnostic | None:
    is_bad = thickness >= 0 if expect_negative else thickness <= 0
    if not is_bad:
        return None
    expected = "negative" if expect_negative else "positive"
    return Diagnostic(
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

        diagnostic = _thickness_sign_diagnostic(
            index, thickness, reflections % 2 == 1, reflections
        )
        if diagnostic is not None:
            findings.append(diagnostic)

    return findings


# --------------------------------------------------------------------------
# OPT012: a probe trace shows no rays reaching the image surface.
# --------------------------------------------------------------------------


def _system_ready_for_probe_trace(lens: Optic) -> bool:
    preconditions = (
        lens.surfaces.num_surfaces >= 2,
        lens.wavelengths.num_wavelengths > 0,
        any(w.is_primary for w in lens.wavelengths),
        lens.aperture is not None,
        any(surf.is_stop for surf in lens.surfaces.surfaces),
        len(lens.fields.fields) > 0,
    )
    return all(preconditions)


def _probe_trace_reaches_image(lens: Optic) -> bool:
    """True if a probe trace reaches the image, or can't be judged."""
    hx, hy = lens.fields.get_field_coords()[0]
    try:
        lens.trace(hx, hy, lens.primary_wavelength, num_rays=8)
    except Exception:
        # Ray tracing failures surface through their own exceptions; a probe
        # trace that raises is not this check's concern to report.
        return True

    intensity = lens.surfaces.intensity[-1, :]
    return bool(be.any(intensity != 0))


def check_no_rays_reach_image(lens: Optic) -> list[Diagnostic]:
    """OPT012: a probe trace shows no rays reaching the image surface.

    Only runs once the system is complete enough to trace: it needs
    wavelengths, a primary wavelength, an aperture, a stop, at least two
    surfaces, and at least one field. Any check failure above already
    reports the root cause, so this is skipped rather than duplicated.
    """
    if not _system_ready_for_probe_trace(lens):
        return []
    if _probe_trace_reaches_image(lens):
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
