"""ISO 10110 Bridge Module

This module converts between the fabrication tolerances of an ISO 10110
drawing specification (``optiland.iso10110.DrawingSpec``) and the tolerances
of a ``Tolerancing`` problem.

Field mapping, drawing to tolerance (surface numbers are the optic's):

- ``SurfaceSpec.r_tolerance`` (mm) to ``"radius"``, in lens units (mm).
  ``"±a"`` or a bare number ``a`` gives ``[-a, +a]``; ``"+a/-b"`` gives
  ``[-b, +a]``. Flat surfaces are skipped.
- ``3/ A(B/C)`` (ISO 10110-5): ``sag_tolerance`` (A) to ``"fringes"``,
  ``irregularity`` (B) to ``"irregularity"`` with ``kind="B"`` and
  ``rotationally_symmetric`` (C) to ``"irregularity"`` with ``kind="C"``,
  each with limits ``[-v, +v]``. The test wavelength is
  ``test_wavelength`` in nm, converted to µm, or 546.07 nm when unset, as the
  drawn ``3/`` code states. With ``form_unit="nm"`` a value in nm of sag is
  converted to fringes: one fringe is half the test wavelength of sag. The
  clear semi-aperture is ``ca_diameter / 2`` when given.
- ``4/ σ'(δ)`` (ISO 10110-6): the tilt angle σ in arc minutes, the angle of
  the surface normal to the reference axis, to ``"surface_tilt"`` about x and
  y, converted to degrees (σ / 60). With ``centering="per_axis"`` (default)
  each axis gets ``[-σ, +σ]``; with ``centering="rss"`` each axis gets
  ``±σ / √2``, so a tilt at both limits has a total angle of σ. The lateral
  decentration δ in mm goes to ``"surface_decenter"`` in x and y, ``[-δ, +δ]``
  each, in lens units.
- ``ElementSpec.ct_tolerance`` (mm) to ``"thickness"`` after the first
  surface of a singlet, parsed like ``r_tolerance``. With
  ``adjust_thickness=True`` (default) the air space after the element absorbs
  the change, so later surfaces do not move. Cemented elements are skipped:
  the drawn tolerance is on the total thickness, not on one component.

The ISO 10110 data model holds no refractive index or Abbe number
tolerances (homogeneity classes are not index offsets), so no ``"index"`` or
``"abbe"`` tolerances are made. Tolerances are added by name through
``tolerance_registry``: a type that is not registered, a flat surface with a
radius tolerance, or a cemented element with a thickness tolerance is skipped
with a ``UserWarning``, or raises ``ValueError`` when ``strict=True``.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

from optiland.tolerancing.registry import tolerance_registry
from optiland.tolerancing.types.irregularity import IrregularityTolerance
from optiland.tolerancing.types.surface import (
    FringesTolerance,
    RadiusTolerance,
    SurfaceDecenterTolerance,
    SurfaceTiltTolerance,
    ThicknessTolerance,
    _scalar,
)

if TYPE_CHECKING:
    from optiland.iso10110 import DrawingSpec
    from optiland.tolerancing.core import Tolerancing
    from optiland.tolerancing.distributions import Distribution
    from optiland.tolerancing.tolerance import Tolerance

#: Test wavelength in nm of a fringe value with no stated wavelength.
DEFAULT_TEST_WAVELENGTH_NM = 546.07

_ARCMIN_PER_DEGREE = 60.0


def _limits(tol: str | float) -> tuple[float, float]:
    """Return ``(minus, plus)`` of a drawing tolerance: ``"±a"``, ``a`` or
    ``"+a/-b"``."""
    s = str(tol).strip().replace("−", "-").replace("+-", "±")
    if s.startswith("±"):
        a = abs(float(s[1:]))
        return -a, a
    if "/" in s:
        hi, lo = s.split("/", 1)
        return -abs(float(lo)), abs(float(hi))
    a = abs(float(s))
    return -a, a


class _Builder:
    """Adds tolerances by registered name, skipping or raising on failure."""

    def __init__(
        self, tolerancing: Tolerancing, distribution: str | Distribution, strict: bool
    ) -> None:
        self.tolerancing = tolerancing
        self.distribution = distribution
        self.strict = strict
        self.added: list[Tolerance] = []

    def skip(self, message: str) -> None:
        if self.strict:
            raise ValueError(message)
        warnings.warn(f"{message} Skipped.", UserWarning, stacklevel=4)

    def add(self, name: str, minus: float, plus: float, **kwargs) -> None:
        if name not in tolerance_registry:
            self.skip(f"Tolerance type {name!r} is not registered.")
            return
        tolerance = self.tolerancing.add_tolerance(
            name, minus=minus, plus=plus, distribution=self.distribution, **kwargs
        )
        self.added.append(tolerance)


def tolerances_from_drawing(
    tolerancing: Tolerancing,
    drawing: DrawingSpec,
    *,
    distribution: str | Distribution = "normal",
    centering: str = "per_axis",
    adjust_thickness: bool = True,
    strict: bool = False,
) -> list[Tolerance]:
    """Add the tolerances stated by an ISO 10110 drawing specification.

    See the module docstring for the field mapping and unit conversions.
    Only surfaces and elements of ``drawing.elements`` are read.

    Args:
        tolerancing: The tolerancing problem to add to. Its optic must have
            the same surfaces as the drawing's optic.
        drawing: The ISO 10110 drawing specification.
        distribution: The distribution of every new tolerance.
        centering: How the 4/ tilt angle σ is split into tilts about x and
            y: ``"per_axis"`` (``±σ`` each) or ``"rss"`` (``±σ/√2`` each).
        adjust_thickness: Let the air space after an element absorb its
            thickness change.
        strict: Raise instead of warning when a field cannot be mapped.

    Returns:
        The new tolerances, in the order they were added.

    Raises:
        ValueError: If the optics differ in surface count, ``centering`` is
            unknown, or, with ``strict=True``, a field cannot be mapped.
            On an error no tolerance is added.
    """
    if centering not in ("per_axis", "rss"):
        raise ValueError(f'centering must be "per_axis" or "rss", got {centering!r}.')
    if tolerancing.optic.surfaces.num_surfaces != drawing.optic.surfaces.num_surfaces:
        raise ValueError("The drawing and the tolerancing optic differ in surfaces.")
    axis_factor = 1.0 if centering == "per_axis" else 1.0 / math.sqrt(2.0)
    builder = _Builder(tolerancing, distribution, strict)
    try:
        _add_all(builder, drawing, axis_factor, adjust_thickness)
    except ValueError:
        # leave the problem as it was
        for tolerance in builder.added:
            tolerancing.perturbations.remove(tolerance)
        raise
    return builder.added


def _add_all(
    builder: _Builder,
    drawing: DrawingSpec,
    axis_factor: float,
    adjust_thickness: bool,
) -> None:
    """Add the tolerances of every element of the drawing."""
    optic = builder.tolerancing.optic
    for e, element in enumerate(drawing.elements):
        for s in element.surface_indices:
            spec = drawing.get_surface_spec(s)
            if spec.r_tolerance is not None:
                if math.isinf(_scalar(optic.surfaces[s].geometry.radius)):
                    builder.skip(
                        f"Surface {s} is flat, so its radius tolerance is not mapped."
                    )
                else:
                    builder.add("radius", *_limits(spec.r_tolerance), surface_number=s)

            wavelength_nm = (
                DEFAULT_TEST_WAVELENGTH_NM
                if spec.test_wavelength is None
                else spec.test_wavelength
            )
            # fringes per form unit: 1 for fringes, 2 / lambda for nm of sag
            per_unit = 1.0 if spec.form_unit == "fr" else 2.0 / wavelength_nm
            form = {
                "surface_number": s,
                "test_wavelength": wavelength_nm * 1e-3,
                "semi_aperture": None
                if spec.ca_diameter is None
                else spec.ca_diameter / 2.0,
            }
            for name, value, extra in (
                ("fringes", spec.sag_tolerance, {}),
                ("irregularity", spec.irregularity, {"kind": "B"}),
                ("irregularity", spec.rotationally_symmetric, {"kind": "C"}),
            ):
                if value is not None:
                    v = abs(value) * per_unit
                    builder.add(name, -v, v, **form, **extra)

            if spec.centration is not None:
                v = spec.centration / _ARCMIN_PER_DEGREE * axis_factor
                for axis in ("x", "y"):
                    builder.add("surface_tilt", -v, v, surface_number=s, axis=axis)
            if spec.centration_decentration is not None:
                v = abs(spec.centration_decentration)
                for axis in ("x", "y"):
                    builder.add("surface_decenter", -v, v, surface_number=s, axis=axis)

        ct_tolerance = drawing.get_element_spec(e).ct_tolerance
        if ct_tolerance is not None:
            if element.is_cemented:
                builder.skip(
                    f"Element {e} is cemented, so its ct_tolerance is not mapped."
                )
            else:
                first, last = element.surface_indices[0], element.surface_indices[-1]
                builder.add(
                    "thickness",
                    *_limits(ct_tolerance),
                    surface_number=first,
                    adjust=last if adjust_thickness else None,
                )


def _dimension(minus: float, plus: float) -> float | str:
    """Return a drawing dimension tolerance: a number if symmetric."""
    if math.isclose(-minus, plus):
        return plus
    return f"{plus:+g}/{minus:+g}"


def drawing_values_from_tolerancing(
    tolerancing: Tolerancing,
) -> dict[str, dict[int, dict[str, float | str]]]:
    """Report the ISO 10110 drawing fields implied by the tolerances.

    The inverse of :func:`tolerances_from_drawing` for the built-in types.
    The result holds keyword arguments for ``DrawingSpec.set_surface`` under
    ``"surfaces"`` (keyed by surface number) and for
    ``DrawingSpec.set_element`` under ``"elements"`` (keyed by element
    index). Radius and thickness limits are reported as a number when
    symmetric and as ``"+a/-b"`` otherwise. Form, tilt and decenter fields
    are magnitudes, so asymmetric limits report the larger magnitude. Tilts
    are read as ``centering="per_axis"``: σ is the larger tilt limit in arc
    minutes. All fringe values of a surface are stated at the test
    wavelength of its first form tolerance. Other tolerance types, and
    thickness tolerances that are not on the first surface of a singlet, are
    not reported. Clear apertures are not reported.

    Args:
        tolerancing: The tolerancing problem.

    Returns:
        ``{"surfaces": {...}, "elements": {...}}``.
    """
    from optiland.iso10110.elements import identify_elements

    singlets = {
        el.surface_indices[0]: e
        for e, el in enumerate(identify_elements(tolerancing.optic))
        if not el.is_cemented
    }
    surfaces: dict[int, dict[str, float | str]] = {}
    elements: dict[int, dict[str, float | str]] = {}

    def widen(fields: dict, key: str, value: float) -> None:
        fields[key] = max(fields.get(key, 0.0), value)

    for tolerance in tolerancing.perturbations:
        t = getattr(tolerance, "type", None)
        if t is None:
            continue
        minus, plus = tolerance.limits
        magnitude = max(-minus, plus)
        s = getattr(t, "surface_number", None)
        if isinstance(t, ThicknessTolerance):
            if s in singlets:
                elements.setdefault(singlets[s], {})["ct_tolerance"] = _dimension(
                    minus, plus
                )
            continue
        fields = surfaces.setdefault(s, {})
        if isinstance(t, RadiusTolerance):
            fields["r_tolerance"] = _dimension(minus, plus)
        elif isinstance(t, (FringesTolerance, IrregularityTolerance)):
            wavelength_nm = fields.setdefault(
                "test_wavelength", t.test_wavelength * 1e3
            )
            if isinstance(t, FringesTolerance):
                key = "sag_tolerance"
            else:
                key = "irregularity" if t.kind == "B" else "rotationally_symmetric"
            widen(fields, key, magnitude * t.test_wavelength * 1e3 / wavelength_nm)
        elif isinstance(t, SurfaceTiltTolerance):
            widen(fields, "centration", magnitude * _ARCMIN_PER_DEGREE)
        elif isinstance(t, SurfaceDecenterTolerance):
            widen(fields, "centration_decentration", magnitude)
        if not fields:
            del surfaces[s]
    return {"surfaces": surfaces, "elements": elements}
