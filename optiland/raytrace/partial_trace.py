"""Partial Trace Eligibility Module

Decides where a real-ray trace may resume after one optimization variable
changes (opt v2 PR3.1). A resumed trace starts from the ray state recorded
before surface ``s`` (see :mod:`optiland.raytrace.snapshot`), so it equals a
full trace only when the variable leaves everything before ``s`` untouched:
the surfaces, the launched rays and the ray aiming.

The rules are deliberately conservative. Any case they do not recognize falls
back to a full trace (``None``). Each rule is a separate private predicate so
it can be tested on its own:

1. The variable mutates exactly one surface ``s`` and triggers no pickup or
   solve that writes to a surface before ``s``.
2. ``s`` is strictly after the aperture stop, and the system aperture does not
   depend on surfaces after the stop.
3. The field type does not depend on image-side quantities.
4. Ray aiming does not depend on surfaces after the stop.
5. No coordinate system or shared geometry upstream of ``s`` is affected.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.optic import Optic

# Ray aiming modes whose launch depends only on surfaces up to the stop:
# paraxial aiming to the entrance pupil and real (iterative) aiming to the
# stop. Robust aiming keeps calibration state across calls, and the cached
# wrapper warm-starts from earlier traces, so both are excluded.
_UPSTREAM_AIMING_MODES = ("paraxial", "iterative")


def partial_trace_start(optic: Optic, variable: Any) -> int | None:
    """Surface index where a trace may resume after ``variable`` changes.

    Args:
        optic: The optic whose trace would be resumed.
        variable: An optimization ``Variable`` or its behavior object.

    Returns:
        The index ``s`` of the one surface the variable mutates when all five
        eligibility rules hold, so a trace may resume at ``s`` from a snapshot
        taken before the change. None when any rule fails or the case is not
        recognized, meaning a full trace.
    """
    behavior = _behavior(variable)
    if getattr(behavior, "optic", None) is not optic:
        return None
    s = _mutated_surface(behavior)
    if s is None or not 0 < s < len(optic.surfaces.surfaces):
        return None
    eligible = (
        _no_upstream_pickup_or_solve(optic, s)
        and _after_stop_with_upstream_aperture(optic, s)
        and _field_is_object_side(optic)
        and _aiming_is_upstream(optic)
        and _coordinates_unaffected(optic, s, behavior)
    )
    return s if eligible else None


def _behavior(variable: Any) -> Any:
    """The variable behavior behind a ``Variable`` wrapper (or itself)."""
    from optiland.optimization.variable.variable import Variable

    return variable.variable if isinstance(variable, Variable) else variable


def _single_surface_types() -> tuple[type, ...]:
    """Variable behaviors that mutate only the surface they name.

    Radius, conic, asphere / polynomial / Zernike coefficients, the thickness
    after the surface, and the material after the surface. Matched by exact
    type: subclasses may mutate more and fall back to a full trace.
    """
    from optiland.optimization.variable.asphere_coeff import AsphereCoeffVariable
    from optiland.optimization.variable.conic import ConicVariable
    from optiland.optimization.variable.index import IndexVariable
    from optiland.optimization.variable.material import MaterialVariable
    from optiland.optimization.variable.polynomial_coeff import (
        PolynomialCoeffVariable,
    )
    from optiland.optimization.variable.radius import RadiusVariable
    from optiland.optimization.variable.reciprocal_radius import (
        ReciprocalRadiusVariable,
    )
    from optiland.optimization.variable.thickness import ThicknessVariable
    from optiland.optimization.variable.zernike_coeff import ZernikeCoeffVariable

    return (
        RadiusVariable,
        ReciprocalRadiusVariable,
        ConicVariable,
        AsphereCoeffVariable,
        PolynomialCoeffVariable,
        ZernikeCoeffVariable,
        ThicknessVariable,
        MaterialVariable,
        IndexVariable,
    )


def _mutated_surface(behavior: Any) -> int | None:
    """Rule 1 (first half): the one surface a variable mutates, if known.

    Args:
        behavior: A variable behavior object.

    Returns:
        The surface index, or None when the variable type is not known to
        mutate exactly one surface.
    """
    if type(behavior) not in _single_surface_types():
        return None
    s = getattr(behavior, "surface_number", None)
    if isinstance(s, bool) or not isinstance(s, int):
        return None
    return s


def _solve_first_written_surface(solve: Any) -> int:
    """Lowest surface index a solve may write to (0 when unknown)."""
    from optiland.solves.curvature import CurvatureSolve
    from optiland.solves.quick_focus import QuickFocusSolve
    from optiland.solves.thickness import ThicknessSolve

    if isinstance(solve, CurvatureSolve | ThicknessSolve):
        # Curvature solves set the radius of surface_idx; thickness solves
        # shift surface_idx and every surface after it.
        return solve.surface_idx
    if isinstance(solve, QuickFocusSolve):
        return len(solve.optic.surfaces.surfaces) - 1
    return 0


def _no_upstream_pickup_or_solve(optic: Optic, s: int) -> bool:
    """Rule 1 (second half): no pickup or solve writes before surface ``s``.

    A pickup is triggered by the variable when it reads from surface ``s`` or
    later of this optic. Solves are re-applied on every update and their
    inputs are not tracked, so any solve that writes before ``s`` counts as
    triggered.

    Args:
        optic: The optic.
        s: The surface the variable mutates.

    Returns:
        True when no pickup or solve can change a surface before ``s``.
    """
    for pickup in optic.pickups.pickups:
        triggered = pickup.source_optic is optic and pickup.source_surface_idx >= s
        if triggered and pickup.target_surface_idx < s:
            return False
    return all(
        _solve_first_written_surface(solve) >= s for solve in optic.solves.solves
    )


def _after_stop_with_upstream_aperture(optic: Optic, s: int) -> bool:
    """Rule 2: ``s`` is after the stop and the aperture is set upstream of it.

    Entrance pupil diameter, float-by-stop size and object-space NA depend
    only on surfaces up to the stop. Image-space F/# (and any other or unknown
    aperture type) depends on the whole system and is ineligible.

    Args:
        optic: The optic.
        s: The surface the variable mutates.

    Returns:
        True when both conditions hold.
    """
    from optiland.aperture.epd import EPDAperture
    from optiland.aperture.float_by_stop import FloatByStopAperture
    from optiland.aperture.object_na import ObjectNAAperture

    if s <= optic.surfaces.stop_index:
        return False
    return type(optic.aperture) in (
        EPDAperture,
        FloatByStopAperture,
        ObjectNAAperture,
    )


def _field_is_object_side(optic: Optic) -> bool:
    """Rule 3: the field type is defined in object space.

    Angle and object-height fields are eligible. Paraxial and real image
    height fields (and any other or unknown type) solve through the whole
    system and are ineligible.

    Args:
        optic: The optic.

    Returns:
        True for an object-side field type.
    """
    from optiland.fields.field_types.angle import AngleField
    from optiland.fields.field_types.object_height import ObjectHeightField

    definition = getattr(optic.fields, "field_definition", None)
    return type(definition) in (AngleField, ObjectHeightField)


def _aiming_is_upstream(optic: Optic) -> bool:
    """Rule 4: ray aiming depends only on surfaces up to the stop.

    Paraxial aiming to the entrance pupil and iterative real aiming to the
    stop are eligible, without the cache wrapper. The configured aimer is
    checked too when the ray generator has already built it.

    Args:
        optic: The optic.

    Returns:
        True for an upstream-only aiming configuration.
    """
    from optiland.rays.ray_aiming.iterative import IterativeRayAimer
    from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer

    tracer = optic.ray_tracer
    config = getattr(tracer, "ray_aiming_config", None)
    if not isinstance(config, dict):
        return False
    mode = config.get("mode")
    if mode not in _UPSTREAM_AIMING_MODES or config.get("cache", False):
        return False
    generator = getattr(tracer, "ray_generator", None)
    if generator is None:
        return False
    if getattr(generator, "_current_config", None) == config:
        expected = ParaxialRayAimer if mode == "paraxial" else IterativeRayAimer
        return type(generator.aimer) is expected
    # The generator rebuilds its aimer from the config on the next trace.
    return True


def _coordinates_unaffected(optic: Optic, s: int, behavior: Any) -> bool:
    """Rule 5: no coordinate system or geometry upstream of ``s`` is affected.

    Surfaces before ``s`` must not share a geometry or a coordinate system
    (directly or through a reference chain) with surface ``s`` or later. A
    thickness variable rebuilds every surface's global ``z`` from the
    thicknesses, so it also needs a global-``z`` system whose upstream
    positions already equal that rebuild exactly.

    Args:
        optic: The optic.
        s: The surface the variable mutates.
        behavior: The variable behavior object.

    Returns:
        True when nothing upstream of ``s`` can move or change shape.
    """
    from optiland.optimization.variable.thickness import ThicknessVariable

    surfaces = list(optic.surfaces.surfaces)
    downstream_geometries = {id(surface.geometry) for surface in surfaces[s:]}
    downstream_cs = {id(surface.geometry.cs) for surface in surfaces[s:]}
    for surface in surfaces[:s]:
        if id(surface.geometry) in downstream_geometries:
            return False
        cs = surface.geometry.cs
        while cs is not None:
            if id(cs) in downstream_cs:
                return False
            cs = getattr(cs, "reference_cs", None)

    if type(behavior) is ThicknessVariable:
        return _upstream_positions_stable(surfaces, s)
    return True


def _upstream_positions_stable(surfaces: list, s: int) -> bool:
    """Whether a thickness update leaves surfaces ``0..s`` where they are.

    ``OpticUpdater.set_thickness`` rebuilds ``cs.z`` of surfaces 1 onward as
    cumulative thicknesses from 0. That is exact for surfaces up to ``s`` only
    when the beam path runs along global ``z`` and their current positions
    already equal the rebuild bit for bit.
    """
    import optiland.backend as be
    from optiland.paraxial_path import build_paraxial_path

    if not build_paraxial_path(surfaces).positions_are_global_z:
        return False
    z = be.array(0.0)
    for k in range(1, s + 1):
        if k > 1:
            t_prev = surfaces[k - 1].thickness
            z = z + (t_prev if hasattr(t_prev, "detach") else be.array(t_prev))
        if be.to_numpy(surfaces[k].geometry.cs.z).item() != be.to_numpy(z).item():
            return False
    return True
