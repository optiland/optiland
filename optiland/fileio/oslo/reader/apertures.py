"""OSLO special apertures and aperture-group Boolean rules (reference pp. 49–50)."""

from __future__ import annotations

import math
from typing import Any

from optiland.physical_apertures import (
    BaseAperture,
    DifferenceAperture,
    EllipticalAperture,
    IntersectionAperture,
    PolygonAperture,
    RadialAperture,
    RectangularAperture,
    RotatedAperture,
    UnclippedAperture,
    UnionAperture,
)


def physical_aperture(
    data: dict[str, Any], scale: float, check: bool = True
) -> BaseAperture | None:
    """Build a surface aperture using OSLO transmission and grouping rules.

    Args:
        data: Resolved surface data, with any unsupported holes removed by the
            converter. Omitted shape coordinates follow OSLO's zero defaults.
        scale: Positive lens-unit conversion factor to native millimeters.
        check: Whether physical boundaries clip rays. Disabling checking keeps
            their display shape in an ``UnclippedAperture``.

    Returns:
        The composed surface boundary, or None if no boundary is defined.

    Raises:
        ValueError: If a shape is unsupported, degenerate or non-convex.
    """
    radius = data.get("AP", 0.0)
    # Omit huge unchecked drawing outlines (typically the object plane).
    # A checked radius must retain its clipping boundary at any lens-unit scale.
    outer = (
        RadialAperture(r_max=radius * scale)
        if radius > 0 and (data.get("aperture_checked") or radius < 1e6)
        else None
    )
    groups: dict[int, list[tuple[BaseAperture, int]]] = {}
    holes = None
    for spec in data.get("special_apertures", {}).values():
        action = int(spec.get("AAC", 4))
        if action not in {1, 2, 4}:
            continue  # The parser has already diagnosed unknown actions.
        kind = int(spec.get("ATP", 1))
        if kind in {1, 2}:
            x0, x1, y0, y1 = [
                spec.get(k, 0.0) * scale for k in ("AX1", "AX2", "AY1", "AY2")
            ]
            if x0 >= x1 or y0 >= y1:
                raise ValueError("OSLO special aperture requires increasing bounds")
            if kind == 1:
                ap = EllipticalAperture(
                    (x1 - x0) / 2, (y1 - y0) / 2, (x1 + x0) / 2, (y1 + y0) / 2
                )
            else:
                ap = RectangularAperture(x0, x1, y0, y1)
            if spec.get("AAN", 0.0):
                # AAN rotates around the aperture centroid, not the surface
                # origin: OSLO Optics Reference, printed p. 105.
                # https://lambdares.com/hubfs/Support/support/OSLOOpticsReference_Sep21.pdf#page=105
                ap = RotatedAperture(
                    ap, math.radians(spec["AAN"]), (x1 + x0) / 2, (y1 + y0) / 2
                )
        elif kind in {3, 4}:
            vertices = [
                (spec.get(f"AVX{i}", 0.0), spec.get(f"AVY{i}", 0.0))
                for i in range(1, kind + 1)
            ]
            # OSLO requires nondegenerate triangles and strictly convex,
            # cyclically ordered quadrangles (Program Reference pp. 49–50).
            # Validate in lens units so unit conversion cannot collapse area.
            # https://lambdares.com/hubfs/Support/support/oslo/oslo_releases/OSLOProgramReference.pdf#page=63
            turns = []
            for i, (ax, ay) in enumerate(vertices):
                bx, by = vertices[(i + 1) % kind]
                cx, cy = vertices[(i + 2) % kind]
                turns.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
            if not (all(turn > 0 for turn in turns) or all(turn < 0 for turn in turns)):
                raise ValueError(
                    "OSLO polygon aperture must be nondegenerate and strictly convex"
                )
            ap = PolygonAperture(
                [x * scale for x, _ in vertices], [y * scale for _, y in vertices]
            )
        else:
            raise ValueError(f"OSLO special aperture type {kind} is not supported")
        if action == 1:
            # The converter only retains holes on neutral surfaces. They do
            # not constrain transmission; keep their shape for display alone.
            holes = ap if holes is None else UnionAperture(holes, ap)
        else:
            groups.setdefault(int(spec.get("AGN", 0)), []).append((ap, action))
    combined = None
    for members in groups.values():
        group = None
        for ap, action in members:
            if action == 4:
                group = ap if group is None else IntersectionAperture(group, ap)
        if group is None:
            group = RadialAperture(r_max=math.inf)
        for ap, action in members:
            if action == 2:
                group = DifferenceAperture(group, ap)
        combined = group if combined is None else UnionAperture(combined, group)
    if combined is None:
        if outer is None and holes is not None:
            return UnclippedAperture(holes)
        return (
            UnclippedAperture(outer)
            if outer is not None and (not check or not data.get("aperture_checked"))
            else outer
        )
    if outer is not None and data.get("aperture_checked"):
        combined = IntersectionAperture(outer, combined)
    return combined if check else UnclippedAperture(combined)
