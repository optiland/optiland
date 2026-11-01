"""Physical Thickness Limits

Turns manufacturing limits on center and edge thickness into operand specs for
``OptimizationProblem.add_physical_constraints`` (opt v2 PR8.4, decision V13).
The helper is opt-in: nothing here runs unless the user asks for it.

A gap is the space between two consecutive lens surfaces, so the object and
image gaps are left out. A gap is air when its index is within 0.01 of 1. Limits
are magnitudes: after a mirror, where thickness is negative, they are mirrored
to the negative side.

Experimental: part of the opt v2 facade, subject to change until M10.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.optimization.failure import ConfigurationError

if TYPE_CHECKING:
    from optiland.optic.optic import Optic

#: Accepted values of ``mode``.
PHYSICAL_MODES: tuple[str, ...] = ("constraint", "penalty")

_AIR_INDEX_TOL = 1e-2


@dataclass(frozen=True)
class ThicknessLimits:
    """Bounds of center and edge thickness, for one medium.

    Attributes:
        min_center: Smallest center thickness.
        max_center: Largest center thickness.
        min_edge: Smallest edge thickness.
        max_edge: Largest edge thickness.
    """

    min_center: float | None = None
    max_center: float | None = None
    min_edge: float | None = None
    max_edge: float | None = None

    @property
    def edge(self) -> bool:
        """Whether any edge bound is set."""
        return self.min_edge is not None or self.max_edge is not None

    def specs(self) -> list[tuple[str, float | None, float | None]]:
        """``(operand type, lower, upper)`` of each bounded thickness."""
        specs = [
            ("center_thickness", self.min_center, self.max_center),
            ("edge_thickness", self.min_edge, self.max_edge),
        ]
        return [s for s in specs if s[1] is not None or s[2] is not None]


def _is_air(optic: Optic, surface_number: int) -> bool:
    material = optic.surfaces[surface_number].material_post
    index = np.ravel(be.to_numpy(material.n(optic.primary_wavelength)))[0]
    return abs(float(index) - 1.0) < _AIR_INDEX_TOL


def thickness_specs(
    optic: Optic, glass: ThicknessLimits, air: ThicknessLimits
) -> list[tuple[str, float | None, float | None, dict]]:
    """Operand specs of the limits, gap by gap.

    Args:
        optic: The optic. Missing semi-apertures are initialized when an edge
            limit needs them.
        glass: Limits of the glass gaps.
        air: Limits of the air gaps.

    Returns:
        ``(operand type, min_val, max_val, input_data)`` tuples.

    Raises:
        ConfigurationError: If no limit is set.
    """
    if not (glass.specs() or air.specs()):
        raise ConfigurationError("add_physical_constraints needs at least one limit.")
    gaps = range(1, len(optic.surfaces.surfaces) - 2)
    if (glass.edge or air.edge) and any(
        optic.surfaces[s].semi_aperture is None for s in (*gaps, gaps.stop)
    ):
        optic.updater.update_paraxial()

    specs = []
    for s in gaps:
        limits = air if _is_air(optic, s) else glass
        thickness = float(np.ravel(be.to_numpy(optic.surfaces.get_thickness(s)))[0])
        for operand, lower, upper in limits.specs():
            if thickness < 0:
                lower, upper = (
                    None if upper is None else -upper,
                    None if lower is None else -lower,
                )
            specs.append((operand, lower, upper, {"optic": optic, "surface_number": s}))
    return specs
