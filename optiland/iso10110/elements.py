"""ISO 10110 Lens Element Inference

Infers lens elements (singlets, cemented doublets, etc.) from an optic's
surface sequence.  Optiland is purely surface-based; there is no first-class
LensElement concept.  This module derives elements by grouping consecutive
glass-bounded surface intervals.

Bernhard Lutzer, 2026
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.optic.optic import Optic
    from optiland.surfaces.standard_surface import Surface


# ---------------------------------------------------------------------------
# Air / glass discrimination
# ---------------------------------------------------------------------------


def is_air(material) -> bool:
    """Return True if *material* is air or vacuum (n ≈ 1)."""
    if material is None:
        return True
    try:
        n = float(np.asarray(be.to_numpy(be.array(material.n(0.5876)))))
        return n < 1.01
    except Exception:
        return False


def _physical_aperture_radius(aperture) -> float | None:
    """Return the max radial extent of a physical aperture, or None.

    Mirrors the priority given to ``surface.aperture.extent`` in
    ``optiland.visualization.system.surface`` — an explicitly configured
    physical aperture (e.g. ``RadialAperture``) defines the actual
    manufactured/clipped edge of the surface, so it must take priority over
    any ray-traced or paraxial estimate: that is the whole point of setting
    one, whether it is larger *or* smaller than the illuminated ray bundle
    (vignetting). Returns None when the aperture has no finite extent (e.g.
    an unbounded boolean combination), so the caller can fall back.
    """
    try:
        extent = np.abs(np.asarray(be.to_numpy(be.array(aperture.extent)), dtype=float))
    except Exception:
        return None
    finite = extent[np.isfinite(extent)]
    if finite.size == 0:
        return None
    return float(finite.max())


# ---------------------------------------------------------------------------
# LensElement
# ---------------------------------------------------------------------------


@dataclass
class LensElement:
    """A logical lens element inferred from a consecutive run of glass surfaces.

    A *singlet* has two bounding surfaces and one glass segment.
    A *cemented doublet* has three surfaces and two glass segments (the middle
    surface is the cement interface).  In general an N-component cemented group
    has N+1 surfaces.

    Attributes:
        surfaces: Ordered list of Surface objects that bound this element,
            including the final air-exit surface.
        surface_indices: Corresponding global indices in the SurfaceGroup.
    """

    surfaces: list[Surface]
    surface_indices: list[int]

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def is_cemented(self) -> bool:
        """True if this element consists of more than one glass component."""
        return len(self.surfaces) > 2

    @property
    def num_components(self) -> int:
        """Number of individual glass pieces (1 for singlet, 2 for doublet…)."""
        return len(self.surfaces) - 1

    @property
    def glass_materials(self):
        """List of glass materials, one per component (excludes trailing air)."""
        return [s.material_post for s in self.surfaces[:-1]]

    def semi_aperture(self, surf_local_idx: int, optic: Optic) -> float:
        """Return the semi-aperture of a surface within this element.

        An explicitly configured physical aperture (e.g. ``RadialAperture``)
        takes priority, since it defines the actual manufactured/clipped
        edge of the surface — this matters both when it is smaller than the
        ray bundle (vignetting: the drawing must show the true, smaller
        mechanical edge, not the unclipped ray height) and when it is
        larger (an intentionally oversized blank). Falls back to
        ``semi_aperture`` populated by ``optic.updater.update_paraxial()``,
        and finally to a direct paraxial marginal-ray estimate when neither
        is available.

        Args:
            surf_local_idx: 0-based index within this element's surface list.
            optic: The parent optic (used for the paraxial fallback).
        """
        surf = self.surfaces[surf_local_idx]

        if surf.aperture is not None:
            r = _physical_aperture_radius(surf.aperture)
            if r is not None:
                return r

        if surf.semi_aperture is not None:
            return float(abs(np.asarray(be.to_numpy(be.array(surf.semi_aperture)))))
        # Paraxial fallback
        from optiland.paraxial import Paraxial

        p = Paraxial(optic)
        heights, _ = p.marginal_ray()
        global_idx = self.surface_indices[surf_local_idx]
        return float(
            abs(np.asarray(be.to_numpy(be.array(heights[global_idx]))).ravel()[0])
        )

    def max_semi_aperture(self, optic: Optic) -> float:
        """Return the maximum semi-aperture across all surfaces of this element."""
        return max(self.semi_aperture(i, optic) for i in range(len(self.surfaces)))

    def center_thickness(self, optic) -> float:
        """Total on-axis thickness of the element (front vertex to rear vertex)."""
        z_front = float(
            np.asarray(be.to_numpy(be.array(self.surfaces[0].geometry.cs.z)))
        )
        z_rear = float(
            np.asarray(be.to_numpy(be.array(self.surfaces[-1].geometry.cs.z)))
        )
        return abs(z_rear - z_front)

    def component_thicknesses(self, optic) -> list[float]:
        """Center thickness of each individual glass component."""
        thicknesses = []
        for i in range(len(self.surfaces) - 1):
            z0 = float(
                np.asarray(be.to_numpy(be.array(self.surfaces[i].geometry.cs.z)))
            )
            z1 = float(
                np.asarray(be.to_numpy(be.array(self.surfaces[i + 1].geometry.cs.z)))
            )
            thicknesses.append(abs(z1 - z0))
        return thicknesses

    def radii(self) -> list[float]:
        """Radius of curvature for every surface (including the air-exit surface)."""
        result = []
        for surf in self.surfaces:
            try:
                r = float(np.asarray(be.to_numpy(be.array(surf.geometry.radius))))
            except AttributeError:
                r = float("inf")
            result.append(r)
        return result


# ---------------------------------------------------------------------------
# Element identification
# ---------------------------------------------------------------------------


def identify_elements(optic: Optic) -> list[LensElement]:
    """Infer lens elements from *optic*'s surface sequence.

    Skips the object surface (index 0) and the image surface (index -1).
    Consecutive surfaces whose ``material_post`` is glass (n > 1.01) are
    grouped into a single :class:`LensElement`.  The final surface of each
    element is the one whose ``material_post`` transitions back to air.

    This grouping is purely a glass/air-transition heuristic and does not
    understand reflective (mirror) surfaces or images immersed in a
    non-air medium — both configurations emit a ``UserWarning`` and may
    yield a missing or incomplete element list rather than raising, since
    mirror support is not implemented yet.

    Returns:
        List of :class:`LensElement` in optical order.
    """
    surfaces = optic.surfaces.surfaces  # tuple, index 0 = object

    elements: list[LensElement] = []
    in_glass = False
    current_surfaces: list[Surface] = []
    current_indices: list[int] = []
    has_reflective = False

    # Walk real surfaces only (skip object [0] and image [-1])
    for idx in range(1, len(surfaces) - 1):
        surf = surfaces[idx]
        if getattr(getattr(surf, "interaction_model", None), "is_reflective", False):
            has_reflective = True
        glass = not is_air(surf.material_post)

        if glass:
            if not in_glass:
                in_glass = True
            current_surfaces.append(surf)
            current_indices.append(idx)
        else:
            # material_post is air
            if in_glass:
                # This surface closes the element
                current_surfaces.append(surf)
                current_indices.append(idx)
                elements.append(
                    LensElement(
                        surfaces=list(current_surfaces),
                        surface_indices=list(current_indices),
                    )
                )
                in_glass = False
                current_surfaces = []
                current_indices = []
            # else: air gap between elements — skip

    if has_reflective:
        warnings.warn(
            "identify_elements() does not support reflective (mirror) surfaces: "
            "a mirror's material_post is aliased to material_pre, so it looks "
            "like an air gap to this glass/air-transition grouping. Elements "
            "adjacent to a mirror may be dropped or grouped incorrectly, and the "
            "resulting ISO 10110 drawings may be incomplete or empty.",
            stacklevel=2,
        )
    if in_glass:
        warnings.warn(
            "The optic's final surface is still immersed in a non-air material "
            "(no closing air transition found before the image surface); the "
            "trailing lens element could not be closed and will be missing from "
            "the ISO 10110 drawings.",
            stacklevel=2,
        )

    return elements
