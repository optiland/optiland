"""ISO 10110 Drawing Generation for Optiland

This package generates ISO 10110-compliant fabrication drawings for optical
elements directly from an optiland :class:`~optiland.optic.optic.Optic`.

Quick start — singlet
---------------------
::

    from optiland.iso10110 import DrawingSpec, ISO10110Report

    spec = DrawingSpec(lens, project_name="50mm Photographic Lens",
                       organisation="Acme Optics")

    # Per-surface tolerances (ISO 10110-5, -6, -7, -8, -9, …)
    spec.set_surface(1, irregularity=0.5, centration=0.5,
                     imperfections="2×0.04", roughness="P2",
                     coating="AR 400-700 nm R<0.5%")
    spec.set_surface(2, irregularity=1.0, imperfections="1×0.063")

    # Per-element glass quality (ISO 10110-2, -3, -4)
    spec.set_element(0, birefringence=10, bubbles="1×0.16",
                     nh_class="NH040", striae_shadowgraph="A")

    # Title-block fields (ISO 7200), kept separate from optical quality
    spec.set_title(0, part_number="OPT-001", drawn_by="BL", revision="A")

    report = ISO10110Report(spec)
    report.save_pdf("output/drawings.pdf")
    report.save_dxf("output/")

Quick start — cemented doublet
-------------------------------
For cemented lenses each glass component may carry independent quality specs::

    spec.set_material(0, 0, birefringence=5,  nh_class="NH040")  # crown
    spec.set_material(0, 1, birefringence=10, nh_class="NH100")  # flint

Scope / limitations
--------------------
:func:`identify_elements` groups surfaces by glass/air transitions. It does
not currently support reflective (mirror) surfaces or images immersed in a
non-air medium — these configurations emit a ``UserWarning`` and may produce
a missing or incomplete element list (and therefore missing or incomplete
drawings) rather than raising an error.
"""

from __future__ import annotations

from optiland.iso10110.drawing import ElementDrawing, draw_element
from optiland.iso10110.elements import LensElement, identify_elements
from optiland.iso10110.notation import ElementSpec, SurfaceSpec
from optiland.iso10110.report import ISO10110Report
from optiland.iso10110.spec import DrawingSpec
from optiland.iso10110.style import DrawingStyle

__all__ = [
    "DrawingSpec",
    "DrawingStyle",
    "ISO10110Report",
    "ElementDrawing",
    "LensElement",
    "SurfaceSpec",
    "ElementSpec",
    "identify_elements",
    "draw_element",
]
