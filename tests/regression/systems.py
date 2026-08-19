"""Representative Optic systems used by the golden-value regression harness.

Each factory returns a freshly constructed ``Optic`` so tests never share
mutable state. Kept intentionally small and diverse: a simple doublet, a
wide-FOV system that exercises the robust ray aimer, a system with a coating
stack, and a freeform (Forbes Q2D) surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.coatings import SimpleCoating
from optiland.optic import Optic
from optiland.samples.objectives import WideAngle170FOV
from optiland.samples.simple import CementedAchromat

if TYPE_CHECKING:
    from collections.abc import Callable


def build_doublet() -> Optic:
    """A cemented achromatic doublet (existing sample, unmodified)."""
    return CementedAchromat()


def build_wide_fov() -> Optic:
    """A 170-degree FOV lens exercising the robust chief-ray-calibrated aimer."""
    return WideAngle170FOV()


def build_coated_doublet() -> Optic:
    """A doublet with an explicit partial-reflection coating on surface 1."""
    optic = Optic(name="Coated Doublet")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=12.38401,
        thickness=0.4340,
        is_stop=True,
        material="N-BAK1",
        coating=SimpleCoating(reflectance=0.02, transmittance=0.98),
    )
    optic.surfaces.add(index=2, radius=-7.94140, thickness=0.3210, material="SF2")
    optic.surfaces.add(index=3, radius=-48.44396, thickness=19.6059)
    optic.surfaces.add(index=4)

    optic.set_aperture(aperture_type="imageFNO", value=6)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0)
    optic.fields.add(y=5)
    optic.wavelengths.add(value=0.58756180, is_primary=True)
    return optic


def build_forbes_q2d_singlet() -> Optic:
    """A singlet with a Forbes Q2D freeform second surface."""
    optic = Optic(name="Forbes Q2D Singlet")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=25.0, thickness=4.0, is_stop=True, material="N-BK7"
    )
    optic.surfaces.add(
        index=2,
        surface_type="forbes_q2d",
        radius=-100.0,
        thickness=50.0,
        freeform_coeffs={("a", 1, 1): 0.01},
        norm_radius=20.0,
    )
    optic.surfaces.add(index=3)

    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0)
    optic.fields.add(y=5)
    optic.wavelengths.add(value=0.55, is_primary=True)
    return optic


GOLDEN_SYSTEMS: dict[str, Callable[[], Optic]] = {
    "doublet": build_doublet,
    "wide_fov": build_wide_fov,
    "coated_doublet": build_coated_doublet,
    "forbes_q2d_singlet": build_forbes_q2d_singlet,
}
