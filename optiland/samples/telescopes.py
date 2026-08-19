"""Defines sample telescope optical systems."""

from __future__ import annotations

import optiland.backend as be
from optiland import optic, physical_apertures


class HubbleTelescope(optic.Optic):
    """Hubble Space Telescope optical design.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 200.
    """

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, thickness=4910.01016)

        obscuration = physical_apertures.RadialAperture(r_max=be.inf, r_min=177.80035)

        self.surfaces.add(
            index=2,
            radius=-11040.02286,
            thickness=-4910.01016,
            material="mirror",
            is_stop=True,
            conic=-1.001152,
            aperture=obscuration,
        )
        self.surfaces.add(
            index=3,
            radius=-1349.31166,
            thickness=6365.20955,
            material="mirror",
            conic=-1.483014,
        )
        self.surfaces.add(index=4, radius=-635.38227)

        self.set_aperture(aperture_type="EPD", value=2400)

        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=0.15)

        self.wavelengths.add(value=0.55, is_primary=True)
