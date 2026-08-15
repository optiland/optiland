"""Defines sample eyepiece optical systems."""

from __future__ import annotations

import optiland.backend as be
from optiland import optic


class EyepieceErfle(optic.Optic):
    """An Erfle eyepiece design.

    Based on U.S. Patent 1,479,229 by Heinrich Erfle.
    """

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=be.inf, thickness=15.224, is_stop=True)
        self.surfaces.add(index=2, radius=269.0, thickness=25.1, material="L-BSL7")
        self.surfaces.add(index=3, radius=-125.9, thickness=36.5)
        self.surfaces.add(index=4, radius=93.6, thickness=18.5, material="N-BAK2")
        self.surfaces.add(index=5, radius=-93.6, thickness=4.1, material="N-F2")
        self.surfaces.add(index=6, radius=2550.0, thickness=0.19)
        self.surfaces.add(index=7, radius=93.6, thickness=18.5, material="N-BAK2")
        self.surfaces.add(index=8, radius=-93.6, thickness=4.1, material="N-F2")
        self.surfaces.add(index=9, radius=2550.0, thickness=32.685)
        self.surfaces.add(index=10)

        self.set_aperture(aperture_type="EPD", value=4.0)

        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=14)
        self.fields.add(y=20)

        self.wavelengths.add(value=0.4861)
        self.wavelengths.add(value=0.5876, is_primary=True)
        self.wavelengths.add(value=0.6563)
