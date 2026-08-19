"""Defines sample infrared optical systems."""

from __future__ import annotations

import optiland.backend as be
from optiland import materials, optic


class InfraredTriplet(optic.Optic):
    """An infrared air-spaced triplet lens.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 54.
    """

    def __init__(self):
        super().__init__()

        germanium = materials.IdealMaterial(n=4.002)

        # https://www.spectral-systems.com/technical-data-sheet/znse-zinc-selenide
        ZnSe = materials.IdealMaterial(n=2.4028)

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(
            index=1,
            radius=10.4578,
            thickness=0.5901,
            material=germanium,
            is_stop=True,
        )
        self.surfaces.add(index=2, radius=14.1079, thickness=4.3909)
        self.surfaces.add(index=3, radius=-15.8842, thickness=0.59, material=ZnSe)
        self.surfaces.add(index=4, radius=-18.2105, thickness=5.6218)
        self.surfaces.add(index=5, radius=2.5319, thickness=0.3918, material=germanium)
        self.surfaces.add(index=6, radius=2.4308, thickness=1.3065)
        self.surfaces.add(index=7)

        self.set_aperture(aperture_type="imageFNO", value=2)

        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=2.8)
        self.fields.add(y=4)

        self.wavelengths.add(value=10.6, is_primary=True)


class InfraredTripletF4(optic.Optic):
    """A 4-inch infrared triplet lens with f/4.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 57.
    """

    def __init__(self):
        super().__init__()

        # refractive index at 4.2 µm
        germanium = materials.IdealMaterial(n=4.002)
        silicon = materials.IdealMaterial(n=3.4222)

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=2.0721, thickness=0.1340, material=silicon)
        self.surfaces.add(index=2, radius=3.5488, thickness=0.2392)
        self.surfaces.add(index=3, thickness=0.6105, is_stop=True)
        self.surfaces.add(index=4, radius=13.7583, thickness=0.1, material=germanium)
        self.surfaces.add(index=5, radius=1.7491, thickness=0.8768)
        self.surfaces.add(index=6, thickness=0.1462, material=silicon)
        self.surfaces.add(index=7, radius=-3.5850, thickness=2.8386)
        self.surfaces.add(index=8)

        self.set_aperture(aperture_type="imageFNO", value=4)

        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=2.45)
        self.fields.add(y=3.5)

        self.wavelengths.add(value=4.2, is_primary=True)
