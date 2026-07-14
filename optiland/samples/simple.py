"""Defines simple sample optical systems."""

from __future__ import annotations

import optiland.backend as be
from optiland import optic


class Edmund_49_847(optic.Optic):
    """Edmund Optics 49-847.

    A specific lens from Edmund Optics.
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(
            index=1,
            thickness=7,
            radius=19.93,
            is_stop=True,
            material="N-SF11",
        )
        self.surfaces.add(index=2, thickness=21.48)
        self.surfaces.add(index=3)

        # add aperture
        self.set_aperture(aperture_type="EPD", value=25.4)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=10)
        self.fields.add(y=14)

        # add wavelength
        self.wavelengths.add(value=0.48613270)
        self.wavelengths.add(value=0.58756180, is_primary=True)
        self.wavelengths.add(value=0.65627250)


class SingletStopSurf2(optic.Optic):
    """A simple singlet lens with the stop on surface 2."""

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, thickness=10.0, radius=63.73364157, material="LAC9")
        self.surfaces.add(
            index=2,
            thickness=92.73834630,
            radius=653.29392320,
            is_stop=True,
        )
        self.surfaces.add(index=3)

        # add aperture
        self.set_aperture(aperture_type="EPD", value=25.0)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=3.5)
        self.fields.add(y=5)

        # add wavelength
        self.wavelengths.add(value=0.48613270)
        self.wavelengths.add(value=0.58756180, is_primary=True)
        self.wavelengths.add(value=0.65627250)


class TelescopeDoublet(optic.Optic):
    """A telescope doublet lens.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 44.
    """

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(
            index=1,
            radius=29.32908,
            thickness=0.7,
            material="N-BK7",
            is_stop=True,
        )
        self.surfaces.add(index=2, radius=-20.06842, thickness=0.032)
        self.surfaces.add(
            index=3,
            radius=-20.08770,
            thickness=0.5780,
            material=("SF2", "schott"),
        )
        self.surfaces.add(index=4, radius=-66.54774, thickness=47.3562)
        self.surfaces.add(index=5)

        self.set_aperture(aperture_type="imageFNO", value=8.0)

        self.fields.set_type(field_type="angle")
        self.fields.add(y=0.0)
        self.fields.add(y=0.7)
        self.fields.add(y=1.0)

        self.wavelengths.add(value=0.4861)
        self.wavelengths.add(value=0.5876, is_primary=True)
        self.wavelengths.add(value=0.6563)

        self.updater.image_solve()


class CementedAchromat(optic.Optic):
    """A cemented achromatic doublet lens.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 45.
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(
            index=1,
            radius=12.38401,
            thickness=0.4340,
            is_stop=True,
            material="N-BAK1",
        )
        self.surfaces.add(
            index=2,
            radius=-7.94140,
            thickness=0.3210,
            material=("SF2", "schott"),
        )
        self.surfaces.add(index=3, radius=-48.44396, thickness=19.6059)
        self.surfaces.add(index=4)

        # add aperture
        self.set_aperture(aperture_type="imageFNO", value=6)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=3.5)
        self.fields.add(y=5)

        # add wavelength
        self.wavelengths.add(value=0.48613270)
        self.wavelengths.add(value=0.58756180, is_primary=True)
        self.wavelengths.add(value=0.65627250)

        self.updater.image_solve()


class AsphericSinglet(optic.Optic):
    """An aspheric singlet lens."""

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(
            index=1,
            thickness=7,
            radius=20.0,
            is_stop=True,
            material="N-SF11",
            surface_type="even_asphere",
            conic=0.0,
            coefficients=[-2.248851e-4, -4.690412e-6, -6.404376e-8],
        )
        self.surfaces.add(index=2, thickness=21.56201105)
        self.surfaces.add(index=3)

        # add aperture
        self.set_aperture(aperture_type="EPD", value=20.0)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)

        # add wavelength
        self.wavelengths.add(value=0.587, is_primary=True)
