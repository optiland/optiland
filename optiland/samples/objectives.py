"""Defines various sample objective lens optical systems."""

from __future__ import annotations

import optiland.backend as be
from optiland import optic


class TripletTelescopeObjective(optic.Optic):
    """A triplet telescope objective lens design."""

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(
            index=1,
            radius=50.098,
            thickness=4.5,
            material="N-BK7",
            is_stop=True,
        )
        self.surfaces.add(index=2, radius=-983.42, thickness=0.1)
        self.surfaces.add(index=3, radius=56.671, thickness=4.5, material="N-BK7")
        self.surfaces.add(index=4, radius=-171.15, thickness=5.571)
        self.surfaces.add(
            index=5,
            radius=-97.339,
            thickness=3.5,
            material=("SF1", "schott"),
        )
        self.surfaces.add(index=6, radius=81.454, thickness=75.132)
        self.surfaces.add(index=7)

        self.set_aperture(aperture_type="imageFNO", value=2.8)

        self.fields.set_type(field_type="angle")
        self.fields.add(y=0.0)
        self.fields.add(y=0.7)
        self.fields.add(y=1.0)

        self.wavelengths.add(value=0.4861)
        self.wavelengths.add(value=0.5876, is_primary=True)
        self.wavelengths.add(value=0.6563)


class CookeTriplet(optic.Optic):
    """A Cooke triplet lens design."""

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=22.01359, thickness=3.25896, material="SK16")
        self.surfaces.add(index=2, radius=-435.76044, thickness=6.00755)
        self.surfaces.add(
            index=3, radius=-22.21328, thickness=0.99997, material=("F2", "schott")
        )
        self.surfaces.add(index=4, radius=20.29192, thickness=4.75041, is_stop=True)
        self.surfaces.add(index=5, radius=79.68360, thickness=2.95208, material="SK16")
        self.surfaces.add(index=6, radius=-18.39533, thickness=42.20778)
        self.surfaces.add(index=7)

        self.set_aperture(aperture_type="EPD", value=10)

        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=14)
        self.fields.add(y=20)

        self.wavelengths.add(value=0.48)
        self.wavelengths.add(value=0.55, is_primary=True)
        self.wavelengths.add(value=0.65)


class DoubleGauss(optic.Optic):
    """A Double Gauss lens design."""

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=56.20238, thickness=8.75, material="N-SSK2")
        self.surfaces.add(index=2, radius=152.28580, thickness=0.5)
        self.surfaces.add(index=3, radius=37.68262, thickness=12.5, material="N-SK2")
        self.surfaces.add(
            index=4,
            radius=be.inf,
            thickness=3.8,
            material=("F5", "schott"),
        )
        self.surfaces.add(index=5, radius=24.23130, thickness=16.369445)
        self.surfaces.add(index=6, radius=be.inf, thickness=13.747957, is_stop=True)
        self.surfaces.add(
            index=7,
            radius=-28.37731,
            thickness=3.8,
            material=("F5", "schott"),
        )
        self.surfaces.add(index=8, radius=be.inf, thickness=11, material="N-SK16")
        self.surfaces.add(index=9, radius=-37.92546, thickness=0.5)
        self.surfaces.add(index=10, radius=177.41176, thickness=7, material="N-SK16")
        self.surfaces.add(index=11, radius=-79.41143, thickness=61.487536)
        self.surfaces.add(index=12)

        self.set_aperture(aperture_type="imageFNO", value=5)
        self.fields.set_type(field_type="angle")

        self.fields.add(y=0)
        self.fields.add(y=10)
        self.fields.add(y=14)

        self.wavelengths.add(value=0.4861)
        self.wavelengths.add(value=0.5876, is_primary=True)
        self.wavelengths.add(value=0.6563)


class ReverseTelephoto(optic.Optic):
    """A reverse telephoto lens design."""

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(
            index=1,
            radius=1.69111096,
            thickness=0.08259680,
            material="N-SK10",
        )
        self.surfaces.add(index=2, radius=0.94414496, thickness=0.8)
        self.surfaces.add(
            index=3,
            radius=4.32100401,
            thickness=0.080256,
            material="SK15",
        )
        self.surfaces.add(index=4, radius=1.78117621, thickness=0.5)
        self.surfaces.add(
            index=5,
            radius=2.64050282,
            thickness=0.27638160,
            material="BASF2",
        )
        self.surfaces.add(index=6, radius=-3.86177348, thickness=0.1)
        self.surfaces.add(index=7, radius=1.05627661, thickness=0.2, material="FK3")
        self.surfaces.add(index=8, radius=-4.06933311, thickness=0.2001384)
        self.surfaces.add(index=9, radius=be.inf, thickness=0.06688, is_stop=True)
        self.surfaces.add(
            index=10,
            radius=-2.61246583,
            thickness=0.064372,
            material=("SF15", "hikari"),
        )
        self.surfaces.add(index=11, radius=0.99117409, thickness=0.3)
        self.surfaces.add(
            index=12,
            radius=9.03045960,
            thickness=0.18743120,
            material="N-LAK12",
        )
        self.surfaces.add(index=13, radius=-1.35680743, thickness=2.35130547)
        self.surfaces.add(index=14)

        self.set_aperture(aperture_type="EPD", value=0.3)

        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=21)
        self.fields.add(y=30)

        self.wavelengths.add(value=0.4861)
        self.wavelengths.add(value=0.5876, is_primary=True)
        self.wavelengths.add(value=0.6563)


class ObjectiveUS008879901(optic.Optic):
    """An objective lens design based on U.S. Patent 8,879,901."""

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(
            index=1,
            radius=47.07125235,
            thickness=5.29811826,
            material="N-LAF32",
        )
        self.surfaces.add(index=2, radius=184.28171667, thickness=0.6)
        self.surfaces.add(
            index=3,
            radius=29.92177645,
            thickness=7.13654863,
            material="H-ZLAF52A",
        )
        self.surfaces.add(index=4, radius=50.4992638, thickness=2.0)
        self.surfaces.add(
            index=5,
            radius=60.5004845,
            thickness=0.99941671,
            material="E-SF1",
        )
        self.surfaces.add(index=6, radius=17.72638376, thickness=9.9)
        self.surfaces.add(index=7, radius=be.inf, thickness=8.7, is_stop=True)
        self.surfaces.add(
            index=8,
            radius=-17.49862241,
            thickness=1.29934579,
            material=("SF4", "hikari"),
        )
        self.surfaces.add(
            index=9,
            radius=1000.00000019,
            thickness=8.44325264,
            material="M-TAF1",
        )
        self.surfaces.add(index=10, radius=-28.00122422, thickness=0.1)
        self.surfaces.add(
            index=11,
            radius=-141.99976777,
            thickness=6.79950254,
            material="M-TAF1",
        )
        self.surfaces.add(index=12, radius=-35.94103045, thickness=0.516)
        self.surfaces.add(
            index=13,
            radius=92.00034667,
            thickness=3.29901361,
            material="Q-LAFPH1S",
        )
        self.surfaces.add(index=14, radius=-277.85210888, thickness=2.13)
        self.surfaces.add(
            index=15,
            radius=-157.24588662,
            thickness=1.29980422,
            material="S-FSL5",
        )
        self.surfaces.add(index=16, radius=740.47397742, thickness=0.25)
        self.surfaces.add(
            index=17,
            radius=19.91929498,
            thickness=5.59345688,
            material="J-LASF015",
        )
        self.surfaces.add(index=18, radius=36.48852623, thickness=0.574)
        self.surfaces.add(
            index=19,
            radius=45.97532235,
            thickness=1.00045731,
            material="E-SF1",
        )
        self.surfaces.add(index=20, radius=16.39521847, thickness=2.951)
        self.surfaces.add(
            index=21,
            radius=33.86131631,
            thickness=3.22444231,
            material="H-LAK52",
        )
        self.surfaces.add(index=22, radius=be.inf, thickness=8.0)
        self.surfaces.add(index=23, radius=be.inf, thickness=4.0, material="H-LAK52")
        self.surfaces.add(index=24, radius=be.inf, thickness=3.15317838)
        self.surfaces.add(index=25)

        self.set_aperture(aperture_type="imageFNO", value=2.0)

        self.fields.set_type(field_type="angle")
        self.fields.add(0.0)
        self.fields.add(7.574)
        self.fields.add(10.82)

        self.wavelengths.add(value=0.4861327)
        self.wavelengths.add(value=0.5875618, is_primary=True)
        self.wavelengths.add(value=0.6562725)


class TelescopeObjective48Inch(optic.Optic):
    """48-in. Telescope Objective

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 48.
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=-12.7172, thickness=0.8, material="N-PSK3")
        self.surfaces.add(index=2, radius=-18.5430, thickness=0.0148)
        self.surfaces.add(index=3, thickness=0.0150, is_stop=True)
        self.surfaces.add(
            index=4,
            radius=15.758,
            thickness=1.6701,
            material=("CAF2", "Daimon-20"),
        )
        self.surfaces.add(index=5, radius=-13.0390, thickness=0.0487)
        self.surfaces.add(index=6, radius=-12.8310, thickness=0.8, material="S-LAL18")
        self.surfaces.add(index=7, radius=-18.5430, thickness=1.1799)
        self.surfaces.add(index=8, radius=9.8197, thickness=0.8, material="N-SK16")
        self.surfaces.add(index=9, radius=8.0010, thickness=44.3502)
        self.surfaces.add(index=10)

        # add aperture
        self.set_aperture(aperture_type="imageFNO", value=6)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=7)
        self.fields.add(y=10)

        # add wavelength
        self.wavelengths.add(value=0.48613270)
        self.wavelengths.add(value=0.58756180, is_primary=True)
        self.wavelengths.add(value=0.65627250)


class HeliarLens(optic.Optic):
    """Heliar Lens f/5

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 63.
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=4.2103, thickness=0.9004, material="N-SK16")
        self.surfaces.add(index=2, radius=-3.6208, thickness=0.2999, material="E-LLF6")
        self.surfaces.add(index=3, radius=29.1869, thickness=0.7587)
        self.surfaces.add(index=4, radius=-3.1715, thickness=0.2, material="E-LLF6")
        self.surfaces.add(index=5, radius=3.2083, thickness=0.1264)
        self.surfaces.add(index=6, radius=be.inf, thickness=0.2629, is_stop=True)
        self.surfaces.add(index=7, radius=43.0710, thickness=0.25, material="E-LLF6")
        self.surfaces.add(index=8, radius=2.4494, thickness=0.8308, material="N-SK16")
        self.surfaces.add(index=9, radius=-3.2576, thickness=8.5066)
        self.surfaces.add(index=10)

        # add aperture
        self.set_aperture(aperture_type="imageFNO", value=5)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=7)
        self.fields.add(y=10)

        # add wavelength
        self.wavelengths.add(value=0.48613270)
        self.wavelengths.add(value=0.58756180, is_primary=True)
        self.wavelengths.add(value=0.65627250)


class TessarLens(optic.Optic):
    """Tessar Lens f/4.5

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 63.
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=1.3329, thickness=0.2791, material="N-SK15")
        self.surfaces.add(index=2, radius=-9.9754, thickness=0.2054)
        self.surfaces.add(
            index=3,
            radius=-2.0917,
            thickness=0.09,
            material=("F2", "schott"),
        )
        self.surfaces.add(index=4, radius=1.2123, thickness=0.0709)
        self.surfaces.add(index=5, radius=be.inf, thickness=0.1534, is_stop=True)
        self.surfaces.add(index=6, radius=-7.5205, thickness=0.09, material="K10")
        self.surfaces.add(index=7, radius=1.3010, thickness=0.3389, material="N-SK15")
        self.surfaces.add(index=8, radius=-1.5218, thickness=3.4025)
        self.surfaces.add(index=9)

        # add aperture
        self.set_aperture(aperture_type="imageFNO", value=4.5)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=10)
        self.fields.add(y=20.5)

        # add wavelength
        self.wavelengths.add(value=0.48613270)
        self.wavelengths.add(value=0.58756180, is_primary=True)
        self.wavelengths.add(value=0.65627250)


class LensWithFieldCorrector(optic.Optic):
    """5-in. Focal Length, f/3.5 Lens with Field Corrector.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 66.
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=1.9863, thickness=0.5, material="N-SK16")
        self.surfaces.add(index=2, radius=6.2901, thickness=0.4878)
        self.surfaces.add(index=3, radius=be.inf, thickness=0.1016, is_stop=True)
        self.surfaces.add(
            index=4,
            radius=-2.5971,
            thickness=0.1843,
            material=("F5", "schott"),
        )
        self.surfaces.add(index=5, radius=2.4073, thickness=0.0719)
        self.surfaces.add(index=6, radius=5.8147, thickness=0.3153, material="N-SK16")
        self.surfaces.add(index=7, radius=-2.1926, thickness=2.6845)
        self.surfaces.add(index=8, radius=1.9071, thickness=0.502, material="N-SK16")
        self.surfaces.add(index=9, radius=2.3148, thickness=0.015)
        self.surfaces.add(index=10, radius=1.1907, thickness=0.2, material="N-SK4")
        self.surfaces.add(index=11, radius=0.9911, thickness=1.159)
        self.surfaces.add(index=12)

        # add aperture
        self.set_aperture(aperture_type="imageFNO", value=3.5)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=5)
        self.fields.add(y=9.65)

        # add wavelength
        self.wavelengths.add(value=0.48613270)
        self.wavelengths.add(value=0.58756180, is_primary=True)
        self.wavelengths.add(value=0.65627250)

        # scale from inches to mm
        self.updater.scale_system(25.4)


class PetzvalLens(optic.Optic):
    """Petzval lens f/1.4.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 75.
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=1.3265, thickness=0.4, material="N-LAK12")
        self.surfaces.add(index=2, radius=-2.6919, thickness=0.06)
        self.surfaces.add(
            index=3,
            radius=-2.0028,
            thickness=0.16,
            material=("SF4", "schott"),
        )
        self.surfaces.add(index=4, radius=5.4499, thickness=0.1)
        self.surfaces.add(index=5, radius=be.inf, thickness=0.8999, is_stop=True)
        self.surfaces.add(index=6, radius=1.1724, thickness=0.3, material="N-LAK12")
        self.surfaces.add(index=7, radius=-2.4602, thickness=0.2221)
        self.surfaces.add(
            index=8,
            radius=-0.8615,
            thickness=0.08,
            material=("LF5", "schott"),
        )
        self.surfaces.add(index=9, radius=3.0039, thickness=0.3921)
        self.surfaces.add(index=10)

        # add aperture
        self.set_aperture(aperture_type="imageFNO", value=1.4)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=3.5)
        self.fields.add(y=7)

        # add wavelength
        self.wavelengths.add(value=0.48613270)
        self.wavelengths.add(value=0.58756180, is_primary=True)
        self.wavelengths.add(value=0.65627250)

        # scale from inches to mm
        self.updater.scale_system(25.4)


class Telephoto(optic.Optic):
    """Telephoto lens f/5.6.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 91.
    """

    def __init__(self):
        super().__init__()

        # add surfaces
        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=0.8589, thickness=0.2391, material="N-BK7")
        self.surfaces.add(index=2, radius=-2.6902, thickness=0.09, material="N-BASF2")
        self.surfaces.add(index=3, radius=3.0318, thickness=0.0481)
        self.surfaces.add(index=4, radius=be.inf, thickness=1.0347, is_stop=True)
        self.surfaces.add(index=5, radius=-0.5715, thickness=0.09, material="N-ZK7")
        self.surfaces.add(index=6, radius=-0.7423, thickness=0.1005, material="N-LAF33")
        self.surfaces.add(index=7, radius=-1.1433, thickness=0.0156)
        self.surfaces.add(
            index=8,
            radius=-17.0388,
            thickness=0.0793,
            material=("SF1", "schott"),
        )
        self.surfaces.add(index=9, radius=-2.7695, thickness=2.4796)
        self.surfaces.add(index=10)

        # add aperture
        self.set_aperture(aperture_type="imageFNO", value=5.6)

        # add field
        self.fields.set_type(field_type="angle")
        self.fields.add(y=0)
        self.fields.add(y=7)
        self.fields.add(y=10)

        # add wavelength
        self.wavelengths.add(value=0.48613270)
        self.wavelengths.add(value=0.58756180, is_primary=True)
        self.wavelengths.add(value=0.65627250)

        # scale from inches to mm
        self.updater.scale_system(25.4)


class WideAngle100FOV(optic.Optic):
    """A wide-angle lens design with a 100° field of view.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 108.
    """

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=8.0107, thickness=0.2500, material="N-SK4")
        self.surfaces.add(index=2, radius=1.1856, thickness=0.9613)
        self.surfaces.add(index=3, radius=1.6747, thickness=0.3578, material="SF1")
        self.surfaces.add(index=4, radius=-7.5157, thickness=0.2136, material="N-SK4")
        self.surfaces.add(index=5, radius=0.9411, thickness=0.4146)
        self.surfaces.add(index=6, radius=-1.7688, thickness=0.3333, material="SF1")
        self.surfaces.add(index=7, radius=-1.5531, thickness=0.3863, material="N-SK4")
        self.surfaces.add(index=8, radius=-2.2281, thickness=0.9842)
        self.surfaces.add(index=9, radius=be.inf, thickness=0.1000, is_stop=True)
        self.surfaces.add(index=10, radius=13.6803, thickness=0.1400, material="N-SK4")
        self.surfaces.add(index=11, radius=-3.4279, thickness=0.0605)
        self.surfaces.add(index=12, radius=20.0257, thickness=0.3332, material="N-SK4")
        self.surfaces.add(index=13, radius=-0.9258, thickness=0.4374, material="SF1")
        self.surfaces.add(index=14, radius=-3.2233, thickness=0.2679)
        self.surfaces.add(index=15, radius=10.3847, thickness=0.1400, material="SF1")
        self.surfaces.add(index=16, radius=2.4272, thickness=0.2609, material="N-SK4")
        self.surfaces.add(index=17, radius=-3.8828, thickness=0.0150)
        self.surfaces.add(index=18, radius=3.3650, thickness=0.3167, material="N-SK4")
        self.surfaces.add(index=19, radius=-14.7547, thickness=1.5604)
        self.surfaces.add(index=20, radius=be.inf, thickness=0.0)

        self.set_aperture("imageFNO", 4.0)

        self.fields.set_type("angle")
        self.fields.add(y=0)
        self.fields.add(y=35.0)
        self.fields.add(y=50.0)

        self.wavelengths.add(value=0.4861)
        self.wavelengths.add(value=0.5876, is_primary=True)
        self.wavelengths.add(value=0.6563)

        self.ray_tracer.set_aiming("iterative", cache=True)


class ProjectionLens120FOV(optic.Optic):
    """A 120-degree FOV projection lens design.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 110.
    """

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=28.2482, thickness=0.5000, material="N-SK4")
        self.surfaces.add(index=2, radius=2.2027, thickness=1.4379)
        self.surfaces.add(index=3, radius=2.7819, thickness=1.1174, material="SF1")
        self.surfaces.add(index=4, radius=-10.4886, thickness=0.3022, material="N-SK4")
        self.surfaces.add(index=5, radius=1.2417, thickness=0.7247)
        self.surfaces.add(index=6, radius=-2.8768, thickness=0.5088, material="SF1")
        self.surfaces.add(index=7, radius=-2.0989, thickness=0.4998, material="N-SK4")
        self.surfaces.add(index=8, radius=-4.2448, thickness=0.0150)
        self.surfaces.add(index=9, radius=8.4162, thickness=1.1279, material="N-SK4")
        self.surfaces.add(index=10, radius=-1.6294, thickness=0.2486, material="SF1")
        self.surfaces.add(index=11, radius=-4.5056, thickness=0.5412)
        self.surfaces.add(index=12, radius=be.inf, thickness=0.0200, is_stop=True)
        self.surfaces.add(index=13, radius=be.inf, thickness=0.1689, material="N-SK4")
        self.surfaces.add(index=14, radius=-3.8691, thickness=0.1065)
        self.surfaces.add(index=15, radius=-6.6827, thickness=0.2007, material="N-SK4")
        self.surfaces.add(index=16, radius=-2.2167, thickness=0.2822)
        self.surfaces.add(index=17, radius=-1.6444, thickness=0.1827, material="SF1")
        self.surfaces.add(index=18, radius=-3.3466, thickness=0.0149)
        self.surfaces.add(index=19, radius=-36.4599, thickness=0.5235, material="SF1")
        self.surfaces.add(index=20, radius=5.3765, thickness=0.0373)
        self.surfaces.add(index=21, radius=6.4204, thickness=0.6243, material="N-SK4")
        self.surfaces.add(index=22, radius=-2.7537, thickness=0.0150)
        self.surfaces.add(index=23, radius=5.1876, thickness=0.8475, material="N-SK4")
        self.surfaces.add(index=24, radius=be.inf, thickness=2.0000)
        self.surfaces.add(index=25, radius=be.inf, thickness=0.0)

        self.set_aperture("imageFNO", 2.8)

        self.fields.set_type("angle")
        self.fields.add(y=0)
        self.fields.add(y=42.0)
        self.fields.add(y=60.0)

        self.wavelengths.add(value=0.4861)
        self.wavelengths.add(value=0.5876, is_primary=True)
        self.wavelengths.add(value=0.6563)

        self.ray_tracer.set_aiming("robust", cache=True)


class ProjectionLens160FOV(optic.Optic):
    """An f/2 160-degree FOV projection lens.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 112.
    """

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=38.9150, thickness=0.5459, material="N-BK7")
        self.surfaces.add(index=2, radius=3.6152, thickness=1.6595)
        self.surfaces.add(index=3, radius=be.inf, thickness=0.6927, material="N-BK7")
        self.surfaces.add(index=4, radius=5.2515, thickness=1.4576)
        self.surfaces.add(index=5, radius=-5.2515, thickness=0.6927, material="N-BK7")
        self.surfaces.add(index=6, radius=be.inf, thickness=4.1494)
        self.surfaces.add(index=7, radius=5.1075, thickness=0.7000, material="SF4")
        self.surfaces.add(index=8, radius=-10.8385, thickness=0.0162)
        self.surfaces.add(index=9, radius=2.3897, thickness=0.7443, material="SF4")
        self.surfaces.add(index=10, radius=2.2789, thickness=0.1136)
        self.surfaces.add(index=11, radius=be.inf, thickness=0.1040, is_stop=True)
        self.surfaces.add(index=12, radius=-2.3536, thickness=0.7038, material="SF1")
        self.surfaces.add(index=13, radius=2.7829, thickness=0.5423, material="N-LAK7")
        self.surfaces.add(index=14, radius=-2.8287, thickness=0.2506)
        self.surfaces.add(index=15, radius=13.1578, thickness=0.6350, material="N-LAK7")
        self.surfaces.add(index=16, radius=-1.4944, thickness=0.2922, material="SF4")
        self.surfaces.add(index=17, radius=-4.8082, thickness=0.0668)
        self.surfaces.add(index=18, radius=4.3447, thickness=0.5059, material="N-LAK21")
        self.surfaces.add(index=19, radius=72.8673, thickness=2.2376)
        self.surfaces.add(index=20, radius=be.inf, thickness=0.0)

        self.set_aperture("imageFNO", 2.0)

        self.fields.set_type("angle")
        self.fields.add(y=0)
        self.fields.add(y=56.0)
        self.fields.add(y=80.0)

        self.wavelengths.add(value=0.4861)
        self.wavelengths.add(value=0.5876, is_primary=True)
        self.wavelengths.add(value=0.6563)

        self.ray_tracer.set_aiming("robust", cache=True)


class WideAngle170FOV(optic.Optic):
    """A 170-degree FOV, f/1.8 camera lens.

    Reference: Milton Laikin, Lens Design, 4th ed., CRC Press, 2007, p. 114.
    """

    def __init__(self):
        super().__init__()

        self.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        self.surfaces.add(index=1, radius=6.0515, thickness=0.1796, material="N-BK7")
        self.surfaces.add(index=2, radius=1.2087, thickness=0.9182)
        self.surfaces.add(index=3, radius=-9.4697, thickness=0.1939, material="N-SK5")
        self.surfaces.add(index=4, radius=1.0053, thickness=0.6402)
        self.surfaces.add(index=5, radius=1.6634, thickness=0.4518, material="SF1")
        self.surfaces.add(index=6, radius=-1.6712, thickness=0.1825, material="N-LAK9")
        self.surfaces.add(index=7, radius=1.1206, thickness=2.5272)
        self.surfaces.add(index=8, radius=1.3895, thickness=0.2500, material="N-LAK9")
        self.surfaces.add(index=9, radius=0.5522, thickness=0.6047, material="N-PSK3")
        self.surfaces.add(index=10, radius=-0.8844, thickness=0.0196)
        self.surfaces.add(index=11, radius=-0.8804, thickness=0.0800, material="SF1")
        self.surfaces.add(index=12, radius=-2.1545, thickness=0.1200)
        self.surfaces.add(index=13, radius=be.inf, thickness=0.1200, is_stop=True)
        self.surfaces.add(index=14, radius=1.0354, thickness=0.4830, material="N-PSK3")
        self.surfaces.add(index=15, radius=-0.8443, thickness=0.0546)
        self.surfaces.add(index=16, radius=-0.7386, thickness=0.1076, material="SF1")
        self.surfaces.add(index=17, radius=-1.5285, thickness=0.8308)
        self.surfaces.add(index=18, radius=be.inf, thickness=0.0)

        self.set_aperture("imageFNO", 1.8)

        self.fields.set_type("angle")
        self.fields.add(y=0)
        self.fields.add(y=60.0)
        self.fields.add(y=85.0)

        self.wavelengths.add(value=0.4861)
        self.wavelengths.add(value=0.5876, is_primary=True)
        self.wavelengths.add(value=0.6563)

        self.ray_tracer.set_aiming("robust", cache=True)
