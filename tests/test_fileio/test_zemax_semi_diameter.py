"""Zemax semi-diameters: fixed ("U") semi-diameters clip, DIAM is a radius.

OpticStudio vignettes rays at a user-fixed semi-diameter (``DIAM <r> 1``)
even when the surface has no explicit aperture, and every DIAM value is a
semi-diameter. The prescriptions here are written for these tests; expected
ray heights follow from the geometry of each system.
"""

from __future__ import annotations

import math

import pytest
from numpy.testing import assert_allclose

import optiland.backend as be
from optiland.fileio import load_zemax_file, save_zemax_file
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture

_HEADER = """VERS 221026 46 20120530 20120530
MODE SEQ
NAME {name}
UNIT MM X W X CM MR CPMM
{aperture}
GCAT SCHOTT
FTYP 0 0 1 1 0 0 0 1
XFLN 0
YFLN 0
FWGN 1
WAVM 1 0.5875618 1
PWAV 1
"""


def _surf(index: int, *lines: str) -> str:
    body = "".join(f"  {line}\n" for line in lines)
    return f"SURF {index}\n{body}"


def _object() -> str:
    return _surf(0, "TYPE STANDARD", 'CURV 0.0 0 0 0 0 ""', "DISZ INFINITY")


def _write(tmp_path, name: str, aperture: str, *surfaces: str) -> str:
    path = tmp_path / f"{name}.zmx"
    path.write_text(
        _HEADER.format(name=name, aperture=aperture) + "".join(surfaces),
        encoding="utf-8",
    )
    return str(path)


def _passes(optic, px: float = 0.0, py: float = 0.0) -> bool:
    """True if an on-axis ray at normalized pupil (px, py) reaches the image."""
    rays = optic.trace_generic(0.0, 0.0, px, py, optic.primary_wavelength)
    return bool(be.to_numpy(rays.i).ravel()[0] > 0)


def _singlet(tmp_path, front_diam: str, *front_extra: str, aperture="ENPD 20"):
    """Singlet behind a stop; the collimated beam has radius 10 at the lens."""
    return _write(
        tmp_path,
        "singlet",
        aperture,
        _object(),
        _surf(
            1,
            "STOP",
            "TYPE STANDARD",
            'CURV 0.0 0 0 0 0 ""',
            "DISZ 10",
            'DIAM 10 0 0 0 1 ""',
        ),
        _surf(
            2,
            "TYPE STANDARD",
            'CURV 0.01 0 0 0 0 ""',
            "DISZ 5",
            "GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0",
            front_diam,
            *front_extra,
        ),
        _surf(
            3, "TYPE STANDARD", 'CURV 0.0 0 0 0 0 ""', "DISZ 100", 'DIAM 12 0 0 0 1 ""'
        ),
        _surf(4, "TYPE STANDARD", 'CURV 0.0 0 0 0 0 ""', "DISZ 0"),
    )


class TestRefractive:
    def test_fixed_semi_diameter_becomes_aperture(self, tmp_path, set_test_backend):
        optic = load_zemax_file(_singlet(tmp_path, 'DIAM 5 1 0 0 1 ""'))
        aperture = optic.surfaces[2].aperture
        assert type(aperture) is RadialAperture
        assert_allclose(aperture.r_max, 5.0)
        assert_allclose(aperture.r_min, 0.0)

    def test_fixed_semi_diameter_vignettes(self, tmp_path, set_test_backend):
        # Collimated in air up to surface 2, so a ray at pupil height py * 10
        # meets the lens at that height.
        optic = load_zemax_file(_singlet(tmp_path, 'DIAM 5 1 0 0 1 ""'))
        assert _passes(optic, py=0.499)
        assert not _passes(optic, py=0.501)
        assert _passes(optic, px=0.499)
        assert not _passes(optic, px=-0.501)

    def test_automatic_semi_diameter_does_not_clip(self, tmp_path, set_test_backend):
        optic = load_zemax_file(_singlet(tmp_path, 'DIAM 5 0 0 0 1 ""'))
        assert optic.surfaces[2].aperture is None
        assert _passes(optic, py=1.0)

    def test_explicit_aperture_takes_precedence(self, tmp_path, set_test_backend):
        optic = load_zemax_file(_singlet(tmp_path, 'DIAM 5 1 0 0 1 ""', "CLAP 0 7 0"))
        assert_allclose(optic.surfaces[2].aperture.r_max, 7.0)
        assert _passes(optic, py=0.69)
        assert not _passes(optic, py=0.71)

    def test_floating_aperture_takes_precedence(self, tmp_path, set_test_backend):
        optic = load_zemax_file(_singlet(tmp_path, 'DIAM 6 1 0 0 1 ""', "FLAP 0 6 0"))
        assert_allclose(optic.surfaces[2].aperture.r_max, 6.0)

    @pytest.mark.parametrize(
        "diam", ['DIAM 0 1 0 0 1 ""', "DIAM 5", 'DIAM 5 0 0 0 1 ""']
    )
    def test_no_aperture_without_a_positive_fixed_value(
        self, tmp_path, diam, set_test_backend
    ):
        optic = load_zemax_file(_singlet(tmp_path, diam))
        assert optic.surfaces[2].aperture is None

    @pytest.mark.parametrize(
        "aperture_line",
        ["SPID 1 3 0", "USAP 0 0 0"],
    )
    def test_unsupported_aperture_replaces_fixed_clip(
        self, tmp_path, aperture_line, set_test_backend
    ):
        # In OpticStudio any surface aperture replaces the semi-diameter clip,
        # so a type the reader cannot build yet must not leave a U clip behind.
        optic = load_zemax_file(_singlet(tmp_path, 'DIAM 5 1 0 0 1 ""', aperture_line))
        assert optic.surfaces[2].aperture is None


def _round_trip(optic, tmp_path):
    """Write ``optic`` after setting its semi-apertures, then read it back.

    ``update_paraxial`` fills ``semi_aperture`` so the writer emits DIAM for
    every surface, the export path a round trip must get right.
    """
    optic.update_paraxial()
    out = tmp_path / "round_trip.zmx"
    save_zemax_file(optic, str(out))
    return load_zemax_file(str(out))


class TestRoundTrip:
    def test_serialized_object_distance_survives_export(
        self, tmp_path, set_test_backend
    ):
        optic = load_zemax_file(_singlet(tmp_path, 'DIAM 5 1 0 0 1 ""'))
        restored = Optic.from_dict(optic.to_dict())
        reloaded = _round_trip(restored, tmp_path)
        assert reloaded.surfaces[0].is_infinite
        assert _passes(reloaded, py=0.499)
        assert not _passes(reloaded, py=0.501)

    def test_fixed_clip_survives(self, tmp_path, set_test_backend):
        # The clip is carried by the surface aperture (written as CLAP), not
        # by the DIAM value, so it reloads with the same radius and vignetting.
        optic = load_zemax_file(_singlet(tmp_path, 'DIAM 5 1 0 0 1 ""'))
        reloaded = _round_trip(optic, tmp_path)
        aperture = reloaded.surfaces[2].aperture
        assert type(aperture) is RadialAperture
        assert_allclose(aperture.r_max, 5.0)
        assert_allclose(aperture.r_min, 0.0)
        assert _passes(reloaded, py=0.499)
        assert not _passes(reloaded, py=0.501)

    def test_automatic_semi_diameter_stays_unclipped(self, tmp_path, set_test_backend):
        # optiland's semi_aperture is recomputed from paraxial rays, so the
        # writer must not export it as a fixed semi-diameter; that would add
        # a clip on reload.
        optic = load_zemax_file(_singlet(tmp_path, 'DIAM 5 0 0 0 1 ""'))
        reloaded = _round_trip(optic, tmp_path)
        assert reloaded.surfaces[2].aperture is None
        assert _passes(reloaded, py=1.0)


class TestFloatingStop:
    def test_stop_semi_diameter_is_doubled(self, tmp_path, set_test_backend):
        # The stop is the first surface, in air, so EPD = stop diameter.
        optic = load_zemax_file(
            _singlet(tmp_path, 'DIAM 5 0 0 0 1 ""', aperture="FLOA")
        )
        assert optic.aperture.ap_type == "float_by_stop_size"
        assert_allclose(optic.aperture.value, 20.0)
        assert_allclose(be.to_numpy(optic.paraxial.EPD()), 20.0)

    def test_round_trip_keeps_stop_size(self, tmp_path, set_test_backend):
        optic = load_zemax_file(
            _singlet(tmp_path, 'DIAM 5 0 0 0 1 ""', aperture="FLOA")
        )
        out = tmp_path / "round_trip.zmx"
        save_zemax_file(optic, str(out))
        reloaded = load_zemax_file(str(out))
        assert_allclose(reloaded.aperture.value, 20.0)
        stop_diam = [
            line.split()[1]
            for line in out.read_text(encoding="utf-8").split("SURF 1")[1].splitlines()
            if line.strip().startswith("DIAM")
        ]
        assert_allclose(float(stop_diam[0]), 10.0)


def _two_mirror(
    tmp_path,
    primary_diam: str,
    secondary_diam: str,
    *primary_extra: str,
    aperture="ENPD 60",
):
    """Paraboloid primary (f = 100) with a flat secondary 70 mm in front of it.

    Light reflected from the primary at height h heads for the focus at
    z = -100, so it meets the secondary at z = -70 at height
    30 h / (100 - h**2 / 400).
    """
    return _write(
        tmp_path,
        "two_mirror",
        aperture,
        _object(),
        _surf(1, "TYPE STANDARD", 'CURV 0.0 0 0 0 0 ""', "DISZ 100"),
        _surf(
            2,
            "STOP",
            "TYPE STANDARD",
            'CURV -0.005 0 0 0 0 ""',
            "CONI -1",
            "DISZ -70",
            "GLAS MIRROR 0 0 1.5 40 0 0 0 0 0 0",
            primary_diam,
            *primary_extra,
        ),
        _surf(
            3,
            "TYPE STANDARD",
            'CURV 0.0 0 0 0 0 ""',
            "DISZ 50",
            "GLAS MIRROR 0 0 1.5 40 0 0 0 0 0 0",
            secondary_diam,
        ),
        _surf(4, "TYPE STANDARD", 'CURV 0.0 0 0 0 0 ""', "DISZ 0"),
    )


def _secondary_height(h: float) -> float:
    return 30.0 * h / (100.0 - h**2 / 400.0)


class TestReflectors:
    def test_primary_mirror_fixed_semi_diameter(self, tmp_path, set_test_backend):
        optic = load_zemax_file(
            _two_mirror(tmp_path, 'DIAM 25 1 0 0 1 ""', 'DIAM 20 0 0 0 1 ""')
        )
        assert_allclose(optic.surfaces[2].aperture.r_max, 25.0)
        assert optic.surfaces[3].aperture is None
        # Beam radius 30; the primary passes heights up to 25.
        assert _passes(optic, py=24.9 / 30)
        assert not _passes(optic, py=25.1 / 30)

    def test_secondary_mirror_fixed_semi_diameter(self, tmp_path, set_test_backend):
        r_secondary = 6.0
        optic = load_zemax_file(
            _two_mirror(
                tmp_path, 'DIAM 40 0 0 0 1 ""', f'DIAM {r_secondary} 1 0 0 1 ""'
            )
        )
        assert optic.surfaces[2].aperture is None
        assert_allclose(optic.surfaces[3].aperture.r_max, r_secondary)
        h_inside, h_outside = 19.5, 20.0
        assert _secondary_height(h_inside) < r_secondary < _secondary_height(h_outside)
        assert _passes(optic, py=h_inside / 30)
        assert not _passes(optic, py=h_outside / 30)

    def test_obscuration_replaces_fixed_clip_on_mirror(
        self, tmp_path, set_test_backend
    ):
        # Checked in OpticStudio: with a central obscuration on the primary,
        # rays beyond the U semi-diameter of 25 still reach the image.
        optic = load_zemax_file(
            _two_mirror(
                tmp_path, 'DIAM 25 1 0 0 1 ""', 'DIAM 20 0 0 0 1 ""', "OBSC 0 5 0"
            )
        )
        # The surface carries the obscuration (see test_zemax_obscuration.py),
        # not a U clip.
        aperture = optic.surfaces[2].aperture
        assert type(aperture) is RadialAperture
        assert_allclose(aperture.r_min, 5.0)
        assert math.isinf(float(aperture.r_max))
        assert _passes(optic, py=26.0 / 30)

    def test_floating_stop_on_mirror(self, tmp_path, set_test_backend):
        # The stop is the primary, with nothing powered in front of it.
        optic = load_zemax_file(
            _two_mirror(
                tmp_path, 'DIAM 25 0 0 0 1 ""', 'DIAM 20 0 0 0 1 ""', aperture="FLOA"
            )
        )
        assert_allclose(optic.aperture.value, 50.0)
        assert_allclose(be.to_numpy(optic.paraxial.EPD()), 50.0)

    def test_fixed_semi_diameter_on_tilted_fold_mirror(
        self, tmp_path, set_test_backend
    ):
        # A 45 degree fold mirror with a 12 mm fixed semi-diameter in a beam of
        # radius 10. The aperture lies in the mirror's own tilted frame, so a
        # ray 9 mm off axis in the tilt plane lands 9 * sqrt(2) = 12.7 mm from
        # the mirror centre and is blocked, while the same ray in the
        # perpendicular plane lands at 9 mm and passes.
        path = _write(
            tmp_path,
            "fold",
            "ENPD 20",
            _object(),
            _surf(
                1,
                "STOP",
                "TYPE STANDARD",
                'CURV 0.0 0 0 0 0 ""',
                "DISZ 30",
                'DIAM 10 0 0 0 1 ""',
            ),
            _surf(
                2,
                "TYPE COORDBRK",
                "PARM 1 0",
                "PARM 2 0",
                "PARM 3 45",
                "PARM 4 0",
                "PARM 5 0",
                "PARM 6 0",
                "DISZ 0",
            ),
            _surf(
                3,
                "TYPE STANDARD",
                'CURV 0.0 0 0 0 0 ""',
                "DISZ 0",
                "GLAS MIRROR 0 0 1.5 40 0 0 0 0 0 0",
                'DIAM 12 1 0 0 1 ""',
            ),
            _surf(
                4,
                "TYPE COORDBRK",
                "PARM 1 0",
                "PARM 2 0",
                "PARM 3 45",
                "PARM 4 0",
                "PARM 5 0",
                "PARM 6 0",
                "DISZ -20",
            ),
            _surf(5, "TYPE STANDARD", 'CURV 0.0 0 0 0 0 ""', "DISZ 0"),
        )
        optic = load_zemax_file(path)
        mirror = next(s for s in optic.surfaces if s.aperture is not None)
        assert_allclose(mirror.aperture.r_max, 12.0)
        assert 9.0 * math.sqrt(2.0) > 12.0
        assert _passes(optic, px=0.9)
        assert not _passes(optic, py=0.9)
        assert _passes(optic, py=0.84)  # 8.4 * sqrt(2) = 11.9
