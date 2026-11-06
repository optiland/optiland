"""Zemax rectangular and elliptical apertures and obscurations.

OpticStudio writes the four types as ``SQAP``, ``SQOB``, ``ELAP`` and ``ELOB``
followed by the X and Y half-widths and a pickup surface (0 for none); an
``OBDC`` record decentres them. The apertures pass rays inside the shape, the
obscurations block them and pass everything else with no outer limit, and any
of them replaces a fixed ("U") semi-diameter clip. The prescription is written
for these tests: a collimated beam of radius 30 falls on a concave mirror, so a
ray at normalized pupil coordinates (px, py) meets the mirror at 30 (px, py).
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.fileio import load_zemax_file, save_zemax_file
from optiland.optic import Optic
from optiland.physical_apertures import (
    DifferenceAperture,
    EllipticalAperture,
    RadialAperture,
    RectangularAperture,
)

_MIRROR = """VERS 221026 46 20120530 20120530
MODE SEQ
NAME Aperture shape test
UNIT MM X W X CM MR CPMM
ENPD 60
GCAT SCHOTT
FTYP 0 0 1 1 0 0 0 1
XFLN 0
YFLN 0
WAVM 1 0.5875618 1
PWAV 1
SURF 0
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ INFINITY
SURF 1
  STOP
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ 50
  DIAM 30 0 0 0 1 ""
SURF 2
  TYPE STANDARD
  CURV -0.005 0 0 0 0 ""
  DISZ -100
  GLAS MIRROR 0 0 1.5 40 0 0 0 0 0 0
  DIAM 25 1 0 0 1 ""
{aperture}
SURF 3
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ 0
"""

_BEAM = 30.0

# Points on the mirror for a 6 x 3 half-width shape decentred by (1, 1): the
# rectangle spans x in [-5, 7] and y in [-2, 4].
_INSIDE_BOTH = [(0.0, 0.0), (1.0, 3.8)]
_RECTANGLE_ONLY = [(6.5, 3.5), (6.0, -1.5)]
_OUTSIDE_BOTH = [(-5.5, 0.0), (0.0, 4.5)]
_BEYOND_CLIP = (0.0, 28.0)  # outside the U semi-diameter of 25


def _load(tmp_path, *aperture_lines: str):
    path = tmp_path / "shape.zmx"
    lines = "\n".join(f"  {line}" for line in aperture_lines)
    path.write_text(_MIRROR.format(aperture=lines), encoding="utf-8")
    return load_zemax_file(str(path))


def _passes(optic, x: float = 0.0, y: float = 0.0) -> bool:
    """True if the on-axis ray meeting the mirror at (x, y) reaches the image."""
    rays = optic.trace_generic(0.0, 0.0, x / _BEAM, y / _BEAM, optic.primary_wavelength)
    return bool(be.to_numpy(rays.i).ravel()[0] > 0)


def _round_trip(optic, tmp_path):
    out = tmp_path / "round_trip.zmx"
    save_zemax_file(optic, str(out))
    return load_zemax_file(str(out)), out.read_text(encoding="utf-8")


def _records(text: str) -> dict[str, list[float]]:
    records = {}
    for line in text.splitlines():
        tokens = line.split()
        if tokens and tokens[0] in ("SQAP", "SQOB", "ELAP", "ELOB", "OBDC", "CLAP"):
            records[tokens[0]] = [float(t) for t in tokens[1:3]]
    return records


class TestRectangularAperture:
    def test_aperture_type(self, tmp_path, set_test_backend):
        aperture = _load(tmp_path, "SQAP 6 3 0", "OBDC 1 1").surfaces[2].aperture
        assert type(aperture) is RectangularAperture
        assert tuple(float(v) for v in aperture.extent) == (-5.0, 7.0, -2.0, 4.0)

    def test_passes_only_inside(self, tmp_path, set_test_backend):
        optic = _load(tmp_path, "SQAP 6 3 0", "OBDC 1 1")
        for point in _INSIDE_BOTH + _RECTANGLE_ONLY:
            assert _passes(optic, *point)
        for point in _OUTSIDE_BOTH:
            assert not _passes(optic, *point)

    def test_centred_without_obdc(self, tmp_path, set_test_backend):
        aperture = _load(tmp_path, "SQAP 6 3 0").surfaces[2].aperture
        assert tuple(float(v) for v in aperture.extent) == (-6.0, 6.0, -3.0, 3.0)

    def test_replaces_fixed_clip(self, tmp_path, set_test_backend):
        assert _passes(_load(tmp_path, "SQAP 30 30 0"), *_BEYOND_CLIP)


class TestRectangularObscuration:
    def test_aperture_type(self, tmp_path, set_test_backend):
        aperture = _load(tmp_path, "SQOB 6 3 0", "OBDC 1 1").surfaces[2].aperture
        assert type(aperture) is DifferenceAperture
        assert type(aperture.a) is RadialAperture
        assert type(aperture.b) is RectangularAperture

    def test_blocks_only_inside(self, tmp_path, set_test_backend):
        optic = _load(tmp_path, "SQOB 6 3 0", "OBDC 1 1")
        for point in _INSIDE_BOTH + _RECTANGLE_ONLY:
            assert not _passes(optic, *point)
        for point in [*_OUTSIDE_BOTH, _BEYOND_CLIP]:
            assert _passes(optic, *point)


class TestEllipticalAperture:
    def test_aperture_type(self, tmp_path, set_test_backend):
        aperture = _load(tmp_path, "ELAP 6 3 0", "OBDC 1 1").surfaces[2].aperture
        assert type(aperture) is EllipticalAperture
        assert (aperture.a, aperture.b) == (6.0, 3.0)
        assert (aperture.offset_x, aperture.offset_y) == (1.0, 1.0)

    def test_passes_only_inside(self, tmp_path, set_test_backend):
        optic = _load(tmp_path, "ELAP 6 3 0", "OBDC 1 1")
        for point in _INSIDE_BOTH:
            assert _passes(optic, *point)
        for point in _RECTANGLE_ONLY + _OUTSIDE_BOTH:
            assert not _passes(optic, *point)

    def test_replaces_fixed_clip(self, tmp_path, set_test_backend):
        assert _passes(_load(tmp_path, "ELAP 30 30 0"), *_BEYOND_CLIP)


class TestEllipticalObscuration:
    def test_aperture_type(self, tmp_path, set_test_backend):
        aperture = _load(tmp_path, "ELOB 6 3 0", "OBDC 1 1").surfaces[2].aperture
        assert type(aperture) is DifferenceAperture
        assert type(aperture.b) is EllipticalAperture

    def test_blocks_only_inside(self, tmp_path, set_test_backend):
        optic = _load(tmp_path, "ELOB 6 3 0", "OBDC 1 1")
        for point in _INSIDE_BOTH:
            assert not _passes(optic, *point)
        for point in [*_RECTANGLE_ONLY, *_OUTSIDE_BOTH, _BEYOND_CLIP]:
            assert _passes(optic, *point)


@pytest.mark.parametrize("keyword", ["SQAP", "SQOB", "ELAP", "ELOB"])
def test_obdc_before_the_shape(tmp_path, keyword, set_test_backend):
    after = _load(tmp_path, f"{keyword} 6 3 0", "OBDC 1 1").surfaces[2].aperture
    before = _load(tmp_path, "OBDC 1 1", f"{keyword} 6 3 0").surfaces[2].aperture
    assert before.to_dict() == after.to_dict()


def test_pickup_keeps_the_written_half_widths(tmp_path, set_test_backend):
    # A nonzero last field picks the aperture up from that surface; the file
    # still carries the picked-up half-widths.
    aperture = _load(tmp_path, "SQAP 7 7 1").surfaces[2].aperture
    assert tuple(float(v) for v in aperture.extent) == (-7.0, 7.0, -7.0, 7.0)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "lines,written",
        [
            (("SQAP 6 3 0",), {"SQAP": [6, 3]}),
            (("SQAP 6 3 0", "OBDC 1 1"), {"SQAP": [6, 3], "OBDC": [1, 1]}),
            (("SQOB 6 3 0", "OBDC 1 1"), {"SQOB": [6, 3], "OBDC": [1, 1]}),
            (("ELAP 6 3 0",), {"ELAP": [6, 3]}),
            (("ELAP 6 3 0", "OBDC 1 1"), {"ELAP": [6, 3], "OBDC": [1, 1]}),
            (("ELOB 6 3 0", "OBDC -2 0.5"), {"ELOB": [6, 3], "OBDC": [-2, 0.5]}),
        ],
    )
    def test_written_back(self, tmp_path, lines, written, set_test_backend):
        optic = _load(tmp_path, *lines)
        reloaded, text = _round_trip(optic, tmp_path)
        assert _records(text) == written
        before = optic.surfaces[2].aperture.to_dict()
        assert reloaded.surfaces[2].aperture.to_dict() == before

    def test_reloaded_lens_traces_the_same(self, tmp_path, set_test_backend):
        reloaded, _ = _round_trip(_load(tmp_path, "SQOB 6 3 0", "OBDC 1 1"), tmp_path)
        assert not _passes(reloaded)
        assert _passes(reloaded, -5.5, 0.0)

    def test_asymmetric_rectangle_written_with_decentre(
        self, tmp_path, set_test_backend
    ):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=be.inf)
        optic.surfaces.add(
            index=1,
            thickness=10,
            is_stop=True,
            aperture=RectangularAperture(-2.0, 6.0, -1.0, 3.0),
        )
        optic.surfaces.add(index=2)
        optic.set_aperture(aperture_type="EPD", value=10)
        optic.fields.set_type(field_type="angle")
        optic.fields.add(y=0)
        optic.wavelengths.add(value=0.55, is_primary=True)
        reloaded, text = _round_trip(optic, tmp_path)
        assert _records(text) == {"SQAP": [4, 2], "OBDC": [2, 1]}
        aperture = reloaded.surfaces[1].aperture
        assert tuple(float(v) for v in aperture.extent) == (-2.0, 6.0, -1.0, 3.0)
