"""Zemax spider apertures.

OpticStudio writes a spider as ``SPID`` followed by the full width of each arm,
the number of arms and a pickup surface (0 for none); an ``OBDC`` record
decentres it. The arms are half-lines from the centre, equally spaced in angle,
the first along local +x. Rays on an arm are blocked and every other ray
passes, with no outer limit, so a spider also replaces a fixed ("U")
semi-diameter clip. The points below were checked against OpticStudio.

The prescription is written for these tests: a collimated beam of radius 30
falls on a concave mirror, so a ray at normalized pupil coordinates (px, py)
meets the mirror at 30 (px, py).
"""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.fileio import load_zemax_file, save_zemax_file
from optiland.physical_apertures import SpiderAperture

_MIRROR = """VERS 221026 46 20120530 20120530
MODE SEQ
NAME Spider test
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


def _load(tmp_path, *aperture_lines: str):
    path = tmp_path / "spider.zmx"
    lines = "\n".join(f"  {line}" for line in aperture_lines)
    path.write_text(_MIRROR.format(aperture=lines), encoding="utf-8")
    return load_zemax_file(str(path))


def _passes(optic, x: float = 0.0, y: float = 0.0) -> bool:
    """True if the on-axis ray meeting the mirror at (x, y) reaches the image."""
    rays = optic.trace_generic(0.0, 0.0, x / _BEAM, y / _BEAM, optic.primary_wavelength)
    return bool(be.to_numpy(rays.i).ravel()[0] > 0)


def _polar(r: float, degrees: float, across: float = 0.0) -> tuple[float, float]:
    """The point r along the direction at the given angle, moved across it."""
    a = math.radians(degrees)
    return (
        r * math.cos(a) - across * math.sin(a),
        r * math.sin(a) + across * math.cos(a),
    )


def test_aperture_type(tmp_path, set_test_backend):
    aperture = _load(tmp_path, "SPID 2 3 0", "OBDC 3 0").surfaces[2].aperture
    assert type(aperture) is SpiderAperture
    assert (aperture.width, aperture.num_arms) == (2.0, 3)
    assert (aperture.offset_x, aperture.offset_y) == (3.0, 0.0)


class TestArms:
    @pytest.mark.parametrize("line,half", [("SPID 2 3 0", 1.0), ("SPID 4 4 0", 2.0)])
    def test_width_is_the_full_width(self, tmp_path, line, half, set_test_backend):
        optic = _load(tmp_path, line)
        for across in (0.0, half - 0.1, -(half - 0.1)):
            assert not _passes(optic, *_polar(8.0, 0.0, across))
        for across in (half + 0.1, -(half + 0.1)):
            assert _passes(optic, *_polar(8.0, 0.0, across))

    def test_arms_are_equally_spaced_from_plus_x(self, tmp_path, set_test_backend):
        optic = _load(tmp_path, "SPID 2 3 0")
        for degrees in (0.0, 120.0, 240.0):
            assert not _passes(optic, *_polar(8.0, degrees))
        for degrees in (60.0, 180.0, 300.0):
            assert _passes(optic, *_polar(8.0, degrees))

    def test_arms_run_one_way_from_the_centre(self, tmp_path, set_test_backend):
        # Opposite an arm of a three-arm spider is open; a four-arm spider
        # has an arm there.
        assert _passes(_load(tmp_path, "SPID 2 3 0"), -8.0, 0.0)
        assert not _passes(_load(tmp_path, "SPID 2 4 0"), -8.0, 0.0)

    def test_no_hub_beyond_the_arms(self, tmp_path, set_test_backend):
        # Midway between two arms, a ray passes once it is more than half a
        # width from both, however close to the centre.
        optic = _load(tmp_path, "SPID 2 3 0")
        assert not _passes(optic, *_polar(1.0, 60.0))
        assert _passes(optic, *_polar(1.2, 60.0))
        assert _passes(optic, -1.2, 0.0)

    def test_no_outer_limit(self, tmp_path, set_test_backend):
        # The surface has a fixed semi-diameter of 25: the spider replaces it.
        optic = _load(tmp_path, "SPID 2 3 0")
        assert _passes(optic, *_polar(28.0, 60.0))
        assert not _passes(optic, 28.0, 0.0)


class TestDecentre:
    def test_moves_the_centre(self, tmp_path, set_test_backend):
        optic = _load(tmp_path, "SPID 2 3 0", "OBDC 3 0")
        for point in [(3.0, 0.0), (2.0, 0.0), (3.0, 1.5), (3.0, -1.5), (8.0, 0.0)]:
            assert not _passes(optic, *point)
        for point in [(0.0, 0.0), (1.5, 0.0), (3.0, 3.0), (3.0, -3.0)]:
            assert _passes(optic, *point)

    def test_obdc_before_the_spider(self, tmp_path, set_test_backend):
        after = _load(tmp_path, "SPID 2 3 0", "OBDC 3 0").surfaces[2].aperture
        before = _load(tmp_path, "OBDC 3 0", "SPID 2 3 0").surfaces[2].aperture
        assert before.to_dict() == after.to_dict()


def test_pickup_field_is_not_a_parameter(tmp_path, set_test_backend):
    picked = _load(tmp_path, "SPID 2 3 1").surfaces[2].aperture
    plain = _load(tmp_path, "SPID 2 3 0").surfaces[2].aperture
    assert picked.to_dict() == plain.to_dict()


class TestRoundTrip:
    @pytest.mark.parametrize(
        "lines,written",
        [
            (("SPID 2 3 0",), {"SPID": [2, 3]}),
            (("SPID 2 3 0", "OBDC 3 0"), {"SPID": [2, 3], "OBDC": [3, 0]}),
            (("SPID 0.5 4 0", "OBDC -1 2"), {"SPID": [0.5, 4], "OBDC": [-1, 2]}),
        ],
    )
    def test_written_back(self, tmp_path, lines, written, set_test_backend):
        optic = _load(tmp_path, *lines)
        out = tmp_path / "round_trip.zmx"
        save_zemax_file(optic, str(out))
        records = {}
        for line in out.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if tokens and tokens[0] in ("SPID", "OBDC", "CLAP", "OBSC"):
                records[tokens[0]] = [float(t) for t in tokens[1:3]]
        assert records == written
        reloaded = load_zemax_file(str(out))
        before = optic.surfaces[2].aperture.to_dict()
        assert reloaded.surfaces[2].aperture.to_dict() == before
