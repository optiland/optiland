"""Zemax circular obscurations (OBSC) import, decentre and round-trip.

OpticStudio's circular obscuration is the complement of the circular aperture:
``OBSC r_min r_max`` blocks rays between the two radii and passes the rest,
with no outer limit, and it replaces a fixed ("U") semi-diameter clip like any
other surface aperture. The prescriptions are written for these tests; a
collimated beam of radius 30 falls on a concave mirror, so a ray at normalized
pupil height p meets the mirror at 30 p.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.fileio import load_zemax_file, save_zemax_file
from optiland.physical_apertures import (
    OffsetRadialAperture,
    RadialAperture,
    UnionAperture,
)

_MIRROR = """VERS 221026 46 20120530 20120530
MODE SEQ
NAME OBSC test
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
    path = tmp_path / "obsc.zmx"
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


class TestCentralObscuration:
    def test_issue_reproduction(self, tmp_path, set_test_backend):
        # OpticStudio 2022 R2.02 on this file: the axial ray is blocked, rays
        # at 10, 20 and 30 mm reach the image. The 30 mm ray is outside the
        # U semi-diameter of 25, which the obscuration replaces.
        optic = _load(tmp_path, "OBSC 0 5 0")
        assert not _passes(optic)
        for y in (-30.0, -20.0, -10.0, 10.0, 20.0, 30.0):
            assert _passes(optic, y=y)
        assert _passes(optic, x=20.0)

    def test_aperture_type(self, tmp_path, set_test_backend):
        aperture = _load(tmp_path, "OBSC 0 5 0").surfaces[2].aperture
        assert type(aperture) is RadialAperture
        assert float(aperture.r_min) == 5.0
        assert float(aperture.r_max) == float("inf")

    def test_edge(self, tmp_path, set_test_backend):
        optic = _load(tmp_path, "OBSC 0 5 0")
        assert not _passes(optic, y=4.9)
        assert _passes(optic, y=5.1)


class TestAnnularObscuration:
    def test_blocks_only_between_the_radii(self, tmp_path, set_test_backend):
        optic = _load(tmp_path, "OBSC 2 5 0")
        assert type(optic.surfaces[2].aperture) is UnionAperture
        assert _passes(optic)
        assert _passes(optic, y=1.5)
        assert not _passes(optic, y=2.5)
        assert not _passes(optic, x=-4.5)
        assert _passes(optic, y=5.5)
        assert _passes(optic, y=-28.0)


class TestDecentredObscuration:
    @pytest.mark.parametrize(
        "lines", [("OBSC 0 5 0", "OBDC 0 8"), ("OBDC 0 8", "OBSC 0 5 0")]
    )
    def test_obdc_moves_the_obscuration(self, tmp_path, lines, set_test_backend):
        optic = _load(tmp_path, *lines)
        aperture = optic.surfaces[2].aperture
        assert type(aperture) is OffsetRadialAperture
        assert (aperture.offset_x, aperture.offset_y) == (0.0, 8.0)
        assert _passes(optic)
        assert not _passes(optic, y=8.0)
        assert not _passes(optic, y=12.0)
        assert _passes(optic, y=14.0)

    def test_annular_decentre_moves_both_radii(self, tmp_path, set_test_backend):
        optic = _load(tmp_path, "OBSC 2 5 0", "OBDC 6 0")
        assert _passes(optic, x=6.0)
        assert not _passes(optic, x=9.5)
        assert not _passes(optic, x=2.5)
        assert _passes(optic)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "lines,written",
        [
            (("OBSC 0 5 0",), {"OBSC": [0, 5]}),
            (("OBSC 2 5 0",), {"OBSC": [2, 5]}),
            (("OBSC 0 5 0", "OBDC 1.5 -2"), {"OBSC": [0, 5], "OBDC": [1.5, -2]}),
            (("OBSC 2 5 0", "OBDC 6 0"), {"OBSC": [2, 5], "OBDC": [6, 0]}),
        ],
    )
    def test_written_back_as_obsc(self, tmp_path, lines, written, set_test_backend):
        optic = _load(tmp_path, *lines)
        reloaded, text = _round_trip(optic, tmp_path)
        records = {}
        for line in text.splitlines():
            tokens = line.split()
            if tokens and tokens[0] in ("OBSC", "OBDC", "CLAP"):
                records[tokens[0]] = [float(t) for t in tokens[1:3]]
        assert records == written
        before = optic.surfaces[2].aperture.to_dict()
        assert reloaded.surfaces[2].aperture.to_dict() == before

    def test_reloaded_lens_traces_the_same(self, tmp_path, set_test_backend):
        reloaded, _ = _round_trip(_load(tmp_path, "OBSC 0 5 0"), tmp_path)
        assert not _passes(reloaded)
        assert _passes(reloaded, y=10.0)
        assert _passes(reloaded, y=30.0)
