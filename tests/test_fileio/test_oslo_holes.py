"""Undeviated holes on neutral planes and explicit limits elsewhere."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.fileio import load_oslo_file
from optiland.rays import RealRays
from tests.utils import assert_allclose


def hole_lens(
    tmp_path, surface="", prefix="", bounds="AX1 A -1; AX2 A 1; AY1 A -2; AY2 A 2"
):
    path = tmp_path / "hole.len"
    path.write_text(
        'LEN NEW "hole" 1 2; EBR 1; TH 10; '
        + prefix
        + "; NXT; "
        + surface
        + "; TH 3; APN 1; ATP A 2; AAC A 1; "
        + bounds
        + "; NXT; END 2"
    )
    return path


def test_neutral_hole_does_not_clip_or_change_direction(tmp_path, set_test_backend):
    optic = load_oslo_file(hole_lens(tmp_path), strict=True)
    rays = RealRays(
        be.array([0.0, 3.0]),
        be.zeros(2),
        be.full(2, -1.0),
        be.full(2, 0.2),
        be.zeros(2),
        be.full(2, (1 - 0.2**2) ** 0.5),
        be.ones(2),
        be.full(2, 0.55),
    )
    optic.surfaces[1].trace(rays)
    assert_allclose(rays.i, [1, 1])
    assert_allclose(rays.L, [0.2, 0.2])
    assert_allclose(optic.surfaces[1].aperture.extent, [-1, 1, -2, 2])


@pytest.mark.parametrize(
    "surface,prefix", [("GLA 1.5", ""), ("RFL", ""), ("RD 30", ""), ("", "GLA 1.5")]
)
def test_non_neutral_hole_warns_or_rejects(tmp_path, surface, prefix):
    path = hole_lens(tmp_path, surface, prefix)
    with pytest.raises(ValueError, match="undeviated hole"):
        load_oslo_file(path, strict=True)
    with pytest.warns(UserWarning, match="undeviated hole"):
        load_oslo_file(path)


def test_neutral_hole_still_validates_shape(tmp_path):
    with pytest.raises(ValueError, match="increasing bounds"):
        load_oslo_file(hole_lens(tmp_path, bounds="AX1 A 1; AX2 A -1"), strict=True)
