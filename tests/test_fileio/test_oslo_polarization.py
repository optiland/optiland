"""OSLO local Cartesian Jones elements, with independent complex-field checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

import optiland.backend as be
from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.optic import Optic
from optiland.rays import PolarizationState, PolarizedRays
from tests.utils import assert_allclose


def polar_lens(tmp_path, commands, tilt=""):
    path = tmp_path / "polar.len"
    path.write_text(
        'LEN NEW "polarization" 1 2; EBR 1; TH 10; NXT; TH 3; '
        + commands
        + "; "
        + tilt
        + "; NXT; END 2"
    )
    return path


def axial_bundle():
    return PolarizedRays(*[be.array([v]) for v in (0, 0, -1, 0, 0, 1, 1, 0.55)])


def test_jones_reflection_uses_local_cartesian_matrix(tmp_path, set_test_backend):
    optic = load_oslo_file(polar_lens(tmp_path, "RFL; JAA .5; JAD .5"), strict=True)
    rays = axial_bundle()
    optic.surfaces[1].trace(rays)
    rays.update_intensity(PolarizationState())
    assert_allclose(rays.N, [-1])
    assert_allclose(rays.i, [0.25])


def test_jones_object_surface_is_rejected(lens_file):
    with pytest.raises(ValueError, match="on the object"):
        load_oslo_file(lens_file(system="JAA .5"), strict=True)


def test_jones_amplitude_phase_and_native_serialization(tmp_path, set_test_backend):
    path = polar_lens(
        tmp_path, "JAA .6; JPA 90; JAB .2; JPB -90; JAC .3; JPC 180; JAD .4; JPD 0"
    )
    optic = load_oslo_file(path, strict=True)
    restored = Optic.from_dict(json.loads(json.dumps(optic.to_dict())))
    expected = np.array([[0.6j, -0.2j, 0], [-0.3, 0.4, 0], [0, 0, 1]])
    for candidate in (optic, restored):
        rays = axial_bundle()
        candidate.surfaces[1].trace(rays)
        assert_allclose(rays.p[0], expected)
        rays.update_intensity(
            PolarizationState(is_polarized=True, Ex=1, Ey=0, phase_x=0, phase_y=0)
        )
        assert rays.i.item() == pytest.approx(0.6**2 + 0.3**2)
    output = tmp_path / "protected.len"
    output.write_text("keep")
    with pytest.raises(NotImplementedError, match="coating"):
        save_oslo_file(optic, output)
    assert output.read_text() == "keep"


def test_jones_identity_defaults_and_surface_axes(tmp_path, set_test_backend):
    optic = load_oslo_file(polar_lens(tmp_path, "JPA 90", "TLC 90"), strict=True)
    rays = axial_bundle()
    optic.surfaces[1].trace(rays)
    # The local X retarder acts on global Y after a 90-degree surface roll.
    assert_allclose(rays.p[0], np.diag([1, 1j, 1]))


@pytest.mark.parametrize("command", ["JAA", "JPA 0 1", "JAB -1", "JPC nan"])
def test_invalid_jones_commands_fail(tmp_path, command):
    with pytest.raises(ValueError, match="J[AP][ABCD]"):
        load_oslo_file(polar_lens(tmp_path, command), strict=True)
