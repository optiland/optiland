"""OSLO polygon boundaries must satisfy the documented vertex geometry."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.fileio import load_oslo_file
from tests.utils import assert_allclose


def polygon_commands(vertices):
    commands = [f"APN 1\nATP A {len(vertices)}"]
    commands.extend(
        f"AVX{index} A {x}\nAVY{index} A {y}"
        for index, (x, y) in enumerate(vertices, 1)
    )
    return "\n".join(commands)


def test_unknown_aperture_action_warns_and_is_not_applied(lens_file):
    path = lens_file(surface="APN 1; AAC A 3")
    with pytest.warns(UserWarning, match="aperture action 3"):
        optic = load_oslo_file(path)
    assert optic.surfaces[1].aperture.extent == (-3, 3, -3, 3)
    with pytest.raises(ValueError, match="aperture action 3"):
        load_oslo_file(path, strict=True)


@pytest.mark.parametrize(
    "vertices",
    [
        [(0, 0), (1, 1), (2, 2)],  # Collinear triangle.
        [(0, 0), (1, 1), (0, 0)],  # Repeated vertex.
        [(-1, -1), (1, 1), (-1, 1), (1, -1)],  # Crossing edges.
        [(0, 0), (2, 0), (0.5, 0.5), (0, 2)],  # Concave quadrangle.
        [(0, 0), (1, 0), (2, 0), (0, 1)],  # Straight interior angle.
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_invalid_polygon_boundaries_are_rejected(lens_file, vertices, strict):
    with pytest.raises(ValueError, match="polygon.*nondegenerate.*convex"):
        load_oslo_file(lens_file(surface=polygon_commands(vertices)), strict=strict)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("units", [1, 10])
@pytest.mark.parametrize(
    "vertices,point",
    [
        ([(0, 0), (2, 0), (0, 2)], (0.5, 0.5)),
        ([(-1, -1), (1, -1), (1, 1), (-1, 1)], (0, 0)),
    ],
)
def test_polygon_winding_and_units_preserve_interior_clipping(
    lens_file, set_test_backend, vertices, point, reverse, units
):
    optic = load_oslo_file(
        lens_file(
            system=f"UNI {units}",
            surface=polygon_commands(vertices[::-1] if reverse else vertices),
        ),
        strict=True,
    )
    aperture = optic.surfaces[1].aperture
    x = be.array([point[0], 3]) * units
    y = be.array([point[1], 3]) * units
    assert_allclose(aperture.contains(x, y), [True, False])
