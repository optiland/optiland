"""Drawn rays follow curved paths through gradient-index media."""

from __future__ import annotations

import numpy as np

from optiland.materials import GradientMaterial
from optiland.optic import Optic
from optiland.visualization.system.ray_path import splice_curved_segments
from optiland.visualization.system.rays import Rays2D, Rays3D


def _rod() -> Optic:
    n0, a = 1.6, 0.01
    optic = Optic()
    optic.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    optic.surfaces.add(
        index=1,
        radius=np.inf,
        thickness=np.pi / np.sqrt(a),  # half pitch: the image inverts
        material=GradientMaterial(n0=n0, nr2=-n0 * a / 2),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=np.inf, thickness=5.0, material="air")
    optic.surfaces.add(index=3, radius=np.inf)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def test_splice_inserts_finite_interior_samples():
    points = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 2]], dtype=float)
    segment = np.array([[[0, 1, 1]], [[0, 1.5, 1.2]], [[np.nan] * 3], [[0, 2, 2]]])
    spliced, intensity, neutral = splice_curved_segments(
        points,
        np.array([1.0, 0.5, 0.0]),
        np.array([False, True, False]),
        {2: segment},
        0,
    )
    np.testing.assert_array_equal(spliced[2], [0, 1.5, 1.2])
    assert len(spliced) == 4
    np.testing.assert_array_equal(intensity, [1.0, 0.5, 0.5, 0.0])
    np.testing.assert_array_equal(neutral, [False, True, False, False])


def test_rays_through_grin_rod_are_curved():
    for plotter_type in (Rays2D, Rays3D):
        plotter = plotter_type(_rod())
        plotter._trace((0.0, 0.0), 0.55, 3, "line_y")
        paths = list(plotter._iter_physical_paths())
        assert len(paths) == 3
        for path in paths:
            assert len(path) > 50  # ~31 mm in 0.1 mm steps, plus surface hits
            assert np.isfinite(path).all()
        # The off-axis rays cross the axis inside the rod and exit inverted.
        top = paths[-1]
        assert top[1, 1] * top[-2, 1] < 0
