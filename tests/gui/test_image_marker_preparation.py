"""Image marker sizing survives detached 2D and 3D layout preparation."""

from __future__ import annotations

import threading

import numpy as np

import optiland.backend as be
from optiland.physical_apertures import RadialAperture
from optiland_gui.services.job_records import OpticSnapshot
from optiland_gui.services.layout_tasks import prepare_2d, prepare_3d


def test_prepared_image_marker_retains_schematic_extent_and_provenance(
    set_test_backend, minimal_optic
):
    minimal_optic.surfaces[2].aperture = RadialAperture(3)
    snapshot = OpticSnapshot.capture(minimal_optic)

    def progress(*args):
        pass

    for count in (3, 9):
        data = prepare_2d(
            snapshot,
            {"num_rays": count, "distribution": "line_y"},
            progress,
            threading.Event(),
        )
        image = next(
            primitive
            for primitive in data["primitives"]
            if primitive["role"] == "surface" and primitive["surfaces"] == (3,)
        )
        assert "schematic image plane" in image["label"]
        assert data["extent_sources"][3] == "schematic"
        np.testing.assert_allclose(image["xy"][:, 1][[0, -1]], [-3, 3])
        np.testing.assert_allclose(data["references"][3][1][[0, -1]], [-3, 3])

    data = prepare_3d(snapshot, {}, progress, threading.Event())
    image = next(mesh for mesh in data["meshes"] if mesh["surfaces"] == (3,))
    # Mesh points are local; the matrix supplies the unchanged image vertex.
    np.testing.assert_allclose(
        [image["points"][:, 1].min(), image["points"][:, 1].max()], [-3, 3]
    )
    np.testing.assert_allclose(
        image["matrix"][:3, 3],
        be.to_numpy(
            minimal_optic.surfaces[-1].geometry.cs.get_effective_transform()[0]
        ).ravel(),
    )
    assert minimal_optic.surfaces[-1].aperture is None
