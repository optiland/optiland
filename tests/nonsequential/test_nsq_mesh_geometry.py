"""MeshGeometry intersection from a point on the mesh surface.

A ray that has just refracted through a mesh face is re-launched from that
face. The mesh must then report the *next* surface along the ray (for a
closed solid, the far wall), not lose the ray because the first candidate
returned by trimesh is the face it starts on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.materials.ideal import IdealMaterial  # noqa: E402
from optiland.nonsequential import (  # noqa: E402
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    MeshGeometry,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
)


def _slab_mesh(engine: str):
    """20 x 20 x 3 mm box, faces at z = +-1.5, using the requested ray caster."""
    mesh = trimesh.creation.box(extents=(20.0, 20.0, 3.0))
    if engine == "embree":
        if not trimesh.ray.has_embree:
            pytest.skip("embree ray caster not installed")
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    else:
        from trimesh.ray.ray_triangle import RayMeshIntersector
    mesh.ray = RayMeshIntersector(mesh)
    return mesh


@pytest.mark.parametrize("engine", ["triangle", "embree"])
def test_ray_starting_on_face_hits_far_wall(engine):
    """From the entrance face, going inwards, the next hit is the back face."""
    geom = MeshGeometry(_slab_mesh(engine))
    origins = np.array([[0.3, -0.2, -1.5], [-4.0, 2.0, -1.5]])
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]])

    t, normals, hit, n_geom = geom.ray_intersect(origins, directions)

    assert np.all(np.asarray(hit))
    np.testing.assert_allclose(np.asarray(t), [3.0, 3.0 / 0.8], rtol=1e-9)
    # n_geom is the raw outward face normal of the back face (+z)
    np.testing.assert_allclose(np.asarray(n_geom), [[0, 0, 1], [0, 0, 1]], atol=1e-12)
    # normals are flipped to face the incoming ray
    assert np.all((np.asarray(normals) * directions).sum(axis=1) < 0)


def test_closed_mesh_slab_transmission():
    """A closed mesh slab transmits (1 - R) / (1 + R), like the analytic slab.

    Before the fix the back face was never seen: transmission was 1 - R.
    NumPy backend only: MeshGeometry hands the ray arrays to trimesh on the
    host and does not support grad-tracking torch tensors (mesh geometry is
    forward-only, see optiland.nonsequential module docs).
    """
    n = 1.5
    r = ((n - 1) / (n + 1)) ** 2
    expected = (1 - r) / (1 + r)  # incoherent multiple reflections, 0.92308
    num_rays = 50_000

    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-5.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=2.0
        ),
    )
    pmma = NSQMaterial(optiland_material=IdealMaterial(n=n))
    scene.add_component(
        "Slab",
        RefractiveComponent(
            CoordinateSystem(),
            MeshGeometry(_slab_mesh("triangle")),
            material_front=pmma,
            material_back=VACUUM,
            name="Slab",
        ),
    )
    scene.add_detector(
        "D", CoordinateSystem(z=5.0), IrradianceDetectorConfig(width=30, height=30)
    )
    result = scene.trace(num_rays=num_rays, seed=1)
    transmitted = float(result.detectors["D"].total_flux)

    sigma = math.sqrt(expected * (1 - expected) / num_rays)  # Monte Carlo, ~0.0012
    assert abs(transmitted - expected) < 5 * sigma
    assert transmitted < 1 - r - 10 * sigma  # clearly not the single-face value
