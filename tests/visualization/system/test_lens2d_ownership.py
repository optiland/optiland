"""Lens ownership metadata stays exact across projections and redraws."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.visualization.system.lens import Lens2D
from optiland.visualization.system.surface import Surface2D


@pytest.mark.parametrize("projection", ["XZ", "YZ"])
def test_cemented_annular_bodies_keep_exact_surface_ownership(
    set_test_backend, projection
):
    optic = Optic()
    optic.surfaces.add(index=0, radius=np.inf, thickness=10)
    for index, material in enumerate(
        (IdealMaterial(1.5), IdealMaterial(1.6), IdealMaterial(1.0))
    ):
        optic.surfaces.add(
            index=index + 1,
            radius=np.inf,
            thickness=3,
            material=material,
            aperture=RadialAperture(r_max=5, r_min=1),
        )
    optic.surfaces.add(index=4, radius=np.inf)
    surfaces = tuple(optic.surfaces)[1:4]
    lens = Lens2D([Surface2D(surface, 5) for surface in surfaces])
    ax = Figure().add_subplot()
    artists = lens.plot(ax, projection=projection)

    assert len(artists) == 4  # Each of the two glass regions is split at the hole.
    assert all(component is lens for component in artists.values())
    assert set(lens.artist_surfaces) == set(artists)
    assert set(lens.artist_surfaces.values()) == {
        surfaces[:2],
        surfaces[1:],
    }
    assert set(lens.boundary_coordinates) == set(surfaces)
    for surface, (z, height) in lens.boundary_coordinates.items():
        finite = np.isfinite(z) & np.isfinite(height)
        np.testing.assert_allclose(z[finite], float(surface.geometry.cs.z))
        assert np.min(np.abs(height[finite])) >= 1
        assert np.max(np.abs(height[finite])) == pytest.approx(5)

    previous_artists = set(artists)
    ax.clear()
    face_on = lens.plot(ax, projection="XY")
    assert set(lens.artist_surfaces) == set(face_on)
    assert not previous_artists.intersection(lens.artist_surfaces)
    assert list(lens.artist_surfaces.values()) == [surfaces]
    assert lens.boundary_coordinates == {}

    ax.clear()
    redrawn = lens.plot(ax, projection=projection)
    assert set(lens.artist_surfaces) == set(redrawn)
    assert set(lens.boundary_coordinates) == set(surfaces)
