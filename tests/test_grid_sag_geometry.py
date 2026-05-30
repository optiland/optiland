"""Unit tests for the GridSagGeometry class."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.grid_sag import GridSagGeometry
from optiland.rays.real_rays import RealRays


@pytest.fixture
def sample_grid_geometry(set_test_backend):
    """Returns a sample GridSagGeometry instance for testing."""
    cs = CoordinateSystem()
    x_coords = [-1.0, 0.0, 1.0]
    y_coords = [-1.0, 0.0, 1.0]
    sag_values = [[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]]  # A simple hill
    return GridSagGeometry(cs, x_coords, y_coords, sag_values)


def test_grid_sag_geometry_init(sample_grid_geometry):
    """Test the initialization of the GridSagGeometry class."""
    assert sample_grid_geometry is not None
    assert be.array_equal(sample_grid_geometry.x_grid, be.asarray([-1.0, 0.0, 1.0]))
    assert be.array_equal(sample_grid_geometry.y_grid, be.asarray([-1.0, 0.0, 1.0]))
    assert be.array_equal(
        sample_grid_geometry.sag_grid,
        be.asarray([[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]]),
    )


def test_grid_sag_sag_calculation(set_test_backend):
    """Test the sag calculation for the GridSagGeometry."""
    cs = CoordinateSystem()
    x_coords = [0.0, 1.0]
    y_coords = [0.0, 1.0]
    sag_values = [[0.0, 0.1], [0.2, 0.3]]
    geometry = GridSagGeometry(cs, x_coords, y_coords, sag_values)

    # Test interpolation
    assert be.isclose(geometry.sag(0.5, 0.5), be.asarray(0.15))
    # Test on grid points
    assert be.isclose(geometry.sag(0.0, 0.0), be.asarray(0.0))
    assert be.isclose(geometry.sag(1.0, 1.0), be.asarray(0.3))
    # Test out of bounds
    assert be.isnan(geometry.sag(-1.0, 0.5))


def test_grid_sag_distance_and_normal(set_test_backend):
    """Test the distance and surface normal calculations."""
    cs = CoordinateSystem()
    x_coords = [-1.0, 1.0]
    y_coords = [-1.0, 1.0]
    # A simple plane tilted in x
    sag_values = [[-0.1, 0.1], [-0.1, 0.1]]
    geometry = GridSagGeometry(cs, x_coords, y_coords, sag_values)

    # Ray starting at (0, 0, -1) and traveling along +z
    rays = RealRays(
        x=[0.0],
        y=[0.0],
        z=[-1.0],
        L=[0.0],
        M=[0.0],
        N=[1.0],
        intensity=1.0,
        wavelength=0.55,
    )

    distance = geometry.distance(rays)
    dtype = rays.x.dtype
    assert be.allclose(distance, be.asarray([1.0], dtype=dtype), atol=1e-5)

    # Move ray to the surface and check normal
    rays.z = rays.z + distance
    nx, ny, nz = geometry.surface_normal(rays)

    # Expected normal for sag = 0.1*x is (-0.1, 0, 1) normalized
    expected_norm_mag = be.sqrt(be.asarray(0.1**2 + 1, dtype=dtype))
    expected_nx = -0.1 / expected_norm_mag
    assert be.allclose(nx, be.asarray([expected_nx], dtype=dtype), atol=1e-5)
    assert be.allclose(ny, be.asarray([0.0], dtype=dtype), atol=1e-5)
    assert be.allclose(nz, be.asarray([1 / expected_norm_mag], dtype=dtype), atol=1e-5)


def test_grid_sag_serialization(set_test_backend):
    """Test the to_dict and from_dict methods."""
    cs = CoordinateSystem()
    x_coords = [-1.0, 0.0, 1.0]
    y_coords = [-1.0, 0.0, 1.0]
    sag_values = [[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]]
    geometry = GridSagGeometry(cs, x_coords, y_coords, sag_values)

    data = geometry.to_dict()
    new_geometry = GridSagGeometry.from_dict(data)

    assert isinstance(new_geometry, GridSagGeometry)
    assert be.array_equal(new_geometry.x_grid, geometry.x_grid)
    assert be.array_equal(new_geometry.y_grid, geometry.y_grid)
    assert be.array_equal(new_geometry.sag_grid, geometry.sag_grid)


def test_grid_sag_flip(sample_grid_geometry):
    """Test the flip method."""
    original_sag = be.copy(sample_grid_geometry.sag_grid)
    sample_grid_geometry.flip()
    assert be.array_equal(sample_grid_geometry.sag_grid, -original_sag)
