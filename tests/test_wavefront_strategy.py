from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Set a backend before importing optiland modules
import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.samples.objectives import DoubleGauss
from optiland.wavefront.strategy import (
    BestFitSphereStrategy,
    CentroidReferenceSphereStrategy,
    ChiefRayStrategy,
    ReferenceStrategy,
    create_strategy,
)
from optiland.wavefront.wavefront_data import WavefrontData

from .utils import assert_allclose


@pytest.fixture
def optic():
    """Provides a DoubleGauss optic instance for testing."""
    return DoubleGauss()


@pytest.fixture
def distribution():
    """Provides a hexapolar distribution with 15 points."""
    dist = create_distribution("hexapolar")
    dist.generate_points(15)
    return dist


class ConcreteReferenceStrategy(ReferenceStrategy):
    """A concrete implementation of the abstract ReferenceStrategy for testing."""

    def compute_wavefront_data(self, field, wavelength):
        """Mock implementation for the abstract method."""
        pass  # Not needed for testing the base class methods

    def _create_reference_geometry(self, rays):
        """Mock implementation for the abstract method."""
        pass


class TestReferenceStrategy:
    """Tests for the abstract ReferenceStrategy base class."""

    @pytest.fixture
    def strategy(self, optic, distribution):
        """Fixture for the concrete strategy implementation."""
        return ConcreteReferenceStrategy(optic, distribution)

    def test_init(self, strategy, optic, distribution):
        """Test the constructor of ReferenceStrategy."""
        assert strategy.optic is optic
        assert strategy.distribution is distribution
        assert strategy.n_image == optic.surfaces.n(optic.primary_wavelength)[-1]

    def test_opd_image_to_xp(self, strategy, set_test_backend):
        """Test the OPD calculation from image to the exit pupil sphere."""
        # Mock ray data at the image plane
        rays_at_image = MagicMock()
        rays_at_image.x = be.array([0.1])
        rays_at_image.y = be.array([0.2])
        rays_at_image.z = be.array([100.0])
        rays_at_image.L = be.array([0.01])
        rays_at_image.M = be.array([0.02])
        rays_at_image.N = be.array([be.sqrt(1 - 0.01**2 - 0.02**2)])

        # Reference sphere parameters
        xc, yc, zc, R = 0.0, 0.0, 110.0, 10.0
        wavelength = 0.55

        opd = strategy._opd_image_to_xp(rays_at_image, xc, yc, zc, R, wavelength)
        assert opd.shape == (1,)
        assert_allclose(opd, -0.00250219)

    def test_opd_image_to_xp_negative_t(self, strategy, set_test_backend):
        """Test _opd_image_to_xp when the ray points away from the sphere."""
        # This ray points in the opposite direction
        rays_at_image = MagicMock()
        rays_at_image.x = be.array([0.0])
        rays_at_image.y = be.array([0.0])
        rays_at_image.z = be.array([0.0])
        rays_at_image.L = be.array([0.0])
        rays_at_image.M = be.array([0.0])
        rays_at_image.N = be.array([-1.0])  # Pointing away

        xc, yc, zc, R = 0.0, 0.0, 10.0, 5.0
        wavelength = 0.55

        opd = strategy._opd_image_to_xp(rays_at_image, xc, yc, zc, R, wavelength)
        # The second root should be chosen, resulting in a positive distance
        assert be.all(opd > 0)

    def test_correct_tilt_angle_field(self, set_test_backend):
        """Test tilt correction for a field specified by angle."""
        optic = DoubleGauss()
        dist = create_distribution("hexapolar")
        dist.generate_points(15)
        strategy = ConcreteReferenceStrategy(optic, dist)
        optic.fields.set_type("angle")
        opd = be.ones(strategy.distribution.x.shape)
        field = (0.5, 0.5)  # Hx, Hy

        corrected_opd = strategy._correct_tilt(field, opd)

        assert corrected_opd.shape == opd.shape
        assert not be.all(corrected_opd == opd)

    def test_correct_tilt_object_height_field(self, strategy, optic, set_test_backend):
        """Test tilt correction when field type is not 'angle'."""
        optic.fields.set_type("object_height")
        opd = be.ones(strategy.distribution.x.shape)
        field = (0.5, 0.5)

        corrected_opd = strategy._correct_tilt(field, opd)
        # No correction should be applied
        assert be.all(corrected_opd == opd)

    def test_correct_tilt_with_custom_coords(self, strategy, optic, set_test_backend):
        """Test tilt correction with explicitly passed coordinates."""
        optic.fields.set_type("angle")
        opd = be.ones(5)
        x = be.linspace(-1, 1, 5)
        y = be.linspace(-1, 1, 5)
        field = (0.5, 0.5)

        corrected_opd = strategy._correct_tilt(field, opd, x=x, y=y)

        assert corrected_opd.shape == opd.shape
        assert not be.all(corrected_opd == opd)


class TestChiefRayStrategy:
    """Tests for the ChiefRayStrategy."""

    @pytest.fixture
    def strategy(self, optic, distribution):
        return ChiefRayStrategy(optic, distribution)

    def test_calculate_sphere_from_chief_ray(self, set_test_backend):
        """Test the reference sphere calculation from a chief ray."""
        optic = DoubleGauss()
        dist = create_distribution("hexapolar")
        dist.generate_points(15)
        strategy = ChiefRayStrategy(optic, dist)
        chief_ray = MagicMock()
        chief_ray.x = be.array(0.1)
        chief_ray.y = be.array(0.2)
        chief_ray.z = be.array(100.0)

        x, y, z, R = strategy._calculate_sphere_from_chief_ray(chief_ray)

        assert x == 0.1
        assert y == 0.2
        assert z == 100.0
        assert isinstance(R, float)
        assert R > 0

    def test_calculate_sphere_from_chief_ray_error(self, strategy, set_test_backend):
        """Test that an error is raised if more than one chief ray is provided."""
        chief_ray = MagicMock()
        chief_ray.x = be.array([0.1, 0.2])  # More than one ray
        chief_ray.y = be.array([0.2, 0.3])
        chief_ray.z = be.array([100.0, 101.0])

        with pytest.raises(ValueError, match="Chief ray cannot be determined"):
            strategy._calculate_sphere_from_chief_ray(chief_ray)

    def test_compute_wavefront_data(self, set_test_backend):
        """Test the full wavefront data computation for ChiefRayStrategy."""
        optic = DoubleGauss()
        dist = create_distribution("uniform")
        dist.generate_points(18)
        strategy = ChiefRayStrategy(optic, dist)

        field = (0.0, 0.1)
        wavelength = 0.55

        # Use the real optic and distribution for an integration test
        wavefront_data = strategy.compute_wavefront_data(field, wavelength)

        assert isinstance(wavefront_data, WavefrontData)
        num_points = len(dist.x)
        assert wavefront_data.pupil_x.shape == (num_points,)
        assert wavefront_data.pupil_y.shape == (num_points,)
        assert wavefront_data.pupil_z.shape == (num_points,)
        assert wavefront_data.opd.shape == (num_points,)
        assert wavefront_data.intensity.shape == (num_points,)
        assert isinstance(wavefront_data.radius, float)
        assert wavefront_data.radius > 0


class TestCentroidReferenceSphereStrategy:
    """Tests for the CentroidReferenceSphereStrategy."""

    @pytest.fixture
    def strategy(self, optic, distribution):
        return CentroidReferenceSphereStrategy(optic, distribution, robust_trim_std=3.0)

    def test_points_from_rays(self, set_test_backend):
        """Test the conversion from ray data to wavefront points."""
        optic = DoubleGauss()
        dist = create_distribution("uniform")
        dist.generate_points(18)
        strategy = CentroidReferenceSphereStrategy(optic, dist)

        num_points = len(dist.x)
        rays = MagicMock()
        rays.x = be.array(be.random_uniform(size=num_points))
        rays.y = be.array(be.random_uniform(size=num_points))
        rays.z = be.array(be.random_uniform(size=num_points) + 100)
        rays.L = be.array(be.random_uniform(size=num_points) * 0.1)
        rays.M = be.array(be.random_uniform(size=num_points) * 0.1)
        rays.N = be.sqrt(1 - rays.L**2 - rays.M**2)
        rays.opd = be.array(be.random_uniform(size=num_points))
        rays.i = be.ones(num_points)

        points, valid_mask = strategy._points_from_rays(rays)
        assert points.shape[1] == 3
        assert be.all(valid_mask)

    def test_points_from_rays_with_invalid(self, set_test_backend):
        """Test _points_from_rays with some invalid ray data."""
        optic = DoubleGauss()
        dist = create_distribution("uniform")
        dist.generate_points(18)
        strategy = CentroidReferenceSphereStrategy(optic, dist)

        num_points = len(dist.x)
        rays = MagicMock()
        rays.x = be.array(be.random_uniform(size=num_points))
        rays.y = be.array(be.random_uniform(size=num_points))
        rays.z = be.array(be.random_uniform(size=num_points) + 100)
        rays.L = be.array(be.random_uniform(size=num_points) * 0.1)
        rays.M = be.array(be.random_uniform(size=num_points) * 0.1)
        rays.N = be.sqrt(1 - rays.L**2 - rays.M**2)
        rays.opd = be.array(be.random_uniform(size=num_points))
        rays.i = be.ones(num_points)

        rays.x[0] = be.nan
        rays.i = be.copy(rays.i)
        rays.i[1] = 0

        points, valid_mask = strategy._points_from_rays(rays)
        assert not valid_mask[0]
        assert not valid_mask[1]
        assert be.sum(valid_mask) == len(rays.x) - 2

    def test_points_from_rays_no_valid(self, set_test_backend):
        """Test _points_from_rays when no valid rays are found."""
        optic = DoubleGauss()
        dist = create_distribution("uniform")
        dist.generate_points(18)
        strategy = CentroidReferenceSphereStrategy(optic, dist)

        num_points = len(dist.x)
        rays = MagicMock()
        rays.x = be.array(be.random_uniform(size=num_points))
        rays.y = be.array(be.random_uniform(size=num_points))
        rays.z = be.array(be.random_uniform(size=num_points) + 100)
        rays.L = be.array(be.random_uniform(size=num_points) * 0.1)
        rays.M = be.array(be.random_uniform(size=num_points) * 0.1)
        rays.N = be.sqrt(1 - rays.L**2 - rays.M**2)
        rays.opd = be.array(be.random_uniform(size=num_points))
        rays.i = be.ones(num_points)

        rays.i = be.zeros_like(rays.i)
        with pytest.raises(ValueError, match="No valid ray samples found"):
            strategy._points_from_rays(rays)

    def test_calculate_reference_sphere(self, set_test_backend):
        """Test the reference sphere calculation."""
        optic = DoubleGauss()
        dist = create_distribution("uniform")
        dist.generate_points(18)
        strategy = CentroidReferenceSphereStrategy(optic, dist)

        rays = optic.trace(Hx=0.2, Hy=0.56, distribution=dist, wavelength=0.55)

        cx, cy, cz, r = strategy._calculate_reference_sphere(rays)
        assert isinstance(cx, float)
        assert isinstance(cy, float)
        assert isinstance(cz, float)
        assert isinstance(r, float)
        assert_allclose(cx, 4.87516947)
        assert_allclose(cy, 13.72537562)
        assert_allclose(cz, 139.454938)
        assert_allclose(r, 190.10010539)

    def test_calculate_reference_sphere_no_trim(self, set_test_backend):
        """Test sphere calculation without robust trimming."""
        optic = DoubleGauss()
        dist = create_distribution("uniform")
        dist.generate_points(18)
        strategy = CentroidReferenceSphereStrategy(optic, dist, robust_trim_std=0)

        rays = optic.trace(Hx=0.2, Hy=0.56, distribution=dist, wavelength=0.55)
        cx, cy, cz, r = strategy._calculate_reference_sphere(rays)
        assert r > 0

    def test_compute_wavefront_data(self, set_test_backend):
        """Test full wavefront data computation for CentroidReferenceSphereStrategy."""
        optic = DoubleGauss()
        dist = create_distribution("uniform")
        dist.generate_points(18)
        strategy = CentroidReferenceSphereStrategy(optic, dist)
        field = (0.0, 0.1)
        wavelength = 0.55

        # Integration test with real optic
        wavefront_data = strategy.compute_wavefront_data(field, wavelength)

        assert isinstance(wavefront_data, WavefrontData)
        num_points = len(dist.x)
        assert wavefront_data.pupil_x.shape == (num_points,)
        assert wavefront_data.opd.shape == (num_points,)
        # Check that piston is removed (mean OPD should be close to zero)
        assert_allclose(be.mean(wavefront_data.opd), 0.0)
        assert isinstance(wavefront_data.radius, float)
        assert wavefront_data.radius > 0


def test_create_strategy(optic, distribution, set_test_backend):
    """Test the factory function for creating strategies."""
    # Test chief_ray strategy creation
    chief_ray_strategy = create_strategy("chief_ray", optic, distribution)
    assert isinstance(chief_ray_strategy, ChiefRayStrategy)

    # Test centroid_sphere strategy creation
    centroid_strategy = create_strategy("centroid_sphere", optic, distribution)
    assert isinstance(centroid_strategy, CentroidReferenceSphereStrategy)
    # Check default kwargs
    assert centroid_strategy.robust_trim_std == 3.0

    # Test centroid_sphere strategy with custom kwargs
    centroid_strategy_custom = create_strategy(
        "centroid_sphere", optic, distribution, robust_trim_std=5.0
    )
    assert isinstance(centroid_strategy_custom, CentroidReferenceSphereStrategy)
    assert centroid_strategy_custom.robust_trim_std == 5.0

    # Test for unknown strategy
    with pytest.raises(
        ValueError, match="Unknown wavefront strategy: invalid_strategy"
    ):
        create_strategy("invalid_strategy", optic, distribution)

    # Test best_fit_sphere strategy creation
    bfs_strategy = create_strategy("best_fit_sphere", optic, distribution)
    assert isinstance(bfs_strategy, BestFitSphereStrategy)


class TestBestFitSphereStrategy:
    """Tests for the BestFitSphereStrategy."""

    def test_strategy_compare_at_best_focus(self, set_test_backend):
        """
        When the image surface is at best focus, BFS and centroid
        strategies should match.
        """
        optic = DoubleGauss()
        # Put image surface at known best focus position for field/wavelength
        optic.image_surface.geometry.cs.z = be.array([139.36352573])
        dist = create_distribution("uniform")
        dist.generate_points(32)
        strategy_centroid = CentroidReferenceSphereStrategy(optic, dist)
        data_centroid = strategy_centroid.compute_wavefront_data((0, 0), 0.5876)
        strategy_bfs = BestFitSphereStrategy(optic, dist)
        data_bfs = strategy_bfs.compute_wavefront_data((0, 0), 0.5876)

        assert isinstance(strategy_bfs.center, tuple)
        assert_allclose(strategy_bfs.center[0], 0)
        assert_allclose(strategy_bfs.center[1], 0)
        assert_allclose(strategy_bfs.center[2], 139.36352573)

        # when at best focus, both strategies should yield similar results
        assert_allclose(data_bfs.radius, data_centroid.radius)
