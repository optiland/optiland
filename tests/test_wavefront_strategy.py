from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Set a backend before importing optiland modules
import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import RealRays
from optiland.samples.objectives import DoubleGauss
from optiland.wavefront import OPD
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

    def test_restore_launch_phase_angle_field(self, set_test_backend):
        """Test launch-phase restoration for an angular field."""
        optic = DoubleGauss()
        dist = create_distribution("hexapolar")
        dist.generate_points(15)
        strategy = ConcreteReferenceStrategy(optic, dist)
        optic.fields.set_type("angle")
        opd = be.ones(strategy.distribution.x.shape)
        field = (0.5, 0.5)  # Hx, Hy

        reference = optic.trace_generic(
            *field,
            Px=0.0,
            Py=0.0,
            wavelength=optic.primary_wavelength,
            retain_launch=True,
        )
        rays = optic.trace(
            *field,
            optic.primary_wavelength,
            None,
            dist,
            retain_launch=True,
        )
        corrected_opd = strategy._restore_launch_phase(
            rays, opd, optic.primary_wavelength, reference
        )

        assert corrected_opd.shape == opd.shape
        assert not be.all(corrected_opd == opd)

    def test_restore_launch_phase_object_height_field(
        self, strategy, optic, set_test_backend
    ):
        """Test launch phase when the field type is not angular."""
        optic.fields.set_type("object_height")
        opd = be.ones(strategy.distribution.x.shape)
        rays = MagicMock(opd=opd)

        corrected_opd = strategy._restore_launch_phase(
            rays, opd, optic.primary_wavelength, rays
        )
        # No correction should be applied
        assert be.all(corrected_opd == opd)

    def test_restore_launch_phase_uses_generated_state(self, set_test_backend):
        """Test launch phase with explicitly traced pupil coordinates."""
        optic = DoubleGauss()
        dist = create_distribution("hexapolar")
        dist.generate_points(15)
        strategy = ConcreteReferenceStrategy(optic, dist)
        optic.fields.set_type("angle")
        opd = be.ones(5)
        x = be.linspace(-1, 1, 5)
        y = be.linspace(-1, 1, 5)
        field = (0.5, 0.5)

        reference = optic.trace_generic(
            *field,
            Px=0.0,
            Py=0.0,
            wavelength=optic.primary_wavelength,
            retain_launch=True,
        )
        rays = optic.trace_generic(
            *field, x, y, optic.primary_wavelength, retain_launch=True
        )
        corrected_opd = strategy._restore_launch_phase(
            rays, opd, optic.primary_wavelength, reference
        )

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

    def test_masks_non_finite_ray_samples(self, set_test_backend):
        """Test that invalid rays cannot contaminate downstream wavefront data."""
        optic = MagicMock()
        optic.primary_wavelength = 0.55
        optic.surfaces.n.return_value = be.array([1.0])
        optic.surfaces.positions = [0.0]
        optic.surfaces.intensity = be.array([[1.0, 1.0]])
        optic.paraxial.XPL.return_value = 0.0
        optic.fields.field_definition = object()

        chief_ray = MagicMock()
        chief_ray.x = be.array(0.0)
        chief_ray.y = be.array(0.0)
        chief_ray.z = be.array(10.0)
        chief_ray.opd = be.array(0.0)
        optic.trace_generic.return_value = chief_ray

        rays = MagicMock()
        rays.x = be.array([0.0, float("nan")])
        rays.y = be.array([0.0, float("nan")])
        rays.z = be.array([10.0, float("nan")])
        rays.L = be.array([0.0, float("nan")])
        rays.M = be.array([0.0, float("nan")])
        rays.N = be.array([1.0, float("nan")])
        rays.opd = be.array([0.0, float("nan")])
        rays.p = None
        rays.get_exit_fields = None
        optic.trace.return_value = rays

        geometry = MagicMock(radius=10.0)
        geometry.path_length.side_effect = [
            be.array(0.0),
            be.array([0.0, float("nan")]),
        ]
        strategy = ChiefRayStrategy(optic, MagicMock())
        strategy._create_reference_geometry = MagicMock(return_value=geometry)

        wavefront_data = strategy.compute_wavefront_data((0.0, 0.0), 0.55)

        assert be.all(be.isfinite(wavefront_data.pupil_x))
        assert be.all(be.isfinite(wavefront_data.pupil_y))
        assert be.all(be.isfinite(wavefront_data.pupil_z))
        assert be.all(be.isfinite(wavefront_data.opd))
        assert_allclose(wavefront_data.intensity, be.array([1.0, 0.0]))


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


# Waves. Float64 noise on a several-hundred-wave signal sits near 1e-11.
ZERO_WAVEFRONT_TOLERANCE = 1e-10

# Millimetres. Curving the index-matched stop adds sag without adding refractive
# power, so the plane wave stays flat while the stop is no longer its own tangent
# plane. Paraxial aiming targets that tangent plane and iterative aiming targets
# the real surface, which is what makes the two aimers select different launch
# coordinates for an oblique field.
AIMING_DIVERGENT_RADIUS = 20.0


def collimated_planes(
    index: float = 1.0,
    field: tuple[float, float] = (0.0, 5.0),
    vignette: tuple[float, float] = (0.0, 0.0),
    first_radius: float = be.inf,
) -> Optic:
    """Create index-matched planes that preserve a collimated plane wave."""
    medium = IdealMaterial(index)
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf, material=medium)
    optic.surfaces.add(
        index=1,
        radius=first_radius,
        thickness=10.0,
        material=medium,
        is_stop=True,
    )
    optic.surfaces.add(index=2, thickness=10.0, material=medium)
    optic.surfaces.add(index=3, material=medium)
    optic.set_aperture("EPD", 4.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.fields.add(x=field[0], y=field[1], vx=vignette[0], vy=vignette[1])
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def max_abs_wavefront(
    optic: Optic, num_rays: int = 6, strategy: str = "chief_ray"
) -> float:
    """Return the peak absolute wavefront error over valid samples."""
    field = optic.fields.get_field_coords()[-1]
    analysis = OPD(
        optic,
        field,
        "primary",
        num_rays=num_rays,
        strategy=strategy,
        afocal=True,
        remove_tilt=False,
    )
    data = analysis.get_data(field, optic.primary_wavelength)
    opd = be.to_numpy(data.opd)
    keep = be.to_numpy(data.intensity) > 0
    return float(abs(opd[keep]).max())


@pytest.mark.parametrize("index", [1.0, 1.33, 1.5, 2.0])
@pytest.mark.parametrize(
    "field", [(0.0, 0.0), (0.0, 5.0), (0.0, -5.0), (5.0, 0.0), (3.0, 4.0)]
)
def test_plane_wave_is_flat_for_any_object_index(
    set_test_backend: None, index: float, field: tuple[float, float]
) -> None:
    """The launch term is an optical path, so it carries the object index."""
    optic = collimated_planes(index=index, field=field)
    assert max_abs_wavefront(optic) < ZERO_WAVEFRONT_TOLERANCE


@pytest.mark.parametrize("vignette", [(0.0, 0.25), (0.25, 0.0), (0.4, 0.1)])
@pytest.mark.parametrize("index", [1.0, 1.5])
def test_plane_wave_is_flat_with_vignetted_pupils(
    set_test_backend: None, index: float, vignette: tuple[float, float]
) -> None:
    """Vignetting scales the launch offsets and their relative phase."""
    optic = collimated_planes(index=index, field=(3.0, 4.0), vignette=vignette)
    assert max_abs_wavefront(optic) < ZERO_WAVEFRONT_TOLERANCE


@pytest.mark.parametrize("first_radius", [be.inf, AIMING_DIVERGENT_RADIUS])
@pytest.mark.parametrize("aiming", ["paraxial", "iterative", "robust"])
@pytest.mark.parametrize("strategy", ["chief_ray", "centroid", "best_fit"])
def test_plane_wave_is_flat_for_each_ray_aimer(
    set_test_backend: None, aiming: str, strategy: str, first_radius: float
) -> None:
    """Launch phase follows the coordinates selected by the active aimer."""
    optic = collimated_planes(
        index=1.5,
        field=(3.0, 4.0),
        vignette=(0.4, 0.1),
        first_radius=first_radius,
    )
    optic.ray_tracer.set_aiming(aiming, max_iter=20, tol=1e-8)
    assert max_abs_wavefront(optic, strategy=strategy) < ZERO_WAVEFRONT_TOLERANCE


def test_iterative_aiming_changes_the_vignetted_launch_map(
    set_test_backend: None,
) -> None:
    """The aiming regression exercises distinct generated launch coordinates."""
    optic = collimated_planes(
        index=1.5,
        field=(3.0, 4.0),
        vignette=(0.4, 0.1),
        first_radius=AIMING_DIVERGENT_RADIUS,
    )
    field = optic.fields.get_field_coords()[-1]
    distribution = create_distribution("hexapolar")
    distribution.generate_points(6)

    optic.ray_tracer.set_aiming("paraxial")
    paraxial = optic.trace(
        *field, optic.primary_wavelength, None, distribution, retain_launch=True
    )
    optic.ray_tracer.set_aiming("iterative", max_iter=20, tol=1e-8)
    iterative = optic.trace(
        *field, optic.primary_wavelength, None, distribution, retain_launch=True
    )

    paraxial_launch = paraxial._launch_state
    iterative_launch = iterative._launch_state
    displacement = be.sqrt(
        (paraxial_launch.x - iterative_launch.x) ** 2
        + (paraxial_launch.y - iterative_launch.y) ** 2
        + (paraxial_launch.z - iterative_launch.z) ** 2
    )
    assert float(be.to_numpy(be.max(displacement))) > 1e-3


@pytest.mark.parametrize("strategy", ["centroid", "best_fit"])
def test_wavefront_is_independent_of_pupil_sample_order(
    set_test_backend: None, strategy: str
) -> None:
    """Reference geometry uses the chief launch rather than an array element."""
    optic = DoubleGauss()
    field = optic.fields.get_field_coords()[-1]
    wavelength = optic.primary_wavelength
    forward = create_distribution("hexapolar")
    forward.generate_points(6)
    reverse = create_distribution("hexapolar")
    reverse.generate_points(6)
    reverse.x = be.flip(reverse.x)
    reverse.y = be.flip(reverse.y)

    forward_data = OPD(
        optic, field, wavelength, distribution=forward, strategy=strategy
    ).get_data(field, wavelength)
    reverse_data = OPD(
        optic, field, wavelength, distribution=reverse, strategy=strategy
    ).get_data(field, wavelength)

    assert_allclose(forward_data.opd, be.flip(reverse_data.opd), atol=1e-9, rtol=0.0)
    assert_allclose(forward_data.radius, reverse_data.radius, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    ("field", "expected_direction"),
    [
        ((0.0, 100.0), (0, 1, -1)),
        ((0.0, -100.0), (0, -1, -1)),
        ((100.0, 0.0), (1, 0, -1)),
        ((-100.0, 0.0), (-1, 0, -1)),
    ],
)
def test_plane_wave_is_flat_for_reverse_propagation(
    set_test_backend: None,
    field: tuple[float, float],
    expected_direction: tuple[int, int, int],
) -> None:
    """The incident direction preserves the selected propagation hemisphere."""
    optic = collimated_planes(index=1.5, field=field)
    normalized_field = optic.fields.get_field_coords()[-1]
    chief = optic.trace_generic(
        *normalized_field,
        Px=0.0,
        Py=0.0,
        wavelength=optic.primary_wavelength,
        retain_launch=True,
    )
    launch = chief._launch_state
    direction = [
        float(be.to_numpy(component)[0]) for component in (launch.L, launch.M, launch.N)
    ]
    for actual, expected in zip(direction, expected_direction, strict=True):
        if expected == 0:
            assert abs(actual) < 1e-12
        else:
            assert actual * expected > 0
    assert max_abs_wavefront(optic) < ZERO_WAVEFRONT_TOLERANCE


@pytest.mark.parametrize(
    ("steps", "first_radius"),
    [([0, 1, 2, 3], be.inf), ([1, 2, 3], 20.0)],
)
def test_plane_wave_is_flat_for_a_forward_sequence(
    set_test_backend: None, steps: list[int], first_radius: float
) -> None:
    """Launch phase is independent of whether the object step is included."""
    optic = collimated_planes(
        index=1.5,
        field=(3.0, 4.0),
        vignette=(0.4, 0.1),
        first_radius=first_radius,
    )
    sequence = optic.add_sequence("forward", steps)

    assert max_abs_wavefront(sequence) < ZERO_WAVEFRONT_TOLERANCE


@pytest.mark.parametrize("strategy", ["chief_ray", "centroid", "best_fit"])
def test_sequence_uses_its_incident_medium_for_launch_phase(
    set_test_backend: None, strategy: str
) -> None:
    """A sequence may begin after a refractive transition it does not trace."""
    air = IdealMaterial(1.0)
    glass = IdealMaterial(1.5)
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf, material=air)
    optic.surfaces.add(index=1, thickness=10.0, material=glass)
    optic.surfaces.add(index=2, thickness=10.0, material=glass, is_stop=True)
    optic.surfaces.add(index=3, material=glass)
    optic.set_aperture("EPD", 4.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.fields.add(y=5.0)
    optic.wavelengths.add(0.55, is_primary=True)
    sequence = optic.add_sequence("inside_glass", [2, 3])

    assert max_abs_wavefront(sequence, strategy=strategy) < ZERO_WAVEFRONT_TOLERANCE


@pytest.mark.parametrize("wavefront_strategy", ["chief_ray", "centroid", "best_fit"])
def test_launch_phase_preserves_first_surface_phase(
    set_test_backend: None, wavefront_strategy: str
) -> None:
    """Surface phase remains present when a sequence omits the object step."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        surface_type="paraxial",
        f=25.0,
        thickness=10.0,
        is_stop=True,
    )
    optic.surfaces.add(index=2)
    optic.set_aperture("EPD", 4.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(0.55, is_primary=True)
    distribution = create_distribution("line_x")
    distribution.generate_points(7)
    with_object = optic.add_sequence("with_object", [0, 1, 2])
    without_object = optic.add_sequence("without_object", [1, 2])

    rays_with = with_object.trace(
        0.0, 0.0, 0.55, None, distribution, retain_launch=True
    )
    rays_without = without_object.trace(
        0.0, 0.0, 0.55, None, distribution, retain_launch=True
    )
    surface_opd = without_object.surfaces[0].opd
    assert float(be.to_numpy(be.max(surface_opd) - be.min(surface_opd))) > 1e-3
    assert_allclose(rays_without.opd, rays_with.opd)

    strategy = ConcreteReferenceStrategy(without_object, distribution)
    reference = strategy._generate_chief_launch((0.0, 0.0), 0.55)
    without_object.surfaces[0].x[0] = be.nan
    corrected = strategy._restore_launch_phase(
        rays_without, rays_without.opd, 0.55, reference
    )
    assert_allclose(corrected, rays_without.opd)

    with_data = OPD(
        with_object,
        (0.0, 0.0),
        0.55,
        num_rays=6,
        strategy=wavefront_strategy,
        afocal=True,
    ).get_data((0.0, 0.0), 0.55)
    without_data = OPD(
        without_object,
        (0.0, 0.0),
        0.55,
        num_rays=6,
        strategy=wavefront_strategy,
        afocal=True,
    ).get_data((0.0, 0.0), 0.55)
    assert_allclose(without_data.opd, with_data.opd, atol=1e-10, rtol=0.0)
    assert float(be.to_numpy(be.max(with_data.opd) - be.min(with_data.opd))) > 1.0


def test_propagated_optical_path_is_common_to_every_ray(
    set_test_backend: None,
) -> None:
    """Confirm propagation does not introduce pupil-dependent optical path."""
    optic = collimated_planes(index=1.5, field=(0.0, 5.0), vignette=(0.0, 0.25))
    rays = optic.trace(0.0, 1.0, 0.55, num_rays=32, distribution="line_y")
    opd = be.to_numpy(rays.opd)
    assert_allclose(opd, opd[0], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("num_rays", [4, 6, 12])
def test_flatness_is_independent_of_pupil_sampling(
    set_test_backend: None, num_rays: int
) -> None:
    optic = collimated_planes(index=1.5, field=(3.0, 4.0), vignette=(0.4, 0.1))
    assert max_abs_wavefront(optic, num_rays=num_rays) < ZERO_WAVEFRONT_TOLERANCE


def test_object_index_scales_launch_phase_numerically(set_test_backend: None) -> None:
    """The launch phase uses the object index at the analysis wavelength."""
    optic = collimated_planes(index=1.5)
    distribution = create_distribution("line_y")
    distribution.generate_points(5)
    strategy = ConcreteReferenceStrategy(optic, distribution)
    reference = strategy._generate_chief_launch((0.0, 1.0), 0.55)
    rays = optic.trace(0.0, 1.0, 0.55, None, distribution, retain_launch=True)
    raw_opd = be.copy(rays.opd)
    optic.object_surface.material_post.n = MagicMock(
        side_effect=lambda wavelength: be.array(1.0 if wavelength == 0.55 else 1.5)
    )

    phase_at_055 = (
        strategy._restore_launch_phase(rays, raw_opd, 0.55, reference) - raw_opd
    )
    phase_at_065 = (
        strategy._restore_launch_phase(rays, raw_opd, 0.65, reference) - raw_opd
    )

    assert not be.all(phase_at_055 == 0)
    assert_allclose(phase_at_065, 1.5 * phase_at_055)


def test_retained_launch_phase_preserves_torch_gradients(
    set_test_backend: None,
) -> None:
    """Copying launch coordinates keeps their autograd connection."""
    if be.get_backend() != "torch":
        pytest.skip("Gradient check requires the Torch backend.")

    optic = collimated_planes(index=1.5)
    distribution = create_distribution("line_x")
    distribution.generate_points(2)
    strategy = ConcreteReferenceStrategy(optic, distribution)
    x = be.array([0.0, 1.0])
    x.requires_grad_(True)
    zeros = be.zeros(2)
    rays = RealRays(x, zeros, zeros, be.ones(2), zeros, zeros, be.ones(2), 0.55)
    rays._capture_launch_state()
    reference = RealRays(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.55)
    reference._capture_launch_state()

    phase = strategy._restore_launch_phase(rays, rays.opd, 0.55, reference)
    be.sum(phase).backward()

    assert x.grad is not None
    assert_allclose(x.grad, be.array([1.5, 1.5]))


def test_non_angular_and_finite_object_fields_are_untouched(
    set_test_backend: None,
) -> None:
    """Only infinite-conjugate angular fields need launch-phase correction."""
    optic = collimated_planes(index=1.5)
    strategy = OPD(
        optic, (0.0, 1.0), "primary", num_rays=4, strategy="chief_ray", afocal=True
    ).strategy
    opd = be.ones(strategy.distribution.x.shape)
    rays = MagicMock(opd=opd)

    optic.fields.set_type("object_height")
    assert_allclose(strategy._restore_launch_phase(rays, opd, 0.55, rays), opd)

    optic.fields.set_type("angle")
    optic.surfaces.surfaces[0].geometry.cs.z = be.array(-100.0)
    assert_allclose(strategy._restore_launch_phase(rays, opd, 0.55, rays), opd)
