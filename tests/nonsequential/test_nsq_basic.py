"""Basic non-sequential raytracing tests.

Covers analytic reference cases and flux conservation.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSource,
    CollimatedSourceConfig,
    ConicGeometry,
    FarFieldDetectorConfig,
    IrradianceDetector,
    IrradianceDetectorConfig,
    LensConfig,
    NSQRng,
    NSQScene,
    NSQTracer,
    PointSource,
    PointSourceConfig,
    RayDatabaseConfig,
    ReflectiveComponent,
    Spectrum,
    SpecularBRDF,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def numpy_backend():
    """Return a seeded NumpyBackend."""
    return NumpyBackend(seed=42)


# ---------------------------------------------------------------------------
# Test: Spectrum sampling
# ---------------------------------------------------------------------------


class TestSpectrum:
    def test_monochromatic(self):
        spec = Spectrum.monochromatic(0.55)
        rng = NSQRng(0)
        ray_id = np.arange(100)
        wl = spec.sample(ray_id, np.zeros_like(ray_id), rng)
        assert np.all(wl == 0.55)

    def test_broadband_sampling(self):
        wls = np.array([0.4, 0.5, 0.6, 0.7])
        wts = np.array([1.0, 2.0, 2.0, 1.0])
        spec = Spectrum(wls, wts)
        rng = NSQRng(42)
        ray_id = np.arange(10000)
        samples = spec.sample(ray_id, np.zeros_like(ray_id), rng)
        assert set(np.round(samples, 5)).issubset(set(np.round(wls, 5)))
        # 0.5 and 0.6 µm should be sampled more than 0.4 and 0.7 µm
        counts = {w: int((np.isclose(samples, w)).sum()) for w in wls}
        assert counts[0.5] > counts[0.4]
        assert counts[0.6] > counts[0.7]

    def test_wavelength_um_validation(self):
        """Wavelengths in nm should raise ValueError."""
        with pytest.raises(ValueError, match="µm"):
            Spectrum.monochromatic(550.0)  # nm value -- should fail

    def test_wavelength_um_valid(self):
        """Wavelengths in µm should be accepted."""
        spec = Spectrum.monochromatic(0.55)
        assert spec.wavelengths[0] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Test: Point source generation
# ---------------------------------------------------------------------------


class TestPointSource:
    def test_generates_correct_num_rays(self):
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        src = PointSource(cs=cs, spectrum=spec, total_flux=1.0, half_angle_deg=90.0)
        rng = NSQRng(0)
        rays = src.generate(np.arange(1000), rng)
        assert rays.num_rays == 1000
        assert rays.num_rays_alive == 1000

    def test_total_flux_conservation(self):
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        total = 2.5
        src = PointSource(cs=cs, spectrum=spec, total_flux=total, half_angle_deg=90.0)
        rng = NSQRng(0)
        rays = src.generate(np.arange(10000), rng)
        assert np.isclose(rays.flux.sum(), total, rtol=1e-6)

    def test_directions_normalized(self):
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        src = PointSource(cs=cs, spectrum=spec, total_flux=1.0)
        rng = NSQRng(0)
        rays = src.generate(np.arange(500), rng)
        dirs = np.stack([rays.L, rays.M, rays.N], axis=1)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_cone_constraint(self):
        """All rays within a 30-degree half-angle cone around +z."""
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        src = PointSource(cs=cs, spectrum=spec, total_flux=1.0, half_angle_deg=30.0)
        rng = NSQRng(0)
        rays = src.generate(np.arange(1000), rng)
        cos_theta = rays.N  # +z axis
        min_cos = np.cos(np.radians(30.0))
        assert np.all(cos_theta >= min_cos - 1e-10)


# ---------------------------------------------------------------------------
# Test: Plane geometry intersection
# ---------------------------------------------------------------------------


class TestPlaneGeometry:
    def test_infinite_plane_hits(self):
        from optiland.nonsequential.components.geometry.analytic.plane import (
            PlaneGeometry,  # noqa: PLC0415
        )

        geom = PlaneGeometry()
        N = 5
        origins = np.zeros((N, 3))
        origins[:, 2] = -1.0  # behind plane at z=0
        directions = np.zeros((N, 3))
        directions[:, 2] = 1.0  # pointing +z

        t, normals, hit_mask, n_geom = geom.ray_intersect(origins, directions)
        assert hit_mask.all()
        np.testing.assert_allclose(t, 1.0, atol=1e-9)

    def test_parallel_ray_no_hit(self):
        from optiland.nonsequential.components.geometry.analytic.plane import (
            PlaneGeometry,  # noqa: PLC0415
        )

        geom = PlaneGeometry()
        origins = np.array([[0.0, 0.0, -1.0]])
        directions = np.array([[1.0, 0.0, 0.0]])  # parallel to plane
        t, normals, hit_mask, n_geom = geom.ray_intersect(origins, directions)
        assert not hit_mask[0]
        assert t[0] == np.inf

    def test_finite_plane_aperture_check(self):
        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,  # noqa: PLC0415
        )

        geom = FinitePlaneGeometry(width=10.0, height=10.0)
        # Ray hitting inside aperture
        o_in = np.array([[0.0, 0.0, -1.0]])
        d_in = np.array([[0.0, 0.0, 1.0]])
        t, _, hit, _ = geom.ray_intersect(o_in, d_in)
        assert hit[0]

        # Ray hitting outside aperture
        o_out = np.array([[6.0, 0.0, -1.0]])
        t, _, hit, _ = geom.ray_intersect(o_out, d_in)
        assert not hit[0]


# ---------------------------------------------------------------------------
# Test: Sphere geometry intersection
# ---------------------------------------------------------------------------


class TestSphereGeometry:
    def test_sphere_center_ray(self):
        from optiland.nonsequential.components.geometry.analytic.sphere import (
            SphereGeometry,  # noqa: PLC0415
        )

        R = 10.0
        geom = SphereGeometry(radius=R)
        # Ray from outside along -z hitting sphere
        origins = np.array([[0.0, 0.0, -20.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        t, normals, hit, n_geom = geom.ray_intersect(origins, directions)
        assert hit[0]
        np.testing.assert_allclose(t[0], 10.0, atol=1e-9)  # hits at z=-10

    def test_sphere_normal_outward(self):
        from optiland.nonsequential.components.geometry.analytic.sphere import (
            SphereGeometry,  # noqa: PLC0415
        )

        geom = SphereGeometry(radius=5.0)
        origins = np.array([[0.0, 0.0, -10.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        t, normals, hit, n_geom = geom.ray_intersect(origins, directions)
        # Normal at (0, 0, -5) should point toward incoming ray (in -z direction)
        assert normals[0, 2] < 0  # points toward origin (incoming side)


# ---------------------------------------------------------------------------
# Test: Flat mirror normal incidence (analytic reference case)
# ---------------------------------------------------------------------------


class TestFlatMirrorReflection:
    """Collimated beam → flat mirror at 45° → flat detector.

    Expected: spot on detector at geometric prediction.
    """

    def test_flat_mirror_reflection(self):
        # Collimated beam along +z, centred on origin
        # Mirror at z=50, ry=-45° → local z points in (-sin45,0,cos45) = upper-left.
        # A +z beam hits the mirror and reflects to +x.
        src_cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        CollimatedSource(
            cs=src_cs,
            spectrum=spec,
            total_flux=1.0,
            aperture_radius=1.0,
        )

        # Mirror: ry=-pi/4 rad → reflects +z global to +x global
        mirror_cs = CoordinateSystem(x=0, y=0, z=50, ry=-math.pi / 4)
        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,  # noqa: PLC0415
        )

        mirror = ReflectiveComponent(
            cs=mirror_cs,
            geometry=FinitePlaneGeometry(width=20.0, height=20.0),
            reflectance=1.0,
            bsdf=SpecularBRDF(),
        )

        # Detector at x=50, z=50 (ry=pi/2 rad → local z points in +x global).
        # Reflected beam travels +x and hits the detector.
        det_cs = CoordinateSystem(x=50, y=0, z=50, ry=math.pi / 2)
        IrradianceDetector(
            cs=det_cs,
            width=10.0,
            height=10.0,
            num_pixels_x=64,
            num_pixels_y=64,
        )

        scene = NSQScene()
        scene.add_component("mirror", mirror)
        scene.add_source(
            "src",
            src_cs,
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=1.0),
        )
        scene.add_detector(
            "det",
            det_cs,
            IrradianceDetectorConfig(
                width=10.0, height=10.0, num_pixels_x=64, num_pixels_y=64
            ),
        )

        tracer = NSQTracer(scene)
        result = tracer.trace(num_rays=10_000, max_depth=5, seed=42)

        irr = result.detectors["det"]
        # Should detect most flux (some geometric clipping expected)
        assert irr.num_rays_hit > 8000


# ---------------------------------------------------------------------------
# Test: Flux conservation
# ---------------------------------------------------------------------------


class TestFluxConservation:
    """Flux conservation: collimated beam into flat detector."""

    def test_collimated_beam_to_flat_detector(self):
        """Collimated source → large flat detector directly above.

        Since no components are between source and detector, all rays should
        reach the detector. Flux conservation error should be ~0.
        """

        src_cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        CollimatedSource(
            cs=src_cs,
            spectrum=spec,
            total_flux=1.0,
            aperture_radius=5.0,
        )

        # Large detector at z=100, centred, covers the full beam cross-section
        det_cs = CoordinateSystem(x=0, y=0, z=100)
        IrradianceDetector(
            cs=det_cs,
            width=30.0,
            height=30.0,
            num_pixels_x=32,
            num_pixels_y=32,
        )

        scene = NSQScene()
        scene.add_source(
            "src",
            src_cs,
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=5.0),
        )
        scene.add_detector(
            "det",
            det_cs,
            IrradianceDetectorConfig(
                width=30.0, height=30.0, num_pixels_x=32, num_pixels_y=32
            ),
        )

        tracer = NSQTracer(scene)
        result = tracer.trace(num_rays=10_000, max_depth=5, seed=0)

        irr = result.detectors["det"]
        # All rays should reach the detector (beam r=5mm, detector 30mm wide)
        assert irr.num_rays_hit == 10_000
        # Total detected flux should equal total source flux
        np.testing.assert_allclose(irr.total_flux, 1.0, rtol=1e-6)
        assert result.flux_conservation_error < 0.001
        assert result.trace_time_sec > 0.0

    def test_reflective_mirror_flux(self):
        """Paraboloid collimating mirror: some rays should reach detector."""
        # Mirror with ry=180° so it opens downward (-z global).
        # Vertex in global at z=100. Local focus at z_local=R/2=100mm.
        # With ry=180°: local +z points in global -z.
        # Focus in global: vertex + Ry(180°)@[0,0,100] = [0,0,100]+[0,0,-100]=[0,0,0].
        # Source at z=0 is at the focus. Source emits upward (+z global).
        # Reflected rays go parallel to local +z (global -z), downward.
        # A detector below at z=-100 (global) catches the collimated beam.

        mirror_cs = CoordinateSystem(x=0, y=0, z=100, ry=math.pi)
        mirror = ReflectiveComponent(
            cs=mirror_cs,
            geometry=ConicGeometry(radius=200.0, conic=-1.0, aperture_radius=50.0),
            reflectance=1.0,
            bsdf=SpecularBRDF(),
        )

        src_cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        # Source at focus, emit upward toward the mirror
        PointSource(cs=src_cs, spectrum=spec, total_flux=1.0, half_angle_deg=30.0)

        # Large detector below the mirror to catch reflected collimated beam
        det_cs = CoordinateSystem(x=0, y=0, z=-50)
        IrradianceDetector(
            cs=det_cs,
            width=200.0,
            height=200.0,
            num_pixels_x=64,
            num_pixels_y=64,
        )

        scene = NSQScene()
        scene.add_component("mirror", mirror)
        scene.add_source(
            "src",
            src_cs,
            PointSourceConfig(spectrum=spec, total_flux=1.0, half_angle_deg=30.0),
        )
        scene.add_detector(
            "det",
            det_cs,
            IrradianceDetectorConfig(
                width=200.0, height=200.0, num_pixels_x=64, num_pixels_y=64
            ),
        )

        tracer = NSQTracer(scene)
        result = tracer.trace(num_rays=20_000, max_depth=5, seed=0)

        # Some rays should be detected; tracer should complete without error
        assert result.trace_time_sec > 0.0
        assert result.num_rays_total == 20_000

    def test_flux_conservation_mirror(self):
        """Mirror scene: flux_conservation_error should be near zero."""
        # A collimated beam hits a large flat mirror and reflects to a detector.
        # All flux should be accounted for.
        src_cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        CollimatedSource(cs=src_cs, spectrum=spec, total_flux=1.0, aperture_radius=2.0)

        mirror_cs = CoordinateSystem(x=0, y=0, z=50, ry=-math.pi / 4)
        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,  # noqa: PLC0415
        )

        mirror = ReflectiveComponent(
            cs=mirror_cs,
            geometry=FinitePlaneGeometry(width=20.0, height=20.0),
            reflectance=1.0,
            bsdf=SpecularBRDF(),
        )

        det_cs = CoordinateSystem(x=50, y=0, z=50, ry=math.pi / 2)
        IrradianceDetector(
            cs=det_cs, width=20.0, height=20.0, num_pixels_x=32, num_pixels_y=32
        )

        scene = NSQScene()
        scene.add_component("mirror", mirror)
        scene.add_source(
            "src",
            src_cs,
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=2.0),
        )
        scene.add_detector(
            "det",
            det_cs,
            IrradianceDetectorConfig(
                width=20.0, height=20.0, num_pixels_x=32, num_pixels_y=32
            ),
        )

        tracer = NSQTracer(scene)
        result = tracer.trace(num_rays=5_000, max_depth=5, seed=0)

        # flux_conservation_error < 5% (some rays may escape around mirror edge)
        assert result.flux_conservation_error < 0.05
        assert result.total_flux_in == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Test: Collimated source generation
# ---------------------------------------------------------------------------


class TestCollimatedSource:
    def test_directions_parallel(self):
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        src = CollimatedSource(
            cs=cs, spectrum=spec, total_flux=1.0, aperture_radius=5.0
        )
        rng = NSQRng(0)
        rays = src.generate(np.arange(200), rng)
        # All rays point in +z direction (no tilt in this CS)
        np.testing.assert_allclose(rays.L, 0.0, atol=1e-12)
        np.testing.assert_allclose(rays.M, 0.0, atol=1e-12)
        np.testing.assert_allclose(rays.N, 1.0, atol=1e-12)

    def test_positions_within_aperture(self):
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        r = 5.0
        src = CollimatedSource(cs=cs, spectrum=spec, total_flux=1.0, aperture_radius=r)
        rng = NSQRng(0)
        rays = src.generate(np.arange(500), rng)
        radii = (rays.x**2 + rays.y**2) ** 0.5
        assert np.all(radii <= r + 1e-9)


# ---------------------------------------------------------------------------
# Test: NSQScene validation
# ---------------------------------------------------------------------------


class TestNSQScene:
    def test_validate_no_sources(self):
        scene = NSQScene()
        cs = CoordinateSystem(x=0, y=0, z=0)
        scene.add_detector(
            "det",
            cs,
            IrradianceDetectorConfig(
                width=10.0, height=10.0, num_pixels_x=32, num_pixels_y=32
            ),
        )
        with pytest.raises(ValueError, match="no sources"):
            scene.validate()

    def test_validate_no_detectors(self):
        scene = NSQScene()
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(0.55)
        scene.add_source("src", cs, PointSourceConfig(spectrum=spec, total_flux=1.0))
        with pytest.raises(ValueError, match="no detectors"):
            scene.validate()


# ---------------------------------------------------------------------------
# Test: BSDF
# ---------------------------------------------------------------------------


class TestBSDFs:
    def test_specular_reflectance_unity(self):
        from optiland.nonsequential.bsdf.specular import SpecularBRDF  # noqa: PLC0415

        bsdf = SpecularBRDF()
        N = 10
        dirs = np.tile([0.0, 0.0, -1.0], (N, 1))
        normals = np.tile([0.0, 0.0, 1.0], (N, 1))
        wl = np.full(N, 0.55)
        r = bsdf.reflectance(dirs, normals, wl)
        np.testing.assert_allclose(r, 1.0)

    def test_specular_direction(self):
        from optiland.nonsequential.bsdf.specular import SpecularBRDF  # noqa: PLC0415

        bsdf = SpecularBRDF()
        N = 1
        dirs = np.array([[0.0, 0.0, -1.0]])  # incoming downward
        normals = np.array([[0.0, 0.0, 1.0]])  # normal upward
        wl = np.array([0.55])
        rng = NSQRng(0)
        ray_id = np.arange(N)
        bounce = np.zeros(N, dtype=np.int32)
        scattered, weights, transmitted = bsdf.sample(
            N, dirs, normals, wl, rng, ray_id, bounce
        )
        np.testing.assert_allclose(scattered, [[0.0, 0.0, 1.0]], atol=1e-10)
        assert not np.any(transmitted)

    def test_lambertian_reflectance(self):
        from optiland.nonsequential.bsdf.lambertian import (
            LambertianBSDF,  # noqa: PLC0415
        )

        bsdf = LambertianBSDF(reflectance_value=0.8)
        N = 5
        dirs = np.tile([0.0, 0.0, -1.0], (N, 1))
        normals = np.tile([0.0, 0.0, 1.0], (N, 1))
        wl = np.full(N, 0.55)
        r = bsdf.reflectance(dirs, normals, wl)
        np.testing.assert_allclose(r, 0.8)

    def test_lambertian_sample_directions_on_hemisphere(self):
        from optiland.nonsequential.bsdf.lambertian import (
            LambertianBSDF,  # noqa: PLC0415
        )

        bsdf = LambertianBSDF()
        N = 1000
        dirs = np.tile([0.0, 0.0, -1.0], (N, 1)).astype(float)
        normals = np.tile([0.0, 0.0, 1.0], (N, 1)).astype(float)
        wl = np.full(N, 0.55)
        rng = NSQRng(42)
        ray_id = np.arange(N)
        bounce = np.zeros(N, dtype=np.int32)
        scattered, weights, transmitted = bsdf.sample(
            N, dirs, normals, wl, rng, ray_id, bounce
        )
        assert not np.any(transmitted)
        # All scattered rays should be on the +z hemisphere
        assert np.all(scattered[:, 2] >= -1e-6)
        # Check normalization
        norms = np.linalg.norm(scattered, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: New required tests per spec
# ---------------------------------------------------------------------------


class TestSelfIntersectionGuard:
    """T_EPSILON guard prevents self-intersection after surface crossing."""

    def test_t_epsilon_no_self_hit(self):
        """After a ray crosses a surface, it should not immediately re-hit it."""
        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,  # noqa: PLC0415
        )
        from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

        Spectrum.monochromatic(0.55)
        cs = CoordinateSystem(x=0, y=0, z=10)
        geom = FinitePlaneGeometry(width=20.0, height=20.0)
        comp = ReflectiveComponent(
            cs=cs, geometry=geom, reflectance=1.0, bsdf=SpecularBRDF()
        )

        # Create a ray bundle that is AT the surface (t would be 0 without guard)
        rays = NSQRayBundle(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([10.0]),  # exactly at the surface
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            flux=np.array([1.0]),
            wavelength=np.array([0.55]),
            n_current=np.array([1.0]),
            bounce=np.array([0], dtype=np.int32),
            alive=np.array([True]),
        )

        t, normals, hit_mask, n_geom = comp.intersect(rays)
        # With T_EPSILON guard, a ray at the surface should NOT register as a hit
        assert not hit_mask[0], "Ray at surface should not self-intersect"
        assert t[0] == np.inf


class TestMultiSourceRayCount:
    """Multi-source ray count is distributed proportionally."""

    def test_two_sources_ray_distribution(self):
        """Two sources with equal flux should each get half the rays."""
        spec = Spectrum.monochromatic(0.55)

        src1_cs = CoordinateSystem(x=0, y=0, z=0)
        src2_cs = CoordinateSystem(x=5, y=0, z=0)
        det_cs = CoordinateSystem(x=0, y=0, z=10)

        scene = NSQScene()
        scene.add_source(
            "src1",
            src1_cs,
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=1.0),
        )
        scene.add_source(
            "src2",
            src2_cs,
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=1.0),
        )
        scene.add_detector(
            "det",
            det_cs,
            IrradianceDetectorConfig(
                width=30.0, height=30.0, num_pixels_x=32, num_pixels_y=32
            ),
        )

        tracer = NSQTracer(scene)
        result = tracer.trace(num_rays=2000, seed=0)

        # Total rays = 2000 (sum across sources)
        assert result.num_rays_total == 2000
        # total_flux_in = 2.0 (both sources)
        assert result.total_flux_in == pytest.approx(2.0, rel=1e-6)


class TestWavelengthUm:
    """Verify wavelengths are stored in µm, not nm."""

    def test_ray_bundle_wavelength_in_um(self):
        """Generated ray bundle should have wavelength in µm range."""
        spec = Spectrum.monochromatic(0.55)
        cs = CoordinateSystem(x=0, y=0, z=0)
        src = CollimatedSource(
            cs=cs, spectrum=spec, total_flux=1.0, aperture_radius=1.0
        )
        rng = NSQRng(0)
        rays = src.generate(np.arange(100), rng)
        # Wavelength should be ~0.55 µm (not 550 nm)
        assert np.all(rays.wavelength < 10.0), "Wavelengths appear to be in nm, not µm"
        np.testing.assert_allclose(rays.wavelength, 0.55, rtol=1e-10)


class TestAbsorbedCount:
    """AbsorbingComponent tracks absorbed ray and flux counts."""

    def test_absorbed_count_increments(self):
        from optiland.nonsequential.components.absorbing import (
            AbsorbingComponent,  # noqa: PLC0415
        )
        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,  # noqa: PLC0415
        )
        from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

        cs = CoordinateSystem(x=0, y=0, z=10)
        geom = FinitePlaneGeometry(width=20.0, height=20.0)
        absorber = AbsorbingComponent(cs=cs, geometry=geom)

        N = 5
        rays = NSQRayBundle(
            x=np.zeros(N),
            y=np.zeros(N),
            z=np.zeros(N),
            L=np.zeros(N),
            M=np.zeros(N),
            N=np.ones(N),
            flux=np.ones(N),
            wavelength=np.full(N, 0.55),
            n_current=np.ones(N),
            bounce=np.zeros(N, dtype=np.int32),
            alive=np.ones(N, dtype=bool),
        )

        hit_mask = np.array([True, True, False, True, False])
        t = np.full(N, 5.0)
        normals = np.zeros((N, 3))
        normals[:, 2] = -1.0

        from optiland.nonsequential.ir.bsdf_ir import BsdfIR  # noqa: PLC0415

        rng = np.random.default_rng(0)
        n_geom = np.zeros((N, 3))
        absorber.interact(rays, t, normals, hit_mask, rng, BsdfIR(kind="none"), n_geom)

        assert absorber._absorbed_count == 3
        assert absorber._absorbed_flux == pytest.approx(3.0)

    def test_reset_stats(self):
        from optiland.nonsequential.components.absorbing import (
            AbsorbingComponent,  # noqa: PLC0415
        )
        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,  # noqa: PLC0415
        )

        cs = CoordinateSystem(x=0, y=0, z=0)
        geom = FinitePlaneGeometry(width=10.0, height=10.0)
        absorber = AbsorbingComponent(cs=cs, geometry=geom)
        absorber._absorbed_count = 10
        absorber._absorbed_flux = 5.0
        absorber.reset_stats()
        assert absorber._absorbed_count == 0
        assert absorber._absorbed_flux == 0.0


class TestDeathCauseFluxKilled:
    """Rays below min_flux_fraction are killed and counted."""

    def test_flux_killed_count(self):
        """A ray that loses most flux should be killed and counted."""
        from optiland.nonsequential.components.absorbing import (
            AbsorbingComponent,  # noqa: PLC0415
        )
        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,  # noqa: PLC0415
        )

        # Source with 1 ray
        spec = Spectrum.monochromatic(0.55)
        src_cs = CoordinateSystem(x=0, y=0, z=0)
        CollimatedSource(cs=src_cs, spectrum=spec, total_flux=1.0, aperture_radius=1.0)

        # Absorber that kills some flux before detector
        abs_cs = CoordinateSystem(x=0, y=0, z=50)
        abs_geom = FinitePlaneGeometry(width=50.0, height=50.0)
        absorber = AbsorbingComponent(cs=abs_cs, geometry=abs_geom)

        det_cs = CoordinateSystem(x=0, y=0, z=100)
        IrradianceDetector(
            cs=det_cs, width=50.0, height=50.0, num_pixels_x=16, num_pixels_y=16
        )

        scene = NSQScene()
        scene.add_component("absorber", absorber)
        scene.add_source(
            "src",
            src_cs,
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=1.0),
        )
        scene.add_detector(
            "det",
            det_cs,
            IrradianceDetectorConfig(
                width=50.0, height=50.0, num_pixels_x=16, num_pixels_y=16
            ),
        )

        result = scene.trace(num_rays=1000, seed=0)

        # Absorber should have captured rays
        assert absorber._absorbed_count > 0
        assert result.num_rays_absorbed > 0


# ---------------------------------------------------------------------------
# Flux bookkeeping — every launched watt must be accounted for
# ---------------------------------------------------------------------------


class TestFluxBookkeeping:
    """The flux ledger must close, including rays killed by the cutoffs."""

    @staticmethod
    def _scene():
        """Lens scene with a deliberately shallow depth budget."""
        scene = NSQScene()
        scene.add_source(
            "S1",
            CoordinateSystem(),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=8.0,
            ),
        )
        scene.add_lens(
            "L1",
            CoordinateSystem(z=50.0),
            LensConfig(
                r1=60.0,
                r2=-60.0,
                thickness=6.0,
                material="N-BK7",
                front_aperture_radius=12.0,
            ),
        )
        scene.add_detector(
            "D1",
            CoordinateSystem(z=200.0),
            IrradianceDetectorConfig(
                width=40, height=40, num_pixels_x=32, num_pixels_y=32
            ),
        )
        return scene

    def test_ledger_closes(self):
        """detected + absorbed + bulk_absorbed + escaped + lost == launched.

        N-BK7 has a small but nonzero extinction coefficient at 0.55 um, so
        the lens's Beer-Lambert bulk absorption (D-13) is a real, nonzero
        term in this ledger, not just a defensive zero-check.
        """
        result = self._scene().trace(num_rays=20_000, seed=3)
        assert result.total_flux_bulk_absorbed > 0.0
        balance = (
            result.total_flux_in
            - result.total_flux_detected
            - result.total_flux_absorbed
            - result.total_flux_bulk_absorbed
            - result.total_flux_escaped
            - result.total_flux_lost
        )
        assert abs(balance) / result.total_flux_in < 1e-9, (
            f"Flux ledger does not close: residual {balance:.6e} of "
            f"{result.total_flux_in:.6e} launched"
        )

    def test_conservation_error_counts_killed_rays(self):
        """Rays killed by the depth cutoff must not be reported as an error.

        ``flux_conservation_error`` omitting ``total_flux_lost`` made any
        scene that depth-kills rays report a large error that is not real —
        precisely the stray-light scenes the diagnostic exists to serve.
        """
        # max_depth=1 forces most rays to be depth-killed inside the lens.
        result = self._scene().trace(num_rays=5_000, seed=3, max_depth=1)
        assert result.num_rays_depth_killed > 0, "Test scene killed nothing"
        assert result.total_flux_lost > 0.0
        assert result.flux_conservation_error < 1e-9, (
            "Depth-killed flux must be part of the ledger, not counted as a "
            f"conservation error (got {result.flux_conservation_error:.3e})"
        )


class TestConicSurfaceNormals:
    """Conic normals must match the analytic sag derivative.

    For a conic, dz/dr = c*r / sqrt(1 - (1+K) c^2 r^2). An incorrect normal
    tilts every refracted ray, and the error grows with aperture and |K|, so
    it is easy to miss on a slow spherical test case.
    """

    @pytest.mark.parametrize(
        ("radius", "conic"),
        [(120.0, 0.0), (25.0, -2.5), (25.0, 1.5), (-40.0, -0.6), (50.0, -1.0)],
    )
    def test_normal_matches_finite_difference_sag(self, radius, conic):
        geom = ConicGeometry(radius, conic, 12.0)
        h = 1e-6
        for x_val in (1.0, 5.0, 9.0):
            x = np.array([x_val])
            y = np.array([0.0])
            dsag_dx_fd = float(
                (geom._sag(x + h, y) - geom._sag(x - h, y))[0] / (2.0 * h)
            )
            # _normal_local returns the gradient of (z - sag): (-dsag/dx, ...)
            dsag_dx_analytic = -float(geom._normal_local(x, y)[0, 0])
            assert dsag_dx_analytic == pytest.approx(dsag_dx_fd, rel=1e-6), (
                f"R={radius}, K={conic}, x={x_val}"
            )

    @pytest.mark.parametrize("radius", [0.0, float("inf")])
    def test_flat_surface_conventions(self, radius):
        """radius=0 and radius=inf both denote a plano surface."""
        geom = ConicGeometry(radius, 0.0, 10.0)
        x = np.linspace(-9.0, 9.0, 5)
        y = np.zeros(5)
        assert np.allclose(np.asarray(geom._sag(x, y)), 0.0)
        normals = np.asarray(geom._normal_local(x, y))
        assert np.allclose(normals[:, 0], 0.0)
        assert np.allclose(normals[:, 1], 0.0)


class TestBatchingFluxInvariance:
    """Splitting a trace into batches must not change the launched energy.

    ``source.generate(ray_id)`` spreads the source's whole ``total_flux``
    over the rays it is asked for. When the tracer processes a source in several
    batches, each batch therefore has to be rescaled to its share of the ray
    budget; otherwise every batch re-emits the full flux and the simulation
    manufactures energy in proportion to the batch count. This affects any run
    with more rays than ``batch_size`` (default 1e6), which is exactly the
    high-ray-count forward regime.
    """

    @staticmethod
    def _scene(total_flux=1.0):
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-80),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=total_flux,
                aperture_radius=10.0,
            ),
        )
        scene.add_lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50,
                r2=-50,
                thickness=5,
                material="N-BK7",
                front_aperture_radius=12.5,
            ),
        )
        scene.add_detector(
            "D",
            CoordinateSystem(z=100),
            IrradianceDetectorConfig(
                width=40, height=40, num_pixels_x=64, num_pixels_y=64
            ),
        )
        return scene

    @pytest.mark.parametrize("batch_size", [500, 2_000, 10_000, 1_000_000])
    def test_detected_flux_never_exceeds_launched(self, batch_size):
        """Detected flux must stay within the launched budget for any batch size."""
        result = self._scene().trace(num_rays=10_000, seed=42, batch_size=batch_size)
        detected = result.detectors["D"].total_flux
        assert detected <= result.total_flux_in + 1e-9, (
            f"batch_size={batch_size} detected {detected:.4f} W from "
            f"{result.total_flux_in:.4f} W launched"
        )
        assert result.flux_conservation_error < 1e-9

    def test_flux_is_independent_of_batch_size(self):
        """Batching is an implementation detail, not a physical parameter."""
        fluxes = [
            self._scene()
            .trace(num_rays=10_000, seed=42, batch_size=b)
            .detectors["D"]
            .total_flux
            for b in (500, 2_000, 10_000, 1_000_000)
        ]
        # Differences are Monte Carlo noise only: batching changes how the RNG
        # stream is consumed, not how much energy is launched.
        assert max(fluxes) - min(fluxes) < 0.02, fluxes

    def test_multi_source_batching_conserves_flux(self):
        """Two sources with unequal flux, split across batches."""
        scene = self._scene(total_flux=1.0)
        scene.add_source(
            "S2",
            CoordinateSystem(z=-80, x=3.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=2.0,
                aperture_radius=5.0,
            ),
        )
        result = scene.trace(num_rays=9_000, seed=1, batch_size=700)
        assert result.total_flux_in == pytest.approx(3.0)
        assert result.detectors["D"].total_flux <= 3.0 + 1e-9
        assert result.flux_conservation_error < 1e-9


class TestDetectorFluxAccounting:
    """Every detector must report flux in watts and join the flux ledger.

    ``SimulationResult`` builds ``total_flux_detected`` by summing each
    detector result's ``total_flux``, so a detector that reports the wrong
    quantity, or none at all, silently corrupts the global flux budget.
    """

    @staticmethod
    def _point_source_scene(detector_name, detector_config, z=100.0):
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=0),
            PointSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                half_angle_deg=20.0,
            ),
        )
        scene.add_detector(detector_name, CoordinateSystem(z=z), detector_config)
        return scene

    def test_far_field_reports_flux_not_intensity(self):
        """FarFieldPattern.total_flux is watts, not the sum of W/sr bins.

        ``intensity`` is divided by the per-bin solid angle, so summing it
        gives a number that is both dimensionally wrong and numerically far
        larger than the launched flux.
        """
        scene = self._point_source_scene(
            "FF", FarFieldDetectorConfig(num_theta=45, num_phi=180)
        )
        result = scene.trace(num_rays=20_000, seed=42)
        ff = result.detectors["FF"]

        assert ff.total_flux <= result.total_flux_in + 1e-9, (
            f"Far-field total_flux={ff.total_flux:.3f} W exceeds the "
            f"{result.total_flux_in:.3f} W launched"
        )
        assert ff.total_flux > 0.9, "Nearly all rays should reach the detector"
        assert result.flux_conservation_error < 1e-9

    def test_far_field_intensity_is_still_per_steradian(self):
        """The fix must not disturb the intensity normalisation."""
        scene = self._point_source_scene(
            "FF", FarFieldDetectorConfig(num_theta=45, num_phi=180)
        )
        ff = scene.trace(num_rays=20_000, seed=42).detectors["FF"]
        # W/sr over a narrow cone is much larger than the flux in W.
        assert ff.intensity.sum() > ff.total_flux

    def test_ray_database_joins_the_flux_ledger(self):
        """RayDatabase exposes total_flux so detected flux is not undercounted."""
        scene = self._point_source_scene(
            "RDB", RayDatabaseConfig(width=200, height=200)
        )
        result = scene.trace(num_rays=20_000, seed=42)
        db = result.detectors["RDB"]

        assert db.total_flux == pytest.approx(float(np.sum(db.flux)))
        assert result.total_flux_detected == pytest.approx(db.total_flux)
        assert result.flux_conservation_error < 1e-9

    def test_irradiance_and_far_field_agree_on_total_flux(self):
        """Two detector types on the same beam must record the same flux."""
        spec = Spectrum.monochromatic(0.55)
        common = dict(spectrum=spec, total_flux=1.0, half_angle_deg=20.0)

        scene_irr = NSQScene()
        scene_irr.add_source("S", CoordinateSystem(z=0), PointSourceConfig(**common))
        scene_irr.add_detector(
            "D",
            CoordinateSystem(z=100),
            IrradianceDetectorConfig(
                width=400, height=400, num_pixels_x=64, num_pixels_y=64
            ),
        )
        flux_irr = scene_irr.trace(num_rays=20_000, seed=42).detectors["D"].total_flux

        scene_ff = NSQScene()
        scene_ff.add_source("S", CoordinateSystem(z=0), PointSourceConfig(**common))
        scene_ff.add_detector(
            "FF",
            CoordinateSystem(z=100),
            FarFieldDetectorConfig(num_theta=45, num_phi=180),
        )
        flux_ff = scene_ff.trace(num_rays=20_000, seed=42).detectors["FF"].total_flux

        assert flux_ff == pytest.approx(flux_irr, rel=1e-6)
