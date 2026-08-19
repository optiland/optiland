"""Comprehensive tests for the extended sources feature.

This module tests:
  - SMFSource ray generation, parameters, and edge cases
  - BaseSource interface contract
  - ExtendedSourceOptic delegation, tracing, drawing, and restrictions
  - Source visualization (SourceViewer)
  - Integration with IncoherentIrradiance analysis
"""

from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.optic import ExtendedSourceOptic, Optic
from optiland.physical_apertures import RectangularAperture
from optiland.rays import RealRays
from optiland.sources import BaseSource, SMFSource

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _simple_singlet():
    """Create a minimal singlet for testing."""
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        thickness=7,
        radius=43.7354,
        is_stop=True,
        material="N-SF11",
    )
    lens.surfaces.add(index=2, radius=-46.2795, thickness=50)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=25)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


def _simple_singlet_with_detector():
    """Create a singlet that has a rectangular aperture on the image surface."""
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        thickness=7,
        radius=43.7354,
        is_stop=True,
        material="N-SF11",
    )
    lens.surfaces.add(index=2, radius=-46.2795, thickness=50)
    lens.surfaces.add(
        index=3,
        aperture=RectangularAperture(x_min=-15, x_max=15, y_min=-15, y_max=15),
    )
    lens.set_aperture(aperture_type="EPD", value=25)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


def _default_source():
    """Create a default SMFSource for testing."""
    return SMFSource(
        mfd_um=10.4,
        wavelength_um=0.55,
        total_power=1.0,
    )


# ===========================================================================
# SMFSource Tests
# ===========================================================================


class TestSMFSource:
    """Tests for the Single-Mode Fiber source."""

    @pytest.fixture(autouse=True)
    def _backend(self, set_test_backend):
        pass

    # --- Initialization & parameter computation ---

    def test_default_divergence_calculation(self):
        """Divergence should be computed from the Gaussian beam formula."""
        src = SMFSource(mfd_um=10.4, wavelength_um=1.55)
        w0 = 10.4 / 2.0
        expected_half_rad = 1.55 / (math.pi * w0)
        expected_deg = 2 * math.degrees(expected_half_rad)
        assert abs(src.divergence_deg_1e2 - expected_deg) < 1e-10

    def test_explicit_divergence_overrides(self):
        """When divergence is explicitly provided it should be used as-is."""
        src = SMFSource(mfd_um=10.4, wavelength_um=1.55, divergence_deg_1e2=20.0)
        assert src.divergence_deg_1e2 == 20.0

    def test_sigma_spatial_computation(self):
        """sigma_spatial_mm should equal (w0_um * 1e-3) / 2."""
        src = SMFSource(mfd_um=10.0, wavelength_um=0.55)
        expected = (10.0 / 2.0 * 1e-3) / 2.0  # 0.0025 mm
        assert abs(src.sigma_spatial_mm - expected) < 1e-15

    def test_sigma_angular_computation(self):
        """sigma_angular_rad should equal half_angle / 2."""
        div_deg = 5.0
        src = SMFSource(mfd_um=10.0, wavelength_um=0.55, divergence_deg_1e2=div_deg)
        expected = math.radians(div_deg / 2.0) / 2.0
        assert abs(src.sigma_angular_rad - expected) < 1e-15

    def test_attributes_stored(self):
        """All constructor args should be stored as attributes."""
        src = SMFSource(
            mfd_um=10.4,
            wavelength_um=1.31,
            divergence_deg_1e2=12.0,
            total_power=0.5,
            position=(1.0, 2.0, 3.0),
            is_point_source=True,
        )
        assert src.mfd_um == 10.4
        assert src.wavelength == 1.31
        assert src.divergence_deg_1e2 == 12.0
        assert src.total_power == 0.5
        assert src.is_point_source is True
        assert float(src.cs.x) == pytest.approx(1.0)
        assert float(src.cs.y) == pytest.approx(2.0)
        assert float(src.cs.z) == pytest.approx(3.0)

    # --- Ray generation ---

    def test_generate_rays_returns_real_rays(self):
        """generate_rays should return a RealRays object."""
        src = _default_source()
        rays = src.generate_rays(100)
        assert isinstance(rays, RealRays)

    def test_generate_rays_power_of_two_rounding(self):
        """Sobol requires power-of-two; 100 → 128."""
        src = _default_source()
        rays = src.generate_rays(100)
        n = be.size(rays.x)
        assert n == 128

    def test_generate_rays_exact_power_of_two(self):
        """If already power of two, no rounding needed."""
        src = _default_source()
        rays = src.generate_rays(64)
        assert be.size(rays.x) == 64

    def test_generate_rays_one_ray(self):
        """Requesting 1 ray should produce exactly 1."""
        src = _default_source()
        rays = src.generate_rays(1)
        assert be.size(rays.x) == 1

    def test_generate_rays_invalid_zero(self):
        """Zero rays should raise ValueError."""
        src = _default_source()
        with pytest.raises(ValueError, match="positive"):
            src.generate_rays(0)

    def test_generate_rays_invalid_negative(self):
        """Negative number of rays should raise ValueError."""
        src = _default_source()
        with pytest.raises(ValueError, match="positive"):
            src.generate_rays(-5)

    def test_ray_direction_cosine_normalization(self):
        """L² + M² + N² should equal 1 for every ray."""
        src = _default_source()
        rays = src.generate_rays(64)
        norm_sq = rays.L**2 + rays.M**2 + rays.N**2
        np.testing.assert_allclose(be.to_numpy(norm_sq), 1.0, atol=1e-12)

    def test_ray_forward_propagation(self):
        """All N direction cosines should be positive (forward propagation)."""
        src = _default_source()
        rays = src.generate_rays(64)
        assert be.all(rays.N > 0)

    def test_ray_wavelength_uniform(self):
        """All rays should carry the source wavelength."""
        src = SMFSource(mfd_um=10.0, wavelength_um=1.31)
        rays = src.generate_rays(16)
        np.testing.assert_allclose(be.to_numpy(rays.w), 1.31, atol=1e-15)

    def test_total_power_conserved(self):
        """Sum of ray intensities should equal total_power."""
        power = 2.5
        src = SMFSource(mfd_um=10.0, wavelength_um=0.55, total_power=power)
        rays = src.generate_rays(64)
        total = float(be.sum(rays.i))
        assert total == pytest.approx(power, rel=1e-10)

    def test_point_source_zero_spatial_extent(self):
        """In point-source mode all x, y should be zero."""
        src = SMFSource(mfd_um=10.0, wavelength_um=0.55, is_point_source=True)
        rays = src.generate_rays(32)
        np.testing.assert_allclose(be.to_numpy(rays.x), 0.0, atol=1e-15)
        np.testing.assert_allclose(be.to_numpy(rays.y), 0.0, atol=1e-15)

    def test_extended_source_nonzero_spatial_extent(self):
        """In extended mode, at least some x, y should be non-zero."""
        src = SMFSource(mfd_um=10.0, wavelength_um=0.55, is_point_source=False)
        rays = src.generate_rays(64)
        assert float(be.max(be.abs(rays.x))) > 0
        assert float(be.max(be.abs(rays.y))) > 0

    def test_position_offset_applied(self):
        """Rays from an offset source should have shifted coordinates."""
        src_origin = SMFSource(mfd_um=10.0, wavelength_um=0.55, is_point_source=True)
        src_offset = SMFSource(
            mfd_um=10.0,
            wavelength_um=0.55,
            is_point_source=True,
            position=(1.0, 2.0, 3.0),
        )
        rays_origin = src_origin.generate_rays(8)
        rays_offset = src_offset.generate_rays(8)
        # Point source at origin should have x≈0, at offset x≈1.0
        assert float(be.mean(rays_origin.x)) == pytest.approx(0.0, abs=1e-12)
        assert float(be.mean(rays_offset.x)) == pytest.approx(1.0, abs=1e-6)

    # --- Representation ---

    def test_repr_contains_key_info(self):
        """repr should contain source type and key parameters."""
        src = SMFSource(mfd_um=10.4, wavelength_um=1.55, total_power=0.5)
        r = repr(src)
        assert "SMFSource" in r
        assert "mfd=10.4" in r
        assert "wavelength=1.55" in r
        assert "power=0.5" in r
        assert "extended" in r

    def test_repr_point_mode(self):
        """repr should say 'point' when is_point_source=True."""
        src = SMFSource(mfd_um=10.0, wavelength_um=0.55, is_point_source=True)
        assert "point" in repr(src)

    # --- Visualization ---

    def test_draw_returns_figure_and_axes(self):
        """Source.draw() should return (Figure, list[Axes])."""
        src = _default_source()
        fig, axes = src.draw(num_rays=32, figsize=(10, 5))
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(axes, list)
        assert len(axes) == 6  # 2×3 grid
        plt.close(fig)


# ===========================================================================
# BaseSource (interface contract) Tests
# ===========================================================================


class TestBaseSourceInterface:
    """Verify the BaseSource abstract interface."""

    @pytest.fixture(autouse=True)
    def _backend(self, set_test_backend):
        pass

    def test_cannot_instantiate_directly(self):
        """BaseSource is abstract and should not be instantiable."""
        with pytest.raises(TypeError):
            BaseSource()

    def test_subclass_must_implement_generate_rays(self):
        """Missing generate_rays should prevent instantiation."""

        class BadSource(BaseSource):
            def __repr__(self):
                return "BadSource"

        with pytest.raises(TypeError):
            BadSource()

    def test_subclass_must_implement_repr(self):
        """Missing __repr__ should prevent instantiation."""

        class BadSource(BaseSource):
            def generate_rays(self, num_rays):
                pass  # pragma: no cover

        with pytest.raises(TypeError):
            BadSource()

    def test_concrete_subclass_works(self):
        """A properly implemented subclass should be instantiable."""

        class DummySource(BaseSource):
            def generate_rays(self, num_rays):
                return RealRays(
                    x=be.zeros(num_rays),
                    y=be.zeros(num_rays),
                    z=be.zeros(num_rays),
                    L=be.zeros(num_rays),
                    M=be.zeros(num_rays),
                    N=be.ones(num_rays),
                    intensity=be.ones(num_rays),
                    wavelength=be.full((num_rays,), 0.55),
                )

            def __repr__(self):
                return "DummySource()"

        src = DummySource(position=(1.0, 2.0, 3.0))
        assert isinstance(src.cs, CoordinateSystem)
        rays = src.generate_rays(4)
        assert be.size(rays.x) == 4

    def test_default_position_is_origin(self):
        """Default position should be (0, 0, 0)."""

        class DummySource(BaseSource):
            def generate_rays(self, num_rays):
                pass  # pragma: no cover

            def __repr__(self):
                return "DummySource()"

        src = DummySource()
        assert float(src.cs.x) == 0.0
        assert float(src.cs.y) == 0.0
        assert float(src.cs.z) == 0.0


# ===========================================================================
# ExtendedSourceOptic Tests
# ===========================================================================


class TestExtendedSourceOptic:
    """Tests for the ExtendedSourceOptic wrapper."""

    @pytest.fixture(autouse=True)
    def _backend(self, set_test_backend):
        pass

    @pytest.fixture()
    def optic(self):
        return _simple_singlet()

    @pytest.fixture()
    def source(self):
        return _default_source()

    @pytest.fixture()
    def ext_optic(self, optic, source):
        return ExtendedSourceOptic(optic, source)

    # --- Initialization ---

    def test_stores_optic_and_source(self, optic, source, ext_optic):
        assert ext_optic.optic is optic
        assert ext_optic.source is source

    # --- Attribute delegation (getattr) ---

    def test_delegates_name(self, optic, ext_optic):
        assert ext_optic.name == optic.name

    def test_delegates_surface_group(self, optic, ext_optic):
        assert ext_optic.surfaces is optic.surfaces

    def test_delegates_fields(self, optic, ext_optic):
        assert ext_optic.fields is optic.fields

    def test_delegates_wavelengths(self, optic, ext_optic):
        assert ext_optic.wavelengths is optic.wavelengths

    def test_delegates_paraxial(self, optic, ext_optic):
        assert ext_optic.paraxial is optic.paraxial

    def test_delegates_aberrations(self, optic, ext_optic):
        assert ext_optic.aberrations is optic.aberrations

    def test_delegates_aperture(self, optic, ext_optic):
        assert ext_optic.aperture is optic.aperture

    def test_delegates_polarization(self, optic, ext_optic):
        assert ext_optic.polarization == optic.polarization

    def test_nonexistent_attr_raises(self, ext_optic):
        with pytest.raises(AttributeError):
            _ = ext_optic.this_attr_does_not_exist

    # --- Attribute delegation (setattr) ---

    def test_setattr_delegates_to_optic(self, optic, ext_optic):
        ext_optic.name = "Modified"
        assert optic.name == "Modified"

    def test_setattr_source_stays_local(self, ext_optic):
        new_source = SMFSource(mfd_um=5.0, wavelength_um=0.55)
        ext_optic.source = new_source
        assert ext_optic.source is new_source

    def test_setattr_optic_stays_local(self, ext_optic):
        new_optic = Optic()
        ext_optic.optic = new_optic
        assert ext_optic.optic is new_optic

    # --- Method delegation ---

    def test_set_aperture_through_wrapper(self, optic, ext_optic):
        ext_optic.set_aperture("EPD", 10)
        assert optic.aperture.value == 10

    def test_add_field_through_wrapper(self, optic, ext_optic):
        initial_count = len(optic.fields.fields)
        ext_optic.fields.add(y=5.0)
        assert len(optic.fields.fields) == initial_count + 1

    def test_add_wavelength_through_wrapper(self, optic, ext_optic):
        initial_count = len(optic.wavelengths.wavelengths)
        ext_optic.wavelengths.add(0.65)
        assert len(optic.wavelengths.wavelengths) == initial_count + 1

    def test_set_radius_through_wrapper(self, optic, ext_optic):
        ext_optic.updater.set_radius(100.0, surface_number=1)
        assert float(optic.surfaces[1].geometry.radius) == pytest.approx(100.0)

    def test_set_thickness_through_wrapper(self, optic, ext_optic):
        ext_optic.updater.set_thickness(99.0, surface_number=1)
        assert optic.surfaces.get_thickness(1).item() == (pytest.approx(99.0))

    def test_paraxial_calculations_through_wrapper(self, ext_optic):
        """Paraxial data should be accessible through the wrapper."""
        f = ext_optic.paraxial.f2()
        assert f is not None and f != 0

    def test_info_through_wrapper(self, ext_optic):
        """info() should not raise when called through wrapper."""
        ext_optic.info()

    # --- Tracing ---

    def test_trace_returns_rays_and_path(self, ext_optic):
        traced_rays, ray_path = ext_optic.trace(num_rays=32)
        assert isinstance(traced_rays, RealRays)
        assert "x" in ray_path and "y" in ray_path and "z" in ray_path

    def test_trace_ray_path_shape(self, ext_optic):
        traced_rays, ray_path = ext_optic.trace(num_rays=32)
        num_surfaces = ext_optic.optic.surfaces.num_surfaces
        actual_num_rays = be.size(traced_rays.x)
        x_path = ray_path["x"]
        assert be.shape(x_path) == (num_surfaces, actual_num_rays)

    def test_trace_rays_have_intensity(self, ext_optic):
        traced_rays, _ = ext_optic.trace(num_rays=32)
        # At least some rays should pass through a simple system
        total_i = float(be.sum(traced_rays.i))
        assert total_i > 0

    def test_trace_respects_source_change(self, ext_optic):
        """After replacing the source, trace should use the new one."""
        new_source = SMFSource(
            mfd_um=50.0,
            wavelength_um=0.55,
            is_point_source=True,
        )
        ext_optic.source = new_source
        traced_rays, _ = ext_optic.trace(num_rays=16)
        assert be.size(traced_rays.x) > 0

    # --- Drawing ---

    def test_draw_returns_figure_and_axes(self, ext_optic):
        fig, ax = ext_optic.draw(num_rays=8, figsize=(6, 3))
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None
        plt.close(fig)

    def test_draw_accepts_title(self, ext_optic):
        fig, ax = ext_optic.draw(num_rays=8, title="Test Title")
        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_draw_auto_title(self, ext_optic):
        fig, ax = ext_optic.draw(num_rays=8)
        assert "SMFSource" in ax.get_title()
        plt.close(fig)

    def test_draw_accepts_existing_axes(self, ext_optic):
        fig_ext, ax_ext = plt.subplots()
        fig, ax = ext_optic.draw(num_rays=8, ax=ax_ext)
        assert ax is ax_ext
        plt.close(fig_ext)

    def test_draw_projection_yz(self, ext_optic):
        """YZ projection should work without error."""
        fig, ax = ext_optic.draw(num_rays=8, projection="YZ")
        plt.close(fig)

    def test_draw_projection_xz(self, ext_optic):
        """XZ projection should work without error."""
        fig, ax = ext_optic.draw(num_rays=8, projection="XZ")
        plt.close(fig)

    def test_draw_projection_xy(self, ext_optic):
        """XY projection should work without error."""
        fig, ax = ext_optic.draw(num_rays=8, projection="XY")
        plt.close(fig)

    def test_draw_with_limits(self, ext_optic):
        fig, ax = ext_optic.draw(num_rays=8, xlim=(-10, 100), ylim=(-20, 20))
        assert ax.get_xlim() == pytest.approx((-10, 100))
        assert ax.get_ylim() == pytest.approx((-20, 20))
        plt.close(fig)

    # --- Restricted methods ---

    def test_trace_generic_raises(self, ext_optic):
        with pytest.raises(NotImplementedError, match="trace_generic"):
            ext_optic.trace_generic(0, 0, 0, 0, 0.55)

    # --- __repr__ ---

    def test_repr_format(self, ext_optic):
        r = repr(ext_optic)
        assert "ExtendedSourceOptic" in r
        assert "SMFSource" in r

    def test_repr_with_named_optic(self, source):
        optic = Optic(name="My Lens")
        optic.surfaces.add(index=0, thickness=10)
        optic.surfaces.add(index=1, thickness=10, is_stop=True)
        optic.surfaces.add(index=2)
        optic.wavelengths.add(0.55, is_primary=True)
        ext = ExtendedSourceOptic(optic, source)
        assert "My Lens" in repr(ext)

    def test_repr_unnamed_optic(self, source):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=10)
        optic.surfaces.add(index=1, thickness=10, is_stop=True)
        optic.surfaces.add(index=2)
        optic.wavelengths.add(0.55, is_primary=True)
        ext = ExtendedSourceOptic(optic, source)
        assert "Unnamed" in repr(ext)


# ===========================================================================
# IncoherentIrradiance Integration Tests
# ===========================================================================


class TestIrradianceWithSource:
    """Test that IncoherentIrradiance works with the source= parameter."""

    @pytest.fixture(autouse=True)
    def _backend(self, set_test_backend):
        pass

    @pytest.fixture()
    def optic_with_detector(self):
        return _simple_singlet_with_detector()

    @pytest.fixture()
    def source(self):
        return _default_source()

    def test_irradiance_with_source(self, optic_with_detector, source):
        from optiland.analysis import IncoherentIrradiance

        analysis = IncoherentIrradiance(
            optic_with_detector,
            source=source,
            num_rays=128,
            detector_surface=-1,
            res=(16, 16),
        )
        assert analysis.data is not None
        assert len(analysis.data) > 0

    def test_irradiance_source_and_user_rays_conflict(
        self, optic_with_detector, source
    ):
        """Providing both source and user_initial_rays should raise."""
        from optiland.analysis import IncoherentIrradiance

        dummy_rays = source.generate_rays(16)
        with pytest.raises(ValueError, match="Cannot specify both"):
            IncoherentIrradiance(
                optic_with_detector,
                source=source,
                user_initial_rays=dummy_rays,
                num_rays=16,
                detector_surface=-1,
                res=(8, 8),
            )

    def test_irradiance_view_smoke(self, optic_with_detector, source):
        """view() should execute without error."""
        from optiland.analysis import IncoherentIrradiance

        analysis = IncoherentIrradiance(
            optic_with_detector,
            source=source,
            num_rays=128,
            detector_surface=-1,
            res=(8, 8),
        )
        fig = analysis.view(figsize=(4, 3))
        plt.close("all")


# ===========================================================================
# Source Visualization Tests
# ===========================================================================


class TestSourceViewer:
    """Tests for the SourceViewer visualization."""

    @pytest.fixture(autouse=True)
    def _backend(self, set_test_backend):
        pass

    def test_viewer_returns_six_panels(self):
        src = _default_source()
        from optiland.sources.visualization import SourceViewer

        viewer = SourceViewer(src)
        fig, axes = viewer.view(num_rays=32)
        assert len(axes) == 6
        plt.close(fig)

    def test_viewer_custom_figsize(self):
        src = _default_source()
        from optiland.sources.visualization import SourceViewer

        viewer = SourceViewer(src)
        fig, axes = viewer.view(num_rays=32, figsize=(12, 4))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12)
        assert h == pytest.approx(4)
        plt.close(fig)

    def test_viewer_propagation_distance(self):
        """Different propagation distances should not raise."""
        src = _default_source()
        from optiland.sources.visualization import SourceViewer

        viewer = SourceViewer(src)
        fig, _ = viewer.view(num_rays=32, propagation_distance=1.0)
        plt.close(fig)
        fig, _ = viewer.view(num_rays=32, propagation_distance=0.01)
        plt.close(fig)


# ===========================================================================
# Standard Analysis Delegation Tests
# ===========================================================================


class TestStandardAnalysesThroughWrapper:
    """Verify that standard Optic capabilities remain accessible when using an
    ExtendedSourceOptic.

    Because ExtendedSourceOptic overrides trace() with a source-based
    signature, standard analyses that call optic.trace(Hx, Hy, ...) should
    be run on the inner optic (ext_optic.optic). Paraxial, aberration, and
    setter methods that do NOT call trace() work directly through the wrapper.
    """

    @pytest.fixture(autouse=True)
    def _backend(self, set_test_backend):
        pass

    @pytest.fixture()
    def ext_optic(self):
        optic = _simple_singlet()
        source = _default_source()
        return ExtendedSourceOptic(optic, source)

    # --- Analyses on the inner optic ---

    def test_spot_diagram_on_inner_optic(self, ext_optic):
        """SpotDiagram runs on the inner optic accessed via .optic."""
        from optiland.analysis import SpotDiagram

        spot = SpotDiagram(ext_optic.optic)
        assert spot.data is not None
        plt.close("all")

    def test_ray_fan_on_inner_optic(self, ext_optic):
        from optiland.analysis import RayFan

        fan = RayFan(ext_optic.optic)
        assert fan.data is not None
        plt.close("all")

    def test_encircled_energy_on_inner_optic(self, ext_optic):
        from optiland.analysis import EncircledEnergy

        ee = EncircledEnergy(ext_optic.optic)
        assert ee.data is not None
        plt.close("all")

    def test_distortion_on_inner_optic(self, ext_optic):
        from optiland.analysis import Distortion

        ext_optic.fields.add(y=5.0)
        ext_optic.fields.add(y=10.0)
        d = Distortion(ext_optic.optic)
        assert d.data is not None
        plt.close("all")

    def test_field_curvature_on_inner_optic(self, ext_optic):
        from optiland.analysis import FieldCurvature

        ext_optic.fields.add(y=5.0)
        ext_optic.fields.add(y=10.0)
        fc = FieldCurvature(ext_optic.optic)
        assert fc.data is not None
        plt.close("all")

    def test_y_ybar_on_inner_optic(self, ext_optic):
        from optiland.analysis import YYbar

        yb = YYbar(ext_optic.optic)
        assert yb.data is not None
        plt.close("all")

    def test_standard_trace_on_inner_optic(self, ext_optic):
        """The standard Optic.trace (field-based) works on .optic."""
        rays = ext_optic.optic.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=16)
        assert isinstance(rays, RealRays)
        assert be.size(rays.x) > 0

    # --- Paraxial and aberration data through the wrapper ---

    def test_paraxial_focal_length(self, ext_optic):
        """Paraxial properties are accessible through the wrapper."""
        f = ext_optic.paraxial.f2()
        assert f is not None and float(f) != 0

    def test_paraxial_fno(self, ext_optic):
        fno = ext_optic.paraxial.FNO()
        assert fno is not None and float(fno) > 0

    def test_aberrations_through_wrapper(self, ext_optic):
        """Seidel aberration coefficients are accessible through the wrapper."""
        sc = ext_optic.aberrations.SC()
        assert sc is not None

    # --- Setters and modifiers through the wrapper ---

    def test_scale_system_through_wrapper(self, ext_optic):
        """scale_system works through the wrapper."""
        f_before = float(ext_optic.paraxial.f2())
        ext_optic.updater.scale_system(2.0)
        f_after = float(ext_optic.paraxial.f2())
        assert f_after == pytest.approx(2.0 * f_before, rel=1e-6)

    def test_update_through_wrapper(self, ext_optic):
        """update() does not raise through the wrapper."""
        ext_optic.updater.update()
