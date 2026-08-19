from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import optiland.backend as be
from optiland.optic import ExtendedSourceOptic, Optic
from optiland.sources import SMFSource


class TestExtendedSourceOptic:
    @pytest.fixture
    def optic(self):
        """Creates a simple optic for testing."""
        optic = Optic()
        optic.surfaces.add(index=0, thickness=10)
        optic.surfaces.add(index=1, thickness=10, is_stop=True)
        optic.surfaces.add(index=2)
        optic.wavelengths.add(1.55, is_primary=True)
        return optic

    @pytest.fixture
    def source(self):
        """Creates an SMF source for testing."""
        return SMFSource(
            mfd_um=10.4,
            wavelength_um=1.55,
            total_power=1.0,
        )

    def test_divergence_calculation(self):
        """Test that divergence is calculated if not provided."""
        import math

        src = SMFSource(mfd_um=10.4, wavelength_um=1.55)

        # Check calculation against known formula: theta_half = wavelength / (pi * w0)
        w0 = 10.4 / 2.0
        expected_rad = 1.55 / (math.pi * w0)
        expected_deg = 2 * math.degrees(expected_rad)

        assert abs(src.divergence_deg_1e2 - expected_deg) < 1e-4

        # Test override
        src_override = SMFSource(
            mfd_um=10.4, wavelength_um=1.55, divergence_deg_1e2=20.0
        )
        assert src_override.divergence_deg_1e2 == 20.0

    @pytest.fixture
    def ext_optic(self, optic, source):
        """Creates an ExtendedSourceOptic instance."""
        return ExtendedSourceOptic(optic, source)

    def test_initialization(self, optic, source, ext_optic):
        """Test initialization and attribute delegation."""
        assert ext_optic.optic is optic
        assert ext_optic.source is source

        # Test delegation
        assert ext_optic.name == optic.name
        assert ext_optic.surfaces is optic.surfaces

        # Test setting attribute on optic via wrapper (should affect optic)
        ext_optic.name = "New Name"
        assert optic.name == "New Name"

    def test_trace(self, ext_optic):
        """Test trace method."""
        num_rays = 100
        traced_rays, ray_path = ext_optic.trace(num_rays=num_rays)

        # Sobol sampler rounds up to nearest power of 2
        assert be.size(traced_rays.x) > 0
        assert "x" in ray_path
        assert "y" in ray_path
        assert "z" in ray_path

        # Check shapes of ray path arrays
        num_surfaces = ext_optic.optic.surfaces.num_surfaces
        actual_num_rays = be.size(traced_rays.x)
        assert be.shape(ray_path["x"]) == (num_surfaces, actual_num_rays)

    def test_draw(self, ext_optic):
        """Test draw method (smoke test)."""
        fig, ax = ext_optic.draw(num_rays=10)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_trace_generic_not_implemented(self, ext_optic):
        """Test that trace_generic raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ext_optic.trace_generic(0, 0, 0, 0, 1.55)

    def test_repr(self, ext_optic):
        """Test string representation."""
        repr_str = repr(ext_optic)
        assert "ExtendedSourceOptic" in repr_str
        assert "SMFSource" in repr_str
