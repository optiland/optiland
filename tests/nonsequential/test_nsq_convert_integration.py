"""Integration smoke tests for sequential_to_nonsequential converter.

These tests run full Monte Carlo traces; they are slower than unit tests.
Mark with @pytest.mark.slow if a slow-test marker is configured.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

from optiland.nonsequential.convert import sequential_to_nonsequential

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _singlet_optic():
    """BK7 singlet: R1=50, R2=-50, t=5, EPD=10."""
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True
    )
    optic.add_surface(index=2, radius=-50.0, thickness=50.0)
    optic.add_surface(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)
    return optic


def _doublet_optic():
    """Simple cemented doublet."""
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1, radius=60.0, thickness=6.0, material="N-BK7", is_stop=True
    )
    optic.add_surface(index=2, radius=-30.0, thickness=2.0, material="N-F2")
    optic.add_surface(index=3, radius=-80.0, thickness=50.0)
    optic.add_surface(index=4)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)
    return optic


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_singlet_trace_receives_flux():
    """A converted singlet must deliver flux > 0 to the detector."""
    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    result = scene.trace(num_rays=5_000, seed=0)
    assert result.detectors["D1"].total_flux > 0


def test_doublet_trace_receives_flux():
    """A converted doublet must deliver flux > 0 to the detector."""
    optic = _doublet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    result = scene.trace(num_rays=5_000, seed=0)
    assert result.detectors["D1"].total_flux > 0
