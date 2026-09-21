from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from optiland import backend as be
from optiland.interactions.phase_interaction_model import PhaseInteractionModel
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.phase import RadialPhaseProfile
from optiland.rays.real_rays import RealRays


@pytest.mark.parametrize("reflective", [False, True])
@pytest.mark.parametrize("gradient", [0.0, 0.1])
def test_phase_normal_orientation_invariance(set_test_backend, reflective, gradient):
    """Reversing a surface normal cannot change the physical outgoing ray."""
    results = []
    for sign in (1.0, -1.0):
        surface = Mock()
        surface.material_pre = IdealMaterial(n=1.0)
        surface.material_post = IdealMaterial(n=1.5)
        surface.geometry.surface_normal.return_value = (
            be.zeros(2), be.zeros(2), sign * be.ones(2)
        )
        model = PhaseInteractionModel(
            surface, RadialPhaseProfile(coefficients=[gradient]),
            is_reflective=reflective,
        )
        model._apply_coating_and_bsdf = lambda rays, *args: rays
        rays = RealRays(
            x=be.ones(2), y=be.zeros(2), z=be.zeros(2),
            L=be.array([0.2, 0.2]), M=be.zeros(2),
            N=be.array([np.sqrt(0.96), -np.sqrt(0.96)]),
            wavelength=0.55, intensity=be.ones(2),
        )
        result = model.interact_real_rays(rays)
        results.append(np.stack([be.to_numpy(result.L), be.to_numpy(result.N)]))
        expected_sign = np.array([-1, 1] if reflective else [1, -1])
        np.testing.assert_array_equal(np.sign(be.to_numpy(result.N)), expected_sign)
    np.testing.assert_allclose(results[0], results[1], atol=1e-12)


def test_zero_phase_curved_singlet_matches_refraction(set_test_backend):
    """Adding a zero phase profile must preserve the complete singlet trace."""
    results = []
    for with_phase in (False, True):
        lens = Optic()
        lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        options = {}
        if with_phase:
            options["phase_profile"] = RadialPhaseProfile(coefficients=[0.0, 0.0])
        lens.surfaces.add(
            index=1, radius=100.0, thickness=5.0, is_stop=True,
            material=IdealMaterial(n=4.0), **options,
        )
        lens.surfaces.add(index=2, radius=be.inf, thickness=100.0)
        lens.surfaces.add(index=3)
        lens.set_aperture("EPD", 20.0)
        lens.fields.set_type("angle")
        lens.fields.add(y=0.0)
        lens.wavelengths.add(value=10.0, is_primary=True)
        py = be.array([0.5, 1.0])
        zero = be.zeros_like(py)
        rays = lens.trace_generic(zero, zero, zero, py, 10.0)
        results.append(np.stack([
            be.to_numpy(getattr(rays, key)) for key in ("y", "M", "N", "opd", "i")
        ]))
        assert np.all(be.to_numpy(lens.surfaces.N)[1] > 0)
    np.testing.assert_allclose(results[0], results[1], rtol=1e-10, atol=1e-10)
