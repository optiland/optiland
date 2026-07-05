"""Tests for the pluggable distortion strategies.

These cover the reference-point strategies (chief ray vs. energy centroid), the
distortion models (rotational vs. affine), the factory, and the integration of
the non-paraxial path into the standard distortion analysis, grid distortion,
and image-simulation warper.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import optiland.backend as be
from optiland import analysis
from optiland.analysis.distortion_strategies import (
    AffineDistortionModel,
    CentroidReferencePoint,
    ChiefRayReferencePoint,
    DistortionModel,
    RotationalDistortionModel,
    create_distortion_model,
)
from optiland.analysis.image_simulation import DistortionWarper
from optiland.samples.objectives import CookeTriplet


@pytest.fixture
def cooke_triplet():
    return CookeTriplet()


class TestReferencePointStrategies:
    def test_chief_ray_locate_shapes(self, set_test_backend, cooke_triplet):
        locator = ChiefRayReferencePoint()
        Hx = be.array([0.0, 0.0, 0.0])
        Hy = be.array([0.0, 0.5, 1.0])
        x, y = locator.locate(cooke_triplet, Hx, Hy, 0.55)
        assert x.shape[0] == 3
        assert y.shape[0] == 3
        # On-axis field lands near the origin for this symmetric system.
        assert_allclose(be.to_numpy(x)[0], 0.0, atol=1e-9)
        assert_allclose(be.to_numpy(y)[0], 0.0, atol=1e-9)

    def test_centroid_matches_chief_for_symmetric_system(
        self, set_test_backend, cooke_triplet
    ):
        # For a well-behaved rotationally symmetric system, the transmitted
        # centroid coincides with the chief-ray intercept to high accuracy.
        chief = ChiefRayReferencePoint()
        centroid = CentroidReferencePoint(num_rays=32)
        Hy = be.array([0.0, 0.7])
        Hx = be.zeros_like(Hy)
        xc, yc = chief.locate(cooke_triplet, Hx, Hy, 0.55)
        xb, yb = centroid.locate(cooke_triplet, Hx, Hy, 0.55)
        assert_allclose(be.to_numpy(xb), be.to_numpy(xc), atol=5e-3)
        assert_allclose(be.to_numpy(yb), be.to_numpy(yc), atol=5e-3)

    def test_centroid_unweighted_runs(self, set_test_backend, cooke_triplet):
        centroid = CentroidReferencePoint(num_rays=20, flux_weighted=False)
        x, y = centroid.locate(cooke_triplet, be.array([0.3]), be.array([0.3]), 0.55)
        assert np.all(np.isfinite(be.to_numpy(x)))
        assert np.all(np.isfinite(be.to_numpy(y)))


class TestRotationalModel:
    def test_invalid_distortion_type(self, set_test_backend):
        with pytest.raises(ValueError):
            RotationalDistortionModel(distortion_type="invalid")

    def test_evaluate_before_fit_raises(self, set_test_backend, cooke_triplet):
        model = RotationalDistortionModel()
        with pytest.raises(RuntimeError):
            model.evaluate(cooke_triplet, be.array([0.0]), be.array([0.5]), 0.55)

    def test_field_center_is_zero_distortion(self, set_test_backend, cooke_triplet):
        model = RotationalDistortionModel()
        result = model.compute(cooke_triplet, be.array([0.0]), be.array([1e-10]), 0.55)
        pct = model.percent(result, signed=True)
        assert_allclose(be.to_numpy(pct)[0], 0.0, atol=1e-6)


class TestAffineModel:
    def test_invalid_projection(self, set_test_backend):
        with pytest.raises(ValueError):
            AffineDistortionModel(field_projection="invalid")

    def test_evaluate_before_fit_raises(self, set_test_backend, cooke_triplet):
        model = AffineDistortionModel(fit_grid_size=5)
        with pytest.raises(RuntimeError):
            model.evaluate(cooke_triplet, be.array([0.0]), be.array([0.5]), 0.55)

    def test_reference_radius_positive(self, set_test_backend, cooke_triplet):
        model = AffineDistortionModel(
            reference_point=CentroidReferencePoint(num_rays=16), fit_grid_size=7
        )
        model.fit(cooke_triplet, 0.55)
        assert float(be.to_numpy(model._reference_radius)) > 0.0

    def test_affine_residual_finite(self, set_test_backend, cooke_triplet):
        model = AffineDistortionModel(
            reference_point=CentroidReferencePoint(num_rays=16), fit_grid_size=7
        )
        result = model.compute(
            cooke_triplet, be.array([0.0, 0.5]), be.array([0.0, 0.5]), 0.55
        )
        pct = model.percent(result)
        assert np.all(np.isfinite(be.to_numpy(pct)))


class TestFactory:
    def test_paraxial_alias(self, set_test_backend):
        model = create_distortion_model("paraxial")
        assert isinstance(model, RotationalDistortionModel)

    def test_nonparaxial_alias(self, set_test_backend):
        model = create_distortion_model("centroid")
        assert isinstance(model, AffineDistortionModel)

    def test_instance_passthrough(self, set_test_backend):
        custom = AffineDistortionModel()
        assert create_distortion_model(custom) is custom

    def test_unknown_method_raises(self, set_test_backend):
        with pytest.raises(ValueError):
            create_distortion_model("bogus")

    def test_distortion_type_forwarded(self, set_test_backend):
        model = create_distortion_model("paraxial", distortion_type="f-theta")
        assert isinstance(model, RotationalDistortionModel)
        assert model.distortion_type == "f-theta"

    def test_is_distortion_model(self, set_test_backend):
        assert isinstance(create_distortion_model("paraxial"), DistortionModel)


class TestNonParaxialIntegration:
    def test_radial_distortion_nonparaxial(self, set_test_backend, cooke_triplet):
        dist = analysis.Distortion(cooke_triplet, method="nonparaxial")
        assert len(dist.data) == len(cooke_triplet.wavelengths.get_wavelengths())
        for arr in dist.data:
            assert np.all(np.isfinite(be.to_numpy(arr)))

    def test_grid_distortion_nonparaxial(self, set_test_backend, cooke_triplet):
        dist = analysis.GridDistortion(cooke_triplet, method="nonparaxial")
        assert dist.data["xr"].shape == (10, 10)
        assert dist.data["yp"].shape == (10, 10)
        assert float(be.to_numpy(dist.data["max_distortion"])) >= 0.0

    def test_grid_distortion_custom_model(self, set_test_backend, cooke_triplet):
        model = AffineDistortionModel(
            reference_point=CentroidReferencePoint(num_rays=16), fit_grid_size=7
        )
        dist = analysis.GridDistortion(cooke_triplet, method=model)
        assert dist.data["xr"].shape == (10, 10)

    def test_warper_with_centroid_reference(self, set_test_backend, cooke_triplet):
        warper = DistortionWarper(
            cooke_triplet, reference_point=CentroidReferencePoint(num_rays=16)
        )
        grid = warper.generate_distortion_map(
            0.55, (16, 16), num_grid_points=7, degree=3
        )
        assert tuple(grid.shape) == (1, 16, 16, 2)
        assert np.all(np.isfinite(be.to_numpy(grid)))
