"""Default merit block (opt v2 PR8.3).

The quadrature estimate is checked against a dense pupil computed through the
public analysis path, independent of the block: an equal-area midpoint rule, so
a plain mean over the rays is the pupil average. A uniform grid is a poor
reference here: it clips the pupil edge and converges as 1/N.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.distribution import BaseDistribution
from optiland.optimization import DefaultMeritBlock, OptimizationProblem, minimize
from optiland.optimization.evaluators import FiniteDiffEvaluator
from optiland.optimization.failure import ConfigurationError
from optiland.samples.objectives import CookeTriplet
from optiland.wavefront import OPD
from optiland.wavefront.reference_geometry import SphericalReference
from optiland.wavefront.strategy import CentroidStrategy

from ..utils import assert_allclose

pytestmark = pytest.mark.filterwarnings("ignore")

RINGS = 96
SPOKES = 192


class EqualAreaPupil(BaseDistribution):
    """Midpoint rule in rho^2 and angle: every ray stands for the same area."""

    def generate_points(self, num_points: int = RINGS) -> None:
        rho = np.sqrt((np.arange(num_points) + 0.5) / num_points)
        theta = 2.0 * np.pi * (np.arange(SPOKES) + 0.5) / SPOKES
        rho, theta = np.meshgrid(rho, theta)
        self.x = be.array((rho * np.cos(theta)).ravel())
        self.y = be.array((rho * np.sin(theta)).ravel())


def _dense_pupil():
    pupil = EqualAreaPupil()
    pupil.generate_points()
    return pupil


def _dense_spot(lens, field, wavelength, reference_xy):
    rays = lens.trace(*field, wavelength, None, _dense_pupil())
    x, y = be.to_numpy(rays.x), be.to_numpy(rays.y)
    keep = np.isfinite(x) & np.isfinite(y)
    return np.mean((x[keep] - reference_xy[0]) ** 2 + (y[keep] - reference_xy[1]) ** 2)


def _reference_xy(lens, field, reference):
    primary = lens.primary_wavelength
    if reference == "chief":
        rays = lens.trace_generic(*field, 0.0, 0.0, primary)
        return float(be.to_numpy(rays.x)[0]), float(be.to_numpy(rays.y)[0])
    rays = lens.trace(*field, primary, None, _dense_pupil())
    return np.nanmean(be.to_numpy(rays.x)), np.nanmean(be.to_numpy(rays.y))


def _dense_spot_merit(lens, reference):
    total = 0.0
    for field, fw in zip(
        lens.fields.get_field_coords(), lens.fields.weights, strict=True
    ):
        ref = _reference_xy(lens, field, reference)
        for wave in lens.wavelengths.wavelengths:
            spot = _dense_spot(lens, field, wave.value, ref)
            total += fw * wave.weight * spot
    return total


def _dense_wavefront_merit(lens, strategy, **kwargs):
    total = 0.0
    for field, fw in zip(
        lens.fields.get_field_coords(), lens.fields.weights, strict=True
    ):
        for wave in lens.wavelengths.wavelengths:
            opd = OPD(
                lens,
                field,
                wave.value,
                num_rays=None,
                distribution=_dense_pupil(),
                strategy=strategy,
                **kwargs,
            )
            total += fw * wave.weight * float(be.to_numpy(opd.rms())) ** 2
    return total


def _merit(lens, **kwargs):
    problem = OptimizationProblem()
    block = problem.add_default_merit(lens, **kwargs)
    return float(be.to_numpy(problem.sum_squared())), block


@pytest.mark.parametrize("reference", ["centroid", "chief"])
def test_spot_matches_dense_grid(set_test_backend, reference):
    lens = CookeTriplet()
    merit, _ = _merit(lens, kind="spot", rings=6, arms=24, reference=reference)
    assert merit == pytest.approx(_dense_spot_merit(lens, reference), rel=1e-3)


def test_default_sampling_is_a_good_estimate(set_test_backend):
    lens = CookeTriplet()
    merit, block = _merit(lens)
    assert merit == pytest.approx(_dense_spot_merit(lens, "centroid"), rel=0.02)
    assert block.size() == 2 * 3 * 3 * 18


def test_wavefront_chief_matches_dense_grid(set_test_backend):
    lens = CookeTriplet()
    merit, _ = _merit(lens, kind="wavefront", rings=6, arms=24, reference="chief")
    assert merit == pytest.approx(_dense_wavefront_merit(lens, "chief_ray"), rel=1e-3)


def test_wavefront_centroid_matches_dense_grid(set_test_backend, monkeypatch):
    # Same definition as the block: no outlier trimming, and the sphere runs
    # through the exit pupil (CentroidStrategy sizes it from the total path).
    def exit_pupil_sphere(self, wavefront_points, centroid, weights):
        center = tuple(float(c) for c in centroid)
        point = self.optic.paraxial.exit_pupil_point_gcs()
        radius = sum(
            (c - float(be.to_numpy(be.ravel(p))[0])) ** 2
            for c, p in zip(center, point, strict=True)
        )
        return SphericalReference(center, radius**0.5)

    monkeypatch.setattr(CentroidStrategy, "_create_spherical_ref", exit_pupil_sphere)
    lens = CookeTriplet()
    merit, _ = _merit(lens, kind="wavefront", rings=6, arms=24, reference="centroid")
    dense = _dense_wavefront_merit(lens, "centroid", robust_trim_std=0.0)
    assert merit == pytest.approx(dense, rel=1e-4)


def test_wavefront_centroid_is_close_to_the_centroid_analysis(set_test_backend):
    lens = CookeTriplet()
    merit, _ = _merit(lens, kind="wavefront", rings=6, arms=24, reference="centroid")
    dense = _dense_wavefront_merit(lens, "centroid", robust_trim_std=0.0)
    assert merit == pytest.approx(dense, rel=2e-3)


def test_weights_scale_rows(set_test_backend):
    lens = CookeTriplet()
    plain, _ = _merit(lens, fields=[1], wavelengths=[1])
    scaled, _ = _merit(lens, fields=[1], wavelengths=[1], weight=3.0)
    assert_allclose(scaled, 9.0 * plain)

    lens.fields.fields[1].weight = 4.0
    lens.wavelengths.wavelengths[1].weight = 0.5
    weighted, _ = _merit(lens, fields=[1], wavelengths=[1])
    assert_allclose(weighted, 2.0 * plain)


def test_zero_weight_rows_are_exact_zeros(set_test_backend):
    lens = CookeTriplet()
    lens.fields.fields[1].weight = 0.0
    problem = OptimizationProblem()
    block = problem.add_default_merit(lens, fields=[0, 1], wavelengths=[1])
    r = be.to_numpy(problem.residual_vector())
    assert np.any(r[: block.size() // 2] != 0.0)
    assert np.all(r[block.size() // 2 :] == 0.0)


def test_spot_keeps_lateral_color(set_test_backend):
    lens = CookeTriplet()
    single = sum(_merit(lens, fields=[2], wavelengths=[w])[0] for w in range(3))
    pooled, _ = _merit(lens, fields=[2])
    assert pooled > single


def test_primary_not_selected_uses_first_selected(set_test_backend):
    lens = CookeTriplet()
    block = DefaultMeritBlock(lens, wavelengths=[2, 0])
    assert block._reference_wavelength == 2


def test_chief_ray_adds_no_row(set_test_backend):
    lens = CookeTriplet()
    spot = DefaultMeritBlock(lens, reference="chief", fields=[0], wavelengths=[0])
    wave = DefaultMeritBlock(lens, kind="wavefront", fields=[0], wavelengths=[0])
    assert spot.trace_requests()[0].size == 19
    assert spot.size() == 36
    assert wave.size() == 18


@pytest.mark.parametrize("batching", [True, False])
def test_unbatched_matches_batched(set_test_backend, batching):
    lens = CookeTriplet()
    reference = OptimizationProblem(batching=True)
    reference.add_default_merit(lens, kind="wavefront", reference="chief")
    problem = OptimizationProblem(batching=batching)
    problem.add_default_merit(lens, kind="wavefront", reference="chief")
    assert_allclose(problem.residual_vector(), reference.residual_vector())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "zernike"},
        {"reference": "vertex"},
        {"rings": 0},
        {"arms": 0},
        {"weight": 0.0},
        {"fields": [99]},
        {"fields": "some"},
        {"wavelengths": []},
    ],
)
def test_bad_arguments_raise(set_test_backend, kwargs):
    problem = OptimizationProblem()
    with pytest.raises(ConfigurationError):
        problem.add_default_merit(CookeTriplet(), **kwargs)


@pytest.mark.parametrize("kind", ["spot", "wavefront"])
@pytest.mark.parametrize("reference", ["centroid", "chief"])
def test_partial_trace_jacobian_matches_full(reference, kind):
    be.set_backend("numpy")
    lens = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_default_merit(lens, kind=kind, reference=reference, fields=[0, 2])
    for surface in (1, 3, 5):
        problem.add_variable(lens, "radius", surface_number=surface)
    problem.add_variable(lens, "conic", surface_number=5)
    partial = FiniteDiffEvaluator(problem)
    full = FiniteDiffEvaluator(problem, partial_trace=False)
    x = partial.read_x()
    J_partial = partial.jacobian(x)
    assert partial.resumed_traces > 0
    assert_allclose(J_partial, full.jacobian(x), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("kind", ["spot", "wavefront"])
@pytest.mark.parametrize("reference", ["centroid", "chief"])
def test_autograd_jacobian_matches_finite_differences(reference, kind):
    pytest.importorskip("torch")
    from optiland.optimization.evaluators import AutogradEvaluator

    be.set_backend("torch")
    be.set_precision("float64")
    try:
        lens = CookeTriplet()
        problem = OptimizationProblem()
        problem.add_default_merit(lens, kind=kind, reference=reference, fields=[2])
        problem.add_variable(lens, "radius", surface_number=3)
        problem.add_variable(lens, "conic", surface_number=5)
        auto = AutogradEvaluator(problem)
        x = auto.read_x()
        J_auto = be.to_numpy(auto.jacobian(x))
        fd = FiniteDiffEvaluator(problem, scheme="central")
        J_fd = fd.jacobian(fd.read_x())
        scale = np.max(np.abs(J_fd))
        assert_allclose(J_auto / scale, J_fd / scale, atol=1e-5)
    finally:
        be.set_backend("numpy")


@pytest.mark.parametrize("kind", ["spot", "wavefront"])
def test_dls_lowers_the_merit(set_test_backend, kind):
    lens = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_default_merit(lens, kind=kind)
    for surface in (1, 2, 3):
        problem.add_variable(lens, "radius", surface_number=surface)
    start = float(be.to_numpy(problem.sum_squared()))
    result = minimize(problem, "dls", maxiter=30)
    assert result.fun < start
    assert_allclose(problem.sum_squared(), result.fun)
