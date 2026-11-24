"""Tests for the rms_wavefront, mtf and boresight operands."""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.mtf import GeometricMTF, SampledMTF
from optiland.optic import Optic
from optiland.optimization.operand import ImageQualityOperand, operand_registry
from optiland.samples.objectives import CookeTriplet
from optiland.tolerancing.core import Tolerancing
from optiland.wavefront import Wavefront
from tests.utils import assert_allclose

WAVELENGTH = 0.55  # µm
FOCAL_LENGTH = 100.0  # mm
EPD = 10.0  # mm, F/10


def parabola(defocus: float = 0.0) -> Optic:
    """On-axis paraboloid mirror: aberration-free at its focus."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=-2 * FOCAL_LENGTH,
        conic=-1.0,
        thickness=-(FOCAL_LENGTH + defocus),
        material="mirror",
        is_stop=True,
    )
    optic.surfaces.add(index=2)
    optic.set_aperture(aperture_type="EPD", value=EPD)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(value=WAVELENGTH, is_primary=True)
    return optic


def thin_lens(dx: float = 0.0, dy: float = 0.0) -> Optic:
    """Ideal (paraxial) lens behind a centered stop, optionally decentered."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=0.0, is_stop=True)
    optic.surfaces.add(
        index=2,
        surface_type="paraxial",
        f=FOCAL_LENGTH,
        thickness=FOCAL_LENGTH,
        dx=dx,
        dy=dy,
    )
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=EPD)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(value=WAVELENGTH, is_primary=True)
    return optic


def diffraction_limited_mtf(frequency: float) -> float:
    """Analytic incoherent MTF of an aberration-free circular pupil."""
    cutoff = 1.0 / (WAVELENGTH * 1e-3 * FOCAL_LENGTH / EPD)
    phi = math.acos(frequency / cutoff)
    return 2.0 / math.pi * (phi - math.cos(phi) * math.sin(phi))


@pytest.mark.parametrize("name", ["rms_wavefront", "mtf", "boresight_x", "boresight_y"])
def test_operands_registered(name):
    assert name in operand_registry
    assert operand_registry.get(name) is getattr(ImageQualityOperand, name)


class TestRmsWavefront:
    def test_perfect_system_is_zero(self, set_test_backend):
        value = ImageQualityOperand.rms_wavefront(parabola(), 0.0, 0.0, 6, WAVELENGTH)
        assert_allclose(value, 0.0, atol=1e-8)

    def test_defocus_matches_analytic(self, set_test_backend):
        # Defocus dz gives W = W020 rho**2 with W020 = dz / (8 N**2); with
        # piston removed its RMS is W020 / (2 sqrt(3)).
        defocus = 0.1
        w020 = defocus / (8 * (FOCAL_LENGTH / EPD) ** 2) / (WAVELENGTH * 1e-3)
        value = ImageQualityOperand.rms_wavefront(
            parabola(defocus), 0.0, 0.0, 6, WAVELENGTH
        )
        # Residual difference is the exact vs paraxial defocus OPD at F/10.
        assert_allclose(value, w020 / (2 * math.sqrt(3)), rtol=5e-3)

    def test_matches_wavefront_data(self, set_test_backend):
        optic = CookeTriplet()
        value = ImageQualityOperand.rms_wavefront(
            optic, 0.0, 1.0, 32, WAVELENGTH, distribution="uniform"
        )
        data = Wavefront(optic, [(0.0, 1.0)], [WAVELENGTH], 32, "uniform").get_data(
            (0.0, 1.0), WAVELENGTH
        )
        opd = data.opd[data.intensity > 0]
        assert_allclose(value, be.sqrt(be.mean((opd - be.mean(opd)) ** 2)))

    def test_remove_tilt_matches_wavefront_detrend(self, set_test_backend):
        optic = CookeTriplet()
        value = ImageQualityOperand.rms_wavefront(
            optic, 0.0, 1.0, 32, WAVELENGTH, distribution="uniform", remove_tilt=True
        )
        data = Wavefront(optic, [(0.0, 1.0)], [WAVELENGTH], 32, "uniform").get_data(
            (0.0, 1.0), WAVELENGTH
        )
        residual = Wavefront.fit_and_remove_tilt(data, remove_piston=True)
        assert_allclose(value, be.sqrt(be.mean(residual**2)))

        with_tilt = ImageQualityOperand.rms_wavefront(
            optic, 0.0, 1.0, 32, WAVELENGTH, distribution="uniform"
        )
        assert value < with_tilt

    def test_primary_wavelength(self, set_test_backend):
        optic = CookeTriplet()
        assert_allclose(
            ImageQualityOperand.rms_wavefront(optic, 0.0, 0.7, 4, "primary"),
            ImageQualityOperand.rms_wavefront(
                optic, 0.0, 0.7, 4, optic.primary_wavelength
            ),
        )


class TestMTF:
    @pytest.mark.parametrize("azimuth", ["tangential", "sagittal"])
    @pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
    def test_diffraction_limited_matches_analytic(
        self, set_test_backend, azimuth, ratio
    ):
        cutoff = EPD / (WAVELENGTH * 1e-3 * FOCAL_LENGTH)
        sampled = ImageQualityOperand.mtf(
            parabola(), 0.0, 0.0, WAVELENGTH, ratio * cutoff, azimuth
        )
        geometric = ImageQualityOperand.mtf(
            parabola(), 0.0, 0.0, WAVELENGTH, ratio * cutoff, azimuth, "geometric"
        )
        expected = diffraction_limited_mtf(ratio * cutoff)
        # Sampled: discretization of the pupil autocorrelation on a 64-ray grid.
        assert_allclose(sampled, expected, atol=5e-3)
        # Geometric: a point image scaled by the diffraction-limited MTF.
        assert_allclose(geometric, expected, atol=1e-6)

    @pytest.mark.parametrize(
        ("azimuth", "freq"), [("tangential", (0.0, 40.0)), ("sagittal", (40.0, 0.0))]
    )
    def test_sampled_matches_sampled_mtf(self, set_test_backend, azimuth, freq):
        optic = CookeTriplet()
        value = ImageQualityOperand.mtf(
            optic, 0.0, 1.0, WAVELENGTH, 40.0, azimuth, num_rays=32
        )
        expected = SampledMTF(optic, (0.0, 1.0), WAVELENGTH, 32).calculate_mtf([freq])
        assert_allclose(value, expected[0])

    def test_geometric_matches_geometric_mtf(self, set_test_backend):
        optic = CookeTriplet()
        mtf = GeometricMTF(optic, [(0.0, 1.0)], WAVELENGTH, 32, max_freq=40.0)
        for index, azimuth in enumerate(["tangential", "sagittal"]):
            value = ImageQualityOperand.mtf(
                optic, 0.0, 1.0, WAVELENGTH, 40.0, azimuth, "geometric", 32
            )
            assert_allclose(value, mtf.mtf[0][index][-1])

    def test_off_axis_azimuths_differ(self, set_test_backend):
        optic = CookeTriplet()
        tangential = ImageQualityOperand.mtf(optic, 0.0, 1.0, WAVELENGTH, 40.0)
        sagittal = ImageQualityOperand.mtf(
            optic, 0.0, 1.0, WAVELENGTH, 40.0, "sagittal"
        )
        assert abs(float(tangential) - float(sagittal)) > 1e-3

    def test_invalid_arguments(self, set_test_backend):
        optic = CookeTriplet()
        with pytest.raises(ValueError, match="azimuth"):
            ImageQualityOperand.mtf(optic, 0.0, 0.0, WAVELENGTH, 40.0, "radial")
        with pytest.raises(ValueError, match="method"):
            ImageQualityOperand.mtf(optic, 0.0, 0.0, WAVELENGTH, 40.0, method="fft")


class TestBoresight:
    def test_centered_system_is_zero(self, set_test_backend):
        optic = thin_lens()
        assert_allclose(ImageQualityOperand.boresight_x(optic, "primary"), 0.0)
        assert_allclose(ImageQualityOperand.boresight_y(optic, "primary"), 0.0)

    @pytest.mark.parametrize(("dx", "dy"), [(0.5, 0.0), (0.0, -0.3), (0.2, 0.4)])
    def test_decentered_lens(self, set_test_backend, dx, dy):
        # Decentering an ideal lens by d moves the on-axis image point by d,
        # i.e. the line of sight deviates by d / f radians.
        optic = thin_lens(dx, dy)
        assert_allclose(
            ImageQualityOperand.boresight_x(optic, WAVELENGTH), 1e3 * dx / FOCAL_LENGTH
        )
        assert_allclose(
            ImageQualityOperand.boresight_y(optic, WAVELENGTH), 1e3 * dy / FOCAL_LENGTH
        )


@pytest.mark.skipif(
    "torch" not in be.list_available_backends(), reason="Requires the Torch backend"
)
@pytest.mark.parametrize("set_test_backend", ["torch"], indirect=True)
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("rms_wavefront", {"Hx": 0.0, "Hy": 0.0, "num_rays": 4}),
        ("mtf", {"Hx": 0.0, "Hy": 0.0, "frequency": 40.0, "num_rays": 32}),
    ],
)
def test_gradient_matches_finite_difference(set_test_backend, name, kwargs):
    import torch

    def evaluate(radius):
        optic = CookeTriplet()
        optic.surfaces[1].geometry.radius = radius
        return getattr(ImageQualityOperand, name)(
            optic, wavelength=WAVELENGTH, **kwargs
        )

    radius0 = float(CookeTriplet().surfaces[1].geometry.radius)
    radius = torch.tensor(radius0, dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(evaluate(radius), radius)

    step = 1e-4
    with torch.no_grad():
        upper = evaluate(torch.tensor(radius0 + step, dtype=torch.float64))
        lower = evaluate(torch.tensor(radius0 - step, dtype=torch.float64))
    assert_allclose(gradient, (upper - lower) / (2 * step), rtol=1e-4)


def test_tolerancing_add_operand(set_test_backend):
    optic = CookeTriplet()
    tolerancing = Tolerancing(optic)
    input_data = {
        "optic": optic,
        "Hx": 0.0,
        "Hy": 1.0,
        "num_rays": 4,
        "wavelength": "primary",
    }
    tolerancing.add_operand("rms_wavefront", input_data)
    tolerancing.add_operand(
        "mtf",
        {"optic": optic, "Hx": 0.0, "Hy": 0.0, "wavelength": 0.55, "frequency": 40.0},
    )
    tolerancing.add_operand("boresight_x", {"optic": optic, "wavelength": 0.55})

    rms, mtf, boresight = tolerancing.operands
    assert_allclose(rms.target, ImageQualityOperand.rms_wavefront(**input_data))
    assert_allclose(rms.value, rms.target)
    assert 0.0 < float(mtf.target) <= 1.0
    assert_allclose(boresight.target, 0.0, atol=1e-10)
