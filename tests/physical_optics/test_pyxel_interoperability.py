"""Optional monochromatic interoperability checks against ESA's real Pyxel.

References use exact SI constants independently of Optiland's implementation.
Float64 tolerances cover rounding in calibration, binning, and FFT convolution;
these deterministic tests do not validate shot noise, broadband response, or ADU.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RectangularAperture
from optiland.physical_optics import ScalarField
from optiland.physical_optics.interoperability import from_hcipy
from optiland.physical_optics.radiometry import (
    field_to_irradiance,
    irradiance_to_photon_rate,
    normalized_psf,
)
from optiland.physical_optics.train import ScalarOpticalTrain

pyxel = pytest.importorskip("pyxel", reason="Optional ESA pyxel-sim is not installed")
# The unrelated game engine also imports as `pyxel`; it is not this dependency.
detectors = pytest.importorskip(
    "pyxel.detectors", reason="ESA pyxel-sim detectors are required"
)
exposure = pytest.importorskip("pyxel.exposure")
charge_generation = pytest.importorskip("pyxel.models.charge_generation")
photon_collection = pytest.importorskip("pyxel.models.photon_collection")
pipelines = pytest.importorskip("pyxel.pipelines")
CCD = detectors.CCD
CCDGeometry = detectors.CCDGeometry
Characteristics = detectors.Characteristics
Environment = detectors.Environment
Exposure = exposure.Exposure
Readout = exposure.Readout
simple_conversion = charge_generation.simple_conversion
load_image = photon_collection.load_image
load_psf = photon_collection.load_psf
DetectionPipeline = pipelines.DetectionPipeline
ModelFunction = pipelines.ModelFunction
Processor = pipelines.Processor

PLANCK_J_S = 6.62607015e-34
LIGHT_M_S = 299792458.0
WAVELENGTH_NM = 500.0
DX_MM = 0.01
DY_MM = 0.02
IRRADIANCE_SCALE_W_PER_MM2 = 1e-12
QE = 0.4
RTOL = 1e-12
ATOL = 1e-12


def _detector(shape, binning=(1, 1), *, dx_mm=DX_MM, dy_mm=DY_MM):
    by, bx = binning
    return CCD(
        geometry=CCDGeometry(
            row=shape[0],
            col=shape[1],
            pixel_vert_size=dy_mm * by * 1000,
            pixel_horz_size=dx_mm * bx * 1000,
        ),
        environment=Environment(wavelength=WAVELENGTH_NM),
        characteristics=Characteristics(quantum_efficiency=QE),
    )


def _export_calibrated_marker(tmp_path, binning):
    # An asymmetric rectangle detects transposition and either-axis reversal.
    marker = np.arange(24, dtype=np.float64).reshape(4, 6)
    field = ScalarField(
        be.array(marker) * np.exp(0.37j),
        dx=DX_MM,
        dy=DY_MM,
        wavelength=WAVELENGTH_NM * 1e-6,  # nm -> mm, as required by the field
    )
    irradiance = field_to_irradiance(
        field, irradiance_scale_w_per_mm2=IRRADIANCE_SCALE_W_PER_MM2
    )
    rate = irradiance_to_photon_rate(
        irradiance,
        dx_mm=DX_MM,
        dy_mm=DY_MM,
        wavelength_nm=WAVELENGTH_NM,
        binning=binning,
    )

    # Independent SI calculation: W/m² * m² / J/photon, then explicit blocks.
    photon_energy_j = PLANCK_J_S * LIGHT_M_S / (WAVELENGTH_NM * 1e-9)
    intensity_w_per_m2 = marker**2 * IRRADIANCE_SCALE_W_PER_MM2 * 1e6
    cell_area_m2 = (DX_MM * 1e-3) * (DY_MM * 1e-3)
    expected_fine = intensity_w_per_m2 * cell_area_m2 / photon_energy_j
    by, bx = binning
    expected = np.array(
        [
            [
                expected_fine[y : y + by, x : x + bx].sum()
                for x in range(0, marker.shape[1], bx)
            ]
            for y in range(0, marker.shape[0], by)
        ]
    )
    exported = be.to_numpy(rate)
    assert exported.dtype == np.float64
    np.testing.assert_allclose(exported, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(exported.sum(), expected_fine.sum(), rtol=RTOL)
    path = tmp_path / "photon_rate.npy"
    np.save(path, exported, allow_pickle=False)
    return path, expected


@pytest.mark.parametrize("interval_s", [0.01, 0.025])
@pytest.mark.parametrize("binning", [(1, 1), (2, 2)])
def test_real_loader_calibrated_rates_orientation_and_qe(
    set_test_backend, tmp_path, interval_s, binning
):
    path, expected_rate = _export_calibrated_marker(tmp_path, binning)
    detector = _detector(expected_rate.shape, binning)
    detector.set_readout(times=[interval_s], non_destructive=False)
    # Standalone model calls do not advance the readout loop themselves.
    detector.readout_properties.time = interval_s
    detector.readout_properties.time_step = interval_s
    load_image(
        detector,
        image_file=str(path),
        convert_to_photons=False,
        time_scale=1.0,
    )
    expected_counts = expected_rate * interval_s
    assert detector.photon.array.shape == expected_rate.shape
    np.testing.assert_allclose(
        detector.photon.array, expected_counts, rtol=RTOL, atol=ATOL
    )
    simple_conversion(detector, binomial_sampling=False)
    np.testing.assert_allclose(
        detector.charge.array, expected_counts * QE, rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("entrypoint", ["run_mode", "processor"])
def test_timed_destructive_pipeline_reloads_rate_after_empty(
    set_test_backend, tmp_path, entrypoint
):
    binning = (2, 2)
    path, expected_rate = _export_calibrated_marker(tmp_path, binning)
    detector = _detector(expected_rate.shape, binning)
    # Exposure discards preloaded buckets; the loader must run inside the pipeline.
    detector.photon.array = np.full(expected_rate.shape, 1e9)
    detector.charge.add_charge_array(np.full(expected_rate.shape, 1e9))
    pipeline = DetectionPipeline(
        photon_collection=[
            ModelFunction(
                name="load_rate",
                func="pyxel.models.photon_collection.load_image",
                arguments={
                    "image_file": str(path),
                    "convert_to_photons": False,
                    "time_scale": 1.0,
                },
            )
        ],
        charge_generation=[
            ModelFunction(
                name="photoelectrons",
                func="pyxel.models.charge_generation.simple_conversion",
                arguments={"binomial_sampling": False},
            )
        ],
        charge_collection=[
            ModelFunction(
                name="collect",
                func="pyxel.models.charge_collection.simple_collection",
            )
        ],
    )
    times_s = [0.01, 0.035, 0.06]
    intervals_s = np.array([0.01, 0.025, 0.025])
    exposure = Exposure(readout=Readout(times=times_s, non_destructive=False))
    if entrypoint == "run_mode":
        result = pyxel.run_mode(mode=exposure, detector=detector, pipeline=pipeline)
    else:
        result = exposure.run_exposure(
            processor=Processor(detector=detector, pipeline=pipeline),
            debug=False,
            with_inherited_coords=True,
        )
    bucket = result["/bucket"]
    np.testing.assert_allclose(bucket["time"].to_numpy(), times_s, rtol=RTOL)
    expected_counts = intervals_s[:, None, None] * expected_rate[None, :, :]
    assert bucket["photon"].dims == ("time", "y", "x")
    np.testing.assert_allclose(
        bucket["photon"].to_numpy(), expected_counts, rtol=RTOL, atol=ATOL
    )
    for name in ("charge", "pixel"):
        np.testing.assert_allclose(
            bucket[name].to_numpy(), expected_counts * QE, rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("position", [(3, 4), (0, 0)], ids=["interior", "corner"])
def test_exported_normalized_psf_real_model_mean_filled_boundary(
    set_test_backend, tmp_path, position
):
    # This is a detector-sampled monochromatic shape, not a throughput estimate.
    intensity = np.array([[0.0, 1.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 3.0]])
    kernel = be.to_numpy(normalized_psf(be.array(intensity)))
    np.testing.assert_allclose(kernel, intensity / 10.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(kernel.sum(), 1.0, rtol=RTOL)
    kernel_path = tmp_path / "psf.npy"
    np.save(kernel_path, kernel, allow_pickle=False)
    stamp = np.zeros((7, 9), dtype=np.float64)
    stamp[position] = 100.0
    stamp_path = tmp_path / "stamp_rate.npy"
    np.save(stamp_path, stamp, allow_pickle=False)
    detector = _detector(stamp.shape)
    detector.set_readout(times=[1.0])
    detector.readout_properties.time_step = 1.0
    load_image(
        detector, image_file=str(stamp_path), convert_to_photons=False, time_scale=1.0
    )
    # Its FFT can produce tiny negative roundoff at zero-valued pixels; Pyxel
    # warns and clips those to zero, covered by the absolute tolerance below.
    load_psf(detector, filename=str(kernel_path), normalize_kernel=False)

    # Pyxel's 2D PSF model fills outside the stamp with its MEAN, not zero.
    # Use an explicit spatial convolution independent of its FFT implementation.
    padded = np.pad(stamp, 1, constant_values=stamp.mean())
    expected = np.empty_like(stamp)
    reference_kernel = (intensity / 10.0)[::-1, ::-1]
    for y in range(stamp.shape[0]):
        for x in range(stamp.shape[1]):
            expected[y, x] = np.sum(padded[y : y + 3, x : x + 3] * reference_kernel)
    np.testing.assert_allclose(detector.photon.array, expected, rtol=RTOL, atol=ATOL)
    if position == (3, 4):
        np.testing.assert_allclose(
            detector.photon.array[2:5, 3:6], 100 * intensity / 10, rtol=RTOL, atol=ATOL
        )
        # Mean-filled edges add flux; whole-stamp unit flux is NOT the contract.
        assert detector.photon.array.sum() > stamp.sum()


@pytest.mark.parametrize(
    "shape,offset_yx,expected_shift_yx",
    [
        ((15, 19), (0.0, 0.0), (0.0, 0.0)),
        ((16, 20), (0.0, 0.0), (-0.5, -0.5)),
        ((15, 20), (0.0, 0.0), (0.0, -0.5)),
        ((16, 19), (0.0, 0.0), (-0.5, 0.0)),
        ((15, 19), (1.0, -0.75), (0.0, 0.0)),
    ],
    ids=["odd", "even", "even-x", "even-y", "odd-offset"],
)
def test_real_psf_loader_native_center_registration(
    set_test_backend, tmp_path, shape, offset_yx, expected_shift_yx
):
    """Record native registration, including the even-grid half-pixel limitation."""
    ny, nx = shape
    # Independent physical axes: the optical origin bisects an even grid.
    y_mm = (np.arange(ny) - (ny - 1) / 2) * DY_MM
    x_mm = (np.arange(nx) - (nx - 1) / 2) * DX_MM
    intensity = np.exp(
        -0.5
        * (
            ((y_mm[:, None] / DY_MM - offset_yx[0]) / 2.0) ** 2
            + ((x_mm[None, :] / DX_MM - offset_yx[1]) / 3.0) ** 2
        )
    )
    reference = intensity / intensity.sum()
    kernel = be.to_numpy(normalized_psf(be.array(intensity)))
    np.testing.assert_allclose(kernel, reference, rtol=RTOL, atol=ATOL)
    physical_centroid_yx = np.array(
        [
            np.sum(reference * y_mm[:, None]) / DY_MM,
            np.sum(reference * x_mm[None, :]) / DX_MM,
        ]
    )
    if offset_yx == (0.0, 0.0):
        np.testing.assert_allclose(physical_centroid_yx, 0.0, rtol=0, atol=1e-10)
    else:
        # Retain a real physical displacement; do not force the centroid to zero.
        assert physical_centroid_yx[0] > 0.5
        assert physical_centroid_yx[1] < -0.5

    kernel_path = tmp_path / "registered_psf.npy"
    np.save(kernel_path, kernel, allow_pickle=False)
    stamp = np.zeros((65, 81), dtype=np.float64)
    position = (32, 40)
    stamp[position] = 100.0
    detector = _detector(stamp.shape)
    detector.photon.array = stamp
    # Exercise the actual .npy loader and default kernel normalization.
    load_psf(detector, filename=str(kernel_path))

    # Native convolution assigns zero offset to index N//2 on each axis.
    native_center_yx = np.array([ny // 2, nx // 2])
    physical_origin_yx = np.array([(ny - 1) / 2, (nx - 1) / 2])
    np.testing.assert_array_equal(
        physical_origin_yx - native_center_yx, expected_shift_yx
    )
    y0, x0 = np.array(position) - native_center_yx
    # Only the interior impulse response: mean-filled outer edges add flux and
    # must not contaminate its centroid or flux measurement.
    response = detector.photon.array[y0 : y0 + ny, x0 : x0 + nx]
    np.testing.assert_allclose(response, 100.0 * reference, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(response.sum(), 100.0, rtol=RTOL, atol=ATOL)
    detector_y = np.arange(y0, y0 + ny) - position[0]
    detector_x = np.arange(x0, x0 + nx) - position[1]
    measured_centroid_yx = (
        np.array(
            [
                np.sum(response * detector_y[:, None]),
                np.sum(response * detector_x[None, :]),
            ]
        )
        / response.sum()
    )
    np.testing.assert_allclose(
        measured_centroid_yx,
        physical_centroid_yx + expected_shift_yx,
        rtol=0,
        atol=1e-10,  # pixels; comfortably above float64 FFT rounding
    )


def test_hcipy_power_normalized_gaussian_train_to_timed_pyxel(
    set_test_backend, tmp_path
):
    hp = pytest.importorskip("hcipy", reason="This chain also requires optional HCIPy")
    ny, nx = 96, 128
    dx_mm, dy_mm, waist_mm = 0.003, 0.004, 0.032
    incident_power_w = 1e-9
    # Both window half-widths exceed five waists; at least eight samples/waist.
    assert min(nx * dx_mm, ny * dy_mm) / 2 > 5 * waist_mm
    assert waist_mm / max(dx_mm, dy_mm) >= 8
    x_mm = (np.arange(nx) - (nx - 1) / 2) * dx_mm
    y_mm = (np.arange(ny) - (ny - 1) / 2) * dy_mm
    xx_mm, yy_mm = np.meshgrid(x_mm, y_mm)
    radius_squared_mm2 = xx_mm**2 + yy_mm**2
    gaussian = np.exp(-radius_squared_mm2 / waist_mm**2)
    cell_area_m2 = (dx_mm * 1e-3) * (dy_mm * 1e-3)
    amplitude_scale = np.sqrt(incident_power_w / (np.sum(gaussian**2) * cell_area_m2))
    # Explicit sqrt(W/m²), not an arbitrary SI electric field in V/m.
    samples_m = amplitude_scale * gaussian * np.exp(0.31j)
    grid = hp.make_uniform_grid(
        [nx, ny], [nx * dx_mm * 1e-3, ny * dy_mm * 1e-3], has_center=False
    )
    source_field = hp.Field(samples_m.ravel(), grid)
    assert isinstance(source_field, np.ndarray)  # Legacy NumPy HCIPy source.
    source = hp.Wavefront(source_field, wavelength=WAVELENGTH_NM * 1e-9)
    np.testing.assert_allclose(source.total_power, incident_power_w, rtol=RTOL, atol=0)
    imported = from_hcipy(source)
    assert imported.shape == (ny, nx)
    np.testing.assert_allclose(
        [imported.dx, imported.dy, imported.wavelength],
        [dx_mm, dy_mm, WAVELENGTH_NM * 1e-6],
        rtol=RTOL,
        atol=0,
    )
    np.testing.assert_allclose(
        be.to_numpy(imported.data), samples_m / 1000, rtol=RTOL, atol=1e-18
    )
    np.testing.assert_allclose(
        be.to_numpy(imported.power), incident_power_w, rtol=RTOL, atol=0
    )

    x_min, x_max, y_min, y_max = -0.037, 0.021, -0.023, 0.041
    mask = (xx_mm >= x_min) & (xx_mm <= x_max) & (yy_mm >= y_min) & (yy_mm <= y_max)
    optic = Optic()
    optic.surfaces.add(index=0, thickness=np.inf, material="air")
    radius_mm = 25.0
    optic.surfaces.add(
        index=1,
        z=0,
        radius=radius_mm,
        conic=-1,
        material=IdealMaterial(1.5),
        aperture=RectangularAperture(x_min, x_max, y_min, y_max),
    )
    # Zero-gap plano exit: phase-screen lens, no FFT or propagation truncation.
    # It is lossless apart from the aperture; Fresnel reflection is not modeled.
    optic.surfaces.add(index=2, z=0, surface_type="plane", material="air")
    output = ScalarOpticalTrain.from_optic(optic).propagate(imported)
    assert output.refractive_index == 1.0
    expected_phase = (
        2
        * np.pi
        / (WAVELENGTH_NM * 1e-6)
        * (1 - 1.5)
        * radius_squared_mm2
        / (2 * radius_mm)
    )
    expected_field_mm = samples_m / 1000 * mask * np.exp(1j * expected_phase)
    np.testing.assert_allclose(
        be.to_numpy(output.data), expected_field_mm, rtol=RTOL, atol=1e-18
    )
    # Independent aperture quadrature, with no normalization after clipping.
    retained_power_w = np.sum(amplitude_scale**2 * gaussian**2 * mask) * cell_area_m2
    assert 0 < retained_power_w < incident_power_w
    np.testing.assert_allclose(
        be.to_numpy(output.power), retained_power_w, rtol=RTOL, atol=0
    )
    irradiance = field_to_irradiance(output, irradiance_scale_w_per_mm2=1.0)
    rate = irradiance_to_photon_rate(
        irradiance, dx_mm=dx_mm, dy_mm=dy_mm, wavelength_nm=WAVELENGTH_NM
    )
    photon_energy_j = PLANCK_J_S * LIGHT_M_S / (WAVELENGTH_NM * 1e-9)
    expected_rate = (
        amplitude_scale**2 * gaussian**2 * mask * cell_area_m2 / photon_energy_j
    )
    exported_rate = be.to_numpy(rate)
    np.testing.assert_allclose(exported_rate, expected_rate, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        exported_rate.sum(), retained_power_w / photon_energy_j, rtol=RTOL, atol=0
    )
    path = tmp_path / "hcipy_train_photon_rate.npy"
    np.save(path, exported_rate, allow_pickle=False)
    pipeline = DetectionPipeline(
        photon_collection=[
            ModelFunction(
                name="load_rate",
                func="pyxel.models.photon_collection.load_image",
                arguments={
                    "image_file": str(path),
                    "convert_to_photons": False,
                    "time_scale": 1.0,
                },
            )
        ],
        charge_generation=[
            ModelFunction(
                name="photoelectrons",
                func="pyxel.models.charge_generation.simple_conversion",
                arguments={"binomial_sampling": False},
            )
        ],
        charge_collection=[
            ModelFunction(
                name="collect",
                func="pyxel.models.charge_collection.simple_collection",
            )
        ],
    )
    detector = _detector((ny, nx), dx_mm=dx_mm, dy_mm=dy_mm)
    times_s = [0.01, 0.035]
    intervals_s = np.array([0.01, 0.025])
    processor = Processor(detector=detector, pipeline=pipeline)
    result = Exposure(
        readout=Readout(times=times_s, non_destructive=False)
    ).run_exposure(processor=processor, debug=False, with_inherited_coords=True)
    np.testing.assert_allclose(detector.time_step, intervals_s[-1], rtol=RTOL, atol=0)
    bucket = result["/bucket"]
    np.testing.assert_allclose(bucket["time"].to_numpy(), times_s, rtol=RTOL, atol=0)
    assert bucket["photon"].dims == ("time", "y", "x")
    expected_counts = intervals_s[:, None, None] * expected_rate[None, :, :]
    np.testing.assert_allclose(
        bucket["photon"].to_numpy(), expected_counts, rtol=RTOL, atol=ATOL
    )
    for name in ("charge", "pixel"):
        np.testing.assert_allclose(
            bucket[name].to_numpy(), expected_counts * QE, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            bucket[name].sum(dim=("y", "x")).to_numpy(),
            QE * retained_power_w / photon_energy_j * intervals_s,
            rtol=RTOL,
            atol=0,
        )
