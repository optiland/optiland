"""Absolute power through a native asphere and explicitly lossy prescription.

The reference is sampled entrance-mask quadrature followed by uniform axial
Beer--Lambert loss, not an angle/sag-dependent path-length model. All sampled
FFT modes propagate on this grid. Float64 tolerances cover rounding, not a
continuous-aperture integral or a physical validation of the paraxial screen.
Independent complex asphere-screen and exact-erf quadrature checks live in
test_scalar_optical_train_aspheres.py and test_aperture_quadrature.py.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import optiland.backend as be
from optiland.geometries.even_asphere import EvenAsphere
from optiland.materials import IdealMaterial, Material
from optiland.optic import Optic
from optiland.physical_apertures import RectangularAperture
from optiland.physical_optics import ScalarField
from optiland.physical_optics.interoperability import from_hcipy
from optiland.physical_optics.radiometry import (
    field_to_irradiance,
    irradiance_to_photon_rate,
)
from optiland.physical_optics.train import ScalarOpticalTrain

# Exact SI definitions, independent of the radiometry implementation.
PLANCK_J_S = 6.62607015e-34
LIGHT_M_S = 299792458.0
WAVELENGTH_NM = 500.0
RTOL = 2e-12
RATE_ATOL = 1e-12  # photons/s or expected counts; accommodates near-zero cells


def test_default_rejects_catalog_and_tiny_synthetic_extinction(set_test_backend):
    """Neither a small catalog kappa nor an arbitrary tolerance implies lossless."""
    field = ScalarField(be.ones((8, 12)) * (1 + 0j), dx=0.01, wavelength=0.0005)
    materials = (
        Material("SK16", reference="hikari", match_policy="strict"),
        IdealMaterial(1.5, k=1e-15),
    )
    for material in materials:
        kappa = be.to_numpy(material.k(0.5)).item()
        assert kappa > 0
        optic = Optic()
        optic.surfaces.add(index=0, thickness=np.inf, material="air")
        optic.surfaces.add(index=1, z=0, surface_type="plane", material=material)
        optic.surfaces.add(index=2, z=0.013, surface_type="plane", material="air")
        # The wavelength-dependent rejection may happen at construction or at
        # propagation; the default is exact rejection, not an isclose test.
        with pytest.raises(ValueError, match="lossless"):
            ScalarOpticalTrain.from_optic(optic).propagate(field)
        output = ScalarOpticalTrain.from_optic(optic, absorption="axial").propagate(
            field
        )
        expected_transmission = np.exp(-4 * np.pi * kappa * 0.013 / field.wavelength)
        np.testing.assert_allclose(
            be.to_numpy(output.power / field.power),
            expected_transmission,
            rtol=RTOL,
            atol=0,
        )


def test_hcipy_lossy_asphere_to_binned_rates_and_timed_pyxel(
    set_test_backend, tmp_path, monkeypatch
):
    """Actual optional packages, two CPU backends, and no upstream exposure/QE."""
    hp = pytest.importorskip("hcipy", reason="This chain requires optional HCIPy")
    pytest.importorskip("pyxel", reason="Optional ESA pyxel-sim is not installed")
    # An unrelated game engine has the same top-level name. Require ESA's
    # detector module before reusing the existing real-Pyxel test helpers.
    pytest.importorskip("pyxel.detectors", reason="ESA pyxel-sim detectors required")
    from tests.physical_optics.test_pyxel_interoperability import (
        QE,
        DetectionPipeline,
        Exposure,
        ModelFunction,
        Processor,
        Readout,
        _detector,
    )

    # Mixed even-y/odd-x rectangular sampling, unequal pitches, and exact 2x3
    # incoherent bins. The sampled HCIPy power is 1 nW, not an infinite Gaussian
    # integral; the independent reference below uses these same known samples.
    ny, nx = 96, 129
    dx_mm, dy_mm, waist_mm = 0.003, 0.004, 0.032
    center_mm = (0.009, -0.006)
    beam_center_mm = (0.008, -0.011)
    incident_power_w = 1e-9
    binning = (2, 3)
    assert ny % 2 == 0 and nx % 2 == 1
    assert ny % binning[0] == nx % binning[1] == 0
    assert min(nx * dx_mm, ny * dy_mm) / 2 > 5 * waist_mm
    assert waist_mm / max(dx_mm, dy_mm) >= 8
    wavelength_mm = WAVELENGTH_NM * 1e-6
    wavelength_m = WAVELENGTH_NM * 1e-9
    photon_energy_j = PLANCK_J_S * LIGHT_M_S / wavelength_m
    x_mm = center_mm[0] + (np.arange(nx) - (nx - 1) / 2) * dx_mm
    y_mm = center_mm[1] + (np.arange(ny) - (ny - 1) / 2) * dy_mm
    xx_mm, yy_mm = np.meshgrid(x_mm, y_mm)
    gaussian = np.exp(
        -((xx_mm - beam_center_mm[0]) ** 2 + (yy_mm - beam_center_mm[1]) ** 2)
        / waist_mm**2
    )
    cell_area_m2 = (dx_mm * 1e-3) * (dy_mm * 1e-3)
    amplitude_scale = np.sqrt(incident_power_w / (np.sum(gaussian**2) * cell_area_m2))
    known_intensity_w_per_m2 = amplitude_scale**2 * gaussian**2
    samples_m = amplitude_scale * gaussian * np.exp(1j * (0.31 + 9 * xx_mm - 7 * yy_mm))
    grid = hp.make_uniform_grid(
        [nx, ny],
        [nx * dx_mm * 1e-3, ny * dy_mm * 1e-3],
        center=np.asarray(center_mm) * 1e-3,
        has_center=False,
    )
    source = hp.Wavefront(hp.Field(samples_m.ravel(), grid), wavelength=wavelength_m)
    source_samples_before = np.array(source.electric_field, copy=True)
    grid_delta_before, grid_zero_before = grid.delta.copy(), grid.zero.copy()
    np.testing.assert_allclose(source.total_power, incident_power_w, rtol=RTOL, atol=0)
    imported = from_hcipy(source)
    imported_before = be.copy(imported.data)
    assert imported.shape == (ny, nx)
    np.testing.assert_allclose(
        [imported.dx, imported.dy, imported.wavelength, *imported.center],
        [dx_mm, dy_mm, wavelength_mm, *center_mm],
        rtol=RTOL,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        be.to_numpy(imported.data), samples_m / 1000, rtol=RTOL, atol=1e-18
    )
    np.testing.assert_allclose(
        be.to_numpy(imported.power), incident_power_w, rtol=RTOL, atol=0
    )
    assert be.to_numpy(imported.data).dtype == np.complex128
    if be.get_backend() == "torch":
        assert imported.data.device.type == "cpu"

    # Physical aperture and native screen share the entrance vertex. There is
    # no propagation before clipping, and no implicit ray-launch stop aperture.
    bounds_mm = (-0.037, 0.021, -0.023, 0.041)
    x_min, x_max, y_min, y_max = bounds_mm
    mask = (xx_mm >= x_min) & (xx_mm <= x_max) & (yy_mm >= y_min) & (yy_mm <= y_max)
    retained_entrance_power_w = np.sum(known_intensity_w_per_m2 * mask) * cell_area_m2
    assert 0 < retained_entrance_power_w < incident_power_w
    # Weak passive ideal slabs with different real indices and kappas, then
    # an air gap to the detector marker. Loss uses vacuum wavelength and the
    # outgoing medium of each vertex gap, without an additional index factor.
    layers = ((1.5, 2e-4, 0.017), (1.3, 3e-4, 0.023), (1.0, 0.0, 0.019))
    optic = Optic()
    optic.surfaces.add(index=0, thickness=np.inf, material="air")
    optic.surfaces.add(
        index=1,
        z=0,
        surface_type="even_asphere",
        radius=25.0,
        conic=-0.7,
        coefficients=[0.0004, -0.002, 0.0008],
        material=IdealMaterial(layers[0][0], k=layers[0][1]),
        aperture=RectangularAperture(*bounds_mm),
    )
    z_mm = layers[0][2]
    for index, (n, kappa, gap_mm) in enumerate(layers[1:], start=2):
        optic.surfaces.add(
            index=index, z=z_mm, surface_type="plane", material=IdealMaterial(n, kappa)
        )
        z_mm += gap_mm
    optic.surfaces.add(index=4, z=z_mm, surface_type="plane", material="air")
    assert type(optic.surfaces[1].geometry) is EvenAsphere
    # Even the air-side Nyquist corner is propagating, so FFT propagation has
    # unit-modulus transfer on every retained mode; only clipping/loss removes power.
    assert (np.pi / dx_mm) ** 2 + (np.pi / dy_mm) ** 2 < (
        2 * np.pi / wavelength_mm
    ) ** 2
    extinction_length_mm = sum(kappa * gap for _, kappa, gap in layers)
    expected_power_w = retained_entrance_power_w * np.exp(
        -4 * np.pi * extinction_length_mm / wavelength_mm
    )
    assert 0 < expected_power_w < retained_entrance_power_w

    materials = [surface.material_post for surface in optic.surfaces]
    for material in materials:
        material.n(0.61)
        material.k(0.61)
    prescription_before = deepcopy(optic.to_dict())
    caches_before = [
        (material._n_cache.copy(), material._k_cache.copy(), material._cache_context)
        for material in materials
    ]
    geometry = optic.surfaces[1].geometry
    geometry_before = vars(geometry).copy()
    coefficients_before = tuple(geometry.coefficients)
    aperture_before = vars(optic.surfaces[1].aperture).copy()
    train = ScalarOpticalTrain.from_optic(optic, absorption="axial")
    train_before = vars(train).copy()
    lookup_calls = []
    native_n, native_k = IdealMaterial.n, IdealMaterial.k

    def checked_n(self, wavelength, **kwargs):
        lookup_calls.append(("n", wavelength))
        return native_n(self, wavelength, **kwargs)

    def checked_k(self, wavelength, **kwargs):
        lookup_calls.append(("k", wavelength))
        return native_k(self, wavelength, **kwargs)

    with monkeypatch.context() as lookup_patch:
        lookup_patch.setattr(IdealMaterial, "n", checked_n)
        lookup_patch.setattr(IdealMaterial, "k", checked_k)
        with pytest.raises(ValueError, match="lossless"):
            ScalarOpticalTrain.from_optic(optic).propagate(imported)
        output = train.propagate(imported)
    assert {name for name, _ in lookup_calls} == {"n", "k"}
    # Small scalar metadata is inspected; no sampled field is downloaded for
    # material preflight. Fixed IdealMaterial alone would not detect wrong units.
    np.testing.assert_allclose(
        [wavelength for _, wavelength in lookup_calls],
        WAVELENGTH_NM * 1e-3,  # nm -> um, NOT the mm field wavelength
        rtol=RTOL,
        atol=0,
    )
    assert output.shape == imported.shape and output.data.dtype == imported.data.dtype
    assert output.center == imported.center
    assert (output.dx, output.dy, output.wavelength) == (
        imported.dx,
        imported.dy,
        imported.wavelength,
    )
    assert output.refractive_index == 1.0
    if be.get_backend() == "torch":
        assert output.data.device == imported.data.device
    np.testing.assert_allclose(
        be.to_numpy(output.power), expected_power_w, rtol=RTOL, atol=0
    )

    # Keep the HCIPy import's absolute calibration of sqrt(W/mm^2). The pixel
    # reference uses actual complex output amplitudes, with explicit incoherent
    # block sums independent of the radiometry implementation's reshape/reductions.
    output_samples_mm = be.to_numpy(output.data)
    expected_irradiance = np.abs(output_samples_mm) ** 2
    irradiance = field_to_irradiance(output, irradiance_scale_w_per_mm2=1.0)
    np.testing.assert_allclose(
        be.to_numpy(irradiance), expected_irradiance, rtol=RTOL, atol=0
    )
    rate = irradiance_to_photon_rate(
        irradiance,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        wavelength_nm=WAVELENGTH_NM,
        binning=binning,
    )
    fine_rate = expected_irradiance * dx_mm * dy_mm / photon_energy_j
    by, bx = binning
    expected_rate = np.array(
        [
            [fine_rate[y : y + by, x : x + bx].sum() for x in range(0, nx, bx)]
            for y in range(0, ny, by)
        ]
    )
    exported_rate = be.to_numpy(rate)
    assert exported_rate.shape == (48, 43) and exported_rate.dtype == np.float64
    np.testing.assert_allclose(exported_rate, expected_rate, rtol=RTOL, atol=RATE_ATOL)
    np.testing.assert_allclose(
        exported_rate.sum(), expected_power_w / photon_energy_j, rtol=RTOL, atol=0
    )
    path = tmp_path / "lossy_prescription_photon_rate.npy"
    np.save(path, exported_rate, allow_pickle=False)
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), exported_rate)

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
                name="collect", func="pyxel.models.charge_collection.simple_collection"
            )
        ],
    )
    detector = _detector(expected_rate.shape, binning, dx_mm=dx_mm, dy_mm=dy_mm)
    np.testing.assert_allclose(
        [detector.geometry.pixel_vert_size, detector.geometry.pixel_horz_size],
        [dy_mm * by * 1000, dx_mm * bx * 1000],
        rtol=RTOL,
    )
    # Exposure clears preloaded buckets: photons/s must be loaded inside each
    # destructive-readout pipeline, not multiplied by QE or duration beforehand.
    detector.photon.array = np.full(expected_rate.shape, 1e9)
    detector.charge.add_charge_array(np.full(expected_rate.shape, 1e9))
    absolute_readout_times_s = [0.01, 0.035, 0.06]
    intervals_s = np.array([0.01, 0.025, 0.025])
    result = Exposure(
        readout=Readout(times=absolute_readout_times_s, non_destructive=False)
    ).run_exposure(
        processor=Processor(detector=detector, pipeline=pipeline),
        debug=False,
        with_inherited_coords=True,
    )
    bucket = result["/bucket"]
    np.testing.assert_allclose(
        bucket["time"].to_numpy(), absolute_readout_times_s, rtol=RTOL, atol=0
    )
    np.testing.assert_allclose(detector.time_step, intervals_s[-1], rtol=RTOL, atol=0)
    assert bucket["photon"].dims == ("time", "y", "x")
    expected_counts = intervals_s[:, None, None] * expected_rate[None, :, :]
    np.testing.assert_allclose(
        bucket["photon"].to_numpy(), expected_counts, rtol=RTOL, atol=RATE_ATOL
    )
    for name in ("charge", "pixel"):
        np.testing.assert_allclose(
            bucket[name].to_numpy(), expected_counts * QE, rtol=RTOL, atol=RATE_ATOL
        )
    for name, efficiency in (("photon", 1.0), ("charge", QE), ("pixel", QE)):
        np.testing.assert_allclose(
            bucket[name].sum(dim=("y", "x")).to_numpy(),
            efficiency * expected_power_w / photon_energy_j * intervals_s,
            rtol=RTOL,
            atol=0,
        )

    # The original HCIPy source, imported field, train, prescription parameters,
    # and source material caches remain read-only throughout the complete chain.
    np.testing.assert_array_equal(
        np.asarray(source.electric_field), source_samples_before
    )
    np.testing.assert_array_equal(grid.delta, grid_delta_before)
    np.testing.assert_array_equal(grid.zero, grid_zero_before)
    assert source.wavelength == wavelength_m
    np.testing.assert_allclose(source.total_power, incident_power_w, rtol=RTOL, atol=0)
    np.testing.assert_array_equal(
        be.to_numpy(imported.data), be.to_numpy(imported_before)
    )
    assert optic.to_dict() == prescription_before
    assert vars(train).keys() == train_before.keys()
    for name, value in train_before.items():
        assert getattr(train, name) is value
    for name, value in geometry_before.items():
        assert getattr(geometry, name) is value
    for before, after in zip(coefficients_before, geometry.coefficients, strict=True):
        assert before is after
    for name, value in aperture_before.items():
        assert getattr(optic.surfaces[1].aperture, name) is value
    for material, (n_cache, k_cache, context) in zip(
        materials, caches_before, strict=True
    ):
        assert material._cache_context == context
        for cache, original in (
            (material._n_cache, n_cache),
            (material._k_cache, k_cache),
        ):
            assert cache.keys() == original.keys()
            for key, value in original.items():
                assert cache[key] is value
