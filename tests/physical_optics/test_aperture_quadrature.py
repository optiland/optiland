"""Physical-power checks against Gaussian rectangle integrals, not mask sums.

Samples are point evaluations of a waist-plane Gaussian, not cell averages.
RectangularAperture.contains includes its boundaries; all stops here lie on
cell edges, at least half a pitch from samples. Thus the clipped sum is genuine
midpoint quadrature, without a fractional-area or inclusive-edge correction.
Threefold nested refinements preserve each grid's odd/even parity. Window
expansion is checked separately, since reducing truncation cannot eliminate
the fixed-pitch quadrature error. No diffraction or reflection is involved.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

import optiland.backend as be
from optiland.optic import Optic
from optiland.physical_apertures import RectangularAperture
from optiland.physical_optics.interoperability import from_hcipy
from optiland.physical_optics.radiometry import (
    field_to_irradiance,
    irradiance_to_photon_rate,
)
from optiland.physical_optics.train import ScalarOpticalTrain

WAIST_MM = 0.05
BEAM_CENTER_MM = (0.010, -0.007)
CAPTURED_POWER_W = 1e-9
WAVELENGTH_NM = 500.0
# Exact SI definitions, independent of the radiometry module's constants.
PHOTON_ENERGY_J = 6.62607015e-34 * 299792458.0 / (WAVELENGTH_NM * 1e-9)
ROUNDING_RTOL = 1e-12


def _axis_integral(lower, upper, beam_center):
    """Integral in mm of exp(-2*(coordinate-beam_center)**2/waist**2)."""
    scale = np.sqrt(2) / WAIST_MM
    return (
        WAIST_MM
        * np.sqrt(np.pi / 8)
        * (erf(scale * (upper - beam_center)) - erf(scale * (lower - beam_center)))
    )


def _rectangle_integral_and_bound(bounds, dx, dy):
    """Independent integral and a conservative midpoint error bound, in mm².

    For the 1D Gaussian, sup(abs(f'')) <= 4/waist². Composite midpoint error
    is at most length*pitch²*sup(abs(f''))/24. The product bound includes
    both axis errors and their cross term. Bounds may broadcast over pixels.
    """
    x_min, x_max, y_min, y_max = bounds
    ix = _axis_integral(x_min, x_max, BEAM_CENTER_MM[0])
    iy = _axis_integral(y_min, y_max, BEAM_CENTER_MM[1])
    ex = (x_max - x_min) * dx**2 / (6 * WAIST_MM**2)
    ey = (y_max - y_min) * dy**2 / (6 * WAIST_MM**2)
    return ix * iy, ex * iy + ey * ix + ex * ey


def _window_bounds(window, center):
    lx, ly = window
    cx, cy = center
    return cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2


def _measure_imported_gaussian(shape, window, center, stop, binning):
    """Exercise existing import, plane aperture, and radiometry interfaces."""
    hp = pytest.importorskip("hcipy", reason="Gaussian import requires optional HCIPy")
    ny, nx = shape
    lx, ly = window
    dx, dy = lx / nx, ly / ny
    cx, cy = center
    x = cx + (np.arange(nx) - (nx - 1) / 2) * dx
    y = cy + (np.arange(ny) - (ny - 1) / 2) * dy
    gaussian = np.exp(
        -((x[None, :] - BEAM_CENTER_MM[0]) ** 2 + (y[:, None] - BEAM_CENTER_MM[1]) ** 2)
        / WAIST_MM**2
    )
    # Normalize the captured sample domain to 1 nW, NOT the infinite beam.
    # The continuous reference therefore divides by the finite-window integral.
    cell_area_m2 = dx * dy * 1e-6
    samples_m = (
        gaussian
        * np.sqrt(CAPTURED_POWER_W / (np.sum(gaussian**2) * cell_area_m2))
        * np.exp(0.31j)
    )  # sqrt(W/m²), never an electric phasor in V/m
    grid = hp.CartesianGrid(
        hp.RegularCoords(
            np.array([dx, dy]) * 1e-3,
            [nx, ny],
            np.array([x[0], y[0]]) * 1e-3,
        )
    )
    source = hp.Wavefront(
        hp.Field(samples_m.ravel(), grid), wavelength=WAVELENGTH_NM * 1e-9
    )
    field = from_hcipy(source)
    np.testing.assert_allclose(
        source.total_power, CAPTURED_POWER_W, rtol=ROUNDING_RTOL, atol=0
    )
    np.testing.assert_allclose(
        be.to_numpy(field.power), CAPTURED_POWER_W, rtol=ROUNDING_RTOL, atol=0
    )
    np.testing.assert_allclose(field.center, center, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        field.wavelength, WAVELENGTH_NM * 1e-6, rtol=ROUNDING_RTOL, atol=0
    )
    actual_x, actual_y = field.coordinates()
    np.testing.assert_allclose(be.to_numpy(actual_x), x, rtol=0, atol=1e-15)
    np.testing.assert_allclose(be.to_numpy(actual_y), y, rtol=0, atol=1e-15)
    for axis, edges, pitch in ((x, stop[:2], dx), (y, stop[2:], dy)):
        for edge in edges:
            assert np.min(np.abs(axis - edge)) >= 0.49 * pitch

    optic = Optic()
    optic.surfaces.add(index=0, thickness=np.inf, material="air")
    optic.surfaces.add(
        index=1,
        z=0,
        surface_type="plane",
        material="air",
        aperture=RectangularAperture(*stop),
    )
    output = ScalarOpticalTrain.from_optic(optic).propagate(field)
    assert output.shape == shape
    assert output.center == field.center
    np.testing.assert_allclose([output.dx, output.dy], [dx, dy], rtol=ROUNDING_RTOL)
    irradiance = field_to_irradiance(output, irradiance_scale_w_per_mm2=1.0)
    fine_rate = irradiance_to_photon_rate(
        irradiance, dx_mm=dx, dy_mm=dy, wavelength_nm=WAVELENGTH_NM
    )
    rate = be.to_numpy(
        irradiance_to_photon_rate(
            irradiance,
            dx_mm=dx,
            dy_mm=dy,
            wavelength_nm=WAVELENGTH_NM,
            binning=binning,
        )
    )
    measured_power = float(be.to_numpy(output.power))
    assert 0 < measured_power < CAPTURED_POWER_W
    np.testing.assert_allclose(
        rate.sum(), measured_power / PHOTON_ENERGY_J, rtol=ROUNDING_RTOL, atol=0
    )
    np.testing.assert_allclose(
        rate.sum(), be.to_numpy(fine_rate).sum(), rtol=ROUNDING_RTOL, atol=0
    )

    window_bounds = _window_bounds(window, center)
    iw, ew = _rectangle_integral_and_bound(window_bounds, dx, dy)
    ia, ea = _rectangle_integral_and_bound(stop, dx, dy)
    assert iw > ew
    expected_power = CAPTURED_POWER_W * ia / iw
    power_bound = CAPTURED_POWER_W * (ea + ia / iw * ew) / (iw - ew)
    assert abs(measured_power - expected_power) <= (
        power_bound + ROUNDING_RTOL * CAPTURED_POWER_W
    )

    # Independent physical bin edges, centered at the imported plane center.
    # Pitch refinements align stops to bins; the window-expansion cases also
    # exercise bins partially clipped at stops (still whole fine sample cells).
    by, bx = binning
    x_edges = cx - lx / 2 + np.arange(nx // bx + 1) * bx * dx
    y_edges = cy - ly / 2 + np.arange(ny // by + 1) * by * dy
    # Mathematically identical edge formulas can differ by a few ulps. Snap
    # only those coincidences before integrating, rather than inventing a tiny
    # transmitted sliver in a dark bin and loosening its photon-rate tolerance.
    edge_roundoff_mm = 8 * np.finfo(float).eps * max(window)
    for edges, boundaries in ((x_edges, stop[:2]), (y_edges, stop[2:])):
        for boundary in boundaries:
            coincident = np.isclose(edges, boundary, rtol=0, atol=edge_roundoff_mm)
            edges[coincident] = boundary
    np.testing.assert_allclose((x_edges[0] + x_edges[-1]) / 2, cx, atol=1e-15)
    np.testing.assert_allclose((y_edges[0] + y_edges[-1]) / 2, cy, atol=1e-15)
    x_lo = np.clip(x_edges[:-1], stop[0], stop[1])[None, :]
    x_hi = np.clip(x_edges[1:], stop[0], stop[1])[None, :]
    y_lo = np.clip(y_edges[:-1], stop[2], stop[3])[:, None]
    y_hi = np.clip(y_edges[1:], stop[2], stop[3])[:, None]
    pixel_integral, pixel_error = _rectangle_integral_and_bound(
        (x_lo, x_hi, y_lo, y_hi), dx, dy
    )
    expected_rate = CAPTURED_POWER_W * pixel_integral / iw / PHOTON_ENERGY_J
    rate_bound = (
        CAPTURED_POWER_W
        / PHOTON_ENERGY_J
        * (pixel_error + pixel_integral / iw * ew)
        / (iw - ew)
    )
    assert rate.shape == (ny // by, nx // bx)
    # Relative float64 rounding allowance; the 1 photon/s floor only protects
    # analytically dark pixels from sub-picophoton/s floating-point noise.
    rounding_rate = ROUNDING_RTOL * np.maximum(expected_rate, 1.0)
    assert np.all(np.abs(rate - expected_rate) <= rate_bound + rounding_rate)
    np.testing.assert_allclose(
        expected_rate.sum(),
        expected_power / PHOTON_ENERGY_J,
        rtol=ROUNDING_RTOL,
        atol=0,
    )
    return measured_power, expected_power, rate


@pytest.mark.parametrize(
    "base_shape,center",
    [
        ((15, 21), (0.0, 0.0)),
        ((18, 24), (0.0, 0.0)),
        ((15, 24), (0.0, 0.0)),
        ((18, 21), (0.0, 0.0)),
        ((15, 21), (0.013, -0.009)),
    ],
    ids=["odd", "even", "even-x", "even-y", "shifted"],
)
def test_fixed_window_aperture_pitch_converges(set_test_backend, base_shape, center):
    window = (0.144, 0.120)  # mm; fixed for all three pitch refinements
    lx, ly = window
    cx, cy = center
    stop = (cx - lx / 6, cx + lx / 6, cy - ly / 6, cy + ly / 6)
    relative_errors = []
    for refinement in (1, 3, 9):
        shape = tuple(size * refinement for size in base_shape)
        measured, reference, _ = _measure_imported_gaussian(
            shape, window, center, stop, binning=(refinement, refinement)
        )
        relative_errors.append(abs(measured - reference) / reference)
    # Aligned stop boundaries remove discontinuous edge-phase oscillations.
    # The smooth midpoint O(pitch²) term should shrink by 3², not merely agree
    # across backends. Broad ratios allow higher-order terms on the coarse grid.
    ratios = np.array(relative_errors[:-1]) / relative_errors[1:]
    assert np.all((ratios > 8) & (ratios < 10)), (relative_errors, ratios)
    assert relative_errors[-1] < 7e-5, relative_errors


def test_fixed_pitch_window_truncation_is_separate(set_test_backend):
    center = (0.0, 0.0)
    stop = (-0.024, 0.024, -0.020, 0.020)  # mm; fixed physical stop
    ia, _ = _rectangle_integral_and_bound(stop, 0.0015, 0.002)
    infinite_reference = CAPTURED_POWER_W * ia / (np.pi * WAIST_MM**2 / 2)
    tail_biases, quadrature_errors, total_errors = [], [], []
    for expansion in (1, 2, 3):
        measured, finite_reference, _ = _measure_imported_gaussian(
            (60 * expansion, 96 * expansion),
            (0.144 * expansion, 0.120 * expansion),
            center,
            stop,
            binning=(2, 3),  # constant dy=2 µm, dx=1.5 µm and detector pitch
        )
        tail_biases.append(
            abs(finite_reference - infinite_reference) / infinite_reference
        )
        quadrature_errors.append(abs(measured - finite_reference) / finite_reference)
        total_errors.append(abs(measured - infinite_reference) / infinite_reference)
    assert 0.02 < tail_biases[0] < 0.03
    assert tail_biases[1] < 4e-6
    assert tail_biases[2] < 3e-12
    assert max(quadrature_errors) < 4e-4
    assert total_errors[0] > 0.02
    assert total_errors[-1] < 4e-4
    # Expanding a resolved window reaches a nonzero fixed-pitch error floor;
    # do not demand convergence to roundoff or assert blind monotonicity.
    assert 2e-4 < quadrature_errors[-1] < 4e-4


def test_converged_aperture_power_to_real_pyxel(set_test_backend, tmp_path):
    detectors = pytest.importorskip("pyxel.detectors", reason="Optional ESA Pyxel")
    models = pytest.importorskip("pyxel.models.photon_collection")
    charge = pytest.importorskip("pyxel.models.charge_generation")
    center = (0.013, -0.009)
    window = (0.144, 0.120)
    stop = (center[0] - 0.024, center[0] + 0.024, center[1] - 0.020, center[1] + 0.020)
    measured, reference, rate = _measure_imported_gaussian(
        (135, 189), window, center, stop, binning=(9, 9)
    )
    path = tmp_path / "aperture_photon_rate.npy"
    np.save(path, rate, allow_pickle=False)
    qe, interval_s = 0.4, 0.025
    detector = detectors.CCD(
        geometry=detectors.CCDGeometry(
            row=15,
            col=21,
            pixel_vert_size=window[1] / 15 * 1000,
            pixel_horz_size=window[0] / 21 * 1000,
        ),
        environment=detectors.Environment(wavelength=WAVELENGTH_NM),
        characteristics=detectors.Characteristics(quantum_efficiency=qe),
    )
    detector.set_readout(times=[interval_s], non_destructive=False)
    detector.readout_properties.time = interval_s
    detector.readout_properties.time_step = interval_s
    models.load_image(
        detector, image_file=str(path), convert_to_photons=False, time_scale=1.0
    )
    np.testing.assert_allclose(
        detector.photon.array, rate * interval_s, rtol=ROUNDING_RTOL, atol=1e-12
    )
    np.testing.assert_allclose(
        detector.photon.array.sum(),
        measured / PHOTON_ENERGY_J * interval_s,
        rtol=ROUNDING_RTOL,
        atol=0,
    )
    np.testing.assert_allclose(
        detector.photon.array.sum(),
        reference / PHOTON_ENERGY_J * interval_s,
        rtol=7e-5,
        atol=0,  # inherited midpoint error, not detector tolerance
    )
    charge.simple_conversion(detector, binomial_sampling=False)
    np.testing.assert_allclose(
        detector.charge.array.sum(),
        qe * measured / PHOTON_ENERGY_J * interval_s,
        rtol=ROUNDING_RTOL,
        atol=0,
    )
