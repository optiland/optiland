from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.physical_optics import GaussianBeam
from tests.utils import assert_allclose

WAVELENGTH = 1.064  # µm
LAM = WAVELENGTH * 1e-3  # mm


def test_from_waist_sets_q(set_test_backend):
    beam = GaussianBeam.from_waist(0.5, 20.0, wavelength=WAVELENGTH)
    z_r = math.pi * 0.5**2 / LAM
    assert_allclose(beam.q_real, -20.0)
    assert_allclose(beam.q_imag, z_r)
    assert_allclose(beam.rayleigh_range, z_r)
    assert_allclose(beam.waist_position, 20.0)
    assert_allclose(beam.waist_radius, 0.5)
    assert_allclose(beam.divergence, LAM / (math.pi * 0.5))


def test_free_space_radius_and_curvature(set_test_backend):
    w0 = 0.2
    z_r = math.pi * w0**2 / LAM
    for z in (-300.0, 50.0, 400.0):
        beam = GaussianBeam.from_waist(w0, wavelength=WAVELENGTH).propagate(z)
        assert_allclose(beam.beam_radius, w0 * math.sqrt(1 + (z / z_r) ** 2))
        assert_allclose(beam.curvature, 1 / (z * (1 + (z_r / z) ** 2)))
    assert_allclose(GaussianBeam.from_waist(w0, wavelength=WAVELENGTH).curvature, 0.0)


def test_thin_lens_matches_self(set_test_backend):
    # Self (1983): 1 mm waist 50 mm before an f = 100 mm thin lens at 1.064 µm.
    beam = GaussianBeam.from_waist(1.0, -50.0, wavelength=WAVELENGTH)
    lens = be.array([[1.0, 0.0], [-1 / 100.0, 1.0]])
    out = beam.transform(lens)
    z_r = math.pi / LAM
    mag2 = 1 / ((1 - 50.0 / 100.0) ** 2 + (z_r / 100.0) ** 2)
    assert_allclose(out.waist_position, 100.0 + mag2 * (50.0 - 100.0))
    assert_allclose(out.waist_radius, math.sqrt(mag2) * 1.0)
    assert_allclose(out.waist_position, 99.942664, atol=1e-6)


def test_plane_interface_moves_waist_to_n_d(set_test_backend):
    n = 1.506635
    beam = GaussianBeam.from_waist(0.05, 20.0, wavelength=WAVELENGTH)
    interface = be.array([[1.0, 0.0], [0.0, 1.0 / n]])
    out = beam.transform(interface, refractive_index=n)
    assert_allclose(out.waist_position, n * 20.0)
    assert_allclose(out.waist_radius, 0.05)


def test_gouy_phase_through_focus(set_test_backend):
    beam = GaussianBeam.from_waist(0.1, 1000.0, wavelength=WAVELENGTH)
    z_r = math.pi * 0.1**2 / LAM
    assert_allclose(beam.gouy_phase(2000.0), 2 * math.atan(1000.0 / z_r))
    assert_allclose(beam.gouy_phase(2000.0), 3.082557, atol=1e-6)


def test_overlap(set_test_backend):
    a = GaussianBeam.from_waist(0.10, wavelength=WAVELENGTH)
    b = GaussianBeam.from_waist(0.12, wavelength=WAVELENGTH)
    assert_allclose(a.overlap(a), 1.0)
    assert_allclose(a.overlap(b), 4 / (0.10 / 0.12 + 0.12 / 0.10) ** 2)
    assert_allclose(a.overlap(b), 0.967482, atol=1e-6)
    shifted = GaussianBeam.from_waist(0.10, 30.0, wavelength=WAVELENGTH)
    z_r = math.pi * 0.10**2 / LAM
    assert_allclose(a.overlap(shifted), 4 * z_r**2 / (30.0**2 + 4 * z_r**2))


def test_m2_uses_embedded_gaussian(set_test_backend):
    beam = GaussianBeam.from_waist(0.3, wavelength=WAVELENGTH, m2=2.0).propagate(500)
    tem00 = GaussianBeam.from_waist(0.3, wavelength=2 * WAVELENGTH).propagate(500)
    assert_allclose(beam.beam_radius, tem00.beam_radius)
    with pytest.raises(ValueError, match="TEM00"):
        beam.overlap(beam)


def _round_trip(length, r1, r2):
    # Unfolded round trip from mirror 1 to mirror 2 and back; a concave mirror
    # of radius r acts as a lens of focal length r / 2.
    def space(d):
        return be.array([[1.0, d], [0.0, 1.0]])

    def mirror(r):
        return be.array([[1.0, 0.0], [-2.0 / r, 1.0]])

    return mirror(r1) @ space(length) @ mirror(r2) @ space(length)


def test_from_round_trip_symmetric_cavity(set_test_backend):
    length, radius = 1000.0, 1500.0
    matrix = _round_trip(length, radius, radius)
    mode = GaussianBeam.from_round_trip(matrix, wavelength=WAVELENGTH)
    assert_allclose(mode.waist_position, length / 2)
    assert_allclose(mode.rayleigh_range, math.sqrt(length * (2 * radius - length)) / 2)
    assert_allclose(1 / mode.curvature, -radius)
    assert_allclose(mode.beam_radius, 0.599355, atol=1e-6)
    again = mode.transform(matrix)
    assert_allclose(again.q_real, mode.q_real, atol=1e-9)
    assert_allclose(again.q_imag, mode.q_imag)


def test_from_round_trip_plano_concave_cavity(set_test_backend):
    # The waist sits on the flat mirror, with z_R = sqrt(L (R - L)).
    length, radius = 1000.0, 3000.0
    matrix = _round_trip(length, be.inf, radius)
    mode = GaussianBeam.from_round_trip(matrix, wavelength=WAVELENGTH)
    assert_allclose(mode.waist_position, 0.0, atol=1e-9)
    assert_allclose(mode.rayleigh_range, math.sqrt(length * (radius - length)))


def test_from_round_trip_rejects_unstable_or_open_matrices(set_test_backend):
    with pytest.raises(ValueError, match="not stable"):
        GaussianBeam.from_round_trip(
            _round_trip(1000.0, 400.0, 400.0), wavelength=WAVELENGTH
        )
    for det in (0.5, 1.001):
        with pytest.raises(ValueError, match="det = 1"):
            GaussianBeam.from_round_trip(
                be.array([[1.0, 0.0], [0.0, det]]), wavelength=WAVELENGTH
            )


@pytest.fixture
def float32(set_test_backend):
    """Run a test in float32, the default precision of the PyTorch backend."""
    be.set_precision("float32")
    yield
    be.set_precision("float64")


def test_from_round_trip_symmetric_cavity_float32(float32):
    length, radius = 1000.0, 1500.0
    mode = GaussianBeam.from_round_trip(
        _round_trip(length, radius, radius), wavelength=WAVELENGTH
    )
    assert be.to_numpy(mode.q_imag).dtype == np.float32
    assert_allclose(mode.waist_position, length / 2, rtol=1e-6)
    assert_allclose(
        mode.rayleigh_range, math.sqrt(length * (2 * radius - length)) / 2, rtol=1e-6
    )


def test_from_round_trip_accepts_large_entries_in_float32(float32):
    # Moving the reference plane of a cavity by `shift` gives another valid
    # round trip, P(shift) M P(-shift), with determinant exactly 1 and entries
    # near 1e7. Its eigenmode is the original one propagated by `shift`.
    length, radius, shift = 1000.0, 1500.0, 1e5

    def space(d):
        return np.array([[1.0, d], [0.0, 1.0]])

    def mirror(r):
        return np.array([[1.0, 0.0], [-2.0 / r, 1.0]])

    cavity = mirror(radius) @ space(length) @ mirror(radius) @ space(length)
    moved = space(shift) @ cavity @ space(-shift)
    matrix = be.array(moved)  # every entry rounded once to float32

    # Rounding alone moves the determinant far beyond an absolute 1e-6 (the
    # determinant of the rounded entries, computed exactly in float64).
    a, b, c, d = be.to_numpy(matrix).astype(np.float64).ravel()
    assert abs(a * d - b * c - 1) > 1e-4

    mode = GaussianBeam.from_round_trip(matrix, wavelength=WAVELENGTH)
    expected = GaussianBeam.from_round_trip(
        be.array(cavity), wavelength=WAVELENGTH
    ).propagate(shift)
    assert_allclose(mode.q_real, expected.q_real, rtol=1e-6)
    assert_allclose(mode.q_imag, expected.q_imag, rtol=1e-4)


def test_from_round_trip_rejects_change_of_medium_in_float32(float32):
    for det in (0.5, 1.001):
        with pytest.raises(ValueError, match="det = 1"):
            GaussianBeam.from_round_trip(
                be.array([[1.0, 0.0], [0.0, det]]), wavelength=WAVELENGTH
            )


def test_overlap_rejects_other_wavelength_or_medium(set_test_backend):
    beam = GaussianBeam.from_waist(0.1, wavelength=WAVELENGTH)
    with pytest.raises(ValueError, match="same wavelength"):
        beam.overlap(GaussianBeam.from_waist(0.1, wavelength=0.532))
    with pytest.raises(ValueError, match="same medium"):
        beam.overlap(
            GaussianBeam.from_waist(0.1, wavelength=WAVELENGTH, refractive_index=1.5)
        )


def test_repr(set_test_backend):
    beam = GaussianBeam.from_waist(0.5, 10.0, wavelength=WAVELENGTH)
    assert repr(beam) == (
        "GaussianBeam(w=0.500046 mm, w0=0.5 mm, waist at +10 mm, "
        "wavelength=1.064 µm, m2=1.0)"
    )


def test_with_refractive_index_keeps_waist(set_test_backend):
    beam = GaussianBeam.from_waist(0.3, 12.0, wavelength=WAVELENGTH)
    water = beam.with_refractive_index(1.33)
    assert_allclose(water.waist_radius, 0.3)
    assert_allclose(water.waist_position, 12.0)
    assert_allclose(water.rayleigh_range, 1.33 * beam.rayleigh_range)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q_real": 0.0, "q_imag": -1.0, "wavelength": WAVELENGTH},
        {"q_real": 0.0, "q_imag": 1.0, "wavelength": -1.0},
        {"q_real": 0.0, "q_imag": 1.0, "wavelength": 1.0, "refractive_index": 0.0},
        {"q_real": 0.0, "q_imag": 1.0, "wavelength": 1.0, "m2": 0.5},
    ],
)
def test_invalid_parameters_raise(set_test_backend, kwargs):
    with pytest.raises(ValueError):
        GaussianBeam(**kwargs)
