from __future__ import annotations

import os
import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import optiland.backend as be
from optiland.physical_optics import (
    ScalarField,
    WraparoundDiagnostic,
    angular_spectrum,
    wraparound_diagnostic,
)
from tests.utils import assert_allclose, assert_array_equal

N = 256
DX = 0.002  # mm
WAVELENGTH = 0.0005  # mm
WIDTH = N * DX  # mm


def _complex_array(data):
    # be.array uses the backend's real default dtype, discarding imaginary data.
    if be.get_backend() == "torch":
        torch = pytest.importorskip("torch")
        return torch.tensor(data, dtype=torch.complex128)
    return np.asarray(data, dtype=np.complex128)


def _fractions(result):
    return (
        float(be.to_numpy(result.lower_fraction)),
        float(be.to_numpy(result.upper_fraction)),
    )


def _gaussian(waist, x0=0.0, cycles=0):
    # An integer number of carrier cycles across the grid keeps the field periodic.
    x = (np.arange(N) - (N - 1) / 2) * DX
    x_grid, y_grid = np.meshgrid(x, x)
    envelope = np.exp(-((x_grid - x0) ** 2 + y_grid**2) / waist**2)
    return envelope * np.exp(2j * np.pi * cycles * x_grid / WIDTH)


def _distance_for_shift(cycles, shift, size=WIDTH, wavelength=WAVELENGTH):
    direction = cycles * wavelength / size
    return shift * np.sqrt(1 - direction**2) / direction


def _wrapped_fraction(data, distance, factor=4, dx=DX, wavelength=WAVELENGTH):
    """Power fraction by which periodic propagation differs from a padded one."""
    ny, nx = data.shape
    top, left = (factor - 1) * ny // 2, (factor - 1) * nx // 2
    padded = np.zeros((factor * ny, factor * nx), dtype=np.complex128)
    padded[top : top + ny, left : left + nx] = data
    periodic = ScalarField(_complex_array(data), dx=dx, wavelength=wavelength)
    linear = ScalarField(_complex_array(padded), dx=dx, wavelength=wavelength)
    window = (slice(top, top + ny), slice(left, left + nx))
    expected = be.to_numpy(angular_spectrum(linear, distance).data)[window]
    error = be.to_numpy(angular_spectrum(periodic, distance).data) - expected
    return np.sum(np.abs(error) ** 2) / np.sum(np.abs(data) ** 2)


def _reference_bounds(data, dx, dy, wavelength, distance, support_threshold):
    """Classify each frequency by moving every occupied cell by its shift."""
    ny, nx = data.shape
    power = np.abs(np.fft.fft2(data)) ** 2
    cosines_x, cosines_y = np.meshgrid(
        np.fft.fftfreq(nx, dx) * wavelength, np.fft.fftfreq(ny, dy) * wavelength
    )
    intensity = np.abs(data) ** 2
    rows, columns = np.nonzero(intensity >= support_threshold * intensity.max())
    lower = upper = 0.0
    for p, ux, uy in zip(
        power.ravel(), cosines_x.ravel(), cosines_y.ravel(), strict=True
    ):
        if ux * ux + uy * uy >= 1:
            continue  # evanescent; generic samples avoid the exact cutoff
        uz = np.sqrt(1 - ux * ux - uy * uy)
        sx, sy = distance * ux / uz, distance * uy / uz
        if abs(sx) >= nx * dx or abs(sy) >= ny * dy:
            lower += p
        # Cell edges after the shift, against the grid edges.
        if (
            np.any((columns - 0.5) * dx + sx < -0.5 * dx)
            or np.any((columns + 0.5) * dx + sx > (nx - 0.5) * dx)
            or np.any((rows - 0.5) * dy + sy < -0.5 * dy)
            or np.any((rows + 0.5) * dy + sy > (ny - 0.5) * dy)
        ):
            upper += p
    return lower / power.sum(), upper / power.sum()


@pytest.mark.parametrize("shape", [(12, 18), (13, 17)])
@pytest.mark.parametrize("refractive_index", [1.0, 1.4])
@pytest.mark.parametrize("backend_distance", [False, True])
def test_bounds_match_independent_reference(
    set_test_backend, shape, refractive_index, backend_distance
):
    rng = np.random.default_rng(20261104)
    dx, dy, wavelength = 0.01, 0.013, 0.021
    for _ in range(6):
        # A random block anywhere on the grid gives asymmetric margins on both
        # axes. The highest frequencies of this sampling are evanescent.
        data = np.zeros(shape, dtype=np.complex128)
        top, left = rng.integers(0, 6), rng.integers(0, 8)
        height, width = rng.integers(2, shape[0] - top), rng.integers(2, 9)
        data[top : top + height, left : left + width] = rng.normal(
            size=(height, width)
        ) + 1j * rng.normal(size=(height, width))
        distance = rng.uniform(-0.3, 0.3)
        support_threshold = rng.choice([1e-3, 0.3])
        field = ScalarField(
            _complex_array(data),
            dx=dx,
            dy=dy,
            wavelength=wavelength,
            refractive_index=refractive_index,
        )
        result = wraparound_diagnostic(
            field,
            be.array(distance) if backend_distance else distance,
            support_threshold=support_threshold,
            threshold=0.1,
        )
        expected = _reference_bounds(
            data, dx, dy, wavelength / refractive_index, distance, support_threshold
        )
        assert_allclose(_fractions(result), expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("shift_ratio", [-1.5, -0.5, 0.5, 1.5])
@pytest.mark.parametrize("refractive_index", [1.0, 1.5])
def test_plane_wave_bounds_are_exact(
    set_test_backend, axis, shift_ratio, refractive_index
):
    shape, dx, dy, cycles = (12, 20), 0.03, 0.05, 3
    rows, columns = np.indices(shape)
    phase = columns * cycles / shape[1] if axis == "x" else rows * cycles / shape[0]
    size = shape[1] * dx if axis == "x" else shape[0] * dy
    distance = _distance_for_shift(
        cycles, shift_ratio * size, size, 0.0005 / refractive_index
    )
    field = ScalarField(
        _complex_array(np.exp(2j * np.pi * phase)),
        dx=dx,
        dy=dy,
        wavelength=0.0005,
        refractive_index=refractive_index,
    )
    result = wraparound_diagnostic(
        field, distance, support_threshold=0.5, threshold=0.1
    )
    assert isinstance(result, WraparoundDiagnostic)
    # The plane wave fills the grid, so any shift carries some of it across.
    # FFT roundoff leaks squared-epsilon power into the other frequency bins.
    expected = (float(abs(shift_ratio) > 1), 1.0)
    assert_allclose(_fractions(result), expected, rtol=0, atol=1e-15)
    assert result.exceeds_threshold


@pytest.mark.parametrize(
    "data, distance",
    [
        # Moves exactly one grid width: no boundary energy, yet all of it wraps.
        (_gaussian(0.03, cycles=20), _distance_for_shift(20, WIDTH)),
        (_gaussian(0.03, cycles=20), _distance_for_shift(20, 0.6 * WIDTH)),
        (_gaussian(0.004), 40.0),
        (_gaussian(0.05), 2.0),
        # Starts near an edge: wraps without power outside any spectral window.
        (_gaussian(0.01, x0=0.2), 3.0),
        (_gaussian(0.02, x0=-0.2, cycles=20), _distance_for_shift(20, 0.7 * WIDTH)),
        (_gaussian(0.02, x0=-0.2, cycles=-20), _distance_for_shift(20, 0.7 * WIDTH)),
    ],
    ids=[
        "full-period-shift",
        "partial-shift",
        "narrow-far-field",
        "wide-near-field",
        "near-edge",
        "toward-centre",
        "toward-edge",
    ],
)
def test_bounds_bracket_wrapped_power(set_test_backend, data, distance):
    field = ScalarField(_complex_array(data), dx=DX, wavelength=WAVELENGTH)
    result = wraparound_diagnostic(
        field, distance, support_threshold=1e-6, threshold=0.01
    )
    lower, upper = _fractions(result)
    wrapped = _wrapped_fraction(data, distance)
    # The bounds follow geometric shifts; Gaussian tails below the support
    # threshold may wrap by a negligible amount beyond them.
    assert lower <= wrapped + 1e-9
    assert wrapped <= upper + 1e-9


def test_full_wrap_is_detected_by_lower_bound(set_test_backend):
    data = _gaussian(0.03, cycles=20)
    field = ScalarField(_complex_array(data), dx=DX, wavelength=WAVELENGTH)
    result = wraparound_diagnostic(
        field, _distance_for_shift(20, WIDTH), support_threshold=1e-6, threshold=0.01
    )
    assert _fractions(result)[0] > 0.5


def test_shift_direction_follows_distance_sign(set_test_backend):
    distance = _distance_for_shift(20, 0.7 * WIDTH)
    inward = ScalarField(
        _complex_array(_gaussian(0.02, x0=-0.2, cycles=20)),
        dx=DX,
        wavelength=WAVELENGTH,
    )
    outward = ScalarField(
        _complex_array(_gaussian(0.02, x0=-0.2, cycles=-20)),
        dx=DX,
        wavelength=WAVELENGTH,
    )
    kwargs = {"support_threshold": 1e-6, "threshold": 0.01}
    toward_edge = _fractions(wraparound_diagnostic(outward, distance, **kwargs))
    assert _fractions(wraparound_diagnostic(inward, distance, **kwargs))[1] < 0.5
    assert toward_edge[1] > 0.99
    assert_allclose(
        _fractions(wraparound_diagnostic(inward, -distance, **kwargs)),
        toward_edge,
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("backend_distance", [False, True])
def test_zero_distance_reports_no_wraparound(set_test_backend, backend_distance):
    rng = np.random.default_rng(20261104)
    data = rng.normal(size=(6, 10)) + 1j * rng.normal(size=(6, 10))
    field = ScalarField(_complex_array(data), dx=0.2, dy=0.7, wavelength=0.5)
    distance = be.array(0.0) if backend_distance else 0.0
    result = wraparound_diagnostic(field, distance, support_threshold=1, threshold=1)
    assert _fractions(result) == (0.0, 0.0)


def test_evanescent_content_does_not_shift(set_test_backend):
    indices = be.arange_indices(32)
    checkerboard = 1 - 2 * (indices % 2)
    field = ScalarField(
        checkerboard[:, None] * checkerboard[None, :], dx=0.1, wavelength=1.0
    )
    result = wraparound_diagnostic(field, 5.0, support_threshold=0.5, threshold=0.1)
    assert _fractions(result) == (0.0, 0.0)


@pytest.mark.parametrize("distance", [-1.0, 0.0, 1.0])
def test_grazing_component_at_cutoff(set_test_backend, distance):
    # Two cycles over 8 unit samples at wavelength 4 give ux = 1 exactly, a
    # grazing wave whose lateral shift diverges at any nonzero distance.
    columns = np.arange(8)
    data = np.exp(2j * np.pi * 2 * columns / 8)[None, :] * np.ones((4, 1))
    field = ScalarField(_complex_array(data), dx=1, wavelength=4)
    result = wraparound_diagnostic(
        field, distance, support_threshold=0.5, threshold=0.1
    )
    expected = float(distance != 0)
    assert_allclose(_fractions(result), (expected, expected), rtol=0, atol=1e-15)


def test_support_threshold_only_changes_upper_bound(set_test_backend):
    field = ScalarField(
        _complex_array(_gaussian(0.01, x0=0.2)), dx=DX, wavelength=WAVELENGTH
    )
    results = [
        _fractions(
            wraparound_diagnostic(
                field, 3.0, support_threshold=support_threshold, threshold=0.01
            )
        )
        for support_threshold in (1e-12, 1e-6, 1e-2, 1)
    ]
    lowers, uppers = zip(*results, strict=True)
    # A larger threshold shrinks the support box and widens its margins.
    assert len(set(lowers)) == 1
    assert all(a >= b for a, b in zip(uppers, uppers[1:], strict=False))
    assert uppers[0] > uppers[-1]


def test_lower_bound_never_exceeds_upper_bound(set_test_backend):
    rng = np.random.default_rng(20261104)
    for _ in range(20):
        data = np.zeros((16, 24), dtype=np.complex128)
        data[4:12, 6:18] = rng.normal(size=(8, 12)) + 1j * rng.normal(size=(8, 12))
        field = ScalarField(_complex_array(data), dx=0.01, dy=0.02, wavelength=0.002)
        result = wraparound_diagnostic(
            field,
            rng.uniform(-5, 5),
            support_threshold=rng.uniform(1e-3, 1),
            threshold=0.1,
        )
        lower, upper = _fractions(result)
        assert 0 <= lower <= upper <= 1


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_amplitude_scale_invariance_without_overflow(set_test_backend, scale):
    data = _gaussian(0.01, x0=0.2)
    kwargs = {"support_threshold": 1e-6, "threshold": 0.01}
    reference = wraparound_diagnostic(
        ScalarField(_complex_array(data), dx=DX, wavelength=WAVELENGTH), 3.0, **kwargs
    )
    scaled = wraparound_diagnostic(
        ScalarField(_complex_array(data * scale), dx=DX, wavelength=WAVELENGTH),
        3.0,
        **kwargs,
    )
    assert_allclose(_fractions(scaled), _fractions(reference), rtol=1e-12, atol=0)


def test_zero_field_reports_no_wraparound(set_test_backend):
    field = ScalarField(be.zeros((6, 10)), dx=1, wavelength=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wraparound_diagnostic(
            field, 10.0, support_threshold=0.5, threshold=0.1, warn=True
        )
    assert _fractions(result) == (0.0, 0.0)
    assert not result.exceeds_threshold
    assert not caught


@pytest.mark.parametrize("warn", [False, True])
@pytest.mark.parametrize("threshold", [0.99, 1.0])
def test_warning_is_optional_and_strict(set_test_backend, warn, threshold):
    rows, columns = np.indices((12, 20))
    field = ScalarField(
        _complex_array(np.exp(2j * np.pi * 3 * columns / 20)), dx=1, wavelength=0.5
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wraparound_diagnostic(
            field, 10.0, support_threshold=0.5, threshold=threshold, warn=warn
        )
    exceeds = threshold < 1
    assert result.exceeds_threshold is exceeds
    assert len(caught) == int(warn and exceeds)
    if caught:
        assert caught[0].category is RuntimeWarning
        assert "periodic FFT" in str(caught[0].message)
        # normcase: Windows may report the drive letter in either case
        assert os.path.normcase(caught[0].filename) == os.path.normcase(__file__)


def test_diagnostic_does_not_mutate_field(set_test_backend):
    field = ScalarField(
        _complex_array(_gaussian(0.01, x0=0.2)), dx=DX, wavelength=WAVELENGTH
    )
    before = be.to_numpy(field.data).copy()
    original_data = field.data
    result = wraparound_diagnostic(field, 3.0, support_threshold=1e-6, threshold=0.5)
    assert field.data is original_data
    assert_array_equal(field.data, before)
    assert (result.support_threshold, result.threshold) == (1e-6, 0.5)
    with pytest.raises(FrozenInstanceError):
        result.threshold = 0.1


@pytest.mark.parametrize("support_threshold", [0, -0.1, 1.5, np.nan, np.inf])
def test_invalid_support_threshold_value(set_test_backend, support_threshold):
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    with pytest.raises(ValueError, match="support_threshold"):
        wraparound_diagnostic(
            field, 1.0, support_threshold=support_threshold, threshold=0.1
        )


@pytest.mark.parametrize("support_threshold", [True, 0.1j, "0.1", None])
def test_invalid_support_threshold_type(set_test_backend, support_threshold):
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    with pytest.raises(TypeError, match="support_threshold"):
        wraparound_diagnostic(
            field, 1.0, support_threshold=support_threshold, threshold=0.1
        )


def test_invalid_reporting_parameters(set_test_backend):
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    with pytest.raises(ValueError, match="threshold"):
        wraparound_diagnostic(field, 1.0, support_threshold=0.5, threshold=0)
    with pytest.raises(TypeError, match="threshold"):
        wraparound_diagnostic(field, 1.0, support_threshold=0.5, threshold="0.1")
    with pytest.raises(TypeError, match="warn"):
        wraparound_diagnostic(
            field, 1.0, support_threshold=0.5, threshold=0.1, warn="yes"
        )


@pytest.mark.parametrize(
    "distance, error",
    [
        (np.nan, ValueError),
        (np.inf, ValueError),
        (1j, TypeError),
        ("1", TypeError),
        ("array", TypeError),
    ],
)
def test_invalid_distance(set_test_backend, distance, error):
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    if distance == "array":
        distance = be.ones((2,))
    with pytest.raises(error, match="distance"):
        wraparound_diagnostic(field, distance, support_threshold=0.5, threshold=0.1)


def test_invalid_field(set_test_backend):
    with pytest.raises(TypeError, match="ScalarField"):
        wraparound_diagnostic(
            be.ones((6, 10)), 1.0, support_threshold=0.5, threshold=0.1
        )
    data = np.ones((6, 10), dtype=complex)
    data[3, 5] = np.nan
    field = ScalarField(_complex_array(data), dx=1, wavelength=0.5)
    with pytest.raises(ValueError, match="finite"):
        wraparound_diagnostic(field, 1.0, support_threshold=0.5, threshold=0.1)
    # Finite parts whose magnitude overflows.
    data = np.full((6, 10), 1.5e308 * (1 + 1j))
    field = ScalarField(_complex_array(data), dx=1, wavelength=0.5)
    with pytest.raises(ValueError, match="magnitudes must be finite"):
        wraparound_diagnostic(field, 1.0, support_threshold=0.5, threshold=0.1)


def test_backend_change_is_rejected(set_test_backend):
    if len(be.list_available_backends()) < 2:
        pytest.skip("requires both backends")
    field = ScalarField(be.ones((6, 10)), dx=1, wavelength=0.5)
    original_backend = be.get_backend()
    try:
        be.set_backend("torch" if original_backend == "numpy" else "numpy")
        with pytest.raises(RuntimeError, match="active backend changed"):
            wraparound_diagnostic(field, 1.0, support_threshold=0.5, threshold=0.1)
    finally:
        be.set_backend(original_backend)


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("zero", [False, True])
def test_torch_bounds_preserve_dtype_device_and_gradients(
    set_test_backend, precision, device, zero
):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific contract")
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    data = torch.tensor(
        _gaussian(0.01, x0=0.2), dtype=getattr(torch, precision), device=device
    )
    if zero:
        data = torch.zeros_like(data)
    data.requires_grad_()
    distance = torch.tensor(3.0, dtype=torch.float64, device=device)
    field = ScalarField(data, dx=DX, wavelength=WAVELENGTH)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wraparound_diagnostic(
            field, distance, support_threshold=1e-6, threshold=0.01, warn=True
        )
    for fraction in (result.lower_fraction, result.upper_fraction):
        assert fraction.dtype == data.real.dtype
        assert fraction.device == data.device
        assert fraction.ndim == 0
        assert fraction.requires_grad
    assert len(caught) == (0 if zero else 1)
    result.upper_fraction.backward()
    assert torch.isfinite(data.grad).all()
    if zero:
        assert torch.count_nonzero(data.grad) == 0
    else:
        assert torch.count_nonzero(data.grad) > 0


@pytest.mark.parametrize("bound", ["lower_fraction", "upper_fraction"])
def test_torch_gradient_matches_finite_difference(set_test_backend, bound):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific contract")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(20261104)
    data = np.zeros((16, 24))
    data[4:12, 6:18] = rng.uniform(0.2, 1.5, size=(8, 12))
    direction = np.zeros_like(data)
    direction[4:12, 6:18] = rng.normal(size=(8, 12))

    def fraction(values):
        field = ScalarField(values, dx=0.01, dy=0.02, wavelength=0.002)
        result = wraparound_diagnostic(
            field, 4.0, support_threshold=0.01, threshold=0.1
        )
        return getattr(result, bound)

    samples = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    value = fraction(samples)
    # Strictly inside (0, 1), so neither clip nor an empty band hides the gradient.
    assert 0 < float(value.detach()) < 1
    value.backward()
    derivative = float(torch.sum(samples.grad * torch.tensor(direction)))
    assert derivative != 0
    step = 1e-6
    difference = float(
        fraction(torch.tensor(data + step * direction))
        - fraction(torch.tensor(data - step * direction))
    ) / (2 * step)
    assert_allclose(derivative, difference, rtol=1e-6, atol=1e-12)


def test_torch_penalty_reduces_true_wraparound_of_phase_design(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific contract")
    torch = pytest.importorskip("torch")
    shape, dx, wavelength, distance = (24, 32), 0.01, 0.002, 3.0
    amplitude = torch.zeros(shape, dtype=torch.float64)
    amplitude[6:18, 8:24] = 1
    generator = torch.Generator().manual_seed(20261106)
    phase = 2 * torch.pi * torch.rand(shape, dtype=torch.float64, generator=generator)
    phase.requires_grad_()

    def design():
        return ScalarField(
            amplitude * torch.exp(1j * phase), dx=dx, wavelength=wavelength
        )

    def penalty():
        result = wraparound_diagnostic(
            design(), distance, support_threshold=0.5, threshold=1
        )
        return result.upper_fraction

    def wrapped():
        data = be.to_numpy(design().data.detach())
        return _wrapped_fraction(data, distance, 8, dx, wavelength)

    initial_penalty, initial_wrapped = float(penalty().detach()), wrapped()
    optimizer = torch.optim.Adam([phase], lr=0.1)
    for _ in range(40):
        optimizer.zero_grad()
        penalty().backward()
        assert torch.count_nonzero(phase.grad) > 0
        optimizer.step()
    # The design must wrap less in the padded reference, not only in the
    # diagnostic's ray model.
    assert float(penalty().detach()) < 0.25 * initial_penalty
    assert wrapped() < 0.25 * initial_wrapped


def test_torch_distance_does_not_enter_the_gradient(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific contract")
    torch = pytest.importorskip("torch")
    data = torch.tensor(_gaussian(0.01, x0=0.2), requires_grad=True)
    distance = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    field = ScalarField(data, dx=DX, wavelength=WAVELENGTH)
    result = wraparound_diagnostic(
        field, distance, support_threshold=1e-6, threshold=0.01
    )
    (result.lower_fraction + result.upper_fraction).backward()
    # The bounds are piecewise constant in distance; only the field is a
    # differentiable design input.
    assert distance.grad is None
    assert torch.count_nonzero(data.grad) > 0
