"""Zero padding preserves imported samples and their physical grid locations."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.physical_optics import ScalarField


def _backend_array(values):
    if be.get_backend() == "torch":
        import torch

        return torch.as_tensor(values)
    return values.copy()


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (2, 2), (4, 6)])
@pytest.mark.parametrize(
    "pad_width,widths",
    [
        (2, ((2, 2), (2, 2))),
        (np.int64(1), ((1, 1), (1, 1))),
        (((1, 2), (3, 1)), ((1, 2), (3, 1))),
        (((0, 3), (2, 0)), ((0, 3), (2, 0))),
        (((np.int32(1), 0), (0, np.int64(2))), ((1, 0), (0, 2))),
        (0, ((0, 0), (0, 0))),
        (((0, 0), (0, 0)), ((0, 0), (0, 0))),
    ],
)
def test_padding_preserves_imported_samples_and_grid(
    set_test_backend, shape, pad_width, widths
):
    # Binary-exact spacings/centers and integer marker values permit exact
    # coordinate, power, and sample comparisons (no propagation roundoff).
    marker = np.arange(np.prod(shape)).reshape(shape)
    values = (marker + 1 + 1j * (2 * marker + 3)).astype(np.complex128)
    imported = _backend_array(values)
    field = ScalarField(
        imported,
        dx=0.125,
        dy=0.25,
        wavelength=0.0005,
        refractive_index=1.4,
        center=(0.375, -0.5),
    )
    old_x, old_y = field.coordinates()
    old_power = be.to_numpy(field.power).copy()
    old_center = field.center
    padded = field.pad(pad_width)
    (yb, ya), (xb, xa) = widths
    ny, nx = shape
    expected = np.zeros((ny + yb + ya, nx + xb + xa), dtype=values.dtype)
    expected[yb : yb + ny, xb : xb + nx] = values

    assert padded is not field
    assert padded.shape == expected.shape
    np.testing.assert_array_equal(be.to_numpy(padded.data), expected)
    np.testing.assert_array_equal(be.to_numpy(field.data), values)
    np.testing.assert_array_equal(be.to_numpy(imported), values)
    assert field.center == old_center
    assert padded.center == (
        old_center[0] + (xa - xb) * field.dx / 2,
        old_center[1] + (ya - yb) * field.dy / 2,
    )
    x, y = padded.coordinates()
    np.testing.assert_array_equal(be.to_numpy(x[xb : xb + nx]), be.to_numpy(old_x))
    np.testing.assert_array_equal(be.to_numpy(y[yb : yb + ny]), be.to_numpy(old_y))
    np.testing.assert_array_equal(
        be.to_numpy(x),
        (np.arange(expected.shape[1]) - xb - (nx - 1) / 2) * field.dx + old_center[0],
    )
    np.testing.assert_array_equal(
        be.to_numpy(y),
        (np.arange(expected.shape[0]) - yb - (ny - 1) / 2) * field.dy + old_center[1],
    )
    np.testing.assert_array_equal(be.to_numpy(padded.power), old_power)
    for attribute in ("dx", "dy", "wavelength", "refractive_index", "_backend"):
        assert getattr(padded, attribute) == getattr(field, attribute)
    assert padded.data.dtype == field.data.dtype
    if be.get_backend() == "torch":
        assert padded.data.device == field.data.device
        assert padded.data.data_ptr() != field.data.data_ptr()
    else:
        assert not np.shares_memory(padded.data, field.data)
    padded.data[yb, xb] = 100 + 200j
    np.testing.assert_array_equal(be.to_numpy(field.data), values)


@pytest.mark.parametrize("precision", [32, 64])
def test_padding_preserves_dtype_after_configured_precision_change(
    set_test_backend, precision
):
    values = np.full((3, 4), 1 + 2j, dtype=f"complex{2 * precision}")
    field = ScalarField(_backend_array(values), dx=0.125, wavelength=0.0005)
    try:
        be.set_precision("float64" if precision == 32 else "float32")
        padded = field.pad(((1, 2), (3, 0)))
        assert padded.data.dtype == field.data.dtype
        x, y = padded.coordinates()
        assert x.dtype == field.data.real.dtype
        assert y.dtype == field.data.real.dtype
        np.testing.assert_array_equal(be.to_numpy(padded.data[1:4, 3:7]), values)
    finally:
        be.set_precision("float64")


@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("pad_width", [0, ((1, 2), (3, 0))])
def test_torch_padding_preserves_device_and_power_gradient(
    set_test_backend, precision, device, pad_width
):
    if be.get_backend() != "torch":
        pytest.skip("Torch-specific gradient reference")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    values = torch.tensor(
        [[1 + 2j, 3 - 1j, 2j], [2 - 3j, -1j, 4 + 2j]],
        dtype=getattr(torch, f"complex{2 * precision}"),
        device=device,
        requires_grad=True,
    )
    field = ScalarField(values, dx=0.125, dy=0.25, wavelength=0.0005)
    padded = field.pad(pad_width)
    assert padded.data.device == values.device
    assert padded.data.dtype == values.dtype
    assert padded.data.requires_grad
    padded.power.backward()
    # For a real intensity integral, the complex gradient is 2*A*dx*dy.
    torch.testing.assert_close(
        values.grad, 2 * values.detach() * field.dx * field.dy, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "pad_width",
    [
        True,
        False,
        np.bool_(True),
        1.0,
        1.5,
        1j,
        "1",
        None,
        (1, 2),
        (1,),
        ((1, 2),),
        ((1, 2), (3, 4), (5, 6)),
        ((1, 2, 3), (4, 5)),
        [[1, 2], [3, 4]],
        ([1, 2], (3, 4)),
        ((1, 2), (3.0, 4)),
        ((1, 2), (True, 4)),
        ((1, np.bool_(False)), (3, 4)),
        np.array([[1, 2], [3, 4]]),
    ],
)
def test_padding_rejects_ambiguous_or_noninteger_widths(set_test_backend, pad_width):
    field = ScalarField(be.ones((2, 3)), dx=0.125, wavelength=0.0005)
    with pytest.raises(TypeError, match="integer|tuple|boolean"):
        field.pad(pad_width)


@pytest.mark.parametrize(
    "pad_width", [-1, np.int64(-2), ((0, -1), (2, 3)), ((1, 0), (-2, 0))]
)
def test_padding_rejects_negative_widths(set_test_backend, pad_width):
    field = ScalarField(be.ones((2, 3)), dx=0.125, wavelength=0.0005)
    with pytest.raises(ValueError, match="nonnegative"):
        field.pad(pad_width)


def test_padding_rejects_active_backend_change(set_test_backend):
    original = be.get_backend()
    alternatives = [name for name in be.list_available_backends() if name != original]
    if not alternatives:
        pytest.skip("A second backend is not available")
    field = ScalarField(be.ones((2, 3)), dx=0.125, wavelength=0.0005)
    try:
        be.set_backend(alternatives[0])
        with pytest.raises(RuntimeError, match="active backend changed"):
            field.pad(1)
    finally:
        be.set_backend(original)
