from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest

import optiland.backend as be
from optiland.colorimetry import core
from optiland.colorimetry.plotting import plot_cie_1931_chromaticity_diagram
from tests.utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture(autouse=True)
def skip_torch_without_numpy_bridge(set_test_backend):
    if be.get_backend() != "torch":
        return

    import torch

    try:
        _ = torch.tensor([0.0]).numpy()
    except Exception as exc:
        pytest.skip(f"Torch backend unavailable in this env: {exc}")


def _as_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def test_spectrum_to_xyz_white_reflectance():
    wavelengths = list(range(380, 781, 5))
    values = [1.0] * len(wavelengths)

    X, Y, Z = core.spectrum_to_xyz(wavelengths=wavelengths, values=values)
    x, y, _ = core.xyz_to_xyY(X, Y, Z)
    r, g, b = core.xyz_to_srgb(X, Y, Z)

    assert_allclose(Y, 100.0, rtol=1e-6, atol=1e-6)
    assert abs(_as_float(x) - 0.3127385736725134) < 1e-6
    assert abs(_as_float(y) - 0.32905204701686674) < 1e-6
    assert _as_float(r) == 254
    assert _as_float(g) == 255
    assert _as_float(b) == 254


def test_spectrum_to_xyz_requires_visible_range():
    wavelengths = list(range(420, 701, 10))
    values = [1.0] * len(wavelengths)

    with pytest.raises(ValueError, match="does not cover"):
        core.spectrum_to_xyz(wavelengths=wavelengths, values=values)


def test_spectrum_to_xyz_illuminant_length_check():
    wavelengths = list(range(380, 781, 5))
    values = [1.0] * len(wavelengths)
    illuminant = [1.0] * (len(wavelengths) - 1)

    with pytest.raises(ValueError, match="Illuminant must have the same size"):
        core.spectrum_to_xyz(
            wavelengths=wavelengths,
            values=values,
            illuminant=illuminant,
        )


def test_xyz_to_xyY_vectorized_inputs():
    xyz = be.array(
        [
            [41.24, 21.26, 1.93],
            [35.76, 71.52, 11.92],
        ]
    )

    x, y, Y = core.xyz_to_xyY(xyz)

    xy_sum = xyz[:, 0] + xyz[:, 1] + xyz[:, 2]
    expected_x = xyz[:, 0] / xy_sum
    expected_y = xyz[:, 1] / xy_sum

    assert_allclose(x, expected_x)
    assert_allclose(y, expected_y)
    assert_allclose(Y, xyz[:, 1])


def test_xyz_to_xyY_handles_zero_sum():
    x, y, Y = core.xyz_to_xyY(0.0, 0.0, 0.0)

    assert _as_float(Y) == 0.0
    assert abs(_as_float(x) - 0.3127) < 1e-6
    assert abs(_as_float(y) - 0.3290) < 1e-6


def test_xyz_to_srgb_white_point():
    # D65 white point in XYZ (scaled for Y=100)
    X, Y, Z = 95.047, 100.0, 108.883
    r, g, b = core.xyz_to_srgb(X, Y, Z)

    r_val = _as_float(r)
    g_val = _as_float(g)
    b_val = _as_float(b)

    assert 254 <= r_val <= 255
    assert 254 <= g_val <= 255
    assert 254 <= b_val <= 255


def test_xyz_to_xyY_packed_scalar_raises():
    with pytest.raises(ValueError, match="Scalar input requires X, Y, Z arguments"):
        core.xyz_to_xyY(1.0)


def test_xyz_to_xyY_packed_first_axis_shape3():
    xyz = be.array(
        [
            [41.24, 35.76],
            [21.26, 71.52],
            [1.93, 11.92],
        ]
    )

    x, y, Y = core.xyz_to_xyY(xyz)

    expected_x = xyz[0] / (xyz[0] + xyz[1] + xyz[2])
    expected_y = xyz[1] / (xyz[0] + xyz[1] + xyz[2])

    assert_allclose(x, expected_x)
    assert_allclose(y, expected_y)
    assert_allclose(Y, xyz[1])


def test_xyz_to_xyY_packed_middle_axis_shape3():
    xyz = be.array(
        [
            [
                [41.24, 35.76],
                [21.26, 71.52],
                [1.93, 11.92],
            ],
            [
                [19.01, 20.00],
                [20.00, 21.00],
                [21.99, 22.00],
            ],
        ]
    )

    x, y, Y = core.xyz_to_xyY(xyz)

    X_expected = xyz[:, 0, :]
    Y_expected = xyz[:, 1, :]
    Z_expected = xyz[:, 2, :]
    denom = X_expected + Y_expected + Z_expected

    assert_allclose(x, X_expected / denom)
    assert_allclose(y, Y_expected / denom)
    assert_allclose(Y, Y_expected)


def test_xyz_to_xyY_packed_without_axis_3_raises():
    xyz = be.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="dimension of size 3"):
        core.xyz_to_xyY(xyz)


def test_xyz_to_srgb_clip_fallback_without_astype():
    r, g, b = core.xyz_to_srgb(95.047, 100.0, 108.883)

    assert 0 <= _as_float(r) <= 255
    assert 0 <= _as_float(g) <= 255
    assert 0 <= _as_float(b) <= 255


def test_plot_cie_1931_diagram_no_legend():
    fig, ax = plot_cie_1931_chromaticity_diagram(show_legend=False)
    assert fig is not None
    assert ax is not None
    assert ax.get_legend() is None
    plt.close(fig)


def test_plot_cie_1931_diagram_with_legend():
    fig, ax = plot_cie_1931_chromaticity_diagram(color="contour", show_legend=True)
    assert fig is not None
    assert ax is not None
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_cie_1931_diagram_fill_mode():
    fig, ax = plot_cie_1931_chromaticity_diagram(color="fill", show_legend=False)
    assert fig is not None
    assert ax is not None
    assert len(ax.images) == 1
    plt.close(fig)


def test_plot_cie_1931_diagram_no_mode_with_existing_axes():
    fig, ax = plt.subplots(figsize=(6, 6))
    fig2, ax2 = plot_cie_1931_chromaticity_diagram(ax=ax, color="no", show_legend=True)
    assert fig2 is fig
    assert ax2 is ax
    assert ax.get_legend() is not None
    plt.close(fig)
