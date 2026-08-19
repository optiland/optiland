from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be


def test_config(set_test_backend):
    # Test setting device/precision (mostly for torch)
    if be.get_backend() == "torch":
        import torch

        # Check current settings
        assert be.get_device() in ["cpu", "cuda"]
        assert be.get_precision() in [32, 64]

        # Test changing precision — get_precision() now returns int (32 or 64)
        be.set_precision("float32")
        assert be.get_precision() == 32
        be.set_precision("float64")
        assert be.get_precision() == 64

        # Test complex precision
        c_prec = be.get_complex_precision()
        assert c_prec == torch.complex128


def test_creation_zeros_ones_full(set_test_backend):
    shape = (2, 3)
    z = be.zeros(shape)
    assert z.shape == shape
    assert be.all(z == 0)

    o = be.ones(shape)
    assert o.shape == shape
    assert be.all(o == 1)

    f = be.full(shape, 5.0)
    assert f.shape == shape
    assert be.all(f == 5.0)

    # Test *_like
    zl = be.zeros_like(o)
    assert zl.shape == shape
    assert be.all(zl == 0)

    ol = be.ones_like(z)
    assert ol.shape == shape
    assert be.all(ol == 1)

    fl = be.full_like(z, 3.0)
    assert fl.shape == shape
    assert be.all(fl == 3.0)


def test_linspace_arange(set_test_backend):
    l = be.linspace(0, 1, 5)
    assert len(l) == 5
    # Use be.abs for backend-agnostic abs
    assert be.abs(l[0] - 0.0) < 1e-6
    assert be.abs(l[-1] - 1.0) < 1e-6

    a = be.arange(5)
    assert len(a) == 5
    assert a[-1] == 4

    a2 = be.arange(0, 5, 2)
    assert len(a2) == 3  # 0, 2, 4


def test_shape_manipulation(set_test_backend):
    x = be.zeros((2, 3))

    # Reshape
    r = be.reshape(x, (3, 2))
    assert r.shape == (3, 2)

    # Stack
    if hasattr(be, "stack"):
        s = be.stack([x, x])
        assert s.shape == (2, 2, 3)

    # Broadcast
    if hasattr(be, "broadcast_to"):
        b = be.broadcast_to(x, (2, 2, 3))
        assert b.shape == (2, 2, 3)

    # Flip
    if hasattr(be, "flip"):
        f = be.flip(x)
        assert f.shape == x.shape

    # Roll
    if hasattr(be, "roll"):
        rl = be.roll(x, 1)
        assert rl.shape == x.shape

    # Tile
    if hasattr(be, "tile"):
        t = be.tile(x, (2, 1))
        assert t.shape == (4, 3)

    # Meshgrid
    if hasattr(be, "meshgrid"):
        xx, yy = be.meshgrid(be.arange(3), be.arange(2))
        assert xx.shape == (2, 3)
        assert yy.shape == (2, 3)


def test_math_ops(set_test_backend):
    x = be.array([-1.0, 0.0, 1.0])

    assert be.all(be.abs(x) >= 0)
    assert be.all(be.exp(x) > 0)

    # Trigonometry
    pi = be.array([np.pi])
    assert be.abs(be.sin(pi)) < 1e-6
    assert be.abs(be.cos(pi) + 1.0) < 1e-6

    # Degrees/Radians
    d = be.degrees(pi)
    assert be.abs(d - 180.0) < 1e-5
    r = be.radians(d)
    assert be.abs(r - np.pi) < 1e-5

    # Min/Max/Mean
    assert be.min(x) == -1.0
    assert be.max(x) == 1.0

    # nanmax
    x_nan = be.array([1.0, float("nan"), 2.0])
    if hasattr(be, "nanmax"):
        assert be.nanmax(x_nan) == 2.0

    # mean
    assert be.mean(be.array([1.0, 3.0])) == 2.0

    # maximum (elementwise)
    m = be.maximum(be.array([1, 5]), be.array([2, 3]))
    val1 = m[0]
    val2 = m[1]
    if hasattr(val1, "item"):
        val1 = val1.item()
    if hasattr(val2, "item"):
        val2 = val2.item()
    assert val1 == 2
    assert val2 == 5

    # where
    w = be.where(x > 0, 1.0, -1.0)
    val = w[2]
    if hasattr(val, "item"):
        val = val.item()
    assert val == 1.0


def test_all_any(set_test_backend):
    t = be.array([True, True])
    f = be.array([True, False])
    assert be.all(t)
    assert not be.all(f)
    assert be.any(f)
    assert not be.any(be.array([False, False]))


def test_cross_product(set_test_backend):
    if hasattr(be, "cross"):
        a = be.array([1.0, 0.0, 0.0])
        b = be.array([0.0, 1.0, 0.0])
        c = be.cross(a, b)
        assert be.allclose(c, be.array([0.0, 0.0, 1.0]))


def test_histogram2d(set_test_backend):
    if hasattr(be, "histogram2d"):
        x = be.array([0.5, 1.5])
        y = be.array([0.5, 1.5])
        bins = (be.array([0, 1, 2]), be.array([0, 1, 2]))
        H, _, _ = be.histogram2d(x, y, bins)
        # Should have count 1 at (0,0) and (1,1)
        # H is (ny, nx) -> (2, 2)
        assert H[0, 0] == 1
        assert H[1, 1] == 1


def test_poly(set_test_backend):
    if hasattr(be, "polyfit") and hasattr(be, "polyval"):
        x = be.array([0.0, 1.0, 2.0])
        y = be.array([0.0, 1.0, 2.0])
        coeffs = be.polyfit(x, y, 1)

        # Test consistency
        val = be.polyval(coeffs, 3.0)
        if hasattr(val, "item"):
            val = val.item()
        assert abs(val - 3.0) < 1e-5


def test_pad(set_test_backend):
    if hasattr(be, "pad"):
        x = be.ones((2, 2))
        # pad width ((top, bottom), (left, right))
        p = be.pad(x, ((1, 1), (1, 1)))
        assert p.shape == (4, 4)
        val = p[0, 0]
        if hasattr(val, "item"):
            val = val.item()
        assert val == 0  # Default constant 0


def test_vectorize(set_test_backend):
    if hasattr(be, "vectorize"):

        def my_func(x):
            return x + 1

        v_func = be.vectorize(my_func)
        res = v_func(be.array([1, 2, 3]))
        assert be.all(res == be.array([2, 3, 4]))


def test_interp(set_test_backend):
    if hasattr(be, "interp"):
        xp = be.array([0.0, 1.0, 2.0])
        fp = be.array([0.0, 1.0, 0.0])
        x = be.array([0.5, 1.5])
        res = be.interp(x, xp, fp)
        assert be.allclose(res, be.array([0.5, 0.5]))


def test_copy_to(set_test_backend):
    if hasattr(be, "copy_to"):
        src = be.array([1.0])
        dst = be.array([0.0])
        be.copy_to(src, dst)
        val = dst[0]
        if hasattr(val, "item"):
            val = val.item()
        assert val == 1.0


def test_arange_indices(set_test_backend):
    res = be.arange_indices(0, 5, 1)
    if be.get_backend() == "numpy":
        assert be.is_array_like(res)
    assert len(res) == 5


def test_cast(set_test_backend):
    arr = be.array([1, 2, 3])
    casted = be.cast(arr)
    if be.get_backend() == "numpy":
        assert casted.dtype == float
    else:
        assert "float" in str(casted.dtype)


def test_array_creation(set_test_backend):
    arr = be.array([1, 2, 3])
    assert be.is_array_like(arr)
    assert len(arr) == 3


def test_transpose(set_test_backend):
    arr = be.array([[1, 2], [3, 4]])
    transposed = be.transpose(arr)
    assert transposed.shape == (2, 2)
    val = transposed[0, 1]
    if hasattr(val, "item"):
        val = val.item()
    assert val == 3


def test_atleast_1d(set_test_backend):
    scalar = 5.0
    arr = be.atleast_1d(scalar)
    assert len(arr) == 1


def test_as_array_1d(set_test_backend):
    res = be.as_array_1d(5.0)
    assert res.shape == (1,)


def test_ravel(set_test_backend):
    if be.get_backend() == "torch" and not hasattr(be, "ravel"):
        pytest.skip("ravel not implemented")
    arr = be.array([[1, 2], [3, 4]])
    raveled = be.ravel(arr)
    assert raveled.shape == (4,)


def test_rotations(set_test_backend):
    if not hasattr(be, "from_euler"):
        pytest.skip("from_euler not implemented")
    euler = be.array([0, 0, 0])
    rot = be.from_euler(euler)
    assert rot is not None


def test_random_functions(set_test_backend):
    u = be.random_uniform(0, 1, size=(2, 2))
    assert u.shape == (2, 2)
    n = be.random_normal(0, 1, size=(2, 2))
    assert n.shape == (2, 2)
    r = be.rand(2, 2)
    assert r.shape == (2, 2)


def test_nearest_nd_interpolator(set_test_backend):
    points = be.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    values = be.array([0.0, 1.0, 2.0, 3.0])
    x = be.array([0.1])
    y = be.array([0.1])
    res = be.nearest_nd_interpolator(points, values, x, y)
    if hasattr(res, "item"):
        val = res.item()
    else:
        val = res[0]
    assert val == 0.0


def test_unsqueeze_last(set_test_backend):
    arr = be.array([[1, 2], [3, 4]])
    unsqueezed = be.unsqueeze_last(arr)
    assert unsqueezed.shape == (2, 2, 1)


def test_to_complex(set_test_backend):
    arr = be.array([1.0, 2.0])
    comp = be.to_complex(arr)
    if be.get_backend() == "numpy":
        assert np.iscomplexobj(comp)
    else:
        assert comp.is_complex()


def test_batched_chain_matmul3(set_test_backend):
    N = 2
    eye = be.eye(3)
    eyes = [eye for _ in range(N)]
    if hasattr(be, "stack"):
        A = be.stack(eyes)
        B = be.stack(eyes)
        C = be.stack(eyes)
    else:
        A = be.array(eyes)
        B = be.array(eyes)
        C = be.array(eyes)

    res = be.batched_chain_matmul3(A, B, C)
    assert res.shape == (N, 3, 3)


def test_factorial(set_test_backend):
    res = be.factorial(be.array([3.0]))
    val = res[0]
    if hasattr(val, "item"):
        val = val.item()
    assert abs(val - 6.0) < 1e-5


def test_path_contains_points(set_test_backend):
    vertices = be.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    points = be.array([[0.5, 0.5], [1.5, 1.5]])
    mask = be.path_contains_points(vertices, points)
    if hasattr(mask, "cpu"):
        mask = mask.detach().cpu().numpy()
    assert mask[0] == True
    assert mask[1] == False


def test_lstsq(set_test_backend):
    A = be.array([[1.0], [2.0], [3.0]])
    B = be.array([2.0, 4.0, 6.0])
    res = be.lstsq(A, B)
    if hasattr(res, "cpu"):
        res = res.detach().cpu().numpy()
    assert np.allclose(res.flatten(), np.array([2.0]), atol=1e-4)


def test_fftconvolve(set_test_backend):
    in1 = be.array([1.0, 2.0, 3.0])
    in2 = be.array([0.0, 1.0, 0.5])
    # Full
    res = be.fftconvolve(in1, in2, mode="full")
    assert res.shape[0] == 5
    # Same
    res = be.fftconvolve(in1, in2, mode="same")
    assert res.shape[0] == 3
    # Valid
    res = be.fftconvolve(in1, in2, mode="valid")
    assert res.shape[0] == 1


def test_grid_sample(set_test_backend):
    data = np.zeros((1, 1, 4, 4), dtype=float)
    data[0, 0, 1, 1] = 1.0
    input_tensor = be.array(data)
    grid_data = np.zeros((1, 1, 1, 2), dtype=float)
    grid = be.array(grid_data)
    res = be.grid_sample(input_tensor, grid, align_corners=False)
    assert res.shape == (1, 1, 1, 1)


def test_get_bilinear_weights(set_test_backend):
    if hasattr(be, "get_bilinear_weights"):
        coords = be.array([[0.5, 0.5]])
        bin_edges = (be.array([0, 1]), be.array([0, 1]))
        indices, weights = be.get_bilinear_weights(coords, bin_edges)
        assert indices.shape == (1, 4, 2)
        assert weights.shape == (1, 4)
