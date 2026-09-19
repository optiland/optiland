"""Tests for ``kernels/reduce.metal`` (df64 reductions; sf64 twins when present).

References are computed on the DECODED inputs (hi + lo summed in float64, which
is exact) with exact arithmetic: ``math.fsum`` for sums, Python integers for
prefix sums and ``fractions.Fraction`` for products. Errors are reported in
units of u^2 = 2^-48.

The sf64 tests need ``sf64_core.h`` (and the vendored softfloat) in the kernels
directory; they are skipped otherwise. ``OPTILAND_SF64_SOURCE_FILES`` (a
``:``-separated list of absolute paths) overrides the sf64 prelude for
development.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import itertools  # noqa: E402
import math  # noqa: E402
import warnings  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("torch MPS backend is not available", allow_module_level=True)

from optiland.backend.torch_backend.metal.compile import (  # noqa: E402
    compile_library,
    kernel_source,
)

U2 = 2.0**-48
TG = 256
KERNEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optiland"
    / "backend"
    / "torch_backend"
    / "metal"
    / "kernels"
)
MPS = torch.device("mps")

# op -> (first-pass kernel, second-pass kernel) for the tree family.
TREE_OPS = {
    "sum": ("sum", "sum"),
    "nansum": ("nansum", "sum"),
    "prod": ("prod", "prod"),
    "max": ("max", "max"),
    "min": ("min", "min"),
    "nanmax": ("nanmax", "nanmax"),
    "nanmin": ("nanmin", "nanmin"),
}


# --------------------------------------------------------------------------
# Library / encoding helpers
# --------------------------------------------------------------------------
def sf64_source(*kernel_files: str) -> str | None:
    """Amalgamate vendor softfloat -> df64_core.h -> sf64_core.h -> kernels.

    Returns None when ``sf64_core.h`` is not available. The environment variable
    ``OPTILAND_SF64_SOURCE_FILES`` (``:``-separated absolute paths) replaces the
    softfloat + sf64_core.h prelude for development against a mock.
    """
    override = os.environ.get("OPTILAND_SF64_SOURCE_FILES")
    if override:
        prelude = "\n".join(Path(p).read_text() for p in override.split(":"))
        return "\n".join(
            [kernel_source("df64_core.h"), prelude, kernel_source(*kernel_files)]
        )
    if not (KERNEL_DIR / "sf64_core.h").exists():
        return None
    return kernel_source(
        "vendor/softfloat64.metal", "df64_core.h", "sf64_core.h", *kernel_files
    )


@pytest.fixture(scope="module")
def lib():
    return compile_library(kernel_source("df64_core.h", "reduce.metal"))


@pytest.fixture(scope="module")
def sflib():
    src = sf64_source("reduce.metal")
    if src is None:
        pytest.skip("sf64_core.h is not available yet")
    return compile_library(src)


def encode(x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Split float64 into (hi, lo) float32 MPS tensors; return the decoded array."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    hi = x.astype(np.float32)
    with np.errstate(invalid="ignore"):  # inf - inf for non-finite entries
        lo = np.where(np.isfinite(hi), (x - hi.astype(np.float64)), 0.0).astype(
            np.float32
        )
    decoded = hi.astype(np.float64) + lo.astype(np.float64)
    return (
        torch.from_numpy(hi).to(MPS),
        torch.from_numpy(lo).to(MPS),
        decoded,
    )


def decode(hi: torch.Tensor, lo: torch.Tensor) -> np.ndarray:
    return hi.cpu().numpy().astype(np.float64) + lo.cpu().numpy().astype(np.float64)


def layout(shape: tuple[int, ...], axis: int) -> tuple[int, int, int]:
    outer = int(np.prod(shape[:axis], dtype=np.int64))
    inner = int(np.prod(shape[axis + 1 :], dtype=np.int64))
    return outer, int(shape[axis]), inner


def choose_groups(n: int, min_groups: int = 1) -> int:
    if n == 0:
        return 1
    return max(min_groups, min(1024, -(-n // 4096)))


# --------------------------------------------------------------------------
# Host drivers (reference implementation of the multi-pass protocol)
# --------------------------------------------------------------------------
def tree_reduce(lib, op, hi, lo, outer, n, inner, min_groups=1):
    first, second = TREE_OPS[op]
    name = f"{first}_df64"
    hi, lo = hi.contiguous(), lo.contiguous()
    while True:
        groups = choose_groups(n, min_groups)
        out_hi = torch.empty(outer * groups * inner, dtype=torch.float32, device=MPS)
        out_lo = torch.empty_like(out_hi)
        getattr(lib, name)(
            hi,
            lo,
            out_hi,
            out_lo,
            outer,
            n,
            inner,
            groups,
            threads=[outer * inner * groups * TG, 1, 1],
            group_size=[TG, 1, 1],
        )
        if groups == 1:
            return decode(out_hi, out_lo).reshape(outer, inner)
        hi, lo, n, name, min_groups = out_hi, out_lo, groups, f"{second}_df64", 1


def seq_reduce(lib, op, hi, lo, outer, n, inner):
    out_hi = torch.empty(outer * inner, dtype=torch.float32, device=MPS)
    out_lo = torch.empty_like(out_hi)
    getattr(lib, f"{op}_seq_df64")(
        hi.contiguous(),
        lo.contiguous(),
        out_hi,
        out_lo,
        outer,
        n,
        inner,
        threads=[outer * inner, 1, 1],
    )
    return decode(out_hi, out_lo).reshape(outer, inner)


def reduce_axis(lib, op, x, axis, mode="tree", min_groups=1):
    outer, n, inner = layout(x.shape, axis)
    hi, lo, decoded = encode(x)
    if mode == "tree":
        r = tree_reduce(lib, op, hi, lo, outer, n, inner, min_groups)
    else:
        r = seq_reduce(lib, op, hi, lo, outer, n, inner)
    return r.reshape(x.shape[:axis] + x.shape[axis + 1 :]), decoded


def tree_arg_reduce(lib, op, hi, lo, outer, n, inner, min_groups=1):
    name = f"{op}_df64"
    hi, lo = hi.contiguous(), lo.contiguous()
    idx = torch.zeros(1, dtype=torch.int64, device=MPS)
    has_idx = 0
    while True:
        groups = choose_groups(n, min_groups)
        out_hi = torch.empty(outer * groups * inner, dtype=torch.float32, device=MPS)
        out_lo = torch.empty_like(out_hi)
        out_idx = torch.empty(outer * groups * inner, dtype=torch.int64, device=MPS)
        getattr(lib, name)(
            hi,
            lo,
            idx,
            out_hi,
            out_lo,
            out_idx,
            outer,
            n,
            inner,
            groups,
            has_idx,
            threads=[outer * inner * groups * TG, 1, 1],
            group_size=[TG, 1, 1],
        )
        if groups == 1:
            return out_idx.cpu().numpy().reshape(outer, inner)
        hi, lo, idx, n, has_idx, min_groups = out_hi, out_lo, out_idx, groups, 1, 1


def seq_arg_reduce(lib, op, hi, lo, outer, n, inner):
    out_hi = torch.empty(outer * inner, dtype=torch.float32, device=MPS)
    out_lo = torch.empty_like(out_hi)
    out_idx = torch.empty(outer * inner, dtype=torch.int64, device=MPS)
    dummy = torch.zeros(1, dtype=torch.int64, device=MPS)
    getattr(lib, f"{op}_seq_df64")(
        hi.contiguous(),
        lo.contiguous(),
        dummy,
        out_hi,
        out_lo,
        out_idx,
        outer,
        n,
        inner,
        0,
        threads=[outer * inner, 1, 1],
    )
    return out_idx.cpu().numpy().reshape(outer, inner)


def arg_reduce_axis(lib, op, x, axis, mode="tree", min_groups=1):
    outer, n, inner = layout(x.shape, axis)
    hi, lo, decoded = encode(x)
    if mode == "tree":
        r = tree_arg_reduce(lib, op, hi, lo, outer, n, inner, min_groups)
    else:
        r = seq_arg_reduce(lib, op, hi, lo, outer, n, inner)
    return r.reshape(x.shape[:axis] + x.shape[axis + 1 :]), decoded


def cumsum_axis(lib, x, axis):
    outer, n, inner = layout(x.shape, axis)
    hi, lo, decoded = encode(x)
    out_hi = torch.empty_like(hi)
    out_lo = torch.empty_like(lo)
    lib.cumsum_df64(
        hi, lo, out_hi, out_lo, outer, n, inner, threads=[outer * inner, 1, 1]
    )
    return decode(out_hi, out_lo).reshape(x.shape), decoded


# --------------------------------------------------------------------------
# Exact references
# --------------------------------------------------------------------------
def fsum_axis(x: np.ndarray, axis: int) -> np.ndarray:
    return np.apply_along_axis(lambda v: math.fsum(v.tolist()), axis, x)


def exact_cumsum_1d(v: np.ndarray) -> np.ndarray:
    """Exact prefix sums of dyadic float64 values via scaled Python integers."""
    scale = 2**120
    ints = [int(math.ldexp(float(t), 120)) for t in v]
    assert all(
        math.ldexp(float(t), 120) == float(i) for t, i in zip(v, ints, strict=True)
    )
    prefix = list(itertools.accumulate(ints))
    return np.array([p / scale for p in prefix], dtype=np.float64)


def rel_err_u2(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return np.abs(got - ref) / np.maximum(np.abs(ref), np.finfo(np.float64).tiny) / U2


def same_values(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_nan=True
    )


# --------------------------------------------------------------------------
# sum / nansum
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1000, 2**17, 2**22])
def test_sum_random_positive_relative(lib, n):
    rng = np.random.default_rng(n)
    x = rng.random(n) * rng.choice([1e-3, 1.0, 1e5], size=n)
    got, decoded = reduce_axis(lib, "sum", x, 0)
    ref = math.fsum(decoded.tolist())
    err = rel_err_u2(got[()], ref)
    assert err <= 8 * math.log2(n), f"sum rel err {err:.2f} u^2 for n={n}"


@pytest.mark.parametrize("n", [4097, 2**20])
def test_sum_random_signed_backward_error(lib, n):
    rng = np.random.default_rng(7 * n)
    x = rng.standard_normal(n) * 10.0 ** rng.integers(-3, 4, size=n)
    got, decoded = reduce_axis(lib, "sum", x, 0)
    ref = math.fsum(decoded.tolist())
    abs_err = abs(float(got[()]) - ref)
    assert abs_err <= 8 * math.log2(n) * U2 * np.abs(decoded).sum()


@pytest.mark.parametrize("mode", ["tree", "seq"])
def test_sum_integer_exact(lib, mode):
    rng = np.random.default_rng(3)
    for n, low, high in (
        (10**6, -1000, 1000),
        (2**20, 0, 2**20),
        (5000, -(2**30), 2**30),
    ):
        x = rng.integers(low, high, size=n).astype(np.float64)
        got, decoded = reduce_axis(lib, "sum", x, 0, mode=mode)
        assert float(got[()]) == math.fsum(decoded.tolist())


@pytest.mark.parametrize("mode", ["tree", "seq"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sum_along_axes(lib, mode, axis):
    rng = np.random.default_rng(axis + 10)
    x = rng.standard_normal((7, 33, 5)) * 3.0
    got, decoded = reduce_axis(lib, "sum", x, axis, mode=mode, min_groups=3)
    ref = fsum_axis(decoded, axis)
    abs_ref = np.abs(decoded).sum(axis=axis)
    assert got.shape == ref.shape
    assert np.all(np.abs(got - ref) <= 8 * 6 * U2 * abs_ref)


def test_sum_multipass_forced_groups(lib):
    rng = np.random.default_rng(11)
    x = rng.random(3000)
    got, decoded = reduce_axis(
        lib, "sum", x, 0, min_groups=5
    )  # 5 groups then a 5-element pass
    err = rel_err_u2(got[()], math.fsum(decoded.tolist()))
    assert err <= 8 * math.log2(3000)


@pytest.mark.parametrize("mode", ["tree", "seq"])
def test_sum_special_values(lib, mode):
    cases = {
        "nan_propagates": (np.array([1.0, np.nan, 2.0]), np.nan),
        "inf": (np.array([1.0, np.inf, 2.0]), np.inf),
        "neg_inf": (np.array([-np.inf, 1.0]), -np.inf),
        "inf_minus_inf": (np.array([np.inf, 3.0, -np.inf]), np.nan),
        "empty": (np.zeros(0), 0.0),
        "single": (np.array([-2.5]), -2.5),
        "cancel": (np.array([1e10, 1.0, -1e10]), 1.0),
        "ones": (np.ones(1000), 1000.0),
    }
    for name, (x, expect) in cases.items():
        got, _ = reduce_axis(lib, "sum", x, 0, mode=mode)
        assert same_values(got[()], expect), (
            f"sum {name}: got {got[()]} expected {expect}"
        )


@pytest.mark.parametrize("mode", ["tree", "seq"])
def test_nansum(lib, mode):
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 5000, 3))
    x[rng.random(x.shape) < 0.1] = np.nan
    got, decoded = reduce_axis(lib, "nansum", x, 1, mode=mode, min_groups=4)
    ref = fsum_axis(np.nan_to_num(decoded, nan=0.0), 1)
    abs_ref = np.nansum(np.abs(decoded), axis=1)
    assert np.all(np.abs(got - ref) <= 8 * 13 * U2 * abs_ref)
    # all-NaN line -> 0; inf partials still propagate through the second pass
    got, _ = reduce_axis(
        lib, "nansum", np.full(700, np.nan), 0, mode=mode, min_groups=3
    )
    assert got[()] == 0.0
    got, _ = reduce_axis(
        lib,
        "nansum",
        np.array([np.inf, np.nan, -np.inf] + [1.0] * 600),
        0,
        mode=mode,
        min_groups=3,
    )
    assert np.isnan(got[()])
    got, _ = reduce_axis(lib, "nansum", np.array([np.nan, np.inf, 2.0]), 0, mode=mode)
    assert got[()] == np.inf


# --------------------------------------------------------------------------
# prod
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["tree", "seq"])
@pytest.mark.parametrize("n", [1, 7, 64])
def test_prod_random(lib, mode, n):
    rng = np.random.default_rng(100 + n)
    x = rng.uniform(0.5, 2.0, size=(3, n)) * rng.choice([-1.0, 1.0], size=(3, n))
    got, decoded = reduce_axis(lib, "prod", x, 1, mode=mode, min_groups=2)
    for row in range(3):
        ref = float(math.prod(Fraction(float(t)) for t in decoded[row]))
        err = rel_err_u2(got[row], ref)
        assert err <= n * 5, f"prod rel err {err:.2f} u^2 for n={n}"


@pytest.mark.parametrize("mode", ["tree", "seq"])
def test_prod_special_values(lib, mode):
    cases = {
        "empty": (np.zeros(0), 1.0),
        "zero": (np.array([2.0, 0.0, 3.0]), 0.0),
        "nan": (np.array([2.0, np.nan]), np.nan),
        "inf": (np.array([2.0, np.inf]), np.inf),
        "inf_zero": (np.array([np.inf, 0.0]), np.nan),
        "neg": (np.array([-1.0, -1.0, -1.0]), -1.0),
    }
    for name, (x, expect) in cases.items():
        got, _ = reduce_axis(lib, "prod", x, 0, mode=mode)
        assert same_values(got[()], expect), f"prod {name}: got {got[()]}"


# --------------------------------------------------------------------------
# max / min / nanmax / nanmin
# --------------------------------------------------------------------------
def _tie_data(rng, shape):
    return rng.integers(-3, 4, size=shape).astype(np.float64) * 0.25


@pytest.mark.parametrize("mode", ["tree", "seq"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_max_min_random_and_ties(lib, mode, axis):
    rng = np.random.default_rng(20 + axis)
    for x in (rng.standard_normal((5, 3000, 4)), _tie_data(rng, (5, 3000, 4))):
        for op, ref_fn in (("max", np.max), ("min", np.min)):
            got, decoded = reduce_axis(lib, op, x, axis, mode=mode, min_groups=3)
            assert same_values(got, ref_fn(decoded, axis=axis)), (
                f"{op} axis={axis} {mode}"
            )


@pytest.mark.parametrize("mode", ["tree", "seq"])
def test_max_min_nan_and_inf(lib, mode):
    rng = np.random.default_rng(21)
    x = rng.standard_normal((6, 2000))
    x[0, 1234] = np.nan
    x[1, 7] = np.inf
    x[2, 9] = -np.inf
    x[3, :] = np.nan
    x[4, 5] = np.nan
    x[4, 1999] = np.inf
    for op, ref_fn, nan_fn in (("max", np.max, np.nanmax), ("min", np.min, np.nanmin)):
        got, decoded = reduce_axis(lib, op, x, 1, mode=mode, min_groups=3)
        assert same_values(got, ref_fn(decoded, axis=1)), f"{op}: {got}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = nan_fn(decoded, axis=1)
        got, _ = reduce_axis(lib, f"nan{op}", x, 1, mode=mode, min_groups=3)
        assert same_values(got, ref), f"nan{op}: {got}"
    # +-0 ties are value-equal
    got, _ = reduce_axis(lib, "max", np.array([-0.0, 0.0, -1.0]), 0, mode=mode)
    assert got[()] == 0.0
    got, _ = reduce_axis(lib, "min", np.array([0.0, -0.0, 1.0]), 0, mode=mode)
    assert got[()] == 0.0


# --------------------------------------------------------------------------
# argmax / argmin
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["tree", "seq"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_argmax_argmin_random_and_ties(lib, mode, axis):
    rng = np.random.default_rng(30 + axis)
    for x in (
        rng.standard_normal((5, 3000, 4)),
        _tie_data(rng, (5, 3000, 4)),
        np.zeros((5, 3000, 4)),
    ):
        for op, ref_fn in (("argmax", np.argmax), ("argmin", np.argmin)):
            got, decoded = arg_reduce_axis(lib, op, x, axis, mode=mode, min_groups=3)
            assert np.array_equal(got, ref_fn(decoded, axis=axis)), (
                f"{op} axis={axis} {mode}"
            )


@pytest.mark.parametrize("mode", ["tree", "seq"])
def test_argmax_argmin_nan_first_wins(lib, mode):
    rng = np.random.default_rng(31)
    x = rng.standard_normal((5, 5000))
    x[0, 4321] = np.nan
    x[1, 12] = np.nan
    x[1, 4000] = np.nan
    x[2, :] = np.nan
    x[3, 0] = np.inf
    x[3, 1] = -np.inf
    x[4, 100] = np.nan
    x[4, 99] = np.inf
    for op, ref_fn in (("argmax", np.argmax), ("argmin", np.argmin)):
        got, decoded = arg_reduce_axis(lib, op, x, 1, mode=mode, min_groups=4)
        assert np.array_equal(got, ref_fn(decoded, axis=1)), f"{op}: {got}"
    # signed-zero ties: first index
    for op in ("argmax", "argmin"):
        got, _ = arg_reduce_axis(
            lib, op, np.array([[-0.0, 0.0], [0.0, -0.0]]), 1, mode=mode
        )
        assert np.array_equal(got, [0, 0])


def test_argmax_long_line_multipass(lib):
    rng = np.random.default_rng(32)
    x = rng.random(2**20)
    x[777_777] = 2.0
    x[999_999] = 2.0  # later tie must lose
    got, decoded = arg_reduce_axis(lib, "argmax", x, 0)
    assert int(got[()]) == 777_777 == int(np.argmax(decoded))
    got, _ = arg_reduce_axis(lib, "argmin", -x, 0)
    assert int(got[()]) == 777_777


# --------------------------------------------------------------------------
# cumsum
# --------------------------------------------------------------------------
def test_cumsum_long_line_prefix_relative(lib):
    rng = np.random.default_rng(40)
    x = rng.random(100_000) * rng.choice([1e-2, 1.0, 1e3], size=100_000)
    got, decoded = cumsum_axis(lib, x, 0)
    ref = exact_cumsum_1d(decoded)
    err = rel_err_u2(got, ref)
    assert err.max() <= 8, f"cumsum max prefix rel err {err.max():.2f} u^2"


def test_cumsum_signed_and_cancellation(lib):
    rng = np.random.default_rng(41)
    x = rng.standard_normal(20_000)
    got, decoded = cumsum_axis(lib, x, 0)
    ref = exact_cumsum_1d(decoded)
    running_abs = np.cumsum(np.abs(decoded))
    assert np.all(np.abs(got - ref) <= 8 * U2 * running_abs)
    # big partial sums that cancel back to O(1)
    x = np.array([1e12, 1.0, -1e12, 2.0, 1e-3, -3.0])
    got, decoded = cumsum_axis(lib, x, 0)
    ref = exact_cumsum_1d(decoded)
    # rounding of each prefix to df64 (<= ~1u^2 of the prefix) plus the third
    # word's rounding (~2^-72 of the running magnitude)
    running_max = np.maximum.accumulate(np.abs(decoded))
    assert np.all(
        np.abs(got - ref) <= 4 * U2 * np.maximum(np.abs(ref), 2.0**-24 * running_max)
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cumsum_axes_and_integers(lib, axis):
    rng = np.random.default_rng(42 + axis)
    x = rng.integers(-1000, 1000, size=(4, 50, 3)).astype(np.float64)
    got, decoded = cumsum_axis(lib, x, axis)
    assert np.array_equal(got, np.cumsum(decoded, axis=axis))
    x = rng.standard_normal((4, 50, 3))
    got, decoded = cumsum_axis(lib, x, axis)
    ref = np.apply_along_axis(exact_cumsum_1d, axis, decoded)
    running_abs = np.cumsum(np.abs(decoded), axis=axis)
    assert np.all(np.abs(got - ref) <= 8 * U2 * running_abs)


def test_cumsum_special_values(lib):
    x = np.array([1.0, np.inf, 2.0, -np.inf, 5.0])
    got, _ = cumsum_axis(lib, x, 0)
    assert same_values(got, [1.0, np.inf, np.inf, np.nan, np.nan])
    x = np.array([1.0, np.nan, 2.0])
    got, _ = cumsum_axis(lib, x, 0)
    assert same_values(got, [1.0, np.nan, np.nan])
    got, _ = cumsum_axis(lib, np.zeros(0), 0)
    assert got.shape == (0,)


# --------------------------------------------------------------------------
# sf64 twins (skipped until sf64_core.h exists)
# --------------------------------------------------------------------------
def sf_encode(x: np.ndarray) -> torch.Tensor:
    x = np.ascontiguousarray(x, dtype=np.float64)
    return torch.from_numpy(x.view(np.int64).copy()).to(MPS)


def sf_decode(bits: torch.Tensor) -> np.ndarray:
    return bits.cpu().numpy().view(np.float64)


def sf_tree_reduce(sflib, op, x, axis, min_groups=1):
    first, second = TREE_OPS[op]
    outer, n, inner = layout(x.shape, axis)
    bits = sf_encode(x)
    name = f"{first}_sf64"
    while True:
        groups = choose_groups(n, min_groups)
        out = torch.empty(outer * groups * inner, dtype=torch.int64, device=MPS)
        getattr(sflib, name)(
            bits,
            out,
            outer,
            n,
            inner,
            groups,
            threads=[outer * inner * groups * TG, 1, 1],
            group_size=[TG, 1, 1],
        )
        if groups == 1:
            return sf_decode(out).reshape(x.shape[:axis] + x.shape[axis + 1 :])
        bits, n, name, min_groups = out, groups, f"{second}_sf64", 1


def sf_seq_reduce(sflib, op, x, axis):
    outer, n, inner = layout(x.shape, axis)
    out = torch.empty(outer * inner, dtype=torch.int64, device=MPS)
    getattr(sflib, f"{op}_seq_sf64")(
        sf_encode(x), out, outer, n, inner, threads=[outer * inner, 1, 1]
    )
    return sf_decode(out).reshape(x.shape[:axis] + x.shape[axis + 1 :])


def sf_arg_reduce(sflib, op, x, axis, mode="tree", min_groups=1):
    outer, n, inner = layout(x.shape, axis)
    bits = sf_encode(x)
    if mode == "seq":
        out = torch.empty(outer * inner, dtype=torch.int64, device=MPS)
        out_idx = torch.empty_like(out)
        getattr(sflib, f"{op}_seq_sf64")(
            bits,
            torch.zeros(1, dtype=torch.int64, device=MPS),
            out,
            out_idx,
            outer,
            n,
            inner,
            0,
            threads=[outer * inner, 1, 1],
        )
        return out_idx.cpu().numpy().reshape(x.shape[:axis] + x.shape[axis + 1 :])
    idx = torch.zeros(1, dtype=torch.int64, device=MPS)
    has_idx = 0
    while True:
        groups = choose_groups(n, min_groups)
        out = torch.empty(outer * groups * inner, dtype=torch.int64, device=MPS)
        out_idx = torch.empty_like(out)
        getattr(sflib, f"{op}_sf64")(
            bits,
            idx,
            out,
            out_idx,
            outer,
            n,
            inner,
            groups,
            has_idx,
            threads=[outer * inner * groups * TG, 1, 1],
            group_size=[TG, 1, 1],
        )
        if groups == 1:
            return out_idx.cpu().numpy().reshape(x.shape[:axis] + x.shape[axis + 1 :])
        bits, idx, n, has_idx, min_groups = out, out_idx, groups, 1, 1


@pytest.mark.parametrize("mode", ["tree", "seq"])
def test_sf64_sum_and_nansum(sflib, mode):
    rng = np.random.default_rng(50)
    x = rng.standard_normal((3, 20_000, 2)) * 10.0 ** rng.integers(
        -3, 4, size=(3, 20_000, 2)
    )
    run = sf_tree_reduce if mode == "tree" else sf_seq_reduce
    got = run(sflib, "sum", x, 1)
    ref = fsum_axis(x, 1)
    # accumulation depth <= (elements per thread) + tree levels + second pass
    depth = (20_000 // (TG * choose_groups(20_000))) + 8 + choose_groups(20_000) + 8
    assert np.all(np.abs(got - ref) <= depth * 2.0**-53 * np.abs(x).sum(axis=1))
    xi = rng.integers(-1000, 1000, size=(3, 20_000, 2)).astype(np.float64)
    assert np.array_equal(run(sflib, "sum", xi, 1), xi.sum(axis=1))
    xn = x.copy()
    xn[rng.random(xn.shape) < 0.1] = np.nan
    got = run(sflib, "nansum", xn, 1)
    ref = fsum_axis(np.nan_to_num(xn, nan=0.0), 1)
    assert np.all(np.abs(got - ref) <= depth * 2.0**-53 * np.nansum(np.abs(xn), axis=1))
    assert np.isnan(run(sflib, "sum", xn, 1)).all()
    assert np.isnan(run(sflib, "nansum", np.array([np.inf, np.nan, -np.inf]), 0))


@pytest.mark.parametrize("mode", ["tree", "seq"])
def test_sf64_prod_max_min_arg(sflib, mode):
    rng = np.random.default_rng(51)
    x = rng.uniform(0.5, 2.0, size=(3, 40))
    run = sf_tree_reduce if mode == "tree" else sf_seq_reduce
    got = run(sflib, "prod", x, 1)
    for row in range(3):
        ref = float(math.prod(Fraction(float(t)) for t in x[row]))
        assert abs(got[row] - ref) <= 40 * 2.0**-53 * abs(ref)
    y = _tie_data(rng, (5, 3000, 4))
    y[0, 100, 0] = np.nan
    y[1, :, 1] = np.nan
    for op, ref_fn in (("max", np.max), ("min", np.min)):
        assert same_values(run(sflib, op, y, 1), ref_fn(y, axis=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = (np.nanmax if op == "max" else np.nanmin)(y, axis=1)
        assert same_values(run(sflib, f"nan{op}", y, 1), ref)
    for op, ref_fn in (("argmax", np.argmax), ("argmin", np.argmin)):
        got = sf_arg_reduce(sflib, op, y, 1, mode=mode, min_groups=3)
        assert np.array_equal(got, ref_fn(y, axis=1))


def test_sf64_cumsum_bit_exact(sflib):
    rng = np.random.default_rng(52)
    x = rng.standard_normal((4, 5000, 3)) * 10.0 ** rng.integers(
        -3, 4, size=(4, 5000, 3)
    )
    x[0, 10, 0] = np.nan
    x[1, 20, 1] = np.inf
    outer, n, inner = layout(x.shape, 1)
    out = torch.empty(x.size, dtype=torch.int64, device=MPS)
    sflib.cumsum_sf64(sf_encode(x), out, outer, n, inner, threads=[outer * inner, 1, 1])
    got = sf_decode(out).reshape(x.shape)
    assert same_values(got, np.cumsum(x, axis=1))  # numpy cumsum is sequential
