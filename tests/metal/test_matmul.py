"""Tests for ``kernels/matmul.metal`` (df64 batched GEMM and dot; sf64 twins).

References are exact: the decoded inputs (hi + lo, exact in float64) are dyadic
rationals, so they are scaled to Python integers, multiplied with object-dtype
``numpy`` matmul, and the exact result is divided back with correctly rounded
integer true division. Errors are reported in units of u^2 = 2^-48.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import math  # noqa: E402
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
SCALE_BITS = 120
KERNEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optiland"
    / "backend"
    / "torch_backend"
    / "metal"
    / "kernels"
)
MPS = torch.device("mps")


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
    return compile_library(kernel_source("df64_core.h", "reduce.metal", "matmul.metal"))


@pytest.fixture(scope="module")
def sflib():
    src = sf64_source("reduce.metal", "matmul.metal")
    if src is None:
        pytest.skip("sf64_core.h is not available yet")
    return compile_library(src)


# --------------------------------------------------------------------------
# Encoding and exact references
# --------------------------------------------------------------------------
def encode(x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    hi = x.astype(np.float32)
    with np.errstate(invalid="ignore"):  # inf - inf for non-finite entries
        lo = np.where(np.isfinite(hi), (x - hi.astype(np.float64)), 0.0).astype(
            np.float32
        )
    decoded = hi.astype(np.float64) + lo.astype(np.float64)
    return torch.from_numpy(hi).to(MPS), torch.from_numpy(lo).to(MPS), decoded


def decode(hi: torch.Tensor, lo: torch.Tensor) -> np.ndarray:
    return hi.cpu().numpy().astype(np.float64) + lo.cpu().numpy().astype(np.float64)


def to_ints(x: np.ndarray) -> np.ndarray:
    """Scale dyadic float64 values to exact Python integers (object array)."""
    flat = [int(math.ldexp(float(t), SCALE_BITS)) for t in x.ravel()]
    assert all(
        math.ldexp(float(t), SCALE_BITS) == float(i)
        for t, i in zip(x.ravel(), flat, strict=True)
    ), "input is not exactly representable at the chosen scale"
    out = np.empty(x.size, dtype=object)
    out[:] = flat
    return out.reshape(x.shape)


def exact_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correctly rounded float64 of the exact product A @ B (batched)."""
    ai, bi = to_ints(a), to_ints(b)
    ci = np.matmul(ai, bi)
    denom = 1 << (2 * SCALE_BITS)
    out = np.array([c / denom for c in ci.ravel()], dtype=np.float64)
    return out.reshape(ci.shape)


def abs_products(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """sum_k |a_mk b_kn| (float64 is plenty for a tolerance scale)."""
    return np.matmul(np.abs(a), np.abs(b))


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
def matmul(lib, a: np.ndarray, b: np.ndarray):
    batch, m, k = a.shape
    _, k2, n = b.shape
    assert k == k2
    a_hi, a_lo, a_dec = encode(a)
    b_hi, b_lo, b_dec = encode(b)
    c_hi = torch.empty(batch * m * n, dtype=torch.float32, device=MPS)
    c_lo = torch.empty_like(c_hi)
    lib.matmul_df64(
        a_hi,
        a_lo,
        b_hi,
        b_lo,
        c_hi,
        c_lo,
        batch,
        m,
        k,
        n,
        threads=[batch * m * n, 1, 1],
    )
    return decode(c_hi, c_lo).reshape(batch, m, n), a_dec, b_dec


def dot(lib, x: np.ndarray, y: np.ndarray, groups: int | None = None):
    n = x.size
    if groups is None:
        groups = max(1, min(1024, -(-n // 4096)))
    x_hi, x_lo, x_dec = encode(x)
    y_hi, y_lo, y_dec = encode(y)
    p_hi = torch.empty(groups, dtype=torch.float32, device=MPS)
    p_lo = torch.empty_like(p_hi)
    lib.dot_df64(
        x_hi,
        x_lo,
        y_hi,
        y_lo,
        p_hi,
        p_lo,
        n,
        groups,
        threads=[groups * TG, 1, 1],
        group_size=[TG, 1, 1],
    )
    if groups > 1:  # finish with the sum kernel from reduce.metal (outer = inner = 1)
        s_hi = torch.empty(1, dtype=torch.float32, device=MPS)
        s_lo = torch.empty_like(s_hi)
        lib.sum_df64(
            p_hi,
            p_lo,
            s_hi,
            s_lo,
            1,
            groups,
            1,
            1,
            threads=[TG, 1, 1],
            group_size=[TG, 1, 1],
        )
        p_hi, p_lo = s_hi, s_lo
    return float(decode(p_hi, p_lo)[0]), x_dec, y_dec


SHAPES = [
    # (batch, M, K, N)
    (1, 8, 4096, 8),
    (2, 5, 37, 3),
    (1, 1, 64, 9),
    (1, 7, 64, 1),
    (3, 1, 100, 1),
    (4, 3, 1, 5),
    (1, 4, 40000, 4),
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_matmul_random_positive_relative(lib, shape):
    batch, m, k, n = shape
    rng = np.random.default_rng(sum(shape))
    a = rng.uniform(0.5, 1.5, size=(batch, m, k))
    b = rng.uniform(0.5, 1.5, size=(batch, k, n))
    got, a_dec, b_dec = matmul(lib, a, b)
    ref = exact_matmul(a_dec, b_dec)
    err = (np.abs(got - ref) / np.abs(ref) / U2).max()
    assert err <= k * 10, f"matmul {shape}: max rel err {err:.2f} u^2"


@pytest.mark.parametrize(
    "shape",
    [(2, 6, 512, 5), (1, 3, 4096, 3), (1, 2, 40000, 2)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_matmul_random_signed_backward_error(lib, shape):
    batch, m, k, n = shape
    rng = np.random.default_rng(1 + sum(shape))
    a = rng.standard_normal((batch, m, k)) * 10.0 ** rng.integers(
        -2, 3, size=(batch, m, k)
    )
    b = rng.standard_normal((batch, k, n)) * 10.0 ** rng.integers(
        -2, 3, size=(batch, k, n)
    )
    got, a_dec, b_dec = matmul(lib, a, b)
    ref = exact_matmul(a_dec, b_dec)
    scale = abs_products(a_dec, b_dec)
    err = (np.abs(got - ref) / scale / U2).max()
    assert err <= k * 10, f"matmul {shape}: max backward err {err:.2f} u^2"


def test_matmul_integer_exact_and_identity(lib):
    rng = np.random.default_rng(9)
    a = rng.integers(-1000, 1000, size=(2, 6, 1000)).astype(np.float64)
    b = rng.integers(-1000, 1000, size=(2, 1000, 4)).astype(np.float64)
    got, a_dec, b_dec = matmul(lib, a, b)
    assert np.array_equal(got, exact_matmul(a_dec, b_dec))
    x = rng.standard_normal((1, 9, 9))
    eye = np.eye(9)[None]
    got, x_dec, _ = matmul(lib, x, eye)
    assert np.array_equal(got, x_dec)
    got, _, x_dec = matmul(lib, eye, x)
    assert np.array_equal(got, x_dec)


def test_matmul_special_values(lib):
    a = np.array([[[1.0, 2.0], [np.nan, 1.0], [np.inf, 0.0], [1.0, 0.0]]])
    b = np.array([[[1.0, 0.0, -np.inf], [0.0, 1.0, 1.0]]])
    got, a_dec, b_dec = matmul(lib, a, b)
    ref = a_dec @ b_dec  # numpy float64 IEEE semantics
    assert np.array_equal(got, ref, equal_nan=True), f"{got} vs {ref}"
    z = np.zeros((1, 3, 5))
    got, _, _ = matmul(lib, z, np.ones((1, 5, 2)))
    assert np.array_equal(got, np.zeros((1, 3, 2)))


@pytest.mark.parametrize("n", [1, 1000, 4097, 2**20])
def test_dot_random(lib, n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    got, x_dec, y_dec = dot(lib, x, y)
    ref = float(exact_matmul(x_dec[None, None, :], y_dec[None, :, None])[0, 0, 0])
    scale = float(np.abs(x_dec) @ np.abs(y_dec))
    assert abs(got - ref) <= 8 * (math.log2(n) + 1) * U2 * scale
    # single-group path and matmul(M=N=1) agree with the multi-group path
    got1, _, _ = dot(lib, x, y, groups=1)
    assert abs(got1 - ref) <= 8 * (math.log2(n) + 1) * U2 * scale
    mm, _, _ = matmul(lib, x[None, None, :], y[None, :, None])
    assert abs(float(mm[0, 0, 0]) - ref) <= 8 * (math.log2(n) + 1) * U2 * scale


# --------------------------------------------------------------------------
# sf64 twins (bit-exact against a sequential float64 loop)
# --------------------------------------------------------------------------
def sf_encode(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(x, dtype=np.float64).view(np.int64).copy()
    ).to(MPS)


def sequential_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    batch, m, k = a.shape
    n = b.shape[2]
    out = np.zeros((batch, m, n))
    for bb in range(batch):
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for kk in range(k):
                    acc += float(a[bb, i, kk]) * float(b[bb, kk, j])
                out[bb, i, j] = acc
    return out


@pytest.mark.parametrize(
    "shape",
    [(2, 3, 64, 3), (1, 2, 4096, 2), (1, 1, 5, 1)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_sf64_matmul_bit_exact(sflib, shape):
    batch, m, k, n = shape
    rng = np.random.default_rng(77 + k)
    a = rng.standard_normal((batch, m, k)) * 10.0 ** rng.integers(
        -3, 4, size=(batch, m, k)
    )
    b = rng.standard_normal((batch, k, n)) * 10.0 ** rng.integers(
        -3, 4, size=(batch, k, n)
    )
    a[0, 0, 0] = np.nan if k > 5 else a[0, 0, 0]
    c = torch.empty(batch * m * n, dtype=torch.int64, device=MPS)
    sflib.matmul_sf64(
        sf_encode(a), sf_encode(b), c, batch, m, k, n, threads=[batch * m * n, 1, 1]
    )
    got = c.cpu().numpy().view(np.float64).reshape(batch, m, n)
    ref = sequential_matmul(a, b)
    assert np.array_equal(got, ref, equal_nan=True)


def test_sf64_dot(sflib):
    rng = np.random.default_rng(78)
    n = 20_000
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    groups = 5
    p = torch.empty(groups, dtype=torch.int64, device=MPS)
    sflib.dot_sf64(
        sf_encode(x),
        sf_encode(y),
        p,
        n,
        groups,
        threads=[groups * TG, 1, 1],
        group_size=[TG, 1, 1],
    )
    s = torch.empty(1, dtype=torch.int64, device=MPS)
    sflib.sum_sf64(p, s, 1, groups, 1, 1, threads=[TG, 1, 1], group_size=[TG, 1, 1])
    got = float(s.cpu().numpy().view(np.float64)[0])
    ref = float(exact_matmul(x[None, None, :], y[None, :, None])[0, 0, 0])
    depth = n // (TG * groups) + 8 + groups + 8
    assert abs(got - ref) <= depth * 2.0**-53 * float(np.abs(x) @ np.abs(y))
