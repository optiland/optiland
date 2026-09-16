"""Fused conic-intersection kernel vs the shared NumPy algorithm (masks exact)."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover
    pytest.skip("Metal GPU required", allow_module_level=True)

from optiland.backend._conic import _conic_candidates  # noqa: E402
from optiland.backend.torch_backend.metal import encode  # noqa: E402
from optiland.backend.torch_backend.metal.conic import conic_candidates  # noqa: E402
from optiland.backend.torch_backend.metal.tensor import MetalFloat64  # noqa: E402

CASES = [
    (12.345, 0.0),
    (-25.0, -0.7),
    (50.0, -1.0),
    (8.0, 2.0),
    (1e6, 0.0),
    (-3.0, -0.5),
]


def _rays(n: int, seed: int = 11) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-8, 8, n)
    y = rng.uniform(-8, 8, n)
    z = rng.uniform(
        -40, -1, n
    )  # origins in front of the vertex, like rays arriving from a previous surface
    L = rng.uniform(-0.3, 0.3, n)
    M = rng.uniform(-0.3, 0.3, n)
    N = np.sqrt(1 - L * L - M * M)
    k = n // 10
    x[:k] = 0
    y[:k] = 0
    L[:k] = 0
    M[:k] = 0
    N[:k] = 1  # axial rays
    z[k : 2 * k] = 0  # origin on the vertex plane (self-hit handling)
    L[2 * k : 3 * k] = 0.999
    M[2 * k : 3 * k] = 0
    N[2 * k : 3 * k] = np.sqrt(1 - 0.999**2)  # grazing rays
    return x, y, z, L, M, N


def _reference(arrays: list[np.ndarray], radius: float, conic: float, eps: float):
    with np.errstate(all="ignore"):
        return _conic_candidates(
            *arrays, radius, conic, np.where, np.sqrt, np.copysign, lambda v: eps
        )


@pytest.mark.parametrize("mode", ["df64", "sf64"])
@pytest.mark.parametrize("radius,conic", CASES)
def test_masks_match_numpy_and_roots_agree(
    mode: str, radius: float, conic: float
) -> None:
    n = 60000
    arrays = list(_rays(n))
    tensors = [MetalFloat64.from_numpy(a, mode) for a in arrays]
    decoded = [t.to_numpy() for t in tensors]  # exact representable inputs
    eps = 2.0**-48 if mode == "df64" else 2.0**-53
    roots = conic_candidates(*tensors, radius, conic)
    ref = _reference(decoded, radius, conic, eps)
    for got, exp, name in [
        (roots.first_valid, ref.first_valid, "valid1"),
        (roots.second_valid, ref.second_valid, "valid2"),
        (roots.pick_second, ref.pick_second, "pick2"),
        (roots.solvable, ref.solvable, "solvable"),
        (roots.regular, ref.regular, "regular"),
    ]:
        assert np.array_equal(got.cpu().numpy(), exp), name
    t1, t2 = roots.first.to_numpy(), roots.second.to_numpy()
    sel = np.where(
        roots.solvable.cpu().numpy(),
        np.where(roots.pick_second.cpu().numpy(), t2, t1),
        np.nan,
    )
    sel_ref = np.where(
        ref.solvable, np.where(ref.pick_second, ref.second, ref.first), np.nan
    )
    assert np.array_equal(np.isnan(sel), np.isnan(sel_ref))
    both = np.isfinite(sel)
    rel = np.abs(sel[both] - sel_ref[both]) / np.maximum(np.abs(sel_ref[both]), 1e-300)
    if mode == "sf64":
        assert rel.max() == 0.0  # same correctly rounded operations in the same order
    else:
        # Per-op df64 error is ~1e-14; the quadratic's own conditioning (origins
        # almost on the surface, grazing incidence) amplifies it for a few rays, as
        # it does for float64 vs exact arithmetic. Require the bulk to be tight
        # and the tail bounded.
        assert np.median(rel) < 1e-14
        assert np.quantile(rel, 0.999) < 1e-11
        assert rel.max() < 1e-8


def test_scalar_encoding_paths() -> None:
    assert encode.df64_scalar(12.345)[0] == np.float32(12.345)
    assert encode.sf64_scalar(-0.0) == np.array(-0.0).view(np.int64)
