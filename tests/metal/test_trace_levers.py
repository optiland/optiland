"""Regression tests for the post-benchmark performance levers.

Three levers were added after the fused trace-interpreter kernel's first
benchmark showed the per-op path *around* the kernel dominating the end-to-end
time (ray generation issued ~620 launches and ~3,700 host-side scalar ops per
trace; the record compiler paid ~40 dispatch-layer ops per row):

1. ``HexagonalDistribution.generate_points`` builds every ring at once
   (``optiland/distribution.py``): a handful of whole-array ops instead of
   four per ring, bit-identical on NumPy.
2. ``SurfaceGroup.paraxial_path_scope`` memoizes ``build_paraxial_path`` for
   the duration of a ray-generation call chain (``RealRayTracer.trace`` /
   ``trace_generic`` and the batch API's bundle generator run inside it).
3. The record compiler reads host-resident emulated scalars as their plain
   CPU float64 tensors (``trace_adapters.host_plain``) and memoizes material
   scalars per compile context (``trace_adapters._material_scalars``).

Each test pins the observable that made the lever worth having, plus the
invariant it must not break.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import numpy as np
import pytest
import torch

import optiland.backend as be
from optiland.backend.torch_backend import metal
from optiland.distribution import create_distribution
from optiland.samples.objectives import CookeTriplet

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal device required"
)


@pytest.fixture
def mps_df64():
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    metal.set_mode("df64")
    be.grad_mode.disable()
    yield
    be.set_backend("numpy")


def _launches() -> int:
    return sum(v for k, v in metal.stats().items() if k.startswith("gpu:"))


def _host_ops() -> int:
    return sum(v for k, v in metal.stats().items() if k.startswith("host:"))


# ---------------------------------------------------------------- lever 1


@pytest.mark.parametrize("num_rings", [3, 42, 182])
def test_hexapolar_matches_per_ring_loop_bitwise_on_numpy(num_rings):
    """The vectorized generator is the per-ring loop, word for word, on NumPy."""
    be.set_backend("numpy")
    d = create_distribution("hexapolar")
    d.generate_points(num_rings)
    x = np.zeros(1)
    y = np.zeros(1)
    r = np.linspace(0, 1, num_rings + 1)
    for i in range(num_rings):
        theta = np.linspace(0, 2 * np.pi, 6 * (i + 1) + 1)[:-1]
        x = np.concatenate([x, r[i + 1] * np.cos(theta)])
        y = np.concatenate([y, r[i + 1] * np.sin(theta)])
    assert np.array_equal(d.x.view(np.int64), x.view(np.int64))
    assert np.array_equal(d.y.view(np.int64), y.view(np.int64))


def test_hexapolar_is_a_handful_of_launches_on_mps(mps_df64):
    """182 rings (about 1e5 points) cost single-digit launches, not ~600."""
    d = create_distribution("hexapolar")
    d.generate_points(182)
    torch.mps.synchronize()
    metal.reset_stats()
    d = create_distribution("hexapolar")
    d.generate_points(182)
    torch.mps.synchronize()
    assert int(d.x.numel()) == 1 + 3 * 182 * 183
    assert _launches() <= 8
    assert bool(torch.isfinite(d.x.to_cpu_float64()).all())


def test_hexapolar_zero_rings_is_the_centre_point():
    be.set_backend("numpy")
    d = create_distribution("hexapolar")
    d.generate_points(0)
    assert d.x.shape == (1,) and d.y.shape == (1,)
    assert float(d.x[0]) == 0.0 and float(d.y[0]) == 0.0


# ---------------------------------------------------------------- lever 2


def test_paraxial_path_scope_memoizes_and_releases(monkeypatch):
    """Inside the scope the path is built once; outside, every call rebuilds."""
    be.set_backend("numpy")
    lens = CookeTriplet()
    group = lens.surfaces
    import optiland.surfaces.surface_group as sg

    calls = {"n": 0}
    real = sg.build_paraxial_path

    def counting(surfaces):
        calls["n"] += 1
        return real(surfaces)

    monkeypatch.setattr(sg, "build_paraxial_path", counting)
    group.build_paraxial_path()
    group.build_paraxial_path()
    assert calls["n"] == 2
    with group.paraxial_path_scope():
        a = group.build_paraxial_path()
        b = group.build_paraxial_path()
        with group.paraxial_path_scope():  # nested scopes share the memo
            c = group.build_paraxial_path()
    assert a is b is c
    assert calls["n"] == 3
    group.build_paraxial_path()
    assert calls["n"] == 4  # released
    assert group._paraxial_path_memo is None


def test_trace_builds_the_paraxial_path_once(monkeypatch):
    """optic.trace rebuilt the path seven times per call before the scope."""
    be.set_backend("numpy")
    lens = CookeTriplet()
    import optiland.surfaces.surface_group as sg

    calls = {"n": 0}
    real = sg.build_paraxial_path

    def counting(surfaces):
        calls["n"] += 1
        return real(surfaces)

    monkeypatch.setattr(sg, "build_paraxial_path", counting)
    lens.trace(Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=6, distribution="hexapolar")
    assert calls["n"] == 1
    # A geometry change after the trace is seen by the next trace (the memo
    # lives only inside the scope): the edge-field chief ray moves in y.
    before = np.array(lens.trace(Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=6).y)
    lens.set_thickness(lens.surfaces.surfaces[2].thickness * 1.05, 2)
    after = np.array(lens.trace(Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=6).y)
    assert not np.array_equal(before, after)


def test_scope_does_not_leak_across_traces(mps_df64):
    """Results with and without the scope are word-identical on the GPU."""
    lens = CookeTriplet()
    rays = lens.trace(Hx=0.0, Hy=0.7, wavelength=0.55, num_rays=12)
    a = rays.x.to_cpu_float64().numpy()
    lens.surfaces._paraxial_path_memo = None
    rays = lens.trace(Hx=0.0, Hy=0.7, wavelength=0.55, num_rays=12)
    b = rays.x.to_cpu_float64().numpy()
    assert np.array_equal(a.view(np.int64), b.view(np.int64))


# ---------------------------------------------------------------- lever 3


def test_host_plain_returns_the_host_tensor_or_the_value(mps_df64):
    from optiland.backend.torch_backend.metal.trace_adapters import host_plain

    small = be.array(1.25)  # host-resident under dual residency
    plain = host_plain(small)
    assert type(plain) is torch.Tensor and plain.dtype == torch.float64
    assert plain.device.type == "cpu"
    assert float(torch.cos(-plain)) == float(be.cos(-small))
    assert host_plain(2.5) == 2.5
    assert host_plain(np.float64(1.0)) == 1.0
    big = be.ones((4096,))  # GPU-resident: returned unchanged
    assert host_plain(big) is big


def test_compile_records_issues_few_host_ops(mps_df64):
    """The record compiler no longer pays ~40 dispatch-layer ops per row."""
    from optiland.backend.torch_backend.metal import trace_record

    lens = CookeTriplet()
    group = lens.surfaces
    w0 = trace_record.canonical_w0(0.55, "df64")
    trace_record.compile_records(group, w0, "df64")
    metal.reset_stats()
    records = trace_record.compile_records(group, w0, "df64")
    assert len(group.surfaces) == records.S
    assert _launches() == 0
    # Two material evaluations per glass (n, k) at most; the pose and conic
    # slots are read from plain host tensors and cost no dispatch at all.
    assert _host_ops() <= 12 * records.S


def test_material_memo_is_per_context_and_identity_keyed(mps_df64, monkeypatch):
    from optiland.backend.torch_backend.metal import trace_record

    lens = CookeTriplet()
    group = lens.surfaces
    w0 = trace_record.canonical_w0(0.55, "df64")
    glass = group.surfaces[1].material_post
    calls = {"n": 0}
    real_n = type(glass).n

    def counting(self, wavelength, **kw):
        if self is glass:
            calls["n"] += 1
        return real_n(self, wavelength, **kw)

    monkeypatch.setattr(type(glass), "n", counting)
    trace_record.compile_records(group, w0, "df64")
    assert calls["n"] == 1  # post of surface 1 and pre of surface 2: one evaluation
    trace_record.compile_records(group, w0, "df64")
    assert calls["n"] == 2  # a new compile context evaluates again
    # Replacing the material (what set_index / set_material do) is a new key.
    lens.set_index(1.7, 1)
    rec = trace_record.compile_records(group, w0, "df64")
    from optiland.backend.torch_backend.metal import trace_layout as L

    assert rec.surf_real[0, 2, L.SR_NPRE] == pytest.approx(1.7)


# ------------------------------------------------ lever 4: kernel occupancy


def test_spherical_twin_is_selected_and_bit_identical(mps_df64):
    """Both entries exist; the twin serves spherical systems, word for word."""
    from optiland.backend.torch_backend.metal import trace, trace_record
    from optiland.raytrace import batch_trace as bt

    assert trace.entry_name("df64", True) == trace.ENTRY["df64"]
    assert trace.entry_name("df64", False) == trace.ENTRY_SPHERICAL["df64"]
    lib = trace._kernel_library("df64")
    assert hasattr(lib, trace.ENTRY["df64"]) and hasattr(
        lib, trace.ENTRY_SPHERICAL["df64"]
    )
    lens = CookeTriplet()
    with torch.no_grad():
        rays = bt._generate_bundle(
            lens, Hx=0.0, Hy=1.0, wavelength=0.55, num_rays=12, distribution="hexapolar"
        )
    n = int(rays.x.numel())
    w0 = trace_record.canonical_w0(0.55, "df64")
    records = trace_record.compile_records(lens.surfaces, w0, "df64", record=True)
    assert records.has_newton is False
    launch = trace._pack_launch(rays, "df64", n)
    outs = []
    for has_newton in (True, False):  # full body, then the spherical twin
        records.has_newton = has_newton
        with torch.no_grad():
            r = trace.launch_trace(
                records, launch, launch_stride=0, N=n, write_final=True, mode="df64"
            )
        torch.mps.synchronize()
        outs.append([c.cpu().numpy().copy() for c in (r.snap + r.final)])
    for a, b in zip(outs[0], outs[1], strict=True):
        assert np.array_equal(a.view(np.int32), b.view(np.int32))


def test_kernel_source_pins_the_occupancy_attribute():
    from pathlib import Path

    src = (
        Path(__file__).parents[2]
        / "optiland/backend/torch_backend/metal/kernels/trace.metal"
    ).read_text()
    assert src.count("[[max_total_threads_per_threadgroup(1024)]]") == 2  # df64 twins
    assert src.count("[[max_total_threads_per_threadgroup(512)]]") == 2  # sf64 twins
    assert "kernel void trace_surfaces_df64_spherical(" in src
    assert "kernel void trace_surfaces_sf64_spherical(" in src


# ------------------------------------------------ lever 5: host fast path


def test_host_fast_path_matches_the_general_path(mps_df64):
    """The fast path returns what the general host path returns, and counts the same."""
    from optiland.backend.torch_backend.metal import tensor as T

    aten = torch.ops.aten
    a = be.array([1.25, -2.5, 0.75])
    b = be.array([0.5, 4.0, -1.0])
    for func, args in (
        (aten.mul.Tensor, (a, b)),
        (aten.add.Tensor, (a, b)),
        (aten.div.Tensor, (a, b)),
        (aten.pow.Tensor_Scalar, (a, 2)),
        (aten.cos.default, (a,)),
        (aten.sqrt.default, (b,)),
        (aten.sum.default, (a,)),
        (aten.where.self, (a > 0, a, b)),
    ):
        metal.reset_stats()
        fast = func(*args)
        fast_stats = dict(metal.stats())
        metal.reset_stats()
        gen = T._run_on_host(func, args, {})
        assert fast_stats == dict(metal.stats())
        assert type(fast) is type(gen)
        if isinstance(fast, T.MetalFloat64):
            assert fast.is_host_resident and gen.is_host_resident
            assert np.array_equal(  # bit patterns: NaN-safe and stricter
                fast._host.numpy().view(np.int64), gen._host.numpy().view(np.int64)
            )
            assert (fast._mode, fast._agnostic, fast.requires_grad) == (
                gen._mode,
                gen._agnostic,
                gen.requires_grad,
            )
        else:
            assert torch.equal(fast, gen)
    assert not any(k.startswith("gpu:") for k in metal.stats())


def test_host_fast_path_defers_what_it_cannot_take(mps_df64):
    """GPU-resident operands, kwargs and mutations take the general path."""
    from optiland.backend.torch_backend.metal import tensor as T

    aten = torch.ops.aten
    small = be.array([1.0, 2.0])
    big = be.ones((4096,))
    assert T._run_on_host_fast(aten.mul.Tensor, (small, big)) is T._NOT_FAST
    assert T._fast_host_ok(aten.mul_.Tensor) is False  # mutating
    assert T._fast_host_ok(aten.view.default) is False  # aliasing
    assert T._fast_host_ok(aten._to_copy.default) is False  # conversion rules
    assert T._fast_host_ok(aten.mul.Tensor) is True
    metal.reset_stats()
    out = big * small.sum()  # a GPU-resident operand: one launch
    assert sum(v for k, v in metal.stats().items() if k.startswith("gpu:")) == 1
    assert not out.is_host_resident


def test_host_fast_path_is_faster(mps_df64):
    """A host-side scalar op costs well under the pre-lever ~19 us (measured 11)."""
    import time

    a = be.array(1.2345)
    b = be.array(0.5)
    for _ in range(300):
        a * b
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(2000):
            a * b
        best = min(best, (time.perf_counter() - t0) / 2000 * 1e6)
    assert best < 16.0, best


def test_host_fast_path_leaves_complex_to_the_general_path(mps_df64):
    """``1j * phase`` (polarized rays) must stay on the CPU complex path."""
    phase = be.array(0.25)
    z = 1j * phase
    assert type(z) is torch.Tensor and z.dtype == torch.complex128
    assert z.device.type == "cpu"
    assert complex(z) == 0.25j
