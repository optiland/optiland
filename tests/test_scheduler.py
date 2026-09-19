"""Tests for the shape-aware scheduler in ``optiland.parallel``.

The planning logic is pure Python and is tested exactly; the pool integration
runs a few tiny jobs through real worker processes (spawn) and, when a Metal
device is present, checks that the jobs the plan sent to the GPU worker really
ran the fused kernel there.
"""

from __future__ import annotations

import math
import sys

import pytest

from optiland.parallel import (
    MEASURED_HOST,
    CostModel,
    Job,
    Plan,
    WorkerConfig,
    default_model,
    gpu_available,
    performance_cores,
    plan_jobs,
    run_scheduled,
    trace_rays,
)

RAYS_1E5 = 1 + 3 * 182 * 183  # 182 hexapolar rings
RAYS_1E6 = 1 + 3 * 577 * 578


def _noop(*args, **kwargs):
    return args, kwargs


# ---------------------------------------------------------------- Job / helpers


def test_job_shape_and_steps():
    j = Job(_noop, kind="trace", rays=100, traces=3, surfaces=8)
    assert j.steps_per_trace == 700 and j.steps == 2100
    b = Job(_noop, kind="batch", rays=100, designs=10, surfaces=5)
    assert b.steps == 4 * 100 * 10
    with pytest.raises(ValueError):
        Job(_noop, kind="tpu")


def test_trace_rays_matches_the_distributions():
    assert trace_rays(182) == RAYS_1E5
    assert trace_rays(6) == 1 + 3 * 6 * 7
    assert trace_rays(50, "random") == 50


def test_performance_cores_is_positive_and_not_more_than_cpu_count():
    import os

    n = performance_cores()
    assert 1 <= n <= (os.cpu_count() or n)
    if sys.platform == "darwin":
        # Apple silicon: the performance cluster is smaller than the whole chip
        # whenever efficiency cores exist (M1 Max: 8 of 10).
        assert n <= (os.cpu_count() or n)


# ---------------------------------------------------------------- cost model


def test_shipped_model_reproduces_the_measured_ranking():
    """Measured on the M1 Max (NOTES/08 section 6): the model must rank the same way."""
    m = CostModel()
    small = Job(_noop, kind="trace", rays=1027)
    mid = Job(_noop, kind="trace", rays=RAYS_1E5)
    big = Job(_noop, kind="trace", rays=RAYS_1E6)
    batch = Job(_noop, kind="batch", rays=RAYS_1E5, designs=100)
    # small traces: CPU per-job time far below the GPU's fixed cost
    assert m.seconds(small, "cpu") < 0.3 * m.seconds(small, "gpu")
    # million-ray traces and batches: GPU well ahead
    assert m.seconds(big, "gpu") < 0.3 * m.seconds(big, "cpu")
    assert m.seconds(batch, "gpu") < 0.1 * m.seconds(batch, "cpu")
    # within 35% of the measured points (pool regime for the CPU)
    assert m.seconds(mid, "cpu") == pytest.approx(0.078, rel=0.35)
    assert m.seconds(mid, "gpu") == pytest.approx(0.072, rel=0.35)
    assert m.seconds(big, "cpu") == pytest.approx(1.1, rel=0.35)
    assert m.seconds(big, "gpu") == pytest.approx(0.177, rel=0.35)
    assert m.seconds(batch, "gpu") == pytest.approx(0.374, rel=0.35)


def test_cost_model_is_monotone_in_work():
    m = CostModel()
    for resource in ("cpu", "gpu"):
        prev = 0.0
        for rays in (10, 1_000, 10_000, 100_000, 1_000_000, 4_000_000):
            t = m.seconds(Job(_noop, kind="trace", rays=rays), resource)
            assert t > prev
            prev = t
    with pytest.raises(ValueError):
        m.seconds(Job(_noop), "tpu")


def test_model_round_trips_through_json(tmp_path):
    m = CostModel(cpu_per_step=2e-7, host="Some Chip")
    path = tmp_path / "model.json"
    m.save(path)
    assert CostModel.load(path) == m


def test_default_model_policy(tmp_path, monkeypatch):
    import optiland.parallel as P

    monkeypatch.setenv("OPTILAND_SCHEDULER_CACHE", str(tmp_path))
    monkeypatch.delenv("OPTILAND_SCHEDULER_CALIBRATE", raising=False)
    # never calibrate: shipped coefficients, host stamped
    monkeypatch.setattr(P, "host_signature", lambda: "Other Chip")
    m = default_model(calibrate=False)
    assert m.host == "Other Chip" and m.cpu_per_step == CostModel().cpu_per_step
    # auto on the measured host: shipped model, no calibration
    monkeypatch.setattr(P, "host_signature", lambda: MEASURED_HOST)
    assert default_model() == CostModel()
    # auto on another host with a cache: the cached model
    monkeypatch.setattr(P, "host_signature", lambda: "Other Chip")
    cached = CostModel(cpu_per_step=9e-7, host="Other Chip")
    cached.save(P._cache_path("Other Chip"))
    assert default_model() == cached
    # auto on another host without a cache: calibrates once and saves
    calls = {"n": 0}

    def fake_calibrate(cls, **kw):
        calls["n"] += 1
        return CostModel(cpu_per_step=5e-7, host="Fresh Chip")

    monkeypatch.setattr(P, "host_signature", lambda: "Fresh Chip")
    monkeypatch.setattr(CostModel, "calibrate", classmethod(fake_calibrate))
    m = default_model()
    assert m.cpu_per_step == 5e-7 and calls["n"] == 1
    assert default_model().cpu_per_step == 5e-7 and calls["n"] == 1  # cached now
    monkeypatch.setenv("OPTILAND_SCHEDULER_CALIBRATE", "1")
    default_model()
    assert calls["n"] == 2  # forced


# ---------------------------------------------------------------- planning


def _plan(jobs, cpu=7, gpu=1):
    return plan_jobs(jobs, cpu_workers=cpu, gpu_workers=gpu)


def test_plan_routes_by_shape():
    jobs = (
        [Job(_noop, kind="trace", rays=1027)] * 40
        + [Job(_noop, kind="batch", rays=RAYS_1E5, designs=100)] * 4
        + [Job(_noop, kind="trace", rays=RAYS_1E6)] * 2
    )
    plan = _plan(jobs)
    assert isinstance(plan, Plan) and len(plan.assignment) == len(jobs)
    assert all(plan.assignment[i] == "cpu" for i in range(40))  # small traces
    assert all(plan.assignment[i] == "gpu" for i in range(40, 44))  # batches
    assert plan.uses_gpu
    # every job appears exactly once across the two queues, longest first
    seen = sorted(plan.order["cpu"] + plan.order["gpu"])
    assert seen == list(range(len(jobs)))
    for res in ("cpu", "gpu"):
        secs = [plan.seconds[i] for i in plan.order[res]]
        assert secs == sorted(secs, reverse=True)
    assert plan.makespan["gpu"] > 0 and plan.makespan["cpu"] > 0


def test_plan_without_gpu_puts_everything_on_the_cpu():
    jobs = [Job(_noop, kind="batch", rays=RAYS_1E5, designs=100)] * 3
    plan = _plan(jobs, cpu=4, gpu=0)
    assert plan.assignment == ["cpu"] * 3 and not plan.uses_gpu
    assert plan.makespan["gpu"] == 0.0


def test_plan_honours_forced_kinds():
    jobs = [
        Job(_noop, kind="cpu", rays=RAYS_1E6),
        Job(_noop, kind="gpu", rays=10),
    ]
    plan = _plan(jobs)
    assert plan.assignment == ["cpu", "gpu"]
    with pytest.warns(RuntimeWarning):
        plan = _plan(jobs, gpu=0)
    assert plan.assignment == ["cpu", "cpu"]


def test_plan_uses_the_gpu_only_when_it_shortens_the_run():
    tiny = [Job(_noop, kind="trace", rays=1027)] * 12
    plan = _plan(tiny, cpu=7, gpu=1)
    assert (
        not plan.uses_gpu
    )  # 12 tiny jobs on 7 slots finish before the GPU's fixed cost pays
    many = [Job(_noop, kind="trace", rays=RAYS_1E5)] * 400
    plan = _plan(many, cpu=7, gpu=1)
    gpu_share = plan.assignment.count("gpu") / len(many)
    # the GPU worker is worth about one CPU slot on 1e5-ray traces: it takes
    # a minority share, never the majority, and the two queues end together
    assert 0.05 < gpu_share < 0.4
    assert plan.makespan["gpu"] == pytest.approx(plan.makespan["cpu"], rel=0.15)


def test_plan_balances_a_million_ray_workload():
    jobs = [Job(_noop, kind="trace", rays=RAYS_1E6)] * 24
    plan = _plan(jobs, cpu=7, gpu=1)
    # the GPU is ~6x a CPU slot here: it should take roughly 6 of every 13 jobs
    assert 0.3 < plan.assignment.count("gpu") / 24 < 0.6
    assert plan.makespan["gpu"] == pytest.approx(plan.makespan["cpu"], rel=0.25)


def test_empty_plan_and_run():
    assert run_scheduled([]) == []
    results, plan = run_scheduled([], return_plan=True)
    assert results == [] and plan.assignment == []


# ---------------------------------------------------------------- pool integration


def _trace_job(hy: float, rings: int) -> tuple[float, str, str, int]:
    import optiland.backend as be
    from optiland.samples.objectives import CookeTriplet

    lens = CookeTriplet()
    rays = lens.trace(Hx=0.0, Hy=hy, wavelength=0.55, num_rays=rings)
    fused = 0
    if be.get_backend() == "torch" and be.get_device() == "mps":
        from optiland.backend.torch_backend import metal

        fused = int(metal.stats().get("fused_trace:traces", 0))
    return (
        float(be.to_numpy(rays.y)[-1]),
        be.get_backend(),
        str(be.get_device()) if be.get_backend() == "torch" else "cpu",
        fused,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="spawn semantics tested on POSIX")
def test_run_scheduled_results_in_job_order_and_on_the_planned_resource():
    gpu = 1 if gpu_available() else 0
    rings_small, rings_big = 6, 120  # 127 rays (host-resident), 43,561 rays
    jobs = [
        Job(_trace_job, (0.0, rings_small), kind="trace", rays=trace_rays(rings_small)),
        Job(
            _trace_job,
            (1.0, rings_big),
            kind="gpu" if gpu else "cpu",
            rays=trace_rays(rings_big),
        ),
        Job(_trace_job, (0.5, rings_small), kind="cpu", rays=trace_rays(rings_small)),
        Job(
            _trace_job,
            (0.7, rings_big),
            kind="gpu" if gpu else "cpu",
            rays=trace_rays(rings_big),
        ),
    ]
    results, plan = run_scheduled(
        jobs,
        cpu_workers=2,
        gpu_workers=gpu,
        model=CostModel(),
        gpu_config=WorkerConfig(
            backend="torch",
            device="mps",
            precision="float64",
            metal_mode="df64",
            fused="require",
        ),
        return_plan=True,
    )
    assert len(results) == 4
    # results arrive in job order with the same physics on both resources
    serial = [
        _trace_job(0.0, rings_small),
        _trace_job(1.0, rings_big),
        _trace_job(0.5, rings_small),
        _trace_job(0.7, rings_big),
    ]
    for got, ref in zip(results, serial, strict=True):
        assert math.isclose(got[0], ref[0], rel_tol=1e-9, abs_tol=1e-12)
    for i, (_value, backend, device, fused) in enumerate(results):
        if plan.assignment[i] == "gpu":
            assert backend == "torch" and device.startswith("mps") and fused >= 1
        else:
            assert backend == "numpy"
    if gpu:
        assert plan.assignment == ["cpu", "gpu", "cpu", "gpu"]
