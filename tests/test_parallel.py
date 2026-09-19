"""Tests for optiland.parallel (spawned worker processes with per-worker backends)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from optiland.parallel import WorkerConfig, evaluate_parallel


def _job(optic_dict: dict, hy: float) -> tuple[float, str, str, int]:
    import optiland.backend as be
    from optiland.backend.utils import to_numpy
    from optiland.optic import Optic

    optic = Optic.from_dict(optic_dict)
    rays = optic.trace(Hx=0.0, Hy=hy, wavelength=0.55, num_rays=6, distribution="hexapolar")
    x, y = to_numpy(rays.x), to_numpy(rays.y)
    m = np.isfinite(x)
    rms = float(np.sqrt(np.mean((x[m] - x[m].mean()) ** 2 + (y[m] - y[m].mean()) ** 2)))
    device = be.get_device() if be.get_backend() == "torch" else "cpu"
    return rms, be.get_backend(), device, os.getpid()


@pytest.mark.skipif(sys.platform == "win32", reason="spawn semantics tested on POSIX")
def test_evaluate_parallel_matches_serial_and_uses_configured_backends():
    from optiland.samples.objectives import CookeTriplet

    d = CookeTriplet().to_dict()
    jobs = [(d, hy) for hy in (0.0, 0.5, 1.0)]
    serial = [_job(*j)[0] for j in jobs]
    configs = [WorkerConfig(backend="numpy"), WorkerConfig("cpu", "float64")]
    results = evaluate_parallel(_job, jobs, workers=configs)
    assert [r[0] for r in results] == pytest.approx(serial, rel=1e-12, abs=1e-15)
    assert {(r[1], r[2]) for r in results} <= {("numpy", "cpu"), ("torch", "cpu")}
    assert len({r[3] for r in results}) >= 1
    assert all(r[3] != os.getpid() for r in results)


def test_worker_config_defaults():
    cfg = WorkerConfig()
    assert (cfg.device, cfg.precision, cfg.backend, cfg.metal_mode, cfg.threads) == ("cpu", "float64", "torch", "df64", 1)
