from __future__ import annotations

import json
import os

import pytest

import optiland.backend as be


def pytest_addoption(parser):
    """Register the --update-golden flag used by the regression snapshot suite."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate tests/regression/fixtures/*.json instead of comparing "
        "against them.",
    )


def _backend_params() -> list[str]:
    """Backends to parametrize over; ``torch-mps`` (emulated float64 on the Apple
    GPU) is added when OPTILAND_TEST_MPS=1 and the device is available."""
    params = list(be.list_available_backends())
    if os.environ.get("OPTILAND_TEST_MPS") == "1" and "torch" in params:
        import torch

        if torch.backends.mps.is_available():
            params.append("torch-mps")
    return params


@pytest.fixture(params=_backend_params(), ids=lambda b: f"backend={b}")
def set_test_backend(request):
    """Set the backend for each test and ensure proper device configuration."""
    backend_name = request.param
    if backend_name == "torch-mps":
        be.set_backend("torch")
        be.set_device("mps")
        # Fork-local: OPTILAND_TEST_MPS_GRAD=0 runs the suite with autograd off,
        # which is what makes a bundle eligible for the fused trace (plan 4/WP4).
        if os.environ.get("OPTILAND_TEST_MPS_GRAD", "1") == "1":
            be.grad_mode.enable()
        else:
            be.grad_mode.disable()
        be.set_precision("float64")
        yield
        be.set_backend("numpy")
        return
    be.set_backend(backend_name)

    if backend_name == "torch":
        # Device and precision can be overridden for accelerator runs, e.g.
        # OPTILAND_TEST_TORCH_DEVICE=mps OPTILAND_TEST_TORCH_PRECISION=float32.
        be.set_device(os.environ.get("OPTILAND_TEST_TORCH_DEVICE", "cpu"))
        be.grad_mode.enable()  # Enable gradient tracking
        be.set_precision(os.environ.get("OPTILAND_TEST_TORCH_PRECISION", "float64"))

    yield

    # Reset the backend to numpy after the test
    be.set_backend("numpy")


# ---------------------------------------------------------------------------
# Fork-local: fused-trace warning filter and independent candidate census
# (plan 4/WP4, 8.2, 8.3).  Inert unless the environment asks for it.
# ---------------------------------------------------------------------------

#: Where the per-test census/stats JSON lines go; unset disables the fixture.
_STATS_FILE = os.environ.get("OPTILAND_TEST_MPS_STATS_FILE")

#: Mirrors ``metal.tensor.HOST_THRESHOLD`` without importing the gate, so the
#: census stays an independent predictor of ``fused_trace:candidates``.
_HOST_THRESHOLD = int(os.environ.get("OPTILAND_METAL_HOST_THRESHOLD", "256"))


def pytest_configure(config):
    """Silence the once-per-process fused-trace unavailability warning."""
    if os.environ.get("OPTILAND_TEST_MPS") == "1":
        config.addinivalue_line(
            "filterwarnings",
            "ignore::optiland.backend.torch_backend.metal.trace"
            ".FusedTraceUnavailableWarning",
        )


def _is_fuse_candidate(group, rays, skip) -> bool:
    """True when this ``SurfaceGroup.trace`` call is a fused-trace candidate.

    The structural checks of plan 1.3, evaluated without importing the gate, so
    the census stays an independent predictor of ``fused_trace:candidates``.
    """
    from optiland.rays import RealRays
    from optiland.surfaces.surface_group import SurfaceGroup

    x = getattr(rays, "x", None)
    return (
        type(group) is SurfaceGroup
        and type(rays) is RealRays
        and skip == 0
        and type(x).__name__ == "MetalFloat64"
        and x.numel() > _HOST_THRESHOLD
        and not be.grad_mode.requires_grad
    )


@pytest.fixture(autouse=True)
def _fused_trace_census(request):
    """Append ``{nodeid, census, stats}`` per test to OPTILAND_TEST_MPS_STATS_FILE.

    ``census`` counts the :func:`_is_fuse_candidate` calls, so the two
    identities of plan 1.3 can be checked against the driver's own counters.
    """
    if _STATS_FILE is None:
        yield
        return
    from optiland.backend.torch_backend import metal  # what be.metal_stats() reads
    from optiland.surfaces.surface_group import SurfaceGroup

    census, original, before = [0], SurfaceGroup.trace, metal.stats()

    def _census_trace(self, rays, skip=0, record=True):
        if _is_fuse_candidate(self, rays, skip):
            census[0] += 1
        return original(self, rays, skip=skip, record=record)

    SurfaceGroup.trace = _census_trace
    try:
        yield
    finally:
        SurfaceGroup.trace = original
        now = metal.stats()
        delta = {k: v - before.get(k, 0) for k, v in now.items() if v != before.get(k)}
        line = {"nodeid": request.node.nodeid, "census": census[0], "stats": delta}
        with open(_STATS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
