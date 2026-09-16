from __future__ import annotations

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
    """Fixture to set the backend for each test and ensure proper device configuration."""
    backend_name = request.param
    if backend_name == "torch-mps":
        be.set_backend("torch")
        be.set_device("mps")
        be.grad_mode.enable()
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
