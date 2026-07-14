from __future__ import annotations

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


@pytest.fixture(params=be.list_available_backends(), ids=lambda b: f"backend={b}")
def set_test_backend(request):
    """Fixture to set the backend for each test and ensure proper device configuration."""
    backend_name = request.param
    be.set_backend(backend_name)

    if backend_name == "torch":
        be.set_device("cpu")  # Use CPU for tests
        be.grad_mode.enable()  # Enable gradient tracking
        be.set_precision("float64")  # Set precision to float64 for tests

    yield

    # Reset the backend to numpy after the test
    be.set_backend("numpy")
