from __future__ import annotations

import pytest

import optiland.backend as be


def _apply_backend(backend_name: str) -> None:
    """Configure the active backend for a test (or test class)."""
    be.set_backend(backend_name)

    if backend_name == "torch":
        be.set_device("cpu")
        be.set_precision("float64")
        be.grad_mode.disable()


@pytest.fixture(params=be.list_available_backends(), ids=lambda b: f"backend={b}")
def set_test_backend(request):
    """Fixture to set the backend for each test and ensure proper device configuration."""
    _apply_backend(request.param)

    yield

    be.set_backend("numpy")


@pytest.fixture(
    scope="class",
    params=be.list_available_backends(),
    ids=lambda b: f"backend={b}",
)
def set_test_backend_class(request):
    """Class-scoped variant of :func:`set_test_backend`."""
    _apply_backend(request.param)

    yield

    be.set_backend("numpy")
