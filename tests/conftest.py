from __future__ import annotations

import pytest

import optiland.backend as be


def _apply_backend(backend_name: str) -> None:
    """Configure the active backend for a test (or test class)."""
    be.set_backend(backend_name)

    if backend_name == "torch":
        be.set_device("cpu")  # Use CPU for tests
        be.grad_mode.enable()  # Enable gradient tracking
        be.set_precision("float64")  # Set precision to float64 for tests


@pytest.fixture(params=be.list_available_backends(), ids=lambda b: f"backend={b}")
def set_test_backend(request):
    """Fixture to set the backend for each test and ensure proper device configuration."""
    _apply_backend(request.param)

    yield

    # Reset the backend to numpy after the test
    be.set_backend("numpy")


@pytest.fixture(
    scope="class",
    params=be.list_available_backends(),
    ids=lambda b: f"backend={b}",
)
def set_test_backend_class(request):
    """Class-scoped variant of :func:`set_test_backend`.

    Lets a test class share an expensive fixture (e.g. an analysis object
    that takes several seconds to construct) across all of its tests for a
    given backend. The backend is configured once per class+backend instead
    of per individual test, which is essential when the heavy computation
    happens during ``__init__``.

    Pair with class-scoped fixtures that build the expensive object so the
    work is amortised over every test in the class:

    .. code-block:: python

        @pytest.fixture(scope="class")
        def cached_analysis(set_test_backend_class, optic):
            return ExpensiveAnalysis(optic)

        class TestThing:
            def test_a(self, cached_analysis): ...
            def test_b(self, cached_analysis): ...
    """
    _apply_backend(request.param)

    yield

    be.set_backend("numpy")
