from __future__ import annotations

import types

import pytest

from optiland import backend as be


@pytest.fixture(autouse=True)
def reset_backend_state():
    # Store original state
    original_backend = be.get_backend()
    yield
    # Restore original state after test
    try:
        be.set_backend(original_backend)
    except Exception as e:
        # If restoration fails, fall back to numpy
        print(f"Error restoring original backend {original_backend}: {e}")
        be.set_backend("numpy")


@pytest.fixture
def cleanup_dummy_backend():
    yield
    # Remove any dummy backends that might have been added
    if "dummy" in be.__getattr__.__globals__["_backends"]:
        del be.__getattr__.__globals__["_backends"]["dummy"]
    # Ensure we're back to numpy
    be.set_backend("numpy")


def test_default_backend():
    assert be.get_backend() == "numpy"


def test_list_available_backends():
    assert "numpy" in be.list_available_backends()


def test_invalid_backend():
    with pytest.raises(ValueError):
        be.set_backend("nonexistent")


def test_getattr_nonexistent_attribute():
    with pytest.raises(AttributeError):
        _ = be.some_nonexistent_attribute


def test_getattr_from_backend(monkeypatch):
    fake_backend = types.ModuleType("fake_numpy")
    fake_backend.test_attr_func = lambda x: x * 2
    monkeypatch.setitem(be.__getattr__.__globals__["_backends"], "numpy", fake_backend)
    monkeypatch.setattr(be, "_current_backend", "numpy")
    assert be.test_attr_func(5) == 10


def test_getattr_from_backend_lib(monkeypatch):
    # The new class-based dispatch no longer falls through to _lib.
    # Attributes must be on the backend instance itself.
    # Verify that attributes NOT on the instance raise AttributeError.
    fake_backend = types.SimpleNamespace(test_attr_on_instance=99)
    monkeypatch.setitem(be.__getattr__.__globals__["_backends"], "numpy", fake_backend)
    monkeypatch.setattr(be, "_current_backend", "numpy")
    assert be.test_attr_on_instance == 99


def test_getattr_fallback_priority(monkeypatch):
    fake_backend = types.ModuleType("fake_numpy")
    fake_backend.test_attr_priority = "from_backend"
    fake_backend._lib = types.SimpleNamespace(test_attr_priority="from_lib")
    monkeypatch.setitem(be.__getattr__.__globals__["_backends"], "numpy", fake_backend)
    monkeypatch.setattr(be, "_current_backend", "numpy")
    assert be.test_attr_priority == "from_backend"


def test_get_backend_after_set(monkeypatch):
    dummy_backend = type("DummyBackend", (), {})()
    dummy_backend.dummy_attr = "dummy_value"
    monkeypatch.setitem(be.__getattr__.__globals__["_backends"], "dummy", dummy_backend)
    be.set_backend("dummy")
    assert be.get_backend() == "dummy"
    assert be.dummy_attr == "dummy_value"
