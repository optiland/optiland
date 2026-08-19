from __future__ import annotations

import pytest

import optiland.backend as be

# Attempt importing torch and skip all tests if not installed.
# Note that some users may not have torch installed, so this avoids unnecessary errors.
torch = pytest.importorskip("torch", reason="torch is not installed")

# Access the TorchBackend singleton through the registry so tests work with
# the class-based design.
_torch_instance = be._backends.get("torch")
if _torch_instance is None:
    pytest.skip("torch backend not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def restore_backend_config():
    """Automatically restore global state after each test."""
    original_backend = be.get_backend()
    original_device = _torch_instance.get_device()
    original_precision = _torch_instance.get_precision()  # int: 32 or 64
    original_grad = _torch_instance.grad_mode.requires_grad

    yield  # run the test

    # Restore the original state
    be.set_backend(original_backend)
    _torch_instance.set_device(original_device)
    precision_str = "float32" if original_precision == 32 else "float64"
    _torch_instance.set_precision(precision_str)
    if original_grad:
        _torch_instance.grad_mode.enable()
    else:
        _torch_instance.grad_mode.disable()


def test_set_and_get_device_cpu():
    _torch_instance.set_device("cpu")
    assert _torch_instance.get_device() == "cpu"


def test_set_and_get_device_cuda():
    if torch.cuda.is_available():
        _torch_instance.set_device("cuda")
        assert _torch_instance.get_device() == "cuda"
    else:
        with pytest.raises(ValueError):
            _torch_instance.set_device("cuda")


def test_set_device_invalid():
    with pytest.raises(ValueError):
        _torch_instance.set_device("tpu")


def test_set_and_get_precision_float32():
    _torch_instance.set_precision("float32")
    assert _torch_instance.get_precision() == 32


def test_set_and_get_precision_float64():
    _torch_instance.set_precision("float64")
    assert _torch_instance.get_precision() == 64


def test_set_precision_invalid():
    with pytest.raises(ValueError):
        _torch_instance.set_precision("float16")


def test_grad_mode_enable_disable():
    _torch_instance.grad_mode.disable()
    assert _torch_instance.grad_mode.requires_grad is False

    _torch_instance.grad_mode.enable()
    assert _torch_instance.grad_mode.requires_grad is True

    _torch_instance.grad_mode.disable()
    assert _torch_instance.grad_mode.requires_grad is False


def test_grad_mode_temporary_enable():
    _torch_instance.grad_mode.disable()
    initial_state = _torch_instance.grad_mode.requires_grad
    assert initial_state is False

    with _torch_instance.grad_mode.temporary_enable():
        assert _torch_instance.grad_mode.requires_grad is True

    assert _torch_instance.grad_mode.requires_grad is initial_state
