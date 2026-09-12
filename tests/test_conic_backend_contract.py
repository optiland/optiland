"""Backend ownership, dispatch, and optional-dependency conic contracts."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import optiland.backend as be
from optiland.geometries.standard import _conic_intersection_distance


@pytest.fixture(autouse=True)
def reset_backend():
    original = be.get_backend()
    yield
    be.set_backend(original)


def test_geometry_delegates_arrays_and_membership_to_backend(monkeypatch):
    be.set_backend("numpy")
    values = tuple(np.array([v]) for v in (0.0, 0.1, -1.0, 0.0, 0.0, 1.0))
    rays = SimpleNamespace(
        **dict(zip(("x", "y", "z", "L", "M", "N"), values, strict=True))
    )
    radius, conic = np.array(2.0), np.array(0.0)
    contains = lambda x, y: x * x + y * y < 1
    aperture = SimpleNamespace(contains=contains)
    expected = np.array([1.002501564456182])

    def kernel(*inputs, **kwargs):
        assert all(
            got is want
            for got, want in zip(inputs, (*values, radius, conic), strict=True)
        )
        assert kwargs == {"contains": contains}
        return expected

    monkeypatch.setattr(be._backends["numpy"], "conic_intersection", kernel)
    assert _conic_intersection_distance(rays, radius, conic, aperture) is expected


@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("with_membership", [False, True])
def test_backend_instance_owns_execution_independent_of_global_selection(
    backend_name, dtype, with_membership
):
    # Native float32 and compiled float64 must both use the selected instance,
    # including when module-level be currently refers to the opposite backend.
    torch = pytest.importorskip("torch")
    be.set_backend("torch" if backend_name == "numpy" else "numpy")
    values = (0.1, 0.2, -1.0, 0.0, 0.0, 1.0, 2.0, 0.0)
    if backend_name == "numpy":
        inputs = tuple(np.array([v], dtype=dtype) for v in values[:6]) + tuple(
            np.array(v, dtype=dtype) for v in values[6:]
        )
    else:
        inputs = tuple(
            torch.tensor([v], dtype=getattr(torch, dtype)) for v in values[:6]
        ) + tuple(
            torch.tensor(v, dtype=getattr(torch, dtype)) for v in values[6:]
        )
    contains = (lambda x, y: x * x + y * y < 1) if with_membership else None
    actual = be._backends[backend_name].conic_intersection(*inputs, contains=contains)
    expected = 3 - np.sqrt(4 - 0.1**2 - 0.2**2)
    if backend_name == "torch":
        assert isinstance(actual, torch.Tensor)
        assert actual.device.type == "cpu"
        actual = actual.numpy()
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        actual, expected, rtol=2e-7 if dtype == "float32" else 2e-14
    )


def test_geometry_and_backend_import_direction():
    root = Path(__file__).resolve().parents[1] / "optiland"
    geometry = ast.parse((root / "geometries/standard.py").read_text())
    imports = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(geometry)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert not any(
        name.split(".")[0] in ("numpy", "torch", "numba") for name in imports
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "get_backend"
        for node in ast.walk(geometry)
    )
    for relative in ("_conic.py", "numpy_backend/conic.py", "torch_backend/conic.py"):
        tree = ast.parse((root / "backend" / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(
                    ("optiland.geometries", "optiland.rays", "optiland.physical_apertures")
                )


def test_numpy_geometry_import_and_execution_without_torch():
    # A fresh process is essential: pytest's global fixtures already import
    # Torch when installed. Block imports as if the optional package is absent.
    script = '''
import importlib.abc
import sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("Torch deliberately unavailable", name=fullname)

sys.meta_path.insert(0, NoTorch())
import numpy as np
from types import SimpleNamespace
import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.standard import StandardGeometry
from optiland.optic import Optic

assert be.list_available_backends() == ["numpy"]
for dtype in (np.float32, np.float64):
    rays = SimpleNamespace(**{
        name: np.array([value], dtype=dtype)
        for name, value in zip(("x", "y", "z", "L", "M", "N"), (0, .2, -1, 0, 0, 1))
    })
    geometry = StandardGeometry(CoordinateSystem(), 2)
    geometry.radius, geometry.k = np.array(2, dtype=dtype), np.array(0, dtype=dtype)
    result = geometry.distance(rays)
    np.testing.assert_allclose(result, 3 - np.sqrt(4 - .2**2), rtol=2e-7)
    assert result.dtype == dtype
optic = Optic()
assert "torch" not in sys.modules
assert not any(name.startswith("optiland.backend.torch_backend") for name in sys.modules)
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
