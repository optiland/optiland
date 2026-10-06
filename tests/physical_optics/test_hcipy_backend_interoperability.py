"""Optional explicit export from HCIPy's backend-aware field implementation."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from optiland.physical_optics.interoperability import from_hcipy, to_hcipy
from tests.utils import assert_allclose

hp = pytest.importorskip("hcipy")
NewStyleField = getattr(hp.field, "NewStyleField", None)
if NewStyleField is None:
    pytest.skip("installed HCIPy has no backend-aware field", allow_module_level=True)
if "xp" not in inspect.signature(hp.make_uniform_grid).parameters:
    pytest.skip(
        "installed HCIPy has no namespace-aware grid factory", allow_module_level=True
    )


@pytest.mark.parametrize("namespace", ["numpy", "array_api_strict"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_explicit_backend_field_export(set_test_backend, namespace, dtype):
    xp = pytest.importorskip(namespace)
    grid = hp.make_uniform_grid(
        [6, 4], [120e-6, 120e-6], center=[30e-6, -20e-6], has_center=True, xp=xp
    )
    values = (np.arange(24) + 1j * (np.arange(24) + 2)).astype(dtype)
    electric = NewStyleField(xp.asarray(values, dtype=getattr(xp, dtype)), grid)
    try:
        source = hp.Wavefront(electric, wavelength=500e-9)
    except (NotImplementedError, AttributeError):
        pytest.skip("this HCIPy version does not yet support backend Wavefront values")
    except TypeError as error:
        if namespace == "array_api_strict" and "dtype" in str(error):
            pytest.skip(
                "this HCIPy version passes string dtypes to the strict namespace"
            )
        raise
    imported = from_hcipy(source)
    assert_allclose(imported.data, values.reshape(4, 6) / 1000, rtol=1e-7, atol=1e-12)
    # Reference uses explicit host quadrature and source single-precision values.
    expected_power = np.sum(np.abs(values.astype(complex)) ** 2) * 20e-6 * 30e-6
    assert_allclose(imported.power, expected_power, rtol=2e-7, atol=1e-15)
    restored = to_hcipy(imported)
    assert_allclose(restored.electric_field, values, rtol=1e-7, atol=1e-6)
    assert_allclose(imported.center, (20e-3, -35e-3), rtol=1e-12, atol=1e-15)
