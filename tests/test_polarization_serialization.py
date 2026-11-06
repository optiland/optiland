"""JSON prescriptions retain usable, backend-independent incident polarization."""

from __future__ import annotations

import json

import numpy as np
import pytest

import optiland.backend as be
from optiland.fileio.optiland_handler import load_optiland_file, save_optiland_file
from optiland.rays import PolarizationState, create_polarization
from tests.test_optic import singlet_infinite_object
from tests.utils import assert_allclose

POLARIZATION_TYPES = ("unpolarized", "H", "V", "L+45", "L-45", "RCP", "LCP")
STATE_VALUES = ("Ex", "Ey", "phase_x", "phase_y")


@pytest.mark.parametrize("kind", POLARIZATION_TYPES)
def test_state_dictionary_is_json_safe(set_test_backend, kind):
    state = create_polarization(kind)
    data = state.to_dict()
    assert set(data) == {"is_polarized", *STATE_VALUES}
    assert type(data["is_polarized"]) is bool
    for name in STATE_VALUES:
        assert data[name] is None or type(data[name]) is float

    restored = PolarizationState.from_dict(json.loads(json.dumps(data)))
    assert restored is not state
    assert restored.is_polarized == state.is_polarized
    for name in STATE_VALUES:
        expected, actual = getattr(state, name), getattr(restored, name)
        if expected is None:
            assert actual is None
        else:
            assert_allclose(actual, expected)


@pytest.mark.parametrize("kind", (None, "ignore", *POLARIZATION_TYPES))
def test_native_json_retains_incident_polarization(tmp_path, set_test_backend, kind):
    optic = singlet_infinite_object()
    optic.polarization = (
        create_polarization(kind) if kind in POLARIZATION_TYPES else kind
    )
    if isinstance(optic.polarization, PolarizationState):
        optic.surfaces.set_fresnel_coatings()
    if kind is not None:
        rays = optic.trace(
            Hx=0, Hy=0, num_rays=3, wavelength=0.5, distribution="line_y"
        )
        expected_intensity = be.to_numpy(rays.i).copy()

    path = tmp_path / "polarized.json"
    save_optiland_file(optic, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    restored = load_optiland_file(path)
    if isinstance(optic.polarization, PolarizationState):
        assert data["wavelengths"]["polarization"] == optic.polarization.to_dict()
        assert isinstance(restored.polarization, PolarizationState)
        assert restored.polarization.to_dict() == pytest.approx(
            optic.polarization.to_dict()
        )
    else:
        assert data["wavelengths"]["polarization"] == kind
        assert restored.polarization == kind

    if kind is not None:
        rays = restored.trace(
            Hx=0, Hy=0, num_rays=3, wavelength=0.5, distribution="line_y"
        )
        np.testing.assert_allclose(be.to_numpy(rays.i), expected_intensity)


@pytest.mark.parametrize("source_backend", be.list_available_backends())
def test_state_can_restore_on_another_backend(set_test_backend, source_backend):
    target_backend = be.get_backend()
    try:
        be.set_backend(source_backend)
        state = PolarizationState(True, 3.0, -4.0, 0.7, -0.2)
        data = json.loads(json.dumps(state.to_dict()))
        be.set_backend(target_backend)
        restored = PolarizationState.from_dict(data)
        assert isinstance(restored.phase_x, type(be.cast(0.7)))
        assert_allclose(restored.Ex, 0.6)
        assert_allclose(restored.Ey, -0.8)
        assert_allclose(restored.phase_x, 0.7)
        assert_allclose(restored.phase_y, -0.2)
    finally:
        be.set_backend(target_backend)


def test_tensor_serialization_does_not_mutate_gradient_state():
    torch = pytest.importorskip("torch")
    previous_backend = be.get_backend()
    try:
        be.set_backend("torch")
        amplitude = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        state = PolarizationState(True, amplitude, 4.0, 0.0, 0.5)
        data = state.to_dict()
        assert data["Ex"] == pytest.approx(0.6)
        assert state.Ex.requires_grad
        state.Ex.backward()
        assert amplitude.grad.item() == pytest.approx(16.0 / 125.0)
    finally:
        be.set_backend(previous_backend)


def test_minimal_unpolarized_state(set_test_backend):
    state = PolarizationState.from_dict({"is_polarized": False})
    assert state.to_dict() == {
        "is_polarized": False,
        "Ex": None,
        "Ey": None,
        "phase_x": None,
        "phase_y": None,
    }


@pytest.mark.parametrize(
    "data, message",
    [
        ({"is_polarized": True, "Ex": 1.0}, "All parameters must be provided"),
        ({"is_polarized": False, "phase_x": 0.0}, "must be None"),
    ],
)
def test_dictionary_restoration_validates_state(set_test_backend, data, message):
    with pytest.raises(ValueError, match=message):
        PolarizationState.from_dict(data)
