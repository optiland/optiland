"""Public JSON files retain incident polarization rather than display strings."""

from __future__ import annotations

import numpy as np
import pytest

from optiland.fileio.optiland_handler import load_optiland_file, save_optiland_file
from optiland.rays import PolarizationState, create_polarization
from tests.test_optic import singlet_infinite_object


@pytest.mark.parametrize("state", [None, "ignore", "unpolarized", "RCP"])
def test_native_json_retains_incident_polarization(tmp_path, set_test_backend, state):
    optic = singlet_infinite_object()
    optic.polarization = (
        create_polarization(state) if state in ("unpolarized", "RCP") else state
    )
    path = tmp_path / "polarized.json"
    save_optiland_file(optic, path)
    restored = load_optiland_file(path)
    if isinstance(optic.polarization, PolarizationState):
        assert isinstance(restored.polarization, PolarizationState)
        assert restored.polarization.is_polarized == optic.polarization.is_polarized
        for name in ("Ex", "Ey", "phase_x", "phase_y"):
            expected, actual = (
                getattr(optic.polarization, name),
                getattr(restored.polarization, name),
            )
            if expected is None:
                assert actual is None
            else:
                import optiland.backend as be

                np.testing.assert_allclose(be.to_numpy(actual), be.to_numpy(expected))
    else:
        assert restored.polarization == state
