"""No-op polarization changes respect the active backend's scalar precision."""

from __future__ import annotations

import math

import pytest

from optiland.rays import PolarizationState
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.services.job_records import BackendConfig


@pytest.mark.parametrize(
    "backend,precision", [("numpy", 64), ("torch", 32), ("torch", 64)]
)
def test_scaled_full_turn_polarization_is_noop_but_real_change_is_edit(
    qapp, backend, precision
):
    if backend == "torch":
        pytest.importorskip("torch")
    original = BackendConfig.capture()
    try:
        BackendConfig(backend, "cpu", precision).apply()
        connector = OptilandConnector()
        optic = connector.get_optic()
        # Existing public states may contain unreduced phases from external code.
        optic.polarization = PolarizationState(True, 1, 0, math.tau, 2.5 * math.pi)
        before = connector.document_state.token, connector.document_state.edit_token
        connector.set_polarization_state("polarized", 2, 0, 0, 90)
        assert (
            connector.document_state.token,
            connector.document_state.edit_token,
        ) == before
        assert not connector._undo_redo_manager._undo_stack
        assert not connector.is_modified()
        connector.set_polarization_state("polarized", 2, 0, 360, 450)
        assert connector.document_state.token == before[0]
        connector.set_polarization_state("polarized", 1, 1, 0, 90)
        assert connector.document_state.token.revision == before[0].revision + 1
        assert len(connector._undo_redo_manager._undo_stack) == 1
        connector.set_polarization_state("polarized", 1, 1, 0, 90.01)
        assert connector.document_state.token.revision == before[0].revision + 2
    finally:
        original.apply()


def test_explicit_setting_can_repair_an_unconfigured_incident_state(qapp):
    connector = OptilandConnector()
    connector.get_optic().polarization = None
    connector.set_polarization_state("unpolarized")
    assert not connector.get_optic().polarization.is_polarized


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_phase_does_not_create_edit(qapp, value):
    connector = OptilandConnector()
    before = connector.document_state.edit_token
    with pytest.raises(ValueError, match="finite"):
        connector.set_polarization_state("polarized", 1, 1, value, 90)
    assert connector.document_state.edit_token == before
    assert connector.get_optic().polarization == "ignore"
