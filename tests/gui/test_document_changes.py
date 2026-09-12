"""Immediate optical invalidation and coherent committed change notifications."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.services.calculation_jobs import CalculationJobs, DocumentState
from optiland_gui.services.job_records import JobRequest


def test_nested_transaction_invalidates_once_before_commit(qapp):
    state = DocumentState()
    invalidations, commits = [], []
    state.changed.connect(invalidations.append)
    state.committed.connect(commits.append)
    initial = state.token
    with state.transaction():
        state.record("optical", surface_indices=(2,), columns=(3,))
        assert state.token.revision == initial.revision + 1
        assert len(invalidations) == 1 and not commits
        with state.transaction():
            state.record("structure", surface_indices=(4,))
        state.record("metadata", surface_indices=(2,), columns=(1,))
    assert len(invalidations) == 1 and len(commits) == 1
    assert commits[0].categories == {"optical", "structure", "metadata"}
    assert commits[0].surface_indices == {2, 4}
    assert not commits[0].columns  # The structural change affects all columns.


def test_global_change_is_not_narrowed_by_another_local_edit(qapp):
    state = DocumentState()
    commits = []
    state.committed.connect(commits.append)
    with state.transaction():
        state.record("optical")
        state.record("metadata", surface_indices=(1,), columns=(1,))
    assert not commits[0].surface_indices and not commits[0].columns
    state.record("metadata", surface_indices=(2,), columns=(1,))
    assert commits[1].surface_indices == {2} and commits[1].columns == {1}


def test_replacement_and_changed_pair_is_one_epoch(qapp):
    state = DocumentState()
    events, commits = [], []
    state.changed.connect(events.append)
    state.committed.connect(commits.append)
    previous = state.token
    with state.transaction(replacement=True):
        state.replace()
        state.change()
    assert len(events) == len(commits) == 1
    assert state.token.document_id != previous.document_id
    assert state.token.revision == 0


def test_metadata_does_not_revoke_calculations(qapp):
    state = DocumentState()
    jobs = CalculationJobs(state)
    request = JobRequest(1, state.token, "layout", 0, "unused", None, {})
    commits = []
    state.committed.connect(commits.append)
    state.record("metadata", surface_indices=(1,), columns=(1,))
    assert jobs.is_current(request)
    assert not commits[0].affects_optics
    with state.transaction():
        state.change()
        assert not jobs.is_current(request)  # Before any timer/event-loop turn.


def test_failed_nested_transaction_never_publishes_success(qapp):
    state = DocumentState()
    commits = []
    state.committed.connect(commits.append)
    with state.transaction():
        state.change()
        with pytest.raises(ValueError), state.transaction():
            state.record("metadata")
            raise ValueError("Prepared mutation rejected")
    assert not commits
    state.record("metadata")
    assert len(commits) == 1 and commits[0].categories == {"metadata"}


def test_canonical_connector_notification_preserves_public_signal_once(qapp):
    connector = OptilandConnector()
    invalidations, commits, legacy = [], [], []
    connector.document_state.changed.connect(invalidations.append)
    connector.document_state.committed.connect(commits.append)
    connector.opticChanged.connect(lambda: legacy.append(True))
    connector.notify_change("metadata", surface_indices=(0,), columns=(1,))
    assert not invalidations and len(commits) == len(legacy) == 1
    connector.notify_change("optical")
    assert len(invalidations) == 1 and len(commits) == len(legacy) == 2
    connector.opticChanged.emit()  # Supported external signal remains conservative.
    assert len(invalidations) == 2


def test_external_replacement_is_detected_even_with_only_changed_signal(qapp):
    from optiland.optic import Optic

    connector = OptilandConnector()
    original = connector.document_state.token
    commits = []
    connector.document_state.committed.connect(commits.append)
    connector._optic = Optic()
    connector.opticChanged.emit()
    assert connector.document_state.token.document_id != original.document_id
    assert commits[0].structural


def test_comment_edit_skips_optical_update_and_noop(qapp, monkeypatch):
    connector = OptilandConnector()
    optic = connector.get_optic()
    original_token = connector.document_state.token
    update = MagicMock(side_effect=AssertionError("A comment recalculated optics"))
    monkeypatch.setattr(optic.updater, "update", update)
    connector.set_surface_data(1, connector.COL_COMMENT, "Renamed surface")
    assert optic.surfaces[1].comment == "Renamed surface"
    assert connector.document_state.token == original_token
    assert connector.is_modified()
    count = len(connector._undo_redo_manager._undo_stack)
    connector.set_surface_data(1, connector.COL_COMMENT, "Renamed surface")
    assert len(connector._undo_redo_manager._undo_stack) == count
    update.assert_not_called()


def test_polarization_noop_compares_normalized_values(qapp, monkeypatch):
    connector = OptilandConnector()
    service = connector._system_service
    service.set_polarization_state("unpolarized")
    initial = connector.document_state.token
    service.set_polarization_state("unpolarized")
    assert connector.document_state.token == initial
    service.set_polarization_state("polarized", 1, 0, 0, 90)
    initial = connector.document_state.token
    service.set_polarization_state("polarized", 2, 0, 360, 450)
    assert connector.document_state.token == initial
    service.set_polarization_state("polarized", 1, 1, 0, 90)
    assert connector.document_state.token.revision == initial.revision + 1
    with pytest.raises(ValueError, match="nonzero"):
        service.set_polarization_state("polarized", 0, 0, 0, 0)


@pytest.mark.parametrize(
    "category, signal_name",
    [("metadata", "opticChanged"), ("replacement", "opticLoaded")],
)
def test_external_change_during_public_notification_still_revokes_results(
    qapp, category, signal_name
):
    connector = OptilandConnector()
    initial = connector.document_state.token
    request = JobRequest(1, initial, "layout", 0, "unused", None, {})
    adjusted = []

    def adjust_once():
        if not adjusted:
            adjusted.append(connector.document_state.token)
            connector.get_optic().surfaces[1].geometry.radius = 75.0
            connector.opticChanged.emit()

    getattr(connector, signal_name).connect(adjust_once)
    connector.notify_change(category, surface_indices=(1,), columns=(1,))
    assert adjusted
    assert connector.document_state.token.revision == adjusted[0].revision + 1
    assert not connector.calculation_jobs.is_current(request)


def test_invalid_transaction_declarations_leave_no_success_notifications(qapp):
    state = DocumentState()
    commits = []
    state.committed.connect(commits.append)
    initial = state.token, state.edit_token
    with pytest.raises(ValueError, match="Unknown"):
        state.record("unsupported")
    assert (state.token, state.edit_token) == initial
    with (
        pytest.raises(ValueError, match="outer"),
        state.transaction(),
        state.transaction(replacement=True),
    ):
        pass
    with pytest.raises(ValueError, match="Declare"), state.transaction():
        state.record("metadata")
        state.replace()
    assert not commits
    state.change()
    assert len(commits) == 1
