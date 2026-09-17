"""Whole-document candidates cannot overwrite newer metadata-only edits."""

from __future__ import annotations

import pytest

from optiland_gui.services.calculation_jobs import CalculationJobs, DocumentState
from optiland_gui.services.job_records import JobRequest


def test_metadata_preserves_preview_but_revokes_whole_candidate_before_commit(qapp):
    state = DocumentState()
    jobs = CalculationJobs(state)
    request = JobRequest(1, state.token, "optimization", 0, "unused", None, {})
    candidate_edit_token = state.edit_token
    commits = []
    state.committed.connect(commits.append)
    with state.transaction():
        state.record("metadata")
        assert jobs.is_current(request)
        assert state.edit_token != candidate_edit_token
        assert not commits
        state.record("metadata")
        assert state.edit_token.revision == candidate_edit_token.revision + 1
    assert commits[0].edit_token == state.edit_token


def test_presentation_is_not_a_persisted_edit(qapp):
    state = DocumentState()
    before = state.token, state.edit_token
    state.record("presentation")
    assert (state.token, state.edit_token) == before


def test_mixed_nested_transaction_has_one_edit_revision(qapp):
    state = DocumentState()
    initial = state.edit_token
    observed = []
    state.changed.connect(lambda _: observed.append(state.edit_token))
    with state.transaction():
        state.record("metadata")
        with state.transaction():
            state.change()
        state.record("metadata")
    assert state.edit_token.revision == initial.revision + 1
    assert observed == [state.edit_token]


def test_replacement_uses_one_shared_epoch_and_revokes_both_guards(qapp):
    state = DocumentState()
    previous = state.edit_token
    with state.transaction(replacement=True):
        state.record("metadata")
        state.replace()
        state.change()
    assert state.token == state.edit_token
    assert state.token.document_id != previous.document_id
    assert state.token.revision == 0


def test_failed_transaction_does_not_restore_candidate_permission(qapp):
    state = DocumentState()
    previous = state.edit_token
    with pytest.raises(ValueError), state.transaction():
        state.record("metadata")
        raise ValueError("Mutation must be rolled back by its owner")
    assert state.edit_token != previous
    failed = state.edit_token
    state.record("metadata")
    assert state.edit_token.revision == failed.revision + 1
