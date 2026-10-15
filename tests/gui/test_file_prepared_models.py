"""File acceptance installs worker models without GUI material reconstruction."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.rays import create_polarization
from optiland_gui.services.file_service import SpecialFloatEncoder
from optiland_gui.services.job_records import BackendConfig, JobResult
from optiland_gui.services.prepared_optic import PreparedOptic

from .test_calculation_jobs import wait_for
from .test_file_operations import file_operations as file_operations


def test_worker_load_never_reconstructs_materials_or_prescription_in_gui(
    qapp, set_test_backend, file_operations, tmp_path, monkeypatch
):
    from optiland.materials.material import Material

    connector, operations, _, _, notifications = file_operations
    original = connector.get_optic()
    original.polarization = create_polarization("RCP")
    path = tmp_path / "polarized.json"
    path.write_text(json.dumps(original.to_dict(), cls=SpecialFloatEncoder))

    def reject(*args, **kwargs):
        pytest.fail("File acceptance repeated model/material construction in the GUI")

    monkeypatch.setattr(Material, "_retrieve_file", reject)
    monkeypatch.setattr(Optic, "from_dict", reject)
    operations.request_load(path)
    wait_for(qapp, lambda: not operations.busy)
    restored = connector.get_optic()
    assert restored is not original, notifications
    assert float(restored.surfaces[1].geometry.radius) == 50
    assert restored.polarization.to_dict() == original.polarization.to_dict()
    assert isinstance(restored.polarization.phase_y, type(be.cast(0.0)))
    assert connector.get_current_filepath() == str(path)
    assert not connector.is_modified()


@pytest.mark.parametrize("invalid", ["backend", "type"])
def test_incompatible_prepared_result_cannot_replace_the_live_document(
    file_operations, tmp_path, monkeypatch, invalid
):
    connector, operations, _, _, notifications = file_operations
    original = connector.get_optic()
    token = connector.document_state.edit_token
    monkeypatch.setattr(operations.jobs, "_dispatch", lambda: None)
    request = operations.request_load(tmp_path / "candidate.json")
    candidate = PreparedOptic.capture(original)
    candidate = (
        replace(candidate, backend=BackendConfig("torch", precision=32))
        if invalid == "backend"
        else {"unexpected": "payload"}
    )
    operations._finished(JobResult(request, "succeeded", candidate, current=True))
    assert not operations.busy
    assert connector.get_optic() is original
    assert connector.document_state.edit_token == token
    assert connector.get_current_filepath() is None
    assert connector.is_modified()
    assert any("Unable to install" in text for text, _ in notifications)
