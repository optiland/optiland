"""Owned file jobs retain current Zemax material and aperture semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from optiland.physical_apertures import RadialAperture
from tests.test_fileio.test_zemax_semi_diameter import _passes, _singlet

from .test_calculation_jobs import wait_for
from .test_file_operations import file_operations as file_operations


def assert_fixed_catalog_singlet(optic):
    front = optic.surfaces[2]
    assert front.material_post.reference == "schott"
    assert front.material_post.material_data["filename"] == "glass/schott/N-BK7.yml"
    assert isinstance(front.aperture, RadialAperture)
    assert float(front.aperture.r_max) == 5.0
    assert _passes(optic, py=0.499)
    assert not _passes(optic, py=0.501)


def test_worker_roundtrip_preserves_declared_glass_and_fixed_aperture(
    qapp, set_test_backend, file_operations, tmp_path
):
    connector, operations, _, _, notifications = file_operations
    source = _singlet(tmp_path, 'DIAM 5 1 0 0 1 ""')
    operations.request_load(source)
    wait_for(qapp, lambda: not operations.busy)
    assert connector.is_modified() and connector.get_current_filepath() is None
    assert_fixed_catalog_singlet(connector.get_optic())

    native = tmp_path / "native.json"
    operations.request_output(native)
    wait_for(qapp, lambda: not operations.busy)
    assert native.exists(), notifications
    assert not connector.is_modified()
    assert connector.get_current_filepath() == str(native)
    operations.request_load(native)
    wait_for(qapp, lambda: not operations.busy)
    assert_fixed_catalog_singlet(connector.get_optic())

    exported = tmp_path / "roundtrip.zmx"
    operations.request_output(exported, "zemax")
    wait_for(qapp, lambda: not operations.busy)
    assert exported.exists(), notifications
    assert connector.get_current_filepath() == str(native)
    assert not connector.is_modified()
    operations.request_load(exported)
    wait_for(qapp, lambda: not operations.busy)
    assert_fixed_catalog_singlet(connector.get_optic())
    assert connector.is_modified() and connector.get_current_filepath() is None


@pytest.mark.parametrize("entrypoint", ["synchronous", "worker"])
def test_unresolvable_foreign_glass_preserves_current_document_and_history(
    qapp, file_operations, tmp_path, entrypoint
):
    connector, operations, _, _, notifications = file_operations
    original = connector.get_optic()
    connector.set_surface_data(1, connector.COL_COMMENT, "preserve this document")
    connector._file_service._current_filepath = "existing.json"
    token = connector.document_state.edit_token
    source = Path(_singlet(tmp_path, 'DIAM 5 1 0 0 1 ""'))
    source.write_text(
        source.read_text().replace(
            "GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0", "GLAS MISSING_GLASS 0 0"
        )
    )
    if entrypoint == "synchronous":
        connector.import_zemax(str(source))
    else:
        operations.request_load(source)
        wait_for(qapp, lambda: not operations.busy)
    assert connector.get_optic() is original
    assert connector.document_state.edit_token == token
    assert connector.get_current_filepath() == "existing.json"
    assert connector.is_modified()
    assert connector._undo_redo_manager.can_undo()
    assert any(
        "GCAT" in message and severity == "error" for message, severity in notifications
    )
