"""File preparation failures preserve the destination and owned-model boundary."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from optiland_gui.services import output_tasks as file_tasks
from optiland_gui.services.file_service import SpecialFloatEncoder
from optiland_gui.services.job_records import BackendConfig, OpticSnapshot


def parameters(tmp_path, file_format="optiland"):
    return {
        "path": str(tmp_path / "saved.json"),
        "staged_path": str(tmp_path / ".optiland-owned.tmp"),
        "format": file_format,
        "backend": BackendConfig.capture(),
        "load_options": {},
    }


def test_preparation_preserves_original_error_if_cleanup_also_fails(
    minimal_optic, tmp_path, monkeypatch
):
    def broken_encoder(*args):
        yield "partial contents"
        raise OSError("simulated full disk")

    def blocked_cleanup(*args, **kwargs):
        raise PermissionError("staging file is locked")

    monkeypatch.setattr(SpecialFloatEncoder, "iterencode", broken_encoder)
    monkeypatch.setattr(file_tasks, "cleanup_output", blocked_cleanup)
    with pytest.raises(OSError, match="simulated full disk") as caught:
        file_tasks.prepare_output(
            OpticSnapshot.capture(minimal_optic),
            parameters(tmp_path),
            lambda *a, **k: None,
            threading.Event(),
        )
    assert caught.value.__notes__ == ["Staging cleanup failed: staging file is locked"]
    assert (tmp_path / ".optiland-owned.tmp").exists()
    assert not (tmp_path / "saved.json").exists()


@pytest.mark.parametrize("file_format", ["unsupported", "optiland"])
def test_failed_preparation_cleans_owned_stage_and_preserves_destination(
    minimal_optic, tmp_path, monkeypatch, file_format
):
    p = parameters(tmp_path, file_format)
    destination = tmp_path / "saved.json"
    destination.write_bytes(b"previous valid contents")
    if file_format == "optiland":

        def broken_encoder(*args):
            yield "partial contents"
            raise OSError("simulated full disk")

        monkeypatch.setattr(SpecialFloatEncoder, "iterencode", broken_encoder)
    with pytest.raises((ValueError, OSError)):
        file_tasks.prepare_output(
            OpticSnapshot.capture(minimal_optic),
            p,
            lambda *a, **k: None,
            threading.Event(),
        )
    assert destination.read_bytes() == b"previous valid contents"
    assert not (tmp_path / ".optiland-owned.tmp").exists()


def test_publication_rejects_changed_staging_contents(minimal_optic, tmp_path):
    p = parameters(tmp_path)
    result = file_tasks.prepare_output(
        OpticSnapshot.capture(minimal_optic), p, lambda *a, **k: None, threading.Event()
    )
    p.update(result)
    stage = tmp_path / ".optiland-owned.tmp"
    stage.write_text("modified after preparation")
    with pytest.raises(ValueError, match="changed before publication"):
        file_tasks.publish_output(None, p, lambda *a, **k: None, threading.Event())
    assert not (tmp_path / "saved.json").exists()
    assert stage.read_text() == "modified after preparation"


def test_array_encoding_and_unknown_values_are_lossless_or_explicitly_rejected():
    assert json.loads(
        json.dumps(
            {"array": np.array([1, 2]), "scalar": np.float32(3)},
            cls=SpecialFloatEncoder,
        )
    ) == {"array": [1, 2], "scalar": 3}
    with pytest.raises(TypeError):
        json.dumps({"unsupported": object()}, cls=SpecialFloatEncoder)


@pytest.mark.parametrize("file_format", ["optiland", "zemax", "codev"])
def test_owned_prepare_publish_and_reconcile_roundtrip(
    minimal_optic, tmp_path, file_format
):
    from optiland.fileio import load_codev_file, load_zemax_file

    from .test_file_save_export import read_optic

    p = parameters(tmp_path, file_format)
    assert file_tasks.reconcile_output(None, p, None, None) == {"matches": False}
    receipt = file_tasks.prepare_output(
        OpticSnapshot.capture(minimal_optic), p, lambda *a, **k: None, threading.Event()
    )
    p.update(receipt)
    assert not (tmp_path / "saved.json").exists()
    file_tasks.publish_output(None, p, None, None)
    assert file_tasks.reconcile_output(None, p, None, None) == {"matches": True}
    restored = (
        read_optic
        if file_format == "optiland"
        else load_zemax_file
        if file_format == "zemax"
        else load_codev_file
    )(Path(p["path"]) if file_format == "optiland" else p["path"])
    assert float(restored.surfaces[1].geometry.radius) == 50
    file_tasks.cleanup_output(None, p, None, None)


def test_unowned_stage_and_invalid_staging_path_are_never_removed(tmp_path):
    p = parameters(tmp_path)
    stage = tmp_path / ".optiland-owned.tmp"
    stage.write_bytes(b"another writer")
    p["staged_identity"] = None
    with pytest.raises(ValueError, match="ownership"):
        file_tasks.cleanup_output(None, p, None, None)
    assert stage.read_bytes() == b"another writer"
    p["staged_path"] = str(tmp_path / "not-owned.tmp")
    with pytest.raises(ValueError, match="owned staging"):
        file_tasks.cleanup_output(None, p, None, None)
