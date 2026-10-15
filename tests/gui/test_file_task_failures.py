"""File preparation failures preserve the destination and owned-model boundary."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from optiland.optic import Optic
from optiland_gui.services import file_tasks
from optiland_gui.services.file_service import SpecialFloatEncoder
from optiland_gui.services.job_records import BackendConfig, OpticSnapshot
from optiland_gui.services.model_initialization import initialize_loaded_optic


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


def test_invalid_stage_and_load_formats_fail_before_file_changes(tmp_path):
    p = parameters(tmp_path)
    p["staged_path"] = str(tmp_path / "unowned.tmp")
    with pytest.raises(ValueError, match="owned staging"):
        file_tasks.cleanup_output(None, p, None, None)
    p["format"] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported file format"):
        file_tasks.load_file(None, p, lambda *a, **k: None, threading.Event())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "sample", [("optiland.optic", "Optic"), ("optiland.samples.objectives", "__doc__")]
)
def test_gallery_rejects_non_builtin_optical_classes(tmp_path, sample):
    p = parameters(tmp_path, "sample")
    p["load_options"]["sample_class"] = sample
    with pytest.raises(ValueError, match="Gallery|Optic class"):
        file_tasks.load_file(None, p, lambda *a, **k: None, threading.Event())


def test_loaded_empty_model_gets_valid_minimal_structure():
    optic = Optic()
    initialize_loaded_optic(optic)
    assert optic.surfaces.num_surfaces == 2
    assert optic.wavelengths.primary_wavelength.value == 0.55
    assert optic.aperture is not None


def test_array_encoding_and_unknown_values_are_lossless_or_explicitly_rejected():
    assert json.loads(
        json.dumps(
            {"array": np.array([1, 2]), "scalar": np.float32(3)},
            cls=SpecialFloatEncoder,
        )
    ) == {"array": [1, 2], "scalar": 3}
    with pytest.raises(TypeError):
        json.dumps({"unsupported": object()}, cls=SpecialFloatEncoder)
