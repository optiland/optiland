"""Stage, verify, publish and clean owned output files in the worker."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from optiland.fileio import save_codev_file, save_zemax_file
from optiland_gui.services.job_records import check_cancelled


def _stage_path(parameters):
    path = Path(parameters["staged_path"])
    destination = Path(parameters["path"])
    if (
        path.parent != destination.parent
        or not path.name.startswith(".optiland-")
        or path.suffix != ".tmp"
    ):
        raise ValueError("Invalid owned staging path.")
    return path


def _digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_output(snapshot, parameters, progress, cancelled):
    """Write an owned sibling file while the destination remains untouched."""
    from optiland_gui.services.file_service import SpecialFloatEncoder

    path = _stage_path(parameters)
    progress("Preparing file contents")
    optic = snapshot.restore()
    created = False
    identity = None
    try:
        # Reserve only our unique sibling file, never truncate the destination.
        with open(path, "x", encoding="utf-8") as stream:
            created = True
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            progress(
                "Writing owned staging file",
                details={"staged_identity": identity},
            )
            if parameters["format"] == "optiland":
                encoder = SpecialFloatEncoder(indent=4)
                for chunk in encoder.iterencode(optic.to_dict()):
                    check_cancelled(cancelled)
                    stream.write(chunk)
        if parameters["format"] == "zemax":
            save_zemax_file(optic, path)
        elif parameters["format"] == "codev":
            save_codev_file(optic, path)
        elif parameters["format"] != "optiland":
            raise ValueError("Unsupported output format.")
        check_cancelled(cancelled)
        progress("Verifying prepared file")
        return {"digest": _digest(path), "staged_identity": identity}
    except BaseException as error:
        if created:
            try:
                cleanup_output(
                    None, dict(parameters, staged_identity=identity), None, None
                )
            except (OSError, ValueError) as cleanup_error:
                error.add_note(f"Staging cleanup failed: {cleanup_error}")
        raise


def publish_output(snapshot, parameters, progress, cancelled):
    """Called only after the GUI accepts the non-cancellable commit phase."""
    path = _stage_path(parameters)
    _confirm_ownership(path.stat(), parameters["staged_identity"])
    if _digest(path) != parameters["digest"]:
        raise ValueError("Prepared file contents changed before publication.")
    os.replace(path, parameters["path"])
    return {"published": True}


def reconcile_output(snapshot, parameters, progress, cancelled):
    """Confirm content after an interrupted publication without claiming rollback."""
    path = Path(parameters["path"])
    return {"matches": path.is_file() and _digest(path) == parameters["digest"]}


def cleanup_output(snapshot, parameters, progress, cancelled):
    """Remove only this operation's own staging file, outside the Qt thread."""
    path = _stage_path(parameters)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    _confirm_ownership(stat, parameters["staged_identity"])
    path.unlink()


def _confirm_ownership(stat, identity):
    if identity is None or tuple(identity) != (stat.st_dev, stat.st_ino):
        raise ValueError(
            "Staging ownership is unconfirmed; the file was left untouched."
        )
