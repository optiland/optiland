"""File I/O service for the Optiland GUI.

Handles loading and saving Optiland JSON files, importing Zemax files,
and all file-path state. ``SpecialFloatEncoder`` and ``json_inf_nan_hook``
live here so that the JSON serialisation logic is co-located with the file
operations that use it.
"""

from __future__ import annotations

import json
import os
import tempfile

import optiland.backend as be
from optiland.fileio import (
    load_codev_file,
    load_zemax_file,
    save_codev_file,
    save_zemax_file,
)
from optiland.optic import Optic


class SpecialFloatEncoder(json.JSONEncoder):
    """JSON encoder that serialises ``inf`` and ``nan`` as strings.

    The standard ``json`` module raises ``ValueError`` for these values.
    This encoder converts them to the string tokens ``"Infinity"``,
    ``"-Infinity"``, and ``"NaN"`` so that round-tripping through JSON is
    lossless when combined with :func:`json_inf_nan_hook`.
    """

    def _encode_special_float(self, f: float) -> str | None:
        """Return a string token for a special float, or ``None`` if normal.

        Args:
            f: The float value to inspect.

        Returns:
            A string token for special floats, or ``None`` for ordinary ones.
        """
        if f == float("inf"):
            return "Infinity"
        if f == float("-inf"):
            return "-Infinity"
        if be.isnan(f):
            return "NaN"
        return None

    def default(self, obj: object) -> object:
        """Encode an object, handling special floats and array-like types.

        Args:
            obj: The object to encode.

        Returns:
            A JSON-serialisable representation of *obj*.
        """
        if isinstance(obj, float):
            encoded = self._encode_special_float(obj)
            if encoded is not None:
                return encoded

        if hasattr(obj, "tolist") and callable(obj.tolist):
            return obj.tolist()
        if hasattr(obj, "item") and callable(obj.item):
            return obj.item()
        # Unknown optical data must fail before publication, never turn into
        # display text that cannot be reconstructed when the file is reopened.
        return super().default(obj)


def json_inf_nan_hook(dct: dict) -> dict:
    """``object_hook`` that converts special float string tokens back to floats.

    Args:
        dct: A decoded JSON object dictionary.

    Returns:
        The same dictionary with ``"Infinity"``, ``"-Infinity"``, and
        ``"NaN"`` string values replaced with the corresponding Python floats.
    """
    for k, v in dct.items():
        if isinstance(v, str):
            if v == "Infinity":
                dct[k] = float("inf")
            elif v == "-Infinity":
                dct[k] = float("-inf")
            elif v == "NaN":
                dct[k] = float("nan")
    return dct


class FileService:
    """Manages all file I/O operations for the Optiland GUI.

    Responsibilities include loading Optiland JSON files, saving to JSON,
    importing Zemax ``.zmx`` files, and tracking the current file path.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance that owns this service. Used to access the optic, the
            undo/redo manager, and to emit signals.
    """

    def __init__(self, connector: object) -> None:
        self._connector = connector
        self._current_filepath: str | None = None
        from optiland_gui.services.file_operations import FileOperations

        self.operations = FileOperations(self, connector)

    # ------------------------------------------------------------------
    # Toast helper
    # ------------------------------------------------------------------

    def _toast(self, message: str, severity: str, sub: str | None = None) -> None:
        """Forward *message* to the application's toast manager if available."""
        tm = getattr(self._connector, "toast_manager", None)
        if tm is not None:
            tm.notify(message, severity, sub_message=sub)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_system(self) -> None:
        """Resets the workspace to a blank default optical system.

        Clears the undo/redo stacks, creates a new :class:`~optiland.optic.Optic`,
        initialises a default 3-surface structure, and emits the appropriate
        signals.
        """
        self.operations.cancel_load()
        self._connector._undo_redo_manager.clear_stacks()
        self._connector._optic = Optic("New Untitled System")
        self._connector._initialize_optic_structure(
            self._connector._optic, is_specific_new_system=True
        )
        self._current_filepath = None
        self._connector.set_modified(False)
        self._connector.notify_change("replacement")

    def load(self, filepath: str) -> None:
        """Load an optical system from *filepath*.

        Supports Optiland JSON (``.json``) and Zemax (``.zmx``) files.
        On success, the undo/redo stack is cleared and ``opticLoaded`` is
        emitted. A failed parse or validation preserves the current document.

        Args:
            filepath: Absolute path to the file to load.
        """
        self.operations.cancel_load()
        try:
            _name, extension = os.path.splitext(filepath)
            if extension.lower() == ".zmx":
                candidate = load_zemax_file(filepath)
                current_filepath = None
            else:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f, object_hook=json_inf_nan_hook)
                candidate = Optic.from_dict(data)
                current_filepath = filepath
            self._publish_candidate(candidate, current_filepath, modified=False)
            self._toast(f"Opened \u2014 {os.path.basename(filepath)}", "info")
        except Exception as e:
            self._toast(f"Load failed: {e}", "error", sub=filepath)

    def save(self, filepath: str) -> None:
        """Save the current optical system to *filepath* as Optiland JSON.

        On success, updates the current file path and clears the modified flag.
        On failure, shows an error message box.

        Args:
            filepath: Absolute path to write to.
        """
        try:
            if self.operations.path_busy(filepath):
                raise RuntimeError(
                    "An asynchronous save to this path is still running."
                )
            sequence = self.operations.next_save_sequence()
            edit_token = self._connector.document_state.edit_token
            data = self._connector._capture_optic_state()
            destination = os.path.abspath(filepath)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=os.path.dirname(destination),
                    prefix=".optiland-",
                    suffix=".tmp",
                    delete=False,
                ) as f:
                    temporary = f.name
                    json.dump(data, f, indent=4, cls=SpecialFloatEncoder)
                os.replace(temporary, destination)
                temporary = None
            finally:
                if temporary is not None:
                    os.unlink(temporary)
            self.operations.confirm_saved_document(filepath, edit_token, sequence)
            self._toast(f"Saved \u2014 {os.path.basename(filepath)}", "success")
        except Exception as e:
            self._toast(f"Save failed: {e}", "error", sub=filepath)

    def load_from_object(self, optic_instance: Optic) -> None:
        """Load an optical system from an already-instantiated Optic object.

        Serialises the optic to a dict and reconstructs it so the connector
        owns its own copy. Clears the undo/redo stack and emits ``opticLoaded``
        and ``opticChanged``.

        Args:
            optic_instance: An instantiated :class:`~optiland.optic.Optic` to load.
        """
        try:
            optic_data = optic_instance.to_dict()
            candidate = Optic.from_dict(optic_data)
            self._publish_candidate(candidate, None, modified=True)
        except Exception as e:
            self._toast(f"Failed to load system from sample object: {e}", "error")

    def import_zemax(self, filepath: str) -> None:
        """Import a Zemax ``.zmx`` file, replacing the current system.

        Clears the undo/redo stack. The current file path is set to ``None``
        because the imported system has no associated JSON file.

        Args:
            filepath: Path to the ``.zmx`` file to import.
        """
        try:
            candidate = load_zemax_file(filepath)
            self._publish_candidate(candidate, None, modified=True)
        except Exception as e:
            self._toast(f"Failed to import Zemax file from {filepath}: {e}", "error")

    def import_codev(self, filepath: str) -> None:
        """Import a CODE V ``.seq`` file, replacing the current system.

        Clears the undo/redo stack. The current file path is set to ``None``
        because the imported system has no associated JSON file.

        Args:
            filepath: Path to the ``.seq`` file to import.
        """
        try:
            candidate = load_codev_file(filepath)
            self._publish_candidate(candidate, None, modified=True)
        except Exception as e:
            self._toast(f"Failed to import CODE V file from {filepath}: {e}", "error")

    def export_zemax(self, filepath: str) -> None:
        """Export the current system to a Zemax ``.zmx`` file.

        This is a non-destructive export: it does not update
        :attr:`_current_filepath` or modify the modified flag.

        Args:
            filepath: Destination path for the ``.zmx`` file.
        """
        try:
            optic = self._connector._optic
            if optic is None:
                self._toast("No optical system loaded to export.", "warning")
                return
            save_zemax_file(optic, filepath)
        except Exception as e:
            self._toast(f"Failed to export Zemax file to {filepath}: {e}", "error")

    def export_codev(self, filepath: str) -> None:
        """Export the current system to a CODE V ``.seq`` file.

        This is a non-destructive export: it does not update
        :attr:`_current_filepath` or modify the modified flag.

        Args:
            filepath: Destination path for the ``.seq`` file.
        """
        try:
            optic = self._connector._optic
            if optic is None:
                self._toast("No optical system loaded to export.", "warning")
                return
            save_codev_file(optic, filepath)
        except Exception as e:
            self._toast(f"Failed to export CODE V file to {filepath}: {e}", "error")

    def get_current_filepath(self) -> str | None:
        """Return the path of the last successfully saved/loaded JSON file.

        Returns:
            The file path string, or ``None`` if the system has never been
            saved to or loaded from a file in this session.
        """
        return self._current_filepath

    def _publish_candidate(self, candidate, filepath, *, modified, validated=False):
        """Validate before changing document ownership, history or path state."""
        if not validated:
            self._connector._initialize_optic_structure(candidate)
        self._connector._optic = candidate
        self._connector._undo_redo_manager.clear_stacks()
        self._current_filepath = filepath
        self._connector.set_modified(modified)
        self._connector.notify_change("replacement")
