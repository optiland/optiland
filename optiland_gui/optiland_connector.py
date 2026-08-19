"""Defines the connector that bridges the GUI and the Optiland core logic.

This module contains the ``OptilandConnector`` class, which acts as a thin
facade delegating to focused service classes.  All public signals and methods
are preserved so that existing panel code continues to work without
modification.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from optiland.optic import Optic
from optiland_gui.services.analysis_runner import AnalysisRunner
from optiland_gui.services.file_service import (
    FileService,
    SpecialFloatEncoder,  # re-exported for backward compat
    json_inf_nan_hook,  # re-exported for backward compat
)
from optiland_gui.services.optimization_service import OptimizationService
from optiland_gui.services.surface_service import SurfaceService
from optiland_gui.services.system_service import SystemService
from optiland_gui.undo_redo_manager import UndoRedoManager

__all__ = [
    "OptilandConnector",
    "SpecialFloatEncoder",
    "json_inf_nan_hook",
]


class OptilandConnector(QObject):
    """Thin facade that delegates all domain logic to focused service classes.

    All existing public signals and method signatures are preserved for
    backward compatibility with panel code.
    """

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    opticLoaded = Signal()
    opticChanged = Signal()
    modifiedStateChanged = Signal(bool)
    surfaceDataChanged = Signal(int, int, object)
    surfaceAdded = Signal(int)
    surfaceRemoved = Signal(int)
    surfaceCountChanged = Signal()
    undoStackAvailabilityChanged = Signal(bool)
    redoStackAvailabilityChanged = Signal(bool)
    optimizationVariablesChanged = Signal()
    requestAddOptimizationVariable = Signal(int, str)  # surface_index, suggested_type

    # ------------------------------------------------------------------
    # Class-level constants (used by LensEditor and other panels)
    # ------------------------------------------------------------------
    COL_TYPE = 0
    COL_COMMENT = 1
    COL_RADIUS = 2
    COL_THICKNESS = 3
    COL_MATERIAL = 4
    COL_CONIC = 5
    COL_SEMI_DIAMETER = 6

    DEFAULT_WAVELENGTH_UM = 0.550

    # Kept for backward compatibility; canonical copies live on SurfaceService.
    AVAILABLE_SURFACE_TYPES = SurfaceService.AVAILABLE_SURFACE_TYPES
    EXTRA_PARAM_MAP = SurfaceService.EXTRA_PARAM_MAP

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()

        self._optic = Optic("Default System")
        self._undo_redo_manager = UndoRedoManager(self)

        # Instantiate services — order does not matter; each receives *self*.
        self._file_service = FileService(self)
        self._surface_service = SurfaceService(self)
        self._system_service = SystemService(self)
        self._analysis_runner = AnalysisRunner(self)
        self._optimization_service = OptimizationService(self)

        self._initialize_optic_structure(self._optic, is_specific_new_system=True)
        self._is_modified = False

        self._undo_redo_manager.undoStackAvailabilityChanged.connect(
            self.undoStackAvailabilityChanged
        )
        self._undo_redo_manager.redoStackAvailabilityChanged.connect(
            self.redoStackAvailabilityChanged
        )
        self.opticLoaded.emit()
        self.opticChanged.emit()
        self._undo_redo_manager.clear_stacks()

    # ------------------------------------------------------------------
    # Shared state utilities (stay on connector; used by services)
    # ------------------------------------------------------------------

    def set_modified(self, modified: bool) -> None:
        """Set the modified flag and emit ``modifiedStateChanged`` if changed.

        Args:
            modified: New modified state.
        """
        if self._is_modified != modified:
            self._is_modified = modified
            self.modifiedStateChanged.emit(self._is_modified)

    def is_modified(self) -> bool:
        """Return whether the current design has unsaved changes.

        Returns:
            ``True`` if the optic has been modified since last save/load.
        """
        return self._is_modified

    def get_optic(self) -> Optic:
        """Return the active :class:`~optiland.optic.Optic` instance.

        Returns:
            The current optic object.
        """
        return self._optic

    # ------------------------------------------------------------------
    # Internal helpers (shared by services via self._connector)
    # ------------------------------------------------------------------

    def _create_new_optic_structure(self, optic: Optic) -> None:
        """Populate *optic* with a default 3-surface structure.

        Args:
            optic: The optic to initialise.
        """
        optic.surfaces.clear()
        optic.wavelengths.wavelengths.clear()
        optic.surfaces.add(
            index=0,
            surface_type="standard",
            radius=float("inf"),
            thickness=float("inf"),
            comment="Object",
            material="Air",
        )
        optic.surfaces.add(
            index=1,
            surface_type="standard",
            radius=float("inf"),
            thickness=20.0,
            comment="Stop",
            material="Air",
            is_stop=True,
        )
        optic.surfaces.add(
            index=2,
            surface_type="standard",
            radius=float("inf"),
            thickness=0.0,
            comment="Image",
            material="Air",
        )
        optic.wavelengths.add(self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um")
        optic.fields.set_type("angle")
        optic.fields.add(y=0)
        optic.set_aperture("EPD", 10.0)

    def _ensure_valid_optic_structure(self, optic: Optic) -> None:
        """Ensure a loaded optic has a minimally valid structure.

        Args:
            optic: The optic to validate and repair if necessary.
        """
        if optic.surfaces.num_surfaces < 2:
            optic.surfaces.clear()
            optic.surfaces.add(
                surface_type="standard",
                radius=float("inf"),
                thickness=10.0,
                comment="Object",
                material="Air",
            )
            optic.surfaces.add(
                surface_type="standard",
                radius=float("inf"),
                thickness=0.0,
                comment="Image",
                material="Air",
            )

        if optic.wavelengths.num_wavelengths == 0:
            optic.wavelengths.add(
                self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
            )
        elif optic.wavelengths.primary_index is None:
            optic.wavelengths.wavelengths[0].is_primary = True

        if not hasattr(optic, "aperture") or optic.aperture is None:
            try:
                optic.set_aperture("EPD", 10.0)
            except Exception as e:
                print(f"Warning: Failed to set aperture for loaded system: {e}")

    def _initialize_optic_structure(
        self,
        optic_instance: Optic,
        is_specific_new_system: bool = False,
    ) -> None:
        """Initialise *optic_instance*, creating a new structure or validating it.

        Args:
            optic_instance: The optic to initialise.
            is_specific_new_system: If ``True``, a fresh 3-surface structure is
                created; otherwise the existing structure is validated.
        """
        if is_specific_new_system:
            self._create_new_optic_structure(optic_instance)
        else:
            self._ensure_valid_optic_structure(optic_instance)
        optic_instance.updater.update()

    def _capture_optic_state(self) -> dict:
        """Serialise the current optic state for undo/redo.

        Returns:
            A dict representation of the current optic.
        """
        if self._optic.wavelengths.num_wavelengths == 0:
            self._optic.wavelengths.add(
                self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
            )
        elif (
            self._optic.wavelengths.primary_index is None
            and self._optic.wavelengths.num_wavelengths > 0
        ):
            self._optic.wavelengths.wavelengths[0].is_primary = True
        self._optic.updater.update()
        return self._optic.to_dict()

    def _restore_optic_state(self, state_data: dict) -> None:
        """Restore the optic from a previously captured state dict.

        Args:
            state_data: A dict returned by :meth:`_capture_optic_state`.
        """
        self._optic = Optic.from_dict(state_data)
        self._initialize_optic_structure(self._optic, is_specific_new_system=False)
        self.opticLoaded.emit()

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def undo(self) -> None:
        """Revert to the previous design state."""
        if self._undo_redo_manager.can_undo():
            state = self._undo_redo_manager.undo(self._capture_optic_state())
            if state:
                self._restore_optic_state(state)

    def redo(self) -> None:
        """Re-apply the next design state."""
        if self._undo_redo_manager.can_redo():
            state = self._undo_redo_manager.redo(self._capture_optic_state())
            if state:
                self._restore_optic_state(state)

    # ------------------------------------------------------------------
    # FileService delegation
    # ------------------------------------------------------------------

    def new_system(self) -> None:
        """Reset the workspace to a blank default optical system."""
        self._file_service.new_system()

    def load_optic_from_file(self, filepath: str) -> None:
        """Load an optical system from *filepath*.

        Args:
            filepath: Path to an Optiland JSON or Zemax file.
        """
        self._file_service.load(filepath)

    def save_optic_to_file(self, filepath: str) -> None:
        """Save the current optical system to *filepath*.

        Args:
            filepath: Destination path for the Optiland JSON file.
        """
        self._file_service.save(filepath)

    def load_optic_from_object(self, optic_instance: Optic) -> None:
        """Load an optical system from an instantiated Optic object.

        Args:
            optic_instance: An :class:`~optiland.optic.Optic` to load.
        """
        self._file_service.load_from_object(optic_instance)

    def get_current_filepath(self) -> str | None:
        """Return the path of the last successfully saved/loaded file.

        Returns:
            The file path, or ``None`` if no file has been saved/loaded.
        """
        return self._file_service.get_current_filepath()

    def import_zemax(self, filepath: str) -> None:
        """Import a Zemax ``.zmx`` file, replacing the current system.

        Args:
            filepath: Path to the ``.zmx`` file.
        """
        self._file_service.import_zemax(filepath)

    def import_codev(self, filepath: str) -> None:
        """Import a CODE V ``.seq`` file, replacing the current system.

        Args:
            filepath: Path to the ``.seq`` file.
        """
        self._file_service.import_codev(filepath)

    def export_zemax(self, filepath: str) -> None:
        """Export the current system to a Zemax ``.zmx`` file.

        Args:
            filepath: Destination path for the ``.zmx`` file.
        """
        self._file_service.export_zemax(filepath)

    def export_codev(self, filepath: str) -> None:
        """Export the current system to a CODE V ``.seq`` file.

        Args:
            filepath: Destination path for the ``.seq`` file.
        """
        self._file_service.export_codev(filepath)

    # ------------------------------------------------------------------
    # SurfaceService delegation
    # ------------------------------------------------------------------

    def get_surface_count(self) -> int:
        """Return the number of surfaces in the active optic."""
        return self._surface_service.get_surface_count()

    def get_column_headers(self, row: int = -1) -> list[str]:
        """Return LDE column headers, dynamically adjusted for *row*.

        Args:
            row: LDE row index, or ``-1`` for generic headers.

        Returns:
            A list of seven header strings.
        """
        return self._surface_service.get_column_headers(row)

    def get_available_surface_types(self) -> list[str]:
        """Return the list of supported surface type strings."""
        return self._surface_service.get_available_surface_types()

    def get_surface_type_info(self, row: int) -> dict:
        """Return display metadata for a surface row.

        Args:
            row: LDE row index (0-based).

        Returns:
            A dict with ``display_text``, ``is_changeable``, and optionally
            ``has_extra_params``.
        """
        return self._surface_service.get_surface_type_info(row)

    def get_surface_data(self, row: int, col_idx: int) -> object:
        """Return the display value for a given LDE cell.

        Args:
            row: LDE row index (0-based).
            col_idx: Column index constant.

        Returns:
            A string value or ``None``.
        """
        return self._surface_service.get_surface_data(row, col_idx)

    def set_surface_data(self, row: int, col_idx: int, value_str: str) -> None:
        """Write a value to a specific LDE cell.

        Args:
            row: LDE row index (0-based).
            col_idx: Column index constant.
            value_str: The new value as entered by the user.
        """
        self._surface_service.set_surface_data(row, col_idx, value_str)

    def set_surface_type(self, row: int, new_type: str) -> None:
        """Change the geometry type for a surface.

        Args:
            row: LDE row index.
            new_type: New surface type string.
        """
        self._surface_service.set_surface_type(row, new_type)

    def add_surface(self, index: int = -1) -> None:
        """Insert a new standard surface.

        Args:
            index: Insertion index, or ``-1`` to insert before the image surface.
        """
        self._surface_service.add_surface(index)

    def remove_surface(self, lde_row_index: int) -> None:
        """Remove a surface by its LDE row index.

        Args:
            lde_row_index: Row index of the surface to remove.
        """
        self._surface_service.remove_surface(lde_row_index)

    def set_stop_surface(self, row: int) -> None:
        """Set the surface at *row* as the aperture stop.

        Args:
            row: LDE row index of the surface to set as stop.
        """
        self._surface_service.set_stop_surface(row)

    def get_surface_geometry_params(self, row: int) -> dict:
        """Return the extra geometry parameters for the surface properties box.

        Args:
            row: LDE row index.

        Returns:
            A dict of parameter label → value pairs.
        """
        return self._surface_service.get_surface_geometry_params(row)

    def set_surface_geometry_params(self, row: int, params_dict: dict) -> None:
        """Apply extra geometry parameters from the surface properties box.

        Args:
            row: LDE row index.
            params_dict: A dict of parameter label → new value pairs.
        """
        self._surface_service.set_surface_geometry_params(row, params_dict)

    # ------------------------------------------------------------------
    # SystemService delegation
    # ------------------------------------------------------------------

    def get_field_options(self) -> list[tuple[str, str]]:
        """Return field selector options for the analysis panel.

        Returns:
            A list of ``(display_name, value_str)`` tuples.
        """
        return self._system_service.get_field_options()

    def get_wavelength_options(self) -> list[tuple[str, str]]:
        """Return wavelength selector options for the analysis panel.

        Returns:
            A list of ``(display_name, value_str)`` tuples.
        """
        return self._system_service.get_wavelength_options()

    def get_aperture_types(self) -> list[str]:
        """Return all aperture type keys registered with BaseSystemAperture.

        Returns:
            A sorted list of aperture type identifier strings.
        """
        return self._system_service.get_aperture_types()

    def get_field_types(self) -> list[tuple[str, str]]:
        """Return all four supported field types.

        Returns:
            A list of ``(display_name, type_key)`` tuples.
        """
        return self._system_service.get_field_types()

    def set_polarization_state(
        self,
        mode: str,
        Ex: float | None = None,
        Ey: float | None = None,
        phase_x_deg: float | None = None,
        phase_y_deg: float | None = None,
    ) -> None:
        """Set or clear the polarization state on the active optic.

        Args:
            mode: "ignore", "unpolarized", or "polarized".
            Ex: Electric field x-component.
            Ey: Electric field y-component.
            phase_x_deg: Phase of Ex in degrees.
            phase_y_deg: Phase of Ey in degrees.
        """
        self._system_service.set_polarization_state(
            mode, Ex, Ey, phase_x_deg, phase_y_deg
        )

    def get_geometry_types(self) -> list[str]:
        """Return all geometry type keys registered with GeometryFactory.

        Returns:
            A sorted list of geometry type identifier strings.
        """
        return self._surface_service.get_geometry_types()

    # ------------------------------------------------------------------
    # OptimizationService delegation
    # ------------------------------------------------------------------

    def add_optimization_variable(self, var_dict: dict) -> None:
        """Append a variable descriptor and emit variablesChanged.

        Args:
            var_dict: Variable descriptor dict (surface_number, type, etc.).
        """
        self._optimization_service.add_variable(var_dict)
        self.optimizationVariablesChanged.emit()

    def remove_optimization_variable(self, index: int) -> None:
        """Remove a variable by index and emit variablesChanged.

        Args:
            index: Zero-based index of the variable to remove.
        """
        self._optimization_service.remove_variable(index)
        self.optimizationVariablesChanged.emit()

    def get_optimization_variables(self) -> list[dict]:
        """Return the current list of optimization variable descriptors."""
        return self._optimization_service.get_variables()

    def set_optimization_variable(self, index: int, var_dict: dict) -> None:
        """Replace an optimization variable at *index*."""
        self._optimization_service.set_variable(index, var_dict)
        self.optimizationVariablesChanged.emit()

    def get_variable_current_value(self, var_dict: dict) -> float | None:
        """Return the live optic value for a variable descriptor.

        Args:
            var_dict: Variable descriptor dict.

        Returns:
            The current float value, or ``None`` if unavailable.
        """
        return self._optimization_service.get_variable_current_value(var_dict)

    def add_optimization_operand(self, op_dict: dict) -> None:
        """Append an operand descriptor.

        Args:
            op_dict: Operand descriptor dict (type, category, etc.).
        """
        self._optimization_service.add_operand(op_dict)

    def remove_optimization_operand(self, index: int) -> None:
        """Remove an operand by index.

        Args:
            index: Zero-based index.
        """
        self._optimization_service.remove_operand(index)

    def get_optimization_operands(self) -> list[dict]:
        """Return the current list of optimization operand descriptors."""
        return self._optimization_service.get_operands()

    def set_optimization_operand(self, index: int, op_dict: dict) -> None:
        """Replace an optimization operand at *index*."""
        self._optimization_service.set_operand(index, op_dict)

    def get_operand_current_value(self, op_dict: dict) -> float | None:
        """Return the live optic value for an operand descriptor."""
        return self._optimization_service.get_operand_current_value(op_dict)

    def get_optimizer_catalog(self) -> list[tuple[str, type]]:
        """Return available optimizer classes as (display_name, cls) tuples.

        Returns:
            A list of ``(name, class)`` pairs.
        """
        return self._optimization_service.get_optimizer_catalog()

    def get_operand_categories(self) -> dict[str, list[str]]:
        """Return operand types grouped by category.

        Returns:
            A dict mapping category name to list of operand type keys.
        """
        return self._optimization_service.OPERAND_CATEGORIES

    def get_common_variable_types(self) -> list[tuple[str, str]]:
        """Return common variable types as (display_name, type_key) tuples.

        Returns:
            A list of ``(display_name, type_key)`` pairs.
        """
        return self._optimization_service.COMMON_VARIABLE_TYPES

    def get_variable_metadata(self, var_type: str) -> dict:
        """Return graphical configuration metadata for a variable type.

        Args:
            var_type: Variable type key.

        Returns:
            Dict of parameter metadata names and their default values/types.
        """
        return self._optimization_service.get_variable_metadata(var_type)

    def get_default_operand_input_data_str(self, op_type: str) -> str:
        """Return the default JSON input_data string for an operand type.

        Args:
            op_type: Operand type key.

        Returns:
            JSON string (optic excluded).
        """
        return self._optimization_service.get_default_input_data_str(op_type)

    def validate_operand_input_data(
        self, op_type: str, input_data: str | dict | None
    ) -> str | None:
        """Validate that *input_data* contains required keys for *op_type*.

        Args:
            op_type: Operand type key.
            input_data: JSON string or dict of extra parameters.

        Returns:
            An error message string if validation fails, or ``None`` if valid.
        """
        return self._optimization_service.validate_operand_input_data(
            op_type, input_data
        )

    def get_operand_metadata(self, op_type: str) -> dict:
        """Return graphical configuration metadata for an operand type.

        Args:
            op_type: Operand type key.

        Returns:
            Dict of parameter metadata names and their default values/types.
        """
        return self._optimization_service.get_operand_metadata(op_type)

    def get_optimizer_metadata(self, optimizer_cls: type) -> dict:
        """Return optimization parameter metadata for an optimizer class.

        Args:
            optimizer_cls: The optimizer class.

        Returns:
            Dict of parameter metadata.
        """
        return self._optimization_service.get_optimizer_metadata(optimizer_cls)

    def get_optimizer_groups(self) -> dict:
        """Return optimisers organised into Local/Global groups.

        Returns:
            Dict mapping group name → list of (display_name, cls, bounds_mode).
        """
        return self._optimization_service.get_optimizer_groups()

    def validate_bounds_for_optimizer(self, optimizer_cls: type) -> str | None:
        """Check that variable bounds match the optimizer's requirements.

        Args:
            optimizer_cls: The optimizer class to validate against.

        Returns:
            An error message string if validation fails, or ``None`` if valid.
        """
        return self._optimization_service.validate_bounds_for_optimizer(optimizer_cls)

    def run_optimization(
        self,
        optimizer_cls: type,
        optimizer_kwargs: dict,
        on_progress: object | None = None,
        on_finished: object | None = None,
        on_error: object | None = None,
    ) -> None:
        """Start the optimizer in a background thread.

        Args:
            optimizer_cls: Optimizer class.
            optimizer_kwargs: Keyword args for ``optimizer.optimize()``.
            on_progress: Optional ``(n: int) -> None`` callback.
            on_finished: Optional ``(summary: str) -> None`` callback.
            on_error: Optional ``(message: str) -> None`` callback.
        """
        self._optimization_service.run(
            optimizer_cls, optimizer_kwargs, on_progress, on_finished, on_error
        )

    def stop_optimization(self) -> None:
        """Request cancellation of the current optimisation run."""
        self._optimization_service.stop()

    def is_optimization_running(self) -> bool:
        """Return ``True`` if an optimisation thread is active.

        Returns:
            ``True`` while running, ``False`` otherwise.
        """
        return self._optimization_service.is_running
