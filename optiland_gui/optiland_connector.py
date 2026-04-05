"""Defines the connector that bridges the GUI and the Optiland core logic.

This module contains the `OptilandConnector` class, which acts as a vital
intermediary between the user interface and the underlying `optiland` optical
engine. It manages the active optical system (`Optic` object), handles file
I/O, provides data to UI components like the Lens Editor, and manages the
undo/redo stack.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import ast
import json
import os

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox

import optiland.backend as be
from optiland.fileio.zemax_handler import load_zemax_file
from optiland.materials import IdealMaterial
from optiland.materials import Material as OptilandMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.physical_apertures.radial import configure_aperture
from optiland.surfaces.factories.geometry_factory import GeometryFactory
from optiland_gui.undo_redo_manager import UndoRedoManager


class SpecialFloatEncoder(json.JSONEncoder):
    def _encode_special_float(self, f: float):
        """Encodes infinity and NaN floats into strings."""
        if f == float("inf"):
            return "Infinity"
        if f == float("-inf"):
            return "-Infinity"
        if be.isnan(f):
            return "NaN"
        return None

    def default(self, obj):
        if isinstance(obj, float):
            encoded = self._encode_special_float(obj)
            if encoded is not None:
                return encoded

        if hasattr(obj, "item") and isinstance(obj.item(), float):
            encoded = self._encode_special_float(obj.item())
            if encoded is not None:
                return encoded

        try:
            return super().default(obj)
        except TypeError:
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return str(obj)


class OptilandConnector(QObject):
    """The main bridge between the Optiland core logic and the GUI."""

    opticLoaded = Signal()
    opticChanged = Signal()
    modifiedStateChanged = Signal(bool)
    surfaceDataChanged = Signal(int, int, object)
    surfaceAdded = Signal(int)
    surfaceRemoved = Signal(int)
    surfaceCountChanged = Signal()
    undoStackAvailabilityChanged = Signal(bool)
    redoStackAvailabilityChanged = Signal(bool)

    # Column indices
    COL_TYPE = 0
    COL_COMMENT = 1
    COL_RADIUS = 2
    COL_THICKNESS = 3
    COL_MATERIAL = 4
    COL_CONIC = 5
    COL_SEMI_DIAMETER = 6

    DEFAULT_WAVELENGTH_UM = 0.550

    AVAILABLE_SURFACE_TYPES = [
        "standard",
        "paraxial",
        "biconic",
        "chebyshev",
        "even_asphere",
        "odd_asphere",
        "polynomial",
        "toroidal",
        "zernike",
    ]

    # Map for EXTRA parameters that appear ONLY in the properties box
    EXTRA_PARAM_MAP = {
        "BiconicGeometry": {
            "Radius X": "Rx",
            "Conic X": "kx",
        },
        "ChebyshevPolynomialGeometry": {
            "Coefficients": "c",
            "Norm X": "norm_x",
            "Norm Y": "norm_y",
        },
        "EvenAsphere": {"Coefficients": "c"},
        "OddAsphere": {"Coefficients": "c"},
        "PolynomialGeometry": {"Coefficients": "c"},
        "ToroidalGeometry": {
            "Radius of Rotation (XZ)": "R_rot",
            "Coefficients YZ": "coeffs_poly_y",
        },
        "ZernikePolynomialGeometry": {
            "Coefficients": "c",
            "Normalization Radius": "norm_radius",
        },
    }

    def __init__(self):
        super().__init__()
        self._optic = Optic("Default System")
        self._undo_redo_manager = UndoRedoManager(self)
        self._initialize_optic_structure(self._optic, is_specific_new_system=True)
        self._current_filepath = None
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

    def set_modified(self, modified: bool):
        if self._is_modified != modified:
            self._is_modified = modified
            self.modifiedStateChanged.emit(self._is_modified)

    def is_modified(self) -> bool:
        return self._is_modified

    def _create_new_optic_structure(self, optic: Optic):
        """Creates a default 3-surface structure for a new optic."""
        optic.surface_group.clear()
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

    def _ensure_valid_optic_structure(self, optic: Optic):
        """Ensures a loaded or modified optic has a valid basic structure."""
        if optic.surface_group.num_surfaces < 2:
            # If the system is invalid, reset it to a minimal default
            optic.surface_group.clear()
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
        self, optic_instance: Optic, is_specific_new_system: bool = False
    ):
        if is_specific_new_system:
            self._create_new_optic_structure(optic_instance)
        else:
            self._ensure_valid_optic_structure(optic_instance)
        optic_instance.update()

    def _get_safe_primary_wavelength_value(self) -> float:
        if self._optic.wavelengths.num_wavelengths > 0:
            primary_idx = self._optic.wavelengths.primary_index
            if primary_idx is not None:
                try:
                    return self._optic.wavelengths.wavelengths[primary_idx].value
                except IndexError:
                    pass
            if self._optic.wavelengths.num_wavelengths > 0:
                self._optic.wavelengths.wavelengths[0].is_primary = True
                return self._optic.wavelengths.wavelengths[0].value
        if self._optic.wavelengths.num_wavelengths == 0:
            self._optic.wavelengths.add(
                self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
            )
            self._optic.update()
        return self.DEFAULT_WAVELENGTH_UM

    def get_optic(self) -> Optic:
        return self._optic

    def _capture_optic_state(self):
        if self._optic.wavelengths.num_wavelengths == 0:
            self._optic.wavelengths.add(
                self.DEFAULT_WAVELENGTH_UM, is_primary=True, unit="um"
            )
        elif (
            self._optic.wavelengths.primary_index is None
            and self._optic.wavelengths.num_wavelengths > 0
        ):
            self._optic.wavelengths.wavelengths[0].is_primary = True
        self._optic.update()
        return self._optic.to_dict()

    def _restore_optic_state(self, state_data):
        self._optic = Optic.from_dict(state_data)
        self._initialize_optic_structure(self._optic, is_specific_new_system=False)
        self.opticLoaded.emit()

    def new_system(self):
        self._undo_redo_manager.clear_stacks()
        self._optic = Optic("New Untitled System")
        self._initialize_optic_structure(self._optic, is_specific_new_system=True)
        self._current_filepath = None
        self.set_modified(False)
        self.opticLoaded.emit()
        self.opticChanged.emit()

    def load_optic_from_file(self, filepath: str):
        try:
            name, extension = os.path.splitext(filepath)
            if extension == ".zmx":
                self._undo_redo_manager.clear_stacks()
                self._optic = load_zemax_file(filepath)
                self._current_filepath = None
            else:
                with open(filepath) as f:

                    def json_inf_nan_hook(dct):
                        for k, v in dct.items():
                            if isinstance(v, str):
                                if v == "Infinity":
                                    dct[k] = float("inf")
                                elif v == "-Infinity":
                                    dct[k] = float("-inf")
                                elif v == "NaN":
                                    dct[k] = float("nan")
                        return dct

                    data = json.load(f, object_hook=json_inf_nan_hook)
                self._undo_redo_manager.clear_stacks()
                self._optic = Optic.from_dict(data)
                self._current_filepath = filepath
            self._initialize_optic_structure(self._optic, is_specific_new_system=False)
            self.set_modified(False)
            self.opticLoaded.emit()
        except Exception as e:
            QMessageBox.critical(
                None, "Load Error", f"Failed to load system from {filepath}:\n{e}"
            )
            self.new_system()

    def save_optic_to_file(self, filepath: str):
        try:
            data = self._capture_optic_state()
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4, cls=SpecialFloatEncoder)
            self._current_filepath = filepath
            self.set_modified(False)
        except Exception as e:
            QMessageBox.critical(
                None, "Save Error", f"Could not save system to {filepath}:\n{e}"
            )

    def load_optic_from_object(self, optic_instance: Optic):
        """Loads an optical system directly from an instantiated Optic object."""
        try:
            optic_data = optic_instance.to_dict()

            self._undo_redo_manager.clear_stacks()
            self._optic = Optic.from_dict(optic_data)
            self._current_filepath = None
            self._initialize_optic_structure(self._optic)
            self.set_modified(True)
            self.opticLoaded.emit()
            self.opticChanged.emit()
        except Exception as e:
            QMessageBox.critical(
                None, "Load Error", f"Failed to load system from sample object:\n{e}"
            )
            self.new_system()

    def get_current_filepath(self) -> str | None:
        return self._current_filepath

    def get_surface_count(self) -> int:
        return (
            self._optic.surface_group.num_surfaces
            if self._optic and self._optic.surface_group
            else 0
        )

    def get_column_headers(self, row=-1) -> list[str]:
        """Returns column headers, dynamically changing for the selected row
        if provided."""
        headers = [
            "Type",
            "Comment",
            "Radius",
            "Thickness",
            "Material",
            "Conic",
            "Semi-Diameter",
        ]
        if not (0 <= row < self.get_surface_count()):
            return headers

        surface = self._optic.surface_group.surfaces[row]
        if surface.surface_type == "paraxial":
            headers[self.COL_RADIUS] = "Focal Length"
            headers[self.COL_CONIC] = "N/A"
        elif surface.surface_type == "toroidal":
            headers[self.COL_RADIUS] = "Radius YZ"
            headers[self.COL_CONIC] = "Conic YZ"
        elif surface.surface_type == "biconic":
            headers[self.COL_RADIUS] = "Radius Y"
            headers[self.COL_CONIC] = "Conic Y"
        return headers

    def get_available_surface_types(self) -> list[str]:
        return self.AVAILABLE_SURFACE_TYPES

    def get_surface_type_info(self, row: int) -> dict:
        if not (0 <= row < self.get_surface_count()):
            return {"display_text": "Error", "is_changeable": False}
        surface = self._optic.surface_group.surfaces[row]
        if row == 0:
            return {"display_text": "Object", "is_changeable": False}
        if row == self.get_surface_count() - 1:
            return {"display_text": "Image", "is_changeable": False}
        base_type = surface.surface_type or "Standard"
        display_text = (
            f"Stop ({base_type.title()})" if surface.is_stop else base_type.title()
        )
        has_extra_params = bool(self.get_surface_geometry_params(row))
        return {
            "display_text": display_text,
            "is_changeable": not surface.is_stop,
            "has_extra_params": has_extra_params,
        }

    def _get_material_data(self, surface) -> str:
        """Gets the material string for a surface."""
        if surface.interaction_model.is_reflective:
            return "Mirror"
        mat = surface.material_post
        if isinstance(mat, IdealMaterial):
            n = mat.n(self._get_safe_primary_wavelength_value())
            # Use a tolerance for float comparison
            return "Air" if be.isclose(n, 1.0) else f"Ideal n={n:.4f}"
        return mat.name if isinstance(mat, OptilandMaterial) else "Unknown"

    def _get_semi_diameter_data(self, surface) -> str:
        """Gets the semi-diameter string for a surface."""
        if isinstance(surface.aperture, RadialAperture):
            return f"{float(surface.aperture.r_max):.4f}"
        if surface.semi_aperture is not None:
            return f"{float(surface.semi_aperture):.4f}"
        return "Auto"

    def _get_dynamic_radius_data(self, surface) -> str:
        """Gets radius-like data which depends on the surface type."""
        geo = surface.geometry
        val = float("inf")
        if surface.surface_type == "paraxial":
            val = surface.f
        elif surface.surface_type == "toroidal":
            val = getattr(geo, "R_yz", val)
        elif surface.surface_type == "biconic":
            val = getattr(geo, "Ry", val)
        else:
            val = geo.radius
        return "inf" if val == float("inf") else f"{val:.4f}"

    def _get_dynamic_conic_data(self, surface) -> str:
        """Gets conic-like data which depends on the surface type."""
        if surface.surface_type == "paraxial":
            return "N/A"

        geo = surface.geometry
        val = 0.0
        if surface.surface_type == "toroidal":
            val = getattr(geo, "k_yz", val)
        elif surface.surface_type == "biconic":
            val = getattr(geo, "ky", val)
        elif hasattr(geo, "k"):
            val = geo.k
        return f"{val:.4f}"

    def get_surface_data(self, row: int, col_idx: int):
        """Gets data for a specific surface and column using a dispatcher."""
        if not (0 <= row < self.get_surface_count()):
            return None

        surface = self._optic.surface_group.surfaces[row]

        # Dispatcher dictionary mapping column index to a handler function
        column_handlers = {
            self.COL_TYPE: lambda s: self.get_surface_type_info(row)["display_text"],
            self.COL_COMMENT: lambda s: s.comment,
            self.COL_RADIUS: self._get_dynamic_radius_data,
            self.COL_THICKNESS: self._get_thickness_data,
            self.COL_MATERIAL: self._get_material_data,
            self.COL_CONIC: self._get_dynamic_conic_data,
            self.COL_SEMI_DIAMETER: self._get_semi_diameter_data,
        }

        handler = column_handlers.get(col_idx)
        return handler(surface) if handler else None

    def _get_thickness_data(self, surface) -> str:
        """Gets the thickness string for a surface."""
        row = self._optic.surface_group.surfaces.index(surface)
        if row < self.get_surface_count() - 1:
            t = self._optic.surface_group.get_thickness(row)
            return f"{float(t[0]):.4f}" if t is not None and len(t) > 0 else "N/A"
        return "N/A"

    def _get_biconic_config(self, old_geo) -> dict:
        """Creates a config dict for a biconic surface."""
        if hasattr(old_geo, "Rx"):  # Check if converting from another biconic
            return {
                "radius_x": getattr(old_geo, "Rx", 100.0),
                "radius_y": getattr(old_geo, "Ry", 100.0),
                "conic_x": getattr(old_geo, "kx", 0.0),
                "conic_y": getattr(old_geo, "ky", 0.0),
            }
        else:  # Otherwise, use finite defaults to avoid a plane surface
            return {"radius_x": 100.0, "radius_y": 100.0}

    def _get_toroidal_config(self, old_geo) -> dict:
        """Creates a config dict for a toroidal surface."""
        if hasattr(old_geo, "R_rot"):
            return {
                "radius": getattr(old_geo, "R_rot", 100.0),
                "radius_y": getattr(old_geo, "R_yz", 100.0),
                "conic": getattr(old_geo, "k_yz", 0.0),
                "toroidal_coeffs_poly_y": getattr(old_geo, "coeffs_poly_y", []),
            }
        else:
            return {"radius": 100.0, "radius_y": 100.0}

    def _get_polynomial_config(self, old_geo) -> dict:
        """Creates a config dict for polynomial-based surfaces."""
        config = {
            "radius": getattr(old_geo, "radius", float("inf")),
            "conic": getattr(old_geo, "k", 0.0),
            "coefficients": getattr(old_geo, "c", []),
        }
        if config["radius"] == float("inf"):
            config["radius"] = 100.0  # Use a finite default to avoid a plane
        return config

    def _get_chebyshev_config(self, old_geo) -> dict:
        """Creates a config dict for a Chebyshev surface."""
        config = self._get_polynomial_config(old_geo)
        config["norm_x"] = getattr(old_geo, "norm_x", 1.0)
        config["norm_y"] = getattr(old_geo, "norm_y", 1.0)
        return config

    def _get_zernike_config(self, old_geo) -> dict:
        """Creates a config dict for a Zernike surface."""
        config = self._get_polynomial_config(old_geo)
        config["norm_radius"] = getattr(old_geo, "norm_radius", 1.0)
        return config

    def set_surface_type(self, row: int, new_type: str):
        """Sets the geometry type for a surface using a dispatcher."""
        if not (0 < row < self.get_surface_count() - 1):
            return

        new_type = new_type.lower().strip()
        if new_type not in self.AVAILABLE_SURFACE_TYPES:
            return

        old_state = self._capture_optic_state()
        try:
            surface = self._optic.surface_group.surfaces[row]
            old_geo = surface.geometry

            # Base config preserves radius and conic
            config_kwargs = {
                "radius": getattr(old_geo, "radius", float("inf")),
                "conic": getattr(old_geo, "k", 0.0),
            }

            # Map surface types to their specific configuration functions
            config_handlers = {
                "biconic": self._get_biconic_config,
                "toroidal": self._get_toroidal_config,
                "even_asphere": self._get_polynomial_config,
                "odd_asphere": self._get_polynomial_config,
                "polynomial": self._get_polynomial_config,
                "chebyshev": self._get_chebyshev_config,
                "zernike": self._get_zernike_config,
            }

            handler = config_handlers.get(new_type)
            if handler:
                config_kwargs = handler(old_geo)

            # Create the new geometry
            new_geo = GeometryFactory.create(
                surface_type=new_type, cs=old_geo.cs, **config_kwargs
            )

            surface.geometry = new_geo
            surface.surface_type = new_type
            if new_type == "paraxial" and not hasattr(surface, "f"):
                surface.f = float("inf")  # Set a default focal length

            self._optic.update()
            self._undo_redo_manager.add_state(old_state)
            self.set_modified(True)
            self.opticChanged.emit()

        except Exception as e:
            print(f"Error setting surface type: {e}")
            self._restore_optic_state(old_state)

    def _set_comment_data(self, surface, value):
        surface.comment = str(value)

    def _set_material_data(self, surface, value):
        new_material_name = str(value).strip().lower()
        if new_material_name == "mirror":
            surface.is_reflective = True
            surface.material_post = surface.material_pre or IdealMaterial(n=1.0)
        else:
            surface.is_reflective = False
            if new_material_name == "air":
                surface.material_post = IdealMaterial(n=1.0)
            else:
                try:
                    # Try to parse as an ideal material with a given index n
                    n_val = float(new_material_name)
                    surface.material_post = IdealMaterial(n=n_val)
                except ValueError:
                    # Otherwise, treat it as a catalog material name
                    surface.material_post = OptilandMaterial(name=str(value).strip())

        # Propagate the material to the next surface's material_pre
        row = self._optic.surface_group.surfaces.index(surface)
        if row + 1 < self.get_surface_count():
            next_surface = self._optic.surface_group.surfaces[row + 1]
            next_surface.material_pre = surface.material_post

    def _set_thickness_data(self, surface, value):
        row = self._optic.surface_group.surfaces.index(surface)
        if row < self.get_surface_count() - 1:
            self._optic._updater.set_thickness(float(value), row)

    def _set_semi_diameter_data(self, surface, value):
        try:
            # The backend expects diameter, so multiply the semi-diameter by 2
            surface.aperture = configure_aperture(float(value) * 2.0)
        except (ValueError, TypeError):
            surface.aperture = None

    def _set_dynamic_radius_data(self, surface, value):
        """Sets radius-like data which depends on the surface type."""
        val = float(value)
        if surface.surface_type == "paraxial":
            surface.f = val
        elif surface.surface_type == "toroidal":
            surface.geometry.R_yz = be.array(val)
            surface.geometry.c_yz = 1.0 / val if val != 0 else 0.0
        elif surface.surface_type == "biconic":
            surface.geometry.Ry = be.array(val)
            surface.geometry.cy = 1.0 / val if val != 0 else 0.0
        else:
            row = self._optic.surface_group.surfaces.index(surface)
            self._optic._updater.set_radius(val, row)

    def _set_dynamic_conic_data(self, surface, value):
        """Sets conic-like data which depends on the surface type."""
        val = float(value)
        if surface.surface_type == "paraxial":
            return  # N/A, do nothing
        elif surface.surface_type == "toroidal":
            surface.geometry.k_yz = be.array(val)
        elif surface.surface_type == "biconic":
            surface.geometry.ky = be.array(val)
        elif hasattr(surface.geometry, "k"):
            row = self._optic.surface_group.surfaces.index(surface)
            self._optic._updater.set_conic(val, row)

    def set_surface_data(self, row: int, col_idx: int, value_str: str):
        """Sets surface data for a given row and column using a dispatcher."""
        if not (0 <= row < self.get_surface_count()):
            return

        old_state = self._capture_optic_state()
        try:
            surface = self._optic.surface_group.surfaces[row]

            # Map column indices to their corresponding handler functions
            handler_map = {
                self.COL_COMMENT: self._set_comment_data,
                self.COL_MATERIAL: self._set_material_data,
                self.COL_THICKNESS: self._set_thickness_data,
                self.COL_SEMI_DIAMETER: self._set_semi_diameter_data,
                self.COL_RADIUS: self._set_dynamic_radius_data,
                self.COL_CONIC: self._set_dynamic_conic_data,
            }

            handler = handler_map.get(col_idx)
            if handler:
                # For most handlers, the value is the raw string from the UI
                # The helper function itself is responsible for parsing it
                handler(surface, value_str)

            self._optic.update()
            self._undo_redo_manager.add_state(old_state)
            self.set_modified(True)
            self.opticChanged.emit()

        except Exception as e:
            print(
                f"OpticConnector: Error setting data at ({row},{col_idx}) "
                f"to '{value_str}': {e}"
            )
            self._restore_optic_state(old_state)

    def add_surface(self, index: int = -1):
        old_state = self._capture_optic_state()
        num_rows = self.get_surface_count()
        insert_idx = num_rows - 1 if index == -1 or index >= num_rows else index
        if insert_idx <= 0:
            insert_idx = 1

        self._optic.surfaces.add(
            surface_type="standard",
            radius=float("inf"),
            thickness=0.0,
            material="Air",
            comment="New Surface",
            index=insert_idx,
        )
        self._optic.update()
        self._undo_redo_manager.add_state(old_state)
        self.set_modified(True)
        self.opticChanged.emit()

    def remove_surface(self, lde_row_index: int):
        if not (0 < lde_row_index < self.get_surface_count() - 1):
            print("OpticConnector: Cannot remove Object or Image surface.")
            return

        old_state = self._capture_optic_state()
        try:
            self._optic.surface_group.remove_surface(lde_row_index)
            self._optic.update()
            self._undo_redo_manager.add_state(old_state)
            self.set_modified(True)
            self.opticChanged.emit()
        except Exception:
            self._restore_optic_state(old_state)

    def undo(self):
        if self._undo_redo_manager.can_undo():
            state = self._undo_redo_manager.undo(self._capture_optic_state())
            if state:
                self._restore_optic_state(state)

    def redo(self):
        if self._undo_redo_manager.can_redo():
            state = self._undo_redo_manager.redo(self._capture_optic_state())
            if state:
                self._restore_optic_state(state)

    def _parse_param_value(self, value_str: str):
        """Safely parses a string which could be a list, tuple, or float."""
        if isinstance(value_str, str):
            value_str = value_str.strip()
            if value_str.startswith(("[", "(")):
                return ast.literal_eval(value_str)
        return float(value_str)

    def _update_biconic_geometry(self, surface, params_dict):
        """Updates or creates a biconic geometry for a surface."""
        from optiland.geometries.biconic import BiconicGeometry

        # Get existing values from the main table (Y-direction)
        radius_y = getattr(surface.geometry, "radius", float("inf"))
        conic_y = getattr(surface.geometry, "k", 0.0)

        # Get new values from the properties panel (X-direction)
        radius_x = self._parse_param_value(params_dict.get("Radius X", float("inf")))
        conic_x = self._parse_param_value(params_dict.get("Conic X", 0.0))

        # Create a new BiconicGeometry
        new_geo = BiconicGeometry(
            coordinate_system=surface.geometry.cs,
            radius_x=radius_x,
            radius_y=radius_y,
            conic_x=conic_x,
            conic_y=conic_y,
        )
        surface.geometry = new_geo

    def get_surface_geometry_params(self, row: int) -> dict:
        if not (0 < row < self.get_surface_count() - 1):
            return {}
        surface = self._optic.surface_group.surfaces[row]
        geometry = surface.geometry
        geo_class_name = geometry.__class__.__name__

        # if surface_type is biconic but geometry is Plane,
        # show biconic params anyway
        if surface.surface_type == "biconic" and geo_class_name == "Plane":
            # Return default biconic X-direction parameters that can be edited
            # (Y-direction parameters are already in the main LDE table)
            return {
                "Radius X": float("inf"),
                "Conic X": 0.0,
            }

        params = {}
        if geo_class_name in self.EXTRA_PARAM_MAP:
            for label, attr_name in self.EXTRA_PARAM_MAP[geo_class_name].items():
                if hasattr(geometry, attr_name):
                    params[label] = getattr(geometry, attr_name)
        return params

    def set_surface_geometry_params(self, row: int, params_dict: dict):
        """Sets extra geometry parameters for a surface from the properties box."""
        if not (0 < row < self.get_surface_count() - 1):
            return

        old_state = self._capture_optic_state()
        try:
            surface = self._optic.surface_group.surfaces[row]
            geometry = surface.geometry
            geo_class_name = geometry.__class__.__name__

            # Special case: converting a plane to a biconic
            if surface.surface_type == "biconic" and geo_class_name == "Plane":
                self._update_biconic_geometry(surface, params_dict)

            # General case for all other complex surfaces
            elif geo_class_name in self.EXTRA_PARAM_MAP:
                for label, value_str in params_dict.items():
                    attr_name = self.EXTRA_PARAM_MAP[geo_class_name].get(label)
                    if attr_name and hasattr(geometry, attr_name):
                        try:
                            new_value = self._parse_param_value(value_str)
                            setattr(geometry, attr_name, be.array(new_value))

                            # Handle dependent properties
                            if (
                                geo_class_name == "BiconicGeometry"
                                and attr_name == "Rx"
                            ) or (
                                geo_class_name == "ToroidalGeometry"
                                and attr_name == "R_rot"
                            ):
                                geometry.radius = be.array(new_value)

                        except (ValueError, SyntaxError):
                            continue  # Ignore invalid values

            self._optic.update()
            self._undo_redo_manager.add_state(old_state)
            self.set_modified(True)
            self.opticChanged.emit()

        except Exception as e:
            print(f"Error setting geometry params: {e}")
            self._restore_optic_state(old_state)

    def get_field_options(self):
        if not self._optic or self._optic.fields.num_fields == 0:
            return [("all", "'all'")]
        options = [("all", "'all'")]
        field_coords = self._optic.fields.get_field_coords()
        for i, field_tuple in enumerate(field_coords):
            hx, hy = field_tuple[0], field_tuple[1]
            display_name = f"Field {i + 1}: ({hx:.3f}, {hy:.3f})"
            value_str = f"[({hx}, {hy})]"
            options.append((display_name, value_str))
        return options

    def get_wavelength_options(self):
        if not self._optic or self._optic.wavelengths.num_wavelengths == 0:
            return [("all", "'all'"), ("primary", "'primary'")]
        options = [("all", "'all'"), ("primary", "'primary'")]
        wavelength_values = self._optic.wavelengths.get_wavelengths()
        for wl_value in wavelength_values:
            display_name = f"{wl_value:.4f} µm"
            value_str = f"[{wl_value}]"
            options.append((display_name, value_str))
        return options
        return options
