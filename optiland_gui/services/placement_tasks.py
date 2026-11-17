"""Coordinate conversion and owned surface-placement preparation."""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.paraxial_path import (
    UnsupportedParaxialGeometryError,
    require_global_z_geometry,
)
from optiland_gui.services.job_records import check_cancelled
from optiland_gui.services.prepared_optic import PreparedOptic

COMPONENTS = ("x", "y", "z", "rx", "ry", "rz")


def _euler(matrix):
    # At gimbal lock SciPy returns an equivalent, nonunique Euler triplet.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
        return Rotation.from_matrix(matrix).as_euler("xyz")


def placement_values(surface, frame):
    """Return absolute components in the chosen frame; angles are degrees."""
    if frame not in ("reference", "global"):
        raise ValueError("Choose Global or Reference frame.")
    cs = surface.geometry.cs
    values = np.array([float(getattr(cs, key)) for key in COMPONENTS])
    if frame == "global" and cs.reference_cs is not None:
        translation, rotation = cs.get_effective_transform()
        values[:3] = be.to_numpy(translation)
        values[3:] = _euler(be.to_numpy(rotation))
    values[3:] = np.rad2deg(values[3:])
    return dict(zip(COMPONENTS, values.tolist(), strict=True))


def validate_changes(changes):
    """Validate a complete draft before touching any coordinate component."""
    if set(changes) - set(COMPONENTS):
        raise ValueError("Unknown placement component.")
    parsed = {}
    for key, text in changes.items():
        try:
            value = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key.upper()} must be a finite number.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key.upper()} must be a finite number.")
        parsed[key] = value
    return parsed


def apply_placement(optic, row, frame, changes):
    """Edit only this surface; retain its reference frame and unrelated surfaces."""
    if not 0 <= row < len(optic.surfaces):
        raise ValueError("The surface is no longer available.")
    surface = optic.surfaces[row]
    current = placement_values(surface, frame)
    changes = validate_changes(changes)
    changes = {key: value for key, value in changes.items() if value != current[key]}
    if not changes:
        return False
    desired = current | changes
    if not all(math.isfinite(value) for value in desired.values()):
        raise ValueError("Enter a finite position for this surface, including Z.")
    old = surface.geometry.cs
    raw = {key: getattr(old, key) for key in COMPONENTS}
    if frame == "reference" or old.reference_cs is None:
        for key, value in changes.items():
            raw[key] = math.radians(value) if key.startswith("r") else value
    else:
        origin, rotation = old.reference_cs.get_effective_transform()
        origin, rotation = be.to_numpy(origin), be.to_numpy(rotation)
        if any(key in changes for key in COMPONENTS[:3]):
            xyz = rotation.T @ (np.array([desired[k] for k in COMPONENTS[:3]]) - origin)
            raw.update(zip(COMPONENTS[:3], xyz, strict=True))
        if any(key in changes for key in COMPONENTS[3:]):
            global_rotation = Rotation.from_euler(
                "xyz", [desired[k] for k in COMPONENTS[3:]], degrees=True
            ).as_matrix()
            raw.update(
                zip(COMPONENTS[3:], _euler(rotation.T @ global_rotation), strict=True)
            )
    if not all(math.isfinite(float(value)) for value in raw.values()):
        raise ValueError("The requested placement exceeds finite coordinate limits.")
    # A different surface can use old as its parent. Do not move that surface.
    surface.geometry.cs = CoordinateSystem(**raw, reference_cs=old.reference_cs)
    sync_axial_thicknesses(optic)
    return True


def sync_axial_thicknesses(optic):
    """Keep cumulative thickness storage consistent for ordinary unreferenced axes."""
    surfaces = tuple(optic.surfaces)
    if any(surface.geometry.cs.reference_cs is not None for surface in surfaces):
        return
    try:
        require_global_z_geometry(optic.surfaces, "Placement thickness synchronization")
    except UnsupportedParaxialGeometryError:
        return
    for first, second in zip(surfaces, surfaces[1:], strict=False):
        first.thickness = second.geometry.cs.z - first.geometry.cs.z


def prepare_placement(snapshot, parameters, progress, cancelled):
    """Run reconstruction, solves and normalization away from Qt's GUI thread."""
    snapshot.backend.apply()
    check_cancelled(cancelled)
    progress("Preparing surface placement")
    optic = snapshot.restore()
    initially_finite = [
        [math.isfinite(float(getattr(surface.geometry.cs, key))) for key in COMPONENTS]
        for surface in optic.surfaces
    ]
    changed = apply_placement(
        optic, parameters["row"], parameters["frame"], parameters["changes"]
    )
    check_cancelled(cancelled)
    if changed:
        progress("Updating solves and pickups")
        optic.updater.update()
    check_cancelled(cancelled)
    for row, surface in enumerate(optic.surfaces):
        for key, was_finite in zip(COMPONENTS, initially_finite[row], strict=True):
            if (was_finite or row == parameters["row"]) and not math.isfinite(
                float(getattr(surface.geometry.cs, key))
            ):
                raise ValueError(
                    f"Solves produced nonfinite {key.upper()} at surface {row}."
                )
    return PreparedOptic.capture(optic)
