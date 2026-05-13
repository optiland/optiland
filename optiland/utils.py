"""Optiland Utilities Module

This module provides utility functions for optical system analysis, including
the calculation of the working F-number (F/#) of an optical system.

Kramer Harrison, 2025
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

import optiland.backend as be


class FieldPoint(NamedTuple):
    """A resolved field coordinate with its associated weight.

    Attributes:
        coord: (x, y) field coordinate in the field coordinate system.
        weight: Non-negative relative importance scalar. Defaults to 1.0 for
            user-supplied raw coordinates. Refer to optiland weight semantics in
            SPEC_weights.md §2.1.
    """

    coord: tuple[float, float]
    weight: float


class WavelengthPoint(NamedTuple):
    """A resolved wavelength value with its associated weight.

    Attributes:
        value: Wavelength in micrometers.
        weight: Non-negative relative importance scalar. Defaults to 1.0 for
            user-supplied raw values. Refer to optiland weight semantics in
            SPEC_weights.md §2.1.
    """

    value: float
    weight: float


def get_working_FNO(optic, field, wavelength):
    """Calculates the working F-number of the optical system for the
    single defined field point and given wavelength.

    Args:
        optic (Optic): The optic object.
        field (tuple): The field at which to calculate the F/#.
        wavelength (float): The wavelength at which to calculate the F/#.

    Algorithm:
        1. Retrieve the defined given wavelength and field coordinates.
        2. Determine the image-space refractive index 'n' at the given wavelength.
        3. Trace four marginal rays (top, bottom, left, right) at the pupil edges,
            as well as the chief ray.
        4. Compute the angle between each marginal ray and the chief ray.
        4. Calculate the average of the squared numerical apertures from all traced
            marginal rays.
        5. Compute the working F-number as 1 / (2 * be.sqrt(average_NA_squared)).
        6. Cap the calculated F/# at 10,000 if it exceeds this value.

    Returns:
        float: The working F-number.
    """
    MAX_FNUM = 10000.0

    Hx, Hy = field

    n = optic.image_surface.material_post.n(wavelength)
    Px = be.array([0, 0, 0, 1, -1])
    Py = be.array([0, 1, -1, 0, 0])

    rays = optic.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wavelength)

    L0, M0, N0 = rays.L[0], rays.M[0], rays.N[0]
    L, M, N = rays.L[1:], rays.M[1:], rays.N[1:]
    dot = L0 * L + M0 * M + N0 * N
    dot = be.clip(dot, -1.0, 1.0)
    angles = be.arccos(dot)

    numerical_apertures_squared = (n * be.sin(angles)) ** 2

    # Exclude geometrically vignetted marginal rays (intensity == 0)
    marginal_intensities = be.to_numpy(rays.i[1:])
    valid_indices = [i for i, v in enumerate(marginal_intensities) if v > 0]

    if valid_indices:
        valid_na_sq = be.stack([numerical_apertures_squared[i] for i in valid_indices])
        avg_NA_squared = be.mean(valid_na_sq)
    else:
        # Degenerate fallback: all marginal rays vignetted (should not occur in
        # a well-formed system).
        avg_NA_squared = be.mean(be.array(numerical_apertures_squared))

    fno = be.inf if avg_NA_squared <= 0 else 1 / (2 * be.sqrt(avg_NA_squared))

    if fno > MAX_FNUM:
        fno = MAX_FNUM

    if be.isnan(fno):
        raise ValueError("Working F/# could not be calculated due to raytrace errors.")

    return fno


def active_fields(resolved: list[FieldPoint]) -> list[FieldPoint]:
    """Return only FieldPoints with weight > 0. Use in weighted contexts.

    Args:
        resolved: A list of FieldPoint named tuples.

    Returns:
        Filtered list containing only items with positive weight.
    """
    return [fp for fp in resolved if fp.weight > 0.0]


def active_wavelengths(resolved: list[WavelengthPoint]) -> list[WavelengthPoint]:
    """Return only WavelengthPoints with weight > 0. Use in weighted contexts.

    Args:
        resolved: A list of WavelengthPoint named tuples.

    Returns:
        Filtered list containing only items with positive weight.
    """
    return [wp for wp in resolved if wp.weight > 0.0]


def weighted_average(values: list[float], weights: list[float]) -> float:
    """Compute a weighted normalized average: Σ(w_i × x_i) / Σ(w_i).

    Args:
        values: Scalar values to average.
        weights: Non-negative weights (must have same length as values).
            Zero-weight items contribute nothing; Σ(w_i) must be > 0.

    Returns:
        Weighted normalized average.

    Raises:
        ValueError: If all weights are zero.
    """
    total_w = sum(weights)
    if total_w == 0.0:
        raise ValueError("Cannot compute weighted average: all weights are zero.")
    return sum(w * v for w, v in zip(weights, values, strict=False)) / total_w


def resolve_wavelengths(optic, wavelengths) -> list[WavelengthPoint]:
    """Resolve wavelength input into a list of WavelengthPoints (value + weight).

    When wavelengths='all', weights come from optic.wavelengths. For 'primary',
    the primary wavelength's weight is used. For user-supplied raw float values
    (list of floats), weight defaults to 1.0.

    Args:
        optic (Optic): The optical system.
        wavelengths: 'all', 'primary', or a list of float wavelength values in µm.

    Returns:
        List of WavelengthPoint named tuples. Each has .value (float, µm) and .weight.

    Raises:
        ValueError: If wavelengths is an invalid string.
        TypeError: If wavelengths is not a string or list.
    """
    if isinstance(wavelengths, str):
        if wavelengths == "all":
            return [
                WavelengthPoint(value=w.value, weight=w.weight)
                for w in optic.wavelengths.wavelengths
            ]
        elif wavelengths == "primary":
            pw = next(w for w in optic.wavelengths.wavelengths if w.is_primary)
            return [WavelengthPoint(value=pw.value, weight=pw.weight)]
        else:
            raise ValueError("Invalid wavelength string. Must be 'all' or 'primary'.")
    elif isinstance(wavelengths, list):
        return [WavelengthPoint(value=float(v), weight=1.0) for v in wavelengths]
    else:
        raise TypeError("Wavelengths must be a string ('all', 'primary') or a list.")


def resolve_fields(optic, fields) -> list[FieldPoint]:
    """Resolve field input into a list of FieldPoints (coord + weight).

    When fields='all', field weights come from optic.fields. For any
    user-supplied raw coordinates (list of tuples, a single tuple, or an
    integer index), weight defaults to 1.0 because there is no associated
    Field object to look up the weight from.

    Args:
        optic (Optic): The optical system.
        fields: 'all', a list of (x, y) tuples, a single (x, y) tuple, or an
            integer index into optic.fields.

    Returns:
        List of FieldPoint named tuples. Each has .coord (x, y) and .weight.

    Raises:
        ValueError: If fields is an invalid string.
        TypeError: If fields is not one of the supported types.
    """
    if isinstance(fields, str):
        if fields == "all":
            coords = optic.fields.get_field_coords()
            weights_list = optic.fields.weights
            return [
                FieldPoint(coord=c, weight=w)
                for c, w in zip(coords, weights_list, strict=False)
            ]
        else:
            raise ValueError("Invalid field string. Must be 'all'.")
    elif isinstance(fields, list):
        return [FieldPoint(coord=c, weight=1.0) for c in fields]
    elif isinstance(fields, tuple):
        return [FieldPoint(coord=fields, weight=1.0)]
    elif isinstance(fields, int):
        coords = optic.fields.get_field_coords()
        return [FieldPoint(coord=coords[fields], weight=1.0)]
    else:
        raise TypeError("Fields must be a string ('all'), a list, a tuple, or an int.")


def resolve_wavelength(optic, wavelength):
    """Resolves a single wavelength input into a float value.

    Args:
        optic (Optic): The optic object.
        wavelength (str or float or int): The wavelength to resolve.
            Can be 'primary' or a numerical value.

    Returns:
        float: A single wavelength value.
    """
    if isinstance(wavelength, str):
        if wavelength == "primary":
            return optic.primary_wavelength
        else:
            raise ValueError(
                "Invalid wavelength string. For a single wavelength, it must be "
                "'primary'."
            )
    elif isinstance(wavelength, int | float):
        return float(wavelength)
    elif hasattr(wavelength, "item"):
        return float(wavelength.item())
    else:
        raise TypeError("Wavelength must be a string ('primary') or a number.")


def get_attr_by_path(obj: Any, path: str) -> Any:
    """Retrieve an attribute of an object using a dot-separated path.
    Supports list indexing, e.g., 'surfaces[1].geometry.radius'.

    Args:
        obj: The object to retrieve the attribute from.
        path: The dot-separated path to the attribute.

    Returns:
        The value of the attribute.
    """

    def _get_item(current_obj, key):
        # Check for list indexing: name[index]
        match = re.match(r"(\w+)\[(\d+)\]", key)
        if match:
            attr_name, index = match.groups()
            current_obj = getattr(current_obj, attr_name)
            return current_obj[int(index)]
        else:
            return getattr(current_obj, key)

    parts = path.split(".")
    for part in parts:
        obj = _get_item(obj, part)
    return obj


def set_attr_by_path(obj: Any, path: str, value: Any) -> None:
    """Set an attribute of an object using a dot-separated path.
    Supports list indexing, e.g., 'surfaces[1].geometry.radius'.

    Args:
        obj: The object to set the attribute on.
        path: The dot-separated path to the attribute.
        value: The value to set.
    """

    def _get_item_or_list(current_obj, key):
        # Helper to traverse, but stop before setting the final attribute
        # If key is name[index], we get the list item.
        match = re.match(r"(\w+)\[(\d+)\]", key)
        if match:
            attr_name, index = match.groups()
            container = getattr(current_obj, attr_name)
            return container[int(index)]
        else:
            return getattr(current_obj, key)

    parts = path.split(".")
    final_attr = parts[-1]
    parent_path = parts[:-1]

    # Navigate to the parent object
    current_obj = obj
    for part in parent_path:
        current_obj = _get_item_or_list(current_obj, part)

    # Set the value on the final attribute
    # Note: final_attr usually shouldn't have [index] because we set attributes,
    # but if it does (e.g. setting an item in a list directly), handle it.
    match = re.match(r"(\w+)\[(\d+)\]", final_attr)
    if match:
        attr_name, index = match.groups()
        container = getattr(current_obj, attr_name)
        container[int(index)] = value
    else:
        setattr(current_obj, final_attr, value)


def globalize_coordinates(surface, x_local, y_local, z_local):
    """Transform local surface coordinates to global coordinates.

    Args:
        surface: The surface whose coordinate system is used for the
            transformation.
        x_local (BEArray): The local x-coordinates.
        y_local (BEArray): The local y-coordinates.
        z_local (BEArray): The local z-coordinates.

    Returns:
        tuple: (x_global, y_global, z_global) as flattened backend arrays.
    """
    eff_translation, eff_rot_mat = surface.geometry.cs.get_effective_transform()

    points_local = be.stack([x_local, y_local, z_local], axis=0)
    if len(be.shape(points_local)) == 1:
        points_local = be.unsqueeze_last(points_local)

    points_global = be.matmul(eff_rot_mat, points_local) + be.reshape(
        eff_translation, (3, 1)
    )

    x_global = be.ravel(points_global[0, :])
    y_global = be.ravel(points_global[1, :])
    z_global = be.ravel(points_global[2, :])

    return x_global, y_global, z_global
