"""Optiland Utilities Module

This module provides utility functions for optical system analysis, including
the calculation of the working F-number (F/#) of an optical system.

Kramer Harrison, 2025
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, NamedTuple

import optiland.backend as be

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


def machine_eps(value) -> float:
    """Machine epsilon of ``value``'s floating dtype.

    Backend-agnostic: uses ``torch.finfo`` for torch tensors and numpy's
    ``finfo`` otherwise. Numerical thresholds built from this scale correctly
    in float32, where a hardcoded float64-sized constant is below round-off
    and therefore never triggers.

    Args:
        value: A backend array/tensor, or anything without a ``dtype``
            attribute (treated as the default Python float precision).

    Returns:
        float: The machine epsilon for the corresponding dtype.
    """
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return float(be.finfo(float).eps)
    if torch is not None and isinstance(dtype, torch.dtype):
        return float(torch.finfo(dtype).eps)
    return float(be.finfo(dtype).eps)


# Documented multiplier ``C`` on the machine epsilon for the reciprocal
# Frobenius-condition singularity test of a per-element 2x2 matrix. A matrix
# is classified unusable when ``rho_F = |det(A)| / ||A||_F^2 <= C * eps``,
# where ``A`` is the matrix normalized by its largest-magnitude entry. The
# test is scale invariant: a global unit or magnification rescaling cannot
# change the classification. C = 64 (2^6) gives headroom for the few tens of
# rounding operations that accumulate in the normalized determinant and norm
# before the comparison, while staying many orders of magnitude below any
# physically meaningful conditioning.
RCOND_EPS_MULTIPLIER = 64.0


@dataclass
class Jacobian2x2Condition:
    """Scale-invariant conditioning state of per-element ``2x2`` matrices.

    Attributes:
        a, b, c, d: Entries of the normalized matrix ``A = J / s``.
        scale: Per-element normalization ``s = max(|J11|,|J12|,|J21|,|J22|)``,
            replaced by 1 where ``s == 0`` (those elements are singular).
        det: Determinant of the normalized matrix ``A`` (sign preserved).
        rcond: Reciprocal Frobenius-condition estimate
            ``|det(A)| / ||A||_F^2``.
        singular: Per-element mask -- non-finite entries, zero scale, or
            reciprocal condition at round-off level.
    """

    a: Any
    b: Any
    c: Any
    d: Any
    scale: Any
    det: Any
    rcond: Any
    singular: Any


def jacobian_2x2_condition(J11, J12, J21, J22) -> Jacobian2x2Condition:
    """Classify per-element ``2x2`` matrices with a scale-invariant test.

    Normalizes each element's matrix by its largest-magnitude entry and
    computes the reciprocal Frobenius-condition estimate

    ``rho_F = |det(A)| / (a^2 + b^2 + c^2 + d^2)``,

    which for a ``2x2`` matrix bounds ``1 / cond_F(J)``. The classification is
    invariant under a global scaling of the matrix, so a well-conditioned
    but small-magnitude matrix (e.g. ``1e-3 * I`` in float32) is never
    misclassified as singular. This is the single authority for 2x2
    conditioning across Optiland (field solves and ray aiming); do not
    duplicate determinant logic elsewhere.
    """
    finite = be.logical_and(
        be.logical_and(be.isfinite(J11), be.isfinite(J12)),
        be.logical_and(be.isfinite(J21), be.isfinite(J22)),
    )

    magnitude = be.maximum(
        be.maximum(be.abs(J11), be.abs(J12)),
        be.maximum(be.abs(J21), be.abs(J22)),
    )
    zero_scale = magnitude == 0.0
    # Safe placeholder scale for zero-scale entries only; they are singular.
    scale = be.where(zero_scale, be.ones_like(magnitude), magnitude)

    a = J11 / scale
    b = J12 / scale
    c = J21 / scale
    d = J22 / scale
    det = a * d - b * c

    frob_sq = a * a + b * b + c * c + d * d
    safe_frob_sq = be.where(frob_sq > 0.0, frob_sq, be.ones_like(frob_sq))
    rcond = be.abs(det) / safe_frob_sq

    singular = be.logical_or(
        be.logical_or(be.logical_not(finite), zero_scale),
        be.logical_or(
            be.logical_not(be.isfinite(det)),
            rcond <= RCOND_EPS_MULTIPLIER * machine_eps(det),
        ),
    )
    return Jacobian2x2Condition(
        a=a, b=b, c=c, d=d, scale=scale, det=det, rcond=rcond, singular=singular
    )


@dataclass(frozen=True)
class LinearSolve2x2Result:
    """Outcome of a batched, conditioning-aware 2x2 linear solve.

    Attributes:
        x1, x2: Per-element solution of ``J @ [x1, x2] = [r1, r2]``; exact
            zeros where ``valid`` is False (never a clipped step).
        rcond: Scale-invariant reciprocal Frobenius-condition estimate of
            each element's matrix.
        valid: Per-element mask; False where the matrix is singular or
            ill-conditioned at round-off level and no solution was formed.
    """

    x1: Any
    x2: Any
    rcond: Any
    valid: Any


def solve_2x2(J11, J12, J21, J22, r1, r2) -> LinearSolve2x2Result:
    """Solve per-element ``J @ x = r`` via the normalized system.

    Each element's matrix is normalized by its largest-magnitude entry
    (see :func:`jacobian_2x2_condition`), which makes both the singularity
    classification and the solve invariant under global scaling and avoids
    raw-determinant overflow/underflow. The determinant's sign is preserved
    for every valid element -- it is never clamped to an arbitrary positive
    value, so a Newton step formed from the solution can never be silently
    reversed. Singular elements receive an exact zero solution and
    ``valid = False``, leaving the caller to refresh, substitute or hold.

    Args:
        J11, J12, J21, J22: Per-element matrix entries.
        r1, r2: Per-element right-hand sides.

    Returns:
        The :class:`LinearSolve2x2Result` with solutions, reciprocal
        condition estimates and the validity mask.
    """
    cond = jacobian_2x2_condition(J11, J12, J21, J22)

    # Placeholder determinant avoids division by ~0; singular elements are
    # then explicitly zeroed rather than receiving a clipped step.
    safe_det = be.where(cond.singular, be.ones_like(cond.det), cond.det)

    r1_s = r1 / cond.scale
    r2_s = r2 / cond.scale
    x1 = (cond.d * r1_s - cond.b * r2_s) / safe_det
    x2 = (-cond.c * r1_s + cond.a * r2_s) / safe_det

    zero = be.zeros_like(x1)
    x1 = be.where(cond.singular, zero, x1)
    x2 = be.where(cond.singular, zero, x2)
    return LinearSolve2x2Result(
        x1=x1, x2=x2, rcond=cond.rcond, valid=be.logical_not(cond.singular)
    )


class FieldPoint(NamedTuple):
    """A resolved field coordinate with its associated weight.

    Attributes:
        coord: (x, y) field coordinate in the field coordinate system.
        weight: Non-negative relative importance scalar. Defaults to 1.0 for
            user-supplied raw coordinates.
    """

    coord: tuple[float, float]
    weight: float


class WavelengthPoint(NamedTuple):
    """A resolved wavelength value with its associated weight.

    Attributes:
        value: Wavelength in micrometers.
        weight: Non-negative relative importance scalar. Defaults to 1.0 for
            user-supplied raw values.
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
        2. Determine the image-space refractive index 'n' at the given
           wavelength.
        3. Trace four marginal rays (top, bottom, left, right) at the pupil
           edges, as well as the chief ray.
        4. Compute the angle between each marginal ray and the chief ray.
        5. Calculate the average of the squared numerical apertures from all
           traced marginal rays.
        6. Compute the working F-number as 1 / (2 * be.sqrt(average_NA_squared)).
        7. Cap the calculated F/# at 10,000 if it exceeds this value.

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
