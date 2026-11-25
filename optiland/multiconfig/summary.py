"""Multi-Configuration Summary Module

Builds the side-by-side table of a multi-configuration system: the
first-order properties of every configuration, followed by every surface or
optic property whose value differs between configurations (the equivalent of
a multi-configuration editor).

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.multiconfig.multi_configuration import MultiConfiguration
    from optiland.optic import Optic

_REL_TOL = 1e-9
_ABS_TOL = 1e-12

# (label suffix, attribute of the surface's coordinate system)
_CS_ATTRS = (
    ("decenter x", "x"),
    ("decenter y", "y"),
    ("tilt x", "rx"),
    ("tilt y", "ry"),
    ("tilt z", "rz"),
)


def _image_height(optic: Optic) -> Any:
    """Paraxial chief-ray height on the image surface (maximum field)."""
    y_chief, _ = optic.paraxial.chief_ray()
    return be.abs(be.ravel(y_chief)[-1])


# Rows shown for every configuration, whether they differ or not.
_FIRST_ORDER_ROWS: tuple[tuple[str, Callable[[Optic], Any]], ...] = (
    ("EFL", lambda optic: optic.paraxial.f2()),
    ("F/#", lambda optic: optic.paraxial.FNO()),
    ("Image height", _image_height),
    ("Total track", lambda optic: optic.total_track),
    ("EPD", lambda optic: optic.paraxial.EPD()),
)


def configuration_label(index: int) -> str:
    """Return the column label used for a configuration.

    Args:
        index: The configuration index.

    Returns:
        The label, e.g. ``"Config 1"``.
    """
    return f"Config {index}"


def _to_float(value: Any) -> float:
    """Convert a backend scalar (or one-element array) to a Python float."""
    return float(np.asarray(be.to_numpy(value)).ravel()[0])


def _safe_float(fn: Callable[[], Any]) -> float:
    """Evaluate ``fn`` and return a float, or NaN if it raises."""
    try:
        return _to_float(fn())
    except Exception:
        return math.nan


def _material_label(surface: Any) -> str:
    """Readable name of the material after a surface."""
    if getattr(surface.interaction_model, "is_reflective", False):
        return "MIRROR"
    return surface.material_post.display_name


def _optic_properties(optic: Optic) -> dict[str, Any]:
    """Collect the comparable properties of one configuration.

    Args:
        optic: The configuration to inspect.

    Returns:
        An ordered mapping from row label to value (float or string).
    """
    props: dict[str, Any] = {
        "Aperture type": type(optic.aperture).__name__,
        "Aperture value": _safe_float(lambda: optic.aperture.value),
        "Field type": type(optic.fields.field_definition).__name__,
    }
    for i, field in enumerate(optic.fields.fields):
        props[f"Field {i} x"] = _safe_float(lambda f=field: f.x)
        props[f"Field {i} y"] = _safe_float(lambda f=field: f.y)
    for i, wavelength in enumerate(optic.wavelengths.wavelengths):
        props[f"Wavelength {i} (µm)"] = _safe_float(lambda w=wavelength: w.value)

    for k, surface in enumerate(optic.surfaces.surfaces):
        prefix = f"Surface {k}"
        geometry = surface.geometry
        radius = _safe_float(lambda g=geometry: g.radius)
        if hasattr(geometry, "radius"):
            props[f"{prefix} radius"] = radius
        # A conic has no effect on a flat surface, so it is only compared on
        # curved ones (a Plane and a flat standard surface compare equal).
        if hasattr(geometry, "k") and math.isfinite(radius):
            props[f"{prefix} conic"] = _safe_float(lambda g=geometry: g.k)
        props[f"{prefix} thickness"] = _safe_float(lambda s=surface: s.thickness)
        props[f"{prefix} material"] = _material_label(surface)
        for suffix, attr in _CS_ATTRS:
            props[f"{prefix} {suffix}"] = _safe_float(
                lambda a=attr, g=geometry: getattr(g.cs, a)
            )
    return props


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two property values; floats within a tight tolerance."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    return a == b


def configuration_summary(multiconfig: MultiConfiguration) -> pd.DataFrame:
    """Build the side-by-side summary table of a multi-configuration system.

    The table has one column per configuration (``"Config 0"``, ...). Its
    first rows are always ``EFL``, ``F/#``, ``Image height`` (paraxial chief
    ray at the image, maximum field), ``Total track`` and ``EPD``. They are
    followed by every property whose value differs between configurations,
    found by comparing values (not pickups): aperture type and value, field
    type and coordinates, wavelengths, and per surface the radius, conic,
    thickness, material, decenters and tilts (the conic only on curved
    surfaces, where it has an effect). Surfaces are indexed as in
    ``optic.surfaces`` (0 is the object surface); fields and wavelengths are
    0-based.

    A quantity that can not be computed for a configuration (e.g. a paraxial
    value of a folded system) is NaN rather than an error. A property that
    exists in only some configurations is NaN in the others. The table reads
    the current state: call ``optic.updater.update()`` on configurations
    whose pickups may be stale.

    Args:
        multiconfig: The multi-configuration system to summarize.

    Returns:
        A DataFrame indexed by readable row labels (index name
        ``"Parameter"``), with float or string cells.
    """
    configs = multiconfig.configurations
    columns = [configuration_label(i) for i in range(len(configs))]

    data: dict[str, list[Any]] = {}
    for label, fn in _FIRST_ORDER_ROWS:
        data[label] = [_safe_float(lambda o=optic, f=fn: f(o)) for optic in configs]

    per_config = [_optic_properties(optic) for optic in configs]
    labels = list(dict.fromkeys(label for props in per_config for label in props))
    for label in labels:
        values = [props.get(label, math.nan) for props in per_config]
        if not all(_values_equal(values[0], v) for v in values[1:]):
            data[label] = values

    frame = pd.DataFrame.from_dict(data, orient="index", columns=columns)
    frame.index.name = "Parameter"
    return frame
