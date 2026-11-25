"""Thermal copy of an optical system.

This module builds a deep copy of an optic that is geometrically and optically
at another temperature. The copy can be traced and analysed as usual, without
touching the nominal design. The nominal optic is never modified.

Geometry model:
    Every surface's geometry and physical aperture scale isotropically by
    ``1 + alpha * dT``, where ``alpha`` is the coefficient of thermal expansion
    (CTE) of the surrounding material: the glass after the surface, else the
    glass before it, else the mount. Solid thicknesses follow the same rule. A
    gap is held at its edges by the mount, so its *edge* thickness expands by
    the mount's CTE; its centre thickness then follows from the new sags at
    the contact heights.

Variables:
    Every solid in the copy is wrapped in :class:`MaterialAtConditions`, which
    evaluates the glass at the requested temperature and pressure.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.materials import MaterialAtConditions
from optiland.paraxial_path import require_global_z_geometry

if TYPE_CHECKING:
    from optiland.geometries.base import BaseGeometry
    from optiland.optic import Optic
    from optiland.surfaces.standard_surface import Surface

__all__ = ["optic_at_temperature"]

_ONE = 1.0


def _to_float(value) -> float:
    """Convert a backend scalar to a Python float.

    Args:
        value: A Python scalar, NumPy scalar or backend array.

    Returns:
        The value as a Python float.
    """
    return float(be.to_numpy(value).reshape(-1)[0])


def _surface_sag(geometry: BaseGeometry, height: float) -> float:
    """Return the sag of ``geometry`` at ``y = height`` as a Python float.

    Args:
        geometry: The surface geometry.
        height: The height at which the sag is evaluated.

    Returns:
        The sag in millimeters.
    """
    return _to_float(geometry.sag(x=0, y=height))


def _is_gap(material, wavelength: float) -> bool:
    """Return True if ``material`` is a gap, i.e. has an index of exactly 1.

    Args:
        material: The material to test, or ``None`` for a missing medium.
        wavelength: The wavelength, in micrometers, at which the index is
            evaluated.

    Returns:
        True for air and vacuum, False for a solid.
    """
    if material is None:
        return True
    return _to_float(be.asarray(material.n(wavelength))) == _ONE


def _medium_cte(
    surface: Surface,
    index: int,
    *,
    wavelength: float,
    mount_cte: float,
    medium_cte: dict[int, float] | None,
) -> float:
    """Return the CTE of the medium after ``surface``.

    The order of precedence is: an explicit ``medium_cte`` entry, the glass's
    own thermal expansion, and finally the mount CTE for a gap.

    Args:
        surface: The surface whose following medium is queried.
        index: The index of ``surface`` in the optic.
        wavelength: The wavelength, in micrometers, at which the medium is
            classified as a gap or a solid.
        mount_cte: The CTE used for gaps, in 1/K.
        medium_cte: Explicit CTE overrides, keyed by surface index.

    Returns:
        The CTE in 1/K.

    Raises:
        ValueError: If the medium is a solid without CTE data and no
            ``medium_cte`` entry covers it.
    """
    if medium_cte is not None and index in medium_cte:
        return float(medium_cte[index])
    material = surface.material_post
    if _is_gap(material, wavelength):
        return float(mount_cte)
    cte = material.thermal_expansion
    if cte is None:
        raise ValueError(
            f"The material after surface {index} "
            f"({material.display_name}) has no coefficient of thermal "
            "expansion. Pass a value through medium_cte, for example "
            f"medium_cte={{{index}: value}}."
        )
    return float(cte)


def _medium_ctes(
    surfaces,
    *,
    wavelength: float,
    mount_cte: float,
    medium_cte: dict[int, float] | None,
) -> list[float]:
    """Return the CTE of the medium after every surface.

    Args:
        surfaces: The surface sequence.
        wavelength: The wavelength, in micrometers.
        mount_cte: The CTE used for gaps, in 1/K.
        medium_cte: Explicit CTE overrides, keyed by surface index.

    Returns:
        One CTE per surface. The last surface has no following medium; its
        entry is harmless but never used.
    """
    return [
        _medium_cte(
            surface,
            index,
            wavelength=wavelength,
            mount_cte=mount_cte,
            medium_cte=medium_cte,
        )
        for index, surface in enumerate(surfaces)
    ]


def _surface_ctes(
    surfaces,
    *,
    wavelength: float,
    mount_cte: float,
    medium_cte: dict[int, float] | None,
    surface_cte: dict[int, float] | None,
) -> list[float]:
    """Return the CTE that governs each surface's geometry.

    The order of precedence is: an explicit ``surface_cte`` entry, the CTE of
    the solid after the surface (the second glass of a cemented interface),
    the CTE of the solid before the surface (back surfaces and mirrors), and
    finally the mount CTE.

    Args:
        surfaces: The surface sequence.
        wavelength: The wavelength, in micrometers.
        mount_cte: The CTE used for gaps, in 1/K.
        medium_cte: Explicit medium CTE overrides, keyed by surface index.
        surface_cte: Explicit surface geometry CTE overrides, keyed by
            surface index.

    Returns:
        One CTE per surface, in 1/K.
    """
    ctes: list[float] = []
    for index, surface in enumerate(surfaces):
        if surface_cte is not None and index in surface_cte:
            ctes.append(float(surface_cte[index]))
            continue
        after = surface.material_post
        if after is not None and not _is_gap(after, wavelength):
            ctes.append(
                _medium_cte(
                    surface,
                    index,
                    wavelength=wavelength,
                    mount_cte=mount_cte,
                    medium_cte=medium_cte,
                )
            )
            continue
        before = surface.material_pre
        if before is not None and not _is_gap(before, wavelength):
            ctes.append(
                _medium_cte(
                    surfaces[index - 1],
                    index - 1,
                    wavelength=wavelength,
                    mount_cte=mount_cte,
                    medium_cte=medium_cte,
                )
            )
            continue
        ctes.append(float(mount_cte))
    return ctes


def _scaled_geometry(geometry: BaseGeometry, factor: float) -> BaseGeometry:
    """Return a scaled deep copy of ``geometry``, leaving the original intact.

    Args:
        geometry: The geometry to copy and scale.
        factor: The isotropic scale factor.

    Returns:
        The scaled copy.
    """
    scaled = copy.deepcopy(geometry)
    scaled.scale(factor)
    return scaled


def _thermal_scale_factors(
    surfaces,
    delta_t: float,
    *,
    wavelength: float,
    mount_cte: float = 0.0,
    medium_cte: dict[int, float] | None = None,
    surface_cte: dict[int, float] | None = None,
) -> tuple[list[float], list[float | None]]:
    """Compute the thermal geometry factors and the new thicknesses.

    This function has no side effects: the given surfaces are only read.

    Args:
        surfaces: The surface sequence. Every surface must have a
            ``semi_aperture`` and the geometries must be the nominal ones.
        delta_t: The temperature change from the reference, in K.
        wavelength: The wavelength, in micrometers, at which a medium is
            classified as a gap (index exactly 1) or a solid.
        mount_cte: The CTE of the mount, used for gaps, mirrors in air and any
            medium without a more specific CTE, in 1/K.
        medium_cte: Explicit CTE overrides for the medium after a surface,
            keyed by surface index.
        surface_cte: Explicit CTE overrides for a surface's geometry, keyed by
            surface index.

    Returns:
        A tuple ``(geometry_factors, new_thicknesses)``. ``geometry_factors[i]``
        is the scale factor ``1 + alpha_i * delta_t`` of surface ``i``.
        ``new_thicknesses[i]`` is the new centre thickness after surface ``i``,
        or ``None`` for the last surface, which has no following space.
        Infinite thicknesses stay infinite.

    Raises:
        ValueError: If a solid has no CTE data and no ``medium_cte`` entry
            covers it.
    """
    surfaces = list(surfaces)
    num_surfaces = len(surfaces)
    medium_ctes = _medium_ctes(
        surfaces,
        wavelength=wavelength,
        mount_cte=mount_cte,
        medium_cte=medium_cte,
    )
    surface_ctes = _surface_ctes(
        surfaces,
        wavelength=wavelength,
        mount_cte=mount_cte,
        medium_cte=medium_cte,
        surface_cte=surface_cte,
    )

    geometry_factors = [1.0 + cte * delta_t for cte in surface_ctes]

    heights: list[float] = []
    for surface in surfaces:
        if surface.semi_aperture is None:
            raise ValueError(
                "Every surface needs a semi_aperture before its thermal scale "
                "factors are computed. Run optic.updater.update_paraxial() "
                "first."
            )
        heights.append(_to_float(surface.semi_aperture))

    scaled: list[BaseGeometry] = []
    for index, surface in enumerate(surfaces):
        if geometry_factors[index] == 1.0:
            scaled.append(copy.deepcopy(surface.geometry))
        else:
            scaled.append(_scaled_geometry(surface.geometry, geometry_factors[index]))

    new_thicknesses: list[float | None] = []
    for index in range(num_surfaces):
        surface = surfaces[index]
        if index == num_surfaces - 1:
            new_thicknesses.append(None)
            continue

        thickness = _to_float(surface.thickness)
        alpha = medium_ctes[index]
        if not _is_gap(surface.material_post, wavelength):
            new_thicknesses.append(thickness * (1.0 + alpha * delta_t))
            continue
        if math.isinf(thickness):
            new_thicknesses.append(thickness)
            continue

        gap_factor = 1.0 + alpha * delta_t
        next_index = index + 1

        # Edge thickness at the nominal contact height.
        edge = (
            thickness
            + _surface_sag(surfaces[next_index].geometry, heights[next_index])
            - _surface_sag(surface.geometry, heights[index])
        )

        # The mount moves the contact point inward but never past the lens
        # edge, where the lens's own expansion may be slower or faster.
        contact_height = min(
            heights[index] * gap_factor, heights[index] * geometry_factors[index]
        )
        next_contact_height = min(
            heights[next_index] * gap_factor,
            heights[next_index] * geometry_factors[next_index],
        )

        new_thicknesses.append(
            edge * gap_factor
            - _surface_sag(scaled[next_index], next_contact_height)
            + _surface_sag(scaled[index], contact_height)
        )

    return geometry_factors, new_thicknesses


def _copy_optic(optic: Optic) -> Optic:
    """Return an independent copy of ``optic``.

    ``copy.deepcopy`` is the usual path. On the PyTorch backend a paraxial
    update leaves non-leaf tensors, which Torch cannot deepcopy, so the copy
    is made through the public ``to_dict`` / ``from_dict`` serializers
    instead. Those serializers drop per-surface state (``semi_aperture`` and
    ``mirror_thickness``), which is copied over by hand so both paths give
    identical results. The input optic is not modified in either case.

    Args:
        optic: The optic to copy.

    Returns:
        An independent copy of ``optic``.
    """
    try:
        return copy.deepcopy(optic)
    except RuntimeError:
        # Torch cannot deepcopy tensors that are not graph leaves. Only the
        # exception type is relied on: the error text is not a stable API.
        copied = type(optic).from_dict(optic.to_dict())
        _copy_surface_state(optic, copied)
        return copied


def _copy_surface_state(source: Optic, copied: Optic) -> None:
    """Carry per-surface state that ``Surface.to_dict`` does not serialise.

    ``semi_aperture`` (set by the user or by ``update_paraxial``) and
    ``mirror_thickness`` are skipped by the serializers, so they are copied
    from the source surfaces in order. Both are scalars that downstream code
    only replaces, never mutates in place, so the copy may share them with
    the source.

    Args:
        source: The optic copied from.
        copied: The optic rebuilt from ``source.to_dict()``.
    """
    for source_surface, new_surface in zip(
        source.surfaces, copied.surfaces, strict=True
    ):
        new_surface.semi_aperture = source_surface.semi_aperture
        new_surface.mirror_thickness = source_surface.mirror_thickness


def optic_at_temperature(
    optic: Optic,
    temperature: float,
    *,
    pressure: float = 1.0,
    reference_temperature: float = 20.0,
    mount_cte: float = 0.0,
    medium_cte: dict[int, float] | None = None,
    surface_cte: dict[int, float] | None = None,
) -> Optic:
    """Return a deep copy of ``optic`` brought to ``temperature``.

    The input optic is never modified. In the copy:

    - Every surface geometry and physical aperture scales by ``1 + alpha dT``,
      with ``alpha`` the CTE of the surface's geometry: ``surface_cte[i]`` if
      given, else the solid after the surface (the second glass at a cemented
      interface), else the solid before it (back surfaces and mirrors), else
      ``mount_cte``.
    - A solid thickness after surface ``i`` scales by ``1 + alpha dT`` with
      the CTE of that solid.
    - A gap after surface ``i`` is held at its edges by the mount, so its edge
      thickness scales by ``1 + mount_cte dT`` and its centre thickness follows
      from the new sags. The centre gap therefore changes even with
      ``mount_cte = 0`` when the radii change.
    - Every solid is wrapped in :class:`MaterialAtConditions`, so it is
      evaluated at ``temperature`` and ``pressure``. Gaps keep their index of
      one.

    The system aperture, fields and wavelengths are unchanged. Folded systems
    are out of scope.

    Note:
        The nominal optic evaluates each glass at its catalog reference
        temperature (20 °C for Schott, 25 °C for some other vendors), while the
        copy evaluates every glass at ``temperature``. To isolate the effect of
        the temperature change, compare against
        ``optic_at_temperature(optic, reference_temperature)`` rather than
        against the nominal optic.

    Args:
        optic: The optic to copy.
        temperature: The temperature of the copy, in degrees Celsius.
        pressure: The ambient pressure of the copy, in atmospheres.
        reference_temperature: The temperature the nominal optic is designed
            at, in degrees Celsius. The change is
            ``temperature - reference_temperature``.
        mount_cte: The CTE of the mount, in 1/K, used for every air space and
            for any material without a more specific CTE.
        medium_cte: CTE overrides for the medium after a surface, keyed by
            surface index, in 1/K.
        surface_cte: CTE overrides for a surface's geometry, keyed by surface
            index, in 1/K.

    Returns:
        The deep copy of the optic at the requested temperature.

    Raises:
        UnsupportedParaxialGeometryError: If the beam path is folded off the
            global +z axis, as for ``OpticUpdater.scale_system``.
        ValueError: If a solid has no CTE data and no ``medium_cte`` entry
            covers it.

    Example:
        >>> from optiland.samples.objectives import CookeTriplet
        >>> from optiland.environment import optic_at_temperature
        >>> hot = optic_at_temperature(CookeTriplet(), 60.0, mount_cte=23.6e-6)
        >>> hot.paraxial.f2()
    """
    require_global_z_geometry(optic.surfaces, "optic_at_temperature")

    thermal = _copy_optic(optic)
    updater = thermal.updater
    surfaces = list(thermal.surfaces)

    if any(surface.semi_aperture is None for surface in surfaces):
        updater.update_paraxial()

    delta_t = float(temperature) - float(reference_temperature)
    geometry_factors, new_thicknesses = _thermal_scale_factors(
        surfaces,
        delta_t,
        wavelength=float(thermal.primary_wavelength),
        mount_cte=mount_cte,
        medium_cte=medium_cte,
        surface_cte=surface_cte,
    )

    for index, surface in enumerate(surfaces):
        surface.geometry.scale(geometry_factors[index])
        if surface.mirror_thickness is not None:
            surface.mirror_thickness *= geometry_factors[index]
        if surface.aperture is not None:
            surface.aperture.scale(geometry_factors[index])

    for index, thickness in enumerate(new_thicknesses):
        if thickness is None or math.isinf(thickness):
            continue
        updater.set_thickness(thickness, index)

    wavelength = float(thermal.primary_wavelength)
    for index, surface in enumerate(surfaces):
        material = surface.material_post
        if material is not None and not _is_gap(material, wavelength):
            updater.set_material(
                MaterialAtConditions(material, temperature, pressure), index
            )

    return thermal
