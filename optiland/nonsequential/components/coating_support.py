"""Shared coating/reflectance resolution for NSQ components.

RefractiveComponent (an ``optiland.coatings`` coating on a transmissive
interface) and ReflectiveComponent (a required reflectance on a mirror) both
need to validate that a coating is unpolarized and turn it into a per-ray
array. Centralized here so the two components agree on what is accepted.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.coatings import BaseCoating


def reject_polarized_coating(coating: object, *, surface_name: str) -> None:
    """Raise if ``coating`` is a Jones-matrix (polarized) coating.

    NSQ rays carry no polarization state, so a polarized coating cannot be
    evaluated correctly. Rather than silently falling back to some scalar
    average of its Jones matrix, refuse it outright.

    Args:
        coating: The candidate coating, or None / a plain float / a callable.
        surface_name: Name of the surface the coating is attached to, for
            the error message.

    Raises:
        NotImplementedError: If ``coating`` is a
            ``optiland.coatings.BaseCoatingPolarized`` instance.
    """
    from optiland.coatings import BaseCoatingPolarized  # noqa: PLC0415

    if isinstance(coating, BaseCoatingPolarized):
        raise NotImplementedError(
            f"Surface {surface_name!r} was given a polarized coating "
            f"({type(coating).__name__}); NSQ rays carry no polarization "
            "state, so Jones-matrix coatings cannot be evaluated. Use an "
            "unpolarized coating such as optiland.coatings.SimpleCoating, "
            "a constant reflectance, or a callable(wavelength_um) -> "
            "reflectance instead."
        )


def resolve_reflectance(
    reflectance: float | Callable[[be.ndarray], be.ndarray] | BaseCoating,
    wavelength: be.ndarray,
) -> be.ndarray:
    """Turn a mirror's ``reflectance`` spec into a per-ray array.

    Args:
        reflectance: A constant, a ``callable(wavelength_um) -> reflectance``,
            or an unpolarized ``optiland.coatings.BaseCoating`` (read via its
            ``.reflectance`` attribute, e.g. ``SimpleCoating``).
        wavelength: Per-ray wavelength [µm], shape (N,); used only to build
            the broadcast shape for a constant/coating reflectance.

    Returns:
        Per-ray reflectance, shape (N,).
    """
    from optiland.coatings import BaseCoating  # noqa: PLC0415

    if isinstance(reflectance, BaseCoating):
        return be.ones_like(wavelength) * float(reflectance.reflectance)
    if callable(reflectance):
        return be.ones_like(wavelength) * be.array(reflectance(wavelength))
    return be.ones_like(wavelength) * float(reflectance)
