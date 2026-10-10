"""
Provides a phase profile that dispatches to a child profile per wavelength.
"""

from __future__ import annotations

import math
import typing

import numpy as np

from optiland import backend as be
from optiland.phase.base import BasePhaseProfile

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from optiland.surfaces.standard_surface import Surface


def _to_numpy(value: typing.Any) -> np.ndarray:
    """Converts a backend array, or a bare bool/scalar, to a NumPy array."""
    if isinstance(value, bool | np.bool_):
        return np.asarray(value)
    return np.asarray(be.to_numpy(value))


class WavelengthDependentPhaseProfile(BasePhaseProfile):
    """A phase profile that uses a different child profile at each wavelength.

    Each traced wavelength is looked up in a table of child phase profiles and
    the matching child is evaluated. Lookup is exact: a wavelength that is not
    in the table raises a ``ValueError`` listing the wavelengths that are
    defined. There is deliberately no nearest-neighbour fallback, so a
    profile designed for one wavelength is never silently applied to another.

    Matching uses a relative tolerance of ``WAVELENGTH_RTOL`` (and no absolute
    tolerance). This only absorbs floating-point representation error, e.g.
    when rays are traced in single precision; it is orders of magnitude
    tighter than any physically meaningful spectral spacing.

    If one call contains rays at several wavelengths, each child is evaluated
    once and its result is selected element-wise, so the profile works with
    both the NumPy and PyTorch backends and remains differentiable.

    Example::

        profile = WavelengthDependentPhaseProfile(
            {
                0.4861: RadialPhaseProfile(coefficients=[-64.6]),
                0.5876: RadialPhaseProfile(coefficients=[-53.5]),
                0.6563: RadialPhaseProfile(coefficients=[-47.9]),
            }
        )

    Args:
        profiles: Mapping from wavelength in µm to the phase profile used at
            that wavelength. Must contain at least one entry. All children
            must report the same ``efficiency``, because the efficiency of a
            phase profile is not wavelength-dependent in the
            ``BasePhaseProfile`` interface.

    Raises:
        ValueError: If ``profiles`` is empty, a wavelength is not a positive
            finite number, two wavelengths are equal within
            ``WAVELENGTH_RTOL``, or the children report different
            efficiencies.
        TypeError: If a value in ``profiles`` is not a ``BasePhaseProfile``.
    """

    phase_type = "wavelength_dependent"

    WAVELENGTH_RTOL = 1e-6

    def __init__(self, profiles: Mapping[float, BasePhaseProfile]):
        self._profiles = self._validate_profiles(profiles)
        super().__init__()

    @classmethod
    def _validate_profiles(
        cls, profiles: Mapping[float, BasePhaseProfile]
    ) -> dict[float, BasePhaseProfile]:
        if not profiles:
            raise ValueError(
                "WavelengthDependentPhaseProfile requires at least one "
                "(wavelength, phase profile) entry."
            )

        validated: dict[float, BasePhaseProfile] = {}
        for key, profile in profiles.items():
            wavelength = float(key)
            if not math.isfinite(wavelength) or wavelength <= 0.0:
                raise ValueError(
                    f"Wavelengths must be positive and finite, got {key!r}."
                )
            if not isinstance(profile, BasePhaseProfile):
                raise TypeError(
                    f"The profile for wavelength {wavelength} um must be a "
                    f"BasePhaseProfile, got {type(profile).__name__}."
                )
            for existing in validated:
                if math.isclose(
                    wavelength, existing, rel_tol=cls.WAVELENGTH_RTOL, abs_tol=0.0
                ):
                    raise ValueError(
                        f"Wavelengths {existing} um and {wavelength} um are "
                        "equal within the matching tolerance, so lookup "
                        "would be ambiguous."
                    )
            validated[wavelength] = profile

        efficiencies = {float(p.efficiency) for p in validated.values()}
        if len(efficiencies) > 1:
            raise ValueError(
                "All child profiles must have the same efficiency, got "
                f"{sorted(efficiencies)}. Per-wavelength efficiency is not "
                "supported by the BasePhaseProfile interface."
            )

        return dict(sorted(validated.items()))

    @property
    def profiles(self) -> dict[float, BasePhaseProfile]:
        """Mapping from wavelength in µm to its child profile, sorted."""
        return dict(self._profiles)

    @property
    def wavelengths(self) -> list[float]:
        """The wavelengths in µm that have a child profile, sorted."""
        return list(self._profiles)

    @property
    def parent_surface(self) -> Surface | None:
        """The surface this profile is attached to.

        Setting it also attaches every child profile, so children that need
        the surface (e.g. ``HeightProfile``, which reads its materials) work
        unchanged inside the wrapper.
        """
        return self._parent_surface

    @parent_surface.setter
    def parent_surface(self, value: Surface | None) -> None:
        self._parent_surface = value
        for profile in self._profiles.values():
            profile.parent_surface = value

    @property
    def efficiency(self) -> float:
        """The diffraction efficiency shared by all child profiles."""
        return next(iter(self._profiles.values())).efficiency

    def get_phase(self, x: be.Array, y: be.Array, wavelength: be.Array) -> be.Array:
        """Calculates the phase of the matching child profile at (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.
            wavelength: The wavelength of each point in µm.

        Returns:
            The phase at each (x, y) coordinate.

        Raises:
            ValueError: If any wavelength has no child profile.
        """
        return self._dispatch(
            wavelength, lambda profile: profile.get_phase(x, y, wavelength)
        )

    def get_gradient(
        self, x: be.Array, y: be.Array, wavelength: be.Array
    ) -> tuple[be.Array, be.Array, be.Array]:
        """Calculates the phase gradient of the matching child profile.

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.
            wavelength: The wavelength of each point in µm.

        Returns:
            A tuple containing the x, y, and z components of the phase
            gradient (d_phi/dx, d_phi/dy, d_phi/dz).

        Raises:
            ValueError: If any wavelength has no child profile.
        """
        return self._dispatch(
            wavelength, lambda profile: profile.get_gradient(x, y, wavelength)
        )

    def get_paraxial_gradient(self, y: be.Array, wavelength: be.Array) -> be.Array:
        """Calculates the paraxial phase gradient of the matching child.

        Args:
            y: The y-coordinates of the points of interest.
            wavelength: The wavelength of each point in µm.

        Returns:
            The paraxial phase gradient at each y-coordinate.

        Raises:
            ValueError: If any wavelength has no child profile.
        """
        return self._dispatch(
            wavelength,
            lambda profile: profile.get_paraxial_gradient(y, wavelength),
        )

    def _dispatch(
        self,
        wavelength: be.Array,
        evaluate: Callable[[BasePhaseProfile], typing.Any],
    ) -> typing.Any:
        """Evaluates the child matching each wavelength and merges the results.

        Args:
            wavelength: The wavelength of each point in µm.
            evaluate: Calls the relevant method on a child profile. Returns
                an array or a tuple of arrays.

        Returns:
            The child result, selected element-wise by wavelength.
        """
        if wavelength is None:
            raise ValueError(
                "WavelengthDependentPhaseProfile needs the ray wavelength to "
                "select a phase profile, but wavelength=None was passed."
            )

        hits = []
        matched = None
        for key, profile in self._profiles.items():
            mask = be.isclose(wavelength, key, rtol=self.WAVELENGTH_RTOL, atol=0.0)
            if be.any(mask):
                hits.append((profile, mask))
                matched = mask if matched is None else matched | mask

        if matched is None or not be.all(matched):
            raise ValueError(self._missing_wavelength_message(wavelength, matched))

        # Common case: every ray in the call has the same wavelength.
        if len(hits) == 1:
            return evaluate(hits[0][0])

        # Masks are disjoint because the table wavelengths are distinct.
        result = evaluate(hits[0][0])
        for profile, mask in hits[1:]:
            value = evaluate(profile)
            if isinstance(result, tuple):
                result = tuple(
                    be.where(mask, v, r) for v, r in zip(value, result, strict=True)
                )
            else:
                result = be.where(mask, value, result)
        return result

    def _missing_wavelength_message(
        self, wavelength: be.Array, matched: be.Array | None
    ) -> str:
        w = np.atleast_1d(_to_numpy(wavelength))
        if matched is None:
            missing = np.unique(w)
        else:
            hit = np.broadcast_to(np.atleast_1d(_to_numpy(matched)), w.shape)
            missing = np.unique(w[~hit])
        missing_list = ", ".join(f"{v:g}" for v in missing[:10])
        if missing.size > 10:
            missing_list += ", ..."
        defined = ", ".join(f"{v:g}" for v in self._profiles)
        return (
            f"WavelengthDependentPhaseProfile has no phase profile for "
            f"wavelength(s) [{missing_list}] um. Defined wavelengths: "
            f"[{defined}] um. Add a profile for every traced wavelength; "
            "lookup is exact and never falls back to the nearest wavelength."
        )

    def to_dict(self) -> dict:
        """Serializes the phase profile and its children to a dictionary.

        Returns:
            A dictionary representation of the phase profile.
        """
        data = super().to_dict()
        data["profiles"] = [
            {"wavelength": wavelength, "profile": profile.to_dict()}
            for wavelength, profile in self._profiles.items()
        ]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> WavelengthDependentPhaseProfile:
        """Deserializes a phase profile from a dictionary.

        Args:
            data: A dictionary representation of a phase profile.

        Returns:
            An instance of a `WavelengthDependentPhaseProfile`.
        """
        return cls(
            {
                entry["wavelength"]: BasePhaseProfile.from_dict(entry["profile"])
                for entry in data["profiles"]
            }
        )
