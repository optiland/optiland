"""Offset Material

This module contains the OffsetMaterial class, which shifts the refractive
index and the Abbe number of another material while keeping the shape of its
dispersion curve. It models melt-to-melt deviations of a glass, as used by
index and Abbe-number tolerances.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.materials.base import BaseMaterial

if TYPE_CHECKING:
    from optiland.propagation.base import BasePropagationModel

# Fraunhofer lines in µm, as used by BaseMaterial.abbe().
WAVELENGTH_D = 0.5875618
WAVELENGTH_F = 0.4861327
WAVELENGTH_C = 0.6562725


class OffsetMaterial(BaseMaterial):
    """A material with an offset refractive index and Abbe number.

    With ``n_d``, ``n_F`` and ``n_C`` the indices of the wrapped material at
    the Fraunhofer d, F and C lines and ``ν_d = (n_d - 1) / (n_F - n_C)``, the
    index is

    ``n'(λ) = n_d + Δn + s (n(λ) - n_d)``, with
    ``s = (n_d + Δn - 1) / ((ν_d + Δν) (n_F - n_C))``.

    The d-line index becomes exactly ``n_d + Δn`` and the Abbe number exactly
    ``ν_d + Δν``. The dispersion curve is scaled about the d line, so relative
    partial dispersions such as ``P_g,F`` are unchanged. The extinction
    coefficient is passed through unchanged, and keyword arguments given to
    ``n`` or ``k`` (such as ``temperature``) are forwarded to the wrapped
    material. With both offsets zero, the wrapped material's values are
    returned unchanged.

    For a non-dispersive material (``n_F == n_C``), such as an
    :class:`~optiland.materials.ideal.IdealMaterial`, only an index offset is
    defined; it is added at every wavelength.

    Args:
        material: The material to offset. It is not modified.
        delta_nd: The offset of the d-line refractive index. Defaults to 0.0.
        delta_vd: The offset of the Abbe number. Defaults to 0.0.
        propagation_model: The propagation model to use for this material.
            If None, a default HomogeneousPropagation model is created.

    Raises:
        TypeError: If ``material`` is not a material.
        ValueError: If ``delta_vd`` is nonzero and the material is not
            dispersive.
    """

    def __init__(
        self,
        material: BaseMaterial,
        delta_nd: float | be.ndarray = 0.0,
        delta_vd: float | be.ndarray = 0.0,
        propagation_model: BasePropagationModel | None = None,
    ):
        if not isinstance(material, BaseMaterial):
            raise TypeError(
                f"material must be a BaseMaterial, got {type(material).__name__}."
            )
        super().__init__(propagation_model)
        self._material = material
        self._delta_nd = delta_nd
        self._delta_vd = delta_vd
        if not self._is_zero(delta_vd):
            _, n_f, n_c = self._reference_indices()
            self._check_dispersive(n_f, n_c)

    @property
    def material(self) -> BaseMaterial:
        """The wrapped material."""
        return self._material

    @property
    def delta_nd(self) -> float | be.ndarray:
        """The offset of the d-line refractive index."""
        return self._delta_nd

    @property
    def delta_vd(self) -> float | be.ndarray:
        """The offset of the Abbe number."""
        return self._delta_vd

    @property
    def thermal_expansion(self) -> float | None:
        """The wrapped material's coefficient of thermal expansion, in 1/K."""
        return self._material.thermal_expansion

    @property
    def display_name(self) -> str:
        """The wrapped material's name with the offsets."""
        return (
            f"{self._material.display_name} "
            f"(Δnd {_as_float(self._delta_nd):+g}, Δvd {_as_float(self._delta_vd):+g})"
        )

    @staticmethod
    def _is_zero(value) -> bool:
        """True for an exact zero that carries no gradient."""
        return not BaseMaterial._requires_grad(value) and _as_float(value) == 0.0

    @staticmethod
    def _check_dispersive(n_f, n_c) -> None:
        if be.isclose(n_f, n_c, rtol=0.0, atol=1e-12):
            raise ValueError(
                "An Abbe-number offset is undefined for a non-dispersive "
                "material (n_F == n_C)."
            )

    def _reference_indices(self, **kwargs) -> tuple:
        """Return the wrapped material's n_d, n_F and n_C."""
        return tuple(
            self._material.n(wavelength, **kwargs)
            for wavelength in (WAVELENGTH_D, WAVELENGTH_F, WAVELENGTH_C)
        )

    def _cache_state(self) -> tuple | None:
        """Combine the wrapped material's state with the offsets.

        As in :class:`~optiland.materials.at_conditions.MaterialAtConditions`,
        results are cached only when the wrapped material tracks its own state
        and the offsets carry no gradient.
        """
        if "_cache_state" not in type(self._material).__dict__:
            return None
        state = self._material._cache_state()
        offsets = self._state_key((self._delta_nd, self._delta_vd))
        if state is None or offsets is None:
            return None
        return (type(self._material), state, offsets)

    def _calculate_n(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Returns the offset refractive index.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.
            **kwargs: Forwarded to the wrapped material.

        Returns:
            float or be.ndarray: The refractive index at the given wavelength(s).
        """
        n = self._material.n(wavelength, **kwargs)
        if self._is_zero(self._delta_nd) and self._is_zero(self._delta_vd):
            return n
        n_d, n_f, n_c = self._reference_indices(**kwargs)
        if be.isclose(n_f, n_c, rtol=0.0, atol=1e-12):
            if not self._is_zero(self._delta_vd):
                self._check_dispersive(n_f, n_c)
            return n + self._delta_nd
        # (ν_d + Δν)(n_F - n_C) = (n_d - 1) + Δν (n_F - n_C)
        scale = (n_d + self._delta_nd - 1) / (n_d - 1 + self._delta_vd * (n_f - n_c))
        return n_d + self._delta_nd + scale * (n - n_d)

    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Returns the wrapped material's extinction coefficient.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.
            **kwargs: Forwarded to the wrapped material.

        Returns:
            float or be.ndarray: The extinction coefficient at the given
            wavelength(s).
        """
        return self._material.k(wavelength, **kwargs)

    def spectral_range(self, property_name: str = "n") -> tuple[float, float] | None:
        """The wrapped material's validity limits in micrometers, or None."""
        return self._material.spectral_range(property_name)

    def to_dict(self):
        """Returns a dictionary representation of the material.

        The wrapped material is stored with its own serialization.

        Returns:
            dict: A dictionary representation of the material.
        """
        material_dict = super().to_dict()
        material_dict.update(
            {
                "material": self._material.to_dict(),
                "delta_nd": _as_float(self._delta_nd),
                "delta_vd": _as_float(self._delta_vd),
            }
        )
        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            OffsetMaterial: The material.
        """
        return cls(
            BaseMaterial.from_dict(data["material"]),
            data.get("delta_nd", 0.0),
            data.get("delta_vd", 0.0),
        )


def _as_float(value) -> float:
    """Convert a Python or backend scalar to a Python float."""
    if hasattr(value, "detach"):
        value = value.detach()
    return float(be.to_numpy(value).reshape(-1)[0])
