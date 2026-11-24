"""Material at Conditions

This module contains the MaterialAtConditions class, which evaluates another
material at a fixed temperature and pressure. It lets an optic be traced with
its glasses at another temperature without passing the conditions through the
ray trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.materials.base import BaseMaterial

if TYPE_CHECKING:
    import optiland.backend as be
    from optiland.propagation.base import BasePropagationModel

ABSOLUTE_ZERO_C = -273.15


class MaterialAtConditions(BaseMaterial):
    """A material evaluated at a fixed temperature and pressure.

    Every refractive index and extinction coefficient query is forwarded to
    the wrapped material with this wrapper's ``temperature`` and ``pressure``.
    The wrapper's conditions always apply: keyword arguments given to ``n`` or
    ``k`` are not forwarded.

    Only materials whose model depends on the conditions change with them.
    :class:`~optiland.materials.material_file.MaterialFile` (and
    :class:`~optiland.materials.material.Material`) applies the Schott dn/dT
    model when its data includes thermal coefficients. Materials that ignore
    temperature and pressure, such as
    :class:`~optiland.materials.ideal.IdealMaterial` and
    :class:`~optiland.materials.abbe.AbbeMaterial`, return their normal index.

    Args:
        material: The material to evaluate.
        temperature: The temperature in degrees Celsius. Must be above
            absolute zero (-273.15 °C).
        pressure: The ambient pressure in atmospheres. Must not be negative.
            Defaults to 1.0.
        propagation_model: The propagation model to use for this material.
            If None, a default HomogeneousPropagation model is created.

    Raises:
        TypeError: If ``material`` is not a material.
        ValueError: If ``temperature`` is not above absolute zero, or
            ``pressure`` is negative.
    """

    def __init__(
        self,
        material: BaseMaterial,
        temperature: float,
        pressure: float = 1.0,
        propagation_model: BasePropagationModel | None = None,
    ):
        if not isinstance(material, BaseMaterial):
            raise TypeError(
                f"material must be a BaseMaterial, got {type(material).__name__}."
            )
        temperature = float(temperature)
        pressure = float(pressure)
        # Written so that NaN fails both checks.
        if not temperature > ABSOLUTE_ZERO_C:
            raise ValueError(
                f"temperature must be above {ABSOLUTE_ZERO_C} °C, got {temperature}."
            )
        if not pressure >= 0.0:
            raise ValueError(f"pressure must not be negative, got {pressure}.")

        super().__init__(propagation_model)
        self._material = material
        self._temperature = temperature
        self._pressure = pressure

    @property
    def material(self) -> BaseMaterial:
        """The wrapped material."""
        return self._material

    @property
    def temperature(self) -> float:
        """The temperature in degrees Celsius."""
        return self._temperature

    @property
    def pressure(self) -> float:
        """The ambient pressure in atmospheres."""
        return self._pressure

    @property
    def thermal_expansion(self) -> float | None:
        """The wrapped material's coefficient of thermal expansion, in 1/K.

        ``None`` means the wrapped material has no CTE data, not that the CTE
        is zero.
        """
        return self._material.thermal_expansion

    @property
    def display_name(self) -> str:
        """The wrapped material's name with the conditions."""
        return (
            f"{self._material.display_name} @ {self._temperature:.1f} °C, "
            f"{self._pressure:.2f} atm"
        )

    def _cache_state(self) -> tuple | None:
        """Combine the wrapped material's state with the conditions.

        The wrapped material is cached only when its own class defines
        ``_cache_state``, as :meth:`BaseMaterial._evaluate_property` does, so
        this wrapper never caches a material that does not track its state.
        """
        if "_cache_state" not in type(self._material).__dict__:
            return None
        state = self._material._cache_state()
        if state is None:
            return None
        return (type(self._material), state, self._temperature, self._pressure)

    def _calculate_n(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Returns the wrapped material's refractive index at the conditions.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float or be.ndarray: The refractive index at the given wavelength(s).
        """
        return self._material.n(
            wavelength, temperature=self._temperature, pressure=self._pressure
        )

    def _calculate_k(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Returns the wrapped material's extinction coefficient at the conditions.

        Args:
            wavelength (float or be.ndarray): The wavelength(s) of light in microns.

        Returns:
            float or be.ndarray: The extinction coefficient at the given
            wavelength(s).
        """
        return self._material.k(
            wavelength, temperature=self._temperature, pressure=self._pressure
        )

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
                "temperature": self._temperature,
                "pressure": self._pressure,
            }
        )
        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the material.

        Returns:
            MaterialAtConditions: The material.
        """
        return cls(
            BaseMaterial.from_dict(data["material"]),
            data["temperature"],
            data.get("pressure", 1.0),
        )
