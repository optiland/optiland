"""Core tolerancing class for thin film stacks.

Manages operands (optical performance metrics) and perturbations, and provides
``evaluate`` / ``reset`` methods consumed by the analysis classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from optiland.thin_film.optimization.operand.thin_film import ThinFilmOperand

from .perturbation import ThinFilmPerturbation

if TYPE_CHECKING:
    from optiland.thin_film import ThinFilmStack
    from optiland.tolerancing.perturbation import BaseSampler


OpticalProperty = Literal["R", "T", "A"]


@dataclass
class ThinFilmOperandSpec:
    """Specification for a single operand."""

    property: OpticalProperty
    wavelength_nm: float
    aoi_deg: float
    polarization: str
    target: float | None


class ThinFilmTolerancing:
    """Container for thin-film operands and perturbations.

    Args:
        stack: The thin film stack to tolerance.
    """

    def __init__(self, stack: ThinFilmStack):
        self.stack = stack
        self.operands: list[ThinFilmOperandSpec] = []
        self.perturbations: list[ThinFilmPerturbation] = []

    def add_operand(
        self,
        property: OpticalProperty,
        wavelength_nm: float,
        aoi_deg: float = 0.0,
        polarization: str = "u",
        target: float | None = None,
    ) -> ThinFilmTolerancing:
        """Add a performance operand.

        If *target* is ``None`` the current computed value is stored as the
        nominal reference.

        Returns:
            self for method chaining.
        """
        if target is None:
            target = self._evaluate_property(
                property, wavelength_nm, aoi_deg, polarization
            )

        self.operands.append(
            ThinFilmOperandSpec(
                property=property,
                wavelength_nm=wavelength_nm,
                aoi_deg=aoi_deg,
                polarization=polarization,
                target=target,
            )
        )
        return self

    def add_perturbation(
        self,
        layer_index: int,
        perturbation_type: Literal["thickness", "index"] = "thickness",
        sampler: BaseSampler | None = None,
        is_relative: bool = True,
    ) -> ThinFilmTolerancing:
        """Add a perturbation to a layer.

        Returns:
            self for method chaining.
        """
        if sampler is None:
            raise ValueError("A sampler must be provided.")
        self.perturbations.append(
            ThinFilmPerturbation(
                stack=self.stack,
                layer_index=layer_index,
                perturbation_type=perturbation_type,
                sampler=sampler,
                is_relative=is_relative,
            )
        )
        return self

    def evaluate(self) -> list[float]:
        """Evaluate all operands and return their current values."""
        return [
            self._evaluate_property(
                op.property, op.wavelength_nm, op.aoi_deg, op.polarization
            )
            for op in self.operands
        ]

    def reset(self) -> None:
        """Reset all perturbations to nominal."""
        for p in self.perturbations:
            p.reset()

    def _evaluate_property(
        self,
        property: OpticalProperty,
        wavelength_nm: float,
        aoi_deg: float,
        polarization: str,
    ) -> float:
        func = {
            "R": ThinFilmOperand.reflectance,
            "T": ThinFilmOperand.transmittance,
            "A": ThinFilmOperand.absorptance,
        }[property]
        return func(self.stack, wavelength_nm, aoi_deg, polarization)
