"""Thin film perturbation classes.

Applies thickness or refractive-index perturbations to individual layers of a
ThinFilmStack.  Reuses the sampler hierarchy from ``optiland.tolerancing``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from optiland.materials import IdealMaterial

if TYPE_CHECKING:
    from optiland.thin_film import ThinFilmStack
    from optiland.tolerancing.perturbation import BaseSampler


class ThinFilmPerturbation:
    """Perturbation applied to a single layer of a thin-film stack.

    Args:
        stack: The thin film stack containing the layer to perturb.
        layer_index: Index of the layer to perturb.
        perturbation_type: ``"thickness"`` or ``"index"``.
        sampler: A sampler instance (``RangeSampler``, ``DistributionSampler``,
            etc.) that generates perturbation values.
        is_relative: If True (default), perturbation is multiplicative
            (nominal * (1 + delta)).  If False, the sampled value replaces the
            nominal directly.
    """

    def __init__(
        self,
        stack: ThinFilmStack,
        layer_index: int,
        perturbation_type: Literal["thickness", "index"],
        sampler: BaseSampler,
        is_relative: bool = True,
    ):
        self.stack = stack
        self.layer_index = layer_index
        self.perturbation_type = perturbation_type
        self.sampler = sampler
        self.is_relative = is_relative
        self.value: float | None = None

        layer = self.stack.layers[self.layer_index]
        if perturbation_type == "thickness":
            self._nominal = layer.thickness_um
        elif perturbation_type == "index":
            if not isinstance(layer.material, IdealMaterial):
                raise TypeError(
                    "Index perturbation is only supported for IdealMaterial. "
                    f"Got {type(layer.material).__name__}."
                )
            self._nominal = float(layer.material.n(0.55))
        else:
            raise ValueError(
                f"perturbation_type must be 'thickness' or 'index', "
                f"got '{perturbation_type}'."
            )

    def __str__(self) -> str:
        return f"Layer {self.layer_index} {self.perturbation_type}"

    def apply(self) -> None:
        """Sample a perturbation value and apply it to the layer."""
        delta = self.sampler.sample()
        self.value = float(delta)
        layer = self.stack.layers[self.layer_index]

        if self.perturbation_type == "thickness":
            if self.is_relative:
                layer.thickness_um = self._nominal * (1.0 + delta)
            else:
                layer.thickness_um = delta
        elif self.perturbation_type == "index":
            new_n = self._nominal * (1.0 + delta) if self.is_relative else delta
            layer.material = IdealMaterial(n=new_n)

    def reset(self) -> None:
        """Restore the layer to its nominal state."""
        layer = self.stack.layers[self.layer_index]

        if self.perturbation_type == "thickness":
            layer.thickness_um = self._nominal
        elif self.perturbation_type == "index":
            layer.material = IdealMaterial(n=self._nominal)

        self.value = None
