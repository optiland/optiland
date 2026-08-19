"""Abstract base class for propagation models.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays.real_rays import RealRays


class BasePropagationModel(abc.ABC):
    """Abstract base class for all propagation models."""

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in the factory registry."""
        super().__init_subclass__(**kwargs)
        BasePropagationModel._registry[cls.__name__] = cls

    @abc.abstractmethod
    def propagate(self, rays: RealRays, t: float) -> None:
        """Propagates rays a distance t through the medium.

        This method modifies the ray state in-place.

        Args:
            rays: The rays object to be propagated.
            t: The distance to propagate along the z-axis.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serializes the propagation model to a dictionary.

        By default, this only stores the class name, as the model is
        reconstructed and linked by the parent material during its own
        deserialization.
        """
        return {"class": self.__class__.__name__}

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], material: BaseMaterial
    ) -> BasePropagationModel:
        """Deserializes a propagation model from a dictionary using a factory pattern.

        This factory method finds the correct propagation model subclass
        from the registry and delegates the deserialization to it.

        Args:
            d: A dictionary containing the serialized data, including a 'class' key.
            material: The parent material instance, passed to the subclass constructor.

        Returns:
            An instance of a specific propagation model subclass.
        """
        model_class_name = d.get("class")
        if model_class_name not in cls._registry:
            raise ValueError(f"Unknown propagation model class: {model_class_name}")

        # Look up the specific subclass in the registry.
        model_subclass = cls._registry[model_class_name]

        # Delegate the actual object creation to the subclass's from_dict method.
        return model_subclass.from_dict(d, material)
