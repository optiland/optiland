"""NSQ Material adapter -- thin differentiable wrapper over optiland.materials.

Evaluates refractive index as an attached computation node when the backend
is PyTorch, enabling gradients w.r.t. material dispersion parameters.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    import torch

    from optiland.materials import BaseMaterial
    from optiland.nonsequential.bsdf.base import BaseBSDF

# Wavelength input: Python float, numpy array, or torch Tensor
WavelengthInput = Union[float, np.ndarray, "torch.Tensor"]


@dataclass
class NSQMaterial:
    """Thin differentiable adapter over optiland.materials.BaseMaterial.

    Evaluates ``n(wavelength_um)`` without any grad-severing casts
    (no ``float()``, no ``np.asarray()``) so the result stays in the
    autograd graph when using the Torch backend.

    Attributes:
        optiland_material: Underlying material model. None means vacuum (n=1).
        bsdf: Optional surface scatter model.
    """

    optiland_material: BaseMaterial | None = None
    bsdf: BaseBSDF | None = None

    @classmethod
    def from_glass(cls, name: str) -> NSQMaterial:
        """Resolve a glass catalog name to an NSQMaterial.

        Args:
            name: Glass catalog name (e.g. ``'N-BK7'``, ``'SF11'``).

        Returns:
            NSQMaterial wrapping the resolved BaseMaterial.

        Raises:
            ValueError: If the glass name is not found in the catalog.
        """
        from optiland.materials import Material  # noqa: PLC0415

        try:
            mat = Material(name)
        except Exception as exc:
            raise ValueError(
                f"Glass '{name}' not found in the Optiland material catalog."
            ) from exc
        return cls(optiland_material=mat)

    def n(self, wavelength_um: WavelengthInput) -> WavelengthInput:
        """Refractive index at the given wavelength(s).

        Differentiable: when ``wavelength_um`` is a torch Tensor with
        ``requires_grad=True``, the returned value carries attached gradients.
        No ``float()`` or ``np.asarray()`` casts are applied to the result.

        Args:
            wavelength_um: Wavelength(s) in micrometres [µm]. Accepts Python
                float, NumPy ndarray, or torch Tensor.

        Returns:
            Refractive index with the same array type as the input. Returns
            ``1.0`` (scalar) for vacuum when input is a scalar, or a
            ones-like array/tensor matching the input shape for array inputs.
        """
        if self.optiland_material is None:
            # Vacuum: return ones matching the input type/device
            try:
                return be.ones_like(wavelength_um)
            except (TypeError, AttributeError):
                return 1.0
        # Pass wavelength directly -- preserves the grad graph
        return self.optiland_material.n(wavelength_um)

    def k(self, wavelength_um: WavelengthInput) -> WavelengthInput:
        """Extinction coefficient at the given wavelength(s).

        Feeds Beer-Lambert bulk absorption: ``alpha = 4*pi*k/wavelength_um``
        [1/um], matching ``optiland.propagation.homogeneous
        .HomogeneousPropagation`` so NSQ and the sequential engine attenuate
        a glass path by the same amount.

        Args:
            wavelength_um: Wavelength(s) in micrometres [µm].

        Returns:
            Extinction coefficient (dimensionless). Returns ``0.0`` (scalar)
            for vacuum when input is a scalar, or a zeros-like array/tensor
            matching the input shape for array inputs.
        """
        if self.optiland_material is None:
            # Vacuum: non-absorbing.
            try:
                return be.zeros_like(wavelength_um)
            except (TypeError, AttributeError):
                return 0.0
        return self.optiland_material.k(wavelength_um)


# Module-level vacuum constant
VACUUM: NSQMaterial = NSQMaterial(optiland_material=None)


def medium_stack_id(material: NSQMaterial) -> int:
    """Canonical integer id for a medium, for the ray-level medium stack.

    Every vacuum-like material (``optiland_material is None``) maps to the
    same id ``0`` regardless of which ``NSQMaterial`` instance wraps it --
    two separately constructed vacuum wrappers are physically the same
    medium. Any other material is identified by Python object identity: two
    ``NSQMaterial`` instances that happen to wrap the same physical glass
    are only treated as the same medium if the scene reuses one instance
    for both (as :class:`~optiland.nonsequential.components.volume.Volume`
    and the ``Lens``/``Doublet`` builders do for a shared interior). This is
    a conservative default -- reusing distinct instances for the same
    physical glass on either side of a gap can produce a spurious
    ``medium_stack_underflows`` count, never a wrong flux/index result (the
    stack is a diagnostic cross-check; ``n1``/``n2`` are always resolved
    geometrically, never from the stack).

    Args:
        material: The medium to identify.

    Returns:
        A stable (for the process lifetime) non-negative integer id.
    """
    if material.optiland_material is None:
        return 0
    return id(material)
