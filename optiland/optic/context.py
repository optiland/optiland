"""OpticDataContext Protocol

Defines a read-only structural Protocol that captures the subset of Optic
state needed by physics computation classes (Aberrations, Paraxial, etc.).
Accepting this Protocol instead of the full Optic type allows unit testing
without constructing a complete optical system.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from optiland.paraxial import Paraxial
    from optiland.surfaces.surface_group import SurfaceGroup
    from optiland.wavelength import WavelengthGroup


@runtime_checkable
class OpticDataContext(Protocol):
    """Read-only view of Optic state needed by physics computation classes.

    Any object satisfying this Protocol may be passed where an Optic is
    currently required by Aberrations, Paraxial, or similar classes.
    This enables unit testing without constructing a full Optic.
    """

    @property
    def surfaces(self) -> SurfaceGroup: ...

    @property
    def wavelengths(self) -> WavelengthGroup: ...

    @property
    def paraxial(self) -> Paraxial: ...
