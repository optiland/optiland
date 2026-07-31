"""SequencedOptic: a view of a base ``Optic`` traced through a sub-sequence.

Composition, not inheritance: a ``SequencedOptic`` delegates aperture,
fields, wavelengths, polarization, and paraxial analysis to the base optic
untouched, and exposes ``surfaces`` as a
:class:`~optiland.sequences.sequenced_surface_group.SequencedSurfaceGroup`.

Ray definition (conjugates, aperture stop, aiming) always comes from the
base optic's own nominal sequence — a sub-sequence defines traversal only.
Per-sequence first-order analysis is out of scope for v1 (SPEC §4, phase 5).

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.rays import PolarizedRays
from optiland.sequences.sequenced_surface_group import SequencedSurfaceGroup

if TYPE_CHECKING:
    from optiland._types import DistributionType
    from optiland.distribution import BaseDistribution
    from optiland.optic.optic import Optic
    from optiland.rays import PolarizationState, RealRays
    from optiland.sequences.steps import RawStep


class SequencedOptic:
    """A named sub-sequence over a base optic's surfaces.

    Args:
        base_optic: The optic whose surfaces this sequence traverses.
        name: A name for the sequence, unique within the base optic.
        steps: The raw step list. See
            :func:`optiland.sequences.steps.parse_steps`.

    Raises:
        ValueError: If ``steps`` is empty, malformed, or references an
            out-of-range surface index.
        SequenceValidationError: If adjacent steps are not physically
            consistent.
    """

    def __init__(self, base_optic: Optic, name: str, steps: list[RawStep]):
        self.base_optic = base_optic
        self.name = name
        self.raw_steps = steps
        self.surfaces = SequencedSurfaceGroup(base_optic.surfaces.surfaces, steps)

    # -- Delegated to the base optic ---------------------------------------

    @property
    def aperture(self):
        return self.base_optic.aperture

    @property
    def fields(self):
        return self.base_optic.fields

    @property
    def wavelengths(self):
        return self.base_optic.wavelengths

    @property
    def polarization(self) -> PolarizationState | str:
        return self.base_optic.polarization

    @property
    def polarization_state(self) -> PolarizationState | None:
        return self.base_optic.polarization_state

    @property
    def apodization(self):
        return self.base_optic.apodization

    @property
    def paraxial(self):
        return self.base_optic.paraxial

    @property
    def primary_wavelength(self) -> float:
        return self.base_optic.primary_wavelength

    @property
    def object_surface(self):
        """The base optic's object surface (conjugates are never per-sequence)."""
        return self.base_optic.object_surface

    @property
    def surface_group(self):
        """Alias for ``surfaces``, matching internal use of ``Optic.surface_group``."""
        return self.surfaces

    # -- Owned ---------------------------------------------------------------

    @property
    def image_surface(self):
        """The last step's view: wherever this sequence terminates.

        Unlike the base optic, a sub-sequence does not necessarily end at
        the system's physical image plane (e.g. a ghost path that folds
        back on itself), so "image surface" here means the sequence's own
        terminal step.
        """
        return self.surfaces[-1]

    def n(self, wavelength: float | str = "primary"):
        """Get the exit-medium refractive indices at each step of this sequence.

        Args:
            wavelength: The wavelength in microns, or ``"primary"`` to use
                the base optic's primary wavelength.

        Returns:
            be.ndarray: The refractive indices at each step.
        """
        if wavelength == "primary":
            wavelength = self.primary_wavelength
        return self.surfaces.n(wavelength)

    def trace(
        self,
        Hx,
        Hy,
        wavelength: float,
        num_rays: int | None = 100,
        distribution: DistributionType | BaseDistribution | None = "hexapolar",
    ) -> RealRays:
        """Trace a distribution of rays through this sequence.

        Rays are generated exactly as they would be for the base optic
        (its own conjugates, aperture stop, and ray aiming); only the
        traversal through the surfaces differs.

        Args:
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            wavelength: The wavelength of the rays in microns.
            num_rays: The number of rays to trace. Defaults to 100.
            distribution: The distribution of rays. Defaults to 'hexapolar'.

        Returns:
            RealRays: The traced rays.
        """
        tracer = self.base_optic.ray_tracer
        tracer._validate_normalized_coordinates(Hx, Hy, "field")

        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays)
        Px, Py = distribution.x, distribution.y

        Hx = be.atleast_1d(Hx)
        Hy = be.atleast_1d(Hy)
        num_fields = len(Hx)
        num_pupil_points = len(Px)

        Hx_full = be.repeat(Hx, num_pupil_points)
        Hy_full = be.repeat(Hy, num_pupil_points)
        Px_full = be.tile(Px, num_fields)
        Py_full = be.tile(Py, num_fields)

        rays = tracer.ray_generator.generate_rays(
            Hx_full, Hy_full, Px_full, Py_full, wavelength
        )
        rays = self.surfaces.trace(rays)

        if isinstance(rays, PolarizedRays):
            rays.update_intensity(self.polarization_state)

        return rays

    def trace_generic(self, Hx, Hy, Px, Py, wavelength: float) -> RealRays:
        """Trace a single generic ray (given field and pupil coordinates).

        As with :meth:`trace`, the ray is generated exactly as it would be
        for the base optic; only the traversal through the surfaces differs.

        Args:
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            wavelength: The wavelength of the rays in microns.

        Returns:
            RealRays: The traced rays.
        """
        tracer = self.base_optic.ray_tracer
        tracer._validate_normalized_coordinates(Hx, Hy, "field")
        tracer._validate_normalized_coordinates(Px, Py, "pupil")

        vx, vy = self.fields.get_vig_factor(Hx, Hy)
        Px = Px * (1 - vx)
        Py = Py * (1 - vy)

        Hx, Hy, Px, Py = tracer._validate_array_size(Hx, Hy, Px, Py)

        rays = tracer.ray_generator.generate_rays(Hx, Hy, Px, Py, wavelength)
        return self.surfaces.trace(rays)

    def __repr__(self) -> str:
        return (
            f"SequencedOptic(name={self.name!r}, base_optic={self.base_optic!r}, "
            f"steps={len(self.surfaces)})"
        )
