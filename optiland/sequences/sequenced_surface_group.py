"""SequencedSurfaceGroup: the ``SurfaceGroup`` interface over a sequence of views.

Presents a list of :class:`~optiland.sequences.surface_view.SurfaceView`
objects — resolved from a base surface list and a raw step list — through
the read/trace subset of the :class:`~optiland.surfaces.surface_group.SurfaceGroup`
interface that analyses and the tracing pipeline rely on. A sequence is
static once resolved: unlike ``SurfaceGroup``, there is no ``add``/``remove``;
to change a sequence's traversal, resolve a new one.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.coatings import BaseCoatingPolarized
from optiland.sequences.resolver import resolve_sequence

if TYPE_CHECKING:
    from optiland.sequences.steps import RawStep
    from optiland.sequences.surface_view import SurfaceView
    from optiland.surfaces.standard_surface import Surface


class SequencedSurfaceGroup:
    """A traversal-ordered group of ``SurfaceView`` over shared base surfaces.

    Args:
        base_surfaces: The optic's base surfaces, indexed as in ``raw_steps``.
        raw_steps: The raw sequence definition. See
            :func:`optiland.sequences.steps.parse_steps`.

    Raises:
        ValueError: If ``raw_steps`` is empty, malformed, or references an
            out-of-range surface index.
        SequenceValidationError: If adjacent steps are not physically
            consistent.
    """

    def __init__(self, base_surfaces: list[Surface], raw_steps: list[RawStep]):
        self.base_surfaces = base_surfaces
        self.raw_steps = raw_steps
        self._views: list[SurfaceView] = resolve_sequence(base_surfaces, raw_steps)

    @property
    def surfaces(self) -> tuple[SurfaceView, ...]:
        return tuple(self._views)

    def __getitem__(self, index):
        return self._views[index]

    def __iter__(self):
        return iter(self._views)

    def __len__(self):
        return len(self._views)

    @property
    def num_surfaces(self) -> int:
        """int: the number of steps in the sequence."""
        return len(self._views)

    @property
    def stop_index(self) -> int:
        """int: the index (within this sequence) of the aperture stop step."""
        for index, view in enumerate(self._views):
            if view.is_stop:
                return index

        raise ValueError("No stop surface found in this sequence.")

    @property
    def x(self):
        """np.array: x intersection points at each step."""
        return be.stack([v.x for v in self._views if be.size(v.x) > 0])

    @property
    def y(self):
        """np.array: y intersection points at each step."""
        return be.stack([v.y for v in self._views if be.size(v.y) > 0])

    @property
    def z(self):
        """np.array: z intersection points at each step."""
        return be.stack([v.z for v in self._views if be.size(v.z) > 0])

    @property
    def L(self):
        """np.array: x direction cosines at each step."""
        return be.stack([v.L for v in self._views if be.size(v.L) > 0])

    @property
    def M(self):
        """np.array: y direction cosines at each step."""
        return be.stack([v.M for v in self._views if be.size(v.M) > 0])

    @property
    def N(self):
        """np.array: z direction cosines at each step."""
        return be.stack([v.N for v in self._views if be.size(v.N) > 0])

    @property
    def opd(self):
        """np.array: optical path difference recorded at each step."""
        return be.stack([v.opd for v in self._views if be.size(v.opd) > 0])

    @property
    def u(self):
        """np.array: paraxial ray angles at each step."""
        return be.stack([v.u for v in self._views if be.size(v.u) > 0])

    @property
    def intensity(self):
        """np.array: ray intensities at each step."""
        return be.stack([v.intensity for v in self._views if be.size(v.intensity) > 0])

    @property
    def positions(self):
        """np.array: z positions of each step's surface vertex, in traversal order."""
        positions = be.array([v.geometry.cs.position_in_gcs[2] for v in self._views])
        return positions.reshape(-1, 1)

    @property
    def radii(self):
        """np.array: radii of curvature at each step."""
        return be.array([v.geometry.radius for v in self._views])

    @property
    def uses_polarization(self) -> bool:
        """bool: True if any step's interaction uses polarization."""
        return any(
            isinstance(v.interaction_model.coating, BaseCoatingPolarized)
            for v in self._views
        )

    def n(self, wavelength):
        """Get the exit-medium refractive index at each step.

        Args:
            wavelength (float or str): The wavelength for which to
                calculate the refractive indices.

        Returns:
            numpy.ndarray: The refractive indices at each step.
        """
        n = [be.atleast_1d(v.material_post.n(wavelength)) for v in self._views]
        return be.ravel(be.array(n))

    def get_thickness(self, step_number: int):
        """Calculate the (signed) axial distance between two consecutive steps.

        Args:
            step_number (int): The index (within this sequence) of the
                first step.

        Returns:
            float: The distance between the two steps' surface vertices.
                Negative if the sequence traverses backward at this point.
        """
        t = self.positions
        return t[step_number + 1] - t[step_number]

    def trace(self, rays, skip: int = 0):
        """Trace the given rays through the sequence.

        Args:
            rays (BaseRays): The rays to be traced.
            skip (int, optional): Number of steps to skip before tracing.
                Defaults to 0.

        Returns:
            BaseRays: The traced rays.
        """
        self.reset()
        for view in self._views[skip:]:
            view.trace(rays)
        return rays

    def reset(self) -> None:
        """Resets the recorded information owned by every view in the sequence."""
        for view in self._views:
            view.reset()
