"""Through-focus analysis base class

This module provides a base class for performing through-focus analysis.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import optiland.backend as be


class ThroughFocusAnalysis(ABC):
    """Base class for through-focus analysis.

    This class provides the basic framework for analyses that need to be
    performed over a range of focal planes. It handles the iteration through
    different focal positions and delegates the specific analysis at each
    position to subclasses.

    Args:
        optic (optiland.optic.Optic): The optical system to analyze.
        delta_focus (float, optional): The increment of focal shift in mm.
            Defaults to 0.1.
        num_steps (int, optional): The number of focal planes to analyze
            before and after the nominal focus. Defaults to 5.
        fields (list[tuple[float, float]] | str, optional): Fields for
            analysis. If "all", uses all fields from `optic.fields`.
            Otherwise, expects a list of field coordinates.
            Defaults to "all".
        wavelengths (list[float] | str, optional): Wavelengths for
            analysis. If "all", uses all wavelengths from
            `optic.wavelengths`. Otherwise, expects a list of
            wavelength values. Defaults to "all".

    Attributes:
        results (list): A list to store the results from the analysis performed
            at each focal plane. The structure of each item in the list is
            determined by the `_perform_analysis_at_focus` method of the
            subclass.
    """

    MAX_STEPS = 7
    MIN_STEPS = 1

    def __init__(
        self,
        optic,
        delta_focus: float = 0.1,
        num_steps: int = 5,
        fields="all",
        wavelengths: str | list[float] = "all",
    ):
        self.optic = optic
        self.delta_focus = delta_focus
        self._validate_num_steps(num_steps)
        self.num_steps = num_steps
        self.fields = self._resolve_fields(fields)
        self.wavelengths = self._resolve_wavelengths(wavelengths)
        self.positions = self._generate_focus_positions()

        self.nominal_focus = be.copy(self.optic.image_surface.geometry.cs.z)

        self.results = []
        self._calculate_through_focus()

    def _validate_num_steps(self, num_steps):
        if not isinstance(num_steps, int) or num_steps < self.MIN_STEPS:
            raise ValueError("'num_steps' must be a positive integer.")
        if num_steps % 2 == 0:
            raise ValueError("'num_steps' must be an odd integer.")
        if num_steps > self.MAX_STEPS:
            raise ValueError(
                "'num_steps' must be less than or equal to 7 for performance reasons."
            )

    def _resolve_fields(self, fields):
        from optiland.utils import resolve_fields as _resolve_fields

        return _resolve_fields(self.optic, fields)

    def _resolve_wavelengths(self, wavelengths):
        from optiland.utils import resolve_wavelengths as _resolve_wavelengths

        return _resolve_wavelengths(self.optic, wavelengths)

    def _generate_focus_positions(self):
        """Generates a list of focal positions based on the delta focus and number of
        steps."""
        nominal_focus = self.optic.image_surface.geometry.cs.z
        positions = [
            nominal_focus + (i - self.num_steps // 2) * self.delta_focus
            for i in range(self.num_steps)
        ]
        return positions

    def _defocus_image_plane(self, z_position):
        """Applies defocus to the image plane of the optical system.

        Args:
            z_position (float): The new z-coordinate for the image surface
        """
        self.optic.image_surface.geometry.cs.z = z_position

    def _reset_focus(self):
        """Resets the image surface to its nominal focus position."""
        self._defocus_image_plane(self.nominal_focus)

    @abstractmethod
    def view(self, *, show: bool = True):
        """Visualizes or prints the results of the through-focus analysis.

        This abstract method must be implemented by subclasses. It defines
        how the collected `self.results` (containing analysis data from all
        focal planes) should be presented to the user, for example, by
        plotting graphs or printing a formatted table.

        Args:
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _perform_analysis_at_focus(self):
        """Performs the specific analysis at the current focal position.

        This abstract method must be implemented by subclasses. It defines
        the analysis to be performed at each focal position after applying
        the defocus. The results of this analysis should be returned and will
        be stored in `self.results`.

        Returns:
            Any: The result of the analysis at the current focal position.
        """
        pass  # pragma: no cover

    def _calculate_through_focus(self):
        """Performs the through-focus analysis by iterating over the focal positions.

        This method iterates through each focal position, applies the defocus
        to the optical system, and performs the specific analysis defined in
        `_perform_analysis_at_focus`. The results are stored in `self.results`.
        """
        for position in self.positions:
            self._defocus_image_plane(position)
            result = self._perform_analysis_at_focus()
            self.results.append(result)
        self._reset_focus()
