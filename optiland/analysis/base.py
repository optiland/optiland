"""Analysis Base Module

This module contains the abstract base class for all analysis
classes in the Optiland package.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from optiland.utils import resolve_wavelengths

if TYPE_CHECKING:
    from optiland.optic import Optic


class BaseAnalysis(abc.ABC):
    """Base class for all analysis routines.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str or list, optional): The wavelengths to analyze.
            Can be 'all', 'primary', or a list of wavelength values.
            Defaults to 'all'.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The list of wavelengths (in µm) being analyzed.
        data: The generated analysis data. This is populated by the
              `_generate_data` method implemented by subclasses.
    """

    def __init__(self, optic: Optic, wavelengths: str | list = "all"):
        self.optic = optic
        self.wavelengths = resolve_wavelengths(optic, wavelengths)
        self.data = self._generate_data()

    @abc.abstractmethod
    def _generate_data(self):
        """Abstract method to generate analysis-specific data.

        This method must be implemented by subclasses. It should perform
        the necessary calculations and return the data to be stored in
        `self.data`.
        """
        pass

    @abc.abstractmethod
    def view(self, figsize=None, *, show: bool = True, **kwargs):
        """Visualize the analysis data.

        Args:
            figsize (tuple, optional): Figure size passed to matplotlib.
            show (bool): If True (default), calls plt.show(). Set to False
                for headless use (e.g. saving to file, CI environments).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            The matplotlib Figure object (or a tuple starting with the Figure).
        """
        pass
