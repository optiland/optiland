"""Compensator Module

This module contains a class for compensating perturbations in an optical
system. The CompensatorOptimizer class is used to optimize a set of variables
to compensate for perturbations in an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.optimization import (
    LeastSquares,
    OptimizationProblem,
    OptimizerGeneric,
    minimize,
)
from optiland.optimization.failure import ConfigurationError
from optiland.solves import QuickFocusSolve

if TYPE_CHECKING:
    from optiland.optic import Optic


class CompensatorOptimizer(OptimizationProblem):
    """A class representing a compensator for a tolerancing problem. This class
    optimizes a set of variables to compensate for perturbations in an optical
    system.

    Args:
        method (str): The type of optimizer to use. Default is 'generic'.
            Other options: 'least_squares'
        tol (float): The tolerance for the optimizer. Default is 1e-5.

    Attributes:
        method (str): The type of optimizer to use. Default is 'generic'.
            Other options include 'least_squares'.
        tol (float): The tolerance for the optimizer. Default is 1e-5.
        _optimizer_map (dict): A mapping of optimizer types to their
            respective classes.

    """

    def __init__(self, method="generic", tol=1e-5):
        super().__init__()
        self.method = method
        self.tol = tol

        self._optimizer_map = {
            "generic": OptimizerGeneric,
            "least_squares": LeastSquares,
        }

    def get_optimizer(self):
        if self.method not in self._optimizer_map:
            raise ValueError(f'Invalid optimizer method "{self.method}".')
        return self._optimizer_map[self.method]

    @property
    def has_variables(self):
        """Check if the optimizer has variables. If no variables are present,
        then the optimizer will not run.

        Returns:
            bool: True if the optimizer has variables, False otherwise.

        """
        return bool(self.variables)

    def run(self):
        """Run the optimizer for compensation.

        The run goes through ``optiland.optimization.minimize`` with the
        settings the legacy optimizers used: ``"generic"`` is L-BFGS-B with
        at most 1000 iterations, ``"least_squares"`` is
        ``scipy.optimize.least_squares`` with the ``"lm"`` method. Nothing
        is printed. A starting design with non-finite operands is left
        unchanged, so a trial that cannot be traced does not stop a Monte
        Carlo run.

        Returns:
            scipy.optimize.OptimizeResult: The result of the optimizer run, or
            None if the starting design could not be evaluated.

        Raises:
            ValueError: If ``method`` is not a valid compensator method.

        """
        self.get_optimizer()  # validates the method
        try:
            if self.method == "generic":
                return minimize(self, "l-bfgs-b", maxiter=1000, tol=self.tol)
            return minimize(self, "least_squares", tol=self.tol, method_choice="lm")
        except ConfigurationError:
            if np.all(np.isfinite(be.to_numpy(self.fun_array()))):
                raise
            return None


class QuickFocusCompensator:
    """Refocus the image surface to the minimum RMS spot position per trial.

    The image surface is moved along z to the position that minimizes the RMS
    spot size of one field and wavelength (``QuickFocusSolve``). This is a
    closed-form solve, much cheaper than an optimizer compensator. Only
    coaxial systems are supported.

    Args:
        optic: The optic whose image surface is moved.
        Hx: Normalized x field coordinate. Defaults to 0.
        Hy: Normalized y field coordinate. Defaults to 0.
        wavelength: Wavelength in µm, or ``"primary"`` (default) for the
            primary wavelength of the optic.
        num_rays: Pupil sampling; for ``"hexapolar"`` the number of rings.
            Defaults to 5.
        distribution: Ray distribution. Defaults to ``"hexapolar"``.

    Attributes:
        label: Column name of the compensator value in result tables.
    """

    label = "quick_focus: image distance"

    def __init__(
        self,
        optic: Optic,
        Hx: float = 0.0,
        Hy: float = 0.0,
        wavelength: float | str = "primary",
        num_rays: int = 5,
        distribution: str = "hexapolar",
    ) -> None:
        self.optic = optic
        self.Hx = Hx
        self.Hy = Hy
        self.wavelength = wavelength
        self.num_rays = num_rays
        self.distribution = distribution
        self._z_before = None

    def apply(self) -> dict[str, float]:
        """Move the image surface to the best focus.

        Returns:
            The distance from the last surface before the image to the image
            surface after refocusing, keyed by ``label``.

        Raises:
            UnsupportedParaxialGeometryError: If the system is folded.
        """
        wavelength = self.wavelength
        if wavelength == "primary":
            wavelength = self.optic.wavelengths.primary_wavelength.value

        image = self.optic.surfaces[-1].geometry.cs
        z_focus = QuickFocusSolve(self.optic).optimal_focus_distance(
            Hx=self.Hx,
            Hy=self.Hy,
            wavelength=wavelength,
            num_rays=self.num_rays,
            distribution=self.distribution,
        )
        if self._z_before is None:
            self._z_before = image.z
        image.z = z_focus

        previous = self.optic.surfaces[-2].geometry.cs
        return {self.label: float(image.z - previous.z)}

    def reset(self) -> None:
        """Move the image surface back to where it was before ``apply``.

        Call before resetting the perturbations: a thickness perturbation
        moves the image surface again when it is reset.
        """
        if self._z_before is not None:
            self.optic.surfaces[-1].geometry.cs.z = self._z_before
            self._z_before = None
