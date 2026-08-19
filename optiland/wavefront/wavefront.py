"""
This module defines the `Wavefront` class, which is designed to analyze the
wavefront of an optical system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.distribution import BaseDistribution, create_distribution
from optiland.utils import resolve_fields, resolve_wavelengths

from .strategy import create_strategy

if TYPE_CHECKING:
    from optiland._types import DistributionType, Fields, Wavelengths
    from optiland.fields import Field
    from optiland.optic.optic import Optic
    from optiland.wavefront.strategy import WavefrontStrategyType
    from optiland.wavefront.wavefront_data import WavefrontData


class Wavefront:
    """Performs wavefront analysis on an optical system.

    This class acts as a high-level controller that delegates the complex
    work of wavefront analysis to a specified strategy (e.g., 'chief_ray' or
    'centroid_sphere'). It computes ray intersection points with the exit pupil,
    the optical path difference (OPD), ray intensities, and the radius of
    curvature of the reference sphere.

    Args:
        optic (Optic): The optical system to analyze.
        fields (str or list[tuple[float, float]]): The fields to analyze.
            Can be "all" to use all fields defined in the optic.
        wavelengths (str or list[float]): The wavelengths to analyze. Can be
            "all" for all wavelengths or "primary" for the primary wavelength.
        num_rays (int): The number of rays to use for pupil sampling.
        distribution (str or Distribution): The ray distribution pattern. Can
            be a name (e.g., "hexapolar") or a Distribution object.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray", "centroid_sphere", and "best_fit_sphere".
            Defaults to "chief_ray".
        remove_tilt (bool): If True, removes tilt and piston from the OPD data.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the strategy.

    Attributes:
        data (dict): A dictionary containing the computed `WavefrontData` for
            each (field, wavelength) pair.
    """

    def __init__(
        self,
        optic: Optic,
        fields: Fields | list[Field] = "all",
        wavelengths: Wavelengths = "all",
        num_rays: int = 12,
        distribution: DistributionType = "hexapolar",
        strategy: WavefrontStrategyType = "chief_ray",
        afocal: bool = False,
        remove_tilt: bool = False,
        **kwargs,
    ):
        self.optic = optic
        self.fields = resolve_fields(optic, fields)
        self.wavelengths = resolve_wavelengths(optic, wavelengths)
        self.num_rays = num_rays
        self.distribution = self._resolve_distribution(distribution, self.num_rays)

        reference_type = "plane" if afocal else "sphere"
        self.strategy = create_strategy(
            strategy_name=strategy,
            optic=self.optic,
            distribution=self.distribution,
            reference_type=reference_type,
            **kwargs,
        )
        self.remove_tilt = remove_tilt

        self.data: dict[tuple[tuple[float, float], float], WavefrontData] = {}
        self._generate_data()

    def get_data(self, field: tuple[float, float], wl: float) -> WavefrontData:
        """Retrieves precomputed wavefront data for a field and wavelength.

        Args:
            field (tuple[float, float]): The field coordinates, or a FieldPoint.
            wl (float): The wavelength in µm, or a WavelengthPoint.

        Returns:
            WavefrontData: A data container with the computed wavefront results.
        """
        if hasattr(field, "coord"):
            field = field.coord
        if hasattr(wl, "value"):
            wl = wl.value
        return self.data[(field, wl)]

    @staticmethod
    def fit_and_remove_tilt(
        data: WavefrontData, remove_piston: bool = False, ridge: float = 1e-12
    ) -> be.ndarray:
        """
        Removes piston and tilt from OPD data using weighted least squares.

        Args:
            data (WavefrontData): The wavefront data containing pupil coordinates
                and OPD.
            remove_piston (bool, optional): If True, removes piston term as well
                as tilt. Defaults to False.
            ridge (float, optional): Small diagonal regularization for stability.
                Defaults to 1e-12.

        Returns:
            opd_detrended (be.ndarray): OPD with piston and tilt removed, shape (N,).
        """
        x = data.pupil_x
        y = data.pupil_y
        weights = data.intensity
        opd = data.opd

        # weighted design matrix
        one = be.ones_like(x)
        X = be.stack([one, x, y], axis=1)  # (N,3)

        # apply sqrt(weights) to each column
        W = be.sqrt(weights)[:, None]
        Xw = X * W
        yw = opd * be.sqrt(weights)

        XT_X = be.matmul(Xw.T, Xw) + ridge * be.eye(3)
        XT_y = be.matmul(Xw.T, yw)

        # solve for coefficients
        coeffs = be.linalg.solve(XT_X, XT_y)

        if not remove_piston:
            coeffs = be.copy(coeffs)
            coeffs[0] = 0.0

        # subtract fitted plane
        fitted = X @ coeffs
        opd_detrended = opd - fitted

        return opd_detrended

    def _resolve_distribution(
        self, dist: DistributionType | BaseDistribution, num_rays
    ) -> BaseDistribution:
        """Resolves the pupil distribution from the input specification."""
        if isinstance(dist, str):
            dist_obj = create_distribution(dist)
            dist_obj.generate_points(num_rays)
            return dist_obj
        return dist

    def _generate_data(self):
        """Generates wavefront data for all specified fields and wavelengths.

        This method iterates through each field and wavelength pair and
        delegates the computation to the selected strategy object.
        """
        for fp in self.fields:
            field = fp.coord
            for wp in self.wavelengths:
                wl = wp.value
                data = self.strategy.compute_wavefront_data(field, wl)

                if self.remove_tilt:
                    data.opd = self.fit_and_remove_tilt(data)

                self.data[(field, wl)] = data
