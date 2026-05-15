"""Spot Diagram Analysis

This module provides the core spot diagram analysis for optical systems,
including data generation, centering, and metrics calculation.

Kramer Harrison, 2024
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import optiland.backend as be
from optiland.utils import resolve_fields
from optiland.visualization.system.utils import transform

from ..base import BaseAnalysis
from .plotting import (
    calculate_axis_limits,
    finalize_plot,
    handle_no_fields,
    plot_field,
    setup_plot_layout,
)
from .reference import SpotReferenceType, create_reference_strategy

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland._types import BEArray, DistributionType


@dataclass
class SpotData:
    """Stores the x, y coordinates and intensity of a spot.

    Attributes:
        x: Array of x-coordinates.
        y: Array of y-coordinates.
        intensity: Array of intensity values.
    """

    x: be.array
    y: be.array
    intensity: be.array


class SpotDiagram(BaseAnalysis):
    """Generates and plots real ray intersection data on the image surface.

    This class creates spot diagrams, which are purely geometric plots that give
    an indication of the blur produced by aberrations in an optical system.

    Attributes:
        optic: Instance of the optic object to be assessed.
        fields: Fields at which data is generated.
        wavelengths: Wavelengths at which data is generated.
        num_rings: Number of rings in the pupil distribution for ray tracing.
        distribution: The pupil distribution type for ray tracing.
        data: Contains spot data in a nested list, ordered by field, then
            wavelength.
        coordinates: The coordinate system ('global' or 'local') for data and
            plotting.
        reference: The reference point type used for centering spots.
    """

    def __init__(
        self,
        optic,
        fields: str | list = "all",
        wavelengths: str | list = "all",
        num_rings: int = 6,
        distribution: DistributionType = "hexapolar",
        coordinates: Literal["global", "local"] = "local",
        reference: str | SpotReferenceType = SpotReferenceType.CHIEF_RAY,
    ):
        """Initializes the SpotDiagram analysis.

        Note:
            The constructor generates all data that is later used for plotting.

        Args:
            optic: An instance of the optic object to be assessed.
            fields: Fields at which to generate data. If 'all', all defined
                field points are used. Defaults to "all".
            wavelengths: Wavelengths at which to generate data. If 'all', all
                defined wavelengths are used. Defaults to "all".
            num_rings: Number of rings in the pupil distribution for ray
                tracing. Defaults to 6.
            distribution: Pupil distribution type for ray tracing.
                Defaults to "hexapolar".
            coordinates: Coordinate system for data generation and plotting.
                Defaults to "local".
            reference: Reference point type for centering spots. Can be
                "chief_ray" or "centroid". Defaults to "chief_ray".

        Raises:
            ValueError: If `coordinates` is not 'global' or 'local'.
            ValueError: If `reference` is not a valid SpotReferenceType.
        """
        self.fields = resolve_fields(optic, fields)  # list[FieldPoint]

        if coordinates not in ["global", "local"]:
            raise ValueError("Coordinates must be 'global' or 'local'.")
        self.coordinates = coordinates

        self.num_rings = num_rings
        self.distribution: DistributionType = distribution
        self._reference_strategy = create_reference_strategy(reference)

        super().__init__(optic, wavelengths)
        primary_wl_value = self.optic.primary_wavelength
        wl_values = [wp.value for wp in self.wavelengths]
        if primary_wl_value in wl_values:
            self._analysis_ref_wavelength_index = wl_values.index(primary_wl_value)
        else:
            self._analysis_ref_wavelength_index = 0

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (12, 4),
        add_airy_disk: bool = False,
        *,
        show: bool = True,
    ) -> tuple[Figure, list[Axes]]:
        """Displays the spot diagram plot.

        Args:
            fig_to_plot_on: An existing Matplotlib figure to plot on. If None,
                a new figure is created. Defaults to None.
            figsize: The figure size for the output window, applied per row.
                Defaults to (12, 4).
            add_airy_disk: If True, adds the Airy disk visualization to the
                plots. Defaults to False.
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            A tuple containing the Matplotlib figure and a list of its axes.
        """
        if not self.fields:
            return handle_no_fields(fig_to_plot_on)

        centered_data = self._center_spots(self.data)
        airy_disk_data = self._prepare_airy_disk_data() if add_airy_disk else None

        fig, axs = setup_plot_layout(len(self.fields), fig_to_plot_on, figsize)
        axis_lim = calculate_axis_limits(centered_data, self.fields, airy_disk_data)

        for i, field_data in enumerate(centered_data):
            if i >= len(axs):
                break
            plot_field(
                axs[i],
                field_data,
                self.wavelengths,
                self.fields[i].coord,
                axis_lim,
                i,
                self.optic.image_surface,
                airy_disk_data,
            )

        finalize_plot(fig, axs, len(self.fields), self.wavelengths)
        is_gui_embedding = fig_to_plot_on is not None
        if show and not is_gui_embedding:
            import matplotlib.pyplot as plt

            plt.show()
        return fig, fig.get_axes()

    # --- Calculation Methods ---

    def angle_from_cosine(self, a: BEArray, b: BEArray) -> float:
        """Calculates the angle in radians between two direction cosine vectors.

        Args:
            a: The first direction cosine vector.
            b: The second direction cosine vector.

        Returns:
            The angle between the vectors in radians.
        """
        a = a / be.linalg.norm(a)
        b = b / be.linalg.norm(b)
        return be.arccos(be.clip(be.dot(a, b), -1, 1))

    def f_number(self, n: float, theta: float) -> float:
        """Calculates the physical F-number.

        Args:
            n: The refractive index of the medium.
            theta: The half-angle of the cone of light in radians.

        Returns:
            The calculated physical F-number.
        """
        return 1 / (2 * n * be.sin(theta))

    def airy_radius(self, n_w: float, wavelength: float) -> float:
        """Calculates the Airy disk radius.

        Args:
            n_w: The physical F-number.
            wavelength: The wavelength of light in micrometers.

        Returns:
            The Airy disk radius.
        """
        return 1.22 * n_w * wavelength

    def generate_marginal_rays(
        self, H_x: float, H_y: float, wavelength: float
    ) -> tuple:
        """Generates marginal rays at the four cardinal points of the pupil.

        Args:
            H_x: The x-field coordinate.
            H_y: The y-field coordinate.
            wavelength: The wavelength for the rays.

        Returns:
            A tuple containing the traced rays for north, south, east, and west
            pupil points.
        """
        ray_north = self.optic.trace_generic(
            Hx=H_x, Hy=H_y, Px=0, Py=1, wavelength=wavelength
        )
        ray_south = self.optic.trace_generic(
            Hx=H_x, Hy=H_y, Px=0, Py=-1, wavelength=wavelength
        )
        ray_east = self.optic.trace_generic(
            Hx=H_x, Hy=H_y, Px=1, Py=0, wavelength=wavelength
        )
        ray_west = self.optic.trace_generic(
            Hx=H_x, Hy=H_y, Px=-1, Py=0, wavelength=wavelength
        )
        return ray_north, ray_south, ray_east, ray_west

    def generate_marginal_rays_cosines(
        self, H_x: float, H_y: float, wavelength: float
    ) -> tuple:
        """Generates direction cosines for each marginal ray of a given field.

        Args:
            H_x: The x-field coordinate.
            H_y: The y-field coordinate.
            wavelength: The wavelength for the rays.

        Returns:
            A tuple of direction cosine vectors for north, south, east, and
            west rays.
        """
        rays = self.generate_marginal_rays(H_x, H_y, wavelength)
        return tuple(be.array([ray.L, ray.M, ray.N]).ravel() for ray in rays)

    def generate_chief_rays_cosines(self, wavelength: float) -> BEArray:
        """Generates direction cosines for the chief ray of each field.

        Args:
            wavelength: The wavelength for the rays.

        Returns:
            An array of shape (num_fields, 3) containing the direction cosines.
        """
        cosines = [
            be.array([ray.L, ray.M, ray.N]).ravel()
            for fp in self.fields
            for ray in [
                self.optic.trace_generic(
                    Hx=fp.coord[0], Hy=fp.coord[1], Px=0, Py=0, wavelength=wavelength
                )
            ]
        ]
        return be.stack(cosines, axis=0)

    def generate_chief_rays_centers(self, wavelength: float) -> BEArray:
        """Generates the (x, y) intersection points for the chief ray of each
        field.

        Args:
            wavelength: The wavelength for the rays.

        Returns:
            An array of shape (num_fields, 2) with (x, y) coordinates.
        """
        centers = [
            [ray.x.item(), ray.y.item()]
            for fp in self.fields
            for ray in [
                self.optic.trace_generic(
                    Hx=fp.coord[0], Hy=fp.coord[1], Px=0, Py=0, wavelength=wavelength
                )
            ]
        ]
        return be.stack(centers, axis=0)

    def airy_disc_x_y(self, wavelength: float) -> tuple[list[float], list[float]]:
        """Generates the Airy disk radii for the x and y axes for each field.

        Args:
            wavelength: The wavelength for the calculation.

        Returns:
            A tuple of two lists: x-axis radii and y-axis radii per field.
        """
        chief_cosines = self.generate_chief_rays_cosines(wavelength)
        airy_rad_x_list, airy_rad_y_list = [], []

        for i, fp in enumerate(self.fields):
            H_x, H_y = fp.coord
            north, south, east, west = self.generate_marginal_rays_cosines(
                H_x, H_y, wavelength
            )
            chief = chief_cosines[i]

            angle_x = (
                self.angle_from_cosine(chief, north)
                + self.angle_from_cosine(chief, south)
            ) / 2
            angle_y = (
                self.angle_from_cosine(chief, east)
                + self.angle_from_cosine(chief, west)
            ) / 2

            f_num_x = self.f_number(n=1, theta=angle_x)
            f_num_y = self.f_number(n=1, theta=angle_y)

            # Convert radius from µm to mm
            airy_rad_x_list.append(self.airy_radius(f_num_x, wavelength) * 1e-3)
            airy_rad_y_list.append(self.airy_radius(f_num_y, wavelength) * 1e-3)

        return airy_rad_x_list, airy_rad_y_list

    def centroid(self) -> list[tuple[BEArray, BEArray]]:
        """Calculates the geometric centroid of each spot for the reference
        wavelength.

        Returns:
            A list of (x, y) centroid coordinates for each field.
        """
        ref_idx = self._analysis_ref_wavelength_index
        return [
            (be.mean(field_data[ref_idx].x), be.mean(field_data[ref_idx].y))
            for field_data in self.data
        ]

    def geometric_spot_radius(self) -> list[list[BEArray]]:
        """Calculates the maximum geometric spot radius for each spot.

        Returns:
            A nested list of maximum radii for each field and wavelength.
        """
        centered_data = self._center_spots(self.data)
        return [
            [
                be.max(be.sqrt(wave_data.x**2 + wave_data.y**2))
                for wave_data in field_data
            ]
            for field_data in centered_data
        ]

    def rms_spot_radius(self) -> list[list[BEArray]]:
        """Calculates the root-mean-square (RMS) spot radius for each spot.

        Returns:
            A nested list of RMS radii for each field and wavelength.
        """
        centered_data = self._center_spots(self.data)
        return [
            [
                be.sqrt(be.mean(wave_data.x**2 + wave_data.y**2))
                for wave_data in field_data
            ]
            for field_data in centered_data
        ]

    # --- Internal Data Generation Helpers ---

    def _get_reference_centers(
        self, data: list[list[SpotData]]
    ) -> list[tuple[BEArray, BEArray]]:
        """Computes the reference centers using the configured strategy.

        Args:
            data: The spot data to compute centers for.

        Returns:
            A list of (x, y) center tuples, one per field.
        """
        ref_wl = self.wavelengths[self._analysis_ref_wavelength_index].value
        fields_coords = [fp.coord for fp in self.fields]
        return self._reference_strategy.get_centers(
            data,
            self._analysis_ref_wavelength_index,
            self.optic,
            fields_coords,
            ref_wl,
            self.coordinates,
        )

    def _center_spots(self, data: list[list[SpotData]]) -> list[list[SpotData]]:
        """Centers spot data around the configured reference point.

        Args:
            data: The original, uncentered spot data.

        Returns:
            A deep copy of the data, centered around the reference points.
        """
        centers = self._get_reference_centers(data)
        centered_data = []
        for i, field_list in enumerate(data):
            cx, cy = centers[i]
            centered_field = [
                SpotData(
                    x=sd.x - cx,
                    y=sd.y - cy,
                    intensity=be.copy(sd.intensity),
                )
                for sd in field_list
            ]
            centered_data.append(centered_field)
        return centered_data

    def _generate_data(self) -> list[list[SpotData]]:
        """Generates spot data for all configured fields and wavelengths.

        Returns:
            A nested list of spot intersection data.
        """
        return [
            [
                self._generate_field_data(
                    fp.coord,
                    wp.value,
                    self.num_rings,
                    self.distribution,
                    self.coordinates,
                )
                for wp in self.wavelengths
            ]
            for fp in self.fields
        ]

    def _generate_field_data(
        self,
        field: tuple[float, float],
        wavelength: float,
        num_rays: int,
        distribution: DistributionType,
        coordinates: str,
    ) -> SpotData:
        """Generates spot data for a single field and wavelength.

        Args:
            field: The (Hx, Hy) field coordinates.
            wavelength: The wavelength for tracing.
            num_rays: The number of rays to generate, or number of rings if
                distribution is hexapolar.
            distribution: The ray distribution pattern.
            coordinates: The coordinate system ('local' or 'global').

        Returns:
            A SpotData object with the traced ray intersection data.
        """
        self.optic.trace(*field, wavelength, num_rays, distribution)
        surf_group = self.optic.surfaces
        x_g, y_g, z_g, i_g = (
            surf_group.x[-1, :],
            surf_group.y[-1, :],
            surf_group.z[-1, :],
            surf_group.intensity[-1, :],
        )

        # Ignore rays with zero intensity
        mask = i_g > 0
        x_g, y_g, z_g, i_g = x_g[mask], y_g[mask], z_g[mask], i_g[mask]

        if coordinates == "local":
            x_plot, y_plot, _ = transform(
                x_g, y_g, z_g, self.optic.image_surface, is_global=True
            )
        else:
            x_plot, y_plot = x_g, y_g

        return SpotData(x=x_plot, y=y_plot, intensity=i_g)

    def _prepare_airy_disk_data(self) -> dict:
        """Prepares all necessary data for plotting the Airy disk.

        The Airy disk position is determined by the configured reference
        strategy: when using chief ray centering, the Airy disk sits at (0,0);
        when using centroid centering, it is offset by the difference between
        the chief ray and centroid positions.

        Returns:
            A dictionary containing Airy disk radii and center coordinates
            relative to the reference point.
        """
        primary_wl_obj = self.optic.wavelengths.primary_wavelength
        wl_val = primary_wl_obj.value if primary_wl_obj else self.wavelengths[0]

        airy_rad_x, airy_rad_y = self.airy_disc_x_y(wavelength=wl_val)
        chief_centers = self.generate_chief_rays_centers(wavelength=wl_val)
        reference_centers = self._get_reference_centers(self.data)

        # Airy disk is physically at the chief ray position. Compute its
        # offset relative to whichever reference was used for centering.
        airy_centers = be.to_numpy(chief_centers) - be.to_numpy(
            be.stack(reference_centers)
        )

        return {
            "radii_x": be.to_numpy(airy_rad_x),
            "radii_y": be.to_numpy(airy_rad_y),
            "airy_centers": airy_centers,
        }
