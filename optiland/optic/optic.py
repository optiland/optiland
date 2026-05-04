"""Optic Module

This is the core module of Optiland, which provides the class to define
generic optical systems. The Optic class encapsulates the core properties
of an optical system, such as the aperture, fields, surfaces, and
wavelengths. It also provides methods to draw the optical system, trace rays,
and perform paraxial and aberration analyses. Instances of the Optic class
are used as arguments to various analysis, optimization, and visualization
functions in Optiland.

Kramer Harrison, 2024
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

from optiland._deprecation import deprecated
from optiland.aberrations import Aberrations
from optiland.aperture import BaseSystemAperture, make_system_aperture
from optiland.fields import (
    FieldGroup,
)
from optiland.optic.optic_serializer import OpticSerializer
from optiland.optic.optic_updater import OpticUpdater
from optiland.paraxial import Paraxial
from optiland.pickup import PickupManager
from optiland.rays import PolarizationState
from optiland.raytrace.real_ray_tracer import RealRayTracer
from optiland.solves import SolveManager
from optiland.surfaces import ObjectSurface, SurfaceGroup
from optiland.wavelength import WavelengthGroup

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland._types import (
        ApertureType,
        BEArray,
        DistributionType,
        FieldType,
        ReferenceRay,
        ScalarOrArray,
        SurfaceParameters,
        SurfaceType,
        Unpack,
        WavelengthUnit,
    )
    from optiland.apodization import BaseApodization
    from optiland.distribution import BaseDistribution
    from optiland.materials.base import BaseMaterial
    from optiland.rays import RealRays
    from optiland.surfaces.standard_surface import Surface


class Optic:
    """A class for defining and manipulating optical systems.

    The Optic class is central to the Optiland library, providing a comprehensive
    framework for representing optical systems. It encapsulates essential
    properties such as aperture, fields of view, constituent surfaces, and
    operating wavelengths. This class also offers a suite of methods for
    visualizing the optical system, performing ray tracing, and conducting
    paraxial and aberration analyses. Instances of Optic are fundamental objects
    passed to various analysis, optimization, and visualization functions
    throughout Optiland.

    Args:
        name (str, optional): An optional name for the optical system.
            Defaults to None.

    Attributes:
        name (str | None): An optional name for the optical system.
        aperture (Aperture | None): The aperture of the optical system.
        fields (FieldGroup): The group of fields defined for the system.
        wavelengths (WavelengthGroup): The group of wavelengths used for
            analysis.
        surfaces (SurfaceGroup): The group of surfaces that make up the
            optical system.
        paraxial (Paraxial): A helper class for paraxial analysis of the
            optical system.
        aberrations (Aberrations): A helper class for analyzing aberrations.
        ray_tracer (RealRayTracer): The ray tracer for performing real ray
            traces.
        polarization (PolarizationState | Literal['ignore']): The polarization
            state of the light. Defaults to 'ignore'.
        apodization (BaseApodization | None): The apodization function applied
            to the entrance pupil.
        pickups (PickupManager): Manages pickups, which link properties of one
            surface to another.
        solves (SolveManager): Manages solves, which automatically adjust
            surface properties to meet certain constraints.
        obj_space_telecentric (bool): If True, the system is object-space
            telecentric. Defaults to False.


    """

    def __init__(self, name: str | None = None):
        """
        Initializes an Optic instance.

        Args:
            name: An optional name for the optical system.
                Defaults to None.
        """
        self.name = name
        self.reset()

    def _initialize_attributes(self):
        """Initialize the attributes of the optical system."""
        self.aperture: BaseSystemAperture | None = None

        self.surfaces: SurfaceGroup = SurfaceGroup()
        self.fields: FieldGroup = FieldGroup()
        self.wavelengths: WavelengthGroup = WavelengthGroup()

        self.paraxial: Paraxial = Paraxial(self)
        self.aberrations: Aberrations = Aberrations(self)
        self.ray_tracer: RealRayTracer = RealRayTracer(self)

        self.polarization: PolarizationState | Literal["ignore"] = "ignore"

        self.apodization: BaseApodization | None = None
        self.pickups: PickupManager = PickupManager(self)
        self.solves: SolveManager = SolveManager(self)
        self.obj_space_telecentric: bool = False
        self.updater: OpticUpdater = OpticUpdater(self)

    @property
    def surface_group(self) -> SurfaceGroup:
        warnings.warn(
            "Optic.surface_group is deprecated; use Optic.surfaces instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.surfaces

    @surface_group.setter
    def surface_group(self, value):
        warnings.warn(
            "Optic.surface_group is deprecated; use Optic.surfaces instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.surfaces = value

    def __add__(self, other: Optic) -> Optic:
        """Add two Optic objects together.

        This method combines the surfaces of the two Optic objects. The
        properties of the first optic (aperture, fields, etc.) are retained.

        Args:
            other (Optic): The Optic object to add to the current one.

        Returns:
            Optic: A new Optic object containing the combined surfaces.
        """
        new_optic = deepcopy(self)
        new_optic.surfaces += other.surfaces
        return new_optic

    @property
    def primary_wavelength(self) -> float:
        """The primary wavelength in microns."""
        return self.wavelengths.primary_wavelength.value

    @property
    def object_surface(self) -> ObjectSurface | None:
        """The object surface instance (`ObjectSurface` or `None`)."""
        for surface in self.surfaces.surfaces:
            if isinstance(surface, ObjectSurface):
                return surface
        return None

    @property
    def image_surface(self) -> Surface:
        """The image surface instance."""
        return self.surfaces.surfaces[-1]

    @property
    def total_track(self) -> float:
        """The total track length of the system."""
        return self.surfaces.total_track

    @property
    def polarization_state(self) -> PolarizationState | None:
        """The polarization state of the optic.

        Returns:
            PolarizationState | None: The `PolarizationState` object if
            polarization is considered, otherwise `None`.

        Raises:
            ValueError: If the polarization state is invalid.
        """
        if self.polarization == "ignore":
            return None
        elif isinstance(self.polarization, PolarizationState):
            return self.polarization
        else:
            raise ValueError(
                "Invalid polarization state. Must be either "
                'PolarizationState or "ignore".',
            )

    def reset(self):
        """Reset the optical system to its initial state."""
        self._initialize_attributes()

    @deprecated("optic.surfaces.add()")
    def add_surface(
        self,
        new_surface: Surface | None = None,
        surface_type: SurfaceType = "standard",
        comment: str = "",
        index: int | None = None,
        is_stop: bool = False,
        material: str | BaseMaterial = "air",
        **kwargs: Unpack[SurfaceParameters],
    ):
        """Adds a new surface to the optic.

        .. deprecated::
            Use ``optic.surfaces.add()`` instead.

        Args:
            new_surface (Surface, optional): The new surface to add. If not
                provided, a new surface will be created based on the other
                arguments.
            surface_type (str, optional): The type of surface to create.
            index (int, optional): The index at which to insert the new
                surface. If not provided, the surface will be appended to the
                end of the list.
            is_stop (bool, optional): Indicates if the surface is the aperture.
            material (str, optional): The material of the surface.
                Default is 'air'.
            **kwargs: Additional keyword arguments for surface-specific
                parameters such as radius, conic, dx, dy, rx, ry, aperture.

        Raises:
            ValueError: If a new surface is provided and no index is given.
            IndexError: If the index is out of bounds for insertion, or negative.

        """
        self.surfaces.add(
            new_surface=new_surface,
            surface_type=surface_type,
            comment=comment,
            index=index,
            is_stop=is_stop,
            material=material,
            **kwargs,
        )

    @deprecated("optic.surfaces.remove()")
    def remove_surface(
        self,
        index: int,
    ):
        """Removes a surface from the optic.

        .. deprecated::
            Use ``optic.surfaces.remove()`` instead.

        Args:
            index (int, optional): The index of the surface to remove.

        """
        self.surfaces.remove(
            index=index,
        )

    @deprecated("optic.fields.add()")
    def add_field(self, y: float, x: float = 0.0, vx: float = 0.0, vy: float = 0.0):
        """Add a field to the optical system.

        .. deprecated::
            Use ``optic.fields.add()`` instead.

        Args:
            y: The y-coordinate of the field.
            x: The x-coordinate of the field.
                Defaults to 0.0.
            vx: The x-component of the field's vignetting
                factor. Defaults to 0.0.
            vy: The y-component of the field's vignetting
                factor. Defaults to 0.0.

        """
        self.fields.add(y, x, vx, vy)

    @deprecated("optic.wavelengths.add()")
    def add_wavelength(
        self,
        value: float,
        is_primary: bool = False,
        unit: WavelengthUnit = "um",
        weight: float = 1.0,
    ):
        """Add a wavelength to the optical system.

        .. deprecated::
            Use ``optic.wavelengths.add()`` instead.

        Args:
            value (float): The value of the wavelength.
            is_primary (bool, optional): If True, this wavelength is set as the
                primary wavelength. Defaults to False.
            unit (WavelengthUnit, optional): The unit of the wavelength.
                Defaults to 'um'.
            weight (float, optional): The weight of the wavelength for
                polychromatic analysis. Defaults to 1.0.

        """
        self.wavelengths.add(
            value=value, is_primary=is_primary, unit=unit, weight=weight
        )

    def set_aperture(self, aperture_type: ApertureType, value: float):
        """Set the aperture of the optical system.

        Args:
            aperture_type (ApertureType): The type of the aperture. Must be one of
                'EPD', 'imageFNO', or 'objectNA'.
            value (float): The value of the aperture.

        """
        self.aperture = make_system_aperture(aperture_type, value)

    @deprecated("optic.fields.set_type()")
    def set_field_type(self, field_type: FieldType):
        """Set the type of field used in the optical system.

        .. deprecated::
            Use ``optic.fields.set_type()`` instead.

        Args:
            field_type (FieldType): The type of field, e.g., 'angle',
                'object_height', or 'paraxial_image_height'.

        Raises:
            ValueError: If the field type is invalid.
        """
        self.fields.set_type(field_type)

    @deprecated("optic.updater.set_radius()")
    def set_radius(self, value: float, surface_number: int):
        """Set the radius of curvature of a surface.

        .. deprecated::
            Use ``optic.updater.set_radius()`` instead.

        Args:
            value (float): The value of the radius.
            surface_number (int): The index of the surface.

        """
        self.updater.set_radius(value, surface_number)

    @deprecated("optic.updater.set_conic()")
    def set_conic(self, value: float, surface_number: int):
        """Set the conic constant of a surface.

        .. deprecated::
            Use ``optic.updater.set_conic()`` instead.

        Args:
            value (float): The value of the conic constant.
            surface_number (int): The index of the surface.

        """
        self.updater.set_conic(value, surface_number)

    @deprecated("optic.updater.set_thickness()")
    def set_thickness(self, value: float, surface_number: int):
        """Set the thickness of a surface.

        .. deprecated::
            Use ``optic.updater.set_thickness()`` instead.

        Args:
            value (float): The value of the thickness.
            surface_number (int): The index of the surface.

        """
        self.updater.set_thickness(value, surface_number)

    @deprecated("optic.updater.set_index()")
    def set_index(self, value: float, surface_number: int):
        """Set the index of refraction of a surface.

        .. deprecated::
            Use ``optic.updater.set_index()`` instead.

        Args:
            value (float): The value of the index of refraction.
            surface_number (int): The index of the surface.

        """
        self.updater.set_index(value, surface_number)

    @deprecated("optic.updater.set_material()")
    def set_material(self, material: BaseMaterial, surface_number: int):
        """Set the material of a surface.

        .. deprecated::
            Use ``optic.updater.set_material()`` instead.

        Args:
            material (BaseMaterial): The material.
            surface_number (int): The index of the surface.

        """
        self.updater.set_material(material, surface_number)

    @deprecated("optic.updater.set_norm_radius()")
    def set_norm_radius(self, value: float, surface_number: int, is_fixed: bool = True):
        """Set the normalization radius of a surface.

        .. deprecated::
            Use ``optic.updater.set_norm_radius()`` instead.

        Args:
            value (float): The value of the normalization radius.
            surface_number (int): The index of the surface.
            is_fixed (bool, optional): Whether to lock the normalization radius
                from automatic paraxial updates. Defaults to True.
        """
        self.updater.set_norm_radius(value, surface_number, is_fixed)

    @deprecated("optic.updater.set_asphere_coeff()")
    def set_asphere_coeff(
        self, value: float, surface_number: int, aspher_coeff_idx: int
    ):
        """Set an aspheric coefficient on a surface.

        .. deprecated::
            Use ``optic.updater.set_asphere_coeff()`` instead.

        Args:
            value (float): The value of the aspheric coefficient.
            surface_number (int): The index of the surface.
            aspher_coeff_idx (int): The index of the aspheric coefficient to
                set.

        """
        self.updater.set_asphere_coeff(value, surface_number, aspher_coeff_idx)

    @deprecated("optic.updater.set_polarization()")
    def set_polarization(self, polarization: PolarizationState | Literal["ignore"]):
        """Set the polarization state of the optic.

        .. deprecated::
            Use ``optic.updater.set_polarization()`` instead.

        Args:
            polarization (PolarizationState | Literal['ignore']): The polarization
                state to set. It can be either a `PolarizationState` object or
                'ignore'.

        """
        self.updater.set_polarization(polarization)

    @deprecated("optic.updater.set_apodization()")
    def set_apodization(
        self, apodization: BaseApodization | str | dict = None, **kwargs
    ):
        """Sets the apodization for the optical system.

        .. deprecated::
            Use ``optic.updater.set_apodization()`` instead.

        Args:
            apodization (BaseApodization | str | dict, optional): The
                apodization to apply. Defaults to None.
            **kwargs: Additional keyword arguments used to initialize the
                apodization class when `apodization` is a string.
        """
        self.updater.set_apodization(apodization, **kwargs)

    @deprecated("optic.updater.scale_system()")
    def scale_system(self, scale_factor: float):
        """Scales the optical system by a given scale factor.

        .. deprecated::
            Use ``optic.updater.scale_system()`` instead.

        Args:
            scale_factor (float): The factor by which to scale the system.

        """
        self.updater.scale_system(scale_factor)

    @deprecated("optic.updater.update_paraxial()")
    def update_paraxial(self):
        """Update the semi-aperture of the surfaces based on paraxial analysis.

        .. deprecated::
            Use ``optic.updater.update_paraxial()`` instead.
        """
        self.updater.update_paraxial()

    @deprecated("optic.updater.update_normalization()")
    def update_normalization(self, surface: Surface) -> None:
        """Update the normalization radius of surfaces.

        .. deprecated::
            Use ``optic.updater.update_normalization()`` instead.
        """
        self.updater.update_normalization(surface)

    @deprecated("optic.ray_tracer.set_aiming()")
    def set_ray_aiming(
        self, mode: str, max_iter: int = 10, tol: float = 1e-6, **kwargs
    ):
        """Configure the ray aiming strategy.

        .. deprecated::
            Use ``optic.ray_tracer.set_aiming()`` instead.

        Args:
            mode: The aiming mode ("paraxial", "iterative", "robust").
            max_iter: Maximum iterations for iterative solvers.
            tol: Convergence tolerance for iterative solvers.
            **kwargs: Additional configuration parameters.
        """
        self.ray_tracer.set_aiming(mode, max_iter, tol, **kwargs)

    @deprecated("optic.updater.update()")
    def update(self) -> None:
        """Update the surface properties (pickups, solves, paraxial properties).

        .. deprecated::
            Use ``optic.updater.update()`` instead.
        """
        self.updater.update()

    @deprecated("optic.updater.image_solve()")
    def image_solve(self):
        """Update the image position such that the marginal ray crosses the optical axis
        at the image location.

        .. deprecated::
            Use ``optic.updater.image_solve()`` instead.
        """
        self.updater.image_solve()

    @deprecated("optic.updater.flip()")
    def flip(self):
        """Flips the optical system.

        .. deprecated::
            Use ``optic.updater.flip()`` instead.

        This reverses the order of surfaces (excluding object and image planes),
        their geometries, and materials. Pickups and solves referencing surface
        indices are updated accordingly. The coordinate system is adjusted such
        that the new first optical surface (originally the last one in the
        flipped segment) is placed at z=0.0.
        """
        self.updater.flip()

    def draw(
        self,
        fields: list[tuple[float, float]] | Literal["all"] = "all",
        wavelengths: list[float] | Literal["primary"] = "primary",
        num_rays: int = 3,
        distribution: DistributionType | None = None,
        show_apertures: bool = True,
        hide_vignetted: bool = False,
        figsize: tuple[float, float] = (10, 4),
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str | None = None,
        reference: ReferenceRay | None = None,
        projection: Literal["XY", "XZ", "YZ"] = "YZ",
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Draw a 2D representation of the optical system.

        Args:
            fields (list[tuple[float, float]] | Literal['all'], optional): The fields to
                be displayed, specified by their indices. Defaults to 'all'.
            wavelengths (list[float] | Literal['primary'], optional): The
                wavelengths to be displayed, specified by their indices.
                Defaults to 'primary'.
            num_rays (int, optional): The number of rays to trace for each
                field and wavelength. Defaults to 3.
            distribution (str | None, optional): The distribution of rays.
                Defaults to None, which selects a default based on projection.
            show_apertures (bool, optional): If True, overlays aperture graphics
                on the system view. Defaults to True.
            hide_vignetted (bool, optional): If True, rays that vignette at any
                surface are not shown. Defaults to False.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (10, 4).
            xlim (tuple[float, float] | None, optional): The x-axis limits of
                the plot. Defaults to None.
            ylim (tuple[float, float] | None, optional): The y-axis limits of
                the plot. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to
                None.
            reference (ReferenceRay | None, optional): The reference rays to
                plot, e.g., 'chief' or 'marginal'. Defaults to None.
            projection (Literal["XY", "XZ", "YZ"], optional): The projection
                plane. Defaults to "YZ".
            ax (matplotlib.axes.Axes, optional): The axes to plot on.
                If None, a new figure and axes are created. Defaults to None.

        Returns:
            tuple[Figure, Axes]: A tuple containing the matplotlib Figure and
            Axes objects of the plot.

        """
        from optiland.visualization import OpticViewer

        viewer = OpticViewer(self)
        fig, ax, _ = viewer.view(
            fields,
            wavelengths,
            num_rays,
            distribution=distribution,
            show_apertures=show_apertures,
            hide_vignetted=hide_vignetted,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            title=title,
            reference=reference,
            projection=projection,
            ax=ax,
        )
        return fig, ax

    def draw3D(
        self,
        fields: list[tuple[float, float]] | Literal["all"] = "all",
        wavelengths: list[float] | Literal["primary"] = "primary",
        num_rays: int = 24,
        distribution: DistributionType = "ring",
        figsize: tuple[float, float] = (1200, 800),
        dark_mode: bool = False,
        reference: ReferenceRay | None = None,
        hide_vignetted: bool = False,
    ):
        """Draw a 3D representation of the optical system.

        Args:
            fields (list[tuple[float, float]] | Literal['all'], optional): The fields to
                be displayed, specified by their indices. Defaults to 'all'.
            wavelengths (list[int] | Literal['primary'], optional): The
                wavelengths to be displayed, specified by their indices.
                Defaults to 'primary'.
            num_rays (int, optional): The number of rays to trace for each
                field and wavelength. Defaults to 24.
            distribution (DistributionType, optional): The distribution of
                rays to trace. Defaults to 'ring'.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (1200, 800).
            dark_mode (bool, optional): If True, use a dark theme for the
                plot. Defaults to False.
            reference (ReferenceRay | None, optional): The reference rays to
                plot, e.g., 'chief' or 'marginal'. Defaults to None.
            hide_vignetted (bool, optional): If True, rays that vignette at any
                surface are not shown. Defaults to False.

        """
        from optiland.visualization import OpticViewer3D

        viewer = OpticViewer3D(self)
        viewer.view(
            fields,
            wavelengths,
            num_rays,
            distribution=distribution,
            figsize=figsize,
            dark_mode=dark_mode,
            reference=reference,
            hide_vignetted=hide_vignetted,
        )

    def info(self):
        """Display the optical system information."""
        from optiland.visualization import LensInfoViewer

        viewer = LensInfoViewer(self)
        viewer.view()

    @deprecated("optic.surfaces.n(wavelength)")
    def n(self, wavelength: float | Literal["primary"] = "primary") -> BEArray:
        """Get the refractive indices of materials at a given wavelength.

        .. deprecated::
            Use ``optic.surfaces.n(wavelength)`` instead.  To resolve the
            primary wavelength use ``optic.primary_wavelength``.

        Args:
            wavelength (float | Literal['primary'], optional): The wavelength
                in microns for which to calculate the refractive indices.
                Can be a float value or 'primary' to use the system's
                primary wavelength. Defaults to 'primary'.

        Returns:
            be.ndarray: An array of refractive indices for each space.

        """
        if wavelength == "primary":
            wavelength = self.primary_wavelength

        return self.surfaces.n(wavelength)

    def trace(
        self,
        Hx: ScalarOrArray,
        Hy: ScalarOrArray,
        wavelength: float,
        num_rays: int | None = 100,
        distribution: DistributionType | BaseDistribution | None = "hexapolar",
    ) -> RealRays:
        """Trace a distribution of rays through the optical system.

        Args:
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            wavelength (float): The wavelength of the rays in microns.
            num_rays: The number of rays to trace.
                Defaults to 100.
            distribution:
                The distribution of rays. Can be a string identifier (e.g.,
                'hexapolar', 'uniform') or a `BaseDistribution` object.
                Defaults to 'hexapolar'.

        Returns:
            RealRays: A `RealRays` object containing the traced rays.

        """
        return self.ray_tracer.trace(Hx, Hy, wavelength, num_rays, distribution)

    def trace_generic(
        self,
        Hx: ScalarOrArray,
        Hy: ScalarOrArray,
        Px: ScalarOrArray,
        Py: ScalarOrArray,
        wavelength: float,
    ):
        """Trace generic rays through the optical system.

        Args:
            Hx: The normalized x field coordinate(s).
            Hy: The normalized y field coordinate(s).
            Px: The normalized x pupil coordinate(s).
            Py: The normalized y pupil coordinate(s).
            wavelength (float): The wavelength of the rays in microns.

        Returns:
            RealRays: A `RealRays` object containing the traced rays.

        """
        return self.ray_tracer.trace_generic(Hx, Hy, Px, Py, wavelength)

    def plot_surface_sag(
        self,
        surface_index: int,
        y_cross_section: float = 0,
        x_cross_section: float = 0,
        fig_to_plot_on: Figure | None = None,
        max_extent: float | None = None,
        num_points_grid: int = 50,
        buffer_factor: float = 1.1,
    ):
        """Analyzes and visualizes the sag of a given lens surface.

        Args:
            surface_index: The index of the surface to analyze.
            y_cross_section: The y-coordinate for the
                x-sag plot. Defaults to 0.
            x_cross_section: The x-coordinate for the
                y-sag plot. Defaults to 0.
        """
        from optiland.visualization import SurfaceSagViewer

        viewer = SurfaceSagViewer(self)
        viewer.view(
            surface_index=surface_index,
            y_cross_section=y_cross_section,
            x_cross_section=x_cross_section,
            fig_to_plot_on=fig_to_plot_on,
            max_extent=max_extent,
            num_points_grid=num_points_grid,
            buffer_factor=buffer_factor,
        )

    def to_dict(self) -> dict:
        """Convert the optical system to a dictionary.

        Returns:
            The dictionary representation of the optical system.

        """
        return OpticSerializer.to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Optic:
        """Create an optical system from a dictionary.

        Args:
            data: The dictionary representation of the optical system.

        Returns:
            The optical system.

        """
        return OpticSerializer.from_dict(data)
