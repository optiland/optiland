"""Optic Updater Module

This module contains the OpticModifier class, which is responsible for updating the
optical system properties, such as the surface radii of curvature, thicknesses,
materials, conic constants, polarization, etc.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.apodization import BaseApodization
from optiland.geometries import Plane, StandardGeometry
from optiland.materials import IdealMaterial

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays import PolarizationState


class OpticUpdater:
    """Class to update or modify an optical system

    This class is responsible for updating the optical system properties, such as
    the surface radii of curvature, thicknesses, materials, conic constants,
    polarization, etc.

    Args:
        optic (Optic): The optical system to be modified.
    """

    def __init__(self, optic):
        self.optic = optic

    def set_radius(self, value, surface_number):
        """Set the radius of curvature of a surface.

        Args:
            value (float): The new radius of curvature.
            surface_number (int): The index of the surface to modify.

        """
        surface = self.optic.surfaces[surface_number]

        # change geometry from plane to standard
        if isinstance(surface.geometry, Plane):
            cs = surface.geometry.cs
            new_geometry = StandardGeometry(cs, radius=value, conic=0)
            surface.geometry = new_geometry
        else:
            surface.geometry.radius = value

    def set_conic(self, value, surface_number):
        """Set the conic constant of a surface.

        Args:
            value (float): The new conic constant.
            surface_number (int): The index of the surface to modify.

        """
        surface = self.optic.surfaces[surface_number]
        surface.geometry.k = value

    def set_thickness(self, value, surface_number):
        """Set the thickness of a surface.

        Args:
            value (float): The new thickness value to set for the space
                following the specified surface.
            surface_number (int): The index of the surface *before* the
                thickness to be modified.

        """
        if surface_number == 0:
            # First surface thickness sets the object distance.
            # We treat this specially to avoid issues with infinite values.
            self.optic.surfaces[0].thickness = value
            self.optic.surfaces[0].geometry.cs.z = be.array(-value)
            # No need to shift other surfaces as they are relative to S1 at z=0
            return

        # Source of truth for downstream positions is each surface's `.thickness`
        # attribute. Update the requested one first, then rebuild every cs.z
        # downstream from those thickness values (rather than incrementally
        # mutating cs.z). This keeps every current-iteration thickness tensor
        # in the autograd graph — the prior incremental form needed `detach()`
        # to drop stale graphs from earlier iterations, but that also severed
        # in-iteration grad paths when multiple thickness variables were
        # updated sequentially (only the last one kept its gradient). See #569.
        surfaces = self.optic.surfaces
        n = len(surfaces)
        if surface_number < n:
            surfaces[surface_number].thickness = value

        # Surface 1 anchored at z=0; cs.z[k] = sum of upstream thicknesses.
        if n >= 2:
            z = be.array(0.0)
            surfaces[1].geometry.cs.z = be.array(z)
            for k in range(2, n):
                t_prev = surfaces[k - 1].thickness
                z = z + (t_prev if hasattr(t_prev, "detach") else be.array(t_prev))
                surfaces[k].geometry.cs.z = be.array(z)

    def set_index(self, value: float, surface_number: int) -> None:
        """Set the index of refraction of a surface.

        Args:
            value (float): The new refractive index value.
            surface_number (int): The index of the surface whose *post-material*
                (material after the surface) will be updated. This also updates
                the *pre-material* of the subsequent surface.

        """
        new_material = IdealMaterial(n=value, k=0)
        self.set_material(new_material, surface_number)

    def set_material(self, material: BaseMaterial, surface_number: int) -> None:
        """Set the material of a surface.

        Args:
            value (BaseMaterial): The new material.
            surface_number (int): The material of the surface whose *post-material*
                (material after the surface) will be updated. This also updates
                the *pre-material* of the subsequent surface.

        """
        surface = self.optic.surfaces[surface_number]
        surface.material_post = material

    def set_norm_radius(self, value, surface_number, is_fixed=True):
        """Set the normalization radius on a surface.

        Args:
            value (float): The new value for the normalization radius.
            surface_number (int): The index of the surface to modify.
            is_fixed (bool, optional): Whether to lock the normalization radius
                from automatic paraxial updates. Defaults to True.
        """
        surface = self.optic.surfaces[surface_number]
        if hasattr(surface.geometry, "norm_radius"):
            surface.geometry.norm_radius = value
            surface.geometry.normalization_mode = "manual" if is_fixed else "auto"
        else:
            # This error is useful for debugging if used on an incorrect surface type
            raise AttributeError(
                f"Surface {surface_number} with geometry type "
                f"'{surface.geometry.__class__.__name__}' has no "
                "'norm_radius' attribute."
            )

    def set_asphere_coeff(self, value, surface_number, aspher_coeff_idx):
        """Set the asphere coefficient on a surface

        Args:
            value (float): The new value for the aspheric coefficient.
            surface_number (int): The index of the surface to modify.
            aspher_coeff_idx (int): The index of the aspheric coefficient
                within the surface's coefficient list to set.

        """
        surface = self.optic.surfaces[surface_number]
        surface.geometry.coefficients[aspher_coeff_idx] = value

    def set_polarization(self, polarization: PolarizationState | str):
        """Set the polarization state of the optic.

        Args:
            polarization (PolarizationState or str): The polarization
                state to set. Can be a `PolarizationState` object or the string
                'ignore'.

        """
        if isinstance(polarization, str) and polarization != "ignore":
            raise ValueError(
                "Invalid polarization state. Must be either "
                'PolarizationState or "ignore".',
            )
        self.optic.polarization = polarization

    def scale_system(self, scale_factor):
        """Scales the optical system by a given scale factor.

        Args:
            scale_factor (float): The factor by which to scale all relevant
                system dimensions (radii, thicknesses, EPD, physical apertures).

        """
        num_surfaces = self.optic.surfaces.num_surfaces
        thicknesses = [
            self.optic.surfaces.get_thickness(surf_idx)[0]
            for surf_idx in range(num_surfaces - 1)
        ]

        # Scale radii, geometries, and thicknesses
        for surf_idx in range(num_surfaces):
            surface = self.optic.surfaces[surf_idx]
            surface.geometry.scale(scale_factor)

            if surf_idx != num_surfaces - 1 and not be.isinf(thicknesses[surf_idx]):
                self.set_thickness(thicknesses[surf_idx] * scale_factor, surf_idx)

        # Scale aperture if the aperture type supports scaling
        if self.optic.aperture and self.optic.aperture.is_scalable:
            self.optic.aperture = self.optic.aperture.scale(scale_factor)

        # Scale physical apertures
        for surface in self.optic.surfaces:
            if surface.aperture is not None:
                surface.aperture.scale(scale_factor)

    def update_paraxial(self):
        """Update the semi-aperture of all surfaces based on paraxial marginal
        and chief ray heights. Also updates normalization radii for relevant
        surface types.
        """
        ya, _ = self.optic.paraxial.marginal_ray()
        yb, _ = self.optic.paraxial.chief_ray()
        ya = be.abs(be.ravel(ya))
        yb = be.abs(be.ravel(yb))
        for k, surface in enumerate(self.optic.surfaces):
            r_max = ya[k] + yb[k]
            if surface.aperture is not None:
                extent_max = be.max(be.abs(be.array(surface.aperture.extent)))
                if be.isfinite(be.array(extent_max)):
                    r_max = be.max(be.array([r_max, extent_max]))

            surface.set_semi_aperture(r_max=r_max)
            self.update_normalization(surface)

    def update_normalization(self, surface) -> None:
        """Update the normalization radius/factors of a given non-spherical surface.

        Args:
            surface (Surface): The surface whose normalization factors are to be
                updated.
        """
        # Skip updating normalization when the normalization radius is a variable
        if getattr(surface, "is_norm_radius_variable", False):
            return

        # Delegate updating normalization directly to the geometry
        surface.geometry.update_normalization(surface.semi_aperture)

    def update(self) -> None:
        """Update the optical system by applying all defined pickups and solves.
        If certain surface types requiring paraxial updates are present,
        `update_paraxial()` is also called.
        """
        self.optic.pickups.apply()
        self.optic.solves.apply()

        if any(
            surface.surface_type
            in ["chebyshev", "zernike", "forbes_qbfs", "forbes_q2d"]
            for surface in self.optic.surfaces
        ):
            self.update_paraxial()

    def image_solve(self):
        """Adjusts the position of the image surface (last surface) such that
        the paraxial marginal ray crosses the optical axis at this new location.
        This effectively sets the paraxial focus.
        """
        ya, ua = self.optic.paraxial.marginal_ray()
        offset = float(ya[-1, 0] / ua[-1, 0])
        surfaces = self.optic.surfaces
        self.optic.surfaces[-1].geometry.cs.z -= offset
        surfaces[-2].thickness = (
            self.optic.surfaces[-1].geometry.cs.z - surfaces[-2].geometry.cs.z
        )

    def flip(self):
        """Flips the optical system, reversing the order of surfaces (excluding
        object and image planes), their geometries, and materials. Pickups and
        solves referencing surface indices are updated accordingly.
        The new first optical surface (originally the last) is placed at z=0.0.
        """
        if self.optic.surfaces.num_surfaces < 3:
            raise ValueError(
                "Optic flip requires at least 3 surfaces (obj, element, img)"
            )

        # 1. Call SurfaceGroup.flip()
        self.optic.surfaces.flip()

        # 2. Define remapping function for indices
        num_surfaces = self.optic.surfaces.num_surfaces

        def remap_index_func(old_idx):  # pragma: no cover
            if old_idx == 0 or old_idx == num_surfaces - 1:  # Object or Image surface
                return old_idx
            if 1 <= old_idx <= num_surfaces - 2:
                return num_surfaces - 1 - old_idx
            return old_idx  # Should not happen if indices are valid

        # 3. Handle Pickups
        if self.optic.pickups and len(self.optic.pickups.pickups) > 0:
            self.optic.pickups.remap_surface_indices(remap_index_func)

        # 4. Handle Solves
        if (
            hasattr(self.optic, "solves")
            and self.optic.solves
            and hasattr(self.optic.solves, "solves")
            and len(self.optic.solves.solves) > 0
        ):
            self.optic.solves.remap_surface_indices(remap_index_func)

        # 5. Update Optic instance
        self.update()

    def set_apodization(
        self, apodization: BaseApodization | str | dict = None, **kwargs
    ):
        """Sets the apodization for the optical system.

        This method supports setting the apodization in multiple ways:
        1. By providing an instance of a `BaseApodization` subclass.
        2. By providing a string identifier (e.g., "GaussianApodization")
           and keyword arguments for its parameters.
        3. By providing a dictionary that can be passed to `from_dict`.
        4. By passing `None` to remove any existing apodization.

        Args:
            apodization (BaseApodization | str | dict, optional): The
                apodization to apply. Defaults to None.
            **kwargs: Additional keyword arguments used to initialize the
                apodization class when `apodization` is a string.

        Raises:
            TypeError: If the provided `apodization` is not a supported type.
            ValueError: If the string identifier is not found in the registry.
        """
        if apodization is None:
            self.optic.apodization = None
        elif isinstance(apodization, BaseApodization):
            self.optic.apodization = apodization
        elif isinstance(apodization, str):
            if apodization in BaseApodization._registry:
                apodization_class = BaseApodization._registry[apodization]
                self.optic.apodization = apodization_class(**kwargs)
            else:
                raise ValueError(f"Unknown apodization type: {apodization}")
        elif isinstance(apodization, dict):
            self.optic.apodization = BaseApodization.from_dict(apodization)
        else:
            raise TypeError(
                "apodization must be a string, a dict, a BaseApodization "
                "instance, or None."
            )
