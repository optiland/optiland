"""Gaussian Beam Analysis

This module provides GaussianBeamAnalysis, which carries a paraxial Gaussian
beam surface by surface through an optical system with the system's
ray-transfer (ABCD) matrices. It reports the beam before and after every
surface, the Gouy phase gained along the way, and plots the beam envelope.

References:
    1. A. E. Siegman, Lasers (University Science Books, 1986). Ray matrices:
       chapter 15. Gaussian beams in ABCD systems: chapter 20.

Andrei Utina, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.geometries import BiconicGeometry, ToroidalGeometry
from optiland.paraxial_path import UnsupportedParaxialGeometryError

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.optic import Optic
    from optiland.physical_optics.gaussian_beam import GaussianBeam


ANAMORPHIC_SURFACE = "ANAMORPHIC_SURFACE"


def _curvature(radius):
    """Curvature 1/R, zero for a flat (infinite or zero) radius."""
    radius = be.array(radius)
    return be.where(be.isinf(radius) | (radius == 0), 0.0, 1.0 / radius)


class GaussianBeamAnalysis(BaseAnalysis):
    """Propagate a paraxial Gaussian beam through an optical system.

    Mirrors are handled by unfolding the optical axis, so systems folded by
    plane mirrors work. The analysis accepts the same systems as
    ``optic.paraxial.ray_transfer_matrix``.

    Args:
        optic: The optical system.
        beam: The input beam just before surface 1. Its waist position is
            measured from surface 1. If its medium differs from object space,
            the same waist is used in the object-space medium.

    Raises:
        UnsupportedParaxialGeometryError: If the system is outside the
            first-order engine's domain, or if a surface has different radii
            in x and y (a toroidal or biconic surface), because the beam is
            rotationally symmetric. A powered mirror at oblique incidence only
            gives a ``ParaxialDomainWarning``, because its astigmatism is
            ignored.
    """

    def __init__(self, optic: Optic, beam: GaussianBeam) -> None:
        self.beam = beam
        super().__init__(optic, wavelengths=[beam.wavelength])

    def _generate_data(self) -> dict[str, list]:
        """Propagate the beam and record it at every surface.

        Returns:
            dict: ``"before"`` and ``"after"`` hold the beam at each surface,
                from surface 1 to the image. ``"gouy_phase"`` holds the Gouy
                phase gained since surface 1 in radians, and ``"z"`` the
                unfolded distance from surface 1 in mm.
        """
        self._check_rotational_symmetry()
        wavelength = self.beam.wavelength
        path = self.optic.surfaces.build_paraxial_path()
        positions = be.ravel(path.axial_positions)
        n = be.ravel(self.optic.surfaces.n(wavelength))

        beam = self.beam
        if not bool(be.allclose(beam.refractive_index, n[0])):
            beam = beam.with_refractive_index(n[0])

        data = {"before": [], "after": [], "gouy_phase": [], "z": []}
        # Zeros with the backend's dtype and device.
        gouy = n[0] * 0.0
        z = n[0] * 0.0
        for k in range(1, self.optic.surfaces.num_surfaces):
            if k > 1:
                # After a mirror the axial coordinate runs backwards; the
                # parity makes the unfolded leg length positive again.
                length = (positions[k] - positions[k - 1]) * path.parity_after[k - 1]
                gouy = gouy + beam.gouy_phase(length)
                beam = beam.propagate(length)
                z = z + length
            data["before"].append(beam)

            # Flip the slope signs that a reflection introduces, so the matrix
            # acts in the unfolded frame.
            matrix = self.optic.paraxial.ray_transfer_matrix(
                k, k, wavelength, path=path
            )
            s_in = path.parity_before[k]
            s_out = path.parity_after[k]
            unfolded = [
                [matrix[0][0], matrix[0][1] * s_in],
                [matrix[1][0] * s_out, matrix[1][1] * s_in * s_out],
            ]
            beam = beam.transform(unfolded, refractive_index=n[k])
            data["after"].append(beam)
            data["gouy_phase"].append(gouy)
            data["z"].append(z)
        return data

    def _check_rotational_symmetry(self) -> None:
        """Reject surfaces that focus differently in x and y.

        The first-order engine keeps only one of the two radii of a toroidal
        or biconic surface, so a rotationally symmetric beam would silently
        describe one plane only.

        Raises:
            UnsupportedParaxialGeometryError: For the first surface whose x
                and y curvatures differ.
        """
        for k in range(1, self.optic.surfaces.num_surfaces):
            geometry = self.optic.surfaces[k].geometry
            if isinstance(geometry, ToroidalGeometry):
                radius_x, radius_y = geometry.R_rot, geometry.R_yz
            elif isinstance(geometry, BiconicGeometry):
                radius_x, radius_y = geometry.Rx, geometry.Ry
            else:
                continue
            if not bool(be.allclose(_curvature(radius_x), _curvature(radius_y))):
                name = type(geometry).__name__
                rx = float(be.to_numpy(radius_x))
                ry = float(be.to_numpy(radius_y))
                raise UnsupportedParaxialGeometryError(
                    f"[{ANAMORPHIC_SURFACE}] surface {k}: the {name} "
                    f"has different radii in x ({rx:g} mm) and y ({ry:g} mm), "
                    "so it focuses differently in the two planes. "
                    "GaussianBeamAnalysis models rotationally symmetric beams "
                    "only; use real ray tracing for this system."
                )

    def beam_at(self, surface_number: int, side: str = "after") -> GaussianBeam:
        """Return the beam at a surface.

        Args:
            surface_number: Surface index, from 1 to the image surface.
                Negative values count from the end, so -1 is the image.
            side: ``"before"`` or ``"after"`` the surface.

        Returns:
            The beam at that surface.

        Raises:
            ValueError: If the surface index or the side is not valid.
        """
        if side not in ("before", "after"):
            raise ValueError('side must be "before" or "after".')
        num_surfaces = self.optic.surfaces.num_surfaces
        index = num_surfaces + surface_number if surface_number < 0 else surface_number
        if not 1 <= index < num_surfaces:
            raise ValueError(
                f"surface_number must be between 1 and {num_surfaces - 1} "
                f"(or negative), got {surface_number}."
            )
        return self.data[side][index - 1]

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (8, 4),
        *,
        show: bool = True,
        num_points: int = 200,
    ) -> tuple[Figure, Axes]:
        """Plot the beam radius along the unfolded optical axis.

        Args:
            fig_to_plot_on: Existing figure to plot on. Creates a new figure
                if None.
            figsize: Figure size when a new figure is created.
            show: If True (default), calls plt.show(). Set False for headless
                use.
            num_points: Number of samples on each leg between surfaces.

        Returns:
            tuple: Matplotlib Figure and Axes objects.
        """
        is_embedding = fig_to_plot_on is not None
        if is_embedding:
            fig = fig_to_plot_on
            fig.clear()
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(figsize=figsize)

        z, w = self._envelope(num_points)
        ax.plot(z, w * 1e3, color="C0")
        ax.plot(z, -w * 1e3, color="C0")
        for z_surface in self.data["z"]:
            ax.axvline(float(be.to_numpy(z_surface)), color="0.8", linewidth=0.8)
        ax.set_xlabel("Unfolded distance from surface 1 (mm)")
        ax.set_ylabel("Beam radius (µm)")
        ax.set_title(
            f"Gaussian beam (λ = {self.beam.wavelength:.4f} µm, M² = {self.beam.m2:g})"
        )
        fig.tight_layout()

        if is_embedding and hasattr(fig, "canvas"):
            fig.canvas.draw_idle()
        elif show:
            plt.show()
        return fig, ax

    def _envelope(self, num_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample the beam radius on every leg between surfaces."""
        z_parts, w_parts = [], []
        for i in range(len(self.data["after"]) - 1):
            beam = self.data["after"][i]
            z0 = float(be.to_numpy(self.data["z"][i]))
            z1 = float(be.to_numpy(self.data["z"][i + 1]))
            t = np.linspace(0.0, z1 - z0, num_points)
            x = float(be.to_numpy(beam.q_real)) + t
            y = float(be.to_numpy(beam.q_imag))
            w0 = float(be.to_numpy(beam.waist_radius))
            z_parts.append(z0 + t)
            w_parts.append(w0 * np.sqrt(1 + (x / y) ** 2))
        if not z_parts:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(z_parts), np.concatenate(w_parts)
