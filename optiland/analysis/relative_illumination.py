"""Relative Illumination Analysis

This module provides a relative illumination analysis for optical systems,
based on M. Rimmer, "Relative illumination calculations", Proc. SPIE 655, 99
(1986).

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.geometries import Plane

from ._plotting import finalize_figure, prepare_axes
from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.optic import Optic


class RelativeIllumination(BaseAnalysis):
    """Relative illumination: image irradiance versus field, normalized.

    For a uniform, planar, Lambertian object, radiance is conserved through
    the system apart from transmission losses. The irradiance at an image
    point is then proportional to the projected solid angle of the exit pupil
    seen from that point, weighted by transmission. In image space, this
    projected solid angle is the area that the rays cover in direction-cosine
    space ``(L, M)``, measured in the image surface's local frame (Rimmer,
    1986). Vignetting, pupil aberrations, telecentricity and changes of
    F-number across the field all change that area, so no separate cos⁴ term
    is applied.

    For each field point ``Hy`` in ``linspace(0, 1, num_fields)`` (``Hx = 0``),
    rays are traced on a polar grid of normalized pupil coordinates that covers
    the unit disk. Each grid cell maps to a quadrilateral in ``(L, M)``. Its
    signed area (shoelace formula) is weighted by the mean intensity of its
    four corner rays, with vignetted rays counted as zero intensity. This
    captures coatings, apodization and absorption, and gives a cell that is
    partially vignetted a proportional share of its area. The sum over cells
    is the unnormalized irradiance. Each wavelength's curve is normalized by
    its maximum over the field.

    The grid is polar, not a square grid masked to the unit disk, because
    the area of a masked square grid has a staircase boundary whose error
    falls only as ``1 / num_pupil``. The polar grid follows the pupil circle,
    and the remaining chord error is divided out, so its error falls as
    ``1 / num_pupil**2``. The integrated area does not depend on how the pupil
    disk is parameterized, so both grids converge to the same value.

    If the signed cell areas of one field change sign, rays from different
    pupil regions share an image-space direction (a caustic in direction
    space), and the area no longer measures the irradiance. A
    ``RuntimeWarning`` names those fields, and ``data["reliable"]`` marks
    them ``False``.

    The pupil grid follows the optic's ray aiming. With paraxial aiming, the
    off-axis beams fill the paraxial entrance pupil, which can under- or
    overfill the real stop when pupil aberrations are large. Iterative or
    robust aiming (``optic.ray_tracer.set_aiming``) fills the real stop and
    can change the result by several percent or more for such lenses.

    The method assumes that the image surface is a good conjugate of the
    object: rays from one field point should land close together on it.
    Strong defocus, field curvature or astigmatism at the image surface make
    the result differ from the true irradiance, because the cone of rays from
    one field point is then not the cone of rays arriving at one image point.

    Only planar image surfaces are supported. A tilted planar image surface is
    handled through its local frame.

    Args:
        optic (Optic): The optic object to analyze.
        wavelengths (str | list, optional): The wavelengths to analyze.
            Defaults to 'primary'.
        num_fields (int, optional): The number of field points from the axis
            to the full field. Defaults to 32.
        num_pupil (int, optional): The pupil sampling, the number of rays
            across the pupil diameter. As for a square grid masked to the
            unit disk, about ``num_pupil**2 * pi / 4`` rays are traced per
            field point. Defaults to 64. Must be at least 5.

    Attributes:
        optic (Optic): The optic object being analyzed.
        wavelengths (list): The wavelengths being analyzed.
        num_fields (int): The number of field points.
        num_pupil (int): The pupil sampling.
        data (dict): ``"Hy"``, the normalized field coordinates;
            ``"relative_illumination"``, one array per wavelength (same order
            as ``wavelengths``) with values in ``[0, 1]``; ``"irradiance"``,
            the unnormalized values (projected solid angle in sr, weighted by
            transmission); and ``"reliable"``, one boolean array per
            wavelength, ``False`` where a caustic was detected.

    Raises:
        NotImplementedError: If the image surface is not planar.
        ValueError: If ``num_fields`` is less than 1 or ``num_pupil`` is less
            than 5.
    """

    def __init__(
        self,
        optic: Optic,
        wavelengths: str | list = "primary",
        num_fields: int = 32,
        num_pupil: int = 64,
    ):
        if num_fields < 1:
            raise ValueError(f"num_fields must be at least 1, got {num_fields}.")
        if num_pupil < 5:
            raise ValueError(f"num_pupil must be at least 5, got {num_pupil}.")
        if not isinstance(optic.image_surface.geometry, Plane):
            raise NotImplementedError(
                "RelativeIllumination supports planar image surfaces only."
            )
        self.num_fields = num_fields
        self.num_pupil = num_pupil
        super().__init__(optic, wavelengths)

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (8, 5.5),
        *,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """Displays a plot of the relative illumination versus field.

        Fields flagged as unreliable (caustic detected) are marked with an
        ``x``.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (8, 5.5).
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            tuple: The current figure and its axes.
        """
        current_fig, ax, is_gui_embedding = prepare_axes(fig_to_plot_on, figsize)

        field_np = be.to_numpy(self.data["Hy"]) * self.optic.fields.max_field

        for k, wp in enumerate(self.wavelengths):
            ri = be.to_numpy(self.data["relative_illumination"][k])
            reliable = be.to_numpy(self.data["reliable"][k]).astype(bool)
            ax.plot(field_np, ri, f"C{k}", zorder=10, label=f"{wp.value:.4f} µm")
            if not reliable.all():
                ax.plot(
                    field_np[~reliable],
                    ri[~reliable],
                    f"C{k}x",
                    zorder=11,
                    label=f"{wp.value:.4f} µm, caustic",
                )

        ax.set_xlabel("Field")
        ax.set_ylabel("Relative Illumination")
        ax.set_xlim([0, self.optic.fields.max_field])
        ax.set_ylim([0, 1.05])
        ax.set_title("Relative Illumination")
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.grid(True)
        current_fig.tight_layout()

        finalize_figure(current_fig, is_gui_embedding, show)
        return current_fig, ax

    def _generate_data(self) -> dict:
        """Computes the relative illumination for each wavelength.

        Returns:
            dict: ``"Hy"``, ``"relative_illumination"``, ``"irradiance"`` and
                ``"reliable"``, as described in the class docstring.
        """
        Hy = be.linspace(0, 1, self.num_fields)
        relative, irradiance, reliable = [], [], []
        for wp in self.wavelengths:
            e, ok = self._irradiance(Hy, wp.value)
            peak = be.max(e)
            has_light = float(be.to_numpy(peak)) > 0
            relative.append(e / peak if has_light else be.zeros_like(e))
            irradiance.append(e)
            reliable.append(ok)
            _warn_caustics(Hy, ok, wp.value)
        return {
            "Hy": Hy,
            "relative_illumination": relative,
            "irradiance": irradiance,
            "reliable": reliable,
        }

    def _irradiance(self, Hy, wavelength: float):
        """Weighted direction-cosine area for every field point.

        Args:
            Hy (be.ndarray): Normalized y field coordinates.
            wavelength (float): Wavelength in µm.

        Returns:
            tuple: ``(irradiance, reliable)``, one value per field point.
        """
        grid = _PolarPupilGrid(self.num_pupil)
        num_fields = Hy.shape[0]
        num_rays = grid.Px.shape[0]

        Hy_all = be.repeat(Hy, num_rays)
        rays = self.optic.trace_generic(
            be.zeros_like(Hy_all),
            Hy_all,
            be.tile(grid.Px, num_fields),
            be.tile(grid.Py, num_fields),
            wavelength,
        )
        # The returned rays are a fresh object, so localizing them leaves the
        # recorded per-surface data in the global frame.
        self.optic.image_surface.geometry.localize(rays)

        shape = (num_fields, num_rays)
        L = grid.cells(be.reshape(rays.L, shape))
        M = grid.cells(be.reshape(rays.M, shape))
        intensity = grid.cells(be.reshape(be.where(rays.i > 0, rays.i, 0.0), shape))

        # Corners in order around each cell; the diagonal form of the
        # shoelace formula gives the signed area of the quadrilateral.
        L0, L1, L2, L3 = L
        M0, M1, M2, M3 = M
        area = 0.5 * ((L2 - L0) * (M3 - M1) - (L3 - L1) * (M2 - M0))
        weight = 0.25 * (intensity[0] + intensity[1] + intensity[2] + intensity[3])

        finite = be.isfinite(area)
        area = be.reshape(be.where(finite, area, 0.0), (num_fields, -1))
        weight = be.reshape(be.where(finite, weight, 0.0), (num_fields, -1))

        irradiance = be.abs(be.sum(area * weight, axis=1)) * grid.chord_correction
        return irradiance, _no_sign_change(area, weight)


class _PolarPupilGrid:
    """Polar grid of normalized pupil coordinates over the unit disk.

    The grid has ``num_rho`` radial and ``num_theta`` angular intervals. Each
    point is traced once: the pupil center, then ``num_theta`` points on each
    ring. :meth:`cells` rebuilds the closed ``(num_rho + 1) x (num_theta + 1)``
    grid from per-ray values and returns the corners of every cell.

    Args:
        num_pupil (int): Rays across the pupil diameter. The grid uses
            ``num_pupil // 2`` rings of ``round(pi * num_pupil / 2)`` points,
            about ``num_pupil**2 * pi / 4`` rays in all.

    Attributes:
        Px (be.ndarray): Normalized pupil x coordinates of the traced rays.
        Py (be.ndarray): Normalized pupil y coordinates of the traced rays.
        num_rho (int): Number of radial intervals.
        num_theta (int): Number of angular intervals.
        chord_correction (float): ``d / sin(d)`` for the angular step ``d``.
            The outer cells replace the pupil circle by chords, which loses
            the factor ``sin(d) / d`` for a locally linear pupil-to-direction
            map.
    """

    def __init__(self, num_pupil: int):
        self.num_rho = max(2, num_pupil // 2)
        self.num_theta = max(8, round(math.pi * num_pupil / 2))
        step = 2 * math.pi / self.num_theta
        self.chord_correction = step / math.sin(step)

        rho = be.linspace(0, 1, self.num_rho + 1)[1:]
        theta = be.arange(self.num_theta) * step
        theta, rho = be.meshgrid(theta, rho)
        # Clip rounding overshoot so the coordinates stay within [-1, 1].
        zero = be.zeros(1)
        self.Px = be.concatenate([zero, be.ravel(be.clip(rho * be.cos(theta), -1, 1))])
        self.Py = be.concatenate([zero, be.ravel(be.clip(rho * be.sin(theta), -1, 1))])

    def cells(self, values):
        """Corners of every grid cell, in order around the cell.

        Args:
            values (be.ndarray): Per-ray values of shape ``(fields, rays)``,
                in the order of ``Px`` and ``Py``.

        Returns:
            tuple: Four arrays of shape ``(fields, num_rho, num_theta)`` for
            the corners ``(i, j)``, ``(i + 1, j)``, ``(i + 1, j + 1)`` and
            ``(i, j + 1)``, with ``i`` the ring and ``j`` the angle index.
        """
        num_fields = values.shape[0]
        center = values[:, :1, None] * be.ones((1, 1, self.num_theta))
        rings = be.reshape(values[:, 1:], (num_fields, self.num_rho, self.num_theta))
        grid = be.concatenate([center, rings], axis=1)
        # Repeat the first angle at the end to close each ring.
        grid = be.concatenate([grid, grid[:, :, :1]], axis=2)
        return (
            grid[:, :-1, :-1],
            grid[:, 1:, :-1],
            grid[:, 1:, 1:],
            grid[:, :-1, 1:],
        )


def _no_sign_change(area, weight, rel_tol: float = 1e-6):
    """Whether the signed cell areas of each field share one sign.

    Cells that carry no light, and cells far smaller than the largest cell of
    their field (such as the degenerate cells at the pupil center), are
    ignored.

    Args:
        area (be.ndarray): Signed cell areas, shape ``(fields, cells)``.
        weight (be.ndarray): Cell weights, same shape.
        rel_tol (float, optional): Relative area below which a cell is
            ignored. Defaults to 1e-6.

    Returns:
        be.ndarray: One boolean per field, ``True`` without a sign change.
    """
    lit = weight > 0
    largest = be.nanmax(be.where(lit, be.abs(area), 0.0), axis=1)
    counted = lit & (be.abs(area) > rel_tol * largest[:, None])
    positive = be.sum(be.where(counted & (area > 0), 1.0, 0.0), axis=1)
    negative = be.sum(be.where(counted & (area < 0), 1.0, 0.0), axis=1)
    return (positive == 0) | (negative == 0)


def _warn_caustics(Hy, reliable, wavelength: float) -> None:
    """Emits a ``RuntimeWarning`` naming the fields with a caustic.

    Args:
        Hy (be.ndarray): Normalized y field coordinates.
        reliable (be.ndarray): One boolean per field.
        wavelength (float): Wavelength in µm.
    """
    flags = be.to_numpy(reliable).astype(bool)
    if flags.all():
        return
    bad = np.round(be.to_numpy(Hy)[~flags], 4).tolist()
    warnings.warn(
        f"RelativeIllumination: the direction-cosine mapping folds over "
        f"(caustic) at {wavelength:.4f} µm for Hy = {bad}. Relative "
        f"illumination is unreliable at these fields.",
        RuntimeWarning,
        stacklevel=4,
    )
