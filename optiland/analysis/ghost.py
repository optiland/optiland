"""Ghost Analysis

This module enumerates the two-bounce ghost paths of a refractive optical
system and ranks them by how bright they are at the image surface.

A two-bounce ghost is light that reflects off surface ``j``, travels back
toward the object, reflects again off an earlier surface ``i < j``, and then
continues forward to the image. Each path is traced with real rays as a
:class:`~optiland.sequences.sequenced_optic.SequencedOptic`, so the base optic
is never modified.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.sequences import SequencedOptic

from .base import BaseAnalysis

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from optiland.optic import Optic
    from optiland.sequences.steps import RawStep


class GhostAnalysis(BaseAnalysis):
    """Two-bounce ghost analysis: ghost focus, spot size and irradiance.

    Every pair of reflecting surfaces ``(i, j)`` with ``0 < i < j`` gives one
    ghost path: forward to ``j``, reflect, back to ``i``, reflect, forward to
    the image. For each path, a hexapolar ray bundle at one field point is
    traced, and the following are computed at the image surface:

    - ``reflectance``: ``R_i * R_j``, the product of the two surface
      reflectances.
    - ``transmitted_fraction``: the fraction of launched rays that reach the
      image surface along the ghost path (the rest are clipped by apertures or
      fail to intersect a surface).
    - ``energy_fraction``: ``reflectance * transmitted_fraction``, the share
      of the pupil flux carried by the ghost.
    - ``rms_radius``: the RMS radius of the ghost spot about its centroid on
      the image surface, in lens units.
    - ``focus_shift``: the axial distance from the image surface to the plane
      where the ghost spot is smallest, along the global z axis. Positive
      means past the image surface. It is the least-squares focus of the
      straight-line extension of the ghost rays, so it is valid on either
      side of the image surface.
    - ``irradiance``: ``energy_fraction / (pi * r**2)``, the mean ghost
      irradiance on the image surface per unit pupil flux, in inverse lens
      units squared. ``r`` is ``rms_radius``, floored at the Airy radius
      ``1.22 * wavelength * FNO`` so that a ghost focused on the image does
      not give an infinite value. Use it to rank ghosts against each other.

    Ghosts are sorted by ``irradiance``, brightest first. A ghost with fewer
    than two rays at the image surface has NaN ``rms_radius`` and
    ``focus_shift``, and zero ``irradiance``.

    By default, each surface's reflectance is the uncoated Fresnel
    reflectance at normal incidence, ``((n1 - n2) / (n1 + n2)) ** 2``, from
    the refractive indices on either side of it. Coatings on the surfaces are
    not read. Pass ``reflectance`` to set the values yourself, for example to
    model anti-reflection coatings or a reflective sensor.

    Args:
        optic (Optic): The optic to analyze. It must not contain reflective
            surfaces (mirrors).
        field (tuple[float, float], optional): The normalized field
            coordinates ``(Hx, Hy)``. Defaults to ``(0.0, 0.0)``.
        wavelength (str or float, optional): The wavelength in µm, or
            ``"primary"``. Defaults to ``"primary"``.
        num_rings (int, optional): The number of rings of the hexapolar ray
            bundle traced for each ghost. Defaults to 8.
        reflectance (float or dict[int, float], optional): The reflectance of
            the surfaces. A float applies to every surface between the object
            and the image surfaces. A dict maps surface indices to their
            reflectance, and only those surfaces are considered; it may
            include the image surface, to model light reflected by a sensor.
            If None, the uncoated Fresnel reflectance is used. Defaults to
            None.

    Attributes:
        optic (Optic): The optic being analyzed.
        field (tuple[float, float]): The normalized field coordinates.
        num_rings (int): The number of rings of the ray bundle.
        data (dict): The ghost data. ``"surfaces"`` is a list of ``(i, j)``
            tuples. The other keys are ``"reflectance"``,
            ``"transmitted_fraction"``, ``"energy_fraction"``,
            ``"rms_radius"``, ``"focus_shift"`` and ``"irradiance"``, each a
            1D array with one entry per ghost, in the order of
            ``"surfaces"``.

    Raises:
        ValueError: If the optic contains a reflective surface, if a
            reflectance is outside ``[0, 1]``, or if a surface index in
            ``reflectance`` is out of range.
    """

    def __init__(
        self,
        optic: Optic,
        field: tuple[float, float] = (0.0, 0.0),
        wavelength: str | float = "primary",
        num_rings: int = 8,
        reflectance: float | dict[int, float] | None = None,
    ):
        self.field = field
        self.num_rings = num_rings
        self._reflectance = reflectance
        wavelengths = wavelength if wavelength == "primary" else [wavelength]
        super().__init__(optic, wavelengths=wavelengths)

    def sequence(self, i: int, j: int) -> SequencedOptic:
        """Builds the ghost path that reflects off surface ``j``, then ``i``.

        The returned sequence can be passed to other analyses, for example
        ``SpotDiagram``, to inspect one ghost in detail. It is not registered
        on the optic.

        Args:
            i (int): The index of the second reflecting surface.
            j (int): The index of the first reflecting surface, ``j > i``.

        Returns:
            SequencedOptic: The ghost path.

        Raises:
            ValueError: If the indices do not satisfy
                ``0 < i < j < num_surfaces``.
        """
        num_surfaces = self.optic.surfaces.num_surfaces
        if not 0 < i < j < num_surfaces:
            raise ValueError(
                "A ghost path needs 0 < i < j < num_surfaces, got "
                f"i={i}, j={j} with num_surfaces={num_surfaces}."
            )
        return SequencedOptic(
            self.optic, f"ghost_{i}_{j}", self._ghost_steps(i, j, num_surfaces)
        )

    def view(
        self,
        fig_to_plot_on: Figure | None = None,
        figsize: tuple[float, float] = (10, 4.5),
        *,
        max_ghosts: int = 20,
        show: bool = True,
    ) -> tuple[Figure, np.ndarray]:
        """Displays the brightest ghosts: irradiance and focus shift.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple[float, float], optional): The size of the figure.
                Defaults to (10, 4.5).
            max_ghosts (int, optional): The maximum number of ghosts shown,
                brightest first. Defaults to 20.
            show (bool): If True (default), calls plt.show(). Set False for
                headless use.

        Returns:
            tuple: The current figure and an array of its two axes.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            axes = np.array(current_fig.subplots(1, 2))
        else:
            current_fig, axes = plt.subplots(1, 2, figsize=figsize)

        count = min(max_ghosts, len(self.data["surfaces"]))
        labels = [f"{i}-{j}" for i, j in self.data["surfaces"][:count]]
        positions = np.arange(count)
        irradiance = be.to_numpy(self.data["irradiance"])[:count]
        focus_shift = be.to_numpy(self.data["focus_shift"])[:count]

        axes[0].bar(positions, irradiance, color="C0")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Relative irradiance (1/lens unit²)")
        axes[0].set_title("Ghost irradiance at the image")

        axes[1].bar(positions, focus_shift, color="C1")
        axes[1].axhline(y=0, color="k", linewidth=0.5)
        axes[1].set_ylabel("Focus shift (lens units)")
        axes[1].set_title("Ghost focus from the image surface")

        for ax in axes:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=90)
            ax.set_xlabel("Surfaces (i-j)")
            ax.grid(True, axis="y")
        current_fig.tight_layout()

        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        if show and not is_gui_embedding:
            plt.show()
        return current_fig, axes

    def _generate_data(self) -> dict:
        """Traces every two-bounce ghost path and computes its metrics.

        Returns:
            dict: The ghost data, sorted by irradiance, brightest first.
        """
        surfaces = self.optic.surfaces.surfaces
        for index, surface in enumerate(surfaces):
            if surface.interaction_model.is_reflective:
                raise ValueError(
                    "GhostAnalysis supports refractive systems only, but "
                    f"surface {index} is reflective."
                )

        wavelength = self.wavelengths[0].value
        reflectance = self._surface_reflectance(wavelength)
        airy_radius = 1.22 * wavelength * 1e-3 * be.abs(self.optic.paraxial.FNO())

        candidates = sorted(index for index, value in reflectance.items() if value > 0)
        rows = []
        for position, j in enumerate(candidates):
            for i in candidates[:position]:
                rows.append(
                    self._trace_ghost(
                        i, j, reflectance[i] * reflectance[j], wavelength, airy_radius
                    )
                )

        order = np.argsort(-np.array([float(be.to_numpy(r[-1])) for r in rows]))
        rows = [rows[k] for k in order]
        keys = (
            "reflectance",
            "transmitted_fraction",
            "energy_fraction",
            "rms_radius",
            "focus_shift",
            "irradiance",
        )
        data = {"surfaces": [row[0] for row in rows]}
        for column, key in enumerate(keys, start=1):
            values = [be.ravel(be.asarray(row[column])) for row in rows]
            data[key] = be.concatenate(values) if values else be.zeros(0)
        return data

    def _trace_ghost(
        self,
        i: int,
        j: int,
        reflectance: float,
        wavelength: float,
        airy_radius,
    ) -> tuple:
        """Traces one ghost path and computes its metrics at the image.

        Args:
            i (int): The index of the second reflecting surface.
            j (int): The index of the first reflecting surface.
            reflectance (float): The product of the two reflectances.
            wavelength (float): The wavelength in µm.
            airy_radius: The floor on the spot radius, in lens units.

        Returns:
            tuple: ``((i, j), reflectance, transmitted_fraction,
            energy_fraction, rms_radius, focus_shift, irradiance)``.
        """
        Hx, Hy = self.field
        rays = self.sequence(i, j).trace(
            Hx, Hy, wavelength, num_rays=self.num_rings, distribution="hexapolar"
        )

        x, y = rays.x, rays.y
        valid = be.isfinite(x) & be.isfinite(y) & be.isfinite(rays.i) & (rays.i > 0)
        num_valid = int(be.to_numpy(be.sum(valid)))
        transmitted = num_valid / x.shape[0]
        energy = reflectance * transmitted

        if num_valid < 2:
            nan = be.asarray(float("nan"))
            return ((i, j), reflectance, transmitted, energy, nan, nan, be.asarray(0.0))

        x, y = x[valid], y[valid]
        slope_x = rays.L[valid] / rays.N[valid]
        slope_y = rays.M[valid] / rays.N[valid]
        dx, dy = x - be.mean(x), y - be.mean(y)
        dsx, dsy = slope_x - be.mean(slope_x), slope_y - be.mean(slope_y)

        rms_radius = be.sqrt(be.mean(dx**2 + dy**2))

        # Plane of least RMS spot size for the straight-line ray extensions:
        # minimize sum |d + dz * ds|^2 over dz.
        image_z = self.optic.image_surface.geometry.cs.z
        z_offset = be.mean(rays.z[valid]) - image_z
        focus_shift = z_offset - be.sum(dx * dsx + dy * dsy) / be.sum(dsx**2 + dsy**2)

        radius = be.maximum(rms_radius, airy_radius)
        irradiance = energy / (be.pi * radius**2)
        return (
            (i, j),
            reflectance,
            transmitted,
            energy,
            rms_radius,
            focus_shift,
            irradiance,
        )

    def _surface_reflectance(self, wavelength: float) -> dict[int, float]:
        """Resolves the reflectance of every candidate surface.

        Args:
            wavelength (float): The wavelength in µm.

        Returns:
            dict[int, float]: The reflectance of each candidate surface,
            keyed by surface index.

        Raises:
            ValueError: If a reflectance is outside ``[0, 1]`` or a surface
                index is out of range.
        """
        num_surfaces = self.optic.surfaces.num_surfaces
        interior = range(1, num_surfaces - 1)

        if self._reflectance is None:
            surfaces = self.optic.surfaces.surfaces
            values = {}
            for index in interior:
                surface = surfaces[index]
                n1 = surface.material_pre.n(wavelength)
                n2 = surface.material_post.n(wavelength)
                fresnel = be.to_numpy(((n1 - n2) / (n1 + n2)) ** 2)
                values[index] = float(np.ravel(fresnel)[0])
            return values

        if isinstance(self._reflectance, dict):
            values = {int(k): float(v) for k, v in self._reflectance.items()}
            for index in values:
                if not 0 < index < num_surfaces:
                    raise ValueError(
                        f"Surface index {index} in reflectance is out of range: "
                        f"it must be between 1 and {num_surfaces - 1}."
                    )
        else:
            values = dict.fromkeys(interior, float(self._reflectance))

        for index, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"The reflectance of surface {index} must be in [0, 1], "
                    f"got {value}."
                )
        return values

    @staticmethod
    def _ghost_steps(i: int, j: int, num_surfaces: int) -> list[RawStep]:
        """Builds the raw steps of the ghost path that reflects off ``j``, then ``i``.

        Args:
            i (int): The index of the second reflecting surface.
            j (int): The index of the first reflecting surface.
            num_surfaces (int): The number of surfaces in the optic.

        Returns:
            list: The raw step list.
        """
        return [
            *range(j),
            (j, "reflect"),
            *range(j - 1, i, -1),
            (i, "reflect"),
            *range(i + 1, num_surfaces),
        ]
